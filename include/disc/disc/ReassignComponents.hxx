#pragma once

#include <disc/disc/Composition.hxx>
#include <disc/disc/DecomposeContext.hxx>
#include <disc/disc/DiscoverPatternsets.hxx>
#include <disc/utilities/EmptyCallback.hxx>

#include <iterator>
#include <utility>

namespace sd
{
namespace disc
{

template <typename Subset>
bool component_was_removed(const Subset& s, size_t old_label)
{

    return s.size() == 0 || std::all_of(s.begin(), s.end(), [&](const auto& e) {
               return label(e) != old_label;
           });
    /* ||
   !std::any_of(
       s.begin(), s.end(), [&](const auto& e) { return label(e) != label(s[0]); }); */
}

template <typename Trait>
bool remove_one_component_if_empty(Composition<Trait>& c,
                                   size_t              component_index,
                                   size_t              component_label,
                                   size_t              remove_index)
{
    if (component_was_removed(c.data, component_label))
    {
        c.models.erase(c.models.begin() + remove_index);
        c.assignment.erase(c.assignment.begin() + remove_index);
        c.subset_encodings.erase(c.subset_encodings.begin() + remove_index);
        return true;
    }

    return false;
}

template <typename Container>
void erase_from_composition(const itemset<tag_dense>& tombstone, Container& c)
{
    assert(c.size() == tombstone.length());
    [[maybe_unused]] auto expected_size = c.size() - count(tombstone);

    for (size_t i = 0, j = 0; i < tombstone.length(); ++i)
    {
        if (tombstone.test(i))
        {
            c.erase(std::next(c.begin(), j));
        }
        else
        {
            ++j;
        }
    }

    assert(c.size() == expected_size);
}

template <typename Trait, typename Callback>
size_t
remove_empty_components(Composition<Trait>& c, slice<const size_t> rev, Callback&& callback)
{
    // auto d = std::max_element(ids.begin(), ids.end());
    itemset<tag_dense> tombstone(c.data.num_components(), true);

    for (auto x : c.data)
    {
        tombstone.erase(rev[label(x)]);
        if (tombstone.empty())
            return 0;
    }

    erase_from_composition(tombstone, c.models);
    erase_from_composition(tombstone, c.subset_encodings);
    erase_from_composition(tombstone, c.assignment);

    iterate_over(tombstone, callback);

    return count(tombstone);
}

template <typename Trait>
auto only_reassign_rows(Composition<Trait>& c, slice<const size_t> ids)
{
    using float_type = typename Trait::float_type;

    assert(c.assignment.size() == c.data.num_components());
    //
    // any data-point t is assigned to the component D* with highest likelihood
    // D* <- D* \cup t for D* = argmax_{D_i} p(t | S_i)
    //
#pragma omp parallel for
    for (size_t k = 0; k < c.data.size(); ++k)
    {
        const auto& x = c.data.point(k);
        auto        m = std::pair<size_t, float_type>{0, -1};

        for (size_t i = 0; i < c.data.num_components(); ++i)
        {
            const auto p = c.models[i].expected_generalized_frequency(x);

            if (m.second < p)
                m = {i, p};
        }

        c.data.label(k) = ids[m.first];
    }
}

template <typename Trait>
auto make_label_to_component_index_vector(const Composition<Trait>& c)
{
    small_vector<size_t, 16> rids(1 + *std::max_element(c.data.template col<0>().begin(),
                                                        c.data.template col<0>().end()));
    for (size_t i = 0; i < c.data.num_components(); ++i)
    {
        rids[label(c.data.subset(i)[0])] = i;
    }
    return rids;
}

template <typename Trait>
auto component_ids_as_vector(const Composition<Trait>& c)
{
    small_vector<size_t, 16> ids(c.data.num_components());
    for (size_t i = 0; i < ids.size(); ++i)
    {
        ids[i] = label(c.data.subset(i)[0]);
    }
    return ids;
}

template <typename Trait, typename CALLBACK = EmptyCallback>
void reassign_components_step(Composition<Trait>&          c,
                              const DecompositionSettings& settings,
                              CALLBACK&&                   callback = {})
{
    auto ids = component_ids_as_vector(c);
    auto rev = make_label_to_component_index_vector(c);

    only_reassign_rows(c, ids);

    [[maybe_unused]] auto before      = c.data.num_components();
    [[maybe_unused]] auto num_removes = remove_empty_components(c, rev, callback);

    c.data.group_by_label();

    assert(c.data.num_components() == before - num_removes);
    assert(c.models.size() == c.data.num_components());
    assert(check_invariant(c));

    compute_frequency_matrix(c.data, c.summary, c.frequency);
    retrain_models(c);
    c.encoding = encoding_length_mdm(c, settings.use_bic);
}

template <typename Trait, typename CB1 = EmptyCallback, typename CB2 = EmptyCallback>
Composition<Trait> reassign_components(Composition<Trait>    c,
                                       DecompositionSettings settings,
                                       size_t                max_iteration = 20,
                                       CB1&&                 cb_rem        = {},
                                       CB2&&                 cb_gain       = {})
{
    assert(check_invariant(c));

    c.data.group_by_label();
    simplify_labels(c.data);

    assert(check_invariant(c));

    while (c.data.num_components() > 1 && max_iteration-- > 0)
    {
        auto before_encoding = c.encoding;

        reassign_components_step(c, settings, cb_rem);

        cb_gain(before_encoding, c.encoding);

        auto pv = nhc_pvalue(before_encoding.objective(), c.encoding.objective());

        if (pv < 0.05 || std::abs(before_encoding.objective() - c.encoding.objective()) < 1)
            break;
    }

    assert(check_invariant(c));
    return c;
}

template <typename Trait, typename CALLBACK = EmptyCallback>
Composition<Trait> reassign_components(Composition<Trait>&&                 c,
                                       NodewiseDecompositionContext<Trait>& context,
                                       DecompositionSettings                settings,
                                       size_t                               max_iteration = 20,
                                       CALLBACK&&                           cb_gain       = {})
{
    auto remover = [&context](size_t i) {
        auto& dn  = context.decompose_next;
        auto  pos = std::find(dn.begin(), dn.end(), i);
        if (pos != dn.end())
            dn.erase(pos);
    };

    return reassign_components(std::forward<Composition<Trait>>(c),
                               settings,
                               max_iteration,
                               std::move(remover),
                               cb_gain);
}

} // namespace disc
} // namespace sd