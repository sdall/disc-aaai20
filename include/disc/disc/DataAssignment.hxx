#pragma once

#include <disc/desc/Composition.hxx>
#include <disc/disc/Settings.hxx>
#include <disc/utilities/EmptyCallback.hxx>

#include <iterator>
#include <utility>

namespace sd
{
namespace disc
{

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

template <typename Trait>
auto label_to_component_id(const Composition<Trait>& c)
{
    small_vector<size_t, 32> rids(1 + *std::max_element(c.data.template col<0>().begin(),
                                                        c.data.template col<0>().end()));
    for (size_t i = 0; i < c.data.num_components(); ++i)
    {
        rids[label(c.data.subset(i)[0])] = i;
    }
    return rids;
}

template <typename Trait, typename IDs>
auto create_tombstones(Composition<Trait> const& c, const IDs& rev)
{
    itemset<tag_dense> tombstone(c.data.num_components(), true);

    for (auto x : c.data)
    {
        tombstone.erase(rev[label(x)]);
        if (tombstone.empty())
            break;
    }
    return tombstone;
}

template <typename Trait, typename IDs>
size_t remove_empty_components(Composition<Trait>& c, const IDs& rev)
{
    auto tombstone = create_tombstones(c, rev);

    erase_from_composition(tombstone, c.models);
    erase_from_composition(tombstone, c.subset_encodings);
    erase_from_composition(tombstone, c.assignment);

    return count(tombstone);
}

template <typename S>
auto component_index_to_label(const PartitionedData<S>& data)
{
    small_vector<size_t, 32> ids(data.num_components());
    for (size_t i = 0; i < ids.size(); ++i)
    {
        ids[i] = label(data.subset(i)[0]);
    }
    return ids;
}

template <typename Trait>
auto reassign_rows(Composition<Trait>& c)
{
    using float_type = typename Trait::float_type;
    auto ids         = component_index_to_label(c.data);

    assert(c.assignment.size() == c.data.num_components());
    //
    // any data-point t is assigned to the component D* with highest likelihood
    // D* <- D* \cup t for D* = argmax_{D_i} p(t | S_i)
    //
#pragma omp parallel for
    for (size_t k = 0; k < c.data.size(); ++k)
    {
        std::pair<size_t, float_type> best = {0, -1};

        for (size_t i = 0; i < c.data.num_components(); ++i)
        {
            const auto p = c.models[i].expectation(c.data.point(k));

            if (best.second < p)
                best = {i, p};
        }

        c.data.label(k) = ids[best.first];
    }
}

template <typename Trait, typename CALLBACK = EmptyCallback>
void reassign_components_step(Composition<Trait>& c, const DiscConfig& cfg)
{
    auto r = label_to_component_id(c);
    reassign_rows(c);
    remove_empty_components(c, r);
    c.data.group_by_label();
    simplify_labels(c.data);
    characterize_components(c, cfg);
}

template <typename Trait, typename CB = EmptyCallback>
void reassign_components(Composition<Trait>& c,
                         const DiscConfig&   cfg,
                         size_t              max_iteration = 1,
                         CB&&                cb_gain       = {})
{
    c.data.group_by_label();
    simplify_labels(c.data);
    assert(check_invariant(c));

    while (c.data.num_components() > 1 && max_iteration-- > 0)
    {
        auto before_encoding = c.encoding;

        reassign_components_step(c, cfg);
        assert(check_invariant(c));

        cb_gain(before_encoding, c.encoding);

        if (std::abs(before_encoding.objective() - c.encoding.objective()) < 1)
            break;
    }
}

} // namespace disc
} // namespace sd