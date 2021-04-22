#pragma once

#include <desc/Composition.hxx>
#include <disc/Settings.hxx>
#include <desc/utilities/EmptyCallback.hxx>

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
        if (tombstone.test(i)) { c.erase(std::next(c.begin(), j)); }
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
    // small_vector<size_t, 64> map(1 + *std::max_element(c.data.template col<0>().begin(),

    // std::vector<size_t> map;
    // map.reserve(64);
    // for(size_t i = 0; i < c.data.num_components(); ++i) {
    //     auto to = label(c.data.subset(i)[0]);
    //     if (to <= map.size()) map.resize(to + 1);
    //     map[to] = i;
    // }
    size_t largest_label = 0;
    for (size_t i = 0; i < c.data.num_components(); ++i)
    {
        largest_label = std::max(largest_label, label(c.data.subset(i)[0]));
    }
    std::vector<size_t> map(1 + largest_label);
    // small_vector<size_t, 64> map(1 + largest_label);
    for (size_t i = 0; i < c.data.num_components(); ++i)
    {
        map[label(c.data.subset(i)[0])] = i;
    }
    return map;
}

template <typename Trait, typename IDs>
auto create_tombstones(Composition<Trait> const& c, const IDs& rev)
{
    itemset<tag_dense> tombstone(c.data.num_components(), true);

    for (auto x : c.data)
    {
        tombstone.erase(rev[label(x)]);
        if (tombstone.empty()) break;
    }
    return tombstone;
}

template <typename Trait, typename IDs>
size_t remove_empty_components(Composition<Trait>& c, const IDs& rev)
{
    auto tombstone = create_tombstones(c, rev);

    auto cnt = count(tombstone);

    if (cnt == 0) return cnt;

    erase_from_composition(tombstone, c.models);
    // erase_from_composition(tombstone, c.subset_encodings);
    erase_from_composition(tombstone, c.assignment);

    c.data.group_by_label();
    simplify_labels(c.data);

    return cnt;
}

template <typename S>
auto get_component_label(const PartitionedData<S>& data)
{
    std::vector<size_t> ids(data.num_components());
    // small_vector<size_t, 32> ids(data.num_components());
    for (size_t i = 0; i < ids.size(); ++i) { ids[i] = label(data.subset(i)[0]); }
    return ids;
}

template <typename Trait>
auto reassign_rows(Composition<Trait>& c)
{
    using float_type = typename Trait::float_type;
    auto ids         = get_component_label(c.data);

    size_t count = 0;

    //
    // any data-point t is assigned to the component D* with highest likelihood
    // D* <- D* \cup t for D* = argmax_{D_i} p(t | S_i)
    //
    // pstl::for_each(pstl::execution::par_unseq,
    //                c.data.begin(),
    //                c.data.end(),
    //                [&](auto&& x) {
    //                    float_type p_max = 0;
    //                    for (size_t i = 0; i < c.data.num_components(); ++i)
    //                    {
    //                        const auto p = c.models[i].expectation(point(x));
    //                        if (p_max < p)
    //                        {
    //                            p_max    = p;
    //                            label(x) = i;
    //                        }
    //                    }
    //                })

#pragma omp parallel for
    for (size_t k = 0; k < c.data.size(); ++k)
    {
        std::pair<size_t, float_type> best = {0, -1};

        for (size_t i = 0; i < c.data.num_components(); ++i)
        {
            const auto p = c.models[i].expectation(c.data.point(k));

            if (best.second < p) best = {i, p};
        }

        count += c.data.label(k) != ids[best.first];

        c.data.label(k) = ids[best.first];
    }

    return count;
}

// template <typename Trait,
//           typename CALLBACK  = EmptyCallback,
//           typename Interface = DefaultAssignment>
// void reassign_components_step(Composition<Trait>& c, const Config& cfg, Interface&& f = {})
// {
//     auto r = label_to_component_id(c);
//     reassign_rows(c);
//     remove_empty_components(c, r);
//     c.data.group_by_label();
//     simplify_labels(c.data);
//     characterize_components(c, cfg, f);
//     c.encoding = encode(c, cfg);
// }

// template <typename Trait, typename CB = EmptyCallback, typename Interface = DefaultAssignment>
// void reassign_components(Composition<Trait>& c,
//                          const Config&   cfg,
//                          size_t              max_iteration = 1,
//                          Interface&&         f             = {},
//                          CB&&                cb_gain       = {})
// {
//     c.data.group_by_label();
//     simplify_labels(c.data);
//     assert(check_invariant(c));

//     while (c.data.num_components() > 1 && max_iteration-- > 0)
//     {
//         auto before_encoding = c.encoding;

//         reassign_components_step(c, cfg, f);
//         assert(check_invariant(c));

//         cb_gain(before_encoding, c.encoding);

//         if (std::abs(before_encoding.objective() - c.encoding.objective()) < 1) break;
//     }
// }


template <typename Trait, typename Interface = DefaultAssignment>
void reassign_components(Composition<Trait>& c,
                          const Config&   cfg,
                          size_t              max_iteration = 1,
                          Interface&&         f             = {})
{
    assert(check_invariant(c));
    while (c.data.num_components() > 1 && max_iteration-- > 0)
    {
        auto r = label_to_component_id(c);
        if (reassign_rows(c) == 0) break;
        remove_empty_components(c, r);
        characterize_components(c, cfg, f);
        assert(check_invariant(c));
    }
}


} // namespace disc
} // namespace sd