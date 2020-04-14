#pragma once

#include <disc/disc/Composition.hxx>
#include <disc/disc/Encoding.hxx>
#include <disc/disc/Settings.hxx>

namespace sd
{
namespace disc
{

template <typename Trait, typename Pattern, typename float_type_1>
bool is_candidate_assignment_good(const Composition<Trait>& c,
                                  const Pattern&            candidate_pattern,
                                  const size_t              component_index,
                                  const float_type_1        fr,
                                  const bool                bic)
{

    if (fr == 0 || !c.models[component_index].is_itemset_allowed(candidate_pattern))
    {
        return false;
    }

    auto estimate = c.models[component_index].expected_frequency(candidate_pattern);

    if (estimate == 0)
    {
        estimate = float_type_1(1) / (c.data.size() * c.data.dim);
    }

    const auto subset        = c.data.subset(component_index);
    const auto support       = static_cast<size_t>(fr * subset.size());
    const auto expected_gain = support * std::log2(fr / estimate);

    const auto cost = bic ? additional_cost_bic(subset.size()) : additional_cost_mdl(support);
    return expected_gain > cost;
}

template <typename Trait, typename Pattern, typename float_type_1>
bool find_assignment_if(Composition<Trait>& c,
                        const size_t        future_candidate_index,
                        const Pattern&      candidate_pattern,
                        const size_t        component_index,
                        const float_type_1  fr,
                        const bool          bic,
                        const bool          estimate_coefficients = false)
{
    auto do_assignment =
        is_candidate_assignment_good(c, candidate_pattern, component_index, fr, bic);

    if (do_assignment)
    {
        c.assignment[component_index].insert(future_candidate_index);
        c.models[component_index].insert(fr, candidate_pattern, estimate_coefficients);
        return true;
    }
    else
    {
        return false;
    }
}

template <typename Trait, typename Candidate>
bool find_assignment_first_component_only(Composition<Trait>& c,
                                          const Candidate&    candidate,
                                          bool                bic)
{
    using float_type      = typename Trait::float_type;
    const auto overall_fr = static_cast<float_type>(candidate.support) / c.data.size();
    return find_assignment_if(c, c.summary.size(), candidate.pattern, 0, overall_fr, bic, true);
}

template <typename Trait, typename Candidate, typename Masks>
size_t find_assignment_all_components(Composition<Trait>& c,
                                      const Candidate&    candidate,
                                      const Masks&        masks,
                                      bool                bic)
{
    using float_type = typename Trait::float_type;
    assert(masks.size() == c.assignment.size());
    size_t assignment_count = 0;

    for (size_t i = 0; i < c.assignment.size(); ++i)
    {
        auto s = size_of_intersection(candidate.row_ids, masks[i]);
        auto q = static_cast<float_type>(s) / c.data.size();

        if (find_assignment_if(c, c.summary.size(), candidate.pattern, i, q, bic, true))
        {
            assignment_count += 1;
        }
    }
    return assignment_count;
}

template <typename Trait, typename Candidate, typename Masks>
void insert_pattern_to_summary(Composition<Trait>& c,
                               const Candidate&    candidate,
                               const Masks&        masks)
{
    using float_type = typename Trait::float_type;

    assert(c.frequency.extent(1) > 0);

    const auto glob_frequency = static_cast<float_type>(candidate.support) / c.data.size();

    c.summary.insert(glob_frequency, candidate.pattern);
    c.frequency.push_back();
    auto new_q = c.frequency[c.frequency.length() - 1];

    for (size_t j = 0; j < c.data.num_components(); ++j)
    {
        auto s   = size_of_intersection(candidate.row_ids, masks[j]);
        auto q   = static_cast<float_type>(s) / c.data.size();
        new_q(j) = q;
    }
    new_q.back() = glob_frequency;
}

template <typename Trait, typename Candidate, typename Masks>
bool find_assignment(Composition<Trait>& c,
                     const Candidate&    candidate,
                     const Masks&        masks,
                     bool                bic)
{
    size_t assignment_count = 0;

    if (c.data.num_components() == 1)
    {
        assignment_count += find_assignment_first_component_only(c, candidate, bic);
    }
    else
    {
        assignment_count += find_assignment_all_components(c, candidate, masks, bic);
    }

    if (assignment_count > 0)
    {
        insert_pattern_to_summary(c, candidate, masks);
    }

    return assignment_count > 0;
}

template <typename Trait>
auto construct_component_masks(const Composition<Trait>& c)
{
    using tid_container = long_storage_container<typename Trait::pattern_type>;

    assert(check_invariant(c));

    std::vector<tid_container> masks;
    masks.reserve(16);
    if (c.data.size() > 1)
    {
        masks.resize(c.data.num_components(), tid_container{c.data.size()});
        for (size_t s = 0, row = 0, n = c.data.num_components(); s < n; ++s)
        {
            for ([[maybe_unused]] const auto& x : c.data.subset(s))
            {
                masks[s].insert(row);
                ++row;
            }
        }
    }
    return masks;
}

} // namespace disc
} // namespace sd