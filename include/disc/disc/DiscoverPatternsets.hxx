#pragma once

#include <disc/disc/DiscoverPatternset.hxx>
#include <disc/disc/HeuristicScore.hxx>
#include <disc/disc/Settings.hxx>
#include <disc/disc/SlimCandidateGeneration.hxx>
#include <disc/storage/Itemset.hxx>
#include <disc/utilities/EmptyCallback.hxx>

#include <chrono>

namespace sd
{
namespace disc
{

bool pattern_known(size_t future_candidate_index, size_t summary_size)
{
    return future_candidate_index < summary_size;
}

template <typename Trait, typename Pattern, typename float_type_1, typename float_type_2>
bool is_candidate_assignment_good(const Composition<Trait>& c,
                                  const Pattern&            candidate_pattern,
                                  const size_t              component_index,
                                  const float_type_1        fr,
                                  const bool                bic,
                                  const float_type_2        alpha)
{

    if (fr == 0 || !c.models[component_index].is_item_allowed(candidate_pattern))
    {
        return false;
    }

    auto estimate = c.models[component_index].expected_frequency(candidate_pattern);

    if (estimate == 0)
    {
        estimate = float_type_1(1) / (c.data.size() * c.data.dim);
    }

    const auto subset        = c.data.subset(component_index);
    const auto subset_size   = subset.size();
    const auto support       = static_cast<size_t>(fr * subset_size);
    const auto expected_gain = support * std::log2(fr / estimate);
    
    const float_type_1 add_cost = bic ? 0 : additional_cost_mdl_assignment(c,
                                                                           candidate_pattern,
                                                                           support,
                                                                           component_index,
                                                                           subset_size);
    return expected_gain > add_cost;
}

template <typename Trait, typename Pattern, typename float_type_1, typename float_type_2>
bool find_assignment_if(Composition<Trait>& c,
                        const size_t        future_candidate_index,
                        const Pattern&      candidate_pattern,
                        const size_t        component_index,
                        const float_type_1  fr,
                        const bool          bic,
                        const float_type_2  alpha,
                        const bool          estimate_coefficients = false)
{
    auto do_assignment =
        is_candidate_assignment_good(c, candidate_pattern, component_index, fr, bic, alpha);

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

template <typename Trait, typename Candidate, typename Float>
bool find_assignment_first_component_only(Composition<Trait>& c,
                                          const Candidate&    candidate,
                                          bool                bic,
                                          Float               alpha)
{
    using float_type      = typename Trait::float_type;
    const auto overall_fr = static_cast<float_type>(candidate.support) / c.data.size();
    return find_assignment_if(
        c, c.summary.size(), candidate.pattern, 0, overall_fr, bic, float_type(alpha), true);
}

template <typename Trait, typename Candidate, typename Masks, typename Float>
size_t find_assignment_all_components(Composition<Trait>& c,
                                      const Candidate&    candidate,
                                      const Masks&        masks,
                                      bool                bic,
                                      Float               alpha)
{
    using float_type = typename Trait::float_type;
    assert(masks.size() == c.assignment.size());
    size_t assignment_count = 0;

    for (size_t i = 0; i < c.assignment.size(); ++i)
    {
        auto s = size_of_intersection(candidate.row_ids, masks[i]);
        auto q = static_cast<float_type>(s) / c.data.size();

        if (find_assignment_if(
                c, c.summary.size(), candidate.pattern, i, q, bic, float_type(alpha), true))
        {
            assignment_count += 1;
            // estimate_model(c.models[i]);
        }
    }
    return assignment_count;
}

template <typename Trait, typename Candidate, typename Masks, typename Float>
bool find_assignment(Composition<Trait>& c,
                     const Candidate&    candidate,
                     const Masks&        masks,
                     bool                bic,
                     Float               alpha)
{

    using float_type = typename Trait::float_type;

    size_t assignment_count = 0;

    if (c.data.num_components() == 1)
    {
        assignment_count += find_assignment_first_component_only(c, candidate, bic, alpha);
    }
    else
    {
        assignment_count += find_assignment_all_components(c, candidate, masks, bic, alpha);
    }

    if (assignment_count > 0)
    {
        const auto overall_fr = static_cast<float_type>(candidate.support) / c.data.size();
        insert_pattern_to_summary(c, candidate.pattern, overall_fr);
    }

    return assignment_count > 0;
}

template <typename Trait>
auto construct_component_masks(const Composition<Trait>& c)
{
    using tid_container = long_storage_container<typename Trait::pattern_type>;

    assert(check_invariant(c));

    small_vector<tid_container, 16> masks;
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

template <typename Trait, typename Generator, typename CALLBACK = EmptyCallback>
Composition<Trait> discover_patternsets(Composition<Trait>    c,
                                        Generator&            gen,
                                        const MiningSettings& cfg,
                                        CALLBACK&&            callback = {})
{
    assert(c.data.num_components() > 0);
    assert(c.data.dim != 0);
    assert(check_invariant(c));

    c.initial_encoding = c.encoding;
    callback(std::as_const(c));

    if (cfg.max_iteration == 0)
    {
        return c;
    }

    const auto& masks = construct_component_masks(c);

    auto score_function = [&](const auto& x, const auto&...) {
        return mean_score(c, x, cfg.use_bic, masks);
    };
    auto is_pattern_allowed = [&c](const auto& t) {
        return std::any_of(
            begin(c.models), end(c.models), [&](auto& m) { return m.is_item_allowed(t); });
    };

    size_t patience   = cfg.max_patience;
    size_t items_used = 0;

    gen.compute_scores(score_function);
    gen.prune([&](const auto& t) { return t.score <= 0 || !is_pattern_allowed(t.pattern); });
    gen.order_candidates();

    const auto start_time = std::chrono::system_clock::now();

    for (size_t it = 0; it < cfg.max_iteration; ++it)
    {
        if (!gen.has_next())
            break;

        if (auto x = gen.next(); x && find_assignment(c, *x, masks, cfg.use_bic, cfg.alpha))
        {
            items_used = items_used + 1;
            patience   = std::max(patience * 2, cfg.max_patience);

            gen.add_next(*x, score_function);
            gen.prune(
                [&](const auto& t) { return t.score <= 0 || !is_pattern_allowed(t.pattern); });
        }
        else if (patience-- == 0)
            break;

        if (cfg.max_patternset_size && items_used > cfg.max_patternset_size.value())
            break;
        if (cfg.max_time && std::chrono::system_clock::now() - start_time > *cfg.max_time)
            break;
    }

    c.encoding = encoding_length_mdm(c, cfg.use_bic);

    callback(std::as_const(c));

    return c;
}

template <typename Trait, typename CALLBACK = EmptyCallback>
Composition<Trait> discover_patternsets(Composition<Trait>    c,
                                        const MiningSettings& cfg      = {},
                                        CALLBACK&&            callback = {})

{
    using S  = typename Trait::pattern_type;
    using T  = typename Trait::float_type;
    auto gen = SlimGenerator<S, T>(c.data, c.summary, cfg.min_support, cfg.max_pattern_size);
    return discover_patternsets(std::move(c), gen, cfg, callback);
}

} // namespace disc
} // namespace sd