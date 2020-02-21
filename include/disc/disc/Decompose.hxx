#pragma once

#include <disc/disc/CharacterizeComponents.hxx>
#include <disc/disc/Composition.hxx>
#include <disc/disc/DecomposeContext.hxx>
#include <disc/disc/DiscoverPatternsets.hxx>
#include <disc/disc/ReassignComponents.hxx>
#include <disc/disc/TestDivergence.hxx>

#include <disc/disc/DiscoverPatternsetsActualGain.hxx>

namespace sd
{
namespace disc
{

// template <typename Trait>
// Composition<Trait> find_patternsets(Composition<Trait>&& com, DecompositionSettings settings)
// {
//     if (settings.quick)
//     {
//         return discover_patternsets(std::forward<Composition<Trait>>(com),
//         std::move(settings));
//     }
//     else
//     {
//         return discover_patternsets_test_actual_gain(std::forward<Composition<Trait>>(com),
//                                                      std::move(settings));
//     }
// }

// template <typename Trait, typename S, typename T>
// Composition<Trait> find_patternsets(Composition<Trait>&&                   com,
//                                     nonstd::optional<SlimGenerator<S, T>>& g,
//                                     DecompositionSettings                  settings)
// {
//     if (!g.has_value())
//         return discover_patternsets(std::forward<Composition<Trait>>(com),
//         std::move(settings));

//     if (settings.quick)
//     {
//         return discover_patternsets(
//             std::forward<Composition<Trait>>(com), g.value(), std::move(settings));
//     }
//     else
//     {
//         return discover_patternsets_test_actual_gain(
//             std::forward<Composition<Trait>>(com), g.value(), std::move(settings));
//     }
// }

template <typename Trait, typename P>
void just_split(Composition<Trait>& com, const P& x, size_t index, size_t label)
{
    auto set = com.data.subset(index);
    for (auto [y, t, _] : set)
    {
        if (is_subset(x, t))
            y = label;
    }
    com.data.group_by_label();
}

template <typename Trait>
void undo_split_data(Composition<Trait>& com, size_t label_before, size_t label_after)
{
    for (auto [y, t, _] : com.data)
    {
        if (y == label_before)
            y = label_after;
    }
    com.data.group_by_label();
}

template <typename Trait, typename P>
std::pair<bool, Composition<Trait>>
split_no_reassign_points(Composition<Trait>           com,
                         const P&                     x,
                         size_t                       index,
                         size_t                       label_after,
                         const DecompositionSettings& settings)
{
    size_t before       = com.data.num_components();
    size_t label_before = label(com.data.subset(index)[0]);

    just_split(com, x, index, label_after);

    if (com.data.num_components() <= before)
    {
        undo_split_data(com, label_after, label_before);
        return {false, com};
    }
    else
    {
        // com.frequency = make_frequencies(com.data, com.summary);
        // com.initial_encoding = com.encoding;
        // com                  = reassign_patterns_quick(std::move(com));
        // com.encoding         = encoding_length_mdm(com);
        const auto s = com.data.num_components() - 1;
        com          = characterize_split(std::move(com), {index, s}, settings);
        return {true, std::move(com)};
    }
}

template <typename Trait, typename P>
std::pair<bool, Composition<Trait>> split_and_reassign(Composition<Trait>           com,
                                                       const P&                     x,
                                                       size_t                       index,
                                                       size_t                       label,
                                                       const DecompositionSettings& settings)
{
    auto p = split_no_reassign_points(std::move(com), x, index, label, settings);
    if (!p.first)
        return p;
    p.second = reassign_components(
        std::move(p.second), settings, settings.max_em_iterations.value_or(100000));
    return p;
}

template <typename Trait, typename P>
std::pair<bool, Composition<Trait>> do_significant_split(Composition<Trait> com,
                                                         const P&           x,
                                                         size_t             x_index,
                                                         size_t             index,
                                                         size_t             label,
                                                         size_t total_number_of_tests,
                                                         const DecompositionSettings& settings)
{
    using float_type = typename Trait::float_type;

    auto before_encoding = com.encoding;

    SignificanceStatistic<float_type> stat;
    stat.is_significant = true;
    stat.pvalue         = 1;

    const auto alpha_js  = float_type(settings.alpha) / total_number_of_tests;
    const auto alpha_nhc = float_type(settings.alpha);

    const std::pair<size_t, size_t> component = {index, com.data.num_components()};

    if (settings.test_divergence)
    {
        auto [has_split, d] =
            split_no_reassign_points(std::move(com), x, index, label, settings);
        com = std::move(d);

        if (!has_split) // || before_encoding.objective() <= com.encoding.objective())
        {
            return {false, std::move(com)};
        }

        auto stat =
            significance_statistics(com, alpha_js, component.first, component.second, x_index);

        if (!stat.is_significant)
        {
            return {false, std::move(com)};
        }

        com = reassign_components(
            std::move(com), settings, settings.max_em_iterations.value_or(100000));
    }
    else
    {
        auto [has_split, d] = split_and_reassign(std::move(com), x, index, label, settings);
        com                 = std::move(d);
        if (!has_split)
        {
            return {false, std::move(com)};
        }
    }

    auto pvalue         = nhc_pvalue(before_encoding.objective(), com.encoding.objective());
    bool is_significant = pvalue < alpha_nhc;

    if (is_significant && settings.test_divergence)
    {
        stat =
            significance_statistics(com, alpha_js, component.first, component.second, x_index);
        is_significant &= stat.is_significant;
    }

    // info::log_candidate_trial<float_type>(
    //     x, before_encoding, com.encoding, stat.pvalue, pvalue, alpha_nhc);

    return {is_significant, std::move(com)};
}

template <typename T>
bool is_split_fr_low(const T& fr, const T& minfr)
{
    return fr < minfr || 1.0 - fr < minfr;
}

template <typename T, typename S>
void insert_distinct_nonsingletons(const disc::LabeledDataset<T, S>& splitset,
                                   LabeledDataset<T, S>&             summary)
{
    for (const auto& [y, x] : splitset)
    {
        if (!is_singleton(x) && !disc::contains_any(x, summary))
            summary.insert(y, x);
    }
}

template <typename Trait>
auto decompose_step_one_component(const Composition<Trait>&            com,
                                  const size_t                         comp_index,
                                  NodewiseDecompositionContext<Trait>& ctx,
                                  const DecompositionSettings&         settings,
                                  std::atomic_int&                     label_counter)
    -> nonstd::optional<std::pair<size_t, Composition<Trait>>>
{
    constexpr const bool top_level_parallel = true;

    using float_type = typename Trait::float_type;

    nonstd::optional<std::pair<size_t, size_t>> split_on = nonstd::nullopt;
    nonstd::optional<Composition<Trait>>        result   = nonstd::nullopt;

    assert(com.frequency.extent(0) == com.summary.size());
    assert(comp_index < com.assignment.size());
    assert(comp_index < com.data.num_components());

#pragma omp parallel for if (top_level_parallel)
    for (size_t jj = 0; jj < com.assignment[comp_index].size(); ++jj)
    {
        const auto j = com.assignment[comp_index][jj];

        if (ctx.has_rejected.find({comp_index, j}) != ctx.has_rejected.end())
            continue;

        assert(j < com.summary.size());
        assert(com.frequency.shape().contains({j, comp_index}));
        assert(com.frequency.shape().contains({j, com.frequency.extent(1) - 1}));

        const auto q_i   = com.frequency(j, comp_index);
        const auto q     = com.frequency(j, com.frequency.extent(1) - 1);
        const auto minfr = float_type(settings.min_support) / com.data.size();

        if (q_i < minfr || /*(T(1) - q_i) < minfr ||*/ q < minfr)
            continue;
        if (auto test_j = significance_statistics_pattern(com, settings.alpha, comp_index, j);
            !test_j.is_significant)
        {
            ctx.has_rejected.insert({comp_index, j});
            continue;
        }

        const auto  label         = label_counter++;
        const auto& split_pattern = com.summary.point(j);

        auto [is_significant, d] = do_significant_split(
            com, split_pattern, j, comp_index, label, ctx.total_number_of_tests, settings);

        auto& /*clang-7 bug*/ e = d;

        is_significant &= d.data.num_components() > com.data.num_components();

        if (is_significant)
        {
#pragma omp critical
            {
                if (!result ||
                    (result && result->encoding.objective() > e.encoding.objective()))
                {
                    split_on = {j, label};
                    result   = std::move(e);
                }
            }
        }
        else
        {
#pragma omp critical
            {
                ctx.has_rejected.insert({comp_index, j});
            }
        }
    }

    if (split_on.has_value())
    {
        ctx.has_rejected.insert(*split_on);
        return {{split_on.value().first, std::move(result.value())}};
    }
    else
    {
        return {};
    }
}

template <typename Trait, typename CALLBACK = EmptyCallback>
std::pair<bool, Composition<Trait>> decompose_round(Composition<Trait>                   com,
                                                    NodewiseDecompositionContext<Trait>& ctx,
                                                    const DecompositionSettings& settings,
                                                    CALLBACK&&                   callback = {})
{
    // using optional_triple = nonstd::optional<std::tuple<size_t, size_t, size_t>>;

    insert_distinct_nonsingletons(ctx.splitset, com.summary);

    // info::log_overall_progress(com.encoding);

    std::atomic_int label_counter = com.data.num_components() + 1;

    ctx.total_number_of_tests += ctx.to_decompose.size() * com.summary.size();
    /*
     * for any to-be-decomposed component:
     */
    for (size_t ii = 0; ii < ctx.to_decompose.size(); ++ii)
    {
        const auto comp_index = ctx.to_decompose[ii];

        if (comp_index >= com.data.num_components())
            continue;

        auto next = decompose_step_one_component(com, comp_index, ctx, settings, label_counter);
        if (next)
        {
            com = std::move(next->second);

            // if(cfg.intermediate_mining) {
            //     com = discover_patternsets(std::move(com), cfg);
            // }

            callback(std::as_const(com));

            ctx.decompose_next.push_back(comp_index);
            ctx.decompose_next.push_back(com.data.num_components() - 1);

            // info::log_found_local_split(
            //     comp_index, next->first, com.summary.point(next->first));
        }
    }

    ctx.to_decompose.clear();
    std::swap(ctx.to_decompose, ctx.decompose_next);

    simplify_labels(com.data);

    // info::log_overall_progress(com.encoding);

    return {!ctx.to_decompose.empty(), std::move(com)};
}

template <typename Trait, typename Call = EmptyCallback>
Composition<Trait> decompose_maybe_mine(Composition<Trait>                   com,
                                        NodewiseDecompositionContext<Trait>& ctx,
                                        DecompositionSettings                settings,
                                        Call&&                               report = {})
{
    // ctx.total_number_of_tests = 1;
    ctx.to_decompose.resize(com.data.num_components());
    std::iota(ctx.to_decompose.begin(), ctx.to_decompose.end(), 0);

    const auto& max_k = settings.max_cliques;

    report(std::as_const(com));

    while (!ctx.to_decompose.empty())
    {
        const auto k = com.data.num_components();
        if (max_k && k >= *max_k)
            break;

        auto [has_next, d] = decompose_round(std::move(com), ctx, settings, report);
        com                = std::move(d);

        if (settings.intermediate_mining)
        {
            com = discover_patternsets(std::move(com), /*ctx.generator,*/ settings);
            report(std::as_const(com));
        }

        if (!has_next)
        {
            break;
        }
    }

    return com;
}

template <typename Trait, typename Call = EmptyCallback>
Composition<Trait>
decompose_maybe_mine(Composition<Trait> com, DecompositionSettings settings, Call&& report = {})
{
    NodewiseDecompositionContext<Trait> ctx;
    return decompose_maybe_mine(std::move(com), ctx, settings, report);
}

} // namespace disc
} // namespace sd