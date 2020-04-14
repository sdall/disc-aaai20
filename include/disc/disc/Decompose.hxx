#pragma once

#include <disc/disc/CharacterizeSplit.hxx>
#include <disc/disc/Composition.hxx>
#include <disc/disc/DecomposeContext.hxx>
#include <disc/disc/ReassignComponents.hxx>
#include <disc/disc/TestDivergence.hxx>

namespace sd
{
namespace disc
{

template <typename Trait, typename P>
void just_split_no_group_by(Composition<Trait>& com, const P& x, size_t index, size_t label)
{
    auto set = com.data.subset(index);
    for (auto [y, t, _] : set)
    {
        if (is_subset(x, t))
            y = label;
    }
}

template <typename Trait, typename P>
void just_split(Composition<Trait>& com, const P& x, size_t index, size_t label)
{
    just_split_no_group_by(com, x, index, label);
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
bool split_and_characterize(Composition<Trait>&          com,
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
        return false;
    }
    else
    {
        characterize_split(com, {index, com.data.num_components() - 1}, settings);
        return true;
    }
}

template <typename Trait, typename P>
bool split_and_reassign(Composition<Trait>&          com,
                        const P&                     x,
                        size_t                       index,
                        size_t                       label,
                        const DecompositionSettings& settings)

{
    auto found = split_and_characterize(com, x, index, label, settings);
    if (found)
    {
        reassign_components(com, settings, settings.max_em_iterations.value_or(100000));
    }
    return found;
}

template <typename Trait, typename P, typename float_type>
bool split_test_reassign(Composition<Trait>&          com,
                         const P&                     x,
                         size_t                       index,
                         size_t                       label,
                         std::pair<size_t, size_t>    component,
                         size_t                       x_index,
                         const float_type             alpha_js,
                         const DecompositionSettings& settings)

{

    bool has_split = split_and_characterize(com, x, index, label, settings);

    if (!has_split ||
        !test_stat_divergence(com, alpha_js, component.first, component.second, x_index))
    {
        return false;
    }
    else
    {
        reassign_components(com, settings, settings.max_em_iterations.value_or(100000));
        return true;
    }
}

template <typename Trait, typename P, typename F>
bool do_split(Composition<Trait>&          com,
              const P&                     x,
              size_t                       x_index,
              size_t                       index,
              size_t                       label,
              const F&                     alpha_js,
              const DecompositionSettings& settings)
{
    assert(check_invariant(com));
    assert(x_index < com.summary.size());
    assert(index < com.data.num_components());

    if (settings.test_divergence)
    {
        return split_test_reassign(com,
                                   x,
                                   index,
                                   label,
                                   {index, com.data.num_components()},
                                   x_index,
                                   alpha_js,
                                   settings);
    }
    else
    {
        return split_and_reassign(com, x, index, label, settings);
    }
}

template <typename Trait, typename P, typename F>
bool do_significant_split(Composition<Trait>&          com,
                          const P&                     x,
                          size_t                       x_index,
                          size_t                       index,
                          size_t                       label,
                          const F&                     alpha_js,
                          const DecompositionSettings& settings)
{

    const auto enc_before = com.encoding;
    const auto k          = com.data.num_components();
    const bool has_split  = do_split(com, x, x_index, index, label, alpha_js, settings);
    // const auto pvalue     = nhc_pvalue(enc_before.objective(), com.encoding.objective());

    bool is_significant =
        com.encoding.objective() < enc_before.objective(); // pvalue < settings.alpha;

    if (settings.test_divergence && is_significant && has_split &&
        k < com.data.num_components())
    {
        is_significant &=
            test_stat_divergence(com, alpha_js, index, com.data.num_components() - 1, x_index);
    }

    return is_significant;
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
bool is_early_reject(const Composition<Trait>&    com,
                     size_t                       pattern_index,
                     size_t                       comp_index,
                     const DecompositionSettings& settings)
{
    using float_type = typename Trait::float_type;

    assert(pattern_index < com.summary.size());
    assert(com.frequency.shape().contains({pattern_index, comp_index}));
    assert(com.frequency.shape().contains({pattern_index, com.frequency.extent(1) - 1}));

    const auto q_i   = com.frequency(pattern_index, comp_index);
    const auto q     = com.frequency(pattern_index, com.frequency.extent(1) - 1);
    const auto minfr = float_type(settings.min_support) / com.data.size();

    if (q_i < minfr || (float_type(1) - q_i) < minfr || q < minfr)
    {
        return true;
    }
    else
        return false;
}

template <typename Trait>
auto decompose_step_one_component(const Composition<Trait>&            com,
                                  const size_t                         comp_index,
                                  NodewiseDecompositionContext<Trait>& ctx,
                                  const DecompositionSettings&         settings,
                                  std::atomic_int&                     label_counter)
{
    constexpr const bool top_level_parallel = true;

    using float_type = typename Trait::float_type;

    std::optional<std::pair<size_t, size_t>> split_on = std::nullopt;
    std::optional<Composition<Trait>>        result   = std::nullopt;

    assert(check_invariant(com));
    assert(com.frequency.extent(0) == com.summary.size());
    assert(comp_index < com.assignment.size());
    assert(comp_index < com.data.num_components());

    thread_local Composition<Trait> next;

#pragma omp parallel for if (top_level_parallel)
    for (size_t jj = 0; jj < com.assignment[comp_index].size(); ++jj)
    {
        const auto j = com.assignment[comp_index][jj];

        if (ctx.has_rejected.find({comp_index, j}) != ctx.has_rejected.end())
            continue;

        if (is_early_reject(com, j, comp_index, settings))
        {
#pragma omp critical
            {
                ctx.has_rejected.insert({comp_index, j});
            }
            continue;
        }

        const auto  label           = label_counter++;
        const auto& split_pattern   = com.summary.point(j);
        const auto  corrected_alpha = float_type(settings.alpha) / ctx.total_number_of_tests;

        next                = com;
        bool is_significant = do_significant_split(
            next, split_pattern, j, comp_index, label, corrected_alpha, settings);

        if (is_significant)
        {
#pragma omp critical
            {
                if (!result ||
                    (result && result->encoding.objective() > next.encoding.objective()))
                {
                    split_on = {j, label};
                    result   = std::move(next);
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
    }

    return result;
}

template <typename Trait, typename CALLBACK = EmptyCallback>
bool decompose_round(Composition<Trait>&                  com,
                     NodewiseDecompositionContext<Trait>& ctx,
                     const DecompositionSettings&         settings,
                     CALLBACK&&                           callback = {})
{
    insert_distinct_nonsingletons(ctx.splitset, com.summary);

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
            const auto k = com.data.num_components();
            com          = std::move(*next);

            callback(std::as_const(com));

            ctx.decompose_next.push_back(comp_index);
            if (k < com.data.num_components())
            {
                ctx.decompose_next.push_back(com.data.num_components() - 1);
            }
        }
    }

    ctx.to_decompose.clear();
    std::swap(ctx.to_decompose, ctx.decompose_next);
    simplify_labels(com.data);
    return !ctx.to_decompose.empty();
}

template <typename Trait, typename PatternMiner, typename Call = EmptyCallback>
void decompose_dataset(Composition<Trait>&                  com,
                       NodewiseDecompositionContext<Trait>& ctx,
                       PatternMiner&&                       miner,
                       DecompositionSettings                settings,
                       Call&&                               report = {})
{
    ctx.to_decompose.resize(com.data.num_components());
    std::iota(ctx.to_decompose.begin(), ctx.to_decompose.end(), 0);

    const auto& max_k = settings.max_cliques;

    report(std::as_const(com));

    while (!ctx.to_decompose.empty())
    {
        const auto k = com.data.num_components();
        if (max_k && k >= *max_k)
            break;

        auto has_next = decompose_round(com, ctx, settings, report);

        miner(com, settings);

        if (!has_next)
        {
            break;
        }
    }
}

template <bool Intermediate_Mining,
          typename Trait,
          typename Miner,
          typename Call = EmptyCallback>
void decompose_maybe_mine(Composition<Trait>&                  com,
                          NodewiseDecompositionContext<Trait>& ctx,
                          Miner&&                              miner,
                          const DecompositionSettings&         cfg,
                          Call&&                               report = {})
{
    if (Intermediate_Mining)
    {
        decompose_dataset(com, ctx, miner, cfg, std::forward<Call>(report));
    }
    else
    {
        decompose_dataset(com, ctx, [](auto&&...) {}, cfg, std::forward<Call>(report));
    }
}

} // namespace disc
} // namespace sd