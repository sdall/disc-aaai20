#pragma once

#include <disc/desc/CharacterizeComponents.hxx>
#include <disc/desc/Composition.hxx>
#include <disc/disc/CharacterizeSplit.hxx>
#include <disc/disc/DataAssignment.hxx>
#include <disc/disc/Settings.hxx>
#include <disc/disc/TestDivergence.hxx>

#include <containers/random-access-set.hxx>

namespace sd
{
namespace disc
{

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
bool split_and_characterize(Composition<Trait>& com,
                            const P&            x,
                            size_t              index,
                            size_t              label_after,
                            const DiscConfig&   cfg)
{

    if (index > com.data.num_components() || com.data.subset(index).size() == 0)
    {
        return false;
    }

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
        characterize_components(com, cfg);
        // characterize_split(com, {index, com.data.num_components() - 1}, cfg);
        return true;
    }
}

template <typename Trait, typename P>
bool split_and_reassign(
    Composition<Trait>& com, const P& x, size_t index, size_t label, const DiscConfig& cfg)

{
    auto found = split_and_characterize(com, x, index, label, cfg);
    if (found)
    {
        reassign_components(com, cfg);
    }
    return found;
}

template <typename Trait, typename P, typename float_type>
bool split_test_reassign(Composition<Trait>&       com,
                         const P&                  x,
                         size_t                    index,
                         size_t                    label,
                         std::pair<size_t, size_t> component,
                         size_t                    x_index,
                         const float_type          alpha,
                         const DiscConfig&         cfg)

{

    bool has_split = split_and_characterize(com, x, index, label, cfg);

    if (!has_split ||
        !test_stat_divergence(com, alpha, component.first, component.second, x_index))
    {
        return false;
    }
    else
    {
        reassign_components(com, cfg);
        return true;
    }
}

template <typename Trait, typename P, typename F>
bool do_split(Composition<Trait>& com,
              const P&            x,
              size_t              x_index,
              size_t              index,
              size_t              label,
              const F&            alpha,
              const DiscConfig&   cfg)
{
    assert(check_invariant(com));
    assert(x_index < com.summary.size());
    assert(index < com.data.num_components());

    if (cfg.test_divergence)
    {
        return split_test_reassign(
            com, x, index, label, {index, com.data.num_components()}, x_index, alpha, cfg);
    }
    else
    {
        return split_and_reassign(com, x, index, label, cfg);
    }
}

template <typename Trait, typename P, typename F>
bool do_significant_split(Composition<Trait>& com,
                          const P&            x,
                          size_t              x_index,
                          size_t              index,
                          size_t              label,
                          const F&            alpha,
                          const DiscConfig&   cfg)
{

    const auto enc_before = com.encoding;
    const auto k          = com.data.num_components();
    const bool has_split  = do_split(com, x, x_index, index, label, alpha, cfg);

    bool is_significant = com.encoding.objective() < enc_before.objective();

    if (cfg.test_divergence && is_significant && has_split && k < com.data.num_components())
    {
        is_significant &=
            test_stat_divergence(com, alpha, index, com.data.num_components() - 1, x_index);
    }

    return is_significant;
}

template <typename Trait>
bool is_early_reject(const Composition<Trait>& com,
                     size_t                    pattern_index,
                     size_t                    comp_index,
                     const DiscConfig&         cfg)
{
    using float_type = typename Trait::float_type;

    assert(pattern_index < com.summary.size());
    assert(com.frequency.shape().contains({pattern_index, comp_index}));
    assert(com.frequency.shape().contains({pattern_index, com.frequency.extent(1) - 1}));

    const auto q_i   = com.frequency(pattern_index, comp_index);
    const auto q     = com.frequency(pattern_index, com.frequency.extent(1) - 1);
    const auto minfr = float_type(cfg.min_support) / com.data.size();

    if (q_i < minfr || (float_type(1) - q_i) < minfr || q < minfr)
    {
        return true;
    }
    else
    {
        return false;
    }
}

template <typename Trait>
bool disc_decomp_step(Composition<Trait>&                                 c,
                      const DiscConfig&                                   cfg,
                      andres::RandomAccessSet<std::pair<size_t, size_t>>& rejected)
{

    using float_type = typename Trait::float_type;

    auto            best        = c;
    std::atomic_int label       = c.data.num_components() + 1;
    float_type      calpha      = cfg.alpha / (c.data.num_components() * c.summary.size());
    const auto      rejected_ro = rejected;

#pragma omp parallel for collapse(2) schedule(dynamic, 1) shared(best) shared(rejected)        \
    firstprivate(rejected_ro)
    for (size_t i = 0; i < c.data.num_components(); ++i)
    {
        for (size_t j = 0; j < c.summary.size(); ++j)
        {

            if (!c.assignment[i].contains(j) || is_early_reject(c, j, i, cfg) ||
                rejected_ro.find({i, j}) != rejected_ro.end())
            {
                continue;
            }

            auto next = c; // expensive

            const auto& x   = c.summary.point(j);
            bool        sig = do_significant_split(next, x, j, i, label++, calpha, cfg);

            if (sig)
            {
#pragma omp critical
                {
                    if (best.encoding.objective() > next.encoding.objective())
                    {
                        best = std::move(next);
                    }
                }
            }
            else
            {
#pragma omp critical
                {
                    rejected.insert({i, j});
                }
            }
        }
    }

    bool is_better = best.encoding.objective() < c.encoding.objective();

    if (is_better)
    {
        c = std::move(best);
        simplify_labels(c.data);
    }

    return is_better;
}

template <typename Trait, typename Info = EmptyCallback>
void disc_decomp(Composition<Trait>& c, const DiscConfig& cfg, Info&& info = {})
{
    andres::RandomAccessSet<std::pair<size_t, size_t>> rejected;

    while (disc_decomp_step(c, cfg, rejected))
    {
        info(std::as_const(c));
    }
}

template <typename Trait, typename PatternsetMiner, typename Call>
void discover_components(Composition<Trait>& c,
                         const DiscConfig&   cfg,
                         PatternsetMiner&&   patternset_miner,
                         Call&&              report)
{
    using clk = std::chrono::high_resolution_clock;

    const auto& tt = cfg.max_time_total;
    const auto  st = clk::now();

    patternset_miner(c, cfg);

    andres::RandomAccessSet<std::pair<size_t, size_t>> rejected;

    while (true)
    {
        const size_t k_before = c.data.num_components();
        const size_t s_before = c.summary.size();

        while (disc_decomp_step(c, cfg, rejected))
        {
            report(std::as_const(c));

            if (tt && clk::now() > st + *tt)
                return;
        }

        patternset_miner(c, cfg);

        if (tt && clk::now() > st + *tt)
            break;

        if (k_before == c.data.num_components())
            break;

        if (s_before <= c.summary.size())
            break;
    }
}

template <typename Trait, typename PatternsetMiner, typename Call>
void mine_decompose(Composition<Trait>& com,
                    const DiscConfig&   cfg,
                    PatternsetMiner&&   pm,
                    Call&&              report)
{
    pm(com, cfg);
    disc_decomp(com, cfg, report);
}

template <typename Trait, typename PatternsetMiner, typename Call>
void mine_decompose_mine(Composition<Trait>& com,
                         const DiscConfig&   cfg,
                         PatternsetMiner&&   pm,
                         Call&&              report)
{
    mine_decompose(com, cfg, pm, report);
    pm(com, cfg);
}

} // namespace disc
} // namespace sd