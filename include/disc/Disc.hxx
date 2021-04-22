#pragma once

#include <desc/CharacterizeComponents.hxx>
#include <desc/Composition.hxx>
#include <disc/CharacterizeSplit.hxx>
#include <disc/DataAssignment.hxx>
#include <disc/Settings.hxx>
#include <disc/TestDivergence.hxx>
#include <container/random-access-set.hxx>

namespace sd
{
namespace disc
{

template <typename Trait, typename P>
void split_data(Composition<Trait>& com, const P& x, size_t label)
{
    for (auto [y, t, _] : com.data)
    {
        if (is_subset(x, t)) y = label;
    }
    com.data.group_by_label();
}
template <typename Trait, typename P>
void split_component(Composition<Trait>& com, size_t index, const P& x, size_t label)
{
    for (auto [y, t, _] : com.data.subset(index))
    {
        if (is_subset(x, t)) y = label;
    }
    com.data.group_by_label();
}
/*
template <typename Trait, typename P>
void just_split(Composition<Trait>& com, const P& x, size_t index, size_t label)
{
    auto set = com.data.subset(index);
    for (auto [y, t, _] : set)
    {
        if (is_subset(x, t)) y = label;
    }
    com.data.group_by_label();
}

template <typename Trait>
void undo_split_data(Composition<Trait>& com, size_t label_before, size_t label_after)
{
    for (auto [y, t, _] : com.data)
    {
        if (y == label_before) y = label_after;
    }
    com.data.group_by_label();
}

template <typename Trait, typename P, typename Interface = DefaultAssignment>
bool split_and_characterize(Composition<Trait>& com,
                            const P&            x,
                            size_t              index,
                            size_t              label_after,
                            const DiscConfig&   cfg,
                            Interface&&         f = {})
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
        characterize_components(com, cfg, std::forward<Interface>(f));
        com.encoding = encode(com, c.use_bicfg);
        // characterize_split(com, {index, com.data.num_components() - 1}, cfg);
        return true;
    }
}

template <typename Trait, typename P, typename Interface = DefaultAssignment>
bool split_and_reassign(Composition<Trait>& com,
                        const P&            x,
                        size_t              index,
                        size_t              label,
                        const DiscConfig&   cfg,
                        Interface&&         f = {})

{
    auto found = split_and_characterize(com, x, index, label, cfg, f);
    if (found) { reassign_components(com, cfg, 1, f); }
    return found;
}

template <typename Trait,
          typename P,
          typename float_type,
          typename Interface = DefaultAssignment>
bool split_test_reassign(Composition<Trait>&       com,
                         const P&                  x,
                         size_t                    index,
                         size_t                    label,
                         std::pair<size_t, size_t> component,
                         size_t                    x_index,
                         const float_type          alpha,
                         const DiscConfig&         cfg,
                         Interface&&               f = {})

{

    bool has_split = split_and_characterize(com, x, index, label, cfg, f);

    if (!has_split ||
        !test_stat_divergence(com, alpha, component.first, component.second, x_index))
    {
        return false;
    }
    else
    {
        reassign_components(com, cfg, 1, f);
        return true;
    }
}

template <typename Trait, typename P, typename F, typename Interface = DefaultAssignment>
bool do_split(Composition<Trait>& com,
              const P&            x,
              size_t              x_index,
              size_t              index,
              size_t              label,
              const F&            alpha,
              const DiscConfig&   cfg,
              Interface&&         f = {})
{
    assert(check_invariant(com));
    assert(x_index < com.summary.size());
    assert(index < com.data.num_components());

    if (cfg.test_divergence)
    {
        return split_test_reassign(
            com, x, index, label, {index, com.data.num_components()}, x_index, alpha, cfg, f);
    }
    else
    {
        return split_and_reassign(com, x, index, label, cfg, f);
    }
}

template <typename Trait, typename P, typename F, typename Interface = DefaultAssignment>
bool do_significant_split(Composition<Trait>& com,
                          const P&            x,
                          size_t              x_index,
                          size_t              index,
                          size_t              label,
                          const F&            alpha,
                          const DiscConfig&   cfg,
                          Interface&&         f = {})
{

    const auto enc_before = com.encoding;
    const auto k          = com.data.num_components();
    const bool has_split =
        do_split(com, x, x_index, index, label, alpha, cfg, std::forward<Interface>(f));

    bool is_significant = com.encoding.objective() < enc_before.objective();

    if (cfg.test_divergence && is_significant && has_split && k < com.data.num_components())
    {
        is_significant &=
            test_stat_divergence(com, alpha, index, com.data.num_components() - 1, x_index);
    }

    return is_significant;
}
*/
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

    if (q_i < minfr || (float_type(1) - q_i) < minfr || q < minfr) { return true; }
    else
    {
        return false;
    }
}

template <typename Trait, typename Interface = DefaultAssignment>
bool disc_decomp_step(Composition<Trait>&                                 c,
                      const DiscConfig&                                   cfg,
                      EncodingLength<typename Trait::float_type>&         encoding,
                      andres::RandomAccessSet<std::pair<size_t, size_t>>& rejected,
                      Interface&&                                         f = {})
{

    using float_type = typename Trait::float_type;

    auto best          = c;
    auto best_encoding = encoding;

    std::atomic_int label       = c.data.num_components() + 1;
    float_type      calpha      = cfg.alpha / (c.data.num_components() * c.summary.size());
    const auto      rejected_ro = rejected;

#pragma omp parallel for collapse(2) schedule(dynamic, 1) shared(best) shared(rejected)        \
    firstprivate(rejected_ro)
    for (size_t i = 0; i < c.data.num_components(); ++i)
    {
        for (size_t j = 0; j < c.summary.size(); ++j)
        {
            // !c.assignment[i].contains(j) &&
            if (c.confidence(j, i) <= 0 || is_early_reject(c, j, i, cfg) ||
                rejected_ro.find({i, j}) != rejected_ro.end())
            {
                continue;
            }

            auto next = c; // expensive

            const auto& x = c.summary.point(j);
            split_component(next, i, x, label++);
            if (next.data.num_components() <= c.data.num_components()) { continue; }
            characterize_components(next, cfg, f);
            if (!test_stat_divergence(next, calpha, i, next.data.num_components() - 1, j))
            {
                continue;
            }

            reassign_components(next, cfg, 2, f);

            auto next_encoding = encode(next, cfg.use_bic);

            if (!test_stat_divergence(next, calpha, i, next.data.num_components() - 1, j))
            {
                continue;
            }

            bool sig = next_encoding.objective() < encoding.objective();

            // bool        sig = do_significant_split(next, x, j, i, label++, calpha, cfg, f);

            if (sig)
            {
#pragma omp critical
                {
                    if (best_encoding.objective() > next_encoding.objective())
                    {
                        best          = std::move(next);
                        best_encoding = next_encoding;
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

    bool is_better = best_encoding.objective() < encoding.objective();

    if (is_better)
    {
        simplify_labels(best.data);
        c        = std::move(best);
        encoding = best_encoding;
    }

    return is_better;
}

template <typename Trait, typename Info = EmptyCallback, typename Interface = DefaultAssignment>
auto disc_decomp(Composition<Trait>& c,
                 const DiscConfig&   cfg,
                 Info&&              info = {},
                 Interface&&         f    = {})
{
    andres::RandomAccessSet<std::pair<size_t, size_t>> rejected;
    EncodingLength<typename Trait::float_type> encoding;

    while (disc_decomp_step(c, cfg, encoding, rejected, f)) { info(std::as_const(c)); }

    return encoding;
}

template <typename Trait,
          typename PatternsetMiner,
          typename Call,
          typename Interface = DefaultAssignment>
auto discover_components(Composition<Trait>& c,
                         const DiscConfig&   cfg,
                         PatternsetMiner&&   patternset_miner,
                         Call&&              report,
                         Interface&&         f = {})
{
    using clk = std::chrono::high_resolution_clock;

    const auto& tt = cfg.max_time;
    const auto  st = clk::now();

    patternset_miner(c, cfg);
    auto encoding = encode(c, cfg.use_bic);

    andres::RandomAccessSet<std::pair<size_t, size_t>> rejected;

    while (true)
    {
        const size_t k_before = c.data.num_components();
        const size_t s_before = c.summary.size();

        while (disc_decomp_step(c, cfg, encoding, rejected, f))
        {
            report(std::as_const(c));

            if (tt && clk::now() > st + *tt) return encoding;
        }

        patternset_miner(c, cfg);
        encoding = encode(c, cfg.use_bic);

        if (tt && clk::now() > st + *tt) break;

        if (k_before == c.data.num_components()) break;

        if (s_before <= c.summary.size()) break;
    }
    return encoding;
}

template <typename Trait,
          typename PatternsetMiner,
          typename Call,
          typename Interface = DefaultAssignment>
void mine_decompose(Composition<Trait>& com,
                    const DiscConfig&   cfg,
                    PatternsetMiner&&   pm,
                    Call&&              report,
                    Interface&&         f = {})
{
    pm(com, cfg);
    disc_decomp(com, cfg, report, f);
}

template <typename Trait,
          typename PatternsetMiner,
          typename Call,
          typename Interface = DefaultAssignment>
void mine_decompose_mine(Composition<Trait>& com,
                         const DiscConfig&   cfg,
                         PatternsetMiner&&   pm,
                         Call&&              report,
                         Interface&&         f = {})
{
    mine_decompose(com, cfg, pm, report, f);
    pm(com, cfg);
}

} // namespace disc
} // namespace sd