#pragma once

#include <disc/disc/CharacterizeComponents.hxx>
#include <disc/disc/Composition.hxx>
#include <disc/disc/Decompose.hxx>

namespace sd::disc
{

template <typename Trait, typename PatternsetMiner, typename Call>
void decompse_always_mine(Composition<Trait>&          com,
                          const DecompositionSettings& settings,
                          PatternsetMiner&&            patternset_miner,
                          Call&&                       report)
{
    NodewiseDecompositionContext<Trait> ctx;

    auto start_time = std::chrono::high_resolution_clock::now();

    patternset_miner(com, settings);

    while (true)
    {
        auto k_before = com.data.num_components();

        decompose_maybe_mine<true>(com, ctx, patternset_miner, settings, report);

        if (k_before >= com.data.num_components())
            break;

        if (settings.max_time_total && std::chrono::high_resolution_clock::now() >
                                           start_time + settings.max_time_total.value())
            break;
    }
}

template <typename Trait, typename PatternsetMiner, typename Call>
void decompose_until_summary_converges(Composition<Trait>&          com,
                                       const DecompositionSettings& settings,
                                       PatternsetMiner&&            patternset_miner,
                                       Call&&                       report)
{
    NodewiseDecompositionContext<Trait> ctx;

    auto start_time = std::chrono::high_resolution_clock::now();

    patternset_miner(com, settings);

    while (true)
    {
        auto k_before = com.data.num_components();

        decompose_maybe_mine<false>(com, ctx, patternset_miner, settings, report);

        if (k_before == com.data.num_components())
            break;

        const size_t s_before = com.summary.size();

        patternset_miner(com, settings);

        if (s_before <= com.summary.size())
            break;

        if (settings.max_time_total && std::chrono::high_resolution_clock::now() >
                                           start_time + settings.max_time_total.value())
            break;
    }
}

template <typename Trait, typename PatternsetMiner, typename Call>
void mine_decompose(Composition<Trait>&          com,
                    const DecompositionSettings& settings,
                    PatternsetMiner&&            pm,
                    Call&&                       report)
{
    NodewiseDecompositionContext<Trait> ctx;
    pm(com, settings);
    decompose_maybe_mine<false>(com, ctx, pm, settings, report);
}

template <typename Trait, typename PatternsetMiner, typename Call>
void mine_decompose_mine(Composition<Trait>&          com,
                         const DecompositionSettings& settings,
                         PatternsetMiner&&            pm,
                         Call&&                       report)
{
    mine_decompose(com, settings, pm, report);
    pm(com, settings);
}

} // namespace sd::disc