#pragma once

#include <disc/disc/Composition.hxx>
#include <disc/disc/Decompose.hxx>
#include <disc/disc/SlimCandidateGeneration.hxx>

namespace sd
{
namespace disc
{

template <typename Trait>
auto configure_generator(Composition<Trait> const& com, DecompositionSettings const& settings)
{
    using pattern_type = typename Trait::pattern_type;
    using float_type   = typename Trait::float_type;

    return SlimGenerator<pattern_type, float_type>{
        com.data, com.summary, settings.min_support, settings.max_pattern_size};
}

template <typename Trait, typename Call = EmptyCallback>
Composition<Trait>
mine_decompose(Composition<Trait> com, DecompositionSettings settings, Call&& report = {})
{
    settings.intermediate_mining = false;

    com = discover_patternsets(std::move(com), settings);
    com = decompose_maybe_mine(std::move(com), settings, report);

    return com;
}

template <typename Trait, typename Call = EmptyCallback>
Composition<Trait>
mine_decompose_mine(Composition<Trait> com, DecompositionSettings settings, Call&& report = {})
{
    settings.intermediate_mining = false;

    NodewiseDecompositionContext<Trait> ctx;

    com = discover_patternsets(std::move(com), /*ctx.generator,*/ settings);
    com = decompose_maybe_mine(std::move(com), ctx, settings, report);
    com = discover_patternsets(std::move(com), /*ctx.generator,*/ settings);

    return com;
}

template <typename Trait, typename Call = EmptyCallback>
Composition<Trait> mine_decompose_repeat(Composition<Trait>    com,
                                         DecompositionSettings settings,
                                         Call&&                report = {})
{
    settings.intermediate_mining = false;

    NodewiseDecompositionContext<Trait> ctx;
    // ctx.generator = configure_generator(com, settings);

    auto start_time = std::chrono::high_resolution_clock::now();
    com             = discover_patternsets(std::move(com), /*ctx.generator,*/ settings);
    while (true)
    {
        auto before = com.data.num_components();

        com = decompose_maybe_mine(std::move(com), ctx, settings, report);
        com = discover_patternsets(std::move(com), /*ctx.generator,*/ settings);

        if (before >= com.data.num_components())
            break;
        if (settings.max_time_total && std::chrono::high_resolution_clock::now() >
                                           start_time + settings.max_time_total.value())
            break;
        // ctx = NodewiseDecompositionContext<S, T>();
    }
    return com;
}

template <typename Trait, typename Call = EmptyCallback>
Composition<Trait> mine_split_round_repeat(Composition<Trait>    com,
                                           DecompositionSettings settings,
                                           Call&&                report = {})
{
    settings.intermediate_mining = true;

    NodewiseDecompositionContext<Trait> ctx;
    // ctx.generator = configure_generator(com, settings);
    auto start_time = std::chrono::high_resolution_clock::now();
    com             = discover_patternsets(std::move(com), /*ctx.generator,*/ settings);
    while (true)
    {
        auto before = com.data.num_components();

        com = decompose_maybe_mine(std::move(com), ctx, settings, report);

        if (before >= com.data.num_components())
            break;
        if (settings.max_time_total && std::chrono::high_resolution_clock::now() >
                                           start_time + settings.max_time_total.value())
            break;
        // ctx = NodewiseDecompositionContext<S, T>();
    }

    return com;
}

// alias
template <typename Trait, typename Call = EmptyCallback>
auto discover_composition(Composition<Trait>    com,
                          DecompositionSettings settings,
                          Call&&                report = {})

{
    return mine_split_round_repeat(
        std::move(com), std::move(settings), std::forward<Call>(report));
}

} // namespace disc
} // namespace sd