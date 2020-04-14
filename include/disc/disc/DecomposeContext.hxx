#pragma once

#include <containers/random-access-set.hxx>
#include <disc/disc/SlimCandidateGeneration.hxx>
#include <disc/storage/Dataset.hxx>

#include <cstddef>
#include <vector>

namespace sd
{
namespace disc
{

template <typename Trait>
struct NodewiseDecompositionContext
{

    using trait        = Trait;
    using float_type   = typename Trait::float_type;
    using pattern_type = typename Trait::pattern_type;

    std::vector<size_t>                                to_decompose;
    std::vector<size_t>                                decompose_next;
    andres::RandomAccessSet<std::pair<size_t, size_t>> has_rejected;
    sd::disc::LabeledDataset<float_type, pattern_type> splitset;

    std::optional<SlimGenerator<pattern_type, float_type>> generator;

    size_t total_number_of_tests = 1;
};

} // namespace disc
} // namespace sd