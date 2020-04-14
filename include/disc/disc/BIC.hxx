
#pragma once

#include <disc/disc/Composition.hxx>
#include <disc/disc/PatternsetResult.hxx>
#include <disc/distribution/Distribution.hxx>
#include <disc/storage/Dataset.hxx>
#include <disc/utilities/Support.hxx>
#include <disc/utilities/UniversalIntEncoding.hxx>

#include <math/nchoosek.hxx>

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace sd::disc::bic
{

template <typename Trait>
auto encode_model_bic(const Composition<Trait>& c)
{
    // n is constant, add or remove if from score to your desire.
    // times 2 because we consider theta and assignment of patterns as dof

    const auto n = c.data.size();
    const auto s = c.summary.size();
    const auto k = c.data.num_components();
    const auto d = c.data.dim;
    const auto m = s - d;

    using float_type    = typename Trait::float_type;
    const float_type df = k * (2 * m + d) + n;

    return std::log2(n) * df / 2.;
}

template <typename Trait>
auto encode_model_bic(const PatternsetResult<Trait>& c)
{
    return std::log2(c.data.size()) * c.summary.size() / 2.;
}

} // namespace sd::disc::bic