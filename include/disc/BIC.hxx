
#pragma once

#include <desc/Component.hxx>
#include <desc/Composition.hxx>
#include <desc/Support.hxx>
#include <desc/distribution/Distribution.hxx>
#include <desc/storage/Dataset.hxx>

#include <math/nchoosek.hxx>

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace sd::disc::bic
{

template <typename Trait>
auto encode_model_bic_aaai20(const Composition<Trait>& c) -> typename Trait::float_type
{
    // n is constant, add or remove if from score to your desire.
    // times 2 because we consider theta and assignment of patterns as dof

    using std::log2;

    const auto n = c.data.size();
    const auto s = c.summary.size();
    const auto k = c.data.num_components();
    const auto d = c.data.dim;
    const auto m = s - d;

    const auto df = k * (2 * m + d) + n;
    // const float_type df = k * m * 2; // + d + n const

    return log2(n) * df / 2.;
}

template <typename Trait>
auto encode_model_bic_new(const Composition<Trait>& c)
{
    const auto n  = c.data.size();
    const auto k  = c.data.num_components();
    const auto d  = c.data.dim;
    const auto s  = c.summary.size();
    const auto m  = s - d;
    const auto tr = trace(c.assignment) - d * k; // singletons are free

    using float_type    = typename Trait::float_type;
    const float_type df = tr + m * k; // + n

    using std::log2;

    return log2(n) * df / 2.;
}

template <typename Trait>
auto encode_model_bic(const Composition<Trait>& c) -> typename Trait::float_type
{
    // return encode_model_bic_aaai20(c);
    using std::log2;
    const auto k = c.data.num_components();
    const auto m = c.summary.size() - c.data.dim;
    return log2(c.data.size()) * m * k / 2;
}

template <typename Trait>
auto encode_model_bic(const Component<Trait>& c)
{
    using std::log2;
    return log2(c.data.size()) * (c.summary.size() - c.data.dim) / 2.;
}

} // namespace sd::disc::bic