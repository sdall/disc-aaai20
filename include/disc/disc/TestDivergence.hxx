#pragma once

#include <disc/interfaces/BoostMultiprecision.hxx>
#include <math/IncrementalStatistics.hxx>

#include <boost/math/distributions.hpp> // chi-square

namespace sd::disc
{

template <typename T>
T kl1(T q, T p) noexcept
{
    return q < T(1e-15) ? T(0) : T(q * std::log2(q / p));
}

template <typename T>
T js1(const T& p, const T& q) noexcept
{
    auto m = (p + q) / T(2);
    return m < T(1e-15) ? T(0) : (kl1(p, m) + kl1(q, m)) / T(2);
}

double chi_squared_value(double pvalue, size_t dof)
{
    namespace bm = boost::math;
    bm::chi_squared chi(dof);
    return bm::quantile(bm::complement(chi, pvalue));
}

template <typename T>
T chi_squared_pvalue(const T& p, size_t dof)
{
    namespace bm = boost::math;
    bm::chi_squared chi(dof);
    return bm::cdf(chi, p);
}

template <typename T>
std::pair<T, T> chi_squared_confidence_interval(const T& sd, const T& alpha, size_t dof)
{
    namespace bm = boost::math;
    using namespace boost::math;

    bm::chi_squared chi(dof);

    auto dofsd2 = dof * sd * sd;

    T lower_limit = sqrt(dofsd2 / bm::quantile(bm::complement(chi, alpha / 2)));
    T upper_limit = sqrt(dofsd2 / bm::quantile(chi, alpha / 2));

    return {lower_limit, upper_limit};
}

// to test if p is differently distributed from p', it is sufficient to compare \poly and \poly'
template <typename Trait>
auto js_divergence(const Composition<Trait>& c,
                   size_t                    first_index,
                   size_t                    second_index,
                   std::optional<size_t>     x_index = std::nullopt)
{
    using float_t = typename Composition<Trait>::float_type;

    IncrementalDescription<float_t> stat;

    for (size_t i = 0; i < c.summary.size(); ++i)
    {
        if (x_index && i != *x_index)
            stat += js1(c.frequency(i, first_index), c.frequency(i, second_index));
    }

    return std::make_pair(stat.sum(), stat.sd());
}

template <typename Trait, typename T>
auto test_stat_divergence(const Composition<Trait>& c,
                          const T&                  alpha,
                          size_t                    first_index,
                          size_t                    second_index,
                          std::optional<size_t>     x_index = std::nullopt) -> bool
{
    if (c.data.num_components() <= first_index)
    {
        return false;
    }
    if (c.data.num_components() <= second_index)
    {
        return false;
    }
    if (first_index == second_index)
    {
        return false;
    }

    auto [js, sd] = js_divergence(c, first_index, second_index, x_index);

    if (std::isnan(js) || std::isinf(js))
    {
        return false;
    }

    const size_t n    = c.data.subset(first_index).size() + c.data.subset(second_index).size();
    const size_t dof  = c.summary.size() - x_index.has_value() - 1;
    const auto   div  = 2 * n * js;
    const auto   pval = chi_squared_pvalue(div, dof);
    const auto   ci   = chi_squared_confidence_interval(sd, alpha, dof);

    return pval >= (T(1) - alpha) && (ci.first <= sd && sd <= ci.second);
}

} // namespace sd::disc