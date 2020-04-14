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

double chi_squared_value(size_t degrees_of_freedom, double pvalue = 0.05)
{
    namespace bm = boost::math;
    bm::chi_squared chi(degrees_of_freedom);
    return bm::quantile(bm::complement(chi, pvalue));
}

template <typename T>
T chi_squared_pvalue(size_t degrees_of_freedom, const T& p)
{
    namespace bm = boost::math;
    bm::chi_squared chi(degrees_of_freedom);
    return bm::cdf(chi, p);
}

template <typename T>
std::pair<T, T>
chi_squared_confidence_interval(size_t degrees_of_freedom, const T& sd, const T& alpha)
{
    namespace bm = boost::math;
    using namespace boost::math;

    bm::chi_squared chi(degrees_of_freedom - 1);

    T lower_limit =
        sqrt((degrees_of_freedom - 1) * sd * sd / bm::quantile(bm::complement(chi, alpha / 2)));
    T upper_limit = sqrt((degrees_of_freedom - 1) * sd * sd / bm::quantile(chi, alpha / 2));

    return {lower_limit, upper_limit};
}

template <typename T>
auto js_divergence_sd(sd::slice<T> first, sd::slice<T> second, size_t ignored_index)
{
    IncrementalDescription<T> stat;
    for (size_t i = 0; i < first.size(); ++i)
    {
        if (i != ignored_index)
            stat += js1(first[i], second[i]);
    }
    return stat;
}

template <typename Trait>
auto js_fr_divergence_sd(const Composition<Trait>& c,
                         size_t                    first_index,
                         size_t                    second_index,
                         std::optional<size_t>     x_index = std::nullopt)
{
    using float_t = typename Composition<Trait>::float_type;

    IncrementalDescription<float_t> stat;

    for (size_t i = 0; i < c.summary.size(); ++i)
    {
        if (x_index && i == *x_index)
            continue;
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

    auto [js, sd] = js_fr_divergence_sd(c, first_index, second_index, x_index);

    if (std::isnan(js) || std::isinf(js))
    {
        return false;
    }

    const size_t n    = c.data.subset(first_index).size() + c.data.subset(second_index).size();
    const size_t m    = c.summary.size() - x_index.has_value();
    const auto   pval = chi_squared_pvalue(m - 1, 2 * n * js);
    const auto   ci   = chi_squared_confidence_interval(n, sd, alpha);

    const bool is_significant = pval >= (T(1) - alpha) && (ci.first <= sd && sd <= ci.second);

    return is_significant;
}

} // namespace sd::disc