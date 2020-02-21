#pragma once

#include <disc/disc/DiscoverPatternsets.hxx>
#include <disc/disc/Encoding.hxx>
#include <disc/disc/Frequencies.hxx>
#include <disc/disc/Settings.hxx>
#include <disc/interfaces/BoostMultiprecision.hxx>
#include <math/RunningStatistics.hxx>

#include <boost/math/distributions.hpp> // chi-square

namespace sd
{
namespace disc
{

template <typename T = double>
constexpr T g_2x2(std::size_t a, std::size_t b, std::size_t c, std::size_t d) noexcept
{
    if (a == 0 || b == 0)
        return false;

    const auto e1 = c * T(a + b) / (c + d);
    const auto e2 = d * T(a + b) / (c + d);
    const auto g2 = 2 * (a * log(T(a) / e1) + b * log(T(b) / e2));

    return g2;
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
T chi_squared_confidence_width(size_t degrees_of_freedom, const T& sd, const T& alpha)
{
    auto [l, u] = chi_squared_confidence_interval(degrees_of_freedom, sd, alpha);
    return (u - l);
}

template <typename T>
constexpr T ks_test_threshold_factor(const T& alpha)
{
    return sqrt(-T(0.5) * log(alpha));
}

template <typename T>
T menendez_js1(const T& p, const T& q, const T& a = T(0.9464522)) noexcept
{
    const auto m  = a * p + (T(1) - a) * q;
    const auto w0 = T(1) / (T(1) - a);
    const auto w1 = T(1) / a;
    return m == 0 ? 0 : (w0 * kl1(p, m) + w1 * kl1(q, m));
}

template <typename T>
struct SignificanceStatistic
{
    bool is_significant = false;
    T    pvalue         = 0;
    // T               js               = 0;
    // T               js_stat          = 0;
    // T               ci_width         = -1;
    // std::pair<T, T> confidence_interval;
};

template <typename Trait>
auto js_patternset_divergence(const Composition<Trait>& c,
                              size_t                    first_index,
                              size_t                    second_index)
{
    using float_type = typename Trait::float_type;

    float_type js = 0;
    for (const auto& [x, y] : c.summary)
    {
        const auto p_0 = c.models[first_index].expected_frequency(y);
        const auto p_1 = c.models[second_index].expected_frequency(y);
        js += js1(p_0, p_1);
    }
    return js;
}

template <typename Trait>
auto js_patternset_divergence_normalized(const Composition<Trait>& c,
                                         size_t                    first_index,
                                         size_t                    second_index)
{
    using float_type = typename Trait::float_type;

    sd::small_vector<std::pair<float_type, float_type>, 128> p;
    p.reserve(c.summary.size());
    std::array<float_type, 2> sum{{0, 0}};

    float_type js = 0;
    for (const auto& [x, y] : c.summary)
    {
        const auto p_0 = c.models[first_index].expected_frequency(y);
        const auto p_1 = c.models[second_index].expected_frequency(y);
        p.emplace_back(p_0, p_1);
        sum[0] += p_0;
        sum[1] += p_1;
    }

    for (size_t i = 0; i < p.size(); ++i)
    {
        js += js1(p[i].first / sum[0], p[i].second / sum[1]);
    }
    return js;
    // assert(p.size() == n);
}

template <typename Trait>
auto js_fr_divergence_normalized(const Composition<Trait>& c,
                                 size_t                    first_index,
                                 size_t                    second_index,
                                 nonstd::optional<size_t>  x_index = nonstd::nullopt) ->
    typename Trait::float_type
{
    using float_type = typename Trait::float_type;
    std::array<float_type, 2> sum{{0, 0}};
    for (size_t i = 0; i < c.summary.size(); ++i)
    {
        sum[0] += c.frequency(i, first_index);
        sum[1] += c.frequency(i, second_index);
    }

    assert(sum[0] != 0 && sum[1] != 0);

    float_type js = 0;
    for (size_t i = 0; i < c.summary.size(); ++i)
    {
        js += js1(c.frequency(i, first_index) / sum[0], c.frequency(i, second_index) / sum[1]);
    }

    if (x_index)
    {
        js -= js1(c.frequency(*x_index, first_index) / sum[0],
                  c.frequency(*x_index, second_index) / sum[1]);
    }

    return js;
}

template <typename Trait>
auto js_fr_divergence(const Composition<Trait>& c,
                      size_t                    first_index,
                      size_t                    second_index,
                      nonstd::optional<size_t>  x_index = nonstd::nullopt)
{
    typename Composition<Trait>::float_type js = 0;
    for (size_t i = 0; i < c.summary.size(); ++i)
    {
        js += js1(c.frequency(i, first_index), c.frequency(i, second_index));
    }

    if (x_index)
    {
        js -= js1(c.frequency(*x_index, first_index), c.frequency(*x_index, second_index));
    }

    return js;
}

template <typename Trait>
auto js_fr_divergence_sd(const Composition<Trait>& c,
                         size_t                    first_index,
                         size_t                    second_index,
                         nonstd::optional<size_t>  x_index = nonstd::nullopt)
{
    using float_t = typename Composition<Trait>::float_type;

    RunningDescription<float_t> stat;

    for (size_t i = 0; i < c.summary.size(); ++i)
    {
        if (x_index && i == *x_index)
            continue;
        stat += js1(c.frequency(i, first_index), c.frequency(i, second_index));
    }

    return std::make_pair(stat.sum(), stat.sd());
}

template <typename Trait>
auto ks_fr_divergence(const Composition<Trait>& c,
                      size_t                    first_index,
                      size_t                    second_index,
                      nonstd::optional<size_t>  x_index = nonstd::nullopt) ->
    typename Trait::float_type
{
    using float_type = typename Trait::float_type;
    std::array<float_type, 2> sum{{0, 0}};
    for (size_t i = 0; i < c.summary.size(); ++i)
    {
        if (x_index && i == *x_index)
            continue;

        sum[0] += c.frequency(i, first_index);
        sum[1] += c.frequency(i, second_index);
    }
    assert(sum[0] != 0 && sum[1] != 0);

    float_type cdf0 = 0;
    float_type cdf1 = 0;
    float_type ks   = 0;
    for (size_t i = 0; i < c.summary.size(); ++i)
    {
        if (x_index && i == *x_index)
            continue;

        cdf0 += c.frequency(i, first_index) / sum[0];
        cdf1 += c.frequency(i, second_index) / sum[1];

        ks = std::max(ks, std::abs(cdf0 - cdf1));
    }

    return ks;
}

template <typename Trait>
auto js_data_divergence(const Composition<Trait>& c, size_t first_index, size_t second_index)
{
    typename Trait::float_type js = 0;
    for (const auto& x : c.data.subset(first_index))
    {
        const auto p_0 = c.models[first_index].expected_generalized_frequency(point(x));
        const auto p_1 = c.models[second_index].expected_generalized_frequency(point(x));
        js += js1(p_0, p_1);
    }
    for (const auto& x : c.data.subset(second_index))
    {
        const auto p_0 = c.models[first_index].expected_generalized_frequency(point(x));
        const auto p_1 = c.models[second_index].expected_generalized_frequency(point(x));
        js += js1(p_0, p_1);
    }
    return js;
}

template <typename Trait, typename T>
auto significance_statistics(const Composition<Trait>& c,
                             const T&                  alpha,
                             size_t                    first_index,
                             size_t                    second_index,
                             nonstd::optional<size_t>  x_index = nonstd::nullopt)
    -> SignificanceStatistic<typename Trait::float_type>
{
    if (c.data.num_components() <= first_index)
    {
        return {};
    }
    if (c.data.num_components() <= second_index)
    {
        return {};
    }
    if (first_index == second_index)
    {
        return {};
    }

#if 0
    // const auto n              = c.summary.size() - x_index.has_value();
    // const auto ks             = ks_fr_divergence(c, first_index, second_index, x_index);
    // const auto c_alpha        = ks_test_threshold_factor(alpha);
    // const bool is_significant = ks < c_alpha * std::sqrt((n + n) / n * n);
    // return {is_significant, ks};
#else

    auto [js, sd] = js_fr_divergence_sd(c, first_index, second_index, x_index);

    if (std::isnan(js) || std::isinf(js))
    {
        return {};
    }

    // const size_t n              = c.summary.size();
    const size_t n    = c.data.subset(first_index).size() + c.data.subset(second_index).size();
    const size_t m    = c.summary.size() - x_index.has_value();
    const auto   pval = chi_squared_pvalue(m - 1, 2 * n * js);
    const auto   ci   = chi_squared_confidence_interval(n, sd, alpha);

    const bool is_significant = pval >= (T(1) - alpha) && (ci.first <= sd && sd <= ci.second);

    return {is_significant, pval};
#endif
}

template <typename Trait>
auto js_fr_divergence_pattern(const Composition<Trait>& c, size_t s_index, size_t x_index) ->
    typename Trait::float_type
{
    using float_type = typename Trait::float_type;

    const auto& pr = c.models[s_index];

    const auto p = pr.expected_generalized_frequency(c.summary.point(x_index));

    itemset<typename Trait::pattern_type> joined;

    float_type js = 0;
    for (size_t i = 0; i < c.summary.size(); ++i)
    {
        if (i == x_index)
            continue;

        intersection(c.summary.point(i), c.summary.point(x_index), joined);
        const auto pp = pr.expected_frequency(joined);

        js += js1(p, pp);
    }

    return js;
}

template <typename Trait, typename T>
auto significance_statistics_pattern(const Composition<Trait>& c,
                                     const T&                  alpha,
                                     size_t                    component_index,
                                     size_t                    x_index)
    -> SignificanceStatistic<typename Trait::float_type>
{

    const size_t n = c.data.subset(component_index).size();
    const size_t m = c.summary.size() - 1;

    const auto js   = js_fr_divergence_pattern(c, component_index, x_index);
    const auto pval = chi_squared_pvalue(m - 1, 2 * n * js);

    return {pval >= (T(1) - alpha), pval};
}

template <typename Trait, typename T>
auto is_significant(const Composition<Trait>& c,
                    const T&                  alpha,
                    size_t                    first_index,
                    size_t                    second_index,
                    nonstd::optional<size_t>  x_index = nonstd::nullopt)
{
    return significance_statistics(c, alpha, first_index, second_index, x_index).is_significant;
}

template <typename S>
double purity(const PartitionedData<S>&  data,
              const std::vector<size_t>& classes,
              const std::vector<size_t>& class_labels)
{
    assert(classes.size() == data.size());

    double              aggregated_purity = 0;
    std::vector<size_t> h(class_labels.size());
    for (size_t k = 0; k < data.num_components(); ++k)
    {
        auto component = data.subset(k);
        for (size_t i = 0; i < component.size(); ++i)
        {
            size_t original_position = get<2>(component[i]);
            auto   p =
                std::find(begin(class_labels), end(class_labels), classes[original_position]);
            ++h[std::distance(begin(class_labels), p)];
        }
        auto m      = *std::max_element(begin(h), end(h));
        auto purity = double(m); // / component.size();
        aggregated_purity += purity;
        std::fill(begin(h), end(h), 0);
    }
    return aggregated_purity / data.size();
    // return aggregated_purity / data.num_components();
}

template <typename S>
double purity(const PartitionedData<S>& data, const std::vector<size_t>& classes)
{
    assert(classes.size() == data.size());

    auto d = *std::max_element(classes.begin(), classes.end());

    double              aggregated_purity = 0;
    std::vector<size_t> h(d + 1);
    for (size_t k = 0; k < data.num_components(); ++k)
    {
        auto component = data.subset(k);
        for (size_t i = 0; i < component.size(); ++i)
        {
            size_t original_position = get<2>(component[i]);
            ++h[classes[original_position]];
        }
        aggregated_purity += *std::max_element(begin(h), end(h));
        std::fill(begin(h), end(h), 0);
    }
    return aggregated_purity / data.size();
    // return aggregated_purity / data.num_components();
}

template <typename T>
T nchoose2(const T& x)
{
    return (x * (x - 1)) / T(2);
}

template <typename Trait>
auto pairwise_divergence_js(const Composition<Trait>& c)
{
    using float_type = typename Trait::float_type;

    const auto K   = c.data.num_components();
    float_type acc = 0;
    if (K <= 1)
        return acc;
    for (size_t i = 0; i < K; ++i)
    {
        for (size_t j = i + 1; j < K; ++j)
        {
            auto js = js_fr_divergence_normalized(c, i, j);
            acc += js;
        }
    }
    return acc / nchoose2(float_type(K));
}

template <typename T>
T smooth_zero(const T& x, size_t n, size_t m, size_t a = 1)
{
    if (x != 0)
        return x;
    else
        return smooth(x, n, m, a);
}

template <typename Trait>
auto kl_fr_divergence_sym(const Composition<Trait>& c, size_t first_index, size_t second_index)
{
    const auto&                dat = c.data;
    typename Trait::float_type acc = 0;
    for (size_t i = 0; i < c.summary.size(); ++i)
    {
        if (c.summary.label(i) == 0)
            continue;
        const auto p_0 =
            smooth_zero(c.frequency(i, first_index), dat.subset(first_index).size(), dat.dim);
        const auto p_1 =
            smooth_zero(c.frequency(i, second_index), dat.subset(second_index).size(), dat.dim);
        acc += (kl1(p_0, p_1) + kl1(p_1, p_0)) / 2;
    }
    return acc;
}

template <typename Trait>
auto pairwise_divergence_kl(const Composition<Trait>& c)
{
    using float_type = typename Trait::float_type;
    const auto K     = c.data.num_components();
    float_type acc   = 0;

    if (K <= 1)
        return acc;
    for (size_t i = 0; i < K; ++i)
    {
        for (size_t j = i + 1; j < K; ++j)
        {
            acc += kl_fr_divergence_sym(c, i, j);
        }
    }
    return acc / nchoose2(float_type(K));
}

} // namespace disc
} // namespace sd