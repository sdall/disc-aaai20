#pragma once
#include <cmath>
#include <type_traits>

namespace sd
{

template <typename T>
constexpr T nchoosek(T n, T k)
{
    static_assert(std::is_floating_point_v<T>, "Type not floating point");
    if (k > n)
        return 0.;
    if (k * 2 > n)
        k = n - k; // nchoosek(n, n - k);
    if (k == 0)
        return 1.;

    T result = n;
    for (size_t i = 2; i <= k; ++i)
    {
        result *= (n - i + 1) / i;
    }
    return result;
}

template <typename T>
constexpr auto log_nchoosek(T n, T k)
{
    using result_t = std::decay_t<decltype(log(T{}))>;
    using std::log;
    if (k > n)
        return -std::numeric_limits<result_t>::infinity();
    if (k * 2 > n)
        k = n - k; // nchoosek(n, n - k);
    if (k == 0)
        return result_t(0);

    result_t result = log(n);
    for (size_t i = 2; i <= k; ++i)
    {
        result += log((n - i + 1) / i);
    }
    return result;
}

constexpr double log2_nchoosek(double n, double k)
{
    using std::log;
    return log_nchoosek<double>(n, k) / log(2);
}

} // namespace sd
