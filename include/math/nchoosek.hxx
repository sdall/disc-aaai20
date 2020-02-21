#pragma once
#include <cmath>
#include <type_traits>

namespace sd
{
namespace math
{

constexpr double nchoosek(double n, double k)
{
    if (k > n)
        return 0.;
    if (k * 2 > n)
        k = n - k; // nchoosek(n, n - k);
    if (k == 0)
        return 1.;

    double result = n;
    for (size_t i = 2; i <= k; ++i)
    {
        result *= (n - i + 1) / i;
    }
    return result;
}

constexpr double log2_nchoosek(double n, double k)
{
    if (k > n)
        return -std::numeric_limits<double>::infinity();
    if (k * 2 > n)
        k = n - k; // nchoosek(n, n - k);
    if (k == 0)
        return 0.;

    double result = std::log2(n);
    for (size_t i = 2; i <= k; ++i)
    {
        result += std::log2((n - i + 1) / i);
    }
    return result;
}

} // namespace math
} // namespace sd
