#pragma once

#include <math/nchoosek.hxx>

namespace sd
{
namespace disc
{

constexpr size_t iterated_log2(double n)
{
    // iterated log
    if (n <= 1)
        return 0;
    if (n <= 2)
        return 1;
    if (n <= 4)
        return 2;
    if (n <= 16)
        return 3;
    if (n <= 65536)
        return 4;
    return 5; // if (n <= 2^65536)
    // never 6 for our case.
}

constexpr double universal_code(size_t n) { return iterated_log2(n) + 1.5186; }
// 1.5186 = log2(2.865064)

} // namespace disc
} // namespace sd