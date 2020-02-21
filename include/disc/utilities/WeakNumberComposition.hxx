#pragma once

#include <math/nchoosek.hxx>

namespace sd
{
namespace disc
{

size_t log_number_of_weak_compositions(size_t n, size_t k)
{
    return !(n == 0 || k <= 1) ? math::log2_nchoosek(n + k - 1, k - 1) : 0;
}

} // namespace disc
} // namespace sd