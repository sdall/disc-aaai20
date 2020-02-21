#pragma once
#include <cmath>

namespace sd
{
namespace disc
{

template <typename value_type>
value_type nhc_pvalue(value_type h1, value_type h0)
{
    // 2**(-(h0 - h1)),
    // h1 >>> h0 --> ~1
    // h1 <<< h0 --> ~0
    // h1 = ``old value'', h0 = ``next value''
    // based on the non-hypercopressability theorem
    auto m = std::min(h1, h0); // for numerical stability
    auto d = -((h1 - m) - (h0 - m));
    if (d >= 0)
        return 1;
    // if (d < -30) return 0;
    return std::exp2(d);
}

} // namespace disc
} // namespace sd
