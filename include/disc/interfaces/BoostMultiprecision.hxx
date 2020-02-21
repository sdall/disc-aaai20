#pragma once
#ifndef INTERFACE_BOOST_MULTIPRECISION
#define INTERFACE_BOOST_MULTIPRECISION

#if WITH_QUADMATH //&& __has_include(<quadmath.h>) &&
                  //__has_include(<boost/multiprecision/float128.hpp>)

#define HAS_QUADMATH 1

#include <boost/math/special_functions.hpp>
#include <boost/multiprecision/float128.hpp>
#include <cmath>
#include <iostream>

static const boost::multiprecision::float128 float128_type_log_of_2 =
    boost::multiprecision::log(boost::multiprecision::float128(2));

namespace std
{
// I know that this is undefined behaviour; TODO: use ADL in source code
auto log2(boost::multiprecision::float128 x)
{
    return boost::multiprecision::log(x) / float128_type_log_of_2;
}
auto log(boost::multiprecision::float128 x) { return boost::multiprecision::log(x); }
auto abs(boost::multiprecision::float128 x) { return boost::multiprecision::abs(x); }
auto exp2(boost::multiprecision::float128 x) { return boost::multiprecision::pow(2, x); }
auto isnan(boost::multiprecision::float128 x) { return boost::multiprecision::isnan(x); }
auto isinf(boost::multiprecision::float128 x) { return boost::multiprecision::isinf(x); }
auto sqrt(boost::multiprecision::float128 x) { return boost::multiprecision::sqrt(x); }
auto fabs(boost::multiprecision::float128 x) { return boost::multiprecision::abs(x); }
auto fma(const boost::multiprecision::float128& a,
         const boost::multiprecision::float128& x,
         const boost::multiprecision::float128& b)
{
    return a * x + b;
}

} // namespace std

namespace boost::multiprecision
{
auto log2(boost::multiprecision::float128 x) { return log(x) / float128_type_log_of_2; }
} // namespace boost::multiprecision

#pragma omp declare reduction \
  (*:boost::multiprecision::float128:omp_out=omp_out*omp_in) \
  initializer(omp_priv=1)

#pragma omp declare reduction \
  (+:boost::multiprecision::float128:omp_out=omp_out+omp_in) \
  initializer(omp_priv=0)

namespace sd::disc
{
using float_hp_t = boost::multiprecision::float128;
}

#else
namespace sd::disc
{
using float_hp_t = long double;
}
#endif

#endif