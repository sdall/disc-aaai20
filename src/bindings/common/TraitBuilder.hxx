#pragma once

#include <disc/desc/Desc.hxx>
#include <disc/disc/Disc.hxx>

namespace sd::disc
{
template <typename S, typename T, typename Fn>
void select_dist_type(Fn&& fn)
{
    using trait = Trait<S, T, MaxEntDistribution<S, T>>;
    std::forward<Fn>(fn)(trait{});
}

template <typename S, typename Fn>
void select_real_type(bool is_precise, Fn&& fn)
{
    if (is_precise)
    {
        select_dist_type<S, precise_float_t>(std::forward<Fn>(fn));
    }
    else
    {
        select_dist_type<S, double>(std::forward<Fn>(fn));
    }
}

template <typename Fn>
void build_trait(bool is_sparse, bool is_precise, Fn&& fn)
{
    if (is_sparse)
    {
        select_real_type<tag_sparse>(is_precise, std::forward<Fn>(fn));
    }
    else
    {
        select_real_type<tag_dense>(is_precise, std::forward<Fn>(fn));
    }
}

template <typename S>
constexpr const char* storage_type_to_str()
{
    if constexpr (std::is_same_v<S, tag_dense>)
    {
        return "dense";
    }
    else
    {
        return "sparse";
    }
}

template <typename T>
constexpr const char* float_storage_type_to_str()
{
    if constexpr (std::is_same_v<T, double>)
    {
        return "double";
    }
    else if constexpr (std::is_same_v<T, long double>)
    {
        return "ldouble";
    }
#if HAS_QUADMATH
    else if constexpr (std::is_same_v<T, boost::multiprecision::float128>)
    {
        return "float128";
    }
#endif
    else
    {
        return "unknown";
    }
}

} // namespace sd::disc