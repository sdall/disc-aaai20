#pragma once

#include <disc/storage/Dataset.hxx>

namespace sd
{
namespace disc
{

template <typename P, typename Data>
size_t support(const P& x, const Data& es)
{
    return std::count_if(
        es.begin(), es.end(), [&](const auto& e) { return is_subset(x, point(e)); });
}

template <typename P, typename Data>
bool contains_geq(const P& x, const Data& es, size_t bound)
{
    size_t c = 0;
    for (const auto& e : es)
    {
        if (is_subset(x, point(e)) && ++c >= bound)
        {
            return true;
        }
    }
    return false;
}

template <typename P, typename Data>
bool contains_any_superset(const P& x, const Data& es)
{
    return std::any_of(
        es.begin(), es.end(), [&](const auto& e) { return is_subset(x, point(e)); });
}

template <typename P, typename Data>
bool contains_any(const P& x, const Data& es)
{
    return std::any_of(es.begin(), es.end(), [&](const auto& e) { return equal(point(e), x); });
}

template <typename T, typename P, typename E>
T frequency(const P& x, const E& es)
{
    return static_cast<T>(support(x, es)) / es.size();
}

} // namespace disc
} // namespace sd