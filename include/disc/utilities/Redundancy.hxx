#pragma once

#include <cmath>

#include <disc/storage/Dataset.hxx>
#include <math/RunningStatistics.hxx>

namespace sd::disc
{

template <typename T, typename S>
auto self_similarity(LabeledDataset<T, S> const& set)
{
    sd::RunningDescription<T> acc;
    for (size_t i = 0, n = set.size(); i < n; ++i)
    {
        auto        qi = set.label(i);
        const auto& u  = set.point(i);
        if (!is_singleton(u))
        {
            size_t cnt_u = count(u);
            for (size_t j = i + 1; j < n; ++j)
            {
                const auto& v = set.point(j);
                if (!is_singleton(v))
                {
                    size_t cnt_v = count(v);
                    auto   qj    = set.label(j);
                    T      sim   = size_of_intersection(u, v);
                    acc += (T(1) - abs(qi - qj)) * sim / max(cnt_u, cnt_v);
                }
            }
        }
    }
    return acc;
}

template <typename T, typename S>
auto shannon_redundancy(LabeledDataset<T, S> const& set)
{
    if (set.size() <= 1)
        return T(0);

    std::vector<size_t> support(set.dim);
    for (const auto& x : set)
    {
        iterate_over(point(x), [&](size_t j) { ++support[j]; });
    }

    T H = 0;
    for (size_t i = 0, n = support.size(); i < n; ++i)
    {
        if (support[i] == 0)
            continue;
        auto qi = T(support[i]) / set.size();

        H -= qi * std::log2(qi);
    }
    const T Hmax = std::log2(set.dim);
    return T(1) - H / Hmax;
}

} // namespace sd::disc
