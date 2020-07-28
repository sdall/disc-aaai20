#pragma once

#include <disc/storage/Dataset.hxx>
#include <disc/storage/Itemset.hxx>

#include <vector>

namespace sd
{
namespace disc
{

template <typename DataType>
std::vector<size_t> compute_singleton_supports(size_t dim, const DataType& data)
{
    std::vector<size_t> support(dim);
    for (size_t i = 0; i < data.size(); ++i)
    {
        assert(i < data.size());
        foreach(data.point(i), [&](auto j) {
            assert(j < dim);
            support[j]++;
        });
    }
    return support;
}

template <typename DataType, typename ItemsetType>
void insert_missing_singletons(const DataType& data, ItemsetType& summary)
{
    using value_type = typename ItemsetType::label_type;

    itemset<tag_dense> set(std::max(data.dim, summary.dim));

    for (const auto& x : summary)
        if (is_singleton(point(x)))
            set.insert(front(point(x)));

    if (set.count() == data.dim)
    {
        return;
    }

    auto support = compute_singleton_supports(data.dim, data);
    assert(support.size() == data.dim);

    for (size_t i = 0; i < data.dim; ++i)
    {
        if (!set.test(i)) // && support[i] > 0)
        {
            auto fr = static_cast<value_type>(support[i]) / data.size();
            summary.insert(fr, i);
        }
    }
}

} // namespace disc
} // namespace sd