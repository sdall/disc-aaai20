#pragma once

#include <disc/storage/Itemset.hxx>

#include <datatable/data_table.hxx>

namespace sd::disc
{

template <typename T>
decltype(auto) point(std::tuple<T>&& t)
{
    return std::get<0>(t);
}
template <typename T>
decltype(auto) point(std::tuple<T>& t)
{
    return std::get<0>(t);
}
template <typename T>
decltype(auto) point(const std::tuple<T>& t)
{
    return std::get<0>(t);
}

template <typename... Ts>
decltype(auto) point(std::tuple<Ts...>&& t)
{
    return std::get<1>(t);
}
template <typename... Ts>
decltype(auto) point(std::tuple<Ts...>& t)
{
    return std::get<1>(t);
}
template <typename... Ts>
decltype(auto) point(const std::tuple<Ts...>& t)
{
    return std::get<1>(t);
}

template <typename... Ts>
decltype(auto) label(std::tuple<Ts...>&& t)
{
    return std::get<0>(t);
}
template <typename... Ts>
decltype(auto) label(std::tuple<Ts...>& t)
{
    return std::get<0>(t);
}
template <typename... Ts>
decltype(auto) label(const std::tuple<Ts...>& t)
{
    return std::get<0>(t);
}
} // namespace sd::disc

namespace sd::df
{

template <>
struct column_type<disc::tag_dense> : ndarray<disc::storage_container<disc::tag_dense>>
{
};
template <>
struct column_type<disc::tag_sparse> : ndarray<disc::storage_container<disc::tag_sparse>>
{
};

} // namespace sd::df

namespace sd::disc
{

template <typename S>
struct PartitionedData;

template <typename S>
struct Dataset : public sd::df::col_store<S>
{
    using pattern_type = S;

    template <typename T>
    void insert(T&& t)
    {
        buf.clear();
        buf.insert(std::forward<T>(t));
        dim = std::max(dim, get_dim(buf));
        this->push_back(buf);
        if constexpr (std::is_same_v<S, tag_sparse>)
        {
            point(this->size() - 1).container.shrink_to_fit();
        }
    }

    Dataset& operator=(const PartitionedData<S>& rhs)
    {
        for (const auto x : rhs)
            insert(sd::disc::point(x));
        return *this;
    }

    Dataset()               = default;
    Dataset(Dataset&&)      = default;
    Dataset(const Dataset&) = default;
    Dataset& operator=(const Dataset&) = default;
    Dataset& operator=(Dataset&&) = default;

    decltype(auto) point(size_t index) const { return this->template col<0>()[index]; }
    decltype(auto) point(size_t index) { return this->template col<0>()[index]; }

    size_t               dim = 0;
    storage_container<S> buf{};
};

template <typename T, typename S>
struct LabeledDataset : public sd::df::col_store<T, S>
{
    using label_type   = T;
    using pattern_type = S;

    template <typename U>
    void insert(U&& u)
    {
        insert(0, std::forward<U>(u));
    }

    template <typename U>
    void insert(const T& t, U&& u)
    {
        buf.clear();
        buf.insert(std::forward<U>(u));
        dim = std::max(dim, get_dim(buf));
        this->push_back(t, buf);
        if constexpr (std::is_same_v<S, tag_sparse>)
        {
            point(this->size() - 1).container.shrink_to_fit();
        }
    }
    const auto&    labels() const { return this->template col<0>(); }
    decltype(auto) point(size_t index) const { return this->template col<1>()[index]; }
    decltype(auto) label(size_t index) const { return this->template col<0>()[index]; }
    decltype(auto) point(size_t index) { return this->template col<1>()[index]; }
    decltype(auto) label(size_t index) { return this->template col<0>()[index]; }

    size_t               dim = 0;
    storage_container<S> buf{};
};

template <typename S>
struct PartitionedData : public sd::df::col_store<size_t, S, size_t>
{
    using pattern_type = S;

    template <typename T>
    void insert(T&& t, size_t label, size_t original_position)
    {
        buf.clear();
        buf.insert(std::forward<T>(t));
        dim = std::max(dim, get_dim(buf));
        this->push_back(label, buf, original_position);
        if constexpr (std::is_same_v<S, tag_sparse>)
        {
            point(this->size() - 1).container.shrink_to_fit();
        }
    }

    template <typename T>
    void insert(T&& t, size_t label = 0)
    {
        insert(std::forward<T>(t), label, this->size());
    }

    auto subset(size_t index)
    {
        return this->map_cols(
            [&](auto& s) { return s.cut(positions[index], positions[index + 1]); });
    }

    auto subset(size_t index) const
    {
        return this->map_cols(
            [&](auto& s) { return s.cut(positions[index], positions[index + 1]); });
    }

    size_t num_components() const { return positions.empty() ? 0 : positions.size() - 1; }

    const auto& elements() const { return this->template col<1>(); }

    void revert_order()
    {
        std::sort(this->begin(), this->end(), [](const auto& a, const auto& b) {
            return get<2>(a) < get<2>(b);
        });
        positions.clear();
    }

    void group_by_label()
    {
        auto lt = [](const auto& a, const auto& b) { return get<0>(a) < get<0>(b); };

        positions.clear();
        positions.reserve(this->size() * 0.1);

        std::sort(this->begin(), this->end(), lt);

        size_t last_label = std::numeric_limits<size_t>::max();
        auto&  labels     = this->template col<0>();
        for (size_t i = 0; i < labels.size(); ++i)
        {
            const auto& l = labels[i];
            if (last_label != l)
            {
                last_label = l;
                positions.push_back(i);
            }
        }
        positions.push_back(this->size());
        num_components_backup = num_components();
    }

    void reserve(size_t n)
    {
        this->foreach_col([n](auto& c) { c.reserve(n); });
    }

    decltype(auto) point(size_t index) { return this->template col<1>()[index]; }
    decltype(auto) point(size_t index) const { return this->template col<1>()[index]; }
    decltype(auto) label(size_t index) const { return this->template col<0>()[index]; }
    decltype(auto) label(size_t index) { return this->template col<0>()[index]; }
    decltype(auto) original_position(size_t index) const
    {
        return this->template col<2>()[index];
    }

    std::vector<size_t>  positions;
    size_t               dim                   = 0;
    size_t               num_components_backup = 0;
    storage_container<S> buf{};
};

template <typename S>
void simplify_labels(PartitionedData<S>& data)
{
    for (size_t subset = 0, n = data.num_components(); subset < n; ++subset)
    {
        for (auto [y, _1, _2] : data.subset(subset))
        {
            y = subset;
        }
    }
}

} // namespace sd::disc