#pragma once

#include <disc/storage/Itemset.hxx>

#include <datatable/data_table.hxx>

// #include <boost/container/pmr/vector.hpp>

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
struct column_type<disc::tag_dense>
{
    using value = std::vector<disc::storage_container<disc::tag_dense>>;
};
template <>
struct column_type<disc::tag_sparse>
{
    using value = std::vector<disc::storage_container<disc::tag_sparse>>;
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
        this->push_back(storage_container<S>(std::forward<T>(t)));
        if constexpr (std::is_same_v<S, tag_sparse>)
        {
            point(this->size() - 1).container.shrink_to_fit();
        }
        dim = std::max(dim, get_dim(point(this->size() - 1)));
    }

    Dataset& operator=(const PartitionedData<S>& rhs)
    {
        using sd::disc::point;
        for (const auto x : rhs)
            insert(point(x));
        return *this;
    }

    template <typename R, typename = std::enable_if_t<!std::is_same_v<R, Dataset<S>>>>
    Dataset(R&& other)
    {
        this->reserve(other.size());
        for (auto&& o : other)
        {
            using sd::disc::point;
            insert(point(o));
        }
    }

    Dataset()               = default;
    Dataset(Dataset&&)      = default;
    Dataset(const Dataset&) = default;
    Dataset& operator=(const Dataset&) = default;
    Dataset& operator=(Dataset&&) = default;

    decltype(auto) point(size_t index) const { return this->template col<0>()[index]; }
    decltype(auto) point(size_t index) { return this->template col<0>()[index]; }

    size_t capacity() const { return this->template col<0>().capacity(); }

    size_t dim = 0;
};

template <typename L, typename S>
struct LabeledDataset : public sd::df::col_store<L, S>
{
    using label_type   = L;
    using pattern_type = S;

    template <typename T>
    void insert(const L& label, T&& t)
    {
        storage_container<S> buf;
        buf.insert(t);
        this->push_back(label, std::move(buf));
        if constexpr (std::is_same_v<S, tag_sparse>)
        {
            point(this->size() - 1).container.shrink_to_fit();
        }
        dim = std::max(dim, get_dim(point(this->size() - 1)));
    }
    template <typename T>
    void insert(T&& t)
    {
        insert(L{}, std::forward<T>(t));
    }
    const auto&    labels() const { return this->template col<0>(); }
    decltype(auto) point(size_t index) const { return this->template col<1>()[index]; }
    decltype(auto) label(size_t index) const { return this->template col<0>()[index]; }
    decltype(auto) point(size_t index) { return this->template col<1>()[index]; }
    decltype(auto) label(size_t index) { return this->template col<0>()[index]; }

    size_t dim = 0;
};

template <typename S>
struct PartitionedData_Copies : public sd::df::col_store<size_t, S, size_t>
{
    using pattern_type = S;

    template <typename T>
    void insert(T&& t, size_t label, size_t original_position)
    {
        this->push_back(label, storage_container<S>(std::forward<T>(t)), original_position);
        if constexpr (std::is_same_v<S, tag_sparse>)
        {
            point(this->size() - 1).container.shrink_to_fit();
        }
        dim = std::max(dim, get_dim(point(this->size() - 1)));
    }

    template <typename T>
    void insert(T&& t, size_t label = 0)
    {
        insert(std::forward<T>(t), label, this->size());
    }

    auto subset(size_t index)
    {
        return this->map_cols([&](auto& s) {
            return sd::make_cpslice(s).cut(positions[index], positions[index + 1]);
        });
    }

    auto subset(size_t index) const
    {
        return this->map_cols([&](auto& s) {
            return sd::make_cpslice(s).cut(positions[index], positions[index + 1]);
        });
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

    std::vector<size_t> positions;
    size_t              dim                   = 0;
    size_t              num_components_backup = 0;
};

template <typename T>
using itemset_view = std::conditional_t<is_sparse(T{}),
                                        sd::sparse_bit_view<sd::slice<sparse_index_type>>,
                                        sd::bit_view<sd::slice<uint64_t>>>;

template <typename S>
struct PartitionedData : public sd::df::col_store<size_t, itemset_view<S>, size_t>
{
    using pattern_type = S;

    PartitionedData() = default;

    explicit PartitionedData(Dataset<S>&& rhs)
    {
        data = std::make_shared<Dataset<S>>(std::forward<Dataset<S>>(rhs));

        this->resize(data->size());

        auto& ol = this->template col<2>();
        std::iota(ol.begin(), ol.end(), 0);

        reindex();
        dim = data->dim;
        group_by_label();
    }

    explicit PartitionedData(Dataset<S>&& rhs, const std::vector<size_t>& labels) : PartitionedData(std::forward<Dataset<S>>(rhs))
    {
        assert(labels.size() == this->size());
        std::copy_n(labels.begin(), labels.size(), this->template col<0>().begin());
        group_by_label();
    }

    // template <typename T>
    // void insert(T&& t, size_t label, size_t original_position)
    // {
    //     auto c = data->capacity();
    //     data->insert(std::forward<T>(t));

    //     auto& x = data->point(data->size() - 1);

    //     if constexpr (std::is_same_v<S, tag_sparse>)
    //     {
    //         this->push_back(label, itemset_view<S>{x.container}, original_position);
    //     }
    //     else
    //     {
    //         this->push_back(label, itemset_view<S>(x.container, x.length_), original_position);
    //     }

    //     // if (data->capacity() != c)
    //         reindex();

    //     dim = data->dim;
    // }

    // template <typename T>
    // void insert(T&& t, size_t label = 0)
    // {
    //     insert(std::forward<T>(t), label, this->size());
    // }

    auto subset(size_t index)
    {
        return this->map_cols([&](auto& s) {
            return sd::make_cpslice(s).cut(positions[index], positions[index + 1]);
        });
    }

    auto subset(size_t index) const
    {
        return this->map_cols([&](auto& s) {
            return sd::make_cpslice(s).cut(positions[index], positions[index + 1]);
        });
    }

    size_t num_components() const { return positions.empty() ? 0 : positions.size() - 1; }

    const auto& elements() const { return this->template col<1>(); }

    void reindex()
    {
        assert(data->size() == this->size());
        for (size_t i = 0; i < data->size(); ++i)
        {
            if constexpr (std::is_same_v<S, tag_sparse>)
            {
                point(i) = itemset_view<S>{data->point(i).container};
            }
            else
            {
                auto& x  = data->point(i);
                point(i) = itemset_view<S>(x.container, x.length_);
            }
        }
    }

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
        this->data->reserve(n);
    }

    decltype(auto) point(size_t index) { return this->template col<1>()[index]; }
    decltype(auto) point(size_t index) const { return this->template col<1>()[index]; }
    decltype(auto) label(size_t index) const { return this->template col<0>()[index]; }
    decltype(auto) label(size_t index) { return this->template col<0>()[index]; }
    decltype(auto) original_position(size_t index) const
    {
        return this->template col<2>()[index];
    }
  
    std::shared_ptr<Dataset<S>> data{new Dataset<S>()};
    std::vector<size_t>         positions;
    size_t                      dim                   = 0;
    size_t                      num_components_backup = 0;
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

namespace meta
{
template <typename T, typename = void>
struct has_components_member_fn : std::false_type
{
};

template <typename T>
struct has_components_member_fn<T, std::void_t<decltype(std::declval<T>().num_components())>>
    : std::true_type
{
};

} // namespace meta

} // namespace sd::disc