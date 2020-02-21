#pragma once

#include <disc/distribution/BitPermutation.hxx>
#include <disc/distribution/IncrementBitset.hxx>
#include <disc/storage/Itemset.hxx>

#include <nonstd/optional.hpp>

namespace sd
{
namespace viva
{

template <typename Fn>
void iterate_powerset_short(size_t n, Fn&& fn)
{
    const auto one   = size_t(1);
    const auto power = (one << n);
    for (size_t i = power; i-- > 0;)
    {
        std::forward<Fn>(fn)(i);
    }
}

template <typename Fn>
void iterate_powerset_long(size_t n, Fn&& fn)
{
    for (disc::itemset<disc::tag_dense> i(n, false); true; increment(i))
    {
        std::forward<Fn>(fn)(i);
        if (count(i) == n)
            break;
    }
}

template <typename Fn>
void iterate_powerset(size_t n, Fn&& fn)
{
    if (n < std::numeric_limits<size_t>::digits)
    {
        iterate_powerset_short(n, std::forward<Fn>(fn));
    }
    else
    {
        iterate_powerset_long(n, std::forward<Fn>(fn));
    }
}

template <typename U, typename V>
struct Block
{
    using count_type = V;

    count_type       value{0};
    count_type       count{0};
    disc::itemset<U> cover;

    bool operator<(Block const& rhs) const { return count < rhs.count; }
};

template <typename model_type, typename block_container_type>
size_t generate_blocks_long(size_t dim, model_type const& m, block_container_type& blocks)
{
    using count_type   = typename model_type::float_type;
    using pattern_type = typename model_type::pattern_type;
    using storage_type = disc::itemset<pattern_type>;

    const auto size = m.size();

    if (size == 0 || dim == 0)
    {
        const auto k = std::exp2(count_type(dim));
        blocks       = {{k, k, storage_type(dim)}};

        return 1;
    }

    disc::itemset<pattern_type> cover;
    cover.reserve(dim);

    blocks.clear();
    for (disc::itemset<disc::tag_dense> i(size, false); true; increment(i))
    {
        cover.clear();

        iterate_over(i, [&](size_t j) { cover.insert(m.point(j)); });

        const auto k = std::exp2(count_type(dim) - count_type(count(cover)));
        assert(k > 0);
        blocks.push_back({k, k, cover});

        if (count(i) == size)
            break;
    }
    std::sort(blocks.begin(), blocks.end());
    auto it = std::unique(blocks.begin(), blocks.end(), [](const auto& a, const auto& b) {
        return equal(a.cover, b.cover);
    });
    blocks.erase(it, blocks.end());
    return blocks.size();
}

template <typename model_type, typename block_container_type>
void generate_blocks_and_counts_long(size_t                dim,
                                     model_type const&     model,
                                     block_container_type& blocks)
{
    auto n = generate_blocks_long(dim, model, blocks);
    if (n <= 1)
        return;

    for (size_t i = 0; i < n; ++i)
    {
        auto& b = blocks[i];
        b.value = b.count;
        // if(b.cover.empty()) break;
        for (size_t j = 0; j < i; ++j)
        {
            auto& a = blocks[j];
            if (is_subset(b.cover, a.cover))
            {
                if(count(b.cover) == count(a.cover) && 
                equal(a.cover, b.cover)) 
                {
                    b.value = 0;
                    b.cover.clear();
                    break;
                }
                // if(count(b.cover) < count(a.cover)) {
                assert(b.value - a.value >= 0);
                b.value -= a.value;
                // }
            }
        }
    }
    // std::decay_t<decltype(blocks.front().value)> total = 0;
    // std::cout << "### BEGIN OF TRANSACTION ###\n";
    // for(const auto& block : blocks) 
    // {
        // std::cout << "{ ";
        // iterate_over(block.cover, [&](size_t j) { std::cout << j << " "; });
        // std::cout << "} [";
        // std::cout << block.value << "] \n";
        // total += block.value ;
    // }
    // std::cout << total << "\n";
    // std::cout << "### END OF TRANSACTION ### \n\n\n";
}

template <typename model_type, typename block_container_type>
size_t generate_blocks_and_counts(size_t dim, model_type const& m, block_container_type& blocks)
{
    using count_type   = typename model_type::float_type;
    using pattern_type = typename model_type::pattern_type;
    using storage_type = disc::itemset<pattern_type>;

    const auto size  = m.size();
    const auto width = static_cast<count_type>(dim);

    if (size == 0 || dim == 0)
    {
        const auto k = std::exp2(width);
        blocks       = {{k, k, storage_type(dim)}};
        return 1;
    }

    const size_t part_size = std::exp2(m.size());
    blocks.clear();
    blocks.resize(part_size);
    // blocks.clear();
    // blocks.reserve(part_size);
    // std::cout << "### BEGIN OF TRANSACTION ###\n";

    size_t index = 0;
    // count_type total = 0;

    viva::permute_all(m.size(), [&](size_t i) {
        // auto& block = blocks.emplace_back();
        auto& block = blocks[index];

        block.cover.clear();
        iterate_over(i, [&](size_t j) { block.cover.insert(m.point(j)); });
        const auto cover_size = static_cast<count_type>(count(block.cover));
        const auto k          = std::exp2(width - cover_size);
        block.value           = k;
        block.count           = k;
        assert(k > 0);
        assert(!std::isnan(k));

        for (size_t prev = 0; prev < index; ++prev)
        {
            if (is_subset(block.cover, blocks[prev].cover))
            {
                if(count(block.cover) == count(blocks[prev].cover)) {
                    block.value = 0;
                    block.cover.clear();
                    break;
                }
                else {
                    block.value -= blocks[prev].value;
                    assert(block.value >= 0);
                    assert(floor(block.value) == block.value);
                }
            }
        }
                // total += block.value;
        if(block.value != 0) {
            // std::cout << "{ ";
            // iterate_over(block.cover, [&](size_t j) { std::cout << j << " "; });
            // std::cout << "} [";
            // std::cout << block.value << "] \n";
            ++index;
        }
    });
            // std::cout << "sum of values = " << total << '\n';

    blocks.resize(index);
        // std::cout << "### END OF TRANSACTION ### \n\n\n";

    return index;
}

template <typename model_type, typename block_container_type>
size_t compute_counts(size_t dim, model_type const& m, block_container_type& blocks)
{
    blocks.clear();
    if (dim < std::numeric_limits<size_t>::digits)
    {
        return generate_blocks_and_counts(dim, m, blocks);
    }
    else
    {
        generate_blocks_and_counts_long(dim, m, blocks);
        return blocks.size();
    }
}

#if 0
    template <typename model_type, typename block_container_type>
    size_t
    generate_blocks_short(size_t dim, model_type const& m, block_container_type& blocks)
    {
        using count_type = typename model_type::float_type;
        using pattern_type = typename model_type::pattern_type;
        using storage_type = disc::itemset<pattern_type>;

        const auto size = m.size();
        const auto width = static_cast<count_type>(dim);

        if (size == 0 || dim == 0)
        {
            const auto k = std::exp2(width);
            blocks       = {{k, k, storage_type(dim)}};
            return 1;
        }

        const size_t part_size = std::exp2(m.size());
        blocks.resize(part_size);

        // static constexpr auto bits_per_block = std::numeric_limits<block_type>::digits;
        cover_storage cover;
        cover_storage buffer;
        cover.reserve(dim); // std::max(dim, m.point(0).size()));
        buffer.reserve(size);
        blocks.clear();
        const auto one   = size_t(1);
        const auto power = (one << size);
        for (size_t i = power; i-- > 0;) {
            cover.clear();
            buffer.clear();
            iterate_over(i, [&](size_t j) {
                cover.insert(m.point(j));
                buffer.insert(j);
            });

            const auto k =
                std::exp2(count_type(dim) - count_type(count(cover)));
            assert(k > 0);
            blocks.push_back({k, k, cover});
        }
        std::sort(blocks.begin(), blocks.end());
        return blocks.size();
    }
#endif


} // namespace viva
} // namespace sd
