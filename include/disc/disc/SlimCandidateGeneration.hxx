#pragma once

#include <vector>

#include <nonstd/optional.hpp>

#include <disc/storage/Dataset.hxx>
#include <disc/storage/Itemset.hxx>

namespace sd
{
namespace disc
{

template <typename S, typename T, typename... U>
struct SlimCandidate
{
    itemset<S>                pattern;
    long_storage_container<S> row_ids;
    size_t                    support = 0;
    T                         score   = 0;

    std::tuple<U...> optional_payload;
};

template <typename S, typename T, typename... U>
SlimCandidate<S, T, U...> join(SlimCandidate<S, T, U...> next, const SlimCandidate<S, T, U...>& b)
{
    intersection(b.row_ids, next.row_ids);
    next.pattern.insert(b.pattern);
    next.support = count(next.row_ids);
    return next;
}

template <typename S, typename T, typename... U>
struct SlimGenerator
{
    using state_type = SlimCandidate<S, T, U...>;

    struct ordering
    {
        constexpr bool operator()(const state_type& a, const state_type& b) const noexcept
        {
            return a.score < b.score;
        }
    };

    struct ConstantScoreFunction
    {
        constexpr int operator()(const state_type&) const noexcept { return 1; }
    };

    template <typename Data, typename ScoreFn = ConstantScoreFunction>
    SlimGenerator(const Data&              data,
                  size_t                   min_support,
                  nonstd::optional<size_t> max_width = {},
                  ScoreFn&&                score     = {})
        : max_width(max_width), min_support(min_support)
    {
        init(data, std::forward<ScoreFn>(score));
    }

    template <typename Data, typename Patternset, typename ScoreFn = ConstantScoreFunction>
    SlimGenerator(const Data&              data,
                  const Patternset&        summary,
                  size_t                   min_support,
                  nonstd::optional<size_t> max_width = {},
                  ScoreFn&&                score     = {})
        : max_width(max_width), min_support(min_support)
    {
        init(data, summary, score);
    }

    // SlimGenerator(SlimGenerator&&)      = default;
    // SlimGenerator(const SlimGenerator&) = default;
    // SlimGenerator& operator=(SlimGenerator&&) = default;
    // SlimGenerator& operator=(const SlimGenerator&) = default;

    nonstd::optional<state_type> next()
    {
        if (candidates.empty())
            return {};

        // std::pop_heap(candidates.begin(), candidates.end(), ordering{});

        auto ret = std::move(candidates.back());
        candidates.pop_back();

        return ret;
    }

    template <typename ScoreFunction>
    void combine_pairs(const state_type& next, ScoreFunction&& score)
    {
        const size_t count_next = count(next.pattern);

#pragma omp parallel for
        for (size_t i = 0; i < singletons.size(); ++i)
        {
            // clang-format off
            if (is_subset(singletons[i].pattern, next.pattern)) continue;

            auto joined = join(next, singletons[i]);

            // if (is_subset(joined.pattern, next.pattern))         continue;
            if (count(joined.pattern) <= count_next)             continue;

            if (joined.support < min_support)                    continue;
            if (max_width && count(joined.pattern) > *max_width) continue;
            
            joined.score = score(joined);            
            
            if (joined.score <= T(0))                            continue;
                // clang-format on


#pragma omp critical
            {
                candidates.emplace_back(std::move(joined));
                ++stat_num_generated_candidates;
            }
        }
    }

    template <typename ScoreFunction>
    void add_next_only(const state_type& next, ScoreFunction&& score)
    {
        // recompute scores for patterns that are affected
        compute_scores(next, score);
        // prune
        // prune([](auto const &a) { return a.score <= 0; });
        // add pairs into queue
        combine_pairs(next, std::forward<ScoreFunction>(score));
    }

    template <typename ScoreFunction>
    void add_next(const state_type& next, ScoreFunction&& score)
    {
        add_next_only(next, std::forward<ScoreFunction>(score));
        // re-order candidates.
        order_candidates();
    }

    template <typename ScoreFunction>
    void compute_scores(const state_type& joined, ScoreFunction&& score)
    {
#pragma omp parallel for
        for (size_t i = 0; i < candidates.size(); ++i)
        {
            if (intersects(joined.pattern, candidates[i].pattern))
                candidates[i].score = score(candidates[i]);
        }
    }

    void order_candidates()
    {
        // std::make_heap(candidates.begin(), candidates.end(), ordering{});
        std::sort(candidates.begin(), candidates.end(), ordering{});
        // auto cur = std::unique(candidates.begin(), candidates.end(),
        //    [](const auto& a, const auto & b) {
        //         return a.support == b.support && sd::equal(a.pattern, b.pattern);
        //     });
        // candidates.erase(cur, candidates.end());

        // auto cur2 = std::adjacent_find(candidates.begin(), candidates.end(), 
        //     [](const auto& a, const auto & b) {
        //         return a.support == b.support && sd::equal(a.pattern, b.pattern);
        //     });
        // if(cur2 != candidates.end()) throw 0;
    }

    template <typename ScoreFunction>
    void compute_scores(ScoreFunction&& score)
    {
#pragma omp parallel for
        for (size_t i = 0; i < candidates.size(); ++i)
        {
            candidates[i].score = score(candidates[i]);
        }
    }

    template <typename Fn>
    void prune(Fn&& fn)
    {
        auto ptr = std::remove_if(candidates.begin(), candidates.end(), std::forward<Fn>(fn));
        candidates.erase(ptr, candidates.end());
    }

    bool has_next() const { return !candidates.empty(); }
    bool queue_size() const { return candidates.size(); }

    template <typename Data>
    void init_singletons(Data const& data)
    {
        singletons                    = std::vector<state_type>(data.dim);
        stat_num_generated_candidates = 0;

        if (data.size() == 0 || data.dim == 0)
            return;

        size_t i = 0;
        for (auto& s : singletons)
        {
            s.row_ids.reserve(data.size());
            // reserve(s.row_ids, data.size());
            s.pattern.insert(i++);
        }

        size_t row_index = 0;
        for (const auto& x : data)
        {
            iterate_over(point(x), [&](size_t i) { singletons[i].row_ids.insert(row_index); });
            ++row_index;
        }

        for (auto& s : singletons)
        {
            s.support = count(s.row_ids);
        }
    }

    template <typename Data, typename ScoreFn = ConstantScoreFunction>
    void init(Data const& data, ScoreFn&& score = {})
    {
        init_singletons(data);
        candidates.clear();
        candidates.reserve(data.dim * data.dim);

#pragma omp parallel for // collapse (2)
        for (size_t i = 0; i < singletons.size(); ++i)
        {
            if (singletons[i].support >= min_support)
            {
                for (size_t j = i + 1; j < singletons.size(); ++j)
                {
                    if (singletons[j].support >= min_support)
                    {
                        auto joined = join(singletons[i], singletons[j]);
                        if (max_width && count(joined.pattern) > *max_width)
                            continue;
                        if (joined.support >= min_support)
                        {
#pragma omp critical
                            {
                                joined.score = score(joined);
                                if (joined.score > 0)
                                {
                                    candidates.emplace_back(std::move(joined));
                                    ++stat_num_generated_candidates;
                                }
                            }
                        }
                    }
                }
            }
        }
        order_candidates();
    }

    template <typename Data, typename Patternset, typename ScoreFn = ConstantScoreFunction>
    void init(Data const& data, const Patternset& patternset, ScoreFn&& score = {})
    {
        if (patternset.empty())
        {
            init(data);
            return;
        }

        init_singletons(data);
        candidates.clear();
        candidates.reserve(data.dim * data.dim);

        disc::itemset<S> has_seen;

        for (const auto& x : patternset)
        {
            if (is_singleton(point(x)))
                continue;

            has_seen.insert(point(x));

            // SlimCandidate<S, T, U...> next; //  = singletons[front(point(x))];
            state_type next;

            // reserve(next.row_ids, data.size());
            next.row_ids.reserve(data.size());

            iterate_over(point(x), [&](size_t i) {
                next = join(std::move(next), singletons[i]);
                if (!is_singleton(next.pattern))
                {
                    combine_pairs(next, [](auto&&...) { return 0; });
                }
            });
        }

#pragma omp parallel for // collapse (2)
        for (size_t i = 0; i < singletons.size(); ++i)
        {
            if (singletons[i].support >= min_support)
            {
                for (size_t j = i + 1; j < singletons.size(); ++j)
                {
                    if (singletons[j].support >= min_support)
                    {
                        auto joined = join(singletons[i], singletons[j]);
                        if (max_width && count(joined.pattern) > *max_width)
                            continue;
                        if (is_subset(joined.pattern, has_seen))
                            continue;
                        if (joined.support >= min_support)
                        {
#pragma omp critical
                            {
                                joined.score = score(joined);
                                if (joined.score > 0)
                                {
                                    candidates.emplace_back(std::move(joined));
                                    ++stat_num_generated_candidates;
                                }
                            }
                        }
                    }
                }
            }
        }
        order_candidates();
    }

    size_t count_generated_candidates() const { return stat_num_generated_candidates; }
    size_t count_current_candidates() const { return candidates.size(); }

private:
    nonstd::optional<size_t> max_width;
    size_t                   min_support = 1;
    std::vector<state_type>  singletons;
    std::vector<state_type>  candidates;
    size_t                   stat_num_generated_candidates = 0;
};

// template <typename S, typename T>
// struct SlimGenerator

} // namespace disc
} // namespace sd