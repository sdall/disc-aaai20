#pragma once

#include <dbscan/dbscan.hpp>
#include <marray/marray.hxx>

#include <disc/disc/Desc.hxx>
#include <disc/disc/ReassignComponents.hxx>
#include <disc/utilities/MeasureElapsedTime.hxx>

namespace sd::disc 
{
    template<typename S, typename T>
    float distance(const S& a, const T& b) 
    {
        auto na = count(a);
        auto nb = count(b);
        return 1.0 - static_cast<float>(size_of_intersection(a, b)) / std::max(na, nb);
    }

    template<typename Data, typename DistanceMatrix>
    void compute_distances(const Data& data, DistanceMatrix & m) 
    {
        const size_t n = data.size();
        
        m.resize(n, n);

#pragma omp parallel for
        for (size_t i = 0; i < n; ++i)
        {
            m(i, i) = 0;
            for (size_t j = i + 1; j < n; ++j)
            {
                auto d = distance(data.point(i), data.point(j));
                m(i, j) = d;
                m(j, i) = d;
            }
        }
    }

    template<typename Pattern, typename Patterns>  
    void mark_presence_of_patterns(const Pattern& y, const Patterns& patterns, itemset<tag_dense>& marks)
    {
        marks.clear();
        const size_t m = patterns.size();
        for (size_t k = 0; k < m; ++k) {
            const auto& z = patterns.point(k);
            if (!is_singleton(z) && is_subset(z, y)) marks.insert(k); 
        }
    }

    template<typename Data, typename Patterns, typename DistanceMatrix>
    void compute_pattern_distances(const Data& data, const Patterns& patterns, DistanceMatrix & m) 
    {

        const size_t n = data.size();
        
        m.resize(n, n);

        std::vector<itemset<tag_dense>> cover_space(n);
#pragma omp parallel for shared(cover_space)
        for (size_t i = 0; i < n; ++i)
        {
            mark_presence_of_patterns(data.point(i), patterns, cover_space[i]);   
        }

#pragma omp parallel for shared(cover_space)
        for (size_t i = 0; i < n; ++i)
        {
            m(i, i) = 0;

            for (size_t j = i + 1; j < n; ++j)
            {                
                auto d = distance(cover_space[i], cover_space[j]);
                m(i, j) = d;
                m(j, i) = d;
            }
        }
    }

    template<typename Trait>
    void dbscan_desc1(Composition<Trait>& c, const ndarray<float, 2>& d, double eps, size_t min_n, const DecompositionSettings& cfg) {

        auto ys = dbscan::dbscan(d, eps, min_n);
        for (size_t i = 0, n = ys.size(); i < n; ++i) {
            // dbscan marks unidentified points with -1
            c.data.label(i) = ys[i] + 1;
        }

        initialize_composition(c, cfg);
        simplify_labels(c.data);
        c = reassign_components(std::move(c), cfg, cfg.max_em_iterations.value_or(100));
        c = discover_patternsets(std::move(c), cfg);
    }

    template<typename Trait>
    auto dbscan_desc1_gridsearch(Composition<Trait>& c, const ndarray<float, 2>& d, const DecompositionSettings& cfg) 
    {
        nonstd::optional<std::pair<size_t, double>> best_eps;

        constexpr double eps_param[] {.01, .05, .1, .2, .3, .4, .5};

        c.initial_encoding = c.encoding;
        c.encoding.of_data = std::numeric_limits<typename Trait::float_type>::max();
        c.encoding.of_model = 0;

        for (size_t i = 0; i < std::size(eps_param); ++i) {
            auto test = c;

            dbscan_desc1(test, d, eps_param[i], 5, cfg);

            if (test.encoding.objective() < c.encoding.objective()) {
                c = std::move(test);
                best_eps = {i, eps_param[i]};
            }
        } 
        return best_eps;
    }

    template<typename Trait>
    auto dbscan_desc(Composition<Trait>& c, const DecompositionSettings& cfg, bool use_pattern_distance = false) 
    {
        initialize_composition(c, cfg);
        c = discover_patternsets(std::move(c), cfg);
        ndarray<float, 2> d;
        if(use_pattern_distance)
            compute_pattern_distances(c.data, c.summary, d);
        else 
            compute_distances(c.data, d);
        dbscan_desc1_gridsearch(c, d, cfg);
    }

    template<typename Trait>
    auto dbscan_desc1_gridsearch_measured(Composition<Trait>& c, const ndarray<float, 2>& d, const DecompositionSettings& cfg, 
        std::array<std::chrono::milliseconds, 7>& elapsed
    ) 
    {
        constexpr std::array<double, 7> eps_param{{.01, .05, .1, .2, .3, .4, .5}};

        c.initial_encoding = c.encoding;
        c.encoding.of_data = std::numeric_limits<typename Trait::float_type>::max();
        c.encoding.of_model = 0;

        std::pair<size_t, double> best_eps { std::size(eps_param) + 1 , 1};
        auto best_candidate = c;

        for (size_t i = 0; i < std::size(eps_param); ++i) {
            elapsed[i] = measure([&] { 
                auto test = c;

                dbscan_desc1(test, d, eps_param[i], 5, cfg);

                if (test.encoding.objective() < best_candidate.encoding.objective()) {
                    best_candidate = std::move(test);
                    best_eps = std::pair{i, eps_param[i]};
                }
            });
        } 

        c = std::move(best_candidate);
        return best_eps;
    }

    template<typename Trait>
    auto dbscan_desc_measured(Composition<Trait>& c, const DecompositionSettings& cfg, bool use_pattern_distance = false) 
    {
        using ms = std::chrono::milliseconds;

        ndarray<float, 2> d;
        auto time = measure([&] {
            initialize_composition(c, cfg);
            c = discover_patternsets(std::move(c), cfg);

            if (use_pattern_distance)
                compute_pattern_distances(c.data, c.summary, d);
            else 
                compute_distances(c.data, d);
        });

        std::array<ms, 7> times {{}};
        auto [best, eps] = dbscan_desc1_gridsearch_measured(c, d, cfg, times);

        auto time_best = time + times[best];
        auto time_total = time + std::accumulate(begin(times), end(times), ms(), std::plus<>());

        return std::make_tuple(eps, time_best, time_total);
    }
}