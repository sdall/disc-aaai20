#pragma once

#include <algorithm>
#include <vector>

#if DBSCAN_PARALLEL
#include <atomic>
#endif

namespace sd::dbscan
{

template<class DistanceMatrix>
void find_neighbors(const DistanceMatrix& D, size_t k, double eps, std::vector<size_t>& ne)
{
    // inefficient implementation; consider replacing D with a kd- or vp-tree.
    ne.clear();

//     if (D.length() > 2'000'000) 
//     {
// #pragma omp parallel for// if (D.length() > 500'000) //  reduction(insert: ne) 
//         for (size_t j = 0; j < D.length(); ++j)
//         {
//             if (D(k, j) <= eps)
//             {
// #pragma omp critical
//         {
//                 ne.push_back(j);
//         }
//             }
//         }
//     }
//     else {
        for (size_t j = 0; j < D.length(); ++j)
        {
            if (D(k, j) <= eps)
            {
                ne.push_back(j);
            }
        }
    // }
}

template<class DistanceMatrix>
void dbscan(const DistanceMatrix& dist, std::vector<int>& labels, double eps, size_t min_elems)
{
    const size_t len = dist.length();

#if DBSCAN_PARALLEL
    constexpr const bool parallel = true;
    std::vector<std::atomic_bool> visited(len);
    std::atomic_size_t cur_label = 0;
#else
    // constexpr const bool parallel = false;
    std::vector<uint8_t> visited(len);
    size_t cur_label = 0;
#endif
    
    std::vector<size_t> n, n_j;

    n.reserve(len);
    n_j.reserve(len);

// #pragma omp parallel for private(n, n_j) if (parallel)
    for (size_t k = 0; k < len; ++k)
    {
        if (!visited[k])
        {
            visited[k] = 1;
            find_neighbors(dist, k, eps, n);
            if (n.size() >= min_elems)
            {
                labels[k] = cur_label;
                for (size_t i = 0; i < n.size(); ++i)
                {
                    const size_t j = n[i];
                    if (!visited[j])
                    {
                        visited[j] = 1;
                        find_neighbors(dist, j, eps, n_j);
                        if (n_j.size() >= min_elems)
                        {
                            for (const auto& n1 : n_j)
                            {
                                n.push_back(n1);
                            }
                        }
                    }
// #pragma omp critical 
// {
                    if (labels[j] == -1)
                    {
                        labels[j] = cur_label;
                    }
// }
                }

                ++cur_label;
            }
        }
    }
}

// 'eps' is the search space for neighbors in the range [0,1], where 0.0 is exactly self
// and 1.0 is entire dataset
template<class DistanceMatrix>
std::vector<int> dbscan(const DistanceMatrix& D, double eps, size_t min_elems)
{
    std::vector<int> labels(D.length(), -1);
    dbscan(D, labels, eps, min_elems);
    return labels;
}


} // namespace clustering