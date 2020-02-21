#pragma once

#include <disc/disc/Composition.hxx>
#include <disc/storage/Dataset.hxx>
#include <disc/utilities/Support.hxx>
#include <marray/marray.hxx>

namespace sd
{
namespace disc
{

template <typename T>
T smooth(const T& fr, size_t n, size_t m = 1, size_t a = 1)
{
    return T(fr * n + a) / (n + m * a);
}

template <typename T, typename S>
void apply_smoothing(const PartitionedData<S>& data, sd::ndarray<T, 2>& fr)
{
    for (size_t i = 0; i < fr.length(); ++i)
    {
        for (size_t j = 0; j < fr.extent(1) - 1; ++j)
        {
            fr(i, j) = smooth(fr(i, j), data.subset(j).size(), data.dim);
        }
        fr(i, fr.extent(1) - 1) = smooth(fr(i, fr.extent(1) - 1), data.size(), data.dim);
    }
}

template <typename T, typename S>
void compute_frequency_matrix_column(const PartitionedData<S>&   data,
                                     const LabeledDataset<T, S>& summary,
                                     const size_t                comp_index,
                                     sd::ndarray<T, 2>&          fr,
                                     bool                        laplace_smoothing = false)
{
    for (size_t i = 0; i < fr.extent(0); ++i)
        fr(i, comp_index) = 0;

    auto component = data.subset(comp_index);
    for (auto&& x : component)
    {
        iterate_over(point(x), [&](auto i) { fr(i, comp_index) += 1; });
        // for (size_t i = data.dim; i < summary.size(); ++i)
        for (size_t i = 0; i < summary.size(); ++i)
        {
            if (!is_singleton(summary.point(i)) && is_subset(summary.point(i), point(x)))
            {
                fr(i, comp_index) += 1;
            }
        }
    }

    for (size_t i = 0; i < summary.size(); ++i)
    {
        fr(i, comp_index) /= component.size();
        if (laplace_smoothing)
            fr(i, comp_index) = smooth(fr(i, comp_index), component.size(), data.dim);
    }
}

template <typename T, typename S>
void compute_frequency_matrix(const PartitionedData<S>&   data,
                              const LabeledDataset<T, S>& summary,
                              sd::ndarray<T, 2>&          fr,
                              bool                        laplace_smoothing = false)
{
    size_t n_cols = data.num_components();
    if (n_cols > 1)
        n_cols += 1;

    fr.clear();
    fr.resize(sd::layout<2>({summary.size(), n_cols}), 0.0);

    if (fr.extent(1) == 1)
    {
        std::copy_n(summary.labels().begin(), summary.size(), fr[0].begin());
    }
    else
    {
        for (size_t j = 0; j < n_cols - 1; ++j)
        {
            compute_frequency_matrix_column(data, summary, j, fr, laplace_smoothing);
        }
        for (size_t i = 0; i < summary.size(); ++i)
        {
            fr(i, n_cols - 1) = summary.label(i);
            if (laplace_smoothing)
                fr(i, n_cols - 1) = smooth(fr(i, n_cols - 1), data.size(), data.dim);
        }
    }
}

template <typename T, typename S>
sd::ndarray<T, 2> make_frequencies(const PartitionedData<S>&   data,
                                   const LabeledDataset<T, S>& summary,
                                   bool                        laplace_smoothing = false)
{
    sd::ndarray<T, 2> fr;
    compute_frequency_matrix(data, summary, fr);
    if (laplace_smoothing)
        apply_smoothing(data, fr);
    return fr;
}

} // namespace disc
} // namespace sd