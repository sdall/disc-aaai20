#pragma once

#include <disc/desc/Composition.hxx>
#include <disc/desc/Support.hxx>
#include <disc/storage/Dataset.hxx>
#include <ndarray/ndarray.hxx>

namespace sd
{
namespace disc
{

template <typename T, typename S>
void compute_frequency_matrix_column(const PartitionedData<S>&   data,
                                     const LabeledDataset<T, S>& summary,
                                     const size_t                comp_index,
                                     sd::ndarray<T, 2>&          fr)
{
    for (size_t i = 0; i < fr.extent(0); ++i)
        fr(i, comp_index) = 0;

    auto component = data.subset(comp_index);
    for (auto&& x : component)
    {
        for (size_t i = 0; i < summary.size(); ++i)
        {
            if (is_subset(summary.point(i), point(x)))
            {
                fr(i, comp_index) += 1;
            }
        }
    }

    for (size_t i = 0; i < summary.size(); ++i)
    {
        fr(i, comp_index) /= component.size();
    }
}

template <typename T, typename S>
void compute_frequency_matrix(const PartitionedData<S>&   data,
                              const LabeledDataset<T, S>& summary,
                              sd::ndarray<T, 2>&          fr)
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
            compute_frequency_matrix_column(data, summary, j, fr);
        }
        for (size_t i = 0; i < summary.size(); ++i)
        {
            fr(i, n_cols - 1) = summary.label(i);
        }
    }
}

template <typename T, typename S>
sd::ndarray<T, 2> make_frequencies(const PartitionedData<S>&   data,
                                   const LabeledDataset<T, S>& summary)
{
    sd::ndarray<T, 2> fr;
    compute_frequency_matrix(data, summary, fr);
    return fr;
}

} // namespace disc
} // namespace sd