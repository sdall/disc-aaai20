#pragma once

#include <disc/disc/Composition.hxx>
#include <disc/disc/Settings.hxx>
#include <disc/distribution/Distribution.hxx>
#include <disc/storage/Dataset.hxx>
#include <disc/disc/Frequencies.hxx>

#include <nonstd/optional.hpp>

namespace sd::disc {

template <typename Trait>
struct PatternsetResult
{
    using pattern_type      = typename Trait::pattern_type;
    using float_type        = typename Trait::float_type;
    using distribution_type = typename Trait::distribution_type;
    using size_type         = typename Trait::size_type;

    Dataset<pattern_type>                    data;
    LabeledDataset<float_type, pattern_type> summary;
    EncodingLength<float_type>               encoding;
    EncodingLength<float_type>               initial_encoding;

    nonstd::optional<distribution_type> model;

    template <typename DATA>
    PatternsetResult(DATA&& d) : data(std::forward<DATA>(d))
    {
    }

    PatternsetResult()                        = default;
    PatternsetResult(PatternsetResult&&)      = default;
    PatternsetResult(const PatternsetResult&) = default;
    PatternsetResult& operator=(const PatternsetResult&) = default;
    PatternsetResult& operator=(PatternsetResult&&) = default;

    explicit operator Composition<Trait>() const
    {
        Composition<Trait> c;

        c.data.reserve(data.size());
        for (size_t i = 0; i < data.size(); ++i)
        {
            c.data.insert(data.point(i), 0, i);
        }
        c.data.group_by_label();

        c.encoding         = encoding;
        c.initial_encoding = initial_encoding;
        c.summary          = summary;
        if (model.has_value())
            c.models = {model.value()};
        c.subset_encodings = {encoding.of_data};

        auto& a = c.assignment.emplace_back();
        for (size_t i = 0; i < summary.size(); ++i)
        {
            a.insert(i);
        }

        compute_frequency_matrix(c.data, c.summary, c.frequency);
        return c;
    }

    // explicit operator Composition<Trait>() &&
    // {
    //     Composition<Trait> c;

    //     c.data.template col<1>() = std::move(data.template col<0>());
    //     c.data.template col<0>().resize({c.data.template col<1>().size()}, 0);
    //     c.data.template col<2>().resize({c.data.template col<1>().size()}, 0);
    //     std::fill(c.data.template col<2>().begin(), c.data.template col<2>().end(), 0);

    //     c.data.group_by_label();

    //     c.encoding         = encoding;
    //     c.initial_encoding = initial_encoding;
    //     c.summary          = std::move(summary);
    //     if (model.has_value())
    //         c.models = {model.value()};
    //     c.subset_encodings = {encoding.of_data};

    //     auto& a = c.assignment.emplace_back();
    //     for (size_t i = 0; i < summary.size(); ++i)
    //     {
    //         a.insert(i);
    //     }

    //     compute_frequency_matrix(c.data, c.summary, c.frequency);
    //     return c;
    // }
};

}