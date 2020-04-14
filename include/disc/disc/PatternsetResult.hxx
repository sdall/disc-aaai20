#pragma once

#include <disc/disc/Composition.hxx>
#include <disc/disc/Frequencies.hxx>
#include <disc/disc/InsertMissingSingletons.hxx>
#include <disc/disc/Settings.hxx>
#include <disc/distribution/Distribution.hxx>
#include <disc/storage/Dataset.hxx>

namespace sd::disc
{

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
    distribution_type                        model;

    template <typename DATA,
              typename = std::enable_if_t<
                  !std::is_same_v<std::decay_t<DATA>, PatternsetResult<Trait>>>>
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
        c.models           = {model};
        c.subset_encodings = {encoding.of_data};

        auto& a = c.assignment.emplace_back();
        for (size_t i = 0; i < summary.size(); ++i)
        {
            a.insert(i);
        }

        compute_frequency_matrix(c.data, c.summary, c.frequency);
        return c;
    }
};

template <typename Trait>
typename Trait::distribution_type& initialize_model(PatternsetResult<Trait>& com,
                                                    const MiningSettings&    cfg = {})
{
    auto& data    = com.data;
    auto& summary = com.summary;

    if (cfg.with_singletons)
    {
        insert_missing_singletons(data, summary);
    }

    com.model = make_distribution(com, cfg);
    auto& pr  = com.model;

    for (const auto& i : summary)
    {
        pr.insert(label(i), point(i), false);
    }
    estimate_model(pr);

    // com.initial_encoding = encode(com, cfg);

    return pr;
}

} // namespace sd::disc