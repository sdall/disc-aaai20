#pragma once

#include <disc/storage/Dataset.hxx>
#include <marray/marray.hxx>

#include <disc/disc/Settings.hxx>
#include <disc/distribution/Distribution.hxx>

namespace sd
{
namespace disc
{

using AssignmentMatrix = std::vector<sparse_dynamic_bitset<size_t>>;

template <typename T>
struct EncodingLength
{
    T    of_data{std::numeric_limits<T>::max()};
    T    of_model{0};
    auto objective() const { return of_data + of_model; }
};

template <typename Pattern_Type, typename Float_Type, typename Distribution_Type>
struct Trait
{
    using pattern_type      = Pattern_Type;
    using float_type        = Float_Type;
    using distribution_type = Distribution_Type;
    using size_type         = size_t;
};

using DefaultTrait = Trait<tag_dense, double, MEDistribution<tag_dense, double>>;

template <typename Trait>
struct Composition
{
    using pattern_type      = typename Trait::pattern_type;
    using float_type        = typename Trait::float_type;
    using distribution_type = typename Trait::distribution_type;
    using size_type         = typename Trait::size_type;

    PartitionedData<pattern_type>            data;
    LabeledDataset<float_type, pattern_type> summary;
    AssignmentMatrix                         assignment;
    sd::ndarray<float_type, 2>               frequency;
    EncodingLength<float_type>               encoding;
    EncodingLength<float_type>               initial_encoding;
    std::vector<distribution_type>           models;
    std::vector<float_type>                  subset_encodings;
};

template <typename T>
bool check_invariant(const Composition<T>& comp)
{
    return ( // comp.summary.size() >= comp.data.dim &&
        comp.frequency.extent(0) == comp.summary.size() &&
        comp.frequency.extent(1) >= comp.data.num_components() &&
        comp.assignment.size() == comp.data.num_components() &&
        comp.models.size() == comp.assignment.size() &&
        comp.subset_encodings.size() == comp.models.size());
}

template <typename Trait>
void initialize_composition(Composition<Trait>& c, const MiningSettings& cfg)
{
    c.data.group_by_label();

    if (cfg.with_singletons)
    {
        insert_missing_singletons(c.data, c.summary);
    }

    c.subset_encodings.resize(c.data.num_components());

    characterize_no_mining(c, cfg);

    assert(check_invariant(c));
}

template <typename Trait, typename S, typename T>
Composition<Trait> make_composition(PartitionedData<S>&&   data,
                                    LabeledDataset<T, S>&& summary,
                                    const MiningSettings&  cfg)
{
    Composition<Trait> c;
    c.data    = std::forward<PartitionedData<S>>(data);
    c.summary = std::forward<LabeledDataset<T, S>>(summary);
    initialize_composition(c, cfg);
    return c;
}

template <typename Trait, typename S>
Composition<Trait> make_composition(PartitionedData<S>&& data, const MiningSettings& cfg)
{
    using T = typename Trait::float_type;
    return make_composition<Trait, S, T>(std::forward<PartitionedData<S>>(data), {}, cfg);
}

template <typename Trait, typename S>
Composition<Trait> make_composition(PartitionedData<S> data, const MiningSettings& cfg)
{
    using T = typename Trait::float_type;
    return make_composition<Trait, S, T>(std::move(data), {}, cfg);
}

} // namespace disc
} // namespace sd