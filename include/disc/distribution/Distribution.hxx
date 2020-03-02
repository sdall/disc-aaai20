#pragma once

#include <disc/distribution/InferProbabilities.hxx>
#include <disc/distribution/IterativeScaling.hxx>
#include <disc/storage/Itemset.hxx>

#include <disc/disc/Settings.hxx>

namespace sd
{
namespace disc
{

template <typename Inherited>
struct Distribution
{
    void   clear() { get().clear(); }
    size_t dimension() const { return get().dimension(); }
    size_t num_itemsets() const { return get().num_itemsets(); }
    size_t size() const { return get().size(); }

    template <typename S, typename T>
    void insert(S label, const T& t, bool estimate = false)
    {
        get().insert(label, t, estimate);
    }

    template <typename S, typename T>
    void insert_singleton(S label, const T& t, bool estimate = false)
    {
        get().insert_singleton(label, t, estimate);
    }

    template <typename pattern_t>
    decltype(auto) expected_frequency(const pattern_t& t) const
    {
        return get().expected_frequency(t);
    }
    template <typename pattern_t>
    decltype(auto) expected_generalized_frequency(const pattern_t& t) const
    {
        return get().expected_generalized_frequency(t);
    }

    template <typename T>
    bool is_item_allowed(const T& t) const
    {
        return get().is_item_allowed(t);
    }

private:
    const Inherited& get() const { return *static_cast<const Inherited*>(this); }
    Inherited&       get() { return *static_cast<Inherited*>(this); }
};

template <typename Model>
struct MEDistributionImpl : Distribution<MEDistributionImpl<Model>>
{
    using underlying_model_type = Model;
    using float_type            = typename underlying_model_type::float_type;
    using pattern_type          = typename underlying_model_type::pattern_type;
    
    MEDistributionImpl(size_t dimension, size_t length)
        : model(dimension)
        , epsilon(std::min(float_type(1e-16), float_type(1) / (length * dimension)))
    {
        assert(dimension > 0);
        assert(length > 0);
    }

    void   clear() { model.clear(); }
    size_t dimension() const { return model.dimension(); }
    size_t num_itemsets() const { return model.num_itemsets(); }
    size_t size() const { return model.size(); }

    template <typename T>
    void insert(float_type label, const T& t, bool estimate = false)
    {
        label = std::clamp<float_type>(label, epsilon, float_type(1.0) - epsilon);
        model.insert(label, t, estimate);
    }
    template <typename T>
    void insert_singleton(float_type label, const T& t, bool estimate = false)
    {
        label = std::clamp<float_type>(label, epsilon, float_type(1.0) - epsilon);
        model.insert_singleton(label, t, estimate);
    }

    template <typename T>
    bool is_item_allowed(const T& t) const
    {
        return model.is_pattern_feasible(t);
    }

    template <typename pattern_t>
    auto probability_transaction(const pattern_t& t) const
    {
        return viva::probability_transaction(model, t);
    }
    template <typename pattern_t>
    auto expected_frequency(const pattern_t& t) const
    {
        return viva::expected_frequency(model, t);
    }
    template <typename pattern_t>
    auto expected_generalized_frequency(const pattern_t& t) const
    {
        return viva::expected_generalized_frequency(model, t);
    }

    underlying_model_type model;

    /// Laplacian Smoothing
    ///     makes sure that the support all distributions is the complete domain.
    ///     prevents both log p or log (1- p) from being -inf.
    float_type epsilon{1e-12};
};

template <typename M, typename T = double>
auto estimate_model(MEDistributionImpl<M>&                   m,
                    viva::IterativeScalingSettings<T> const& opts = {})
{
    return viva::estimate_model(m.model, opts);
}

template <typename U, typename V>
struct MEDistribution : MEDistributionImpl<viva::FactorizedModel<U, V>>
{
    using base = MEDistributionImpl<viva::FactorizedModel<U, V>>;

    MEDistribution(size_t dimension,
                   size_t length,
                   size_t max_num_itemsets = 5,
                   size_t max_range_size   = 8)
        : base(dimension, length)
    {
        this->model.max_num_itemsets = max_num_itemsets;
        this->model.max_range_size   = max_range_size;
    }

    MEDistribution(size_t dimension, size_t length, const disc::MiningSettings& cfg)
        : MEDistribution(dimension,
                         length,
                         cfg.distribution.max_num_sets_per_factor,
                         cfg.distribution.max_width_per_factor)
    {
    }
};

} // namespace disc
} // namespace sd