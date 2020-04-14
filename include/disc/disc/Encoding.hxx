#pragma once

#include <disc/disc/BIC.hxx>
#include <disc/disc/Composition.hxx>
#include <disc/disc/MDL.hxx>
#include <disc/disc/PatternsetResult.hxx>

namespace sd::disc
{

template <typename Distribution_Type, typename Data_Type>
auto log_likelihood(Distribution_Type const& model, const Data_Type& data)
{
    using float_type = typename Distribution_Type::float_type;

    assert(!data.empty());

    // resillience against unlikely numeric issues using
    // laplace smoothing
    const auto l = float_type(1) / (model.model.dimension() + data.size());
    const auto u = float_type(1) - l;

    float_type acc = 0;
#pragma omp parallel for reduction(+ : acc)
    for (size_t i = 0; i < data.size(); ++i)
    {
        auto p = model.expected_generalized_frequency(point(data[i]));

        assert(!std::isinf(p));
        assert(!std::isnan(p));
        assert(p > 0);
        assert(p <= 1);

        if (!(p > 0) || !(p < 1))
            p = std::clamp(p, l, u);

        acc -= std::log2(p);
    }

    return acc;
}

template <typename distribution_type, typename data_type>
auto encode_subsets(const std::vector<distribution_type>& s, const data_type& data)
{
    using float_type = typename distribution_type::float_type;
    float_type l     = 0;
    for (size_t i = 0; i < s.size(); ++i)
    {
        l += log_likelihood(s[i], data.subset(i));
    }
    return l;
}

template <typename distribution_type, typename data_type, typename value_type>
auto encode_subsets(const std::vector<distribution_type>& s,
                    const data_type&                      data,
                    std::vector<value_type>&              per_subset)
{
    assert(per_subset.size() == s.size());
    assert(data.num_components() == s.size());

    value_type l = 0;
    for (size_t i = 0; i < data.num_components(); ++i)
    {
        per_subset[i] = log_likelihood(s[i], data.subset(i));
        l += per_subset[i];
    }
    return l;
}

template <typename Trait>
auto encode_data(Composition<Trait>& c, bool do_not_update = false)
{
    if (do_not_update)
    {
        return encode_subsets(c.models, c.data);
    }
    else
    {
        return encode_subsets(c.models, c.data, c.subset_encodings);
    }
}

template <typename Trait>
auto encode_data(const Composition<Trait>& c)
{
    return encode_subsets(c.models, c.data);
}

template <typename Trait>
auto encode_data(const PatternsetResult<Trait>& c)
{
    return log_likelihood(c.model, c.data);
}

template <typename C, typename... X>
auto encode_model(const C& c, const bool use_bic, const MEDistribution<X...>&)
{
    return use_bic ? bic::encode_model_bic(c) : mdl::encode_model_mdl(c);
}

template <typename C, typename Config, typename... X>
auto encode_model(const C& c, const Config& cfg, const MEDistribution<X...>& tag)
{
    return encode_model(c, cfg.use_bic, tag);
}

template <typename C, typename Config>
auto encode_model(C&& c, const Config& cfg)
{
    if constexpr (meta::has_components_member_fn<decltype(c.data)>::value)
    {
        return encode_model(c, cfg, c.models[0]);
    }
    else
    {
        return encode_model(c, cfg, c.model);
    }
}

template <typename Trait, typename Config>
auto encode(Composition<Trait>& c, const Config& cfg, bool do_not_update = false)
    -> EncodingLength<typename Trait::float_type>
{
    return {encode_data(c, do_not_update), encode_model(c, cfg)};
}

template <typename C, typename Config>
auto encode(C const& c, const Config& cfg) -> EncodingLength<typename C::float_type>
{
    return {encode_data(c), encode_model(c, cfg)};
}

auto additional_cost_bic(size_t component_size) { return std::log2(component_size) / 2.; }
auto constant_bic_cost_once(size_t n, size_t k) { return std::log2(n) * k / 2.; }

template <typename C, typename X>
auto constant_mdl_cost(const C& c, const X& x)
{
    auto [l, length] = mdl::encode_pattern_by_singletons<typename C::float_type>(
        x, c.data.size(), c.summary.labels());
    l += universal_code(length);
    return l;
}
auto additional_cost_mdl(size_t support)
{
    return universal_code(support); // encode-per-component-support
}

template <typename X, typename... Args>
auto additional_cost_bic(const X&, size_t component_size, const MEDistribution<Args...>&)
{
    return additional_cost_bic(component_size);
}

template <typename... Args>
auto additional_cost_mdl(size_t support, const MEDistribution<Args...>&)
{
    return additional_cost_mdl(support);
}

} // namespace sd::disc