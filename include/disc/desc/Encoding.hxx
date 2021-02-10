#pragma once

#include <disc/desc/BIC.hxx>
#include <disc/desc/Component.hxx>
#include <disc/desc/Composition.hxx>
#include <disc/desc/MDL.hxx>

#if WITH_EXECUTION_POLICIES 
#include <execution>
#endif

namespace sd::disc
{

template <typename Distribution_Type, typename Data_Type>
auto log_likelihood(Distribution_Type const& model, const Data_Type& data)
{
    using float_t = typename Distribution_Type::float_type;

    assert(!data.empty());

#if WITH_EXECUTION_POLICIES
    return std::reduce(std::execution::par_unseq, data.begin(), data.end(), float_t(), [&](auto acc, const auto& x) {
        return acc - model.log_expectation(point(x));
    });
#else
    // resillience against unlikely numeric issues using laplace smoothing
    // const auto l = std::min<float_t>(1e-10, float_t(1) / (model.model.dim + data.size()));
    // const auto l = float_t(1) / (model.model.dim + data.size());
    float_t acc = 0;
#pragma omp parallel for reduction(+ : acc)
    for (size_t i = 0; i < data.size(); ++i)
    {
        // auto p = model.expectation(point(data[i]));
        // if (!(p > 0) || !(p <= 1))
        // {
        //     p = std::clamp(p, l, 1 - l);
        // }
        acc -= model.log_expectation(point(data[i]));
    }
    return acc;
#endif
}

// template <typename distribution_type, typename data_type>
// auto encode_subsets(const std::vector<distribution_type>& s, const data_type& data)
// {
//     using float_type = typename distribution_type::float_type;
//     float_type l     = 0;
//     for (size_t i = 0; i < s.size(); ++i)
//     {
//         l += log_likelihood(s[i], data.subset(i));
//     }
//     return l;
// }
// template <typename distribution_type, typename data_type, typename value_type>
// auto encode_subsets(const std::vector<distribution_type>& s,
//                     const data_type&                      data,
//                     std::vector<value_type>&              per_subset)
// {
//     assert(per_subset.size() == s.size());
//     assert(data.num_components() == s.size());

//     value_type l = 0;
//     for (size_t i = 0; i < data.num_components(); ++i)
//     {
//         per_subset[i] = log_likelihood(s[i], data.subset(i));
//         l += per_subset[i];
//     }
//     return l;
// }

// template <typename Trait>
// auto encode_data(Composition<Trait>& c)
// {
//      return encode_subsets(c.models, c.data, c.subset_encodings);
// }
template <typename Trait>
auto encode_data(const Composition<Trait>& c)
{
    using float_type = typename Trait::float_type;
    float_type l     = 0;
    for (size_t i = 0; i < c.models.size(); ++i)
    {
        l += log_likelihood(c.models[i], c.data.subset(i));
    }
    return l;
}
template <typename Trait>
auto encode_data(const Component<Trait>& c)
{
    return log_likelihood(c.model, c.data);
}

template <typename C, typename... X>
auto encode_model(const C& c, const Config& cfg, const MaxEntDistribution<X...>&)
{
    return cfg.use_bic ? bic::encode_model_bic(c) : mdl::encode_model_mdl(c);
}
template <typename T>
auto encode_model(const Component<T>& c, const Config& cfg)
{
    return encode_model(c, cfg, c.model);
}
template <typename T>
auto encode_model(const Composition<T>& c, const Config& cfg)
{
    return encode_model(c, cfg, c.models.front());
}

template <typename Trait>
auto encode(Composition<Trait>& c, const Config& cfg)
    -> EncodingLength<typename Trait::float_type>
{
    return {encode_data(c), encode_model(c, cfg)};
}

template <typename C>
auto encode(C const& c, const Config& cfg) -> EncodingLength<typename C::float_type>
{
    return {encode_data(c), encode_model(c, cfg)};
}

// auto additional_cost_bic(size_t component_size) { return std::log2(component_size) / 2.; }

// template <typename X, typename... Args>
// auto additional_cost_bic(const X&, size_t component_size, const MaxEntDistribution<Args...>&)
// {
//     return additional_cost_bic(component_size);
// }

// template <typename... Args>
// auto additional_cost_mdl(size_t support, const MaxEntDistribution<Args...>&)
// {
//     return additional_cost_mdl(support);
// }

} // namespace sd::disc