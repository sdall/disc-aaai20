#pragma once

#include <desc/Component.hxx>
#include <desc/Composition.hxx>
#include <disc/BIC.hxx>
#include <disc/MDL.hxx>

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
    return std::reduce(
        std::execution::par_unseq,
        data.begin(),
        data.end(),
        float_t(),
        [&](auto acc, const auto& x) { return acc - model.log_expectation(point(x)); });
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

template <typename C>
auto encode(C const& c, bool bic = true) -> EncodingLength<typename C::float_type>
{
    return {encode_data(c), bic ? bic::encode_model_bic(c) : mdl::encode_model_mdl(c)};
}

} // namespace sd::disc