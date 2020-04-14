#pragma once

#include <disc/distribution/InferProbabilities.hxx>

#include <algorithm>
#include <cmath>
#include <vector>

namespace sd
{
namespace viva
{

template <typename float_type>
struct IterativeScalingSettings
{
    float_type sensitivity   = 1e-15;
    float_type epsilon       = 1e-18;
    size_t     max_iteration = 200;
    bool       warmstart     = false;
    bool       normalize     = false;
};

template <typename T>
bool bad_scaling_factor(const T& a, const T& z)
{
    const auto r = z * a;
    return std::isinf(r) || std::isnan(r) || r <= 0 || (r != r);
}

template <typename T>
bool bad_scaling_factor(const T& p, const T& q, const T& z)
{
    const auto r = z * (q / p);
    return std::isinf(r) || std::isnan(r) || r <= 0 || (r != r);
}

template <typename model_type, typename Transactions>
void update_precomputed_probabilities(model_type& model, [[maybe_unused]] const Transactions& t)
{
    for (size_t i = 0; i < model.size(); ++i)
    {
        model.probability(i) = expected_frequency_known(t[i], model, model.point(i));
    }
}

template <typename U, typename V>
void reset_normalizer(MultiModel<U, V>& model)
{
    model.singletons.theta0 = 1;
    // model.itemsets.theta0   = 1;
    model.itemsets.theta0 = std::exp2(-V(dimension_of_factor(model)));
}

template <typename U, typename V>
void reset_coefficients(MultiModel<U, V>& model)
{
    for (auto& x : model.itemsets.set)
        x.theta = x.frequency;
    for (auto& x : model.singletons.set)
        x.theta = x.frequency;
    reset_normalizer(model);
}

template <typename Model, typename Transactions, typename AllTransactions, typename F>
auto iterative_scaling(Model&                           model,
                       const std::vector<Transactions>& transactions,
                       const AllTransactions&,
                       IterativeScalingSettings<F> options)
{
    using float_type = typename Model::float_type;

    if (!options.warmstart)
    {
        reset_coefficients(model);
    }
    else
    {
        reset_normalizer(model);
    }

    float_type last_dif = std::numeric_limits<float_type>::max();

    for (size_t it = 0; it < options.max_iteration; ++it)
    {
        // if (options.normalize && !all_transactions.empty())
        //     normalize(model, model.normalizer(), all_transactions);

        float_type dif = 0;

        for (size_t i = 0; i < model.size(); ++i)
        {
            auto q = model.frequency(i);
            auto p = expected_frequency_known(transactions[i], model, model.point(i));
            model.probability(i) = p;

            dif += std::abs(q - p);

            if (std::abs(q - p) < options.sensitivity)
                continue;
            if (bad_scaling_factor(p, q, model.coefficient(i)))
                continue;
            // if (bad_condition_number(p, q, model.normalizer())) continue;

            model.coefficient(i) *= q / p; // * ((1 - p) / (1 - q));
            // model.normalizer() *= (1 - q) / (1 - p);
        }

        if (dif < options.sensitivity || std::abs(dif - last_dif) < options.epsilon)
        {
            last_dif = dif;
            break;
        }

        last_dif = dif;
    }

    // if (options.normalize && !all_transactions.empty())
    //     normalize(model, model.normalizer(), all_transactions);

    return last_dif;
}

template <typename Model, typename T = double>
auto estimate_model(Model& m, IterativeScalingSettings<T> const& opts = {})
{
    using float_type   = typename Model::float_type;
    using pattern_type = typename Model::pattern_type;

    using block_t = Block<pattern_type, float_type>;

    std::vector<std::vector<block_t>> t(m.size());

    // compute_transactions(m, m.point(i), m.is_pattern_known(i),  m.itemsets.partitions);

    m.itemsets.num_singletons = m.singletons.set.size();
    m.itemsets.update_partitions();

    // #pragma omp parallel for shared(t) if (m.size() > 16)
    for (size_t i = 0; i < m.size(); ++i)
    {
        if (m.is_pattern_known(i))
        {
            t[i] = m.itemsets.partitions;
        }
        else
        {
            compute_transactions(m, m.point(i), false, t[i]);
        }

        auto it = std::remove_if(t[i].begin(), t[i].end(), [&](const auto& x) {
            return x.value == 0 || !is_subset(m.point(i), x.cover);
        });
        t[i].erase(it, t[i].end());
    }

    std::vector<block_t> all;
    if (opts.normalize)
    {
        all = m.itemsets.partitions;
    }

    const auto dif = iterative_scaling(m, t, all, opts);

    update_precomputed_probabilities(m, t);

    return dif;
}

template <typename U, typename V, typename W, typename F = double>
void estimate_model(FactorizedModel<U, V, W>& m, IterativeScalingSettings<F> const& opts = {})
{
#pragma omp parallel for
    for (size_t i = 0; i < m.factors.size(); ++i)
    {
        estimate_model(m.factors[i].factor, opts);
    }
}

} // namespace viva
} // namespace sd
