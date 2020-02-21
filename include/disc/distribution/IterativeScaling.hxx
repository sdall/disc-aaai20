#pragma once

#include <disc/distribution/InferProbabilities.hxx>

#include <algorithm>
#include <cmath>
#include <vector>

namespace sd
{
namespace viva
{

// struct ISStandardDistribution
// {
//     template <typename Model, typename Query>
//     auto expected_frequency(const Model& model, const Query& x) const
//     {
//         return viva::expected_frequency(model, x);
//     }

//     template <typename Model, typename Blocks, typename Query>
//     auto expected_frequency(const Model& model, const Blocks& blocks, const Query& x) const
//     {
//         return expected_frequency_known(blocks, model, x);
//     }

//     template <typename Model, typename Query, typename Blocks>
//     void generate_counts(const Model& model, const Query& x, bool known, Blocks& blocks)
//     const
//     {
//         compute_transactions(model, x, known, blocks);
//     }
// };

template <typename float_type>
struct IterativeScalingSettings
{
    float_type sensitivity   = 1e-18;
    float_type epsilon       = 1e-16;
    size_t     max_iteration = 500;
    bool       warmstart     = false;
    bool       normalize     = false;
};

template <typename T>
bool bad_condition_number(const T& p, const T& q, const T& z)
{
    const auto cond_num = std::abs(z * (q / p) - z) / z;
    return std::isinf(cond_num) || std::isnan(cond_num) || T(1) / cond_num < 1e-4;
}

template <typename T>
bool bad_scaling_factor(const T& a, const T& z)
{
    const auto cond_num = z * a;
    return std::isinf(cond_num) || std::isnan(cond_num) || cond_num <= 0 ||
           (cond_num != cond_num);
}

template <typename T>
bool bad_scaling_factor(const T& p, const T& q, const T& z)
{
    const auto cond_num = z * (q / p);
    return std::isinf(cond_num) || std::isnan(cond_num) || cond_num <= 0 ||
           (cond_num != cond_num);
}

template <typename model_type, typename float_type, typename AllTransactions>
void normalize(model_type& m, float_type& theta0, const AllTransactions& all_transactions)
{
#if 1
    // float_type sum = 0;
    // float_type sum0 = 0;
    float_type sum1 = 0;
    // float_type sum2 = 0;
    // float_type sum3 = 0;
    // float_type sum4 = 0;
    for (const auto& t : all_transactions)
    {
        // auto p = expected_frequency(m, t.cover);
        auto pp = probability_transaction(m, t.cover);
        // auto g =  probability_of_absent_items(m, t.cover);

        // sum += p;
        // sum0 += t.value * p;
        // if(pp > 2) pp = 1.2;
        sum1 += std::clamp<float_type>(pp, 0.0001, 0.9999);
        // sum2 += t.value * pp;
        // sum3 += p * g;
        // sum4 += t.value * p * g;
    }

    //  std::cout << " sum of probabilities\n";
    //  std::cout << sum  << " ";
    //  std::cout << sum0 << " ";
    //  std::cout << sum1 << " ";
    //  std::cout << sum2 << " ";
    //  std::cout << sum3 << " ";
    //  std::cout << sum4 << "\n";

    // do not normalize if (1) p is already normalized or (2) if p is not even close to be a
    // valid distribution this helps with mitigating numerical issues in the iterative scaling
    // algorithm (not division-free after all)
    if (std::abs(sum1 - float_type(1)) > 1e-4 && !bad_scaling_factor(theta0, sum1) &&
        1e-6 < sum1 && sum1 < 5)
    {
        theta0 /= sum1;

        for (size_t i = 0; i < m.size(); ++i)
        {
            m.coefficient(i) /= sum1;
        }
    }

#else
    float_type sum = 0;
    for (size_t i = 0; i < m.size(); ++i)
    {
        sum += m.probability(i);
    }
    for (size_t i = 0; i < m.size(); ++i)
    {
        m.coefficient(i) /= sum;
    }
    theta0 /= sum;
#endif
}

template <typename model_type, typename float_type, typename AllTransactions>
void normalize_coefficients(model_type&            m,
                            float_type&            theta0,
                            const AllTransactions& all_transactions)
{
    float_type sum = 0;
    for (const auto& t : all_transactions)
    {
        auto p = expected_frequency(m, t.cover);
        sum += p;
    }
    for (size_t i = 0; i < m.size(); ++i)
    {
        m.coefficient(i) /= sum;
    }
}

template <typename model_type, typename Transactions>
void update_precomputed_probabilities(model_type& model, [[maybe_unused]] const Transactions& t)
{
    for (size_t i = 0; i < model.size(); ++i)
    {
        model.probability(i) = expected_frequency_known(t[i], model, model.point(i));
        // model.probability(i) = std::clamp<float_type>(model.probability(i), 1e-16, 1.0 -
        // 1e-16);
    }
}

template <typename U, typename V>
void reset_coefficients(MultiModel<U, V>& model)
{
    for (auto& x : model.itemsets.set)
        x.theta = x.frequency;
    for (auto& x : model.singletons.set)
        x.theta = x.frequency;
    model.singletons.theta0 = 1;
    // model.itemsets.theta0   = 1;
    // model.itemsets.theta0   = std::exp2(-V(model.dimension()));
    // model.itemsets.theta0   = std::exp2(-V(model.singletons.size()));
    model.itemsets.theta0   = std::exp2(-V(dimension_of_factor(model)));
}

template <typename Model, typename Transactions, typename AllTransactions, typename F>
auto iterative_scaling(Model&                           model,
                       const std::vector<Transactions>& transactions,
                       const AllTransactions&           all_transactions,
                       IterativeScalingSettings<F>      options)
{
    using float_type = typename Model::float_type;

    if (!options.warmstart)
        reset_coefficients(model);

    float_type last_dif = std::numeric_limits<float_type>::max();

    for (size_t it = 0; it < options.max_iteration; ++it)
    {
        if (options.normalize && !all_transactions.empty())
            normalize(model, model.normalizer(), all_transactions);

        float_type dif = 0;

        for (size_t i = 0; i < model.size(); ++i)
        {
            auto q = model.frequency(i);
            auto p = expected_frequency_known(transactions[i], model, model.point(i));
            model.probability(i) = p;

            dif += std::abs(q - p);
            // std::cout << p << "\n";

            // if (std::abs(q - p) < options.sensitivity)
            //     continue;
            // if (bad_scaling_factor(p, q, model.coefficient(i)))
            //     continue;
            // if (bad_condition_number(p, q, model.normalizer())) continue;

            model.coefficient(i) *= q / p;// * ((1 - p) / (1 - q));
            // model.normalizer() *= (1 - q) / (1 - p);
        }

        if (dif < options.sensitivity /*||
            std::abs(dif - last_dif) < options.epsilon*/)
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
        if(m.is_pattern_known(i)) {
            t[i] = m.itemsets.partitions;
        }
        else {
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
