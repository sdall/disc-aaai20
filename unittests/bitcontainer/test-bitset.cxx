#include <TrivialTest.hxx>

#include <disc/storage/Itemset.hxx>

#include <random>

using namespace sd;

template <typename S, typename... Args>
auto& insert(S& s, Args... args)
{
    [[maybe_unused]] auto tmp =
        std::initializer_list<int>{(s.insert(std::forward<Args>(args)), 0)...};
    return s;
}

template <typename S, typename... Args>
auto& assign(S& s, Args... args)
{
    s.clear();
    return insert(s, std::forward<Args>(args)...);
}

template <typename pattern_type>
void run_test()
{
    size_t max = 1000;

    pattern_type bits(max);
    pattern_type testing(max + 200);

    TEST(bits.count() == 0);

    auto superset = assign(bits, 1, 2, 3);
    auto subset   = assign(testing);
    TEST(is_subset(superset, superset));

    TEST(is_subset(assign(subset, 1), superset));
    TEST(is_subset(assign(subset, 2), superset));
    TEST(is_subset(assign(subset, 3), superset));
    TEST(is_subset(assign(subset, 1, 2), superset));
    TEST(is_subset(assign(subset, 2, 3), superset));
    TEST(is_subset(assign(subset, 1, 3), superset));
    TEST(is_subset(assign(subset, 1, 2, 3), superset));
    TEST(!is_subset(assign(subset, 0), superset));
    TEST(!is_proper_subset(assign(subset, 1, 2, 3), superset));
    TEST(is_subset(assign(subset), superset));
    TEST(is_subset(assign(subset), assign(superset)));
    TEST(!is_subset(assign(subset, 0), assign(superset)));

    {
        pattern_type s, t;
        assign(s, 1, 2, 3, 4, 5);
        assign(t, 2, 3, 6);
        TEST(size_of_intersection(s, t) == 2);
        intersection(s, t);
        TEST(size_of_intersection(s, t) == 2);
        TEST(t.count() == 2);
        TEST(intersects(s, t));

        assign(s, 1, 2, 3, 4, 5);
        assign(t, 6);
        TEST(size_of_intersection(s, t) == 0);
        intersection(s, t);
        TEST(size_of_intersection(s, t) == 0);
        TEST(t.count() == 0);
        TEST(!intersects(s, t));
    }

    {
        sd::dynamic_bitset<size_t> s, t;
        TEST(equal(s, t));

        assign(s, 1);
        assign(t, 1);
        TEST(equal(s, t));

        t.erase(1);
        TEST(!equal(s, t));

        t.insert(1);
        t.resize(10000, false);
        TEST(equal(s, t));

        s.resize(10001, false);
        TEST(equal(s, t));
        s.resize(10000, false);
        TEST(equal(s, t));
        s.resize(500, false);
        TEST(equal(s, t));
        s.resize(2, false);
        TEST(equal(s, t));
    }

    {
        pattern_type s;
        s.insert(4);
        TEST(s.count() == 1);

        s.clear();
        s.insert(5);
        TEST(s.count() == 1);

        pattern_type t;
        t.insert(64 * 4 - 1);

        auto k = last_entry(t);
        auto l = front(t);

        TEST(k == l);
    }

    {
        pattern_type a(5), b(3);
        assign(a, 1, 2, 3);
        assign(b, 2, 3, 4);

        intersection(a, b);
        TEST(!b.contains(0));
        TEST(!b.contains(1));
        TEST(b.contains(2));
        TEST(b.contains(3));
        TEST(!b.contains(4));
        TEST(!b.contains(5));
    }

    {
        pattern_type a;
        insert(a, 1, 2, 3);

        auto b = a;
        b.insert(99);

        insert(a, b);
        TEST(a.contains(99));
        TEST(a.count() == 4);
    }

    {
        pattern_type a;
        assign(a, 1, 3, 4);
        auto b = a;
        // b.resize(512);
        TEST(equal(a, b));
    }

    {
        pattern_type a;
        assign(a, 1, 2, 3, 4, 5, 6, 7, 8, 9);

        auto b = a;
        setminus(b, a);
        TEST(count(b) == 0);
        b = a;
        a.erase(5);
        TEST(a.count() == b.count() - 1);
        setminus(b, a);
        TEST(count(b) == 1);

        size_t cnt = 0;
        foreach(b, [&cnt](...) { cnt++; });
        TEST(cnt == b.count());
    }

    {
        pattern_type        a;
        std::vector<size_t> b{1, 2, 3, 4, 5, 7, 99};

        for (auto i : b)
            a.insert(i);

        for (auto i : b)
            TEST(a.contains(i));

        TEST(a.count() == b.size());
        TEST(!is_subset(999, a));
        TEST(is_subset(99, a));
    }

    {
        pattern_type a, b, c, d;

        assign(a, 1, 2, 3, 4, 5);
        assign(b, 3, 4);
        assign(c, 5);
        assign(d, 99);

        TEST(intersects(a, b));
        TEST(intersects(b, a));
        TEST(!intersects(c, b));
        TEST(!intersects(b, c));
        TEST(intersects(a, c));
        TEST(intersects(c, a));
        TEST(!intersects(d, a));
    }

    for (size_t i = 0; i < max; ++i)
    {
        assign(bits, i);
        assign(testing, i);
        TEST(is_subset(bits, testing));
        TEST(is_subset(i, bits));
        TEST(bits.count() == 1);
    }

    std::minstd_rand                       rng;
    std::uniform_int_distribution<size_t>  iuniform(0, max - 1);
    std::uniform_real_distribution<double> uniform(0, 1);

    for (size_t i = 0; i < 100000; ++i)
    {
        auto n = uniform(rng) * bits.count() / 2;
        bits.clear();
        for (size_t j = 0; j < n; ++j)
        {
            bits.insert(iuniform(rng), true);
        }
        testing = bits;
        testing.insert(iuniform(rng));
        // bits.erase(front(bits));
        // testing.flip(iuniform(rng));
        // TEST(is_subset(bits, testing) || is_subset(testing, bits));
        TEST(is_subset(bits, testing));
    }

    for (size_t i = 0; i < 100000; ++i)
    {
        auto n = uniform(rng) * max / 2;
        bits.clear();
        for (size_t j = 0; j < n; ++j)
        {
            bits.insert(iuniform(rng), true);
        }
        testing = bits;
        testing.erase(iuniform(rng));
        TEST(is_subset(testing, bits));
    }

    size_t                                half = max / 2;
    std::uniform_int_distribution<size_t> iuniform_half(0, half - 1);

    // for (size_t i = 0; i < 100000; ++i)
    // {
    //     auto n = uniform(rng) * max / 2;
    //     bits.clear();
    //     for (size_t j = 0; j < n; ++j) { bits.insert(iuniform_half(rng)); }

    //     // auto view = get_observer_type<size_t, is_sparse(tag_dense{})>({bits.container},
    //     //                                                               bits.length() / 2);
    //     auto view = get_observer_type<size_t, true>{
    //         cpslice<size_t>{bits.container.data(), bits.container.size() / 2}};

    //     TEST(is_subset(view, bits));
    //     TEST(size_of_intersection(view, bits) == count(view));
    // }

    for (size_t i = 0; i < 100000; ++i)
    {
        auto n = uniform(rng) * (max / 2);
        bits.clear();
        for (size_t j = 0; j < n; ++j)
        {
            bits.insert(iuniform(rng));
        }
        auto testing = bits;
        auto idx     = iuniform(rng);
        testing.insert(idx);
        bits.erase(idx);
        setminus(testing, bits);
        TEST(count(testing) <= 1);
        TEST(is_subset(idx, testing));
    }
}

int main(void)
{
    run_test<sd::sparse_dynamic_bitset<size_t>>();
    run_test<sd::dynamic_bitset<size_t>>();
}
