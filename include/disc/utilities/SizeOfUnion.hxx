#include <disc/storage/Itemset.hxx>
namespace sd
{
template <typename Model_Type, typename T>
size_t union_size(const Model_Type& m, const T& x)
{
    thread_local disc::itemset<typename Model_Type::pattern_type> joined_cover;
    joined_cover.clear();
    joined_cover.insert(x);
    for (size_t j = 0; j < m.itemsets.set.size(); ++j)
    {
        const auto& t = m.itemsets.set[j].point;
        if (intersects(x, t))
            joined_cover.insert(t);
    }
    return count(joined_cover);

    // size_t cnt = count(x);
    // for (size_t j = 0; j < m.size(); ++j)
    // {
    //     const auto& t = m.point(j);
    //     if(!intersects(x, t)) continue;

    //     cnt += count(t) - size_of_intersection(t, x);
    //     for (size_t k = j; k-- > 0;)
    //     {
    //         const auto& s = m.point(k);
    //         // if(!intersects(x, s)) continue;
    //         cnt -= size_of_intersection(s, t);
    //     }
    // }
    // return cnt;
}
} // namespace sd