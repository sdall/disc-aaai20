#pragma once

#include "flat_hash_map/flat_hash_map.hpp"
#include <vector>

namespace sd
{
/**
 * The class implements the Trie data type.
 */
template <typename K, typename V, template <class...> class TMap = ska::flat_hash_map>
class Trie
{
public:
    using Key   = K;
    using Value = V;
    using Map   = TMap<Key, Trie>;

    std::size_t size() const { return count; }
    bool        empty() const { return count == 0 && !is_value; }
                operator bool() const { return empty(); }
    // bool        contains_value() const { return is_value; }
    void clear() { *this = {}; }

    /**
     * query a sequence starting from start position and returns a reference to the
     * corresponding value in the trie.
     * If the sequence is not contained in the Trie, a node is added using the default
     * constructor of Value.
     */
    template <typename Sequence>
    Value& operator[](const Sequence& s)
    {
        bool  inserted = false;
        auto& v        = query_and_insert(s, 0, inserted);
        if (inserted)
            count++;
        return v;
    }

    template <typename Sequence>
    const Value& operator[](const Sequence& s) const
    {
        return at(s);
    }

    template <typename Sequence>
    Value& operator()(const Sequence& s)
    {
        return operator[](s);
    }

    template <typename Sequence>
    const Value& operator()(const Sequence& s) const
    {
        return operator[](s);
    }

    template <typename Sequence>
    void insert(const Sequence& s, const Value& value = Value{})
    {
        operator[](s) = value;
    }

    template <typename Sequence>
    void insert(std::pair<const Sequence&, const Value&> p)
    {
        operator[](p.first) = p.second;
    }

    template <typename Sequence>
    Value& at(const Sequence& s)
    {
        return query(s, 0);
    }

    template <typename Sequence>
    const Value& at(const Sequence& s) const
    {
        return query(s, 0);
    }

    template <typename Sequence, typename Visitor>
    void prefix(const Sequence& s, std::size_t start, Visitor& visit)
    {
        auto& child = children.at(s[start]);
        visit(child.value);
        if (s.size() < start)
            child.prefix(s, start + 1, visit);
        // try {
        //     auto& child = children.at(s[start]);
        //     visit(child.value);
        //     if (s.size() < start) child.prefix(s, start + 1, visit);
        // } catch (std::out_of_range& e) {
        //     throw std::out_of_range("Trie::query: sequence is not contained in Trie.");
        // }
    }

    template <typename Sequence>
    Value& query_and_insert(const Sequence& s, std::size_t start, bool& inserted)
    {
        if (s.size() == start)
        {
            is_value = true;
            return value;
        }

        if (inserted)
        {
            return children[s[start]].query_and_insert(s, start + 1, inserted);
        }

        const auto oldSize = children.size();
        auto&      child   = children[s[start]];
        if (children.size() > oldSize)
        {
            inserted |= true;
        }
        return child.query_and_insert(s, start + 1, inserted);
    }

    template <typename Sequence>
    Value& query(const Sequence& s, std::size_t start)
    {
        return s.size() == start ? value : children.at(s[start]).query(s, start + 1);
    }

    template <typename Sequence>
    const Value& query(const Sequence& s, std::size_t start) const
    {
        return s.size() == start ? value : children.at(s[start]).query(s, start + 1);
    }

    /** TRUE if the trie contains the sequence from start position on */
    template <typename Sequence>
    bool contains(const Sequence& s) const
    {
        try
        {
            at(s);
        }
        catch (std::out_of_range& e)
        {
            return false;
        }
        return true;
    }

    template <typename VISITOR>
    void visit_prefixes_df(VISITOR&& call) const
    {
        std::vector<Key> buf;
        buf.reserve((children.size() == 0u || size() == 0) ? 1 : size() / children.size());
        visit_prefixes_df(buf, call);
    }

    template <typename Sequence, typename VISITOR>
    void visit_prefixes_df(Sequence& buf, VISITOR&& call) const
    {
        for (auto& child : children)
        {
            buf.push_back(child.first);
            call(buf, child.second.value);
            child.second.visit_prefixes_df(buf, call);
            buf.pop_back();
        }
    }

    template <typename VISITOR>
    void visit_values(VISITOR&& call) const
    {
        std::vector<Key> buf;
        buf.reserve((children.size() == 0u || size() == 0) ? 1 : size() / children.size());
        visit_values(buf, call);
    }

    template <typename Sequence, typename VISITOR>
    void visit_values(Sequence& buf, VISITOR&& call) const
    {
        for (auto& child : children)
        {
            buf.push_back(child.first);
            if (child.second.is_value)
                call(buf, child.second.value);
            child.second.visit_values(buf, call);
            buf.pop_back();
        }
    }

private:
    Value value{};
    /** true if this is a word */
    bool is_value{false};
    /** number of elements */
    std::size_t count{0};
    /** Holds the children */
    Map children;
};
} // namespace sd
