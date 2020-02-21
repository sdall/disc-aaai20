#pragma once

#include <chrono>

namespace sd::disc
{

template <typename Fn>
auto measure(Fn&& fn)
{
    namespace ch = std::chrono;
    // using namespace std::chrono_literals;

    auto before = ch::high_resolution_clock::now();
    std::forward<Fn>(fn)();
    auto after = ch::high_resolution_clock::now();
    // return (after - before) / 1ms;
    return ch::duration_cast<ch::milliseconds>(after - before);
}

} // namespace sd::disc
