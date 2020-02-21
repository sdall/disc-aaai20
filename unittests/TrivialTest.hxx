#pragma once

#include <iostream>
#include <stdexcept>

namespace sd
{

struct failed_test_error : public std::logic_error
{
    using std::logic_error::logic_error;
};

void test(bool cond, const char* msg = "test failed!")
{
    if (!cond)
    {
        std::cerr << msg << std::endl;
        throw failed_test_error{msg};
    }
}
} // namespace sd

#define QUOTE(name) #name
#define STR(macro) QUOTE(macro)
#define TEST(_Expression)                                                                      \
    sd::test(_Expression, "[" __FILE__ ":" STR(__LINE__) "] " #_Expression)
#define TESTCASE(_Expression)                                                                  \
    [&] {                                                                                      \
        try                                                                                    \
        {                                                                                      \
            TEST(_Expression);                                                                 \
        }                                                                                      \
        catch (...)                                                                            \
        {                                                                                      \
            return true;                                                                       \
        }                                                                                      \
        return false;                                                                          \
    }()
//__DATE__ " " __TIME__ " ; "

// #define TEST_CASE(STR__)
// #define SUBCASE(STR__)
// #define CHECK(...) TEST((__VA_ARGS__))
