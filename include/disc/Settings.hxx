#pragma once

#include <desc/Settings.hxx>

namespace sd::disc
{

struct DiscConfig : public Config
{
    bool use_bic         = true;
};

} // namespace sd::disc