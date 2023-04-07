#pragma once

#include "sl/Camera.hpp"
#include "json.hpp"


namespace sk {
    nlohmann::json serialize(sl::Bodies&);
    sl::Bodies deserialize(nlohmann::json &);

}
