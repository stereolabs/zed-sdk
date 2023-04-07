#include "TimestampUtils.h"

/**
 * @brief Get the current datetime in human readable string
 *
 * @return std::string current_data in human readable string
 */
std::string getCurrentDatetime()
{
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    std::ostringstream oss;
    const char *date_format = "%d-%m-%Y_%H-%M-%S";
    oss << std::put_time(&tm, date_format);
    std::string datetime_str = oss.str();
    return datetime_str;
}