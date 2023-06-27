#include "exporter/TimestampUtils.h"

#include <chrono>

#include <stdio.h>
#include <time.h>

/**
 * @brief Get the current datetime in human readable string
 *
 * @return std::string current_data in human readable string
 */
std::string getCurrentDatetime()
{
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    // Visit http://en.cppreference.com/w/cpp/chrono/c/strftime
    // for more information about date/time format
    strftime(buf, sizeof(buf), "%d-%m-%Y_%H-%M-%S", &tstruct);

    return buf;
}