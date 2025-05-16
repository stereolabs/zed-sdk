find_path(GPS_INCLUDE_DIR NAMES gps.h DOC "libgps include directory")
find_library(GPS_LIBRARY NAMES gps DOC "libgps library")

if(GPS_INCLUDE_DIR)
    file(STRINGS ${GPS_INCLUDE_DIR}/gps.h _version_lines REGEX "GPSD_API_(MAJOR|MINOR)_VERSION")
    string(REGEX MATCH "MAJOR_VERSION[ \t]+([0-9]+)" _version_major ${_version_lines})
    set(GPS_VERSION_MAJOR ${CMAKE_MATCH_1})
    string(REGEX MATCH "MINOR_VERSION[ \t]+([0-9]+)" _version_minor ${_version_lines})
    set(GPS_VERSION_MINOR ${CMAKE_MATCH_1})
    set(GPS_VERSION_STRING "${GPS_VERSION_MAJOR}.${GPS_VERSION_MINOR}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GPS
    REQUIRED_VARS GPS_INCLUDE_DIR GPS_LIBRARY
    VERSION_VAR GPS_VERSION_STRING)

if(GPS_FOUND AND NOT TARGET GPS::GPS)
    add_library(GPS::GPS UNKNOWN IMPORTED)
    set_target_properties(GPS::GPS PROPERTIES
        IMPORTED_LOCATION "${GPS_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${GPS_INCLUDE_DIR}")
endif()

mark_as_advanced(GPS_INCLUDE_DIR GPS_LIBRARY)
set(GPS_INCLUDE_DIRS ${GPS_INCLUDE_DIR})
set(GPS_LIBRARIES ${GPS_LIBRARY})
