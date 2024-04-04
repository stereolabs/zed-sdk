cmake_minimum_required(VERSION 3.5)
PROJECT(ZED_Global_Localization)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_BUILD_TYPE "Release")

find_package(ZED 4 REQUIRED)
find_package(CUDA ${ZED_CUDA_VERSION} REQUIRED)
IF(NOT MSVC)
    SET(SPECIAL_OS_LIBS "pthread")
ENDIF()

# Include external lib header:
include_directories(${ZED_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} main.cpp)
set(LIBS ${ZED_LIBRARIES} )
target_link_libraries(${PROJECT_NAME} ${LIBS} ${SPECIAL_OS_LIBS})