cmake_minimum_required(VERSION 3.6)

project(ZED_Export_Sensors)

find_package(ZED 4 REQUIRED)
find_package(CUDA ${ZED_CUDA_VERSION} REQUIRED)

include_directories(${ZED_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(include)

link_directories(${ZED_LIBRARY_DIR})
link_directories(${CUDA_LIBRARY_DIRS})

add_executable(${PROJECT_NAME} src/main.cpp)
set(LIBS ${ZED_LIBRARIES} )

target_link_libraries(${PROJECT_NAME} ${LIBS})