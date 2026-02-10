CMAKE_MINIMUM_REQUIRED(VERSION 3.5)
PROJECT(ZED_One_SVO_External_Data)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_BUILD_TYPE "RelWithDebInfo")

option(LINK_SHARED_ZED "Link with the ZED SDK shared executable" ON)

if (NOT LINK_SHARED_ZED AND MSVC)
    message(FATAL_ERROR "LINK_SHARED_ZED OFF : ZED SDK static libraries not available on Windows")
endif()

find_package(ZED REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDA ${ZED_CUDA_VERSION} REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${ZED_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

link_directories(${ZED_LIBRARY_DIR})
link_directories(${OpenCV_LIBRARY_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})

SET(ZED_LIBS ${ZED_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY})

# Recording sample
ADD_EXECUTABLE(ZED_One_SVO_External_Data_Recording include/utils.hpp src/recording.cpp)
TARGET_LINK_LIBRARIES(ZED_One_SVO_External_Data_Recording ${ZED_LIBS} ${OpenCV_LIBRARIES})

# Playback sample
ADD_EXECUTABLE(ZED_One_SVO_External_Data_Playback include/utils.hpp src/playback.cpp)
TARGET_LINK_LIBRARIES(ZED_One_SVO_External_Data_Playback ${ZED_LIBS} ${OpenCV_LIBRARIES})

if(INSTALL_SAMPLES)
    LIST(APPEND SAMPLE_LIST ZED_One_SVO_External_Data_Recording)
    LIST(APPEND SAMPLE_LIST ZED_One_SVO_External_Data_Playback)
    SET(SAMPLE_LIST "${SAMPLE_LIST}" PARENT_SCOPE)
endif()
