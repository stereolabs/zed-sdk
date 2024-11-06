CMAKE_MINIMUM_REQUIRED(VERSION 3.5)
PROJECT(ZED_One_live)

if (WIN32)
	message(FATAL_ERROR "ZED One is not yet supported on windows.")
endif(WIN32)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_BUILD_TYPE "RelWithDebInfo")

option(LINK_SHARED_ZED "Link with the ZED SDK shared executable" ON)

if (NOT LINK_SHARED_ZED AND MSVC)
    message(FATAL_ERROR "LINK_SHARED_ZED OFF : ZED SDK static libraries not available on Windows")
endif()

find_package(ZED 4 REQUIRED)
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

ADD_EXECUTABLE(${PROJECT_NAME} include/utils.hpp live.cpp)
TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${ZED_LIBS} ${OpenCV_LIBRARIES})

ADD_EXECUTABLE(ZED_One_SVO_playback include/utils.hpp svo_playback.cpp)
TARGET_LINK_LIBRARIES(ZED_One_SVO_playback ${ZED_LIBS} ${OpenCV_LIBRARIES})

ADD_EXECUTABLE(ZED_One_SVO_recording include/utils.hpp svo_recording.cpp)
TARGET_LINK_LIBRARIES(ZED_One_SVO_recording ${ZED_LIBS} ${OpenCV_LIBRARIES})

ADD_EXECUTABLE(ZED_One_streaming_sender include/utils.hpp streaming_sender.cpp)
TARGET_LINK_LIBRARIES(ZED_One_streaming_sender ${ZED_LIBS} ${OpenCV_LIBRARIES})

ADD_EXECUTABLE(ZED_One_streaming_receiver include/utils.hpp streaming_receiver.cpp)
TARGET_LINK_LIBRARIES(ZED_One_streaming_receiver ${ZED_LIBS} ${OpenCV_LIBRARIES})

if(INSTALL_SAMPLES)
    LIST(APPEND SAMPLE_LIST ${PROJECT_NAME})
    SET(SAMPLE_LIST "${SAMPLE_LIST}" PARENT_SCOPE)
endif()
