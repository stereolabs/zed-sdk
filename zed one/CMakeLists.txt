CMAKE_MINIMUM_REQUIRED(VERSION 3.5)
PROJECT(ZED_One_Samples)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_BUILD_TYPE "RelWithDebInfo")

# Add all subprojects
add_subdirectory(live/cpp)
add_subdirectory(svo_recording/cpp)
add_subdirectory(svo_playback/cpp)
add_subdirectory(svo_external_data/cpp)
add_subdirectory(streaming_sender/cpp)
add_subdirectory(streaming_receiver/cpp)
add_subdirectory(custom_inference/cpp)
