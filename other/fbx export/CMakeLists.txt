CMAKE_MINIMUM_REQUIRED(VERSION 3.5)

PROJECT(ZED_FBX_Export)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_BUILD_TYPE "Release")

option(LINK_SHARED_ZED "Link with the ZED SDK shared executable" ON)

if (NOT LINK_SHARED_ZED AND MSVC)
    message(FATAL_ERROR "LINK_SHARED_ZED OFF : ZED SDK static libraries not available on Windows")
endif()

find_package(ZED 3 REQUIRED)
find_package(CUDA ${ZED_CUDA_VERSION} REQUIRED)

# change this according to your FBX install directory
if(WIN32)
	set(FBX_DIR "C:/Program Files/Autodesk/FBX/FBX SDK/2020.3.1")
else()
	set(FBX_DIR "/usr/local/lib/FBX")
endif()

include("${FBX_DIR}/samples/CMakeSettings.txt")

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${ZED_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

include_directories("${FBX_ROOT}/include")
set(FBX_COMMON_DIR "${FBX_CMAKE_ROOT}/Common")
include_directories("${FBX_COMMON_DIR}")

link_directories(${ZED_LIBRARY_DIR})
link_directories(${CUDA_LIBRARY_DIRS})

FILE(GLOB_RECURSE HDR_FILES include/*.h*)

SET(EXPORT_CAMERA_PATH "ZED_FBX_Camera")
SET(EXPORT_SKELETONS "ZED_FBX_Skeletons")

ADD_EXECUTABLE(${EXPORT_CAMERA_PATH} ${HDR_FILES} src/mainCamera.cpp ${FBX_COMMON_DIR}/Common.cxx)
ADD_EXECUTABLE(${EXPORT_SKELETONS} ${HDR_FILES} src/mainSkeletons.cpp ${FBX_COMMON_DIR}/Common.cxx)

if (LINK_SHARED_ZED)
    SET(ZED_LIBS ${ZED_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY})
else()
    SET(ZED_LIBS ${ZED_STATIC_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_LIBRARY})
endif()

IF(NOT WIN32)
    SET(SPECIAL_OS_LIBS "pthread;dl")
ENDIF()

TARGET_LINK_LIBRARIES(${EXPORT_CAMERA_PATH} ${ZED_LIBS} ${FBXSDK_LIBRARY} ${FBX_REQUIRED_LIBS_DEPENDENCY} ${SPECIAL_OS_LIBS})
TARGET_LINK_LIBRARIES(${EXPORT_SKELETONS} ${ZED_LIBS} ${FBXSDK_LIBRARY} ${FBX_REQUIRED_LIBS_DEPENDENCY} ${SPECIAL_OS_LIBS})
