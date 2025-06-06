CMAKE_MINIMUM_REQUIRED(VERSION 3.5)

PROJECT(ZED_Body_export_FBX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_BUILD_TYPE "Release")

option(LINK_SHARED_ZED "Link with the ZED SDK shared executable" ON)

if (NOT LINK_SHARED_ZED AND MSVC)
    message(FATAL_ERROR "LINK_SHARED_ZED OFF : ZED SDK static libraries not available on Windows")
endif()

find_package(ZED REQUIRED)
find_package(CUDA ${ZED_CUDA_VERSION} REQUIRED)

# change this according to your FBX install directory
if(WIN32)
	set(FBX_DIR "C:/Program Files/Autodesk/FBX/FBX SDK/2020.3.1")
else()
	set(FBX_DIR "/usr/local/lib/FBX")
endif()

if(EXISTS "${FBX_DIR}")
	include("${FBX_DIR}/samples/CMakeSettings.txt")
	include_directories("${FBX_ROOT}/include")
	set(FBX_COMMON_DIR "${FBX_CMAKE_ROOT}/Common")
	include_directories("${FBX_COMMON_DIR}")
	set(FBX_SRC ${FBX_COMMON_DIR}/Common.cxx)
	add_definitions(-DFBX_EXPORT)
else()
	message(AUTHOR_WARNING "FBX library not found")
endif()

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${ZED_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

link_directories(${ZED_LIBRARY_DIR})
link_directories(${CUDA_LIBRARY_DIRS})


ADD_EXECUTABLE(${PROJECT_NAME} include/utils.hpp src/main.cpp ${FBX_SRC})

if (LINK_SHARED_ZED)
	SET(ZED_LIBS ${ZED_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY})
else()
	SET(ZED_LIBS ${ZED_STATIC_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_LIBRARY})
endif()

TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${ZED_LIBS} ${FBXSDK_LIBRARY} ${FBX_REQUIRED_LIBS_DEPENDENCY})

if(INSTALL_SAMPLES)
    LIST(APPEND SAMPLE_LIST ${PROJECT_NAME})
    SET(SAMPLE_LIST "${SAMPLE_LIST}" PARENT_SCOPE)
endif()
