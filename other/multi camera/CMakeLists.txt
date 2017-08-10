SET(execName ZED_Multi_Camera)
CMAKE_MINIMUM_REQUIRED(VERSION 2.4)
PROJECT(${execName})

if(COMMAND cmake_policy)
	cmake_policy(SET CMP0003 OLD)
	cmake_policy(SET CMP0015 OLD)
endif(COMMAND cmake_policy)

SET(EXECUTABLE_OUTPUT_PATH ".")
SET(VERSION_REQ_OCV "3")
SET(VERSION_REQ_CUDA "8.0")

IF(WIN32) # Windows
    message(FATAL_ERROR "Multi ZED not available on Windows")
    EXIT()
ELSE() # Linux
    find_package(ZED 2.0 REQUIRED)
    
    ##For Jetson, OpenCV4Tegra is based on OpenCV2.4
    exec_program(uname ARGS -p OUTPUT_VARIABLE CMAKE_SYSTEM_NAME2)
    if ( CMAKE_SYSTEM_NAME2 MATCHES "aarch64" OR CMAKE_SYSTEM_NAME2 MATCHES "armv7l" )
        SET(VERSION_REQ_OCV "2.4")
        SET(VERSION_REQ_CUDA "")
    endif()
ENDIF(WIN32)

find_package(OpenCV ${VERSION_REQ_OCV} COMPONENTS core highgui imgproc REQUIRED)
find_package(CUDA ${VERSION_REQ_CUDA} REQUIRED)
    
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${ZED_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

link_directories(${ZED_LIBRARY_DIR})
link_directories(${OpenCV_LIBRARY_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})

FILE(GLOB_RECURSE SRC_FILES src/*.cpp)

ADD_EXECUTABLE(${execName} ${SRC_FILES})
add_definitions(-std=c++0x)

TARGET_LINK_LIBRARIES(${execName}
			${ZED_LIBRARIES}
			${OpenCV_LIBRARIES}
			${CUDA_LIBRARIES} ${CUDA_npps_LIBRARY} ${CUDA_nppi_LIBRARY} )

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3" ) # Release Perf mode
