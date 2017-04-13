SET(execName ZED_Multi_Camera)

CMAKE_MINIMUM_REQUIRED(VERSION 2.4)

if(COMMAND cmake_policy)
	cmake_policy(SET CMP0003 OLD)
	cmake_policy(SET CMP0015 OLD)
endif(COMMAND cmake_policy)

SET(EXECUTABLE_OUTPUT_PATH ".")

IF(WIN32) # Windows
    message(FATAL_ERROR "Multi ZED not available on Windows")
    EXIT()
ELSE() # Linux
    find_package(ZED 2.0 REQUIRED)
    
    ##For Jetson, OpenCV4Tegra is based on OpenCV2.4
    exec_program(uname ARGS -p OUTPUT_VARIABLE CMAKE_SYSTEM_NAME2)
    if ( CMAKE_SYSTEM_NAME2 MATCHES "aarch64" OR CMAKE_SYSTEM_NAME2 MATCHES "armv7l" )
        SET(OCV_VERSION "2.4")
        SET(CUDA_VERSION "")
    else()
        SET(CUDA_VERSION "8")
        SET(OCV_VERSION "3.1")
    endif()
ENDIF(WIN32)

find_package(OpenCV ${OCV_VERSION} COMPONENTS core highgui imgproc REQUIRED)
find_package(CUDA ${CUDA_VERSION} REQUIRED)
    
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${ZED_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

link_directories(${ZED_LIBRARY_DIR})
link_directories(${OpenCV_LIBRARY_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})

SET(SRC_FOLDER src)
FILE(GLOB_RECURSE SRC_FILES "${SRC_FOLDER}/*.cpp")

ADD_EXECUTABLE(${execName} ${SRC_FILES})
set_property(TARGET ${execName} PROPERTY OUTPUT_NAME "ZED Multi Camera")

add_definitions(-std=c++0x)

TARGET_LINK_LIBRARIES(${execName}
                        ${ZED_LIBRARIES}
                        ${OpenCV_LIBRARIES}
                        ${CUDA_LIBRARIES} ${CUDA_npps_LIBRARY} ${CUDA_nppi_LIBRARY}
                    )

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3" ) # Release Perf mode
