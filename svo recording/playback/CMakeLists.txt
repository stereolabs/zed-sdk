SET(execName ZED_SVO_Playback)

CMAKE_MINIMUM_REQUIRED(VERSION 2.4)

if(COMMAND cmake_policy)
	cmake_policy(SET CMP0003 OLD)
	cmake_policy(SET CMP0015 OLD)
endif(COMMAND cmake_policy)

SET(EXECUTABLE_OUTPUT_PATH ".")

SET(VERSION_REQ_OCV "3.1")
SET(VERSION_REQ_CUDA "8.0")

IF(WIN32) # Windows
    if (CMAKE_CL_64) 
        SET(ZED_INCLUDE_DIRS $ENV{ZED_INCLUDE_DIRS})
        SET(ZED_LIBRARIES $ENV{ZED_LIBRARIES_64})
        SET(ZED_LIBRARY_DIR $ENV{ZED_LIBRARY_DIR})
        SET(OPENCV_DIR $ENV{OPENCV_DIR})
    else()
        message(FATAL_ERROR "You've selected the 32bit version of ${CMAKE_GENERATOR}. \n Please delete the cache (file->Delete Cache) and use the 64bit version. (${CMAKE_GENERATOR} Win64)")
    endif()
ELSE() # Linux
    find_package(ZED 2 REQUIRED)

    ##For Jetson, OpenCV4Tegra is based on OpenCV2.4
    exec_program(uname ARGS -p OUTPUT_VARIABLE CMAKE_SYSTEM_NAME2)
    if ( CMAKE_SYSTEM_NAME2 MATCHES "aarch64" )
       SET(VERSION_REQ_OCV "2.4")
       SET(VERSION_REQ_CUDA "")

       unset(CUDA_USE_STATIC_CUDA_RUNTIME CACHE)
       option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
       message ("CUDA_USE_STATIC_CUDA_RUNTIME : ${CUDA_USE_STATIC_CUDA_RUNTIME}")
       ##to prevent from opencv_dep_cudart dependencies error...
       ## cmake with -DCUDA_USE_STATIC_CUDA_RUNTIME=false can also be called.
    endif()

    SET(SPECIAL_OS_LIBS "pthread" "X11")
ENDIF(WIN32)

find_package(OpenCV REQUIRED)
find_package(CUDA ${VERSION_REQ_CUDA} REQUIRED)
 
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${ZED_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

link_directories(${ZED_LIBRARY_DIR})
link_directories(${CUDA_LIBRARY_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})

ADD_EXECUTABLE(${execName} src/main.cpp)
add_definitions(-std=c++0x -g -O3)

# Add the required libraries for linking:
TARGET_LINK_LIBRARIES(${execName}
                        ${ZED_LIBRARIES}
                        ${SPECIAL_OS_LIBS}
                        ${OpenCV_LIBS}
                        ${CUDA_LIBRARIES} ${CUDA_npps_LIBRARY} ${CUDA_nppi_LIBRARY} )
