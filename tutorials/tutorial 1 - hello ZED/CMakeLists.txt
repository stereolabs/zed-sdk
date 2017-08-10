SET(execName ZED_Tutorial_1)
CMAKE_MINIMUM_REQUIRED(VERSION 2.4)
PROJECT(${execName})

if(COMMAND cmake_policy)
	cmake_policy(SET CMP0003 OLD)
	cmake_policy(SET CMP0015 OLD)
endif(COMMAND cmake_policy)

SET(EXECUTABLE_OUTPUT_PATH ".")
SET(VERSION_REQ_CUDA "8.0")

IF(WIN32) # Windows
    if (CMAKE_CL_64)  
        SET(ZED_INCLUDE_DIRS $ENV{ZED_INCLUDE_DIRS})
        SET(ZED_LIBRARIES $ENV{ZED_LIBRARIES_64})
        SET(ZED_LIBRARY_DIR $ENV{ZED_LIBRARY_DIR})
    else()
        message(FATAL_ERROR "You've selected the 32bit version of ${CMAKE_GENERATOR}. \n Please delete the cache (file->Delete Cache) and use the 64bit version. (${CMAKE_GENERATOR} Win64)")
    endif()
ELSE() # Linux
    find_package(ZED 2.1 REQUIRED)
ENDIF(WIN32)

find_package(CUDA ${VERSION_REQ_CUDA} REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${ZED_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

link_directories(${ZED_LIBRARY_DIR})
link_directories(${CUDA_LIBRARY_DIRS})

ADD_EXECUTABLE(${execName} "main.cpp")
add_definitions(-std=c++0x -g -O3)

TARGET_LINK_LIBRARIES(${execName}
                        ${ZED_LIBRARIES}
                        ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY} ${CUDA_npp_LIBRARY} )
