SET(execName ZED_CUDA_Refocus)
CMAKE_MINIMUM_REQUIRED(VERSION 2.4)
PROJECT(${execName})

if(COMMAND cmake_policy)
	cmake_policy(SET CMP0003 OLD)
	cmake_policy(SET CMP0002 OLD)
	cmake_policy(SET CMP0015 OLD)
endif(COMMAND cmake_policy)

SET(EXECUTABLE_OUTPUT_PATH ".")
SET(VERSION_REQ_CUDA "8.0")

IF(WIN32) # Windows
    SET(ZED_INCLUDE_DIRS $ENV{ZED_INCLUDE_DIRS})
    SET(ZED_LIBRARIES $ENV{ZED_LIBRARIES_64})
    SET(GLUT_LIBRARY_DIRS $ENV{GLUT_LIBRARY_DIRS})
    SET(GLEW_LIBRARY_DIRS $ENV{GLEW_LIBRARY_DIRS})
    SET(GLEW_LIBRARIES $ENV{GLEW_LIBRARY})
    SET(ZED_LIBRARY_DIR $ENV{ZED_LIBRARY_DIR})
    SET(GLUT_INCLUDE_DIRS $ENV{GLUT_INCLUDE_DIRS})
    SET(GLEW_INCLUDE_DIRS $ENV{GLEW_INCLUDE_DIRS})
ELSE() # Linux
    find_package(ZED REQUIRED)
    find_package(GLUT REQUIRED)
    find_package(GLEW REQUIRED)

    exec_program(uname ARGS -p OUTPUT_VARIABLE CMAKE_SYSTEM_NAME2)
    if ( CMAKE_SYSTEM_NAME2 MATCHES "aarch64" OR CMAKE_SYSTEM_NAME2 MATCHES "armv7l" )
        SET(VERSION_REQ_CUDA "")
        link_directories("/usr/lib/aarch64-linux-gnu/tegra") ## on Jetson TX1 64bits, the correct libGL.so is located here (the default one will lead to linkin error (undef ref)) ##ignore warning
    endif()

    ##For Jetson TX1
    if ( CMAKE_SYSTEM_NAME2 MATCHES "aarch64")
        unset(CUDA_USE_STATIC_CUDA_RUNTIME CACHE)
        option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
        message ("CUDA_USE_STATIC_CUDA_RUNTIME : ${CUDA_USE_STATIC_CUDA_RUNTIME}")
        ##to prevent from opencv_dep_cudart dependencies error...
        ## cmake with -DCUDA_USE_STATIC_CUDA_RUNTIME=false can also be called.
    endif()

     SET(SPECIAL_OS_LIBS "X11" "pthread")
ENDIF(WIN32)

find_package(CUDA ${VERSION_REQ_CUDA} REQUIRED)
find_package(OpenGL REQUIRED)

include_directories(${ZED_INCLUDE_DIRS})
include_directories(${GLUT_INCLUDE_DIRS})
include_directories(${GLEW_INCLUDE_DIRS})
include_directories(${OPENGL_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

link_directories(${ZED_LIBRARY_DIR})
link_directories(${CUDA_LIBRARY_DIRS})
link_directories(${GLUT_LIBRARY_DIRS})
link_directories(${GLEW_LIBRARY_DIRS})
link_directories(${OpenGL_LIBRARY_DIRS})

FILE(GLOB_RECURSE SRC_FILES src/*.c*)
FILE(GLOB_RECURSE HRD_FILES include/*.h)

ADD_EXECUTABLE(${execName} ${HRD_FILES} ${SRC_FILES})

add_definitions(-std=c++0x -g -O3)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++11) 
cuda_add_executable(${execName} ${SRC_FILES}) 

# Add the required libraries for linking:
TARGET_LINK_LIBRARIES(${execName}
			${ZED_LIBRARIES}
			${SPECIAL_OS_LIBS}
			${GLEW_LIBRARIES}
			${OPENGL_LIBRARIES}
			${GLUT_LIBRARY}
			${CUDA_LIBRARIES} ${CUDA_nppi_LIBRARY} ${CUDA_npps_LIBRARY} )
