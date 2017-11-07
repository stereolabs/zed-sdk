SET(execName ZED_CUDA_Refocus)
CMAKE_MINIMUM_REQUIRED(VERSION 2.4)
PROJECT(${execName})

if(COMMAND cmake_policy)
	cmake_policy(SET CMP0003 OLD)
	cmake_policy(SET CMP0002 OLD)
	cmake_policy(SET CMP0015 OLD)
endif(COMMAND cmake_policy)

SET(EXECUTABLE_OUTPUT_PATH ".")

find_package(ZED 2 REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)

IF(NOT WIN32)
     SET(SPECIAL_OS_LIBS "X11" "pthread")
ENDIF(WIN32)

include_directories(${ZED_INCLUDE_DIRS})
include_directories(${GLUT_INCLUDE_PATH})
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
add_definitions(-std=c++11 -g -O3)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++11) 
cuda_add_executable(${execName} ${SRC_FILES}) 

# Add the required libraries for linking:
TARGET_LINK_LIBRARIES(${execName}
			${ZED_LIBRARIES}
			${SPECIAL_OS_LIBS}
			${GLEW_LIBRARIES}
			${OPENGL_LIBRARIES}
			${GLUT_LIBRARY}
			${CUDA_LIBRARIES} ${CUDA_nppi_LIBRARY} ${CUDA_NPP_LIBRARIES_ZED} )
