CMAKE_MINIMUM_REQUIRED(VERSION 2.4)
PROJECT(ZED_Depth_Sensing)

option(LINK_SHARED_ZED "Link with the ZED SDK shared executable" ON)

if (NOT LINK_SHARED_ZED AND MSVC)
    message(FATAL_ERROR "LINK_SHARED_ZED OFF : ZED SDK static libraries not available on Windows")
endif()

if(COMMAND cmake_policy)
	cmake_policy(SET CMP0003 OLD)
	cmake_policy(SET CMP0015 OLD)
endif(COMMAND cmake_policy)

SET(EXECUTABLE_OUTPUT_PATH ".")
 
find_package(ZED 3 REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)

IF(NOT WIN32)
    SET(SPECIAL_OS_LIBS "pthread" "X11")
    add_definitions(-Wno-write-strings -fpermissive)
ENDIF()

find_package(CUDA ${ZED_CUDA_VERSION} EXACT REQUIRED)

include_directories(${ZED_INCLUDE_DIRS})
include_directories(${GLEW_INCLUDE_DIRS})
include_directories(${GLUT_INCLUDE_PATH})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

link_directories(${ZED_LIBRARY_DIR})
link_directories(${GLEW_LIBRARY_DIRS})
link_directories(${GLUT_LIBRARY_DIRS})
link_directories(${OpenGL_LIBRARY_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)

FILE(GLOB_RECURSE SRC_FILES src/*.cpp)
FILE(GLOB_RECURSE HDR_FILES include/*.hpp)

ADD_EXECUTABLE(${PROJECT_NAME} ${HDR_FILES} ${SRC_FILES})
add_definitions(-std=c++14 -O3 )

if (LINK_SHARED_ZED)
    SET(ZED_LIBS ${ZED_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY})
else()
    SET(ZED_LIBS ${ZED_STATIC_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_LIBRARY})
endif()

TARGET_LINK_LIBRARIES(${PROJECT_NAME} 
                        ${SPECIAL_OS_LIBS} 
                        ${ZED_LIBS} 
                        ${OPENGL_LIBRARIES}
                        ${GLUT_LIBRARY}
                        ${GLEW_LIBRARIES})

if(INSTALL_SAMPLES)
    LIST(APPEND SAMPLE_LIST ${PROJECT_NAME})
    SET(SAMPLE_LIST "${SAMPLE_LIST}" PARENT_SCOPE)
endif()