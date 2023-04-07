cmake_minimum_required(VERSION 3.5)
PROJECT(ZED_GNSS_recording)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_BUILD_TYPE "Release")

# Find external package
# OpenGL
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)
SET(OpenGL_GL_PREFERENCE GLVND)
find_package(OpenGL REQUIRED)
# ZED
find_package(ZED 4 REQUIRED)
find_package(CUDA ${ZED_CUDA_VERSION} REQUIRED)
# GPSD - If found
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)
find_package(GPS)
if(GPS_FOUND)
    add_definitions(-DGPSD_FOUND)
    message("Found GPSD library")
else()
    message("GPSD was not FOUND ... the resulted sample will not be able to read live GNSS data")
endif()

# Include external lib header:
include_directories(${ZED_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})

# Include project headers:
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/display)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/exporter)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/gnss_reader)

# Add execuable and link
ADD_EXECUTABLE(${PROJECT_NAME} src/main.cpp src/display/GLViewer.cpp src/display/GenericDisplay.cpp src/gnss_reader/GPSDReader.cpp src/exporter/KMLExporter.cpp src/exporter/GNSSSaver.cpp src/exporter/TimestampUtils.cpp)
set(LIBS ${ZED_LIBRARIES} ${OpenCV_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} ${GLEW_LIBRARIES} pthread util)
if(GPS_FOUND)
    list(APPEND LIBS, ${GPS_LIBRARIES})
endif()
target_link_libraries(${PROJECT_NAME} ${LIBS} ${SPECIAL_OS_LIBS})