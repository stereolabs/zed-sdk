cmake_minimum_required(VERSION 3.14)
project(SensorPlacer VERSION 1.0 LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Qt5
find_package(Qt5 REQUIRED COMPONENTS Widgets OpenGL)

# ZED SDK
find_package(ZED 5 REQUIRED)
find_package(CUDA ${ZED_CUDA_VERSION} REQUIRED)

# OpenGL
set(OpenGL_GL_PREFERENCE GLVND)
find_package(OpenGL REQUIRED)

include_directories(${ZED_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

link_directories(${ZED_LIBRARY_DIR})
link_directories(${CUDA_LIBRARY_DIRS})

# Generate ui_mainwindow.h
execute_process(COMMAND uic ${CMAKE_CURRENT_SOURCE_DIR}/src/mainwindow.ui
                -o ${CMAKE_CURRENT_SOURCE_DIR}/src/ui_mainwindow.h)

set(PROJECT_SOURCES
    src/main.cpp
    src/mainwindow.cpp
    src/mainwindow.h
    src/mainwindow.ui
    src/GlViewer.cpp
    src/GlViewer.h
    src/SensorWorker.cpp
    src/SensorWorker.h
)

add_executable(${PROJECT_NAME} ${PROJECT_SOURCES})

target_link_libraries(${PROJECT_NAME} PRIVATE
    Qt5::Widgets
    Qt5::OpenGL
    ${ZED_LIBRARIES}
    ${CUDA_CUDA_LIBRARY}
    ${CUDA_CUDART_LIBRARY}
    ${OPENGL_LIBRARIES}
    pthread
)
