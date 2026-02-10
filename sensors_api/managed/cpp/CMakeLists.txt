CMAKE_MINIMUM_REQUIRED(VERSION 3.5)
PROJECT(SensorsAPISample)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE "Release")

option(WITH_RERUN "Build with Rerun SDK support (auto-downloads if not found)" OFF)
option(BUILD_WITH_SANITIZER_ADDRESS "Build with address sanitizer" OFF)

find_package(ZED 5 REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)

if(WITH_RERUN)
    message(STATUS "Downloading Rerun SDK via FetchContent...")
    include(FetchContent)
    
    set(RERUN_SDK_URL "https://github.com/rerun-io/rerun/releases/download/0.28.2/rerun_cpp_sdk.zip")
    
    # CMake 3.24+ supports DOWNLOAD_EXTRACT_TIMESTAMP
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
        FetchContent_Declare(
            rerun_sdk
            URL ${RERUN_SDK_URL}
            DOWNLOAD_EXTRACT_TIMESTAMP ON
        )
    else()
        FetchContent_Declare(
            rerun_sdk
            URL ${RERUN_SDK_URL}
        )
    endif()
    FetchContent_MakeAvailable(rerun_sdk)
endif()

SET(SOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/GLViewer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/RerunViewer.cpp
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${ZED_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${GLEW_INCLUDE_DIRS})
include_directories(${GLUT_INCLUDE_DIRS})

link_directories(${ZED_LIBRARY_DIR})
link_directories(${GLEW_LIBRARY_DIRS})
link_directories(${GLUT_LIBRARY_DIRS})
link_directories(${OpenGL_LIBRARY_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})

ADD_EXECUTABLE(${PROJECT_NAME} ${SOURCES})
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

target_link_libraries(${PROJECT_NAME} 
    ${ZED_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${OPENGL_LIBRARIES}
    ${GLUT_LIBRARIES}
    ${GLEW_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${CUDA_npps_LIBRARY}
    ${CUDA_nppig_LIBRARY}
)

if(WITH_RERUN)
    target_compile_definitions(${PROJECT_NAME} PRIVATE WITH_RERUN)
    target_link_libraries(${PROJECT_NAME} rerun_sdk)
endif()

if(BUILD_WITH_SANITIZER_ADDRESS)
    message("Building with address sanitizer enabled")
    target_compile_options(${PROJECT_NAME} PRIVATE -fno-omit-frame-pointer -fsanitize=address)
    target_link_options(${PROJECT_NAME} PRIVATE -fno-omit-frame-pointer -fsanitize=address)
endif()
