CMAKE_MINIMUM_REQUIRED(VERSION 3.5)
PROJECT(ZED_SVO_Recording)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_BUILD_TYPE "RelWithDebInfo")

option(LINK_SHARED_ZED "Link with the ZED SDK shared executable" ON)

if (NOT LINK_SHARED_ZED AND MSVC)
    message(FATAL_ERROR "LINK_SHARED_ZED OFF : ZED SDK static libraries not available on Windows")
endif()

find_package(ZED 3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDA ${ZED_CUDA_VERSION} REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${ZED_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

link_directories(${ZED_LIBRARY_DIR})
link_directories(${CUDA_LIBRARY_DIRS})
IF(NOT WIN32)
    IF (BUILD_WITH_SANITIZER)
        message("!! Building with address sanitizer and -g !!")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer -fsanitize=address -Wall -Wextra -g")
        set (CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} -fno-omit-frame-pointer -fsanitize=address -Wall -Wextra -g")
        add_definitions(-g -fsanitize=address -fno-omit-frame-pointer -static-libasan -Wall -Wextra)
    ENDIF()

    IF (BUILD_WITH_DEBUGINFOS)
        message("!! Building with -g !!")
        add_definitions(-g)
    ENDIF()
    ENDIF()
    
ADD_EXECUTABLE(${PROJECT_NAME} include/utils.hpp src/main.cpp)

if (LINK_SHARED_ZED)
    SET(ZED_LIBS ${ZED_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY} ${OpenCV_LIBRARIES}) 
else()
    SET(ZED_LIBS ${ZED_STATIC_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_LIBRARY})
endif()

TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${ZED_LIBS} ${OpenCV_LIBRARIES})

if(INSTALL_SAMPLES)
    LIST(APPEND SAMPLE_LIST ${PROJECT_NAME})
    SET(SAMPLE_LIST "${SAMPLE_LIST}" PARENT_SCOPE)
endif()
