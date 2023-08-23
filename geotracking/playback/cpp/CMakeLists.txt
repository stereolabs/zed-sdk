cmake_minimum_required(VERSION 3.5)
PROJECT(ZED_GNSS_playback)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_BUILD_TYPE "Release")

find_package(ZED 4 REQUIRED)
find_package(CUDA ${ZED_CUDA_VERSION} REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)
SET(OpenGL_GL_PREFERENCE GLVND)
find_package(OpenGL REQUIRED)
find_package(OpenCV REQUIRED)

# ZEDHub - if found
find_package(SL_HUB)

if(SL_HUB_FOUND)
    message("Found ZEDHub library")
    add_definitions(-DCOMPILE_WITH_ZEDHUB=TRUE)
    link_directories(${SL_HUB_LIB_DIR})
    include_directories(${SL_HUB_INCLUDE_DIR})
endif()

IF(NOT MSVC)
    SET(SPECIAL_OS_LIBS "pthread")
ENDIF()

# Include external lib header:
include_directories(${ZED_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${GLEW_INCLUDE_DIRS})
include_directories(${GLUT_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${OpenCV_INCLUDE_DIRS})

link_directories(${ZED_LIBRARY_DIR})
link_directories(${GLEW_LIBRARY_DIRS})
link_directories(${GLUT_LIBRARY_DIRS})
link_directories(${OpenGL_LIBRARY_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})


option(BUILD_WITH_SANITIZER "Build with sanitizer" OFF)
IF(NOT WIN32)

	add_definitions(-Werror=return-type)
	IF (BUILD_WITH_SANITIZER)
	message("!! Building with address sanitizer and -g !!")
	set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -static-libasan -fsanitize=address")
	set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -static-libasan -fsanitize=address")
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer -static-libasan -fsanitize=address -Wall -Wextra -g")
	set (CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} -fno-omit-frame-pointer -static-libasan -fsanitize=address -Wall -Wextra -g")
	add_definitions(-g -fsanitize=address -fno-omit-frame-pointer -static-libasan -Wall -Wextra)
	ENDIF()

	IF (BUILD_WITH_DEBUGINFOS)
	message("!! Building with -g !!")
	add_definitions(-g)
	ENDIF()
ENDIF()

FILE(GLOB_RECURSE SRC_FILES src/*.c*)
FILE(GLOB_RECURSE HDR_FILES include/*.h*)

add_executable(${PROJECT_NAME} ${HDR_FILES} ${SRC_FILES})

if(SL_HUB_FOUND)
    set(LIBS sl_hub util ${ZED_LIBRARIES} ${OpenCV_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} ${GLEW_LIBRARIES})
else()
    set(LIBS ${ZED_LIBRARIES} ${OpenCV_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} ${GLEW_LIBRARIES})
endif()


target_link_libraries(${PROJECT_NAME} ${LIBS} ${SPECIAL_OS_LIBS})
