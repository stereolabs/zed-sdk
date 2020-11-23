cmake_minimum_required(VERSION 3.1)
PROJECT(Tutos)

OPTION(BUILD_CPP "Build c++ samples else c#" ON)

if(${BUILD_CPP})
    SET(TYPE "cpp")
else()
    SET(TYPE "csharp")
endif()

add_subdirectory("tutorial 1 - hello ZED/${TYPE}")
add_subdirectory("tutorial 2 - image capture/${TYPE}")
add_subdirectory("tutorial 3 - depth sensing/${TYPE}")
add_subdirectory("tutorial 4 - positional tracking/${TYPE}")
add_subdirectory("tutorial 5 - spatial mapping/${TYPE}")
add_subdirectory("tutorial 6 - object detection/${TYPE}")
add_subdirectory("tutorial 7 - sensor data/${TYPE}")