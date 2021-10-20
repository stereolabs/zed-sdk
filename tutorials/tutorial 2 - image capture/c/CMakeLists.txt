CMAKE_MINIMUM_REQUIRED(VERSION 3.5)
PROJECT(ZED_C_Tutorial_2)

if(COMMAND cmake_policy)
	cmake_policy(SET CMP0003 OLD)
	cmake_policy(SET CMP0015 OLD)
endif(COMMAND cmake_policy)

SET(EXECUTABLE_OUTPUT_PATH ".")

if (WIN32)
include_directories($ENV{ZED_SDK_ROOT_DIR}/include/)
link_directories($ENV{ZED_SDK_ROOT_DIR}/lib/)
else (WIN32)
include_directories("/usr/local/zed/include/")
link_directories("/usr/local/zed/lib/")
endif(WIN32)


ADD_EXECUTABLE(${PROJECT_NAME} main.c)

target_link_libraries(${PROJECT_NAME} sl_zed_c)
