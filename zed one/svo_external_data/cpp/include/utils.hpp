#pragma once

// ZED include
#include <sl/CameraOne.hpp>

// Standard includes
#include <iostream>
#include <iomanip>

/**
 * @brief Prints a formatted message with error code information
 * @param msg_prefix The message prefix
 * @param err_code The error code (default: SUCCESS)
 * @param msg_suffix Optional message suffix
 */
inline void print(std::string msg_prefix, sl::ERROR_CODE err_code = sl::ERROR_CODE::SUCCESS, std::string msg_suffix = "") {
    std::cout << "[Sample]";
    if (err_code > sl::ERROR_CODE::SUCCESS)
        std::cout << "[Error] ";
    else if (err_code < sl::ERROR_CODE::SUCCESS)
        std::cout << "[Warning] ";
    else
        std::cout << " ";
    std::cout << msg_prefix << " ";
    if (err_code != sl::ERROR_CODE::SUCCESS) {
        std::cout << " | " << sl::toString(err_code) << " : ";
        std::cout << sl::toVerbose(err_code);
    }
    if (!msg_suffix.empty())
        std::cout << " " << msg_suffix;
    std::cout << std::endl;
}

// Flag for graceful exit
static bool exit_app = false;

// Handle the CTRL-C keyboard signal
#ifdef _WIN32
    #include <Windows.h>
void CtrlHandler(DWORD fdwCtrlType) {
    exit_app = (fdwCtrlType == CTRL_C_EVENT);
}
#else
    #include <signal.h>
void nix_exit_handler(int s) {
    exit_app = true;
}
#endif

inline void SetCtrlHandler() {
#ifdef _WIN32
    SetConsoleCtrlHandler((PHANDLER_ROUTINE)CtrlHandler, TRUE);
#else
    struct sigaction sigIntHandler;
    sigIntHandler.sa_handler = nix_exit_handler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;
    sigaction(SIGINT, &sigIntHandler, NULL);
#endif
}
