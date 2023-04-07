///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2022, STEREOLABS.
//
// All rights reserved.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
///////////////////////////////////////////////////////////////////////////

#pragma once

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

// Set the function to handle the CTRL-C
void SetCtrlHandler() {
#ifdef _WIN32
    SetConsoleCtrlHandler((PHANDLER_ROUTINE) CtrlHandler, TRUE);
#else // unix
    struct sigaction sigIntHandler;
    sigIntHandler.sa_handler = nix_exit_handler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;
    sigaction(SIGINT, &sigIntHandler, NULL);
#endif
}


// Array containing keypoint indexes of each joint's parent. Used to build skeleton node hierarchy.
const int parentsIdx[] = {
	-1,
	0,
	1,
	2,
	2,
	4,
	5,
	6,
	7,
	8,
	7,
	2,
	11,
	12,
	13,
	14,
	15,
	14,
	0,
	18,
	19,
	20,
	0,
	22,
	23,
	24,
	3,
	26,
	26,
	26,
	26,
	26,
	20,
	24
};

// List of children of each joint. Used to build skeleton node hierarchy.
std::vector<std::vector<int>> childenIdx{
	{1,18,22},
	{2},
	{11,3,4},
	{26},
	{5},
	{6},
	{7},
	{8,10},
	{9},
	{},
	{},
	{12},
	{13},
	{14},
	{15,17},
	{16},
	{},
	{},
	{19},
	{20},
	{21,32},
	{},
	{23},
	{24},
	{25,33},
	{},
	{27,28,29,30,31},
	{},
	{},
	{},
	{},
	{},
	{},
	{}
};

// Local joint position of each joint. Used to build skeleton rest pose.
std::vector<sl::float3> local_joints_translations{
	sl::float3(0,0,0),              // 0
	sl::float3(0,20,0),				// 1
	sl::float3(0,20,0),				// 2
	sl::float3(0,20,0),				// 3
	sl::float3(-5,20,0),			// 4
	sl::float3(-15,0,0),			// 5
	sl::float3(-26,0,0),			// 6
	sl::float3(-25,0,0),			// 7
	sl::float3(-5,0,0),				// 8
	sl::float3(-10,0,0),			// 9
	sl::float3(-10,-6,0),			// 10
	sl::float3(5,20,0),				// 11
	sl::float3(15,0,0),				// 12
	sl::float3(26,0,0),				// 13
	sl::float3(25,0,0),				// 14
	sl::float3(5, 0, 0),			// 15
	sl::float3(10,0,0),				// 16
	sl::float3(10,-6,0),			// 17
	sl::float3(-10,0,0),			// 18
	sl::float3(0,-45,0),			// 19
	sl::float3(0,-40,0),			// 20
	sl::float3(0,-10,12),			// 21
	sl::float3(10,0,0),				// 22
	sl::float3(0,-45,0),			// 23
	sl::float3(0,-40,0),			// 24
	sl::float3(0,-10,12),			// 25
	sl::float3(0,15,0),				// 26
	sl::float3(0,10,0),				// 27
	sl::float3(-3,13,0),			// 28
	sl::float3(-8.5,10,-10),		// 29
	sl::float3(3,13,0),				// 30
	sl::float3(8.5,10,-10),			// 31
	sl::float3(0,-10,-4),			// 32
	sl::float3(0,-10,-4),			// 33
};