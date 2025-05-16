///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2025, STEREOLABS.
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

sl::String toSlString(const sl::BODY_38_PARTS& body_part) {
	sl::String out;

	switch (body_part) {
	case sl::BODY_38_PARTS::PELVIS:
		out = "PELVIS";
		break;
	case sl::BODY_38_PARTS::SPINE_1:
		out = "SPINE 1";
		break;
	case sl::BODY_38_PARTS::SPINE_2:
		out = "SPINE 2";
		break;
	case sl::BODY_38_PARTS::SPINE_3:
		out = "SPINE 3";
		break;
	case sl::BODY_38_PARTS::NECK:
		out = "NECK";
		break;
	case sl::BODY_38_PARTS::LEFT_CLAVICLE:
		out = "LEFT CLAVICLE";
		break;
	case sl::BODY_38_PARTS::LEFT_SHOULDER:
		out = "LEFT SHOULDER";
		break;
	case sl::BODY_38_PARTS::LEFT_ELBOW:
		out = "LEFT ELBOW";
		break;
	case sl::BODY_38_PARTS::LEFT_WRIST:
		out = "LEFT WRIST";
		break;
	case sl::BODY_38_PARTS::RIGHT_CLAVICLE:
		out = "RIGHT CLAVICLE";
		break;
	case sl::BODY_38_PARTS::RIGHT_SHOULDER:
		out = "RIGHT SHOULDER";
		break;
	case sl::BODY_38_PARTS::RIGHT_ELBOW:
		out = "RIGHT ELBOW";
		break;
	case sl::BODY_38_PARTS::RIGHT_WRIST:
		out = "RIGHT WRIST";
		break;
	case sl::BODY_38_PARTS::LEFT_HIP:
		out = "LEFT HIP";
		break;
	case sl::BODY_38_PARTS::LEFT_KNEE:
		out = "LEFT KNEE";
		break;
	case sl::BODY_38_PARTS::LEFT_ANKLE:
		out = "LEFT ANKLE";
		break;
	case sl::BODY_38_PARTS::RIGHT_HIP:
		out = "RIGHT HIP";
		break;
	case sl::BODY_38_PARTS::RIGHT_KNEE:
		out = "RIGHT KNEE";
		break;
	case sl::BODY_38_PARTS::RIGHT_ANKLE:
		out = "RIGHT ANKLE";
		break;
	case sl::BODY_38_PARTS::NOSE:
		out = "NOSE";
		break;
	case sl::BODY_38_PARTS::LEFT_EYE:
		out = "LEFT EYE";
		break;
	case sl::BODY_38_PARTS::LEFT_EAR:
		out = "LEFT EAR";
		break;
	case sl::BODY_38_PARTS::RIGHT_EYE:
		out = "RIGHT EYE";
		break;
	case sl::BODY_38_PARTS::RIGHT_EAR:
		out = "RIGHT EAR";
		break;
	case sl::BODY_38_PARTS::LEFT_HEEL:
		out = "LEFT HEEL";
		break;
	case sl::BODY_38_PARTS::RIGHT_HEEL:
		out = "RIGHT HEEL";
		break;
	case sl::BODY_38_PARTS::LEFT_BIG_TOE:
		out = "LEFT BIG TOE";
		break;
	case sl::BODY_38_PARTS::LEFT_SMALL_TOE:
		out = "LEFT SMALL TOE";
		break;
	case sl::BODY_38_PARTS::RIGHT_BIG_TOE:
		out = "RIGHT BIG TOE";
		break;
	case sl::BODY_38_PARTS::RIGHT_SMALL_TOE:
		out = "RIGHT SMALL TOE";
		break;
	case sl::BODY_38_PARTS::LEFT_HAND_THUMB_4:
		out = "LEFT HAND THUMB 4";	
		break;
	case sl::BODY_38_PARTS::RIGHT_HAND_THUMB_4:
		out = "RIGHT HAND THUMB 4";
		break;
	case sl::BODY_38_PARTS::LEFT_HAND_INDEX_1:
		out = "LEFT HAND INDEX 1";
		break;
	case sl::BODY_38_PARTS::RIGHT_HAND_INDEX_1:
		out = "RIGHT HAND INDEX 1";
		break;
	case sl::BODY_38_PARTS::LEFT_HAND_MIDDLE_4:
		out = "LEFT HAND MIDDLE 4";
		break;
	case sl::BODY_38_PARTS::RIGHT_HAND_MIDDLE_4:
		out = "RIGHT HAND MIDDLE 4";
		break;
	case sl::BODY_38_PARTS::LEFT_HAND_PINKY_1:
		out = "LEFT HAND PINKY 1";
		break;
	case sl::BODY_38_PARTS::RIGHT_HAND_PINKY_1:
		out = "RIGHT HAND PINKY 1";
		break;
	}

	return out;
}

// List of children of each joint. Used to build skeleton node hierarchy.
std::vector<std::vector<int>> childrenIdx34{
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


std::vector<std::vector<int>> childrenIdx38{
	{1,18,19},
	{2},
	{3},
	{4, 10, 11},
	{5},
	{6, 7},
	{8},
	{9},
	{},
	{},
	{12},
	{13},
	{14},
	{15},
	{16},
	{17},
	{30, 32, 34, 36},
	{31, 33, 35, 37},
	{20},
	{21},
	{22},
	{23},
	{24, 26, 28},
	{25, 27, 29},
	{},
	{},
	{},
	{},
	{},
	{},
	{},
	{},
	{},
	{},
	{},
	{},
	{},
	{}
};

// Local joint position of each joint. Used to build skeleton rest pose.
std::vector<sl::float3> local_joints_translations_body34{
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

std::vector<sl::float3> local_joints_translations_body38{
	sl::float3(0,0,0),              // 0
	sl::float3(0,7,0),				// 1
	sl::float3(0,13,0),				// 2
	sl::float3(0,13,0),				// 3
	sl::float3(0,17,0),				// 4
	sl::float3(0,20,-8),			// 5
	sl::float3(-3,6,4),				// 6
	sl::float3(3,6,4),				// 7
	sl::float3(-4,-6,9),			// 8
	sl::float3(4,-6,9),				// 9
	sl::float3(-7,12,5),			// 10
	sl::float3(7,12,5),				// 11
	sl::float3(-11,0,0),			// 12
	sl::float3(11,0,0),				// 13
	sl::float3(-26,0,0),			// 14
	sl::float3(26,0,0),				// 15
	sl::float3(-26,0,0),			// 16
	sl::float3(26,0,0),				// 17
	sl::float3(-9,-6,0),			// 18
	sl::float3(9,-6,0),				// 19
	sl::float3(0,-45,0),			// 20
	sl::float3(0,-45,0),			// 21
	sl::float3(1,-41,0),			// 22
	sl::float3(-1,-41,0),			// 23
	sl::float3(4,-6,-14),			// 24
	sl::float3(-3,-6,-14),			// 25
	sl::float3(-3,-6,-10),			// 26
	sl::float3(4,-6,-10),			// 27
	sl::float3(0,-6,6),				// 28
	sl::float3(0,-6,6),				// 29
	sl::float3(-11,-4,-6),			// 30
	sl::float3(11,-4,-6),			// 31
	sl::float3(11,-4,-6),			// 31
	sl::float3(-11,0,-3),			// 32
	sl::float3(11,0,-3),			// 33
	sl::float3(-19,0,-1),			// 34
	sl::float3(19,0,-1),			// 35
	sl::float3(-11,-1,4),			// 36
	sl::float3(11,-1,4),			// 37
};