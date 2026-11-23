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

#include <chrono>
#include <thread>

#include <sl/Camera.hpp>

//
// Plays an SVO file back at the speed it was recorded at: an SVO is read as fast as the machine
// can process it, which for a recording of a moving camera is faster than real time. Only useful
// to watch a replay; a batch run does not need it.
//
// InitParameters::svo_real_time_mode does the same at the SDK level, and is the shorter answer for
// an application that only plays a recording back. It is not used here because it skips images
// when the machine cannot keep up, which for a tracking sample would silently change the result
// being demonstrated; waiting, as below, always processes every image.
//
class SvoPlaybackClock {
public:
    // Sleeps until as much wall-clock time has passed as there is between the previous image and
    // this one. Does nothing on the first image, which has no earlier image to pace against.
    void waitForRealTime(sl::Timestamp imageTimestamp) {
        if (lastImageTimestamp.getNanoseconds() == 0) {
            return;
        }

        sl::Timestamp expectedElapsed = imageTimestamp - lastImageTimestamp;
        sl::Timestamp actualElapsed = sl::getCurrentTimeStamp() - lastClockTimestamp;

        if (expectedElapsed > actualElapsed) {
            sl::Timestamp sleepTime = expectedElapsed - actualElapsed;
            std::this_thread::sleep_for(std::chrono::milliseconds(sleepTime.getMilliseconds()));
        }
    }

    // Call once the image has been processed: pairs its own timestamp with the wall-clock time it
    // was done at, which is what the next wait is measured against.
    void markProcessed(sl::Timestamp imageTimestamp) {
        lastImageTimestamp = imageTimestamp;
        lastClockTimestamp = sl::getCurrentTimeStamp();
    }

private:
    sl::Timestamp lastImageTimestamp = sl::Timestamp(0);
    sl::Timestamp lastClockTimestamp = sl::Timestamp(0);
};
