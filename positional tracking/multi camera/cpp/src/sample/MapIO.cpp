///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2026, STEREOLABS.
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

#include "MapIO.hpp"

#include <chrono>
#include <thread>

#include "SampleConsole.hpp"

using namespace sl;

void saveAreaMap(Sensors& sensors, SensorDeviceIdentifier trackingCamera, const std::string& path) {
    print("Saving area map to " + path + " ...");

    SENSORS_ERROR_CODE status = sensors.saveAreaMap(path.c_str(), trackingCamera);

    if (status > SENSORS_ERROR_CODE::SUCCESS) {
        print("Failed to save area map", status);
        return;
    }

    AREA_EXPORTING_STATE exportState = sensors.getAreaExportState(trackingCamera);

    while (exportState == AREA_EXPORTING_STATE::RUNNING) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        exportState = sensors.getAreaExportState(trackingCamera);
    }

    if (exportState == AREA_EXPORTING_STATE::SUCCESS) {
        print("Successfully saved area map to " + path);
    } else {
        print("Failed to save area map: " + std::string(toString(exportState).c_str()), ERROR_CODE::FAILURE, false);
    }
}
