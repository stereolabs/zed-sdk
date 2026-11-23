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

#include "LandmarkOverlay.hpp"

#include <algorithm>

#include <opencv2/opencv.hpp>

using namespace sl;

namespace {

    const cv::Scalar kStaticLandmarkColor(63, 255, 67, 255);
    const cv::Scalar kDynamicLandmarkColor(100, 100, 255, 255);
    const cv::Scalar kUnknownLandmarkColor(240, 25, 25, 255);

    // Green when the landmark looks static, blue when it looks like it belongs to something that
    // moves, red while the tracking has no opinion.
    cv::Scalar landmarkColor(float dynamicConfidence) {
        if (dynamicConfidence == -1.f) {
            return kUnknownLandmarkColor;
        }

        const float confidence = std::clamp(dynamicConfidence, 0.0f, 1.0f);
        cv::Scalar color;

        for (int channel = 0; channel < 4; channel++) {
            color[channel] = kStaticLandmarkColor[channel] * confidence + kDynamicLandmarkColor[channel] * (1 - confidence);
        }

        return color;
    }

} // namespace

void drawLandmarks2D(Mat& image, const std::vector<Landmark2D>& landmarks, Resolution cameraResolution) {
    const float widthRatio = image.getWidth() / (float)cameraResolution.width;
    const float heightRatio = image.getHeight() / (float)cameraResolution.height;

    // Wraps the image in a cv::Mat sharing the same CPU buffer: no copy, so drawing here draws on
    // the image the viewer displays. retrieveImage() gives a 4-channel 8-bit image.
    cv::Mat imageCVMat(image.getHeight(), image.getWidth(), CV_8UC4, image.getPtr<sl::uchar1>(MEM::CPU), image.getStepBytes(MEM::CPU));

    for (const Landmark2D& landmark : landmarks) {
        cv::Scalar color = landmarkColor(landmark.dynamic_confidence);

        cv::circle(
            imageCVMat,
            cv::Point2f(landmark.image_position[0] * widthRatio, landmark.image_position[1] * heightRatio),
            2,
            color,
            -1
        );
    }
}
