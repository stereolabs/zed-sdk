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

#include <optional>
#include <string>

#include <sl/Camera.hpp>

//
// Console output for the sample. None of this is part of the ZED API: it only exists so the
// sample tells you what it is doing.
//

// Centered title between two horizontal rules.
void printHeader(const std::string& text);

// One "[Sample]" line. Passing an error code tags the line [Error] or [Warning] and, unless
// showErrorDetail is false, appends the SDK explanation of that code.
void print(const std::string& message, std::optional<sl::ERROR_CODE> errorCode = std::nullopt, bool showErrorDetail = true);

// Recap of the tracking configuration the camera was started with, then the viewer shortcuts.
void printTrackingParameters(const sl::PositionalTrackingParameters& trackingParameters);
