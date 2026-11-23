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

#include <algorithm>
#include <cmath>
#include <string>

#include <GL/glew.h>
#include <GL/freeglut.h>

#include <sl/Camera.hpp>

//
// Flat 2D drawing over the 3D view: panels, badges and the timeline strip. Window pixels, origin at
// the bottom left as OpenGL has it (GLUT mouse positions have it at the top left: flip them first).
// Everything between begin() and end() is drawn with the fixed pipeline, depth test off.
//
namespace overlay {

    inline void begin(int width, int height) {
        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();
        gluOrtho2D(0, width, 0, height);

        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadIdentity();
    }

    inline void end() {
        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
        glMatrixMode(GL_MODELVIEW);
        glPopMatrix();
    }

    inline void fillRect(float x, float y, float width, float height, const sl::float4& color) {
        glColor4f(color.x, color.y, color.z, color.w);
        glBegin(GL_QUADS);
        glVertex2f(x, y);
        glVertex2f(x + width, y);
        glVertex2f(x + width, y + height);
        glVertex2f(x, y + height);
        glEnd();
    }

    inline void strokeRect(float x, float y, float width, float height, const sl::float4& color, float lineWidth = 1.f) {
        glLineWidth(lineWidth);
        glColor4f(color.x, color.y, color.z, color.w);
        glBegin(GL_LINE_LOOP);
        glVertex2f(x, y);
        glVertex2f(x + width, y);
        glVertex2f(x + width, y + height);
        glVertex2f(x, y + height);
        glEnd();
    }

    // Dashed when dash > 0: dash pixels drawn, then as many skipped.
    inline void line(float x0, float y0, float x1, float y1, const sl::float4& color, float lineWidth = 1.f, float dash = 0.f) {
        glLineWidth(lineWidth);
        glColor4f(color.x, color.y, color.z, color.w);
        glBegin(GL_LINES);

        const float length = std::sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0));
        if (dash <= 0.f || length <= dash) {
            glVertex2f(x0, y0);
            glVertex2f(x1, y1);
        } else {
            for (float from = 0.f; from < length; from += 2.f * dash) {
                const float to = std::min(from + dash, length);
                glVertex2f(x0 + (x1 - x0) * from / length, y0 + (y1 - y0) * from / length);
                glVertex2f(x0 + (x1 - x0) * to / length, y0 + (y1 - y0) * to / length);
            }
        }

        glEnd();
    }

    inline void triangle(float x0, float y0, float x1, float y1, float x2, float y2, const sl::float4& color) {
        glColor4f(color.x, color.y, color.z, color.w);
        glBegin(GL_TRIANGLES);
        glVertex2f(x0, y0);
        glVertex2f(x1, y1);
        glVertex2f(x2, y2);
        glEnd();
    }

    inline void diamond(float x, float y, float radius, const sl::float4& color) {
        glColor4f(color.x, color.y, color.z, color.w);
        glBegin(GL_QUADS);
        glVertex2f(x, y - radius);
        glVertex2f(x + radius, y);
        glVertex2f(x, y + radius);
        glVertex2f(x - radius, y);
        glEnd();
    }

    // y is the baseline.
    inline void text(float x, float y, void* font, const sl::float4& color, const std::string& str) {
        glColor4f(color.x, color.y, color.z, color.w);
        glRasterPos2f(x, y);
        glutBitmapString(font, reinterpret_cast<const unsigned char*>(str.c_str()));
    }

    inline int textWidth(void* font, const std::string& str) {
        return glutBitmapLength(font, reinterpret_cast<const unsigned char*>(str.c_str()));
    }

} // namespace overlay
