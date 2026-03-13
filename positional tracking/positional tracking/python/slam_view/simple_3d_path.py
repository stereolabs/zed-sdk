###########################################################################
#
# Copyright (c) 2025, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
###########################################################################

import numpy as np
from OpenGL.GL import *

from .gl_shader import GLShader

#
# Simple3DPath
#
class Simple3DPath:

    DEFAULT_MAX_POINTS = 100000
    
    def __init__(self, max_points=DEFAULT_MAX_POINTS):
        self._points = []
        self._max_points = max_points
        self._color = (1.0, 1.0, 1.0, 1.0)
        self._mvp = np.identity(4, dtype=np.float32)
        self._vao = 0
        self._vbo = 0
        self._drawing_type = GL_LINE_STRIP
        self._shader = GLShader()
        self._is_dirty = False

    def add_point(self, point):
        self._points.append(point)

        # Prevent unbounded memory growth by trimming old points
        if self._max_points > 0 and len(self._points) > self._max_points:
            excess = len(self._points) - self._max_points
            del self._points[:excess]

        self._is_dirty = True

    def set_color(self, color):
        self._color = color

    def set_mvp(self, mvp):
        self._mvp = mvp

    def draw(self):
        self._push_to_gpu()

        # Stash GL state
        prev_program = glGetIntegerv(GL_CURRENT_PROGRAM)
        prev_vao = glGetIntegerv(GL_VERTEX_ARRAY_BINDING)

        self._shader.use()

        mvp_loc = glGetUniformLocation(self._shader.program, "uMVP")
        glUniformMatrix4fv(mvp_loc, 1, GL_TRUE, self._mvp)

        color_loc = glGetUniformLocation(self._shader.program, "uColor")
        glUniform4f(color_loc, self._color[0], self._color[1], self._color[2], self._color[3])

        glBindVertexArray(self._vao)
        glDrawArrays(self._drawing_type, 0, GLsizei(len(self._points)))

        # Pop GL state
        glBindVertexArray(prev_vao)
        glUseProgram(prev_program)
    
    def clear(self):
        self._points.clear()
        self._is_dirty = True

    def release(self):
        if glIsBuffer(self._vbo):
            glDeleteBuffers(1, [self._vbo])
        
        if glIsVertexArray(self._vao):
            glDeleteVertexArrays(1, [self._vao])

        self._shader.release()

    def _push_to_gpu(self):
        self._create_resources_if_necessary()

        if not self._is_dirty:
            return

        points = np.array(self._points, dtype=np.float32)

        glBindVertexArray(self._vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glBufferData(GL_ARRAY_BUFFER, points.nbytes, points, GL_DYNAMIC_DRAW);

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        self._is_dirty = False

    def _create_resources_if_necessary(self):
        if glIsVertexArray(self._vao):
            return
        
        self._vao = glGenVertexArrays(1)
        self._vbo = glGenBuffers(1)

        self._shader.load(GLShader.MVP_VERTEX, GLShader.UNIFORM_COLOR_FRAGMENT)
