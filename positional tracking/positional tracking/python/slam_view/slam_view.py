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

from enum import Enum
import time
import numpy as np
import pyzed.sl as sl

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.GLUT.fonts import GLUT_BITMAP_HELVETICA_18

from .gl_shader import GLShader
from .gl_camera import GLCamera
from .simple_3d_object import Simple3DObject
from .simple_3d_path import Simple3DPath
from .point_cloud import PointCloud

#
# Style
#
TO_F = 1.0 / 255.0
COLOR_LIME = [217 * TO_F, 255 * TO_F, 66 * TO_F, 1.0]
COLOR_PEARL = [242 * TO_F, 242 * TO_F, 242 * TO_F, 1.0]
COLOR_IRON = [194 * TO_F, 194 * TO_F, 194 * TO_F, 1.0]
COLOR_SOIL = [25 * TO_F, 25 * TO_F, 25 * TO_F, 1.0]
COLOR_GREEN = [67 * TO_F, 255 * TO_F, 63 * TO_F, 1.0]
COLOR_RED = [255 * TO_F, 100 * TO_F, 100 * TO_F, 1.0]

FONT = GLUT_BITMAP_HELVETICA_18

#
# Shaders
#
MESH_VERTEX_SHADER = """
#version 330 core

layout(location=0) in vec3 in_Vertex;
layout(location=1) in vec4 in_Color;
uniform mat4 u_mvpMatrix;
out vec4 v_color;

void main() {
    v_color = in_Color;
    gl_Position = u_mvpMatrix * vec4(in_Vertex, 1.0);
}
"""

MESH_FRAGMENT_SHADER = """
#version 330 core

in vec4 v_color;
out vec4 fragColor;

void main() { 
    fragColor = v_color; 
}
"""

#
# Interaction definitions
#
class MOUSE_WHEEL(Enum):
    UP = 3
    DOWN = 4

#
# SLAMView
#
class SLAMView:

    #
    # Initialization
    #
    def __init__(self, frame: sl.Mat, point_cloud: sl.Mat, title="ZED Positional Tracking"):
        self._frame = frame
        self._dark_mode = True
        self._side_by_side_mode = True
        self._point_cloud_mode = True
        self._landmark_mode = False
        self._follow_mode = False
        self._frame_texture_id = 0
        self._pose_transform = sl.Transform()
        self._positional_tracking_status = sl.PositionalTrackingStatus()

        self._start_time = time.monotonic()

        self._mouse_button_pressed = False
        self._ctrl_pressed = False
        self._mouse_position = [0.0, 0.0]

        glutInit([])
        glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION)
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA)

        screen_width = glutGet(GLUT_SCREEN_WIDTH)
        screen_height = glutGet(GLUT_SCREEN_HEIGHT)

        self._width = 0.6 * screen_width
        self._height = 0.6 * screen_height

        glutInitWindowSize(int(self._width), int(self._height))
        glutInitWindowPosition(int(0.2 * screen_width), int(0.2 * screen_height))
        glutCreateWindow(title)

        glutIdleFunc(self._on_idle)
        glutDisplayFunc(self._on_display)
        glutReshapeFunc(self._on_reshape)
        glutKeyboardFunc(self._on_key)
        glutMouseFunc(self._on_mouse_button_pressed)
        glutMotionFunc(self._on_mouse_motion)
        glutCloseFunc(self._on_close)

        self._shader = GLShader()
        self._shader.load(MESH_VERTEX_SHADER, MESH_FRAGMENT_SHADER)
        self._mvp_matrix = glGetUniformLocation(self._shader.program, "u_mvpMatrix")

        self._camera = GLCamera(target=(-1.4, 0.8, -0.9), distance=3.5, yaw=-18, pitch=27)

        self._origin_axes = Simple3DObject(drawing_type=GL_LINES, is_static=True)
      
        self._origin_axes.add_point((0, 0, 0), (1, 0, 0, 1))
        self._origin_axes.add_point((1, 0, 0), (1, 0, 0, 1))

        self._origin_axes.add_point((0, 0, 0), (0, 1, 0, 1))
        self._origin_axes.add_point((0, 1, 0), (0, 1, 0, 1))

        self._origin_axes.add_point((0, 0, 0), (0, 0, 1, 1))
        self._origin_axes.add_point((0, 0, 1), (0, 0, 1, 1))

        fx = 1400.0
        fy = 1400.0
        width = 2208
        height = 1242
        cx = width / 2.0
        cy = height / 2.0
        z = 0.15

        # Axis flip to OpenGL convention (x, -y, -z)
        to_opengl = np.array([1.0, -1.0, -1.0], dtype=np.float32)

        # Project pixel (u,v) at depth z into camera space with axis transform
        def pixel_to_camera(u, v, z, fx, fy, cx, cy, axis_transform):
            x_projected = (u - cx) * (z / fx)
            y_projected = (v - cy) * (z / fy)
            projection = np.array([x_projected, y_projected, z], dtype=np.float32)

            return projection * np.array(axis_transform, dtype=np.float32)

        cam_0 = np.zeros(3, dtype=np.float32)
        cam_1 = pixel_to_camera(0, 0, z, fx, fy, cx, cy, to_opengl)
        cam_2 = pixel_to_camera(width, 0, z, fx, fy, cx, cy, to_opengl)
        cam_3 = pixel_to_camera(width, height, z, fx, fy, cx, cy, to_opengl)
        cam_4 = pixel_to_camera(0, height, z, fx, fy, cx, cy, to_opengl)

        self._camera_frustum = Simple3DObject(drawing_type=GL_LINES)

        self._camera_frustum.add_point(cam_0, COLOR_LIME)
        self._camera_frustum.add_point(cam_1, COLOR_LIME)

        self._camera_frustum.add_point(cam_0, COLOR_LIME)
        self._camera_frustum.add_point(cam_2, COLOR_LIME)
        
        self._camera_frustum.add_point(cam_0, COLOR_LIME)
        self._camera_frustum.add_point(cam_3, COLOR_LIME)
        
        self._camera_frustum.add_point(cam_0, COLOR_LIME)
        self._camera_frustum.add_point(cam_4, COLOR_LIME)

        self._point_cloud = PointCloud(point_cloud)

        self._landmarks = Simple3DObject(drawing_type=GL_POINTS)

        self._camera_path = Simple3DPath()

    #
    # Public API
    #
    @property
    def is_landmark_mode_enabled(self) -> bool:
        return self._landmark_mode
    
    def update_pose_transform(self, pose_transform: sl.Transform):
        self._pose_transform = pose_transform
        translation = pose_transform.get_translation()
        self._camera_path.add_point(tuple(translation.get()))

    def update_positional_tracking_status(self, positional_tracking_status):
        self._positional_tracking_status = positional_tracking_status

    def update_landmarks(self, landmarks):
        self._landmarks.clear()

        for landmark in landmarks.values():
            self._landmarks.add_point(landmark.position, COLOR_LIME)

    def run(self, callback):
        self._callback = callback
        glutMainLoop()

    def stop(self):
        glutLeaveMainLoop()

    #
    # GLUT Callbacks
    #
    def _on_idle(self):
        self._callback()
        glutPostRedisplay()

    def _on_display(self):
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_POINT_SMOOTH)
        glEnable(GL_DEPTH_TEST)

        color = COLOR_SOIL if self._dark_mode else COLOR_PEARL
        glClearColor(color[0], color[1], color[2], color[3])
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        vp_matrix = self._camera.projection @ self._camera.view

        glPolygonMode(GL_FRONT, GL_LINE)
        glPolygonMode(GL_BACK, GL_LINE)
        glLineWidth(3.0)

        glUseProgram(self._shader.program)

        if self._follow_mode:
            pose_w_t = sl.Transform(self._pose_transform)
            euler_angles = pose_w_t.get_euler_angles()
            pose_w_t.set_euler_angles(0, euler_angles[1], 0)
            sl.Transform.inverse(pose_w_t)
            vp_matrix = vp_matrix @ pose_w_t.m

        glUniformMatrix4fv(self._mvp_matrix, 1, GL_TRUE, vp_matrix)
        self._origin_axes.draw()

        self._camera_path.set_color(COLOR_PEARL if self._dark_mode else COLOR_SOIL)
        self._camera_path.set_mvp(vp_matrix)
        self._camera_path.draw()

        if self._landmark_mode:
            glPointSize(1.5)
            self._landmarks.draw()

        glPointSize(1.0)

        transformed_pose = vp_matrix @ self._pose_transform.m
        glUniformMatrix4fv(self._mvp_matrix, 1, GL_TRUE, transformed_pose)
        self._camera_frustum.draw()
        glUseProgram(0)

        if self._point_cloud_mode:
            self._point_cloud.draw(transformed_pose)

        glDisable(GL_DEPTH_TEST)

        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        self._render_positional_tracking_status()

        if self._side_by_side_mode:
            self._render_image_frame()

        glutSwapBuffers()

    def _on_reshape(self, width: int, height: int):
        self._width = width
        self._height = height

        glViewport(0, 0, width, height)

        aspect_ratio = width / height
        self._camera.resize_viewport(aspect_ratio)

    def _on_key(self, key, x, y):
        if key == b' ':
            self._side_by_side_mode = not(self._side_by_side_mode)
        elif key == b'd':
            self._dark_mode = not(self._dark_mode)
        elif key == b'p':
            self._point_cloud_mode = not(self._point_cloud_mode)
        elif key == b'l':
            self._landmark_mode = not(self._landmark_mode)
        elif key == b'f':
            self._follow_mode = not(self._follow_mode)
            self._camera.reset()
        elif key == b'z':
            self._camera.reset()
        elif key == b'\x1b':
            self.stop()

    def _on_mouse_button_pressed(self, button, state, x, y):
        zoomSensitivity = 0.75

        if button == GLUT_LEFT_BUTTON:
            self._mouse_button_pressed = (state == GLUT_DOWN)

            if self._mouse_button_pressed:
                self._mouse_position[0] = x
                self._mouse_position[1] = y
                self._ctrl_pressed = glutGetModifiers() & GLUT_ACTIVE_CTRL
        
        elif button == MOUSE_WHEEL.DOWN.value:
            self._camera.zoom(-zoomSensitivity)
        elif button == MOUSE_WHEEL.UP.value:
            self._camera.zoom(zoomSensitivity)

    def _on_mouse_motion(self, x, y):
        if not self._mouse_button_pressed:
            return

        delta_x = x - self._mouse_position[0]
        delta_y = y - self._mouse_position[1]

        if self._ctrl_pressed:
            rotate_sensitivity = 0.05
            self._camera.rotate(-delta_x * rotate_sensitivity, delta_y * rotate_sensitivity)
        else:
            translate_sensitivity = 0.007
            self._camera.pan(delta_x * translate_sensitivity, delta_y * translate_sensitivity)

        self._mouse_position[0] = x
        self._mouse_position[1] = y

    def _on_close(self):
        self._origin_axes.release()
        self._camera_frustum.release()
        self._camera_path.release()
        self._point_cloud.release()

        if glIsTexture(self._frame_texture_id):
            glDeleteTextures(1, self._frame_texture_id)

        self._shader.release()

    #
    # Drawing Specifics
    #
    def _render_positional_tracking_status(self):
        # Save current projection and modelview matrices
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluOrtho2D(0, self._width, 0, self._height)

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        start_x = 50
        start_y = self._height - 50
        vertical_spacing = 25
        status_value_x = start_x + 150

        text_color = COLOR_PEARL if self._dark_mode else COLOR_SOIL

        # Fusion status
        self._render_text(start_x, start_y, FONT, text_color, "Fusion:")

        status_color = COLOR_GREEN if self._positional_tracking_status.tracking_fusion_status != sl.POSITIONAL_TRACKING_FUSION_STATUS.UNAVAILABLE else COLOR_RED

        self._render_text(status_value_x, start_y, FONT, status_color, str(self._positional_tracking_status.tracking_fusion_status))

        # Spatial memory status
        self._render_text(start_x, start_y - vertical_spacing, FONT, text_color, "Spatial Memory:");

        status_color = COLOR_GREEN if self._positional_tracking_status.spatial_memory_status != sl.SPATIAL_MEMORY_STATUS.OFF else COLOR_RED

        spatial_memory_status_text = str(self._positional_tracking_status.spatial_memory_status)

        self._render_text(status_value_x, start_y - vertical_spacing, FONT, status_color, spatial_memory_status_text)

        # Odometry status
        self._render_text(start_x, start_y - 2 * vertical_spacing, FONT, text_color, "Odometry:")

        status_color = COLOR_GREEN if self._positional_tracking_status.odometry_status != sl.ODOMETRY_STATUS.UNAVAILABLE else COLOR_RED

        odometry_status_text = str(self._positional_tracking_status.odometry_status)

        self._render_text(status_value_x, start_y - 2 * vertical_spacing, FONT, status_color, odometry_status_text)

        # Up time
        elapsed = time.monotonic() - self._start_time
        total_seconds = int(elapsed)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        uptime_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        self._render_text(start_x, start_y - 4 * vertical_spacing, FONT, text_color, "Up Time:")
        self._render_text(status_value_x, start_y - 4 * vertical_spacing, FONT, text_color, uptime_str)

        # Pose transform
        self._render_text(start_x, start_y - 6 * vertical_spacing, FONT, text_color, "Translation (m):")
        self._render_text(status_value_x, start_y - 6 * vertical_spacing, FONT, text_color, self._format_numeric_text(self._pose_transform.get_translation().get()))

        self._render_text(start_x, start_y - 7 * vertical_spacing, FONT, text_color, "Rotation   (rad):")
        self._render_text(status_value_x, start_y - 7 * vertical_spacing, FONT, text_color, self._format_numeric_text(self._pose_transform.get_rotation_vector()))

        # Restore matrices
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

    def _format_numeric_text(self, values):
        return f"{values[0]:.2f}, {values[1]:.2f}, {values[2]:.2f}"

    def _render_text(self, x, y, font, color, text):
        glColor3f(color[0], color[1], color[2])
        glRasterPos2i(x, y)
        glutBitmapString(font, text.encode("utf-8"))

    def _render_image_frame(self):
        if self._frame.get_width() == 0 or self._frame.get_height() == 0:
            return

        self._frame_texture_id = self._upload_image_to_texture(self._frame, self._frame_texture_id)

        max_width_fraction = 0.3
        min_width = 512

        max_width = self._width * max_width_fraction
        display_width = max(min_width, max_width)

        scale = display_width / self._frame.get_width()
        scaled_width = display_width
        scaled_height = self._frame.get_height() * scale

        self._render_texture(self._frame_texture_id, 50, 50, scaled_width, scaled_height)

    def _upload_image_to_texture(self, frame, texture):
        if not glIsTexture(texture):
            texture = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texture)

            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

            glTexImage2D(
                GL_TEXTURE_2D,
                0,
                GL_RGBA,
                frame.get_width(),
                frame.get_height(),
                0,
                GL_BGRA,
                GL_UNSIGNED_BYTE,
                frame.get_data()
            )
        else:
            glBindTexture(GL_TEXTURE_2D, texture)
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, frame.get_width(), frame.get_height(), GL_BGRA, GL_UNSIGNED_BYTE, frame.get_data())

        return texture

    def _render_texture(self, texture_id, x, y, width, height):
        # Save current projection and model view
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluOrtho2D(0, self._width, 0, self._height)

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        # Bind to texture
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glColor3f(1.0, 1.0, 1.0)

        # Render in rect
        glBegin(GL_QUADS);
        glTexCoord2f(0.0, 1.0)
        glVertex2i(x, y)
        glTexCoord2f(1.0, 1.0)
        glVertex2i(x + int(width), y)
        glTexCoord2f(1.0, 0.0)
        glVertex2i(x + int(width), y + int(height))
        glTexCoord2f(0.0, 0.0)
        glVertex2i(x, y + int(height))
        glEnd()

        glDisable(GL_TEXTURE_2D)

        # Restore projection and model view
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
