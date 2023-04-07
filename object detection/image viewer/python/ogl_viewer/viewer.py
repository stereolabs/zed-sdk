from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

from threading import Lock
import numpy as np
import sys
import array
import math
import ctypes
import pyzed.sl as sl


VERTEX_SHADER = """
# version 330 core
layout(location = 0) in vec3 in_Vertex;
layout(location = 1) in vec4 in_Color;
uniform mat4 u_mvpMatrix;
out vec4 b_color;
void main() {
    b_color = in_Color;
    gl_Position = u_mvpMatrix * vec4(in_Vertex, 1);
}
"""

FRAGMENT_SHADER = """
# version 330 core
in vec4 b_color;
layout(location = 0) out vec4 out_Color;
void main() {
   out_Color = b_color;
}
"""

M_PI = 3.1415926

GRID_SIZE = 15.0

CLASS_COLORS = np.array([
	[44, 117, 255]          # People
	, [255, 0, 255]         # Vehicle
	, [0, 0, 255]
    , [0, 255, 255]
    , [0, 255, 0]
    , [255, 255, 255]]
    , np.float32)

ID_COLORS = np.array([
	[0.231, 0.909, 0.69]
    , [0.098, 0.686, 0.816]
    , [0.412, 0.4, 0.804]
    , [1, 0.725, 0]
    , [0.989, 0.388, 0.419]]
    , np.float32)

def get_color_class(_idx):
    _idx = min(5, _idx)
    clr = [CLASS_COLORS[_idx][0], CLASS_COLORS[_idx][1], CLASS_COLORS[_idx][2], 1.0]
    return np.divide(clr, 255.0)

def generate_color_id(_idx):
    clr = []
    if _idx < 0:
        clr = [236, 184, 36, 255]
        clr = np.divide(clr, 255.0)
    else:
        offset = _idx % 5
        clr = [ID_COLORS[offset][0], ID_COLORS[offset][1], ID_COLORS[offset][2], 1]
    return clr

class Shader:
    def __init__(self, _vs, _fs):

        self.program_id = glCreateProgram()
        vertex_id = self.compile(GL_VERTEX_SHADER, _vs)
        fragment_id = self.compile(GL_FRAGMENT_SHADER, _fs)

        glAttachShader(self.program_id, vertex_id)
        glAttachShader(self.program_id, fragment_id)
        glBindAttribLocation( self.program_id, 0, "in_vertex")
        glBindAttribLocation( self.program_id, 1, "in_texCoord")
        glLinkProgram(self.program_id)

        if glGetProgramiv(self.program_id, GL_LINK_STATUS) != GL_TRUE:
            info = glGetProgramInfoLog(self.program_id)
            glDeleteProgram(self.program_id)
            glDeleteShader(vertex_id)
            glDeleteShader(fragment_id)
            raise RuntimeError('Error linking program: %s' % (info))
        glDeleteShader(vertex_id)
        glDeleteShader(fragment_id)

    def compile(self, _type, _src):
        try:
            shader_id = glCreateShader(_type)
            if shader_id == 0:
                print("ERROR: shader type {0} does not exist".format(_type))
                exit()

            glShaderSource(shader_id, _src)
            glCompileShader(shader_id)
            if glGetShaderiv(shader_id, GL_COMPILE_STATUS) != GL_TRUE:
                info = glGetShaderInfoLog(shader_id)
                glDeleteShader(shader_id)
                raise RuntimeError('Shader compilation failed: %s' % (info))
            return shader_id
        except:
            glDeleteShader(shader_id)
            raise

    def get_program_id(self):
        return self.program_id


class Simple3DObject:
    """
    Class that manages simple 3D objects to render with OpenGL
    """
    def __init__(self, _is_static):
        self.vaoID = 0
        self.drawing_type = GL_TRIANGLES
        self.is_static = _is_static
        self.elementbufferSize = 0

        self.vertices = array.array('f')
        self.colors = array.array('f')
        self.normals = array.array('f')
        self.indices = array.array('I')

    def __del__(self):
        if self.vaoID:
            self.vaoID = 0


    """
    Add a unique point to the list of points
    """
    def add_pt(self, _pts):
        for pt in _pts:
            self.vertices.append(pt)

    """
    Add a unique color to the list of colors
    """
    def add_clr(self, _clrs):
        for clr in _clrs:
            self.colors.append(clr)

    """
    Add a unique normal to the list of normals
    """
    def add_normal(self, _normals):
        for normal in _normals:
            self.normals.append(normal)

    """
    Add a set of points to the list of points and their corresponding color
    """
    def add_points(self, _pts, _base_clr):
        for i in range(len(_pts)):
            pt = _pts[i]
            self.add_pt(pt)
            self.add_clr(_base_clr)
            current_size_index = (len(self.vertices)/3)-1
            self.indices.append(current_size_index)
            self.indices.append(current_size_index+1)

    """
    Add a point and its corresponding color to the list of points
    """
    def add_point_clr(self, _pt, _clr):
        self.add_pt(_pt)
        self.add_clr(_clr)
        self.indices.append(len(self.indices))

    """
    Define a line from two points
    """
    def add_line(self, _p1, _p2, _clr):
        self.add_point_clr(_p1, _clr)
        self.add_point_clr(_p2, _clr)

    def add_full_edges(self, _pts, _clr):
        start_id = int(len(self.vertices) / 3)
        _clr[3] = 0.2

        for i in range(len(_pts)):
            self.add_pt(_pts[i])
            self.add_clr(_clr)

        box_links_top = np.array([0, 1, 1, 2, 2, 3, 3, 0])
        i = 0
        while i < box_links_top.size:
            self.indices.append(start_id + box_links_top[i])
            self.indices.append(start_id + box_links_top[i+1])
            i = i + 2

        box_links_bottom = np.array([4, 5, 5, 6, 6, 7, 7, 4])
        i = 0
        while i < box_links_bottom.size:
            self.indices.append(start_id + box_links_bottom[i])
            self.indices.append(start_id + box_links_bottom[i+1])
            i = i + 2

    def __add_single_vertical_line(self, _top_pt, _bottom_pt, _clr):
        current_pts = np.array(
            [_top_pt,
            ((GRID_SIZE - 1) * np.array(_top_pt) + np.array(_bottom_pt)) / GRID_SIZE,
            ((GRID_SIZE - 2) * np.array(_top_pt) + np.array(_bottom_pt) * 2) / GRID_SIZE,
            (2 * np.array(_top_pt) + np.array(_bottom_pt) * (GRID_SIZE - 2)) / GRID_SIZE,
            (np.array(_top_pt) + np.array(_bottom_pt) * (GRID_SIZE - 1)) / GRID_SIZE,
            _bottom_pt
            ], np.float32)
        start_id = int(len(self.vertices) / 3)
        for i in range(len(current_pts)):
            self.add_pt(current_pts[i])
            if (i == 2 or i == 3):
                _clr[3] = 0
            else:
                _clr[3] = 0.2
            self.add_clr(_clr)

        box_links = np.array([0, 1, 1, 2, 2, 3, 3, 4, 4, 5])
        i = 0
        while i < box_links.size:
            self.indices.append(start_id + box_links[i])
            self.indices.append(start_id + box_links[i+1])
            i = i + 2

    def add_vertical_edges(self, _pts, _clr):
        self.__add_single_vertical_line(_pts[0], _pts[4], _clr)
        self.__add_single_vertical_line(_pts[1], _pts[5], _clr)
        self.__add_single_vertical_line(_pts[2], _pts[6], _clr)
        self.__add_single_vertical_line(_pts[3], _pts[7], _clr)

    def add_top_face(self, _pts, _clr):
        _clr[3] = 0.25
        for pt in _pts:
            self.add_point_clr(pt, _clr)

    def __add_quad(self, _quad_pts, _alpha1, _alpha2, _clr):
        for i in range(len(_quad_pts)):
            self.add_pt(_quad_pts[i])
            if i < 2:
                _clr[3] = _alpha1
            else:
                _clr[3] = _alpha2
            self.add_clr(_clr)

        self.indices.append(len(self.indices))
        self.indices.append(len(self.indices))
        self.indices.append(len(self.indices))
        self.indices.append(len(self.indices))

    def add_vertical_faces(self, _pts, _clr):
    	# For each face, we need to add 4 quads (the first 2 indexes are always the top points of the quad)
        quads = [[0, 3, 7, 4]       # Front face
                , [3, 2, 6, 7]      # Right face
                , [2, 1, 5, 6]      # Back face
                , [1, 0, 4, 5]]     # Left face

        alpha = 0.25

        # Create gradually fading quads
        for quad in quads:
            quad_pts_1 = [
                _pts[quad[0]],
                _pts[quad[1]],
                ((GRID_SIZE - 0.5) * np.array(_pts[quad[1]]) + 0.5 * np.array(_pts[quad[2]])) / GRID_SIZE,
                ((GRID_SIZE - 0.5) * np.array(_pts[quad[0]]) + 0.5 * np.array(_pts[quad[3]])) / GRID_SIZE
            ]
            self.__add_quad(quad_pts_1, alpha, alpha, _clr)

            quad_pts_2 = [
                ((GRID_SIZE - 0.5) * np.array(_pts[quad[0]]) + 0.5 * np.array(_pts[quad[3]])) / GRID_SIZE,
                ((GRID_SIZE - 0.5) * np.array(_pts[quad[1]]) + 0.5 * np.array(_pts[quad[2]])) / GRID_SIZE,
                ((GRID_SIZE - 1.0) * np.array(_pts[quad[1]]) + np.array(_pts[quad[2]])) / GRID_SIZE,
                ((GRID_SIZE - 1.0) * np.array(_pts[quad[0]]) + np.array(_pts[quad[3]])) / GRID_SIZE
            ]
            self.__add_quad(quad_pts_2, alpha, 2 * alpha / 3, _clr)

            quad_pts_3 = [
                ((GRID_SIZE - 1.0) * np.array(_pts[quad[0]]) + np.array(_pts[quad[3]])) / GRID_SIZE,
                ((GRID_SIZE - 1.0) * np.array(_pts[quad[1]]) + np.array(_pts[quad[2]])) / GRID_SIZE,
                ((GRID_SIZE - 1.5) * np.array(_pts[quad[1]]) + 1.5 * np.array(_pts[quad[2]])) / GRID_SIZE,
                ((GRID_SIZE - 1.5) * np.array(_pts[quad[0]]) + 1.5 * np.array(_pts[quad[3]])) / GRID_SIZE
            ]
            self.__add_quad(quad_pts_3, 2 * alpha / 3, alpha / 3, _clr)

            quad_pts_4 = [
                ((GRID_SIZE - 1.5) * np.array(_pts[quad[0]]) + 1.5 * np.array(_pts[quad[3]])) / GRID_SIZE,
                    ((GRID_SIZE - 1.5) * np.array(_pts[quad[1]]) + 1.5 * np.array(_pts[quad[2]])) / GRID_SIZE,
                    ((GRID_SIZE - 2.0) * np.array(_pts[quad[1]]) + 2.0 * np.array(_pts[quad[2]])) / GRID_SIZE,
                    ((GRID_SIZE - 2.0) * np.array(_pts[quad[0]]) + 2.0 * np.array(_pts[quad[3]])) / GRID_SIZE
            ]
            self.__add_quad(quad_pts_4, alpha / 3, 0.0, _clr)

            quad_pts_5 = [
                (np.array(_pts[quad[1]]) * 2.0 + (GRID_SIZE - 2.0) * np.array(_pts[quad[2]])) / GRID_SIZE,
                (np.array(_pts[quad[0]]) * 2.0 + (GRID_SIZE - 2.0) * np.array(_pts[quad[3]])) / GRID_SIZE,
                (np.array(_pts[quad[0]]) * 1.5 + (GRID_SIZE - 1.5) * np.array(_pts[quad[3]])) / GRID_SIZE,
                (np.array(_pts[quad[1]]) * 1.5 + (GRID_SIZE - 1.5) * np.array(_pts[quad[2]])) / GRID_SIZE
            ]
            self.__add_quad(quad_pts_5, 0.0, alpha / 3, _clr)

            quad_pts_6 = [
                (np.array(_pts[quad[1]]) * 1.5 + (GRID_SIZE - 1.5) * np.array(_pts[quad[2]])) / GRID_SIZE,
                (np.array(_pts[quad[0]]) * 1.5 + (GRID_SIZE - 1.5) * np.array(_pts[quad[3]])) / GRID_SIZE,
                (np.array(_pts[quad[0]]) + (GRID_SIZE - 1.0) * np.array(_pts[quad[3]])) / GRID_SIZE,
                (np.array(_pts[quad[1]]) + (GRID_SIZE - 1.0) * np.array(_pts[quad[2]])) / GRID_SIZE
            ]
            self.__add_quad(quad_pts_6, alpha / 3, 2 * alpha / 3, _clr)

            quad_pts_7 = [
                (np.array(_pts[quad[1]]) + (GRID_SIZE - 1.0) * np.array(_pts[quad[2]])) / GRID_SIZE,
                (np.array(_pts[quad[0]]) + (GRID_SIZE - 1.0) * np.array(_pts[quad[3]])) / GRID_SIZE,
                (np.array(_pts[quad[0]]) * 0.5 + (GRID_SIZE - 0.5) * np.array(_pts[quad[3]])) / GRID_SIZE,
                (np.array(_pts[quad[1]]) * 0.5 + (GRID_SIZE - 0.5) * np.array(_pts[quad[2]])) / GRID_SIZE
            ]
            self.__add_quad(quad_pts_7, 2 * alpha / 3, alpha, _clr)

            quad_pts_8 = [
                (np.array(_pts[quad[0]]) * 0.5 + (GRID_SIZE - 0.5) * np.array(_pts[quad[3]])) / GRID_SIZE,
                (np.array(_pts[quad[1]]) * 0.5 + (GRID_SIZE - 0.5) * np.array(_pts[quad[2]])) / GRID_SIZE,
                np.array(_pts[quad[2]]),
                np.array(_pts[quad[3]])
            ]
            self.__add_quad(quad_pts_8, alpha, alpha, _clr)

    def push_to_GPU(self):
        self.vboID = glGenBuffers(4)

        if len(self.vertices):
            glBindBuffer(GL_ARRAY_BUFFER, self.vboID[0])
            glBufferData(GL_ARRAY_BUFFER, len(self.vertices) * self.vertices.itemsize, (GLfloat * len(self.vertices))(*self.vertices), GL_STATIC_DRAW)

        if len(self.colors):
            glBindBuffer(GL_ARRAY_BUFFER, self.vboID[1])
            glBufferData(GL_ARRAY_BUFFER, len(self.colors) * self.colors.itemsize, (GLfloat * len(self.colors))(*self.colors), GL_STATIC_DRAW)

        if len(self.indices):
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.vboID[2])
            glBufferData(GL_ELEMENT_ARRAY_BUFFER,len(self.indices) * self.indices.itemsize,(GLuint * len(self.indices))(*self.indices), GL_STATIC_DRAW)

        if len(self.normals):
            glBindBuffer(GL_ARRAY_BUFFER, self.vboID[3])
            glBufferData(GL_ARRAY_BUFFER, len(self.normals) * self.normals.itemsize, (GLfloat * len(self.normals))(*self.normals), GL_STATIC_DRAW)
            glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, 0)
            glEnableVertexAttribArray(2)

        self.elementbufferSize = len(self.indices)

    def clear(self):
        self.vertices = array.array('f')
        self.colors = array.array('f')
        self.normals = array.array('f')
        self.indices = array.array('I')

    def set_drawing_type(self, _type):
        self.drawing_type = _type

    def draw(self):
        if (self.elementbufferSize):
            glEnableVertexAttribArray(0)
            glBindBuffer(GL_ARRAY_BUFFER, self.vboID[0])
            glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,0,None)

            glEnableVertexAttribArray(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.vboID[1])
            glVertexAttribPointer(1,4,GL_FLOAT,GL_FALSE,0,None)

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.vboID[2])
            glDrawElements(self.drawing_type, self.elementbufferSize, GL_UNSIGNED_INT, None)

            glDisableVertexAttribArray(0)
            glDisableVertexAttribArray(1)

IMAGE_FRAGMENT_SHADER = """
# version 330 core
in vec2 UV;
out vec4 color;
uniform sampler2D texImage;
uniform bool revert;
uniform bool rgbflip;
void main() {
    vec2 scaler  =revert?vec2(UV.x,1.f - UV.y):vec2(UV.x,UV.y);
    vec3 rgbcolor = rgbflip?vec3(texture(texImage, scaler).zyx):vec3(texture(texImage, scaler).xyz);
    float gamma = 1.0/1.65;
    vec3 color_rgb = pow(rgbcolor, vec3(1.0/gamma));
    color = vec4(color_rgb,1);
}
"""

IMAGE_VERTEX_SHADER = """
# version 330
layout(location = 0) in vec3 vert;
out vec2 UV;
void main() {
   UV = (vert.xy+vec2(1,1))/2;
	gl_Position = vec4(vert, 1);
}
"""

class ImageHandler:
    """
    Class that manages the image stream to render with OpenGL
    """
    def __init__(self):
        self.tex_id = 0
        self.image_tex = 0
        self.quad_vb = 0
        self.is_called = 0

    def close(self):
        if self.image_tex:
            self.image_tex = 0

    def initialize(self, _res):
        self.shader_image = Shader(IMAGE_VERTEX_SHADER, IMAGE_FRAGMENT_SHADER)
        self.tex_id = glGetUniformLocation( self.shader_image.get_program_id(), "texImage")

        g_quad_vertex_buffer_data = np.array([-1, -1, 0,
                                                1, -1, 0,
                                                -1, 1, 0,
                                                -1, 1, 0,
                                                1, -1, 0,
                                                1, 1, 0], np.float32)

        self.quad_vb = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.quad_vb)
        glBufferData(GL_ARRAY_BUFFER, g_quad_vertex_buffer_data.nbytes,
                     g_quad_vertex_buffer_data, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        # Create and populate the texture
        glEnable(GL_TEXTURE_2D)

        # Generate a texture name
        self.image_tex = glGenTextures(1)

        # Select the created texture
        glBindTexture(GL_TEXTURE_2D, self.image_tex)

        # Set the texture minification and magnification filters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        # Fill the texture with an image
        # None means reserve texture memory, but texels are undefined
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, _res.width, _res.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)

        # Unbind the texture
        glBindTexture(GL_TEXTURE_2D, 0)

    def push_new_image(self, _zed_mat):
        glBindTexture(GL_TEXTURE_2D, self.image_tex)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, _zed_mat.get_width(), _zed_mat.get_height(), GL_RGBA, GL_UNSIGNED_BYTE,  ctypes.c_void_p(_zed_mat.get_pointer()))
        glBindTexture(GL_TEXTURE_2D, 0)

    def draw(self):
        glUseProgram(self.shader_image.get_program_id())
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.image_tex)
        glUniform1i(self.tex_id, 0)

        # invert y axis and color for this image
        glUniform1i(glGetUniformLocation(self.shader_image.get_program_id(), "revert"), 1)
        glUniform1i(glGetUniformLocation(self.shader_image.get_program_id(), "rgbflip"), 1)

        glEnableVertexAttribArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, self.quad_vb)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glDisableVertexAttribArray(0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glUseProgram(0)

class GLViewer:
    """
    Class that manages the rendering in OpenGL
    """
    def __init__(self):
        self.available = False
        self.objects_name = []
        self.mutex = Lock()
        # Show tracked objects only
        self.is_tracking_on = False

    def init(self, _params, _is_tracking_on):
        glutInit()
        wnd_w = glutGet(GLUT_SCREEN_WIDTH)
        wnd_h = glutGet(GLUT_SCREEN_HEIGHT)
        width = (int)(wnd_w*0.9)
        height = (int)(wnd_h*0.9)

        glutInitWindowSize(width, height)
        glutInitWindowPosition((int)(wnd_w*0.05),(int)(wnd_h*0.05))
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_SRGB)
        glutCreateWindow("ZED Object detection")
        glViewport(0, 0, width, height)

        glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE,
                      GLUT_ACTION_CONTINUE_EXECUTION)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

        # Initialize image renderer
        self.image_handler = ImageHandler()
        self.image_handler.initialize(_params.image_size)

        glEnable(GL_FRAMEBUFFER_SRGB)

        # Compile and create the shader for 3D objects
        self.shader_image = Shader(VERTEX_SHADER, FRAGMENT_SHADER)
        self.shader_MVP = glGetUniformLocation(self.shader_image.get_program_id(), "u_mvpMatrix")

        # Create the rendering camera
        self.projection = array.array('f')
        self.set_render_camera_projection(_params, 0.5, 20)

        # Create the bounding box object
        self.BBox_edges = Simple3DObject(False)
        self.BBox_edges.set_drawing_type(GL_LINES)

        self.BBox_faces = Simple3DObject(False)
        self.BBox_faces.set_drawing_type(GL_QUADS)

        self.is_tracking_on = _is_tracking_on

        # Set OpenGL settings
        glDisable(GL_DEPTH_TEST)    # avoid occlusion with bbox
        glLineWidth(1.5)

        # Register the drawing function with GLUT
        glutDisplayFunc(self.draw_callback)
        # Register the function called when nothing happens
        glutIdleFunc(self.idle)
        # Register the function called on key pressed
        glutKeyboardFunc(self.keyPressedCallback)
        # Register the closing function
        glutCloseFunc(self.close_func)
        self.available = True

    def set_render_camera_projection(self, _params, _znear, _zfar):
        # Just slightly move up the ZED camera FOV to make a small black border
        fov_y = (_params.v_fov + 0.5) * M_PI / 180
        fov_x = (_params.h_fov + 0.5) * M_PI / 180

        self.projection.append( 1 / math.tan(fov_x * 0.5) )  # Horizontal FoV.
        self.projection.append( 0)
        # Horizontal offset.
        self.projection.append( 2 * ((_params.image_size.width - _params.cx) / _params.image_size.width) - 1)
        self.projection.append( 0)

        self.projection.append( 0)
        self.projection.append( 1 / math.tan(fov_y * 0.5))  # Vertical FoV.
        # Vertical offset.
        self.projection.append(-(2 * ((_params.image_size.height - _params.cy) / _params.image_size.height) - 1))
        self.projection.append( 0)

        self.projection.append( 0)
        self.projection.append( 0)
        # Near and far planes.
        self.projection.append( -(_zfar + _znear) / (_zfar - _znear))
        # Near and far planes.
        self.projection.append( -(2 * _zfar * _znear) / (_zfar - _znear))

        self.projection.append( 0)
        self.projection.append( 0)
        self.projection.append( -1)
        self.projection.append( 0)


    def is_available(self):
        if self.available:
            glutMainLoopEvent()
        return self.available

    def render_object(self, _object_data):      # _object_data of type sl.ObjectData
        if self.is_tracking_on:
            return _object_data.tracking_state == sl.OBJECT_TRACKING_STATE.OK
        else:
            return _object_data.tracking_state == sl.OBJECT_TRACKING_STATE.OK or _object_data.tracking_state == sl.OBJECT_TRACKING_STATE.OFF

    def update_view(self, _image, _objs):       # _objs of type sl.Objects
        self.mutex.acquire()

        # update image
        self.image_handler.push_new_image(_image)

        # Clear frame objects
        self.BBox_edges.clear()
        self.BBox_faces.clear()
        self.objects_name = []

        for i in range(len(_objs.object_list)):
            if self.render_object(_objs.object_list[i]):
                bounding_box = np.array(_objs.object_list[i].bounding_box)
                if bounding_box.any():
                    color_class = get_color_class(0)
                    color_id = generate_color_id(_objs.object_list[i].id)
                    if _objs.object_list[i].tracking_state != sl.OBJECT_TRACKING_STATE.OK:
                        color_id = color_class
                    else:
                        pos = [_objs.object_list[i].position[0], _objs.object_list[i].bounding_box[0][1], _objs.object_list[i].position[2]]
                        self.create_id_rendering(pos, color_id, _objs.object_list[i].id)

                    self.create_bbox_rendering(bounding_box, color_id)

        self.mutex.release()

    def create_bbox_rendering(self, _bbox, _bbox_clr):
        # First create top and bottom full edges
	    self.BBox_edges.add_full_edges(_bbox, _bbox_clr)
	    # Add faded vertical edges
	    self.BBox_edges.add_vertical_edges(_bbox, _bbox_clr)
	    # Add faces
	    self.BBox_faces.add_vertical_faces(_bbox, _bbox_clr)
	    # Add top face
	    self.BBox_faces.add_top_face(_bbox, _bbox_clr)

    def create_id_rendering(self, _center, _clr, _id):
        tmp = ObjectClassName()
        tmp.name = "ID: " + str(_id)
        tmp.color = _clr
        tmp.position = np.array([_center[0], _center[1], _center[2]], np.float32)
        self.objects_name = np.append(self.objects_name, tmp)

    def idle(self):
        if self.available:
            glutPostRedisplay()

    def exit(self):
        if self.available:
            self.available = False
            self.image_handler.close()

    def close_func(self):
        if self.available:
            self.available = False
            self.image_handler.close()

    def keyPressedCallback(self, key, x, y):
        if ord(key) == 113 or ord(key) == 27:
            self.close_func()

    def draw_callback(self):
        if self.available:
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            self.mutex.acquire()
            self.update()
            self.draw()
            self.print_text()
            self.mutex.release()

            glutSwapBuffers()
            glutPostRedisplay()

    def update(self):
        self.BBox_edges.push_to_GPU()
        self.BBox_faces.push_to_GPU()

    def draw(self):
        self.image_handler.draw()

        glUseProgram(self.shader_image.get_program_id())
        glUniformMatrix4fv(self.shader_MVP, 1, GL_TRUE,  (GLfloat * len(self.projection))(*self.projection))
        self.BBox_edges.draw()
        self.BBox_faces.draw()
        glUseProgram(0)

    def print_text(self):
        glDisable(GL_BLEND)

        wnd_size = sl.Resolution()
        wnd_size.width = glutGet(GLUT_WINDOW_WIDTH)
        wnd_size.height = glutGet(GLUT_WINDOW_HEIGHT)

        if len(self.objects_name) > 0:
            for obj in self.objects_name:
                pt2d = self.compute_3D_projection(obj.position, self.projection, wnd_size)
                glColor4f(obj.color[0], obj.color[1], obj.color[2], obj.color[3])
                glWindowPos2f(pt2d[0], pt2d[1])
                for i in range(len(obj.name)):
                    glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ctypes.c_int(ord(obj.name[i])))
            glEnable(GL_BLEND)

    def compute_3D_projection(self, _pt, _cam, _wnd_size):
        pt4d = np.array([_pt[0],_pt[1],_pt[2], 1], np.float32)
        _cam_mat = np.array(_cam, np.float32).reshape(4,4)

        proj3D_cam = np.matmul(pt4d, _cam_mat)     # Should result in a 4 element row vector
        proj3D_cam[1] = proj3D_cam[1] + 0.25
        proj2D = [((proj3D_cam[0] / pt4d[3]) * _wnd_size.width) / (2. * proj3D_cam[3]) + (_wnd_size.width * 0.5)
                , ((proj3D_cam[1] / pt4d[3]) * _wnd_size.height) / (2. * proj3D_cam[3]) + (_wnd_size.height * 0.5)]
        return proj2D

########################################################################
class ObjectClassName:
    def __init__(self):
        self.position = [0,0,0] # [x,y,z]
        self.name = ""
        self.color = [0,0,0,0]  # [r,g,b,a]
