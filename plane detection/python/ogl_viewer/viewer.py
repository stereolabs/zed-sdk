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

M_PI = 3.1415926

MESH_VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 in_Vertex;
layout(location = 1) in float in_dist;
uniform mat4 u_mvpMatrix;
uniform vec3 u_color;
out vec3 b_color;
out float distance;
void main() {
    b_color = u_color;
    distance = in_dist;
    gl_Position = u_mvpMatrix * vec4(in_Vertex, 1);
}
"""

MESH_FRAGMENT_SHADER = """
#version 330 core
in vec3 b_color;
in float distance;
layout(location = 0) out vec4 color;
void main() {
   color = vec4(b_color,distance);
};
"""

IMAGE_FRAGMENT_SHADER = """
#version 330 core
in vec2 UV;
out vec4 color;
uniform sampler2D texImage;
uniform bool revert;
uniform bool rgbflip;
void main() {
    vec2 scaler  =revert?vec2(UV.x,1.f - UV.y):vec2(UV.x,UV.y);
    vec3 rgbcolor = rgbflip?vec3(texture(texImage, scaler).zyx):vec3(texture(texImage, scaler).xyz);
    color = vec4(rgbcolor,1);
}
"""

IMAGE_VERTEX_SHADER = """
#version 330
layout(location = 0) in vec3 vert;
out vec2 UV;
void main() {
    UV = (vert.xy+vec2(1,1))/2;
    gl_Position = vec4(vert, 1);
}
"""

def get_plane_color(_type):
    """
    Get plane color according to its type
    """
    plane_clr_map = {
        sl.PLANE_TYPE.HORIZONTAL : [0.65, 0.95, 0.35],
        sl.PLANE_TYPE.VERTICAL : [0.95, 0.35, 0.65],
        sl.PLANE_TYPE.UNKNOWN : [0.35, 0.65, 0.95],
        sl.PLANE_TYPE.LAST : [0.,0.,0.]
        }
    return plane_clr_map.get(_type, "Unknown type")

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

class MeshObject:
    def __init__(self):
        self.current_fc = 0
        self.need_update = False
        self.vert = []
        self.edge_dist = array.array('f')
        self.tri = []
        self.type = sl.PLANE_TYPE.UNKNOWN

    def alloc(self):
        self.vboID = glGenBuffers(3)
        self.shader_image = Shader(MESH_VERTEX_SHADER, MESH_FRAGMENT_SHADER)
        self.shader_MVP = glGetUniformLocation(self.shader_image.get_program_id(), "u_mvpMatrix")
        self.shader_color_loc = glGetUniformLocation(self.shader_image.get_program_id(), "u_color")

    def update_mesh(self, _vertices, _triangles, _border):
        if not self.need_update:
            self.vert = _vertices.flatten().astype(np.float32)
            self.tri = _triangles.flatten().astype(np.uintc)

            # Resize edge_dist vector
            self.edge_dist = array.array('f')
            for i in range(len(_vertices)):
                self.edge_dist.append(0.)
                                     
            for i in range(len(self.edge_dist)):
                d_min = sys.float_info.max
                v_current = _vertices[i]
                for j in range(len(_border)):
                    dist_current = np.linalg.norm(v_current - _vertices[_border[j]]) 
                    if dist_current < d_min:
                        d_min = dist_current
                if d_min >= 0.002:
                    self.edge_dist[i] = 0.4
                else:
                    self.edge_dist[i] = 0.0
            self.need_update = True

    def push_to_GPU(self):
        if self.need_update:
            if len(self.vert):
                glBindBuffer(GL_ARRAY_BUFFER, self.vboID[0])
                glBufferData(GL_ARRAY_BUFFER, len(self.vert) * self.vert.itemsize , (GLfloat * len(self.vert))(*self.vert), GL_DYNAMIC_DRAW)

            if len(self.edge_dist):
                glBindBuffer(GL_ARRAY_BUFFER, self.vboID[1])
                glBufferData(GL_ARRAY_BUFFER, len(self.edge_dist) * self.edge_dist.itemsize , (GLfloat * len(self.edge_dist))(*self.edge_dist), GL_DYNAMIC_DRAW)

            if len(self.tri):
                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.vboID[2])
                glBufferData(GL_ELEMENT_ARRAY_BUFFER, len(self.tri) * self.tri.itemsize , (GLuint * len(self.tri))(*self.tri), GL_DYNAMIC_DRAW)
                self.current_fc = self.tri.size  

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            self.need_update = False

    def draw(self):
        if self.current_fc:
            glEnableVertexAttribArray(0)
            glBindBuffer(GL_ARRAY_BUFFER, self.vboID[0])
            glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,0,None)

            glEnableVertexAttribArray(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.vboID[1])
            glVertexAttribPointer(1,1,GL_FLOAT,GL_FALSE,0,None)

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.vboID[2])
            glDrawElements(GL_TRIANGLES, self.current_fc, GL_UNSIGNED_INT, None)      
            
            glDisableVertexAttribArray(0)
            glDisableVertexAttribArray(1)

class UserAction:
    def __init__(self):
        self.press_space = False
        self.hit = False
        self.hit_coord = []

    def clear(self):
        self.press_space = False
        self.hit = False

class ImageHandler:
    """
    Class that manages the image stream to render with OpenGL
    """
    def __init__(self):
        self.tex_id = 0
        self.image_tex = 0
        self.quad_vb = 0

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
        self.mutex = Lock()
        self.projection = sl.Matrix4f()
        self.projection.set_identity()
        self.znear = 0.1        
        self.zfar = 100.        
        self.image_handler = ImageHandler()
        self.pose = sl.Transform().set_identity()
        self.tracking_state = sl.POSITIONAL_TRACKING_STATE.OFF

        self.mesh_object = MeshObject()
        self.user_action = UserAction()
        self.new_data = False
        self.wnd_w = 0
        self.wnd_h = 0

    def init(self, _params, _has_imu): 
        glutInit()
        wnd_w = glutGet(GLUT_SCREEN_WIDTH)
        wnd_h = glutGet(GLUT_SCREEN_HEIGHT)
        width = wnd_w*0.9
        height = wnd_h*0.9
     
        if width > _params.image_size.width and height > _params.image_size.height:
            width = _params.image_size.width
            height = _params.image_size.height

        glutInitWindowSize(width, height)
        glutInitWindowPosition(0, 0) # The window opens at the upper left corner of the screen
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA)
        glutCreateWindow("ZED Plane Detection")
        
        self.reshape_callback(width, height)

        glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE,
                      GLUT_ACTION_CONTINUE_EXECUTION)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

        # Initialize image renderer
        self.image_handler.initialize(_params.image_size)

        # Create the rendering camera
        self.set_render_camera_projection(_params)

        self.mesh_object.alloc()

        # Register the drawing function with GLUT
        glutDisplayFunc(self.draw_callback)
        # Register the function called when nothing happens
        glutIdleFunc(self.idle)   
        # Register the function called when a key is pressed
        glutKeyboardUpFunc(self.keyReleasedCallback)
        # Register the function on window resizing
        glutReshapeFunc(self.reshape_callback)
        # Register the mouse callback function
        glutMouseFunc(self.mouse_button_callback)
        # Register the closing function
        glutCloseFunc(self.close_func)

        self.use_imu = _has_imu

        self.user_action.hit_coord = [0.5,0.5]

        self.available = True

    def set_render_camera_projection(self, _params):
        # Just slightly move up the ZED camera FOV to make a small black border
        fov_y = (_params.v_fov + 0.5) * M_PI / 180
        fov_x = (_params.h_fov + 0.5) * M_PI / 180

        self.projection[(0,0)] = 1. / math.tan(fov_x * .5)
        self.projection[(1,1)] = 1. / math.tan(fov_y * .5)
        self.projection[(2,2)] = -(self.zfar + self.znear) / (self.zfar - self.znear)
        self.projection[(3,2)] = -1.
        self.projection[(2,3)] = -(2. * self.zfar * self.znear) / (self.zfar - self.znear)
        self.projection[(3,3)] = 0.
    
    def print_GL(self, _x, _y, _string):
        glRasterPos(_x, _y)
        for i in range(len(_string)):
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ctypes.c_int(ord(_string[i])))

    def is_available(self):
        if self.available:
            glutMainLoopEvent()
        return self.available
    
    def update_mesh(self, _mesh, _type):
        self.mutex.acquire()
        edges = _mesh.get_boundaries()
        self.mesh_object.update_mesh(_mesh.vertices, _mesh.triangles, edges)
        self.mesh_object.type = _type
        self.mutex.release()

    def update_view(self, _image, _pose, _tracking_state):     
        self.mutex.acquire()
        if self.available:
            # update image
            self.image_handler.push_new_image(_image)
            self.pose = _pose
            self.tracking_state = _tracking_state
        self.new_data = True
        self.mutex.release()

        copy = self.user_action
        self.user_action.clear()
        return copy

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

    def keyReleasedCallback(self, key, x, y):
        if ord(key) == 113 or ord(key) == 27:   # Esc or 'q' key
            self.close_func()
        if ord(key) == 32:      # space bar
            self.user_action.press_space = True
        if ord(key) == 112:     # 'p' key
            self.user_action.hit = True

    def draw_callback(self):
        if self.available:
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glClearColor(0, 0, 0, 1.0)

            self.mutex.acquire()
            self.update()
            self.draw()
            self.print_text()
            self.mutex.release()  

            glutSwapBuffers()
            glutPostRedisplay()

    def reshape_callback(self, _width, _height):
        glViewport(0, 0, _width, _height)
        self.wnd_w = _width
        self.wnd_h = _height

    def mouse_button_callback(self, _button, _state, _x, _y):
        # Action right, left, middle click
        if _button < 3:
            if _state == GLUT_DOWN:
                self.user_action.hit = True
                self.user_action.hit_coord[0] = (_x / (1. * self.wnd_w))
                self.user_action.hit_coord[1] = (_y / (1. * self.wnd_h))

    def update(self):
        # Update GPU data
        if self.new_data:
            self.mesh_object.push_to_GPU()
            self.new_data = False

    def draw(self):  
        if self.available:
            self.image_handler.draw()

            # If the Positional tracking is good, we can draw the mesh over the current image
            if self.tracking_state == sl.POSITIONAL_TRACKING_STATE.OK:
                glDisable(GL_TEXTURE_2D)
                # Send the projection and the Pose to the GLSL shader to make the projection of the 2D image
                tmp = self.pose
                tmp.inverse()
                proj = (self.projection * tmp).m
                vpMat = proj.flatten()
                
                glUseProgram(self.mesh_object.shader_image.get_program_id())
                glUniformMatrix4fv(self.mesh_object.shader_MVP, 1, GL_TRUE, (GLfloat * len(vpMat))(*vpMat))

                # Get plane color according to its type
                clr_plane = get_plane_color(self.mesh_object.type)
                glLineWidth(0.5)
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
                glUniform3fv(self.mesh_object.shader_color_loc, 1, (GLfloat * len(clr_plane))(*clr_plane))
                self.mesh_object.draw()

                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
                glUniform3fv(self.mesh_object.shader_color_loc, 1, (GLfloat * len(clr_plane))(*clr_plane))
                self.mesh_object.draw()
                glUseProgram(0)
            
            # Draw hit
            cx = self.user_action.hit_coord[0] * 2.0 - 1.0
            cy = (self.user_action.hit_coord[1] * 2.0 - 1.0) * -1.0

            lx = 0.02
            ly = lx * (self.wnd_w/(1.0*self.wnd_h))

            glLineWidth(2)
            glColor3f(0.2, 0.45, 0.9)
            glBegin(GL_LINES)
            glVertex3f(cx - lx, cy, 0.0)
            glVertex3f(cx + lx, cy, 0.0)
            glVertex3f(cx, cy - ly, 0.0)
            glVertex3f(cx, cy + ly, 0.0)
            glEnd()        

    def print_text(self):
        if self.available:
            if self.tracking_state != sl.POSITIONAL_TRACKING_STATE.OK:
                glColor4f(0.85, 0.25, 0.15, 1.)
                state_str = ""
                positional_tracking_state_str = "POSITIONAL TRACKING STATE : "
                state_str = positional_tracking_state_str + str(self.tracking_state)
                self.print_GL(-0.99, 0.9, str(state_str))
            else: 
                glColor4f(0.82, 0.82, 0.82, 1.0)
                self.print_GL(-0.99, 0.9, "Press Space Bar to detect floor PLANE.")
                self.print_GL(-0.99, 0.85, "Press 'p' key to get plane at hit.")

            if self.use_imu:
                y_start = -0.99
                for plane_type in sl.PLANE_TYPE:
                    if plane_type != sl.PLANE_TYPE.LAST:
                        clr = get_plane_color(plane_type)
                        glColor4f(clr[0], clr[1], clr[2], 1.0)
                        self.print_GL(-0.99, y_start, str(plane_type))
                        y_start = y_start + 0.05
                glColor4f(0.22, 0.22, 0.22, 1.)
                self.print_GL(-0.99, y_start, "PLANES ORIENTATION :")
