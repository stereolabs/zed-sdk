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

MESH_VERTEX_SHADER ="""
#version 330 core
layout(location = 0) in vec3 in_Vertex;
uniform mat4 u_mvpMatrix;
uniform vec3 u_color;
out vec3 b_color;
void main() {
    b_color = u_color;
    gl_Position = u_mvpMatrix * vec4(in_Vertex, 1);
}
"""

FPC_VERTEX_SHADER ="""
#version 330 core
layout(location = 0) in vec4 in_Vertex;
uniform mat4 u_mvpMatrix;
uniform vec3 u_color;
out vec3 b_color;
void main() {
    b_color = u_color;
    gl_Position = u_mvpMatrix * vec4(in_Vertex.xyz, 1);
}
"""

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
#version 330 core
in vec3 b_color;
layout(location = 0) out vec4 color;
void main() {
   color = vec4(b_color,1);
}
"""

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
        self.draw_mesh = False
        self.new_chunks = False
        self.chunks_pushed = False
        self.change_state = False
        self.sub_maps = []
        self.pose = sl.Transform().set_identity()
        self.tracking_state = sl.POSITIONAL_TRACKING_STATE.OFF
        self.mapping_state = sl.SPATIAL_MAPPING_STATE.NOT_ENABLED

    def init(self, _params, _mesh): 
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
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_SRGB)
        glutCreateWindow("ZED Spatial Mapping")
        glViewport(0, 0, width, height)

        glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE,
                      GLUT_ACTION_CONTINUE_EXECUTION)

        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

        # Initialize image renderer
        self.image_handler = ImageHandler()
        self.image_handler.initialize(_params.image_size)

        # glEnable(GL_FRAMEBUFFER_SRGB)

        # Init mesh (TODO fused point cloud)
        self.init_mesh(_mesh)

        # Compile and create the shader 
        if(self.draw_mesh):
            self.shader_image = Shader(MESH_VERTEX_SHADER, FRAGMENT_SHADER)
        # TODO else FPC
        self.shader_MVP = glGetUniformLocation(self.shader_image.get_program_id(), "u_mvpMatrix")
        self.shader_color_loc = glGetUniformLocation(self.shader_image.get_program_id(), "u_color")
        # Create the rendering camera
        self.projection = array.array('f')
        self.set_render_camera_projection(_params, 0.5, 20)

        glLineWidth(1.)
        glPointSize(4.)

        # Register the drawing function with GLUT
        glutDisplayFunc(self.draw_callback)
        # Register the function called when nothing happens
        glutIdleFunc(self.idle)   

        glutKeyboardUpFunc(self.keyReleasedCallback)

        # Register the closing function
        glutCloseFunc(self.close_func)

        self.ask_clear = False
        self.available = True

        # Set color for wireframe
        self.vertices_color = [0.35,0.36,0.95] 
        # self.vertices_color = [1.0,0.0,0.0] 
        
        # Ready to start
        self.chunks_pushed = True

    # TODO for FPC
    def init_mesh(self, _mesh):
        self.draw_mesh = True
        self.mesh = _mesh

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
    
    def print_GL(self, _x, _y, _string):
        glRasterPos(_x, _y)
        for i in range(len(_string)):
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ctypes.c_int(ord(_string[i])))


    def is_available(self):
        if self.available:
            glutMainLoopEvent()
        return self.available

    def render_object(self, _object_data):      # _object_data of type sl.ObjectData
        if _object_data.tracking_state == sl.OBJECT_TRACKING_STATE.OK or _object_data.tracking_state == sl.OBJECT_TRACKING_STATE.OFF:
            return True
        else:
            return False

    def update_chunks(self):
        self.new_chunks = True
        self.chunks_pushed = False
    
    def chunks_updated(self):
        return self.chunks_pushed

    def clear_current_mesh(self):
        self.ask_clear = True
        self.new_chunks = True

    def update_view(self, _image, _pose, _tracking_state, _mapping_state):     
        self.mutex.acquire()
        if self.available:
            # update image
            self.image_handler.push_new_image(_image)
            self.pose = _pose
            self.tracking_state = _tracking_state
            self.mapping_state = _mapping_state
        self.mutex.release()
        copy_state = self.change_state
        self.change_state = False
        return copy_state

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
        print("key released : {}\tunicode : {}".format(key, ord(key)))
        if ord(key) == 113 or ord(key) == 27:
            self.close_func()
        if  ord(key) == 32:     # space bar
            self.change_state = True

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

    def update(self):
        if self.new_chunks:
            if self.ask_clear:
                self.sub_maps = []
                self.ask_clear = False
            
            nb_c = 0
            if self.draw_mesh:
                nb_c = len(self.mesh.chunks)
            # TODO else FPC

            if nb_c > len(self.sub_maps):
                step = 500.0
                new_size = (int)(((nb_c / step) + 1) * step)
                self.sub_maps = [SubMapObj()] * new_size   

            if self.draw_mesh:
                c = 0
                for sub_map in self.sub_maps:
                    if (c < nb_c) and self.mesh.chunks[c].has_been_updated:
                        sub_map.update(self.mesh.chunks[c])
                        c = c + 1 
            # TODO else FPC

            self.new_chunks = False
            self.chunks_pushed = True

    def draw(self):  
        if self.available:
            self.image_handler.draw()

            # If the Positional tracking is good, we can draw the mesh over the current image
            if self.tracking_state == sl.POSITIONAL_TRACKING_STATE.OK and len(self.sub_maps) > 0:
                # Draw the mesh in GL_TRIANGLES with a polygon mode in line (wire)
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

                # Send the projection and the Pose to the GLSL shader to make the projection of the 2D image
                vp_matrix = sl.Transform()
                if self.pose.inverse():
                    proj_4d = np.array(self.projection, np.float32).reshape(4,4)

                    vp_matrix = np.matmul(proj_4d, self.pose.m) 

                glUseProgram(self.shader_image.get_program_id())
                vp_matrix_flat = vp_matrix.flatten()         # Turn matrix into 1D array
                glUniformMatrix4fv(self.shader_MVP, 1, GL_TRUE,  vp_matrix_flat)            
                glUniform3fv(self.shader_color_loc, 1, (GLfloat * len(self.vertices_color))(*self.vertices_color))
        
                for sub_map in self.sub_maps:
                    sub_map.draw()

                glUseProgram(0)
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

    def print_text(self):
        if self.available:
            if self.mapping_state == sl.SPATIAL_MAPPING_STATE.NOT_ENABLED:
                glColor3f(0.15, 0.15, 0.15)
                self.print_GL(-0.99, 0.9, "Hit Space Bar to activate Spatial Mapping.")
            else:
                glColor3f(0.25, 0.25, 0.25)
                self.print_GL(-0.99, 0.9, "Hit Space Bar to stop spatial mapping.")


    def compute_3D_projection(self, _pt, _cam, _wnd_size):
        pt4d = np.array([_pt[0],_pt[1],_pt[2], 1], np.float32)
        _cam_mat = np.array(_cam, np.float32).reshape(4,4)
           
        proj3D_cam = np.matmul(pt4d, _cam_mat)     # Should result in a 4 element row vector
        proj3D_cam[1] = proj3D_cam[1] + 0.25
        proj2D = [((proj3D_cam[0] / pt4d[3]) * _wnd_size.width) / (2. * proj3D_cam[3]) + (_wnd_size.width * 0.5)
                , ((proj3D_cam[1] / pt4d[3]) * _wnd_size.height) / (2. * proj3D_cam[3]) + (_wnd_size.height * 0.5)]
        return proj2D

class SubMapObj:
    def __init__(self):
        self.current_fc = 0
        self.vaoID = 0      # see if necessary ?
        self.index = array.array('I')

    def update(self, _chunk): 
        self.vboID = glGenBuffers(2)

        if len(_chunk.vertices):
            vert_float = _chunk.vertices.astype(np.float32)
            vert = vert_float.flatten()      # transform _chunk.vertices into 1D array 
            glBindBuffer(GL_ARRAY_BUFFER, self.vboID[0])
            glBufferData(GL_ARRAY_BUFFER, len(vert_float) * vert_float.itemsize * 3, (GLfloat * len(vert))(*vert), GL_DYNAMIC_DRAW)                  


        if len(_chunk.triangles):
            triangles_uint = _chunk.triangles.astype(np.uintc)      # Force triangle array data type
            triangles = triangles_uint.flatten()
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.vboID[1])
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, len(triangles_uint) * triangles_uint.itemsize * 3, (GLuint * len(triangles))(*triangles), GL_DYNAMIC_DRAW)
            
            # self.current_fc = _chunk.triangles.size * 3
            self.current_fc = len(_chunk.triangles) * 3

    
    def draw(self): 
        if self.current_fc:
            glEnableVertexAttribArray(0)
            glBindBuffer(GL_ARRAY_BUFFER, self.vboID[0])
            glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,0,None)

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.vboID[1])
            glDrawElements(GL_TRIANGLES, self.current_fc, GL_UNSIGNED_INT, None)      
            
            glDisableVertexAttribArray(0)


########################################################################
class ObjectClassName:
    def __init__(self):
        self.position = [0,0,0] # [x,y,z]
        self.name = ""
        self.color = [0,0,0,0]  # [r,g,b,a]
