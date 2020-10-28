from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

import ctypes
import sys
import math
from threading import Lock
import numpy as np
import array

import positional_tracking.utils as ut
import positional_tracking.zed_model as zm
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
    def __init__(self, _is_static):
        self.vaoID = 0
        self.drawing_type = GL_TRIANGLES
        self.is_static = _is_static
        self.elementbufferSize = 0

        self.vertices = array.array('f')
        self.colors = array.array('f')
        self.indices = array.array('I')

    def add_pt(self, _pts):  # _pts [x,y,z]
        for pt in _pts:
            self.vertices.append(pt)

    def add_clr(self, _clrs):    # _clr [r,g,b]
        for clr in _clrs:
            self.colors.append(clr)

    def add_point_clr(self, _pt, _clr):
        self.add_pt(_pt)
        self.add_clr(_clr)
        self.indices.append(len(self.indices))
            
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

        self.elementbufferSize = len(self.indices)

    def clear(self):        
        self.vertices = array.array('f')
        self.colors = array.array('f')
        self.indices = array.array('I')
        self.elementbufferSize = 0

    def set_drawing_type(self, _type):
        self.drawing_type = _type

    def draw(self):
        if (self.elementbufferSize):            
            glEnableVertexAttribArray(0)
            glBindBuffer(GL_ARRAY_BUFFER, self.vboID[0])
            glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,0,None)

            glEnableVertexAttribArray(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.vboID[1])
            glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,0,None)
            
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.vboID[2])
            glDrawElements(self.drawing_type, self.elementbufferSize, GL_UNSIGNED_INT, None)      
            
            glDisableVertexAttribArray(0)
            glDisableVertexAttribArray(1)

def addVert(obj, i_f, limit, clr) :
    obj.addLine([i_f, 0, -limit], [i_f, 0, limit], clr)
    obj.addLine([-limit, 0, i_f],[limit, 0, i_f], clr)

class GLViewer:
    def __init__(self):
        self.available = False
        # TODO : might not be necessary to init it here
        self.objects_name = []
        self.mutex = Lock()

    def init(self, _argc, _argv, camera_model): # _params = sl.CameraParameters
        glutInit(_argc, _argv)
        wnd_w = glutGet(GLUT_SCREEN_WIDTH)
        wnd_h = glutGet(GLUT_SCREEN_HEIGHT) *0.9
        glutInitWindowSize(wnd_w*0.9, wnd_h*0.9)
        glutInitWindowPosition(wnd_w*0.05, wnd_h*0.05)

        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH)
        glutCreateWindow("ZED Positional Tracking")
        glViewport(0, 0, wnd_w*0.9, wnd_h*0.9)

        glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE,
                      GLUT_ACTION_CONTINUE_EXECUTION)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

        # Compile and create the shader for 3D objects
        self.shader_image = Shader(VERTEX_SHADER, FRAGMENT_SHADER)
        self.shader_MVP = glGetUniformLocation(self.shader_image.get_program_id(), "u_mvpMatrix")
        
        self.bckgrnd_clr = np.array([223/255., 230/255., 233/255.])

        # Create the bounding box object
        self.floor_grid = Simple3DObject(False)
        self.floor_grid.set_drawing_type(GL_LINES)
        
        limit = 20.
        clr1 = np.array([218/255., 223/255., 225/255.])        
        clr2 = np.array([108/255., 122/255., 137/255.])
        
        for i in range (limit * -5, limit * 5):
            i_f = i / 5.
            if((i % 5) == 0):
                addVert(self.floor_grid, i_f, limit, clr2)
            else:
                addVert(self.floor_grid, i_f, limit, clr1)
        self.floor_grid.push_to_GPU()

        self.zedPath = Simple3DObject(False)
        self.zedPath.set_drawing_type(GL_LINE_STRIP)

        self.zedModel = Simple3DObject(False)

        # Register the drawing function with GLUT
        glutDisplayFunc(self.draw_callback)
        # Register the function called when nothing happens
        glutIdleFunc(self.idle)   

        glutKeyboardFunc(self.keyPressedCallback)
        # Register the closing function
        glutCloseFunc(self.close_func)
        self.available = True

    def is_available(self):
        if self.available:
            glutMainLoopEvent()
        return self.available

    def updateData(self, zed_rt, str_t, str_r, state):
        self.mutex.acquire()
        self.zedPath.add_point_clr(zed_rt.get_translation(), [1,0,0.5])
        self.mutex.release()

    def idle(self):
        if self.available:
            glutPostRedisplay()

    def exit(self):
        if self.available:
            self.available = False

    def close_func(self):
        if self.available:
            self.available = False

    def keyPressedCallback(self, key, x, y):
        if ord(key) == 27:
            self.close_func()        

    def draw_callback(self):
        if self.available:
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glClearColor(self.bckgrnd_clr[0], self.bckgrnd_clr[1], self.bckgrnd_clr[2], 1.)

            self.mutex.acquire()
            self.update()
            self.draw()
            #self.print_text()
            self.mutex.release()  

            glutSwapBuffers()
            glutPostRedisplay()

    def update(self):
        self.zedPath.push_to_GPU()

    def draw(self): 
        glPointSize(1.)
        glUseProgram(self.shader_image.get_program_id())
        glUniformMatrix4fv(self.shader_MVP, 1, GL_TRUE,  (GLfloat * len(self.projection))(*self.projection))    

        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glLineWidth(2)
        self.zedPath.draw()
        glUseProgram(0)


class CameraGL:
    def __init__(self):
        self.ORIGINAL_FORWARD = sl.Translation()
        self.ORIGINAL_FORWARD.init_vector(0,0,1)
        self.ORIGINAL_UP = sl.Translation()
        self.ORIGINAL_UP.init_vector(0,1,0)
        self.ORIGINAL_RIGHT = sl.Translation()
        self.ORIGINAL_RIGHT.init_vector(1,0,0)
        self.znear = 0.5
        self.zfar = 100.
        self.rotation = sl.Orientation()
        self.position_ = sl.Translation()
        self.forward_ = sl.Translation()
        self.up_ = sl.Translation()
        self.right_ = sl.Translation()
        self.vertical_ = sl.Translation()

    def setPosition(self, p):
        self.position_ = p

    def setDirection(self, d, v):
        dirNormalized = d
        dirNormalized.normalize()
        self.rotation = sl.Orientation(self.ORIGINAL_FORWARD, dirNormalized*-1.)
        updateVectors()
        self.vertical_ = v
        


    def update(self):
        vertical_.do