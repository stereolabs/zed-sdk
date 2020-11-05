from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

import ctypes
import sys
import math
from threading import Lock
import numpy as np
import array

import ogl_viewer.zed_model as zm
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

POINTCLOUD_VERTEX_SHADER ="""
#version 330 core
layout(location = 0) in vec4 in_VertexRGBA;
uniform mat4 u_mvpMatrix;
out vec4 b_color;
void main() {
    uint vertexColor = floatBitsToUint(in_VertexRGBA.w);
    vec3 clr_int = vec3((vertexColor & uint(0x000000FF)), (vertexColor & uint(0x0000FF00)) >> 8, (vertexColor & uint(0x00FF0000)) >> 16);
    b_color = vec4(clr_int.r / 255.0f, clr_int.g / 255.0f, clr_int.b / 255.0f, 1.f);
    gl_Position = u_mvpMatrix * vec4(in_VertexRGBA.xyz, 1);
}
"""

POINTCLOUD_FRAGMENT_SHADER = """
#version 330 core
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
    def __init__(self, _is_static, pts_size = 3, clr_size = 3):
        self.is_init = False
        self.drawing_type = GL_TRIANGLES
        self.is_static = _is_static
        self.clear()
        self.pt_type = pts_size
        self.clr_type = clr_size


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

    def add_line(self, _p1, _p2, _clr):
        self.add_point_clr(_p1, _clr)
        self.add_point_clr(_p2, _clr)
            
    def addFace(self, p1, p2, p3, clr) :
        self.add_point_clr(p1, clr)
        self.add_point_clr(p2, clr)
        self.add_point_clr(p3, clr)
        
    def push_to_GPU(self):
        if(self.is_init == False):
            self.vboID = glGenBuffers(3)
            self.is_init = True

        if(self.is_static):
            type_draw = GL_STATIC_DRAW
        else:
            type_draw = GL_DYNAMIC_DRAW

        if len(self.vertices):
            glBindBuffer(GL_ARRAY_BUFFER, self.vboID[0])
            glBufferData(GL_ARRAY_BUFFER, len(self.vertices) * self.vertices.itemsize, (GLfloat * len(self.vertices))(*self.vertices), type_draw)
        
        if len(self.colors):
            glBindBuffer(GL_ARRAY_BUFFER, self.vboID[1])
            glBufferData(GL_ARRAY_BUFFER, len(self.colors) * self.colors.itemsize, (GLfloat * len(self.colors))(*self.colors), type_draw)

        if len(self.indices):
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.vboID[2])
            glBufferData(GL_ELEMENT_ARRAY_BUFFER,len(self.indices) * self.indices.itemsize,(GLuint * len(self.indices))(*self.indices), type_draw)

        self.elementbufferSize = len(self.indices)
        
    def init(self, res):
        if(self.is_init == False):
            self.vboID = glGenBuffers(3)
            self.is_init = True

        if(self.is_static):
            type_draw = GL_STATIC_DRAW
        else:
            type_draw = GL_DYNAMIC_DRAW

        self.elementbufferSize = res.width * res.height

        glBindBuffer(GL_ARRAY_BUFFER, self.vboID[0])
        glBufferData(GL_ARRAY_BUFFER, self.elementbufferSize * self.pt_type * self.vertices.itemsize, None, type_draw)
        
        if(self.clr_type):
            glBindBuffer(GL_ARRAY_BUFFER, self.vboID[1])
            glBufferData(GL_ARRAY_BUFFER, self.elementbufferSize * self.clr_type * self.colors.itemsize, None, type_draw)

        for i in range (0, self.elementbufferSize):
            self.indices.append(i+1)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.vboID[2])
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,len(self.indices) * self.indices.itemsize,(GLuint * len(self.indices))(*self.indices), type_draw)
        
    def setPoints(self, pc):
        glBindBuffer(GL_ARRAY_BUFFER, self.vboID[0])
        glBufferSubData(GL_ARRAY_BUFFER, 0, self.elementbufferSize * self.pt_type * self.vertices.itemsize, ctypes.c_void_p(pc.get_pointer()))
        glBindBuffer(GL_ARRAY_BUFFER, 0)

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
            glVertexAttribPointer(0,self.pt_type,GL_FLOAT,GL_FALSE,0,None)

            if(self.clr_type):
                glEnableVertexAttribArray(1)
                glBindBuffer(GL_ARRAY_BUFFER, self.vboID[1])
                glVertexAttribPointer(1,self.clr_type,GL_FLOAT,GL_FALSE,0,None)
            
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.vboID[2])
            glDrawElements(self.drawing_type, self.elementbufferSize, GL_UNSIGNED_INT, None)      
            
            glDisableVertexAttribArray(0)
            glDisableVertexAttribArray(1)
    
class GLViewer:
    def __init__(self):
        self.available = False
        self.mutex = Lock()
        self.camera = CameraGL()
        self.wheelPosition = 0.
        self.mouse_button = [False, False]
        self.mouseCurrentPosition = [0., 0.]
        self.previousMouseMotion = [0., 0.]
        self.mouseMotion = [0., 0.]
        self.zedModel = Simple3DObject(True)
        self.point_cloud = Simple3DObject(False, 4)

    def init(self, _argc, _argv, camera_model, res): # _params = sl.CameraParameters
        glutInit(_argc, _argv)
        wnd_w = int(glutGet(GLUT_SCREEN_WIDTH)*0.9)
        wnd_h = int(glutGet(GLUT_SCREEN_HEIGHT) *0.9)
        glutInitWindowSize(wnd_w, wnd_h)
        glutInitWindowPosition(int(wnd_w*0.05), int(wnd_h*0.05))

        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH)
        glutCreateWindow("ZED Depth Sensing")
        glViewport(0, 0, wnd_w, wnd_h)

        glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE,
                      GLUT_ACTION_CONTINUE_EXECUTION)

        glEnable(GL_DEPTH_TEST)
        
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

        # Compile and create the shader for 3D objects
        self.shader_image = Shader(VERTEX_SHADER, FRAGMENT_SHADER)
        self.shader_image_MVP = glGetUniformLocation(self.shader_image.get_program_id(), "u_mvpMatrix")
        
        self.shader_pc = Shader(POINTCLOUD_VERTEX_SHADER, POINTCLOUD_FRAGMENT_SHADER)
        self.shader_pc_MVP = glGetUniformLocation(self.shader_pc.get_program_id(), "u_mvpMatrix")
        
        self.bckgrnd_clr = np.array([223/255., 230/255., 233/255.])

        if(camera_model == sl.MODEL.ZED):
            for i in range(0, zm.NB_ALLUMINIUM_TRIANGLES * 3, 3):
                for j in range(3):
                    index = int(zm.alluminium_triangles[i + j] - 1)
                    self.zedModel.add_point_clr([zm.vertices[index * 3], zm.vertices[index * 3 + 1], zm.vertices[index * 3 + 2]], [zm.ALLUMINIUM_COLOR.r, zm.ALLUMINIUM_COLOR.g, zm.ALLUMINIUM_COLOR.b] )

            for i in range(0, zm.NB_DARK_TRIANGLES * 3, 3):
                for j in range(3):
                    index = int(zm.dark_triangles[i + j] - 1)
                    self.zedModel.add_point_clr([zm.vertices[index * 3], zm.vertices[index * 3 + 1], zm.vertices[index * 3 + 2]], [zm.DARK_COLOR.r, zm.DARK_COLOR.g, zm.DARK_COLOR.b] )
        elif(camera_model == sl.MODEL.ZED_M):
            for i in range(0, zm.NB_AL_ZEDM_TRI * 3, 3):
                for j in range(3):
                    index = int(zm.al_triangles_m[i + j] - 1)
                    self.zedModel.add_point_clr([zm.vertices_m[index * 3], zm.vertices_m[index * 3 + 1], zm.vertices_m[index * 3 + 2]], [zm.ALLUMINIUM_COLOR.r, zm.ALLUMINIUM_COLOR.g, zm.ALLUMINIUM_COLOR.b] )

            for i in range(0, zm.NB_DARK_ZEDM_TRI * 3, 3):
                for j in range(3):
                    index = int(zm.dark_triangles_m[i + j] - 1)
                    self.zedModel.add_point_clr([zm.vertices_m[index * 3], zm.vertices_m[index * 3 + 1], zm.vertices_m[index * 3 + 2]], [zm.DARK_COLOR.r, zm.DARK_COLOR.g, zm.DARK_COLOR.b] )

            for i in range(0, zm.NB_GRAY_ZEDM_TRI * 3, 3):
                for j in range(3):
                    index = int(zm.gray_triangles_m[i + j] - 1)
                    self.zedModel.add_point_clr([zm.vertices_m[index * 3], zm.vertices_m[index * 3 + 1], zm.vertices_m[index * 3 + 2]], [zm.GRAY_COLOR.r, zm.GRAY_COLOR.g, zm.GRAY_COLOR.b] )

            for i in range(0, zm.NB_YELLOW_ZEDM_TRI * 3, 3):
                for j in range(3):
                    index = int(zm.yellow_triangles_m[i + j] - 1)
                    self.zedModel.add_point_clr([zm.vertices_m[index * 3], zm.vertices_m[index * 3 + 1], zm.vertices_m[index * 3 + 2]], [zm.YELLOW_COLOR.r, zm.YELLOW_COLOR.g, zm.YELLOW_COLOR.b] )

        elif(camera_model == sl.MODEL.ZED2):
            for i in range(0, zm.NB_ALLUMINIUM_TRIANGLES * 3, 3):
                for j in range(3):
                    index = int(zm.alluminium_triangles[i + j] - 1)
                    self.zedModel.add_point_clr([zm.vertices[index * 3], zm.vertices[index * 3 + 1], zm.vertices[index * 3 + 2]], [zm.DARK_COLOR.r, zm.DARK_COLOR.g, zm.DARK_COLOR.b] )

            for i in range(0, zm.NB_DARK_TRIANGLES * 3, 3):
                for j in range(3):
                    index = int(zm.dark_triangles[i + j] - 1)
                    self.zedModel.add_point_clr([zm.vertices[index * 3], zm.vertices[index * 3 + 1], zm.vertices[index * 3 + 2]], [zm.GRAY_COLOR.r, zm.GRAY_COLOR.g, zm.GRAY_COLOR.b] )
        self.zedModel.set_drawing_type(GL_TRIANGLES)
        self.zedModel.push_to_GPU()

        self.point_cloud.init(res)
        self.point_cloud.set_drawing_type(GL_POINTS)
        
        # Register GLUT callback functions
        glutDisplayFunc(self.draw_callback)
        glutIdleFunc(self.idle)
        glutKeyboardFunc(self.keyPressedCallback)
        glutCloseFunc(self.close_func)
        glutMouseFunc(self.on_mouse)
        glutMotionFunc(self.on_mousemove)
        glutReshapeFunc(self.on_resize)  
        
        self.available = True

    def is_available(self):
        if self.available:
            glutMainLoopEvent()
        return self.available

    def updateData(self, pc):
        self.mutex.acquire()
        self.point_cloud.setPoints(pc)
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

    def on_mouse(self,*args,**kwargs):
        (key,Up,x,y) = args
        if key==0:
            self.mouse_button[0] = (Up == 0)
        elif key==2 :
            self.mouse_button[1] = (Up == 0)  
        elif(key == 3):
            self.wheelPosition = self.wheelPosition + 1
        elif(key == 4):
            self.wheelPosition = self.wheelPosition - 1
        
        self.mouseCurrentPosition = [x, y]
        self.previousMouseMotion = [x, y]
        
    def on_mousemove(self,*args,**kwargs):
        (x,y) = args
        self.mouseMotion[0] = x - self.previousMouseMotion[0]
        self.mouseMotion[1] = y - self.previousMouseMotion[1]
        self.previousMouseMotion = [x, y]
        glutPostRedisplay()

    def on_resize(self,Width,Height):
        glViewport(0, 0, Width, Height)
        self.camera.setProjection(Height / Width)

    def draw_callback(self):
        if self.available:
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glClearColor(self.bckgrnd_clr[0], self.bckgrnd_clr[1], self.bckgrnd_clr[2], 1.)

            self.mutex.acquire()
            self.update()
            self.draw()
            self.mutex.release()  

            glutSwapBuffers()
            glutPostRedisplay()

    def update(self):
        if(self.mouse_button[0]):
            r = sl.Rotation()
            vert=self.camera.vertical_
            tmp = vert.get()
            vert.init_vector(tmp[0] * 1.,tmp[1] * 1., tmp[2] * 1.)
            r.init_angle_translation(self.mouseMotion[0] * 0.002, vert)
            self.camera.rotate(r)

            r.init_angle_translation(self.mouseMotion[1] * 0.002, self.camera.right_)
            self.camera.rotate(r)
        
        if(self.mouse_button[1]):
            t = sl.Translation()
            tmp = self.camera.right_.get()
            scale = self.mouseMotion[0] *-0.01
            t.init_vector(tmp[0] * scale, tmp[1] * scale, tmp[2] * scale)
            self.camera.translate(t)
            
            tmp = self.camera.up_.get()
            scale = self.mouseMotion[1] * 0.01
            t.init_vector(tmp[0] * scale, tmp[1] * scale, tmp[2] * scale)
            self.camera.translate(t)

        if (self.wheelPosition != 0):          
            t = sl.Translation()  
            tmp = self.camera.forward_.get()
            scale = self.wheelPosition * -0.065
            t.init_vector(tmp[0] * scale, tmp[1] * scale, tmp[2] * scale)
            self.camera.translate(t)

        self.camera.update()

        self.mouseMotion = [0., 0.]
        self.wheelPosition = 0

    def draw(self):
        vpMatrix = self.camera.getViewProjectionMatrix()
        glUseProgram(self.shader_image.get_program_id())
        glUniformMatrix4fv(self.shader_image_MVP, 1, GL_TRUE,  (GLfloat * len(vpMatrix))(*vpMatrix))    
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        self.zedModel.draw() 
        glUseProgram(0)
        
        glUseProgram(self.shader_pc.get_program_id())
        glUniformMatrix4fv(self.shader_pc_MVP, 1, GL_TRUE,  (GLfloat * len(vpMatrix))(*vpMatrix))
        glPointSize(1.)
        self.point_cloud.draw()
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
        self.horizontalFOV = 70.
        self.orientation_ = sl.Orientation()
        self.position_ = sl.Translation()
        self.forward_ = sl.Translation()
        self.up_ = sl.Translation()
        self.right_ = sl.Translation()
        self.vertical_ = sl.Translation()
        self.vpMatrix_ = sl.Matrix4f()
        self.offset_ = sl.Translation()
        self.offset_.init_vector(0,0,5)
        self.projection_ = sl.Matrix4f()
        self.projection_.set_identity()
        self.setProjection(1.78)

        self.position_.init_vector(0., 0., 0.)
        tmp = sl.Translation()
        tmp.init_vector(0, 0, -.1)
        tmp2 = sl.Translation()
        tmp2.init_vector(0, 1, 0)
        self.setDirection(tmp, tmp2)        

    def update(self): 
        dot_ = sl.Translation.dot_translation(self.vertical_, self.up_)
        if(dot_ < 0.):
            tmp = self.vertical_.get()
            self.vertical_.init_vector(tmp[0] * -1.,tmp[1] * -1., tmp[2] * -1.)
        transformation = sl.Transform()

        tmp_position = self.position_.get()
        tmp = (self.offset_ * self.orientation_).get()
        new_position = sl.Translation()
        new_position.init_vector(tmp_position[0] + tmp[0], tmp_position[1] + tmp[1], tmp_position[2] + tmp[2])
        transformation.init_orientation_translation(self.orientation_, new_position)
        transformation.inverse()
        self.vpMatrix_ = self.projection_ * transformation
        
    def setProjection(self, im_ratio):
        fov_x = self.horizontalFOV * 3.1416 / 180.
        fov_y = self.horizontalFOV * im_ratio * 3.1416 / 180.

        self.projection_[(0,0)] = 1. / math.tan(fov_x * .5)
        self.projection_[(1,1)] = 1. / math.tan(fov_y * .5)
        self.projection_[(2,2)] = -(self.zfar + self.znear) / (self.zfar - self.znear)
        self.projection_[(3,2)] = -1.
        self.projection_[(2,3)] = -(2. * self.zfar * self.znear) / (self.zfar - self.znear)
        self.projection_[(3,3)] = 0.
    
    def getViewProjectionMatrix(self):
        tmp = self.vpMatrix_.m
        vpMat = array.array('f')
        for row in tmp:
            for v in row:
                vpMat.append(v)
        return vpMat
        
    def getViewProjectionMatrixRT(self, tr):
        tmp = self.vpMatrix_
        tmp.transpose()
        tr.transpose()
        tmp =  (tr * tmp).m
        vpMat = array.array('f')
        for row in tmp:
            for v in row:
                vpMat.append(v)
        return vpMat

    def setDirection(self, dir, vert):
        dir.normalize()
        tmp = dir.get()
        dir.init_vector(tmp[0] * -1.,tmp[1] * -1., tmp[2] * -1.)
        self.orientation_.init_translation(self.ORIGINAL_FORWARD, dir)
        self.updateVectors()
        self.vertical_ = vert
        if(sl.Translation.dot_translation(self.vertical_, self.up_) < 0.):
            tmp = sl.Rotation()
            tmp.init_angle_translation(3.14, self.ORIGINAL_FORWARD)
            self.rotate(tmp)
    
    def translate(self, t):
        ref = self.position_.get()
        tmp = t.get()
        self.position_.init_vector(ref[0] + tmp[0], ref[1] + tmp[1], ref[2] + tmp[2])

    def setPosition(self, p):
        self.position_ = p

    def rotate(self, r): 
        tmp = sl.Orientation()
        tmp.init_rotation(r)
        self.orientation_ = tmp * self.orientation_
        self.updateVectors()

    def setRotation(self, r):
        self.orientation_.init_rotation(r)
        self.updateVectors()

    def updateVectors(self):
        self.forward_ = self.ORIGINAL_FORWARD * self.orientation_
        self.up_ = self.ORIGINAL_UP * self.orientation_
        right = self.ORIGINAL_RIGHT
        tmp = right.get()
        right.init_vector(tmp[0] * -1.,tmp[1] * -1., tmp[2] * -1.)
        self.right_ = right * self.orientation_
