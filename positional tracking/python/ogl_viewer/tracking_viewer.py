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

def safe_glutBitmapString(font, str_):
    for i in range(len(str_)):
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(str_[i]))

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

    def add_line(self, _p1, _p2, _clr):
        self.add_point_clr(_p1, _clr)
        self.add_point_clr(_p2, _clr)
            
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
    obj.add_line([i_f, 0, -limit], [i_f, 0, limit], clr)
    obj.add_line([-limit, 0, i_f],[limit, 0, i_f], clr)

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
        self.pose = sl.Transform()
        self.trackState = sl.POSITIONAL_TRACKING_STATE
        self.txtT = ""
        self.txtR = ""

    def init(self, _argc, _argv, camera_model): # _params = sl.CameraParameters
        glutInit(_argc, _argv)
        wnd_w = int(glutGet(GLUT_SCREEN_WIDTH)*0.9)
        wnd_h = int(glutGet(GLUT_SCREEN_HEIGHT) *0.9)
        glutInitWindowSize(wnd_w, wnd_h)
        glutInitWindowPosition(int(wnd_w*0.05), int(wnd_h*0.05))

        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH)
        glutCreateWindow("ZED Positional Tracking")
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
        self.shader_MVP = glGetUniformLocation(self.shader_image.get_program_id(), "u_mvpMatrix")
        
        self.bckgrnd_clr = np.array([223/255., 230/255., 233/255.])

        # Create the bounding box object
        self.floor_grid = Simple3DObject(False)
        self.floor_grid.set_drawing_type(GL_LINES)
        
        limit = 20
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

    def updateData(self, zed_rt, str_t, str_r, state):
        self.mutex.acquire()
        self.pose = zed_rt
        self.zedPath.add_point_clr(zed_rt.get_translation().get(), [0.1,0.36,0.84])
        self.trackState = state
        self.txtT = str_t
        self.txtR = str_r
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
            self.print_text()
            self.mutex.release()  

            glutSwapBuffers()
            glutPostRedisplay()

    def update(self):
        self.zedPath.push_to_GPU()

        if(self.mouse_button[0]):
            r = sl.Rotation()
            vert=self.camera.vertical_
            tmp = vert.get()
            vert.init_vector(tmp[0] * -1.,tmp[1] * -1., tmp[2] * -1.)
            r.init_angle_translation(self.mouseMotion[0] * 0.002, vert)
            self.camera.rotate(r)

            r.init_angle_translation(self.mouseMotion[1] * 0.002, self.camera.right_)
            self.camera.rotate(r)
        
        if(self.mouse_button[1]):
            t = sl.Translation()
            tmp = self.camera.right_.get()
            scale = self.mouseMotion[0] * -0.01
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
        glPointSize(1.)
        glUseProgram(self.shader_image.get_program_id())

        vpMatrix = self.camera.getViewProjectionMatrix()
        glUniformMatrix4fv(self.shader_MVP, 1, GL_TRUE,  (GLfloat * len(vpMatrix))(*vpMatrix))    

        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glLineWidth(2)
        self.zedPath.draw()
        self.floor_grid.draw()
        
        vpMatrix = self.camera.getViewProjectionMatrixRT(self.pose)
        glUniformMatrix4fv(self.shader_MVP, 1, GL_FALSE,  (GLfloat * len(vpMatrix))(*vpMatrix))

        self.zedModel.draw()
        glUseProgram(0)
        
    def print_text(self):
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        w_wnd = glutGet(GLUT_WINDOW_WIDTH)
        h_wnd = glutGet(GLUT_WINDOW_HEIGHT)
        glOrtho(0, w_wnd, 0, h_wnd, -1., 1.)

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        start_w = 20
        start_h = h_wnd - 40

        if(self.trackState == sl.POSITIONAL_TRACKING_STATE.OK):
            glColor3f(0.2, 0.65, 0.2)
        else:
            glColor3f(0.85, 0.2, 0.2)

        glRasterPos2i(start_w, start_h)

        safe_glutBitmapString(GLUT_BITMAP_HELVETICA_18,  "POSITIONAL TRACKING : " + str(self.trackState))

        dark_clr = 0.12
        glColor3f(dark_clr, dark_clr, dark_clr)
        glRasterPos2i(start_w, start_h - 25)
        safe_glutBitmapString(GLUT_BITMAP_HELVETICA_18, "Translation (m) :")

        glColor3f(0.4980, 0.5490, 0.5529)
        glRasterPos2i(155, start_h - 25)

        safe_glutBitmapString(GLUT_BITMAP_HELVETICA_18, self.txtT)

        glColor3f(dark_clr, dark_clr, dark_clr)
        glRasterPos2i(start_w, start_h - 50)
        safe_glutBitmapString(GLUT_BITMAP_HELVETICA_18, "Rotation   (rad) :")

        glColor3f(0.4980, 0.5490, 0.5529)
        glRasterPos2i(155, start_h - 50)
        safe_glutBitmapString(GLUT_BITMAP_HELVETICA_18, self.txtR)

        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

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
        self.projection_ = sl.Matrix4f()
        self.projection_.set_identity()
        self.setProjection(1.78)

        self.position_.init_vector(0., 5., -3.)
        tmp = sl.Translation()
        tmp.init_vector(0, 0, -4)
        tmp2 = sl.Translation()
        tmp2.init_vector(0, 1, 0)
        self.setDirection(tmp, tmp2)
        cam_rot = sl.Rotation()
        cam_rot.set_euler_angles(-50., 180., 0., False)
        self.setRotation(cam_rot)

    def update(self): 
        dot_ = sl.Translation.dot_translation(self.vertical_, self.up_)
        if(dot_ < 0.):
            tmp = self.vertical_.get()
            self.vertical_.init_vector(tmp[0] * -1.,tmp[1] * -1., tmp[2] * -1.)
        transformation = sl.Transform()
        transformation.init_orientation_translation(self.orientation_, self.position_)
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
