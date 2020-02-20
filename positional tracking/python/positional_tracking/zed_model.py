from OpenGL.GL import *
import numpy as np
import pyzed.sl as sl
import positional_tracking.utils as ut


class Color:

    def __init__(self, pr, pg, pb):
        self.r = pr
        self.g = pg
        self.b = pb

NB_VERTICES = 56
NB_ALLUMINIUM_TRIANGLES = 54
NB_DARK_TRIANGLES = 54
ALLUMINIUM_COLOR = Color(0.64, 0.64, 0.64)
DARK_COLOR = Color(0.052218, 0.052218, 0.052218)
I = 0

vertices = np.array([
    -0.068456, -0.016299, 0.016299
    , -0.068456, 0.016299, 0.016299
    , -0.068456, 0.016299, -0.016299
    , -0.068456, -0.016299, -0.016299
    , -0.076606, 0.014115, 0.016299
    , -0.082572, 0.008150, 0.016299
    , -0.084755, -0.000000, 0.016299
    , -0.082572, -0.008150, 0.016299
    , -0.076606, -0.014115, 0.016299
    , -0.076606, -0.014115, -0.016299
    , -0.082572, -0.008150, -0.016299
    , -0.084755, -0.000000, -0.016299
    , -0.082572, 0.008150, -0.016299
    , -0.076606, 0.014115, -0.016299
    , -0.053494, -0.009779, -0.016299
    , -0.048604, -0.008469, -0.016299
    , -0.045024, -0.004890, -0.016299
    , -0.043714, 0.000000, -0.016299
    , -0.045024, 0.004890, -0.016299
    , -0.048604, 0.008469, -0.016299
    , -0.053494, 0.009779, -0.016299
    , -0.058383, 0.008469, -0.016299
    , -0.061963, 0.004890, -0.016299
    , -0.063273, 0.000000, -0.016299
    , -0.061963, -0.004890, -0.016299
    , -0.058383, -0.008469, -0.016299
    , 0.000000, -0.016299, -0.016299
    , 0.068456, -0.016299, 0.016299
    , 0.000000, 0.016299, -0.016299
    , 0.068456, 0.016299, 0.016299
    , 0.068456, 0.016299, -0.016299
    , 0.068456, -0.016299, -0.016299
    , 0.076606, 0.014115, 0.016299
    , 0.082572, 0.008150, 0.016299
    , 0.084755, -0.000000, 0.016299
    , 0.082572, -0.008150, 0.016299
    , 0.076606, -0.014115, 0.016299
    , 0.076606, -0.014115, -0.016299
    , 0.082572, -0.008150, -0.016299
    , 0.084755, -0.000000, -0.016299
    , 0.082572, 0.008150, -0.016299
    , 0.076606, 0.014115, -0.016299
    , 0.053494, -0.009779, -0.016299
    , 0.048604, -0.008469, -0.016299
    , 0.045024, -0.004890, -0.016299
    , 0.043714, 0.000000, -0.016299
    , 0.045024, 0.004890, -0.016299
    , 0.048604, 0.008469, -0.016299
    , 0.053494, 0.009779, -0.016299
    , 0.058383, 0.008469, -0.016299
    , 0.061963, 0.004890, -0.016299
    , 0.063273, 0.000000, -0.016299
    , 0.061963, -0.004890, -0.016299
    , 0.058383, -0.008469, -0.016299
    , 0.053494, 0.000000, -0.016299
    , -0.053494, 0.000000, -0.016299
])

alluminium_triangles = np.array([
    1, 10, 4
    , 6, 14, 13
    , 7, 13, 12
    , 8, 12, 11
    , 9, 11, 10
    , 5, 3, 14
    , 44, 45, 55
    , 47, 48, 55
    , 43, 44, 55
    , 46, 47, 55
    , 52, 53, 55
    , 48, 49, 55
    , 54, 43, 55
    , 50, 51, 55
    , 53, 54, 55
    , 49, 50, 55
    , 45, 46, 55
    , 51, 52, 55
    , 27, 32, 28
    , 38, 28, 32
    , 42, 34, 41
    , 41, 35, 40
    , 40, 36, 39
    , 39, 37, 38
    , 31, 33, 42
    , 27, 1, 4
    , 20, 19, 56
    , 22, 21, 56
    , 23, 22, 56
    , 24, 23, 56
    , 19, 18, 56
    , 21, 20, 56
    , 17, 16, 56
    , 26, 25, 56
    , 15, 26, 56
    , 18, 17, 56
    , 16, 15, 56
    , 25, 24, 56
    , 2, 29, 3
    , 31, 29, 30
    , 1, 9, 10
    , 6, 5, 14
    , 7, 6, 13
    , 8, 7, 12
    , 9, 8, 11
    , 5, 2, 3
    , 38, 37, 28
    , 42, 33, 34
    , 41, 34, 35
    , 40, 35, 36
    , 39, 36, 37
    , 31, 30, 33
    , 27, 28, 1
    , 2, 30, 29
])

dark_triangles = np.array([
    23, 3, 22
    , 13, 10, 11
    , 4, 14, 3
    , 11, 12, 13
    , 9, 6, 8
    , 1, 5, 9
    , 8, 6, 7
    , 1, 30, 2
    , 21, 22, 3
    , 23, 24, 3
    , 24, 25, 4
    , 3, 24, 4
    , 25, 26, 4
    , 26, 15, 4
    , 16, 17, 27
    , 17, 18, 27
    , 18, 19, 29
    , 27, 18, 29
    , 19, 20, 29
    , 20, 21, 29
    , 3, 29, 21
    , 16, 27, 15
    , 27, 4, 15
    , 51, 50, 31
    , 38, 41, 39
    , 32, 42, 38
    , 39, 41, 40
    , 34, 37, 36
    , 28, 33, 30
    , 36, 35, 34
    , 49, 31, 50
    , 51, 31, 52
    , 52, 32, 53
    , 31, 32, 52
    , 53, 32, 54
    , 54, 32, 43
    , 44, 27, 45
    , 45, 27, 46
    , 46, 29, 47
    , 27, 29, 46
    , 47, 29, 48
    , 48, 29, 49
    , 31, 49, 29
    , 44, 43, 27
    , 27, 43, 32
    , 13, 14, 10
    , 4, 10, 14
    , 9, 5, 6
    , 1, 2, 5
    , 1, 28, 30
    , 38, 42, 41
    , 32, 31, 42
    , 34, 33, 37
    , 28, 37, 33
])


class Zed3D:
    def __init__(self):
        self.body_io = []
        self.path_mem = []
        self.path = sl.Transform()

        self.set_path(self.path, self.path_mem)

    def set_path(self, path, path_history):
        self.body_io.clear()
        for i in range(0, NB_ALLUMINIUM_TRIANGLES * 3, 3):
            for j in range(3):
                tmp = ut.Double3colorStruct()
                index = int(alluminium_triangles[i + j] - 1)
                tmp.set_coord(vertices[index * 3], vertices[index * 3 + 1], vertices[index * 3 + 2])
                tmp.set_color(ALLUMINIUM_COLOR.r, ALLUMINIUM_COLOR.g, ALLUMINIUM_COLOR.b)
                tmp.transform(path)
                self.body_io.append(tmp)

        for i in range(0, NB_DARK_TRIANGLES * 3, 3):
            for j in range(3):
                tmp = ut.Double3colorStruct()
                index = dark_triangles[i + j] - 1
                tmp.set_coord(vertices[index * 3], vertices[index * 3 + 1], vertices[index * 3 + 2])
                tmp.set_color(DARK_COLOR.r, DARK_COLOR.g, DARK_COLOR.b)
                tmp.transform(path)
                self.body_io.append(tmp)

        self.path_mem = path_history

    def draw(self):
        glPushMatrix()

        glBegin(GL_TRIANGLES)
        for i in range(NB_ALLUMINIUM_TRIANGLES * 3):
            tmp = self.body_io[i]
            glColor3f(tmp.r, tmp.g, tmp.b)
            glVertex3f(tmp.x, tmp.y, tmp.z)

        for i in range(NB_ALLUMINIUM_TRIANGLES * 3, NB_ALLUMINIUM_TRIANGLES * 3 + NB_DARK_TRIANGLES * 3):
            tmp = self.body_io[i]
            glColor3f(tmp.r, tmp.g, tmp.b)
            glVertex3f(tmp.x, tmp.y, tmp.z)

        glEnd()

        if len(self.path_mem) > 1:
            glBegin(GL_LINES)
            for i in range(1, len(self.path_mem)):
                glColor3f(0.1, 0.5, 0.9)
                glVertex3f(self.path_mem[i-1].get()[0], self.path_mem[i-1].get()[1], self.path_mem[i-1].get()[2])
                glVertex3f(self.path_mem[i].get()[0], self.path_mem[i].get()[1], self.path_mem[i].get()[2])
            glEnd()

        glPopMatrix()
