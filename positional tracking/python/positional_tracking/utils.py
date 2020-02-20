import math

M_PI = 3.1416


class Double3colorStruct:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.z = 0
        self.r = 0
        self.g = 0
        self.b = 0

    def init_xyz(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.r = 0
        self.g = 255
        self.b = 255

    def init_xyzrgb(self, x, y, z, r, g, b):
        self.x = x
        self.y = y
        self.z = z
        self.r = r
        self.g = g
        self.b = b

    def __mul__(self, other):
        self.x = self. x * other
        self.y = self.y * other
        self.z = self.z * other
        return self

    def __div__(self, other):
        self.x = self.x / other
        self.y = self.y / other
        self.z = self.z / other
        return self

    def set_color(self, r, g, b):
        self.r = r
        self.g = g
        self.b = b

    def set_coord(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def transform(self, path):
        x_tmp = self.x * path[0, 0] + self.y * path[0, 1] + self.z * path[0, 2] + path[0, 3]
        y_tmp = self.x * path[1, 0] + self.y * path[1, 1] + self.z * path[1, 2] + path[1, 3]
        self.z = self.x * path[2, 0] + self.y * path[2, 1] + self.z * path[2, 2] + path[2, 3]
        self.x = x_tmp
        self.y = y_tmp


def d2r(degree):
    return degree * M_PI/180


def r2d(radians):
    return radians * 180/M_PI


class Vect3:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.z = 0

    def init_vect3(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        return self

    def normalise(self):
        length = math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
        if length != 0:
            self.x = self.x / length
            self.y = self.y / length
            self.z = self.z / length
        else:
            self.x = 0
            self.y = 0
            self.z = 0

    def rotate(self, angle, axis):
        rangle = d2r(angle)
        cc = math.cos(rangle)
        ss = math.sin(rangle)
        a = axis.x * axis.y + (1 - axis.x * axis.x) * cc
        b = axis.x * axis.y * (1 - cc) - axis.z * ss
        c = axis.x * axis.z * (1 - cc) + axis.y * ss
        d = axis.x * axis.y * (1 - cc) + axis.z * ss
        e = axis.y * axis.y + (1 - axis.y * axis.y) * cc
        f = axis.y * axis.z * (1 - cc) - axis.x * ss
        g = axis.x * axis.z * (1 - cc) - axis.y * ss
        h = axis.y * axis.z * (1 - cc) + axis.x * ss
        i = axis.z * axis.z + (1 - axis.z * axis.z) * cc

        nx = self.x * a + self.y * b + self.z * c
        ny = self.x * d + self.y * e + self.z * f
        nz = self.x * g + self.y * h + self.z * i

        self.x = nx
        self.y = ny
        self.z = nz

    def length(self, u):
        return math.sqrt(u.x * u.x + u.y * u.y + u.z * u.z)

    def dot(self, u, v):
        return u.x * v.x + u.y * v.y + u.z * v.z

    def get_angle(self, a, o, b):
        oa = Vect3().init_vect3(a.x - o.x, a.y - o.y, a.z - o.z)
        ob = Vect3().init_vect3(b.x - o.x, b.y - o.y, b.z - o.z)
        s = math.acos(self.dot(oa, ob) / (self.length(oa) * self.length(ob)))
        return r2d(s)
