from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

import ctypes
import sys
import math
import threading

import positional_tracking.utils as ut
import positional_tracking.zed_model as zm
import pyzed.sl as sl


def safe_glut_bitmap_string(w):
    for x in w:
        glutBitmapCharacter(GLUT_BITMAP_9_BY_15, ctypes.c_int(ord(x)))


class TrackBallCamera:
    def __init__(self, p=ut.Vect3(), la=ut.Vect3()):
        self.position = ut.Vect3()
        self.look_at = ut.Vect3()
        self.forward = ut.Vect3()
        self.up = ut.Vect3()
        self.left = ut.Vect3()

        self.position.x = p.x
        self.position.y = p.y
        self.position.z = p.z

        self.look_at.x = la.x
        self.look_at.t = la.y
        self.look_at.z = la.z

        self.angle_x = 0.0
        self.apply_transformations()

    def apply_transformations(self):
        self.forward.init_vect3(self.look_at.x - self.position.x,
                                self.look_at.y - self.position.y,
                                self.look_at.z - self.position.z)

        self.left = ut.Vect3().init_vect3(self.forward.z, 0, - self.forward.x)

        self.up = ut.Vect3().init_vect3(self.left.y * self.forward.z - self.left.z * self.forward.y,
                                        self.left.z * self.forward.x - self.left.x * self.forward.z,
                                        self.left.x * self.forward.y - self.left.y * self.forward.x)

        self.forward.normalise()
        self.left.normalise()
        self.up.normalise()

    def show(self):
        gluLookAt(self.position.x, self.position.y, self.position.z,
                  self.look_at.x, self.look_at.y, self.look_at.z,
                  0.0, 1.0, 0.0)

    def rotation(self, angle, v):
        self.translate(ut.Vect3().init_vect3(- self.look_at.x, - self.look_at.y, - self.look_at.z))
        self.position.rotate(angle, v)
        self.translate(ut.Vect3().init_vect3(self.look_at.x, self.look_at.y, self.look_at.z))
        self.set_angle_x()

    def rotate(self, speed, v):
        angle = speed / 360.0
        if v.x != 0.0:
            tmp_a = self.angle_x - 90.0 + angle
        else:
            tmp_a = self.angle_x - 90.0
        if -89.5 < tmp_a < 89.5:
            self.translate(ut.Vect3().init_vect3(- self.look_at.x, - self.look_at.y, - self.look_at.z))
            self.position.rotate(angle, v)
            self.translate(ut.Vect3().init_vect3(self.look_at.x, self.look_at.y, self.look_at.z))
        self.set_angle_x()

    def translate(self, v):
        self.position.x = self.position.x + v.x
        self.position.y = self.position.y + v.y
        self.position.z = self.position.z + v.z

    def translate_look_at(self, v):
        self.look_at.x = self.look_at.x + v.x
        self.look_at.y = self.look_at.y + v.y
        self.look_at.z = self.look_at.z + v.z

    def translate_all(self, v):
        self.translate(v)
        self.translate_look_at(v)

    def zoom(self, z):
        dist = ut.Vect3().init_vect3(self.position.x - self.look_at.x, self.position.y - self.look_at.y,
                                     self.position.z - self.look_at.z)
        dist = dist.length(dist)
        if dist - z > z:
            self.translate(ut.Vect3().init_vect3(self.forward.x * z, self.forward.y * z, self.forward.z * z))

    def get_position(self):
        return ut.Vect3().init_vect3(self.position.x, self.position.y, self.position.z)

    def get_position_from_look_at(self):
        return ut.Vect3().init_vect3(self.position.x - self.look_at.x, self.position.y - self.look_at.y,
                                     self.position.z - self.look_at.z)

    def get_look_at(self):
        return ut.Vect3().init_vect3(self.look_at.x, self.look_at.y, self.look_at.z)

    def get_forward(self):
        return ut.Vect3().init_vect3(self.forward.x, self.forward.y, self.forward.z)

    def get_up(self):
        return ut.Vect3().init_vect3(self.up.x, self.up.y, self.up.z)

    def get_left(self):
        return ut.Vect3().init_vect3(self.left.x, self.left.y, self.left.z)

    def set_position(self, p):
        self.position.x = p.x
        self.position.y = p.y
        self.position.z = p.z
        self.set_angle_x()

    def set_look_at(self, p):
        self.look_at.x = p.x
        self.look_at.y = p.y
        self.look_at.z = p.z
        self.set_angle_x()

    def set_angle_x(self):
        self.angle_x = ut.Vect3().get_angle(ut.Vect3().init_vect3(self.position.x, self.position.y + 1,
                                                                  self.position.z),
                                            ut.Vect3().init_vect3(self.position.x, self.position.y, self.position.z),
                                            ut.Vect3().init_vect3(self.look_at.x, self.look_at.y, self.look_at.z))


class PyTrackingViewer:
    def __init__(self):
        self.is_init = False
        self.run = False
        self.current_instance = self
        self.camera = TrackBallCamera(ut.Vect3().init_vect3(2.56, 1.2, 0.6), ut.Vect3().init_vect3(0.79, 0.02, -1.53))
        self.translate = False
        self.rotate = False
        self.zoom = False
        self.zed_path = []
        self.path_locker = threading.Lock()
        self.zed3d = zm.Zed3D()
        self.track_state = sl.POSITIONAL_TRACKING_STATE
        self.txt_t = ""
        self.txt_r = ""
        self.startx = 0
        self.starty = 0

    def redraw_callback(self):
        self.current_instance.redraw()
        glutPostRedisplay()

    def mouse_callback(self, button, state, x, y):
        self.current_instance.mouse(button, state, x, y)

    def key_callback(self, c, x, y):
        self.current_instance.key(c, x, y)

    def special_key_callback(self, key, x, y):
        self.current_instance.special_key(key, x, y)

    def motion_callback(self, x, y):
        self.current_instance.motion(x, y)

    def reshape_callback(self, width, height):
        self.current_instance.reshape(width, height)

    def close_callback(self):
        self.current_instance.exit()

    def init(self):
        sys.argv[0] = '\0'
        argc = 1
        glutInit(argc, sys.argv)

        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH)

        w = glutGet(GLUT_SCREEN_WIDTH)
        h = glutGet(GLUT_SCREEN_HEIGHT)
        glutInitWindowSize(w, h)

        glutCreateWindow(b"ZED Tracking Viewer")

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)
        glMatrixMode(GL_PROJECTION)
        gluPerspective(75.0, 1.0, .002, 25.0)
        glMatrixMode(GL_MODELVIEW)
        gluLookAt(0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, .10, 0.0)

        glShadeModel(GL_SMOOTH)
        glDepthFunc(GL_LEQUAL)
        glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)

        glutDisplayFunc(self.redraw_callback)
        glutMouseFunc(self.mouse_callback)
        glutKeyboardFunc(self.key_callback)
        glutMotionFunc(self.motion_callback)
        glutReshapeFunc(self.reshape_callback)
        glutSpecialFunc(self.special_key_callback)

        glClearDepth(1.0)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)

        self.zed_path.clear()
        self.is_init = True
        self.run = True

    def draw_repere(self):
        num_segments = 60
        rad = 0.2

        c1 = 0.09 * 0.5
        c2 = 0.725 * 0.5
        c3 = 0.925 * 0.5

        glLineWidth(2.)

        glBegin(GL_LINE_LOOP)
        for ii in range(num_segments):
            theta = 2.0 * 3.1415926 * float(ii) / float(num_segments)
            glColor3f(c1, c2, c3)
            glVertex3f(rad * math.cos(theta), rad * math.sin(theta), 0)
        glEnd()

        glBegin(GL_LINE_LOOP)
        for ii in range(num_segments):
            theta = 2.0 * 3.1415926 * float(ii) / float(num_segments)
            c1, c2, c3 = get_color(num_segments, ii)
            glColor3f(c3, c2, c2)
            glVertex3f(0, rad * math.sin(theta), rad * math.cos(theta))
        glEnd()

        glBegin(GL_LINE_LOOP)
        for ii in range(num_segments):
            theta = 2.0 * ut.M_PI * (ii + num_segments / 4.) / float(num_segments)
            if theta > (2. * ut.M_PI):
                theta = theta - (2. * ut.M_PI)
                c1, c2, c3 = get_color(num_segments, ii)
            glColor3f(c2, c3, c1)
            glVertex3f(rad * math.cos(theta), 0, rad * math.sin(theta))
        glEnd()

    def draw_line(self, a, b, c, d, c1, c2):
        glBegin(GL_LINES)
        glColor3f(c1.r, c1.g, c1.b)
        glVertex3f(a, 0, b)
        glColor3f(c2.r, c2.g, c2.b)
        glVertex3f(c, 0, d)
        glEnd()

    def draw_grid_plan(self):
        c1 = zm.Color(13/255, 17/255, 20/255)
        c2 = zm.Color(213/255, 207/255, 200/255)
        span = 20
        for i in range(- span, span):
            self.draw_line(i, -span, i, span, c1, c2)
            clr = (i + span) / (span * 2)
            c3 = zm.Color(clr, clr, clr)
            self.draw_line(-span, i, span, i, c3, c3)

    def update_zed_position(self, pose):
        if not self.get_viewer_state():
            return
        self.zed_path.append(pose.get_translation())
        self.path_locker.acquire()
        self.zed3d.set_path(pose, self.zed_path)
        self.path_locker.release()

    def redraw(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glPushMatrix()

        self.camera.apply_transformations()
        self.camera.show()

        glDisable(GL_LIGHTING)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glClearColor(0.12, 0.12, 0.12, 1.0)

        self.path_locker.acquire()
        self.draw_grid_plan()
        self.draw_repere()
        self.zed3d.draw()
        self.print_text()
        self.path_locker.release()
        glutSwapBuffers()
        glPopMatrix()

    def idle(self):
        glutPostRedisplay()

    def exit(self):
        self.run = False
        glutLeaveMainLoop()

    def mouse(self, button, state, x, y):
        if button == GLUT_LEFT_BUTTON:
            if state == GLUT_DOWN:
                self.rotate = True
                self.startx = x
                self.starty = y
            elif state == GLUT_UP:
                self.rotate = False
        if button == GLUT_RIGHT_BUTTON:
            if state == GLUT_DOWN:
                self.translate = True
                self.startx = x
                self.starty = y
            elif state == GLUT_UP:
                self.translate = False
        if button == GLUT_MIDDLE_BUTTON:
            if state == GLUT_DOWN:
                self.zoom = True
                self.startx = x
                self.starty = y
            elif state == GLUT_UP:
                self.zoom = False
        if button == 3 or button == 4:
            if state == GLUT_UP:
                return
            if button == 3:
                self.camera.zoom(0.5)
            else:
                self.camera.zoom(-0.5)

    def motion(self, x, y):
        if self.translate:
            trans_x = (x - self.startx) / 30.0
            trans_y = (y - self.starty) / 30.0
            left = self.camera.get_left()
            up = self.camera.get_up()
            self.camera.translate_all(ut.Vect3().init_vect3(left.x * trans_x, left.y * trans_x, left.z * trans_x))
            self.camera.translate_all(ut.Vect3().init_vect3(up.x * (- trans_y), up.y * (- trans_y), up.z * (- trans_y)))
            self.startx = x
            self.starty = y
        if self.zoom:
            self.camera.zoom((y - self.starty) / 10.0)
            self.starty = y
        if self.rotate:
            sensitivity = 100.0
            rot = y - self.starty
            tmp = self.camera.get_position_from_look_at()
            tmp.y = tmp.x
            tmp.x = - tmp.z
            tmp.z = tmp.y
            tmp.y = 0.0
            tmp.normalise()
            self.camera.rotate(rot * sensitivity, tmp)
        glutPostRedisplay()

    def reshape(self, width, height):
        window_width = width
        window_height = height
        glViewport(0, 0, window_width, window_height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(75.0, window_width / window_height, .002, 40.0)
        glMatrixMode(GL_MODELVIEW)

    def key(self, bkey, x, y):
        key = bkey.decode("utf-8")
        if key == 'o':
            self.camera.set_position(ut.Vect3().init_vect3(2.56, 1.2, 0.6))
            self.camera.set_look_at(ut.Vect3().init_vect3(0.79, 0.02, - 1.53))
        elif key == 'q' or key == 'Q' or key == 27:
            self.current_instance.run = False
            glutLeaveMainLoop()
        glutPostRedisplay()

    def special_key(self, key, x, y):
        sensitivity = 150.0
        tmp = self.camera.get_position_from_look_at()

        tmp.y = tmp.x
        tmp.x = - tmp.z
        tmp.z = tmp.y
        tmp.y = 0.0
        tmp.normalise()

        if key == GLUT_KEY_UP:
            self.camera.rotate(-sensitivity, tmp)
        elif key == GLUT_KEY_DOWN:
            self.camera.rotate(sensitivity, tmp)
        elif key == GLUT_KEY_LEFT:
            self.camera.rotate(sensitivity, ut.Vect3().init_vect3(0.0, 1.0, 0.0))
        elif key == GLUT_KEY_RIGHT:
            self.camera.rotate(-sensitivity, ut.Vect3().init_vect3(0.0, 1.0, 0.0))

    def print_text(self):
        if not self.is_init:
            return
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        w_wnd = glutGet(GLUT_WINDOW_WIDTH)
        h_wnd = glutGet(GLUT_WINDOW_HEIGHT)
        glOrtho(0, w_wnd, 0, h_wnd, -1.0, 1.0)

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        start_w = 20
        start_h = h_wnd - 40

        if self.track_state == sl.POSITIONAL_TRACKING_STATE.OK:
            tracking_is_ok = 1
        else:
            tracking_is_ok = 0

        if tracking_is_ok:
            glColor3f(0.2, 0.65, 0.2)
        else:
            glColor3f(0.85, 0.2, 0.2)
        glRasterPos2i(start_w, start_h)
        safe_glut_bitmap_string(repr(self.track_state))

        glColor3f(0.9255, 0.9412, 0.9451)
        glRasterPos2i(start_w, start_h - 25)
        safe_glut_bitmap_string("Translation(m):")

        glColor3f(0.4980, 0.5490, 0.5529)
        glRasterPos2i(155, start_h - 25)
        safe_glut_bitmap_string(self.txt_t)

        glColor3f(0.9255, 0.9412, 0.9451)
        glRasterPos2i(start_w, start_h - 50)
        safe_glut_bitmap_string("Rotation(rad):")

        glColor3f(0.4980, 0.5490, 0.5529)
        glRasterPos2i(155, start_h - 50)
        safe_glut_bitmap_string(self.txt_r)

        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

    def update_text(self, string_t, string_r, state):
        if not self.get_viewer_state():
            return

        self.txt_t = string_t
        self.txt_r = string_r
        self.track_state = state

    def get_viewer_state(self):
        return self.is_init


def get_color(num_segments, i):
    r = math.fabs(1. - (float(i) * 2.) / float(num_segments))
    c1 = 0.1 * r
    c2 = 0.3 * r
    c3 = 0.8 * r
    return c1, c2, c3
