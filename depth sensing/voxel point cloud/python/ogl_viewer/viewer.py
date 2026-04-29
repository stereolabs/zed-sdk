from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

import ctypes
import sys
import math
from threading import Lock
import numpy as np
import array

import pyzed.sl as sl

M_PI = 3.1415926

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


try:
    from cuda.bindings import runtime as cudart
    import cupy as cp

    GPU_ACCELERATION_AVAILABLE = True

    def format_cudart_err(err):
        return (
            f"{cudart.cudaGetErrorName(err)[1].decode('utf-8')}({int(err)}): "
            f"{cudart.cudaGetErrorString(err)[1].decode('utf-8')}"
        )

    def check_cudart_err(args):
        if isinstance(args, tuple):
            assert len(args) >= 1
            err = args[0]
            if len(args) == 1:
                ret = None
            elif len(args) == 2:
                ret = args[1]
            else:
                ret = args[1:]
        else:
            err = args
            ret = None

        assert isinstance(err, cudart.cudaError_t), type(err)
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(format_cudart_err(err))

        return ret

    class CudaOpenGLMappedBuffer:
        def __init__(self, gl_buffer, flags=0):
            self._gl_buffer = int(gl_buffer)
            self._flags = int(flags)
            self._graphics_ressource = None
            self._cuda_buffer = None
            self.register()

        @property
        def gl_buffer(self):
            return self._gl_buffer

        @property
        def cuda_buffer(self):
            assert self.mapped
            return self._cuda_buffer

        @property
        def graphics_ressource(self):
            assert self.registered
            return self._graphics_ressource

        @property
        def registered(self):
            return self._graphics_ressource is not None

        @property
        def mapped(self):
            return self._cuda_buffer is not None

        def __enter__(self):
            return self.map()

        def __exit__(self, exc_type, exc_value, trace):
            self.unmap()
            return False

        def __del__(self):
            try:
                self.unregister()
            except:
                # Ignore errors during cleanup (e.g., during Python shutdown)
                pass

        def register(self):
            if self.registered:
                return self._graphics_ressource
            self._graphics_ressource = check_cudart_err(
                cudart.cudaGraphicsGLRegisterBuffer(self._gl_buffer, self._flags)
            )
            return self._graphics_ressource

        def unregister(self):
            if not self.registered:
                return self
            try:
                self.unmap()
                if cudart is not None:  # Check if cudart is still available
                    check_cudart_err(
                        cudart.cudaGraphicsUnregisterResource(self._graphics_ressource)
                    )
                self._graphics_ressource = None
            except Exception:
                # Ignore errors during cleanup (e.g., during Python shutdown)
                self._graphics_ressource = None
            return self

        def map(self, stream=None):
            if not self.registered:
                raise RuntimeError("Cannot map an unregistered buffer.")
            if self.mapped:
                return self._cuda_buffer

            check_cudart_err(
                cudart.cudaGraphicsMapResources(1, self._graphics_ressource, stream)
            )

            ptr, size = check_cudart_err(
                cudart.cudaGraphicsResourceGetMappedPointer(self._graphics_ressource)
            )

            self._cuda_buffer = cp.cuda.MemoryPointer(
                cp.cuda.UnownedMemory(ptr, size, self), 0
            )
            return self._cuda_buffer

        def unmap(self, stream=None):
            if not self.registered:
                raise RuntimeError("Cannot unmap an unregistered buffer.")
            if not self.mapped:
                return self

            try:
                if cudart is not None:  # Check if cudart is still available
                    check_cudart_err(
                        cudart.cudaGraphicsUnmapResources(1, self._graphics_ressource, stream)
                    )
                self._cuda_buffer = None
            except Exception:
                # Force cleanup even if unmap fails
                self._cuda_buffer = None
            return self

    class CudaOpenGLMappedArray(CudaOpenGLMappedBuffer):
        def __init__(self, dtype, shape, gl_buffer, flags=0, strides=None, order='C'):
            super().__init__(gl_buffer, flags)
            self._dtype = dtype
            self._shape = shape
            self._strides = strides
            self._order = order

        @property
        def cuda_array(self):
            assert self.mapped
            return cp.ndarray(
                shape=self._shape,
                dtype=self._dtype,
                strides=self._strides,
                order=self._order,
                memptr=self._cuda_buffer,
            )

        def map(self, *args, **kwargs):
            super().map(*args, **kwargs)
            return self.cuda_array

except ImportError:
    GPU_ACCELERATION_AVAILABLE = False


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
            if (self.program_id is not None) and (self.program_id > 0) and glIsProgram(self.program_id):
                glDeleteProgram(self.program_id)
            if (vertex_id is not None) and (vertex_id > 0) and glIsShader(vertex_id):
                glDeleteShader(vertex_id)
            if (fragment_id is not None) and (fragment_id > 0) and glIsShader(fragment_id):
                glDeleteShader(fragment_id)
            raise RuntimeError('Error linking program: %s' % (info))
        if (vertex_id is not None) and (vertex_id > 0) and glIsShader(vertex_id):
            glDeleteShader(vertex_id)
        if (fragment_id is not None) and (fragment_id > 0) and glIsShader(fragment_id):
            glDeleteShader(fragment_id)

    def compile(self, _type, _src):
        shader_id = None
        try:
            shader_id = glCreateShader(_type)
            if shader_id == 0:
                print("ERROR: shader type {0} does not exist".format(_type))
                exit()

            glShaderSource(shader_id, _src)
            glCompileShader(shader_id)
            if glGetShaderiv(shader_id, GL_COMPILE_STATUS) != GL_TRUE:
                info = glGetShaderInfoLog(shader_id)
                if (shader_id is not None) and (shader_id > 0) and glIsShader(shader_id):
                    glDeleteShader(shader_id)
                raise RuntimeError('Shader compilation failed: %s' % (info))
            return shader_id
        except:
            if (shader_id is not None) and (shader_id > 0) and glIsShader(shader_id):
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
        self.data = sl.Mat()
        self.cuda_mapped_buffer = None
        self.use_gpu = GPU_ACCELERATION_AVAILABLE and not _is_static
        self.currentCount = 0

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

    def addFace(self, p1, p2, p3, clr):
        self.add_point_clr(p1, clr)
        self.add_point_clr(p2, clr)
        self.add_point_clr(p3, clr)

    def push_to_GPU(self):
        if not self.is_init:
            self.vboID = glGenBuffers(3)
            self.is_init = True

        if self.is_static:
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
        if not self.is_init:
            self.vboID = glGenBuffers(3)
            self.is_init = True

        if self.is_static:
            type_draw = GL_STATIC_DRAW
        else:
            type_draw = GL_DYNAMIC_DRAW

        self.elementbufferSize = res.width * res.height

        # Initialize vertex buffer (for XYZRGBA data)
        glBindBuffer(GL_ARRAY_BUFFER, self.vboID[0])
        glBufferData(GL_ARRAY_BUFFER, self.elementbufferSize * self.pt_type * self.vertices.itemsize, None, type_draw)

        # Try to set up GPU acceleration if available
        if self.use_gpu:
            try:
                flags = cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard
                self.cuda_mapped_buffer = CudaOpenGLMappedArray(
                    dtype=np.float32, 
                    shape=(self.elementbufferSize, self.pt_type), 
                    gl_buffer=self.vboID[0], 
                    flags=flags
                )
            except Exception as e:
                print(f"Failed to initialize GPU acceleration, falling back to CPU: {e}")
                self.use_gpu = False
                self.cuda_mapped_buffer = None

        # Initialize color buffer (not used for point clouds with XYZRGBA)
        if self.clr_type:
            glBindBuffer(GL_ARRAY_BUFFER, self.vboID[1])
            glBufferData(GL_ARRAY_BUFFER, self.elementbufferSize * self.clr_type * self.colors.itemsize, None, type_draw)

        for i in range (0, self.elementbufferSize):
            self.indices.append(i)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.vboID[2])
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,len(self.indices) * self.indices.itemsize,(GLuint * len(self.indices))(*self.indices), type_draw)

    def setPoints(self, pc):
        """Update point cloud data from sl.Mat"""
        if not pc.is_init():
            return

        try:
            if self.use_gpu and self.cuda_mapped_buffer and pc.get_memory_type() in (sl.MEM.GPU, sl.MEM.BOTH):
                self.setPointsGPU(pc)
            else:
                self.setPointsCPU(pc)
        except Exception as e:
            print(f"Error setting points: {e}")
            # Fallback to CPU if GPU fails
            if self.use_gpu:
                print("Falling back to CPU processing")
                self.use_gpu = False
                self.setPointsCPU(pc)

    def setPointsGPU(self, pc):
        """Set points using GPU acceleration with CUDA-OpenGL interop"""
        try:
            # Get point cloud data from GPU memory
            cupy_arr = pc.get_data(sl.MEM.GPU)

            # Map OpenGL buffer to CUDA memory
            with self.cuda_mapped_buffer as cuda_array:
                # Reshape point cloud data to match buffer format
                if cupy_arr.ndim == 3:  # (height, width, channels)
                    pc_flat = cupy_arr.reshape(-1, cupy_arr.shape[-1])
                else:
                    pc_flat = cupy_arr

                # Copy data to GPU buffer (optimized GPU-to-GPU copy with continuous memory)
                points_to_copy = min(pc_flat.shape[0], cuda_array.shape[0])
                self.currentCount = points_to_copy
                cuda_array[:points_to_copy] = pc_flat[:points_to_copy]

                # Zero out remaining buffer if needed
                if points_to_copy < cuda_array.shape[0]:
                    cuda_array[points_to_copy:] = 0

        except Exception as e:
            print(f"GPU point cloud update failed: {e}")
            raise

    def setPointsCPU(self, pc):
        """Fallback CPU method for setting points"""
        try:
            # Ensure data is available on CPU
            if pc.get_memory_type() == sl.MEM.GPU:
                pc.update_cpu_from_gpu()

            # Get actual point count (may be less than buffer capacity for voxels)
            actual_res = pc.get_resolution()
            actual_count = actual_res.width * actual_res.height
            self.currentCount = min(actual_count, self.elementbufferSize)

            # Get CPU pointer and upload to GPU buffer
            glBindBuffer(GL_ARRAY_BUFFER, self.vboID[0])
            data_ptr = pc.get_pointer(sl.MEM.CPU)
            buffer_size = self.currentCount * self.pt_type * 4  # 4 bytes per float32
            glBufferSubData(GL_ARRAY_BUFFER, 0, buffer_size, ctypes.c_void_p(data_ptr))
            glBindBuffer(GL_ARRAY_BUFFER, 0)

        except Exception as e:
            print(f"CPU point cloud update failed: {e}")
            raise

    def clear(self):
        self.vertices = array.array('f')
        self.colors = array.array('f')
        self.indices = array.array('I')
        self.elementbufferSize = 0

    def set_drawing_type(self, _type):
        self.drawing_type = _type

    def draw(self):
        draw_count = self.currentCount if self.currentCount > 0 else self.elementbufferSize
        if draw_count:
            glEnableVertexAttribArray(0)
            glBindBuffer(GL_ARRAY_BUFFER, self.vboID[0])
            glVertexAttribPointer(0,self.pt_type,GL_FLOAT,GL_FALSE,0,None)

            if(self.clr_type):
                glEnableVertexAttribArray(1)
                glBindBuffer(GL_ARRAY_BUFFER, self.vboID[1])
                glVertexAttribPointer(1,self.clr_type,GL_FLOAT,GL_FALSE,0,None)
            
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.vboID[2])
            glDrawElements(self.drawing_type, draw_count, GL_UNSIGNED_INT, None)      
            
            glDisableVertexAttribArray(0)
            if self.clr_type:
                glDisableVertexAttribArray(1)

    def __del__(self):
        """Cleanup GPU resources"""
        if hasattr(self, 'cuda_mapped_buffer') and self.cuda_mapped_buffer:
            try:
                self.cuda_mapped_buffer.unregister()
            except:
                pass

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
        self.save_data = False
        self.use_voxels = True
        self.key_pressed = 0
        self.paused = False
        self.seekOffset = 0
        self.pointSize = 3.0
        self.confidenceThreshold = 50
        self.windowW = 0
        self.windowH = 0
        # Voxel params (mirrors C++ GLViewer::voxelParams_)
        self.voxel_size = 50.0  # in mm (coordinate units)
        self.resolution_scale = 0.2
        self.resolution_mode = 1  # 0=FIXED, 1=STEREO, 2=LINEAR
        self.centroid = True
        self.mode_names = ["FIXED", "STEREO", "LINEAR"]
        # Image preview texture
        self._preview_tex = 0
        self._preview_quad_vao = 0
        self._preview_quad_vbo = 0
        self._preview_shader = None
        self._preview_res = None

    def init(self, _argc, _argv, res): # _params = sl.CameraParameters
        glutInit(_argc, _argv)
        wnd_w = int(glutGet(GLUT_SCREEN_WIDTH)*0.9)
        wnd_h = int(glutGet(GLUT_SCREEN_HEIGHT) *0.9)
        glutInitWindowSize(wnd_w, wnd_h)
        glutInitWindowPosition(int(wnd_w*0.05), int(wnd_h*0.05))

        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH)
        glutCreateWindow(b"ZED Voxel Point Cloud")
        self.windowW = wnd_w
        self.windowH = wnd_h
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

        self.bckgrnd_clr = np.array([25/255., 25/255., 25/255.])  # sl_soil

        # Create the camera model
        Z_ = -150
        Y_ = Z_ * math.tan(95. * M_PI / 180. / 2.)
        X_ = Y_ * 16./9.

        A = np.array([0, 0, 0])
        B = np.array([X_, Y_, Z_])
        C = np.array([-X_, Y_, Z_])
        D = np.array([-X_, -Y_, Z_])
        E = np.array([X_, -Y_, Z_])

        lime_clr = np.array([217 / 255, 255/255, 66/255])

        self.zedModel.add_line(A, B, lime_clr)
        self.zedModel.add_line(A, C, lime_clr)
        self.zedModel.add_line(A, D, lime_clr)
        self.zedModel.add_line(A, E, lime_clr)

        self.zedModel.add_line(B, C, lime_clr)
        self.zedModel.add_line(C, D, lime_clr)
        self.zedModel.add_line(D, E, lime_clr)
        self.zedModel.add_line(E, B, lime_clr)

        self.zedModel.set_drawing_type(GL_LINES)
        self.zedModel.push_to_GPU()

        self.point_cloud.init(res)
        self.point_cloud.set_drawing_type(GL_POINTS)

        # Register GLUT callback functions
        glutDisplayFunc(self.draw_callback)
        glutIdleFunc(self.idle)
        glutKeyboardFunc(self.keyPressedCallback)
        glutSpecialFunc(self.specialKeyCallback)
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
        try:
            self.point_cloud.setPoints(pc)
        finally:
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
        if ord(key) == 27 or ord(key) == ord('q') or ord(key) == ord('Q'):
            self.close_func()
            return
        if ord(key) == 32:  # Space
            self.paused = not self.paused
        if (ord(key) == 83 or ord(key) == 115):  # 's' or 'S'
            self.save_data = True
        if (ord(key) == 86 or ord(key) == 118):  # 'v' or 'V'
            self.use_voxels = not self.use_voxels
        # Forward key to main loop for voxel parameter controls
        self.key_pressed = ord(key)

    def specialKeyCallback(self, key, x, y):
        if key == GLUT_KEY_LEFT:
            self.seekOffset = -200
        elif key == GLUT_KEY_RIGHT:
            self.seekOffset = 200

    def on_mouse(self, *args, **kwargs):
        (key, Up, x, y) = args
        # Check HUD button clicks first
        if key == 0 and Up == 0:
            if self._handle_button_click(x, y):
                return
        if key == 0:
            self.mouse_button[0] = (Up == 0)
        elif key == 2:
            self.mouse_button[1] = (Up == 0)
        elif key == 3:
            self.wheelPosition = self.wheelPosition + 1
        elif key == 4:
            self.wheelPosition = self.wheelPosition - 1

        self.mouseCurrentPosition = [x, y]
        self.previousMouseMotion = [x, y]

    def on_mousemove(self,*args,**kwargs):
        (x,y) = args
        self.mouseMotion[0] = x - self.previousMouseMotion[0]
        self.mouseMotion[1] = y - self.previousMouseMotion[1]
        self.previousMouseMotion = [x, y]
        glutPostRedisplay()

    def on_resize(self, Width, Height):
        self.windowW = Width
        self.windowH = Height
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
            r.init_angle_translation(self.mouseMotion[0] * 0.03, vert)
            self.camera.rotate(r)

            r.init_angle_translation(self.mouseMotion[1] * 0.03, self.camera.right_)
            self.camera.rotate(r)

        if(self.mouse_button[1]):
            t = sl.Translation()
            tmp = self.camera.right_.get()
            scale = self.mouseMotion[0] * 50
            t.init_vector(tmp[0] * scale, tmp[1] * scale, tmp[2] * scale)
            self.camera.translate(t)

            tmp = self.camera.up_.get()
            scale = self.mouseMotion[1] * 50
            t.init_vector(tmp[0] * scale, tmp[1] * scale, tmp[2] * scale)
            self.camera.translate(t)

        if (self.wheelPosition != 0):
            t = sl.Translation()
            tmp = self.camera.forward_.get()
            scale = self.wheelPosition * -500
            t.init_vector(tmp[0] * scale, tmp[1] * scale, tmp[2] * scale)
            self.camera.translate(t)

        self.camera.update()

        self.mouseMotion = [0., 0.]
        self.wheelPosition = 0

    def draw(self):
        vpMatrix = self.camera.getViewProjectionMatrix()
        glViewport(0, 0, self.windowW, self.windowH)

        glUseProgram(self.shader_image.get_program_id())
        glUniformMatrix4fv(self.shader_image_MVP, 1, GL_TRUE, (GLfloat * len(vpMatrix))(*vpMatrix))
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        self.zedModel.draw()
        glUseProgram(0)

        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glUseProgram(self.shader_pc.get_program_id())
        glUniformMatrix4fv(self.shader_pc_MVP, 1, GL_TRUE, (GLfloat * len(vpMatrix))(*vpMatrix))
        glPointSize(self.pointSize)
        self.point_cloud.draw()
        glUseProgram(0)

        # Image PIP (bottom-left, 25% of window)
        if self._preview_tex:
            pipW = self.windowW // 4
            pipH = self.windowH // 4
            glViewport(10, 10, pipW, pipH)
            glDisable(GL_DEPTH_TEST)
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            self._draw_preview_tex()
            glEnable(GL_DEPTH_TEST)
            glViewport(0, 0, self.windowW, self.windowH)

        # HUD overlay
        self._draw_hud()

    def updateImage(self, image):
        """Upload a CPU sl.Mat (BGRA) as the PIP preview texture."""
        if not image.is_init():
            return
        w = image.get_width()
        h = image.get_height()
        # Init texture on first call or resolution change
        if self._preview_tex == 0 or self._preview_res != (w, h):
            if self._preview_tex:
                glDeleteTextures(1, [self._preview_tex])
            self._preview_tex = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, self._preview_tex)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_BGRA, GL_UNSIGNED_BYTE, None)
            glBindTexture(GL_TEXTURE_2D, 0)
            self._preview_res = (w, h)
        # Upload pixel data
        data = image.get_data()
        if data is not None:
            glBindTexture(GL_TEXTURE_2D, self._preview_tex)
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_BGRA, GL_UNSIGNED_BYTE, data)
            glBindTexture(GL_TEXTURE_2D, 0)

    def _draw_preview_tex(self):
        """Draw the preview texture as a fullscreen quad in the current viewport."""
        glBindVertexArray(0)
        glUseProgram(0)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self._preview_tex)
        glColor4f(1, 1, 1, 1)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 1); glVertex2f(-1, -1)
        glTexCoord2f(1, 1); glVertex2f(1, -1)
        glTexCoord2f(1, 0); glVertex2f(1, 1)
        glTexCoord2f(0, 0); glVertex2f(-1, 1)
        glEnd()
        glBindTexture(GL_TEXTURE_2D, 0)
        glDisable(GL_TEXTURE_2D)

    def _draw_hud(self):
        """Draw 2D HUD overlay with current params, shortcuts, and clickable buttons."""
        glDisable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.windowW, 0, self.windowH, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        W, H = self.windowW, self.windowH
        btn_h = 22
        gap = 5
        sm_w = 28
        mode_w = 70
        tog_w = 100
        left = 10
        row_h = btn_h + gap
        top_y = H - 15

        mode_idx = self.resolution_mode
        is_fixed = (mode_idx == 0)

        # Title
        glColor4f(217/255., 255/255., 66/255., 1)  # sl_lime
        self._draw_str(left, top_y, "Voxel Point Cloud")
        top_y -= row_h

        # Row 0b: Voxel / Full PC toggle
        self._draw_btn(left, top_y, 80, btn_h, "Voxel", self.use_voxels)
        self._draw_btn(left + 80 + gap, top_y, 80, btn_h, "Full PC", not self.use_voxels)
        top_y -= row_h

        # Grey out voxel-specific controls when in Full PC mode
        voxel_on = self.use_voxels

        # Row 1: Voxel Size
        a = 1.0 if voxel_on else 0.5
        glColor4f(194/255. * a, 194/255. * a, 194/255. * a, a)  # sl_iron
        self._draw_str(left, top_y + 6, f"Size: {self.voxel_size:.1f} mm")
        bx = left + 110
        if voxel_on:
            self._draw_btn(bx, top_y, sm_w, btn_h, "-")
            self._draw_btn(bx + sm_w + gap, top_y, sm_w, btn_h, "+")
        else:
            glColor4f(45/255., 45/255., 45/255., 0.5)  # sl_steel disabled
            self._draw_rect(bx, top_y, sm_w, btn_h)
            self._draw_rect(bx + sm_w + gap, top_y, sm_w, btn_h)
        top_y -= row_h

        # Row 2: Scale (greyed when FIXED or Full PC)
        scale_on = voxel_on and not is_fixed
        a = 1.0 if scale_on else 0.5
        glColor4f(194/255. * a, 194/255. * a, 194/255. * a, a)  # sl_iron
        self._draw_str(left, top_y + 6, f"Scale: {self.resolution_scale:.2f}")
        bx = left + 110
        if scale_on:
            self._draw_btn(bx, top_y, sm_w, btn_h, "-")
            self._draw_btn(bx + sm_w + gap, top_y, sm_w, btn_h, "+")
        else:
            glColor4f(45/255., 45/255., 45/255., 0.5)  # sl_steel disabled
            self._draw_rect(bx, top_y, sm_w, btn_h)
            self._draw_rect(bx + sm_w + gap, top_y, sm_w, btn_h)
        top_y -= row_h

        # Row 3: Mode buttons
        a = 1.0 if voxel_on else 0.5
        glColor4f(194/255. * a, 194/255. * a, 194/255. * a, a)  # sl_iron
        self._draw_str(left, top_y + 6, "Mode:")
        bx = left + 50
        if voxel_on:
            for i in range(3):
                self._draw_btn(bx + i * (mode_w + gap), top_y, mode_w, btn_h, self.mode_names[i], i == mode_idx)
        else:
            for i in range(3):
                glColor4f(45/255., 45/255., 45/255., 0.5)  # sl_steel disabled
                self._draw_rect(bx + i * (mode_w + gap), top_y, mode_w, btn_h)
        top_y -= row_h

        # Row 4: Centroid toggle
        a = 1.0 if voxel_on else 0.5
        glColor4f(194/255. * a, 194/255. * a, 194/255. * a, a)  # sl_iron
        self._draw_str(left, top_y + 6, "Pos:")
        bx = left + 50
        if voxel_on:
            self._draw_btn(bx, top_y, tog_w, btn_h, "Centroid", self.centroid)
            self._draw_btn(bx + tog_w + gap, top_y, tog_w, btn_h, "Grid Center", not self.centroid)
        else:
            glColor4f(45/255., 45/255., 45/255., 0.5)  # sl_steel disabled
            self._draw_rect(bx, top_y, tog_w, btn_h)
            self._draw_rect(bx + tog_w + gap, top_y, tog_w, btn_h)
        top_y -= row_h

        # Row 5: Point size
        glColor4f(194/255., 194/255., 194/255., 1)  # sl_iron
        self._draw_str(left, top_y + 6, f"Pt size: {self.pointSize:.1f} px")
        bx = left + 110
        self._draw_btn(bx, top_y, sm_w, btn_h, "-")
        self._draw_btn(bx + sm_w + gap, top_y, sm_w, btn_h, "+")
        top_y -= row_h

        # Row 6: Confidence
        glColor4f(194/255., 194/255., 194/255., 1)  # sl_iron
        self._draw_str(left, top_y + 6, f"Depth Conf: {self.confidenceThreshold}")
        bx = left + 110
        self._draw_btn(bx, top_y, sm_w, btn_h, "-")
        self._draw_btn(bx + sm_w + gap, top_y, sm_w, btn_h, "+")
        top_y -= row_h

        # Row 7: Save + Reset
        self._draw_btn(left, top_y, 80, btn_h, "Save PLY", False, True)
        self._draw_btn(left + 85 + gap, top_y, 65, btn_h, "Reset")

        # Keyboard shortcuts (top-right)
        tx = W - 270
        ty = H - 25
        glColor4f(194/255., 194/255., 194/255., 0.6)  # sl_iron
        self._draw_str(tx, ty,      "+/-  Size    ,/.  Scale", GLUT_BITMAP_HELVETICA_10)
        self._draw_str(tx, ty - 14, "1/2/3 Mode   c Centroid", GLUT_BITMAP_HELVETICA_10)
        self._draw_str(tx, ty - 28, "d/D Depth Conf  p/P Pt size", GLUT_BITMAP_HELVETICA_10)
        self._draw_str(tx, ty - 42, "v Toggle  Space Pause  </>/< Seek", GLUT_BITMAP_HELVETICA_10)
        self._draw_str(tx, ty - 56, "s Save  r Reset  q Quit", GLUT_BITMAP_HELVETICA_10)

        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        glEnable(GL_DEPTH_TEST)

    @staticmethod
    def _draw_str(x, y, text, font=GLUT_BITMAP_HELVETICA_12):
        glWindowPos2f(x, y)
        for ch in text:
            glutBitmapCharacter(font, ord(ch))

    @staticmethod
    def _draw_rect(x, y, w, h):
        glBegin(GL_QUADS)
        glVertex2f(x, y); glVertex2f(x + w, y)
        glVertex2f(x + w, y + h); glVertex2f(x, y + h)
        glEnd()

    @staticmethod
    def _draw_btn(x, y, w, h, label, active=False, highlight=False):
        # Stereolabs brand: sl_lime active, sl_charcoal normal, sl_steel border
        if active:
            glColor4f(217/255., 255/255., 66/255., 0.9)   # sl_lime
        elif highlight:
            glColor4f(60/255., 60/255., 60/255., 0.85)    # sl_charcoal
        else:
            glColor4f(45/255., 45/255., 45/255., 0.8)     # sl_steel
        glBegin(GL_QUADS)
        glVertex2f(x, y); glVertex2f(x + w, y)
        glVertex2f(x + w, y + h); glVertex2f(x, y + h)
        glEnd()
        glColor4f(137/255., 137/255., 137/255., 0.6)      # sl_ash border
        glBegin(GL_LINE_LOOP)
        glVertex2f(x, y); glVertex2f(x + w, y)
        glVertex2f(x + w, y + h); glVertex2f(x, y + h)
        glEnd()
        # Text: dark on lime, light on dark buttons
        if active:
            glColor4f(25/255., 25/255., 25/255., 1)       # sl_soil text
        else:
            glColor4f(242/255., 242/255., 242/255., 1)    # sl_pearl text
        tx = x + w * 0.5 - 4 if len(label) <= 2 else x + 4
        glWindowPos2f(tx, y + 6)
        for ch in label:
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10, ord(ch))

    def _handle_button_click(self, x, y):
        """Check if click hits a HUD button, return True if consumed."""
        gy = self.windowH - y  # GLUT y → GL y
        btn_h, gap, sm_w, mode_w, tog_w, left = 22, 5, 28, 70, 100, 10
        row_h = btn_h + gap
        top_y = self.windowH - 15 - row_h  # skip title

        def hit(bx, by, bw):
            return bx <= x <= bx + bw and by <= gy <= by + btn_h

        # Row 0b: Voxel / Full PC toggle
        if hit(left, top_y, 80):
            self.use_voxels = True
            return True
        if hit(left + 80 + gap, top_y, 80):
            self.use_voxels = False
            return True
        top_y -= row_h

        # Row 1: Size [-] [+] (only when voxel mode)
        if self.use_voxels:
            bx = left + 110
            if hit(bx, top_y, sm_w):
                self.voxel_size = max(5.0, self.voxel_size * 0.8)
                return True
            if hit(bx + sm_w + gap, top_y, sm_w):
                self.voxel_size *= 1.25
                return True
        top_y -= row_h

        # Row 2: Scale [-] [+] (only when voxel mode and not FIXED)
        if self.use_voxels and self.resolution_mode != 0:
            bx = left + 110
            if hit(bx, top_y, sm_w):
                self.resolution_scale = max(0.01, self.resolution_scale * 0.8)
                return True
            if hit(bx + sm_w + gap, top_y, sm_w):
                self.resolution_scale = min(3.0, self.resolution_scale * 1.25)
                return True
        top_y -= row_h

        # Row 3: Mode [FIXED] [STEREO] [LINEAR] (only when voxel mode)
        if self.use_voxels:
            bx = left + 50
            for i in range(3):
                if hit(bx + i * (mode_w + gap), top_y, mode_w):
                    self.resolution_mode = i
                    return True
        top_y -= row_h

        # Row 4: [Centroid] [Grid Center] (only when voxel mode)
        if self.use_voxels:
            bx = left + 50
            if hit(bx, top_y, tog_w):
                self.centroid = True
                return True
            if hit(bx + tog_w + gap, top_y, tog_w):
                self.centroid = False
                return True
        top_y -= row_h

        # Row 5: Pt size [-] [+]
        bx = left + 110
        if hit(bx, top_y, sm_w):
            self.pointSize = max(0.5, self.pointSize - 0.2)
            return True
        if hit(bx + sm_w + gap, top_y, sm_w):
            self.pointSize = min(20.0, self.pointSize + 0.2)
            return True
        top_y -= row_h

        # Row 6: Confidence [-] [+]
        bx = left + 110
        if hit(bx, top_y, sm_w):
            self.confidenceThreshold = max(1, self.confidenceThreshold - 5)
            return True
        if hit(bx + sm_w + gap, top_y, sm_w):
            self.confidenceThreshold = min(100, self.confidenceThreshold + 5)
            return True
        top_y -= row_h

        # Row 7: Save / Reset
        if hit(left, top_y, 80):
            self.save_data = True
            return True
        if hit(left + 85 + gap, top_y, 65):
            self.voxel_size = 50.0
            self.resolution_scale = 0.2
            self.resolution_mode = 1
            self.centroid = True
            self.pointSize = 3.0
            self.confidenceThreshold = 50
            return True

        return False

class CameraGL:
    def __init__(self):
        self.ORIGINAL_FORWARD = sl.Translation()
        self.ORIGINAL_FORWARD.init_vector(0,0,1)
        self.ORIGINAL_UP = sl.Translation()
        self.ORIGINAL_UP.init_vector(0,1,0)
        self.ORIGINAL_RIGHT = sl.Translation()
        self.ORIGINAL_RIGHT.init_vector(1,0,0)
        self.znear = 200.
        self.zfar = 50000.
        self.horizontalFOV = 90.
        self.orientation_ = sl.Orientation()
        self.position_ = sl.Translation()
        self.forward_ = sl.Translation()
        self.up_ = sl.Translation()
        self.right_ = sl.Translation()
        self.vertical_ = sl.Translation()
        self.vpMatrix_ = sl.Matrix4f()
        self.offset_ = sl.Translation()
        self.offset_.init_vector(0,0,0)
        self.projection_ = sl.Matrix4f()
        self.projection_.set_identity()
        self.setProjection(1.78)

        self.position_.init_vector(0., 2000., 3000.)
        tmp = sl.Translation()
        tmp.init_vector(0, 0, -100)
        tmp2 = sl.Translation()
        tmp2.init_vector(0, 1, 0)
        self.setDirection(tmp, tmp2)
        # Set absolute camera orientation to match C++ viewer
        r = sl.Rotation()
        r.set_euler_angles(-25., 0., 0., False)
        self.setRotation(r)

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
