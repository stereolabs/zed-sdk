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

SK_VERTEX_SHADER = """
# version 330 core
layout(location = 0) in vec3 in_Vertex;
layout(location = 1) in vec4 in_Color;
layout(location = 2) in vec3 in_Normal;
out vec4 b_color;
out vec3 b_position;
out vec3 b_normal;
uniform mat4 u_mvpMatrix;
uniform vec4 u_color;
void main() {
   b_color = in_Color;
   b_position = in_Vertex;
   b_normal = in_Normal;
   gl_Position =  u_mvpMatrix * vec4(in_Vertex, 1);
}
"""

SK_FRAGMENT_SHADER = """
# version 330 core
in vec4 b_color;
in vec3 b_position;
in vec3 b_normal;
out vec4 out_Color;
void main() {
	vec3 lightPosition = vec3(0, 2, 1);
	float ambientStrength = 0.3;
	vec3 lightColor = vec3(0.75, 0.75, 0.9);
	vec3 ambient = ambientStrength * lightColor;
	vec3 lightDir = normalize(lightPosition - b_position);
	float diffuse = (1 - ambientStrength) * max(dot(b_normal, lightDir), 0.0);
    out_Color = vec4(b_color.rgb * (diffuse + ambient), 1);
}
"""

ID_COLORS = np.array([
	[0.231, 0.909, 0.69]	
    , [0.098, 0.686, 0.816]	
    , [0.412, 0.4, 0.804]	
    , [1, 0.725, 0]	
    , [0.989, 0.388, 0.419]]
    , np.float32)

def generate_color_id(_idx):
    clr = []
    if _idx < 0:
        clr = [0.84, 0.52, 0.1, 1.]
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
        self.is_init = False

        self.vertices = array.array('f')
        self.colors = array.array('f')
        self.normals = array.array('f')
        self.indices = array.array('I')

    def __del__(self):
        self.is_init = False
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
            current_size_index = int((len(self.vertices)/3))-1
            self.indices.append(current_size_index)
            self.indices.append(current_size_index+1)

    """
    Add a point and its corresponding color to the list of points
    """
    def add_point_clr(self, _pt, _clr):
        self.add_pt(_pt)
        self.add_clr(_clr)
        self.indices.append(len(self.indices))

    def add_point_clr_norm(self, _pt, _clr, _norm):
        self.add_pt(_pt)
        self.add_clr(_clr)
        self.add_normal(_norm)
        self.indices.append(len(self.indices))

    """
    Define a line from two points
    """
    def add_line(self, _p1, _p2, _clr):
        self.add_point_clr(_p1, _clr)
        self.add_point_clr(_p2, _clr)
    
    """
    Define a cylinder from its top and bottom faces and its corresponding color
    """
    def add_cylinder(self, _start_position, _end_position, _clr):
        # Compute rotation matrix
        m_radius = 0.010  

        dir = np.array([_end_position[0] - _start_position[0], _end_position[1] -
                       _start_position[1], _end_position[2] - _start_position[2]])

        m_height = np.linalg.norm(dir)

        dir = np.divide(dir, m_height)

        yAxis = np.array([0, 1, 0])
        v = np.cross(dir, yAxis)

        rotation = sl.Matrix3f()   # identity matrix
        rotation.set_identity()

        data = sl.Matrix3f()

        if(np.linalg.norm(v) > 0.00001):
            cosTheta = np.dot(dir, yAxis)
            scale = (1-cosTheta)/(1-(cosTheta*cosTheta))

            data.set_zeros()
            data[0, 1] = v[2]
            data[0, 2] = -v[1]
            data[1, 0] = -v[2]
            data[1, 2] = v[0]
            data[2, 0] = v[1]
            data[2, 1] = -v[0]

            vx2 = np.matmul(data.r, data.r) * scale 

            for i in range(3):
                for j in range(3):
                    rotation[i,j] = rotation[i,j] + data[i,j]

            for i in range(3):
                for j in range(3):
                    rotation[i,j] = rotation[i,j] + vx2[i,j]
                    
        ######################################
        rot = rotation.r

        NB = 8
        for j in range(NB):
            i = 2. * M_PI * (j/NB)
            i1 = 2. * M_PI * ((j+1)/NB)
            v1_vec = [m_radius * math.cos(i), 0, m_radius * math.sin(i)]
            v1 = np.matmul(rot, v1_vec) + _start_position
            v2_vec = [v1_vec[0], m_height, v1_vec[2]]
            v2 = np.matmul(rot, v2_vec) + _start_position
            v3_vec = [m_radius * math.cos(i1), 0, m_radius * math.sin(i1)]
            v3 = np.matmul(rot, v3_vec) + _start_position
            v4_vec =  [v3_vec[0], m_height, v3_vec[2]]
            v4 = np.matmul(rot, v4_vec) + _start_position
            
            normal = np.cross((v2 - v1), (v3 - v1))
            normal = np.divide(normal, np.linalg.norm(normal))

            self.add_point_clr_norm(v1, _clr, normal)
            self.add_point_clr_norm(v2, _clr, normal)
            self.add_point_clr_norm(v4, _clr, normal)
            self.add_point_clr_norm(v3, _clr, normal)

    def add_sphere(self, _position, _clr): 
        m_radius = 0.02

        m_stack_count = 6
        m_sector_count = 6

        for i in range(m_stack_count+1):
            lat0 = M_PI * (-0.5 + (i - 1) / m_stack_count)
            z0 = math.sin(lat0)
            zr0 = math.cos(lat0)

            lat1 = M_PI * (-0.5 + i / m_stack_count)
            z1 = math.sin(lat1)
            zr1 = math.cos(lat1)
            for j in range(m_sector_count):
                lng = 2 * M_PI * (j - 1) / m_sector_count
                x = math.cos(lng)
                y = math.sin(lng)

                v = [m_radius * x * zr0, m_radius * y * zr0, m_radius * z0] + _position
                normal = [x * zr0, y * zr0, z0]
                self.add_point_clr_norm(v, _clr, normal)

                v = [m_radius * x * zr1, m_radius * y * zr1, m_radius * z1] + _position
                normal = [x * zr1, y * zr1, z1]
                self.add_point_clr_norm(v, _clr, normal)

                lng = 2 * M_PI * j / m_sector_count
                x = math.cos(lng)
                y = math.sin(lng)

                v= [m_radius * x * zr1, m_radius *
                    y * zr1, m_radius * z1] + _position
                normal = [x * zr1, y * zr1, z1]
                self.add_point_clr_norm(v, _clr, normal)

                v = [m_radius * x * zr0, m_radius *
                    y * zr0, m_radius * z0] + _position
                normal = [x * zr0, y * zr0, z0]
                
                self.add_point_clr_norm(v, _clr, normal)

    def push_to_GPU(self):
        if( self.is_init == False):
            self.vboID = glGenBuffers(4)
            self.is_init = True

        if len(self.vertices):
            glBindBuffer(GL_ARRAY_BUFFER, self.vboID[0])
            glBufferData(GL_ARRAY_BUFFER, len(self.vertices) * self.vertices.itemsize, (GLfloat * len(self.vertices))(*self.vertices), GL_STATIC_DRAW)         
            
        if len(self.colors):
            glBindBuffer(GL_ARRAY_BUFFER, self.vboID[1])
            glBufferData(GL_ARRAY_BUFFER, len(self.colors) * self.colors.itemsize, (GLfloat * len(self.colors))(*self.colors), GL_STATIC_DRAW)

        if len(self.normals):
            glBindBuffer(GL_ARRAY_BUFFER, self.vboID[2])
            glBufferData(GL_ARRAY_BUFFER, len(self.normals) * self.normals.itemsize, (GLfloat * len(self.normals))(*self.normals), GL_STATIC_DRAW)

        if len(self.indices):
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.vboID[3])
            glBufferData(GL_ELEMENT_ARRAY_BUFFER,len(self.indices) * self.indices.itemsize,(GLuint * len(self.indices))(*self.indices), GL_STATIC_DRAW)
            
        self.elementbufferSize = len(self.indices)

    def clear(self):        
        self.vertices = array.array('f')
        self.colors = array.array('f')
        self.normals = array.array('f')
        self.indices = array.array('I')

    def set_drawing_type(self, _type):
        self.drawing_type = _type

    def draw(self):
        if (self.elementbufferSize > 0) and self.is_init:            
            glEnableVertexAttribArray(0)
            glBindBuffer(GL_ARRAY_BUFFER, self.vboID[0])
            glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,0,None)

            glEnableVertexAttribArray(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.vboID[1])
            glVertexAttribPointer(1,4,GL_FLOAT,GL_FALSE,0,None)
            
            glEnableVertexAttribArray(2)
            glBindBuffer(GL_ARRAY_BUFFER, self.vboID[2])
            glVertexAttribPointer(2,3,GL_FLOAT,GL_FALSE,0,None)

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.vboID[3])
            glDrawElements(self.drawing_type, self.elementbufferSize, GL_UNSIGNED_INT, None)      
            
            glDisableVertexAttribArray(0)
            glDisableVertexAttribArray(1)
            glDisableVertexAttribArray(2)

IMAGE_FRAGMENT_SHADER = """
# version 330 core
in vec2 UV;
out vec4 color;
uniform sampler2D texImage;
void main() {
    vec2 scaler =vec2(UV.x,1.f - UV.y);
    vec3 rgbcolor = vec3(texture(texImage, scaler).zyx);
    vec3 color_rgb = pow(rgbcolor, vec3(1.65f));
    color = vec4(color_rgb,1.f);
}
"""

IMAGE_VERTEX_SHADER = """
# version 330
layout(location = 0) in vec3 vert;
out vec2 UV;
void main() {
    UV = (vert.xy+vec2(1.f,1.f))*.5f;
    gl_Position = vec4(vert, 1.f);
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

        glEnableVertexAttribArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, self.quad_vb)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glDisableVertexAttribArray(0)
        glBindTexture(GL_TEXTURE_2D, 0)            
        glUseProgram(0)

class GLViewer:
    """
    Class that manages input events, window and OpenGL rendering pipeline
    """
    def __init__(self):
        self.available = False
        self.objects_name = []
        self.mutex = Lock()
        # Create bone objects
        self.bones = Simple3DObject(False)
        # Create joint objects
        self.joints = Simple3DObject(False)
        self.image_handler = ImageHandler()
        # Create the rendering camera
        self.projection = array.array('f')

    def init(self, _params): 
        glutInit()
        wnd_w = glutGet(GLUT_SCREEN_WIDTH)
        wnd_h = glutGet(GLUT_SCREEN_HEIGHT)
        width = (int)(wnd_w*0.9)
        height = (int)(wnd_h*0.9)
     
        glutInitWindowSize(width, height)
        glutInitWindowPosition((int)(wnd_w*0.05), (int)(wnd_h*0.05)) # The window opens at the upper left corner of the screen
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_SRGB)
        glutCreateWindow("ZED Body Tracking")
        glViewport(0, 0, width, height)

        glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE,
                      GLUT_ACTION_CONTINUE_EXECUTION)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

        # Initialize image renderer
        self.image_handler.initialize(_params.image_size)

        glEnable(GL_FRAMEBUFFER_SRGB)

        # Compile and create the shader for 3D objects
        self.shader_sk_image = Shader(SK_VERTEX_SHADER, SK_FRAGMENT_SHADER)
        self.shader_sk_MVP = glGetUniformLocation(self.shader_sk_image.get_program_id(), "u_mvpMatrix")

        self.set_render_camera_projection(_params, 0.5, 20)
        
        self.bones.set_drawing_type(GL_QUADS)
        
        self.joints.set_drawing_type(GL_QUADS)

        self.floor_plane_set = False

        # Register the drawing function with GLUT
        glutDisplayFunc(self.draw_callback)
        # Register the function called when nothing happens
        glutIdleFunc(self.idle)   

        glutKeyboardFunc(self.keyPressedCallback)
        # Register the closing function
        glutCloseFunc(self.close_func)

        self.available = True

    def set_floor_plane_equation(self, _eq):
        self.floor_plane_set = True
        self.floor_plane_eq = _eq

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
        if _object_data.tracking_state == sl.OBJECT_TRACKING_STATE.OK or _object_data.tracking_state == sl.OBJECT_TRACKING_STATE.OFF:
            return True
        else:
            return False

    def update_view(self, _image, _objs):       # _objs of type sl.Objects
        self.mutex.acquire()

        # update image
        self.image_handler.push_new_image(_image)

        # Clear frame objects
        self.bones.clear()
        self.joints.clear()
        self.objects_name = []      

        # Only show tracked objects
        for obj in _objs.object_list:
            if self.render_object(obj):
                color_class = generate_color_id(obj.id)
                # Draw skeletons
                if obj.keypoint.size > 0:
                    # Bones
                    for bone in sl.BODY_BONES:
                        kp_1 = obj.keypoint[bone[0].value]
                        kp_2 = obj.keypoint[bone[1].value]
                        if math.isfinite(kp_1[0]) and math.isfinite(kp_2[0]):
                            self.bones.add_cylinder(kp_1, kp_2, color_class)

                    # Joints
                    for part in range(len(sl.BODY_PARTS)-1):    # -1 to avoid LAST
                        kp = obj.keypoint[part]
                        norm = np.linalg.norm(kp)
                        if math.isfinite(norm):
                            self.joints.add_sphere(kp, color_class)

        self.mutex.release()

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
            self.mutex.release()  

            glutSwapBuffers()
            glutPostRedisplay()

    def update(self):
        self.bones.push_to_GPU()
        self.joints.push_to_GPU()

    def draw(self):
        glDisable(GL_DEPTH_TEST)
        self.image_handler.draw()

        glUseProgram(self.shader_sk_image.get_program_id())
        glUniformMatrix4fv(self.shader_sk_MVP, 1, GL_TRUE,  (GLfloat * len(self.projection))(*self.projection))    
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glEnable(GL_DEPTH_TEST)
        self.bones.draw()
        self.joints.draw()
        glUseProgram(0)
