///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2018, STEREOLABS.
//
// All rights reserved.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
///////////////////////////////////////////////////////////////////////////

/*************************************************************************
 ** This sample shows how to find the floor plane. It's displayed as a mesh on top of      **
 * the left image using OpenGL. The detection can be triggered with the Space Bar key   **
 *************************************************************************/

// Standard includes
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// OpenGL includes
#include <GL/glew.h>
#include "GL/freeglut.h"

// ZED includes
#include <sl_zed/Camera.hpp>

// Sample includes
#include "GLObject.hpp"
#include "utils.hpp"
#include "cuda_gl_interop.h"

using namespace sl;

// ZED object (camera, mesh, pose)
Camera zed;
Mat left_image; //  to hold images
Pose pose; //  to hold pose data
Plane plane;
TRACKING_STATE tracking_state;
bool ask_for_detection;

RuntimeParameters runtime_parameters;

// For CUDA-OpenGL interoperability
cudaGraphicsResource* cuda_gl_ressource; //cuda GL resource           

// OpenGL mesh container
MeshObject mesh_object; // Opengl mesh container

// OpenGL camera projection matrix
Transform camera_projection;

// Opengl object
Shader* shader_mesh = NULL; //GLSL Shader for mesh
Shader* shader_image = NULL; //GLSL Shader for image
GLuint imageTex; //OpenGL texture mapped with a cuda array (opengl gpu interop)
GLuint shMVPMatrixLoc; //Shader variable loc
GLuint shColorLoc; //Shader variable loc
GLuint texID; //Shader variable loc (sampler/texture)
GLuint fbo = 0; //FBO
GLuint renderedTexture = 0; //Render Texture for FBO
GLuint quad_vb; //buffer for vertices/coords for image

// OpenGL Viewport size
int wWnd = 1280;
int hWnd = 720;

// Spatial Mapping status
std::chrono::high_resolution_clock::time_point t_last;

//// Sample functions
void close();
void run();
void keyPressedCallback(unsigned char c, int x, int y);
void mouseCallback(int button, int state, int x, int y);
int initGL();
void drawGL();

enum class PLANE_DETECTION {
    NONE, FLOOR, HIT
};

sl::uint2 hit_coord;
PLANE_DETECTION plane_detection = PLANE_DETECTION::NONE;

int main(int argc, char** argv) {
    // Init GLUT window
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);

    // Setup configuration parameters for the ZED    
    InitParameters parameters;
    if (argc > 1) parameters.svo_input_filename = argv[1];

    parameters.depth_mode = DEPTH_MODE_QUALITY;
    parameters.coordinate_units = UNIT_METER;
    parameters.coordinate_system = COORDINATE_SYSTEM_RIGHT_HANDED_Y_UP; // OpenGL coordinates system

    // Open the ZED
    ERROR_CODE err = zed.open(parameters);
    if (err != ERROR_CODE::SUCCESS) {
        std::cout << toString(err) << std::endl;
        zed.close();
        return -1;
    }

    wWnd = (int) zed.getResolution().width;
    hWnd = (int) zed.getResolution().height;

    // Create GLUT window
    glutInitWindowSize(wWnd, hWnd);
    glutCreateWindow("ZED Planes Detection");

    // Initialize OpenGL
    int res = initGL();
    if (res != 0) {
        std::cout << "Failed to initialize OpenGL" << std::endl;
        zed.close();
        return -1;
    }

    ask_for_detection = false;
    zed.enableTracking();

    hit_coord.x = wWnd / 2;
    hit_coord.y = hWnd / 2;

    runtime_parameters.sensing_mode = SENSING_MODE_STANDARD;
    runtime_parameters.measure3D_reference_frame = REFERENCE_FRAME_WORLD;

    t_last = std::chrono::high_resolution_clock::now();

    // Set glut callback before start
    glutKeyboardFunc(keyPressedCallback);
    glutMouseFunc(mouseCallback);
    glutDisplayFunc(run); // Callback that updates mesh data
    glutCloseFunc(close); // Close callback

    // Start the glut main loop thread
    glutMainLoop();

    return 0;
}

/**
This function close the sample (when a close event is generated)
 **/
void close() {
    left_image.free();

    if (shader_mesh) delete shader_mesh;
    if (shader_image) delete shader_image;

    zed.close();
}

/**
Update the mesh and draw image and wireframe using OpenGL
 **/
void run() {
    if (zed.grab(runtime_parameters) == SUCCESS) {
        // Retrieve image in GPU memory
        zed.retrieveImage(left_image, VIEW_LEFT, MEM_GPU);

        tracking_state = zed.getPosition(pose);

        // CUDA - OpenGL interop : copy the GPU buffer to a CUDA array mapped to the texture.
        cudaArray_t ArrIm;
        cudaGraphicsMapResources(1, &cuda_gl_ressource, 0);
        cudaGraphicsSubResourceGetMappedArray(&ArrIm, cuda_gl_ressource, 0, 0);
        cudaMemcpy2DToArray(ArrIm, 0, 0, left_image.getPtr<sl::uchar1>(MEM_GPU), left_image.getStepBytes(MEM_GPU), left_image.getPixelBytes() * left_image.getWidth(), left_image.getHeight(), cudaMemcpyDeviceToDevice);
        cudaGraphicsUnmapResources(1, &cuda_gl_ressource, 0);

        // Compute elapse time since the last call of plane detection
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - t_last).count();
        // Ask for a mesh update 
        if (ask_for_detection) {

            if (plane_detection == PLANE_DETECTION::HIT) {
                sl::ERROR_CODE plane_status = zed.findPlaneAtHit(hit_coord, plane);
                if (plane_status == sl::SUCCESS) {
                    Mesh mesh = plane.extractMesh();
                    std::vector<int> boundaries = mesh.getBoundaries();
                    mesh_object.updateMesh(mesh.vertices, mesh.triangles, boundaries);
                    mesh_object.type = plane.type;
                }
            }

            if ((duration > 500) && plane_detection == PLANE_DETECTION::FLOOR) { //if 500ms have spend since last request
                // Update pose data (used for projection of the mesh over the current image)
                sl::Transform resetTrackingFloorFrame;

                sl::ERROR_CODE plane_status = zed.findFloorPlane(plane, resetTrackingFloorFrame);
                if (plane_status == sl::SUCCESS) {
                    Mesh mesh = plane.extractMesh();
                    std::vector<int> boundaries = mesh.getBoundaries();
                    mesh_object.updateMesh(mesh.vertices, mesh.triangles, boundaries);
                    mesh_object.type = plane.type;
                }
                t_last = std::chrono::high_resolution_clock::now();
            }

            ask_for_detection = false;
        }
        // Display using OpenGL 
        drawGL();
    }

    // If SVO input is enabled, close the window and stop mapping if video reached the end
    if (zed.getSVOPosition() > 0 && zed.getSVOPosition() == zed.getSVONumberOfFrames() - 1)
        glutLeaveMainLoop();

    // Prepare next update
    glutPostRedisplay();
}

/**
Initialize OpenGL window and objects
 **/
int initGL() {

    // Init glew after window has been created
    glewInit();
    glClearColor(0.0, 0.0, 0.0, 0.0);

    // Create and Register OpenGL Texture for Image (RGBA -- 4channels)
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &imageTex);
    glBindTexture(GL_TEXTURE_2D, imageTex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, wWnd, hWnd, 0, GL_BGRA_EXT, GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);
    cudaError_t err1 = cudaGraphicsGLRegisterImage(&cuda_gl_ressource, imageTex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
    if (err1 != cudaError::cudaSuccess) return -1;

    // Create GLSL Shaders for Mesh and Image
    shader_mesh = new Shader((GLchar*) MESH_VERTEX_SHADER, (GLchar*) MESH_FRAGMENT_SHADER);
    shMVPMatrixLoc = glGetUniformLocation(shader_mesh->getProgramId(), "u_mvpMatrix");
    shColorLoc = glGetUniformLocation(shader_mesh->getProgramId(), "u_color");
    shader_image = new Shader((GLchar*) IMAGE_VERTEX_SHADER, (GLchar*) IMAGE_FRAGMENT_SHADER);
    texID = glGetUniformLocation(shader_image->getProgramId(), "texImage");

    // Create Frame Buffer for offline rendering
    // Here we render the composition of the image and the projection of the mesh on top of it in a texture (using FBO - Frame Buffer Object)
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    // Generate a render texture (which will contain the image and mesh in wireframe overlay)
    glGenTextures(1, &renderedTexture);
    glBindTexture(GL_TEXTURE_2D, renderedTexture);

    // Give an empty image to OpenGL ( the last "0" as pointer )
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, wWnd, hWnd, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

    // Set "renderedTexture" as our color attachment #0
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, renderedTexture, 0);

    // Set the list of draw buffers.
    GLenum DrawBuffers[1] = {GL_COLOR_ATTACHMENT0};
    glDrawBuffers(1, DrawBuffers);

    // Always check that our framebuffer is ok
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        std::cout << "invalid FrameBuffer" << std::endl;
        return -1;
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // Create Projection Matrix for OpenGL. We will use this matrix in combination with the Pose (on REFERENCE_FRAME_WORLD) to project the mesh on the 2D Image.
    CameraParameters camLeft = zed.getCameraInformation().calibration_parameters.left_cam;
    camera_projection(0, 0) = 1.0f / tanf(camLeft.h_fov * M_PI / 180.f * 0.5f);
    camera_projection(1, 1) = 1.0f / tanf(camLeft.v_fov * M_PI / 180.f * 0.5f);
    float znear = 0.001f;
    float zfar = 100.f;
    camera_projection(2, 2) = -(zfar + znear) / (zfar - znear);
    camera_projection(2, 3) = -(2.f * zfar * znear) / (zfar - znear);
    camera_projection(3, 2) = -1.f;
    camera_projection(0, 2) = (camLeft.image_size.width - 2.f * camLeft.cx) / camLeft.image_size.width;
    camera_projection(1, 2) = (-1.f * camLeft.image_size.height + 2.f * camLeft.cy) / camLeft.image_size.height;
    camera_projection(3, 3) = 0.f;

    // Generate the Quad for showing the image in a full viewport
    static const GLfloat g_quad_vertex_buffer_data[] = {
        -1.0f, -1.0f, 0.0f,
        1.0f, -1.0f, 0.0f,
        -1.0f, 1.0f, 0.0f,
        -1.0f, 1.0f, 0.0f,
        1.0f, -1.0f, 0.0f,
        1.0f, 1.0f, 0.0f
    };

    // Generate a buffer to handle vertices for the GLSL shader
    glGenBuffers(1, &quad_vb);
    glBindBuffer(GL_ARRAY_BUFFER, quad_vb);
    glBufferData(GL_ARRAY_BUFFER, sizeof (g_quad_vertex_buffer_data), g_quad_vertex_buffer_data, GL_STATIC_DRAW);

    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    return 0;
}

/**
This function draws a text with OpenGL
 **/
void printGL(float x, float y, const char *string) {
    glRasterPos2f(x, y);
    int len = (int) strlen(string);
    for (int i = 0; i < len; i++)
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, string[i]);
}

sl::float3 getPlaneColor(PLANE_TYPE type) {
    sl::float3 clr;
    switch (type) {
        case PLANE_TYPE_HORIZONTAL: clr = sl::float3(0.65f, 0.95f, 0.35f);
            break;
        case PLANE_TYPE_VERTICAL: clr = sl::float3(0.95f, 0.35f, 0.65f);
            break;
        case PLANE_TYPE_UNKNOWN: clr = sl::float3(0.35f, 0.65f, 0.95f);
            break;
        default: break;
    }
    return clr;
}

void drawCursorHit() {
    float cx = (hit_coord.x / (float) wWnd) * 2 - 1;
    float cy = ((hit_coord.y / (float) hWnd) * 2 - 1)*-1.f;

    float lx = 0.02f;
    float ly = lx * 16. / 9.;

    glLineWidth(1.5);
    glColor3f(0.25f, 0.55f, 0.85f);
    glBegin(GL_LINES);
    glVertex3f(cx - lx, cy, 0.0);
    glVertex3f(cx + lx, cy, 0.0);
    glVertex3f(cx, cy - ly, 0.0);
    glVertex3f(cx, cy + ly, 0.0);
    glEnd();
}

/**
OpenGL draw function
Render Image and wireframe mesh into a texture using the FrameBuffer
 **/
void drawGL() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_TEXTURE_2D);
    glActiveTexture(GL_TEXTURE0);

    glViewport(0, 0, wWnd, hWnd);

    // Render image and wireframe mesh into a texture using frame buffer
    // Bind the frame buffer and specify the viewport (full screen)
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    // Render the ZED view (Left) in the framebuffer
    glUseProgram(shader_image->getProgramId());
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, imageTex);
    glUniform1i(texID, 0);
    //invert y axis and color for this image (since its reverted from cuda array)
    glUniform1i(glGetUniformLocation(shader_image->getProgramId(), "revert"), 1);
    glUniform1i(glGetUniformLocation(shader_image->getProgramId(), "rgbflip"), 1);

    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, quad_vb);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*) 0);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glDisableVertexAttribArray(0);
    glUseProgram(0);

    // If the Positional tracking is good, we can draw the mesh over the current image
    if (tracking_state == TRACKING_STATE_OK) {
        glDisable(GL_TEXTURE_2D);
        // Send the projection and the Pose to the GLSL shader to make the projection of the 2D image.
        Transform vpMatrix = Transform::transpose(camera_projection * Transform::inverse(pose.pose_data));
        glUseProgram(shader_mesh->getProgramId());
        glUniformMatrix4fv(shMVPMatrixLoc, 1, GL_FALSE, vpMatrix.m);

        // Draw the mesh in GL_TRIANGLES with a polygon mode in line (wire)
        sl::float3 clr_plane = getPlaneColor(mesh_object.type);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        glUniform3fv(shColorLoc, 1, clr_plane.v);
        mesh_object.draw(GL_TRIANGLES);

        glLineWidth(0.5);
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        glUniform3fv(shColorLoc, 1, clr_plane.v);
        mesh_object.draw(GL_TRIANGLES);
        glUseProgram(0);
    }

    // Unbind the framebuffer since the texture is now updated
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // Render the texture to the screen
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glUseProgram(shader_image->getProgramId());
    glBindTexture(GL_TEXTURE_2D, renderedTexture);
    glUniform1i(texID, 0);
    glUniform1i(glGetUniformLocation(shader_image->getProgramId(), "revert"), 0);
    glUniform1i(glGetUniformLocation(shader_image->getProgramId(), "rgbflip"), 0);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, quad_vb);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*) 0);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glDisableVertexAttribArray(0);
    glUseProgram(0);
    glDisable(GL_TEXTURE_2D);

    drawCursorHit();

    // Show actions
    // Show mapping state if not okay
    if (tracking_state != TRACKING_STATE_OK) {
        glColor4f(0.85f, 0.25f, 0.15f, 1.f);
        std::string state_str;
        std::string positional_tracking_state_str("POSITIONAL TRACKING STATE : ");
        state_str = positional_tracking_state_str + toString(tracking_state).c_str();
        printGL(-0.99f, 0.9f, state_str.c_str());
    } else {
        glColor4f(0.82f, 0.82f, 0.82f, 1.f);
        printGL(-0.99f, 0.9f, "Press Space Bar to detect floor PLANE.");
        printGL(-0.99f, 0.85f, "Press Space 'h' to get plane at hit.");
    }

    if (zed.getCameraInformation().camera_model == MODEL_ZED_M) {
        float y_start = -0.99f;
        for (int t = 0; t < PLANE_TYPE_LAST; t++) {
            PLANE_TYPE type = static_cast<PLANE_TYPE> (t);
            sl::float3 clr = getPlaneColor(type);
            glColor4f(clr.r, clr.g, clr.b, 1.f);
            printGL(-0.99f, y_start, toString(type).c_str());
            y_start += 0.05;
        }
        glColor4f(0.22, 0.22, 0.22, 1.f);
        printGL(-0.99f, y_start, "PLANES ORIENTATION :");
    }

    // Swap buffers
    glutSwapBuffers();
}

/**
This function handles keyboard events
 **/
void keyPressedCallback(unsigned char c, int x, int y) {
    switch (c) {
        case 32:
            plane_detection = PLANE_DETECTION::FLOOR;
            ask_for_detection = true;
            break;

        case 'h':
        case 'H':
            plane_detection = PLANE_DETECTION::HIT;
            ask_for_detection = true;
            break;

        case 'q':
            glutLeaveMainLoop(); // End the process	
            break;
        default:
            break;
    }
}

/**
This function handles mouse events
 **/
void mouseCallback(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
        hit_coord.x = x;
        hit_coord.y = y;
        plane_detection = PLANE_DETECTION::HIT;
        ask_for_detection = true;
    }
}
