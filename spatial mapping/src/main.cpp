///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2017, STEREOLABS.
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
 ** This sample shows how to capture a real-time 3D reconstruction      **
 ** of the scene using the Spatial Mapping API. The resulting mesh      **
 ** is displayed as a wireframe on top of the left image using OpenGL.  **
 ** Spatial Mapping can be started and stopped with the Space Bar key   **
 *************************************************************************/

 // Standard includes
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// OpenGL includes
#include <GL/glew.h>
#include "GL/freeglut.h"

// ZED includes
#include <sl/Camera.hpp>

// Sample includes
#include "GLObject.hpp"
#include "utils.hpp"
#include "cuda_gl_interop.h"

// Define if you want to use the mesh as a set of chunks or as a global entity.
#define USE_CHUNKS 1

// ZED object (camera, mesh, pose)
sl::Camera zed;
sl::Mat left_image; // sl::Mat to hold images
sl::Pose pose;      // sl::Pose to hold pose data
sl::Mesh mesh;      // sl::Mesh to hold the mesh generated during spatial mapping
sl::SpatialMappingParameters spatial_mapping_params;
sl::MeshFilterParameters filter_params;
sl::TRACKING_STATE tracking_state;

// For CUDA-OpenGL interoperability
cudaGraphicsResource* cuda_gl_ressource;//cuda GL resource           

// OpenGL mesh container
std::vector<MeshObject> mesh_object;    // Opengl mesh container
sl::float3 vertices_color;              // Defines the color of the mesh

// OpenGL camera projection matrix
sl::Transform camera_projection;

// Opengl object
Shader* shader_mesh = NULL; //GLSL Shader for mesh
Shader* shader_image = NULL;//GLSL Shader for image
GLuint imageTex;            //OpenGL texture mapped with a cuda array (opengl gpu interop)
GLuint shMVPMatrixLoc;      //Shader variable loc
GLuint shColorLoc;          //Shader variable loc
GLuint texID;               //Shader variable loc (sampler/texture)
GLuint fbo = 0;             //FBO
GLuint renderedTexture = 0; //Render Texture for FBO
GLuint quad_vb;             //buffer for vertices/coords for image

// OpenGL Viewport size
int wWnd = 1280;
int hWnd = 720;

// Spatial Mapping status
bool mapping_is_started = false;
std::chrono::high_resolution_clock::time_point t_last;

//// Sample functions
void close();
void run();
void startMapping();
void stopMapping();
void keyPressedCallback(unsigned char c, int x, int y);
int initGL();
void drawGL();

int main(int argc, char** argv) {
    // Init GLUT window
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);

    // Setup configuration parameters for the ZED    
    sl::InitParameters parameters;
    if (argc > 1) parameters.svo_input_filename = argv[1];

    parameters.depth_mode = sl::DEPTH_MODE_PERFORMANCE; // Use QUALITY depth mode to improve mapping results
    parameters.coordinate_units = sl::UNIT_METER;
    parameters.coordinate_system = sl::COORDINATE_SYSTEM_RIGHT_HANDED_Y_UP; // OpenGL coordinates system

    // Open the ZED
    sl::ERROR_CODE err = zed.open(parameters);
    if (err != sl::ERROR_CODE::SUCCESS) {
        std::cout << sl::errorCode2str(err) << std::endl;
        zed.close();
        return -1;
    }

    wWnd = (int) zed.getResolution().width;
    hWnd = (int) zed.getResolution().height;

    // Create GLUT window
    glutInitWindowSize(wWnd, hWnd);
    glutCreateWindow("ZED Spatial Mapping");

    // Configure Spatial Mapping and filtering parameters
    spatial_mapping_params.range_meter.second = sl::SpatialMappingParameters::get(sl::SpatialMappingParameters::RANGE_FAR);
    spatial_mapping_params.resolution_meter = sl::SpatialMappingParameters::get(sl::SpatialMappingParameters::RESOLUTION_LOW);
    spatial_mapping_params.save_texture = false;
    spatial_mapping_params.max_memory_usage = 512;
    spatial_mapping_params.keep_mesh_consistent = !USE_CHUNKS; // If we use chunks we do not need to keep the mesh consistent

    filter_params.set(sl::MeshFilterParameters::FILTER_LOW);

    // Initialize OpenGL
    int res = initGL();
    if (res != 0) {
        std::cout << "Failed to initialize OpenGL" << std::endl;
        zed.close();
        return -1;
    }

    std::cout << "*************************************************************" << std::endl;
    std::cout << "**      Press the Space Bar key to start and stop          **" << std::endl;
    std::cout << "*************************************************************" << std::endl;

    // Set glut callback before start
    glutKeyboardFunc(keyPressedCallback);// Callback that starts spatial mapping when space bar is pressed
    glutDisplayFunc(run); // Callback that updates mesh data
    glutCloseFunc(close);// Close callback

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

    mesh_object.clear();
    zed.close();
}

/**
Start the spatial mapping process
**/
void startMapping() {
    // clear previously used objects
    mesh.clear();
    mesh_object.clear();

#if !USE_CHUNKS
    // Create only one object that will contain the full mesh.
    // Otherwise, different MeshObject will be created for each chunk when needed
    mesh_object.emplace_back();
#endif

    // Enable positional tracking before starting spatial mapping
    zed.enableTracking();
    // Enable spatial mapping
    zed.enableSpatialMapping(spatial_mapping_params);

    // Start a timer, we retrieve the mesh every XXms.
    t_last = std::chrono::high_resolution_clock::now();

    mapping_is_started = true;
    std::cout << "** Spatial Mapping is started ... **" << std::endl;
    return;
}

/**
Stop the spatial mapping process
**/
void stopMapping() {
    // Stop the mesh request and extract the whole mesh to filter it and save it as an obj file
    mapping_is_started = false;
    std::cout << "** Stop Spatial Mapping ... **" << std::endl;

    // Extract the whole mesh
    sl::Mesh wholeMesh;
    zed.extractWholeMesh(wholeMesh);
    std::cout << ">> Mesh has been extracted..." << std::endl;

    // Filter the extracted mesh
    wholeMesh.filter(filter_params, !USE_CHUNKS);
    std::cout << ">> Mesh has been filtered..." << std::endl;

    // If textures have been saved during spatial mapping, apply them to the mesh
    if (spatial_mapping_params.save_texture) {
        wholeMesh.applyTexture(sl::MESH_TEXTURE_RGB);
        std::cout << ">> Mesh has been textured..." << std::endl;
    }

    //Save as an OBJ file
    std::string saveName = getDir() + "mesh_gen.obj";
    bool t = wholeMesh.save(saveName.c_str());
    if (t) std::cout << ">> Mesh has been saved under " << saveName << std::endl;
    else std::cout << ">> Failed to save the mesh under  " << saveName << std::endl;

    // Update the displayed Mesh
    mesh_object.clear();
    mesh_object.resize(wholeMesh.chunks.size());
    for (int c = 0; c < wholeMesh.chunks.size(); c++)
        mesh_object[c].updateMesh(wholeMesh.chunks[c].vertices, wholeMesh.chunks[c].triangles);
    return;
}

/**
Update the mesh and draw image and wireframe using OpenGL
**/
void run() {
    if (zed.grab() == sl::SUCCESS) {
        // Retrieve image in GPU memory
        zed.retrieveImage(left_image, sl::VIEW_LEFT, sl::MEM_GPU);

        // CUDA - OpenGL interop : copy the GPU buffer to a CUDA array mapped to the texture.
        cudaArray_t ArrIm;
        cudaGraphicsMapResources(1, &cuda_gl_ressource, 0);
        cudaGraphicsSubResourceGetMappedArray(&ArrIm, cuda_gl_ressource, 0, 0);
        cudaMemcpy2DToArray(ArrIm, 0, 0, left_image.getPtr<sl::uchar1>(sl::MEM_GPU), left_image.getStepBytes(sl::MEM_GPU), left_image.getPixelBytes()*left_image.getWidth(), left_image.getHeight(), cudaMemcpyDeviceToDevice);
        cudaGraphicsUnmapResources(1, &cuda_gl_ressource, 0);

        // Update pose data (used for projection of the mesh over the current image)
        tracking_state = zed.getPosition(pose);

        if (mapping_is_started) {

            // Compute elapse time since the last call of sl::Camera::requestMeshAsync()
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - t_last).count();
            // Ask for a mesh update if 500ms have spend since last request
            if (duration > 500) {
                zed.requestMeshAsync();
                t_last = std::chrono::high_resolution_clock::now();
            }

            if (zed.getMeshRequestStatusAsync() == sl::SUCCESS) {

                // Get the current mesh generated and send it to opengl
                if (zed.retrieveMeshAsync(mesh) == sl::SUCCESS) {
#if USE_CHUNKS
                    for (int c = 0; c < mesh.chunks.size(); c++) {
                        // If the chunk does not exist in the rendering process -> add it in the rendering list
                        if (mesh_object.size() < mesh.chunks.size()) mesh_object.emplace_back();
                        // If the chunck has been updated by the spatial mapping, update it for rendering
                        if (mesh.chunks[c].has_been_updated)
                            mesh_object[c].updateMesh(mesh.chunks[c].vertices, mesh.chunks[c].triangles);
                    }
#else
                    mesh_object[0].updateMesh(mesh.vertices, mesh.triangles);
#endif
                }
            }
        }

        // Display image and mesh using OpenGL 
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
    sl::CameraParameters camLeft = zed.getCameraInformation().calibration_parameters.left_cam;
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
        1.0f, 1.0f, 0.0f};

    // Color of wireframe (soft blue)
    vertices_color.r = 0.35f;
    vertices_color.g = 0.65f;
    vertices_color.b = 0.95f;

    // Generate a buffer to handle vertices for the GLSL shader
    glGenBuffers(1, &quad_vb);
    glBindBuffer(GL_ARRAY_BUFFER, quad_vb);
    glBufferData(GL_ARRAY_BUFFER, sizeof(g_quad_vertex_buffer_data), g_quad_vertex_buffer_data, GL_STATIC_DRAW);

    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);

    return 0;
}

/**
This function draws a text with OpenGL
**/
void printGL(float x, float y, const char *string) {
    glRasterPos2f(x, y);
    int len = (int) strlen(string);
    for (int i = 0; i < len; i++) {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, string[i]);
    }
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
    if (tracking_state == sl::TRACKING_STATE_OK && mesh_object.size()) {
        glDisable(GL_TEXTURE_2D);
        // Send the projection and the Pose to the GLSL shader to make the projection of the 2D image.
        sl::Transform vpMatrix = sl::Transform::transpose(camera_projection * sl::Transform::inverse(pose.pose_data));
        glUseProgram(shader_mesh->getProgramId());
        glUniformMatrix4fv(shMVPMatrixLoc, 1, GL_FALSE, vpMatrix.m);

        glUniform3fv(shColorLoc, 1, vertices_color.v);
        // Draw the mesh in GL_TRIANGLES with a polygon mode in line (wire)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

#if USE_CHUNKS
        for (int c = 0; c < mesh.chunks.size(); c++)
            mesh_object[c].draw(GL_TRIANGLES);
#else
        mesh_object[0].draw(GL_TRIANGLES);
#endif

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

    // Show actions
    glColor4f(0.25f, 0.99f, 0.25f, 1.f);
    if (!mapping_is_started)
        printGL(-0.99f, 0.9f, "* Press Space Bar to activate Spatial Mapping.");
    else
        printGL(-0.99f, 0.9f, "* Press Space Bar to stop spatial mapping.");

    // Show mapping state
    if ((tracking_state == sl::TRACKING_STATE_OK)) {
        sl::SPATIAL_MAPPING_STATE state = zed.getSpatialMappingState();
        if (state == sl::SPATIAL_MAPPING_STATE_OK || state == sl::SPATIAL_MAPPING_STATE_INITIALIZING)
            glColor4f(0.25f, 0.99f, 0.25f, 1.f);
        else if (state == sl::SPATIAL_MAPPING_STATE_NOT_ENABLED)
            glColor4f(0.55f, 0.65f, 0.55f, 1.f);
        else
            glColor4f(0.95f, 0.25f, 0.25f, 1.f);
        printGL(-0.99f, 0.83f, (std::string("** ") + sl::spatialMappingState2str(state)).c_str());
    } else {
        if (mapping_is_started) {
            glColor4f(0.95f, 0.25f, 0.25f, 1.f);
            printGL(-0.99f, 0.83f, (std::string("** ") + sl::trackingState2str(tracking_state)).c_str());
        } else {
            glColor4f(0.55f, 0.65f, 0.55f, 1.f);
            printGL(-0.99f, 0.83f, (std::string("** ") + sl::spatialMappingState2str(sl::SPATIAL_MAPPING_STATE_NOT_ENABLED)).c_str());
        }
    }

    // Swap buffers
    glutSwapBuffers();
}

/**
This function handles keyboard events (especially space bar to start the mapping)
**/
void keyPressedCallback(unsigned char c, int x, int y) {
    switch (c) {
        case 32: // Space bar id	
        if (!mapping_is_started) // User press the space bar and spatial mapping is not started 
            startMapping();
        else // User press the space bar and spatial mapping is started 
            stopMapping();
        break;
        case 'q':
        glutLeaveMainLoop(); // End the process	
        break;
        default:
        break;
    }
}
