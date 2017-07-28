#ifndef __UTILS_HPP__
#define __UTILS_HPP__

////Shaders GLSL for image and mesh //////////

const char* MESH_VERTEX_SHADER =
"#version 330 core\n"
"layout(location = 0) in vec3 in_Vertex;\n"
"uniform mat4 u_mvpMatrix;\n"
"uniform vec3 u_color;\n"
"out vec3 b_color;\n"
"void main() {\n"
"   b_color = u_color;\n"
"   gl_Position = u_mvpMatrix * vec4(in_Vertex, 1);\n"
"}";

const char* MESH_FRAGMENT_SHADER =
"#version 330 core\n"
"in vec3 b_color;\n"
"layout(location = 0) out vec4 color;\n"
"void main() {\n"
"   color = vec4(b_color,1);\n"
"}";


const char* IMAGE_FRAGMENT_SHADER =
"#version 330 core\n"
" in vec2 UV;\n"
" out vec4 color;\n"
" uniform sampler2D texImage;\n"
" uniform bool revert;\n"
" uniform bool rgbflip;\n"
" void main() {\n"
"    vec2 scaler  =revert?vec2(UV.x,1.f - UV.y):vec2(UV.x,UV.y);\n"
"    vec3 rgbcolor = rgbflip?vec3(texture(texImage, scaler).zyx):vec3(texture(texImage, scaler).xyz);\n"
"    color = vec4(rgbcolor,1);\n"
"}";

const char* IMAGE_VERTEX_SHADER =
"#version 330\n"
"layout(location = 0) in vec3 vert;\n"
"out vec2 UV;"
"void main() {\n"
"   UV = (vert.xy+vec2(1,1))/2;\n"
"	gl_Position = vec4(vert, 1);\n"
"}\n";


#ifndef M_PI
#define M_PI 3.141592653f
#endif

#if _WIN32
#include <windows.h>
#include <iostream>
#include <shlobj.h>
#pragma comment(lib, "shell32.lib")
#endif 

/* Find MyDocuments directory for windows platforms.*/
std::string getDir() {
    std::string myDir;
#if _WIN32
    CHAR my_documents[MAX_PATH];
    HRESULT result = SHGetFolderPath(NULL, CSIDL_PERSONAL, NULL, SHGFP_TYPE_CURRENT, my_documents);
    if (result == S_OK)
        myDir = std::string(my_documents) + '/';
#else
    myDir = "./";
#endif
    return myDir;
}

#endif
