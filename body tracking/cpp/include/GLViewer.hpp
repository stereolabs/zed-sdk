#ifndef __VIEWER_INCLUDE__
#define __VIEWER_INCLUDE__

#include <vector>
#include <mutex>
#include <map>
#include <deque>
#include <vector>
#include <sl/Camera.hpp>

#include <GL/glew.h>
#include <GL/freeglut.h>

#include <cuda.h>
#include <cuda_gl_interop.h>

#ifndef M_PI
#define M_PI 3.141592653f
#endif

#define MOUSE_R_SENSITIVITY 0.03f
#define MOUSE_UZ_SENSITIVITY 0.75f
#define MOUSE_DZ_SENSITIVITY 1.25f
#define MOUSE_T_SENSITIVITY 50.f
#define KEY_T_SENSITIVITY 0.1f

using namespace sl;

const std::vector<std::pair< BODY_PARTS, BODY_PARTS>> SKELETON_BONES
{
	{
		BODY_PARTS::NOSE, BODY_PARTS::NECK
	},
	{
		BODY_PARTS::NECK, BODY_PARTS::RIGHT_SHOULDER
	},
	{
		BODY_PARTS::RIGHT_SHOULDER, BODY_PARTS::RIGHT_ELBOW
	},
	{
		BODY_PARTS::RIGHT_ELBOW, BODY_PARTS::RIGHT_WRIST
	},
	{
		BODY_PARTS::NECK, BODY_PARTS::LEFT_SHOULDER
	},
	{
		BODY_PARTS::LEFT_SHOULDER, BODY_PARTS::LEFT_ELBOW
	},
	{
		BODY_PARTS::LEFT_ELBOW, BODY_PARTS::LEFT_WRIST
	},
	{
		BODY_PARTS::RIGHT_HIP, BODY_PARTS::RIGHT_KNEE
	},
	{
		BODY_PARTS::RIGHT_KNEE, BODY_PARTS::RIGHT_ANKLE
	},
	{
		BODY_PARTS::LEFT_HIP, BODY_PARTS::LEFT_KNEE
	},
	{
		BODY_PARTS::LEFT_KNEE, BODY_PARTS::LEFT_ANKLE
	},
	{
		BODY_PARTS::RIGHT_SHOULDER, BODY_PARTS::LEFT_SHOULDER
	},
	{
		BODY_PARTS::RIGHT_HIP, BODY_PARTS::LEFT_HIP
	},
	{
		BODY_PARTS::NOSE, BODY_PARTS::RIGHT_EYE
	},
	{
		BODY_PARTS::RIGHT_EYE, BODY_PARTS::RIGHT_EAR
	},
	{
		BODY_PARTS::NOSE, BODY_PARTS::LEFT_EYE
	},
	{
		BODY_PARTS::LEFT_EYE, BODY_PARTS::LEFT_EAR
	}
};

///////////////////////////////////////////////////////////////////////////////////////////////

class Shader {
public:

	Shader() {}
	Shader(GLchar* vs, GLchar* fs);
	~Shader();
	GLuint getProgramId();

	static const GLint ATTRIB_VERTICES_POS = 0;
	static const GLint ATTRIB_COLOR_POS = 1;
	static const GLint ATTRIB_NORMAL = 2;
private:
	bool compile(GLuint &shaderId, GLenum type, GLchar* src);
	GLuint verterxId_;
	GLuint fragmentId_;
	GLuint programId_;
};

struct ShaderData {
	Shader it;
	GLuint MVP_Mat;
};

class Simple3DObject {
public:

	Simple3DObject();
	Simple3DObject(sl::Translation position, bool isStatic);
	~Simple3DObject();

	void addPt(sl::float3 pt);
	void addClr(sl::float4 clr);
	void addNormal(sl::float3 normal);
	void addPoints(std::vector<sl::float3> pts, sl::float4 base_clr);
	void addBoundingBox(std::vector<sl::float3> bbox, sl::float4 base_clr);
	void addPoint(sl::float3 pt, sl::float4 clr);
	void addLine(sl::float3 p1, sl::float3 p2, sl::float3 clr);
	void addCylinder(sl::float3 startPosition, sl::float3 endPosition, sl::float4 clr);
	void addSphere(sl::float3 position, sl::float4 clr);
	void pushToGPU();
	void clear();

	void setDrawingType(GLenum type);

	void draw();

	void translate(const sl::Translation& t);
	void setPosition(const sl::Translation& p);

	void setRT(const sl::Transform& mRT);

	void rotate(const sl::Orientation& rot);
	void rotate(const sl::Rotation& m);
	void setRotation(const sl::Orientation& rot);
	void setRotation(const sl::Rotation& m);

	const sl::Translation& getPosition() const;

	sl::Transform getModelMatrix() const;

private:
	std::vector<float> vertices_;
	std::vector<float> colors_;
	std::vector<unsigned int> indices_;
	std::vector<float> normals_;

	bool isStatic_;

	GLenum drawingType_;

	GLuint vaoID_;
	/*
	Vertex buffer IDs:
	- [0]: Vertices coordinates;
	- [1]: Colors;
	- [2]: Indices;
	- [3]: Normals
	*/
	GLuint vboID_[4];

	sl::Translation position_;
	sl::Orientation rotation_;
};

class CameraGL {
public:

	CameraGL() {
	}

	enum DIRECTION {
		UP, DOWN, LEFT, RIGHT, FORWARD, BACK
	};
	CameraGL(sl::Translation position, sl::Translation direction, sl::Translation vertical = sl::Translation(0, 1, 0)); // vertical = Eigen::Vector3f(0, 1, 0)
	~CameraGL();

	void update();
	void setProjection(float horizontalFOV, float verticalFOV, float znear, float zfar);
	const sl::Transform& getViewProjectionMatrix() const;

	float getHorizontalFOV() const;
	float getVerticalFOV() const;

	// Set an offset between the eye of the camera and its position
	// Note: Useful to use the camera as a trackball camera with z>0 and x = 0, y = 0
	// Note: coordinates are in local space
	void setOffsetFromPosition(const sl::Translation& offset);
	const sl::Translation& getOffsetFromPosition() const;

	void setDirection(const sl::Translation& direction, const sl::Translation &vertical);
	void translate(const sl::Translation& t);
	void setPosition(const sl::Translation& p);
	void rotate(const sl::Orientation& rot);
	void rotate(const sl::Rotation& m);
	void setRotation(const sl::Orientation& rot);
	void setRotation(const sl::Rotation& m);

	const sl::Translation& getPosition() const;
	const sl::Translation& getForward() const;
	const sl::Translation& getRight() const;
	const sl::Translation& getUp() const;
	const sl::Translation& getVertical() const;
	float getZNear() const;
	float getZFar() const;

	static const sl::Translation ORIGINAL_FORWARD;
	static const sl::Translation ORIGINAL_UP;
	static const sl::Translation ORIGINAL_RIGHT;

	sl::Transform projection_;
	bool usePerspective_;
private:
	void updateVectors();
	void updateView();
	void updateVPMatrix();

	sl::Translation offset_;
	sl::Translation position_;
	sl::Translation forward_;
	sl::Translation up_;
	sl::Translation right_;
	sl::Translation vertical_;

	sl::Orientation rotation_;

	sl::Transform view_;
	sl::Transform vpMatrix_;
	float horizontalFieldOfView_;
	float verticalFieldOfView_;
	float znear_;
	float zfar_;
};

class PointCloud {
public:
	PointCloud();
	~PointCloud();

	// Initialize Opengl and Cuda buffers
	// Warning: must be called in the Opengl thread
	void initialize(sl::Resolution res);
	// Push a new point cloud
	// Warning: can be called from any thread but the mutex "mutexData" must be locked
	void pushNewPC(sl::Mat &matXYZRGBA);
	// Update the Opengl buffer
	// Warning: must be called in the Opengl thread
	void update();
	// Draw the point cloud
	// Warning: must be called in the Opengl thread
	void draw(const sl::Transform& vp);
	// Close (disable update)
	void close();

private:
	sl::Mat matGPU_;
	bool hasNewPCL_ = false;
	ShaderData shader;
	size_t numBytes_;
	float* xyzrgbaMappedBuf_;
	GLuint bufferGLID_;
	cudaGraphicsResource* bufferCudaID_;
};

class ImageHandler {
public:
	ImageHandler();
	~ImageHandler();

	// Initialize Opengl and Cuda buffers
	bool initialize(sl::Resolution res);
	// Push a new Image + Z buffer and transform into a point cloud
	void pushNewImage(sl::Mat &image);
	// Draw the Image
	void draw();
	// Close (disable update)
	void close();

private:
	GLuint texID;
	GLuint imageTex;
	cudaGraphicsResource* cuda_gl_ressource;//cuda GL resource
	Shader shader;
	GLuint quad_vb;
};

struct ObjectClassName {
	sl::float3 position;
	std::string name_lineA;
	std::string name_lineB;
	sl::float4 color;
};

// This class manages input events, window and Opengl rendering pipeline
class GLViewer {
public:
	GLViewer();
	~GLViewer();
	bool isAvailable();
	void init(int argc, char **argv, sl::CameraParameters& param, bool isTrackingON, sl::BODY_FORMAT body_format);
	void updateData(sl::Mat &matXYZRGBA, std::vector<sl::ObjectData> &objs, sl::Transform& pose);
	void exit();
	void setFloorPlaneEquation(sl::float4 eq);

private:
	void render();
	void update();
	void draw();
	void clearInputs();
	void setRenderCameraProjection(sl::CameraParameters params, float znear, float zfar);

	void printText();

	// Glut functions callbacks
	static void drawCallback();
	static void mouseButtonCallback(int button, int state, int x, int y);
	static void mouseMotionCallback(int x, int y);
	static void reshapeCallback(int width, int height);
	static void keyPressedCallback(unsigned char c, int x, int y);
	static void keyReleasedCallback(unsigned char c, int x, int y);
	static void idle();

	bool available;
	bool drawBbox = false;

	enum MOUSE_BUTTON {
		LEFT = 0,
		MIDDLE = 1,
		RIGHT = 2,
		WHEEL_UP = 3,
		WHEEL_DOWN = 4
	};

	enum KEY_STATE {
		UP = 'u',
		DOWN = 'd',
		FREE = 'f'
	};

	bool mouseButton_[3];
	int mouseWheelPosition_;
	int mouseCurrentPosition_[2];
	int mouseMotion_[2];
	int previousMouseMotion_[2];
	KEY_STATE keyStates_[256];

	std::mutex mtx;
	
	ShaderData shaderLine;
	ShaderData shaderSK;

	sl::Transform projection_;
	sl::float3 bckgrnd_clr;

	std::vector<ObjectClassName> objectsName;

	PointCloud pointCloud_;
	CameraGL camera_;
	Simple3DObject skeletons;
	Simple3DObject floor_grid;

	sl::Transform cam_pose;

	bool floor_plane_set = false;
	sl::float4 floor_plane_eq;

	bool showPC = false;
	bool isTrackingON_ = false;
	sl::BODY_FORMAT body_format_ = sl::BODY_FORMAT::POSE_18;
};

#endif /* __VIEWER_INCLUDE__ */
