#include "GLViewer.hpp"
#include <random>

#if defined(_DEBUG) && defined(_WIN32)
#error "This sample should not be built in Debug mode, use RelWithDebInfo if you want to do step by step."
#endif

GLchar* VERTEX_SHADER =
"#version 330 core\n"
"layout(location = 0) in vec3 in_Vertex;\n"
"layout(location = 1) in vec4 in_Color;\n"
"uniform mat4 u_mvpMatrix;\n"
"out vec4 b_color;\n"
"void main() {\n"
"   b_color = in_Color;\n"
"	gl_Position = u_mvpMatrix * vec4(in_Vertex, 1);\n"
"}";

GLchar* FRAGMENT_SHADER =
"#version 330 core\n"
"in vec4 b_color;\n"
"layout(location = 0) out vec4 out_Color;\n"
"void main() {\n"
" float gamma = 2.2;\n"
"   out_Color = b_color;//pow(b_color, vec4(1.0/gamma));;\n"
"}";

GLchar* SK_VERTEX_SHADER =
"#version 330 core\n"
"layout(location = 0) in vec3 in_Vertex;\n"
"layout(location = 1) in vec4 in_Color;\n"
"layout(location = 2) in vec3 in_Normal;\n"
"out vec4 b_color;\n"
"out vec3 b_position;\n"
"out vec3 b_normal;\n"
"uniform mat4 u_mvpMatrix;\n"
"uniform vec4 u_color;\n"
"void main() {\n"
"   b_color = in_Color;\n"
"   b_position = in_Vertex;\n"
"   b_normal = in_Normal;\n"
"	gl_Position =  u_mvpMatrix * vec4(in_Vertex, 1);\n"
"}";

GLchar* SK_FRAGMENT_SHADER =
"#version 330 core\n"
"in vec4 b_color;\n"
"in vec3 b_position;\n"
"in vec3 b_normal;\n"
"out vec4 out_Color;\n"
"void main() {\n"
"	vec3 lightPosition = vec3(0, 10, 0);\n"
"	vec3 lightColor = vec3(1,1,1);\n"
"	float ambientStrength = 0.3;\n"
"	vec3 ambient = ambientStrength * lightColor;\n"
"	vec3 lightDir = normalize(lightPosition - b_position);\n"
"	float diffuse = (1 - ambientStrength) * max(dot(b_normal, lightDir), 0.0);\n"
"   out_Color = vec4(b_color.rgb * (diffuse + ambient), 1);\n"
"}";


GLViewer* currentInstance_ = nullptr;

float const colors[5][3] = {
	{.231f, .909f, .69f},
	{.098f, .686f, .816f},
	{.412f, .4f, .804f},
	{1, .725f, .0f},
	{.989f, .388f, .419f}
};

inline sl::float4 generateColorClass(int idx) {
	int const offset = std::max(0 ,idx % 5);
	return  sl::float4(colors[offset][0], colors[offset][1], colors[offset][2], 1.f);
}

GLViewer::GLViewer() : available(false) {
	currentInstance_ = this;
	clearInputs();
}

GLViewer::~GLViewer() {}

void GLViewer::exit() {
	if (currentInstance_) {
		image_handler.close();
		available = false;
	}
}

bool GLViewer::isAvailable() {
	if (available)
		glutMainLoopEvent();
	return available;
}

void CloseFunc(void) { if (currentInstance_) currentInstance_->exit(); }

void GLViewer::init(int argc, char **argv, sl::CameraParameters param) {

	glutInit(&argc, argv);
	int wnd_w = glutGet(GLUT_SCREEN_WIDTH);
	int wnd_h = glutGet(GLUT_SCREEN_HEIGHT);
	int width = wnd_w * 0.9;
	int height = wnd_h * 0.9;
	if (width > param.image_size.width && height > param.image_size.height) {
		width = param.image_size.width;
		height = param.image_size.height;
	}

	glutInitWindowSize(width, height);
	glutInitWindowPosition(wnd_w*0.05, wnd_h*0.05);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_SRGB);
	glutCreateWindow("ZED Body Tracking Viewer");
	glViewport(0, 0, width, height);

	GLenum err = glewInit();
	if (GLEW_OK != err)
		std::cout << "ERROR: glewInit failed: " << glewGetErrorString(err) << "\n";

	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glEnable(GL_LINE_SMOOTH);
	glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);

	bool status_ = image_handler.initialize(param.image_size);
	if (!status_)
		std::cout << "ERROR: Failed to initialized Image Renderer" << std::endl;

	glEnable(GL_FRAMEBUFFER_SRGB);

	// Compile and create the shader for 3D objects
	shaderBasic.it = Shader(VERTEX_SHADER, FRAGMENT_SHADER);
	shaderBasic.MVP_Mat = glGetUniformLocation(shaderBasic.it.getProgramId(), "u_mvpMatrix");

	shaderSK.it = Shader(SK_VERTEX_SHADER, SK_FRAGMENT_SHADER);
	shaderSK.MVP_Mat = glGetUniformLocation(shaderSK.it.getProgramId(), "u_mvpMatrix");

	// Create the rendering camera
	setRenderCameraProjection(param, 0.5f, 20);

	// Create the bounding box object
	BBox_obj.init();
	BBox_obj.setDrawingType(GL_QUADS);

	bones.init();
	bones.setDrawingType(GL_QUADS);

	joints.init();
	joints.setDrawingType(GL_QUADS);

	floor_plane_set = false;
	// Set background color (black)
	bckgrnd_clr = sl::float3(0, 0, 0);

	// Set OpenGL settings
	glDisable(GL_DEPTH_TEST); //avoid occlusion with bbox

	// Map glut function on this class methods
	glutDisplayFunc(GLViewer::drawCallback);
	glutReshapeFunc(GLViewer::reshapeCallback);
	glutKeyboardFunc(GLViewer::keyPressedCallback);
	glutKeyboardUpFunc(GLViewer::keyReleasedCallback);
	glutCloseFunc(CloseFunc);
	available = true;
}

void GLViewer::setRenderCameraProjection(sl::CameraParameters params, float znear, float zfar) {
	// Just slightly up the ZED camera FOV to make a small black border
	float fov_y = (params.v_fov + 0.5f) * M_PI / 180.f;
	float fov_x = (params.h_fov + 0.5f) * M_PI / 180.f;

	projection_(0, 0) = 1.0f / tanf(fov_x * 0.5f);
	projection_(1, 1) = 1.0f / tanf(fov_y * 0.5f);
	projection_(2, 2) = -(zfar + znear) / (zfar - znear);
	projection_(3, 2) = -1;
	projection_(2, 3) = -(2.f * zfar * znear) / (zfar - znear);
	projection_(3, 3) = 0;

	projection_(0, 0) = 1.0f / tanf(fov_x * 0.5f); //Horizontal FoV.
	projection_(0, 1) = 0;
	projection_(0, 2) = 2.0f * ((params.image_size.width - 1.0f * params.cx) / params.image_size.width) - 1.0f; //Horizontal offset.
	projection_(0, 3) = 0;

	projection_(1, 0) = 0;
	projection_(1, 1) = 1.0f / tanf(fov_y * 0.5f); //Vertical FoV.
	projection_(1, 2) = -(2.0f * ((params.image_size.height - 1.0f * params.cy) / params.image_size.height) - 1.0f); //Vertical offset.
	projection_(1, 3) = 0;

	projection_(2, 0) = 0;
	projection_(2, 1) = 0;
	projection_(2, 2) = -(zfar + znear) / (zfar - znear); //Near and far planes.
	projection_(2, 3) = -(2.0f * zfar * znear) / (zfar - znear); //Near and far planes.

	projection_(3, 0) = 0;
	projection_(3, 1) = 0;
	projection_(3, 2) = -1;
	projection_(3, 3) = 0.0f;
}

void GLViewer::render() {
	if (available) {
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glClearColor(bckgrnd_clr.r, bckgrnd_clr.g, bckgrnd_clr.b, 1.f);
		mtx.lock();
		update();
		draw();
		printText();
		mtx.unlock();
		glutSwapBuffers();
		glutPostRedisplay();
	}
}

void GLViewer::setFloorPlaneEquation(sl::float4 eq) {
	floor_plane_set = true;
	floor_plane_eq = eq;
}

inline bool renderObject(const sl::ObjectData& i) {
	return (i.tracking_state == sl::OBJECT_TRACKING_STATE::OK || i.tracking_state == sl::OBJECT_TRACKING_STATE::OFF);
}

void GLViewer::updateView(sl::Mat image, sl::Objects &objects) {
	mtx.lock();
	// Update Image
	image_handler.pushNewImage(image);

	// Clear frames object
	BBox_obj.clear();
	objectsName.clear();
	bones.clear();
	joints.clear();

	// For each object
    for (auto i = objects.object_list.rbegin(); i != objects.object_list.rend(); ++i) {
        sl::ObjectData& obj = (*i);

		// Only show tracked objects
		if (renderObject(obj)) {
			auto clr_id = generateColorClass(obj.id);

			// draw skeletons
			if (obj.keypoint.size()) {
				for (auto& limb : sl::BODY_BONES) {
					sl::float3 kp_1 = obj.keypoint[getIdx(limb.first)];
					sl::float3 kp_2 = obj.keypoint[getIdx(limb.second)];
					float norm_1 = kp_1.norm();
					float norm_2 = kp_2.norm();
					// draw cylinder between two keypoints
					if (std::isfinite(norm_1) && std::isfinite(norm_2)) {
						bones.addCylinder(kp_1, kp_2, clr_id);
					}
				}
				for (int i = 0; i < static_cast<int>(sl::BODY_PARTS::LAST); i++) {
					sl::float3 kp = obj.keypoint[i];
					if (std::isfinite(kp.norm())) joints.addSphere(kp, clr_id);
				}
			}

			// Draw Labels
			if (drawBbox) {
				auto &bb_ = obj.bounding_box;
				if(bb_.size() > 0) {
					if (floor_plane_set) {// expand bounding box to the floor
						for (int i = 4; i < 8; i++)
							bb_[i].y = (floor_plane_eq.x * bb_[i].x + floor_plane_eq.z * bb_[i].z + floor_plane_eq.w) / (floor_plane_eq.y * -1.f);
					}
					BBox_obj.addBoundingBox(bb_, clr_id);

					objectsName.emplace_back();
					objectsName.back().name_lineA = "ID : " + std::to_string(obj.id);
					std::stringstream ss_vel;
					ss_vel << std::fixed << std::setprecision(1) << obj.velocity.norm();
					objectsName.back().name_lineB = ss_vel.str() + " m/s";
					objectsName.back().color = clr_id;
					objectsName.back().position = obj.position;
					objectsName.back().position.y = (bb_[0].y + bb_[1].y + bb_[2].y + bb_[3].y) / 4.f + 0.2f;
				}
			}
		}
	}

	mtx.unlock();
}

void GLViewer::update() {
	if (keyStates_['q'] == KEY_STATE::UP || keyStates_['Q'] == KEY_STATE::UP || keyStates_[27] == KEY_STATE::UP)
		currentInstance_->exit();

	if (keyStates_['b'] == KEY_STATE::UP || keyStates_['B'] == KEY_STATE::UP)
		currentInstance_->drawBbox = !currentInstance_->drawBbox;

	// Update BBox
	BBox_obj.pushToGPU();
	bones.pushToGPU();
	joints.pushToGPU();

	//Clear inputs
	clearInputs();
}

void GLViewer::draw() {
	glDisable(GL_DEPTH_TEST);
	image_handler.draw();

	glUseProgram(shaderSK.it.getProgramId());
	glUniformMatrix4fv(shaderSK.MVP_Mat, 1, GL_TRUE, projection_.m);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glEnable(GL_DEPTH_TEST);
	bones.draw();
	joints.draw();
	glUseProgram(0);

	glUseProgram(shaderBasic.it.getProgramId());
	glUniformMatrix4fv(shaderBasic.MVP_Mat, 1, GL_TRUE, projection_.m);

	BBox_obj.draw();
	glUseProgram(0);
}

sl::float2 compute3Dprojection(sl::float3 &pt, const sl::Transform &cam, sl::Resolution wnd_size) {
	sl::float4 pt4d(pt.x, pt.y, pt.z, 1.);
	auto proj3D_cam = pt4d * cam;
	sl::float2 proj2D;
	proj2D.x = ((proj3D_cam.x / pt4d.w) * wnd_size.width) / (2.f * proj3D_cam.w) + wnd_size.width / 2.f;
	proj2D.y = ((proj3D_cam.y / pt4d.w) * wnd_size.height) / (2.f * proj3D_cam.w) + wnd_size.height / 2.f;
	return proj2D;
}

void GLViewer::printText() {
	glDisable(GL_BLEND);
	sl::Resolution wnd_size(glutGet(GLUT_WINDOW_WIDTH), glutGet(GLUT_WINDOW_HEIGHT));
	for (auto &it : objectsName) {
		auto pt2d = compute3Dprojection(it.position, projection_, wnd_size);
		glColor4f(it.color.r, it.color.g, it.color.b, .85f);
		const auto *string = it.name_lineA.c_str();
		glWindowPos2f(pt2d.x - 40, pt2d.y + 20);
		int len = (int)strlen(string);
		for (int i = 0; i < len; i++)
			glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, string[i]);

		string = it.name_lineB.c_str();
		glWindowPos2f(pt2d.x - 40, pt2d.y);
		len = (int)strlen(string);
		for (int i = 0; i < len; i++)
			glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, string[i]);
	}
	glEnable(GL_BLEND);
}

void GLViewer::clearInputs() {
	for (unsigned int i = 0; i < 256; ++i)
		keyStates_[i] = KEY_STATE::FREE;
}

void GLViewer::drawCallback() {
	currentInstance_->render();
}

void GLViewer::reshapeCallback(int width, int height) {
	glViewport(0, 0, width, height);
}

void GLViewer::keyPressedCallback(unsigned char c, int x, int y) {
	currentInstance_->keyStates_[c] = KEY_STATE::DOWN;
	glutPostRedisplay();
}

void GLViewer::keyReleasedCallback(unsigned char c, int x, int y) {
	currentInstance_->keyStates_[c] = KEY_STATE::UP;
}

void GLViewer::idle() {
	glutPostRedisplay();
}

Simple3DObject::Simple3DObject() {
	is_init = can_draw = false;
}

Simple3DObject::~Simple3DObject() {
	if (vaoID_ != 0) {
		glDeleteBuffers(4, vboID_);
		glDeleteVertexArrays(1, &vaoID_);
		vaoID_ = 0;
		is_init = can_draw = false;
	}
}

bool Simple3DObject::isInit()
{
	return is_init;
}

void Simple3DObject::init() {
	vaoID_ = 0;
	isStatic_ = false;
	drawingType_ = GL_TRIANGLES;
	rotation_.setIdentity();
	is_init = true;
}

void Simple3DObject::addPt(sl::float3 pt) {
	vertices_.push_back(pt.x);
	vertices_.push_back(pt.y);
	vertices_.push_back(pt.z);
}

void Simple3DObject::addClr(sl::float4 clr) {
	colors_.push_back(clr.r);
	colors_.push_back(clr.g);
	colors_.push_back(clr.b);
	colors_.push_back(clr.a);
}

void Simple3DObject::addNormal(sl::float3 normal) {
	normals_.push_back(normal.x);
	normals_.push_back(normal.y);
	normals_.push_back(normal.z);
}
void Simple3DObject::addPoints(std::vector<sl::float3> pts, sl::float4 base_clr)
{
	for (int k = 0; k < pts.size(); k++) {
		sl::float3 pt = pts.at(k);
		vertices_.push_back(pt.x);
		vertices_.push_back(pt.y);
		vertices_.push_back(pt.z);
		colors_.push_back(base_clr.r);
		colors_.push_back(base_clr.g);
		colors_.push_back(base_clr.b);
		colors_.push_back(1.f);
		int current_size_index = (vertices_.size() / 3 - 1);
		indices_.push_back(current_size_index);
		indices_.push_back(current_size_index + 1);
	}
}

void Simple3DObject::addPoint(sl::float3 pt, sl::float4 clr) {
	addPt(pt);
	addClr(clr);
	indices_.push_back((int)indices_.size());
}

const std::vector<int> boxLinks = { 0,1,5,4,1,2,6,5,2,3,7,6,3,0,4,7 };

void Simple3DObject::addBoundingBox(std::vector<sl::float3> bbox, sl::float4 base_clr) {

	int start_id = vertices_.size() / 3;

	float ratio = 1.f / 5.f;

	std::vector<sl::float3> bbox_;
	// generate TOP BOX
	for (int i = 0; i < 4; i++)
		bbox_.push_back(bbox[i]);
	// generate TOP BOX FADE
	for (int i = 4; i < 8; i++) {
		auto midd = bbox[i - 4] - (bbox[i - 4] - bbox[i]) * ratio;
		bbox_.push_back(midd);
	}

	// generate BOTTOM FADE
	for (int i = 4; i < 8; i++) {
		auto midd = bbox[i] + (bbox[i - 4] - bbox[i]) * ratio;
		bbox_.push_back(midd);
	}
	// generate BOTTOM BOX
	for (int i = 4; i < 8; i++)
		bbox_.push_back(bbox[i]);

	for (int i = 0; i < bbox_.size(); i++) {
		vertices_.push_back(bbox_[i].x);
		vertices_.push_back(bbox_[i].y);
		vertices_.push_back(bbox_[i].z);

		colors_.push_back(base_clr.r);
		colors_.push_back(base_clr.g);
		colors_.push_back(base_clr.b);
		colors_.push_back(((i > 3) && (i < 12)) ? 0 : base_clr.a); //fading
	}

	for (int i = 0; i < boxLinks.size(); i += 4) {
		indices_.push_back(start_id + boxLinks[i]);
		indices_.push_back(start_id + boxLinks[i + 1]);
		indices_.push_back(start_id + boxLinks[i + 2]);
		indices_.push_back(start_id + boxLinks[i + 3]);
	}

	for (int i = 0; i < boxLinks.size(); i += 4) {
		indices_.push_back(start_id + 8 + boxLinks[i]);
		indices_.push_back(start_id + 8 + boxLinks[i + 1]);
		indices_.push_back(start_id + 8 + boxLinks[i + 2]);
		indices_.push_back(start_id + 8 + boxLinks[i + 3]);
	}
}

void Simple3DObject::addLine(sl::float3 p1, sl::float3 p2, sl::float3 clr) {
	vertices_.push_back(p1.x);
	vertices_.push_back(p1.y);
	vertices_.push_back(p1.z);

	vertices_.push_back(p2.x);
	vertices_.push_back(p2.y);
	vertices_.push_back(p2.z);

	colors_.push_back(clr.r);
	colors_.push_back(clr.g);
	colors_.push_back(clr.b);
	colors_.push_back(1.f);

	colors_.push_back(clr.r);
	colors_.push_back(clr.g);
	colors_.push_back(clr.b);
	colors_.push_back(1.f);

	indices_.push_back((int)indices_.size());
	indices_.push_back((int)indices_.size());
}

void Simple3DObject::addCylinder(sl::float3 startPosition, sl::float3 endPosition, sl::float4 clr) {
	const float m_radius = 0.010f;

	sl::float3 dir = endPosition - startPosition;
	float m_height = dir.norm();
	dir = dir / m_height;

	sl::float3 yAxis(0, 1, 0);
	sl::float3 v = sl::float3::cross(dir, yAxis);
	sl::Transform rotation;

	if (v.norm() < 0.00001f)
		rotation.setIdentity();
	else {
		float cosTheta = sl::float3::dot(dir, yAxis);
		float scale = (1.f - cosTheta) / (1.f - (cosTheta * cosTheta));

		float data[] = { 0    , v[2] , -v[1], 0,
						-v[2], 0    , v[0] , 0,
						v[1] , -v[0], 0    , 0,
						0    , 0    , 0    , 1.f };

		sl::Transform vx = sl::Transform(data);
		rotation.setIdentity();
		rotation = rotation + vx;
		rotation = rotation + vx * vx * scale;
	}

	/////////////////////////////

	sl::float3 v1;
	sl::float3 v2;
	sl::float3 v3;
	sl::float3 v4;
	sl::float3 normal;

	const int NB_SEG = 32;
	const float scale_seg = 1.f / NB_SEG;
	auto rot = rotation.getRotationMatrix();
	for(int j = 0; j < NB_SEG; j++){
		float i = 2.f * M_PI * (j * scale_seg);
		float i1 = 2.f * M_PI * ((j + 1) * scale_seg);
		v1 = sl::float3(m_radius * cos(i), 0, m_radius * sin(i)) * rot + startPosition;
		v2 = sl::float3(m_radius * cos(i), m_height, m_radius * sin(i)) * rot + startPosition;
		v3 = sl::float3(m_radius * cos(i1), 0, m_radius * sin(i1)) * rot + startPosition;
		v4 = sl::float3(m_radius * cos(i1), m_height, m_radius * sin(i1)) * rot + startPosition;

		addPoint(v1, clr);
		addPoint(v2, clr);
		addPoint(v4, clr);
		addPoint(v3, clr);

		normal = sl::float3::cross((v2 - v1), (v3 - v1));
		normal = normal / normal.norm();

		addNormal(normal);
		addNormal(normal);
		addNormal(normal);
		addNormal(normal);
	}
}

void Simple3DObject::addSphere(sl::float3 position, sl::float4 clr) {
	const float m_radius = 0.02f;
	const int m_stackCount = 16;
	const int m_sectorCount = 16;

	sl::float3 point;
	sl::float3 normal;

	int i, j;
	for (i = 0; i <= m_stackCount; i++) {
		double lat0 = M_PI * (-0.5 + (double)(i - 1) / m_stackCount);
		double z0 = sin(lat0);
		double zr0 = cos(lat0);

		double lat1 = M_PI * (-0.5 + (double)i / m_stackCount);
		double z1 = sin(lat1);
		double zr1 = cos(lat1);
		for (j = 0; j <= m_sectorCount - 1; j++) {
			double lng = 2 * M_PI * (double)(j - 1) / m_sectorCount;
			double x = cos(lng);
			double y = sin(lng);

			point = sl::float3(m_radius * x * zr0, m_radius * y * zr0, m_radius * z0) + position;
			normal = sl::float3(x * zr0, y * zr0, z0);
			normal = normal / normal.norm();
			addPoint(point, clr);
			addNormal(normal);

			point = sl::float3(m_radius * x * zr1, m_radius * y * zr1, m_radius * z1) + position;
			normal = sl::float3(x * zr1, y * zr1, z1);
			normal = normal / normal.norm();
			addPoint(point, clr);
			addNormal(normal);

			lng = 2 * M_PI * (double)(j) / m_sectorCount;
			x = cos(lng);
			y = sin(lng);

			point = sl::float3(m_radius * x * zr1, m_radius * y * zr1, m_radius * z1) + position;
			normal = sl::float3(x * zr1, y * zr1, z1);
			normal = normal / normal.norm();
			addPoint(point, clr);
			addNormal(normal);

			point = sl::float3(m_radius * x * zr0, m_radius * y * zr0, m_radius * z0) + position;
			normal = sl::float3(x * zr0, y * zr0, z0);
			normal = normal / normal.norm();
			addPoint(point, clr);
			addNormal(normal);
		}
	}
}

void Simple3DObject::pushToGPU() {
	if (!isStatic_ || vaoID_ == 0) {
		if (vaoID_ == 0) {
			glGenVertexArrays(1, &vaoID_);
			glGenBuffers(4, vboID_);
		}
		glShadeModel(GL_SMOOTH);
		if (vertices_.size() > 0) {
			glBindVertexArray(vaoID_);
			glBindBuffer(GL_ARRAY_BUFFER, vboID_[0]);
			glBufferData(GL_ARRAY_BUFFER, vertices_.size() * sizeof(float), &vertices_[0], isStatic_ ? GL_STATIC_DRAW : GL_DYNAMIC_DRAW);
			glVertexAttribPointer(Shader::ATTRIB_VERTICES_POS, 3, GL_FLOAT, GL_FALSE, 0, 0);
			glEnableVertexAttribArray(Shader::ATTRIB_VERTICES_POS);
		}
		if (colors_.size() > 0) {
			glBindBuffer(GL_ARRAY_BUFFER, vboID_[1]);
			glBufferData(GL_ARRAY_BUFFER, colors_.size() * sizeof(float), &colors_[0], isStatic_ ? GL_STATIC_DRAW : GL_DYNAMIC_DRAW);
			glVertexAttribPointer(Shader::ATTRIB_COLOR_POS, 4, GL_FLOAT, GL_FALSE, 0, 0);
			glEnableVertexAttribArray(Shader::ATTRIB_COLOR_POS);
		}
		if (indices_.size() > 0) {
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboID_[2]);
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices_.size() * sizeof(unsigned int), &indices_[0], isStatic_ ? GL_STATIC_DRAW : GL_DYNAMIC_DRAW);
			can_draw = true;
		}
		if (normals_.size() > 0) {
			glBindBuffer(GL_ARRAY_BUFFER, vboID_[3]);
			glBufferData(GL_ARRAY_BUFFER, normals_.size() * sizeof(float), &normals_[0], isStatic_ ? GL_STATIC_DRAW : GL_DYNAMIC_DRAW);
			glVertexAttribPointer(Shader::ATTRIB_NORMAL, 3, GL_FLOAT, GL_FALSE, 0, 0);
			glEnableVertexAttribArray(Shader::ATTRIB_NORMAL);
		}

		glBindVertexArray(0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}
}

void Simple3DObject::clear() {
	vertices_.clear();
	colors_.clear();
	indices_.clear();
	normals_.clear();
	can_draw = false;
}

void Simple3DObject::setDrawingType(GLenum type) {
	drawingType_ = type;
}

void Simple3DObject::draw() {
	if (can_draw && vaoID_) {
		glBindVertexArray(vaoID_);
		glDrawElements(drawingType_, (GLsizei)indices_.size(), GL_UNSIGNED_INT, 0);
		glBindVertexArray(0);
	}
}

void Simple3DObject::translate(const sl::Translation& t) {
	position_ = position_ + t;
}

void Simple3DObject::setPosition(const sl::Translation& p) {
	position_ = p;
}

void Simple3DObject::setRT(const sl::Transform& mRT) {
	position_ = mRT.getTranslation();
	rotation_ = mRT.getOrientation();
}

void Simple3DObject::rotate(const sl::Orientation& rot) {
	rotation_ = rot * rotation_;
}

void Simple3DObject::rotate(const sl::Rotation& m) {
	this->rotate(sl::Orientation(m));
}

void Simple3DObject::setRotation(const sl::Orientation& rot) {
	rotation_ = rot;
}

void Simple3DObject::setRotation(const sl::Rotation& m) {
	this->setRotation(sl::Orientation(m));
}

const sl::Translation& Simple3DObject::getPosition() const {
	return position_;
}

sl::Transform Simple3DObject::getModelMatrix() const {
	sl::Transform tmp;
	tmp.setOrientation(rotation_);
	tmp.setTranslation(position_);
	return tmp;
}

Shader::Shader(GLchar* vs, GLchar* fs) {
	if (!compile(verterxId_, GL_VERTEX_SHADER, vs)) {
		std::cout << "ERROR: while compiling vertex shader" << std::endl;
	}
	if (!compile(fragmentId_, GL_FRAGMENT_SHADER, fs)) {
		std::cout << "ERROR: while compiling fragment shader" << std::endl;
	}

	programId_ = glCreateProgram();

	glAttachShader(programId_, verterxId_);
	glAttachShader(programId_, fragmentId_);

	glBindAttribLocation(programId_, ATTRIB_VERTICES_POS, "in_vertex");
	glBindAttribLocation(programId_, ATTRIB_COLOR_POS, "in_texCoord");

	glLinkProgram(programId_);

	GLint errorlk(0);
	glGetProgramiv(programId_, GL_LINK_STATUS, &errorlk);
	if (errorlk != GL_TRUE) {
		std::cout << "ERROR: while linking Shader :" << std::endl;
		GLint errorSize(0);
		glGetProgramiv(programId_, GL_INFO_LOG_LENGTH, &errorSize);

		char *error = new char[errorSize + 1];
		glGetShaderInfoLog(programId_, errorSize, &errorSize, error);
		error[errorSize] = '\0';
		std::cout << error << std::endl;

		delete[] error;
		glDeleteProgram(programId_);
	}
}

Shader::~Shader() {
	if (verterxId_ != 0)
		glDeleteShader(verterxId_);
	if (fragmentId_ != 0)
		glDeleteShader(fragmentId_);
	if (programId_ != 0)
		glDeleteShader(programId_);
}

GLuint Shader::getProgramId() {
	return programId_;
}

bool Shader::compile(GLuint &shaderId, GLenum type, GLchar* src) {
	shaderId = glCreateShader(type);
	if (shaderId == 0) {
		std::cout << "ERROR: shader type (" << type << ") does not exist" << std::endl;
		return false;
	}
	glShaderSource(shaderId, 1, (const char**)&src, 0);
	glCompileShader(shaderId);

	GLint errorCp(0);
	glGetShaderiv(shaderId, GL_COMPILE_STATUS, &errorCp);
	if (errorCp != GL_TRUE) {
		std::cout << "ERROR: while compiling Shader :" << std::endl;
		GLint errorSize(0);
		glGetShaderiv(shaderId, GL_INFO_LOG_LENGTH, &errorSize);

		char *error = new char[errorSize + 1];
		glGetShaderInfoLog(shaderId, errorSize, &errorSize, error);
		error[errorSize] = '\0';
		std::cout << error << std::endl;

		delete[] error;
		glDeleteShader(shaderId);
		return false;
	}
	return true;
}

GLchar* IMAGE_FRAGMENT_SHADER =
"#version 330 core\n"
" in vec2 UV;\n"
" out vec4 color;\n"
" uniform sampler2D texImage;\n"
" void main() {\n"
"	vec2 scaler  =vec2(UV.x,1.f - UV.y);\n"
"	vec3 rgbcolor = vec3(texture(texImage, scaler).zyx);\n"
"	vec3 color_rgb = pow(rgbcolor, vec3(1.65f));\n"
"	color = vec4(color_rgb,1);\n"
"}";

GLchar* IMAGE_VERTEX_SHADER =
"#version 330\n"
"layout(location = 0) in vec3 vert;\n"
"out vec2 UV;"
"void main() {\n"
"	UV = (vert.xy+vec2(1,1))* .5f;\n"
"	gl_Position = vec4(vert, 1);\n"
"}\n";


ImageHandler::ImageHandler() {}

ImageHandler::~ImageHandler() {
	close();
}

void ImageHandler::close() {
	glDeleteTextures(1, &imageTex);
}

bool ImageHandler::initialize(sl::Resolution res) {
	shader = Shader(IMAGE_VERTEX_SHADER, IMAGE_FRAGMENT_SHADER);
	texID = glGetUniformLocation(shader.getProgramId(), "texImage");
	static const GLfloat g_quad_vertex_buffer_data[] = {
		-1.0f, -1.0f, 0.0f,
		1.0f, -1.0f, 0.0f,
		-1.0f, 1.0f, 0.0f,
		-1.0f, 1.0f, 0.0f,
		1.0f, -1.0f, 0.0f,
		1.0f, 1.0f, 0.0f };

	glGenBuffers(1, &quad_vb);
	glBindBuffer(GL_ARRAY_BUFFER, quad_vb);
	glBufferData(GL_ARRAY_BUFFER, sizeof(g_quad_vertex_buffer_data), g_quad_vertex_buffer_data, GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glEnable(GL_TEXTURE_2D);
	glGenTextures(1, &imageTex);
	glBindTexture(GL_TEXTURE_2D, imageTex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, res.width, res.height, 0, GL_BGRA_EXT, GL_UNSIGNED_BYTE, NULL);
	glBindTexture(GL_TEXTURE_2D, 0);
	cudaError_t err = cudaGraphicsGLRegisterImage(&cuda_gl_ressource, imageTex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
	return (err == cudaSuccess);
}

void ImageHandler::pushNewImage(sl::Mat &image) {
	cudaArray_t ArrIm;
	cudaGraphicsMapResources(1, &cuda_gl_ressource, 0);
	cudaGraphicsSubResourceGetMappedArray(&ArrIm, cuda_gl_ressource, 0, 0);
	cudaMemcpy2DToArray(ArrIm, 0, 0, image.getPtr<sl::uchar1>(sl::MEM::GPU), image.getStepBytes(sl::MEM::GPU), image.getPixelBytes()*image.getWidth(), image.getHeight(), cudaMemcpyDeviceToDevice);
	cudaGraphicsUnmapResources(1, &cuda_gl_ressource, 0);
}

void ImageHandler::draw() {
	const auto id_shade = shader.getProgramId();
	glUseProgram(id_shade);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, imageTex);
	glUniform1i(texID, 0);
	//invert y axis and color for this image (since its reverted from cuda array)

	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, quad_vb);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
	glDrawArrays(GL_TRIANGLES, 0, 6);
	glDisableVertexAttribArray(0);
	glUseProgram(0);
}
