#include "TrackingViewer.hpp"

static void safe_glutBitmapString(void *font, const char *str) {
    for (size_t x = 0; x < strlen(str); ++x)
        glutBitmapCharacter(font, str[x]);
}

TrackingViewer* TrackingViewer::currentInstance_ = nullptr;

TrackBallCamera::TrackBallCamera(vect3 p, vect3 la) {
    position.x = p.x;
    position.y = p.y;
    position.z = p.z;

    lookAt.x = la.x;
    lookAt.y = la.y;
    lookAt.z = la.z;

    angleX = 0.0f;
    applyTransformations();
}

void TrackBallCamera::applyTransformations() {
    forward = vect3(lookAt.x - position.x,
                    lookAt.y - position.y,
                    lookAt.z - position.z);
    left = vect3(forward.z, 0, -forward.x);
    up = vect3(left.y * forward.z - left.z * forward.y,
               left.z * forward.x - left.x * forward.z,
               left.x * forward.y - left.y * forward.x);
    forward.normalise();
    left.normalise();
    up.normalise();
}

void TrackBallCamera::show() {
    gluLookAt(position.x, position.y, position.z,
              lookAt.x, lookAt.y, lookAt.z,
              0.0, 1.0, 0.0);
}

void TrackBallCamera::rotation(float angle, vect3 v) {
    translate(vect3(-lookAt.x, -lookAt.y, -lookAt.z));
    position.rotate(angle, v);
    translate(vect3(lookAt.x, lookAt.y, lookAt.z));
    setAngleX();
}

void TrackBallCamera::rotate(float speed, vect3 v) {
    float tmpA;
    float angle = speed / 360.0f;
    (v.x != 0.0f) ? tmpA = angleX - 90.0f + angle : tmpA = angleX - 90.0f;
    if (tmpA < 89.5f && tmpA > -89.5f) {
        translate(vect3(-lookAt.x, -lookAt.y, -lookAt.z));
        position.rotate(angle, v);
        translate(vect3(lookAt.x, lookAt.y, lookAt.z));
    }
    setAngleX();
}

void TrackBallCamera::translate(vect3 v) {
    position.x += v.x;
    position.y += v.y;
    position.z += v.z;
}

void TrackBallCamera::translateLookAt(vect3 v) {
    lookAt.x += v.x;
    lookAt.y += v.y;
    lookAt.z += v.z;
}

void TrackBallCamera::translateAll(vect3 v) {
    translate(v);
    translateLookAt(v);
}

void TrackBallCamera::zoom(float z) {
    float dist = vect3::length(vect3(position.x - lookAt.x, position.y - lookAt.y, position.z - lookAt.z));

    if (dist - z > z)
        translate(vect3(forward.x * z, forward.y * z, forward.z * z));
}

vect3 TrackBallCamera::getPosition() {
    return vect3(position.x, position.y, position.z);
}

vect3 TrackBallCamera::getPositionFromLookAt() {
    return vect3(position.x - lookAt.x, position.y - lookAt.y, position.z - lookAt.z);
}

vect3 TrackBallCamera::getLookAt() {
    return vect3(lookAt.x, lookAt.y, lookAt.z);
}

vect3 TrackBallCamera::getForward() {
    return vect3(forward.x, forward.y, forward.z);
}

vect3 TrackBallCamera::getUp() {
    return vect3(up.x, up.y, up.z);
}

vect3 TrackBallCamera::getLeft() {
    return vect3(left.x, left.y, left.z);
}

void TrackBallCamera::setPosition(vect3 p) {
    position.x = p.x;
    position.y = p.y;
    position.z = p.z;
    setAngleX();
}

void TrackBallCamera::setLookAt(vect3 p) {
    lookAt.x = p.x;
    lookAt.y = p.y;
    lookAt.z = p.z;
    setAngleX();
}

void TrackBallCamera::setAngleX() {
    angleX = vect3::getAngle(vect3(position.x, position.y + 1, position.z),
                             vect3(position.x, position.y, position.z),
                             vect3(lookAt.x, lookAt.y, lookAt.z));
}

TrackingViewer::TrackingViewer() {
    isInit = false;
    if (currentInstance_ != nullptr)
        delete currentInstance_;
    currentInstance_ = this;
    camera = TrackBallCamera(vect3(2.56f, 1.2f, 0.6f), vect3(0.79f, 0.02f, -1.53f));
    Translate = false;
    Rotate = false;
    Zoom = false;
}

TrackingViewer::~TrackingViewer() {
    isInit = false;
    run = false;
}

void TrackingViewer::redrawCallback() {
    currentInstance_->redraw();
    glutPostRedisplay();
}

void TrackingViewer::mouseCallback(int button, int state, int x, int y) {
    currentInstance_->mouse(button, state, x, y);
}

void TrackingViewer::keyCallback(unsigned char c, int x, int y) {
    currentInstance_->key(c, x, y);
}

void TrackingViewer::specialKeyCallback(int key, int x, int y) {
    currentInstance_->specialkey(key, x, y);
}

void TrackingViewer::motionCallback(int x, int y) {
    currentInstance_->motion(x, y);
}

void TrackingViewer::reshapeCallback(int width, int height) {
    currentInstance_->reshape(width, height);
}

void TrackingViewer::closeCallback() {
    currentInstance_->exit();
}

void TrackingViewer::init() {
    char *argv[1];
    argv[0] = '\0';
    int argc = 1;
    glutInit(&argc, argv);

    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);

    int w = glutGet(GLUT_SCREEN_WIDTH);
    int h = glutGet(GLUT_SCREEN_HEIGHT);
    glutInitWindowSize(w, h);

    glutCreateWindow("ZED Tracking Viewer");

    glewInit();

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_TEXTURE_2D);
    glMatrixMode(GL_PROJECTION);
    gluPerspective(75.0, 1.0, .002, 25.0);
    glMatrixMode(GL_MODELVIEW);
    gluLookAt(0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, .10, 0.0);

    glShadeModel(GL_SMOOTH);
    glDepthFunc(GL_LEQUAL);
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

    glutDisplayFunc(redrawCallback);
    glutMouseFunc(mouseCallback);
    glutKeyboardFunc(keyCallback);
    glutMotionFunc(motionCallback);
    glutReshapeFunc(reshapeCallback);
    glutSpecialFunc(specialKeyCallback);

    glClearDepth(1.0f);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

    zed_path.clear();
    isInit = true;
    run = true;
}

void getColor(int num_segments, int i, float &c1, float &c2, float &c3) {
    float r = fabs(1.f - (float(i)*2.f) / float(num_segments));
    c1 = (0.1f * r);
    c2 = (0.3f * r);
    c3 = (0.8f * r);
}

void TrackingViewer::drawRepere() {
    int num_segments = 60;
    float rad = 0.2f;

    float c1 = (0.09f * 0.5f);
    float c2 = (0.725f * 0.5f);
    float c3 = (0.925f * 0.5f);

    glLineWidth(2.f);

    glBegin(GL_LINE_LOOP);
    for (int ii = 0; ii < num_segments; ii++) {
        float theta = 2.0f * 3.1415926f * float(ii) / float(num_segments);
        glColor3f(c1, c2, c3);
        glVertex3f(rad * cosf(theta), rad * sinf(theta), 0); //output vertex
    }
    glEnd();

    glBegin(GL_LINE_LOOP);
    for (int ii = 0; ii < num_segments; ii++) {
        float theta = 2.0f * 3.1415926f * float(ii) / float(num_segments); //get the current angle
        getColor(num_segments, ii, c1, c2, c3);
        glColor3f(c3, c2, c2);
        glVertex3f(0, rad * sinf(theta), rad * cosf(theta)); //output vertex
    }
    glEnd();

    glBegin(GL_LINE_LOOP);
    for (int ii = 0; ii < num_segments; ii++) {
        float theta = 2.0f * M_PI * (ii + num_segments / 4.f) / float(num_segments); //get the current angle
        theta = theta > (2.f * M_PI) ? theta - (2.f * M_PI) : theta;
        getColor(num_segments, ii, c1, c2, c3);
        glColor3f(c2, c3, c1);
        glVertex3f(rad * cosf(theta), 0, rad * sinf(theta)); //output vertex
    }
    glEnd();
}

void drawLine(float a, float b, float c, float d, color c1, color c2) {
    glBegin(GL_LINES);
    glColor3f(c1.r, c1.g, c1.b);
    glVertex3f(a, 0, b);
    glColor3f(c2.r, c2.g, c2.b);
    glVertex3f(c, 0, d);
    glEnd();
}

void TrackingViewer::drawGridPlan() {
    color c1(13.f / 255.f, 17.f / 255.f, 20.f / 255.f);
    color c2(213.f / 255.f, 207.f / 255.f, 200.f / 255.f);
    float span = 20.f;
    for (int i = (int) -span; i <= (int) span; i++) {
        drawLine(i, -span, i, span, c1, c2);
        float clr = (i + span) / (span * 2);
        color c3(clr, clr, clr);
        drawLine(-span, i, span, i, c3, c3);
    }
}

void TrackingViewer::updateZEDPosition(sl::Transform pose) {

    if (!getViewerState())
        return;

    zed_path.push_back(pose.getTranslation());

    path_locker.lock();
    zed3d.setPath(pose, zed_path);
    path_locker.unlock();
}

void TrackingViewer::redraw() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();
    glPushMatrix();

    camera.applyTransformations();
    camera.show();

    glDisable(GL_LIGHTING);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glClearColor(0.12f, 0.12f, 0.12f, 1.0f);

    path_locker.lock();
    drawGridPlan();
    drawRepere();
    zed3d.draw();
    printText();
    path_locker.unlock();
    glutSwapBuffers();
}

void TrackingViewer::idle() {
    glutPostRedisplay();
}

void TrackingViewer::exit() {
    run = false;
    glutLeaveMainLoop();
}

void TrackingViewer::mouse(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON) {
        if (state == GLUT_DOWN) {
            Rotate = true;
            startx = x;
            starty = y;
        }
        if (state == GLUT_UP)
            Rotate = false;
    }
    if (button == GLUT_RIGHT_BUTTON) {
        if (state == GLUT_DOWN) {
            Translate = true;
            startx = x;
            starty = y;
        }
        if (state == GLUT_UP)
            Translate = false;
    }

    if (button == GLUT_MIDDLE_BUTTON) {
        if (state == GLUT_DOWN) {
            Zoom = true;
            startx = x;
            starty = y;
        }
        if (state == GLUT_UP)
            Zoom = false;
    }

    if ((button == 3) || (button == 4)) {
        if (state == GLUT_UP) return;
        if (button == 3)
            camera.zoom(0.5f);
        else
            camera.zoom(-0.5f);
    }
}

void TrackingViewer::motion(int x, int y) {
    if (Translate) {
        float Trans_x = (x - startx) / 30.0f;
        float Trans_y = (y - starty) / 30.0f;
        vect3 left = camera.getLeft();
        vect3 up = camera.getUp();
        camera.translateAll(vect3(left.x * Trans_x, left.y * Trans_x, left.z * Trans_x));
        camera.translateAll(vect3(up.x * -Trans_y, up.y * -Trans_y, up.z * -Trans_y));
        startx = x;
        starty = y;
    }

    if (Zoom) {
        camera.zoom((float) (y - starty) / 10.0f);
        starty = y;
    }

    if (Rotate) {
        float sensitivity = 100.0f;
        float Rot = (float) (y - starty);
        vect3 tmp = camera.getPositionFromLookAt();
        tmp.y = tmp.x;
        tmp.x = -tmp.z;
        tmp.z = tmp.y;
        tmp.y = 0.0f;
        tmp.normalise();
        camera.rotate(Rot * sensitivity, tmp);

        Rot = (float) (x - startx);
        camera.rotate(-Rot * sensitivity, vect3(0.0f, 1.0f, 0.0f));

        startx = x;
        starty = y;
    }
    glutPostRedisplay();
}

void TrackingViewer::reshape(int width, int height) {
    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(75.0, (double) (width / (double) height), .002, 40.0);
    glMatrixMode(GL_MODELVIEW);
}

void TrackingViewer::key(unsigned char c, int x, int y) {
    switch (c) {
        case 'o':
        camera.setPosition(vect3(2.56f, 1.2f, 0.6f));
        camera.setLookAt(vect3(0.79f, 0.02f, -1.53f));
        break;

        case 'q':
        case 'Q':
        case 27:
        currentInstance_->run = false;
        glutLeaveMainLoop();
        break;

        default:
        break;
    }
    glutPostRedisplay();
}

void TrackingViewer::specialkey(int key, int x, int y) {
    float sensitivity = 150.0f;
    vect3 tmp = camera.getPositionFromLookAt();

    tmp.y = tmp.x;
    tmp.x = -tmp.z;
    tmp.z = tmp.y;
    tmp.y = 0.0f;
    tmp.normalise();

    switch (key) {
        case GLUT_KEY_UP:
        camera.rotate(-sensitivity, tmp);
        break;
        case GLUT_KEY_DOWN:
        camera.rotate(sensitivity, tmp);
        break;
        case GLUT_KEY_LEFT:
        camera.rotate(sensitivity, vect3(0.0f, 1.0f, 0.0f));
        break;
        case GLUT_KEY_RIGHT:
        camera.rotate(-sensitivity, vect3(0.0f, 1.0f, 0.0f));
        break;
    }
}

void TrackingViewer::printText() {
    if (!isInit)
        return;
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    int w_wnd = glutGet(GLUT_WINDOW_WIDTH);
    int h_wnd = glutGet(GLUT_WINDOW_HEIGHT);
    glOrtho(0, w_wnd, 0, h_wnd, -1.0f, 1.0f);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    int start_w = 20;
    int start_h = h_wnd - 40;

    bool trackingIsOK = trackState == sl::TRACKING_STATE_OK;

    trackingIsOK ? glColor3f(0.2f, 0.65f, 0.2f) : glColor3f(0.85f, 0.2f, 0.2f);
    glRasterPos2i(start_w, start_h);
    safe_glutBitmapString(GLUT_BITMAP_HELVETICA_18, sl::trackingState2str(trackState).c_str());

    glColor3f(0.9255f, 0.9412f, 0.9451f);
    glRasterPos2i(start_w, start_h - 25);
    safe_glutBitmapString(GLUT_BITMAP_HELVETICA_18, "Translation (m) :");

    glColor3f(0.4980f, 0.5490f, 0.5529f);
    glRasterPos2i(155, start_h - 25);
    safe_glutBitmapString(GLUT_BITMAP_HELVETICA_18, txtT.c_str());

    glColor3f(0.9255f, 0.9412f, 0.9451f);
    glRasterPos2i(start_w, start_h - 50);
    safe_glutBitmapString(GLUT_BITMAP_HELVETICA_18, "Rotation   (rad) :");

    glColor3f(0.4980f, 0.5490f, 0.5529f);
    glRasterPos2i(155, start_h - 50);
    safe_glutBitmapString(GLUT_BITMAP_HELVETICA_18, txtR.c_str());

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

}

void TrackingViewer::updateText(std::string stringT, std::string stringR, sl::TRACKING_STATE state) {

    if (!getViewerState())
        return;

    txtT = stringT;
    txtR = stringR;
    trackState = state;
}
