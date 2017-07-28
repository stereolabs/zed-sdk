#ifndef __CLOUD_VIEWER_INCLUDE__
#define __CLOUD_VIEWER_INCLUDE__

#include "utils.hpp"

#include <math.h>
#include <thread>         // std::thread
#include <mutex>          // std::mutex

#include "ZEDModel.hpp"    /* OpenGL Utility Toolkit header */
#include <sl/Camera.hpp>


class TrackBallCamera {
public:

    TrackBallCamera() {};
    TrackBallCamera(vect3 p, vect3 la);
    void applyTransformations();
    void show();
    void rotation(float angle, vect3 v);
    void rotate(float speed, vect3 v);
    void translate(vect3 v);
    void translateLookAt(vect3 v);
    void translateAll(vect3 v);
    void zoom(float z);

    vect3 getPosition();
    vect3 getPositionFromLookAt();
    vect3 getLookAt();
    vect3 getForward();
    vect3 getUp();
    vect3 getLeft();

    void setPosition(vect3 p);
    void setLookAt(vect3 p);

private:
    vect3 position;
    vect3 lookAt;
    vect3 forward;
    vect3 up;
    vect3 left;
    float angleX;

    void setAngleX();
};

class TrackingViewer {
public:

    TrackingViewer();
    virtual ~TrackingViewer();


    void init();
    unsigned char getKey();
    void updateText(std::string stringT, std::string stringR, sl::TRACKING_STATE stringState);
    void updateZEDPosition(sl::Transform);

    bool getViewerState() {
        return isInit;
    }

    bool runs() {
        return run;
    }

    void exit();
    std::mutex path_locker;

private:
    static TrackingViewer* currentInstance_;
    //OGL functions
    static void redrawCallback();
    static void mouseCallback(int button, int state, int x, int y);
    static void keyCallback(unsigned char c, int x, int y);
    static void specialKeyCallback(int key, int x, int y);
    static void motionCallback(int x, int y);
    static void reshapeCallback(int width, int height);
    static void closeCallback();

    // drawing
    void drawGridPlan();
    void drawRepere();

    // ZED model
    Zed3D zed3d;

    void idle();
    void redraw();
    void mouse(int button, int state, int x, int y);
    void key(unsigned char c, int x, int y);
    void specialkey(int key, int x, int y);
    void motion(int x, int y);
    void reshape(int width, int height);

    //text
    void printText();
    std::string txtT;
    std::string txtR;
    sl::TRACKING_STATE trackState;
    std::vector<sl::Translation> zed_path;
    //int trackConf;

    //! Mouse Save Position
    bool Rotate;
    bool Translate;
    bool Zoom;
    int startx;
    int starty;

    TrackBallCamera camera;

    bool isInit;
    bool run;
};


#endif
