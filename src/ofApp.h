#pragma once

#include "ofMain.h"
#include "ofxCv.h"
#include <vector>
#include <algorithm>

using namespace cv;
using namespace ofxCv;

class ofApp : public ofBaseApp {
public:
    void setup();
    void update();
    void draw();
    void fftshift(cv::Mat& in, cv::Mat& out);

private:
    void updateFbos();

    ofImage originalImg;
    int width, height;
    vector<pair<int, int>> sortedIndices;

    // OpenCV Matrices
    cv::Mat complexImg;
    cv::Mat fftMat;
    cv::Mat frameMat;
    cv::Mat waveMat;

    // FBOs
    ofFbo reconImageFbo;
    ofFbo waveFbo;
    ofFbo amplitudeFbo;
    ofFbo phaseFbo;

    int currentIndex;
};