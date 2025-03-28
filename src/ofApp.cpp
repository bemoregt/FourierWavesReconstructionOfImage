#include "ofApp.h"


//-----------------------------------------------------
void ofApp::fftshift(cv::Mat& in, cv::Mat& out) {
    out = in.clone();
    int mx1, my1, mx2, my2;
    mx1 = out.cols / 2;
    my1 = out.rows / 2;
    mx2 = int(ceil(out.cols / 2.0));
    my2 = int(ceil(out.rows / 2.0));

    Mat q0(out, cv::Rect(0, 0, mx2, my2));
    Mat q1(out, cv::Rect(mx2, 0, mx1, my2));
    Mat q2(out, cv::Rect(0, my2, mx2, my1));
    Mat q3(out, cv::Rect(mx2, my2, mx1, my1));

    Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q2.copyTo(tmp);
    q1.copyTo(q2);
    tmp.copyTo(q1);

    vconcat(q1, q3, out);
    vconcat(q0, q2, tmp);
    hconcat(tmp, out, out);
}

//----------------------------------------------------
void ofApp::setup() {
    // 이미지 로드 및 초기화
    string path1 = "C:\\Users\\wwpark\\Pictures\\lena512.jpg";
    originalImg.load(path1);
    originalImg.setImageType(OF_IMAGE_GRAYSCALE);

    width = originalImg.getWidth();
    height = originalImg.getHeight();

    // OpenCV Mat으로 변환
    Mat imgMat = toCv(originalImg);
    imgMat.convertTo(imgMat, CV_32F);

    // 평균값 제거
    Scalar mean = cv::mean(imgMat);
    imgMat -= mean[0];

    // DFT 수행을 위한 최적 크기 계산
    Mat padded;
    int m = getOptimalDFTSize(imgMat.rows);
    int n = getOptimalDFTSize(imgMat.cols);
    copyMakeBorder(imgMat, padded, 0, m - imgMat.rows, 0, n - imgMat.cols,
        BORDER_CONSTANT, Scalar::all(0));

    // DFT 수행
    Mat planes[] = { padded, Mat::zeros(padded.size(), CV_32F) };
    merge(planes, 2, complexImg);
    dft(complexImg, fftMat, DFT_COMPLEX_OUTPUT);

    // 진폭 계산 및 정렬
    Mat magMat;
    vector<tuple<float, int, int>> magnitudes;
    split(fftMat, planes);
    magnitude(planes[0], planes[1], magMat);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            float mag = magMat.at<float>(i, j);
            magnitudes.push_back(make_tuple(mag, i, j));
        }
    }

    sort(magnitudes.begin(), magnitudes.end(), greater<tuple<float, int, int>>());

    for (const auto& mag : magnitudes) {
        sortedIndices.push_back({ get<1>(mag), get<2>(mag) });
    }

    // 결과 저장용 Mat 초기화
    frameMat = Mat::zeros(fftMat.size(), CV_32FC2);
    waveMat = Mat::zeros(fftMat.size(), CV_32FC2);

    // FBO 초기화
    reconImageFbo.allocate(width, height);
    waveFbo.allocate(width, height);
    amplitudeFbo.allocate(width, height);
    phaseFbo.allocate(width, height);

    currentIndex = 0;
}

//-----------------------------------------------
void ofApp::update() {
    if (currentIndex >= sortedIndices.size()) return;

    // 현재 주파수 성분 추가
    int i = sortedIndices[currentIndex].first;
    int j = sortedIndices[currentIndex].second;

    // 복소수 값 추출 및 설정
    Vec2f value = fftMat.at<Vec2f>(i, j);
    Vec2f conjugate(value[0], -value[1]);

    frameMat.at<Vec2f>(i, j) = value;
    frameMat.at<Vec2f>((height - i) % height, (width - j) % width) = conjugate;

    waveMat = Mat::zeros(fftMat.size(), CV_32FC2);
    waveMat.at<Vec2f>(i, j) = value;
    waveMat.at<Vec2f>((height - i) % height, (width - j) % width) = conjugate;

    updateFbos();

    currentIndex += 2;
}

//--------------------------------------
void ofApp::updateFbos() {
    // 복원 이미지
    Mat inverseFrame;
    dft(frameMat, inverseFrame, DFT_INVERSE | DFT_SCALE);
    Mat planes[2];
    split(inverseFrame, planes);
    Mat reconMat = planes[0];
    normalize(reconMat, reconMat, 0, 255, NORM_MINMAX);
    reconMat.convertTo(reconMat, CV_8U);
    reconImageFbo.begin();
    drawMat(reconMat, 0, 0);
    reconImageFbo.end();

    // 웨이브 이미지
    Mat inverseWave;
    dft(waveMat, inverseWave, DFT_INVERSE | DFT_SCALE);
    split(inverseWave, planes);
    Mat waveMat = planes[0];
    normalize(waveMat, waveMat, 0, 255, NORM_MINMAX);
    waveMat.convertTo(waveMat, CV_8U);
    waveFbo.begin();
    drawMat(waveMat, 0, 0);
    waveFbo.end();

    // 진폭 스펙트럼
    Mat shiftedFrame;
    fftshift(frameMat, shiftedFrame);
    Mat planes1[2];
    split(shiftedFrame, planes1);
    Mat magMat;
    magnitude(planes1[0], planes1[1], magMat);
    magMat += Scalar::all(1);
    log(magMat, magMat);
    normalize(magMat, magMat, 0, 255, NORM_MINMAX);
    magMat.convertTo(magMat, CV_8U);
    amplitudeFbo.begin();
    drawMat(magMat, 0, 0);
    amplitudeFbo.end();

    // 위상 스펙트럼
    Mat phaseMat;
    phase(planes1[0], planes1[1], phaseMat);
    normalize(phaseMat, phaseMat, 0, 255, NORM_MINMAX);
    phaseMat.convertTo(phaseMat, CV_8U);
    phaseFbo.begin();
    drawMat(phaseMat, 0, 0);
    phaseFbo.end();
}

// -------------------------------------------
void ofApp::draw() {
    ofBackground(0);

    float scale = 0.5;
    int titlePadding = 20; // 제목과 이미지 사이의 간격

    // 제목 스타일 설정
    ofSetColor(255);
    float fontSize = 14;

    // 왼쪽 상단: 복원된 이미지
    reconImageFbo.draw(0, titlePadding, width * scale, height * scale);
    ofDrawBitmapStringHighlight("Reconstructed Image",
        10,
        titlePadding - 5);

    // 오른쪽 상단: 현재 주파수 성분
    waveFbo.draw(width * scale, titlePadding,
        width * scale, height * scale);
    ofDrawBitmapStringHighlight("Current Frequency Component",
        width * scale + 10,
        titlePadding - 5);

    // 왼쪽 하단: 진폭 스펙트럼
    amplitudeFbo.draw(0, height * scale + titlePadding * 2,
        width * scale, height * scale);
    ofDrawBitmapStringHighlight("Amplitude Spectrum",
        10,
        height * scale + titlePadding * 2 - 5);

    // 오른쪽 하단: 위상 스펙트럼
    phaseFbo.draw(width * scale, height * scale + titlePadding * 2,
        width * scale, height * scale);
    ofDrawBitmapStringHighlight("Phase Spectrum",
        width * scale + 10,
        height * scale + titlePadding * 2 - 5);

    // 진행 상황 표시
    ofDrawBitmapStringHighlight("Progress: " + ofToString(currentIndex) + "/" +
        ofToString(sortedIndices.size()),
        10,
        (height * scale) * 2 + titlePadding * 2 + 15);
}