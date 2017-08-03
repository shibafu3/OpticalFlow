#ifdef _DEBUG
//Debugモードの場合
#pragma comment(lib,"C:\\opencv\\build\\x86\\vc12\\lib\\opencv_world300d.lib")            // opencv_core
#else
//Releaseモードの場合
#pragma comment(lib,"C:\\opencv\\build\\x86\\vc12\\lib\\opencv_world300.lib") 
#endif

#include "opencv/cv.h"
#include "opencv2/highgui/highgui_c.h"
#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/superres/optical_flow.hpp>
#include <time.h>
#include <windows.h>

#define IMAGE "C:\\aaa\\02.jpg"
#define XML "C:\\aaa\\4car400x4_03.xml"
#define SCALE_FACTOR 1.2
#define MIN_NEIGHBORS 10
#define MIN_OBJECT_SIZE Size(50,50)

using namespace std;
using namespace cv;
using namespace cv::superres;

int main(){

	// 動画ファイルの読み込み
	cv::VideoCapture capture;
	capture.open(0);
	if (!capture.isOpened()) {
		std::cerr << "cannot find camera" << std::endl;
		return -1;
	}

	// TV-L1アルゴリズムによるオプティカルフロー計算オブジェクトの生成
	Ptr<DenseOpticalFlowExt> opticalFlow = superres::createOptFlow_DualTVL1();

	// 前のフレームを保存しておく
	Mat src_image;
	capture >> src_image;
	//	resize(prev, prev, Size(), 0.05, 0.05);
	Mat prev_image(src_image, Rect(0, 0, 50, 50));
	cvtColor(prev_image, prev_image, CV_RGB2GRAY);
	clock_t time;

	while (waitKey(1) == -1)
	{
		// 現在のフレームを保存
		time = clock();
		capture >> src_image;
		Mat curr_image(src_image, Rect(0, 0, 50, 50));
		//		resize(curr, curr, Size(), 0.05, 0.05);
		cvtColor(curr_image, curr_image, CV_RGB2GRAY);

		// オプティカルフローの計算
		Mat flowX, flowY;
		opticalFlow->calc(prev_image, curr_image, flowX, flowY);

		// オプティカルフローの可視化（色符号化）
		//  オプティカルフローを極座標に変換（角度は[deg]）
		Mat magnitude, angle;
		cartToPolar(flowX, flowY, magnitude, angle, true);
		//  色相（H）はオプティカルフローの角度
		//  彩度（S）は0～1に正規化したオプティカルフローの大きさ
		//  明度（V）は1
		Mat hsvPlanes[3];
		hsvPlanes[0] = angle;
		normalize(magnitude, magnitude, 0, 1, NORM_MINMAX); // 正規化
		hsvPlanes[1] = magnitude;
		hsvPlanes[2] = Mat::ones(magnitude.size(), CV_32F);
		//  HSVを合成して一枚の画像にする
		Mat hsv;
		merge(hsvPlanes, 3, hsv);
		//  HSVからBGRに変換
		Mat flowBgr;
		cvtColor(hsv, flowBgr, cv::COLOR_HSV2BGR);

		// 表示
		cv::imshow("input", curr_image);
		cv::imshow("optical flow", flowBgr);

		// 前のフレームを保存
		prev_image = curr_image;

		cout << clock() - time << endl;
	}
	return 0;
}