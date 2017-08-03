#ifdef _DEBUG
//Debug���[�h�̏ꍇ
#pragma comment(lib,"C:\\opencv\\build\\x86\\vc12\\lib\\opencv_world300d.lib")            // opencv_core
#else
//Release���[�h�̏ꍇ
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

	// ����t�@�C���̓ǂݍ���
	cv::VideoCapture capture;
	capture.open(0);
	if (!capture.isOpened()) {
		std::cerr << "cannot find camera" << std::endl;
		return -1;
	}

	// TV-L1�A���S���Y���ɂ��I�v�e�B�J���t���[�v�Z�I�u�W�F�N�g�̐���
	Ptr<DenseOpticalFlowExt> opticalFlow = superres::createOptFlow_DualTVL1();

	// �O�̃t���[����ۑ����Ă���
	Mat src_image;
	capture >> src_image;
	//	resize(prev, prev, Size(), 0.05, 0.05);
	Mat prev_image(src_image, Rect(0, 0, 50, 50));
	cvtColor(prev_image, prev_image, CV_RGB2GRAY);
	clock_t time;

	while (waitKey(1) == -1)
	{
		// ���݂̃t���[����ۑ�
		time = clock();
		capture >> src_image;
		Mat curr_image(src_image, Rect(0, 0, 50, 50));
		//		resize(curr, curr, Size(), 0.05, 0.05);
		cvtColor(curr_image, curr_image, CV_RGB2GRAY);

		// �I�v�e�B�J���t���[�̌v�Z
		Mat flowX, flowY;
		opticalFlow->calc(prev_image, curr_image, flowX, flowY);

		// �I�v�e�B�J���t���[�̉����i�F�������j
		//  �I�v�e�B�J���t���[���ɍ��W�ɕϊ��i�p�x��[deg]�j
		Mat magnitude, angle;
		cartToPolar(flowX, flowY, magnitude, angle, true);
		//  �F���iH�j�̓I�v�e�B�J���t���[�̊p�x
		//  �ʓx�iS�j��0�`1�ɐ��K�������I�v�e�B�J���t���[�̑傫��
		//  ���x�iV�j��1
		Mat hsvPlanes[3];
		hsvPlanes[0] = angle;
		normalize(magnitude, magnitude, 0, 1, NORM_MINMAX); // ���K��
		hsvPlanes[1] = magnitude;
		hsvPlanes[2] = Mat::ones(magnitude.size(), CV_32F);
		//  HSV���������Ĉꖇ�̉摜�ɂ���
		Mat hsv;
		merge(hsvPlanes, 3, hsv);
		//  HSV����BGR�ɕϊ�
		Mat flowBgr;
		cvtColor(hsv, flowBgr, cv::COLOR_HSV2BGR);

		// �\��
		cv::imshow("input", curr_image);
		cv::imshow("optical flow", flowBgr);

		// �O�̃t���[����ۑ�
		prev_image = curr_image;

		cout << clock() - time << endl;
	}
	return 0;
}