#ifndef __COMMON_HPP__
#define __COMMON_HPP__

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <functional>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <iterator>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <sstream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <valarray>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/src/Core/IO.h>
#include <eigen3/Eigen/SVD>
#include <eigen3/Eigen/LU>
#include <eigen3/unsupported/Eigen/MatrixFunctions>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video.hpp>
#include <boost/filesystem.hpp>

typedef struct Bucket{
	uint32_t safety = 20;
	uint32_t max_features = 400;
	cv::Size grid;
	cv::Size size;
	Eigen::MatrixXd mass;
	Eigen::MatrixXd prob;
}Bucket;

typedef struct Feature{
	static uint32_t new_feature_id;
	uint32_t id;
	uint32_t life;
	uint32_t frame_init;
	std::vector< cv::Point2f > uv;
	cv::Point bucket;
	Eigen::Vector4d point;
	Eigen::Vector4d point_init;
	double point_var;
	bool is_matched;
	bool is_wide;
	bool is_2D_inliered;
	bool is_3D_reconstructed;
	bool is_3D_init;
}Feature;

#endif //__COMMON_HPP__
