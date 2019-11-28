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
#include <memory>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/SVD>
#include <eigen3/Eigen/QR>
#include <eigen3/Eigen/LU>
#include <eigen3/Eigen/Eigen>
#include <eigen3/unsupported/Eigen/MatrixFunctions>
#include <eigen3/Eigen/StdVector>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <boost/filesystem.hpp>

#include "core/DepthFilter.hpp"

#define D_METER 1.2
#define D_RADIAN M_PI/24
#define YAML_VERSION 2.3

enum Type{Unknown,Dynamic,Road,};

class DepthFilter;

typedef struct Bucket{
	int safety = 20;
	int max_features = 400;
	cv::Size grid;
	cv::Size size;
	cv::Mat cv_mass;
	cv::Mat cv_prob;
	Eigen::MatrixXd mass;
	Eigen::MatrixXd prob;
	Eigen::MatrixXd saturated;
}Bucket;

typedef struct Feature{
	static uint32_t new_feature_id;
	uint32_t id;
	uint32_t life;
	uint32_t frame_init;
	// cv::Mat desc;
	std::vector< cv::Point2f > uv;
	cv::Point2f uv_pred;
	cv::Point bucket;
	Eigen::Vector4d point_curr;
	Eigen::Vector4d point_init;
	double point_var;
	bool is_alive;
	bool is_matched;
	bool is_wide;
	bool is_2D_inliered;
	bool is_3D_reconstructed;
	bool is_3D_init;
	Type type;
	DepthFilter * depthfilter;
}Feature;

#endif //__COMMON_HPP__
