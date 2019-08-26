#ifndef __COMMON_HPP__
#define __COMMON_HPP__

#include <iostream>
#include <iomanip>
#include <map>
#include <string>
#include <sstream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <time.h>
#include <vector>
#include <valarray>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <chrono>
#include <functional>

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
	unsigned int safety = 20;
	unsigned int max_features = 400;
	std::vector<unsigned int> grid;
	std::vector<unsigned int> size;
	Eigen::MatrixXd mass;
	Eigen::MatrixXd prob;
}Bucket;

typedef struct Feature{
	static unsigned int new_feature_id;
	unsigned int id;
	unsigned int life;
	unsigned int frame_init;
	std::vector< cv::Point2d > uv;
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