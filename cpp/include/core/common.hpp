#ifndef __COMMON_HPP__
#define __COMMON_HPP__

#include <vector>
#include <memory>
#include <eigen3/Eigen/Core>
#include <opencv2/core/core.hpp>

#define D_METER 1.2 // for key interrupt
#define D_RADIAN M_PI/24 // for key interrupt
#define YAML_VERSION 2.5 // compatibility check

enum Type{Unknown,Dynamic,Road,};

class DepthFilter;

typedef struct Feature{
	static uint32_t new_feature_id;
	uint32_t id;
	uint32_t life;
	int frame_2d_init;
	int frame_3d_init;
	// cv::Mat desc;
	double parallax;
	std::vector< cv::Point2f > uv;
	cv::Point2f uv_pred;
	cv::Point bucket;
	Eigen::Vector4d point_curr;
	Eigen::Vector4d point_init;
	bool is_alive;
	bool is_matched;
	bool is_wide;
	bool is_2D_inliered;
	bool is_3D_reconstructed;
	bool is_3D_init;
	Type type;
	std::shared_ptr<DepthFilter> depthfilter;
}Feature;

#endif //__COMMON_HPP__
