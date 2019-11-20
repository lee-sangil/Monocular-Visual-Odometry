#ifndef __DEPTHFILTER_HPP__
#define __DEPTHFILTER_HPP__

#include "core/common.hpp"
#define DEPTH_MIN 0.1 // meter
#define PX_NOISE 1 // px

class depthFilter{ // inverse-depth estimator
	private:

	bool initialize = false;
	double mean;
	double sigma;
	double a = 10;
	double b = 10;
	static double px_error_angle;
	static double max;
	// static double min = 0; // 1/inf

	public:

	depthFilter(){};
	void update(const double meas, const double tau);
	double get_mean() const;
	static double computeTau(const Eigen::Matrix4d& Toc, const Eigen::Vector3d& uv, const double z, const double px_error_angle);
	static double computeInverseTau(const double z, const double tau);
};

#endif