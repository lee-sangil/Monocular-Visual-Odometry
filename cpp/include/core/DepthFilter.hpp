#ifndef __DEPTHFILTER_HPP__
#define __DEPTHFILTER_HPP__

#include "core/common.hpp"

class DepthFilter{ // inverse-depth estimator
	private:

	bool initialize = false;
	double mean;
	double sigma;
	double a = 10;
	double b = 10;
	
	public:

	static double px_error_angle;
	static double meas_max;
	// static double min = 0; // 1/inf

	public:

	DepthFilter(){};
	void seed(const double a, const double b){this->a = a; this->b = b;};

	// tau: standard variation of meas (=measurement), in usage, meas = inverse-depth, tau: standard variation of inverse-depth
	void update(const double meas, const double tau);
	double get_mean() const;
	double get_variance() const;

	// tau: standard variation of z, depth
	static double computeTau(const Eigen::Matrix4d& Toc, const Eigen::Vector3d& p);
	// inverse tau: standard variation of inverse-depth
	static double computeInverseTau(const double z, const double tau);
};

#endif