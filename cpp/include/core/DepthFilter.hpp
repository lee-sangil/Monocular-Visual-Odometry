#ifndef __DEPTHFILTER_HPP__
#define __DEPTHFILTER_HPP__

#include "core/common.hpp"

class DepthFilter{ // inverse-depth estimator
	private:

	bool initialize_ = false;
	double mean_;
	double sigma_;
	double a_ = 10;
	double b_ = 10;
	
	public:

	static double s_px_error_angle_;
	static double s_meas_max_;
	// static double min = 0; // 1/inf

	public:

	DepthFilter(){};
	void seed(const double a, const double b){this->a_ = a; this->b_ = b;};

	// tau: standard variation of meas (=measurement), in usage, meas = inverse-depth, tau: standard variation of inverse-depth
	void update(const double meas, const double tau);
	double getMean() const;
	double getVariance() const;
	double getA() const;
	double getB() const;
	void reset();

	// tau: standard variation of z, depth
	static double computeTau(const Eigen::Matrix4d& Toc, const Eigen::Vector3d& p);
	// inverse tau: standard variation of inverse-depth
	static double computeInverseTau(const double z, const double tau);
};

#endif