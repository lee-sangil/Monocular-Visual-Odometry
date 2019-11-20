#ifndef __DEPTHFILTER_HPP__
#define __DEPTHFILTER_HPP__

#include "core/common.hpp"
#define DEPTH_MIN 0.1 // meter

class depthFilter{ // inverse-depth estimator
	private:

	bool initialize = false;
	double mean;
	double sigma;
	double a = 10;
	double b = 10;
	static double max;
	// static double min = 0; // 1/inf

	public:

	depthFilter(){};
	void update(double meas, double tau);
	double get_mean() const;
};

#endif