#ifndef __DEPTHFILTER_HPP__
#define __DEPTHFILTER_HPP__

#include "core/common.hpp"

class depthFilter{
	private:

	double mean;
	double variance;
	double a;
	double b;
	double min, max;

	public:

	depthFilter(double mean, double variance, double a, double b, double min, double max):mean(mean),variance(variance),a(a),b(b),min(min),max(max){}
	void update(double measurement, double dt);
	double get_mean() const;
};

#endif