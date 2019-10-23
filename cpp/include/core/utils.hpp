#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include "core/common.hpp"

namespace lsi{
	double rand();
	void idx_randselect(Eigen::MatrixXd weight, Eigen::MatrixXd& mask, int& i, int& j);
}

#endif //__UTILS_HPP__