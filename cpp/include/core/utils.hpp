#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include "core/common.hpp"

namespace lsi{
	double rand();
	void idx_randselect(Eigen::MatrixXd weight, uint32_t& i, uint32_t& j);
}

#endif //__UTILS_HPP__