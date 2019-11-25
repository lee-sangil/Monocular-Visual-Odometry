#ifndef __RANDOM_HPP__
#define __RANDOM_HPP__

#include "core/common.hpp"

namespace lsi{
	void seed();
	double rand();
	double randn();
	void idx_randselect(Eigen::MatrixXd weight, Eigen::MatrixXd& mask, int& i, int& j);
	std::vector<uint32_t> randperm(uint32_t ptNum, int minPtNum);
	std::vector<uint32_t> randweightedpick(const std::vector<double>& h, int n = 1);
}

#endif //__RANDOM_HPP__