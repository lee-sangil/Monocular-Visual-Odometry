#ifndef __RANSAC_HPP__
#define __RANSAC_HPP__

#include "core/common.hpp"

class RANSAC{
	public:

	template <typename DATA, typename FUNC>
	struct RansacCoef{
		int iterMax = 1e4;
		double minPtNum = 5;
		double thInlrRatio = 0.9;
		double thDist = 0.5;
		double thDistOut = 5.0;
		std::vector<double> weight;
		std::function<void(const std::vector<DATA>&, FUNC&)> calculate_func;
		std::function<void(const FUNC, const std::vector<DATA>&, std::vector<double>&)> calculate_dist;
	};