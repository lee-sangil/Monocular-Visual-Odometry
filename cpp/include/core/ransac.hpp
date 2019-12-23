#ifndef __RANSAC_HPP__
#define __RANSAC_HPP__

#include <functional>
#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>

#include "core/random.hpp"

namespace lsi{
	template <typename DATA, typename FUNC>
	struct RansacCoef{
		int max_iteration = 1e4;
		double min_num_point = 5;
		double th_inlier_ratio = 0.9;
		double th_dist = 0.5;
		double th_dist_outlier = 5.0;
		std::vector<double> th_dist_arr;
		std::vector<double> weight;
		std::function<void(const std::vector<DATA>&, FUNC&)> calculate_func;
		std::function<void(const FUNC, const std::vector<DATA>&, std::vector<double>&)> calculate_dist;
	};

	template <typename DATA, typename FUNC>
	void ransac(const std::vector<DATA>& samples, const lsi::RansacCoef<DATA, FUNC>& param, FUNC& val, std::vector<bool>& inlier, std::vector<bool>& outlier){
		uint32_t num_pts = samples.size();

		std::vector<uint32_t> sample_idx;
		std::vector<DATA> sample;
		std::vector<double> dist;
		dist.reserve(num_pts);

		bool use_threshold_array = (param.th_dist_arr.size() > 0);

		int num_iteration = 1e5;
		uint32_t num_max_inlier = 0;
		uint32_t num_inlier = 0;
		double inlier_ratio;
		std::vector<bool> in1;
		in1.reserve(num_pts);

		FUNC max_val;

		for( int it = 0; it < std::min(param.max_iteration, num_iteration); it++ ){
			// choose random samples
			if (param.weight.size() > 0)
				sample_idx = lsi::randweightedpick(param.weight, param.min_num_point);
			else
				sample_idx = lsi::randperm(num_pts, param.min_num_point);

			sample.clear();
			for (uint32_t i = 0; i < sample_idx.size(); i++){
				sample.push_back(samples[sample_idx[i]]);
			}

			// generate model using chosen samples
			param.calculate_func(sample, val);

			// evaluate model
			param.calculate_dist(val, samples, dist);

			in1.clear();
			num_inlier = 0;

			// count inliers
			if( use_threshold_array ){
				for (uint32_t i = 0; i < num_pts; i++){
					if( dist[i] < param.th_dist_arr[i] ){
						in1.push_back( true );
						num_inlier++;
					}else{
						in1.push_back( false );
					}
				}
			}else{
				for (uint32_t i = 0; i < num_pts; i++){
					if( dist[i] < param.th_dist ){
						in1.push_back( true );
						num_inlier++;
					}else{
						in1.push_back( false );
					}
				}
			}
			
			// find model with maximum inlier
			if (num_inlier > num_max_inlier){
				num_max_inlier = num_inlier;
				inlier = in1;
				inlier_ratio = (double)num_max_inlier / (double)num_pts + 1e-16;
				num_iteration = static_cast<int>(std::floor(std::log(1 - param.th_inlier_ratio) / std::log(1 - std::pow(inlier_ratio, param.min_num_point))));
				max_val = val;
			}
		}
		
		if (num_max_inlier == 0){
			inlier.clear();
			outlier.clear();
		}else{
			// // With refinement
			// sample.clear();
			// for (uint32_t i = 0; i < inlier.size(); i++)
			//     if (inlier[i])
			//         sample.push_back(samples[i]);
			
			// param.calculate_func(sample, val);
			// param.calculate_dist(val, samples, dist);

			// inlier.clear();
			// outlier.clear();
			// for (uint32_t i = 0; i < num_pts; i++){
			//     inlier.push_back(dist[i] < param.th_dist);
			//     outlier.push_back(dist[i] > param.th_dist_outlier);
			// }

			// Without refinement
			param.calculate_dist(max_val, samples, dist);

			// count outliers
			outlier.clear();
			if( use_threshold_array ){
				for (uint32_t i = 0; i < num_pts; i++)
					outlier.push_back(dist[i] > param.th_dist_arr[i]);
			}else{
				for (uint32_t i = 0; i < num_pts; i++)
					outlier.push_back(dist[i] > param.th_dist_outlier);
			}
			
			val = max_val;
		}
	}
	void calculateScale(const std::vector<std::pair<cv::Point3f,cv::Point3f>>& pts, double& scale, double reference_value, double reference_weight);
	void calculateScaleError(const double scale, const std::vector<std::pair<cv::Point3f,cv::Point3f>>& pts, std::vector<double>& dist);
	void calculatePlane(const std::vector<cv::Point3f>& pts, std::vector<double>& plane);
	void calculatePlaneError(const std::vector<double>& plane, const std::vector<cv::Point3f>& pts, std::vector<double>& dist);
}

#endif //__RANSAC_HPP__