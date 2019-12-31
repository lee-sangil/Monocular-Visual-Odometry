#ifndef __RANSAC_HPP__
#define __RANSAC_HPP__

#include <functional>
#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>

#include "core/random.hpp"

/**
 * @author Sangil Lee (sangillee724@gmail.com)
 * @date 27-Dec-2019
 */
namespace lsi{
	/** @brief RANSAC 알고리즘 수행에 필요한 데이터들을 정리 */
	template <typename DATA, typename FUNC>
	struct RansacCoef{
		int max_iteration = 1e4; /**< @brief iteration 최대 횟수 */
		double min_num_point = 5; /**< @brief inlier로 판별되기 위한 최소 점 개수 */
		double th_inlier_ratio = 0.9; /**< @brief inlier 비율 */
		double th_dist = 0.5; /**< @brief 거리 threshold (m 단위)  */
		double th_dist_outlier = 5.0; /**< @brief outlier로 판별되는 거리 threshold */
		std::vector<double> th_dist_arr; /**< @brief 거리 threshold 벡터 */
		std::vector<double> weight; /**< @brief 랜덤추출 weight */
		std::function<void(const std::vector<DATA>&, FUNC&)> calculate_func; /**< @brief RANSAC 모델 함수 */
		std::function<void(const FUNC, const std::vector<DATA>&, std::vector<double>&)> calculate_dist; /**< @brief RANSAC 에러 계산 함수 */
	};

	/**
	 * @brief RANSAC 알고리즘을 수행하는 함수
	 * @details param 옵션을 통하여 sample 들중 inlier와 outlier를 판별하는 함수. 모델 결과 값은 val에 저장한다.
	 * @param samples RANSAC을 수행할 데이터 샘플
	 * @param param RANSAC에 사용되는 옵션 (interation, threshold, model functino, error function, ...)
	 * @param val 모델 결과 값
	 * @param inlier sample의 inlier mask
	 * @param outlier sample의 oulier mask
	 * @author Sangil Lee (sangillee724@gmail.com)
	 * @date 24-Dec-2019
	 */
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
			// 1. fit using random points
			if (param.weight.size() > 0)
				sample_idx = lsi::randweightedpick(param.weight, param.min_num_point);
			else
				sample_idx = lsi::randperm(num_pts, param.min_num_point);

			sample.clear();
			for (uint32_t i = 0; i < sample_idx.size(); i++){
				sample.push_back(samples[sample_idx[i]]);
			}

			param.calculate_func(sample, val);
			param.calculate_dist(val, samples, dist);

			in1.clear();
			num_inlier = 0;

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
	void calculateScale(const std::vector<std::pair<cv::Point3f,cv::Point3f>>& pts, double& scale, double reference_value, double reference_weight); /**< @brief scale을 계산하는 함수 */
	void calculateScaleError(const double scale, const std::vector<std::pair<cv::Point3f,cv::Point3f>>& pts, std::vector<double>& dist); /**< @brief 두 점들에 scale을 적용하였을 때의 거리를 계산하는 함수 */
	void calculatePlane(const std::vector<cv::Point3f>& pts, std::vector<double>& plane); /**< @brief 두 점들에 scale을 적용하였을 때의 거리를 계산하는 함수 */
	void calculatePlaneError(const std::vector<double>& plane, const std::vector<cv::Point3f>& pts, std::vector<double>& dist); /**< @brief s점과 평면사이의 거리를 계산하는 함수 */
}

#endif //__RANSAC_HPP__