#ifndef __COMMON_HPP__
#define __COMMON_HPP__

#include <vector>
#include <memory>
#include <eigen3/Eigen/Core>
#include <opencv2/core/core.hpp>

#define D_METER 1.2 // for key interrupt
#define D_RADIAN M_PI/24 // for key interrupt
#define YAML_VERSION 2.5 // compatibility check

/**
 * @brief 특징점의 타입 열거 클래스.
 * @details Unknown은 특징이 따로 정해지지 않은 종류이며, Dynamic은 자동차나 보행자에 속하는 특징점을 분류하기 위한 용도이고, Road는 지면에 속하는 특징점이다.
 * @author Sangil Lee (sangillee724@gmail.com)
 * @date 24-Dec-2019
 */
enum Type{Unknown,Dynamic,Road,};

class DepthFilter;

/**
 * @brief 랜드마크 클래스.
 * @details 랜드마크를 BA에 활용하기 위해 만들어진 클래스이다.
 * @author Sangil Lee (sangillee724@gmail.com)
 * @date 13-May-2020
 */
typedef struct Landmark{
	Eigen::Vector4d point_init; /**< @brief 월드 좌표계에서 특징점 객체의 3차원 좌표 */
	Type type; /**< @brief 특징점의 타입 */
	double std;

	static uint32_t getNewID() {return new_landmark_id++;}; /**< @brief 랜드마크가 생성되는 순에 따라 부여되는 id */
	static uint32_t new_landmark_id; /**< @brief 특징점이 생성되는 순에 따라 부여되는 id */
}Landmark;

/**
 * @brief 특징점 클래스.
 * @details 특징점이 독립적으로 추출 및 추적되고, 2차원, 3차원 모션 계산에 사용되도록 만들어진 클래스이다.
 * @author Sangil Lee (sangillee724@gmail.com)
 * @date 24-Dec-2019
 */
typedef struct Feature{
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	uint32_t id; /**< @brief 특징점 객체가 부여받은 id */
	uint32_t life; /**< @brief 특징점 객체가 나타나는 프레임 횟수 */
	int frame_2d_init; /**< @brief 특징점 객체가 처음 추적되기 시작한 step */
	int frame_3d_init; /**< @brief 특징점 객체의 3차원 좌표가 처음 생성되었을때의 step */
	// cv::Mat desc;
	double parallax; /**< @brief 추적된 특징점 객체가 이미지 내에서 나타내는 시차 */
	std::vector< cv::Point2f > uv; /**< @brief 이미지 내에서 특징점 객체가 추적된 궤적 */
	cv::Point2f uv_pred; /**< @brief 지난 uv를 이용하여 예측한 현재 프레임에서의 uv 위치 */
	cv::Point bucket; /**< @brief 특징점이 속해있는 bucket의 위치 */
	Eigen::Vector4d point_curr; /**< @brief 현재 프레임에서 특징점 객체의 3차원 좌표 */
	bool is_alive; /**< @brief 다음 회차에서 삭제할지 */
	bool is_wide; /**< @brief 시차가 충분이 커졌는지 */
	bool is_matched; /**< @brief 인접한 프레임에서 추적되었는지 */
	bool is_2D_inliered; /**< @brief essential constraint를 만족하는지 */
	bool is_3D_reconstructed; /**< @brief 3차원 좌표가 복원되었는지 */
	Type type; /**< @brief 특징점의 타입 */
	std::shared_ptr<DepthFilter> depthfilter; /**< @brief 깊이 필터 클래스 */
	std::shared_ptr<Landmark> landmark; /**< @brief 유효한 3차원 좌표가 생성되었는지 */

	static uint32_t getNewID() {return new_feature_id++;}; /**< @brief 특징점이 생성되는 순에 따라 부여되는 id */
	static uint32_t new_feature_id; /**< @brief 특징점이 생성되는 순에 따라 부여되는 id */
}Feature;

namespace std{
	template<class _InputIterator, class T>
	std::vector<_InputIterator>
	find_all(_InputIterator begin, _InputIterator end, const T& val)
	{
		std::vector<_InputIterator> matches;
		while(begin != end)
		{
			if((*begin) == val)
				matches.push_back(begin);
			++begin;
		}
		
		return matches;
	}

	template<class _InputIterator, class _Predicate>
	std::vector<_InputIterator>
	find_all_if(_InputIterator begin, _InputIterator end, _Predicate pred)
	{
		std::vector<_InputIterator> matches;
		while(begin != end)
		{
			if(pred(*begin))
				matches.push_back(begin);
			++begin;
		}
		
		return matches;
	}
}

#endif //__COMMON_HPP__
