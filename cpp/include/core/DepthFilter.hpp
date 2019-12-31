#ifndef __DEPTHFILTER_HPP__
#define __DEPTHFILTER_HPP__

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

/**
 * @brief 깊이 필터 클래스.
 * @details 깊이 필터는 입력되는 깊이값이 가우시안과 유니폼의 합으로 분포되어 있다고 가정한다. 가우시안 분포의 평균값은 깊이의 참값이며, 유니폼은 깊이값의 이상값(outlier) 분포를 의미한다. 깊이 필터는 1차원 깊이값과 분산을 입력받아 모델을 업데이트하며, 빠른 속도로 수렴하며 깊이의 참값을 추정한다.
 * @author Sangil Lee (sangillee724@gmail.com)
 * @date 24-Dec-2019
 */
class DepthFilter{ // inverse-depth estimator
	private:

	bool initialize_ = false; /**< @brief 깊이 필터 초기화 파라미터 */
	double mean_; /**< @brief 가우시안 평균값 */
	double sigma_ = 1e9; /**< @brief 가우시안 표준 편차 */
	double a_ = 10; /**< @brief 가우시안 분포의 비중 */
	double b_ = 10; /**< @brief 유니폼 분포의 비중 */
	
	public:

	static double s_px_error_angle_; /**< @brief 픽셀 노이즈에 따른 시차 분산 크기 */
	static double s_meas_max_; /**< @brief 측정치의 최대값 */
	// static double min = 0; // 1/inf

	public:

	DepthFilter(){};
	void seed(const double a, const double b){this->a_ = a; this->b_ = b;};

	void update(const double meas, const double tau);
	double getMean() const;
	double getVariance() const;
	double getA() const;
	double getB() const;
	void reset();

	static double computeTau(const Eigen::Matrix4d& Toc, const Eigen::Vector3d& p);
	static double computeInverseTau(const double z, const double tau);
};

#endif