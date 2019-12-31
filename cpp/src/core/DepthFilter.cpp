#include "core/DepthFilter.hpp"

/**
 * @brief 측정치와 그의 표준 편차를 입력으로 하여 깊이 필터의 가우시안 분포의 파라미터를 업데이트함.
 * @details 알고리즘에서 meas로 inverse-depth를, tau로 inverse-depth의 표준 편차를 입력으로 한다.
 * @param meas 깊이 필터에서 추정하고자 하는 값의 측정치
 * @param tau 측정치의 표준 편차
 * @return 없음
 * @author Sangil Lee (sangillee724@gmail.com)
 * @date 25-Dec-2019
 */
void DepthFilter::update(const double meas, const double tau){
	
	if( initialize_ ){
		double tau2 = tau * tau;
		double sigma2 = sigma_ * sigma_;
		double C1 = a_ / (a_+b_) / std::sqrt(2*M_PI*(sigma2+tau2)) * std::exp(-(meas-mean_)*(meas-mean_)/2/(sigma2+tau2));
		double C2 = b_ / (a_+b_) / (DepthFilter::s_meas_max_-0);
		double C12 = C1 + C2;
		
		C1 = C1 / C12;
		C2 = C2 / C12;

		double s2 = 1 / ( 1/sigma2 + 1/tau2 );
		double m = s2 * (mean_/sigma2 + meas/tau2);

		double mean_new = C1 * m + C2 * mean_;
		double sigma_new = std::sqrt( C1 * (s2 + m*m) + C2 * (sigma2 + mean_*mean_) - mean_new*mean_new);

		double F = C1 * (a_+1)/(a_+b_+1) + C2 * a_/(a_+b_+1);
		double E = C1 * (a_+1)/(a_+b_+1) * (a_+2)/(a_+b_+2) + C2 * a_/(a_+b_+1) * (a_+1)/(a_+b_+2);

		a_ = (E-F)/(F-E/F);
		b_ = a_ * (1-F)/F;
		mean_ = mean_new;
		sigma_ = sigma_new;
	}else{
		mean_ = meas;
		sigma_ = tau; // 1/DEPTH_MIN/DEPTH_MIN/36
		initialize_ = true;
	}
}

double DepthFilter::getMean() const {return mean_;}
double DepthFilter::getVariance() const {return sigma_ * sigma_;}
double DepthFilter::getA() const {return a_;}
double DepthFilter::getB() const {return b_;}

/**
 * @brief 깊이 필터 강제 초기화.
 * @return 없음
 * @author Sangil Lee (sangillee724@gmail.com)
 * @date 25-Dec-2019
 */
void DepthFilter::reset() {initialize_ = false; sigma_ = 1e9;}

/**
 * @brief 측정치의 표준 편차를 계산함.
 * @details 3차원 좌표의 위치와 변환 행렬을 바탕으로 3차원 깊이의 표준 편차를 계산한다.
 * @param Toc 3차원 좌표를 복원할 때의 변환 행렬
 * @param p 3차원 좌표 위치 벡터
 * @return 3차원 좌표 깊이값의 표준 편차
 * @author Sangil Lee (sangillee724@gmail.com)
 * @date 25-Dec-2019
 */
double DepthFilter::computeTau(const Eigen::Matrix4d& Toc, const Eigen::Vector3d& p){
	Eigen::Vector3d t = Toc.block(0,3,3,1);
	Eigen::Vector3d a = p-t;
	double t_norm = t.norm();
	double a_norm = a.norm();
	double p_norm = p.norm();
	double alpha = acos(p.dot(t)/(t_norm*p_norm)); // dot product
	double beta = acos(a.dot(-t)/(t_norm*a_norm)); // dot product
	double beta_plus = beta + DepthFilter::s_px_error_angle_;
	double gamma_plus = M_PI-alpha-beta_plus; // triangle angles sum to PI
	double z_plus = t_norm*sin(beta_plus)/sin(gamma_plus); // law of sines
	return std::abs(z_plus - p(2)); // tau
}

/**
 * @brief 측정치의 역함수의 표준 편차를 계산함.
 * @details 깊이값 z와 그 표준편차에 따른 inverse-depth의 표준 편차를 계산한다.
 * @param z 3차원 좌표의 깊이값
 * @param tau 3차원 좌표 깊이값의 표준 편차
 * @return 3차원 좌표 깊이 역수의 표준 편차
 * @author Sangil Lee (sangillee724@gmail.com)
 * @date 25-Dec-2019
 */
double DepthFilter::computeInverseTau(const double z, const double tau){
	// return tau*tau/std::pow(z,4)+2*tau*tau*tau*tau/std::pow(z,6);
	return std::max(1.0/(z-tau)-1.0/z, 1.0/z-1.0/(z+tau));
	// return (0.5 * (1.0/std::max(0.0000001, z-tau) - 1.0/(z+tau)));
}