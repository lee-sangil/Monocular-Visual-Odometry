#include "core/depthFilter.hpp"

depthFilter::max = 1.0/DEPTH_MIN;

void depthFilter::update(const double meas, const double tau){
	
	if( this->initialize ){
		double tau2 = tau * tau;
		double sigma2 = this->sigma * this->sigma;
		double C1 = a / (a+b) / std::sqrt(2*M_PI*(sigma2+tau2)) * std::exp(-(meas-mean)*(meas-mean)/2/(sigma2+tau2));
		double C2 = b / (a+b) / (max-min);
		double C12 = C1 + C2;
		
		C1 = C1 / C12;
		C2 = C2 / C12;

		double s2 = 1 / ( 1/sigma2 + 1/tau2 );
		double m = s2 * (mean/sigma2 + meas/tau2);

		double mean_new = C1 * m + C2 * mean;
		double sigma_new = std::sqrt( C1 * (s2 + m*m) + C2 * (sigma2 + mean*mean) - mean_new*mean_new);

		double F = C1 * (a+1)/(a+b+1) + C2 * a/(a+b+1);
		double E = C1 * (a+1)/(a+b+1) * (a+2)/(a+b+2) + C2 * a/(a+b+1) * (a+1)/(a+b+2);

		this->a = (E-F)/(F-E/F);
		this->b = this->a * (1-F)/F;
		this->mean = mean_new;
		this->sigma = sigma_new;
	}else{
		this->mean = meas;
		this->sigma = tau; // 1/DEPTH_MIN/DEPTH_MIN/36
		this->initialize = true;
	}
}

double depthFilter::get_mean() const {
	return this->mean;
}

double depthFilter::computeTau(const Eigen::Matrix4d& Toc, const Eigen::Vector3d& uv, const double z, const double px_error_angle){
	Eigen::Vector3d t = Toc.block(0,3,3,1);
	Eigen::Vector3d a = uv*z-t;
	double t_norm = t.norm();
	double a_norm = a.norm();
	double alpha = acos(f.dot(t)/t_norm); // dot product
	double beta = acos(a.dot(-t)/(t_norm*a_norm)); // dot product
	double beta_plus = beta + px_error_angle;
	double gamma_plus = PI-alpha-beta_plus; // triangle angles sum to PI
	double z_plus = t_norm*sin(beta_plus)/sin(gamma_plus); // law of sines
	return (z_plus - z); // tau
}

double depthFilter::computeInverseTau(const double z, const double tau){
	return (0.5 * (1.0/std::max(0.0000001, z-tau) - 1.0/(z+tau)));
}