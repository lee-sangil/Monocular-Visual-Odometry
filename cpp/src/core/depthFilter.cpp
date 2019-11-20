#include "core/depthFilter.hpp"

depthFilter::max = 1.0/DEPTH_MIN;

void depthFilter::update(double meas, double tau){
	
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