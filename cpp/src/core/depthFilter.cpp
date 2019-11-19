#include "core/depthFilter.hpp"

void depthFilter::update(double measurement, double dt){
	double C1 = a / (a+b) / std::sqrt(2*M_PI*(variance*variance+dt*dt)) * std::exp(-(measurement - mean)*(measurement - mean)/2/(variance*variance+dt*dt));
	double C2 = b / (a+b) / (max-min);
	double C12 = C1 + C2;
	
	C1 = C1 / C12;
	C2 = C2 / C12;

	double s2 = 1/(1/(variance*variance)+1/(dt*dt));
	double m = s2 * (mean / (variance*variance) + measurement / (dt*dt));

	double mean_new = C1 * m + C2 * mean;
	double variance_new = std::sqrt( C1 * (s2 + m*m) + C2 * (variance*variance + mean*mean) - mean_new*mean_new);

	double F = C1 * (a+1)/(a+b+1) + C2 * a/(a+b+1);
	double E = C1 * (a+1)/(a+b+1) * (a+2)/(a+b+2) + C2 * a/(a+b+1) * (a+1)/(a+b+2);

	this->a = (E-F)/(F-E/F);
	this->b = this->a * (1-F)/F;
	this->mean = mean_new;
	this->variance = variance_new;
}

double depthFilter::get_mean() const {
	return this->mean;
}