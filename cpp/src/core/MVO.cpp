#include "core/MVO.hpp"

unsigned int Feature::new_feature_id = 1;

MVO::MVO(){
	this->step = 0;

	this->bucket = Bucket();
}
MVO::MVO(Parameter params){
	this->params.fx = params.fx;
	this->params.fy = params.fy;
	this->params.cx = params.cx;
	this->params.cy = params.cy;
	this->params.k1 = params.k1;
	this->params.k2 = params.k2;
	this->params.p1 = params.p1;
	this->params.p2 = params.p2;
	this->params.k3 = params.k3;
	this->params.width = params.width;
	this->params.height = params.height;

	this->params.K << params.fx, 0, params.cx,
						0, params.fy, params.cy,
						0, 0, 1;

	this->params.imSize.push_back(params.width);
	this->params.imSize.push_back(params.height);
	this->params.radialDistortion.push_back(params.k1);
	this->params.radialDistortion.push_back(params.k2);
	this->params.radialDistortion.push_back(params.k3);
	this->params.tangentialDistortion.push_back(params.p1);
	this->params.tangentialDistortion.push_back(params.p2);
}

void MVO::set_image(const cv::Mat image){
	this->prev_image = this->cur_image.clone();
	this->cur_image = image.clone(); 
}