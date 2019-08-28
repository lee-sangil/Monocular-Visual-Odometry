#include "core/MVO.hpp"

void MVO::plot(){
	cv::Mat img(this->cur_image.size(), CV_8UC3);
	cvtColor(this->cur_image, img, CV_GRAY2BGR);

	// buckets
	for( int c = 0; c < this->bucket.grid.width; c++ ){
		cv::line(img, cv::Point(c*this->bucket.size.width,0), cv::Point(c*this->bucket.size.width,this->params.height), cv::Scalar(180,180,180), 1, 16);
	}

	for( int r = 0; r < this->bucket.grid.height; r++ ){
		cv::line(img, cv::Point(0,r*this->bucket.size.height), cv::Point(this->params.width,r*this->bucket.size.height), cv::Scalar(180,180,180), 1, 16);
	}

	// feature points
	for( uint32_t i = 0; i < this->nFeature; i++ ){
		cv::circle(img, cv::Point(this->features[i].uv.back().x, this->features[i].uv.back().y), 3, cv::Scalar(0,255,0), 1);
		// cv::rectangle(img, cv::Point(11+12*i,460), cv::Point(20+12*i,470), this->param.cmap[i],(this->sgm.isObserved[i])?-1:1);
	}
	cv::imshow("MVO", img);
}