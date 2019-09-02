#include "core/MVO.hpp"

void MVO::plot(){
	/*******************************************
	 * 			Image seen by camera
	 * *****************************************/
	cv::Mat img(this->cur_image.size(), CV_8UC3);
	cvtColor(this->cur_image, img, CV_GRAY2BGR);

	// buckets
	for( int c = 0; c < this->bucket.grid.width; c++ ){
		cv::line(img, cv::Point(c*this->bucket.size.width,0), cv::Point(c*this->bucket.size.width,this->params.imSize.height), cv::Scalar(180,180,180), 1, 16);
	}

	for( int r = 0; r < this->bucket.grid.height; r++ ){
		cv::line(img, cv::Point(0,r*this->bucket.size.height), cv::Point(this->params.imSize.width,r*this->bucket.size.height), cv::Scalar(180,180,180), 1, 16);
	}

	// feature points
	for( int i = 0; i < this->nFeature; i++ ){
		cv::circle(img, cv::Point(this->features[i].uv.back().x, this->features[i].uv.back().y), 3, cv::Scalar(0,255,0), 1);
		// cv::rectangle(img, cv::Point(11+12*i,460), cv::Point(20+12*i,470), this->param.cmap[i],(this->sgm.isObserved[i])?-1:1);
	}
	cv::imshow("MVO", img);

	/*******************************************
	 * 				Trajectory
	 * *****************************************/
	cv::Mat traj = cv::Mat::zeros(600,600,CV_8UC3);

	// feature points
	for( int i = 0; i < this->nFeature; i++ ){
		if( this->features[i].is_3D_init )
			cv::circle(traj, cv::Point(300 + this->params.plotScale * this->features[i].point(0), 300 - this->params.plotScale * this->features[i].point(2)), 1, cv::Scalar(255,255,255), CV_FILLED);
	}

	Eigen::Matrix4d currTco = this->TocRec.back().inverse();
	Eigen::Matrix4d prevTco = Eigen::Matrix4d::Identity();
	Eigen::Matrix4d nextTco;
	for( uint32_t i = 0; i < this->TocRec.size(); i++ ){
		nextTco = currTco * this->TocRec[i];
		cv::line(traj, cv::Point2f(300 + this->params.plotScale * prevTco(0,3), 300 - this->params.plotScale * prevTco(2,3)), 
						cv::Point2f(300 + this->params.plotScale * nextTco(0,3), 300 - this->params.plotScale * nextTco(2,3)), cv::Scalar(0,0,255), 2);
		prevTco = nextTco;
	}
	cv::circle(traj, cv::Point(300, 300), 3, cv::Scalar(0,0,255), 3);

	cv::imshow("Trajectory", traj);
}