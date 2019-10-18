#include "core/MVO.hpp"

void MVO::updateView(){
	// feature points
	// double plotScale;
	// int n = 0;
	// for( int i = (int) this->TRec.size()-1; i >= std::max( (int) this->TRec.size()-10, 0 ); i--, n++ ){
	// 	plotScale += std::min(std::max(this->TRec[i].block(0,3,3,1).norm(), 0.01), 100.0);
	// }
	// plotScale = n/plotScale;

	Eigen::Matrix3d rotx, rotz;
	rotx << 1, 0, 0,
			0, std::cos(this->params.view.pitch), -std::sin(this->params.view.pitch),
			0, std::sin(this->params.view.pitch), std::cos(this->params.view.pitch);
	rotz << std::cos(this->params.view.roll), -std::sin(this->params.view.roll), 0,
			std::sin(this->params.view.roll), std::cos(this->params.view.roll), 0,
			0, 0, 1;
	this->params.view.R = (rotz * rotx).transpose();
	this->params.view.t = -(Eigen::Vector3d() << 0,0,-this->params.view.height).finished();
	this->params.view.P = (Eigen::Matrix<double,3,4>() << this->params.view.K * this->params.view.R, this->params.view.K * this->params.view.t).finished();
}

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
	}
	cv::imshow("MVO", img);

	/*******************************************
	 * 				Trajectory
	 * *****************************************/
	this->updateView();

	cv::Mat traj = cv::Mat::zeros(this->params.view.imSize.height,this->params.view.imSize.width,CV_8UC3);

	// Points
	Eigen::Vector4d point;
	Eigen::Vector3d uv;
	for( uint32_t i = 0; i < this->features_dead.size(); i++ ){
		if( this->features_dead[i].is_3D_init ){
			point = this->TocRec.back().inverse() * this->features_dead[i].point_init;
			uv = this->params.view.P * point;
			if( uv(2) > 1 )
				cv::circle(traj, cv::Point(uv(0)/uv(2), uv(1)/uv(2)), 1, cv::Scalar(128,128,128), CV_FILLED);
		}
	}
	for( uint32_t i = 0; i < this->features.size(); i++ ){
		if( this->features[i].is_3D_init ){
			point = this->TocRec.back().inverse() * this->features[i].point_init;
			uv = this->params.view.P * point;
			if( uv(2) > 1 ){
				cv::circle(traj, cv::Point(uv(0)/uv(2), uv(1)/uv(2)), 1, cv::Scalar(128,128,128), CV_FILLED);
				if( this->features[i].is_3D_reconstructed ){
					if( this->features[i].frame_init < this->step )
						cv::circle(traj, cv::Point(uv(0)/uv(2), uv(1)/uv(2)), 3, cv::Scalar(0,200,255));
				}
			}
		}
		if( this->features[i].is_3D_reconstructed){
			uv = this->params.view.P * this->features[i].point;
			if( uv(2) > 1 )
				cv::circle(traj, cv::Point(uv(0)/uv(2), uv(1)/uv(2)), 2, cv::Scalar(0,255,0), CV_FILLED);
		}
	}

	// Trajectory
	Eigen::Matrix4d currTco = this->TocRec.back().inverse();
	Eigen::Matrix4d prevTco = Eigen::Matrix4d::Identity();
	Eigen::Matrix4d nextTco;
	Eigen::Vector4d prevPos, nextPos;
	Eigen::Vector3d uv_next;
	for( uint32_t i = 0; i < this->TocRec.size()-1; i++ ){
		prevTco = currTco * this->TocRec[i];
		nextTco = currTco * this->TocRec[i+1];
		prevPos = prevTco.block(0,3,3,1);
		nextPos = nextTco.block(0,3,3,1);
		uv = this->params.view.P * prevPos;
		uv_next = this->params.view.P * nextPos;
		if( uv(2) > 1 && uv_next(2) > 1)
			cv::line(traj, cv::Point(uv(0)/uv(2), uv(1)/uv(2)), cv::Point(uv_next(0)/uv_next(2), uv_next(1)/uv_next(2)), cv::Scalar(0,0,255), 2);
	}

	// Camera
	Eigen::Vector3d uv0, uv1, uv2, uv3, uv4;
	uv0 = this->params.view.P * (Eigen::Vector4d() << 0,0,0,1).finished();
	uv1 = this->params.view.P * (Eigen::Vector4d() << this->params.view.upperLeft.x,this->params.view.upperLeft.y,this->params.view.upperLeft.z,1).finished();
	uv2 = this->params.view.P * (Eigen::Vector4d() << this->params.view.upperRight.x,this->params.view.upperRight.y,this->params.view.upperRight.z,1).finished();
	uv3 = this->params.view.P * (Eigen::Vector4d() << this->params.view.lowerRight.x,this->params.view.lowerRight.y,this->params.view.lowerRight.z,1).finished();
	uv4 = this->params.view.P * (Eigen::Vector4d() << this->params.view.lowerLeft.x,this->params.view.lowerLeft.y,this->params.view.lowerLeft.z,1).finished();
	cv::line(traj, cv::Point(uv0(0)/uv0(2), uv0(1)/uv0(2)), cv::Point(uv1(0)/uv1(2), uv1(1)/uv1(2)), cv::Scalar(0,0,255), 2);
	cv::line(traj, cv::Point(uv0(0)/uv0(2), uv0(1)/uv0(2)), cv::Point(uv2(0)/uv2(2), uv2(1)/uv2(2)), cv::Scalar(0,0,255), 2);
	cv::line(traj, cv::Point(uv0(0)/uv0(2), uv0(1)/uv0(2)), cv::Point(uv3(0)/uv3(2), uv3(1)/uv3(2)), cv::Scalar(0,0,255), 2);
	cv::line(traj, cv::Point(uv0(0)/uv0(2), uv0(1)/uv0(2)), cv::Point(uv4(0)/uv4(2), uv4(1)/uv4(2)), cv::Scalar(0,0,255), 2);
	cv::line(traj, cv::Point(uv1(0)/uv1(2), uv1(1)/uv1(2)), cv::Point(uv2(0)/uv2(2), uv2(1)/uv2(2)), cv::Scalar(0,0,255), 2);
	cv::line(traj, cv::Point(uv2(0)/uv2(2), uv2(1)/uv2(2)), cv::Point(uv3(0)/uv3(2), uv3(1)/uv3(2)), cv::Scalar(0,0,255), 2);
	cv::line(traj, cv::Point(uv3(0)/uv3(2), uv3(1)/uv3(2)), cv::Point(uv4(0)/uv4(2), uv4(1)/uv4(2)), cv::Scalar(0,0,255), 2);
	cv::line(traj, cv::Point(uv4(0)/uv4(2), uv4(1)/uv4(2)), cv::Point(uv1(0)/uv1(2), uv1(1)/uv1(2)), cv::Scalar(0,0,255), 2);
	cv::imshow("Trajectory", traj);
}