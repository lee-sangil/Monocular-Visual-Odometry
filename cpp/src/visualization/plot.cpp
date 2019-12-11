#include "core/MVO.hpp"

#define RATIO 0.5
#define GAP 30

void MVO::updateView(){
	Eigen::Matrix3d rotx, rotz;
	rotx << 1, 0, 0,
			0, std::cos(params_.view.pitch), -std::sin(params_.view.pitch),
			0, std::sin(params_.view.pitch), std::cos(params_.view.pitch);
	rotz << std::cos(params_.view.roll), -std::sin(params_.view.roll), 0,
			std::sin(params_.view.roll), std::cos(params_.view.roll), 0,
			0, 0, 1;
	params_.view.R = (rotz * rotx).transpose();
	params_.view.t = -(Eigen::Vector3d() << 0,0,-params_.view.height).finished();
	params_.view.P = (Eigen::Matrix<double,3,4>() << params_.view.K * params_.view.R, params_.view.K * params_.view.t).finished();
}

void MVO::plot() const {
	/*******************************************
	 * 		Figure 1: Image seen by camera
	 * *****************************************/
	cv::Mat img(curr_frame_.size(), CV_8UC3);
	cvtColor(curr_frame_.image, img, CV_GRAY2BGR);

	double ratio = 1;
	if( img.rows > 500 ){
		ratio = RATIO;
		cv::resize(img, img, cv::Size(img.cols*ratio, img.rows*ratio));
	}

	cv::Mat mvo(cv::Size(img.cols + img.cols*0.5 + 3*GAP, img.rows + 2*GAP), CV_8UC3, cv::Scalar::all(0));

	// buckets
	for( int c = 0; c < bucket_.grid.width; c++ ){
		cv::line(img, cv::Point(c*bucket_.size.width*ratio,0), cv::Point(c*bucket_.size.width*ratio,params_.im_size.height*ratio), cv::Scalar(180,180,180), 1, 16);
	}

	for( int r = 0; r < bucket_.grid.height; r++ ){
		cv::line(img, cv::Point(0,r*bucket_.size.height*ratio), cv::Point(params_.im_size.width*ratio,r*bucket_.size.height*ratio), cv::Scalar(180,180,180), 1, 16);
	}

	// feature points
	for( int i = 0; i < num_feature_; i++ ){
		if( features_[i].type == Type::Dynamic || features_[i].is_2D_inliered == false ){
			cv::circle(img, cv::Point(features_[i].uv.back().x*ratio, features_[i].uv.back().y*ratio), 3, cv::Scalar(255,0,0), 1);
			if( features_[i].uv_pred.x > 0 && features_[i].uv_pred.y > 0 )
				cv::drawMarker(img, cv::Point(features_[i].uv_pred.x*ratio, features_[i].uv_pred.y*ratio), cv::Scalar(255,0,0), cv::MARKER_CROSS, 5);
		}else if( features_[i].type == Type::Road ){
			cv::circle(img, cv::Point(features_[i].uv.back().x*ratio, features_[i].uv.back().y*ratio), 3, cv::Scalar(50,50,255), 1);
			if( features_[i].uv_pred.x > 0 && features_[i].uv_pred.y > 0 )
				cv::drawMarker(img, cv::Point(features_[i].uv_pred.x*ratio, features_[i].uv_pred.y*ratio), cv::Scalar(50,50,255), cv::MARKER_CROSS, 5);
		}else{
			cv::circle(img, cv::Point(features_[i].uv.back().x*ratio, features_[i].uv.back().y*ratio), 3, cv::Scalar(0,200,0), 1);
			if( features_[i].uv_pred.x > 0 && features_[i].uv_pred.y > 0 )
				cv::drawMarker(img, cv::Point(features_[i].uv_pred.x*ratio, features_[i].uv_pred.y*ratio), cv::Scalar(0,200,0), cv::MARKER_CROSS, 5);
		}
		if( MVO::s_file_logger.is_open() ){
			int key_idx = features_[i].life - 1 - (step_ - keystep_);
			if( key_idx >= 0 )
				cv::line(img, features_[i].uv.back()*ratio, features_[i].uv[key_idx]*ratio, cv::Scalar::all(0), 1, CV_AA);
			cv::putText(img, std::to_string(features_[i].id), features_[i].uv.back()*ratio, cv::FONT_HERSHEY_DUPLEX, 0.4, cv::Scalar(0,255,0), 1, CV_AA);
		}
	}
	img.copyTo(mvo(cv::Rect(GAP,GAP,img.cols,img.rows)));

	/*******************************************
	 * 		Figure 1: Keyframe
	 * *****************************************/
	cv::Mat img_keyframe(curr_keyframe_.size(), CV_8UC3);
	cvtColor(curr_keyframe_.image, img_keyframe, CV_GRAY2BGR);
	cv::resize(img_keyframe, img_keyframe, cv::Size(img.cols*0.5,img.rows*0.5));
	img_keyframe.copyTo(mvo(cv::Rect(img.cols+2*GAP,GAP,img_keyframe.cols,img_keyframe.rows)));

	/*******************************************
	 * 		Figure 1: Reconstructed Depth
	 * *****************************************/
	cv::Mat distance(img.rows*0.5, img.cols*0.5, CV_8UC3, cv::Scalar::all(0));
	Eigen::Matrix4d Tco = TocRec_.back().inverse();
	int r,g,b, depth;
	Eigen::Vector4d point;
	for( uint32_t i = 0; i < features_.size(); i++ ){
		if( params_.output_filtered_depth ){
			if( features_[i].is_3D_init && features_[i].type != Type::Dynamic){
				point = Tco * features_[i].point_init;
				depth = (int) point(2);

				r = std::exp(-depth/150) * std::min(depth*18, 255);
				g = std::exp(-depth/150) * std::max(255 - depth*8, 30);
				b = std::exp(-depth/150) * std::max(100 - depth, 0);
				cv::circle(distance, cv::Point(features_[i].uv.back().x*ratio*0.5, features_[i].uv.back().y*ratio*0.5), std::ceil(5*ratio), cv::Scalar(b, g, r), CV_FILLED);
			}
		}else{
			if( features_[i].is_3D_reconstructed && features_[i].type != Type::Dynamic ){
				point = features_[i].point_curr;
				depth = point(2);

				r = std::exp(-depth/150) * std::min(depth*18, 255);
				g = std::exp(-depth/150) * std::max(255 - depth*8, 30);
				b = std::exp(-depth/150) * std::max(100 - depth, 0);
				cv::circle(distance, cv::Point(features_[i].uv.back().x*ratio*0.5, features_[i].uv.back().y*ratio*0.5), std::ceil(5*ratio), cv::Scalar(b, g, r), CV_FILLED);
			}
		}
	}
	distance.copyTo(mvo(cv::Rect(img.cols+2*GAP,img_keyframe.rows+2*GAP,distance.cols,distance.rows)));

	cv::putText(mvo, "Features", cv::Point(img.cols*0.5 + GAP,GAP*0.7), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar::all(255), 1, CV_AA);
	cv::putText(mvo, "Keyframe", cv::Point(img_keyframe.cols*0.5+img.cols+2*GAP,GAP*0.7), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar::all(255), 1, CV_AA);
	cv::putText(mvo, "Depth", cv::Point(img_keyframe.cols*0.5+img.cols+2*GAP,img_keyframe.rows+GAP*1.7), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar::all(255), 1, CV_AA);
	cv::imshow("MVO", mvo);

	/*******************************************
	 * 		Figure 2: Trajectory
	 * *****************************************/
	cv::Mat traj = cv::Mat::zeros(params_.view.im_size.height,params_.view.im_size.width,CV_8UC3);

	// Points
	Eigen::Vector3d uv;
	for( uint32_t i = 0; i < features_dead_.size(); i++ ){
		if( features_dead_[i].is_3D_init ){
			point = Tco * features_dead_[i].point_init;
			uv = params_.view.P * point;
			if( uv(2) > 1 ){
				switch (features_dead_[i].type){
				case Type::Unknown:
					cv::circle(traj, cv::Point(uv(0)/uv(2), uv(1)/uv(2)), 1, cv::Scalar(128,128,128), CV_FILLED);
					break;
				case Type::Road:
					cv::circle(traj, cv::Point(uv(0)/uv(2), uv(1)/uv(2)), 1, cv::Scalar(128,128,200), CV_FILLED);
					break;
				case Type::Dynamic:
					cv::circle(traj, cv::Point(uv(0)/uv(2), uv(1)/uv(2)), 1, cv::Scalar(100,50,50), CV_FILLED);
					break;
				}
			}
		}
	}
	for( uint32_t i = 0; i < features_.size(); i++ ){
		if( features_[i].is_3D_init ){
			point = Tco * features_[i].point_init;
			uv = params_.view.P * point;
			if( uv(2) > 1 ){
				switch (features_[i].type){
				case Type::Unknown:
					cv::circle(traj, cv::Point(uv(0)/uv(2), uv(1)/uv(2)), 1, cv::Scalar(128,128,128), CV_FILLED);
					break;
				case Type::Road:
					cv::circle(traj, cv::Point(uv(0)/uv(2), uv(1)/uv(2)), 1, cv::Scalar(128,128,200), CV_FILLED);
					break;
				case Type::Dynamic:
					cv::circle(traj, cv::Point(uv(0)/uv(2), uv(1)/uv(2)), 1, cv::Scalar(100,50,50), CV_FILLED);
					break;
				}
				if( params_.output_filtered_depth ){
					if( features_[i].is_3D_init && features_[i].frame_3d_init < step_ && features_[i].type != Type::Dynamic )
						cv::circle(traj, cv::Point(uv(0)/uv(2), uv(1)/uv(2)), 1, cv::Scalar(0,255,0), CV_FILLED);
				}else{
					if( features_[i].is_3D_reconstructed && features_[i].frame_3d_init < step_ && features_[i].type != Type::Dynamic )
						cv::circle(traj, cv::Point(uv(0)/uv(2), uv(1)/uv(2)), 1, cv::Scalar(0,255,0), CV_FILLED);
				}
			}
		}
	}

	// Trajectory
	// Eigen::Matrix4d prevTco = Eigen::Matrix4d::Identity();
	// Eigen::Matrix4d nextTco;
	// Eigen::Vector4d prevPos, nextPos;
	// Eigen::Vector3d uv_next;
	// for( uint32_t i = 0; i < TocRec_.size()-1; i++ ){
	// 	prevTco = Tco * TocRec_[i];
	// 	nextTco = Tco * TocRec_[i+1];
	// 	prevPos = prevTco.block(0,3,4,1);
	// 	nextPos = nextTco.block(0,3,4,1);
	// 	uv = params_.view.P * prevPos;
	// 	uv_next = params_.view.P * nextPos;
	// 	if( uv(2) > 1 && uv_next(2) > 1)
	// 		cv::line(traj, cv::Point(uv(0)/uv(2), uv(1)/uv(2)), cv::Point(uv_next(0)/uv_next(2), uv_next(1)/uv_next(2)), cv::Scalar(0,0,255), 2);
	// }

	// Camera
	Eigen::Vector3d uv0, uv1, uv2, uv3, uv4;
	uv0 = params_.view.P * (Eigen::Vector4d() << 0,0,0,1).finished();
	uv1 = params_.view.P * (Eigen::Vector4d() << params_.view.upper_left.x,params_.view.upper_left.y,params_.view.upper_left.z,1).finished();
	uv2 = params_.view.P * (Eigen::Vector4d() << params_.view.upper_right.x,params_.view.upper_right.y,params_.view.upper_right.z,1).finished();
	uv3 = params_.view.P * (Eigen::Vector4d() << params_.view.lower_right.x,params_.view.lower_right.y,params_.view.lower_right.z,1).finished();
	uv4 = params_.view.P * (Eigen::Vector4d() << params_.view.lower_left.x,params_.view.lower_left.y,params_.view.lower_left.z,1).finished();
	if( uv0(2) > 1 && uv1(2) > 1 && uv2(2) > 1 && uv3(2) > 1 && uv4(2) > 1){
		cv::line(traj, cv::Point(uv0(0)/uv0(2), uv0(1)/uv0(2)), cv::Point(uv1(0)/uv1(2), uv1(1)/uv1(2)), cv::Scalar(0,0,255), 2);
		cv::line(traj, cv::Point(uv0(0)/uv0(2), uv0(1)/uv0(2)), cv::Point(uv2(0)/uv2(2), uv2(1)/uv2(2)), cv::Scalar(0,0,255), 2);
		cv::line(traj, cv::Point(uv0(0)/uv0(2), uv0(1)/uv0(2)), cv::Point(uv3(0)/uv3(2), uv3(1)/uv3(2)), cv::Scalar(0,0,255), 2);
		cv::line(traj, cv::Point(uv0(0)/uv0(2), uv0(1)/uv0(2)), cv::Point(uv4(0)/uv4(2), uv4(1)/uv4(2)), cv::Scalar(0,0,255), 2);
		cv::line(traj, cv::Point(uv1(0)/uv1(2), uv1(1)/uv1(2)), cv::Point(uv2(0)/uv2(2), uv2(1)/uv2(2)), cv::Scalar(0,0,255), 2);
		cv::line(traj, cv::Point(uv2(0)/uv2(2), uv2(1)/uv2(2)), cv::Point(uv3(0)/uv3(2), uv3(1)/uv3(2)), cv::Scalar(0,0,255), 2);
		cv::line(traj, cv::Point(uv3(0)/uv3(2), uv3(1)/uv3(2)), cv::Point(uv4(0)/uv4(2), uv4(1)/uv4(2)), cv::Scalar(0,0,255), 2);
		cv::line(traj, cv::Point(uv4(0)/uv4(2), uv4(1)/uv4(2)), cv::Point(uv1(0)/uv1(2), uv1(1)/uv1(2)), cv::Scalar(0,0,255), 2);
	}

	// Keyframes
	for( uint32_t i = 0; i < keystep_array_.size(); i++ ){
		Eigen::Matrix4d T = TocRec_[keystep_array_[i]];
		uv0 = params_.view.P * Tco * T * (Eigen::Vector4d() << 0,0,0,1).finished();
		uv1 = params_.view.P * Tco * T * (Eigen::Vector4d() << params_.view.upper_left.x,params_.view.upper_left.y,params_.view.upper_left.z,1).finished();
		uv2 = params_.view.P * Tco * T * (Eigen::Vector4d() << params_.view.upper_right.x,params_.view.upper_right.y,params_.view.upper_right.z,1).finished();
		uv3 = params_.view.P * Tco * T * (Eigen::Vector4d() << params_.view.lower_right.x,params_.view.lower_right.y,params_.view.lower_right.z,1).finished();
		uv4 = params_.view.P * Tco * T * (Eigen::Vector4d() << params_.view.lower_left.x,params_.view.lower_left.y,params_.view.lower_left.z,1).finished();
		if( uv0(2) > 1 && uv1(2) > 1 && uv2(2) > 1 && uv3(2) > 1 && uv4(2) > 1){
			cv::line(traj, cv::Point(uv0(0)/uv0(2), uv0(1)/uv0(2)), cv::Point(uv1(0)/uv1(2), uv1(1)/uv1(2)), cv::Scalar(255,0,255), 1);
			cv::line(traj, cv::Point(uv0(0)/uv0(2), uv0(1)/uv0(2)), cv::Point(uv2(0)/uv2(2), uv2(1)/uv2(2)), cv::Scalar(255,0,255), 1);
			cv::line(traj, cv::Point(uv0(0)/uv0(2), uv0(1)/uv0(2)), cv::Point(uv3(0)/uv3(2), uv3(1)/uv3(2)), cv::Scalar(255,0,255), 1);
			cv::line(traj, cv::Point(uv0(0)/uv0(2), uv0(1)/uv0(2)), cv::Point(uv4(0)/uv4(2), uv4(1)/uv4(2)), cv::Scalar(255,0,255), 1);
			cv::line(traj, cv::Point(uv1(0)/uv1(2), uv1(1)/uv1(2)), cv::Point(uv2(0)/uv2(2), uv2(1)/uv2(2)), cv::Scalar(255,0,255), 1);
			cv::line(traj, cv::Point(uv2(0)/uv2(2), uv2(1)/uv2(2)), cv::Point(uv3(0)/uv3(2), uv3(1)/uv3(2)), cv::Scalar(255,0,255), 1);
			cv::line(traj, cv::Point(uv3(0)/uv3(2), uv3(1)/uv3(2)), cv::Point(uv4(0)/uv4(2), uv4(1)/uv4(2)), cv::Scalar(255,0,255), 1);
			cv::line(traj, cv::Point(uv4(0)/uv4(2), uv4(1)/uv4(2)), cv::Point(uv1(0)/uv1(2), uv1(1)/uv1(2)), cv::Scalar(255,0,255), 1);
		}
	}

	// Horizontal Grid
	// Eigen::Vector3d uv_start, uv_end;
	// Eigen::Matrix4d Rco;
	// Rco = Tco;
	// Rco.block(0,3,3,1) = Eigen::Vector3d::Zero();

	// double s, y;
	// Eigen::Matrix3d KRinv = (params_.view.K * params_.view.R).inverse();
	// Eigen::Vector3d KRinv_row = KRinv.row(1);
	// Eigen::Vector3d Kt = params_.view.K * params_.view.t;
	// y = KRinv_row.transpose() * Kt;

	// s = KRinv_row.transpose() * (Eigen::Vector3d() << 1,1,1).finished();
	// s /= y;
	// Eigen::Vector3d topLeftMargin = KRinv * ( (Eigen::Vector3d() << 1,1,1).finished() - Kt * s );
	// topLeftMargin /= s;

	// s = KRinv_row.transpose() * (Eigen::Vector3d() << params_.view.im_size.width,1,1).finished();
	// s /= y;
	// Eigen::Vector3d topRightMargin = KRinv * ( (Eigen::Vector3d() << params_.view.im_size.width,1,1).finished() - Kt * s );
	// topRightMargin /= s;

	// s = KRinv_row.transpose() * (Eigen::Vector3d() << params_.view.im_size.width,params_.view.im_size.height,1).finished();
	// s /= y;
	// Eigen::Vector3d bottomRightMargin = KRinv * ( (Eigen::Vector3d() << params_.view.im_size.width,params_.view.im_size.height,1).finished() - Kt * s);
	// bottomRightMargin /= s;
	
	// s = KRinv_row.transpose() * (Eigen::Vector3d() << 1,params_.view.im_size.height,1).finished();
	// s /= y;
	// Eigen::Vector3d bottomLeftMargin = KRinv * ( (Eigen::Vector3d() << 1,params_.view.im_size.height,1).finished() - Kt * s);
	// bottomLeftMargin /= s;

	// for( int x = 0; x < bottomRightMargin(0) || x < topRightMargin(0); x+=10 ){
	// 	uv_start = params_.view.P * Rco * (Eigen::Vector4d() << x,0,topLeftMargin(2),1).finished();
	// 	uv_end = params_.view.P * Rco * (Eigen::Vector4d() << x,0,bottomRightMargin(2),1).finished();
	// 	if( uv_start(2) > 0 && uv_end(2) > 0 )
	// 		cv::line(traj, cv::Point(uv_start(0)/uv_start(2), uv_start(1)/uv_start(2)), cv::Point(uv_end(0)/uv_end(2), uv_end(1)/uv_end(2)), cv::Scalar(255,255,255), 1);
	// }
	// for( int x = 0; x > topLeftMargin(0) || x > bottomLeftMargin(0); x-=10 ){
	// 	uv_start = params_.view.P * Rco * (Eigen::Vector4d() << x,0,topLeftMargin(2),1).finished();
	// 	uv_end = params_.view.P * Rco * (Eigen::Vector4d() << x,0,bottomRightMargin(2),1).finished();
	// 	if( uv_start(2) > 0 && uv_end(2) > 0 )
	// 		cv::line(traj, cv::Point(uv_start(0)/uv_start(2), uv_start(1)/uv_start(2)), cv::Point(uv_end(0)/uv_end(2), uv_end(1)/uv_end(2)), cv::Scalar(255,255,255), 1);
	// }

	// for( int z = 0; z < topLeftMargin(0); z+=10 ){
	// 	uv_start = params_.view.P * Rco * (Eigen::Vector4d() << topLeftMargin(2),0,z,1).finished();
	// 	uv_end = params_.view.P * Rco * (Eigen::Vector4d() << bottomRightMargin(2),0,z,1).finished();
	// 	if( uv_start(2) > 0 && uv_end(2) > 0 )
	// 		cv::line(traj, cv::Point(uv_start(0)/uv_start(2), uv_start(1)/uv_start(2)), cv::Point(uv_end(0)/uv_end(2), uv_end(1)/uv_end(2)), cv::Scalar(255,255,255), 1);
	// }
	// for( int z = 0; z > bottomRightMargin(0); z-=10 ){
	// 	uv_start = params_.view.P * Rco * (Eigen::Vector4d() << topLeftMargin(2),0,z,1).finished();
	// 	uv_end = params_.view.P * Rco * (Eigen::Vector4d() << bottomRightMargin(2),0,z,1).finished();
	// 	if( uv_start(2) > 0 && uv_end(2) > 0 )
	// 		cv::line(traj, cv::Point(uv_start(0)/uv_start(2), uv_start(1)/uv_start(2)), cv::Point(uv_end(0)/uv_end(2), uv_end(1)/uv_end(2)), cv::Scalar(255,255,255), 1);
	// }
	
	/* simple method */
	// for( int x = -50; x <= 50; x+=10 ){
	// 	uv_start = params_.view.P * Rco * (Eigen::Vector4d() << x,0,50,1).finished();
	// 	uv_end = params_.view.P * Rco * (Eigen::Vector4d() << x,0,-50,1).finished();
	// 	if( uv_start(2) > 0 || uv_end(2) > 0 )
	// 		cv::line(traj, cv::Point(uv_start(0)/uv_start(2), uv_start(1)/uv_start(2)), cv::Point(uv_end(0)/uv_end(2), uv_end(1)/uv_end(2)), cv::Scalar(255,255,255), 1);
	// }
	// for( int z = -50; z <= 50; z+=10 ){
	// 	uv_start = params_.view.P * Rco * (Eigen::Vector4d() << 50,0,z,1).finished();
	// 	uv_end = params_.view.P * Rco * (Eigen::Vector4d() << -50,0,z,1).finished();
	// 	if( uv_start(2) > 0 || uv_end(2) > 0 )
	// 		cv::line(traj, cv::Point(uv_start(0)/uv_start(2), uv_start(1)/uv_start(2)), cv::Point(uv_end(0)/uv_end(2), uv_end(1)/uv_end(2)), cv::Scalar(255,255,255), 1);
	// }

	cv::imshow("Trajectory", traj);
}