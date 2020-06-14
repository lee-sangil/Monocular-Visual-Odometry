#include "core/MVO.hpp"
#include "core/DepthFilter.hpp"
#include <stack>

#define RATIO 0.5
#define GAP 30

/**
 * @brief 숫자의 형식을 바꿈.
 * @details 10진수를 36진수로 변환한다.
 * @param num 10진수 값
 * @return 36진수 문자열 (0, 1, ..., 9, A, ..., Z)
 * @author Sangil Lee (sangillee724@gmail.com)
 * @date 29-Dec-2019
 */
std::string decimalTo36Base(int num){
	char rem;
	std::stack<char> base;
	while( num > 0 ){
		rem = num % 36;
		num /= 36;

		if( rem < 10 )
			base.push(48+rem);
		else
			base.push(55+rem);
	}
	std::stringstream ss;
	while( !base.empty() ){
		ss << base.top();
		base.pop();
	}
	return ss.str();
}

/**
 * @brief 촬영 카메라의 파라미터 변경.
 * @details main.cpp에서 키 인터럽트를 통해 변경된 촬영 카메라의 앵글 및 거리를 적용한다.
 * @return 없음
 * @author Sangil Lee (sangillee724@gmail.com)
 * @date 29-Dec-2019
 */
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

/**
 * @brief 알고리즘의 진행 상황을 나타냄.
 * @details 카메라에서 취득한 이미지와 키프레임 및 깊이 추정값을 그리고, 3차원 카메라 위치를 나타낸다.
 * @param depthMap 참값과 추정값의 비교를 위한 깊이 참값. 깊이 참값이 주어지지 않으면, NULL.
 * @return 없음
 * @author Sangil Lee (sangillee724@gmail.com)
 * @date 29-Dec-2019
 */
void MVO::plot(const Eigen::MatrixXd * const depthMap) const {
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
		if( features_[i].life > 2 ){
			if( features_[i].type == Type::Dynamic || features_[i].is_2D_inliered == false ){
				cv::circle(img, cv::Point(features_[i].uv.back().x*ratio, features_[i].uv.back().y*ratio), 3, cv::Scalar(255,0,0), CV_FILLED);
				if( features_[i].uv_pred.x > 0 && features_[i].uv_pred.y > 0 )
					cv::drawMarker(img, cv::Point(features_[i].uv_pred.x*ratio, features_[i].uv_pred.y*ratio), cv::Scalar(255,0,0), cv::MARKER_CROSS, 5);
			}else if( features_[i].type == Type::Road ){
				cv::circle(img, cv::Point(features_[i].uv.back().x*ratio, features_[i].uv.back().y*ratio), 3, cv::Scalar(50,50,255), CV_FILLED);
				if( features_[i].uv_pred.x > 0 && features_[i].uv_pred.y > 0 )
					cv::drawMarker(img, cv::Point(features_[i].uv_pred.x*ratio, features_[i].uv_pred.y*ratio), cv::Scalar(50,50,255), cv::MARKER_CROSS, 5);
			}else{
				cv::circle(img, cv::Point(features_[i].uv.back().x*ratio, features_[i].uv.back().y*ratio), 3, cv::Scalar(0,200,0), CV_FILLED);
				if( features_[i].uv_pred.x > 0 && features_[i].uv_pred.y > 0 )
					cv::drawMarker(img, cv::Point(features_[i].uv_pred.x*ratio, features_[i].uv_pred.y*ratio), cv::Scalar(0,200,0), cv::MARKER_CROSS, 5);
			}
			if( MVO::s_file_logger_.is_open() ){
				int key_idx = features_[i].life - 1 - (step_ - keystep_);
				if( key_idx >= 0 )
					cv::line(img, features_[i].uv.back()*ratio, features_[i].uv[key_idx]*ratio, cv::Scalar(0,255,255), 1, CV_AA);
				// cv::putText(img, decimalTo36Base(features_[i].id), (features_[i].uv.back()+cv::Point2f(5,5))*ratio, cv::FONT_HERSHEY_DUPLEX, 0.4, cv::Scalar(0,255,0), 1, CV_AA);
			}
		}
	}
	// cv::rectangle(img, cv::Rect(200*ratio,200*ratio,200*ratio,200*ratio),cv::Scalar(0,0,255));
	// cv::rectangle(img, cv::Rect(400*ratio,400*ratio,200*ratio,200*ratio),cv::Scalar(0,255,0));
	// cv::rectangle(img, cv::Rect(600*ratio,400*ratio,200*ratio,200*ratio),cv::Scalar(0,0,255));
	img.copyTo(mvo(cv::Rect(GAP,GAP,img.cols,img.rows)));
	cv::putText(mvo, "Features", cv::Point(img.cols*0.5 + GAP,GAP*0.7), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar::all(255), 1, CV_AA);

	if( depthMap != NULL ){
		/*******************************************
		 * 		Figure 1: Groundtruth Depth
		 * *****************************************/
		cv::Mat gt_dist(img.rows*0.5, img.cols*0.5, CV_8UC3, cv::Scalar::all(0));
		int r, g, b, depth;
		for( uint32_t i = 0; i < features_.size(); i++ ){
			depth = (*depthMap)(features_[i].uv.back().y, features_[i].uv.back().x);

			if( depth > 0 ){
				r = std::exp(-depth/150) * std::min(depth*18, 255);
				g = std::exp(-depth/150) * std::max(255 - depth*8, 30);
				b = std::exp(-depth/150) * std::max(100 - depth, 0);
				cv::circle(gt_dist, features_[i].uv.back()*ratio*0.5, std::ceil(5*ratio), cv::Scalar(b, g, r), CV_FILLED);
			}
		}
		gt_dist.copyTo(mvo(cv::Rect(img.cols+2*GAP,GAP,gt_dist.cols,gt_dist.rows)));
		cv::putText(mvo, "Groundtruth", cv::Point(gt_dist.cols*0.5+img.cols+2*GAP,GAP*0.7), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar::all(255), 1, CV_AA);
	}else{
		/*******************************************
		 * 		Figure 1: Keyframe
		 * *****************************************/
		cv::Mat img_keyframe(curr_keyframe_.size(), CV_8UC3);
		cvtColor(curr_keyframe_.image, img_keyframe, CV_GRAY2BGR);
		cv::resize(img_keyframe, img_keyframe, cv::Size(img.cols*0.5,img.rows*0.5));
		img_keyframe.copyTo(mvo(cv::Rect(img.cols+2*GAP,GAP,img_keyframe.cols,img_keyframe.rows)));
		cv::putText(mvo, "Keyframe", cv::Point(img_keyframe.cols*0.5+img.cols+2*GAP,GAP*0.7), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar::all(255), 1, CV_AA);
	}

	/*******************************************
	 * 		Figure 1: Reconstructed Depth
	 * *****************************************/
	cv::Mat distance(img.rows*0.5, img.cols*0.5, CV_8UC3, cv::Scalar::all(0));
	Eigen::Matrix4d Tco = TocRec_.back().inverse();
	Eigen::Vector4d point;
	int r, g, b, depth;
	for( uint32_t i = 0; i < features_.size(); i++ ){
		if( params_.output_filtered_depth ){
			if( features_[i].landmark && features_[i].type != Type::Dynamic){
				point = Tco * features_[i].landmark->point_init;
				depth = (int) point(2) - 5;

				r = std::exp(-depth/150) * std::min(depth*18, 255);
				g = std::exp(-depth/150) * std::max(255 - depth*8, 30);
				b = std::exp(-depth/150) * std::max(100 - depth, 0);
				cv::circle(distance, features_[i].uv.back()*ratio*0.5, std::ceil(5*ratio), cv::Scalar(b, g, r), CV_FILLED);
				// if( MVO::s_file_logger_.is_open() && point(2) > 0 ) cv::putText(distance, std::to_string(point(2)).substr(0, std::to_string(point(2)).find(".") + 2), (features_[i].uv.back()+cv::Point2f(8,8))*ratio*0.5, cv::FONT_HERSHEY_DUPLEX, 0.3, cv::Scalar(128,128,128), 1, CV_AA);
			}
		}else{
			if( features_[i].is_3D_reconstructed && features_[i].type != Type::Dynamic ){
				point = features_[i].point_curr;
				depth = (int) point(2) - 5;

				r = std::exp(-depth/150) * std::min(depth*18, 255);
				g = std::exp(-depth/150) * std::max(255 - depth*8, 30);
				b = std::exp(-depth/150) * std::max(100 - depth, 0);
				cv::circle(distance, features_[i].uv.back()*ratio*0.5, std::ceil(5*ratio), cv::Scalar(b, g, r), CV_FILLED);
				// if( MVO::s_file_logger_.is_open() && point(2) > 0 ) cv::putText(distance, std::to_string(point(2)).substr(0, std::to_string(point(2)).find(".") + 2), (features_[i].uv.back()+cv::Point2f(8,8))*ratio*0.5, cv::FONT_HERSHEY_DUPLEX, 0.3, cv::Scalar(128,128,128), 1, CV_AA);
			}
		}
	}
	distance.copyTo(mvo(cv::Rect(img.cols+2*GAP,distance.rows+2*GAP,distance.cols,distance.rows)));
	cv::putText(mvo, "Depth", cv::Point(distance.cols*0.5+img.cols+2*GAP,distance.rows+GAP*1.7), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar::all(255), 1, CV_AA);
	
	cv::imshow("MVO", mvo);

	/*******************************************
	 * 		Figure 2: Trajectory - debug
	 * *****************************************/
	if( MVO::s_file_logger_.is_open() ){
		cv::Mat traj = cv::Mat::zeros(params_.view.im_size.height,params_.view.im_size.width,CV_8UC3);

		// Points
		Eigen::Vector3d uv;
		for( const auto & landmark : landmark_ ){
			point = Tco * landmark.second->point_init;
			uv = params_.view.P * point;
			if( uv(2) > 1 && landmark.second->variance < params_.max_point_var ){
				switch (landmark.second->type){
				case Type::Unknown:
					cv::circle(traj, cv::Point(uv(0)/uv(2), uv(1)/uv(2)), 1, cv::Scalar(128,128,128), CV_FILLED);
					break;
				// case Type::Road:
				// 	cv::circle(traj, cv::Point(uv(0)/uv(2), uv(1)/uv(2)), 1, cv::Scalar(128,128,200), CV_FILLED);
				// 	break;
				// case Type::Dynamic:
				// 	cv::circle(traj, cv::Point(uv(0)/uv(2), uv(1)/uv(2)), 1, cv::Scalar(100,50,50), CV_FILLED);
				// 	break;
				}
			}
		}

		for( uint32_t i = 0; i < features_.size(); i++ ){
			// std::cout << features_[i].depthfilter->getVariance() << ' ';
			if( features_[i].landmark && features_[i].depthfilter->getVariance() < params_.max_point_var && features_[i].life > 2 ){
				point = Tco * features_[i].landmark->point_init;
				uv = params_.view.P * point;
				if( uv(2) > 1 ){
					// switch (features_[i].type){
					// case Type::Unknown:
					// 	cv::circle(traj, cv::Point(uv(0)/uv(2), uv(1)/uv(2)), 1, cv::Scalar(128,128,128), CV_FILLED);
					// 	break;
					// case Type::Road:
					// 	cv::circle(traj, cv::Point(uv(0)/uv(2), uv(1)/uv(2)), 1, cv::Scalar(128,128,200), CV_FILLED);
					// 	break;
					// case Type::Dynamic:
					// 	cv::circle(traj, cv::Point(uv(0)/uv(2), uv(1)/uv(2)), 1, cv::Scalar(100,50,50), CV_FILLED);
					// 	break;
					// }
					if( params_.output_filtered_depth ){
						if( features_[i].landmark && features_[i].frame_3d_init < step_ && features_[i].type == Type::Unknown ){
							cv::circle(traj, cv::Point(uv(0)/uv(2), uv(1)/uv(2)), 1, cv::Scalar(1e8*features_[i].depthfilter->getVariance(),255,0), CV_FILLED);
						}
					}else{
						if( features_[i].is_3D_reconstructed && features_[i].frame_3d_init < step_ && features_[i].type == Type::Unknown )
							cv::circle(traj, cv::Point(uv(0)/uv(2), uv(1)/uv(2)), 1, cv::Scalar(0,255,0), CV_FILLED);
					}
				}
			}
		}
		// std::cout << std::endl;

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

		cv::imshow("Trajectory", traj);
	}
}