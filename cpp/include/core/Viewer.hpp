#include "core/MVO.hpp"
#include "core/DepthFilter.hpp"
#include <stack>
#include <mutex>

#define RATIO 0.5
#define GAP 30

/** @brief 시각화를 위한 촬영 카메라의 파라미터 */
class Viewer{
    public:

    double height; /**< @brief 촬영 카메라의 지면으로부터 높이 */
    double roll; /**< @brief 촬영 카메라의 roll 방향 기울기 */
    double pitch; /**< @brief 촬영 카메라의 pitch 방향 기울기 */
    double height_default = 20; /**< @brief 촬영 카메라의 지면으로부터 초기 높이 */
    double roll_default = -M_PI/2; /**< @brief 촬영 카메라의 roll 방향 초기 기울기 */
    double pitch_default = 0; /**< @brief 촬영 카메라의 pitch 방향 초기 기울기 */
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity(); /**< @brief 원점 좌표계 기준 촬영 카메라의 회전 행렬 */
    Eigen::Vector3d t = Eigen::Vector3d::Zero(); /**< @brief 원점 좌표계 기준 촬영 카메라의 변위 벡터 */
    Eigen::Matrix3d K; /**< @brief 촬영 카메라의 내부 파라미터 */
    Eigen::Matrix<double,3,4> P; /**< @brief 촬영 카메라의 투영 파라미터 */
    cv::Size im_size; /**< @brief 촬영 카메라의 해상도 */
    cv::Point3d upper_left, upper_right, lower_left, lower_right; /**< @brief 촬영 카메라 프레임의 절두체 크기 */

    public:

    Viewer(std::string yaml){
        /**************************************************************************
         *  Read .yaml file
         **************************************************************************/
        cv::FileStorage fSettings(yaml, cv::FileStorage::READ);
        if (!fSettings.isOpened()){
            std::cout << "Failed to open: " << yaml << std::endl;
            abort();
        }

        double version = fSettings["Version"];
        if( version != YAML_VERSION ){
            std::cout << "YAML file is an old version (your version is \"" << std::setfill('0') << std::setprecision(1) << std::fixed << version << "\", required is \"" << YAML_VERSION << "\"" << std::endl;
            abort();
        }
        
        this->height_default =  fSettings["viewCam.height"]; // in world coordinate
        this->roll_default =    (double) fSettings["viewCam.roll"] * M_PI/180; // radian
        this->pitch_default =   (double) fSettings["viewCam.pitch"] * M_PI/180; // radian
        this->height =          this->height_default;
        this->roll =            this->roll_default;
        this->pitch =           this->pitch_default;
        this->im_size = cv::Size(600,600);
        this->K <<  300,   0, 300,
                    0, 300, 300,
                    0,   0,   1;
        this->upper_left =   cv::Point3d(-1,-.5, 1);
        this->upper_right =  cv::Point3d( 1,-.5, 1);
        this->lower_left =   cv::Point3d(-1, .5, 1);
        this->lower_right =  cv::Point3d( 1, .5, 1);

        Eigen::Matrix3d rotx, rotz;
        rotx << 1, 0, 0,
                0, std::cos(this->pitch), -std::sin(this->pitch),
                0, std::sin(this->pitch), std::cos(this->pitch);
        rotz << std::cos(this->roll), -std::sin(this->roll), 0,
                std::sin(this->roll), std::cos(this->roll), 0,
                0, 0, 1;
        this->R = (rotz * rotx).transpose();
        this->t = -(Eigen::Vector3d() << 0,0,-this->height).finished();
        this->P = (Eigen::Matrix<double,3,4>() << this->K * this->R, this->K * this->t).finished();
    };
    void updateView();
    void plot(bool& bRun, std::unique_ptr<MVO>& vo, const Eigen::MatrixXd * const depthMap, std::mutex * m);
};

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
void Viewer::updateView(){
	Eigen::Matrix3d rotx, rotz;
	rotx << 1, 0, 0,
			0, std::cos(this->pitch), -std::sin(this->pitch),
			0, std::sin(this->pitch), std::cos(this->pitch);
	rotz << std::cos(this->roll), -std::sin(this->roll), 0,
			std::sin(this->roll), std::cos(this->roll), 0,
			0, 0, 1;
	this->R = (rotz * rotx).transpose();
	this->t = -(Eigen::Vector3d() << 0,0,-this->height).finished();
	this->P = (Eigen::Matrix<double,3,4>() << this->K * this->R, this->K * this->t).finished();
}

/**
 * @brief 알고리즘의 진행 상황을 나타냄.
 * @details 카메라에서 취득한 이미지와 키프레임 및 깊이 추정값을 그리고, 3차원 카메라 위치를 나타낸다.
 * @param depthMap 참값과 추정값의 비교를 위한 깊이 참값. 깊이 참값이 주어지지 않으면, NULL.
 * @return 없음
 * @author Sangil Lee (sangillee724@gmail.com)
 * @date 29-Dec-2019
 */
void Viewer::plot(bool& bRun, std::unique_ptr<MVO>& vo, const Eigen::MatrixXd * const depthMap, std::mutex * m) {
    while( bRun ){
        if( vo->getCurrFrame().image.cols == 0 )
            continue;

        /*******************************************
         * 		Figure 1: Image seen by camera
         * *****************************************/
        cv::Mat img(vo->params_.im_size, CV_8UC3);
        cvtColor(vo->getCurrFrame().image, img, CV_GRAY2BGR);
        
        double ratio = 1;
        if( img.rows > 500 ){
            ratio = RATIO;
            cv::resize(img, img, cv::Size(img.cols*ratio, img.rows*ratio));
        }

        cv::Mat mvo(cv::Size(img.cols + img.cols*0.5 + 3*GAP, img.rows + 2*GAP), CV_8UC3, cv::Scalar::all(0));

        // buckets
        for( int c = 0; c < vo->getBuckets().grid.width; c++ ){
            cv::line(img, cv::Point(c*vo->getBuckets().size.width*ratio,0), cv::Point(c*vo->getBuckets().size.width*ratio,vo->params_.im_size.height*ratio), cv::Scalar(180,180,180), 1, 16);
        }

        for( int r = 0; r < vo->getBuckets().grid.height; r++ ){
            cv::line(img, cv::Point(0,r*vo->getBuckets().size.height*ratio), cv::Point(vo->params_.im_size.width*ratio,r*vo->getBuckets().size.height*ratio), cv::Scalar(180,180,180), 1, 16);
        }

        // feature points
        m->lock();
        int num_features = vo->getFeatures().size();
        for( int i = 0; i < num_features; i++ ){
            if( vo->getFeatures()[i].life > 2 ){
                if( vo->getFeatures()[i].type == Type::Dynamic || vo->getFeatures()[i].is_2D_inliered == false ){
                    cv::circle(img, cv::Point(vo->getFeatures()[i].uv.back().x*ratio, vo->getFeatures()[i].uv.back().y*ratio), 3, cv::Scalar(255,0,0), 1);
                    if( vo->getFeatures()[i].uv_pred.x > 0 && vo->getFeatures()[i].uv_pred.y > 0 )
                        cv::drawMarker(img, cv::Point(vo->getFeatures()[i].uv_pred.x*ratio, vo->getFeatures()[i].uv_pred.y*ratio), cv::Scalar(255,0,0), cv::MARKER_CROSS, 5);
                }else if( vo->getFeatures()[i].type == Type::Road ){
                    cv::circle(img, cv::Point(vo->getFeatures()[i].uv.back().x*ratio, vo->getFeatures()[i].uv.back().y*ratio), 3, cv::Scalar(50,50,255), 1);
                    if( vo->getFeatures()[i].uv_pred.x > 0 && vo->getFeatures()[i].uv_pred.y > 0 )
                        cv::drawMarker(img, cv::Point(vo->getFeatures()[i].uv_pred.x*ratio, vo->getFeatures()[i].uv_pred.y*ratio), cv::Scalar(50,50,255), cv::MARKER_CROSS, 5);
                }else{
                    cv::circle(img, cv::Point(vo->getFeatures()[i].uv.back().x*ratio, vo->getFeatures()[i].uv.back().y*ratio), 3, cv::Scalar(0,200,0), 1);
                    if( vo->getFeatures()[i].uv_pred.x > 0 && vo->getFeatures()[i].uv_pred.y > 0 )
                        cv::drawMarker(img, cv::Point(vo->getFeatures()[i].uv_pred.x*ratio, vo->getFeatures()[i].uv_pred.y*ratio), cv::Scalar(0,200,0), cv::MARKER_CROSS, 5);
                }
                if( MVO::s_file_logger_.is_open() ){
                    int key_idx = vo->getFeatures()[i].life - 1 - (vo->getCurrStep() - vo->getCurrKeyStep());
                    if( key_idx >= 0 )
                        cv::line(img, vo->getFeatures()[i].uv.back()*ratio, vo->getFeatures()[i].uv[key_idx]*ratio, cv::Scalar::all(0), 1, CV_AA);
                    cv::putText(img, decimalTo36Base(vo->getFeatures()[i].id), (vo->getFeatures()[i].uv.back()+cv::Point2f(5,5))*ratio, cv::FONT_HERSHEY_DUPLEX, 0.4, cv::Scalar(0,255,0), 1, CV_AA);
                }
            }
        }
        m->unlock();

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

            m->lock();
            num_features = vo->getFeatures().size();
            for( uint32_t i = 0; i < num_features; i++ ){
                depth = (*depthMap)(vo->getFeatures()[i].uv.back().y, vo->getFeatures()[i].uv.back().x);

                r = std::exp(-depth/150) * std::min(depth*18, 255);
                g = std::exp(-depth/150) * std::max(255 - depth*8, 30);
                b = std::exp(-depth/150) * std::max(100 - depth, 0);
                cv::circle(gt_dist, vo->getFeatures()[i].uv.back()*ratio*0.5, std::ceil(5*ratio), cv::Scalar(b, g, r), CV_FILLED);
            }
            m->unlock();

            gt_dist.copyTo(mvo(cv::Rect(img.cols+2*GAP,GAP,gt_dist.cols,gt_dist.rows)));
            cv::putText(mvo, "Groundtruth", cv::Point(gt_dist.cols*0.5+img.cols+2*GAP,GAP*0.7), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar::all(255), 1, CV_AA);
        }else{
            /*******************************************
             * 		Figure 1: Keyframe
             * *****************************************/
            cv::Mat img_keyframe(vo->getCurrKeyFrame().size(), CV_8UC3);
            cvtColor(vo->getCurrKeyFrame().image, img_keyframe, CV_GRAY2BGR);
            cv::resize(img_keyframe, img_keyframe, cv::Size(img.cols*0.5,img.rows*0.5));
            img_keyframe.copyTo(mvo(cv::Rect(img.cols+2*GAP,GAP,img_keyframe.cols,img_keyframe.rows)));
            cv::putText(mvo, "Keyframe", cv::Point(img_keyframe.cols*0.5+img.cols+2*GAP,GAP*0.7), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar::all(255), 1, CV_AA);
        }

        /*******************************************
         * 		Figure 1: Reconstructed Depth
         * *****************************************/
        cv::Mat distance(img.rows*0.5, img.cols*0.5, CV_8UC3, cv::Scalar::all(0));
        Eigen::Matrix4d Tco = vo->getCurrentMotion().inverse();
        Eigen::Vector4d point;
        int r, g, b, depth;

        m->lock();
        num_features = vo->getFeatures().size();
        for( uint32_t i = 0; i < num_features; i++ ){
            if( vo->params_.output_filtered_depth ){
                if( vo->getFeatures()[i].landmark && vo->getFeatures()[i].type != Type::Dynamic){
                    point = Tco * vo->getFeatures()[i].landmark->point_init;
                    depth = (int) point(2) - 5;

                    r = std::exp(-depth/150) * std::min(depth*18, 255);
                    g = std::exp(-depth/150) * std::max(255 - depth*8, 30);
                    b = std::exp(-depth/150) * std::max(100 - depth, 0);
                    cv::circle(distance, vo->getFeatures()[i].uv.back()*ratio*0.5, std::ceil(5*ratio), cv::Scalar(b, g, r), CV_FILLED);
                    if( MVO::s_file_logger_.is_open() && point(2) > 0 ) cv::putText(distance, std::to_string(point(2)).substr(0, std::to_string(point(2)).find(".") + 2), (vo->getFeatures()[i].uv.back()+cv::Point2f(8,8))*ratio*0.5, cv::FONT_HERSHEY_DUPLEX, 0.3, cv::Scalar(128,128,128), 1, CV_AA);
                }
            }else{
                if( vo->getFeatures()[i].is_3D_reconstructed && vo->getFeatures()[i].type != Type::Dynamic ){
                    point = vo->getFeatures()[i].point_curr;
                    depth = (int) point(2) - 5;

                    r = std::exp(-depth/150) * std::min(depth*18, 255);
                    g = std::exp(-depth/150) * std::max(255 - depth*8, 30);
                    b = std::exp(-depth/150) * std::max(100 - depth, 0);
                    cv::circle(distance, vo->getFeatures()[i].uv.back()*ratio*0.5, std::ceil(5*ratio), cv::Scalar(b, g, r), CV_FILLED);
                    if( MVO::s_file_logger_.is_open() && point(2) > 0 ) cv::putText(distance, std::to_string(point(2)).substr(0, std::to_string(point(2)).find(".") + 2), (vo->getFeatures()[i].uv.back()+cv::Point2f(8,8))*ratio*0.5, cv::FONT_HERSHEY_DUPLEX, 0.3, cv::Scalar(128,128,128), 1, CV_AA);
                }
            }
        }
        m->unlock();

        distance.copyTo(mvo(cv::Rect(img.cols+2*GAP,distance.rows+2*GAP,distance.cols,distance.rows)));
        cv::putText(mvo, "Depth", cv::Point(distance.cols*0.5+img.cols+2*GAP,distance.rows+GAP*1.7), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar::all(255), 1, CV_AA);
        
        cv::imshow("MVO", mvo);

        /*******************************************
         * 		Figure 2: Trajectory - debug
         * *****************************************/
        if( MVO::s_file_logger_.is_open() ){
            cv::Mat traj = cv::Mat::zeros(this->im_size.height,this->im_size.width,CV_8UC3);

            // Points
            m->lock();

            Eigen::Vector3d uv;
            for( const auto & landmark : vo->getLandmark() ){
                point = Tco * landmark.second->point_init;
                uv = this->P * point;
                if( uv(2) > 1 ) cv::circle(traj, cv::Point(uv(0)/uv(2), uv(1)/uv(2)), 1, cv::Scalar(128,128,128), CV_FILLED);
            }

            num_features = vo->getFeatures().size();
            for( uint32_t i = 0; i < num_features; i++ ){
                // std::cout << vo->getFeatures()[i].depthfilter->getVariance() << ' ';
                if( vo->getFeatures()[i].landmark && vo->getFeatures()[i].depthfilter->getVariance() < vo->params_.max_point_var && vo->getFeatures()[i].life > 2 ){
                    point = Tco * vo->getFeatures()[i].landmark->point_init;
                    uv = this->P * point;
                    if( uv(2) > 1 ){
                        switch (vo->getFeatures()[i].type){
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
                        if( vo->params_.output_filtered_depth ){
                            if( vo->getFeatures()[i].landmark && vo->getFeatures()[i].frame_3d_init < vo->getCurrStep() && vo->getFeatures()[i].type != Type::Dynamic ){
                                cv::circle(traj, cv::Point(uv(0)/uv(2), uv(1)/uv(2)), 1, cv::Scalar(1e8*vo->getFeatures()[i].depthfilter->getVariance(),255,0), CV_FILLED);
                            }
                        }else{
                            if( vo->getFeatures()[i].is_3D_reconstructed && vo->getFeatures()[i].frame_3d_init < vo->getCurrStep() && vo->getFeatures()[i].type != Type::Dynamic )
                                cv::circle(traj, cv::Point(uv(0)/uv(2), uv(1)/uv(2)), 1, cv::Scalar(0,255,0), CV_FILLED);
                        }
                    }
                }
            }
            std::cout << std::endl;
            m->unlock();

            // Camera
            Eigen::Vector3d uv0, uv1, uv2, uv3, uv4;
            uv0 = this->P * (Eigen::Vector4d() << 0,0,0,1).finished();
            uv1 = this->P * (Eigen::Vector4d() << this->upper_left.x,this->upper_left.y,this->upper_left.z,1).finished();
            uv2 = this->P * (Eigen::Vector4d() << this->upper_right.x,this->upper_right.y,this->upper_right.z,1).finished();
            uv3 = this->P * (Eigen::Vector4d() << this->lower_right.x,this->lower_right.y,this->lower_right.z,1).finished();
            uv4 = this->P * (Eigen::Vector4d() << this->lower_left.x,this->lower_left.y,this->lower_left.z,1).finished();
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
            m->lock();
            for( uint32_t i = 0; i < vo->getKeySteps().size(); i++ ){
                Eigen::Matrix4d T = vo->getTocRec()[vo->getKeySteps()[i]];
                uv0 = this->P * Tco * T * (Eigen::Vector4d() << 0,0,0,1).finished();
                uv1 = this->P * Tco * T * (Eigen::Vector4d() << this->upper_left.x,this->upper_left.y,this->upper_left.z,1).finished();
                uv2 = this->P * Tco * T * (Eigen::Vector4d() << this->upper_right.x,this->upper_right.y,this->upper_right.z,1).finished();
                uv3 = this->P * Tco * T * (Eigen::Vector4d() << this->lower_right.x,this->lower_right.y,this->lower_right.z,1).finished();
                uv4 = this->P * Tco * T * (Eigen::Vector4d() << this->lower_left.x,this->lower_left.y,this->lower_left.z,1).finished();
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
            m->unlock();

            cv::imshow("Trajectory", traj);
        }
    }
}