#include "core/MVO.hpp"
#include "core/random.hpp"
#include "core/time.hpp"
#include "core/DepthFilter.hpp"

double DepthFilter::s_px_error_angle_;
double DepthFilter::s_meas_max_;
std::ofstream MVO::s_file_logger_;
std::ofstream MVO::s_point_logger_;
uint32_t Feature::new_feature_id = 0;

MVO::MVO(){
    step_ = -1;
    keystep_ = 0;
    keystep_array_.push_back(0);

    is_start_ = false;
    is_scale_initialized_ = false;
    is_speed_provided_ = false;
    is_rotate_provided_ = false;
    cvClahe_ = cv::createCLAHE();

    // Variables
    num_feature_ = 0;
    num_feature_matched_ = 0;
    num_feature_2D_inliered_ = 0;
    num_feature_3D_reconstructed_ = 0;
    num_feature_inlier_ = 0;

    // Initial position
    TRec_.push_back(Eigen::Matrix4d::Identity());
    TocRec_.push_back(Eigen::Matrix4d::Identity());
    PocRec_.push_back((Eigen::Vector4d() << 0,0,0,1).finished());

    R_vec_.reserve(4);
	t_vec_.reserve(4);

    rotate_prior_ = Eigen::Matrix3d::Identity();

    // file_logger
    if( MVO::s_file_logger_.is_open() ) std::cout << "# Generate log.txt" << std::endl;
    if( MVO::s_point_logger_.is_open() ) std::cout << "# Generate pointcloud.txt" << std::endl;

    // random seed
    lsi::seed();
}

MVO::MVO(std::string yaml):MVO(){

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

	params_.fx =			    fSettings["Camera.fx"];
	params_.fy =			    fSettings["Camera.fy"];
	params_.cx =			    fSettings["Camera.cx"];
	params_.cy =			    fSettings["Camera.cy"];
	params_.k1 =			    fSettings["Camera.k1"];
	params_.k2 =			    fSettings["Camera.k2"];
	params_.p1 =			    fSettings["Camera.p1"];
	params_.p2 =			    fSettings["Camera.p2"];
	params_.k3 =			    fSettings["Camera.k3"];
	params_.im_size.width =	    fSettings["Camera.width"];
	params_.im_size.height =	fSettings["Camera.height"];

    params_.K << params_.fx, 0, params_.cx,
						0, params_.fy, params_.cy,
						0, 0, 1;
    params_.Kinv = params_.K.inverse();
    cv::eigen2cv(params_.K, params_.Kcv);
    excludeMask_ = cv::Mat(params_.im_size, CV_8UC1, cv::Scalar(255));

	params_.radial_distortion.push_back(params_.k1);
	params_.radial_distortion.push_back(params_.k2);
	params_.radial_distortion.push_back(params_.k3);
	params_.tangential_distortion.push_back(params_.p1);
	params_.tangential_distortion.push_back(params_.p2);
    params_.dist_coeffs.insert(params_.dist_coeffs.begin(), params_.radial_distortion.begin(), params_.radial_distortion.begin()+1);
    params_.dist_coeffs.insert(params_.dist_coeffs.end(), params_.tangential_distortion.begin(), params_.tangential_distortion.end());
    params_.dist_coeffs.insert(params_.dist_coeffs.begin(), params_.radial_distortion.begin()+2, params_.radial_distortion.end());

    cv::initUndistortRectifyMap(params_.Kcv, params_.dist_coeffs, cv::Mat(), params_.Kcv, params_.im_size, CV_32FC1, distort_map1_, distort_map2_);

    if( !fSettings["Camera.T_IC"].empty() ){
        cv::Mat Tic;
        fSettings["Camera.T_IC"] >> Tic;
        cv::cv2eigen(Tic, params_.Tic);
        params_.Tci = params_.Tic.inverse();
    }else if( !fSettings["Camera.T_CI"].empty() ){
        cv::Mat Tci;
        fSettings["Camera.T_CI"] >> Tci;
        cv::cv2eigen(Tci, params_.Tci);
        params_.Tic = params_.Tci.inverse();
    }else if( !fSettings["Camera.T_IV"].empty() && !fSettings["Camera.T_VC"].empty() ){
        Eigen::Matrix4d Tiv_, Tvc_;
        cv::Mat Tiv, Tvc;
        fSettings["Camera.T_IV"] >> Tiv;
        fSettings["Camera.T_VC"] >> Tvc;
        cv::cv2eigen(Tiv, Tiv_);
        cv::cv2eigen(Tvc, Tvc_);

        params_.Tci = Tvc_ * Tiv_;
        params_.Tic = params_.Tci.inverse();
    }

    // descriptor = cv::BRISK::create();

    params_.th_inlier =             fSettings["Feature.minimum_num_inlier"];
    params_.th_ratio_keyframe =     fSettings["Feature.minimum_matching_ratio"];
    params_.min_px_dist =           fSettings["Feature.minimum_distance_between"];
    params_.th_px_wide =            fSettings["Feature.minimum_triangulation_baseline"];
    params_.max_dist =              fSettings["Feature.maximum_prediction_distance"];
    params_.th_parallax =           fSettings["Feature.threshold_parallax"];
	params_.percentile_parallax =   fSettings["Feature.percentile_parallax"];

    // RANSAC parameter
    params_.ransac_coef_scale.max_iteration =       fSettings["RANSAC.maximum_iteration"];
    params_.ransac_coef_scale.th_inlier_ratio =     fSettings["RANSAC.threshold_inlier_ratio"];
    params_.ransac_coef_scale.min_num_point =       fSettings["RANSAC.scale.num_sample"];
    params_.ransac_coef_scale.th_dist =             fSettings["RANSAC.scale.threshold_dist_inlier"]; // standard deviation
    params_.ransac_coef_scale.th_dist_outlier =     fSettings["RANSAC.scale.threshold_dist_outlier"]; // three times of standard deviation
    params_.ransac_coef_scale.calculate_dist =      lsi::calculateScaleError;
    
    params_.ransac_coef_plane.max_iteration =       fSettings["RANSAC.maximum_iteration"];
    params_.ransac_coef_plane.th_inlier_ratio =     fSettings["RANSAC.threshold_inlier_ratio"];
    params_.ransac_coef_plane.min_num_point =       3;
    params_.ransac_coef_plane.th_dist =             fSettings["RANSAC.plane.threshold_dist_inlier"]; // standard deviation
    params_.ransac_coef_plane.th_dist_outlier =     fSettings["RANSAC.plane.threshold_dist_outlier"]; // three times of standard deviation
    params_.ransac_coef_plane.calculate_func =      lsi::calculatePlane;
    params_.ransac_coef_plane.calculate_dist =      lsi::calculatePlaneError;

    // Bucket
    bucket_ = MVO::Bucket();
    bucket_.max_features =          fSettings["Feature.number"];
    bucket_.safety =                fSettings["Bucket.border_safety"];
    int bucket_grid_rows =          fSettings["Bucket.rows"];
    int bucket_grid_cols =          fSettings["Bucket.cols"];

	bucket_.grid = cv::Size(bucket_grid_cols,bucket_grid_rows);
	bucket_.size = cv::Size(params_.im_size.width/bucket_.grid.width, params_.im_size.height/bucket_.grid.height);

    if( std::min(bucket_.size.width, bucket_.size.height) < 2*bucket_.safety ){
        std::cout << bucket_.safety << std::endl;
        bucket_.safety = static_cast<int>(std::floor(std::min(bucket_.size.width, bucket_.size.height)*0.125));
        std::cout << bucket_.safety << std::endl;
    }

	bucket_.mass.setZero(bucket_.grid.height, bucket_.grid.width);
	bucket_.prob.resize(bucket_.grid.height, bucket_.grid.width);
    bucket_.prob.fill(1.0);
    bucket_.saturated.setZero(bucket_.grid.height, bucket_.grid.width);

    keypoints_of_bucket_.resize(bucket_grid_rows*bucket_grid_cols);
    visit_bucket_ = std::vector<bool>(bucket_grid_rows*bucket_grid_cols, false);
    features_.reserve(bucket_.max_features);
    // prev_pyramid_template_.reserve(10);
    // curr_pyramid_template_.reserve(10);

    // 3D reconstruction
    params_.init_scale =                1;
    params_.vehicle_height =            fSettings["Scale.reference_height"]; // in meter
    params_.weight_scale_ref =          fSettings["Scale.reference_weight"];
	params_.weight_scale_reg =          fSettings["Scale.regularization_weight"];
    params_.reproj_error =              fSettings["PnP.reprojection_error"];
    params_.update_init_point =         fSettings["Debug.update_init_points"];
    params_.output_filtered_depth =     fSettings["Debug.output_filtered_depths"];
    params_.mapping_option =            fSettings["Debug.mapping_options"];

    eigen_solver_ = std::shared_ptr< Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> > (new Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>());

    switch( fSettings["Triangulation.method"] ){
        case 0:
            params_.triangulation_method = MVO::TRIANGULATION::MIDP;
            break;
        case 1:
            params_.triangulation_method = MVO::TRIANGULATION::LLS;
            break;
        default:
            abort();
    }
    
    switch( fSettings["Triangulation.SVD"] ){
        case 0:
            params_.svd_method = MVO::SVD::JACOBI;
            break;
        case 1:
            params_.svd_method = MVO::SVD::BDC;
            break;
        case 2:
            params_.svd_method = MVO::SVD::OpenCV;
            break;
        case 3:
            params_.svd_method = MVO::SVD::Eigen;
            break;
        default:
            abort();
    }

    switch( fSettings["PnP.method"] ){
        case 0:
            params_.pnp_method = MVO::PNP::LM;
            break;
        case 1:
            params_.pnp_method = MVO::PNP::ITERATIVE;
            break;
        case 2:
            params_.pnp_method = MVO::PNP::AP3P;
            break;
        case 3:
            params_.pnp_method = MVO::PNP::EPNP;
            break;
        case 4:
            params_.pnp_method = MVO::PNP::DLS;
            break;
        case 5:
            params_.pnp_method = MVO::PNP::UPNP;
            break;
        default:
            abort();
    }

    params_.max_point_var =         fSettings["DepthFilter.maximum_variance"];
    double depth_min =              fSettings["DepthFilter.minimum_depth"];
    double px_noise =               fSettings["DepthFilter.pixel_noise"];
    DepthFilter::s_meas_max_ = 1.0/depth_min;
    DepthFilter::s_px_error_angle_ = std::atan(px_noise/(params_.fx+params_.fy))*2.0; // law of chord (sehnensatz)

    params_.view.height_default =   fSettings["viewCam.height"]; // in world coordinate
    params_.view.roll_default =     (double) fSettings["viewCam.roll"] * M_PI/180; // radian
    params_.view.pitch_default =    (double) fSettings["viewCam.pitch"] * M_PI/180; // radian
    params_.view.height =          params_.view.height_default;
    params_.view.roll =            params_.view.roll_default;
    params_.view.pitch =           params_.view.pitch_default;
    params_.view.im_size = cv::Size(600,600);
    params_.view.K <<  300,   0, 300,
						      0, 300, 300,
						      0,   0,   1;
    params_.view.upper_left =   cv::Point3d(-1,-.5, 1);
    params_.view.upper_right =  cv::Point3d( 1,-.5, 1);
    params_.view.lower_left =   cv::Point3d(-1, .5, 1);
    params_.view.lower_right =  cv::Point3d( 1, .5, 1);

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

void MVO::refresh(){
    num_feature_matched_ = 0;
    num_feature_2D_inliered_ = 0;
    num_feature_3D_reconstructed_ = 0;
    num_feature_inlier_ = 0;

    for( int i = 0; i < num_feature_; i++ ){
        features_[i].is_matched = false;
        features_[i].is_2D_inliered = false;
        features_[i].is_3D_reconstructed = false;
    }

    keypoints_of_bucket_.clear();
    visit_bucket_.assign(visit_bucket_.size(), false);
}

void MVO::restart(){
    step_ = 0;
    keystep_ = 0;
    keystep_array_.clear();
    keystep_array_.push_back(0);

    prev_frame_.id = -1;
    curr_frame_.id = 0;
    prev_keyframe_.id = -1;
    curr_keyframe_.id = 0;
    next_keyframe_.id = 0;

    is_start_ = false;
    is_scale_initialized_ = false;
    
    deleteDeadFeatures();
    for( uint32_t i = 0; i < features_.size(); i++ ){
        features_[i].life = 1;
        features_[i].frame_2d_init = 0;
        if( features_[i].is_3D_init )
            features_[i].frame_3d_init = 0;
        features_[i].uv.erase(features_[i].uv.begin(), features_[i].uv.end()-1);
        // features_[i].depthfilter->reset();
    }

    // Variables
    num_feature_ = features_.size();
    num_feature_matched_ = 0;
    num_feature_2D_inliered_ = 0;
    num_feature_3D_reconstructed_ = 0;
    num_feature_inlier_ = 0;

    // Initial position
    TRec_.erase(TRec_.begin(), TRec_.end()-1);
    TocRec_.erase(TocRec_.begin(), TocRec_.end()-1);
    PocRec_.erase(PocRec_.begin(), PocRec_.end()-1);

    // restartKeyframeLogger();
}

void MVO::setImage(const cv::Mat& image, double timestamp){
    step_++;
    keystep_ = keystep_array_.back();
    
    if( trigger_keystep_increase_ )
        curr_keyframe_.assign(next_keyframe_);

    trigger_keystep_decrease_previous_ = trigger_keystep_decrease_;
    trigger_keystep_decrease_ = false;
    trigger_keystep_increase_ = false;

    if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "============ Iteration: " << step_ << " (keystep: " << keystep_ << ')' << " ============" << std::endl;
    if( MVO::s_point_logger_.is_open() ) MVO::s_point_logger_ << "============ Iteration: " << step_ << " (keystep: " << keystep_ << ')' << " ============" << std::endl;

    prev_frame_.assign(curr_frame_);

    cv::remap(image, image, distort_map1_, distort_map2_, CV_INTER_LINEAR);
    
    if( params_.apply_clahe )
        cvClahe_->apply(image, curr_frame_.image);
    else
        curr_frame_.image = image.clone();
    curr_frame_.id = step_;
    curr_frame_.timestamp = timestamp;

    if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "# Grab image: " << lsi::toc() << std::endl;
}

void MVO::run(const cv::Mat& image, double timestamp = 0){
    
    lsi::tic();
    setImage(image, timestamp);
    refresh();

    if( !extractFeatures() ) { restart(); return; }
    if( !calculateEssential() ) { restart(); return; }
    if( !calculateMotion() ) { restart(); return; }
    
    // std::vector<cv::Rect> rois;
    // std::vector<int> num_feature;
    // rois.push_back(cv::Rect(200,200,200,200));
    // rois.push_back(cv::Rect(400,400,200,200));
    // rois.push_back(cv::Rect(600,400,200,200));
    // num_feature.push_back(-1);
    // num_feature.push_back(10);
    // num_feature.push_back(-1);
    // updateRoiFeatures(rois, num_feature); // Extract extra features in rois
}

void MVO::updateGyro(const double timestamp, const Eigen::Vector3d& gyro){
    is_rotate_provided_ = true;

    if( trigger_keystep_increase_ ){
        next_keyframe_.angular_velocity_since_.emplace_back(timestamp,gyro);
    }else{
        curr_keyframe_.angular_velocity_since_.emplace_back(timestamp,gyro);
    }
}

void MVO::updateVelocity(const double timestamp, const double speed){
    is_speed_provided_ = true;
    
    if( trigger_keystep_increase_ ){
        next_keyframe_.linear_velocity_since_.emplace_back(timestamp,speed);
    }else{
        curr_keyframe_.linear_velocity_since_.emplace_back(timestamp,speed);
    }
}

const std::vector<Feature>& MVO::getFeatures() const {return features_;}
const Eigen::Matrix4d& MVO::getCurrentMotion() const {return TRec_.back();}

std::vector< std::tuple<uint32_t, cv::Point2f, Eigen::Vector3d, double> > MVO::getPoints() const
{
    Eigen::Matrix4d Tco = TocRec_.back().inverse();
    std::vector< std::tuple<uint32_t, cv::Point2f, Eigen::Vector3d, double> > pts;
    cv::Point2f uv_curr;

    if( params_.output_filtered_depth ){
        for( uint32_t i = 0; i < features_.size(); i++ ){
            uv_curr = features_[i].uv.back();
            
            if( features_[i].is_3D_init && features_[i].life > 1 )
                pts.push_back( std::make_tuple(features_[i].id, uv_curr, Tco.block(0,0,3,4) * features_[i].point_init, features_[i].depthfilter->getVariance() ) );
        }
    }else{
        for( uint32_t i = 0; i < features_.size(); i++ ){
            uv_curr = features_[i].uv.back();
            
            if( features_[i].is_3D_reconstructed && features_[i].life > 1 )
                pts.push_back( std::make_tuple(features_[i].id, uv_curr, features_[i].point_curr.block(0, 0, 3, 1), features_[i].depthfilter->getVariance() ) );
        }
    }
    return pts;
}

void MVO::getPointsInRoi(const cv::Rect& roi, std::vector<uint32_t>& idx) const {
    idx.clear();
    idx.reserve(num_feature_*0.5);

    for( uint32_t i = 0; i < features_.size(); i++ ){
        if( features_[i].uv.size() > 0 && features_[i].uv.back().x >= roi.x && features_[i].uv.back().y >= roi.y && features_[i].uv.back().x <= roi.x + roi.width && features_[i].uv.back().y <= roi.y + roi.height )
            idx.push_back(i);
    }
}

std::vector< std::tuple<uint32_t, cv::Point2f, cv::Point2f> > MVO::getMotions() const
{
    std::vector< std::tuple<uint32_t, cv::Point2f, cv::Point2f> > pts;
    cv::Point2f uv_curr, uv_prev;
    for( uint32_t i = 0; i < features_.size(); i++ ){
        uv_curr = features_[i].uv.back();
        if( features_[i].uv.size() > 1)
            uv_prev = features_[i].uv[features_[i].life - 2];
        else
            uv_prev = cv::Point2f(-1,-1);
        
        if( params_.output_filtered_depth )
            pts.push_back( std::make_tuple(features_[i].id, uv_prev, uv_curr ) );
        else
            pts.push_back( std::make_tuple(features_[i].id, uv_prev, uv_curr ) );
    }
    return pts;
}

void MVO::printFeatures() const {
    if( MVO::s_file_logger_.is_open() ){
        std::stringstream filename;
        filename << "FeatureLogFiles/" << keystep_ << "_to_" << step_ << ".md";
        std::ofstream fid(filename.str());

        if( fid.is_open() ){
            int key_idx;
            for( const auto & feature : features_ ){
                fid << "ID: " << std::setw(4) << feature.id << "\tINIT: " << std::setw(3) << feature.frame_2d_init << "\tLIFE: " << std::setw(3) << feature.life;
                
                key_idx = feature.life - 1 - (step_ - keystep_);
                if( key_idx == 0 )
                    fid << "\tUV: **[" << std::setw(4) << (int) feature.uv[0].x << ", " << std::setw(4) << (int) feature.uv[0].y << "]**";
                else
                    fid << "\tUV: [" << std::setw(4) << (int) feature.uv[0].x << ", " << std::setw(4) << (int) feature.uv[0].y << ']';

                for( uint i = 1; i < feature.uv.size()-1; i++ ){
                    if( key_idx == i )
                        fid << " => **[" << std::setw(4) << (int) feature.uv[i].x << ", " << std::setw(4) << (int) feature.uv[i].y << "]**";
                    else
                        fid << " => [" << std::setw(4) << (int) feature.uv[i].x << ", " << std::setw(4) << (int) feature.uv[i].y << ']';
                }
                if( feature.uv.size() > 1 ){
                    if( key_idx >= 0 ) fid << " => **[" << std::setw(4) << (int) feature.uv.back().x << ", " << std::setw(4) << (int) feature.uv.back().y << "]**";
                    else fid << " => [" << std::setw(4) << (int) feature.uv.back().x << ", " << std::setw(4) << (int) feature.uv.back().y << ']';
                }
                fid << std::endl;
            }
        }
    }
}

void MVO::restartKeyframeLogger() {
    // if( is_speed_provided_ ){
    //     double last_timestamp = timestamp_speed_since_keyframe_.back();
    //     double last_speed = speed_since_keyframe_.back();

    //     timestamp_speed_since_keyframe_.clear();
    //     speed_since_keyframe_.clear();

    //     timestamp_speed_since_keyframe_.push_back(last_timestamp);
    //     speed_since_keyframe_.push_back(last_speed);
    // }

    // if( is_rotate_provided_ ){
    //     double last_timestamp = timestamp_imu_since_keyframe_.back();
    //     Eigen::Vector3d last_gyro = gyro_since_keyframe_.back();

    //     timestamp_imu_since_keyframe_.clear();
    //     gyro_since_keyframe_.clear();

    //     timestamp_imu_since_keyframe_.push_back(last_timestamp);
    //     gyro_since_keyframe_.push_back(last_gyro);
    // }
}