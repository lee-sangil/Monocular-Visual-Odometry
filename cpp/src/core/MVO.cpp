#include "core/MVO.hpp"
#include "core/random.hpp"
#include "core/numerics.hpp"
#include "core/time.hpp"
#include "core/DepthFilter.hpp"

double DepthFilter::s_px_error_angle_;
double DepthFilter::s_meas_max_;
double MVO::s_scale_reference_ = -1;
double MVO::s_scale_reference_weight_;
uint32_t Feature::new_feature_id = 0;

MVO::MVO(){
    this->step_ = -1;
    this->keystep_ = 0;
    this->keystep_array_.push_back(0);

    this->is_start_ = false;
    this->is_scale_initialized_ = false;
    this->is_speed_provided_ = false;
    this->is_rotate_provided_ = false;
    this->cvClahe_ = cv::createCLAHE();

    // Variables
    this->num_feature_ = 0;
    this->num_feature_matched_ = 0;
    this->num_feature_2D_inliered_ = 0;
    this->num_feature_3D_reconstructed_ = 0;
    this->num_feature_inlier_ = 0;

    // Initial position
    this->TRec_.push_back(Eigen::Matrix4d::Identity());
    this->TocRec_.push_back(Eigen::Matrix4d::Identity());
    this->PocRec_.push_back((Eigen::Vector4d() << 0,0,0,1).finished());

    this->R_vec_.reserve(4);
	this->t_vec_.reserve(4);

    this->rotate_prior_ = Eigen::Matrix3d::Identity();

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

	this->params_.fx =			    fSettings["Camera.fx"];
	this->params_.fy =			    fSettings["Camera.fy"];
	this->params_.cx =			    fSettings["Camera.cx"];
	this->params_.cy =			    fSettings["Camera.cy"];
	this->params_.k1 =			    fSettings["Camera.k1"];
	this->params_.k2 =			    fSettings["Camera.k2"];
	this->params_.p1 =			    fSettings["Camera.p1"];
	this->params_.p2 =			    fSettings["Camera.p2"];
	this->params_.k3 =			    fSettings["Camera.k3"];
	this->params_.im_size.width =	fSettings["Camera.width"];
	this->params_.im_size.height =	fSettings["Camera.height"];

    this->params_.K << this->params_.fx, 0, this->params_.cx,
						0, this->params_.fy, this->params_.cy,
						0, 0, 1;
    this->params_.Kinv = this->params_.K.inverse();
    cv::eigen2cv(this->params_.K, this->params_.Kcv);

	this->params_.radial_distortion.push_back(this->params_.k1);
	this->params_.radial_distortion.push_back(this->params_.k2);
	this->params_.radial_distortion.push_back(this->params_.k3);
	this->params_.tangential_distortion.push_back(this->params_.p1);
	this->params_.tangential_distortion.push_back(this->params_.p2);
    this->params_.dist_coeffs.insert(this->params_.dist_coeffs.begin(), this->params_.radial_distortion.begin(), this->params_.radial_distortion.begin()+1);
    this->params_.dist_coeffs.insert(this->params_.dist_coeffs.end(), this->params_.tangential_distortion.begin(), this->params_.tangential_distortion.end());
    this->params_.dist_coeffs.insert(this->params_.dist_coeffs.begin(), this->params_.radial_distortion.begin()+2, this->params_.radial_distortion.end());

    cv::initUndistortRectifyMap(this->params_.Kcv, this->params_.dist_coeffs, cv::Mat(), this->params_.Kcv, this->params_.im_size, CV_32FC1, this->distort_map1_, this->distort_map2_);

    if( !fSettings["Camera.T_IC"].empty() ){
        cv::Mat Tic;
        fSettings["Camera.T_IC"] >> Tic;
        cv::cv2eigen(Tic, this->params_.Tic);
        this->params_.Tci = this->params_.Tic.inverse();
    }else if( !fSettings["Camera.T_CI"].empty() ){
        cv::Mat Tci;
        fSettings["Camera.T_CI"] >> Tci;
        cv::cv2eigen(Tci, this->params_.Tci);
        this->params_.Tic = this->params_.Tci.inverse();
    }else if( !fSettings["Camera.T_IV"].empty() && !fSettings["Camera.T_VC"].empty() ){
        Eigen::Matrix4d Tiv_, Tvc_;
        cv::Mat Tiv, Tvc;
        fSettings["Camera.T_IV"] >> Tiv;
        fSettings["Camera.T_VC"] >> Tvc;
        cv::cv2eigen(Tiv, Tiv_);
        cv::cv2eigen(Tvc, Tvc_);

        this->params_.Tci = Tvc_ * Tiv_;
        this->params_.Tic = this->params_.Tci.inverse();
    }

    // this->descriptor = cv::BRISK::create();

    this->params_.th_inlier =          fSettings["Feature.minimum_num_inlier"];
    this->params_.th_ratio_keyframe =  fSettings["Feature.minimum_matching_ratio"];
    this->params_.min_px_dist =        fSettings["Feature.minimum_distance_between"];
    this->params_.th_px_wide =         fSettings["Feature.minimum_triangulation_baseline"];
    this->params_.max_epiline_dist =   fSettings["Feature.maximum_epiline_distance"];

    // RANSAC parameter
    this->params_.ransac_coef_scale.max_iteration =      fSettings["RANSAC.maximum_iteration"];
    this->params_.ransac_coef_scale.th_inlier_ratio =    fSettings["RANSAC.threshold_inlier_ratio"];
    this->params_.ransac_coef_scale.min_num_point =      fSettings["RANSAC.scale.num_sample"];
    this->params_.ransac_coef_scale.th_dist =            fSettings["RANSAC.scale.threshold_dist_inlier"]; // standard deviation
    this->params_.ransac_coef_scale.th_dist_outlier =    fSettings["RANSAC.scale.threshold_dist_outlier"]; // three times of standard deviation
    this->params_.ransac_coef_scale.calculate_func = MVO::calculateScale;
    this->params_.ransac_coef_scale.calculate_dist = MVO::calculateScaleError;
    
    this->params_.ransac_coef_plane.max_iteration =      fSettings["RANSAC.maximum_iteration"];
    this->params_.ransac_coef_plane.th_inlier_ratio =    fSettings["RANSAC.threshold_inlier_ratio"];
    this->params_.ransac_coef_plane.min_num_point =      3;
    this->params_.ransac_coef_plane.th_dist =            fSettings["RANSAC.plane.threshold_dist_inlier"]; // standard deviation
    this->params_.ransac_coef_plane.th_dist_outlier =    fSettings["RANSAC.plane.threshold_dist_outlier"]; // three times of standard deviation
    this->params_.ransac_coef_plane.calculate_func = MVO::calculatePlane;
    this->params_.ransac_coef_plane.calculate_dist = MVO::calculatePlaneError;

    // Bucket
    this->bucket_ = Bucket();
    this->bucket_.max_features =     fSettings["Feature.number"];
    this->bucket_.safety =           fSettings["Bucket.border_safety"];
    int bucket_grid_rows =          fSettings["Bucket.rows"];
    int bucket_grid_cols =          fSettings["Bucket.cols"];

	this->bucket_.grid = cv::Size(bucket_grid_cols,bucket_grid_rows);
	this->bucket_.size = cv::Size(this->params_.im_size.width/this->bucket_.grid.width, this->params_.im_size.height/this->bucket_.grid.height);
	this->bucket_.mass.setZero(this->bucket_.grid.height, this->bucket_.grid.width);
	this->bucket_.prob.resize(this->bucket_.grid.height, this->bucket_.grid.width);
    this->bucket_.prob.fill(1.0);
    this->bucket_.saturated.setZero(this->bucket_.grid.height, this->bucket_.grid.width);

    this->features_.reserve(this->bucket_.max_features);
    this->features_backup_.reserve(this->bucket_.max_features);
    this->prev_pyramid_template_.reserve(10);
    this->curr_pyramid_template_.reserve(10);

    // 3D reconstruction
    this->params_.init_scale =           1;
    this->params_.vehicle_height =       fSettings["Scale.reference_height"]; // in meter
    this->params_.weight_scale_ref =     fSettings["Scale.reference_weight"];
	this->params_.weight_scale_reg =     fSettings["Scale.regularization_weight"];
    this->params_.reproj_error =         fSettings["PnP.reprojection_error"];
    this->params_.update_init_point =    fSettings["Debug.update_init_points"];
    this->params_.mapping_option =       fSettings["Debug.mapping_options"];

    this->eigen_solver_ = new Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>();

    switch( fSettings["Triangulation.method"] ){
        case 0:
            this->params_.triangulation_method = MVO::TRIANGULATION::MIDP;
            break;
        case 1:
            this->params_.triangulation_method = MVO::TRIANGULATION::LLS;
            break;
        default:
            abort();
    }
    
    switch( fSettings["Triangulation.SVD"] ){
        case 0:
            this->params_.svd_method = MVO::SVD::JACOBI;
            break;
        case 1:
            this->params_.svd_method = MVO::SVD::BDC;
            break;
        case 2:
            this->params_.svd_method = MVO::SVD::OpenCV;
            break;
        case 3:
            this->params_.svd_method = MVO::SVD::Eigen;
            break;
        default:
            abort();
    }

    switch( fSettings["PnP.method"] ){
        case 0:
            this->params_.pnp_method = MVO::PNP::LM;
            break;
        case 1:
            this->params_.pnp_method = MVO::PNP::ITERATIVE;
            break;
        case 2:
            this->params_.pnp_method = MVO::PNP::AP3P;
            break;
        case 3:
            this->params_.pnp_method = MVO::PNP::EPNP;
            break;
        case 4:
            this->params_.pnp_method = MVO::PNP::DLS;
            break;
        case 5:
            this->params_.pnp_method = MVO::PNP::UPNP;
            break;
        default:
            abort();
    }

    this->params_.max_point_var =   fSettings["DepthFilter.maximum_variance"];
    double depth_min =              fSettings["DepthFilter.minimum_depth"];
    double px_noise =               fSettings["DepthFilter.pixel_noise"];
    DepthFilter::s_meas_max_ = 1.0/depth_min;
    DepthFilter::s_px_error_angle_ = std::atan(px_noise/(this->params_.fx+this->params_.fy))*2.0; // law of chord (sehnensatz)

    cv::namedWindow("MVO");
    cv::moveWindow("MVO", 20, 20);

    cv::namedWindow("Trajectory");
    cv::moveWindow("Trajectory", 1320, 20);

    cv::namedWindow("Depth");
    cv::moveWindow("Depth", 20, 440);

    this->params_.view.height_default =   fSettings["viewCam.height"]; // in world coordinate
    this->params_.view.roll_default =     (double) fSettings["viewCam.roll"] * M_PI/180; // radian
    this->params_.view.pitch_default =    (double) fSettings["viewCam.pitch"] * M_PI/180; // radian
    this->params_.view.height =          this->params_.view.height_default;
    this->params_.view.roll =            this->params_.view.roll_default;
    this->params_.view.pitch =           this->params_.view.pitch_default;
    this->params_.view.im_size = cv::Size(600,600);
    this->params_.view.K <<  300,   0, 300,
						      0, 300, 300,
						      0,   0,   1;
    this->params_.view.upper_left =   cv::Point3d(-1,-.5, 1);
    this->params_.view.upper_right =  cv::Point3d( 1,-.5, 1);
    this->params_.view.lower_left =   cv::Point3d(-1, .5, 1);
    this->params_.view.lower_right =  cv::Point3d( 1, .5, 1);
}

void MVO::refresh(){
    this->num_feature_matched_ = 0;
    this->num_feature_2D_inliered_ = 0;
    this->num_feature_3D_reconstructed_ = 0;
    this->num_feature_inlier_ = 0;

    for( int i = 0; i < this->num_feature_; i++ ){
        this->features_[i].is_matched = false;
        this->features_[i].is_2D_inliered = false;
        this->features_[i].is_3D_reconstructed = false;
    }
}

void MVO::restart(){
    this->step_ = -1;
    this->keystep_ = 0;
    this->keystep_array_.clear();
    this->keystep_array_.push_back(0);

    this->is_start_ = false;
    this->is_scale_initialized_ = false;
    
    this->features_.clear();
    this->features_dead_.clear();

    // Variables
    this->num_feature_ = 0;
    this->num_feature_matched_ = 0;
    this->num_feature_2D_inliered_ = 0;
    this->num_feature_3D_reconstructed_ = 0;
    this->num_feature_inlier_ = 0;

    // Initial position
    this->TRec_.clear();
    this->TocRec_.clear();
    this->PocRec_.clear();

    this->TRec_.push_back(Eigen::Matrix4d::Identity());
    this->TocRec_.push_back(Eigen::Matrix4d::Identity());
    this->PocRec_.push_back((Eigen::Vector4d() << 0,0,0,1).finished());
}

void MVO::setImage(cv::Mat& image){
    this->prev_image_ = this->curr_image_.clone();
    cv::remap(image, this->undistorted_image_, this->distort_map1_, this->distort_map2_, cv::INTER_AREA);
    
    if( this->params_.apply_clahe )
        cvClahe_->apply(this->undistorted_image_, this->curr_image_);
    else
        this->curr_image_ = this->undistorted_image_.clone();

    this->step_++;
    this->keystep_ = this->keystep_array_.back();
}

void MVO::run(cv::Mat& image){
    
    lsi::tic();
    this->setImage(image);
    std::cerr << "============ Iteration: " << this->step_ << " ============" << std::endl;
    std::cerr << "# Grab image: " << lsi::toc() << std::endl;
    this->refresh();

    // this->extract_roi_features(rois, num_feature_);   // Extract extra features in rois

    std::array<bool,3> success;
    success[0] = this->extractFeatures();       // Extract and update features
    success[1] = this->calculateEssential();    // RANSAC for calculating essential/fundamental matrix
    success[2] = this->calculateMotion();       // Extract rotational and translational from fundamental matrix

    if( !std::all_of(success.begin(), success.end(), [](bool b){return b;}) )
        this->restart();
}

void MVO::updateTimestamp(double timestamp){
    this->timestamp_since_keyframe_.push_back(timestamp);
}

void MVO::updateGyro(Eigen::Vector3d& gyro){
    this->is_rotate_provided_ = true;
    this->gyro_since_keyframe_.push_back(gyro);

    Eigen::Vector3d radian = Eigen::Vector3d::Zero();
    for( uint32_t i = 0; i < this->gyro_since_keyframe_.size()-1; i++ )
        radian += (this->gyro_since_keyframe_[i]+this->gyro_since_keyframe_[i+1])/2 * (this->timestamp_since_keyframe_[i+1]-this->timestamp_since_keyframe_[i]);

    this->rotate_prior_ = this->params_.Tci.block(0,0,3,3) * skew(-radian).exp() * this->params_.Tic.block(0,0,3,3);
}

void MVO::updateVelocity(double speed){
    this->is_speed_provided_ = true;
    this->speed_since_keyframe_.push_back(speed);

    double scale = 0.0;
    for( uint32_t i = 0; i < this->speed_since_keyframe_.size()-1; i++ )
        scale += (this->speed_since_keyframe_[i]+this->speed_since_keyframe_[i+1])/2 * (this->timestamp_since_keyframe_[i+1]-this->timestamp_since_keyframe_[i]);
    
    this->updateScaleReference(scale);
}

std::vector<Feature> MVO::getFeatures() const {
    return this->features_;
}

std::vector< std::tuple<cv::Point2f, Eigen::Vector3d> > MVO::getPoints() const
{
    std::vector< std::tuple<cv::Point2f, Eigen::Vector3d> > ptsROI;
    for( uint32_t i = 0; i < this->features_.size(); i++ ){
        ptsROI.push_back( std::make_tuple(this->features_[i].uv.back(), this->features_[i].point_curr.block(0, 0, 3, 1) ) );
    }
    return ptsROI;
}