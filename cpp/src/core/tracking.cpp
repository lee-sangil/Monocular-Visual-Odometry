#include "core/MVO.hpp"
#include "core/random.hpp"
#include "core/time.hpp"
#include "core/DepthFilter.hpp"

#include <exception>

bool MVO::extractFeatures(){
    // Update features using KLT tracker
    if( this->updateFeatures() ){
        std::cerr << "# Update features: " << lsi::toc() << std::endl;
        
        // Delete features which is failed to track by KLT tracker
        this->deleteDeadFeatures();
        std::cerr << "# Delete features: " << lsi::toc() << std::endl;
        
        // Add features to the number of the lost features
        this->addFeatures();

        // Add extra feature points
        this->addExtraFeatures();

        std::cerr << "# Add features: " << lsi::toc() << std::endl;
        
        return true;
    }
    else
        return false;
}

bool MVO::updateFeatures(){

    if( this->num_feature_ ){
        // Track the points
        std::vector<cv::Point2f> points;
        std::vector<bool> validity;
        this->kltTracker(points, validity);
        std::cerr << "## KLT tracker: " << lsi::toc() << std::endl;

        for( int i = 0; i < this->num_feature_; i++ ){
            if( validity[i] && this->features_[i].is_alive ){
                cv::Point2f uv_prev = this->features_[i].uv.back();
                this->features_[i].life++;
                if( this->is_rotate_provided_ )
                    this->features_[i].uv_pred = this->calculateRotWarp(uv_prev);
                this->features_[i].uv.push_back(points[i]);
                this->features_[i].is_matched = true;
                this->num_feature_matched_++;

                Eigen::Vector3d epiLine = this->fundamental_ * (Eigen::Vector3d() << uv_prev.x, uv_prev.y, 1).finished();
                double dist_from_epiline = std::abs(epiLine(0)*this->features_[i].uv_pred.x + epiLine(1)*this->features_[i].uv_pred.y + epiLine(2)) / epiLine.topRows(2).norm();
                if( this->is_rotate_provided_ && this->is_start_ && dist_from_epiline > this->params_.max_epiline_dist )
                    this->features_[i].type = Type::Dynamic;
            }else
                this->features_[i].is_alive = false;
        }

        if( this->num_feature_matched_ < this->params_.th_inlier ){
            std::cerr << "There are a few FEATURE MATCHES" << std::endl;
            return false;
        }else{
            return true;
        }
    }else
        return true;
}

cv::Point2f MVO::calculateRotWarp(cv::Point2f uv){
    Eigen::Vector3d pixel, warpedPixel;
    cv::Point2f warpedUV;
    pixel << uv.x, uv.y, 1;
    warpedPixel = this->params_.K * this->rotate_prior_ * this->params_.Kinv * pixel;
    warpedUV.x = warpedPixel(0)/warpedPixel(2);
    warpedUV.y = warpedPixel(1)/warpedPixel(2);
    return warpedUV;
}

void MVO::kltTracker(std::vector<cv::Point2f>& fwd_pts, std::vector<bool>& validity){
    std::vector<cv::Point2f> pts, bwd_pts;
    pts.reserve(this->num_feature_);
    bwd_pts.reserve(this->num_feature_);
    for( int i = 0; i < this->num_feature_; i++ )
        pts.push_back(this->features_[i].uv.back());
    
    // Forward-backward error evaluation
    std::vector<cv::Mat>& prevPyr = this->prev_pyramid_template_;
    std::vector<cv::Mat>& currPyr = this->curr_pyramid_template_;
    std::cerr << "### Prepare variables: " << lsi::toc() << std::endl;

    cv::buildOpticalFlowPyramid(this->prev_image_, prevPyr, cv::Size(21,21), 3, true);
    cv::buildOpticalFlowPyramid(this->curr_image_, currPyr, cv::Size(21,21), 3, true);
    std::cerr << "### Build pyramids: " << lsi::toc() << std::endl;

    cv::Mat status, err;
    cv::calcOpticalFlowPyrLK(prevPyr, currPyr, pts, fwd_pts, status, err);
    cv::calcOpticalFlowPyrLK(currPyr, prevPyr, fwd_pts, bwd_pts, status, err);
    std::cerr << "### Calculate optical flows: " << lsi::toc() << std::endl;
    
    // Calculate bi-directional error( = validity ): validity = ~border_invalid & error_valid
    
    // WARNING: heavy computational load
    // cv::Mat desc;
    // std::vector<cv::KeyPoint> keypoints;
    // for( uint32_t i = 0; i < pts.size(); i++ )
    //     keypoints.emplace_back(fwd_pts[i],1.0);
    // this->descriptor->compute(this->curr_image_, keypoints, desc);
    // bool desc_valid;

    bool border_invalid, error_valid;
    validity.reserve(pts.size());
    for( uint32_t i = 0; i < pts.size(); i++ ){
        border_invalid = (fwd_pts[i].x <= 0) | (fwd_pts[i].x >= this->params_.im_size.width) | (fwd_pts[i].y <= 0) | (fwd_pts[i].y >= this->params_.im_size.height);
        error_valid = cv::norm(pts[i] - bwd_pts[i]) < std::min( (double) cv::norm(pts[i] - fwd_pts[i])/5.0, 1.0);
        // desc_valid = cv::norm(this->features_[i].desc - desc.row(i));

        validity.push_back(!border_invalid & error_valid);
        // bool valid = ~border_invalid & status.at<uchar>(i);// & err.at<float>(i) < std::min( cv::norm(pts[i] - fwd_pts[i])/5.0, 2.0);
        // validity.push_back(valid);
    }
}

void MVO::deleteDeadFeatures(){
    for( uint32_t i = 0; i < this->features_.size(); ){
        if( this->features_[i].is_alive == false ){
            if( this->features_[i].life > 2 ){
                this->features_dead_.push_back(this->features_[i]);
            }
            this->features_.erase(this->features_.begin()+i);
        }else{
            i++;
        }
    }
    this->num_feature_ = this->features_.size();
}

void MVO::addFeatures(){
    this->updateBucket();
    std::cerr << "## Update bucket: " << lsi::toc() << std::endl;

    while( this->num_feature_ < this->bucket_.max_features && this->bucket_.saturated.any() == true )
        this->addFeature();

    // WARNING: descriptor may remove keypoint whose description cannot be extracted
    // cv::Mat desc;
    // std::vector<cv::KeyPoint> keypoints;
    // for( int i = 0; i < this->num_feature_; i++ )
    //     if( this->features_[i].life == 1 )
    //         keypoints.emplace_back(cv::Point2f(this->features_[i].uv.back().x,this->features_[i].uv.back().y),1.0);
    // this->descriptor->compute(this->curr_image_, keypoints, desc);
    // int j = 0;
    // for( int i = 0; i < this->num_feature_; i++ )
    //     if( this->features_[i].life == 1 )
    //         this->features_[i].desc = desc.row(j++).clone();
}

void MVO::updateBucket(){
    this->bucket_.mass.fill(0.0);
    this->bucket_.saturated.fill(1.0);
    for( int i = 0; i < this->num_feature_; i++ ){
        uint32_t row_bucket = std::floor(this->features_[i].uv.back().y / this->params_.im_size.height * this->bucket_.grid.height);
        uint32_t col_bucket = std::floor(this->features_[i].uv.back().x / this->params_.im_size.width * this->bucket_.grid.width);
        this->features_[i].bucket = cv::Point(col_bucket, row_bucket);
        this->bucket_.mass(row_bucket, col_bucket)++;
    }
}

void MVO::addFeature(){
    // Load bucket parameters
    cv::Size bkSize = this->bucket_.size;
    uint32_t bucket_safety = this->bucket_.safety;

    // Choose ROI based on the probabilistic approaches with the mass of bucket
    int row, col;
    lsi::idx_randselect(this->bucket_.prob, this->bucket_.saturated, row, col);
    cv::Rect roi = cv::Rect(col*bkSize.width+1, row*bkSize.height+1, bkSize.width, bkSize.height);

    roi.x = std::max(bucket_safety, (uint32_t)roi.x);
    roi.y = std::max(bucket_safety, (uint32_t)roi.y);
    roi.width = std::min(this->params_.im_size.width-bucket_safety, (uint32_t)roi.x+roi.width)-roi.x;
    roi.height = std::min(this->params_.im_size.height-bucket_safety, (uint32_t)roi.y+roi.height)-roi.y;

    // Seek index of which feature is extracted specific bucket
    std::vector<uint32_t> idx_belong_to_bucket;
    idx_belong_to_bucket.reserve(this->num_feature_);

    for( int l = 0; l < this->num_feature_; l++ ){
        for( int ii = std::max(col-1,0); ii <= std::min(col+1,this->bucket_.grid.width-1); ii++){
            for( int jj = std::max(row-1,0); jj <= std::min(row+1,this->bucket_.grid.height-1); jj++){
                if( (this->features_[l].bucket.x == ii) & (this->features_[l].bucket.y == jj)){
                    idx_belong_to_bucket.push_back(l);
                }
            }
        }
    }
    uint32_t num_feature_inside_bucket = idx_belong_to_bucket.size();
    
    // Try to find a seperate feature
    cv::Mat crop_image;
    std::vector<cv::Point2f> keypoints;

    crop_image = this->curr_image_(roi);
    cv::goodFeaturesToTrack(crop_image, keypoints, 50, 0.1, 2.0, cv::noArray(), 3, true);
    
    if( keypoints.size() == 0 ){
        this->bucket_.saturated(row,col) = 0.0;
        return;
    }else{
        for( uint32_t l = 0; l < keypoints.size(); l++ ){
            keypoints[l].x = keypoints[l].x + roi.x - 1;
            keypoints[l].y = keypoints[l].y + roi.y - 1;
        }
    }

    bool success;
    double dist, min_dist, max_min_dist = 0;
    cv::Point2f best_keypoint;
    for( uint32_t l = 0; l < keypoints.size(); l++ ){
        success = true;
        min_dist = 1e9; // enough-large number
        for( uint32_t f = 0; f < num_feature_inside_bucket; f++ ){
            dist = cv::norm(keypoints[l] - this->features_[idx_belong_to_bucket[f]].uv.back());
            
            if( dist < min_dist )
                min_dist = dist;
            
            if( dist < this->params_.min_px_dist ){
                success = false;
                break;
            }
        }
        if( success ){
            if( min_dist > max_min_dist){
                max_min_dist = min_dist;
                best_keypoint = keypoints[l];
            }
        }
    }
    
    if( max_min_dist > 0.0 ){
        // Add new feature to VO object
        Feature newFeature;

        newFeature.id = Feature::new_feature_id; // unique id of the feature
        newFeature.frame_init = 0; // frame step when the 3d point is initialized
        newFeature.uv.emplace_back(best_keypoint.x, best_keypoint.y); // uv point in pixel coordinates
        newFeature.uv_pred = cv::Point2f(-1,-1);
        newFeature.life = 1; // the number of frames in where the feature is observed
        newFeature.bucket = cv::Point(col, row); // the location of bucket where the feature belong to
        newFeature.point_curr << 0,0,0,1; // 3-dim homogeneous point in the local coordinates
        newFeature.is_alive = true;
        newFeature.is_matched = false; // matched between both frame
        newFeature.is_wide = false; // verify whether features btw the initial and current are wide enough
        newFeature.is_2D_inliered = false; // belong to major (or meaningful) movement
        newFeature.is_3D_reconstructed = false; // triangulation completion
        newFeature.is_3D_init = false; // scale-compensated
        newFeature.point_init << 0,0,0,1; // scale-compensated 3-dim homogeneous point in the global coordinates
        newFeature.point_var = 1e9;
        newFeature.type = Type::Unknown;
        newFeature.depthfilter = new DepthFilter();

        this->features_.push_back(newFeature);
        this->num_feature_++;
    
        Feature::new_feature_id++;

        // Update bucket
        this->bucket_.mass(row, col)++;

        cv::eigen2cv(this->bucket_.mass, this->bucket_.cv_mass);
        cv::GaussianBlur(this->bucket_.cv_mass, this->bucket_.cv_prob, cv::Size(21,21), 3.0);
        cv::cv2eigen(this->bucket_.cv_prob, this->bucket_.prob);

        this->bucket_.prob.array() += 0.05;
        this->bucket_.prob = this->bucket_.prob.cwiseInverse();

        // Assign high weight for ground
        for( int i = 0; i < this->bucket_.prob.rows(); i++ ){
            this->bucket_.prob.block(i,0,1,this->bucket_.prob.cols()) *= i+1;
        }
    }else{
        this->bucket_.saturated(row,col) = 0.0;
    }
}

// haram
bool MVO::calculateEssential()
{
    if (this->step_ == 0)
        return true;

    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;
    points1.reserve(this->num_feature_);
    points2.reserve(this->num_feature_);

    std::vector<uint32_t> idx_static;
    int num_wide_feature = 0;
    int key_idx;
    for( int i = 0; i < this->num_feature_; i++ ){
        key_idx = this->features_[i].life - 1 - (this->step_ - this->keystep_);
        if( key_idx >= 0 && this->features_[i].type != Type::Dynamic ){
            points1.push_back(this->features_[i].uv[key_idx]);
            points2.push_back(this->features_[i].uv.back());   // latest
            idx_static.push_back(i);
            if( cv::norm(this->features_[i].uv[key_idx] - this->features_[i].uv.back()) > this->params_.th_px_wide ){
                this->features_[i].is_wide = true;
                num_wide_feature++;
            }else{
                this->features_[i].is_wide = false;
            }
        }
    }

    if( points1.size() <= this->num_feature_ * this->params_.th_ratio_keyframe ){
        this->keystep_array_.push_back(this->step_);

        if( this->is_speed_provided_ ){
            double last_timestamp = this->timestamp_since_keyframe_.back();
            double last_speed = this->speed_since_keyframe_.back();

            this->timestamp_since_keyframe_.clear();
            this->speed_since_keyframe_.clear();

            this->timestamp_since_keyframe_.push_back(last_timestamp);
            this->speed_since_keyframe_.push_back(last_speed);
        }

        if( this->is_rotate_provided_ ){
            double last_timestamp = this->timestamp_since_keyframe_.back();
            Eigen::Vector3d last_gyro = this->gyro_since_keyframe_.back();

            this->timestamp_since_keyframe_.clear();
            this->gyro_since_keyframe_.clear();

            this->timestamp_since_keyframe_.push_back(last_timestamp);
            this->gyro_since_keyframe_.push_back(last_gyro);
        }

        std::cerr << "key step: " << this->keystep_ << ' ' << std::endl;
    }

    cv::Mat inlier_mat;
    this->essential_ = cv::findEssentialMat(points1, points2, this->params_.Kcv, cv::RANSAC, 0.999, 1.5, inlier_mat);
    std::cerr << "# Calculate essential: " << lsi::toc() << std::endl;
    
    Eigen::Matrix3d E_;
    cv::cv2eigen(this->essential_, E_);
    this->fundamental_ = this->params_.Kinv.transpose() * E_ * this->params_.Kinv;

    double error;
    std::vector<double> essential_error;
    for( uint32_t i = 0; i < points1.size(); i++ ){
        error = (Eigen::Vector3d() << points2[i].x,points2[i].y,1).finished().transpose() * this->params_.Kinv.transpose() * E_ * this->params_.Kinv * (Eigen::Vector3d() << points1[i].x,points1[i].y,1).finished();
        essential_error.push_back(error);
    }

    uint32_t inlier_cnt = 0;
    bool* inlier = inlier_mat.ptr<bool>(0);
    for (int i = 0; i < inlier_mat.rows; i++){
        if (inlier[i]){
            this->features_[idx_static[i]].is_2D_inliered = true;
            inlier_cnt++;
        // }else if( essential_error[i] > 1e-4 ){
        }else{
            this->features_[idx_static[i]].type = Type::Dynamic;
        }
    }
    this->num_feature_2D_inliered_ = inlier_cnt;
    std::cerr << "num_feature_2D_inliered_: " << (double) this->num_feature_2D_inliered_ / this->num_feature_ * 100 << '%' << std::endl;

    Eigen::Matrix3d U,V;
    switch( this->params_.svd_method){
        case MVO::SVD::JACOBI:{
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(E_, Eigen::ComputeThinU | Eigen::ComputeThinV);
            U = svd.matrixU();
            V = svd.matrixV();
            break;
        }
        case MVO::SVD::OpenCV:{
            cv::Mat Vt, U_, W;
            cv::SVD::compute(this->essential_, W, U_, Vt);

            Eigen::MatrixXd Vt_;
            cv::cv2eigen(Vt, Vt_);
            V = Vt_.transpose();
            cv::cv2eigen(U_, U);
            break;
        }
        case MVO::SVD::BDC:
        default:{
            Eigen::Matrix3d E_;
            cv::cv2eigen(this->essential_, E_);
            Eigen::BDCSVD<Eigen::MatrixXd> svd(E_, Eigen::ComputeThinU | Eigen::ComputeThinV);
            U = svd.matrixU();
            V = svd.matrixV();
            break;
        }
    }

    if (U.determinant() < 0)
        U.block(0, 2, 3, 1) = -U.block(0, 2, 3, 1);
    if (V.determinant() < 0)
        V.block(0, 2, 3, 1) = -V.block(0, 2, 3, 1);

    Eigen::Matrix3d W;
    W << 0, -1, 0, 1, 0, 0, 0, 0, 1;

    this->R_vec_.clear();
    this->t_vec_.clear();
    this->R_vec_.push_back(U * W * V.transpose());
    this->R_vec_.push_back(U * W * V.transpose());
    this->R_vec_.push_back(U * W.transpose() * V.transpose());
    this->R_vec_.push_back(U * W.transpose() * V.transpose());
    this->t_vec_.push_back(U.block(0, 2, 3, 1));
    this->t_vec_.push_back(-U.block(0, 2, 3, 1));
    this->t_vec_.push_back(U.block(0, 2, 3, 1));
    this->t_vec_.push_back(-U.block(0, 2, 3, 1));

    std::cerr << "# Extract R, t: " << lsi::toc() << std::endl;

    if (this->num_feature_2D_inliered_ < this->params_.th_inlier){
        std::cerr << " There are a few inliers matching features in 2D." << std::endl;
        return false;
    }else{
        this->is_start_ = true;
        return true;
    }
}

void MVO::addExtraFeatures(){

    if( this->features_extra_.size() > 0 ){
        for( uint32_t i = 0; i < this->features_extra_.size(); i++ ){
            this->features_extra_[i].id = Feature::new_feature_id;
            this->features_.push_back(this->features_extra_[i]);
            this->num_feature_++;
        
            Feature::new_feature_id++;

            // Update bucket
            this->bucket_.mass(this->features_extra_[i].bucket.y, this->features_extra_[i].bucket.x)++;
        }
        cv::eigen2cv(this->bucket_.mass, this->bucket_.cv_mass);
        cv::GaussianBlur(this->bucket_.cv_mass, this->bucket_.cv_prob, cv::Size(21,21), 3.0);
        cv::cv2eigen(this->bucket_.cv_prob, this->bucket_.prob);

        this->bucket_.prob.array() += 0.05;
        this->bucket_.prob = this->bucket_.prob.cwiseInverse();

        // Assign high weight for ground
        for( int i = 0; i < this->bucket_.prob.rows(); i++ ){
            this->bucket_.prob.block(i,0,1,this->bucket_.prob.cols()) *= i+1;
        }

        this->features_extra_.clear();
    }
}

void MVO::extractRoiFeatures(std::vector<cv::Rect> rois, std::vector<int> num_feature_){

    cv::Rect roi;
    for( uint32_t i = 0; i < rois.size(); i++ ){
        roi = rois[i];
        int bucket_safety = this->bucket_.safety;
        roi.x = std::max(bucket_safety, roi.x);
        roi.y = std::max(bucket_safety, roi.y);
        roi.width = std::min(this->params_.im_size.width-bucket_safety, roi.x+roi.width)-roi.x;
        roi.height = std::min(this->params_.im_size.height-bucket_safety, roi.y+roi.height)-roi.y;

        int num_success = 0;
        int num_try = 0;
        while( num_success < num_feature_[i] && num_try++ < 3 )
            if( this->extractRoiFeature(roi) )
                num_success++;
    }
}

bool MVO::extractRoiFeature(cv::Rect& roi){

    int row, col;
    row = (roi.x-1) / this->bucket_.size.height;
    col = (roi.y-1) / this->bucket_.size.width;

    // Seek index of which feature is extracted specific bucket
    std::vector<uint32_t> idx_belong_to_bucket;
    idx_belong_to_bucket.reserve(this->num_feature_);

    for( int l = 0; l < this->num_feature_; l++ ){
        for( int ii = std::max(col-1,0); ii <= std::min(col+1,this->bucket_.grid.width-1); ii++){
            for( int jj = std::max(row-1,0); jj <= std::min(row+1,this->bucket_.grid.height-1); jj++){
                if( (this->features_[l].bucket.x == ii) & (this->features_[l].bucket.y == jj)){
                    idx_belong_to_bucket.push_back(l);
                }
            }
        }
    }
    uint32_t num_feature_inside_bucket = idx_belong_to_bucket.size();
    
    // Try to find a seperate feature
    cv::Mat crop_image;
    std::vector<cv::Point2f> keypoints;

    try{
        crop_image = this->curr_image_(roi);
        cv::goodFeaturesToTrack(crop_image, keypoints, 10, 0.1, 2.0, cv::noArray(), 3, true);
    }catch(std::exception& msg){
        std::cerr << msg.what() << std::endl;
        return false;
    }
    
    if( keypoints.size() > 0 ){
        for( uint32_t l = 0; l < keypoints.size(); l++ ){
            keypoints[l].x = keypoints[l].x + roi.x - 1;
            keypoints[l].y = keypoints[l].y + roi.y - 1;
        }
    }else{
        return false;
    }

    bool success;
    double dist, min_dist, max_min_dist = 0;
    cv::Point2f best_keypoint;
    for( uint32_t l = 0; l < keypoints.size(); l++ ){
        success = true;
        min_dist = 1e9; // enough-large number
        for( uint32_t f = 0; f < num_feature_inside_bucket; f++ ){
            dist = cv::norm(keypoints[l] - this->features_[idx_belong_to_bucket[f]].uv.back());
            
            if( dist < min_dist )
                min_dist = dist;

            if( dist < this->params_.min_px_dist/2+1 ){
                success = false;
                break;
            }
        }
        if( success ){
            if( min_dist > max_min_dist){
                max_min_dist = min_dist;
                best_keypoint = keypoints[l];
            }
        }
    }
    
    if( max_min_dist > 0.0 ){
        // Add new feature to VO object
        Feature newFeature;

        newFeature.frame_init = 0; // frame step when the 3d point is initialized
        newFeature.uv.emplace_back(best_keypoint.x, best_keypoint.y); // uv point in pixel coordinates
        newFeature.uv_pred = cv::Point2f(-1,-1);
        newFeature.life = 1; // the number of frames in where the feature is observed
        newFeature.bucket = cv::Point(col, row); // the location of bucket where the feature belong to
        newFeature.point_curr << 0,0,0,1; // 3-dim homogeneous point in the local coordinates
        newFeature.is_alive = true;
        newFeature.is_matched = false; // matched between both frame
        newFeature.is_wide = false; // verify whether features btw the initial and current are wide enough
        newFeature.is_2D_inliered = false; // belong to major (or meaningful) movement
        newFeature.is_3D_reconstructed = false; // triangulation completion
        newFeature.is_3D_init = false; // scale-compensated
        newFeature.point_init << 0,0,0,1; // scale-compensated 3-dim homogeneous point in the global coordinates
        newFeature.point_var = 1e9;
        newFeature.type = Type::Unknown;
        newFeature.depthfilter = new DepthFilter();

        this->features_extra_.push_back(newFeature);
        return true;
    }else{
        return false;
    }
    
}