#include "core/MVO.hpp"
#include "core/random.hpp"
#include "core/time.hpp"
#include "core/DepthFilter.hpp"

#include <exception>

bool MVO::extractFeatures(){
    // Update features using KLT tracker
    if( updateFeatures() ){
        if( MVO::s_print_log ) std::cerr << "# Update features: " << lsi::toc() << std::endl;
        
        if( MVO::s_print_log ) std::cerr << "* Tracking rate: " << 100.0 * num_feature_matched_ / num_feature_ << std::endl;

        // Delete features which is failed to track by KLT tracker
        deleteDeadFeatures();
        if( MVO::s_print_log ) std::cerr << "# Delete features: " << lsi::toc() << std::endl;
        
        // Add features to the number of the lost features
        addFeatures();

        // Add extra feature points
        addExtraFeatures();

        if( MVO::s_print_log ) std::cerr << "# Add features: " << lsi::toc() << std::endl;
        
        return true;
    }
    else
        return false;
}

bool MVO::updateFeatures(){

    if( num_feature_ ){
        // Track the points
        std::vector<cv::Point2f> points;
        std::vector<bool> validity;
        if( !kltTracker(points, validity) ){
            if( keystep_array_.size() > 0 ){
                keystep_array_.pop_back();
                keystep_ = keystep_array_.back();
                kltTracker(points, validity);
            }
        }
        if( MVO::s_print_log ) std::cerr << "## KLT tracker: " << lsi::toc() << std::endl;

        cv::Point2f uv_prev;
        Eigen::Matrix<double,3,4> Tco = TocRec_.back().inverse().block(0,0,3,4);
        for( int i = 0; i < num_feature_; i++ ){
            if( validity[i] && features_[i].is_alive ){
                uv_prev = features_[i].uv.back();
                features_[i].life++;

                if( is_rotate_provided_ ) features_[i].uv_pred = warpWithIMU(uv_prev);
                else if( features_[i].is_3D_init ) features_[i].uv_pred = warpWithPreviousMotion(Tco * features_[i].point_init);

                features_[i].uv.push_back(points[i]);
                features_[i].is_matched = true;
                num_feature_matched_++;

                if( is_rotate_provided_ ){
                    Eigen::Vector3d epiLine = fundamental_ * (Eigen::Vector3d() << uv_prev.x, uv_prev.y, 1).finished();
                    double dist_from_epiline = std::abs(epiLine(0)*features_[i].uv_pred.x + epiLine(1)*features_[i].uv_pred.y + epiLine(2)) / epiLine.topRows(2).norm();
                    if( is_start_){
                        // if( dist_from_epiline > params_.max_dist ){
                        //     features_[i].type = Type::Dynamic;
                        //     features_[i].is_3D_init = false;
                        // }else{
                        //     if( features_[i].type == Type::Dynamic ) features_[i].type = Type::Unknown;
                        // }
                    }
                }else if( features_[i].is_3D_init ){
                    double dist_from_predicted_point = cv::norm(features_[i].uv_pred - features_[i].uv.back());
                    if( is_start_ ){
                        // if( dist_from_predicted_point > params_.max_dist ){
                        //     features_[i].type = Type::Dynamic;
                        //     features_[i].is_3D_init = false;
                        // }else{
                        //     if( features_[i].type == Type::Dynamic ) features_[i].type = Type::Unknown;
                        // }
                    }
                }
                
            }else
                features_[i].is_alive = false;
        }

        if( MVO::s_print_log ){
            std::cerr << "* parallax: ";
            for( const auto & feature : features_ )
                std::cerr << feature.parallax << ' ';
            std::cerr << std::endl;
        }

        if( num_feature_matched_ < params_.th_inlier ){
            if( MVO::s_print_log ) std::cerr << "Warning: There are a few feature matches" << std::endl;
            return false;
        }else{
            return true;
        }
    }else
        return true;
}

cv::Point2f MVO::warpWithIMU(const cv::Point2f& uv){
    Eigen::Vector3d pixel, warpedPixel;
    cv::Point2f warpedUV;
    pixel << uv.x, uv.y, 1;
    warpedPixel = params_.K * rotate_prior_ * params_.Kinv * pixel;
    warpedUV.x = warpedPixel(0)/warpedPixel(2);
    warpedUV.y = warpedPixel(1)/warpedPixel(2);
    return warpedUV;
}

cv::Point2f MVO::warpWithPreviousMotion(const Eigen::Vector3d& p){
    Eigen::Vector3d warpedPixel;
    cv::Point2f warpedUV;
    Eigen::Matrix3d Rinv = TRec_.back().block(0,0,3,3).transpose();
    Eigen::Vector3d t = TRec_.back().block(0,3,3,1);
    
    warpedPixel = params_.K * Rinv * (p-t);
    warpedUV.x = warpedPixel(0)/warpedPixel(2);
    warpedUV.y = warpedPixel(1)/warpedPixel(2);
    return warpedUV;
}

bool MVO::kltTracker(std::vector<cv::Point2f>& fwd_pts, std::vector<bool>& validity){
    std::vector<cv::Point2f> pts, bwd_pts, fwd_bwd_pts;
    pts.reserve(num_feature_);
    bwd_pts.reserve(num_feature_);
    for( const auto & feature : features_ )
        pts.push_back(feature.uv.back());
    
    // Forward-backward error evaluation
    std::vector<cv::Mat>& prevPyr = prev_pyramid_template_;
    std::vector<cv::Mat>& currPyr = curr_pyramid_template_;
    if( MVO::s_print_log ) std::cerr << "### Prepare variables: " << lsi::toc() << std::endl;

    cv::buildOpticalFlowPyramid(prev_image_, prevPyr, cv::Size(21,21), 3, true);
    cv::buildOpticalFlowPyramid(curr_image_, currPyr, cv::Size(21,21), 3, true);
    if( MVO::s_print_log ) std::cerr << "### Build pyramids: " << lsi::toc() << std::endl;

    cv::Mat status, err;
    cv::calcOpticalFlowPyrLK(prevPyr, currPyr, pts, fwd_pts, status, err);
    cv::calcOpticalFlowPyrLK(currPyr, prevPyr, fwd_pts, bwd_pts, status, err);
    // cv::calcOpticalFlowPyrLK(prevPyr, currPyr, bwd_pts, fwd_bwd_pts, status, err);
    if( MVO::s_print_log ) std::cerr << "### Calculate optical flows: " << lsi::toc() << std::endl;
    
    // Calculate bi-directional error( = validity ): validity = ~border_invalid & error_valid
    
    // WARNING: heavy computational load
    // cv::Mat desc;
    // std::vector<cv::KeyPoint> keypoints;
    // for( uint32_t i = 0; i < pts.size(); i++ )
    //     keypoints.emplace_back(fwd_pts[i],1.0);
    // descriptor->compute(curr_image_, keypoints, desc);
    // bool desc_valid;

    bool border_invalid, error_valid;
    validity.reserve(pts.size());
    for( uint32_t i = 0; i < pts.size(); i++ ){
        border_invalid = (fwd_pts[i].x <= 0) | (fwd_pts[i].x >= params_.im_size.width) | (fwd_pts[i].y <= 0) | (fwd_pts[i].y >= params_.im_size.height);
        error_valid = cv::norm(pts[i] - bwd_pts[i]) < std::min( (double) cv::norm(pts[i] - fwd_pts[i])/5.0, 1.0);
        // desc_valid = cv::norm(features_[i].desc - desc.row(i));
        // if( !error_valid ){
        //     error_valid = cv::norm(fwd_pts[i] - fwd_bwd_pts[i]) < std::min( (double) cv::norm(fwd_pts[i] - bwd_pts[i])/5.0, 1.0);
        //     features_[i].uv.back() = bwd_pts[i];
        // }

        validity.push_back(!border_invalid & error_valid);
        // bool valid = !border_invalid & status.at<uchar>(i);// & err.at<float>(i) < std::min( cv::norm(pts[i] - fwd_pts[i])/5.0, 2.0);
        // validity.push_back(valid);
    }

    // Validate parallax
    int key_idx;
    cv::Point2f uv_keystep;
    for( uint32_t i = 0; i < num_feature_; i++ ){
        if( validity[i] && features_[i].is_alive ){
            key_idx = features_[i].life - 1 - (step_ - keystep_);
            if( key_idx >= 0 ){
                uv_keystep = features_[i].uv[key_idx];
                features_[i].parallax = std::acos((fwd_pts[i].dot(uv_keystep)+1)/std::sqrt(fwd_pts[i].x*fwd_pts[i].x + fwd_pts[i].y*fwd_pts[i].y + 1)/std::sqrt(uv_keystep.x*uv_keystep.x + uv_keystep.y*uv_keystep.y + 1));
            }else
                features_[i].parallax = 0;
        }else
            features_[i].parallax = 0;
    }

    double mean = 0.0;
    int n = 0;
    for( const auto & feature : features_ ){
        if( feature.parallax != 0 && !std::isnan(feature.parallax) ){
            mean += feature.parallax;
            n++;
        }
    }
    mean /= n;
    std::cout << "parallax mean: " << mean << std::endl;

    if( mean < 0.01 ) return false;
    return true;
}

void MVO::deleteDeadFeatures(){
    for( uint32_t i = 0; i < features_.size(); ){
        if( features_[i].is_alive == false ){
            if( features_[i].life > 2 ){
                features_dead_.push_back(features_[i]);
            }
            features_.erase(features_.begin()+i);
        }else{
            i++;
        }
    }
    num_feature_ = features_.size();
}

void MVO::addFeatures(){
    updateBucket();
    if( MVO::s_print_log ) std::cerr << "## Update bucket: " << lsi::toc() << std::endl;

    while( num_feature_ < bucket_.max_features && bucket_.saturated.any() == true )
        addFeature();

    // WARNING: descriptor may remove keypoint whose description cannot be extracted
    // cv::Mat desc;
    // std::vector<cv::KeyPoint> keypoints;
    // for( int i = 0; i < num_feature_; i++ )
    //     if( features_[i].life == 1 )
    //         keypoints.emplace_back(cv::Point2f(features_[i].uv.back().x,features_[i].uv.back().y),1.0);
    // descriptor->compute(curr_image_, keypoints, desc);
    // int j = 0;
    // for( int i = 0; i < num_feature_; i++ )
    //     if( features_[i].life == 1 )
    //         features_[i].desc = desc.row(j++).clone();
}

void MVO::updateBucket(){
    bucket_.mass.fill(0.0);
    bucket_.saturated.fill(1.0);

    uint32_t row_bucket, col_bucket;
    for( int i = 0; i < num_feature_; i++ ){
        row_bucket = std::floor(features_[i].uv.back().y / params_.im_size.height * bucket_.grid.height);
        col_bucket = std::floor(features_[i].uv.back().x / params_.im_size.width * bucket_.grid.width);
        features_[i].bucket = cv::Point(col_bucket, row_bucket);
        bucket_.mass(row_bucket, col_bucket)++;
    }
}

void MVO::addFeature(){
    // Load bucket parameters
    cv::Size bkSize = bucket_.size;
    int bucket_safety = bucket_.safety;

    // Choose ROI based on the probabilistic approaches with the mass of bucket
    int row, col;
    lsi::idx_randselect(bucket_.prob, bucket_.saturated, row, col);
    cv::Rect roi = cv::Rect(col*bkSize.width+1, row*bkSize.height+1, bkSize.width, bkSize.height);

    roi.x = std::max(bucket_safety, roi.x);
    roi.y = std::max(bucket_safety, roi.y);
    roi.width = std::min(params_.im_size.width-bucket_safety, roi.x+roi.width)-roi.x;
    roi.height = std::min(params_.im_size.height-bucket_safety, roi.y+roi.height)-roi.y;

    // Seek index of which feature is extracted specific bucket
    std::vector<uint32_t> idx_belong_to_bucket;
    idx_belong_to_bucket.reserve(num_feature_);

    for( int l = 0; l < num_feature_; l++ ){
        for( int ii = std::max(col-1,0); ii <= std::min(col+1,bucket_.grid.width-1); ii++){
            for( int jj = std::max(row-1,0); jj <= std::min(row+1,bucket_.grid.height-1); jj++){
                if( (features_[l].bucket.x == ii) & (features_[l].bucket.y == jj)){
                    idx_belong_to_bucket.push_back(l);
                }
            }
        }
    }
    uint32_t num_feature_inside_bucket = idx_belong_to_bucket.size();
    
    // Try to find a seperate feature
    std::vector<cv::Point2f> keypoints;
    if( visit_bucket_[row + bucket_.grid.height * col] ){
        keypoints = keypoints_of_bucket_[row + bucket_.grid.height * col];
    }else{
        cv::goodFeaturesToTrack(curr_image_(roi), keypoints, 50, 0.01, params_.min_px_dist, cv::noArray(), 3, true);
        keypoints_of_bucket_[row + bucket_.grid.height * col] = keypoints;
        visit_bucket_[row + bucket_.grid.height * col] = true;
    }
    
    if( keypoints.size() == 0 ){
        bucket_.saturated(row,col) = 0.0;
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
            dist = cv::norm(keypoints[l] - features_[idx_belong_to_bucket[f]].uv.back());
            
            if( dist < min_dist )
                min_dist = dist;
            
            if( dist < params_.min_px_dist ){
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
    
    if( success ){
        // Add new feature to VO object
        Feature newFeature;

        newFeature.id = Feature::new_feature_id; // unique id of the feature
        newFeature.frame_init = 0; // frame step when the 3d point is initialized
        newFeature.parallax = 0; // parallax between associated features
        newFeature.uv.push_back(best_keypoint); // uv point in pixel coordinates
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
        newFeature.depthfilter = std::shared_ptr<DepthFilter>(new DepthFilter());

        features_.push_back(newFeature);
        num_feature_++;
    
        Feature::new_feature_id++;

        // Update bucket
        bucket_.mass(row, col)++;

        cv::eigen2cv(bucket_.mass, bucket_.cv_mass);
        cv::GaussianBlur(bucket_.cv_mass, bucket_.cv_prob, cv::Size(21,21), 3.0);
        cv::cv2eigen(bucket_.cv_prob, bucket_.prob);

        bucket_.prob.array() += 0.05;
        bucket_.prob = bucket_.prob.cwiseInverse();

        // Assign high weight for ground
        for( int i = 0; i < bucket_.prob.rows(); i++ ){
            bucket_.prob.block(i,0,1,bucket_.prob.cols()) *= i+1;
        }
    }else{
        bucket_.saturated(row,col) = 0.0;
    }
}

// haram
bool MVO::calculateEssential()
{
    if (step_ == 0)
        return true;

    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;
    points1.reserve(num_feature_);
    points2.reserve(num_feature_);

    std::vector<uint32_t> idx_static;
    int num_wide_feature = 0;
    int key_idx;
    for( int i = 0; i < num_feature_; i++ ){
        key_idx = features_[i].life - 1 - (step_ - keystep_);
        if( key_idx >= 0 && features_[i].type != Type::Dynamic ){
            points1.push_back(features_[i].uv[key_idx]);
            points2.push_back(features_[i].uv.back());   // latest
            idx_static.push_back(i);
            if( cv::norm(features_[i].uv[key_idx] - features_[i].uv.back()) > params_.th_px_wide ){
                features_[i].is_wide = true;
                num_wide_feature++;
            }else{
                features_[i].is_wide = false;
            }
        }
    }

    if( points1.size() <= num_feature_ * params_.th_ratio_keyframe ){
        keystep_array_.push_back(step_);

        if( is_speed_provided_ ){
            double last_timestamp = timestamp_speed_since_keyframe_.back();
            double last_speed = speed_since_keyframe_.back();

            timestamp_speed_since_keyframe_.clear();
            speed_since_keyframe_.clear();

            timestamp_speed_since_keyframe_.push_back(last_timestamp);
            speed_since_keyframe_.push_back(last_speed);
        }

        if( is_rotate_provided_ ){
            double last_timestamp = timestamp_imu_since_keyframe_.back();
            Eigen::Vector3d last_gyro = gyro_since_keyframe_.back();

            timestamp_imu_since_keyframe_.clear();
            gyro_since_keyframe_.clear();

            timestamp_imu_since_keyframe_.push_back(last_timestamp);
            gyro_since_keyframe_.push_back(last_gyro);
        }
        if( MVO::s_print_log ) std::cerr << "key step: " << keystep_ << ' ' << std::endl;
    }

    if( (int) points1.size() < params_.th_inlier ){
        if( MVO::s_print_log ) std::cerr << "Warning: There are a few stable features" << std::endl;
        return false;
    }

    cv::Mat inlier_mat;
    essential_ = cv::findEssentialMat(points1, points2, params_.Kcv, cv::RANSAC, 0.999, 1.5, inlier_mat);
    if( MVO::s_print_log ) std::cerr << "# Calculate essential: " << lsi::toc() << std::endl;
    
    Eigen::Matrix3d E_;
    cv::cv2eigen(essential_, E_);
    fundamental_ = params_.Kinv.transpose() * E_ * params_.Kinv;

    // double error;
    // std::vector<double> essential_error;
    // for( uint32_t i = 0; i < points1.size(); i++ ){
    //     error = (Eigen::Vector3d() << points2[i].x,points2[i].y,1).finished().transpose() * params_.Kinv.transpose() * E_ * params_.Kinv * (Eigen::Vector3d() << points1[i].x,points1[i].y,1).finished();
    //     essential_error.push_back(error);
    // }

    uint32_t inlier_cnt = 0;
    bool* inlier = inlier_mat.ptr<bool>(0);
    for (int i = 0; i < inlier_mat.rows; i++){
        if (inlier[i]){
            features_[idx_static[i]].is_2D_inliered = true;
            inlier_cnt++;
        // }else if( essential_error[i] > 1e-4 ){
        // }else{
        //     features_[idx_static[i]].type = Type::Dynamic;
        }
    }
    num_feature_2D_inliered_ = inlier_cnt;
    if( MVO::s_print_log ) std::cerr << "num_feature_2D_inliered_: " << (double) num_feature_2D_inliered_ / num_feature_ * 100 << '%' << std::endl;

    Eigen::Matrix3d U,V;
    switch( params_.svd_method){
        case MVO::SVD::JACOBI:{
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(E_, Eigen::ComputeThinU | Eigen::ComputeThinV);
            U = svd.matrixU();
            V = svd.matrixV();
            break;
        }
        case MVO::SVD::OpenCV:{
            cv::Mat Vt, U_, W;
            cv::SVD::compute(essential_, W, U_, Vt);

            Eigen::MatrixXd Vt_;
            cv::cv2eigen(Vt, Vt_);
            V = Vt_.transpose();
            cv::cv2eigen(U_, U);
            break;
        }
        case MVO::SVD::BDC:
        default:{
            Eigen::Matrix3d E_;
            cv::cv2eigen(essential_, E_);
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

    R_vec_.clear();
    t_vec_.clear();
    R_vec_.push_back(U * W * V.transpose());
    R_vec_.push_back(U * W * V.transpose());
    R_vec_.push_back(U * W.transpose() * V.transpose());
    R_vec_.push_back(U * W.transpose() * V.transpose());
    t_vec_.push_back(U.block(0, 2, 3, 1));
    t_vec_.push_back(-U.block(0, 2, 3, 1));
    t_vec_.push_back(U.block(0, 2, 3, 1));
    t_vec_.push_back(-U.block(0, 2, 3, 1));

    if( MVO::s_print_log ) std::cerr << "# Extract R, t: " << lsi::toc() << std::endl;

    if (num_feature_2D_inliered_ < params_.th_inlier){
        if( MVO::s_print_log ) std::cerr << "Warning: There are a few inliers matching features in 2D" << std::endl;
        return false;
    }else{
        is_start_ = true;
        return true;
    }
}

void MVO::addExtraFeatures(){

    if( features_extra_.size() > 0 ){
        for( uint32_t i = 0; i < features_extra_.size(); i++ ){
            features_extra_[i].id = Feature::new_feature_id;
            features_.push_back(features_extra_[i]);
            num_feature_++;
        
            Feature::new_feature_id++;

            // Update bucket
            bucket_.mass(features_extra_[i].bucket.y, features_extra_[i].bucket.x)++;
        }
        cv::eigen2cv(bucket_.mass, bucket_.cv_mass);
        cv::GaussianBlur(bucket_.cv_mass, bucket_.cv_prob, cv::Size(21,21), 3.0);
        cv::cv2eigen(bucket_.cv_prob, bucket_.prob);

        bucket_.prob.array() += 0.05;
        bucket_.prob = bucket_.prob.cwiseInverse();

        // Assign high weight for ground
        for( int i = 0; i < bucket_.prob.rows(); i++ ){
            bucket_.prob.block(i,0,1,bucket_.prob.cols()) *= i+1;
        }

        features_extra_.clear();
    }
}

void MVO::extractRoiFeatures(const std::vector<cv::Rect>& rois, const std::vector<int>& num_feature_){

    cv::Rect roi;
    std::vector<cv::Point2f> keypoints;
    const int& bucket_safety = bucket_.safety;
    for( uint32_t i = 0; i < rois.size(); i++ ){
        roi = rois[i];
        roi.x = std::max(bucket_safety, roi.x);
        roi.y = std::max(bucket_safety, roi.y);
        roi.width = std::min(params_.im_size.width-bucket_safety, roi.x+roi.width)-roi.x;
        roi.height = std::min(params_.im_size.height-bucket_safety, roi.y+roi.height)-roi.y;

        try{
            cv::goodFeaturesToTrack(curr_image_(roi), keypoints, 50, 0.01, params_.min_px_dist, cv::noArray(), 3, true);
        }catch(std::exception& msg){
            if( MVO::s_print_log ) std::cerr << "Warning: " << msg.what() << std::endl;
            continue;
        }
        
        if( keypoints.size() > 0 ){
            for( uint32_t l = 0; l < keypoints.size(); l++ ){
                keypoints[l].x = keypoints[l].x + roi.x - 1;
                keypoints[l].y = keypoints[l].y + roi.y - 1;
            }
        }else{
            if( MVO::s_print_log ) std::cerr << "Warning: There is no keypoints within the bucket" << std::endl;
            try{
                cv::goodFeaturesToTrack(curr_image_(roi), keypoints, 50, 0.1, params_.min_px_dist, cv::noArray(), 3, true);
            }catch(std::exception& msg){
                if( MVO::s_print_log ) std::cerr << "Warning: " << msg.what() << std::endl;
                continue;
            }
        }

        int num_success = 0;
        int num_try = 0;
        while( num_success < num_feature_[i] && num_try++ < 3 )
            if( extractRoiFeature(roi, keypoints) )
                num_success++;
    }
}

bool MVO::extractRoiFeature(const cv::Rect& roi, const std::vector<cv::Point2f>& keypoints){

    int row, col;
    row = (roi.x-1) / bucket_.size.height;
    col = (roi.y-1) / bucket_.size.width;

    // Seek index of which feature is extracted specific bucket
    std::vector<uint32_t> idx_belong_to_bucket;
    idx_belong_to_bucket.reserve(num_feature_);

    for( int l = 0; l < num_feature_; l++ ){
        for( int ii = std::max(col-1,0); ii <= std::min(col+1,bucket_.grid.width-1); ii++){
            for( int jj = std::max(row-1,0); jj <= std::min(row+1,bucket_.grid.height-1); jj++){
                if( (features_[l].bucket.x == ii) & (features_[l].bucket.y == jj)){
                    idx_belong_to_bucket.push_back(l);
                }
            }
        }
    }
    uint32_t num_feature_inside_bucket = idx_belong_to_bucket.size();
    
    // Try to find a seperate feature
    bool success;
    double dist, min_dist, max_min_dist = 0;
    cv::Point2f best_keypoint;
    for( uint32_t l = 0; l < keypoints.size(); l++ ){
        success = true;
        min_dist = 1e9; // enough-large number
        for( uint32_t f = 0; f < num_feature_inside_bucket; f++ ){
            dist = cv::norm(keypoints[l] - features_[idx_belong_to_bucket[f]].uv.back());
            
            if( dist < min_dist )
                min_dist = dist;

            if( dist < params_.min_px_dist/2+1 ){
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
        newFeature.parallax = 0; // parallax between associated features
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
        newFeature.depthfilter = std::shared_ptr<DepthFilter>(new DepthFilter());

        features_extra_.push_back(newFeature);
        return true;
    }else{
        if( MVO::s_print_log ) std::cerr << "Warning: There is no best-match keypoint which is far from the adjacent feature" << std::endl;
        return false;
    }
    
}