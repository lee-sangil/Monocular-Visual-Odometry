#include "core/MVO.hpp"
#include "core/time.hpp"
#include "core/DepthFilter.hpp"
#include "core/numerics.hpp"
#include <eigen3/unsupported/Eigen/MatrixFunctions>
#include <exception>

/**
 * @brief 특징점을 추출하는 프로세스
 * @details 기존 특징점의 위치를 업데이트하고, 추적이 실패한 특징점은 제거하며, 부족분을 새로 추출하여 특징점의 개수를 유지한다.
 * @return 에러가 발생하지 않으면, true
 * @author Sangil Lee (sangillee724@gmail.com)
 * @date 29-Dec-2019
 */
bool MVO::extractFeatures(){
    // Update features using KLT tracker
    if( updateFeatures() ){
        if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "# Update features: " << lsi::toc() << std::endl;
        
        if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "* Tracking rate: " << 100.0 * num_feature_matched_ / num_feature_ << std::endl;

        // Delete features which is failed to track by KLT tracker
        deleteDeadFeatures();
        if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "# Delete features: " << lsi::toc() << std::endl;
        
        // Add features to the number of the lost features
        addFeatures();

        // Add extra feature points
        addExtraFeatures();

        if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "# Add features: " << lsi::toc() << std::endl;
        
        // Print features for debugging
        printFeatures();

        return true;
    }
    else
        return false;
}

/**
 * @brief 특징점의 위치를 갱신
 * @details KLT를 이용하여 특징점을 추적한다.
 * @return 에러가 발생하지 않으면, true
 * @author Sangil Lee (sangillee724@gmail.com)
 * @date 29-Dec-2019
 */
bool MVO::updateFeatures(){

    if( num_feature_ ){
        // Track the points
        std::vector<cv::Point2f> points;
        std::vector<bool> validity;

        // rough klt tracker
        kltTrackerRough(points, validity);
        if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "## KLT tracker: " << lsi::toc() << std::endl;

        selectKeyframeNow(); // update the current keyframe by triggering low parallax
        if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "## Select keyframe: " << lsi::toc() << std::endl;

        // precise klt tracker
        kltTrackerPrecise(points, validity);
        if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "## KLT tracker: " << lsi::toc() << std::endl;

        selectKeyframeAfter(); // update the next keyframe by triggering low tracking ratio
        if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "## Select keyframe: " << lsi::toc() << std::endl;

        if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "! Update features between: " << curr_keyframe_.id << " <--> " << curr_frame_.id << std::endl;

        // if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "key_time: " << std::setprecision(19) << curr_keyframe_.timestamp << ", curr_time: " << curr_frame_.timestamp << std::endl;
        // if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "logging time:"; 
        // for( int i = 0; i < curr_keyframe_.linear_velocity_since_.size(); i++ )
        //     if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << " " << std::setprecision(19) << curr_keyframe_.linear_velocity_since_[i].first;
        // if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << std::endl;

        // calculate and update rotate information from gyro logging data
        updateRotatePrior();

        // update 2d attributes of the matched features
        int key_idx;
        cv::Point2f uv_prev;
        Eigen::Matrix<double,3,4> Tko = TocRec_[keystep_].inverse().block(0,0,3,4);
        for( int i = 0; i < num_feature_; i++ ){
            if( validity[i] && features_[i].is_alive ){
                key_idx = features_[i].life - (step_ - keystep_); // before feature.life increasement

                // memorize the previous uv point
                if( key_idx >= 0 )
                    uv_prev = features_[i].uv[key_idx];
                else
                    uv_prev = features_[i].uv.back();
                
                // Flag KLT matching
                if( features_[i].frame_2d_init < 0 ) features_[i].frame_2d_init = keystep_;
                features_[i].is_matched = true;
                
                // Update features properties: life, uv
                while( features_[i].frame_2d_init + features_[i].life - 1 - step_ < -1 ){
                    features_[i].life++;
                    features_[i].uv.push_back(cv::Point2f(-1,-1)); // In the case of newly-registered-features in the keyframe
                }
                if( features_[i].frame_2d_init + features_[i].life - 1 - step_ == -1 ){
                    features_[i].life++;
                    features_[i].uv.push_back(points[i]);
                }

                // // Predict the current uv point from the previous uv point
                // if( is_rotate_provided_ ) {
                //     if( is_speed_provided_ && features_[i].is_3D_init )
                //         features_[i].uv_pred = warpWithCAN(Tko * features_[i].point_init);
                //     else
                //         features_[i].uv_pred = warpWithIMU(uv_prev);
                // }else if( features_[i].is_3D_init ) features_[i].uv_pred = warpWithPreviousMotion(Tko * features_[i].point_init);
                
                // // Reject unpredicted motion using the previous uv or point and prior knowledge
                // if( is_rotate_provided_  && !is_speed_provided_){
                //     Eigen::Vector3d epiLine = fundamental * (Eigen::Vector3d() << uv_prev.x, uv_prev.y, 1).finished();
                //     double dist_from_epiline = std::abs(epiLine(0)*features_[i].uv_pred.x + epiLine(1)*features_[i].uv_pred.y + epiLine(2)) / epiLine.topRows(2).norm();
                //     if( is_start_){
                //         if( dist_from_epiline > params_.max_dist ){
                //             features_[i].type = Type::Dynamic;
                //             features_[i].is_3D_init = false;
                //         }else{
                //             if( features_[i].type == Type::Dynamic ) features_[i].type = Type::Unknown;
                //         }
                //     }
                // }else if( features_[i].is_3D_init ){
                //     double dist_from_predicted_point = cv::norm(features_[i].uv_pred - features_[i].uv.back());
                //     if( is_start_ ){
                //         if( dist_from_predicted_point > params_.max_dist ){
                //             features_[i].type = Type::Dynamic;
                //             features_[i].is_3D_init = false;
                //         }else{
                //             if( features_[i].type == Type::Dynamic ) features_[i].type = Type::Unknown;
                //         }
                //     }
                // }

            }else
                features_[i].is_alive = false;
        }

        if( num_feature_matched_ < params_.th_inlier ){
            if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "Warning: There are a few feature matches" << std::endl;
            return false;
        }else{
            return true;
        }
    }else{
        curr_keyframe_.assign(curr_frame_); // for logging velocity and gyro
        next_keyframe_.assign(curr_frame_); // for extracting new features

        return true;
    }
}

/**
 * @brief 키프레임 삭제
 * @details 현재 이미지 프레임과 키프레임 사이의 시차가 작은 경우, 현재 키프레임을 삭제하고 이전 키프레임을 현재 키프레임으로 변경한다.
 * @return 없음
 * @author Sangil Lee (sangillee724@gmail.com)
 * @date 29-Dec-2019
 */
void MVO::selectKeyframeNow(){

    // Low parallax
    std::vector<double> parallax;
    parallax.reserve(num_feature_);
    for( const auto & feature : features_ )
        if( feature.parallax >= 0 && !std::isnan(feature.parallax) )
            parallax.push_back(feature.parallax);
    
    // grab the specific percentile of parallaxes
    if( parallax.size() > 0 ){
        std::sort(parallax.begin(), parallax.end());
        double parallax_percentile = (parallax[std::floor(parallax.size()*params_.percentile_parallax)]+parallax[std::ceil(parallax.size()*params_.percentile_parallax)])/2;
        // std::cout << "parallax_percentile:" << parallax_percentile << std::endl;

        // if low parallax appears
        if( parallax_percentile < params_.th_parallax ){
            if( keystep_array_.size() > 0 && !prev_keyframe_.image.empty() ){

                // // Delete features which is created at keyframe and not be used in the next iteration
                // for( uint32_t i = 0; i < num_feature_; i++ )
                //     if( features_[i].life == 1 )
                //         features_[i].is_alive = false;

                // remove the current keyframe
                keystep_array_.pop_back();

                // Extract features in the current iteration, prev_keyframe is now the current keyframe
                next_keyframe_.assign(prev_keyframe_);

                // Select keyframe in the current iteration
                keystep_ = keystep_array_.back();
                curr_keyframe_.merge(prev_keyframe_);

                trigger_keystep_decrease_ = true;
                
                if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "! <keystep> low parallax: " << parallax_percentile << std::endl;
            }
        }
    }

    // if( MVO::s_file_logger_.is_open() ){
    //     MVO::s_file_logger_ << "* parallax: ";
    //     for( const auto & feature : features_ )
    //         MVO::s_file_logger_ << feature.parallax << ' ';
    //     MVO::s_file_logger_ << std::endl;
    // }
}

/**
 * @brief 키프레임 추가
 * @details 현재 이미지 프레임과 키프레임 사이의 특징점 매칭률이 낮으면, 현재 이미지 프레임을 다음 이터레이션에서의 키프레임으로 추가한다.
 * @return 없음
 * @author Sangil Lee (sangillee724@gmail.com)
 * @date 29-Dec-2019
 */
void MVO::selectKeyframeAfter(){
    // Low tracking ratio
    if( !trigger_keystep_decrease_ && !trigger_keystep_decrease_previous_ ){ // only if there is no trigger about low parallax
        if( num_feature_matched_ <= num_feature_ * params_.th_ratio_keyframe ){

            // add new keystep and keyframe
            keystep_array_.push_back(step_);

            // update the previous keyframe
            prev_keyframe_.assign(curr_keyframe_);

            // Extract features in the current iteration, but curr_frame will be chosen as the current keyframe in the next iteration
            next_keyframe_.assign(curr_frame_);

            trigger_keystep_increase_ = true;
            
            if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "! <keystep> tracking loss: " << (double) num_feature_matched_ / num_feature_ << std::endl;
        }
    }
}

/**
 * @brief 특징점 위치 예상
 * @details IMU의 값과 직전 특징점의 위치를 이용하여 현재 이미지 프레임에서 나타날 특징점의 위치를 예상한다.
 * @param uv 직전 특징점의 uv 위치
 * @return 특징점의 예상 uv 위치
 * @author Sangil Lee (sangillee724@gmail.com)
 * @date 29-Dec-2019
 */
cv::Point2f MVO::warpWithIMU(const cv::Point2f& uv) const {
    Eigen::Vector3d pixel, warpedPixel;
    Eigen::Vector3d t = (TocRec_[keystep_].inverse() * TocRec_.back() * TRec_.back()).block(0,3,3,1);
    pixel << uv.x, uv.y, 1;
    warpedPixel = params_.K * rotate_prior_ * (params_.Kinv * pixel - t);
    return cv::Point2f(warpedPixel(0)/warpedPixel(2), warpedPixel(1)/warpedPixel(2));
}

/**
 * @brief 특징점 위치 예상
 * @details CAN 차속의 값과 직전 특징점의 위치를 이용하여 현재 이미지 프레임에서 나타날 특징점의 위치를 예상한다.
 * @param uv 직전 특징점의 uv 위치
 * @return 특징점의 예상 uv 위치
 * @author Sangil Lee (sangillee724@gmail.com)
 * @date 29-Dec-2019
 */
cv::Point2f MVO::warpWithCAN(const Eigen::Vector3d& p) const {
    Eigen::Vector3d pixel, warpedPixel;
    Eigen::Vector3d t = (TocRec_[keystep_].inverse() * TocRec_.back() * TRec_.back()).block(0,3,3,1);
    warpedPixel = params_.K * rotate_prior_ * (p-t);
    return cv::Point2f(warpedPixel(0)/warpedPixel(2), warpedPixel(1)/warpedPixel(2));
}

/**
 * @brief 특징점 위치 예상
 * @details 최근 변환 행렬과 직전 특징점의 위치를 이용하여 현재 이미지 프레임에서 나타날 특징점의 위치를 예상한다.
 * @param uv 직전 특징점의 uv 위치
 * @return 특징점의 예상 uv 위치
 * @author Sangil Lee (sangillee724@gmail.com)
 * @date 29-Dec-2019
 */
cv::Point2f MVO::warpWithPreviousMotion(const Eigen::Vector3d& p) const {
    Eigen::Vector3d warpedPixel;
    Eigen::Matrix3d Rinv = (TocRec_[keystep_].inverse() * TocRec_.back() * TRec_.back()).block(0,0,3,3).transpose();
    Eigen::Vector3d t = (TocRec_[keystep_].inverse() * TocRec_.back() * TRec_.back()).block(0,3,3,1);
    
    warpedPixel = params_.K * Rinv * (p-t);
    return cv::Point2f(warpedPixel(0)/warpedPixel(2), warpedPixel(1)/warpedPixel(2));
}

/**
 * @brief 특징점 추적
 * @details 간단한 방법으로 특징점의 위치를 빠르게 추적한다.
 * @param points 추적된 uv 벡터
 * @param validity 추적 성공 여부
 * @return 없음
 * @author Sangil Lee (sangillee724@gmail.com)
 * @date 29-Dec-2019
 */
void MVO::kltTrackerRough(std::vector<cv::Point2f>& points, std::vector<bool>& validity){
    points.clear();
    points.reserve(num_feature_);

    validity.clear();
    validity.reserve(num_feature_);

    std::vector<cv::Point2f> pts, fwd_pts;
    std::vector<uint32_t> idx_track;

    pts.reserve(num_feature_);
    fwd_pts.reserve(num_feature_);
    idx_track.reserve(num_feature_);

    int key_idx;
    for( uint32_t i = 0; i < num_feature_; i++ ){
        key_idx = features_[i].life - (step_ - keystep_); // before feature.life increasement
        if( key_idx >= 0 ){
            idx_track.push_back(i);
            pts.push_back(features_[i].uv[key_idx]);
        }else if( features_[i].frame_2d_init == -1 ){
            idx_track.push_back(i);
            pts.push_back(features_[i].uv.back());
        }
    }
    
    cv::Mat status, err;
    cv::calcOpticalFlowPyrLK(curr_keyframe_.image, curr_frame_.image, pts, fwd_pts, status, err, cv::Size(5,5), 3);
    if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "### Calculate optical flows roughly: " << lsi::toc() << std::endl;

    // Validate parallax and matched number
    std::vector<bool> border_valid(pts.size(), false);
    for( uint32_t i = 0; i < pts.size(); i++ )
        border_valid[i] = (fwd_pts[i].x > 0) && (fwd_pts[i].x < params_.im_size.width) && (fwd_pts[i].y > 0) && (fwd_pts[i].y < params_.im_size.height);

    // check validity
    cv::Point2f uv_keystep;
    uint32_t j = 0;
    for( uint32_t i = 0; i < num_feature_; i++ ){
        if( i == idx_track[j] && j < pts.size() ){
            if( *(status.data+j) && border_valid[j] && features_[i].is_alive ){
                points.push_back(fwd_pts[j]);
                validity.push_back(true);
                
                key_idx = features_[i].life - (step_ - keystep_); // before feature.life increasement
                uv_keystep = features_[i].uv[key_idx];
                features_[i].parallax = std::acos((fwd_pts[j].dot(uv_keystep)+1)/std::sqrt(fwd_pts[j].x*fwd_pts[j].x + fwd_pts[j].y*fwd_pts[j].y + 1)/std::sqrt(uv_keystep.x*uv_keystep.x + uv_keystep.y*uv_keystep.y + 1));
            }else{
                points.push_back(cv::Point2f(-1,-1));
                validity.push_back(false);
                features_[i].parallax = -1;
            }

            j++;
        }else{
            points.push_back(cv::Point2f(-1,-1));
            validity.push_back(false);
            features_[i].parallax = -1;
        }
    }
}

/**
 * @brief 특징점 추적
 * @details 정교한 방법으로 특징점의 위치를 정밀하게 추적한다.
 * @param points 추적된 uv 벡터
 * @param validity 추적 성공 여부
 * @return 없음
 * @author Sangil Lee (sangillee724@gmail.com)
 * @date 29-Dec-2019
 */
void MVO::kltTrackerPrecise(std::vector<cv::Point2f>& points, std::vector<bool>& validity){
    points.clear();
    points.reserve(num_feature_);

    validity.clear();
    validity.reserve(num_feature_);

    std::vector<cv::Point2f> pts, fwd_pts, bwd_pts, fwd_bwd_pts;
    std::vector<uint32_t> idx_track;

    pts.reserve(num_feature_);
    fwd_pts.reserve(num_feature_);
    bwd_pts.reserve(num_feature_);
    idx_track.reserve(num_feature_);

    int key_idx;
    for( uint32_t i = 0; i < num_feature_; i++ ){
        key_idx = features_[i].life - (step_ - keystep_); // before feature.life increasement
        if( key_idx >= 0 ){
            idx_track.push_back(i);
            pts.push_back(features_[i].uv[key_idx]);
        }else if( features_[i].frame_2d_init == -1 ){
            idx_track.push_back(i);
            pts.push_back(features_[i].uv.back());
        }
    }

    // Forward-backward error evaluation
    // std::vector<cv::Mat> prevPyr;// = prev_pyramid_template_;
    // std::vector<cv::Mat> currPyr;// = curr_pyramid_template_;
    // if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "### Prepare variables: " << lsi::toc() << std::endl;

    // cv::buildOpticalFlowPyramid(curr_keyframe_.image, prevPyr, cv::Size(15,15), 3, true);
    // cv::buildOpticalFlowPyramid(curr_frame_.image, currPyr, cv::Size(15,15), 3, true);
    // if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "### Build pyramids: " << lsi::toc() << std::endl;

    cv::Mat status, err;
    cv::calcOpticalFlowPyrLK(curr_keyframe_.image, curr_frame_.image, pts, fwd_pts, status, err, cv::Size(9,9), 4);
    cv::calcOpticalFlowPyrLK(curr_frame_.image, curr_keyframe_.image, fwd_pts, bwd_pts, status, err, cv::Size(9,9), 4);
    // cv::calcOpticalFlowPyrLK(prevPyr, currPyr, bwd_pts, fwd_bwd_pts, status, err);
    if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "### Calculate optical flows: " << lsi::toc() << std::endl;

    std::vector<bool> border_valid, error_valid;
    border_valid.reserve(pts.size());
    error_valid.reserve(pts.size());
    for( uint32_t i = 0; i < pts.size(); i++ ){
        border_valid.emplace_back((fwd_pts[i].x > 0) && (fwd_pts[i].x < params_.im_size.width) && (fwd_pts[i].y > 0) && (fwd_pts[i].y < params_.im_size.height));
        error_valid.emplace_back(cv::norm(pts[i] - bwd_pts[i]) < std::min( (double) cv::norm(pts[i] - fwd_pts[i])/5.0, 1.0));
    }

    // Validate parallax and matched number
    cv::Point2f uv_keystep;
    uint32_t j = 0;
    for( uint32_t i = 0; i < num_feature_; i++ ){
        if( i == idx_track[j] && j < pts.size() ){
            if( border_valid[j] && error_valid[j] && features_[i].is_alive){
                points.push_back(fwd_pts[j]);
                validity.push_back(true);
                num_feature_matched_++;                           // increase the number of feature matched
                
                key_idx = features_[i].life - (step_ - keystep_); // before feature.life increasement
                uv_keystep = features_[i].uv[key_idx];
                features_[i].parallax = std::acos((fwd_pts[j].dot(uv_keystep)+1)/std::sqrt(fwd_pts[j].x*fwd_pts[j].x + fwd_pts[j].y*fwd_pts[j].y + 1)/std::sqrt(uv_keystep.x*uv_keystep.x + uv_keystep.y*uv_keystep.y + 1));
            }else{
                points.push_back(cv::Point2f(-1,-1));
                validity.push_back(false);
                features_[i].parallax = -1;
            }

            j++;
        }else{
            points.push_back(cv::Point2f(-1,-1));
            validity.push_back(false);
            features_[i].parallax = -1;
        }
    }
}

/**
 * @brief 특징점 삭제
 * @details 더 이상 추적하지 않을 특징점을 삭제한다.
 * @return 없음
 * @author Sangil Lee (sangillee724@gmail.com)
 * @date 29-Dec-2019
 */
void MVO::deleteDeadFeatures(){
    for( uint32_t i = 0; i < features_.size(); ){
        if( features_[i].is_alive == false ){
            if( features_[i].landmark && MVO::s_file_logger_.is_open() ){ // add features_dead for plotting if debugging mode
                features_dead_.push_back(features_[i]);
            }
            features_.erase(features_.begin()+i);
        }else{
            i++;
        }
    }
    // features_.erase(std::remove_if(features_.begin(), features_.end(), [](const auto & f){return f.life > 2 && MVO::s_file_logger_.is_open();}),features_.end());
    num_feature_ = features_.size();
}

/**
 * @brief 특징점 추가
 * @details 특정 갯수가 될 때까지, 또는 유의미한 특징점을 더 추출할 수 없을 때까지 특징점을 추가한다.
 * @return 없음
 * @author Sangil Lee (sangillee724@gmail.com)
 * @date 29-Dec-2019
 */
void MVO::addFeatures(){
    updateBucket();
    if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "## Update bucket: " << lsi::toc() << std::endl;

    // add feature until the number of feature reaches the desired value or all buckets are fail to extract separable feature
    while( num_feature_ < bucket_.max_features && bucket_.saturated.any() == true )
        addFeature();

    if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "! Add features in: " << next_keyframe_.id << std::endl;
}

/**
 * @brief bucket 업데이트
 * @details 특징점의 업데이트된 위치를 바탕으로, bucket의 변수들을 업데이트한다.
 * @return 없음
 * @author Sangil Lee (sangillee724@gmail.com)
 * @date 29-Dec-2019
 */
void MVO::updateBucket(){
    bucket_.mass.fill(0.0);
    bucket_.saturated.fill(1.0);

    uint32_t row_bucket, col_bucket;
    for( int i = 0; i < num_feature_; i++ ){
        row_bucket = std::floor(features_[i].uv.back().y / params_.im_size.height * bucket_.grid.height);
        col_bucket = std::floor(features_[i].uv.back().x / params_.im_size.width * bucket_.grid.width);
        features_[i].bucket = cv::Point(col_bucket, row_bucket);

        if( col_bucket < 0 || col_bucket >= bucket_.grid.width || row_bucket < 0 || row_bucket >= bucket_.grid.height ){
            printFeatures();
            std::cout << "Wrong bucket index" << std::endl;
        }

        bucket_.mass(row_bucket, col_bucket)++;
    }
}

/**
 * @brief 특징점 추가
 * @details 특징점 사이의 최소 거리를 유지하도록 새로운 특징점을 추출 및 추가한다.
 * @return 없음
 * @author Sangil Lee (sangillee724@gmail.com)
 * @date 29-Dec-2019
 */
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
                if( (features_[l].bucket.x == ii) && (features_[l].bucket.y == jj)){
                    idx_belong_to_bucket.push_back(l);
                }
            }
        }
    }
    uint32_t num_feature_inside_bucket = idx_belong_to_bucket.size();
    
    // Try to find a seperate feature
    std::vector<cv::Point2f> keypoints;
    if( visit_bucket_[row + bucket_.grid.height * col] ){
        // if the bucket is visited before, use the previous keypoints extracted at the first time
        keypoints = keypoints_of_bucket_[row + bucket_.grid.height * col];
    }else{
        // if the bucket is visited for the first time, extract keypoint from the cropped image
        cv::goodFeaturesToTrack(next_keyframe_.image(roi), keypoints, 50, 0.01, params_.min_px_dist, excludeMask_(roi), 3, true);
        keypoints_of_bucket_[row + bucket_.grid.height * col] = keypoints;
        visit_bucket_[row + bucket_.grid.height * col] = true;
    }
    
    if( keypoints.size() == 0 ){
        bucket_.saturated(row,col) = 0.0;
        return;
    }else{
        // compensate the location of the extracted keypoints
        for( uint32_t l = 0; l < keypoints.size(); l++ ){
            keypoints[l].x = keypoints[l].x + roi.x - 1;
            keypoints[l].y = keypoints[l].y + roi.y - 1;
        }
    }

    int key_idx;
    int prev_keystep = keystep_array_.back(); // Used when keystep is decreased only

    bool success;
    double dist, min_dist, max_min_dist = 0;
    cv::Point2f best_keypoint;
    for( uint32_t l = 0; l < keypoints.size(); l++ ){
        success = true;
        min_dist = 1e9; // enough-large number
        for( uint32_t f = 0; f < num_feature_inside_bucket; f++ ){
            dist = 1e9;
            if( trigger_keystep_decrease_ ){ // the previous keyframe will be the keyframe in the next iteration
                key_idx = features_[idx_belong_to_bucket[f]].life - 1 - (step_ - prev_keystep); // after feature.life increasement
                if( key_idx >= 0 )
                    dist = cv::norm(keypoints[l] - features_[idx_belong_to_bucket[f]].uv[key_idx]);
                else if( features_[idx_belong_to_bucket[f]].frame_2d_init < 0 )
                    dist = cv::norm(keypoints[l] - features_[idx_belong_to_bucket[f]].uv.back());
            
            }else if( trigger_keystep_increase_ ){ // the current frame will be the keyframe in the next iteration
                dist = cv::norm(keypoints[l] - features_[idx_belong_to_bucket[f]].uv.back());
            
            }else{ // the current keyframe will be the keyframe in the next iteration
                key_idx = features_[idx_belong_to_bucket[f]].life - 1 - (step_ - keystep_); // after feature.life increasement
                if( key_idx >= 0 )
                    dist = cv::norm(keypoints[l] - features_[idx_belong_to_bucket[f]].uv[key_idx]);
                else if( features_[idx_belong_to_bucket[f]].frame_2d_init < 0 )
                    dist = cv::norm(keypoints[l] - features_[idx_belong_to_bucket[f]].uv.back());
            }
            
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

        newFeature.id = Feature::getNewID(); // unique id of the feature
        newFeature.frame_2d_init = -1; // frame step when the 2d point is tracked
        newFeature.frame_3d_init = -1; // frame step when the 3d point is initialized
        newFeature.parallax = 0; // parallax between associated features
        newFeature.uv.push_back(best_keypoint); // uv point in pixel coordinates
        newFeature.uv_pred = cv::Point2f(-1,-1); // uv point predicted before
        newFeature.life = 1; // the number of frames in where the feature is observed
        newFeature.bucket = cv::Point(col, row); // the location of bucket where the feature belong to
        newFeature.point_curr << 0,0,0,1; // 3-dim homogeneous point in the local coordinates
        newFeature.is_alive = true; // if false, the feature is deleted
        newFeature.is_matched = false; // matched between both frame
        newFeature.is_wide = false; // verify whether features btw the initial and current are wide enough
        newFeature.is_2D_inliered = false; // belong to major (or meaningful) movement
        newFeature.is_3D_reconstructed = false; // triangulation completion
        newFeature.landmark = NULL; // landmark point in world coordinates
        newFeature.type = Type::Unknown; // type of feature
        newFeature.depthfilter = std::make_shared<DepthFilter>(); // depth filter

        features_.push_back(newFeature);
        num_feature_++;

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

/**
 * @brief 프레임 사이의 움직임을 계산하는 프로세스
 * @details 특징점 쌍의 essential matrix를 계산하고, R, t를 추출한다.
 * @return 에러가 발생하지 않으면, true
 * @author Sangil Lee (sangillee724@gmail.com) Haram Kim (rlgkfka614@gmail.com)
 * @date 29-Dec-2019
 */
bool MVO::calculateEssential(Eigen::Matrix3d & R, Eigen::Vector3d & t)
{
    if (step_ == 0){
        return true;
    }

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
            points1.push_back(features_[i].uv[key_idx]); // uv of keyframe
            points2.push_back(features_[i].uv.back());   // latest uv
            idx_static.push_back(i);
            if( cv::norm(features_[i].uv[key_idx] - features_[i].uv.back()) > params_.th_px_wide ){
                features_[i].is_wide = true;
                num_wide_feature++;
            }else{
                features_[i].is_wide = false;
            }
        }
    }

    if( (int) points1.size() < params_.th_inlier ){
        if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "Warning: There are a few stable features" << std::endl;
        return false;
    }

    // calculate essential matrix with ransac
    cv::Mat inlier_mat, essential;
    essential = cv::findEssentialMat(points1, points2, params_.Kcv, CV_RANSAC, 0.999, 1.5, inlier_mat);
    if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "# Calculate essential: " << lsi::toc() << std::endl;
    
    Eigen::Matrix3d E_, fundamental;
    cv::cv2eigen(essential, E_);
    fundamental = params_.Kinv.transpose() * E_ * params_.Kinv;

    // double error;
    // std::vector<double> essential_error;
    // for( uint32_t i = 0; i < points1.size(); i++ ){
    //     error = (Eigen::Vector3d() << points2[i].x,points2[i].y,1).finished().transpose() * params_.Kinv.transpose() * E_ * params_.Kinv * (Eigen::Vector3d() << points1[i].x,points1[i].y,1).finished();
    //     essential_error.push_back(error);
    // }

    // check inliers
    uint32_t inlier_cnt = 0;
    bool* inlier = inlier_mat.ptr<bool>(0);
    std::vector<uint32_t> idx_2d_inlier;
    for (int i = 0; i < inlier_mat.rows; i++){
        if (inlier[i]){
            features_[idx_static[i]].is_2D_inliered = true;

            idx_2d_inlier.push_back(idx_static[i]);
            inlier_cnt++;
        // }else if( essential_error[i] > 1e-4 ){
        // }else{
        //     features_[idx_static[i]].type = Type::Dynamic;
        }
    }
    num_feature_2D_inliered_ = inlier_cnt;
    if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "num_feature_2D_inliered_: " << (double) num_feature_2D_inliered_ / num_feature_ * 100 << '%' << std::endl;

    // extract R, t
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
            cv::SVD::compute(essential, W, U_, Vt);

            Eigen::MatrixXd Vt_;
            cv::cv2eigen(Vt, Vt_);
            V = Vt_.transpose();
            cv::cv2eigen(U_, U);
            break;
        }
        case MVO::SVD::BDC:
        default:{
            Eigen::Matrix3d E_;
            cv::cv2eigen(essential, E_);
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

    std::vector<Eigen::Matrix3d> R_vec;
    std::vector<Eigen::Vector3d> t_vec;
    R_vec.clear();
    t_vec.clear();
    R_vec.push_back(U * W * V.transpose());
    R_vec.push_back(U * W * V.transpose());
    R_vec.push_back(U * W.transpose() * V.transpose());
    R_vec.push_back(U * W.transpose() * V.transpose());
    t_vec.push_back(U.block(0, 2, 3, 1));
    t_vec.push_back(-U.block(0, 2, 3, 1));
    t_vec.push_back(U.block(0, 2, 3, 1));
    t_vec.push_back(-U.block(0, 2, 3, 1));

    /**************************************************
     * Solve two-fold ambiguity
     **************************************************/
    std::vector<bool> max_inlier;
    std::vector<Eigen::Vector3d> X_curr;
    if( !verifySolutions(R_vec, t_vec, R, t, max_inlier, X_curr) ) return false;
    if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "# Verify unique pose: " << lsi::toc() << std::endl;

    for( int i = 0; i < max_inlier.size(); i++ ){
        if( max_inlier[i] ){
            features_[idx_2d_inlier[i]].point_curr = (Eigen::Vector4d() << X_curr[i], 1).finished();
            features_[idx_2d_inlier[i]].is_3D_reconstructed = true;
            num_feature_3D_reconstructed_++;
        }
    }

    if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "# Extract R, t: " << lsi::toc() << std::endl;
    if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "! Extract essential between: " << curr_keyframe_.id << " <--> " << curr_frame_.id << std::endl;

    if (num_feature_2D_inliered_ < params_.th_inlier){
        if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "Warning: There are a few inliers matching features in 2D" << std::endl;
        return false;
    }else{
        is_start_ = true;
        return true;
    }
}

/**
 * @brief 프레임 사이의 움직임을 계산하는 프로세스
 * @details 3D 특징점 쌍 사이의 R, t를 추출한다.
 * @return 에러가 발생하지 않으면, true
 * @author Sangil Lee (sangillee724@gmail.com)
 * @date 14-Apr-2020
 */
bool MVO::calculateEssentialStereo(Eigen::Matrix3d & R, Eigen::Vector3d & t)
{
    if (step_ == 0){
        return true;
    }

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
            points1.push_back(features_[i].uv[key_idx]); // uv of keyframe
            points2.push_back(features_[i].uv.back());   // latest uv
            idx_static.push_back(i);
            if( cv::norm(features_[i].uv[key_idx] - features_[i].uv.back()) > params_.th_px_wide ){
                features_[i].is_wide = true;
                num_wide_feature++;
            }else{
                features_[i].is_wide = false;
            }
        }
    }

    if( (int) points1.size() < params_.th_inlier ){
        if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "Warning: There are a few stable features" << std::endl;
        return false;
    }

    // calculate disparity
    std::vector<cv::Point2f> fwd_pts, bwd_pts, fwd_bwd_pts;

    fwd_pts.reserve(num_feature_);
    bwd_pts.reserve(num_feature_);

    cv::Mat status, err;
    cv::calcOpticalFlowPyrLK(curr_frame_.image, curr_frame_right_.image, points2, fwd_pts, status, err, cv::Size(9,9), 4);
    cv::calcOpticalFlowPyrLK(curr_frame_right_.image, curr_frame_.image, fwd_pts, bwd_pts, status, err, cv::Size(9,9), 4);

    bool border_valid, error_valid, stereo_valid;
    std::vector<bool> validity;
    validity.reserve(num_feature_);
    for( uint32_t i = 0; i < num_feature_; i++ ){
        border_valid = (fwd_pts[i].x > 0) && (fwd_pts[i].x < params_.im_size.width) && (fwd_pts[i].y > 0) && (fwd_pts[i].y < params_.im_size.height);
        error_valid = cv::norm(points2[i] - bwd_pts[i]) < std::min( (double) cv::norm(points2[i] - fwd_pts[i])/5.0, 1.0);
        stereo_valid = (fwd_pts[i].x < points2[i].x) && (std::abs(fwd_pts[i].y - points2[i].y) < 1);
        validity.emplace_back(border_valid && error_valid && stereo_valid);
    }

    // calculate essential matrix with ransac to reject outliers
    cv::Mat inlier_mat, essential;
    essential = cv::findEssentialMat(points1, points2, params_.Kcv, CV_RANSAC, 0.999, 1.5, inlier_mat);
    if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "# Calculate essential: " << lsi::toc() << std::endl;
    
    Eigen::Matrix3d E_, fundamental;
    cv::cv2eigen(essential, E_);
    fundamental = params_.Kinv.transpose() * E_ * params_.Kinv;

    // check inliers
    int len;
    std::vector<cv::Point2f> uv_prev, uv_curr;
    uint32_t inlier_cnt = 0;
    bool* inlier = inlier_mat.ptr<bool>(0);
    std::vector<uint32_t> idx_2d_inlier;
    for (int i = 0; i < inlier_mat.rows; i++){
        if (inlier[i]){
            len = features_[idx_static[i]].life;
            features_[idx_static[i]].is_2D_inliered = true;
            uv_prev.emplace_back(features_[idx_static[i]].uv[len-2]);
            uv_curr.emplace_back(features_[idx_static[i]].uv.back());

            idx_2d_inlier.push_back(idx_static[i]);
            inlier_cnt++;
        }
    }
    num_feature_2D_inliered_ = inlier_cnt;
    if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "num_feature_2D_inliered_: " << (double) num_feature_2D_inliered_ / num_feature_ * 100 << '%' << std::endl;

    // extract R, t
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
            cv::SVD::compute(essential, W, U_, Vt);

            Eigen::MatrixXd Vt_;
            cv::cv2eigen(Vt, Vt_);
            V = Vt_.transpose();
            cv::cv2eigen(U_, U);
            break;
        }
        case MVO::SVD::BDC:
        default:{
            Eigen::Matrix3d E_;
            cv::cv2eigen(essential, E_);
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

    std::vector<Eigen::Matrix3d> R_vec;
    std::vector<Eigen::Vector3d> t_vec;
    R_vec.clear();
    t_vec.clear();
    R_vec.push_back(U * W * V.transpose());
    R_vec.push_back(U * W * V.transpose());
    R_vec.push_back(U * W.transpose() * V.transpose());
    R_vec.push_back(U * W.transpose() * V.transpose());
    t_vec.push_back(U.block(0, 2, 3, 1));
    t_vec.push_back(-U.block(0, 2, 3, 1));
    t_vec.push_back(U.block(0, 2, 3, 1));
    t_vec.push_back(-U.block(0, 2, 3, 1));

    std::vector<bool> depth_inlier;
    std::vector<Eigen::Vector3d> X_curr;
    if( !verifySolutions(R_vec, t_vec, R, t, depth_inlier, X_curr) ) return false;
    if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "# Verify unique pose: " << lsi::toc() << std::endl;

    /**************************************************
     * Find proper scale from disparity map
     **************************************************/
    double scale = 0;
    std::vector<std::pair<cv::Point3f,cv::Point3f>> Points;
    float x, y, z;
    for( int i = 0; i < depth_inlier.size(); i++ ){
        if( depth_inlier[i] && validity[idx_2d_inlier[i]] ){
            // Get 3D point with disparity
            z = 0.537*params_.fx/(points2[idx_2d_inlier[i]].x - fwd_pts[idx_2d_inlier[i]].x);
            x = (uv_curr[i].x - params_.cx) / params_.fx * z;
            y = (uv_curr[i].y - params_.cy) / params_.fx * z;
            
            // Set expected 3D point as a result of triangulation
            Points.emplace_back(cv::Point3f(X_curr[i](0),X_curr[i](1),X_curr[i](2)),cv::Point3f(x,y,z));
            features_[idx_2d_inlier[i]].point_curr = (Eigen::Vector4d() << x,y,z,1).finished();
            features_[idx_2d_inlier[i]].is_3D_reconstructed = true;
            num_feature_3D_reconstructed_++;
        }
    }

    params_.ransac_coef_scale.th_dist = 1;
    params_.ransac_coef_scale.calculate_func = std::bind(lsi::calculateScale, std::placeholders::_1, std::placeholders::_2, scale_reference_, params_.weight_scale_ref);

    std::vector<bool> scale_inlier, scale_outlier;
    lsi::ransac<std::pair<cv::Point3f,cv::Point3f>,double>(Points, params_.ransac_coef_scale, scale, scale_inlier, scale_outlier);

    t *= scale;

    if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "# Extract R, t: " << lsi::toc() << std::endl;
    if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "! Extract essential between: " << curr_keyframe_.id << " <--> " << curr_frame_.id << std::endl;

    if (num_feature_2D_inliered_ < params_.th_inlier){
        if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "Warning: There are a few inliers matching features in 2D" << std::endl;
        return false;
    }else{
        is_start_ = true;
        return true;
    }
}

/**
 * @brief 4개의 R, t에서 올바른 값을 선택
 * @details 4개의 R, t 후보군에서 양의 깊이값들을 가지게 하는 R, t를 선택한다.
 * @param R_vec 회전 행렬 후보
 * @param t_vec 변위 벡터 후보
 * @param R 올바른 회전 행렬
 * @param t 올바른 변위 벡터
 * @return 에러가 발생하지 않으면, true
 * @author Sangil Lee (sangillee724@gmail.com)
 * @date 29-Dec-2019
 */
bool MVO::verifySolutions(const std::vector<Eigen::Matrix3d>& R_vec, const std::vector<Eigen::Vector3d>& t_vec, Eigen::Matrix3d& R, Eigen::Vector3d& t, std::vector<bool>& max_inlier, std::vector<Eigen::Vector3d>& opt_X_curr){
    
	bool success;

	// Extract homogeneous 2D point which is inliered with essential constraint
	std::vector<int> idx_2D_inlier;
	for( int i = 0; i < num_feature_; i++ )
		if( features_[i].is_2D_inliered )
			idx_2D_inlier.push_back(i);

    int key_idx;
	std::vector<cv::Point2f> uv_prev, uv_curr;
    for (uint32_t i = 0; i < idx_2D_inlier.size(); i++){
        key_idx = features_[idx_2D_inlier[i]].life - 1 - (step_ - keystep_);
        uv_prev.emplace_back(features_[idx_2D_inlier[i]].uv[key_idx]);
		uv_curr.emplace_back(features_[idx_2D_inlier[i]].uv.back());
    }

	// Find reasonable rotation and translational vector
	int max_num = 0;
	for( uint32_t i = 0; i < R_vec.size(); i++ ){
		Eigen::Matrix3d R1 = R_vec[i];
		Eigen::Vector3d t1 = t_vec[i];
		
		std::vector<Eigen::Vector3d> X_prev, X_curr;
		std::vector<bool> inlier;
		constructDepth(uv_prev, uv_curr, R1, t1, X_prev, X_curr, inlier);

		int num_inlier = 0;
		for( uint32_t i = 0; i < inlier.size(); i++ )
			if( inlier[i] )
				num_inlier++;

		if( num_inlier > max_num ){
			max_num = num_inlier;
			max_inlier = inlier;
            opt_X_curr = X_curr;
			
			R = R1;
			t = t1;
		}
	}

	// Store 3D position in features
	if( max_num < num_feature_2D_inliered_*0.5 ){
        if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "Warning: There is no verified solution" << std::endl;
		success = false;
	}else{
		success = true;
	}

	return success;
}

// add extra feature extracted from the designated roi such as sign or crosswalk
/**
 * @brief 특수 특징점 추가 혹은 삭제
 * @details 요구에 따라 특별히 더 추출하거나 제거하여야 하는 특징점에 대해 업데이트한다.
 * @return 없음
 * @author Sangil Lee (sangillee724@gmail.com)
 * @date 29-Dec-2019
 */
void MVO::addExtraFeatures(){

    if( features_extra_.size() > 0 ){
        for( uint32_t i = 0; i < features_extra_.size(); i++ ){
            features_extra_[i].id = Feature::getNewID();
            features_.push_back(features_extra_[i]);
            num_feature_++;

            // Update bucket
            if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "@@@@@ extra uv: " << features_extra_[i].uv.back().x << ", " << features_extra_[i].uv.back().y << ", bucket: " << features_extra_[i].bucket.x << ", " << features_extra_[i].bucket.y << std::endl;
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

/**
 * @brief 특징점 검색
 * @details ROI 내에 속하는 특징점을 추출 혹은 삭제한다.
 * @param rois 특징점을 추출 혹은 삭제하고자 하는 ROI
 * @param num_feature 양수: 해당 개수만큼 특징점 추가, 음수: 영역 내 특징점 모두 삭제
 * @return 없음
 * @author Sangil Lee (sangillee724@gmail.com)
 * @date 29-Dec-2019
 */
void MVO::updateRoiFeatures(const std::vector<cv::Rect>& rois, const std::vector<int>& num_feature){

    cv::Rect roi;
    std::vector<cv::Point2f> keypoints;
    std::vector<uint32_t> idx_belong_to_roi;
    const int& bucket_safety = bucket_.safety;
    excludeMask_.setTo(cv::Scalar(255));
    
    for( uint32_t i = 0; i < rois.size(); i++ ){
        roi = rois[i];
        if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "@@@@@ roi[x,y,w,h]: " << roi.x << ", " << roi.y << ", " << roi.width << ", " << roi.height << ", num: " << num_feature[i] << std::endl;
        
        // Seek index of which feature is extracted specific roi
        getPointsInRoi(roi, idx_belong_to_roi);
        
        if( num_feature[i] < 0 ){ // remove all features within the roi
            for( uint32_t j = 0; j < idx_belong_to_roi.size(); j++ ){
                if( features_[idx_belong_to_roi[j]-j].landmark && MVO::s_file_logger_.is_open() ){
                    features_dead_.push_back(features_[idx_belong_to_roi[j]-j]);
                }
                features_.erase(features_.begin()+idx_belong_to_roi[j]-j);
            }
            num_feature_ = features_.size();

            try{
                excludeMask_(roi) = cv::Scalar(0); // update mask to not extract feature within the roi
            }catch(std::exception& msg){
                if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "Warning: " << msg.what() << std::endl;
                continue;
            }
        }else if( num_feature[i] > 0 ){ // add features within the roi
            try{
                cv::goodFeaturesToTrack(next_keyframe_.image(roi), keypoints, 50, 0.01, params_.min_px_dist, excludeMask_(roi), 3, true);
            }catch(std::exception& msg){
                if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "Warning: " << msg.what() << std::endl;
                continue;
            }
            
            if( keypoints.size() > 0 ){
                for( uint32_t l = 0; l < keypoints.size(); l++ ){
                    keypoints[l].x = keypoints[l].x + roi.x - 1;
                    keypoints[l].y = keypoints[l].y + roi.y - 1;
                }
            }else{
                if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "Warning: There is no keypoints within the bucket" << std::endl;
                try{
                    cv::goodFeaturesToTrack(next_keyframe_.image(roi), keypoints, 50, 0.1, params_.min_px_dist, excludeMask_(roi), 3, true);
                }catch(std::exception& msg){
                    if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "Warning: " << msg.what() << std::endl;
                    continue;
                }
            }

            int num_success = 0;
            int num_fail = 0;
            while( num_success < num_feature[i] && num_fail < 3 )
                if( updateRoiFeature(roi, keypoints, idx_belong_to_roi) )
                    num_success++;
                else
                    num_fail++;
        }
    }
}

/**
 * @brief 특징점 검색
 * @details ROI 내에 속하는 특징점 하나를 추가한다.
 * @param rois 특징점 하나를 추가하고자 하는 ROI
 * @param keypoints ROI 내에서 추출된 특징점들
 * @param idx_compared ROI 내에서 추가할 특징점과 거리를 비교할 대상들
 * @return 특징점이 성공적으로 추가되면, true
 * @author Sangil Lee (sangillee724@gmail.com)
 * @date 29-Dec-2019
 */
bool MVO::updateRoiFeature(const cv::Rect& roi, const std::vector<cv::Point2f>& keypoints, std::vector<uint32_t>& idx_compared){
    
    // Try to find a seperate feature
    bool success;
    double dist, min_dist, max_min_dist = 0;
    cv::Point2f best_keypoint;
    for( const auto & keypoint : keypoints ){
        success = true;
        min_dist = 1e9; // enough-large number
        for( const auto & idx : idx_compared ){
            if( features_[idx].uv.size() > 0 ){
                dist = cv::norm(keypoint - features_[idx].uv.back());
                
                if( dist < min_dist )
                    min_dist = dist;

                if( dist < params_.min_px_dist/2+1 ){
                    success = false;
                    break;
                }
            }
        }
        for( const auto & feature : features_extra_ ){
            if( feature.uv.size() > 0 ){
                dist = cv::norm(keypoint - feature.uv.back());
                
                if( dist < min_dist )
                    min_dist = dist;

                if( dist < params_.min_px_dist/2+1 ){
                    success = false;
                    break;
                }
            }
        }
        if( success ){
            if( min_dist > max_min_dist){
                max_min_dist = min_dist;
                best_keypoint = keypoint;
            }
        }
    }
    
    if( max_min_dist > 0.0 ){
        int row = std::floor((double) best_keypoint.y / params_.im_size.height * bucket_.grid.height);
        int col = std::floor((double) best_keypoint.x / params_.im_size.width * bucket_.grid.width);

        // Add new feature to VO object
        Feature newFeature;

        newFeature.frame_2d_init = -1; // frame step when the 2d point is tracked
        newFeature.frame_3d_init = -1; // frame step when the 3d point is initialized
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
        newFeature.landmark = NULL; // landmark point in world coordinates
        newFeature.type = Type::Unknown;
        newFeature.depthfilter = std::shared_ptr<DepthFilter>(new DepthFilter());

        features_extra_.push_back(newFeature);
        return true;
    }else{
        if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "Warning: There is no best-match keypoint which is far from the adjacent feature" << std::endl;
        return false;
    }
    
}

/**
 * @brief 영상 모듈 회전 행렬 사전 정보 입력
 * @details 각속도 데이터로부터 회전 행렬 사전 정보값을 입력한다.
 * @return 없음
 * @author Sangil Lee (sangillee724@gmail.com)
 * @date 29-Dec-2019
 */
void MVO::updateRotatePrior(){
    if( is_rotate_provided_ ){
        Eigen::Vector3d radian = Eigen::Vector3d::Zero();
        auto & stack = curr_keyframe_.angular_velocity_since_;
        if( stack.size() ) {
            for( uint32_t i = 0; i < stack.size()-1; i++ )
                radian += (stack[i].second+stack[i+1].second)/2 * (stack[i+1].first-stack[i].first);
            rotate_prior_ = params_.Tci.block(0,0,3,3) * skew(-radian).exp() * params_.Tic.block(0,0,3,3);
        }
        if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "rotate_prior: " << radian.transpose() << std::endl;
    }
}