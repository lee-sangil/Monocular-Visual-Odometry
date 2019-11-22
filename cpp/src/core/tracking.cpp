#include "core/MVO.hpp"
#include "core/utils.hpp"
#include "core/numerics.hpp"
#include "core/time.hpp"
#include "core/DepthFilter.hpp"

#include <exception>

uint32_t Feature::new_feature_id = 0;

bool MVO::extract_features(){
    // Update features using KLT tracker
    if( this->update_features() ){
        std::cerr << "# Update features: " << lsi::toc() << std::endl;
        
        // Delete features which is failed to track by KLT tracker
        this->delete_dead_features();
        std::cerr << "# Delete features: " << lsi::toc() << std::endl;
        
        // Add features to the number of the lost features
        this->add_features();

        // Add extra feature points
        this->add_extra_features();

        std::cerr << "# Add features: " << lsi::toc() << std::endl;
        
        return true;
    }
    else
        return false;
}

bool MVO::update_features(){

    if( this->nFeature ){
        // Track the points
        std::vector<cv::Point2f> points;
        std::vector<bool> validity;
        this->klt_tracker(points, validity);
        std::cerr << "## KLT tracker: " << lsi::toc() << std::endl;

        for( int i = 0; i < this->nFeature; i++ ){
            if( validity[i] && this->features[i].is_alive ){
                cv::Point2f uv_prev = this->features[i].uv.back();
                this->features[i].life++;
                if( this->rotate_provided )
                    this->features[i].uv_pred = this->calculateRotWarp(uv_prev);
                this->features[i].uv.push_back(points[i]);
                this->features[i].is_matched = true;
                this->nFeatureMatched++;

                Eigen::Vector3d epiLine = this->fundamentalMat * (Eigen::Vector3d() << uv_prev.x, uv_prev.y, 1).finished();
                double distFromEpiLine = std::abs(epiLine(0)*this->features[i].uv_pred.x + epiLine(1)*this->features[i].uv_pred.y + epiLine(2)) / epiLine.topRows(2).norm();
                if( this->rotate_provided && this->is_start && distFromEpiLine > this->params.max_epiline_dist )
                    this->features[i].type = Type::Dynamic;
            }else
                this->features[i].is_alive = false;
        }

        if( this->nFeatureMatched < this->params.thInlier ){
            std::cerr << "There are a few FEATURE MATCHES" << std::endl;
            return false;
        }else{
            return true;
        }
    }else
        return true;
}

void MVO::klt_tracker(std::vector<cv::Point2f>& fwd_pts, std::vector<bool>& validity){
    std::vector<cv::Point2f> pts, bwd_pts;
    pts.reserve(this->nFeature);
    bwd_pts.reserve(this->nFeature);
    for( int i = 0; i < this->nFeature; i++ )
        pts.push_back(this->features[i].uv.back());
    
    // Forward-backward error evaluation
    std::vector<cv::Mat>& prevPyr = this->prevPyramidTemplate;
    std::vector<cv::Mat>& currPyr = this->currPyramidTemplate;
    prevPyr.clear();
    currPyr.clear();
    std::cerr << "### Prepare variables: " << lsi::toc() << std::endl;

    cv::buildOpticalFlowPyramid(this->prev_image, prevPyr, cv::Size(21,21), 3, true);
    cv::buildOpticalFlowPyramid(this->cur_image, currPyr, cv::Size(21,21), 3, true);
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
    // this->descriptor->compute(this->cur_image, keypoints, desc);
    // bool desc_valid;

    bool border_invalid, error_valid;
    validity.reserve(pts.size());
    for( uint32_t i = 0; i < pts.size(); i++ ){
        border_invalid = (fwd_pts[i].x <= 0) | (fwd_pts[i].x >= this->params.imSize.width) | (fwd_pts[i].y <= 0) | (fwd_pts[i].y >= this->params.imSize.height);
        error_valid = cv::norm(pts[i] - bwd_pts[i]) < std::min( (double) cv::norm(pts[i] - fwd_pts[i])/5.0, 1.0);
        // desc_valid = cv::norm(this->features[i].desc - desc.row(i));

        validity.push_back(!border_invalid & error_valid);
        // bool valid = ~border_invalid & status.at<uchar>(i);// & err.at<float>(i) < std::min( cv::norm(pts[i] - fwd_pts[i])/5.0, 2.0);
        // validity.push_back(valid);
    }
}

void MVO::delete_dead_features(){
    for( uint32_t i = 0; i < this->features.size(); ){
        if( this->features[i].is_alive == false ){
            if( this->features[i].life > 2 ){
                this->features_dead.push_back(this->features[i]);
                // std::cout << "features_dead increase, ";
            }
            this->features.erase(this->features.begin()+i);
            // std::cout << "features decreases" << std::endl;
        }else{
            i++;
        }
    }
    this->nFeature = this->features.size();
}

void MVO::add_features(){
    this->update_bucket();
    std::cerr << "## Update bucket: " << lsi::toc() << std::endl;

    while( this->nFeature < this->bucket.max_features && this->bucket.saturated.any() == true )
        this->add_feature();

    // WARNING: descriptor may remove keypoint whose description cannot be extracted
    // cv::Mat desc;
    // std::vector<cv::KeyPoint> keypoints;
    // for( int i = 0; i < this->nFeature; i++ )
    //     if( this->features[i].life == 1 )
    //         keypoints.emplace_back(cv::Point2f(this->features[i].uv.back().x,this->features[i].uv.back().y),1.0);
    // this->descriptor->compute(this->cur_image, keypoints, desc);
    // int j = 0;
    // for( int i = 0; i < this->nFeature; i++ )
    //     if( this->features[i].life == 1 )
    //         this->features[i].desc = desc.row(j++).clone();
}

void MVO::update_bucket(){
    this->bucket.mass.fill(0.0);
    this->bucket.saturated.fill(1.0);
    for( int i = 0; i < this->nFeature; i++ ){
        uint32_t row_bucket = std::floor(this->features[i].uv.back().y / this->params.imSize.height * this->bucket.grid.height);
        uint32_t col_bucket = std::floor(this->features[i].uv.back().x / this->params.imSize.width * this->bucket.grid.width);
        this->features[i].bucket = cv::Point(col_bucket, row_bucket);
        this->bucket.mass(row_bucket, col_bucket)++;
    }
}

void MVO::add_feature(){
    // Load bucket parameters
    cv::Size bkSize = this->bucket.size;
    uint32_t bkSafety = this->bucket.safety;

    // Choose ROI based on the probabilistic approaches with the mass of bucket
    int row, col;
    lsi::idx_randselect(this->bucket.prob, this->bucket.saturated, row, col);
    cv::Rect roi = cv::Rect(col*bkSize.width+1, row*bkSize.height+1, bkSize.width, bkSize.height);
    
    // std::cout << "mask:" << std::endl << this->bucket.saturated << std::endl;
    // std::cout << "row: " << row << ", col: " << col << std::endl;

    roi.x = std::max(bkSafety, (uint32_t)roi.x);
    roi.y = std::max(bkSafety, (uint32_t)roi.y);
    roi.width = std::min(this->params.imSize.width-bkSafety, (uint32_t)roi.x+roi.width)-roi.x;
    roi.height = std::min(this->params.imSize.height-bkSafety, (uint32_t)roi.y+roi.height)-roi.y;

    // Seek index of which feature is extracted specific bucket
    std::vector<uint32_t>& idxBelongToBucket = this->idxTemplate;
    idxBelongToBucket.clear();

    for( int l = 0; l < this->nFeature; l++ ){
        for( int ii = std::max(col-1,0); ii <= std::min(col+1,this->bucket.grid.width-1); ii++){
            for( int jj = std::max(row-1,0); jj <= std::min(row+1,this->bucket.grid.height-1); jj++){
                if( (this->features[l].bucket.x == ii) & (this->features[l].bucket.y == jj)){
                    idxBelongToBucket.push_back(l);
                }
            }
        }
    }
    uint32_t nInBucket = idxBelongToBucket.size();
    
    // Try to find a seperate feature
    cv::Mat crop_image;
    std::vector<cv::Point2f> keypoints;

    crop_image = this->cur_image(roi);
    cv::goodFeaturesToTrack(crop_image, keypoints, 50, 0.1, 2.0, cv::noArray(), 3, true);
    
    if( keypoints.size() == 0 ){
        this->bucket.saturated(row,col) = 0.0;
        // std::cout << "Feature cannot be found!" << std::endl;
        return;
    }else{
        for( uint32_t l = 0; l < keypoints.size(); l++ ){
            keypoints[l].x = keypoints[l].x + roi.x - 1;
            keypoints[l].y = keypoints[l].y + roi.y - 1;
        }
    }

    bool success;
    double dist, minDist, maxMinDist = 0;
    cv::Point2f bestKeypoint;
    for( uint32_t l = 0; l < keypoints.size(); l++ ){
        success = true;
        minDist = 1e9; // enough-large number
        for( uint32_t f = 0; f < nInBucket; f++ ){
            dist = cv::norm(keypoints[l] - this->features[idxBelongToBucket[f]].uv.back());
            
            if( dist < minDist )
                minDist = dist;
            
            if( dist < this->params.min_px_dist ){
                success = false;
                break;
            }
        }
        if( success ){
            if( minDist > maxMinDist){
                maxMinDist = minDist;
                bestKeypoint = keypoints[l];
            }
        }
    }
    
    if( maxMinDist > 0.0 ){
        // Add new feature to VO object
        Feature newFeature;

        newFeature.id = Feature::new_feature_id; // unique id of the feature
        newFeature.frame_init = 0; // frame step when the 3d point is initialized
        newFeature.uv.emplace_back(bestKeypoint.x, bestKeypoint.y); // uv point in pixel coordinates
        newFeature.uv_pred = cv::Point2f(-1,-1);
        newFeature.life = 1; // the number of frames in where the feature is observed
        newFeature.bucket = cv::Point(col, row); // the location of bucket where the feature belong to
        newFeature.point.setZero(4,1); // 3-dim homogeneous point in the local coordinates
        newFeature.point(3) = 1;
        newFeature.is_alive = true;
        newFeature.is_matched = false; // matched between both frame
        newFeature.is_wide = false; // verify whether features btw the initial and current are wide enough
        newFeature.is_2D_inliered = false; // belong to major (or meaningful) movement
        newFeature.is_3D_reconstructed = false; // triangulation completion
        newFeature.is_3D_init = false; // scale-compensated
        newFeature.point_init.setZero(4,1); // scale-compensated 3-dim homogeneous point in the global coordinates
        newFeature.point_init(3) = 1;
        newFeature.type = Type::Unknown;
        newFeature.depth = new DepthFilter();

        this->features.push_back(newFeature);
        this->nFeature++;
    
        Feature::new_feature_id++;

        // Update bucket
        this->bucket.mass(row, col)++;

        cv::eigen2cv(this->bucket.mass, this->bucket.cvMass);
        cv::GaussianBlur(this->bucket.cvMass, this->bucket.cvProb, cv::Size(21,21), 3.0);
        cv::cv2eigen(this->bucket.cvProb, this->bucket.prob);

        this->bucket.prob.array() += 0.05;
        this->bucket.prob = this->bucket.prob.cwiseInverse();

        // Assign high weight for ground
        for( int i = 0; i < this->bucket.prob.rows(); i++ ){
            // weight.block(i,0,1,weight.cols()) /= std::pow(weight.rows(),2);
            this->bucket.prob.block(i,0,1,this->bucket.prob.cols()) *= i+1;
        }
        // std::cout << "Feature is added!" << std::endl;
    }else{
        this->bucket.saturated(row,col) = 0.0;
        // std::cout << "Feature cannot be found!" << std::endl;
    }
}

// haram
bool MVO::calculate_essential()
{
    if (this->step == 0)
        return true;

    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;
    points1.reserve(this->nFeature);
    points2.reserve(this->nFeature);

    std::vector<uint32_t> idx_static;
    int nWideFeature = 0;
    int key_idx;
    for( int i = 0; i < this->nFeature; i++ ){
        key_idx = this->features[i].life - 1 - (this->step - this->key_step);
        if( key_idx >= 0 && this->features[i].type != Type::Dynamic ){
            points1.push_back(this->features[i].uv[key_idx]);
            points2.push_back(this->features[i].uv.back());   // latest
            idx_static.push_back(i);
            if( cv::norm(this->features[i].uv[key_idx] - this->features[i].uv.back()) > this->params.px_wide ){
                this->features[i].is_wide = true;
                nWideFeature++;
            }else{
                this->features[i].is_wide = false;
            }
        }
    }

    if( points1.size() <= this->nFeature * this->params.thRatioKeyFrame ){
        this->keystepVec.push_back(this->step);

        if( this->speed_provided ){
            double last_timestamp = this->timestampSinceKeyframe.back();
            double last_speed = this->speedSinceKeyframe.back();

            this->timestampSinceKeyframe.clear();
            this->speedSinceKeyframe.clear();

            this->timestampSinceKeyframe.push_back(last_timestamp);
            this->speedSinceKeyframe.push_back(last_speed);
        }

        if( this->rotate_provided ){
            double last_timestamp = this->timestampSinceKeyframe.back();
            Eigen::Vector3d last_gyro = this->gyroSinceKeyframe.back();

            this->timestampSinceKeyframe.clear();
            this->gyroSinceKeyframe.clear();

            this->timestampSinceKeyframe.push_back(last_timestamp);
            this->gyroSinceKeyframe.push_back(last_gyro);
        }

        std::cerr << "key step: " << this->key_step << ' ' << std::endl;
    }

    cv::Mat inlier_mat;
    this->essentialMat = cv::findEssentialMat(points1, points2, this->params.Kcv, cv::RANSAC, 0.999, 1.5, inlier_mat);
    std::cerr << "# Calculate essential: " << lsi::toc() << std::endl;
    
    Eigen::Matrix3d E_;
    cv::cv2eigen(this->essentialMat, E_);
    this->fundamentalMat = this->params.Kinv.transpose() * E_ * this->params.Kinv;

    double error;
    std::vector<double> essentialError;
    for( uint32_t i = 0; i < points1.size(); i++ ){
        error = (Eigen::Vector3d() << points2[i].x,points2[i].y,1).finished().transpose() * this->params.Kinv.transpose() * E_ * this->params.Kinv * (Eigen::Vector3d() << points1[i].x,points1[i].y,1).finished();
        essentialError.push_back(error);
    }

    uint32_t inlier_cnt = 0;
    bool* inlier = inlier_mat.ptr<bool>(0);
    for (int i = 0; i < inlier_mat.rows; i++){
        if (inlier[i]){
            this->features[idx_static[i]].is_2D_inliered = true;
            inlier_cnt++;
        // }else if( essentialError[i] > 1e-4 ){
        }else{
            this->features[idx_static[i]].type = Type::Dynamic;
        }
    }
    this->nFeature2DInliered = inlier_cnt;
    std::cerr << "nFeature2DInliered: " << (double) this->nFeature2DInliered / this->nFeature * 100 << '%' << std::endl;

    Eigen::Matrix3d U,V;
    switch( this->params.SVDMethod){
        case MVO::SVD::JACOBI:{
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(E_, Eigen::ComputeThinU | Eigen::ComputeThinV);
            U = svd.matrixU();
            V = svd.matrixV();
            break;
        }
        case MVO::SVD::OpenCV:{
            cv::Mat Vt, U_, W;
            cv::SVD::compute(this->essentialMat, W, U_, Vt);

            Eigen::MatrixXd Vt_;
            cv::cv2eigen(Vt, Vt_);
            V = Vt_.transpose();
            cv::cv2eigen(U_, U);
            break;
        }
        case MVO::SVD::BDC:
        default:{
            Eigen::Matrix3d E_;
            cv::cv2eigen(this->essentialMat, E_);
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

    this->R_vec.clear();
    this->t_vec.clear();
    this->R_vec.push_back(U * W * V.transpose());
    this->R_vec.push_back(U * W * V.transpose());
    this->R_vec.push_back(U * W.transpose() * V.transpose());
    this->R_vec.push_back(U * W.transpose() * V.transpose());
    this->t_vec.push_back(U.block(0, 2, 3, 1));
    this->t_vec.push_back(-U.block(0, 2, 3, 1));
    this->t_vec.push_back(U.block(0, 2, 3, 1));
    this->t_vec.push_back(-U.block(0, 2, 3, 1));

    std::cerr << "# Extract R, t: " << lsi::toc() << std::endl;

    if (this->nFeature2DInliered < this->params.thInlier){
        std::cerr << " There are a few inliers matching features in 2D." << std::endl;
        return false;
    }else{
        this->is_start = true;
        return true;
    }
}

void MVO::add_extra_features(){

    if( this->features_extra.size() > 0 ){
        for( uint32_t i = 0; i < this->features_extra.size(); i++ ){
            this->features_extra[i].id = Feature::new_feature_id;
            this->features.push_back(this->features_extra[i]);
            this->nFeature++;
        
            Feature::new_feature_id++;

            // Update bucket
            this->bucket.mass(this->features_extra[i].bucket.y, this->features_extra[i].bucket.x)++;
        }
        cv::eigen2cv(this->bucket.mass, this->bucket.cvMass);
        cv::GaussianBlur(this->bucket.cvMass, this->bucket.cvProb, cv::Size(21,21), 3.0);
        cv::cv2eigen(this->bucket.cvProb, this->bucket.prob);

        this->bucket.prob.array() += 0.05;
        this->bucket.prob = this->bucket.prob.cwiseInverse();

        // Assign high weight for ground
        for( int i = 0; i < this->bucket.prob.rows(); i++ ){
            // weight.block(i,0,1,weight.cols()) /= std::pow(weight.rows(),2);
            this->bucket.prob.block(i,0,1,this->bucket.prob.cols()) *= i+1;
        }

        this->features_extra.clear();
    }
}

void MVO::extract_roi_features(std::vector<cv::Rect> rois, std::vector<int> nFeature){

    cv::Rect roi;
    for( uint32_t i = 0; i < rois.size(); i++ ){
        roi = rois[i];
        int bkSafety = this->bucket.safety;
        roi.x = std::max(bkSafety, roi.x);
        roi.y = std::max(bkSafety, roi.y);
        roi.width = std::min(this->params.imSize.width-bkSafety, roi.x+roi.width)-roi.x;
        roi.height = std::min(this->params.imSize.height-bkSafety, roi.y+roi.height)-roi.y;

        int nSuccess = 0;
        int nTry = 0;
        while( nSuccess < nFeature[i] && nTry++ < 3 )
            if( this->extract_roi_feature(roi) )
                nSuccess++;
    }
}

bool MVO::extract_roi_feature(cv::Rect& roi){

    int row, col;
    row = (roi.x-1) / this->bucket.size.height;
    col = (roi.y-1) / this->bucket.size.width;

    // Seek index of which feature is extracted specific bucket
    std::vector<uint32_t>& idxBelongToBucket = this->idxTemplate;
    idxBelongToBucket.clear();

    for( int l = 0; l < this->nFeature; l++ ){
        for( int ii = std::max(col-1,0); ii <= std::min(col+1,this->bucket.grid.width-1); ii++){
            for( int jj = std::max(row-1,0); jj <= std::min(row+1,this->bucket.grid.height-1); jj++){
                if( (this->features[l].bucket.x == ii) & (this->features[l].bucket.y == jj)){
                    idxBelongToBucket.push_back(l);
                }
            }
        }
    }
    uint32_t nInBucket = idxBelongToBucket.size();
    
    // Try to find a seperate feature
    cv::Mat crop_image;
    std::vector<cv::Point2f> keypoints;

    try{
        crop_image = this->cur_image(roi);
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
    double dist, minDist, maxMinDist = 0;
    cv::Point2f bestKeypoint;
    for( uint32_t l = 0; l < keypoints.size(); l++ ){
        success = true;
        minDist = 1e9; // enough-large number
        for( uint32_t f = 0; f < nInBucket; f++ ){
            dist = cv::norm(keypoints[l] - this->features[idxBelongToBucket[f]].uv.back());
            
            if( dist < minDist )
                minDist = dist;

            if( dist < this->params.min_px_dist/2+1 ){
                success = false;
                break;
            }
        }
        if( success ){
            if( minDist > maxMinDist){
                maxMinDist = minDist;
                bestKeypoint = keypoints[l];
            }
        }
    }
    
    if( maxMinDist > 0.0 ){
        // Add new feature to VO object
        Feature newFeature;

        newFeature.frame_init = 0; // frame step when the 3d point is initialized
        newFeature.uv.emplace_back(bestKeypoint.x, bestKeypoint.y); // uv point in pixel coordinates
        newFeature.uv_pred = cv::Point2f(-1,-1);
        newFeature.life = 1; // the number of frames in where the feature is observed
        newFeature.bucket = cv::Point(col, row); // the location of bucket where the feature belong to
        newFeature.point.setZero(4,1); // 3-dim homogeneous point in the local coordinates
        newFeature.point(3) = 1;
        newFeature.is_alive = true;
        newFeature.is_matched = false; // matched between both frame
        newFeature.is_wide = false; // verify whether features btw the initial and current are wide enough
        newFeature.is_2D_inliered = false; // belong to major (or meaningful) movement
        newFeature.is_3D_reconstructed = false; // triangulation completion
        newFeature.is_3D_init = false; // scale-compensated
        newFeature.point_init.setZero(4,1); // scale-compensated 3-dim homogeneous point in the global coordinates
        newFeature.point_init(3) = 1;
        newFeature.type = Type::Unknown;
        newFeature.depth = new DepthFilter();

        MVO::features_extra.push_back(newFeature);
        return true;
    }else{
        return false;
    }
    
}