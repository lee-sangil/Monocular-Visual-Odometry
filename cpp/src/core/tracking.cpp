#include "core/MVO.hpp"
#include "core/utils.hpp"
#include "core/numerics.hpp"
#include "core/time.hpp"
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
        std::cerr << "# Add features: " << lsi::toc() << std::endl;
        std::cerr << "nFeatures = " << this->nFeature << std::endl;
        
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
            if( validity[i] & (this->features[i].life > 0) ){
                this->features[i].life++;
                this->features[i].uv.push_back(points[i]);
                this->features[i].is_matched = true;
                
                if( cv::norm(this->features[i].uv.front() - this->features[i].uv.back()) > this->params.px_wide ){
                    this->features[i].is_wide = true;
                }else{
                    this->features[i].is_wide = false;
                }
                this->nFeatureMatched++;

            }else
                this->features[i].life = 0;
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
    for( int i = 0; i < this->nFeature; i++ ){
        pts.push_back(this->features[i].uv.back());
    }
    
    // Forward-backward error evaluation
    std::vector<cv::Mat>& prevPyr = this->prevPyramidTemplate;
    std::vector<cv::Mat>& currPyr = this->currPyramidTemplate;
    prevPyr.clear();
    currPyr.clear();
    std::cerr << "### Prepare variables: " << lsi::toc() << std::endl;

    cv::buildOpticalFlowPyramid(this->prev_image, prevPyr, cv::Size(21,21), 2, true);
    cv::buildOpticalFlowPyramid(this->cur_image, currPyr, cv::Size(21,21), 2, true);
    std::cerr << "### Build pyramids: " << lsi::toc() << std::endl;

    cv::Mat status, err;
    cv::calcOpticalFlowPyrLK(prevPyr, currPyr, pts, fwd_pts, status, err);
    cv::calcOpticalFlowPyrLK(currPyr, prevPyr, fwd_pts, bwd_pts, status, err);
    std::cerr << "### Calculate optical flows: " << lsi::toc() << std::endl;
    
    // cv::calcOpticalFlowPyrLK(this->prev_image, this->cur_image, pts, fwd_pts, status, err, cv::Size(21,21), 2);
    // cv::calcOpticalFlowPyrLK(this->cur_image, this->prev_image, fwd_pts, bwd_pts, status, err, cv::Size(21,21), 2);
    // std::cerr << "### Calculate optical flows with build: " << lsi::toc() << std::endl;

    // Calculate bi-directional error( = validity ): validity = ~border_invalid & error_valid
    validity.reserve(this->nFeature);
    for( int i = 0; i < this->nFeature; i++ ){
        bool border_invalid = (fwd_pts[i].x < 0) | (fwd_pts[i].x > this->params.imSize.width) | (fwd_pts[i].y < 0) | (fwd_pts[i].y > this->params.imSize.height);
        bool error_valid = cv::norm(pts[i] - bwd_pts[i]) < std::min( cv::norm(pts[i] - fwd_pts[i])/5.0, 2.0);
        validity.push_back(!border_invalid & error_valid);
        // bool valid = ~border_invalid & status.at<uchar>(i);// & err.at<float>(i) < std::min( cv::norm(pts[i] - fwd_pts[i])/5.0, 2.0);
        // validity.push_back(valid);
    }
}

void MVO::delete_dead_features(){
    // this->features_temp.clear();
    // for( uint32_t i = 0; i < this->features.size(); i++){
    //     if( this->features[i].life > 0 ){
    //         this->features_temp.push_back(this->features[i]);
    //     }
    // }
    // // this->features.swap(this->features_temp);
    // this->nFeature = this->features_temp.size();
    // std::cerr << "# Delete features with new vector: " << lsi::toc() << std::endl;

    for( uint32_t i = 0; i < this->features.size(); ){
        if( this->features[i].life <= 0 ){
            this->features.erase(this->features.begin()+i);
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
}

void MVO::update_bucket(){
    this->bucket.mass.fill(0.0);
    this->bucket.saturated.fill(1.0);
    for( int i = 0; i < this->nFeature; i++ ){
        uint32_t row_bucket = std::ceil(this->features[i].uv.back().y / this->params.imSize.height * this->bucket.grid.height);
        uint32_t col_bucket = std::ceil(this->features[i].uv.back().x / this->params.imSize.width * this->bucket.grid.width);
        this->features[i].bucket = cv::Point(row_bucket, col_bucket);
        this->bucket.mass(row_bucket-1, col_bucket-1)++;
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
        for( int ii = std::max(col-1,0); ii < std::min(col+1,this->bucket.grid.width); ii++){
            for( int jj = std::max(row-1,0); jj < std::min(row+1,this->bucket.grid.height); jj++){
                if( (this->features[l].bucket.x == ii) & (this->features[l].bucket.y == jj)){
                    idxBelongToBucket.push_back(l);
                }
            }
        }
    }
    uint32_t nInBucket = idxBelongToBucket.size();
    
    // Try to find a seperate feature
    double filterSize = 5.0;
    cv::Mat crop_image;
    std::vector<cv::Point2f> keypoints;

    while( true ){

        if( filterSize < 3.0 ){
            this->bucket.saturated(row,col) = 0.0;
            // std::cout << "Feature cannot be found!" << std::endl;
            return;
        }

        crop_image = this->cur_image(roi);
        cv::goodFeaturesToTrack(crop_image, keypoints, 50, 0.01, 2.0, cv::noArray(), 3, true);
        
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
            newFeature.frame_init = this->step; // frame step when the feature is created
            newFeature.uv.emplace_back(bestKeypoint.x, bestKeypoint.y); // uv point in pixel coordinates
            newFeature.life = 1; // the number of frames in where the feature is observed
            newFeature.bucket = cv::Point(row, col); // the location of bucket where the feature belong to
            newFeature.point.setZero(4,1); // 3-dim homogeneous point in the local coordinates
            newFeature.is_matched = false; // matched between both frame
            newFeature.is_wide = false; // verify whether features btw the initial and current are wide enough
            newFeature.is_2D_inliered = false; // belong to major (or meaningful) movement
            newFeature.is_3D_reconstructed = false; // triangulation completion
            newFeature.is_3D_init = false; // scale-compensated
            newFeature.point_init.setZero(4,1); // scale-compensated 3-dim homogeneous point in the global coordinates
            newFeature.point_var = 0;

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

            return;
        }
        filterSize = filterSize - 2;
    }
}

// haram
bool MVO::calculate_essential()
{
    Eigen::Matrix3d K = this->params.K;

    // below define should go in constructor and class member variable
    double focal = (K(0, 0) + K(1, 1)) / 2;
    cv::Point2f principle_point(K(0, 2), K(1, 2));
    Eigen::Matrix3d W;
    W << 0, -1, 0, 1, 0, 0, 0, 0, 1;

    if (this->step == 0)
        return true;

    // Extract homogeneous 2D point which is matched with corresponding feature
    // TODO: define of find function

    std::vector<uint32_t>& idx = this->idxTemplate;
    idx.clear();

    for (uint32_t i = 0; i < this->features.size(); i++)
    {
        if (this->features[i].is_matched)
            idx.push_back(i);
    }

    uint32_t nInlier = idx.size();

    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;
    points1.reserve(nInlier);
    points2.reserve(nInlier);

    // initialize the points here ... */
    int len;
    for (uint32_t i = 0; i < nInlier; i++)
    {
        len = this->features[idx[i]].uv.size();
        points1.push_back(this->features[idx[i]].uv[len - 2]); // second to latest
        points2.push_back(this->features[idx[i]].uv.back());                             // latest
    }

    cv::Mat inlier_mat;
    this->essentialMat = cv::findEssentialMat(points1, points2, focal, principle_point, cv::RANSAC, 0.999, 0.5, inlier_mat);
    std::cerr << "# Calculate essential: " << lsi::toc() << std::endl;

    Eigen::Matrix3d U,V;
    switch( this->params.SVDMethod){
        case MVO::SVD::JACOBI:{
            Eigen::Matrix3d E_;
            cv::cv2eigen(this->essentialMat, E_);
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

    uint32_t inlier_cnt = 0;
    for (int i = 0; i < inlier_mat.rows; i++){
        if (inlier_mat.at<char>(i)){
            this->features[i].is_2D_inliered = true;
            inlier_cnt++;
        }
    }
    this->nFeature2DInliered = inlier_cnt;
    std::cerr << "nFeature2DInliered: " << this->nFeature2DInliered << std::endl;

    if (this->nFeature2DInliered < this->params.thInlier)
    {
        std::cerr << " There are a few inliers matching features in 2D." << std::endl;
        return false;
    }
    else
    {
        return true;
    }
}