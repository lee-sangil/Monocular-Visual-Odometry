#include "core/MVO.hpp"
#include "core/utils.hpp"
#include "core/numerics.hpp"

uint32_t Feature::new_feature_id = 1;

MVO::MVO(Parameter params){
    this->step = 0;
    
    this->nFeature = 0;

	this->params.fx = params.fx;
	this->params.fy = params.fy;
	this->params.cx = params.cx;
	this->params.cy = params.cy;
	this->params.k1 = params.k1;
	this->params.k2 = params.k2;
	this->params.p1 = params.p1;
	this->params.p2 = params.p2;
	this->params.k3 = params.k3;
	this->params.width = params.width;
	this->params.height = params.height;

	this->params.K << params.fx, 0, params.cx,
						0, params.fy, params.cy,
						0, 0, 1;

	this->params.imSize.width = params.width;
	this->params.imSize.height = params.height;
	this->params.radialDistortion.push_back(params.k1);
	this->params.radialDistortion.push_back(params.k2);
	this->params.radialDistortion.push_back(params.k3);
	this->params.tangentialDistortion.push_back(params.p1);
	this->params.tangentialDistortion.push_back(params.p2);
    obj.scale_initialized = false;

    // Bucket
    this->bucket = Bucket();
    this->bucket.safety = 20;
	this->bucket.max_features = 400;
	this->bucket.grid = cv::Size(32,8);
	this->bucket.size = cv::Size(this->params.imSize.width/this->bucket.grid.width, this->params.imSize.height/this->bucket.grid.height);
	this->bucket.mass.setZero(this->bucket.grid.width,this->bucket.grid.height);
	this->bucket.prob.setZero(this->bucket.grid.width,this->bucket.grid.height);

    // Variables
    obj.nFeature = 0;
    obj.nFeatureMatched = 0;
    obj.nFeature2DInliered = 0;
    obj.nFeature3DReconstructed = 0;
    obj.nFeatureInlier = 0;
    obj.params.thInlier = 5;
    obj.params.min_px_dist = 7;
    obj.new_feature_id = 1;
    
    % Initial position
    obj.TRec{1} = eye(4);
    obj.TocRec{1} = eye(4);
    obj.PocRec(:,1) = [0;0;0;1];
    
    % Gaussian kernel
    std = 3;
    sz = 3*std;
    d = (repmat(-sz : sz, 2*sz+1,1).^2 + (repmat(-sz : sz, 2*sz+1,1).^2).').^0.5;
    obj.params.kernel = normpdf(d, zeros(size(d)), std*ones(size(d)));
    
    % RANSAC parameter			
    ransacCoef_scale_prop.iterMax = 1e4;
    ransacCoef_scale_prop.minPtNum = obj.params.thInlier;
    ransacCoef_scale_prop.thInlrRatio = 0.9;
    ransacCoef_scale_prop.thDist = .5; % standard deviation
    ransacCoef_scale_prop.thDistOut = 5; % three times of standard deviation
    ransacCoef_scale_prop.funcFindF = @obj.calculate_scale;
    ransacCoef_scale_prop.funcDist = @obj.calculate_scale_error;
    obj.params.ransacCoef_scale_prop = ransacCoef_scale_prop;
    
    % Statistical model
    obj.params.var_theta = (90 / 1241 / 2)^2;
    obj.params.var_point = 1;
    
    % 3D reconstruction
    obj.params.vehicle_height = 1.5; % in meter
    obj.params.initScale = 1;
    obj.params.reprojError = 1.2;
}

void MVO::refresh(){
    this->nFeatureMatched = 0;
    this->nFeature2DInliered = 0;
    this->nFeature3DReconstructed = 0;
    this->nFeatureInlier = 0;

    for( uint32_t i = 0; i < this->nFeature; i++ ){
        this->features[i].is_matched = false;
        this->features[i].is_2D_inliered = false;
        this->features[i].is_3D_reconstructed = false;
    }
}

void MVO::backup(){
    this->features_backup.clear();
    this->features_backup.assign(this->features.begin(), this->features.end());
}

void MVO::reload(){
    this->features.clear();
    this->features.assign(this->features_backup.begin(), this->features_backup.end());

    this->nFeature = this->features.size();

    this->TRec[this->step] = Eigen::Matrix4d::Identity();
    this->TocRec[this->step] = this->TocRec[this->step-1];
    this->PocRec[this->step] = this->PocRec[this->step-1];
}

void MVO::set_image(const cv::Mat image){
	this->prev_image = this->cur_image.clone();
	this->cur_image = image.clone(); 
}

void MVO::run(const cv::Mat image){
    this->step++;

    this->set_image(image);
    this->refresh();

    if( this->extract_features() & // Extract and update features
	    this->calculate_essential() & // RANSAC for calculating essential/fundamental matrix
		this->calculate_motion() ) // Extract rotational and translational from fundamental matrix
        
        this->backup();
        
    else
        this->reload();
}

bool MVO::extract_features(){
    // Update features using KLT tracker
    if( this->update_features() ){
        
        // Delete features which is failed to track by KLT tracker
        this->delete_dead_features();
        
        // Add features to the number of the lost features
        this->add_features();
        
        return true;
    }
    else
        return false;
}

bool MVO::update_features(){
    
    std::vector<Feature> features;
    features.assign(this->features.begin(),this->features.end());

    if( this->nFeature ){
        // Track the points
        std::vector<cv::Point2d> points;
        std::vector<bool> validity;
        this->klt_tracker(points, validity);
        
        for( uint32_t i = 0; i < this->nFeature; i++ ){
            if( validity[i] & (features[i].life > 0) ){
                features[i].life++;
                features[i].uv.push_back(points[i]);
                features[i].is_matched = true;
                
                if( cv::norm(features[i].uv.front() - features[i].uv.back()) > this->params.min_px_dist ){
                    features[i].is_wide = true;
                }else{
                    features[i].is_wide = false;
                }
                this->nFeatureMatched++;

            }else
                features[i].life = 0;
        }

        if( this->nFeatureMatched < this->params.thInlier ){
            std::cerr << "There are a few FEATURE MATCHES" << std::endl;
            return false;
        }else{
            return true;
            this->features.assign(features.begin(), features.end());
        }
    }else
        return true;
}

void MVO::klt_tracker(std::vector<cv::Point2d>& fwd_pts, std::vector<bool>& validity){
    std::vector<cv::Point2d> pts;
    for( uint32_t i = 0; i < this->nFeature; i++ ){
        pts.push_back(this->features[i].uv.back());
    }
    
    // Forward-backward error evaluation
    std::vector<cv::Point2d> bwd_pts;
    std::vector<cv::Mat> prevPyr, currPyr;
    cv::Mat status, err;
    cv::buildOpticalFlowPyramid(this->prev_image, prevPyr, cv::Size(21,21), 3, true);
    cv::buildOpticalFlowPyramid(this->cur_image, currPyr, cv::Size(21,21), 3, true);
    cv::calcOpticalFlowPyrLK(prevPyr, currPyr, pts, fwd_pts, status, err);
    cv::calcOpticalFlowPyrLK(currPyr, prevPyr, fwd_pts, bwd_pts, status, err);
    
    // Calculate bi-directional error( = validity ): validity = ~border_invalid & error_valid
    for( uint32_t i = 0; i < this->nFeature; i++ ){
        bool border_invalid = (fwd_pts[i].x < 0) | (fwd_pts[i].x > this->params.imSize.width) | (fwd_pts[i].y < 0) | (fwd_pts[i].y > this->params.imSize.height);
        bool error_valid = cv::norm(pts[i] - bwd_pts[i]) < std::min( cv::norm(pts[i] - fwd_pts[i])/5.0, 1.0);
        validity.push_back(!border_invalid & error_valid);
    }
}

void MVO::delete_dead_features(){
    for( uint32_t i = 0; i < this->nFeature; ){
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
    while( this->nFeature < this->bucket.max_features )
        this->add_feature();
}

void MVO::update_bucket(){
    this->bucket.mass.fill(0.0);
    for( uint32_t i = 0; i < this->nFeature; i++ ){
        cv::Point2d uv = this->features[i].uv.back();
        uint32_t row_bucket = std::ceil(uv.x / this->params.imSize.width * this->bucket.grid.width);
        uint32_t col_bucket = std::ceil(uv.y / this->params.imSize.height * this->bucket.grid.height);
        this->features[i].bucket = cv::Size(row_bucket, col_bucket);
        this->bucket.mass(row_bucket-1, col_bucket-1)++;
    }
}

void MVO::add_feature(){
    // Load bucket parameters
    cv::Size bkSize = this->bucket.size;
    uint32_t bkSafety = this->bucket.safety;

    // Choose ROI based on the probabilistic approaches with the mass of bucket
    uint32_t i, j;
    lsi::idx_randselect(this->bucket.prob, i, j);
    cv::Rect roi = cv::Rect((i-1)*bkSize.width+1, (j-1)*bkSize.height+1, bkSize.width, bkSize.height);
    
    roi.x = std::max(bkSafety, (uint32_t)roi.x);
    roi.y = std::max(bkSafety, (uint32_t)roi.y);
    roi.width = std::min(this->params.imSize.width-bkSafety, (uint32_t)roi.x + roi.width)-roi.x;
    roi.height = std::min(this->params.imSize.height-bkSafety, (uint32_t)roi.y+roi.height)-roi.y;

    // Seek index of which feature is extracted specific bucket
    std::vector<uint32_t> idxBelongToBucket;
    for( uint32_t it = 0; it < this->nFeature; it++ ){
        if( ((uint32_t)this->features[it].bucket.x == i) & ((uint32_t)this->features[it].bucket.y == j)){
            idxBelongToBucket.push_back(i);
        }
    }
    uint32_t nInBucket = idxBelongToBucket.size();
    
    // Try to find a seperate feature
    double filterSize = 5.0;

    while( true ){

        if( filterSize < 3.0 )
            return;

        cv::Mat crop_image = this->cur_image(roi);
        std::vector<cv::Point2d> keypoints;
        cv::goodFeaturesToTrack(crop_image, keypoints, 1000, 0.01, 2.0, 0, 3, true);
        
        if( keypoints.size() == 0 )
            return;
        else{
            for( uint32_t l = 0; l < keypoints.size(); l++ ){
                keypoints[l].x = keypoints[l].x + roi.x - 1;
                keypoints[l].y = keypoints[l].y + roi.y - 1;
            }
        }

        double dist;
        for( uint32_t l = 1; l < keypoints.size(); l++ ){
            bool success = true;
            for( uint32_t f = 1; f < nInBucket; f++ ){
                dist = cv::norm(keypoints[l] - this->features[idxBelongToBucket[f]].uv.back());
                if( dist < this->params.min_px_dist ){
                    success = false;
                    break;
                }
            }
            if( success ){
                // Add new feature to VO object
                uint32_t newIdx = this->nFeature + 1;

                this->features[newIdx].id = this->new_feature_id; // unique id of the feature
                this->features[newIdx].frame_init = this->step; // frame step when the feature is created
                this->features[newIdx].uv.push_back(keypoints[l]); // uv point in pixel coordinates
                this->features[newIdx].life = 1; // the number of frames in where the feature is observed
                this->features[newIdx].bucket = cv::Point(i, j); // the location of bucket where the feature belong to
                this->features[newIdx].point.setZero(4,1); // 3-dim homogeneous point in the local coordinates
                this->features[newIdx].is_matched = false; // matched between both frame
                this->features[newIdx].is_wide = false; // verify whether features btw the initial and current are wide enough
                this->features[newIdx].is_2D_inliered = false; // belong to major (or meaningful) movement
                this->features[newIdx].is_3D_reconstructed = false; // triangulation completion
                this->features[newIdx].is_3D_init = false; // scale-compensated
                this->features[newIdx].point_init.setZero(4,1); // scale-compensated 3-dim homogeneous point in the global coordinates
                this->features[newIdx].point_var = this->params.var_point;

                this->new_feature_id++;
                this->nFeature++;

                // Update bucket
                // this->bucket.prob = conv2(obj.bucket.mass, obj.params.kernel, 'same');
                this->bucket.mass(i, j)++;
            }
        }
        
        filterSize = filterSize - 2;
    }
}

// haram
bool MVO::calculate_essential()
{
    Eigen::Matrix3d K = params.K;
    // below define should go in constructor and class member variable
    double focal = (K(1, 1) + K(2, 2)) / 2;
    cv::Point2d principle_point(K(1, 3), K(2, 3));
    Eigen::Matrix3d W;
    W << 0, -1, 0, 1, 0, 0, 0, 0, 1;
    //

    if (step == 1)
        return true;

    // Extract homogeneous 2D point which is matched with corresponding feature
    // TODO: define of find function

    std::vector<int> idx;
    for (uint32_t i = 0; i < features.size(); i++)
    {
        if (features.at(i).is_matched)
            idx.push_back(i);
    }

    uint32_t nInlier = idx.size();

    int point_count = 100;
    std::vector<cv::Point2d> points1(point_count);
    std::vector<cv::Point2d> points2(point_count);

    // initialize the points here ... */
    for (uint32_t i = 0; i < nInlier; i++)
    {
        points1[i] = features.at(idx.at(i)).uv.at(1); // second to latest
        points2[i] = features.at(idx.at(i)).uv.at(0); // latest
    }

    std::vector<bool> inlier;

    cv::Mat E;
    Eigen::Matrix3d E_, U, V;
    E = findEssentialMat(points1, points2, focal, principle_point, cv::RANSAC, 0.999, 1.0, inlier);
    cv2eigen(E, E_);
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(E_, Eigen::ComputeThinU | Eigen::ComputeThinV);
    U = svd.matrixU();
    V = svd.matrixV();

    if (U.determinant() < 0)
        U.block(0, 2, 3, 1) = -U.block(0, 2, 3, 1);
    if (V.determinant() < 0)
        V.block(0, 2, 3, 1) = -V.block(0, 2, 3, 1);
    R_vec.at(0) = U * W * V.transpose();
    R_vec.at(1) = U * W * V.transpose();
    R_vec.at(2) = U * W.transpose() * V;
    R_vec.at(3) = U * W.transpose() * V;
    t_vec.at(0) = U.block(0, 2, 3, 1);
    t_vec.at(1) = -U.block(0, 2, 3, 1);
    t_vec.at(2) = U.block(0, 2, 3, 1);
    t_vec.at(3) = -U.block(0, 2, 3, 1);

    uint32_t inlier_cnt = 0;
    for (uint32_t i = 0; i < inlier.size(); i++)
    {
        if (inlier.at(i))
        {
            features.at(i).is_2D_inliered = true;
            inlier_cnt++;
        }
    }
    nFeature2DInliered = inlier_cnt;

    if (nFeature2DInliered < params.thInlier)
    {
        std::cerr << " There are a few inliers matching features in 2D." << std::endl;
        return false;
    }
    else
    {
        return true;
    }
}

bool MVO::calculate_motion()
{
    if (step == 0)
        return true;

    Eigen::Matrix3d R, R_;
    Eigen::Vector3d t, t_;
    Eigen::Matrix4d T, Toc;
    Eigen::Vector4d Poc;
    
    bool success1 = this->findPoseFrom3DPoints(R_, t_);
    if (!success1){
        // Verity 4 solutions
        bool success2 = this->verify_solutions(R_vec, t_vec, R_, t_);
        
        if (!success2){
            std::cerr << "There are no meaningful R, t." << std::endl;
            return false;
        }

        // Update 3D points
        std::vector<bool> inlier, outlier;
        bool success3 = this->scale_propagation(R_ ,t_, R, t, inlier, outlier);

        if (!success3){
            std::cerr << "There are few inliers matching scale." << std::endl;
            return false;
        }

        this->update3DPoints(R, t, inlier, outlier, T, Toc, Poc); // overloading function
    } // if (!success1)
    else{
        this->verify_solutions(R_vec, t_vec, R_, t_);

        // Update 3D points
        std::vector<bool> inlier, outlier;
        bool success3 = this->scale_propagation(R_, t_, inlier, outlier);

        // Update 3D points
        this->update3DPoints(R, t, inlier, outlier, R_, t_, success3, T, Toc, Poc); // overloading function
    } // if (!success1)

    scale_initialized = true;

    if (nFeature3DReconstructed < params.thInlier){
        std::cerr << "There are few inliers reconstructed in 3D." << std::endl;
        return false;
    }
    else{
        // Save solution
        TRec.push_back(T);
        TocRec.push_back(Toc);
        PocRec.push_back(Poc);

        return true;
    }

    if (T.block(0, 3, 3, 1).norm() > 100){
        std::cout << "the current position: " <<  Poc(1) << ", " << Poc(2) << ", " << Poc(3) << std::endl;
        std::cerr << "Stop!" << std::endl;
    }
}

bool MVO::verify_solutions(std::vector<Eigen::Matrix3d>& R_vec, std::vector<Eigen::Vector3d>& t_vec, Eigen::Matrix3d& R, Eigen::Vector3d& t){
    return true;
}
bool MVO::scale_propagation(Eigen::Matrix3d& R_, Eigen::Vector3d& t_, Eigen::Matrix3d& R, Eigen::Vector3d& t, std::vector<bool>& inlier, std::vector<bool>& outlier){
    return true;
}
bool MVO::scale_propagation(Eigen::Matrix3d& R_, Eigen::Vector3d& t_, std::vector<bool>& inlier, std::vector<bool>& outlier){
    return true;
}
bool MVO::findPoseFrom3DPoints(Eigen::Matrix3d& R, Eigen::Vector3d& t){
    return true;
}
void MVO::contructDepth(const std::vector<cv::Point2d> x_prev, const std::vector<cv::Point2d> x_curr, const Eigen::Matrix3d R, const Eigen::Vector3d t, std::vector<Eigen::Vector4d>& X_prev, std::vector<Eigen::Vector4d>& X_curr, std::vector<double>& lambda_prev, std::vector<double>& lambda_curr){

}
void MVO::update3DPoints(const Eigen::Matrix3d& R, const Eigen::Vector3d& t,const std::vector<bool>& inlier, const std::vector<bool>& outlier,Eigen::Matrix4d& T, Eigen::Matrix4d& Toc, Eigen::Vector4d& Poc){

}
void MVO::update3DPoints(const Eigen::Matrix3d& R, const Eigen::Vector3d& t,const std::vector<bool>& inlier, const std::vector<bool>& outlier,const Eigen::Matrix3d& R_E, const Eigen::Vector3d& t_E, const bool& success_E,Eigen::Matrix4d& T, Eigen::Matrix4d& Toc, Eigen::Vector4d& Poc){

}