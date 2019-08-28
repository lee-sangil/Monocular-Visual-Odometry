#include "core/MVO.hpp"
#include "core/utils.hpp"
#include "core/numerics.hpp"

uint32_t Feature::new_feature_id = 0;

MVO::MVO(){
    this->step = 0;
    this->scale_initialized = false;
    this->cvClahe = cv::createCLAHE();

    // Variables
    this->nFeature = 0;
    this->nFeatureMatched = 0;
    this->nFeature2DInliered = 0;
    this->nFeature3DReconstructed = 0;
    this->nFeatureInlier = 0;

    // Initial position
    this->TRec.push_back(Eigen::Matrix4d::Identity());
    this->TocRec.push_back(Eigen::Matrix4d::Identity());
    this->PocRec.push_back(Eigen::Vector4d::Zero());
    this->PocRec[0](3) = 1;
}

MVO::MVO(std::string yaml):MVO(){

    /**************************************************************************
	 *  Read .yaml file
	 **************************************************************************/
    cv::FileStorage fSettings(yaml, cv::FileStorage::READ);
	if (!fSettings.isOpened()){
		std::cerr << "Failed to open: " << yaml << std::endl;
		return;
	}

	this->params.fx =			    fSettings["Camera.fx"];
	this->params.fy =			    fSettings["Camera.fy"];
	this->params.cx =			    fSettings["Camera.cx"];
	this->params.cy =			    fSettings["Camera.cy"];
	this->params.k1 =			    fSettings["Camera.k1"];
	this->params.k2 =			    fSettings["Camera.k2"];
	this->params.p1 =			    fSettings["Camera.p1"];
	this->params.p2 =			    fSettings["Camera.p2"];
	this->params.k3 =			    fSettings["Camera.k3"];
	this->params.imSize.width =	    fSettings["Camera.width"];
	this->params.imSize.height =	fSettings["Camera.height"];

    this->params.K << this->params.fx, 0, this->params.cx,
						0, this->params.fy, this->params.cy,
						0, 0, 1;

	this->params.radialDistortion.push_back(params.k1);
	this->params.radialDistortion.push_back(params.k2);
	this->params.radialDistortion.push_back(params.k3);
	this->params.tangentialDistortion.push_back(params.p1);
	this->params.tangentialDistortion.push_back(params.p2);

    this->params.thInlier =         fSettings["Feature.thInlier"];
    this->params.min_px_dist =      fSettings["Feature.min_px_dist"];

    // RANSAC parameter
    this->params.ransacCoef_scale_prop.iterMax =        fSettings["RANSAC.iterMax"];
    this->params.ransacCoef_scale_prop.minPtNum =       fSettings["RANSAC.nSample"];
    this->params.ransacCoef_scale_prop.thInlrRatio =    fSettings["RANSAC.thInlrRatio"];
    this->params.ransacCoef_scale_prop.thDist =         fSettings["RANSAC.thDist"]; // standard deviation
    this->params.ransacCoef_scale_prop.thDistOut =      fSettings["RANSAC.thDistOut"]; // three times of standard deviation
    
    // 3D reconstruction
    this->params.vehicle_height =   fSettings["Scale.height"]; // in meter
    this->params.reprojError =      fSettings["Scale.error"];
    this->params.initScale = 1;

    // Bucket
    this->bucket = Bucket();
    this->bucket.safety =           fSettings["Bucket.safety"];
	this->bucket.max_features =     fSettings["Feature.num"];
    int bucketGridRows =            fSettings["Bucket.rows"];
    int bucketGridCols =            fSettings["Bucket.cols"];

	this->bucket.grid = cv::Size(bucketGridRows,bucketGridCols);
	this->bucket.size = cv::Size(this->params.imSize.width/this->bucket.grid.width, this->params.imSize.height/this->bucket.grid.height);
	this->bucket.mass.setZero(this->bucket.grid.width,this->bucket.grid.height);
	this->bucket.prob.setZero(this->bucket.grid.width,this->bucket.grid.height);
    
}

void MVO::refresh(){
    this->nFeatureMatched = 0;
    this->nFeature2DInliered = 0;
    this->nFeature3DReconstructed = 0;
    this->nFeatureInlier = 0;

    for( int i = 0; i < this->nFeature; i++ ){
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
    if( this->params.applyCLAHE )
        cvClahe->apply(image, this->cur_image);
    else
        this->cur_image = image.clone();    
}

void MVO::run(const cv::Mat image){
    this->step++;

    this->set_image(image);
    this->refresh();

    if( this->extract_features() & // Extract and update features
	    this->calculate_essential())// & // RANSAC for calculating essential/fundamental matrix
//		this->calculate_motion() ) // Extract rotational and translational from fundamental matrix
        
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
    
    std::vector<Feature> _features = this->features;

    if( this->nFeature ){
        // Track the points
        std::vector<cv::Point2f> points;
        std::vector<bool> validity;
        this->klt_tracker(points, validity);
        
        for( int i = 0; i < this->nFeature; i++ ){
            if( validity[i] & (_features[i].life > 0) ){
                _features[i].life++;
                _features[i].uv.push_back(points[i]);
                _features[i].is_matched = true;
                
                if( cv::norm(_features[i].uv.front() - _features[i].uv.back()) > this->params.min_px_dist ){
                    _features[i].is_wide = true;
                }else{
                    _features[i].is_wide = false;
                }
                this->nFeatureMatched++;

            }else
                _features[i].life = 0;
        }

        if( this->nFeatureMatched < this->params.thInlier ){
            std::cerr << "There are a few FEATURE MATCHES" << std::endl;
            return false;
        }else{
            this->features = _features;
            return true;
        }
    }else
        return true;
}

void MVO::klt_tracker(std::vector<cv::Point2f>& fwd_pts, std::vector<bool>& validity){
    std::vector<cv::Point2f> pts;
    for( int i = 0; i < this->nFeature; i++ ){
        pts.push_back(this->features[i].uv.back());
    }
    
    // Forward-backward error evaluation
    std::vector<cv::Point2f> bwd_pts;
    std::vector<cv::Mat> prevPyr, currPyr;
    cv::Mat status, err;
    cv::buildOpticalFlowPyramid(this->prev_image, prevPyr, cv::Size(21,21), 4, true);
    cv::buildOpticalFlowPyramid(this->cur_image, currPyr, cv::Size(21,21), 4, true);
    cv::calcOpticalFlowPyrLK(prevPyr, currPyr, pts, fwd_pts, status, err);
    cv::calcOpticalFlowPyrLK(currPyr, prevPyr, fwd_pts, bwd_pts, status, err);
    
    // Calculate bi-directional error( = validity ): validity = ~border_invalid & error_valid
    for( int i = 0; i < this->nFeature; i++ ){
        bool border_invalid = (fwd_pts[i].x < 0) | (fwd_pts[i].x > this->params.imSize.width) | (fwd_pts[i].y < 0) | (fwd_pts[i].y > this->params.imSize.height);
        bool error_valid = cv::norm(pts[i] - bwd_pts[i]) < std::min( cv::norm(pts[i] - fwd_pts[i])/5.0, 2.0);
        validity.push_back(!border_invalid & error_valid);
    }
}

void MVO::delete_dead_features(){
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
    while( this->nFeature < this->bucket.max_features )
        this->add_feature();
}

void MVO::update_bucket(){
    this->bucket.mass.fill(0.0);
    for( int i = 0; i < this->nFeature; i++ ){
        cv::Point2f uv = this->features[i].uv.back();
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
    int i, j;
    lsi::idx_randselect(this->bucket.prob, i, j);
    cv::Rect roi = cv::Rect(i*bkSize.width+1, j*bkSize.height+1, bkSize.width, bkSize.height);
    
    roi.x = std::max(bkSafety, (uint32_t)roi.x);
    roi.y = std::max(bkSafety, (uint32_t)roi.y);
    roi.width = std::min(this->params.imSize.width-bkSafety, (uint32_t)roi.x + roi.width)-roi.x;
    roi.height = std::min(this->params.imSize.height-bkSafety, (uint32_t)roi.y+roi.height)-roi.y;

    // Seek index of which feature is extracted specific bucket
    std::vector<uint32_t> idxBelongToBucket;
    for( int l = 0; l < this->nFeature; l++ ){
        for( int ii = std::max(i-1,0); ii < std::min(i+1,this->bucket.grid.width); ii++){
            for( int jj = std::max(j-1,0); jj < std::min(j+1,this->bucket.grid.height); jj++){
                if( (this->features[l].bucket.x == ii) & (this->features[l].bucket.y == jj)){
                    idxBelongToBucket.push_back(l);
                }
            }
        }
    }
    uint32_t nInBucket = idxBelongToBucket.size();
    
    // Try to find a seperate feature
    double filterSize = 5.0;

    while( true ){

        if( filterSize < 3.0 )
            return;

        cv::Mat crop_image = this->cur_image(roi);
        std::vector<cv::Point2f> keypoints;
        cv::goodFeaturesToTrack(crop_image, keypoints, 1000, 0.01, 2.0, cv::Mat(), 3, true);
        
        if( keypoints.size() == 0 )
            return;
        else{
            for( uint32_t l = 0; l < keypoints.size(); l++ ){
                keypoints[l].x = keypoints[l].x + roi.x - 1;
                keypoints[l].y = keypoints[l].y + roi.y - 1;
            }
        }

        bool success;
        double dist;
        double minDist;
        double maxMinDist = 0;
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
            newFeature.uv.push_back(bestKeypoint); // uv point in pixel coordinates
            newFeature.life = 1; // the number of frames in where the feature is observed
            newFeature.bucket = cv::Point(i, j); // the location of bucket where the feature belong to
            newFeature.point.setZero(4,1); // 3-dim homogeneous point in the local coordinates
            newFeature.is_matched = false; // matched between both frame
            newFeature.is_wide = false; // verify whether features btw the initial and current are wide enough
            newFeature.is_2D_inliered = false; // belong to major (or meaningful) movement
            newFeature.is_3D_reconstructed = false; // triangulation completion
            newFeature.is_3D_init = false; // scale-compensated
            newFeature.point_init.setZero(4,1); // scale-compensated 3-dim homogeneous point in the global coordinates
            newFeature.point_var = this->params.var_point;

            this->features.push_back(newFeature);
            this->nFeature++;

            Feature::new_feature_id++;

            // Update bucket
            this->bucket.mass(i, j)++;

            cv::Mat bucketMass, bucketMassBlur;
            cv::eigen2cv(this->bucket.mass, bucketMass);
            cv::GaussianBlur(bucketMass, bucketMassBlur, cv::Size(21,21), 3.0);
            cv::cv2eigen(bucketMassBlur, this->bucket.prob);

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
    //

    if (this->step == 1)
        return true;

    // Extract homogeneous 2D point which is matched with corresponding feature
    // TODO: define of find function

    std::vector<int> idx;
    for (uint32_t i = 0; i < this->features.size(); i++)
    {
        if (this->features[i].is_matched)
            idx.push_back(i);
    }

    uint32_t nInlier = idx.size();

    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;

    // initialize the points here ... */
    int len;
    for (uint32_t i = 0; i < nInlier; i++)
    {
        len = this->features[idx[i]].uv.size();
        points1.push_back(this->features[idx[i]].uv[len - 2]); // second to latest
        points2.push_back(this->features[idx[i]].uv.back());                             // latest
    }

    cv::Mat inlier_mat;
    cv::Mat E;
    Eigen::Matrix3d E_, U, V;
    E = cv::findEssentialMat(points1, points2, focal, principle_point, cv::RANSAC, 0.999, 0.1, inlier_mat);
    
    cv::cv2eigen(E, E_);
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(E_, Eigen::ComputeThinU | Eigen::ComputeThinV);
    U = svd.matrixU();
    V = svd.matrixV();

    if (U.determinant() < 0)
        U.block(0, 2, 3, 1) = -U.block(0, 2, 3, 1);
    if (V.determinant() < 0)
        V.block(0, 2, 3, 1) = -V.block(0, 2, 3, 1);

    this->R_vec.clear();
    this->t_vec.clear();
    this->R_vec.push_back(U * W * V.transpose());
    this->R_vec.push_back(U * W * V.transpose());
    this->R_vec.push_back(U * W.transpose() * V);
    this->R_vec.push_back(U * W.transpose() * V);
    this->t_vec.push_back(U.block(0, 2, 3, 1));
    this->t_vec.push_back(-U.block(0, 2, 3, 1));
    this->t_vec.push_back(U.block(0, 2, 3, 1));
    this->t_vec.push_back(-U.block(0, 2, 3, 1));

    uint32_t inlier_cnt = 0;
    for (int i = 0; i < inlier_mat.rows; i++)
    {
        if (inlier_mat.at<char>(i))
        {
            this->features[i].is_2D_inliered = true;
            inlier_cnt++;
        }
    }
    this->nFeature2DInliered = inlier_cnt;

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

bool MVO::calculate_motion()
{
    if (this->step == 1)
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
        // std::cout << "the current position: " <<  Poc(1) << ", " << Poc(2) << ", " << Poc(3) << std::endl;
        std::cerr << "Stop!" << std::endl;
    }
}

bool MVO::verify_solutions(std::vector<Eigen::Matrix3d>& R_vec, std::vector<Eigen::Vector3d>& t_vec, Eigen::Matrix3d& R, Eigen::Vector3d& t){
    return true;
}
bool MVO::findPoseFrom3DPoints(Eigen::Matrix3d &R, Eigen::Vector3d &t, std::vector<bool>& inlier, std::vector<bool>& outlier)
{
    // Seek index of which feature is 3D reconstructed currently,
    // and 3D initialized previously
    std::vector<int> idx;
    for (unsigned int i = 0; i < features.size(); i++){
        if (features[i].is_3D_init)
            idx.push_back(i);
    }
    unsigned int nPoint = idx.size();

    // Use RANSAC to find suitable scale
    if (nPoint > params.thInlier)
    {
        std::vector<cv::Point3d> objectPoints;
        std::vector<cv::Point2d> imagePoints;
        Eigen::Vector3d Eigen_point;
        for (unsigned int i = 0; i < nPoint; i++)
        {
            Eigen_point = (features[idx[i]].point_init).block(0, 0, 3, 1);
            objectPoints.emplace_back(Eigen_point(0), Eigen_point(1), Eigen_point(2));
            imagePoints.push_back(features[idx[i]].uv.back()); // return last element of uv
        }
        std::vector<double> r_vec, t_vec;
        cv::Mat cv_K;
        cv::eigen2cv(params.K, cv_K);
        bool success = cv::solvePnPRansac(objectPoints, imagePoints, cv_K, cv::noArray(),
                                          r_vec, t_vec, false, 1e4,
                                          params.reprojError, 0.99, inlier, cv::SOLVEPNP_AP3P);
        if (!success)
        {
            bool success = cv::solvePnPRansac(objectPoints, imagePoints, cv_K, cv::noArray(),
                                              r_vec, t_vec, false, 1e4,
                                              std::pow(params.reprojError, 2), 0.99, inlier, cv::SOLVEPNP_AP3P);
        }

        R = skew(Eigen::Vector3d(r_vec.data())).exp();
        t = Eigen::Vector3d(t_vec.data());
        outlier = inlier;
        outlier.flip();
        flag = success;
    }
    else
    {
        inlier.clear();
        outlier.clear();
        R = Eigen::Matrix3;
        t = ;
        flag = false;
    }
    return flag;
}
void MVO::constructDepth(const std::vector<cv::Point2f> x_prev, const std::vector<cv::Point2f> x_curr, const Eigen::Matrix3d R, const Eigen::Vector3d t, std::vector<Eigen::Vector4d>& X_prev, std::vector<Eigen::Vector4d>& X_curr, std::vector<double>& lambda_prev, std::vector<double>& lambda_curr){

}
void MVO::update3DPoints(const Eigen::Matrix3d& R, const Eigen::Vector3d& t,const std::vector<bool>& inlier, const std::vector<bool>& outlier,Eigen::Matrix4d& T, Eigen::Matrix4d& Toc, Eigen::Vector4d& Poc){

}
void MVO::update3DPoints(const Eigen::Matrix3d& R, const Eigen::Vector3d& t,const std::vector<bool>& inlier, const std::vector<bool>& outlier,const Eigen::Matrix3d& R_E, const Eigen::Vector3d& t_E, const bool& success_E,Eigen::Matrix4d& T, Eigen::Matrix4d& Toc, Eigen::Vector4d& Poc){

}
	 
bool MVO::scale_propagation(const Eigen::Matrix3d &R_, const Eigen::Vector3d &t_, Eigen::Matrix3d &R, Eigen::Vector3d &t, std::vector<bool> &inlier, std::vector<bool> &outlier)
{
    inlier.clear();
    outlier.clear();

    double scale = 0;
    bool flag;
    Eigen::Matrix4d T_;
    T_.setIdentity();
    T_.block(0,0,3,3) = R_;
    T_.block(0,3,3,1) = t_;

    unsigned int nPoint;
    if (scale_initialized)
    {
        // Seek index of which feature is 3D reconstructed currently,
        // and 3D initialized previously
        std::vector<int> idx;
        for (unsigned int i = 0; i < features.size(); i++){
            if( features[i].is_3D_reconstructed && features[i].is_3D_init)
                idx.push_back(i);
        }
        nPoint = idx.size();

        // Use RANSAC to find suitable scale
        if ( nPoint > params.ransacCoef_scale_prop.minPtNum){
            // std::vector<Eigen::Vector3d> P1_ini;
            // std::vector<Eigen::Vector3d> P1_exp;
            std::vector<cv::Point3d> P1_ini;
            std::vector<cv::Point3d> P1_exp;
            Eigen::Vector3d Eigen_point;
            Eigen::Vector4d temp_point;
            for (unsigned int i = 0; i < nPoint; i++){
                temp_point = features[idx[i]].point;

                // Get initialized 3D point
                Eigen_point = ( TocRec[step-1].inverse() * features[idx[i]].point_init ).block(0,0,3,1);
                P1_ini.emplace_back(Eigen_point(0), Eigen_point(1), Eigen_point(2));
                
                // Get expected 3D point by transforming the coordinates of the observed 3d point
                Eigen_point = ( T_.inverse() * temp_point ).block(0,0,3,1);
                P1_exp.emplace_back(Eigen_point(0), Eigen_point(1), Eigen_point(2));
                
                // RANSAC weight
                params.ransacCoef_scale_prop.weight.push_back( std::atan( -temp_point(2)/10 + 3 ) + 3.141592 / 2 );
            }

            scale = ransac(P1_exp, P1_ini, params.ransacCoef_scale_prop, inlier, outlier);

            nFeatureInlier = inlier.size();

        }

        // Use the previous scale, if the scale cannot be found
        if (nPoint <= params.ransacCoef_scale_prop.minPtNum 
            || inlier.size() < (std::size_t)params.thInlier || scale == 0)
        {
            std::cerr << "warning('there are a few SCALE FACTOR INLIERS')" << std::endl;
            idx.clear();
            for (unsigned int i = 0; i < features.size(); i++){
                if (features[i].is_3D_init)
                    idx.push_back(i);
            }
            nPoint = idx.size();

            // Use RANSAC to find wuitable scale
            if (nPoint > (uint32_t)params.thInlier)
            {
                std::vector<cv::Point3d> objectPoints;
                std::vector<cv::Point2d> imagePoints;
                Eigen::Vector3d Eigen_point;
                for (unsigned int i = 0; i < nPoint; i++){
                    Eigen_point = ( T_.inverse() * features[idx[i]].point ).block(0,0,3,1);
                    objectPoints.emplace_back( Eigen_point(0), Eigen_point(1), Eigen_point(2) );
                    imagePoints.push_back( features[idx[i]].uv.back() ); // return last element of uv
                }
                std::vector<double> r_vec, t_vec;
                cv::Mat cv_K;
                cv::eigen2cv(params.K, cv_K);
                bool success = cv::solvePnPRansac( objectPoints, imagePoints, cv_K, cv::noArray(),
                                                    r_vec,  t_vec, false, 1e4, 
                                                    params.reprojError, 0.99, inlier);
                if (!success){
                    bool success = cv::solvePnPRansac( objectPoints, imagePoints,cv_K, cv::noArray(),
                                                    r_vec,  t_vec, false, 1e4, 
                                                    std::pow(params.reprojError,2), 0.99, inlier);
                }

                Eigen::Matrix4d temp_T;
                temp_T.setIdentity();
                temp_T.block(0,0,3,3) = skew(Eigen::Vector3d(r_vec.data())).exp();
                temp_T.block(0,3,3,1) = Eigen::Vector3d(t_vec.data());
                temp_T = TRec[step-1].inverse() * temp_T; 
                R = temp_T.block(0,0,3,3);
                t = temp_T.block(0,3,3,1);

                flag = success;
                outlier = inlier;
                outlier.flip();
                
            }
            else{
                inlier.clear();
                outlier.clear();

                scale = (TRec[step-1].block(0,0,3,1)).norm();

                // Update scale
                t = scale * t_;
                flag = false;
            }
        }
        else{
            // Update scale
            t = scale * t_;
            flag = true;

        }

    }

    // Initialization
    // initialze scale, in the case of the first time
    if (!scale_initialized){
        cv::Point2d uv_curr;
        Eigen::Vector4d point_curr;
        std::vector<double> y_vals_road;
        for (int i = 0; i < nFeature; i++){
            uv_curr = features[i].uv.back(); //latest feature
            point_curr = features[i].point;
            if (uv_curr.y > params.imSize.height * 0.5 
                && uv_curr.y > params.imSize.height - 0.7 * uv_curr.x 
                && uv_curr.y > params.imSize.height + 0.7 * uv_curr.x - params.imSize.width 
                && !std::isnan(point_curr(2)))
                y_vals_road.push_back(point_curr(1));
        }
        std::nth_element(y_vals_road.begin(), y_vals_road.begin() + y_vals_road.size()/2, y_vals_road.end());
        scale = params.vehicle_height / y_vals_road[y_vals_road.size()/2];

        t = scale * t_;

        nFeatureInlier = nFeature3DReconstructed;
        inlier.clear();
        for (unsigned int i = 0; i < features.size(); i++)
            inlier.push_back( features[i].is_3D_reconstructed );
        flag = true;
    }

    return flag;
}

// OVERLOADING
bool MVO::scale_propagation(Eigen::Matrix3d &R, Eigen::Vector3d &t, std::vector<bool> &inlier, std::vector<bool> &outlier)
{       
    inlier.clear();
    outlier.clear();

    double scale = 0;
    bool flag;
    Eigen::Matrix4d T_;
    T_.setIdentity();
    T_.block(0,0,3,3) = R;
    T_.block(0,3,3,1) = t;

    unsigned int nPoint;
    if (scale_initialized)
    {
        // Seek index of which feature is 3D reconstructed currently,
        // and 3D initialized previously
        std::vector<int> idx;
        for (unsigned int i = 0; i < features.size(); i++){
            if( features[i].is_3D_reconstructed && features[i].is_3D_init)
                idx.push_back(i);
        }
        nPoint = idx.size();

        // Use RANSAC to find suitable scale
        if ( nPoint > params.ransacCoef_scale_prop.minPtNum){
            // std::vector<Eigen::Vector3d> P1_ini;
            // std::vector<Eigen::Vector3d> P1_exp;
            std::vector<cv::Point3d> P1_ini;
            std::vector<cv::Point3d> P1_exp;
            Eigen::Vector3d Eigen_point;
            Eigen::Vector4d temp_point;
            for (unsigned int i = 0; i < nPoint; i++){
                temp_point = features[idx[i]].point;

                // Get initialized 3D point
                Eigen_point = ( TocRec[step-1].inverse() * features[idx[i]].point_init ).block(0,0,3,1);
                P1_ini.emplace_back(Eigen_point(0), Eigen_point(1), Eigen_point(2));
                
                // Get expected 3D point by transforming the coordinates of the observed 3d point
                Eigen_point = ( T_.inverse() * temp_point ).block(0,0,3,1);
                P1_exp.emplace_back(Eigen_point(0), Eigen_point(1), Eigen_point(2));
                
                // RANSAC weight
                params.ransacCoef_scale_prop.weight.push_back( std::atan( -temp_point(2)/10 + 3 ) + 3.141592 / 2 );
            }

            scale = ransac(P1_exp, P1_ini, params.ransacCoef_scale_prop, inlier, outlier);

            nFeatureInlier = inlier.size();

        }

        // Use the previous scale, if the scale cannot be found
        if (nPoint <= params.ransacCoef_scale_prop.minPtNum 
            || inlier.size() < (std::size_t)params.thInlier || scale == 0)
        {
            std::cerr << "warning('there are a few SCALE FACTOR INLIERS')" << std::endl;
            idx.clear();
            for (unsigned int i = 0; i < features.size(); i++){
                if (features[i].is_3D_init)
                    idx.push_back(i);
            }
            nPoint = idx.size();

            // Use RANSAC to find wuitable scale
            if (nPoint > (uint32_t)params.thInlier)
            {
                std::vector<cv::Point3d> objectPoints;
                std::vector<cv::Point2d> imagePoints;
                Eigen::Vector3d Eigen_point;
                for (unsigned int i = 0; i < nPoint; i++){
                    Eigen_point = ( T_.inverse() * features[idx[i]].point ).block(0,0,3,1);
                    objectPoints.emplace_back( Eigen_point(0), Eigen_point(1), Eigen_point(2) );
                    imagePoints.push_back( features[idx[i]].uv.back() ); // return last element of uv
                }
                std::vector<double> r_vec, t_vec;
                cv::Mat cv_K;
                cv::eigen2cv(params.K, cv_K);
                bool success = cv::solvePnPRansac( objectPoints, imagePoints, cv_K, cv::noArray(),
                                                    r_vec,  t_vec, false, 1e4, 
                                                    params.reprojError, 0.99, inlier);
                if (!success){
                    bool success = cv::solvePnPRansac( objectPoints, imagePoints,cv_K, cv::noArray(),
                                                    r_vec,  t_vec, false, 1e4, 
                                                    std::pow(params.reprojError,2), 0.99, inlier);
                }

                Eigen::Matrix4d temp_T;
                temp_T.setIdentity();
                temp_T.block(0,0,3,3) = skew(Eigen::Vector3d(r_vec.data())).exp();
                temp_T.block(0,3,3,1) = Eigen::Vector3d(t_vec.data());
                temp_T = TRec[step-1].inverse() * temp_T; 
                R = temp_T.block(0,0,3,3);
                t = temp_T.block(0,3,3,1);

                flag = success;
                outlier = inlier;
                outlier.flip();
                
            }
            else{
                inlier.clear();
                outlier.clear();

                scale = (TRec[step-1].block(0,0,3,1)).norm();

                // Update scale
                t = scale * t;
                flag = false;
            }
        }
        else{
            // Update scale
            t = scale * t;
            flag = true;

        }

    }

    // Initialization
    // initialze scale, in the case of the first time
    if (!scale_initialized){
        cv::Point2d uv_curr;
        Eigen::Vector4d point_curr;
        std::vector<double> y_vals_road;
        for (int i = 0; i < nFeature; i++){
            uv_curr = features[i].uv.back(); //latest feature
            point_curr = features[i].point;
            if (uv_curr.y > params.imSize.height * 0.5 
                && uv_curr.y > params.imSize.height - 0.7 * uv_curr.x 
                && uv_curr.y > params.imSize.height + 0.7 * uv_curr.x - params.imSize.width 
                && !std::isnan(point_curr(2)))
                y_vals_road.push_back(point_curr(1));
        }
        std::nth_element(y_vals_road.begin(), y_vals_road.begin() + y_vals_road.size()/2, y_vals_road.end());
        scale = params.vehicle_height / y_vals_road[y_vals_road.size()/2];

        t = scale * t;

        nFeatureInlier = nFeature3DReconstructed;
        inlier.clear();
        for (unsigned int i = 0; i < features.size(); i++)
            inlier.push_back( features[i].is_3D_reconstructed );
        flag = true;
    }
    return flag;
}

double MVO::ransac(const std::vector<cv::Point3d> &x, const std::vector<cv::Point3d> &y,
                   MVO::RansacCoef ransacCoef,
                   std::vector<bool> &inlier, std::vector<bool> &outlier)
{
    unsigned int ptNum = x.size();

    std::vector<int> sampleIdx;
    std::vector<cv::Point3d> x_sample, y_sample;

    int iterNUM = 1e8;
    std::size_t max_inlier = 0;

    int it = 0;

    while (it < std::min(ransacCoef.iterMax, iterNUM))
    {
        // 1. fit using random points
        if (ransacCoef.weight.size() > 0)
        {
            sampleIdx = randweightedpick(ransacCoef.weight, ransacCoef.minPtNum);
        }
        else
        {
            sampleIdx = randperm(ptNum, ransacCoef.minPtNum);
        }
        int tempIdx = 0;
        x_sample.clear();
        y_sample.clear();
        for (unsigned int i = 0; i < sampleIdx.size(); i++)
        {
            tempIdx = sampleIdx[i];
            x_sample.push_back(x[tempIdx]);
            y_sample.push_back(y[tempIdx]);
        }
        double f1 = calculate_scale(x_sample, y_sample);
        std::vector<double> dist1 = calculate_scale_error(f1, x, y);
        std::vector<bool> in1;
        for (unsigned int i = 0; i < dist1.size(); i++)
        {
                in1.push_back( dist1[i] < ransacCoef.thDist );
        }

        if (in1.size() > max_inlier)
        {
            max_inlier = in1.size();
            inlier = in1;
            double InlrRatio = (double)max_inlier / (double)ptNum + 1e-16;
            iterNUM = static_cast<int>(std::floor(std::log(1 - ransacCoef.thInlrRatio) / std::log(1 - std::pow(InlrRatio, ransacCoef.minPtNum))));
        }
        it++;
    }

    if (inlier.size() == 0)
    {
        inlier.clear();
        outlier.clear();
        return 0;
    }
    else
    {
        x_sample.clear();
        y_sample.clear();
        for (unsigned int i = 0; i < inlier.size(); i++){
            if (inlier[i]){
                x_sample.push_back(x[i]);
                y_sample.push_back(y[i]);
            }
        }
        double f1 = calculate_scale(x_sample, y_sample);

        std::vector<double> dist = calculate_scale_error(f1, x, y);

        inlier.clear();
        outlier.clear();
        for (unsigned int i = 0; i < dist.size(); i++){
                inlier.push_back(dist[i] < ransacCoef.thDist);
                outlier.push_back(dist[i] > ransacCoef.thDistOut);
        }

        return f1;
    }
}

std::vector<int> MVO::randperm(unsigned int ptNum, int minPtNum)
{
    std::vector<int> result;
    for (unsigned int i = 0; i < ptNum; i++)
        result.push_back(i);
    std::random_shuffle(result.begin(), result.begin() + minPtNum);
    return result;
}

std::vector<int> MVO::randweightedpick(const std::vector<double> &h, int n /*=1*/)
{
    /*
    RANDWEIGHTEDPICK: Randomly pick n from size(h)>=n elements,
    biased with linear weights as given in h, without replacement.
    Works with infinity and zero weighting entries,
    but always picks them sequentially in this case.

    Syntax: Y=randweightedpick(h,n)

    Example:

    randweightedpick([2,0,5,Inf,3,Inf,0],7)

    returns [4,6,...,2,7] all the time as [4,6] are infinities and [2,7] are
    zeros; in between, the results will be a permutation of [1,3,5]:

    [...,3,5,1,...] 30.0% of the time {5/(2+3+5) * 3/(2+3)}
    [...,3,1,5,...] 20.0% of the time {5/(2+3+5) * 2/(2+3)}
    [...,5,3,1,...] 21.4% of the time {3/(2+3+5) * 5/(2+5)}
    [...,1,3,5,...] 12.5% of the time {2/(2+3+5) * 5/(3+5)}
    [...,5,1,3,...] 8.6% of the time {3/(2+3+5) * 2/(2+5)}
    [...,1,5,3,...] 7.5% of the time {2/(2+3+5) * 3/(3+5)}

    Y returns the vector of indices corresponding to the weights in h.

    Author: Adam W. Gripton (a.gripton -AT- hw.ac.uk) 2012/03/21
    */

    int u = h.size();
    int s_under;
    double sum, rand_num;
    std::vector<double> H = h;
    //std::list<double> H;
    // std::copy(h.cbegin(), h.cend(), std::back_inserter(H));
    std::vector<double> Hs, Hsc;
    std::vector<int> result;

    n = std::min(std::max(1, n), u);
    std::vector<int> HI(u, 0);          // vector with #u ints.
    std::iota(HI.begin(), HI.end(), 1); // Fill with 1, ..., u.

    // std::transform(H.begin(), H.end(), Hs.begin(),
    // [](double elem) -> double{ return elem/});

    for (int i = 0; i < n; i++)
    {
        // initial variables
        Hs.clear();
        Hsc.clear();
        // random weight
        sum = std::accumulate(H.begin(), H.end(), 0);
        std::transform(H.begin(), H.end(), std::back_inserter(Hs),
                       std::bind(std::multiplies<double>(), std::placeholders::_1, 1 / sum)); // divdie elements in H with the value of sum
        std::partial_sum(Hs.begin(), Hs.end(), std::back_inserter(Hsc), std::plus<double>()); // cummulative sum.

        // generate rand num btw 0 to 1
        rand_num = ((double)rand() / (RAND_MAX));
        // increase s_under if Hsc is lower than rand_num
        s_under = std::count_if(Hsc.begin(), Hsc.end(), [&](double elem) { return elem < rand_num; });

        result.push_back(HI[s_under]);
        H.erase(H.begin() + s_under);
        HI.erase(HI.begin() + s_under);
    }
    // this function may have issue.
    // about vector dynamic allocation.
    // about header include.
    return result;
}

double MVO::calculate_scale(const std::vector<cv::Point3d> &pt1, const std::vector<cv::Point3d> &pt2)
{
    double sum = 0;
    for (unsigned int i = 0; i < pt1.size(); i++)
    {
        sum += (pt1[i].x * pt2[i].x + pt1[i].y * pt2[i].y + pt1[i].z * pt2[i].z) / (pt1[i].x * pt1[i].x + pt1[i].y * pt1[i].y + pt1[i].z * pt1[i].z + 1e-10);
    }
    return sum / pt1.size();
}

std::vector<double> MVO::calculate_scale_error(double scale, const std::vector<cv::Point3d> &pt1, const std::vector<cv::Point3d> &pt2)
{
    std::vector<double> dist;
    for (unsigned int i = 0; i < pt1.size(); i++)
        dist.push_back(cv::norm(pt2[i] - scale * pt1[i]));
    return dist;
}
