#include "core/MVO.hpp"
#include "core/utils.hpp"
#include "core/numerics.hpp"

bool MVO::calculate_motion()
{
    if (this->step == 0)
        return true;

    Eigen::Matrix3d R, R_;
    Eigen::Vector3d t, t_;
    Eigen::Matrix4d T, Toc;
    Eigen::Vector4d Poc;
	std::vector<bool> inlier, outlier;

    // bool success2 = this->verify_solutions(R_vec, t_vec, R_, t_);
    // std::cout << "==================" << std::endl;
    // std::cout << R_ << std::endl;
    // std::cout << "------------------" << std::endl;
    // std::cout << t_.transpose() << std::endl;
    // return true;
    
    bool success1 = this->findPoseFrom3DPoints(R, t, inlier, outlier);
    if (!success1){
        // Verity 4 solutions
        bool success2 = this->verify_solutions(R_vec, t_vec, R_, t_);
        
        if (!success2){
            std::cerr << "There are no meaningful R, t." << std::endl;
            return false;
        }

        // Update 3D points
        std::vector<bool> inlier, outlier;
        bool success3 = this->scale_propagation(R_ ,t_, inlier, outlier);

        if (!success3){
            std::cerr << "There are few inliers matching scale." << std::endl;
            return false;
        }

        this->update3DPoints(R_, t_, inlier, outlier, T, Toc, Poc); // overloading function
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
}

bool MVO::verify_solutions(std::vector<Eigen::Matrix3d>& R_vec, std::vector<Eigen::Vector3d>& t_vec, Eigen::Matrix3d& R, Eigen::Vector3d& t){
    
	bool success;

	// Extract homogeneous 2D point which is inliered with essential constraint
	std::vector<int> idx_2DInlier;
	for( int i = 0; i < this->nFeature; i++ ){
		if( this->features[i].is_2D_inliered )
			idx_2DInlier.push_back(i);
	}

	std::vector<cv::Point2f> uv_prev, uv_curr;
	std::vector<Eigen::Vector3d> x_prev, x_curr;

    int len;
	cv::Point2f uv_prev_i, uv_curr_i;
    for (uint32_t i = 0; i < idx_2DInlier.size(); i++){
		len = this->features[idx_2DInlier[i]].uv.size();
		uv_prev_i = this->features[idx_2DInlier[i]].uv[len-2];
		uv_curr_i = this->features[idx_2DInlier[i]].uv.back();
        uv_prev.push_back(uv_prev_i); // second to latest
        uv_curr.push_back(uv_curr_i); // latest

		Eigen::Vector3d x_prev_i, x_curr_i;
		x_prev_i << (uv_prev_i.x - this->params.cx)/this->params.fx, (uv_prev_i.y - this->params.cy)/this->params.fy, 1;
		x_curr_i << (uv_curr_i.x - this->params.cx)/this->params.fx, (uv_curr_i.y - this->params.cy)/this->params.fy, 1;;
		x_prev.push_back(x_prev_i);
		x_curr.push_back(x_curr_i);
    }

	// Find reasonable rotation and translational vector
	int max_num = 0;
	std::vector<bool> max_inlier;
	std::vector<Eigen::Vector4d> point_curr;
	for( uint32_t i = 0; i < R_vec.size(); i++ ){
		Eigen::Matrix3d R1 = R_vec[i];
		Eigen::Vector3d t1 = t_vec[i];
		
		std::vector<Eigen::Vector3d> X_prev, X_curr;
		std::vector<double> lambda_prev, lambda_curr;
		this->constructDepth(x_prev, x_curr, R1, t1, X_prev, X_curr,  lambda_prev, lambda_curr);

		std::vector<bool> inlier;
		int nInlier = 0;
		for( uint32_t i = 0; i < lambda_prev.size(); i++ ){
			if( lambda_curr[i] > 0 && lambda_prev[i] > 0 ){
				inlier.push_back(true);
				nInlier++;
			}else
				inlier.push_back(false);
		}

		if( nInlier > max_num ){
			max_num = nInlier;
			max_inlier = inlier;

            point_curr.clear();
			for( int i = 0; i < nInlier; i++ ){
				if( inlier[i] ){
					Eigen::Vector4d point_curr_i;
					point_curr_i << X_curr[i], 1;
					point_curr.push_back(point_curr_i);
				}
			}
			
			R = R1;
			t = t1;
		}
	}

	// Store 3D characteristics in features
	if( max_num < this->nFeature2DInliered*0.5 )
		success = false;
	else{
		for( int i = 0; i < this->nFeature2DInliered; i++ ){
			if( max_inlier[idx_2DInlier[i]] ){
				this->features[idx_2DInlier[i]].point = point_curr[i];
				this->features[idx_2DInlier[i]].is_3D_reconstructed = true;
			}
			this->nFeature3DReconstructed++;
		}
		success = true;
	}

	return success;
}
bool MVO::findPoseFrom3DPoints(Eigen::Matrix3d &R, Eigen::Vector3d &t, std::vector<bool>& inlier, std::vector<bool>& outlier)
{
    // Seek index of which feature is 3D reconstructed currently,
    // and 3D initialized previously
	bool flag;

    std::vector<int> idx;
    for (unsigned int i = 0; i < features.size(); i++){
        if (features[i].is_3D_init)
            idx.push_back(i);
    }
    unsigned int nPoint = idx.size();

    // Use RANSAC to find suitable scale
    if (nPoint > (uint32_t)params.thInlier)
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
            cv::solvePnPRansac(objectPoints, imagePoints, cv_K, cv::noArray(),
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
        R = Eigen::Matrix3d::Identity();
        t = Eigen::Vector3d::Zero();
        flag = false;
    }
    return flag;
}

void MVO::constructDepth(const std::vector<Eigen::Vector3d> x_prev, const std::vector<Eigen::Vector3d> x_curr, 
                        const Eigen::Matrix3d R, const Eigen::Vector3d t, 
                        std::vector<Eigen::Vector3d> &X_prev, std::vector<Eigen::Vector3d> &X_curr, 
                        std::vector<double> &lambda_prev, std::vector<double> &lambda_curr)
{
    const int nPoints = x_prev.size();
    Eigen::MatrixXd M_matrix(3*nPoints, nPoints+1);
    M_matrix.setZero();

    for (int i = 0; i < nPoints; i++){
        M_matrix.block(3*i,i,3,1) = skew(x_curr[i])*R*x_prev[i];
        M_matrix.block(3*i,nPoints,3,1) = skew(x_curr[i])*t;
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(M_matrix.transpose()*M_matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::MatrixXd V = svd.matrixV();

    for (int i = 0; i < nPoints; i++){
        lambda_prev.push_back( V(i,nPoints) / V(nPoints,nPoints) );
        X_prev.push_back( lambda_prev.back() * x_prev[i] );
        X_curr.push_back( R*X_prev.back() + t );
        lambda_curr.push_back( X_curr.back()(2) );
    }
}
	
// without PnP
void MVO::update3DPoints(const Eigen::Matrix3d &R, const Eigen::Vector3d &t, 
                        const std::vector<bool> &inlier, const std::vector<bool> &outlier, 
                        Eigen::Matrix4d &T, Eigen::Matrix4d &Toc, Eigen::Vector4d &Poc)
{
    // without PnP
    double scale = t.norm();
    
    // Seek index of which feature is 3D reconstructed currently
    // and 3D initialized previously
    std::vector<int> idx3D;
    for (unsigned int i = 0; i < features.size(); i++){
        if (this->features[i].is_3D_reconstructed)
            idx3D.push_back(i);
    }
    unsigned int nPoint = idx3D.size();

    if (this->scale_initialized){
        // Get expected 3D point by transforming the coordinates of the
        // observed 3d point
        std::vector<cv::Point3d> P1_exp;
        Eigen::Vector3d Eigen_point;
        for (unsigned int i = 0; i < nPoint; i++){
            // Get expected 3D point by transforming the coordinates of the observed 3d point
            Eigen_point = (this->features[idx3D[i]].point ).block(0,0,3,1);
            P1_exp.emplace_back(Eigen_point(0), Eigen_point(1), Eigen_point(2));
            if( outlier[i] )
                features[i].life = 0;
        }
        T.setIdentity();
        T.block(0,0,3,3) = R.transpose(); 
        T.block(0,3,3,1) = -R.transpose()*t; 
        Toc = this->TocRec.back() * T;
        Poc = Toc.block(0,3,4,1);

        // Update features which 3D points is initialized currently
        for (unsigned int i = 0; i < nPoint; i++){
            this->features[idx3D[i]].point.block(0,0,3,1) *= scale;

            // Initialize 3D point of each features when it is initialized for
            // the first time
            if ( this->features[idx3D[i]].is_3D_init && this->features[idx3D[i]].is_wide ){
                this->features[idx3D[i]].point_init = Toc * this->features[idx3D[i]].point;
                this->features[idx3D[i]].is_3D_init = true;
            }
        }
    }

    T.setIdentity();
    T.block(0,0,3,3) = R.transpose();
    T.block(0,3,3,1) = -R.transpose()*t;
    T(3,3) = 1;
    Toc = this->TocRec.back() * T;
    Poc = Toc.block(0,3,4,1);

    for( uint32_t i = 0; i < nPoint; i++ ){
        this->features[idx3D[i]].point.block(0,0,3,1) *= scale;

        if( this->features[idx3D[i]].is_3D_init && this->features[idx3D[i]].is_wide ){
            this->features[idx3D[i]].point_init = Toc * this->features[idx3D[i]].point;
            this->features[idx3D[i]].is_3D_init = true;
        }
    }
}
	
// with pnp
void MVO::update3DPoints(const Eigen::Matrix3d &R, const Eigen::Vector3d &t, 
                        const std::vector<bool> &inlier, const std::vector<bool> &outlier, 
                        const Eigen::Matrix3d &R_E, const Eigen::Vector3d &t_E, const bool &success_E, 
                        Eigen::Matrix4d &T, Eigen::Matrix4d &Toc, Eigen::Vector4d &Poc)
{
    // with PnP
    // Extract homogeneous 2D point which is inliered with essential constraint
    std::vector<int> idx3D;
    for (unsigned int i = 0; i < features.size(); i++){
        if (features[i].is_3D_init)
            idx3D.push_back(i);
    }
    unsigned int nPoint = idx3D.size();

    std::vector<Eigen::Vector3d> x_init_;
    for (unsigned int i = 0; i < nPoint; i++){
        x_init_.push_back(features[idx3D[i]].point_init.block(0,0,3,1));
    }

    Toc.setIdentity();
    Toc.block(0, 0, 3, 3) = R.transpose();
    Toc.block(0, 3, 3, 1) = -R.transpose() * t;
    T = TocRec.back().inverse() * Toc;
    Poc = Toc.block(0,3,4,1);

    Eigen::Matrix4d Tco;
    Tco.setIdentity();
    Tco.block(0, 0, 3, 3) = R;
    Tco.block(0, 3, 3, 1) = t;
    std::vector<Eigen::Vector4d> point_curr, point_prev;
    for (unsigned int i = 0; i < nPoint; i++){
        Eigen::Vector4d temp_vec;
        temp_vec << x_init_[i], 1;
        point_curr.push_back( Tco*temp_vec );
        point_prev.push_back( T*Tco*temp_vec );
    }

    // Update 3D points in local coordinates
    for (unsigned int i = 0; i < nPoint; i++){
        features[idx3D[i]].point = point_prev[i];
        features[idx3D[i]].is_3D_reconstructed = true;
    }

    // Initialize 3D points in global coordinates
    // Extract Homogeneous 2D point which is inliered with essential constraint
    std::vector<int> idx2D;
    for (unsigned int i = 0; i < features.size(); i++){
        if (features[i].is_2D_inliered)
            idx2D.push_back(i);
    }

    Eigen::Matrix3d Rinv;
    Eigen::Vector3d tinv;
    if(success_E){
        Rinv = R_E;
        tinv = t_E;
    }
    else{
        Rinv = T.block(0,0,3,3).transpose();
        tinv = -T.block(0,0,3,3).transpose()*T.block(0,3,3,1);
    }

    nPoint = idx2D.size();

    std::vector<cv::Point2f> uv_prev, uv_curr;
	std::vector<Eigen::Vector3d> x_prev, x_curr;

    int len;
	cv::Point2f uv_prev_i, uv_curr_i;
    for (uint32_t i = 0; i < idx2D.size(); i++){
		len = this->features[idx2D[i]].uv.size();
		uv_prev_i = this->features[idx2D[i]].uv[len-2];
		uv_curr_i = this->features[idx2D[i]].uv.back();
        uv_prev.push_back(uv_prev_i); // second to latest
        uv_curr.push_back(uv_curr_i); // latest

		Eigen::Vector3d x_prev_i, x_curr_i;
		x_prev_i << (uv_prev_i.x - this->params.cx)/this->params.fx, (uv_prev_i.y - this->params.cy)/this->params.fy, 1;
		x_curr_i << (uv_curr_i.x - this->params.cx)/this->params.fx, (uv_curr_i.y - this->params.cy)/this->params.fy, 1;;
		x_prev.push_back(x_prev_i);
		x_curr.push_back(x_curr_i);
    }
    
    std::vector<Eigen::Vector3d> X_prev, X_curr;
    std::vector<double> lambda_prev, lambda_curr;
    constructDepth(x_prev, x_curr, Rinv, tinv, X_prev, X_curr, lambda_prev, lambda_curr);

    point_prev.clear();
    point_curr.clear();
    Eigen::Vector4d temp_vec4_prev, temp_vec4_curr;
    temp_vec4_prev(3) = 1;
    temp_vec4_curr(3) = 1;
    int iterIdx2DInlier = 0;
    for (unsigned int i = 0; i < nPoint; i++){
        // 2d inliers
        if(lambda_prev[i] > 0 && lambda_curr[i] > 0){
            temp_vec4_prev.block(0,0,3,1) = X_prev[i];
            temp_vec4_curr.block(0,0,3,1) = X_curr[i];
            point_prev.push_back(temp_vec4_prev);
            point_curr.push_back(temp_vec4_curr);
            
            if( !features[idx2D[i]].is_3D_init && features[idx2D[i]].is_wide ){
                features[idx2D[i]].point = point_curr[iterIdx2DInlier];
                features[idx2D[i]].is_3D_reconstructed = true;

                features[idx2D[i]].point_init = Toc*point_curr[iterIdx2DInlier];
                features[idx2D[i]].is_3D_init = true;
            }
            iterIdx2DInlier ++;
        } // if(lambda_prev > 0 && lambda_curr > 0)
    } // for
    int Feature3Dconstructed = 0;
    for (unsigned int i = 0; i < features.size(); i++){
        if(features[i].is_3D_reconstructed)
            Feature3Dconstructed++;
    }
    this->nFeature3DReconstructed = Feature3Dconstructed;
}

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
    if (this->scale_initialized)
    {
        // Seek index of which feature is 3D reconstructed currently,
        // and 3D initialized previously
        std::vector<int> idx;
        for (unsigned int i = 0; i < this->features.size(); i++){
            if( this->features[i].is_3D_reconstructed && features[i].is_3D_init)
                idx.push_back(i);
        }
        nPoint = idx.size();

        // Use RANSAC to find suitable scale
        if ( nPoint > this->params.ransacCoef_scale_prop.minPtNum){
            // std::vector<Eigen::Vector3d> P1_ini;
            // std::vector<Eigen::Vector3d> P1_exp;
            std::vector<cv::Point3d> P1_ini;
            std::vector<cv::Point3d> P1_exp;
            Eigen::Vector3d Eigen_point;
            Eigen::Vector4d temp_point;
            for (unsigned int i = 0; i < nPoint; i++){
                temp_point = features[idx[i]].point;

                // Get initialized 3D point
                Eigen_point = ( this->TocRec[step-1].inverse() * this->features[idx[i]].point_init ).block(0,0,3,1);
                P1_ini.emplace_back(Eigen_point(0), Eigen_point(1), Eigen_point(2));
                
                // Get expected 3D point by transforming the coordinates of the observed 3d point
                Eigen_point = ( T_.inverse() * temp_point ).block(0,0,3,1);
                P1_exp.emplace_back(Eigen_point(0), Eigen_point(1), Eigen_point(2));
                
                // RANSAC weight
                this->params.ransacCoef_scale_prop.weight.push_back( std::atan( -temp_point(2)/10 + 3 ) + 3.141592 / 2 );
            }

            scale = ransac(P1_exp, P1_ini, this->params.ransacCoef_scale_prop, inlier, outlier);

            this->nFeatureInlier = inlier.size();

        }

        // Use the previous scale, if the scale cannot be found
        if (nPoint <= this->params.ransacCoef_scale_prop.minPtNum 
            || inlier.size() < (std::size_t)this->params.thInlier || scale == 0)
        {
            std::cerr << "warning('there are a few SCALE FACTOR INLIERS')" << std::endl;
            idx.clear();
            for (unsigned int i = 0; i < features.size(); i++){
                if (features[i].is_3D_init)
                    idx.push_back(i);
            }
            nPoint = idx.size();

            // Use RANSAC to find wuitable scale
            if (nPoint > (uint32_t)this->params.thInlier)
            {
                std::vector<cv::Point3d> objectPoints;
                std::vector<cv::Point2d> imagePoints;
                Eigen::Vector3d Eigen_point;
                for (unsigned int i = 0; i < nPoint; i++){
                    Eigen_point = ( T_.inverse() * this->features[idx[i]].point ).block(0,0,3,1);
                    objectPoints.emplace_back( Eigen_point(0), Eigen_point(1), Eigen_point(2) );
                    imagePoints.push_back( this->features[idx[i]].uv.back() ); // return last element of uv
                }
                std::vector<double> r_vec, t_vec;
                cv::Mat cv_K;
                cv::eigen2cv(this->params.K, cv_K);
                bool success = cv::solvePnPRansac( objectPoints, imagePoints, cv_K, cv::noArray(),
                                                    r_vec,  t_vec, false, 1e4, 
                                                    this->params.reprojError, 0.99, inlier);
                if (!success){
                    cv::solvePnPRansac( objectPoints, imagePoints,cv_K, cv::noArray(),
                                                    r_vec,  t_vec, false, 1e4, 
                                                    std::pow(this->params.reprojError,2), 0.99, inlier);
                }

                Eigen::Matrix4d temp_T;
                temp_T.setIdentity();
                temp_T.block(0,0,3,3) = skew(Eigen::Vector3d(r_vec.data())).exp();
                temp_T.block(0,3,3,1) = Eigen::Vector3d(t_vec.data());
                temp_T = this->TRec[this->step-1].inverse() * temp_T; 
                R = temp_T.block(0,0,3,3);
                t = temp_T.block(0,3,3,1);

                flag = success;
                outlier = inlier;
                outlier.flip();
                
            }
            else{
                inlier.clear();
                outlier.clear();

                scale = (this->TRec[this->step-1].block(0,0,3,1)).norm();

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
    if (!this->scale_initialized){
        cv::Point2d uv_curr;
        Eigen::Vector4d point_curr;
        std::vector<double> y_vals_road;
        for (int i = 0; i < nFeature; i++){
            uv_curr = features[i].uv.back(); //latest feature
            point_curr = features[i].point;
            if (uv_curr.y > params.imSize.height * 0.5 
                && uv_curr.y > params.imSize.height - 0.7 * uv_curr.x 
                && uv_curr.y > params.imSize.height + 0.7 * (uv_curr.x - params.imSize.width)
                && this->features[i].is_3D_reconstructed)
                y_vals_road.push_back(point_curr(1));
        }
        std::nth_element(y_vals_road.begin(), y_vals_road.begin() + y_vals_road.size()/2, y_vals_road.end());
        scale = params.vehicle_height / y_vals_road[y_vals_road.size()/2];

        t = scale * t;

        this->nFeatureInlier = this->nFeature3DReconstructed;
        inlier.clear();
        for (unsigned int i = 0; i < features.size(); i++)
            inlier.push_back( this->features[i].is_3D_reconstructed );
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