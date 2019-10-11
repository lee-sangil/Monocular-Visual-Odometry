#include "core/MVO.hpp"
#include "core/utils.hpp"
#include "core/numerics.hpp"
#include "core/time.hpp"

bool MVO::calculate_motion()
{
    if (this->step == 0)
        return true;

    /**************************************************
     * Solve two-fold ambiguity
     **************************************************/
    Eigen::Matrix3d R_;
    Eigen::Vector3d t_;

    /**** exact criteria ****/
    this->verify_solutions(this->R_vec, this->t_vec, R_, t_);
    
    /**** simple criteria ****/
    // Eigen::Matrix3d Identity = Eigen::Matrix3d::Identity();
    // if( (this->R_vec[0] - Identity).norm() < (this->R_vec[2] - Identity).norm() ) R_ = this->R_vec[0];
    // else R_ = this->R_vec[2];

    // if( this->t_vec[0](2) > 0) t_ = this->t_vec[1];
    // else t_ = this->t_vec[0];
    // this->verify_solutions(R_, t_);

    std::cerr << "# Verify unique pose: " << lsi::toc() << std::endl;
    std::cerr << "R diag: " << R_.diagonal().transpose() << std::endl;
    std::cerr << "t: " << t_.transpose() << std::endl;

    /**************************************************
     * Mapping
     **************************************************/
    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    Eigen::Matrix4d T, Toc;
    Eigen::Vector4d Poc;
    std::vector<bool> inlier, outlier;
	std::vector<int> idxInlier, idxOutlier;

    /**** no mapping and scaling ****/
    // T.setIdentity();
    // T.block(0,0,3,3) = R_.transpose(); 
    // T.block(0,3,3,1) = -R_.transpose()*t_;
    // Toc = this->TocRec.back() * T;
    // Poc = Toc.block(0,3,4,1);

    // TRec.push_back(T);
    // TocRec.push_back(Toc);
    // PocRec.push_back(Poc);
    // return true;

    /**** mapping without scaling ****/
    // this->update3DPoints(R_, t_, inlier, outlier, T, Toc, Poc);
    // return true;

    /**** mapping and scaling with essential 3d reconstruction only ****/
    // this->scale_propagation(R_ ,t_, inlier, outlier);
    // this->update3DPoints(R_, t_, inlier, outlier, T, Toc, Poc); // overloading function

    /**** mapping and scaling with PnP only ****/
    // if( this->scale_initialized == true ){
    //     this->findPoseFrom3DPoints(R, t, idxInlier, idxOutlier);
    // }else{
    //     R = R_;
    //     t = t_;
    // }
    // this->update3DPoints(R, t, inlier, outlier, R_, t_, false, T, Toc, Poc); // overloading function

    /**** use both PnP and essential 3d reconstruction - original****/
    if (this->findPoseFrom3DPoints(R, t, idxInlier, idxOutlier)){
        std::cerr << "# Find pose from PnP: " << lsi::toc() << std::endl;

        // Update 3D points
        bool success = this->scale_propagation(R_, t_, inlier, outlier);
    
        // Update 3D points
        this->update3DPoints(R, t, inlier, outlier, R_, t_, success, T, Toc, Poc); // overloading function
        std::cout << "Update 3D Points with PnP, ";
    }else{
        std::cerr << "# Find pose from PnP: " << lsi::toc() << std::endl;

        // Update 3D points
        bool success = this->scale_propagation(R_ ,t_, inlier, outlier);
    
        if (!success){
            std::cerr << "There are few inliers matching scale." << std::endl;
            return false;
        }

        this->update3DPoints(R_, t_, inlier, outlier, T, Toc, Poc); // overloading function
        std::cout << "Update 3D Points with Essential Constraint, ";
    }

    /**** use both PnP and essential 3d reconstruction - modified****/
    // if( this->scale_initialized == true ){
    //     this->findPoseFrom3DPoints(R, t, idxInlier, idxOutlier);
    //     this->update3DPoints(R, t, inlier, outlier, R_, t_, false, T, Toc, Poc); // overloading function
    // }else{
    //     this->scale_propagation(R_, t_, inlier, outlier);
    //     this->update3DPoints(R_, t_, inlier, outlier, T, Toc, Poc); // overloading function
    // }
    
    /**** ****/
    if (this->nFeature3DReconstructed < this->params.thInlier){
        std::cerr << "There are few inliers reconstructed in 3D." << std::endl;
        this->scale_initialized = false;
        return false;
    }else{
        std::cerr << "Temporal velocity: " << T.block(0,3,3,1).norm() << std::endl;
        std::cerr << "nFeatures3DReconstructed: " << this->nFeature3DReconstructed << std::endl;

        // Save solution
        TRec.push_back(T);
        TocRec.push_back(Toc);
        PocRec.push_back(Poc);

        this->scale_initialized = true;
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

    int len;
	std::vector<cv::Point2f> uv_prev, uv_curr;
    for (uint32_t i = 0; i < idx_2DInlier.size(); i++){
		len = this->features[idx_2DInlier[i]].uv.size();
        uv_prev.emplace_back(this->features[idx_2DInlier[i]].uv[len-2]);
		uv_curr.emplace_back(this->features[idx_2DInlier[i]].uv.back());
    }

	// Find reasonable rotation and translational vector
	int max_num = 0;
	std::vector<bool> max_inlier;
	std::vector<Eigen::Vector3d> opt_X_curr;
	for( uint32_t i = 0; i < R_vec.size(); i++ ){
		Eigen::Matrix3d R1 = R_vec[i];
		Eigen::Vector3d t1 = t_vec[i];
		
		std::vector<Eigen::Vector3d> X_prev, X_curr;
		std::vector<bool> inlier;
		this->constructDepth(uv_prev, uv_curr, R1, t1, X_prev, X_curr, inlier);

		int nInlier = 0;
		for( uint32_t i = 0; i < inlier.size(); i++ )
			if( inlier[i] )
				nInlier++;

		if( nInlier > max_num ){
			max_num = nInlier;
			max_inlier = inlier;
            opt_X_curr = X_curr;
			
			R = R1;
			t = t1;
		}
	}

	// Store 3D characteristics in features
	if( max_num < this->nFeature2DInliered*0.5 )
		success = false;
	else{
		for( int i = 0; i < max_num; i++ ){
			if( max_inlier[i] ){
				this->features[idx_2DInlier[i]].point = (Eigen::Vector4d() << opt_X_curr[i], 1).finished();
				this->features[idx_2DInlier[i]].is_3D_reconstructed = true;
                this->nFeature3DReconstructed++;
			}
		}
		success = true;
	}

	return success;
}

bool MVO::verify_solutions(Eigen::Matrix3d& R, Eigen::Vector3d& t){
    
	bool success;

	// Extract homogeneous 2D point which is inliered with essential constraint
	std::vector<int> idx_2DInlier;
	for( int i = 0; i < this->nFeature; i++ )
		if( this->features[i].is_2D_inliered )
			idx_2DInlier.push_back(i);

    int len;
	std::vector<cv::Point2f> uv_prev, uv_curr;
    for (uint32_t i = 0; i < idx_2DInlier.size(); i++){
		len = this->features[idx_2DInlier[i]].uv.size();
        uv_prev.emplace_back(this->features[idx_2DInlier[i]].uv[len-2]);
		uv_curr.emplace_back(this->features[idx_2DInlier[i]].uv.back());
    }

	// Find reasonable rotation and translational vector
    std::vector<Eigen::Vector3d> X_prev, X_curr;
    std::vector<bool> inlier;
    this->constructDepth(uv_prev, uv_curr, R, t, X_prev, X_curr, inlier);

    int nInlier = 0;
    for( uint32_t i = 0; i < inlier.size(); i++ )
        if( inlier[i] )
            nInlier++;

	// Store 3D characteristics in features
	if( nInlier < this->nFeature2DInliered*0.5 )
		success = false;
	else{
		for( uint32_t i = 0; i < inlier.size(); i++ ){
			if( inlier[i] ){
				this->features[idx_2DInlier[i]].point = (Eigen::Vector4d() << X_curr[i], 1).finished();
				this->features[idx_2DInlier[i]].is_3D_reconstructed = true;
                this->nFeature3DReconstructed++;
			}
		}
        std::cerr << "nFeature3DReconstructed: " << this->nFeature3DReconstructed << std::endl;
		success = true;
	}

	return success;
}

bool MVO::findPoseFrom3DPoints(Eigen::Matrix3d &R, Eigen::Vector3d &t, std::vector<int>& idxInlier, std::vector<int>& idxOutlier){    
    // Seek index of which feature is 3D reconstructed currently,
    // and 3D initialized previously
	bool flag;

    std::vector<int> idx;
    for (int i = 0; i < this->nFeature; i++)
        if (this->features[i].is_3D_init)
            idx.push_back(i);
    
    const uint32_t nPoint = idx.size();

    // Use RANSAC to find suitable scale
    if (nPoint > (uint32_t)params.thInlier){
        std::vector<cv::Point3f> objectPoints;
        std::vector<cv::Point2f> imagePoints;
        for (uint32_t i = 0; i < nPoint; i++)
        {
            objectPoints.emplace_back(features[idx[i]].point_init(0), features[idx[i]].point_init(1), features[idx[i]].point_init(2));
            imagePoints.emplace_back(features[idx[i]].uv.back().x, features[idx[i]].uv.back().y); // return last element of uv
        }

        // r_vec = cv::Mat::zeros(3,1,CV_32F);
        // t_vec = cv::Mat::zeros(3,1,CV_32F);

        R = this->TocRec.back().block(0,0,3,3).transpose();
        t = -R * this->TocRec.back().block(0,3,3,1);

        cv::Mat R_, r_vec, t_vec;
        cv::eigen2cv(R, R_);
        cv::eigen2cv(t, t_vec);
        cv::Rodrigues(R_, r_vec);
        
        cv::solvePnPRefineLM(objectPoints, imagePoints, this->params.Kcv, cv::noArray(), r_vec, t_vec, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 100, 1e-7));

        // bool success = cv::solvePnP(objectPoints, imagePoints, this->params.Kcv, cv::noArray(), r_vec, t_vec, true, cv::SOLVEPNP_ITERATIVE);

        // bool success = cv::solvePnPRansac(objectPoints, imagePoints, this->params.Kcv, cv::noArray(),
        //                                   r_vec, t_vec, false, 1e3,
        //                                   this->params.reprojError, 0.6, idxInlier, cv::SOLVEPNP_AP3P);
        // if (!success){
        //     success = cv::solvePnPRansac(objectPoints, imagePoints, this->params.Kcv, cv::noArray(),
        //                                       r_vec, t_vec, false, 1e3,
        //                                       2 * params.reprojError, 0.6, idxInlier, cv::SOLVEPNP_AP3P);
        // }
        std::cerr << "## Solve PnP: " << lsi::toc() << std::endl;

        cv::Mat R_cv;
        cv::Rodrigues(r_vec, R_cv);
        cv::cv2eigen(R_cv, R);
        cv::cv2eigen(t_vec, t);
        
        idxOutlier.clear();
        if( idxInlier.empty() ){
            for (int i = 0; i < (int)nPoint; i++)
                idxOutlier.push_back(i);
        }else{
            for (int i = 0, it = 0; i < (int)nPoint; i++){
                if (idxInlier[it] == i)
                    it++;
                else
                    idxOutlier.push_back(i);
            }
        }
        flag = true;
    }
    else{
        idxInlier.clear();
        idxOutlier.clear();
        R = Eigen::Matrix3d::Identity();
        t = Eigen::Vector3d::Zero();
        flag = false;
    }
    return flag;
}

void MVO::constructDepth(const std::vector<cv::Point2f> uv_prev, const std::vector<cv::Point2f> uv_curr, 
                        const Eigen::Matrix3d R, const Eigen::Vector3d t, 
                        std::vector<Eigen::Vector3d> &X_prev, std::vector<Eigen::Vector3d> &X_curr, 
                        std::vector<bool> &inlier){

    std::cerr << "## Init constructDepth: " << lsi::toc() << std::endl;
    const uint32_t nPoint = uv_prev.size();

    switch(this->params.triangulationMethod){
        case MVO::TRIANGULATION::MIDP : {
            Eigen::Matrix<double,3,3> A, A0, A1;
            Eigen::Vector3d b;
            
            Eigen::Matrix<double,3,4> P0, P1;
            P0 << this->params.K, Eigen::Vector3d::Zero();
            P1 << this->params.K * R, this->params.K * t;

            Eigen::Matrix3d M0, M1;
            Eigen::Vector3d c0, c1, u0, u1;

            M0 = P0.block(0,0,3,3).inverse();
            c0 = -M0*P0.block(0,3,3,1);
            M1 = P1.block(0,0,3,3).inverse();
            c1 = -M1*P1.block(0,3,3,1);

            for( uint32_t i = 0; i < nPoint; i++ ){
                u0 = M0 * (Eigen::Vector3d() << uv_prev[i].x, uv_prev[i].y, 1).finished();
                u1 = M1 * (Eigen::Vector3d() << uv_curr[i].x, uv_curr[i].y, 1).finished();

                A0 = Eigen::Matrix3d::Identity() - u0 * u0.transpose() / (u0.cwiseProduct(u0)).sum();
                A1 = Eigen::Matrix3d::Identity() - u1 * u1.transpose() / (u1.cwiseProduct(u1)).sum();

                A = A0 + A1;
                b = A0 * c0 + A1 * c1;

                X_prev.push_back(A.inverse() * b);
                X_curr.push_back(R * X_prev.back() + t);
                inlier.push_back(X_prev.back()(2) > 0 && X_curr.back()(2) > 0);
            }
            std::cerr << "## Reconstruct 3D points: " << lsi::toc() << std::endl;
            break;
        }

        case MVO::TRIANGULATION::LLS : {
            std::vector<Eigen::Vector3d> x_prev, x_curr;
            for( uint32_t i = 0; i < uv_prev.size(); i++ ){
                x_prev.emplace_back((uv_prev[i].x - this->params.cx)/this->params.fx, (uv_prev[i].y - this->params.cy)/this->params.fy, 1);
                x_curr.emplace_back((uv_curr[i].x - this->params.cx)/this->params.fx, (uv_curr[i].y - this->params.cy)/this->params.fy, 1);
            }

            Eigen::MatrixXd& M_matrix = this->MapMatrixTemplate;
            M_matrix.resize(3*nPoint, nPoint+1);
            M_matrix.setZero();

            for (uint32_t i = 0; i < nPoint; i++){
                M_matrix.block(3*i,i,3,1) = skew(x_curr[i])*R*x_prev[i];
                M_matrix.block(3*i,nPoint,3,1) = skew(x_curr[i])*t;
            }
            std::cerr << "## Construct MtM: " << lsi::toc() << std::endl;

            Eigen::MatrixXd V;
            uint32_t idxMinEigen = nPoint;
            switch( this->params.SVDMethod){
                case MVO::SVD::JACOBI:{
                    Eigen::JacobiSVD<Eigen::MatrixXd> svd(M_matrix, Eigen::ComputeThinV);
                    V = svd.matrixV();
                    break;
                }
                case MVO::SVD::BDC:{
                    Eigen::HouseholderQR<Eigen::MatrixXd> qrdecomposer(M_matrix);
                    Eigen::MatrixXd R = qrdecomposer.matrixQR().triangularView<Eigen::Upper>();
                    Eigen::MatrixXd compactR;
                    
                    for( int i = R.rows()-1; i >= 0; i-- ){
                        if( R.row(i).any() ){
                            compactR = R.topRows(i+1);
                            break;
                        }
                    }
                    
                    Eigen::BDCSVD<Eigen::MatrixXd> svd(compactR, Eigen::ComputeThinV);
                    V = svd.matrixV();
                    break;
                }
                case MVO::SVD::OpenCV:{
                    cv::Mat Vt, U, W;
                    cv::Mat M;
                    cv::eigen2cv(M_matrix, M);
                    cv::SVD::compute(M, W, U, Vt, cv::SVD::MODIFY_A);

                    Eigen::MatrixXd Vt_;
                    cv::cv2eigen(Vt, Vt_);
                    V = Vt_.transpose();
                    break;
                }
                case MVO::SVD::Eigen:{
                    Eigen::MatrixXd MtM_ = M_matrix.transpose()*M_matrix;
                    this->eigenSolver = new Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(MtM_, Eigen::ComputeEigenvectors);
                    // this->eigenSolver->compute(MtM_, Eigen::ComputeEigenvectors);
                    // Eigen::VectorXd D = eig.eigenvalues().real();
                    V = this->eigenSolver->eigenvectors();
                    idxMinEigen = 0;

                    // std::cout << D.transpose() << std::endl;
                    // D.minCoeff(&idxMinEigen);

                    break;
                }
            }
            std::cerr << "## Compute eigenvector: " << lsi::toc() << std::endl;

            for (uint32_t i = 0; i < nPoint; i++){
                X_prev.push_back( V(i,idxMinEigen) / V(nPoint,idxMinEigen) * x_prev[i] );
                X_curr.push_back( R*X_prev.back() + t );
                inlier.push_back( X_prev.back()(2) > 0 && X_curr.back()(2) > 0 );
            }
        }
    }
}
	
// without PnP
void MVO::update3DPoints(const Eigen::Matrix3d &R, const Eigen::Vector3d &t, 
                        const std::vector<bool> &inlier, const std::vector<bool> &outlier, 
                        Eigen::Matrix4d &T, Eigen::Matrix4d &Toc, Eigen::Vector4d &Poc){
    // without PnP
    double scale = t.norm();
    
    // Seek index of which feature is 3D reconstructed currently
    // and 3D initialized previously
    T.setIdentity();
    T.block(0,0,3,3) = R.transpose();
    T.block(0,3,3,1) = -R.transpose()*t;
    Toc = this->TocRec.back() * T;
    Poc = Toc.block(0,3,4,1);

    for( uint32_t i = 0; i < this->features.size(); i++ ){
        if( this->features[i].is_3D_reconstructed ){
            this->features[i].point.block(0,0,3,1) *= scale;

            if( !this->features[i].is_3D_init && this->features[i].is_wide ){
                this->features[i].point_init = Toc * this->features[i].point;
                this->features[i].is_3D_init = true;
            }
        }
    }

    std::cerr << "# Update 3D points with Essential: " << lsi::toc() << std::endl;
}
	
// with pnp
void MVO::update3DPoints(const Eigen::Matrix3d &R, const Eigen::Vector3d &t, 
                        const std::vector<bool> &inlier, const std::vector<bool> &outlier, 
                        const Eigen::Matrix3d &R_E, const Eigen::Vector3d &t_E, const bool &success_E, 
                        Eigen::Matrix4d &T, Eigen::Matrix4d &Toc, Eigen::Vector4d &Poc){
    // with PnP
    // Extract homogeneous 2D point which is inliered with essential constraint
    Toc.setIdentity();
    Toc.block(0,0,3,3) = R.transpose();
    Toc.block(0,3,3,1) = -R.transpose()*t;
    T = TocRec.back().inverse() * Toc;
    Poc = Toc.block(0,3,4,1);

    Eigen::Matrix4d Tco;
    Tco.setIdentity();
    Tco.block(0,0,3,3) = R;
    Tco.block(0,3,3,1) = t;

    if(success_E){
        double scale = t_E.norm();

        for( uint32_t i = 0; i < features.size(); i++ ){
            if( features[i].is_3D_reconstructed ){
                features[i].point.block(0,0,3,1) *= scale;

                if( !features[i].is_3D_init && features[i].is_wide ){
                    features[i].point_init = Toc*features[i].point;
                    features[i].is_3D_init = true;
                }
            }
        }
    }else{
        Eigen::Matrix3d Rinv;
        Eigen::Vector3d tinv;

        Rinv = T.block(0,0,3,3).transpose();
        tinv = -T.block(0,0,3,3).transpose()*T.block(0,3,3,1);

        // Initialize 3D points in global coordinates
        // Extract Homogeneous 2D point which is inliered with essential constraint
        std::vector<int> idx2D;
        for (uint32_t i = 0; i < features.size(); i++){
            if (features[i].is_2D_inliered)
                idx2D.push_back(i);
        }
        
        const uint32_t nPoint = idx2D.size();
        int len;
        std::vector<cv::Point2f> uv_prev, uv_curr;
        for (uint32_t i = 0; i < nPoint; i++){
            len = this->features[idx2D[i]].uv.size();
            uv_prev.emplace_back(this->features[idx2D[i]].uv[len-2]);
            uv_curr.emplace_back(this->features[idx2D[i]].uv.back());
        }
        
        std::vector<Eigen::Vector3d> X_prev, X_curr;
        std::vector<bool> inliers;
        this->constructDepth(uv_prev, uv_curr, Rinv, tinv, X_prev, X_curr, inliers);

        for (uint32_t i = 0; i < nPoint; i++){
            // 2d inliers
            if(inliers[i]){
                features[idx2D[i]].point = (Eigen::Vector4d() << X_curr[i], 1).finished();
				features[idx2D[i]].is_3D_reconstructed = true;
                
                if( !features[idx2D[i]].is_3D_init && features[idx2D[i]].is_wide ){
                    features[idx2D[i]].point_init = Toc*features[idx2D[i]].point;
                    features[idx2D[i]].is_3D_init = true;
                }
            } // if(lambda_prev > 0 && lambda_curr > 0)
        } // for
    }

    int Feature3Dconstructed = 0;
    for (uint32_t i = 0; i < features.size(); i++){
        if(features[i].is_3D_reconstructed)
            Feature3Dconstructed++;
    }
    this->nFeature3DReconstructed = Feature3Dconstructed;
    std::cerr << "# Update 3D points with PnP: " << lsi::toc() << std::endl;
}

bool MVO::scale_propagation(Eigen::Matrix3d &R, Eigen::Vector3d &t, std::vector<bool> &inlier, std::vector<bool> &outlier){
    inlier.clear();
    outlier.clear();

    double scale = 0;
    bool flag;
    Eigen::Matrix4d T_;
    T_.setIdentity();
    T_.block(0,0,3,3) = R;
    T_.block(0,3,3,1) = t;

    uint32_t nPoint;
    if (this->scale_initialized)
    {
        // Seek index of which feature is 3D reconstructed currently,
        // and 3D initialized previously
        std::vector<int> idx;
        for (uint32_t i = 0; i < this->features.size(); i++){
            if( this->features[i].is_3D_reconstructed && features[i].is_3D_init)
                idx.push_back(i);
        }
        nPoint = idx.size();

        // Use RANSAC to find suitable scale
        if ( nPoint > this->params.ransacCoef_scale_prop.minPtNum){
            // std::vector<Eigen::Vector3d> P1_ini;
            // std::vector<Eigen::Vector3d> P1_exp;
            std::vector<cv::Point3f> P1_ini;
            std::vector<cv::Point3f> P1_exp;
            Eigen::Vector3d Eigen_point;
            Eigen::Vector4d temp_point;
            for (uint32_t i = 0; i < nPoint; i++){
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
            this->nFeatureInlier = std::count(inlier.begin(), inlier.end(), true);

        }

        // Use the previous scale, if the scale cannot be found
        if (nPoint <= this->params.ransacCoef_scale_prop.minPtNum 
            || inlier.size() < (std::size_t)this->params.thInlier || scale == 0)
        {
            std::cerr << "There are a few SCALE FACTOR INLIERS" << std::endl;

            inlier.clear();
            outlier.clear();

            scale = (this->TRec[this->step-1].block(0,0,3,1)).norm();

            // Update scale
            t = scale * t;
            flag = false;
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
        cv::Point2f uv_curr;
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
        for (uint32_t i = 0; i < features.size(); i++)
            inlier.push_back( this->features[i].is_3D_reconstructed );
        flag = true;
    }

    std::cerr << "# Propagate scale: " << lsi::toc() << std::endl;

    if( t.hasNaN() )
        return false;
    else
        return flag;
}

double MVO::ransac(const std::vector<cv::Point3f> &x, const std::vector<cv::Point3f> &y,
                   MVO::RansacCoef ransacCoef,
                   std::vector<bool> &inlier, std::vector<bool> &outlier){
    uint32_t ptNum = x.size();

    std::vector<uint32_t> sampleIdx;
    std::vector<cv::Point3f> x_sample, y_sample;

    int iterNUM = 1e8;
    std::size_t max_inlier = 0;

    int it = 0;

    while (it < std::min(ransacCoef.iterMax, iterNUM))
    {
        // 1. fit using random points
        if (ransacCoef.weight.size() > 0)
            sampleIdx = randweightedpick(ransacCoef.weight, ransacCoef.minPtNum);
        else
            sampleIdx = randperm(ptNum, ransacCoef.minPtNum);

        int tempIdx = 0;
        x_sample.clear();
        y_sample.clear();
        for (uint32_t i = 0; i < sampleIdx.size(); i++)
        {
            tempIdx = sampleIdx[i];
            x_sample.push_back(x[tempIdx]);
            y_sample.push_back(y[tempIdx]);
        }
        double f1 = calculate_scale(x_sample, y_sample);
        std::vector<double> dist1 = calculate_scale_error(f1, x, y);
        std::vector<bool> in1;
        uint32_t nInlier = 0;
        for (uint32_t i = 0; i < dist1.size(); i++)
        {
            if( dist1[i] < ransacCoef.thDist ){
                in1.push_back( true );
                nInlier++;
            }else{
                in1.push_back( false );
            }
        }

        if (nInlier > max_inlier)
        {
            max_inlier = nInlier;
            inlier = in1;
            double InlrRatio = (double)max_inlier / (double)ptNum + 1e-16;
            iterNUM = static_cast<int>(std::floor(std::log(1 - ransacCoef.thInlrRatio) / std::log(1 - std::pow(InlrRatio, ransacCoef.minPtNum))));
        }
        it++;
    }
    std::cerr << "Ransac iterations: " << it << std::endl;

    if (max_inlier == 0)
    {
        inlier.clear();
        outlier.clear();
        return 0;
    }
    else
    {
        x_sample.clear();
        y_sample.clear();
        for (uint32_t i = 0; i < inlier.size(); i++){
            if (inlier[i]){
                x_sample.push_back(x[i]);
                y_sample.push_back(y[i]);
            }
        }
        double f1 = calculate_scale(x_sample, y_sample);

        std::vector<double> dist = calculate_scale_error(f1, x, y);

        inlier.clear();
        outlier.clear();
        for (uint32_t i = 0; i < dist.size(); i++){
            inlier.push_back(dist[i] < ransacCoef.thDist);
            outlier.push_back(dist[i] > ransacCoef.thDistOut);
        }

        return f1;
    }
}

std::vector<uint32_t> MVO::randperm(uint32_t ptNum, int minPtNum){
    std::vector<uint32_t> result;
    for (uint32_t i = 0; i < ptNum; i++)
        result.push_back(i);
    std::random_shuffle(result.begin(), result.begin() + minPtNum);
    return result;
}

std::vector<uint32_t> MVO::randweightedpick(const std::vector<double> &h, int n /*=1*/){
    int u = h.size();
    int s_under;
    double sum, rand_num;
    std::vector<double> H = h;
    //std::list<double> H;
    // std::copy(h.cbegin(), h.cend(), std::back_inserter(H));
    std::vector<double> Hs, Hsc;
    std::vector<uint32_t> result;

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

double MVO::calculate_scale(const std::vector<cv::Point3f> &pt1, const std::vector<cv::Point3f> &pt2){
    double sum = 0;
    for (uint32_t i = 0; i < pt1.size(); i++)
    {
        sum += (pt1[i].x * pt2[i].x + pt1[i].y * pt2[i].y + pt1[i].z * pt2[i].z) / (pt1[i].x * pt1[i].x + pt1[i].y * pt1[i].y + pt1[i].z * pt1[i].z + 1e-10);
    }
    return sum / pt1.size();
}

std::vector<double> MVO::calculate_scale_error(double scale, const std::vector<cv::Point3f> &pt1, const std::vector<cv::Point3f> &pt2){
    std::vector<double> dist;
    for (uint32_t i = 0; i < pt1.size(); i++)
        dist.push_back(cv::norm(pt2[i] - scale * pt1[i]));
    return dist;
}