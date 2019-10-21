#include "core/MVO.hpp"
#include "core/utils.hpp"
#include "core/numerics.hpp"
#include "core/time.hpp"

double scale_reference;

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
    // std::cerr << "R diag: " << R_.diagonal().transpose() << std::endl;
    // std::cerr << "t: " << t_.transpose() << std::endl;

    /**************************************************
     * Mapping
     **************************************************/
    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    Eigen::Matrix4d T, Toc;
    Eigen::Vector4d Poc;
    std::vector<bool> inlier, outlier;
	std::vector<int> idxInlier, idxOutlier;
    
    switch (this->params.mappingOption) {
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

    case 0:
        /**** mapping without scaling ****/
        this->update3DPoints(R_, t_, inlier, outlier, T, Toc, Poc);
        break;

    case 1:
        /**** mapping and scaling with essential 3d reconstruction only ****/
        this->scale_propagation(R_ ,t_, inlier, outlier);
        std::cerr << "Essential 3D error: " << this->calcReconstructionError(R_, t_) << std::endl;
        this->update3DPoints(R_, t_, inlier, outlier, T, Toc, Poc); // overloading function
        break;

    case 2:
        /**** mapping and scaling with PnP only ****/
        if( this->scale_initialized == true ){
            this->findPoseFrom3DPoints(R, t, idxInlier, idxOutlier);
        }else{
            R = R_;
            t = t_;
        }
        this->update3DPoints(R, t, inlier, outlier, R_, t_, false, T, Toc, Poc); // overloading function
        break;

    case 3:
        /**** use both PnP and essential 3d reconstruction - original ****/
        if (this->findPoseFrom3DPoints(R, t, idxInlier, idxOutlier)){
            std::cerr << "# Find pose from PnP: " << lsi::toc() << std::endl;
            std::cerr << "PnP 3D error: " << this->calcReconstructionError(R, t) << std::endl;

            // Update 3D points
            bool success = this->scale_propagation(R_, t_, inlier, outlier);
            std::cerr << "# Find scale from Essential: " << lsi::toc() << std::endl;
            
            // Update 3D points
            this->update3DPoints(R, t, inlier, outlier, R_, t_, success, T, Toc, Poc); // overloading function
            std::cout << "Update 3D Points with PnP, ";
        }else{
            std::cerr << "# Find pose from PnP: " << lsi::toc() << std::endl;

            // Update 3D points
            bool success = this->scale_propagation(R_ ,t_, inlier, outlier);
            std::cerr << "# Find scale from Essential: " << lsi::toc() << std::endl;

            if (!success){
                std::cerr << "There are few inliers matching scale." << std::endl;
                return false;
            }
            std::cerr << "Essential 3D error: " << this->calcReconstructionError(R_, t_) << std::endl;

            this->update3DPoints(R_, t_, inlier, outlier, T, Toc, Poc); // overloading function
            std::cout << "Update 3D Points with Essential Constraint, ";
        }
        break;

    case 4:
        /**** use both PnP and essential 3d reconstruction - modified ****/
        if( this->scale_propagation(R_, t_, inlier, outlier) ){
            std::cerr << "# Find scale from Essential: " << lsi::toc() << std::endl;

            R = R_;
            t = t_;

            this->findPoseFrom3DPoints(R, t, idxInlier, idxOutlier);
            std::cerr << "# Find pose from PnP: " << lsi::toc() << std::endl;

            // Update 3D points
            this->update3DPoints(R, t, inlier, outlier, R_, t_, true, T, Toc, Poc);
            std::cout << "Update 3D Points with PnP, ";
        }else{
            std::cerr << "# Find scale from Essential: " << lsi::toc() << std::endl;

            this->findPoseFrom3DPoints(R, t, idxInlier, idxOutlier);
            std::cerr << "# Find pose from PnP: " << lsi::toc() << std::endl;

            // Update 3D points
            this->update3DPoints(R, t, inlier, outlier, R_, t_, false, T, Toc, Poc);
            std::cout << "Update 3D Points with Essential Constraint, ";
        }
        break;
    }
    std::cerr << "nFeature3DReconstructed: " << (double) this->nFeature / this->nFeature * 100 << '%' << std::endl;

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

bool MVO::verify_solutions(const std::vector<Eigen::Matrix3d>& R_vec, const std::vector<Eigen::Vector3d>& t_vec, Eigen::Matrix3d& R, Eigen::Vector3d& t){
    
	bool success;

	// Extract homogeneous 2D point which is inliered with essential constraint
	std::vector<int> idx_2DInlier;
	for( int i = 0; i < this->nFeature; i++ )
		if( this->features[i].is_2D_inliered )
			idx_2DInlier.push_back(i);

    int key_idx;
	std::vector<cv::Point2f> uv_prev, uv_curr;
    for (uint32_t i = 0; i < idx_2DInlier.size(); i++){
        key_idx = this->features[idx_2DInlier[i]].life - 1 - (this->step - this->key_step);
        uv_prev.emplace_back(this->features[idx_2DInlier[i]].uv[key_idx]);
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

bool MVO::verify_solutions(const Eigen::Matrix3d& R, const Eigen::Vector3d& t){
    
	bool success;

	// Extract homogeneous 2D point which is inliered with essential constraint
	std::vector<int> idx_2DInlier;
	for( int i = 0; i < this->nFeature; i++ )
		if( this->features[i].is_2D_inliered )
			idx_2DInlier.push_back(i);

    int key_idx;
	std::vector<cv::Point2f> uv_prev, uv_curr;
    for (uint32_t i = 0; i < idx_2DInlier.size(); i++){
        key_idx = this->features[idx_2DInlier[i]].life - 1 - (this->step - this->key_step);
        uv_prev.emplace_back(this->features[idx_2DInlier[i]].uv[key_idx]);
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
    if (nPoint > (uint32_t)this->params.thInlier){
        std::vector<cv::Point3f> objectPoints;
        std::vector<cv::Point2f> imagePoints;
        for (uint32_t i = 0; i < nPoint; i++)
        {
            objectPoints.emplace_back(this->features[idx[i]].point_init(0), this->features[idx[i]].point_init(1), this->features[idx[i]].point_init(2));
            imagePoints.emplace_back(this->features[idx[i]].uv.back().x, this->features[idx[i]].uv.back().y); // return last element of uv
        }

        // r_vec = cv::Mat::zeros(3,1,CV_32F);
        // t_vec = cv::Mat::zeros(3,1,CV_32F);

        cv::Mat R_, r_vec, t_vec;
        if( R.determinant() < .1){
            Eigen::Matrix3d R_prev = this->TocRec.back().block(0,0,3,3).transpose();
            Eigen::Vector3d t_prev = -R_prev * this->TocRec.back().block(0,3,3,1);
            Eigen::Vector3d speed_prev = this->TRec.back().block(0,3,3,1);

            cv::eigen2cv(R_prev, R_);
            cv::eigen2cv((Eigen::Vector3d) (t_prev - speed_prev), t_vec);
            cv::Rodrigues(R_, r_vec);
        }else{
            cv::eigen2cv(R, R_);
            cv::eigen2cv(t, t_vec);
            cv::Rodrigues(R_, r_vec);
        }

        switch( this->params.pnpMethod ){
            case MVO::PNP::LM : 
            case MVO::PNP::ITERATIVE : {
                bool success = cv::solvePnP(objectPoints, imagePoints, this->params.Kcv, cv::noArray(), r_vec, t_vec, true, cv::SOLVEPNP_ITERATIVE);
                flag = success;
                break;
            }
            case MVO::PNP::AP3P : {
                bool success = cv::solvePnPRansac(objectPoints, imagePoints, this->params.Kcv, cv::noArray(),
                                                r_vec, t_vec, true, 1e3,
                                                this->params.reprojError, 0.99, idxInlier, cv::SOLVEPNP_AP3P);
                if (!success){
                    success = cv::solvePnPRansac(objectPoints, imagePoints, this->params.Kcv, cv::noArray(),
                                                r_vec, t_vec, true, 1e3,
                                                2 * this->params.reprojError, 0.99, idxInlier, cv::SOLVEPNP_AP3P);
                }
                flag = success;
                break;
            }
            case MVO::PNP::EPNP : {
                bool success = cv::solvePnPRansac(objectPoints, imagePoints, this->params.Kcv, cv::noArray(),
                                                r_vec, t_vec, true, 1e3,
                                                this->params.reprojError, 0.99, idxInlier, cv::SOLVEPNP_EPNP);
                if (!success){
                    success = cv::solvePnPRansac(objectPoints, imagePoints, this->params.Kcv, cv::noArray(),
                                                r_vec, t_vec, true, 1e3,
                                                2 * this->params.reprojError, 0.99, idxInlier, cv::SOLVEPNP_EPNP);
                }
                flag = success;
                break;
            }
            case MVO::PNP::DLS : {
                bool success = cv::solvePnPRansac(objectPoints, imagePoints, this->params.Kcv, cv::noArray(),
                                                r_vec, t_vec, true, 1e3,
                                                this->params.reprojError, 0.99, idxInlier, cv::SOLVEPNP_DLS);
                if (!success){
                    success = cv::solvePnPRansac(objectPoints, imagePoints, this->params.Kcv, cv::noArray(),
                                                r_vec, t_vec, true, 1e3,
                                                2 * this->params.reprojError, 0.99, idxInlier, cv::SOLVEPNP_DLS);
                }
                flag = success;
                break;
            }
            case MVO::PNP::UPNP : {
                bool success = cv::solvePnPRansac(objectPoints, imagePoints, this->params.Kcv, cv::noArray(),
                                                r_vec, t_vec, true, 1e3,
                                                this->params.reprojError, 0.99, idxInlier, cv::SOLVEPNP_UPNP);
                if (!success){
                    success = cv::solvePnPRansac(objectPoints, imagePoints, this->params.Kcv, cv::noArray(),
                                                r_vec, t_vec, true, 1e3,
                                                2 * this->params.reprojError, 0.99, idxInlier, cv::SOLVEPNP_UPNP);
                }
                flag = success;
                break;
            }
        }
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
            
            // Eigen::Matrix<double,3,4> P0, P1;
            // P0 << this->params.K, Eigen::Vector3d::Zero();
            // P1 << this->params.K * R, this->params.K * t;

            Eigen::Matrix3d M0, M1;
            Eigen::Vector3d c0, c1, u0, u1;

            // M0 = P0.block(0,0,3,3).inverse();
            // c0 = -M0*P0.block(0,3,3,1);
            // M1 = P1.block(0,0,3,3).inverse();
            // c1 = -M1*P1.block(0,3,3,1);
            M0 = this->params.Kinv;
            M1 = R.inverse() * this->params.Kinv;
            // c0 = Eigen::Vector3d::Zero();
            c1 = -R.inverse() * t;

            for( uint32_t i = 0; i < nPoint; i++ ){
                u0 = M0 * (Eigen::Vector3d() << uv_prev[i].x, uv_prev[i].y, 1).finished();
                u1 = M1 * (Eigen::Vector3d() << uv_curr[i].x, uv_curr[i].y, 1).finished();

                A0 = Eigen::Matrix3d::Identity() - u0 * u0.transpose() / (u0.cwiseProduct(u0)).sum();
                A1 = Eigen::Matrix3d::Identity() - u1 * u1.transpose() / (u1.cwiseProduct(u1)).sum();

                A = A0 + A1;
                b = A1 * c1; // A0 * c0 + A1 * c1 = A1 * c1, because c0 = 0

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
                    this->eigenSolver->compute(MtM_, Eigen::ComputeEigenvectors);

                    V = this->eigenSolver->eigenvectors();
                    idxMinEigen = 0;
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
    Eigen::Matrix4d T_;
    T_.setIdentity();
    T_.block(0,0,3,3) = R.transpose();
    T_.block(0,3,3,1) = -R.transpose()*t;
    Toc = this->TocRec[this->key_step] * T_;
    Poc = Toc.block(0,3,4,1);
    T = this->TocRec.back().inverse() * Toc;
    
    int key_idx;
    double cur_var, prv_var, new_var, diffX;
    for( uint32_t i = 0; i < this->features.size(); i++ ){
        key_idx = this->features[i].life - 1 - (this->step - this->key_step);
        if( this->features[i].is_3D_reconstructed ){
            this->features[i].point.block(0,0,3,1) *= scale;

            cur_var = 5/(cv::norm(this->features[i].uv[key_idx] - this->features[i].uv.back()));
            if( this->features[i].is_3D_init ){
                if( this->params.updateInitPoint ){
                    diffX = (Toc * this->features[i].point - this->features[i].point_init).norm();
                    if( diffX < 3*this->features[i].point_var){
                        prv_var = this->features[i].point_var;
                        new_var = 1 / ( (1 / prv_var) + (1 / cur_var) );
                        
                        this->features[i].point_init = new_var/cur_var * Toc * this->features[i].point + new_var/prv_var * this->features[i].point_init;
                        this->features[i].point_var = new_var;
                    }
                }
            }else if( this->features[i].is_wide ){
                this->features[i].point_init = Toc * this->features[i].point;
                this->features[i].point_var = cur_var;
                this->features[i].is_3D_init = true;
                this->features[i].frame_init = this->step;
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
    Poc = Toc.block(0,3,4,1);
    T = this->TocRec.back().inverse() * Toc;

    Eigen::Matrix4d Tco;
    Tco.setIdentity();
    Tco.block(0,0,3,3) = R;
    Tco.block(0,3,3,1) = t;

    if(success_E){
        double scale = t_E.norm();

        int key_idx;
        double cur_var, prv_var, new_var, diffX;
        for( uint32_t i = 0; i < this->features.size(); i++ ){
            if( this->features[i].is_3D_reconstructed ){
                this->features[i].point.block(0,0,3,1) *= scale;

                key_idx = this->features[i].life - 1 - (this->step - this->key_step);
                cur_var = 5/(cv::norm(this->features[i].uv[key_idx] - this->features[i].uv.back()));
                if( this->features[i].is_3D_init ){
                    if( this->params.updateInitPoint ){
                        diffX = (Toc * this->features[i].point - this->features[i].point_init).norm();
                        if( diffX < 3*this->features[i].point_var){
                            prv_var = this->features[i].point_var;
                            new_var = 1 / ( (1 / prv_var) + (1 / cur_var) );
                            
                            this->features[i].point_init = new_var/cur_var * Toc * this->features[i].point + new_var/prv_var * this->features[i].point_init;
                            this->features[i].point_var = new_var;
                        }
                    }
                }else if( this->features[i].is_wide ){
                    this->features[i].point_init = Toc * this->features[i].point;
                    this->features[i].point_var = cur_var;
                    this->features[i].is_3D_init = true;
                    this->features[i].frame_init = this->step;
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
        for (uint32_t i = 0; i < this->features.size(); i++){
            if (this->features[i].is_2D_inliered)
                idx2D.push_back(i);
        }
        
        const uint32_t nPoint = idx2D.size();
        int len;
        std::vector<cv::Point2f> uv_prev, uv_curr;
        for (uint32_t i = 0; i < nPoint; i++){
            len = this->features[idx2D[i]].life;
            uv_prev.emplace_back(this->features[idx2D[i]].uv[len-2]);
            uv_curr.emplace_back(this->features[idx2D[i]].uv.back());
        }
        
        std::vector<Eigen::Vector3d> X_prev, X_curr;
        std::vector<bool> inliers;
        this->constructDepth(uv_prev, uv_curr, Rinv, tinv, X_prev, X_curr, inliers);

        double cur_var, prv_var, new_var, diffX;
        for (uint32_t i = 0; i < nPoint; i++){
            // 2d inliers
            if(inliers[i]){
                len = this->features[idx2D[i]].life;
                cur_var = 5/(cv::norm(this->features[idx2D[i]].uv[len-2] - this->features[idx2D[i]].uv.back()));
                this->features[idx2D[i]].point = (Eigen::Vector4d() << X_curr[i], 1).finished();
				this->features[idx2D[i]].is_3D_reconstructed = true;
                
                if( this->features[idx2D[i]].is_3D_init ){
                    if( this->params.updateInitPoint ){
                        diffX = (Toc * this->features[idx2D[i]].point - this->features[idx2D[i]].point_init).norm();
                        if( diffX < 3*this->features[idx2D[i]].point_var){
                            prv_var = this->features[idx2D[i]].point_var;
                            new_var = 1 / ( (1 / prv_var) + (1 / cur_var) );
                            
                            this->features[idx2D[i]].point_init = new_var/cur_var * Toc * this->features[idx2D[i]].point + new_var/prv_var * this->features[idx2D[i]].point_init;
                            this->features[idx2D[i]].point_var = new_var;
                        }
                    }
                }else if( this->features[idx2D[i]].is_wide ){
                    this->features[idx2D[i]].point_init = Toc * this->features[idx2D[i]].point;
                    this->features[idx2D[i]].point_var = cur_var;
                    this->features[idx2D[i]].is_3D_init = true;
                    this->features[idx2D[i]].frame_init = this->step;
                }
            } // if(lambda_prev > 0 && lambda_curr > 0)
        } // for
    }

    int Feature3Dconstructed = 0;
    for (uint32_t i = 0; i < this->features.size(); i++){
        if(this->features[i].is_3D_reconstructed)
            Feature3Dconstructed++;
    }
    this->nFeature3DReconstructed = Feature3Dconstructed;
    std::cerr << "# Update 3D points with PnP: " << lsi::toc() << std::endl;
}

bool MVO::scale_propagation(const Eigen::Matrix3d &R, Eigen::Vector3d &t, std::vector<bool> &inlier, std::vector<bool> &outlier){
    inlier.clear();
    outlier.clear();

    double scale = 0, scale_from_height = 0;
    bool flag;

    // Initialization
    // initialze scale, in the case of the first time
    cv::Point2f uv_curr;
    Eigen::Vector4d point_curr;
    std::vector<cv::Point3f> roadCandidate;
    std::vector<uint32_t> roadIdx;

    for (int i = 0; i < nFeature; i++){
        if( this->features[i].is_3D_reconstructed ){
            uv_curr = this->features[i].uv.back(); //latest feature

            if (uv_curr.y > this->params.imSize.height * 0.5 
            && uv_curr.y > this->params.imSize.height - 0.7 * uv_curr.x 
            && uv_curr.y > this->params.imSize.height + 0.7 * (uv_curr.x - this->params.imSize.width)){
                point_curr = this->features[i].point;
                roadCandidate.emplace_back(point_curr(0),point_curr(1),point_curr(2));
                roadIdx.push_back(i);
            }
        }
    }

    std::vector<double> plane;
    std::vector<bool> planeInlier, planeOutlier;
    this->ransac<cv::Point3f, std::vector<double>>(roadCandidate, this->params.ransacCoef_plane, plane, planeInlier, planeOutlier);

    for( uint32_t i = 0; i < planeInlier.size(); i++ ){
        if( planeInlier[roadIdx[i]] )
            this->features[roadIdx[i]].type = Type::Road;
        else
            this->features[roadIdx[i]].type = Type::Other;
    }

    scale_from_height = this->params.vehicle_height / std::abs(plane[3]);
    ::scale_reference = scale_from_height;

    if (this->scale_initialized)
    {
        // Seek index of which feature is 3D reconstructed currently,
        // and 3D initialized previously
        std::vector<int> idx;
        for (uint32_t i = 0; i < this->features.size(); i++){
            if( this->features[i].is_3D_reconstructed && this->features[i].is_3D_init)
                idx.push_back(i);
        }
        uint32_t nPoint = idx.size();

        // Use RANSAC to find suitable scale
        if ( nPoint > this->params.ransacCoef_scale.minPtNum){
            Eigen::Matrix4d T_;
            T_.setIdentity();
            T_.block(0,0,3,3) = R;
            T_.block(0,3,3,1) = t;

            std::vector<std::pair<cv::Point3f,cv::Point3f>> Points;
            Eigen::Vector4d init_point, expt_point, curr_point;
            this->params.ransacCoef_scale.weight.clear();
            
            for (uint32_t i = 0; i < nPoint; i++){
                curr_point = this->features[idx[i]].point;

                // Get initialized 3D point
                init_point = this->TocRec[this->key_step].inverse() * this->features[idx[i]].point_init;
                
                // Get expected 3D point by transforming the coordinates of the observed 3d point
                expt_point = T_.inverse() * curr_point;

                Points.emplace_back(cv::Point3f(expt_point(0),expt_point(1),expt_point(2)),
                                    cv::Point3f(init_point(0),init_point(1),init_point(2)));
                
                // RANSAC weight
                this->params.ransacCoef_scale.weight.push_back( std::atan( -curr_point(2)/5 + 3 ) + PI / 2 );
            }

            this->ransac<std::pair<cv::Point3f,cv::Point3f>,double>(Points, this->params.ransacCoef_scale, scale, inlier, outlier);
            this->nFeatureInlier = std::count(inlier.begin(), inlier.end(), true);
        }

        // Use the previous scale, if the scale cannot be found
        if (nPoint <= this->params.ransacCoef_scale.minPtNum 
            || inlier.size() < (std::size_t)this->params.thInlier || scale == 0)
        {
            std::cerr << "There are a few SCALE FACTOR INLIERS" << std::endl;

            inlier.clear();
            outlier.clear();

            scale = (this->step - this->key_step) * (this->TRec[this->step-1].block(0,0,3,1)).norm();

            // Update scale
            t = scale * t;
            flag = false;
        }
        else{
            std::cerr << "@ scale_from_height: " << scale_from_height << ", " << "scale: " << scale << std::endl;

            // Update scale
            t = scale * t;
            flag = true;
        }

    }else{
        t = scale_from_height * t;

        this->nFeatureInlier = this->nFeature3DReconstructed;
        flag = true;
    }

    std::cerr << "# Propagate scale: " << lsi::toc() << std::endl;

    if( t.hasNaN() )
        return false;
    else
        return flag;
}

double MVO::calcReconstructionError(Eigen::Matrix4d& Toc){
    std::vector<double> error;
    error.reserve(this->features.size());
    for( uint32_t i = 0; i < this->features.size(); i++ ){
        error.push_back((Toc * this->features[i].point - this->features[i].point_init).norm());
    }
    std::sort(error.begin(), error.end());
    return error[std::floor(error.size()/2)];
}

double MVO::calcReconstructionError(Eigen::Matrix3d& R, Eigen::Vector3d& t){
    Eigen::Matrix4d Toc;
    Toc.block(0,0,3,3) = R;
    Toc.block(0,3,3,1) = t;
    Toc(3,3) = 1;

    return this->calcReconstructionError(Toc);
}

double MVO::calcReconstructionErrorGT(Eigen::MatrixXd& depth){
    std::vector<double> error;
    for( uint32_t i = 0; i < this->features.size(); i++ ){
        if( depth(this->features[i].uv.back().y, this->features[i].uv.back().x) > 0)
            error.push_back(this->features[i].point(2) - depth(this->features[i].uv.back().y, this->features[i].uv.back().x));
    }
    std::sort(error.begin(), error.end());
    return error[std::floor(error.size()/2)];
}

template <typename DATA, typename FUNC>
void MVO::ransac(const std::vector<DATA> &samples, const MVO::RansacCoef<DATA, FUNC> ransacCoef, FUNC& val, std::vector<bool> &inlier, std::vector<bool> &outlier){
    uint32_t ptNum = samples.size();

    std::vector<uint32_t> sampleIdx;
    std::vector<DATA> sample;
    std::vector<double> dist;
    dist.reserve(ptNum);

    int iterNUM = 1e5;
    uint32_t max_inlier = 0;
    double InlrRatio;

    for( int it = 0; it < std::min(ransacCoef.iterMax, iterNUM); it++ ){
        // 1. fit using random points
        if (ransacCoef.weight.size() > 0)
            sampleIdx = randweightedpick(ransacCoef.weight, ransacCoef.minPtNum);
        else
            sampleIdx = randperm(ptNum, ransacCoef.minPtNum);

        sample.clear();
        for (uint32_t i = 0; i < sampleIdx.size(); i++){
            sample.push_back(samples[sampleIdx[i]]);
        }

        ransacCoef.calculate_func(sample, val);
        ransacCoef.calculate_dist(val, samples, dist);

        std::vector<bool> in1;
        uint32_t nInlier = 0;
        for (uint32_t i = 0; i < dist.size(); i++){
            if( dist[i] < ransacCoef.thDist ){
                in1.push_back( true );
                nInlier++;
            }else{
                in1.push_back( false );
            }
        }

        if (nInlier > max_inlier){
            max_inlier = nInlier;
            inlier = in1;
            InlrRatio = (double)max_inlier / (double)ptNum + 1e-16;
            iterNUM = static_cast<int>(std::floor(std::log(1 - ransacCoef.thInlrRatio) / std::log(1 - std::pow(InlrRatio, ransacCoef.minPtNum))));
        }
    }
    std::cerr << "Ransac iterations: " << std::min(ransacCoef.iterMax, iterNUM) << std::endl;

    if (max_inlier == 0){
        inlier.clear();
        outlier.clear();
    }else{
        sample.clear();
        for (uint32_t i = 0; i < inlier.size(); i++)
            if (inlier[i])
                sample.push_back(samples[i]);
        
        ransacCoef.calculate_func(sample, val);
        ransacCoef.calculate_dist(val, samples, dist);

        inlier.clear();
        outlier.clear();
        for (uint32_t i = 0; i < dist.size(); i++){
            inlier.push_back(dist[i] < ransacCoef.thDist);
            outlier.push_back(dist[i] > ransacCoef.thDistOut);
        }
    }
}

std::vector<uint32_t> MVO::randperm(uint32_t ptNum, int minPtNum){
    std::vector<uint32_t> vector;
    for (uint32_t i = 0; i < ptNum; i++)
        vector.push_back(i);
    std::random_shuffle(vector.begin(), vector.end());
    std::vector<uint32_t> sample(vector.begin(), vector.begin()+minPtNum);
    return sample;
}

std::vector<uint32_t> MVO::randweightedpick(const std::vector<double> &h, int n /*=1*/){
    int u = h.size();
    int s_under;
    double sum, rand_num;
    std::vector<double> H = h;
    std::vector<double> Hs, Hsc;
    std::vector<uint32_t> result;

    n = std::min(std::max(1, n), u);
    std::vector<int> HI(u, 0);          // vector with #u ints.
    std::iota(HI.begin(), HI.end(), 0); // Fill with 0, ..., u-1.
    
    for (int i = 0; i < n; i++){
        // initial variables
        Hs.clear();
        Hsc.clear();
        // random weight
        sum = std::accumulate(H.begin(), H.end(), 0.0);
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

void MVO::calculate_scale(const std::vector<std::pair<cv::Point3f,cv::Point3f>> &pts, double& scale){
    double sum = 0;
    for (uint32_t i = 0; i < pts.size(); i++){
        sum += (pts[i].first.x * pts[i].second.x + pts[i].first.y * pts[i].second.y + pts[i].first.z * pts[i].second.z + ::scale_reference) / 
        (pts[i].first.x * pts[i].first.x + pts[i].first.y * pts[i].first.y + pts[i].first.z * pts[i].first.z + 1);
    }
    scale = sum / pts.size();

    // double num = 0, den = 0;
    // for (uint32_t i = 0; i < pts.size(); i++){
    //     num += (pts[i].first.x * pts[i].second.x + pts[i].first.y * pts[i].second.y + pts[i].first.z * pts[i].second.z);
    //     den += (pts[i].first.x * pts[i].first.x + pts[i].first.y * pts[i].first.y + pts[i].first.z * pts[i].first.z);
    // }
    // scale = (num + pts.size() * ::scale_reference) / (den + pts.size());

    // scale = ::scale_reference;
}

void MVO::calculate_scale_error(const double& scale, const std::vector<std::pair<cv::Point3f,cv::Point3f>> &pts, std::vector<double>& dist){
    dist.clear();
    for (uint32_t i = 0; i < pts.size(); i++)
        dist.push_back(cv::norm(pts[i].second - scale * pts[i].first));
}

void MVO::calculate_plane(const std::vector<cv::Point3f>& pts, std::vector<double>& plane){
    // need exact three points
    // return plane's unit normal vector (a, b, c) and distance from origin (d): ax + by + cz + d = 0
    plane.clear();

    if( pts.size() == 3 ){
        Eigen::Vector3d a, b;
        a << pts[1].x - pts[0].x, pts[1].y - pts[0].y, pts[1].z - pts[0].z;
        b << pts[2].x - pts[0].x, pts[2].y - pts[0].y, pts[2].z - pts[0].z;
        
        Eigen::Vector3d n = a.cross(b);
        n /= n.norm();

        double d = - n(0)*pts[0].x - n(1)*pts[0].y - n(2)*pts[0].z;
        plane.push_back(n(0));
        plane.push_back(n(1));
        plane.push_back(n(2));
        plane.push_back(d);

    }else{
        cv::Point3f centroid(0,0,0);
        for( uint32_t i = 0; i < pts.size(); i++ ){
            centroid.x += pts[i].x;
            centroid.y += pts[i].y;
            centroid.z += pts[i].z;
        }
        centroid.x = centroid.x/pts.size();
        centroid.y = centroid.y/pts.size();
        centroid.z = centroid.z/pts.size();

        double xx = 0, xy = 0, xz = 0, yy = 0, yz = 0, zz = 0;
        for( uint32_t i = 0; i < pts.size(); i++ ){
            cv::Point3f r = pts[i] - centroid;
            xx += r.x * r.x;
            xy += r.x * r.y;
            xz += r.x * r.z;
            yy += r.y * r.y;
            yz += r.y * r.z;
            zz += r.z * r.z;
        }

        double det_x = yy*zz - yz*yz;
        double det_y = xx*zz - xz*xz;
        double det_z = xx*yy - xy*xy;

        double max = std::max(std::max(det_x, det_y), det_z);

        if( max > 0 ){
            Eigen::Vector3d n;
            if( max == det_x )
                n << det_x, xz*yz - xy*zz, xy*yz - xz*yy;
            else if( max == det_y )
                n << xz*yz - xy*zz, det_y, xy*xz - yz*xx;
            else if( max == det_z )
                n << xy*yz - xz*yy, xy*xz - yz*xx, det_z;

            double norm = n.norm();
            n /= norm;
            
            double d;
            d = - n(0)*centroid.x - n(1)*centroid.y - n(2)*centroid.z;

            plane.push_back(n(0));
            plane.push_back(n(1));
            plane.push_back(n(2));
            plane.push_back(d);
        }else{
            plane.push_back(0);
            plane.push_back(0);
            plane.push_back(0);
            plane.push_back(0);
        }
    }
}

void MVO::calculate_plane_error(const std::vector<double>& plane, const std::vector<cv::Point3f>& pts, std::vector<double>& dist){
    double norm = std::sqrt(std::pow(plane[0],2) + std::pow(plane[1],2) + std::pow(plane[2],2));
    double a = plane[0]/norm;
    double b = plane[1]/norm;
    double c = plane[2]/norm;
    double d = plane[3]/norm;

    dist.clear();
    for( uint32_t i = 0; i < pts.size(); i++ )
        dist.push_back(std::abs(a * pts[i].x + b * pts[i].y + c * pts[i].z + d));
}