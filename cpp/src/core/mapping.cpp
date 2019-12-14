#include "core/MVO.hpp"
#include "core/ransac.hpp"
#include "core/random.hpp"
#include "core/numerics.hpp"
#include "core/time.hpp"
#include "core/DepthFilter.hpp"

bool MVO::calculateMotion()
{
    if (!is_start_)
        return true;

    /**************************************************
     * Solve two-fold ambiguity
     **************************************************/
    Eigen::Matrix3d R_unique;
    Eigen::Vector3d t_unique;

    if( !verifySolutions(R_vec_, t_vec_, R_unique, t_unique) ) return false;
    if( MVO::s_file_logger.is_open() ) MVO::s_file_logger << "# Verify unique pose: " << lsi::toc() << std::endl;

    /**************************************************
     * Mapping
     **************************************************/
    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    Eigen::Matrix4d T, Toc;
    Eigen::Vector4d Poc;
    std::vector<bool> inlier, outlier;
	std::vector<int> idx_inlier, idx_outlier;
    
    switch (params_.mapping_option) {
    case 0:
        num_feature_inlier_ = num_feature_3D_reconstructed_;

        /**** mapping without scaling ****/
        update3DPoints(R_unique, t_unique, inlier, outlier, T, Toc, Poc);
        if( MVO::s_file_logger.is_open() ) MVO::s_file_logger << "# Update 3D Points with Essential: " << lsi::toc() << std::endl;

        break;

    case 1:
        /**** mapping and scaling with essential 3d reconstruction only ****/
        if( !scalePropagation(R_unique ,t_unique, inlier, outlier) ) return false;

        update3DPoints(R_unique, t_unique, inlier, outlier, T, Toc, Poc); // overloading function
        if( MVO::s_file_logger.is_open() ) MVO::s_file_logger << "# Update 3D Points with Essential: " << lsi::toc() << std::endl;

        break;

    case 2:
        /**** mapping and scaling with PnP only ****/
        if( is_scale_initialized_ == true ){
            findPoseFrom3DPoints(R, t, idx_inlier, idx_outlier);
        }else{
            R = R_unique;
            t = t_unique;
            num_feature_inlier_ = features_.size();
        }

        update3DPoints(R, t, inlier, outlier, R_unique, t_unique, false, T, Toc, Poc); // overloading function
        break;

    case 3:
        /**** use both PnP and essential 3d reconstruction - original ****/
        if (findPoseFrom3DPoints(R, t, idx_inlier, idx_outlier)){
            if( MVO::s_file_logger.is_open() ) MVO::s_file_logger << "# Find pose from PnP: " << lsi::toc() << std::endl;

            // Update 3D points
            bool success = scalePropagation(R_unique, t_unique, inlier, outlier);
            
            // Update 3D points
            update3DPoints(R, t, inlier, outlier, R_unique, t_unique, success, T, Toc, Poc); // overloading function
            if( MVO::s_file_logger.is_open() ) MVO::s_file_logger << "# Update 3D Points with PnP: " << lsi::toc() << std::endl;

        }else{
            if( MVO::s_file_logger.is_open() ) MVO::s_file_logger << "# Find pose from PnP: " << lsi::toc() << std::endl;

            // Update 3D points
            if( !scalePropagation(R_unique ,t_unique, inlier, outlier) ) return false;

            update3DPoints(R_unique, t_unique, inlier, outlier, T, Toc, Poc); // overloading function
            if( MVO::s_file_logger.is_open() ) MVO::s_file_logger << "# Update 3D Points with Essential: " << lsi::toc() << std::endl;
        }
        break;

    case 4:
        /**** use both PnP and essential 3d reconstruction - modified ****/
        if( scalePropagation(R_unique, t_unique, inlier, outlier) ){
            if( MVO::s_file_logger.is_open() ) MVO::s_file_logger << "# Find scale from Essential: " << lsi::toc() << std::endl;

            R = R_unique;
            t = t_unique;

            findPoseFrom3DPoints(R, t, idx_inlier, idx_outlier);
            if( MVO::s_file_logger.is_open() ) MVO::s_file_logger << "# Find pose from PnP: " << lsi::toc() << std::endl;

            // Update 3D points
            update3DPoints(R, t, inlier, outlier, R_unique, t_unique, true, T, Toc, Poc);
        }else{
            if( MVO::s_file_logger.is_open() ) MVO::s_file_logger << "# Find scale from Essential: " << lsi::toc() << std::endl;

            findPoseFrom3DPoints(R, t, idx_inlier, idx_outlier);
            if( MVO::s_file_logger.is_open() ) MVO::s_file_logger << "# Find pose from PnP: " << lsi::toc() << std::endl;

            // Update 3D points
            update3DPoints(R, t, inlier, outlier, R_unique, t_unique, false, T, Toc, Poc);
        }
        break;
    }
    if( MVO::s_file_logger.is_open() ) MVO::s_file_logger << "percentage_feature_3D_reconstructed_: " << (double) num_feature_3D_reconstructed_ / num_feature_ * 100 << '%' << std::endl;
    if( MVO::s_file_logger.is_open() ) MVO::s_file_logger << "percentage_feature_3D_inliered: " << (double) num_feature_inlier_ / num_feature_ * 100 << '%' << std::endl;
    if( MVO::s_file_logger.is_open() ) MVO::s_file_logger << "num_feature_inlier_: " << num_feature_inlier_ << " " << std::endl;

    /**** ****/
    if (num_feature_inlier_ < params_.th_inlier){
        if( MVO::s_file_logger.is_open() ) MVO::s_file_logger << "Warning: There are few inliers reconstructed and accorded in 3D" << std::endl;
        return false;
    }else{
        if( MVO::s_file_logger.is_open() ) MVO::s_file_logger << "Temporal velocity: " << T.block(0,3,3,1).norm() << std::endl;
        if( MVO::s_file_logger.is_open() ) MVO::s_file_logger << "Optimized scale: " << t_unique.norm() << std::endl;

        // Save solution
        TRec_.push_back(T);
        TocRec_.push_back(Toc);
        PocRec_.push_back(Poc);

        is_scale_initialized_ = true;
        return true;
    }
}

bool MVO::verifySolutions(const std::vector<Eigen::Matrix3d>& R_vec, const std::vector<Eigen::Vector3d>& t_vec, Eigen::Matrix3d& R, Eigen::Vector3d& t){
    
	bool success;

	// Extract homogeneous 2D point which is inliered with essential constraint
	std::vector<int> idx_2D_inlier;
	for( int i = 0; i < num_feature_; i++ )
		if( features_[i].is_2D_inliered && features_[i].life - 1 - (step_ - keystep_) >= 0)
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
	std::vector<bool> max_inlier;
	std::vector<Eigen::Vector3d> opt_X_curr;
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

	// Store 3D characteristics in features
	if( max_num < num_feature_2D_inliered_*0.5 ){
        if( MVO::s_file_logger.is_open() ) MVO::s_file_logger << "Warning: There is no verified solution" << std::endl;
		success = false;
	}else{
		for( int i = 0; i < max_num; i++ ){
			if( max_inlier[i] ){
				features_[idx_2D_inlier[i]].point_curr = (Eigen::Vector4d() << opt_X_curr[i], 1).finished();
				features_[idx_2D_inlier[i]].is_3D_reconstructed = true;
                num_feature_3D_reconstructed_++;
            }
		}
		success = true;
	}

	return success;
}

bool MVO::findPoseFrom3DPoints(Eigen::Matrix3d &R, Eigen::Vector3d &t, std::vector<int>& idx_inlier, std::vector<int>& idx_outlier){    
    // Seek index of which feature is 3D reconstructed currently,
    // and 3D initialized previously
	bool flag;

    std::vector<int> idx;
    for (int i = 0; i < num_feature_; i++)
        if (features_[i].is_3D_init)
            idx.push_back(i);
    
    const uint32_t num_pts = idx.size();

    // Use RANSAC to find suitable scale
    if (num_pts > (uint32_t)params_.th_inlier){
        std::vector<cv::Point3f> object_pts;
        std::vector<cv::Point2f> image_pts;
        for (uint32_t i = 0; i < num_pts; i++)
        {
            object_pts.emplace_back(features_[idx[i]].point_init(0), features_[idx[i]].point_init(1), features_[idx[i]].point_init(2));
            image_pts.push_back(features_[idx[i]].uv.back()); // return last element of uv
        }

        // r_vec = cv::Mat::zeros(3,1,CV_32F);
        // t_vec = cv::Mat::zeros(3,1,CV_32F);

        cv::Mat R_cv, r_vec, t_vec;
        if( R.determinant() < .1){
            Eigen::Matrix3d R_prev = TocRec_.back().block(0,0,3,3).transpose();
            Eigen::Vector3d t_prev = -R_prev * TocRec_.back().block(0,3,3,1);
            Eigen::Vector3d speed_prev = TRec_.back().block(0,3,3,1);

            cv::eigen2cv(R_prev, R_cv);
            cv::eigen2cv((Eigen::Vector3d) (t_prev - speed_prev), t_vec);
            cv::Rodrigues(R_cv, r_vec);
        }else{
            cv::eigen2cv(R, R_cv);
            cv::eigen2cv(t, t_vec);
            cv::Rodrigues(R_cv, r_vec);
        }

        switch( params_.pnp_method ){
            case MVO::PNP::LM : 
            case MVO::PNP::ITERATIVE : {
                flag = cv::solvePnP(object_pts, image_pts, params_.Kcv, cv::noArray(), r_vec, t_vec, true, cv::SOLVEPNP_ITERATIVE);
                break;
            }
            case MVO::PNP::AP3P : {
                flag = cv::solvePnPRansac(object_pts, image_pts, params_.Kcv, cv::noArray(),
                                                r_vec, t_vec, true, 1e3,
                                                params_.reproj_error, 0.99, idx_inlier, cv::SOLVEPNP_AP3P);
                if (!flag){
                    flag = cv::solvePnPRansac(object_pts, image_pts, params_.Kcv, cv::noArray(),
                                                r_vec, t_vec, true, 1e3,
                                                2 * params_.reproj_error, 0.99, idx_inlier, cv::SOLVEPNP_AP3P);
                }
                break;
            }
            case MVO::PNP::EPNP : {
                flag = cv::solvePnPRansac(object_pts, image_pts, params_.Kcv, cv::noArray(),
                                                r_vec, t_vec, true, 1e3,
                                                params_.reproj_error, 0.99, idx_inlier, cv::SOLVEPNP_EPNP);
                if (!flag){
                    flag = cv::solvePnPRansac(object_pts, image_pts, params_.Kcv, cv::noArray(),
                                                r_vec, t_vec, true, 1e3,
                                                2 * params_.reproj_error, 0.99, idx_inlier, cv::SOLVEPNP_EPNP);
                }
                break;
            }
            case MVO::PNP::DLS : {
                flag = cv::solvePnPRansac(object_pts, image_pts, params_.Kcv, cv::noArray(),
                                                r_vec, t_vec, true, 1e3,
                                                params_.reproj_error, 0.99, idx_inlier, cv::SOLVEPNP_DLS);
                if (!flag){
                    flag = cv::solvePnPRansac(object_pts, image_pts, params_.Kcv, cv::noArray(),
                                                r_vec, t_vec, true, 1e3,
                                                2 * params_.reproj_error, 0.99, idx_inlier, cv::SOLVEPNP_DLS);
                }
                break;
            }
            case MVO::PNP::UPNP : {
                flag = cv::solvePnPRansac(object_pts, image_pts, params_.Kcv, cv::noArray(),
                                                r_vec, t_vec, true, 1e3,
                                                params_.reproj_error, 0.99, idx_inlier, cv::SOLVEPNP_UPNP);
                if (!flag){
                    flag = cv::solvePnPRansac(object_pts, image_pts, params_.Kcv, cv::noArray(),
                                                r_vec, t_vec, true, 1e3,
                                                2 * params_.reproj_error, 0.99, idx_inlier, cv::SOLVEPNP_UPNP);
                }
                break;
            }
        }
        if( MVO::s_file_logger.is_open() ) MVO::s_file_logger << "## Solve PnP: " << lsi::toc() << std::endl;

        cv::Rodrigues(r_vec, R_cv);
        cv::cv2eigen(R_cv, R);
        cv::cv2eigen(t_vec, t);

        if( params_.pnp_method == MVO::PNP::LM || params_.pnp_method == MVO::PNP::ITERATIVE ){
            num_feature_inlier_ = object_pts.size();
        }
        else{
            idx_outlier.clear();
            if( idx_inlier.empty() ){
                for (int i = 0; i < (int)num_pts; i++)
                    idx_outlier.push_back(i);
            }else{
                int num_inlier = 0;
                for (int i = 0; i < (int)num_pts; i++){
                    if (idx_inlier[num_inlier] == i)
                        num_inlier++;
                    else
                        idx_outlier.push_back(i);
                }
                num_feature_inlier_ = num_inlier;
            }
        }
    }
    else{
        idx_inlier.clear();
        idx_outlier.clear();
        R = Eigen::Matrix3d::Identity();
        t = Eigen::Vector3d::Zero();
        flag = false;
    }
    return flag;
}

void MVO::constructDepth(const std::vector<cv::Point2f>& uv_prev, const std::vector<cv::Point2f>& uv_curr, 
                        const Eigen::Matrix3d& R, const Eigen::Vector3d& t, 
                        std::vector<Eigen::Vector3d> &X_prev, std::vector<Eigen::Vector3d> &X_curr, 
                        std::vector<bool> &inlier){

    if( MVO::s_file_logger.is_open() ) MVO::s_file_logger << "## Init constructDepth: " << lsi::toc() << std::endl;
    const uint32_t num_pts = uv_prev.size();

    switch(params_.triangulation_method){
        case MVO::TRIANGULATION::MIDP : {
            Eigen::Matrix<double,3,3> A, A0, A1;
            Eigen::Vector3d b;
            
            // Eigen::Matrix<double,3,4> P0, P1;
            // P0 << params_.K, Eigen::Vector3d::Zero();
            // P1 << params_.K * R, params_.K * t;

            Eigen::Matrix3d M0, M1;
            Eigen::Vector3d c0, c1, u0, u1;
            double lambda0, lambda1;
            double u0_norm, u1_norm, u0tu1;

            // M0 = P0.block(0,0,3,3).inverse();
            // c0 = -M0*P0.block(0,3,3,1);
            // M1 = P1.block(0,0,3,3).inverse();
            // c1 = -M1*P1.block(0,3,3,1);
            M0 = params_.Kinv;
            M1 = R.inverse() * params_.Kinv;
            // c0 = Eigen::Vector3d::Zero();
            c1 = -R.inverse() * t;

            for( uint32_t i = 0; i < num_pts; i++ ){
                u0 = M0 * (Eigen::Vector3d() << uv_prev[i].x, uv_prev[i].y, 1).finished();
                u1 = M1 * (Eigen::Vector3d() << uv_curr[i].x, uv_curr[i].y, 1).finished();

                // Simplified verseion
                A0 = Eigen::Matrix3d::Identity() - u0 * u0.transpose() / (u0.cwiseProduct(u0)).sum();
                A1 = Eigen::Matrix3d::Identity() - u1 * u1.transpose() / (u1.cwiseProduct(u1)).sum();

                A = A0 + A1;
                b = A1 * c1; // A0 * c0 + A1 * c1 = A1 * c1, because c0 = 0

                X_prev.push_back(A.inverse() * b);
                X_curr.push_back(R * X_prev.back() + t);
                inlier.push_back(X_prev.back()(2) > 0 && X_curr.back()(2) > 0);

                // Our version
                // u0_norm = (u0.cwiseProduct(u0)).sum();
                // u1_norm = (u1.cwiseProduct(u1)).sum();
                // u0tu1 = (u0.cwiseProduct(u1)).sum();

                // lambda0 = ( u0tu1*u1.transpose() - u1_norm*u0.transpose() ) / (u0_norm*u1_norm - u0tu1*u0tu1) * (c0 - c1);
                // lambda1 = ( u0_norm*u1.transpose() - u0tu1*u0.transpose() ) / (u0_norm*u1_norm - u0tu1*u0tu1) * (c0 - c1);

                // X_prev.push_back(lambda0 * u0);
                // X_curr.push_back(lambda1 * u1);
                // inlier.push_back(X_prev.back()(2) > 0 && X_curr.back()(2) > 0);
            }
            if( MVO::s_file_logger.is_open() ) MVO::s_file_logger << "## Reconstruct 3D points: " << lsi::toc() << std::endl;
            break;
        }

        case MVO::TRIANGULATION::LLS : {
            std::vector<Eigen::Vector3d> x_prev, x_curr;
            for( uint32_t i = 0; i < uv_prev.size(); i++ ){
                x_prev.emplace_back((uv_prev[i].x - params_.cx)/params_.fx, (uv_prev[i].y - params_.cy)/params_.fy, 1);
                x_curr.emplace_back((uv_curr[i].x - params_.cx)/params_.fx, (uv_curr[i].y - params_.cy)/params_.fy, 1);
            }

            Eigen::MatrixXd& M_matrix = map_matrix_template_;
            M_matrix.resize(3*num_pts, num_pts+1);
            M_matrix.setZero();

            for (uint32_t i = 0; i < num_pts; i++){
                M_matrix.block(3*i,i,3,1) = skew(x_curr[i])*R*x_prev[i];
                M_matrix.block(3*i,num_pts,3,1) = skew(x_curr[i])*t;
            }
            if( MVO::s_file_logger.is_open() ) MVO::s_file_logger << "## Construct MtM: " << lsi::toc() << std::endl;

            Eigen::MatrixXd V;
            uint32_t idx_minimum_eigenval = num_pts;
            switch( params_.svd_method){
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
                    eigen_solver_->compute(MtM_, Eigen::ComputeEigenvectors);

                    V = eigen_solver_->eigenvectors();
                    idx_minimum_eigenval = 0;
                    break;
                }
            }
            if( MVO::s_file_logger.is_open() ) MVO::s_file_logger << "## Compute eigenvector: " << lsi::toc() << std::endl;

            for (uint32_t i = 0; i < num_pts; i++){
                X_prev.push_back( V(i,idx_minimum_eigenval) / V(num_pts,idx_minimum_eigenval) * x_prev[i] );
                X_curr.push_back( R*X_prev.back() + t );
                inlier.push_back( X_prev.back()(2) > 0 && X_curr.back()(2) > 0 );
            }
        }
    }
}

void MVO::update3DPoint(Feature& feature, const Eigen::Matrix4d& Toc, const Eigen::Matrix4d& T){
    // // Update point under Gaussian model assumption
    // int key_idx = feature.life - 1 - (step_ - this>keystep_);
    // double cur_var = 5/(cv::norm(feature.uv[key_idx] - feature.uv.back()));
    // if( feature.is_3D_init ){
    //     if( params_.update_init_point ){
    //         double diffX = (Toc * feature.point_curr - feature.point_init).norm();
    //         if( diffX < 3*feature.point_var){
    //             double prv_var = feature.point_var;
    //             double new_var = 1 / ( (1 / prv_var) + (1 / cur_var) );
                
    //             feature.point_init = new_var/cur_var * Toc * feature.point_curr + new_var/prv_var * feature.point_init;
    //             feature.point_var = new_var;
    //         }
    //     }
    // }else if( feature.is_wide ){
    //     feature.point_init = Toc * feature.point;
    //     feature.point_var = cur_var;
    //     feature.is_3D_init = true;
    //     feature.frame_3d_init = keystep_;
    // }

    // Use Depthfilter - combination of Gaussian and uniform model
    Eigen::Vector3d point_initframe;
    double z, tau, tau_inverse;

    if( feature.is_3D_init ){
        Eigen::Matrix4d Tkc = TocRec_[feature.frame_3d_init].inverse() * Toc;
        point_initframe = Tkc.block(0,0,3,4) * feature.point_curr;
        z = point_initframe(2);

        tau = DepthFilter::computeTau(Tkc, point_initframe);
        tau_inverse = DepthFilter::computeInverseTau(z, tau);

        if( params_.update_init_point ){
            feature.depthfilter->update(1/z, tau_inverse);
            feature.point_init = TocRec_[feature.frame_3d_init] * (Eigen::Vector4d() << point_initframe / z / feature.depthfilter->getMean(), 1).finished();
            feature.point_var = feature.depthfilter->getVariance();
        }

    }else{ // if( feature.is_wide ){
        point_initframe = T.block(0,0,3,4) * feature.point_curr;
        z = point_initframe(2);

        tau = DepthFilter::computeTau(T, point_initframe);
        tau_inverse = DepthFilter::computeInverseTau(z, tau);

        feature.depthfilter->update(1/z, tau_inverse);
        feature.point_init = Toc * T.inverse() * (Eigen::Vector4d() << point_initframe / z / feature.depthfilter->getMean(), 1).finished();
        feature.point_var = feature.depthfilter->getVariance();
        feature.is_3D_init = true;
        feature.frame_3d_init = keystep_;
    }

    if( feature.point_var > params_.max_point_var )
        feature.is_alive = false;

    // if( feature.id < 100)
    //     if( MVO::s_file_logger.is_open() ) MVO::s_file_logger << "& " << feature.id << " " << z << " " << feature.depthfilter->getMean() << " " << tau << " " << tau_inverse << " " << feature.point_var << " " << feature.depthfilter->getA() << " " << feature.depthfilter->getB() << std::endl;
}

// without PnP
void MVO::update3DPoints(const Eigen::Matrix3d &R, const Eigen::Vector3d &t, 
                        const std::vector<bool> &inlier, const std::vector<bool> &outlier, 
                        Eigen::Matrix4d &T, Eigen::Matrix4d &Toc, Eigen::Vector4d &Poc){
    // without PnP
    double scale = t.norm();
    
    // Seek index of which feature is 3D reconstructed currently
    // and 3D initialized previously
    Eigen::Matrix4d tform;
    tform.setIdentity();
    tform.block(0,0,3,3) = R.transpose();
    tform.block(0,3,3,1) = -R.transpose()*t;
    Toc = TocRec_[keystep_] * tform;
    Poc = Toc.block(0,3,4,1);
    T = TocRec_.back().inverse() * Toc;
    
    for( uint32_t i = 0; i < features_.size(); i++ ){
        // TODO: features_[i].is_3D_reconstructed && inlier[i]: inlier is classified by hard-manner, recommend soft-manner using point-variance
        if( features_[i].is_3D_reconstructed ){
            features_[i].point_curr.block(0,0,3,1) *= scale;
            update3DPoint(features_[i], Toc, tform);
        }
    }
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
    T = TocRec_.back().inverse() * Toc;

    Eigen::Matrix4d Tco;
    Tco.setIdentity();
    Tco.block(0,0,3,3) = R;
    Tco.block(0,3,3,1) = t;

    if(success_E){
        double scale = t_E.norm();

        for( uint32_t i = 0; i < features_.size(); i++ ){
            if( features_[i].is_3D_reconstructed ){
                features_[i].point_curr.block(0,0,3,1) *= scale;
                update3DPoint(features_[i], Toc, TocRec_[keystep_].inverse() * Toc);
            }
        }
    }else{
        Eigen::Matrix3d Rinv;
        Eigen::Vector3d tinv;

        Rinv = T.block(0,0,3,3).transpose();
        tinv = -T.block(0,0,3,3).transpose()*T.block(0,3,3,1);
        
        // Initialize 3D points in global coordinates
        // Extract Homogeneous 2D point which is inliered with essential constraint
        std::vector<int> idx_2D;
        for (uint32_t i = 0; i < features_.size(); i++){
            if (features_[i].is_2D_inliered)
                idx_2D.push_back(i);
        }
        
        const uint32_t num_pts = idx_2D.size();
        int len;
        std::vector<cv::Point2f> uv_prev, uv_curr;
        for (uint32_t i = 0; i < num_pts; i++){
            len = features_[idx_2D[i]].life;
            uv_prev.emplace_back(features_[idx_2D[i]].uv[len-2]);
            uv_curr.emplace_back(features_[idx_2D[i]].uv.back());
        }
        
        std::vector<Eigen::Vector3d> X_prev, X_curr;
        std::vector<bool> pnp_inlier;
        constructDepth(uv_prev, uv_curr, Rinv, tinv, X_prev, X_curr, pnp_inlier);

        for (uint32_t i = 0; i < num_pts; i++){
            // 2d_inlier
            if(pnp_inlier[i]){
                features_[idx_2D[i]].point_curr = (Eigen::Vector4d() << X_curr[i], 1).finished();
				features_[idx_2D[i]].is_3D_reconstructed = true;
                update3DPoint(features_[idx_2D[i]], Toc, T);
            } // if(lambda_prev > 0 && lambda_curr > 0)
        } // for
        num_feature_inlier_ = std::count(pnp_inlier.begin(), pnp_inlier.end(), true);
    }

    int num_reconstructed = 0;
    for (uint32_t i = 0; i < features_.size(); i++){
        if(features_[i].is_3D_reconstructed)
            num_reconstructed++;
    }
    num_feature_3D_reconstructed_ = num_reconstructed;
}

bool MVO::scalePropagation(const Eigen::Matrix3d &R, Eigen::Vector3d &t, std::vector<bool> &inlier, std::vector<bool> &outlier){
    inlier.clear();
    outlier.clear();

    double scale = 0, scale_from_height = 0;
    bool flag;

    // Initialization
    // initialze scale, in the case of the first time
    if( !is_speed_provided_ ){
        cv::Point2f uv_curr;
        Eigen::Vector4d point_curr;
        std::vector<cv::Point3f> road_candidate;
        std::vector<uint32_t> road_idx;

        for (int i = 0; i < num_feature_; i++){
            if( features_[i].is_3D_reconstructed ){
                uv_curr = features_[i].uv.back(); //latest feature

                if (uv_curr.y > params_.im_size.height * 0.5 
                && uv_curr.y > params_.im_size.height - 0.7 * uv_curr.x 
                && uv_curr.y > params_.im_size.height + 0.7 * (uv_curr.x - params_.im_size.width)){
                    point_curr = features_[i].point_curr;
                    road_candidate.emplace_back(point_curr(0),point_curr(1),point_curr(2));
                    road_idx.push_back(i);
                }
            }
        }

        if( road_idx.size() > (uint32_t) params_.th_inlier ){
            std::vector<double> plane;
            std::vector<bool> plane_inlier, plane_outlier;
            lsi::ransac<cv::Point3f, std::vector<double>>(road_candidate, params_.ransac_coef_plane, plane, plane_inlier, plane_outlier);

            if( std::count(plane_inlier.begin(), plane_inlier.end(), true) > (uint32_t) params_.th_inlier ){
                for( uint32_t i = 0; i < road_idx.size(); i++ ){
                    if( plane_inlier[i] )
                        features_[road_idx[i]].type = Type::Road;
                    // else
                    //     features_[road_idx[i]].type = Type::Unknown;
                }
                scale_from_height = params_.vehicle_height / std::abs(plane[3]);
                if( MVO::s_file_logger.is_open() ) MVO::s_file_logger << "Plane: " << plane[0] << ',' << plane[1] << ',' << plane[2] << std::endl;

                updateScaleReference(scale_from_height);
            }
        }
    }

    if (is_scale_initialized_)
    {
        // Seek index of which feature is 3D reconstructed currently,
        // and 3D initialized previously
        std::vector<int> idx;
        for (uint32_t i = 0; i < features_.size(); i++){
            if( features_[i].is_3D_reconstructed && features_[i].is_3D_init )
                idx.push_back(i);
        }
        uint32_t num_pts = idx.size();

        std::vector<bool> scale_inlier, scale_outlier;
        if( params_.weight_scale_ref < 0 ){ // Use reference scale directly
            scale = scale_reference_;
            if( MVO::s_file_logger.is_open() ) MVO::s_file_logger << "@ scale_from_velocity: " << scale_reference_ << std::endl;
            num_feature_inlier_ = num_feature_3D_reconstructed_;
        }
        else if ( num_pts > params_.ransac_coef_scale.min_num_point){ // Use RANSAC to find suitable scale
            Eigen::Matrix4d tform;
            tform.setIdentity();
            tform.block(0,0,3,3) = R;
            tform.block(0,3,3,1) = t;

            std::vector<std::pair<cv::Point3f,cv::Point3f>> Points;
            Eigen::Vector4d init_point, expt_point, curr_point;
            params_.ransac_coef_scale.weight.clear();
            
            for (uint32_t i = 0; i < num_pts; i++){
                curr_point = features_[idx[i]].point_curr;

                // Get initialized 3D point
                init_point = TocRec_[keystep_].inverse() * features_[idx[i]].point_init;
                
                // Get expected 3D point by transforming the coordinates of the observed 3d point
                expt_point = tform.inverse() * curr_point;

                Points.emplace_back(cv::Point3f(expt_point(0),expt_point(1),expt_point(2)),
                                    cv::Point3f(init_point(0),init_point(1),init_point(2)));
                
                // RANSAC weight
                params_.ransac_coef_scale.weight.push_back( std::atan( -curr_point(2)/5 + 3 ) + M_PI / 2 );
                // params_.ransac_coef_scale.weight.push_back( std::atan( -features_[i].point_var * 1e4 ) + M_PI / 2 );
                // params_.ransac_coef_scale.weight.push_back( 1/features_[i].point_var );
            }

            params_.ransac_coef_scale.calculate_func = std::bind(lsi::calculateScale, std::placeholders::_1, std::placeholders::_2, scale_reference_, params_.weight_scale_ref);
            lsi::ransac<std::pair<cv::Point3f,cv::Point3f>,double>(Points, params_.ransac_coef_scale, scale, scale_inlier, scale_outlier);
            num_feature_inlier_ = std::count(scale_inlier.begin(), scale_inlier.end(), true);

            if( MVO::s_file_logger.is_open() ) MVO::s_file_logger << "num_feature_inlier_: " << num_feature_inlier_ << std::endl;
        }

        // Use the previous scale, if the scale cannot be found
        // But do not use the previous scale when the velocity reference is fetched directly (params_.weight_scale_ref < 0)
        if( params_.weight_scale_ref < 0 ){
            // Update scale
            t = scale * t;
            flag = true;
        }else if ( params_.weight_scale_ref >= 0 && (num_pts <= params_.ransac_coef_scale.min_num_point || num_feature_inlier_ < (std::size_t)params_.th_inlier || scale == 0) )
        {
            if( MVO::s_file_logger.is_open() ) MVO::s_file_logger << "Warning: There are a few scale factor inliers" << std::endl;

            scale = (TRec_.back().block(0,3,3,1)).norm();

            // Update scale
            t = scale * t;
            flag = false;

        }else{
            if( MVO::s_file_logger.is_open() ) MVO::s_file_logger << "@ scale_from_height: " << scale_reference_ << ", " << "scale: " << scale << std::endl;

            for( uint32_t i = 0, j = 0; i < features_.size(), j < idx.size(); i++ ){
                if( i == idx[j] ){
                    if( scale_inlier[j] == true )
                        inlier.push_back(true);
                    else
                        inlier.push_back(false);
                    j++;
                }else
                    inlier.push_back(false);
            }

            // Update scale
            t = scale * t;
            flag = true;
        }

    }else{
        if( is_speed_provided_ )
            t = scale_reference_ * t;
        else
            t = scale_from_height * t;

        for( uint32_t i = 0; i < features_.size(); i++ ){
            if( features_[i].is_3D_reconstructed )
                inlier.push_back(true);
            else
                inlier.push_back(false);
        }
        num_feature_inlier_ = num_feature_3D_reconstructed_;
        flag = true;
    }

    if( MVO::s_file_logger.is_open() ) MVO::s_file_logger << "# Propagate scale: " << lsi::toc() << std::endl;

    if( t.hasNaN() ){
        if( MVO::s_file_logger.is_open() ) MVO::s_file_logger << "Warning: A scale value is nan" << std::endl;
        return false;
    }else
        return flag;
}

void MVO::updateScaleReference(const double scale){
    // scale_reference_ = scale;
    if( scale_reference_ < 0 || is_start_ == false )
        scale_reference_ = scale;
    else{
        // low-pass filter
        // scale_reference_ = params_.weightScaleReg * scale_reference_ + (1-params_.weightScaleReg) * scale;

        // limit slope
        scale_reference_ = scale_reference_ + ((scale > scale_reference_)?1:-1) * std::min(std::abs(scale - scale_reference_), params_.weight_scale_reg);
    }
}