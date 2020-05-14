#include "core/MVO.hpp"
#include "core/numerics.hpp"
#include "core/time.hpp"
#include "core/DepthFilter.hpp"

/**
 * @brief 프레임 사이의 변환 행렬의 스케일을 추정 및 보정하는 프로세스
 * @details 변환 행렬의 scale을 계산하고, R, t를 출력한다.
 * @return 에러가 발생하지 않으면, true
 * @author Sangil Lee (sangillee724@gmail.com)
 * @date 29-Dec-2019
 */
bool MVO::calculateMotion(const Eigen::Matrix3d & R_e, Eigen::Vector3d & t_e)
{
    if (!is_start_)
        return true;

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
        update3DPoints(R_e, t_e, inlier, outlier, T, Toc, Poc);
        if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "# Update 3D Points with Essential: " << lsi::toc() << std::endl;

        break;

    case 1:
        /**** mapping and scaling with essential 3d reconstruction only ****/
        if( !scalePropagation(R_e ,t_e, inlier, outlier) ) return false;

        update3DPoints(R_e, t_e, inlier, outlier, T, Toc, Poc); // overloading function
        if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "# Update 3D Points with Essential: " << lsi::toc() << std::endl;

        break;

    case 2:
        /**** mapping and scaling with PnP only ****/
        if( is_scale_initialized_ == true ){
            findPoseFrom3DPoints(R, t, idx_inlier, idx_outlier);
        }else{
            R = R_e;
            t = t_e;
            num_feature_inlier_ = features_.size();
        }

        update3DPoints(R, t, inlier, outlier, R_e, t_e, false, T, Toc, Poc); // overloading function
        break;

    case 3:
        /**** use both PnP and essential 3d reconstruction - original ****/
        if (findPoseFrom3DPoints(R, t, idx_inlier, idx_outlier)){
            if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "# Find pose from PnP: " << lsi::toc() << std::endl;

            // Update 3D points
            bool success = scalePropagation(R_e, t_e, inlier, outlier);
            
            // Update 3D points
            update3DPoints(R, t, inlier, outlier, R_e, t_e, success, T, Toc, Poc); // overloading function
            if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "# Update 3D Points with PnP: " << lsi::toc() << std::endl;

        }else{
            if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "# Find pose from PnP: " << lsi::toc() << std::endl;

            // Update 3D points
            if( !scalePropagation(R_e ,t_e, inlier, outlier) ) return false;

            update3DPoints(R_e, t_e, inlier, outlier, T, Toc, Poc); // overloading function
            if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "# Update 3D Points with Essential: " << lsi::toc() << std::endl;
        }
        break;

    case 4:
        /**** use both PnP and essential 3d reconstruction - modified ****/
        if( scalePropagation(R_e, t_e, inlier, outlier) ){
            if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "# Find scale from Essential: " << lsi::toc() << std::endl;

            R = R_e;
            t = t_e;

            findPoseFrom3DPoints(R, t, idx_inlier, idx_outlier);
            if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "# Find pose from PnP: " << lsi::toc() << std::endl;

            // Update 3D points
            update3DPoints(R, t, inlier, outlier, R_e, t_e, true, T, Toc, Poc);
        }else{
            if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "# Find scale from Essential: " << lsi::toc() << std::endl;

            findPoseFrom3DPoints(R, t, idx_inlier, idx_outlier);
            if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "# Find pose from PnP: " << lsi::toc() << std::endl;

            // Update 3D points
            update3DPoints(R, t, inlier, outlier, R_e, t_e, false, T, Toc, Poc);
        }
        break;
    }
    if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "percentage_feature_3D_reconstructed_: " << (double) num_feature_3D_reconstructed_ / num_feature_ * 100 << '%' << std::endl;
    if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "percentage_feature_3D_inliered: " << (double) num_feature_inlier_ / num_feature_ * 100 << '%' << std::endl;
    if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "num_feature_inlier_: " << num_feature_inlier_ << " " << std::endl;

    /**** return success or failure ****/
    if (num_feature_inlier_ < params_.th_inlier){
        if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "Warning: There are few inliers reconstructed and accorded in 3D" << std::endl;
        return false;
    }else{
        if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "Temporal velocity: " << T.block(0,3,3,1).norm() << std::endl;
        if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "Optimized scale: " << t_e.norm() << std::endl;

        // Save solution
        TRec_.push_back(T);
        TocRec_.push_back(Toc);
        PocRec_.push_back(Poc);

        is_scale_initialized_ = true;
        return true;
    }
}

/**
 * @brief 프레임 사이의 변환 행렬의 스케일을 추정 및 보정하는 프로세스
 * @details 변환 행렬의 scale을 계산하고, R, t를 출력한다.
 * @return 에러가 발생하지 않으면, true
 * @author Sangil Lee (sangillee724@gmail.com)
 * @date 8-May-2020
 */
bool MVO::calculateMotionStereo(const Eigen::Matrix3d & R_e, Eigen::Vector3d & t_e)
{
    if (!is_start_)
        return true;

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
        update3DPointsStereo(R_e, t_e, inlier, outlier, T, Toc, Poc);
        if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "# Update 3D Points with Essential: " << lsi::toc() << std::endl;

        break;

    case 1:
        /**** mapping and scaling with essential 3d reconstruction only ****/
        if( !scalePropagation(R_e ,t_e, inlier, outlier) ) return false;

        update3DPointsStereo(R_e, t_e, inlier, outlier, T, Toc, Poc); // overloading function
        if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "# Update 3D Points with Essential: " << lsi::toc() << std::endl;

        break;
    }
    if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "percentage_feature_3D_reconstructed_: " << (double) num_feature_3D_reconstructed_ / num_feature_ * 100 << '%' << std::endl;
    if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "percentage_feature_3D_inliered: " << (double) num_feature_inlier_ / num_feature_ * 100 << '%' << std::endl;
    if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "num_feature_inlier_: " << num_feature_inlier_ << " " << std::endl;

    /**** return success or failure ****/
    if (num_feature_inlier_ < params_.th_inlier){
        if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "Warning: There are few inliers reconstructed and accorded in 3D" << std::endl;
        return false;
    }else{
        if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "Temporal velocity: " << T.block(0,3,3,1).norm() << std::endl;
        if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "Optimized scale: " << t_e.norm() << std::endl;

        // Save solution
        TRec_.push_back(T);
        TocRec_.push_back(Toc);
        PocRec_.push_back(Poc);

        is_scale_initialized_ = true;
        return true;
    }
}

/**
 * @brief PnP 알고리즘
 * @details 3차원 좌표가 생성된 특징점들에 대해 PnP 알고리즘을 수행하여 자세값을 계산한다.
 * @param R 회전 행렬
 * @param t 변위 벡터
 * @param idx_inlier PnP 인라이어
 * @param idx_outlier PnP 아웃라이어
 * @return 성공적으로 자세를 계산하면, true
 * @author Sangil Lee (sangillee724@gmail.com)
 * @date 29-Dec-2019
 */
bool MVO::findPoseFrom3DPoints(Eigen::Matrix3d &R, Eigen::Vector3d &t, std::vector<int>& idx_inlier, std::vector<int>& idx_outlier){    
    // Seek index of which feature is 3D reconstructed currently,
    // and 3D initialized previously
	bool flag;

    // indices whose feature is initialized in 3d world coordinates
    std::vector<int> idx;
    for (int i = 0; i < num_feature_; i++)
        if (features_[i].landmark)
            idx.push_back(i);
    
    const uint32_t num_pts = idx.size();

    // Use RANSAC to find suitable scale
    if (num_pts > (uint32_t)params_.th_inlier){
        std::vector<cv::Point3f> object_pts;
        std::vector<cv::Point2f> image_pts;
        for (uint32_t i = 0; i < num_pts; i++)
        {
            object_pts.emplace_back(features_[idx[i]].landmark->point_init(0), 
                                    features_[idx[i]].landmark->point_init(1), 
                                    features_[idx[i]].landmark->point_init(2));
            image_pts.push_back(features_[idx[i]].uv.back()); // return last element of uv
        }

        // r_vec = cv::Mat::zeros(3,1,CV_32F);
        // t_vec = cv::Mat::zeros(3,1,CV_32F);

        // if the currently-estimated R, t is not accurate, use the previous R, t
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

        // solve PnP
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
        if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "## Solve PnP: " << lsi::toc() << std::endl;

        // change format
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

/**
 * @brief 삼각 측량
 * @details 인접한 프레임에서의 uv 좌표값과 R, t를 이용하여, 삼각 측량 후 포인트를 반환한다.
 * @param uv_prev 이전 uv 좌표값
 * @param uv_curr 현재 uv 좌표값
 * @param R 회전 행렬
 * @param t 변위 벡터
 * @param X_prev 이전 프레임에서의 xyz 좌표값
 * @param X_curr 현재 프레임에서의 xyz 좌표값
 * @param inlier 양의 깊이 값을 가지는 인라이어
 * @return 없음
 * @author Sangil Lee (sangillee724@gmail.com)
 * @date 29-Dec-2019
 */
void MVO::constructDepth(const std::vector<cv::Point2f>& uv_prev, const std::vector<cv::Point2f>& uv_curr, 
                        const Eigen::Matrix3d& R, const Eigen::Vector3d& t, 
                        std::vector<Eigen::Vector3d> &X_prev, std::vector<Eigen::Vector3d> &X_curr, 
                        std::vector<bool> &inlier){

    if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "## Init constructDepth: " << lsi::toc() << std::endl;
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
            if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "## Reconstruct 3D points: " << lsi::toc() << std::endl;
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
            if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "## Construct MtM: " << lsi::toc() << std::endl;

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
            if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "## Compute eigenvector: " << lsi::toc() << std::endl;

            for (uint32_t i = 0; i < num_pts; i++){
                X_prev.push_back( V(i,idx_minimum_eigenval) / V(num_pts,idx_minimum_eigenval) * x_prev[i] );
                X_curr.push_back( R*X_prev.back() + t );
                inlier.push_back( X_prev.back()(2) > 0 && X_curr.back()(2) > 0 );
            }
        }
    }
}

/**
 * @brief 3차원 좌표 업데이트
 * @details 각 특징점의 3차원 좌표값을 깊이 필터를 통해 업데이트한다.
 * @param feature 업데이트하고자 하는 특징점
 * @param Toc 원점 좌표계 기준 현재 자세 및 위치
 * @param T 키프레임 좌표계 기준 현재 자세 및 위치
 * @return 없음
 * @author Sangil Lee (sangillee724@gmail.com)
 * @date 29-Dec-2019
 */
void MVO::update3DPoint(Feature& feature, const Eigen::Matrix4d& Toc, const Eigen::Matrix4d& T){
    // Use Depthfilter - combination of Gaussian and uniform model
    Eigen::Vector3d point_initframe;
    double z, tau, tau_inverse;

    if( feature.landmark ){
        // compute depth seen from the keyframe
        Eigen::Matrix4d Tkc = TocRec_[feature.frame_3d_init].inverse() * Toc;
        point_initframe = Tkc.block(0,0,3,4) * feature.point_curr;
        z = point_initframe(2);

        // compute variance of inverse-depth
        tau = DepthFilter::computeTau(Tkc, point_initframe);
        tau_inverse = DepthFilter::computeInverseTau(z, tau);

        if( params_.update_init_point ){
            // apply the depth of depth filter; point_initframe/z is a homogeneous form of the current 3d position
            feature.depthfilter->update(1/z, tau_inverse);
            feature.landmark->point_init = TocRec_[feature.frame_3d_init] * (Eigen::Vector4d() << point_initframe / z / feature.depthfilter->getMean(), 1).finished();
        }

    }else{ // if( feature.is_wide ){
        // compute depth seen from the keyframe
        point_initframe = T.block(0,0,3,4) * feature.point_curr;
        z = point_initframe(2);

        tau = DepthFilter::computeTau(T, point_initframe);
        tau_inverse = DepthFilter::computeInverseTau(z, tau);

        feature.depthfilter->update(1/z, tau_inverse);

        // remove feature with abnormally-high variance
        if( feature.depthfilter->getVariance() > params_.max_point_var )
            feature.is_alive = false;
        else{
            feature.landmark = std::make_shared<Landmark>();
            feature.landmark->point_init = Toc * T.inverse() * (Eigen::Vector4d() << point_initframe / z / feature.depthfilter->getMean(), 1).finished();
            feature.frame_3d_init = keystep_; // reference frame to update depth filter
            landmark_.insert((std::make_pair(Landmark::getNewID(), feature.landmark)));
        }
    }

    // if( feature.id < 100)
    //     if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "& " << feature.id << " " << z << " " << feature.depthfilter->getMean() << " " << tau << " " << tau_inverse << " " << feature.point_var << " " << feature.depthfilter->getA() << " " << feature.depthfilter->getB() << std::endl;
}

/**
 * @brief 3차원 좌표 업데이트
 * @details 인접한 이미지 사이에서 essential constraint로 계산한 R, t를 이용해 각 특징점의 3차원 좌표값을 업데이트한다.
 * @param R 회전 행렬
 * @param t 변위 벡터
 * @param inlier 3차원 복원 인라이어
 * @param outlier 3차원 복원 실패 아웃라이어
 * @param T 직전 이미지 프레임 기준 변환 행렬
 * @param Toc 첫 이미지 프레임 기준 변환 행렬
 * @param Poc 첫 이미지 프레임 기준 위치 벡터
 * @return 없음
 * @author Sangil Lee (sangillee724@gmail.com)
 * @date 29-Dec-2019
 */
void MVO::update3DPoints(const Eigen::Matrix3d &R, const Eigen::Vector3d &t, 
                        const std::vector<bool> &inlier, const std::vector<bool> &outlier, 
                        Eigen::Matrix4d &T, Eigen::Matrix4d &Toc, Eigen::Vector4d &Poc){
    // without PnP
    double scale = t.norm();
    
    // Seek index of which feature is 3D reconstructed currently
    // and 3D initialized previously

    /* Find pose by accumulating transform, T */
    Eigen::Matrix4d tform;
    tform.setIdentity();
    tform.block(0,0,3,3) = R.transpose();
    tform.block(0,3,3,1) = -R.transpose()*t;
    Toc = TocRec_[keystep_] * tform;
    Poc = Toc.block(0,3,4,1);
    T = TocRec_.back().inverse() * Toc;
    
    /* Update point */
    for( uint32_t i = 0; i < features_.size(); i++ ){
        if( features_[i].is_3D_reconstructed ){
            features_[i].point_curr.block(0,0,3,1) *= scale;
            update3DPoint(features_[i], Toc, tform);
        }
    }
}

/**
 * @brief 3차원 좌표 업데이트
 * @details 인접한 이미지 사이에서 essential constraint로 계산한 R, t를 이용해 각 특징점의 3차원 좌표값을 업데이트한다.
 * @param R 회전 행렬
 * @param t 변위 벡터
 * @param inlier 3차원 복원 인라이어
 * @param outlier 3차원 복원 실패 아웃라이어
 * @param T 직전 이미지 프레임 기준 변환 행렬
 * @param Toc 첫 이미지 프레임 기준 변환 행렬
 * @param Poc 첫 이미지 프레임 기준 위치 벡터
 * @return 없음
 * @author Sangil Lee (sangillee724@gmail.com)
 * @date 8-May-2020
 */
void MVO::update3DPointsStereo(const Eigen::Matrix3d &R, const Eigen::Vector3d &t, 
                        const std::vector<bool> &inlier, const std::vector<bool> &outlier, 
                        Eigen::Matrix4d &T, Eigen::Matrix4d &Toc, Eigen::Vector4d &Poc){
    // Seek index of which feature is 3D reconstructed currently
    // and 3D initialized previously

    /* Find pose by accumulating transform, T */
    Eigen::Matrix4d tform;
    tform.setIdentity();
    tform.block(0,0,3,3) = R.transpose();
    tform.block(0,3,3,1) = -R.transpose()*t;
    Toc = TocRec_[keystep_] * tform;
    Poc = Toc.block(0,3,4,1);
    T = TocRec_.back().inverse() * Toc;
    
    /* Update point */
    for( uint32_t i = 0; i < features_.size(); i++ ){
        if( features_[i].is_3D_reconstructed ){
            update3DPoint(features_[i], Toc, tform);
        }
    }
}
	
/**
 * @brief 3차원 좌표 업데이트
 * @details 원점 좌표계의 3차원 좌표와 현재 프레임 사이에서 계산한 PnP의 자세값을 이용해 각 특징점의 3차원 좌표값을 업데이트한다.
 * @param R PnP로 얻은 회전 행렬
 * @param t PnP로 얻은 변위 벡터
 * @param inlier 3차원 복원 인라이어
 * @param outlier 3차원 복원 실패 아웃라이어
 * @param R_E 인접한 이미지 사이의 회전 행렬
 * @param t_E 인접한 이미지 사이의 변위 벡터
 * @param success_E 인접한 이미지 사이의 R, t 계산 성공 여부
 * @param T 직전 이미지 프레임 기준 변환 행렬
 * @param Toc 첫 이미지 프레임 기준 변환 행렬
 * @param Poc 첫 이미지 프레임 기준 위치 벡터
 * @return 없음
 * @author Sangil Lee (sangillee724@gmail.com)
 * @date 29-Dec-2019
 */
void MVO::update3DPoints(const Eigen::Matrix3d &R, const Eigen::Vector3d &t, 
                        const std::vector<bool> &inlier, const std::vector<bool> &outlier, 
                        const Eigen::Matrix3d &R_E, const Eigen::Vector3d &t_E, const bool &success_E, 
                        Eigen::Matrix4d &T, Eigen::Matrix4d &Toc, Eigen::Vector4d &Poc){
    // with PnP
    // Extract homogeneous 2D point which is inliered with essential constraint

    /* Find pose from PnP directly */
    Toc.setIdentity();
    Toc.block(0,0,3,3) = R.transpose();
    Toc.block(0,3,3,1) = -R.transpose()*t;
    Poc = Toc.block(0,3,4,1);
    T = TocRec_.back().inverse() * Toc;

    Eigen::Matrix4d Tco;
    Tco.setIdentity();
    Tco.block(0,0,3,3) = R;
    Tco.block(0,3,3,1) = t;

    /* Update point */
    if(success_E){ // using essential constraint
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

/**
 * @brief 변위 벡터 스케일 업데이트
 * @details 원점 좌표계의 3차원 좌표값과 현재 프레임에서의 3차원 좌표값을 비교하여 에러를 최소화하는 최적 스케일 값을 계산한다.
 * @param R 회전 행렬
 * @param t 변위 벡터
 * @param inlier 최적 스케일값을 만족하는 인라이어
 * @param outlier 아웃라이어
 * @return 최적값이 존재할 경우, true
 * @author Sangil Lee (sangillee724@gmail.com)
 * @date 29-Dec-2019
 */
bool MVO::scalePropagation(const Eigen::Matrix3d &R, Eigen::Vector3d &t, std::vector<bool> &inlier, std::vector<bool> &outlier){
    inlier.clear();
    outlier.clear();

    bool flag;

    /**
     * Update scale_reference value
     */
    if( is_speed_provided_ ){
        // compute and update scale reference from velocity stacks if speed raw data are provided
        updateScaleReference();

    }else{
        // calculate distance between road and the camera without speed data
        cv::Point2f uv_curr;
        Eigen::Vector4d point_curr;
        std::vector<cv::Point3f> road_candidate;
        std::vector<uint32_t> road_idx;

        for (int i = 0; i < num_feature_; i++){
            if( features_[i].is_3D_reconstructed ){
                uv_curr = features_[i].uv.back(); //latest feature

                // road region candidate in image plane
                if (uv_curr.y > params_.im_size.height * 0.5 
                && uv_curr.y > params_.im_size.height - 0.7 * uv_curr.x 
                && uv_curr.y > params_.im_size.height + 0.7 * (uv_curr.x - params_.im_size.width)){
                    point_curr = features_[i].point_curr;
                    road_candidate.emplace_back(point_curr(0),point_curr(1),point_curr(2));
                    road_idx.push_back(i);
                }
            }
        }

        // do ransac
        if( road_idx.size() > (uint32_t) params_.th_inlier ){
            std::vector<double> plane;
            std::vector<bool> plane_inlier, plane_outlier;
            lsi::ransac<cv::Point3f, std::vector<double>>(road_candidate, params_.ransac_coef_plane, plane, plane_inlier, plane_outlier);

            if( std::count(plane_inlier.begin(), plane_inlier.end(), true) > (uint32_t) params_.th_inlier ){
                double scale_from_height = 0;
                for( uint32_t i = 0; i < road_idx.size(); i++ ){
                    if( plane_inlier[i] )
                        features_[road_idx[i]].type = Type::Road;
                    // else
                    //     features_[road_idx[i]].type = Type::Unknown;
                }
                scale_from_height = params_.vehicle_height / std::abs(plane[3]);
                if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "Plane: " << plane[0] << ',' << plane[1] << ',' << plane[2] << std::endl;

                updateScaleReference(scale_from_height);
                // std::cout << "scale_from_height: " << scale_from_height << std::endl;
            }
        }
    }

    /**
     * Update scale value
     */
    if (is_scale_initialized_)
    {
        double scale = 0;
        std::vector<bool> scale_inlier, scale_outlier;
        if ( params_.weight_scale_ref >= 0 ){ // Use RANSAC to find suitable scale
            // Seek index of which feature is 3D reconstructed currently,
            // and 3D initialized previously
            std::vector<int> idx;
            for (uint32_t i = 0; i < features_.size(); i++){
                if( features_[i].is_3D_reconstructed && features_[i].landmark )
                    idx.push_back(i);
            }
            uint32_t num_pts = idx.size();

            if( num_pts > params_.ransac_coef_scale.min_num_point ){

                Eigen::Matrix4d tform;
                tform.setIdentity();
                tform.block(0,0,3,3) = R;
                tform.block(0,3,3,1) = t;

                std::vector<std::pair<cv::Point3f,cv::Point3f>> Points;
                Eigen::Vector4d init_point, expt_point, curr_point;
                params_.ransac_coef_scale.weight.clear();
                params_.ransac_coef_scale.th_dist_arr.clear();
                double std_inv_z, inv_z, std_z;
                
                for (uint32_t i = 0; i < num_pts; i++){
                    curr_point = features_[idx[i]].point_curr;

                    // Get initialized 3D point
                    init_point = TocRec_[keystep_].inverse() * features_[idx[i]].landmark->point_init;
                    
                    // Get expected 3D point by transforming the coordinates of the observed 3d point
                    expt_point = tform.inverse() * curr_point;

                    Points.emplace_back(cv::Point3f(expt_point(0),expt_point(1),expt_point(2)),
                                        cv::Point3f(init_point(0),init_point(1),init_point(2)));
                    
                    // RANSAC weight
                    // the smaller depth is, the larger weight is
                    params_.ransac_coef_scale.weight.push_back( std::atan( -curr_point(2)/5 + 3 ) + M_PI / 2 );
                    // params_.ransac_coef_scale.weight.push_back( std::atan( -features_[i].depthfilter->getVariance() * 1e4 ) + M_PI / 2 );
                    // params_.ransac_coef_scale.weight.push_back( 1.0/features_[i].depthfilter->getVariance() );

                    // RANSAC threshold
                    // the larger variance is, the larger threshold is
                    std_inv_z = std::sqrt(features_[i].depthfilter->getVariance());
                    inv_z = features_[i].depthfilter->getMean();
                    std_z = 0.5 * std::min(std::abs(std_inv_z/(inv_z*inv_z+std_inv_z*inv_z)), std::abs(std_inv_z/(inv_z*inv_z-std_inv_z*inv_z)));
                    params_.ransac_coef_scale.th_dist_arr.push_back( std::max(params_.ransac_coef_scale.th_dist, std_z) );
                }

                // do ransac
                params_.ransac_coef_scale.calculate_func = std::bind(lsi::calculateScale, std::placeholders::_1, std::placeholders::_2, scale_reference_, params_.weight_scale_ref);
                lsi::ransac<std::pair<cv::Point3f,cv::Point3f>,double>(Points, params_.ransac_coef_scale, scale, scale_inlier, scale_outlier);
                num_feature_inlier_ = std::count(scale_inlier.begin(), scale_inlier.end(), true);

                if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "num_feature_inlier_: " << num_feature_inlier_ << std::endl;
            }
            if( num_pts <= params_.ransac_coef_scale.min_num_point || num_feature_inlier_ < (std::size_t)params_.th_inlier || scale == 0 ){
                if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "Warning: There are a few scale factor inliers" << std::endl;

                // Use the previous scale, if the scale cannot be found
                scale = (TRec_.back().block(0,3,3,1)).norm();

                // Update scale
                t = scale * t;
                flag = false;
            }else{
                if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "@ scale_from_height: " << scale_reference_ << ", " << "scale: " << scale << std::endl;

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
            // Update scale
            t = scale_reference_ * t;
            num_feature_inlier_ = num_feature_3D_reconstructed_;
            flag = true;
        }

    }else{
        // update scale directly
        t = scale_reference_ * t;

        // find inliers
        for( uint32_t i = 0; i < features_.size(); i++ ){
            if( features_[i].is_3D_reconstructed )
                inlier.push_back(true);
            else
                inlier.push_back(false);
        }
        num_feature_inlier_ = num_feature_3D_reconstructed_;

        if( scale_reference_ < 0 )
            flag = false;
        else
            flag = true;
    }

    if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "# Propagate scale: " << lsi::toc() << std::endl;

    if( t.hasNaN() ){
        if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "Warning: A scale value is nan" << std::endl;
        return false;
    }else
        return flag;
}

/**
 * @brief scale 참조값 업데이트
 * @details 지면으로부터의 높이 또는 속력계로부터 scale의 참조값을 업데이트한다.
 * @param scale 계산한 scale
 * @return 없음
 * @author Sangil Lee (sangillee724@gmail.com)
 * @date 29-Dec-2019
 */
void MVO::updateScaleReference(double scale){
    if( is_speed_provided_ ){
        scale = 0;
        auto & stack = curr_keyframe_.linear_velocity_since_;
        if( stack.size() > 1 ){
            for( uint32_t i = 0; i < stack.size()-1; i++ )
                scale += (stack[i].second+stack[i+1].second)/2 * (stack[i+1].first-stack[i].first);
        }
        // std::cout << "scale_from_speed: " << scale << std::endl;
    }

    if( params_.weight_scale_reg < 0 )
        scale_reference_ = scale;
    else{
        if( scale_reference_ < 0 || is_start_ == false )
            scale_reference_ = scale;
        else{
            // low-pass filter
            // scale_reference_ = params_.weightScaleReg * scale_reference_ + (1-params_.weightScaleReg) * scale;

            // limit slope
            scale_reference_ = scale_reference_ + ((scale > scale_reference_)?1:-1) * std::min(std::abs(scale - scale_reference_), params_.weight_scale_reg);
        }
    }
}