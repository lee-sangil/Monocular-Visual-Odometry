#include "core/MVO.hpp"

/*********************
 *   For debugging   *
 *********************/

/**
 * @brief 3차원 복원 에러 계산.
 * @details 현재 복원된 3차원 좌표와 초기화된 3차원 좌표 사이의 오차를 계산한다.
 * @param Toc 이미지 프레임 사이의 변환 행렬
 * @return 오차값
 * @author Sangil Lee (sangillee724@gmail.com)
 * @date 28-Dec-2019
 */
double MVO::calcReconstructionError(Eigen::Matrix4d& Toc) const {
    std::vector<double> error;
    error.reserve(features_.size());
    for( const auto & feature : features_ )
        if( feature.landmark )
            error.push_back((Toc * feature.point_curr - feature.landmark->point_init).norm());

    std::sort(error.begin(), error.end());
    return error[std::floor(error.size()/2)];
}

/**
 * @brief 3차원 복원 에러 계산.
 * @details 현재 복원된 3차원 좌표와 초기화된 3차원 좌표 사이의 오차를 계산한다.
 * @param R 이미지 프레임 사이의 회전 행렬
 * @param t 이미지 프레임 사이의 변위 벡터
 * @return 오차값
 * @author Sangil Lee (sangillee724@gmail.com)
 * @date 28-Dec-2019
 */
double MVO::calcReconstructionError(Eigen::Matrix3d& R, Eigen::Vector3d& t) const {
    Eigen::Matrix4d Toc;
    Toc.block(0,0,3,3) = R;
    Toc.block(0,3,3,1) = t;
    Toc(3,3) = 1;

    return calcReconstructionError(Toc);
}

/**
 * @brief 3차원 복원 에러 계산.
 * @details 이미지 프레임의 깊이 참값과 3차원 좌표 사이의 거리 오차를 계산한다.
 * @param depth 깊이값 프레임
 * @return 없음
 * @author Sangil Lee (sangillee724@gmail.com)
 * @date 28-Dec-2019
 */
void MVO::calcReconstructionErrorGT(Eigen::MatrixXd& depth) const {
    std::vector<double> idx;
    if( params_.output_filtered_depth ){
        for( uint32_t i = 0; i < features_.size(); i++ )
            if( depth(features_[i].uv.back().y, features_[i].uv.back().x) > 0 && features_[i].landmark != NULL && features_[i].uv.back().y > params_.im_size.height / 2.0 && features_[i].life > 1 )
                idx.push_back(i);
    }else{
        for( uint32_t i = 0; i < features_.size(); i++ )
            if( depth(features_[i].uv.back().y, features_[i].uv.back().x) > 0 && features_[i].is_3D_reconstructed == true && features_[i].uv.back().y > params_.im_size.height / 2.0 && features_[i].life > 1 )
                idx.push_back(i);
    }

    Eigen::Matrix4d Tco = TocRec_.back().inverse();
    Eigen::Vector3d point;

    std::vector<double> relative_error;
    
    if( idx.size() > 0 ){
        if( MVO::s_point_logger_.is_open() ) MVO::s_point_logger_ << "* Reconstruction:";
        for( const auto & i : idx ){
            if( params_.output_filtered_depth ){
                point = Tco.block(0,0,3,4) * features_[i].landmark->point_init;
                if( MVO::s_point_logger_.is_open() ) MVO::s_point_logger_ << ' ' << point(0) << ' ' << point(1) << ' ' << point(2);
            }else{
                if( MVO::s_point_logger_.is_open() ) MVO::s_point_logger_ << ' ' << features_[i].point_curr(0) << ' ' << features_[i].point_curr(1) << ' ' << features_[i].point_curr(2);
            }
        }
        if( MVO::s_point_logger_.is_open() ) MVO::s_point_logger_ << std::endl;

        if( MVO::s_point_logger_.is_open() ) MVO::s_point_logger_ << "* Groundtruth:";
        for( const auto & i : idx )
            if( MVO::s_point_logger_.is_open() ) MVO::s_point_logger_ << ' ' << features_[i].uv.back().x << ' ' << features_[i].uv.back().y << ' ' << depth(features_[i].uv.back().y, features_[i].uv.back().x);
        if( MVO::s_point_logger_.is_open() ) MVO::s_point_logger_ << std::endl;

        for( const auto & i : idx){
            double gt_depth = depth(features_[i].uv.back().y, features_[i].uv.back().x);
            if( params_.output_filtered_depth ){
                point = Tco.block(0,0,3,4) * features_[i].landmark->point_init;
                relative_error.push_back(std::abs(point(2)-gt_depth)/gt_depth);
            }else{
                relative_error.push_back(std::abs(features_[i].point_curr(2)-gt_depth)/gt_depth);
            }
        }
        if( relative_error.size() > 0 ){
            std::sort(relative_error.begin(), relative_error.end());
            std::cout << "Relative error at 25\% from the lowest: " << std::setw(8) << relative_error.at(std::floor(relative_error.size()*0.25)) << ", the lowest: " << relative_error.front() << ", the median: " << relative_error.at(std::floor(relative_error.size()*0.5)) << ", the greatest: " << relative_error.back() << std::endl;
        }
    }
}