#include "core/MVO.hpp"
#include "core/random.hpp"

double MVO::calcReconstructionError(Eigen::Matrix4d& Toc) const {
    std::vector<double> error;
    error.reserve(features_.size());
    for( const auto & feature : features_ )
        error.push_back((Toc * feature.point_curr - feature.point_init).norm());

    std::sort(error.begin(), error.end());
    return error[std::floor(error.size()/2)];
}

double MVO::calcReconstructionError(Eigen::Matrix3d& R, Eigen::Vector3d& t) const {
    Eigen::Matrix4d Toc;
    Toc.block(0,0,3,3) = R;
    Toc.block(0,3,3,1) = t;
    Toc(3,3) = 1;

    return calcReconstructionError(Toc);
}

void MVO::calcReconstructionErrorGT(Eigen::MatrixXd& depth) const {
    std::vector<double> idx;
    for( uint32_t i = 0; i < features_.size(); i++ )
        if( depth(features_[i].uv.back().y, features_[i].uv.back().x) > 0 && features_[i].is_3D_reconstructed == true )
            idx.push_back(i);

    if( idx.size() > 0 ){
        if( MVO::s_file_logger.is_open() ) MVO::s_file_logger << "* Reconstruction depth: ";
        for( const auto & i : idx )
            if( MVO::s_file_logger.is_open() ) MVO::s_file_logger << features_[i].point_curr(2) << ' ';
        if( MVO::s_file_logger.is_open() ) MVO::s_file_logger << std::endl;

        if( MVO::s_file_logger.is_open() ) MVO::s_file_logger << "* Groundtruth depth: ";
        for( const auto & i : idx )
            if( MVO::s_file_logger.is_open() ) MVO::s_file_logger << depth(features_[i].uv.back().y, features_[i].uv.back().x) << ' ';
        if( MVO::s_file_logger.is_open() ) MVO::s_file_logger << std::endl;
    }
}