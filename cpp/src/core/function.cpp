#include "core/MVO.hpp"
#include "core/random.hpp"

void MVO::calculateScale(const std::vector<std::pair<cv::Point3f,cv::Point3f>> &pts, double& scale){
    double num = 0, den = 0;
    for ( const auto & pt : pts ){
        num += (pt.first.x * pt.second.x + pt.first.y * pt.second.y + pt.first.z * pt.second.z);
        den += (pt.first.x * pt.first.x + pt.first.y * pt.first.y + pt.first.z * pt.first.z);
    }
    scale = (num / pts.size() + MVO::s_scale_reference_weight_ * MVO::s_scale_reference_) / (den / pts.size() + MVO::s_scale_reference_weight_ + 1e-10);

    // double sum = 0;
    // for ( const auto & pt : pts ){
    //     sum += (pt.first.x * pt.second.x + pt.first.y * pt.second.y + pt.first.z * pt.second.z + MVO::s_scale_reference_weight_ * ::scale_reference) / 
    //     (pt.first.x * pt.first.x + pt.first.y * pt.first.y + pt.first.z * pt.first.z + MVO::s_scale_reference_weight_ + 1e-10);
    // }
    // scale = sum / pts.size();
}

void MVO::calculateScaleError(const double scale, const std::vector<std::pair<cv::Point3f,cv::Point3f>> &pts, std::vector<double>& dist){
    dist.clear();
    for( const auto & pt : pts )
        dist.push_back(cv::norm(pt.second - scale * pt.first));
}

void MVO::calculatePlane(const std::vector<cv::Point3f>& pts, std::vector<double>& plane){
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
        for( const auto & pt : pts ){
            centroid.x += pt.x;
            centroid.y += pt.y;
            centroid.z += pt.z;
        }
        centroid.x = centroid.x/pts.size();
        centroid.y = centroid.y/pts.size();
        centroid.z = centroid.z/pts.size();

        double xx = 0, xy = 0, xz = 0, yy = 0, yz = 0, zz = 0;
        for( const auto & pt : pts ){
            cv::Point3f r = pt - centroid;
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

    if( std::abs(plane[1]) < 0.5 ){ // perpendicular to y-axis
        plane.clear();
        plane.push_back(0);
        plane.push_back(0);
        plane.push_back(0);
        plane.push_back(0);
    }
}

void MVO::calculatePlaneError(const std::vector<double>& plane, const std::vector<cv::Point3f>& pts, std::vector<double>& dist){
    double norm = std::sqrt(std::pow(plane[0],2) + std::pow(plane[1],2) + std::pow(plane[2],2));
    if( std::abs(norm) < 1e-10 ){
        dist.clear();
        for( const auto & pt : pts )
            dist.push_back(1e10);
    }else{
        double a = plane[0]/norm;
        double b = plane[1]/norm;
        double c = plane[2]/norm;
        double d = plane[3]/norm;

        dist.clear();
        for( const auto & pt : pts )
            dist.push_back(std::abs(a * pt.x + b * pt.y + c * pt.z + d));
    }
}

double MVO::calcReconstructionError(Eigen::Matrix4d& Toc){
    std::vector<double> error;
    error.reserve(features_.size());
    for( const auto & feature : features_ )
        error.push_back((Toc * feature.point_curr - feature.point_init).norm());

    std::sort(error.begin(), error.end());
    return error[std::floor(error.size()/2)];
}

double MVO::calcReconstructionError(Eigen::Matrix3d& R, Eigen::Vector3d& t){
    Eigen::Matrix4d Toc;
    Toc.block(0,0,3,3) = R;
    Toc.block(0,3,3,1) = t;
    Toc(3,3) = 1;

    return calcReconstructionError(Toc);
}

void MVO::calcReconstructionErrorGT(Eigen::MatrixXd& depth){
    std::vector<double> idx;
    for( uint32_t i = 0; i < features_.size(); i++ )
        if( depth(features_[i].uv.back().y, features_[i].uv.back().x) > 0 && features_[i].is_3D_reconstructed == true )
            idx.push_back(i);

    if( idx.size() > 0 ){
        std::cerr << "* Reconstruction depth: ";

        // median value
        // std::sort(error.begin(), error.end());
        // std::cerr << error[std::floor(error.size()/2)];

        // all elements
        for( const auto & i : idx )
            std::cerr << features_[i].point_curr(2) << ' ';
        std::cerr << std::endl;

        std::cerr << "* Groundtruth depth: ";
        for( const auto & i : idx )
            std::cerr << depth(features_[i].uv.back().y, features_[i].uv.back().x) << ' ';
        std::cerr << std::endl;
    }
}