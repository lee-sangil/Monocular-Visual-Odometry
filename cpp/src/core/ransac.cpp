#include "core/ransac.hpp"

void lsi::calculateScale(const std::vector<std::pair<cv::Point3f,cv::Point3f>> &pts, double& scale, double reference_value, double reference_weight){
    double num = 0, den = 0;
    for ( const auto & pt : pts ){
        num += (pt.first.x * pt.second.x + pt.first.y * pt.second.y + pt.first.z * pt.second.z);
        den += (pt.first.x * pt.first.x + pt.first.y * pt.first.y + pt.first.z * pt.first.z);
    }
    scale = (num / pts.size() + reference_weight * reference_value) / (den / pts.size() + reference_weight + 1e-10);

    // double sum = 0;
    // for ( const auto & pt : pts ){
    //     sum += (pt.first.x * pt.second.x + pt.first.y * pt.second.y + pt.first.z * pt.second.z + reference_weight * reference_value) / 
    //     (pt.first.x * pt.first.x + pt.first.y * pt.first.y + pt.first.z * pt.first.z + reference_weight + 1e-10);
    // }
    // scale = sum / pts.size();
}

void lsi::calculateScaleError(const double scale, const std::vector<std::pair<cv::Point3f,cv::Point3f>> &pts, std::vector<double>& dist){
    dist.clear();
    for( const auto & pt : pts )
        dist.push_back(cv::norm(pt.second - scale * pt.first));
}

void lsi::calculatePlane(const std::vector<cv::Point3f>& pts, std::vector<double>& plane){
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

void lsi::calculatePlaneError(const std::vector<double>& plane, const std::vector<cv::Point3f>& pts, std::vector<double>& dist){
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