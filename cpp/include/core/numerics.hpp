#ifndef __NUMERICS__HPP__
#define __NUMERICS__HPP__

#include "core/common.hpp"

inline Eigen::Matrix3d skew(const Eigen::Vector3d& v)
{
    Eigen::Matrix3d M;
    M << 0, -v(2), v(1),
           v(2), 0, -v(0),
           -v(1), v(0), 0;

    return M;
}

inline Eigen::Vector3d vec(const Eigen::Matrix3d& M)
{
    Eigen::Vector3d v;
    v << M(2,1), M(0,2), M(1,0);
    return v;
}

#endif //__NUMERICS__HPP__