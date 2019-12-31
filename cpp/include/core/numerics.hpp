#ifndef __NUMERICS__HPP__
#define __NUMERICS__HPP__

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

/**
 * @brief 벡터를 외적 행렬로 변환
 * @param v 3x1 벡터
 * @return 3x3 행렬
 * @author Sangil Lee (sangillee724@gmail.com)
 * @date 25-Dec-2019
 */
inline Eigen::Matrix3d skew(const Eigen::Vector3d& v)
{
    Eigen::Matrix3d M;
    M << 0, -v(2), v(1),
           v(2), 0, -v(0),
           -v(1), v(0), 0;

    return M;
}

/**
 * @brief 외적 행렬을 벡터로 변환
 * @param M 3x3 행렬
 * @return 3x1 벡터
 * @author Sangil Lee (sangillee724@gmail.com)
 * @date 25-Dec-2019
 */
inline Eigen::Vector3d vec(const Eigen::Matrix3d& M)
{
    Eigen::Vector3d v;
    v << M(2,1), M(0,2), M(1,0);
    return v;
}

#endif //__NUMERICS__HPP__