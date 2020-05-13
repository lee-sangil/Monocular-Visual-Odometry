
#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <random>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/MatrixFunctions>

#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/factory.h"
#include "g2o/types/slam3d/parameter_camera.h"
#include "g2o/types/slam3d/vertex_se3.h"
#include "g2o/types/slam3d/vertex_pointxyz.h"
#include "g2o/types/slam3d/edge_se3_pointxyz.h"
#include "g2o/types/sba/types_six_dof_expmap.h"
#include "g2o/stuff/sampler.h"

/**
 * @author Sangil Lee (sangillee724@gmail.com)
 * @date 11-May-2020
 */
namespace lsi{
    void rotToQuat(const Eigen::Matrix3d& R, Eigen::Quaterniond& quat); /**< @brief 회전 행렬을 쿼터니언(x,y,z,w)으로 변환하는 함수 */
    void quatToRot(const Eigen::Quaterniond& quat, Eigen::Matrix3d& R); /**< @brief 쿼터니언(x,y,z,w)을 회전 행렬로 변환하는 함수 */
    int getNewID(); /**< @brief 새로운 ID 정수값을 읽는 함수 */
    g2o::VertexSE3Expmap * addPoseVertex(g2o::SparseOptimizer * optimizer, g2o::SE3Quat& pose, bool set_fixed = true); /**< @brief 자세 노드 추가하는 함수 */
    g2o::VertexSBAPointXYZ * addPointVertex(g2o::SparseOptimizer * optimizer, const g2o::Vector3D& point); /**< @brief 3차원 랜드마크 노드 추가하는 함수 */
    g2o::EdgeProjectXYZ2UV * addEdgePointPose(g2o::SparseOptimizer * optimizer, g2o::VertexSBAPointXYZ * v0, g2o::VertexSE3Expmap * v1, g2o::Vector2D uv); /**< @brief 3차원 랜드마크 - 자세 사이의 엣지를 추가하는 함수 */
}