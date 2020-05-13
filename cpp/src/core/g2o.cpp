#include "core/g2o.hpp"

void rotToQuat(const Eigen::Matrix3d& R, Eigen::Quaterniond& quat){ // qx qy qz qw
    double qw = 0.5 * std::sqrt(1+R(0,0)+R(1,1)+R(2,2));
    double qx = 0.25 * (1/qw) * (R(2,1)-R(1,2));
    double qy = 0.25 * (1/qw) * (R(0,2)-R(2,0));
    double qz = 0.25 * (1/qw) * (R(1,0)-R(0,1));
    quat.coeffs() = (Eigen::Vector4d() << qx, qy, qz, qw).finished(); // x y z w
}

void quatToRot(const Eigen::Quaterniond& quat, Eigen::Matrix3d& R){
    double qx = quat.x();
    double qy = quat.y();
    double qz = quat.z();
    double qw = quat.w();
    R << qw*qw+qx*qx-qy*qy-qz*qz, 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy),
         2*(qx*qy+qw*qz), qw*qw-qx*qx+qy*qy-qz*qz, 2*(qy*qz-qw*qx),
         2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), qw*qw-qx*qx-qy*qy+qz*qz;
}

int getNewID(){
    static int vertex_id = 0;
    return vertex_id++;
}

g2o::VertexSE3Expmap * addPoseVertex(g2o::SparseOptimizer * optimizer, g2o::SE3Quat& pose, bool set_fixed = true){
    try{
        g2o::VertexSE3Expmap * v_se3 = new g2o::VertexSE3Expmap;
        int id = getNewID();
        v_se3->setId(id);
        if( set_fixed ) v_se3->setEstimate(pose);
        v_se3->setFixed(set_fixed);
        optimizer->addVertex(v_se3);
        return v_se3;
    }catch(int except){
        return NULL;
    }
}

g2o::VertexSBAPointXYZ * addPointVertex(g2o::SparseOptimizer * optimizer, const g2o::Vector3D& point){
    try{
        g2o::VertexSBAPointXYZ * v_xyz = new g2o::VertexSBAPointXYZ;
        int id = getNewID();
        v_xyz->setId(id);
        v_xyz->setFixed(false);
        v_xyz->setMarginalized(true);
        v_xyz->setEstimate(point);
        optimizer->addVertex(v_xyz);
        return v_xyz;
    }catch(int except){
        return NULL;
    }
}

g2o::EdgeProjectXYZ2UV * addEdgePointPose(g2o::SparseOptimizer * optimizer, g2o::VertexSBAPointXYZ * v0, g2o::VertexSE3Expmap * v1, g2o::Vector2D uv){
    try{
        g2o::EdgeProjectXYZ2UV * edge = new g2o::EdgeProjectXYZ2UV;
        edge->setVertex(0, static_cast<g2o::OptimizableGraph::Vertex*>(v0));
        edge->setVertex(1, static_cast<g2o::OptimizableGraph::Vertex*>(v1));
        edge->setMeasurement(uv);
        edge->setInformation(Eigen::MatrixXd::Identity(2,2));
        edge->setParameterId(0,1); // EdgeProjectXYZ2UV의 Parameter[0]이 Optimizable Graph의 Parameter[1]에 대응된다.
        optimizer->addEdge(edge);
        return edge;
    }catch(int except){
        return NULL;
    }
}