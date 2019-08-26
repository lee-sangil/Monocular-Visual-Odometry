#include "core/MVO.hpp"

unsigned int Feature::new_feature_id = 1;

MVO::MVO(){
	this->step = 0;

	this->bucket = Bucket();
}
MVO::MVO(Parameter params){
	this->params.fx = params.fx;
	this->params.fy = params.fy;
	this->params.cx = params.cx;
	this->params.cy = params.cy;
	this->params.k1 = params.k1;
	this->params.k2 = params.k2;
	this->params.p1 = params.p1;
	this->params.p2 = params.p2;
	this->params.k3 = params.k3;
	this->params.width = params.width;
	this->params.height = params.height;

	this->params.K << params.fx, 0, params.cx,
						0, params.fy, params.cy,
						0, 0, 1;

	this->params.imSize.push_back(params.width);
	this->params.imSize.push_back(params.height);
	this->params.radialDistortion.push_back(params.k1);
	this->params.radialDistortion.push_back(params.k2);
	this->params.radialDistortion.push_back(params.k3);
	this->params.tangentialDistortion.push_back(params.p1);
	this->params.tangentialDistortion.push_back(params.p2);
}

void MVO::set_image(const cv::Mat image){
	this->prev_image = this->cur_image.clone();
	this->cur_image = image.clone(); 
}


// haram
bool MVO::calculate_essential()
{
    Eigen::Matrix3d K = params.K;
    // below define should go in constructor and class member variable
    double focal = (K(1, 1) + K(2, 2)) / 2;
    cv::Point2d principle_point(K(1, 3), K(2, 3));
    Eigen::Matrix3d W;
    W << 0, -1, 0, 1, 0, 0, 0, 0, 1;
    //

    if (step == 1)
        return true;

    // Extract homogeneous 2D point which is matched with corresponding feature
    // TODO: define of find function

    std::vector<int> idx;
    for (unsigned int i = 0; i < features.size(); i++)
    {
        if (features.at(i).is_matched)
            idx.push_back(i);
    }

    unsigned int nInlier = idx.size();

    int point_count = 100;
    std::vector<cv::Point2d> points1(point_count);
    std::vector<cv::Point2d> points2(point_count);

    // initialize the points here ... */
    for (unsigned int i = 0; i < nInlier; i++)
    {
        points1[i] = features.at(idx.at(i)).uv.at(1); // second to latest
        points2[i] = features.at(idx.at(i)).uv.at(0); // latest
    }

    std::vector<bool> inlier;

    cv::Mat E;
    Eigen::Matrix3d E_, U, V;
    E = findEssentialMat(points1, points2, focal, principle_point, cv::RANSAC, 0.999, 1.0, inlier);
    cv2eigen(E, E_);
    JacobiSVD<MatrixXf> svd(E_, ComputeThinU | ComputeThinV);
    U = svd.matrixU();
    V = svd.matrixV();

    if (U.determinant < 0)
        U.block(0, 2, 3, 1) = -U.block(0, 2, 3, 1);
    if (V.determinant < 0)
        V.block(0, 2, 3, 1) = -V.block(0, 2, 3, 1);
    R_vec.at(0) = U * W * V.transpose();
    R_vec.at(1) = U * W * V.transpose();
    R_vec.at(2) = U * W.transpose() * V;
    R_vec.at(3) = U * W.transpose() * V;
    t_vec.at(0) = U.block(0, 2, 3, 1);
    t_vec.at(1) = -U.block(0, 2, 3, 1);
    t_vec.at(2) = U.block(0, 2, 3, 1);
    t_vec.at(3) = -U.block(0, 2, 3, 1);

    unsigned int inlier_cnt = 0;
    for (unsigned int i = 0; i < inlier.size(); i++)
    {
        if (inlier(i))
        {
            features.at(i).is_2D_inliered = true;
            inlier_cnt++;
        }
    }
    nFeature2Dinliered = inlier_cnt;

    if (nFeature2Dinliered < params.thInlier)
    {
        std::cerr << " There are a few inliers matching features in 2D." << std::endl;
        return false;
    }
    else
    {
        return true;
    }
}

bool MVO::calculate_motion()
{
    if (step == 1)
        return true;

    Eigen::Matrix3d R,R_;
    Eigen::Vector3d t,t_;
    
    success1 = findPoseFrom3DPoints(R_,t_);
    if (!success1){
        // Verity 4 solutions
        success2 = verify_solutions(R_vec, t_vec, R_, t_);
        
        if (!success2){
            std::cerr << "There are no meaningful R, t." << std::endl;
            return false;
        }

        // Update 3D points
        std::vector<bool> inlier, outlier;
        success3 = scale_propagation(R_,t_, R, t, inlier, outlier);

        if (!success3){
            std::cerr << "There are few inliers matching scale." << std::endl;
            return false;
        }

        Eigen::Matrix4d T, Toc;
        Eigen::Vector4d Poc;
        update3Dpoints(R, t, inlier, outlier, 'w/oPnP', T, Toc, Poc); // overloading function
    } // if (!success1)
    else{
        success2 = verify_solutions(R_vec, t_vec, R_, t_);

        // Update 3D points
        std::vector<bool> inlier, outlier;
        success3 = scale_propagation(R_, t_, inlier, outlier);

        // Update 3D points
        Eigen::Matrix4d T, Toc;
        Eigen::Vector4d Poc;

        update3Dpoints(R, t, inlier, outlier, 'w/PnP', R_, t_, success3, T, Toc, Poc); // overloading function
    } // if (!success1)

    scale_initialized = true;

    if (nFeature3DReconstructed < params.thInlier){
        std::cerr << "There are few inliers reconstructed in 3D." << std::endl;
        return false;
    }
    else{
        // Save solution
        TRec = T;
        TocRec = Toc;
        PocRec = Poc;

        return true;
    }

    if (T(1:3,4).norm() > 100){
        std::cout << "the current position: " <<  Poc(1) << ", " << Poc(2) << ", " << Poc(3) << std::endl;
        std::cerr << "Stop!" << std::endl;
    }
}
