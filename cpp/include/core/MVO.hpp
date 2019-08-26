#ifndef __MVO_HPP__
#define __MVO_HPP__

#include "core/common.hpp"

class MVO{
	public:

	struct RansacCoef{
		int iterMax = 1e4;
		double minPtNum = 5;
		double thInlrRatio = 0.9;
		double thDist = 0.5;
		double thDistOut = 5.0;
		// funcFindF = @obj.calculate_scale;
		// funcDist = @obj.calculate_scale_error;
	};

	struct Parameter{
		double fx, fy, cx, cy;
		double k1, k2, p1, p2, k3;
		double width, height;
		Eigen::Matrix3d K;
		std::vector<double> radialDistortion;
		std::vector<double> tangentialDistortion;
		std::vector<unsigned int> imSize;

		RansacCoef ransacCoef_scale_prop;
		
		unsigned int thInlier = 5;
		double min_px_dist = 7.0;

		// Statistical model
		double var_theta = std::pow(90.0 / 1241.0 / 2, 2);
		double var_point = 1.0;
		
		// 3D reconstruction
		double vehicle_height;
		double initScale;
		double reprojError;
	};
	
	public:

	MVO();
	MVO(Parameter params);
	// Set image
	void set_image(const cv::Mat image);

	// Script operations
	void backup();
	void reload();
	void refresh();
	void run(const cv::Mat image);

	// Feature operations
	void klt_tracker(std::vector<cv::Point>& points, std::vector<bool>& validity);
	void update_bucket();
	bool extract_features();
	bool update_features();
	void delete_dead_features();
	void add_features();
	void add_feature();

	// Calculations
	bool calculate_essential();
	bool calculate_motion();
	bool verify_solutions(Eigen::Matrix3d& R, Eigen::Vector3d& t);
	bool scale_propagation(Eigen::Matrix3d& R, Eigen::Vector3d& t, std::vector<bool> inlier, std::vector<bool> outlier);
	bool findPoseFrom3DPoints(Eigen::Matrix3d& R, Eigen::Vector3d& t, std::vector<bool> inlier, std::vector<bool> outlier);
	void contructDepth(const std::vector<cv::Point2d> x_prev, const std::vector<cv::Point2d> x_curr, const Eigen::Matrix3d R, const Eigen::Vector3d t, 
						std::vector<Eigen::Vector4d>& X_prev, std::vector<Eigen::Vector4d>& X_curr, std::vector<double>& lambda_prev, std::vector<double>& lambda_curr);
	void update3DPoints(const Eigen::Matrix3d R, const Eigen::Vector3d t, const std::vector<bool> inlier, const std::vector<bool> outlier);
	void update3DPoints(const Eigen::Matrix3d R, const Eigen::Vector3d t, const std::vector<bool> inlier, const std::vector<bool> outlier, const Eigen::Matrix3d R_E, const Eigen::Vector3d t_E, const bool success_E);

	// RANSAC
	double calculate_scale(const std::vector<Eigen::Vector3d> P1, const std::vector<Eigen::Vector3d> P1_ref);
	double calculate_scale_error(const double scale, const std::vector<Eigen::Vector3d> P1, const std::vector<Eigen::Vector3d> P1_ref);

	private:

	unsigned int step;
	
	cv::Mat undist_image;
	cv::Mat cur_image;
	cv::Mat prev_image;

	Bucket bucket;
	std::vector<Feature> features;
	std::vector<Feature> features_backup;

	bool scale_initialized;

	unsigned int nFeature;
	unsigned int nFeatureMatched;
	unsigned int nFeature2DInliered;
	unsigned int nFeature3DReconstructed;
	unsigned int nFeatureInlier;
	unsigned int new_feature_id;

	std::vector<Eigen::Vector3d> R_vec;
	std::vector<Eigen::Vector3d> t_vec;
	std::vector<Eigen::Matrix4d> TRec;
	std::vector<Eigen::Matrix4d> TocRec;
	std::vector<Eigen::Vector4d> PocRec;

	Parameter params;
};

#endif //__MVO_HPP__