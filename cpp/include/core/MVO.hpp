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
		std::vector<double> weight;
	};

	struct Parameter{
		double fx, fy, cx, cy;
		double k1, k2, p1, p2, k3;
		Eigen::Matrix3d K;
		std::vector<double> radialDistortion;
		std::vector<double> tangentialDistortion;
		cv::Size imSize;
		bool applyCLAHE = false;

		RansacCoef ransacCoef_scale_prop;
		
		int thInlier = 5;
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
	MVO(std::string yaml);
	
	// Set image
	void set_image(const cv::Mat image);

	// Script operations
	void backup();
	void reload();
	void refresh();
	void run(const cv::Mat image);
	void plot();

	// Feature operations
	void klt_tracker(std::vector<cv::Point2f>& points, std::vector<bool>& validity);
	void update_bucket();
	bool extract_features();
	bool update_features();
	void delete_dead_features();
	void add_features();
	void add_feature();

	// Calculations
	bool calculate_essential();
	bool calculate_motion();
	bool verify_solutions(std::vector<Eigen::Matrix3d>& R_vec, std::vector<Eigen::Vector3d>& t_vec,
						  Eigen::Matrix3d& R, Eigen::Vector3d& t);
	bool scale_propagation(const Eigen::Matrix3d& R_, const Eigen::Vector3d& t_,
						   Eigen::Matrix3d& R, Eigen::Vector3d& t,
						   std::vector<bool>& inlier, std::vector<bool>& outlier);
	bool scale_propagation(Eigen::Matrix3d& R, Eigen::Vector3d& t,
						   std::vector<bool>& inlier, std::vector<bool>& outlier);
	bool findPoseFrom3DPoints(Eigen::Matrix3d &R, Eigen::Vector3d &t, std::vector<bool>& inlier, std::vector<bool>& outlier);
	void constructDepth(const std::vector<cv::Point2f> x_prev, const std::vector<cv::Point2f> x_curr, const Eigen::Matrix3d R, const Eigen::Vector3d t, 
						std::vector<Eigen::Vector4d>& X_prev, std::vector<Eigen::Vector4d>& X_curr, std::vector<double>& lambda_prev, std::vector<double>& lambda_curr);
	void update3DPoints(const Eigen::Matrix3d& R, const Eigen::Vector3d& t,
						const std::vector<bool>& inlier, const std::vector<bool>& outlier,
						Eigen::Matrix4d& T, Eigen::Matrix4d& Toc, Eigen::Vector4d& Poc);
	void update3DPoints(const Eigen::Matrix3d& R, const Eigen::Vector3d& t,
						const std::vector<bool>& inlier, const std::vector<bool>& outlier,
						const Eigen::Matrix3d& R_E, const Eigen::Vector3d& t_E, const bool& success_E,
						Eigen::Matrix4d& T, Eigen::Matrix4d& Toc, Eigen::Vector4d& Poc);

	// RANSAC
	static double ransac(const std::vector<cv::Point3d>& x, const std::vector<cv::Point3d>& y,
				MVO::RansacCoef ransacCoef,
				std::vector<bool>& inlier, std::vector<bool>& outlier);
	static double calculate_scale(const std::vector<cv::Point3d>& pt1, const std::vector<cv::Point3d>& pt2);
	static std::vector<double> calculate_scale_error(double scale, const std::vector<cv::Point3d>& pt1, const std::vector<cv::Point3d>& pt2);
	static std::vector<int> randperm(unsigned int ptNum, int minPtNum);
	static std::vector<int> randweightedpick(const std::vector<double>& h, int n = 1);
	
	private:

	int step;
	
	cv::Mat undist_image;
	cv::Mat cur_image;
	cv::Mat prev_image;
	cv::Ptr<cv::CLAHE> cvClahe;

	Bucket bucket;
	std::vector<Feature> features;
	std::vector<Feature> features_backup;

	bool scale_initialized;

	int nFeature;
	int nFeatureMatched;
	int nFeature2DInliered;
	int nFeature3DReconstructed;
	int nFeatureInlier;

	std::vector<Eigen::Matrix3d> R_vec;
	std::vector<Eigen::Vector3d> t_vec;
	std::vector<Eigen::Matrix4d> TRec;
	std::vector<Eigen::Matrix4d> TocRec;
	std::vector<Eigen::Vector4d> PocRec;

	Parameter params;
};

#endif //__MVO_HPP__
