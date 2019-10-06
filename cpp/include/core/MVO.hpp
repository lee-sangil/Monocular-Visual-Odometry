#ifndef __MVO_HPP__
#define __MVO_HPP__

#include "core/common.hpp"

typedef  std::vector< std::tuple<cv::Point2f, Eigen::Vector3d> > ptsROI_t;

class MVO{
	public:

	enum SVD{
		JACOBI,
		BDC,
		OpenCV,
		Eigen
	};

	enum TRIANGULATION{
		MIDP,
		LLS
	};
	
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
		cv::Mat Kcv;
		std::vector<double> radialDistortion;
		std::vector<double> tangentialDistortion;
		std::vector<double> distCoeffs;
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
		double plotScale;
		double reprojError;
		MVO::SVD SVDMethod;
		MVO::TRIANGULATION triangulationMethod;
	};
	
	public:

	MVO();
	MVO(std::string yaml);
	~MVO(){
		delete this->eigenSolver;
	}
	
	// Set image
	void set_image(cv::Mat& image);

	// Get feature 
	ptsROI_t get_points();
	
	// Script operations
	void backup();
	void reload();
	void refresh();
	void run(cv::Mat& image);
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
	bool verify_solutions(Eigen::Matrix3d& R, Eigen::Vector3d& t);
	bool verify_solutions(std::vector<Eigen::Matrix3d>& R_vec, std::vector<Eigen::Vector3d>& t_vec,
						  Eigen::Matrix3d& R, Eigen::Vector3d& t);
	bool scale_propagation(Eigen::Matrix3d& R, Eigen::Vector3d& t,
						   std::vector<bool>& inlier, std::vector<bool>& outlier);
	bool findPoseFrom3DPoints(Eigen::Matrix3d &R, Eigen::Vector3d &t, std::vector<int>& inlier, std::vector<int>& outlier);
	void constructDepth(const std::vector<cv::Point2f> uv_prev, const std::vector<cv::Point2f> uv_curr, 
                        const Eigen::Matrix3d R, const Eigen::Vector3d t, 
                        std::vector<Eigen::Vector3d> &X_prev, std::vector<Eigen::Vector3d> &X_curr, 
                        std::vector<bool> &inlier);
	void update3DPoints(const Eigen::Matrix3d& R, const Eigen::Vector3d& t,
						const std::vector<bool>& inlier, const std::vector<bool>& outlier,
						Eigen::Matrix4d& T, Eigen::Matrix4d& Toc, Eigen::Vector4d& Poc);
	void update3DPoints(const Eigen::Matrix3d& R, const Eigen::Vector3d& t,
						const std::vector<bool>& inlier, const std::vector<bool>& outlier,
						const Eigen::Matrix3d& R_E, const Eigen::Vector3d& t_E, const bool& success_E,
						Eigen::Matrix4d& T, Eigen::Matrix4d& Toc, Eigen::Vector4d& Poc);

	// RANSAC
	static double ransac(const std::vector<cv::Point3f>& x, const std::vector<cv::Point3f>& y,
				MVO::RansacCoef ransacCoef,
				std::vector<bool>& inlier, std::vector<bool>& outlier);
	static double calculate_scale(const std::vector<cv::Point3f>& pt1, const std::vector<cv::Point3f>& pt2);
	static std::vector<double> calculate_scale_error(double scale, const std::vector<cv::Point3f>& pt1, const std::vector<cv::Point3f>& pt2);
	static std::vector<uint32_t> randperm(unsigned int ptNum, int minPtNum);
	static std::vector<uint32_t> randweightedpick(const std::vector<double>& h, int n = 1);
	
	private:

	int step;
	
	cv::Mat undist_image;
	cv::Mat cur_image;
	cv::Mat prev_image;
	cv::Mat temp_image;
	cv::Ptr<cv::CLAHE> cvClahe;

	Bucket bucket;
	std::vector<Feature> features;
	std::vector<Feature> features_temp;
	std::vector<Feature> features_backup;

	bool scale_initialized;

	int nFeature;
	int nFeatureMatched;
	int nFeature2DInliered;
	int nFeature3DReconstructed;
	int nFeatureInlier;

	cv::Mat Identity3x3;
	cv::Mat essentialMat;
	std::vector<Eigen::Matrix3d> R_vec;
	std::vector<Eigen::Vector3d> t_vec;
	std::vector<Eigen::Matrix4d> TRec;
	std::vector<Eigen::Matrix4d> TocRec;
	std::vector<Eigen::Vector4d> PocRec;

	std::vector<uint32_t> idxTemplate;
	std::vector<bool> inlierTemplate;
	std::vector<cv::Mat> prevPyramidTemplate, currPyramidTemplate;
	Eigen::MatrixXd MapMatrixTemplate;
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> * eigenSolver;

	Parameter params;
};

#endif //__MVO_HPP__
