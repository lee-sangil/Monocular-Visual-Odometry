#ifndef __MVO_HPP__
#define __MVO_HPP__

#include "core/common.hpp"
#define PI 3.1415926536

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
	
	enum PNP{
		LM,
		ITERATIVE,
		AP3P,
		EPNP,
		DLS,
		UPNP
	};

	template <typename DATA, typename FUNC>
	struct RansacCoef{
		int iterMax = 1e4;
		double minPtNum = 5;
		double thInlrRatio = 0.9;
		double thDist = 0.5;
		double thDistOut = 5.0;
		std::vector<double> weight;
		std::function<void(const std::vector<DATA>&, FUNC&)> calculate_func;
		std::function<void(const FUNC, const std::vector<DATA>&, std::vector<double>&)> calculate_dist;
	};

	struct ViewCam{
		double height;
		double roll;
		double pitch;
		double heightDefault = 20;
		double rollDefault = -PI/2;
		double pitchDefault = 0;
		Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
		Eigen::Vector3d t = Eigen::Vector3d::Zero();
		Eigen::Matrix3d K;
		Eigen::Matrix<double,3,4> P;
		cv::Size imSize;
		cv::Point3d upperLeft, upperRight, lowerLeft, lowerRight;
	};

	struct Parameter{
		double fx, fy, cx, cy;
		double k1, k2, p1, p2, k3;
		Eigen::Matrix3d K, Kinv;
		cv::Mat Kcv;
		std::vector<double> radialDistortion;
		std::vector<double> tangentialDistortion;
		std::vector<double> distCoeffs;
		cv::Size imSize;
		bool applyCLAHE = false;

		Eigen::Matrix4d Tci, Tic;

		MVO::RansacCoef<std::pair<cv::Point3f,cv::Point3f>, double> ransacCoef_scale;
		MVO::RansacCoef<cv::Point3f, std::vector<double>> ransacCoef_plane;
		
		int thInlier = 5;
		double thRatioKeyFrame = 0.9;
		double min_px_dist = 7.0;
		double px_wide = 12.0;
		double max_epiline_dist = 20.0;

		// Statistical model
		double var_theta = std::pow(90.0 / 1241.0 / 2, 2);
		double var_point = 1.0;
		
		// 3D reconstruction
		double vehicle_height;
		double initScale;
		double weightScaleRef;
		double weightScaleReg;
		double reprojError;
		MVO::SVD SVDMethod;
		MVO::TRIANGULATION triangulationMethod;
		MVO::PNP pnpMethod;
		int updateInitPoint;
		int mappingOption;

		// drawing
		MVO::ViewCam view;
	};

	public:

	Parameter params;
	
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
	cv::Point2f calculateRotWarp(cv::Point2f uv);
	
	// Script operations
	void restart();
	void refresh();
	void run(cv::Mat& image);
	void update_gyro(double timestamp, Eigen::Vector3d& gyro);
	void update_velocity(double timestamp, double speed);
	void plot();
	void updateView();

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
	bool verify_solutions(const Eigen::Matrix3d& R, const Eigen::Vector3d& t);
	bool verify_solutions(const std::vector<Eigen::Matrix3d>& R_vec, const std::vector<Eigen::Vector3d>& t_vec,
						  Eigen::Matrix3d& R, Eigen::Vector3d& t);
	bool scale_propagation(const Eigen::Matrix3d& R, Eigen::Vector3d& t,
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
	double calcReconstructionError(Eigen::Matrix4d& Toc);
	double calcReconstructionError(Eigen::Matrix3d& R, Eigen::Vector3d& t);
	double calcReconstructionErrorGT(Eigen::MatrixXd& depth);
	void update_scale_reference(const double scale);

	// RANSAC
	template <typename DATA, typename FUNC>
	static void ransac(const std::vector<DATA>& sample, const MVO::RansacCoef<DATA, FUNC> ransacCoef, FUNC& val, std::vector<bool>& inlier, std::vector<bool>& outlier);
	static std::vector<uint32_t> randperm(uint32_t ptNum, int minPtNum);
	static std::vector<uint32_t> randweightedpick(const std::vector<double>& h, int n = 1);
	static void calculate_scale(const std::vector<std::pair<cv::Point3f,cv::Point3f>>& pts, double& scale);
	static void calculate_scale_error(const double& scale, const std::vector<std::pair<cv::Point3f,cv::Point3f>>& pts, std::vector<double>& dist);
	static void calculate_plane(const std::vector<cv::Point3f>& pts, std::vector<double>& plane);
	static void calculate_plane_error(const std::vector<double>& plane, const std::vector<cv::Point3f>& pts, std::vector<double>& dist);
	static double scale_reference;
	static double scale_reference_weight;
	static Eigen::Matrix3d rotate_prior;

	// Add additional feature within bound-box
	void add_extra_features();
	void extract_roi_features(std::vector<cv::Rect> rois, std::vector<int> nFeature);
	bool extract_roi_feature(cv::Rect& roi);
	std::vector<Feature> features_extra;

	private:

	uint32_t step;
	uint32_t key_step;
	std::vector<uint32_t> keystepVec;

	std::vector<double> timestampSinceKeyframe;
	std::vector<double> speedSinceKeyframe;
	std::vector<Eigen::Vector3d> gyroSinceKeyframe;
	
	cv::Mat prev_image;
	cv::Mat cur_image;
	cv::Mat undist_image;

	cv::Ptr<cv::CLAHE> cvClahe;
	cv::Mat distMap1, distMap2;
	// cv::Ptr<cv::DescriptorExtractor> descriptor;

	Bucket bucket;
	std::vector<Feature> features;
	std::vector<Feature> features_backup;
	std::vector<Feature> features_dead; // for debugging with plot

	bool is_start;
	bool scale_initialized;
	bool rotate_provided;
	bool speed_provided;

	int nFeature;
	int nFeatureMatched;
	int nFeature2DInliered;
	int nFeature3DReconstructed;
	int nFeatureInlier;

	cv::Mat essentialMat;
	Eigen::Matrix3d fundamentalMat;
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
};

#endif //__MVO_HPP__
