#ifndef __MVO_HPP__
#define __MVO_HPP__

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include "core/common.hpp"
#include "core/ransac.hpp"

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

	struct Bucket{
		int safety = 20;
		int max_features = 400;
		cv::Size grid;
		cv::Size size;
		cv::Mat cv_mass;
		cv::Mat cv_prob;
		Eigen::MatrixXd mass;
		Eigen::MatrixXd prob;
		Eigen::MatrixXd saturated;
	};

	struct ViewCam{
		double height;
		double roll;
		double pitch;
		double height_default = 20;
		double roll_default = -M_PI/2;
		double pitch_default = 0;
		Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
		Eigen::Vector3d t = Eigen::Vector3d::Zero();
		Eigen::Matrix3d K;
		Eigen::Matrix<double,3,4> P;
		cv::Size im_size;
		cv::Point3d upper_left, upper_right, lower_left, lower_right;
	};

	struct Parameter{
		double fx, fy, cx, cy;
		double k1, k2, p1, p2, k3;
		Eigen::Matrix3d K, Kinv;
		cv::Mat Kcv;
		std::vector<double> radial_distortion;
		std::vector<double> tangential_distortion;
		std::vector<double> dist_coeffs;
		cv::Size im_size;
		bool apply_clahe = false;

		Eigen::Matrix4d Tci, Tic;

		lsi::RansacCoef<std::pair<cv::Point3f,cv::Point3f>, double> ransac_coef_scale;
		lsi::RansacCoef<cv::Point3f, std::vector<double>> ransac_coef_plane;
		
		int th_inlier = 5;
		double th_ratio_keyframe = 0.9;
		double min_px_dist = 7.0;
		double th_px_wide = 12.0;
		double max_dist = 20.0;
		double max_point_var = 0.1;
		double th_parallax = 0.01;
		double percentile_parallax = 0.75;

		// 3D reconstruction
		double vehicle_height;
		double init_scale;
		double weight_scale_ref;
		double weight_scale_reg;
		double reproj_error;
		MVO::SVD svd_method;
		MVO::TRIANGULATION triangulation_method;
		MVO::PNP pnp_method;

		// Debug
		int update_init_point;
		int output_filtered_depth;
		int mapping_option;

		// drawing
		MVO::ViewCam view;
	};

	struct Frame{
		cv::Mat image;
		int id = -1;

		void copy(const MVO::Frame& im){
			this->image = im.image.clone();
			this->id = im.id;
		}
		cv::Size size() const{
			return this->image.size();
		}
	};

	public:

	Parameter params_;
	static std::ofstream s_file_logger;
	
	public:

	MVO();
	MVO(std::string yaml);
	~MVO(){
		// cv::destroyAllWindows();
	}
	
	// Set image
	void setImage(const cv::Mat& image);

	// Get feature
	std::vector< std::tuple<uint32_t, cv::Point2f, Eigen::Vector3d, double> > getPoints() const;
	std::vector< std::tuple<uint32_t, cv::Point2f, cv::Point2f> > getMotions() const;
	void getPointsInRoi(const cv::Rect& roi, std::vector<uint32_t>& idx) const;
	const std::vector<Feature>& getFeatures() const;
	const Eigen::Matrix4d& getCurrentMotion() const;
	cv::Point2f warpWithIMU(const cv::Point2f& uv) const;
	cv::Point2f warpWithPreviousMotion(const Eigen::Vector3d& p) const;
	
	// Script operations
	void restart();
	void refresh();
	void run(const cv::Mat& image);
	void updateGyro(const double timestamp, const Eigen::Vector3d& gyro);
	void updateVelocity(const double timestamp, const double speed);
	void restartKeyframeLogger();
	void updateView();
	void plot() const;
	void printFeatures() const;

	// Feature operations
	void kltTrackerRough(std::vector<cv::Point2f>& points, std::vector<bool>& validity);
	void kltTrackerPrecise(std::vector<cv::Point2f>& points, std::vector<bool>& validity);
	void selectKeyframeNow(const std::vector<cv::Point2f>& points, const std::vector<bool>& validity);
	void selectKeyframeAfter(const std::vector<cv::Point2f>& points, const std::vector<bool>& validity);
	void updateBucket();
	bool extractFeatures();
	bool updateFeatures();
	void deleteDeadFeatures();
	void addFeatures();
	void addFeature();

	// Calculations
	bool calculateEssential();
	bool calculateMotion();
	bool verifySolutions(const std::vector<Eigen::Matrix3d>& R_vec, const std::vector<Eigen::Vector3d>& t_vec,
						  Eigen::Matrix3d& R, Eigen::Vector3d& t);
	bool scalePropagation(const Eigen::Matrix3d& R, Eigen::Vector3d& t,
						   std::vector<bool>& inlier, std::vector<bool>& outlier);
	bool findPoseFrom3DPoints(Eigen::Matrix3d &R, Eigen::Vector3d &t, std::vector<int>& inlier, std::vector<int>& outlier);
	void constructDepth(const std::vector<cv::Point2f>& uv_prev, const std::vector<cv::Point2f>& uv_curr, 
                        const Eigen::Matrix3d& R, const Eigen::Vector3d& t, 
                        std::vector<Eigen::Vector3d> &X_prev, std::vector<Eigen::Vector3d> &X_curr, 
                        std::vector<bool> &inlier);
	void update3DPoints(const Eigen::Matrix3d& R, const Eigen::Vector3d& t,
						const std::vector<bool>& inlier, const std::vector<bool>& outlier,
						Eigen::Matrix4d& T, Eigen::Matrix4d& Toc, Eigen::Vector4d& Poc);
	void update3DPoints(const Eigen::Matrix3d& R, const Eigen::Vector3d& t,
						const std::vector<bool>& inlier, const std::vector<bool>& outlier,
						const Eigen::Matrix3d& R_E, const Eigen::Vector3d& t_E, const bool& success_E,
						Eigen::Matrix4d& T, Eigen::Matrix4d& Toc, Eigen::Vector4d& Poc);
	void update3DPoint(Feature& feature, const Eigen::Matrix4d& Toc, const Eigen::Matrix4d& T);
	void updateScaleReference(const double scale);

	// Error
	double calcReconstructionError(Eigen::Matrix4d& Toc) const;
	double calcReconstructionError(Eigen::Matrix3d& R, Eigen::Vector3d& t) const;
	void calcReconstructionErrorGT(Eigen::MatrixXd& depth) const;

	// Add additional feature within bound-box
	void addExtraFeatures();
	void updateRoiFeatures(const std::vector<cv::Rect>& rois, const std::vector<int>& nFeature);
	bool updateRoiFeature(const cv::Rect& roi, const std::vector<cv::Point2f>& keypoints, std::vector<uint32_t>& idx_compared);
	std::vector<Feature> features_extra_;

	private:

	uint32_t step_;
	uint32_t keystep_;
	std::vector<uint32_t> keystep_array_;
	bool trigger_keystep_decrease_ = false;
	bool trigger_keystep_increase_ = false;
	bool trigger_keystep_decrease_previous_ = false;

	std::vector<double> timestamp_speed_since_keyframe_;
	std::vector<double> timestamp_imu_since_keyframe_;
	std::vector<double> speed_since_keyframe_;
	std::vector<Eigen::Vector3d> gyro_since_keyframe_;
	
	MVO::Frame prev_keyframe_;
	MVO::Frame curr_keyframe_;
	MVO::Frame next_keyframe_;
	MVO::Frame prev_frame_;
	MVO::Frame curr_frame_;
	cv::Mat excludeMask_;

	cv::Ptr<cv::CLAHE> cvClahe_;
	cv::Mat distort_map1_, distort_map2_;
	// cv::Ptr<cv::DescriptorExtractor> descriptor;

	MVO::Bucket bucket_;
	std::vector<bool> visit_bucket_;
	std::vector<std::vector<cv::Point2f>> keypoints_of_bucket_;
	std::vector<Feature> features_;
	std::vector<Feature> features_dead_; // for debugging with plot

	bool is_start_;
	bool is_scale_initialized_;
	bool is_rotate_provided_;
	bool is_speed_provided_;

	Eigen::Matrix3d rotate_prior_;
	double scale_reference_ = -1;

	int num_feature_;
	int num_feature_matched_;
	int num_feature_2D_inliered_;
	int num_feature_3D_reconstructed_;
	int num_feature_inlier_;

	cv::Mat essential_;
	Eigen::Matrix3d fundamental_;
	std::vector<Eigen::Matrix3d> R_vec_;
	std::vector<Eigen::Vector3d> t_vec_;
	std::vector<Eigen::Matrix4d> TRec_;
	std::vector<Eigen::Matrix4d> TocRec_;
	std::vector<Eigen::Vector4d> PocRec_;

	// std::vector<cv::Mat> prev_pyramid_template_, curr_pyramid_template_;
	Eigen::MatrixXd map_matrix_template_;
	std::shared_ptr< Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> > eigen_solver_;
};

#endif //__MVO_HPP__
