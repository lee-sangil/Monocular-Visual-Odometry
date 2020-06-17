#ifndef __MVO_HPP__
#define __MVO_HPP__

#include <fstream>
#include <iostream>
#include <string>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/StdVector>

#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include "core/common.hpp"
#include "core/ransac.hpp"
// #include "core/g2o.hpp"

/**
 * @brief 영상 항법 모듈 클래스.
 * @details 차량에 부착된 단안 카메라로부터 얻은 이미지, 그리고 옵션에 따라 IMU, CAN 차속 데이터를 입력으로 하여 차량의 위치를 추정하고, 동시에 제작된 지역적 지도를 이용하여 단안 카메라의 물리적인 한계인 스케일 모호성(scale ambiguity)과 스케일 표류(scale drift)를 줄이면서 깊이 필터를 통해 이미지 내 물체의 깊이를 추정한다.
 * @author Sangil Lee (sangillee724@gmail.com)
 * @date 25-Dec-2019
 */
class MVO{
	public:

	/** @brief SVD 계산 방법. */
	enum SVD{
		JACOBI,
		BDC,
		OpenCV,
		Eigen
	};

	/** @brief 삼각측량 계산 방법. */
	enum TRIANGULATION{
		MIDP, /**< mid-point */
		LLS /**< linear least square */
	};
	
	/** @brief PnP 계산 방법. */
	enum PNP{
		LM, /**< levenberg-marquardt */
		ITERATIVE, /**< iterative */
		AP3P, /**< AP3P */
		EPNP, /**< Efficient PnP */
		DLS, /**< Direct Least-Squares */
		UPNP /**< Unified PnP */
	};

	/** @brief 이미지 내 영역 분할 구조체 */
	struct Bucket{
		int safety = 20; /**< @brief 특징점이 추출되지 않는 Bucket 가장자리의 너비 */
		int max_features = 400; /**< @brief 이미지 내에서 특징점 추출 시, 특징점의 최대 개수 */
		cv::Size grid; /**< @brief Bucket의 가로/세로 개수 */
		cv::Size size; /**< @brief Bucket의 픽셀 사이즈 */
		cv::Mat cv_mass; /**< @brief Bucket의 특징점 추출 시 가중치 */
		cv::Mat cv_prob; /**< @brief Bucket의 특징점 추출 시 선택 확률 */
		Eigen::MatrixXd mass;  /**< @brief Bucket의 특징점 추출 시 가중치 */
		Eigen::MatrixXd prob;  /**< @brief Bucket의 특징점 추출 시 선택 확률 */
		bool * saturated; /**< @brief 유효한 특징점이 더이상 추출되지 않는 Bucket 표시 */
	};

	/** @brief 시각화를 위한 촬영 카메라의 파라미터 */
	struct ViewCam{
		double height; /**< @brief 촬영 카메라의 지면으로부터 높이 */
		double roll; /**< @brief 촬영 카메라의 roll 방향 기울기 */
		double pitch; /**< @brief 촬영 카메라의 pitch 방향 기울기 */
		double height_default = 20; /**< @brief 촬영 카메라의 지면으로부터 초기 높이 */
		double roll_default = -M_PI/2; /**< @brief 촬영 카메라의 roll 방향 초기 기울기 */
		double pitch_default = 0; /**< @brief 촬영 카메라의 pitch 방향 초기 기울기 */
		Eigen::Matrix3d R = Eigen::Matrix3d::Identity(); /**< @brief 원점 좌표계 기준 촬영 카메라의 회전 행렬 */
		Eigen::Vector3d t = Eigen::Vector3d::Zero(); /**< @brief 원점 좌표계 기준 촬영 카메라의 변위 벡터 */
		Eigen::Matrix3d K; /**< @brief 촬영 카메라의 내부 파라미터 */
		Eigen::Matrix<double,3,4> P; /**< @brief 촬영 카메라의 투영 파라미터 */
		cv::Size im_size; /**< @brief 촬영 카메라의 해상도 */
		cv::Point3d upper_left, upper_right, lower_left, lower_right; /**< @brief 촬영 카메라 프레임의 절두체 크기 */
	};

	/** @brief 영상 항법 모듈 파라미터 */
	struct Parameter{
		double fx, fy, cx, cy; /**< @brief 카메라 내부 파라미터 */
		double k1, k2, p1, p2, k3; /**< @brief 카메라 왜곡 파라미터 */
		Eigen::Matrix3d K, Kinv; /**< @brief 카메라 내부 파라미터 행렬과 역행렬 */
		cv::Mat Kcv; /**< @brief 카메라 내부 파라미터 cv::Mat 형식 */
		std::vector<double> radial_distortion; /**< @brief 카메라 방사형 왜곡 파라미터 */
		std::vector<double> tangential_distortion; /**< @brief 카메라 접선 왜곡 파라미터 */
		std::vector<double> dist_coeffs; /**< @brief 카메라 왜곡 파라미터 벡터 */
		cv::Size im_size; /**< @brief 카메라 이미지 크기 */
		bool apply_clahe = false; /**< @brief Clahe 알고리즘 적용 유무 */

		Eigen::Matrix4d Tci, Tic; /**< @brief 카메라와 IMU 사이의 변환 행렬 */

		lsi::RansacCoef<std::pair<cv::Point3f,cv::Point3f>, double> ransac_coef_scale; /**< @brief 스케일 추정시 RANSAC 파라미터 */
		lsi::RansacCoef<cv::Point3f, std::vector<double>> ransac_coef_plane; /**< @brief 지면 탐색시 RANSAC 파라미터 */
		
		int th_inlier = 5; /**< @brief 각 프로세스를 통과하기 위한 최소 특징점 인라이어 개수 */
		double th_ratio_keyframe = 0.9; /**< @brief 키프레임을 유지하기 위한 최소 특징점 매칭률 */
		double min_px_dist = 7.0; /**< @brief 특징점 사이의 최소 픽셀 거리 */
		double th_px_wide = 12.0; /**< @brief 특징점 사이의 시차가 충분한지 판단 */
		double max_dist = 20.0; /**< @brief 특징점과 예상 특징점 위치 사이의 오차 픽셀 거리 */
		double max_point_var = 0.1; /**< @brief 3차원 좌표의 최대 분산 */
		double th_parallax = 0.01; /**< @brief 키프레임을 유지하기 위한 최대 특징점 시차 */
		double percentile_parallax = 0.75; /**< @brief 특징점 시차 분위수 */

		// 3D reconstruction
		double vehicle_height; /**< @brief 가정한 차체 높이 */
		double weight_scale_ref; /**< @brief 스케일 참조값의 가중치 */
		double weight_scale_reg; /**< @brief 스케일 참조값의 저주파 필터 (LPF) 파라미터 */
		double reproj_error; /**< @brief PnP 재투영 에러 경계값 */
		MVO::SVD svd_method; /**< @brief SVD 계산 방법 */
		MVO::TRIANGULATION triangulation_method; /**< @brief 삼각측량 계산 방법 */
		MVO::PNP pnp_method; /**< @brief PnP 계산 방법 */

		// Debug
		int update_init_point; /**< @brief 3차원 좌표 깊이 필터 업데이트 유무 */
		int output_filtered_depth; /**< @brief 표시할 3차원 좌표 선택 */
		int mapping_option; /**< @brief 스케일 보원 및 자세 추정 방법 선택 */

		// drawing
		MVO::ViewCam view; /**< @brief 촬영 카메라 파라미터 */
	};

	/** @brief 이미지 프레임 파라미터 */
	struct Frame{
		cv::Mat image; /**< 이미지 */
		double timestamp = 0; /**< 타임스탬프 */
		int id = -1; /**< ID */
		std::vector<uint32_t> landmarks; /**< 프레임에서 보이는 landmark의 Key */

		bool is_keyframe;

		std::vector<std::pair<double,double>> linear_velocity_since_; /**< 이미지 프레임 이후 입력된 선속도 데이터 */
		std::vector<std::pair<double,Eigen::Vector3d>> angular_velocity_since_; /**< 이미지 프레임 이후 입력된 각속도 데이터 */
		
		/**
		 * @brief 이미지 프레임 복사.
		 * @details 이미지 프레임의 이미지와 타임스탬프, ID, 선속도, 각속도 데이터를 복사한다.
		 * @param im 이미지 프레임
		 * @return 없음
		 * @author Sangil Lee (sangillee724@gmail.com)
		 * @date 25-Dec-2019
		 */
		void assign(const MVO::Frame& im){
			this->image = im.image.clone();
			this->timestamp = im.timestamp;
			this->id = im.id;
			this->angular_velocity_since_.assign(im.angular_velocity_since_.begin(),im.angular_velocity_since_.end());
			this->linear_velocity_since_.assign(im.linear_velocity_since_.begin(),im.linear_velocity_since_.end());
		}

		/**
		 * @brief 이미지 프레임 병합.
		 * @details 이미지 프레임의 이미지와 타임스탬프, ID를 복사하고, 선속도, 각속도 데이터를 기존 데이터 앞에 삽입한다.
		 * @param im 이미지 프레임
		 * @return 없음
		 * @author Sangil Lee (sangillee724@gmail.com)
		 * @date 25-Dec-2019
		 */
		void merge(const MVO::Frame& im){
			this->image = im.image.clone();
			this->timestamp = im.timestamp;
			this->id = im.id;
			this->angular_velocity_since_.insert(this->angular_velocity_since_.begin(),im.angular_velocity_since_.begin(),im.angular_velocity_since_.end());
			this->linear_velocity_since_.insert(this->linear_velocity_since_.begin(),im.linear_velocity_since_.begin(),im.linear_velocity_since_.end());
		}

		/**
		 * @brief 이미지 사이즈 출력.
		 * @return 이미지 사이즈
		 * @author Sangil Lee (sangillee724@gmail.com)
		 * @date 25-Dec-2019
		 */
		cv::Size size() const{
			return this->image.size();
		}
	};

	public:

	Parameter params_; /**< @brief 영상 항법 모듈 파라미터 */
	static std::ofstream s_point_logger_; /**< @brief 포인트 클라우드 텍스트 파일 출력 */
	static std::ofstream s_file_logger_; /**< @brief 알고리즘 로그 출력 */
	
	public:

	MVO();
	MVO(std::string yaml);
	~MVO(){
		// cv::destroyAllWindows();
	}
	
	// Set image
	void setImage(const cv::Mat& image, double timestamp = -1);
	void setImage(const cv::Mat& image, const cv::Mat& image_right, double timestamp = -1); // for stereo

	// Get feature
	std::vector< std::tuple<uint32_t, cv::Point2f, Eigen::Vector3d, double> > getPoints() const;
	std::vector< std::tuple<uint32_t, cv::Point2f, cv::Point2f> > getMotions() const;
	void getPointsInRoi(const cv::Rect& roi, std::vector<uint32_t>& idx) const;
	const std::vector<Feature>& getFeatures() const;
	const Eigen::Matrix4d& getCurrentMotion() const;
	cv::Point2f warpWithIMU(const cv::Point2f& uv) const;
	cv::Point2f warpWithCAN(const Eigen::Vector3d& p) const;
	cv::Point2f warpWithPreviousMotion(const Eigen::Vector3d& p) const;
	
	// Script operations
	void restart();
	void refresh();
	void run(const cv::Mat& image, double timestamp = -1);
	void run(const cv::Mat& image, const cv::Mat& image_right, double timestamp = -1); // for stereo
	void updateGyro(const double timestamp, const Eigen::Vector3d& gyro);
	void updateVelocity(const double timestamp, const double speed);
	void updateView();
	void plot(const Eigen::MatrixXd * const depthMap = NULL) const;
	void printFeatures() const;
	void printPose(std::ofstream& os) const;

	// Feature operations
	void kltTrackerRough(std::vector<cv::Point2f>& points, std::vector<bool>& validity);
	void kltTrackerPrecise(std::vector<cv::Point2f>& points, std::vector<bool>& validity);
	void selectKeyframeNow();
	void selectKeyframeAfter();
	void updateBucket();
	bool extractFeatures();
	bool updateFeatures();
	void deleteDeadFeatures();
	void addFeatures();
	void addFeature();

	// Calculations
	bool calculateEssential();
	bool calculateEssentialStereo();
	bool calculateMotion();
	bool calculateMotionStereo();

	bool verifySolutions(const std::vector<Eigen::Matrix3d>& R_vec, const std::vector<Eigen::Vector3d>& t_vec,
						  Eigen::Matrix3d& R, Eigen::Vector3d& t, std::vector<bool>& max_inlier, std::vector<Eigen::Vector3d>& opt_X_curr);
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
	void update3DPointsStereo(const Eigen::Matrix3d& R, const Eigen::Vector3d& t,
						const std::vector<bool>& inlier, const std::vector<bool>& outlier,
						Eigen::Matrix4d& T, Eigen::Matrix4d& Toc, Eigen::Vector4d& Poc);
	void update3DPoint(Feature& feature, const Eigen::Matrix4d& Toc, const Eigen::Matrix4d& T);
	void updateScaleReference(double scale = -1);
	void updateRotatePrior();

	// Error
	double calcReconstructionError(Eigen::Matrix4d& Toc) const;
	double calcReconstructionError(Eigen::Matrix3d& R, Eigen::Vector3d& t) const;
	void calcReconstructionErrorGT(Eigen::MatrixXd& depth) const;

	// Add additional feature within bound-box
	void addExtraFeatures();
	void updateRoiFeatures(const std::vector<cv::Rect>& rois, const std::vector<int>& nFeature);
	bool updateRoiFeature(const cv::Rect& roi, const std::vector<cv::Point2f>& keypoints, std::vector<uint32_t>& idx_compared);
	std::vector<Feature> features_extra_; /**< @brief ROI 내에서 추출된 특징점 */

	private:

	uint32_t step_; /**< @brief 현재 스텝 */
	uint32_t keystep_; /**< @brief 현재 키프레임 스텝 */
	std::vector<uint32_t> keystep_array_; /**< @brief 키프레임 스텝 벡터 */
	bool trigger_keystep_decrease_ = false; /**< @brief 키프레임이 줄었을때, true */
	bool trigger_keystep_increase_ = false; /**< @brief 키프레임이 늘었을때, true */
	bool trigger_keystep_decrease_previous_ = false; /**< @brief trigger_keystep_decrease_의 이전값 */
	
	MVO::Frame prev_keyframe_; /**< @brief 이전 키프레임 */
	MVO::Frame curr_keyframe_; /**< @brief 현재 키프레임 */
	MVO::Frame next_keyframe_; /**< @brief 다음 키프레임 */
	std::vector<MVO::Frame> keyframes_; /**< @brief 키프레임들 */

	MVO::Frame prev_frame_; /**< @brief 이전 이미지 프레임 */
	MVO::Frame curr_frame_; /**< @brief 현재 이미지 프레임 */
	MVO::Frame curr_frame_right_; /**< @brief 현재 스테레오 이미지 프레임 */
	cv::Mat excludeMask_; /**< @brief 특징점을 추출할 영역 */
	cv::Mat disparity_; /**< @brief KITTI disparity 프레임 */

	cv::Ptr<cv::CLAHE> cvClahe_; /**< @brief Clahe 객체 */
	cv::Mat distort_map1_, distort_map2_; /**< @brief 이미지 왜곡 변환 매핑 */
	// cv::Ptr<cv::DescriptorExtractor> descriptor;

	MVO::Bucket bucket_; /**< @brief bucket 객체 */
	std::vector<bool> visit_bucket_; /**< @brief 특징점 추출 시 bucket을 방문하면, true */
	std::vector<std::vector<cv::Point2f>> keypoints_of_bucket_; /**< @brief bucket에서 추출된 특징점 벡터 */
	std::vector<Feature> features_; /**< @brief 현재 사용중인 특징점 */
	std::vector<Feature> features_dead_; /**< @brief 사라진 특징점(디버깅용), 3차원 초기화된 특징점만 누적됨. */
	std::unordered_map<uint32_t, std::shared_ptr<Landmark>> landmark_; /**< @brief 랜드마크 */

	bool is_start_; /**< @brief 카메라가 움직이기 시작하면, true */
	bool is_scale_initialized_; /**< @brief 3차원 스케일이 초기화되면, true */
	bool is_rotate_provided_; /**< @brief 자이로값이 제공되면, true */
	bool is_speed_provided_; /**< @brief 차속값이 제공되면, true */

	Eigen::Matrix3d rotate_prior_; /**< @brief 자이로로부터 계산한 예상 회전 행렬 */
	double scale_reference_ = -1; /**< @brief 차속값으로부터 계산한 변위값 */

	int num_feature_; /**< @brief 특징점의 개수 */
	int num_feature_matched_; /**< @brief 인접한 이미지에서 매칭된 특징점의 개수 */
	int num_feature_2D_inliered_; /**< @brief 주류 움직임에 속하는 특징점의 개수 */
	int num_feature_3D_reconstructed_; /**< @brief 3차원 복원된 특징점의 개수 */
	int num_feature_inlier_; /**< @brief 스케일이 정상적으로 복원된 특징점의 개수 */

	std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> TRec_; /**< @brief 두 이미지 사이의 변환 행렬 */
	std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> TocRec_; /**< @brief 초기 위치로부터의 변환 행렬 */
	std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> PocRec_; /**< @brief 초기 위치로부터의 변위 */
	Eigen::Matrix3d R_;
	Eigen::Vector3d t_;

	// std::vector<cv::Mat> prev_pyramid_template_, curr_pyramid_template_;
	Eigen::MatrixXd map_matrix_template_; /**< @brief 빠른 SVD 계산을 위한 임시 메모리 공간 할당 */
	std::shared_ptr< Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> > eigen_solver_; /**< @brief eigen 벡터 계산기 객체 */

	// g2o::BlockSolverX::LinearSolverType * linear_solver_ = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();
    // g2o::BlockSolverX * block_solver_ = new g2o::BlockSolverX(linear_solver_);
    // g2o::OptimizationAlgorithm * algorithm_ = new g2o::OptimizationAlgorithmLevenberg(block_solver_);
    // g2o::SparseOptimizer * optimizer_ = new g2o::SparseOptimizer;
	// g2o::CameraParameters * cam_params_;
};

#endif //__MVO_HPP__
