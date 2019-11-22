#include "core/MVO.hpp"
#include "core/utils.hpp"
#include "core/numerics.hpp"
#include "core/time.hpp"
#include "core/DepthFilter.hpp"

Eigen::Matrix3d MVO::rotate_prior;
double DepthFilter::px_error_angle;
double DepthFilter::meas_max;

MVO::MVO(){
    this->step = -1;
    this->key_step = 0;
    this->keystepVec.push_back(0);

    this->is_start = false;
    this->scale_initialized = false;
    this->speed_provided = false;
    this->rotate_provided = false;
    this->cvClahe = cv::createCLAHE();

    // Variables
    this->nFeature = 0;
    this->nFeatureMatched = 0;
    this->nFeature2DInliered = 0;
    this->nFeature3DReconstructed = 0;
    this->nFeatureInlier = 0;

    // Initial position
    this->TRec.push_back(Eigen::Matrix4d::Identity());
    this->TocRec.push_back(Eigen::Matrix4d::Identity());
    this->PocRec.push_back((Eigen::Vector4d() << 0,0,0,1).finished());

    this->R_vec.reserve(4);
	this->t_vec.reserve(4);
}

MVO::MVO(std::string yaml):MVO(){

    /**************************************************************************
	 *  Read .yaml file
	 **************************************************************************/
    cv::FileStorage fSettings(yaml, cv::FileStorage::READ);
	if (!fSettings.isOpened()){
		std::cout << "Failed to open: " << yaml << std::endl;
		abort();
	}

    double version = fSettings["Version"];
    if( version != YAML_VERSION ){
        std::cout << "YAML file is an old version (your version is \"" << std::setfill('0') << std::setprecision(1) << std::fixed << version << "\", required is \"" << YAML_VERSION << "\"" << std::endl;
        abort();
    }

	this->params.fx =			    fSettings["Camera.fx"];
	this->params.fy =			    fSettings["Camera.fy"];
	this->params.cx =			    fSettings["Camera.cx"];
	this->params.cy =			    fSettings["Camera.cy"];
	this->params.k1 =			    fSettings["Camera.k1"];
	this->params.k2 =			    fSettings["Camera.k2"];
	this->params.p1 =			    fSettings["Camera.p1"];
	this->params.p2 =			    fSettings["Camera.p2"];
	this->params.k3 =			    fSettings["Camera.k3"];
	this->params.imSize.width =	    fSettings["Camera.width"];
	this->params.imSize.height =	fSettings["Camera.height"];

    this->params.K << this->params.fx, 0, this->params.cx,
						0, this->params.fy, this->params.cy,
						0, 0, 1;
    this->params.Kinv = this->params.K.inverse();
    cv::eigen2cv(this->params.K, this->params.Kcv);

	this->params.radialDistortion.push_back(params.k1);
	this->params.radialDistortion.push_back(params.k2);
	this->params.radialDistortion.push_back(params.k3);
	this->params.tangentialDistortion.push_back(params.p1);
	this->params.tangentialDistortion.push_back(params.p2);
    this->params.distCoeffs.insert(this->params.distCoeffs.begin(), this->params.radialDistortion.begin(), this->params.radialDistortion.begin()+1);
    this->params.distCoeffs.insert(this->params.distCoeffs.end(), this->params.tangentialDistortion.begin(), this->params.tangentialDistortion.end());
    this->params.distCoeffs.insert(this->params.distCoeffs.begin(), this->params.radialDistortion.begin()+2, this->params.radialDistortion.end());

    cv::initUndistortRectifyMap(this->params.Kcv, this->params.distCoeffs, cv::Mat(), this->params.Kcv, this->params.imSize, CV_32FC1, this->distMap1, this->distMap2);

    if( !fSettings["Camera.T_IC"].empty() ){
        cv::Mat Tic;
        fSettings["Camera.T_IC"] >> Tic;
        cv::cv2eigen(Tic, this->params.Tic);
        this->params.Tci = this->params.Tic.inverse();
    }else if( !fSettings["Camera.T_CI"].empty() ){
        cv::Mat Tci;
        fSettings["Camera.T_CI"] >> Tci;
        cv::cv2eigen(Tci, this->params.Tci);
        this->params.Tic = this->params.Tci.inverse();
    }else if( !fSettings["Camera.T_IV"].empty() && !fSettings["Camera.T_VC"].empty() ){
        Eigen::Matrix4d Tiv_, Tvc_;
        cv::Mat Tiv, Tvc;
        fSettings["Camera.T_IV"] >> Tiv;
        fSettings["Camera.T_VC"] >> Tvc;
        cv::cv2eigen(Tiv, Tiv_);
        cv::cv2eigen(Tvc, Tvc_);

        this->params.Tci = Tvc_ * Tiv_;
        this->params.Tic = this->params.Tci.inverse();
    }

    // this->descriptor = cv::BRISK::create();

    this->params.thInlier =         fSettings["Feature.thInlier"];
    this->params.thRatioKeyFrame =  fSettings["Feature.thRatioKeyFrame"];
    this->params.min_px_dist =      fSettings["Feature.min_px_dist"];
    this->params.px_wide =          fSettings["Feature.px_wide"];
    this->params.max_epiline_dist = fSettings["Feature.max_epiline_dist"];

    // RANSAC parameter
    this->params.ransacCoef_scale.iterMax =        fSettings["RANSAC.iterMax"];
    this->params.ransacCoef_scale.thInlrRatio =    fSettings["RANSAC.thInlrRatio"];
    this->params.ransacCoef_scale.minPtNum =       fSettings["RANSAC.scale.nSample"];
    this->params.ransacCoef_scale.thDist =         fSettings["RANSAC.scale.thDist"]; // standard deviation
    this->params.ransacCoef_scale.thDistOut =      fSettings["RANSAC.scale.thDistOut"]; // three times of standard deviation
    this->params.ransacCoef_scale.calculate_func = MVO::calculate_scale;
    this->params.ransacCoef_scale.calculate_dist = MVO::calculate_scale_error;
    
    this->params.ransacCoef_plane.iterMax =        fSettings["RANSAC.iterMax"];
    this->params.ransacCoef_plane.thInlrRatio =    fSettings["RANSAC.thInlrRatio"];
    this->params.ransacCoef_plane.minPtNum =       3;
    this->params.ransacCoef_plane.thDist =         fSettings["RANSAC.plane.thDist"]; // standard deviation
    this->params.ransacCoef_plane.thDistOut =      fSettings["RANSAC.plane.thDistOut"]; // three times of standard deviation
    this->params.ransacCoef_plane.calculate_func = MVO::calculate_plane;
    this->params.ransacCoef_plane.calculate_dist = MVO::calculate_plane_error;

    // Bucket
    this->bucket = Bucket();
    this->bucket.max_features =     fSettings["Feature.num"];
    this->bucket.safety =           fSettings["Bucket.safety"];
    int bucketGridRows =            fSettings["Bucket.rows"];
    int bucketGridCols =            fSettings["Bucket.cols"];

	this->bucket.grid = cv::Size(bucketGridCols,bucketGridRows);
	this->bucket.size = cv::Size(this->params.imSize.width/this->bucket.grid.width, this->params.imSize.height/this->bucket.grid.height);
	this->bucket.mass.setZero(this->bucket.grid.height, this->bucket.grid.width);
	this->bucket.prob.resize(this->bucket.grid.height, this->bucket.grid.width);
    this->bucket.prob.fill(1.0);
    this->bucket.saturated.setZero(this->bucket.grid.height, this->bucket.grid.width);

    this->features.reserve(this->bucket.max_features);
    this->features_backup.reserve(this->bucket.max_features);
    this->idxTemplate.reserve(this->bucket.max_features);
    this->inlierTemplate.reserve(this->bucket.max_features);
    this->prevPyramidTemplate.reserve(10);
    this->currPyramidTemplate.reserve(10);

    // 3D reconstruction
    this->params.initScale =        1;
    this->params.vehicle_height =   fSettings["Scale.height"]; // in meter
    this->params.weightScaleRef =   fSettings["Scale.referenceWeight"];
	this->params.weightScaleReg =   fSettings["Scale.regularizationWeight"];
    this->params.reprojError =      fSettings["PnP.threshold"];
    this->params.updateInitPoint =  fSettings["Debug.updateInitPoints"];
    this->params.mappingOption =    fSettings["Debug.mappingOptions"];

    this->eigenSolver = new Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>();

    switch( fSettings["Triangulation.Method"] ){
        case 0:
            this->params.triangulationMethod = MVO::TRIANGULATION::MIDP;
            break;
        case 1:
            this->params.triangulationMethod = MVO::TRIANGULATION::LLS;
            break;
        default:
            abort();
    }
    
    switch( fSettings["Triangulation.SVD"] ){
        case 0:
            this->params.SVDMethod = MVO::SVD::JACOBI;
            break;
        case 1:
            this->params.SVDMethod = MVO::SVD::BDC;
            break;
        case 2:
            this->params.SVDMethod = MVO::SVD::OpenCV;
            break;
        case 3:
            this->params.SVDMethod = MVO::SVD::Eigen;
            break;
        default:
            abort();
    }

    switch( fSettings["PNP.Method"] ){
        case 0:
            this->params.pnpMethod = MVO::PNP::LM;
            break;
        case 1:
            this->params.pnpMethod = MVO::PNP::ITERATIVE;
            break;
        case 2:
            this->params.pnpMethod = MVO::PNP::AP3P;
            break;
        case 3:
            this->params.pnpMethod = MVO::PNP::EPNP;
            break;
        case 4:
            this->params.pnpMethod = MVO::PNP::DLS;
            break;
        case 5:
            this->params.pnpMethod = MVO::PNP::UPNP;
            break;
        default:
            abort();
    }

    double depth_min = fSettings["DepthFilter.depth_min"];
    double px_noise = fSettings["DepthFilter.px_noise"];
    DepthFilter::meas_max = 1.0/depth_min;
    DepthFilter::px_error_angle = std::atan(px_noise/(this->params.fx+this->params.fy))*2.0; // law of chord (sehnensatz)

    cv::namedWindow("MVO");
    cv::moveWindow("MVO", 20, 20);

    cv::namedWindow("Trajectory");
    cv::moveWindow("Trajectory", 1320, 20);

    cv::namedWindow("Depth");
    cv::moveWindow("Depth", 20, 440);

    this->params.view.heightDefault =   fSettings["viewCam.height"]; // in world coordinate
    this->params.view.rollDefault =     (double) fSettings["viewCam.roll"] * M_PI/180; // radian
    this->params.view.pitchDefault =    (double) fSettings["viewCam.pitch"] * M_PI/180; // radian
    this->params.view.height =          this->params.view.heightDefault;
    this->params.view.roll =            this->params.view.rollDefault;
    this->params.view.pitch =           this->params.view.pitchDefault;
    this->params.view.imSize = cv::Size(600,600);
    this->params.view.K <<  300,   0, 300,
						      0, 300, 300,
						      0,   0,   1;
    this->params.view.upperLeft =   cv::Point3d(-1,-.5, 1);
    this->params.view.upperRight =  cv::Point3d( 1,-.5, 1);
    this->params.view.lowerLeft =   cv::Point3d(-1, .5, 1);
    this->params.view.lowerRight =  cv::Point3d( 1, .5, 1);
}

void MVO::refresh(){
    this->nFeatureMatched = 0;
    this->nFeature2DInliered = 0;
    this->nFeature3DReconstructed = 0;
    this->nFeatureInlier = 0;

    for( int i = 0; i < this->nFeature; i++ ){
        this->features[i].is_matched = false;
        this->features[i].is_2D_inliered = false;
        this->features[i].is_3D_reconstructed = false;
    }
}

void MVO::restart(){
    this->step = -1;
    this->key_step = 0;
    this->keystepVec.clear();
    this->keystepVec.push_back(0);

    this->is_start = false;
    this->scale_initialized = false;
    
    this->features.clear();
    this->features_dead.clear();

    // Variables
    this->nFeature = 0;
    this->nFeatureMatched = 0;
    this->nFeature2DInliered = 0;
    this->nFeature3DReconstructed = 0;
    this->nFeatureInlier = 0;

    // Initial position
    this->TRec.clear();
    this->TocRec.clear();
    this->PocRec.clear();

    this->TRec.push_back(Eigen::Matrix4d::Identity());
    this->TocRec.push_back(Eigen::Matrix4d::Identity());
    this->PocRec.push_back((Eigen::Vector4d() << 0,0,0,1).finished());
}

void MVO::set_image(cv::Mat& image){
    this->prev_image = this->cur_image.clone();
    cv::remap(image, this->undist_image, this->distMap1, this->distMap2, cv::INTER_AREA);
    
    if( this->params.applyCLAHE )
        cvClahe->apply(this->undist_image, this->cur_image);
    else
        this->cur_image = this->undist_image.clone();

    this->step++;
    this->key_step = this->keystepVec.back();
}

void MVO::run(cv::Mat& image){
    
    lsi::tic();
    this->set_image(image);
    std::cerr << "============ Iteration: " << this->step << " ============" << std::endl;
    std::cerr << "# Grab image: " << lsi::toc() << std::endl;
    this->refresh();

    // this->extract_roi_features(rois, nFeature);   // Extract extra features in rois

    std::array<bool,3> success;
    success[0] = this->extract_features();       // Extract and update features
    success[1] = this->calculate_essential();    // RANSAC for calculating essential/fundamental matrix
    success[2] = this->calculate_motion();       // Extract rotational and translational from fundamental matrix

    if( !std::all_of(success.begin(), success.end(), [](bool b){return b;}) )
        this->restart();
    
    // std::cout << "start: " << this->is_start << ", key_step: " << this->key_step << " " << std::endl;
}

void MVO::update_gyro(double timestamp, Eigen::Vector3d& gyro){
    this->rotate_provided = true;
    this->timestampSinceKeyframe.push_back(timestamp);
    this->gyroSinceKeyframe.push_back(gyro);

    std::vector<double> timestampDiff;
    Eigen::Vector3d radian = Eigen::Vector3d::Zero();
    for( uint32_t i = 0; i < this->gyroSinceKeyframe.size()-1; i++ )
        radian += (this->gyroSinceKeyframe[i]+this->gyroSinceKeyframe[i+1])/2 * (this->timestampSinceKeyframe[i+1]-this->timestampSinceKeyframe[i]);

    MVO::rotate_prior = this->params.Tci.block(0,0,3,3) * skew(-radian).exp() * this->params.Tic.block(0,0,3,3);
}

void MVO::update_velocity(double timestamp, double speed){
    this->speed_provided = true;
    this->timestampSinceKeyframe.push_back(timestamp);
    this->speedSinceKeyframe.push_back(speed);

    std::vector<double> timestampDiff;
    double scale = 0.0;
    for( uint32_t i = 0; i < this->speedSinceKeyframe.size()-1; i++ )
        scale += (this->speedSinceKeyframe[i]+this->speedSinceKeyframe[i+1])/2 * (this->timestampSinceKeyframe[i+1]-this->timestampSinceKeyframe[i]);
    
    this->update_scale_reference(scale);
}

std::vector<Feature> MVO::get_features() const {
    return this->features;
}

ptsROI_t MVO::get_points() const
{
    ptsROI_t ptsROI;
    for( uint32_t i = 0; i < this->features.size(); i++ ){
        ptsROI.push_back( std::make_tuple(this->features[i].uv.back(), this->features[i].point.block(0, 0, 3, 1) ) );
    }
    return ptsROI;
}

cv::Point2f MVO::calculateRotWarp(cv::Point2f uv){
    Eigen::Vector3d pixel, warpedPixel;
    cv::Point2f warpedUV;
    pixel << uv.x, uv.y, 1;
    warpedPixel = this->params.K * MVO::rotate_prior * this->params.Kinv * pixel;
    warpedUV.x = warpedPixel(0)/warpedPixel(2);
    warpedUV.y = warpedPixel(1)/warpedPixel(2);
    return warpedUV;
}