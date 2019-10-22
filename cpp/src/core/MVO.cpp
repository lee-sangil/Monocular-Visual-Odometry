#include "core/MVO.hpp"
#include "core/utils.hpp"
#include "core/numerics.hpp"
#include "core/time.hpp"

MVO::MVO(){
    this->step = -1;
    this->key_step = 0;
    this->next_key_step = 0;
    this->scale_initialized = false;
    this->groundtruth_provided = false;
    this->speed_provided = false;
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
		std::cerr << "Failed to open: " << yaml << std::endl;
		return;
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

    // this->descriptor = cv::BRISK::create();

    this->params.thInlier =         fSettings["Feature.thInlier"];
    this->params.thRatioKeyFrame =  fSettings["Feature.thRatioKeyFrame"];
    this->params.min_px_dist =      fSettings["Feature.min_px_dist"];
    this->params.px_wide =          fSettings["Feature.px_wide"];

    // RANSAC parameter
    this->params.ransacCoef_scale.iterMax =        fSettings["RANSAC.iterMax"];
    this->params.ransacCoef_scale.minPtNum =       fSettings["RANSAC.nSample"];
    this->params.ransacCoef_scale.thInlrRatio =    fSettings["RANSAC.thInlrRatio"];
    this->params.ransacCoef_scale.thDist =         fSettings["RANSAC.thDist"]; // standard deviation
    this->params.ransacCoef_scale.thDistOut =      fSettings["RANSAC.thDistOut"]; // three times of standard deviation
    this->params.ransacCoef_scale.calculate_func = MVO::calculate_scale;
    this->params.ransacCoef_scale.calculate_dist = MVO::calculate_scale_error;
    
    this->params.ransacCoef_plane.iterMax =        fSettings["RANSAC.iterMax"];
    this->params.ransacCoef_plane.minPtNum =       3;
    this->params.ransacCoef_plane.thInlrRatio =    fSettings["RANSAC.thInlrRatio"];
    this->params.ransacCoef_plane.thDist =         fSettings["RANSAC.thDist"]; // standard deviation
    this->params.ransacCoef_plane.thDistOut =      fSettings["RANSAC.thDistOut"]; // three times of standard deviation
    this->params.ransacCoef_plane.calculate_func = MVO::calculate_plane;
    this->params.ransacCoef_plane.calculate_dist = MVO::calculate_plane_error;

    // Bucket
    this->bucket = Bucket();
    this->bucket.safety =           fSettings["Bucket.safety"];
	this->bucket.max_features =     fSettings["Feature.num"];
    int bucketGridRows =            fSettings["Bucket.rows"];
    int bucketGridCols =            fSettings["Bucket.cols"];

	this->bucket.grid = cv::Size(bucketGridRows,bucketGridCols);
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

    cv::namedWindow("MVO");
    cv::moveWindow("MVO", 20, 20);

    cv::namedWindow("Trajectory");
    cv::moveWindow("Trajectory", 1280, 20);

    this->params.view.heightDefault =   fSettings["viewCam.height"]; // in world coordinate
    this->params.view.rollDefault =     (double) fSettings["viewCam.roll"] * PI/180; // radian
    this->params.view.pitchDefault =    (double) fSettings["viewCam.pitch"] * PI/180; // radian
    this->params.view.height =          this->params.view.heightDefault;
    this->params.view.roll =            this->params.view.rollDefault;
    this->params.view.pitch =           this->params.view.pitchDefault;
    this->params.view.imSize = cv::Size(600,600);
    this->params.view.K <<  300,   0, 300,
						      0, 300, 300,
						      0,   0,   1;
    this->params.view.upperLeft =   cv::Point3d(-3,-2, 3);
    this->params.view.upperRight =  cv::Point3d( 3,-2, 3);
    this->params.view.lowerLeft =   cv::Point3d(-3, 2, 3);
    this->params.view.lowerRight =  cv::Point3d( 3, 2, 3);
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

void MVO::backup(){
    this->features_backup.clear();
    this->features_backup.assign(this->features.begin(), this->features.end());
}

void MVO::reload(){
    this->features.clear();
    this->features.assign(this->features_backup.begin(), this->features_backup.end());
    this->nFeature = this->features.size();

    this->TRec.push_back(Eigen::Matrix4d::Identity());
    this->TocRec.push_back(this->TocRec.back());
    this->PocRec.push_back(this->PocRec.back());
}

void MVO::set_image(cv::Mat& image){
    this->prev_image = this->cur_image.clone();
    cv::remap(image, this->undist_image, this->distMap1, this->distMap2, cv::INTER_AREA);
    
    if( this->params.applyCLAHE )
        cvClahe->apply(this->undist_image, this->cur_image);
    else
        this->cur_image = this->undist_image.clone();

    this->step++;
    this->key_step = this->next_key_step;
}

void MVO::run(cv::Mat& image){
    
    lsi::tic();
    this->set_image(image);
    std::cerr << "============ Iteration: " << this->step << " ============" << std::endl;
    std::cerr << "# Grab image: " << lsi::toc() << std::endl;
    this->refresh();

    std::vector<bool> success;
    success.reserve(3);
    success.push_back(this->extract_features());       // Extract and update features
    success.push_back(this->calculate_essential());    // RANSAC for calculating essential/fundamental matrix
    success.push_back(this->calculate_motion());       // Extract rotational and translational from fundamental matrix

    if( !std::all_of(success.begin(), success.end(), [](bool b){return b;}) )
        this->scale_initialized = false;
}

ptsROI_t MVO::get_points()
{
    ptsROI_t ptsROI;
    for( uint32_t i = 0; i < this->features.size(); i++ ){
        ptsROI.push_back( std::make_tuple(this->features[i].uv.back(), this->features[i].point.block(0, 0, 3, 1) ) );
    }
    return ptsROI;
}
