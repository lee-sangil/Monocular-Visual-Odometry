#include "core/MVO.hpp"
#include "core/utils.hpp"
#include "core/numerics.hpp"

MVO::MVO(){
    this->step = 0;
    this->scale_initialized = false;
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
    this->PocRec.push_back(Eigen::Vector4d::Zero());
    this->PocRec[0](3) = 1;
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

	this->params.radialDistortion.push_back(params.k1);
	this->params.radialDistortion.push_back(params.k2);
	this->params.radialDistortion.push_back(params.k3);
	this->params.tangentialDistortion.push_back(params.p1);
	this->params.tangentialDistortion.push_back(params.p2);

    this->params.thInlier =         fSettings["Feature.thInlier"];
    this->params.min_px_dist =      fSettings["Feature.min_px_dist"];

    // RANSAC parameter
    this->params.ransacCoef_scale_prop.iterMax =        fSettings["RANSAC.iterMax"];
    this->params.ransacCoef_scale_prop.minPtNum =       fSettings["RANSAC.nSample"];
    this->params.ransacCoef_scale_prop.thInlrRatio =    fSettings["RANSAC.thInlrRatio"];
    this->params.ransacCoef_scale_prop.thDist =         fSettings["RANSAC.thDist"]; // standard deviation
    this->params.ransacCoef_scale_prop.thDistOut =      fSettings["RANSAC.thDistOut"]; // three times of standard deviation
    
    // 3D reconstruction
    this->params.vehicle_height =   fSettings["Scale.height"]; // in meter
    this->params.reprojError =      fSettings["Scale.error"];
    this->params.initScale = 1;

    // Bucket
    this->bucket = Bucket();
    this->bucket.safety =           fSettings["Bucket.safety"];
	this->bucket.max_features =     fSettings["Feature.num"];
    int bucketGridRows =            fSettings["Bucket.rows"];
    int bucketGridCols =            fSettings["Bucket.cols"];

	this->bucket.grid = cv::Size(bucketGridRows,bucketGridCols);
	this->bucket.size = cv::Size(this->params.imSize.width/this->bucket.grid.width, this->params.imSize.height/this->bucket.grid.height);
	this->bucket.mass.setZero(this->bucket.grid.width,this->bucket.grid.height);
	this->bucket.prob.setZero(this->bucket.grid.width,this->bucket.grid.height);
    
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

    this->TRec[this->step] = Eigen::Matrix4d::Identity();
    this->TocRec[this->step] = this->TocRec[this->step-1];
    this->PocRec[this->step] = this->PocRec[this->step-1];
}

void MVO::set_image(const cv::Mat image){
	this->prev_image = this->cur_image.clone();
    if( this->params.applyCLAHE )
        cvClahe->apply(image, this->cur_image);
    else
        this->cur_image = image.clone();    
}

void MVO::run(const cv::Mat image){
    this->step++;

    this->set_image(image);
    this->refresh();

    if( this->extract_features() &      // Extract and update features
	    this->calculate_essential() &   // RANSAC for calculating essential/fundamental matrix
		this->calculate_motion() )      // Extract rotational and translational from fundamental matrix
        
        this->backup();
        
    else
        this->reload();
}