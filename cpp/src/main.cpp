#include "core/common.hpp"
#include "core/parser.hpp"
#include "core/imageProc.hpp"
#include "core/MVO.hpp"
#include "visualization/plot.hpp"

int main(int argc, char * argv[]){
	
	Parser::init(argc, argv);
    if(!Parser::hasOption("-i")){
		std::cout << "Error, invalid arguments.\n\n"
				"\tMandatory -i: Input directory.\n"
				"\tOptional -c: Camera setting .yaml file (default: /path/to/input_directory/camera.yaml).\n"
				"\tOptional -o: Output directory (default path: ./CamTrajectory.txt).\n\n"
				"Example: [./divo_dataset -c /path/to/setting.yaml -i /path/to/input_folder/ -o /path/to/output_folder/]" << std::endl;
        return 1;
    }

	std::string inputFile = Parser::getOption("-i");
	std::string outputDir = Parser::getOption("-o");

    if(!boost::filesystem::exists(inputFile)){
		std::cerr << "Invalid parameters." << std::endl;
        return 1;
    }
	if(!Parser::hasOption("-o") || outputDir == "") outputDir = "./";
	if(!boost::filesystem::exists(outputDir)){
		if( std::system((std::string("mkdir -p ") + outputDir).c_str()) == -1 ){
			std::cerr << "Cannot make directory" << std::endl;
			return 1;
		}
	}

	/**************************************************************************
	 *  Read .yaml file
	 **************************************************************************/
	std::string fsname;
	if(Parser::hasOption("-c")){
		fsname = Parser::getOption("-c");
	}else{
		fsname = inputFile + "/camera.yaml";
	}

	cv::FileStorage fSettings(fsname, cv::FileStorage::READ);
	if (!fSettings.isOpened()){
		std::cerr << "Failed to open: " << fsname << std::endl;
		return 1;
	}

	double fx =			fSettings["Camera.fx"];
	double fy =			fSettings["Camera.fy"];
	double cx =			fSettings["Camera.cx"];
	double cy =			fSettings["Camera.cy"];
	double k1 =			fSettings["Camera.k1"];
	double k2 =			fSettings["Camera.k2"];
	double p1 =			fSettings["Camera.p1"];
	double p2 =			fSettings["Camera.p2"];
	double k3 =			fSettings["Camera.k3"];
	double width =		fSettings["Camera.width"];
	double height =		fSettings["Camera.height"];
	
	/**************************************************************************
	 *  Read dataset
	 **************************************************************************/
	std::cout << "# inputFile: " << inputFile << std::endl;
	
	// file name reading
	std::vector<std::string> rgbNameRaw;
	std::vector<std::array<double,6>> imuDataRaw;
	std::vector<double> timeRgb, timeImu;
	
	std::string input_path, image_path, imu_path;
	input_path.append(inputFile);
	image_path.append(inputFile).append("image.txt");
	imu_path.append(inputFile).append("imu.txt");

	chk::getImageFile(image_path.c_str(), timeRgb, rgbNameRaw);
	chk::getIMUFile(imu_path.c_str(), timeImu, imuDataRaw);
	std::cout << "read done." << std::endl;

	std::vector<int> sensorID;
	lsi::sortImageAndImu(timeImu, timeRgb, sensorID);

	/**************************************************************************
	 *  Construct MVO object
	 **************************************************************************/
	MVO::Parameter params;

	if( fx!=0 )			params.fx = fx;
	if( fy!=0 )			params.fy = fy;
	if( cx!=0 )			params.cx = cx;
	if( cy!=0 )			params.cy = cy;
	if( k1!=0 )			params.k1 = k1;
	if( k2!=0 )			params.k2 = k2;
	if( p1!=0 )			params.p1 = p1;
	if( p2!=0 )			params.p2 = p2;
	if( k3!=0 )			params.k3 = k3;
	if( width!=0 )		params.width = width;
	if( height!=0 )		params.height = height;

	MVO * vo = new MVO(params);

	/**************************************************************************
	 *  Run MVO object
	 **************************************************************************/
	std::ofstream statusLogger;
	statusLogger.open(outputDir + "CamTrajectory.txt");
	std::string dirRgb;
	cv::Mat image;
	bool bRun = true;
	int length = sensorID.size();
	
	int it_imu = 0, it_rgb = 0;
	for( int it = 0; it < length && bRun; it++ ){

		switch (sensorID[it]) {
			case 1:
				// Fetch imu
				// imuGrabber.GrabImu(imuDataRaw[it_imu], timeImu[it_imu]);
				
				it_imu++;
				break;
			case 2:
				// Fetch images
				dirRgb.clear();
				dirRgb.append(inputFile).append(rgbNameRaw[it_rgb]);
				chk::getImgTUMdataset(dirRgb, image);

				vo->run(image);
				vo->plot();

				it_rgb++;
				break;
		}

		//KeyBoard Process
		switch (cv::waitKey(1)) {
			case 'q':	// press q to quit
			case 'Q':
				bRun = false;
				break;
			default:
				break;
		}
	}

	if( statusLogger.is_open() ) statusLogger.close();
	image.release();
	cv::waitKey(0);

	return 0;
}