#include "core/common.hpp"
#include "core/parser.hpp"
#include "core/imageProc.hpp"
#include "core/MVO.hpp"
#include "core/time.hpp"

int main(int argc, char * argv[]){

	Parser::init(argc, argv);
    if(!Parser::hasOption("-i")){
		std::cout << "Error, invalid arguments.\n\n"
				"\tMandatory -i: Input directory.\n"
				"\tOptional -c: Camera setting .yaml file (default: /path/to/input_directory/camera.yaml).\n"
				"\tOptional -o: Output directory (default path: ./CamTrajectory.txt).\n"
				"\tOptional -fi: initial frame (default: 0).\n"
				"\tOptional -fl: length frame (default: eof).\n"
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
	
	/**************************************************************************
	 *  Read dataset
	 **************************************************************************/
	std::cout << "# inputFile: " << inputFile << std::endl;
	int initFrame = 0;
	if( Parser::hasOption("-fi") ){
		initFrame = Parser::getIntOption("-fi");
	}
	if( initFrame < 0 ){
		std::cerr << "Init frame should not be smaller than zero" << std::endl;
		return 1;
	}

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
	std::cout << "Read successfully." << std::endl;

	rgbNameRaw.erase(rgbNameRaw.begin(), rgbNameRaw.begin()+initFrame);
	timeRgb.erase(timeRgb.begin(), timeRgb.begin()+initFrame);

	std::vector<int> sensorID;
	lsi::sortImageAndImu(timeImu, timeRgb, sensorID);

	/**************************************************************************
	 *  Construct MVO object
	 **************************************************************************/
	MVO * vo = new MVO(fsname);

	/**************************************************************************
	 *  Run MVO object
	 **************************************************************************/
	std::ofstream statusLogger;
	statusLogger.open(outputDir + "CamTrajectory.txt");

	std::string dirRgb;
	cv::Mat image;
	bool bRun = true;
	int length;

	if( Parser::hasOption("-fl") ){
		length = Parser::getIntOption("-fl");
	}else{
		length = sensorID.size();
	}
	
	std::cout << "Press S to stop/restart or Q to quit." << std::endl;

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

				lsi::tic();
				std::cout << "                                                                                                        " << '\r';
				vo->run(image);
				std::cout << "Iteration: " << it_rgb << ", Execution time: " << lsi::toc()/1e3 << "ms";
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
			case 's':
			case 'S':
				while( true ){
					char key = cv::waitKey(0);
					if( key == 's' || key == 'S' ){
						break;
					}else if( key == 'q' || key == 'Q' ){
						bRun = false;
						break;
					}
				}
				break;
		}
	}

	if( statusLogger.is_open() ) statusLogger.close();

	std::cout << std::endl << "Press Q again to quit." << std::endl;
	while( true ){
		switch (cv::waitKey(0)) {
			case 'q':	// press q to quit
			case 'Q':
				return 0;
		}
	}
}
