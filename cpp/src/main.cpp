#include "core/common.hpp"
#include "core/parser.hpp"
#include "core/imageProc.hpp"
#include "core/MVO.hpp"
#include "core/time.hpp"
#include <opencv2/imgcodecs.hpp>
#define D_METER 5.0
#define D_RADIAN PI/24

Eigen::MatrixXd read_binary(const char* filename, const int rows, const int cols){
    Eigen::MatrixXd matrix(rows, cols);
    std::ifstream in(filename, std::ios::in | std::ios::binary);
    if(!in.is_open())
        return matrix;
    in.read(reinterpret_cast<char*>(matrix.data()), rows*cols*sizeof(double));
    in.close();
    return matrix;
}

int main(int argc, char * argv[]){

	Parser::init(argc, argv);
    if(!Parser::hasOption("-i")){
		std::cout << "Error, invalid arguments.\n\n"
				"\tMandatory -i: Input directory.\n"
				"\tOptional -c: Camera setting .yaml file (default: /path/to/input_directory/camera.yaml).\n"
				"\tOptional -o: Output directory (default path: ./CamTrajectory.txt).\n"
				"\tOptional -fi: initial frame (default: 0).\n"
				"\tOptional -fl: length frame (default: eof).\n"
				"\tOptional -gt: compare with ground-truth if exists (default: false).\n"
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
			std::cerr << "Error: cannot make directory" << std::endl;
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
		std::cerr << "Error: init frame should not be smaller than zero" << std::endl;
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
	std::cout << "- Read successfully." << std::endl;

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
	bool bRun = true, bStep = false;
	int length;

	if( Parser::hasOption("-fl") ){
		length = Parser::getIntOption("-fl");
	}else{
		length = sensorID.size();
	}
	
	std::cout << "# Key descriptions: " << std::endl;
	std::cout << "- s: pause and process a one frame" << std::endl << 
	"- w: play continuously" << std::endl << 
	"- a/d: control zoom in/out of view camera" << std::endl << 
	"- h/l: control roll of view camera" << std::endl << 
	"- j/k: control tilt of view camera" << std::endl << 
	"- e: reset to default parameters of view camera" << std::endl << 
	"- q: quit" << std::endl;

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

				std::cout << "                                                                                                        " << '\r';

				if( Parser::hasOption("-gt") ){
					std::ostringstream dirDepth;
					dirDepth << inputFile << "full_depth/" << std::setfill('0') << std::setw(10) << it_rgb << ".bin";
					Eigen::MatrixXd depth = read_binary(dirDepth.str().c_str(), vo->params.imSize.height, vo->params.imSize.width);
					vo->run(image, depth);
				}else{
					vo->run(image);
				}
				
				std::cout << "Iteration: " << it_rgb << ", Execution time: " << lsi::toc()/1e3 << "ms";
				vo->plot();

				it_rgb++;
				break;
		}

		if( bStep ){
			while(true){
				char key = cv::waitKey(0);
				if( key == 's' ){
					break;
				}else if( key == 'q' ){
					bRun = false;
					break;
				}else if( key == 'w' ){
					bStep = false;
					break;
				}else if( key == 'a' ){
					vo->params.view.height -= D_METER;
					vo->params.view.height = std::max(vo->params.view.height,0.0);
					vo->plot();
					continue;
				}else if( key == 'd' ){
					vo->params.view.height += D_METER;
					vo->plot();
					continue;
				}else if( key == 'h' ){
					vo->params.view.roll += D_RADIAN;
					vo->plot();
					continue;
				}else if( key == 'j' ){
					vo->params.view.pitch -= D_RADIAN;
					vo->plot();
					continue;
				}else if( key == 'k' ){
					vo->params.view.pitch += D_RADIAN;
					vo->plot();
					continue;
				}else if( key == 'l' ){
					vo->params.view.roll -= D_RADIAN;
					vo->plot();
					continue;
				}else if( key == 'e' ){
					vo->params.view.height =  vo->params.view.heightDefault;
					vo->params.view.roll =    vo->params.view.rollDefault;
					vo->params.view.pitch =   vo->params.view.pitchDefault;
					vo->plot();
					continue;
				}
			}
		}

		//KeyBoard Process
		switch (cv::waitKey(1)) {
			case 'q':	// press q to quit
				bRun = false;
				break;
			case 's':
				bStep = true;
				break;
			case 'a':
				vo->params.view.height -= D_METER;
				vo->params.view.height = std::max(vo->params.view.height,0.0);
				break;
			case 'd':
				vo->params.view.height += D_METER;
				break;
			case 'h':
				vo->params.view.roll += D_RADIAN;
				break;
			case 'j':
				vo->params.view.pitch -= D_RADIAN;
				break;
			case 'k':
				vo->params.view.pitch += D_RADIAN;
				break;
			case 'l':
				vo->params.view.roll -= D_RADIAN;
				break;
			case 'e':
				vo->params.view.height =  vo->params.view.heightDefault;
				vo->params.view.roll =    vo->params.view.rollDefault;
				vo->params.view.pitch =   vo->params.view.pitchDefault;
				break;
		}
	}

	if( statusLogger.is_open() ) statusLogger.close();
	std::cout << std::endl;
	return 0;
}
