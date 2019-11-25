#include "core/common.hpp"
#include "core/parser.hpp"
#include "core/imageProc.hpp"
#include "core/MVO.hpp"
#include "core/time.hpp"
#include "core/numerics.hpp"
#include <opencv2/imgcodecs.hpp>
#include <dirent.h>

Eigen::MatrixXd read_binary(const char* filename, const int rows, const int cols);
std::vector<double> oxtsReader(const char * filePath, const std::string & txtName);
void timeReader(const char * filePath, std::vector<double>& timestamp);
void computeVehicleSpeed( std::vector<std::vector<double> > oxtsData, std::vector<double>& speed);
void computeImuRotation( std::vector<std::vector<double> > oxtsData, std::vector<Eigen::Vector3d>& imuRot);
void directoryReader(const char * filePath, std::vector<std::vector<double> >& oxtsData);
bool grabActiveKey(MVO * vo, char key);

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

	// Read OxTS data
	std::vector<double> timestamp, speed;
	std::vector<Eigen::Vector3d> imuRot;
	std::vector<std::vector<double> > oxtsData;

	if( Parser::hasOption("-vel") || Parser::hasOption("-imu") ){
		std::string oxts_path;
		oxts_path.append(inputFile).append("oxts/data/");
		directoryReader(oxts_path.c_str(), oxtsData);

		if( Parser::hasOption("-vel") )
			computeVehicleSpeed(oxtsData, speed);
		
		if( Parser::hasOption("-imu") )
			computeImuRotation(oxtsData, imuRot);

		std::string time_path;
		time_path.append(inputFile).append("oxts/timestamps.txt");
		timeReader(time_path.c_str(), timestamp);

		timestamp.erase(timestamp.begin(), timestamp.begin()+initFrame);
		speed.erase(speed.begin(), speed.begin()+initFrame);
	}

	/**************************************************************************
	 *  Construct MVO object
	 **************************************************************************/
	MVO * vo = new MVO(fsname);

	/**************************************************************************
	 *  Run MVO object
	 **************************************************************************/
	std::ofstream statusLogger;
	statusLogger.open(outputDir + "CamTrajectory.txt");

	char key;
	std::cout << "# Key descriptions: " << std::endl;
	std::cout << "- s: pause and process a one frame" << std::endl << 
	"- w: play continuously" << std::endl << 
	"- a/d: control zoom in/out of view camera" << std::endl << 
	"- h/l: control roll of view camera" << std::endl << 
	"- j/k: control tilt of view camera" << std::endl << 
	"- e: reset to default parameters of view camera" << std::endl << 
	"- q: quit" << std::endl;

	cv::Mat image;
	std::string dirRgb;

	int length;
	if( Parser::hasOption("-fl") )
		length = Parser::getIntOption("-fl");
	else
		length = sensorID.size();

	bool bRun = true, bStep = false;
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

				if( Parser::hasOption("-vel") || Parser::hasOption("-imu") ){
					vo->update_timestamp(timestamp[it_rgb]);

					if( Parser::hasOption("-vel"))
						vo->update_velocity(speed[it_rgb]);
					if( Parser::hasOption("-imu"))
						vo->update_gyro(imuRot[it_rgb]);
				}
				
				vo->run(image);
				
				if( Parser::hasOption("-gt") ){
					std::ostringstream dirDepth;
					dirDepth << inputFile << "full_depth/" << std::setfill('0') << std::setw(10) << it_rgb+initFrame << ".bin";
					Eigen::MatrixXd depth = read_binary(dirDepth.str().c_str(), vo->params.imSize.height, vo->params.imSize.width);
					vo->calcReconstructionErrorGT(depth);
				}
				
				std::cout << "Iteration: " << it_rgb << ", Execution time: " << lsi::toc()/1e3 << "ms       " << '\r';
				vo->plot();

				it_rgb++;
				break;
		}

		if( bStep ){
			while(true){
				key = cv::waitKey(0);
				if( key == 's' ){
					break;
				}else if( key == 'q' ){
					bRun = false;
					break;
				}else if( key == 'w' ){
					bStep = false;
					break;
				}else if( grabActiveKey(vo, key) ){
					vo->plot();
					continue;
				}
			}
		}

		//KeyBoard Process
		key = cv::waitKey(1);
		switch (key) {
			case 'q':	// press q to quit
				bRun = false;
				break;
			case 's':
				bStep = true;
				break;
			default:
				grabActiveKey(vo, key);
				break;
		}
	}

	if( statusLogger.is_open() ) statusLogger.close();
	std::cout << std::endl;
	return 0;
}

Eigen::MatrixXd read_binary(const char* filename, const int rows, const int cols){
    Eigen::MatrixXd matrix(rows, cols);
    std::ifstream in(filename, std::ios::in | std::ios::binary);
    if(!in.is_open())
        return matrix;
    in.read(reinterpret_cast<char*>(matrix.data()), rows*cols*sizeof(double));
    in.close();
    return matrix;
}

void timeReader(const char * filePath, std::vector<double>& timestamp){
	std::ifstream openFile(filePath);
    if( !openFile.is_open() ){
        std::cout << "text file open error" << std::endl;
    }
    else{
		std::string str;
		int year, month, days, hours, minutes;
		double seconds;
        while(std::getline(openFile,str)){
			if(str[0] != '#'){
				sscanf(str.c_str(), "%04d-%02d-%02d %02d:%02d:%lf", &year, &month, &days, &hours, &minutes, &seconds);
				timestamp.push_back(days*86400+hours*3600+minutes*60+seconds);
			}
		}
        openFile.close();
    }
}

std::vector<double> oxtsReader(const char * filePath, const std::string & txtName){
    std::string filename;
    filename.append(filePath);
    filename.append(txtName);
    std::vector<double> Data;

    std::ifstream openFile(filename.c_str());
    if( !openFile.is_open() ){
        std::cout << "text file open error" << std::endl;
    }
    else{
        std::string line;
        getline(openFile, line);
        std::string token;
        char * sz;
        std::stringstream ss(line);

        while (getline(ss, token, ' ')){
            double temp = std::strtod(token.c_str(), & sz);
            Data.push_back( temp );
        }
        
        openFile.close();
    }
    return Data;
}

void computeVehicleSpeed( std::vector<std::vector<double> > oxtsData, std::vector<double>& speed){
    double temp = 0; 
    for (uint32_t i = 0; i < oxtsData.size(); i++){
        temp = 0;
        if (oxtsData[i].size()<30){
            std::cout << "ERROR: # of data is less than 30, iter: " << i << " # of data: "<< oxtsData[i].size() << std::endl;
        }
        else{
            temp = sqrt( pow(oxtsData[i][6],2)+pow(oxtsData[i][7],2) );
            speed.push_back(temp);
        }
    }
}

void computeImuRotation( std::vector<std::vector<double> > oxtsData, std::vector<Eigen::Vector3d>& imuRot){
    for (uint32_t i = 0; i < oxtsData.size(); i++){
		if(i == 0){
			imuRot.push_back(Eigen::Vector3d::Zero());
			continue;
		}
		
		imuRot.push_back( (Eigen::Vector3d() << oxtsData[i][17], oxtsData[i][18], oxtsData[i][19]).finished() );
	}
}

void directoryReader(const char * filePath, std::vector<std::vector<double> >& oxtsData){
    std::vector<std::string> txtList;
    DIR *pdir = NULL;
    pdir = opendir (filePath);
    struct dirent * pent = NULL;

    if (pdir == NULL){
        std::cout << "\n ERROR! OXTS dir could not be opened" << std::endl;
    }

	// the length of txtList limits to 1000. WHY?
    while ((pent = readdir(pdir))){
        if (pent == NULL){
            std::cout << "\n ERROR " << std::endl;
            break;
        }
        std::string temp_str(pent->d_name);
        txtList.push_back(temp_str);
    }

    sort(txtList.begin(), txtList.end());
    txtList.erase(txtList.begin(),txtList.begin()+2);
    
    for (uint32_t i = 0; i < txtList.size(); i++){
        std::vector<double> temp_data = oxtsReader(filePath, txtList[i]);
        oxtsData.push_back( temp_data );
    }
}

bool grabActiveKey(MVO * vo, char key){
	bool bRtn = true;
	switch (key){
	case 'a':
	case 'A':
		vo->params.view.height /= D_METER;
		vo->params.view.height = std::max(vo->params.view.height,5.0);
		break;
	case 'd':
	case 'D':
		vo->params.view.height *= D_METER;
		break;
	case 'h':
	case 'H':
		vo->params.view.roll += D_RADIAN;
		break;
	case 'j':
	case 'J':
		vo->params.view.pitch -= D_RADIAN;
		break;
	case 'k':
	case 'K':
		vo->params.view.pitch += D_RADIAN;
		break;
	case 'l':
	case 'L':
		vo->params.view.roll -= D_RADIAN;
		break;
	case 'e':
	case 'E':
		vo->params.view.height =  vo->params.view.heightDefault;
		vo->params.view.roll =    vo->params.view.rollDefault;
		vo->params.view.pitch =   vo->params.view.pitchDefault;
		break;
	default:
		return false;
	}
	return bRtn;
}