#include "core/common.hpp"
#include "core/parser.hpp"
#include "core/imageProc.hpp"
#include "core/MVO.hpp"
#include "core/time.hpp"
#include "core/numerics.hpp"
#include <boost/filesystem.hpp>
#include <opencv2/imgcodecs.hpp>
#include <dirent.h>

std::vector<std::vector<double> > fileReader(const std::string&);
Eigen::MatrixXd readDepth(const char*, const int, const int);
std::vector<double> oxtsReader(const char*, const std::string & txtName);
void CANReader(const std::string&, std::vector<double>&, std::vector<double>&, std::vector<Eigen::Vector3d>&);
void timeReader(const char *, std::vector<double>&);
void computeVehicleSpeed( std::vector<std::vector<double> >, std::vector<double>&);
void computeImuRotation( std::vector<std::vector<double> >, std::vector<Eigen::Vector3d>&);
void directoryReader(const char *, std::vector<std::vector<double> >&);
bool grabActiveKey(std::unique_ptr<MVO>&, char);

int main(int argc, char * argv[]){

	Parser::init(argc, argv);
    if(!Parser::hasOption("-i")){
		std::cout << "Error, invalid arguments.\n\n"
				"\tMandatory -i: Input directory.\n"
				"\tOptional -kitti: Reading kitti format.\n"
                "\tOptional -jetson: Reading jetson format.\n"
                "\tOptional -hyundai: Reading hyundai format (default).\n"
				"\tOptional -c: Camera setting .yaml file (default: /path/to/input_directory/camera.yaml).\n"
				"\tOptional -fi: initial frame (default: 0).\n"
				"\tOptional -fl: length frame (default: eof).\n"
				"\tOptional -gt: compare with ground-truth if exists (default: false).\n"
				"\tOptional -db: print log (default: false).\n"
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
	std::vector<double> timestamp_image;
	
	std::string input_path, image_path, imu_path;
	input_path.append(inputFile);
	image_path.append(inputFile).append("image.txt");
	imu_path.append(inputFile).append("imu.txt");

	chk::getImageFile(image_path.c_str(), timestamp_image, rgbNameRaw, Parser::hasOption("-jetson"));
	std::cout << "- Read successfully." << std::endl;

	rgbNameRaw.erase(rgbNameRaw.begin(), rgbNameRaw.begin()+initFrame);
	timestamp_image.erase(timestamp_image.begin(), timestamp_image.begin()+initFrame);

	// Read OxTS data
	std::vector<double> timestamp, timestamp_speed, timestamp_imu, data_speed;
	std::vector<Eigen::Vector3d> data_gyro;

	// chk::getIMUFile(imu_path.c_str(), timestamp_imu, imuDataRaw);

	if( Parser::hasOption("-kitti") ){
		std::string time_path;
		time_path.append(inputFile).append("image_00/timestamps.txt");
		timeReader(time_path.c_str(), timestamp_image);
		timestamp_image.erase(timestamp_image.begin(), timestamp_image.begin()+initFrame);

		if( Parser::hasOption("-v") || Parser::hasOption("-w") ){
			std::vector<std::vector<double> > oxtsData;
			std::string oxts_path;
			oxts_path.append(inputFile).append("oxts/data/");
			directoryReader(oxts_path.c_str(), oxtsData);

			if( Parser::hasOption("-v") ){
				computeVehicleSpeed(oxtsData, data_speed);
				data_speed.erase(data_speed.begin(), data_speed.begin()+initFrame);
			}
			
			if( Parser::hasOption("-w") ){
				computeImuRotation(oxtsData, data_gyro);
				data_gyro.erase(data_gyro.begin(), data_gyro.begin()+initFrame);
			}

			std::string time_path;
			time_path.append(inputFile).append("oxts/timestamps.txt");
			timeReader(time_path.c_str(), timestamp_speed);
			timestamp_speed.erase(timestamp_speed.begin(), timestamp_speed.begin()+initFrame);
			timestamp_imu.assign(timestamp_speed.begin(), timestamp_speed.end());
		}
	}else if( Parser::hasOption("-jetson") ){
		if( Parser::hasOption("-v") || Parser::hasOption("-w") ){
			std::string can_path;
			can_path.append(inputFile).append("/can.txt");
			CANReader(can_path, timestamp, data_speed, data_gyro);
			auto it = std::find_if(timestamp.begin(), timestamp.end(), [&](const double val){return val > timestamp_image[0];});

			if( Parser::hasOption("-v") ){
				timestamp_speed.assign(it,timestamp.end());
				data_speed.erase(data_speed.begin(),data_speed.begin()+data_speed.size()-timestamp_speed.size()-1);
			}
			if( Parser::hasOption("-w") ){
				timestamp_imu.assign(it,timestamp.end());
				data_gyro.erase(data_gyro.begin(),data_gyro.begin()+data_gyro.size()-timestamp_imu.size()-1);
			}
		}
	}

	std::vector<int> sensorID;
	lsi::sortTimestamps(timestamp_speed, timestamp_imu, timestamp_image, sensorID);

	/**************************************************************************
	 *  Construct MVO object
	 **************************************************************************/
	if( Parser::hasOption("-db") )
		MVO::s_file_logger_.open("log.txt");
	if( Parser::hasOption("-gt") )
		MVO::s_point_logger_.open("pointcloud.txt");
	
	std::unique_ptr<MVO> vo(new MVO(fsname));

	if( Parser::hasOption("-jetson") && Parser::hasOption("-w") ){
		vo->params_.Tic = Eigen::Matrix4d::Identity();
		vo->params_.Tci = Eigen::Matrix4d::Identity();
	}

	/**************************************************************************
	 *  Run MVO object
	 **************************************************************************/
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
	if( Parser::hasOption("-fl") ) length = Parser::getIntOption("-fl");
	else length = sensorID.size();

	bool bRun = true, bStep = false;
	int it_imu = 0, it_vel = 0, it_rgb = 0;

	for( int it = 0; it < length && bRun; it++ ){
		switch (sensorID[it]) {
			case 0:
				// Fetch speed
				if( it_rgb > 0 && Parser::hasOption("-v"))
					vo->updateVelocity(timestamp_speed[it_vel], data_speed[it_vel]);
				
				it_vel++;
				break;

			case 1:
				// Fetch imu
				if( it_rgb > 0 && Parser::hasOption("-w"))
					vo->updateGyro(timestamp_imu[it_imu], data_gyro[it_imu]);
				
				it_imu++;
				break;

			case 2:
				// // Fetch velocity synchronously (erase switch statement)
				// if( Parser::hasOption("-v"))
				// 	vo->updateVelocity(timestamp_image[it_rgb], data_speed[it_rgb]);

				// Fetch images
				dirRgb.clear();
				dirRgb.append(inputFile).append(rgbNameRaw[it_rgb]);
				chk::getImgTUMdataset(dirRgb, image);
				
				vo->run(image, timestamp_image[it_rgb]);
				std::cout << "Iteration: " << it_rgb << ", Execution time: " << lsi::toc()/1e3 << "ms       " << '\r' << std::flush;

				vo->updateView();
				if( Parser::hasOption("-gt") ){
					std::ostringstream dirDepth;
					dirDepth << inputFile << "full_depth/" << std::setfill('0') << std::setw(10) << it_rgb+initFrame << ".bin";
					Eigen::MatrixXd depth = readDepth(dirDepth.str().c_str(), vo->params_.im_size.height, vo->params_.im_size.width);
					vo->calcReconstructionErrorGT(depth);
					vo->plot(&depth);
				}else{
					vo->plot();
				}

				it_rgb++;

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

				break;
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
	std::cout << std::endl;
	MVO::s_file_logger_.close();
	
	return 0;
}

Eigen::MatrixXd readDepth(const char* filename, const int rows, const int cols){
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
	timestamp.clear();
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

void computeImuRotation( std::vector<std::vector<double> > oxtsData, std::vector<Eigen::Vector3d>& data_gyro){
    for (uint32_t i = 0; i < oxtsData.size(); i++){
		if(i == 0){
			data_gyro.push_back(Eigen::Vector3d::Zero());
			continue;
		}
		
		data_gyro.push_back( (Eigen::Vector3d() << oxtsData[i][17], oxtsData[i][18], oxtsData[i][19]).finished() );
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

bool grabActiveKey(std::unique_ptr<MVO>& vo, char key){
	bool bRtn = true;
	switch (key){
	case 'a':
	case 'A':
		vo->params_.view.height /= D_METER;
		vo->params_.view.height = std::max(vo->params_.view.height,5.0);
		break;
	case 'd':
	case 'D':
		vo->params_.view.height *= D_METER;
		break;
	case 'h':
	case 'H':
		vo->params_.view.roll += D_RADIAN;
		break;
	case 'j':
	case 'J':
		vo->params_.view.pitch -= D_RADIAN;
		break;
	case 'k':
	case 'K':
		vo->params_.view.pitch += D_RADIAN;
		break;
	case 'l':
	case 'L':
		vo->params_.view.roll -= D_RADIAN;
		break;
	case 'e':
	case 'E':
		vo->params_.view.height =  vo->params_.view.height_default;
		vo->params_.view.roll =    vo->params_.view.roll_default;
		vo->params_.view.pitch =   vo->params_.view.pitch_default;
		break;
	default:
		return false;
	}
	return bRtn;
}

std::vector<std::vector<double> > fileReader(const std::string &filename)
{
    std::vector<std::vector<double> > Data;
    std::ifstream openFile(filename.c_str());
    if (!openFile.is_open())
    {
        std::cout << "text file open error" << std::endl;
    }
    else
    {
        std::string line;
        std::string token;
        char *sz;

        while (getline(openFile, line))
        {
            std::vector<double> data;
            std::stringstream ss(line);

            while (getline(ss, token, '\t'))
            {
                double temp = std::strtod(token.c_str(), &sz);
                data.push_back(temp);
            }
        Data.push_back(data);
        }

        openFile.close();
    }
    return Data;
}

void CANReader(const std::string& filePath, std::vector<double>& timestamp, std::vector<double>& speed, std::vector<Eigen::Vector3d>& angular)
{
    std::vector<std::vector<double> > CAN_data = fileReader(filePath);

    double mSpeed;
	for (unsigned int CAN_iter = 0; CAN_iter < CAN_data.size(); CAN_iter ++){
		timestamp.push_back(CAN_data[CAN_iter][1]/1e9);

		for(std::vector<double>::iterator it = CAN_data[CAN_iter].begin()+2; it != CAN_data[CAN_iter].end(); it++)
			mSpeed += *it;
		mSpeed /= 4 * 3.6; // km/h -> m/s
		speed.push_back(mSpeed);
		angular.push_back( (Eigen::Vector3d() << 0,(CAN_data[CAN_iter].at(4)-CAN_data[CAN_iter].at(5))/3.6/3.0,0).finished() ); // assume car-width is 3 meter while converting km/h -> m/s
	}
}
