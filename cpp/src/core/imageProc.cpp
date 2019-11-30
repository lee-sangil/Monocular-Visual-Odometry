#include "core/imageProc.hpp"

void chk::downScaleImg(const cv::Mat& img, cv::Mat& img_lower){ // input : double , output : double
	img_lower.create(cv::Size(img.size().width/2, img.size().height / 2), img.type());
	int x0, x1, y0, y1;
	for (int y=0;y<img_lower.rows;++y){
		for(int x=0;x<img_lower.cols;++x){
			x0 = x * 2;
			x1 = x0 + 1;
			y0 = y * 2;
			y1 = y0 + 1;
			img_lower.at<double>(y,x) = (img.at<double>(y0,x0) + img.at<double>(y0,x1) + img.at<double>(y1,x0) + img.at<double>(y1,x1)) / 4.0;
		}
	}
}

void chk::getImageFile(const std::string& nameFile, std::vector<double>& timestampFile, std::vector<std::string>& file_contents, bool flag){
	std::ifstream file(nameFile);
	std::cout << "- Read: " << nameFile << std::endl;
	std::string str;
	
	while(std::getline(file,str)){
		if(str[0] != '#'){
			if( flag ){
				// Dataset recorded with embedded board
				std::string seq = str.substr(0, str.find('\t'));
				std::stringstream ss;
				ss << std::setw(6) << std::setfill('0') << seq << ".jpg";
				file_contents.push_back(ss.str());
				
				str.erase(0,seq.length()+1);
				// str = str.substr(0,str.length()-1); // erase '\r'
				timestampFile.push_back(std::stod(str)/1e9);
			}else{
				// Dataset provided by Hyundai MNsoft
				std::string seq = str.substr(0, str.find(' '));
				timestampFile.push_back(std::stod(seq));

				str.erase(0,seq.length()+1);
				file_contents.push_back(str);
			}
		}
	}
//	std::cout<<"read complete. length: "<<file_contents.size()<<std::endl;
}

void chk::getIMUFile(const std::string& nameFile, std::vector<double>& timestampFile, std::vector<std::array<double,6>>& file_contents){
	std::ifstream file(nameFile);
	std::cout << "- Read: " << nameFile << std::endl;
	std::string str;
	
	while(std::getline(file,str)){
		if(str[0] != '#'){
			std::string dummy = str.substr(0, str.find('\t'));
			str = str.erase(0,dummy.length()+1);

			std::string timeStr = str.substr(0, str.find('\t'));
			double time = std::stod(timeStr);
			time = (time < 0)? 0 : time;
			timestampFile.push_back(time);

			str.erase(0,timeStr.length()+1);
			std::size_t pos = 0;
			double d = 0.0;
			while (pos < str.size() ){
				if( (pos = str.find_first_of('\t', pos)) != std::string::npos ) {
					str[pos] = ' ';
				}
			}

			std::array<double,6> imu = {0,0,0,0,0,0};
			std::stringstream ss(str);
			int it = 0;
			while( ss >> d ){
				imu[it++] = d;
			}
			file_contents.push_back(imu);
		}
	}
//	std::cout<<"read complete. length: "<<file_contents.size()<<std::endl;
}

void chk::getImgTUMdataset(const std::string& imgFileName, cv::Mat& outputGreyImg){//output rgb : 8uc1

	// read color image
	outputGreyImg = cv::imread(imgFileName, CV_8U);      // read a grayscale image

	if( outputGreyImg.empty() )
		std::cout << "Error: empty image!" << std::endl;
	
}

std::string chk::dtos(double x) {
	std::stringstream s;  // Allocates memory on stack
	s << std::fixed << x;
	return s.str();       // returns a s.str() as a string by value
	// Frees allocated memory of s
}

typedef struct sensor{
	double time;
	int id;
	sensor(double time, int id):time(time),id(id){};
}sensor;

void lsi::sortTimestamps(const std::vector<double> time_0, const std::vector<double> time_1, std::vector<int>& sensorID){
	std::vector<sensor> sensorArray;

	for( uint32_t i = 0; i < time_0.size(); i++){
		sensor data = sensor(time_0[i],0);
		sensorArray.push_back(data);
	}
	for( uint32_t i = 0; i < time_1.size(); i++){
		sensor data = sensor(time_1[i],1);
		sensorArray.push_back(data);
	}
	
	std::sort(sensorArray.begin(), sensorArray.end(), [](const sensor &a, const sensor &b){return a.time < b.time;});

	for( uint32_t i = 0; i < sensorArray.size(); i++){
		sensorID.push_back(sensorArray[i].id);
	}
}

void lsi::sortTimestamps(const std::vector<double> time_0, const std::vector<double> time_1, const std::vector<double> time_2, std::vector<int>& sensorID){
	std::vector<sensor> sensorArray;

	for( uint32_t i = 0; i < time_0.size(); i++){
		sensor data = sensor(time_0[i],0);
		sensorArray.push_back(data);
	}
	for( uint32_t i = 0; i < time_1.size(); i++){
		sensor data = sensor(time_1[i],1);
		sensorArray.push_back(data);
	}
	for( uint32_t i = 0; i < time_2.size(); i++){
		sensor data = sensor(time_2[i],2);
		sensorArray.push_back(data);
	}

	std::sort(sensorArray.begin(), sensorArray.end(), [](const sensor &a, const sensor &b){return a.time < b.time;});

	for( uint32_t i = 0; i < sensorArray.size(); i++){
		sensorID.push_back(sensorArray[i].id);
	}
}