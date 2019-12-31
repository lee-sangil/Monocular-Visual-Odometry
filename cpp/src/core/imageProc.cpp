

#include "core/imageProc.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>

/**
 * @brief 이미지 크기를 줄여주는 함수
 * @details 이미지 크기를 절반으로 줄여준다. 
 * @param img 입력된 소스 이미지
 * @param img_lower 크기가 줄여진 결과 이미지
 * @author Changhyeon Kim (hyun91015@gmail.com)
 * @date 24-Dec-2019
 */
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

/**
 * @brief 디렉토리 경로에서 이미지 파일 경로를 읽어내는 함수
 * @details 파일 경로안에 있는 이미지 파일들의 이름, 시간을 저장한다. 
 * flag로 데이터셋의 종류 (KITTI, jetson board)에 맞추어 파일을 변환한다.
 * image.txt의 형식이 변화되거나, image의 이름이 000001.jpg 에서 다른 형식으로 저장될 경우 맞춰주어야 이미지 이름을 올바르게 저장할 수 있다.  
 * @param nameFile 이미지 파일이 포함된 디렉토리 이름
 * @param timestampFile timestamp를 저장할 벡터
 * @param file_contents 이미지 파일의 경로를 저장한 벡터
 * @param flag 주어진 이미지 파일의 종류 (Ex: KITTI dataset format, jetson dataset)
 * @author Sangil Lee (sangillee724@gmail.com) Changhyeon Kim (hyun91015@gmail.com)
 * @date 24-Dec-2019
 */
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
//  std::cout<<"read complete. length: "<<file_contents.size()<<std::endl;
}

/**
 * @brief 파일 경로에서 IMU 값을 읽어내는 함수
 * @details 파일 경로안에 있는 IMU 값 및 시간을 저장한다. 
 * @param nameFile 이미지 파일이 포함된 디렉토리 이름
 * @param timestampFile timestamp를 저장할 벡터
 * @param file_contents IMU 값을 저장한 벡터
 * @author Sangil Lee (sangillee724@gmail.com) Changhyeon Kim (hyun91015@gmail.com)
 * @date 24-Dec-2019
 */
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
//  std::cout<<"read complete. length: "<<file_contents.size()<<std::endl;
}


/**
 * @brief 파일 경로에서 Gray image 하나를 읽어내는 함수
 * @details TUM 형식 데이터의 이미지를 저장한다. 
 * @param imgFileName 이미지 파일이 포함된 디렉토리 이름
 * @param outputGreyImg 개별 gray image 
 * @author Changhyeon Kim (hyun91015@gmail.com)
 * @date 24-Dec-2019
 */
void chk::getImgTUMdataset(const std::string& imgFileName, cv::Mat& outputGreyImg){//output rgb : 8uc1

    // read color image
    outputGreyImg = cv::imread(imgFileName, CV_8U);      // read a grayscale image

    if( outputGreyImg.empty() )
        std::cout << "Error: empty image!" << std::endl;
    
}

/**
 * @brief double 값을 string으로 변환하는 함수
 * @details double을 string으로 캐스팅한다. 
 * @param x 캐스팅할 double 변수
 * @return string으로 캐스팅된 결과값
 * @author Changhyeon Kim (hyun91015@gmail.com)
 * @date 24-Dec-2019
 */
std::string chk::dtos(double x) {
    std::stringstream s;  // Allocates memory on stack
    s << std::fixed << x;
    return s.str();       // returns a s.str() as a string by value
    // Frees allocated memory of s
}

/**
 * @brief 센서 구조체
 * @details 센서의 시간, id를 저장한다. 
 * @author Sangil Lee (sangillee724@gmail.com)
 * @date 24-Dec-2019
 */
typedef struct Sensor{
    double time; /**< @brief 시간 */
    int id; /**< @brief id */
    Sensor(double time, int id):time(time),id(id){}; /**< @brief 시간, id를 할당하는 constructor */
}Sensor;


/**
 * @brief Timestamp 및 센서 ID를 정렬하는 함수
 * @details  timestamp 순서로 센서의 ID를 정렬한다.
 * @param time_0 0번 ID를 갖는 센서의 timestamp
 * @param time_1 1번 ID를 갖는 센서의 timestamp
 * @param sensorID 정렬된 센서 ID 벡터
 * @param sensorArray 정렬된 Sensor struct를 갖는 벡터
 * @author Sangil Lee (sangillee724@gmail.com)
 * @date 24-Dec-2019
 */
void lsi::sortTimestamps(const std::vector<double> time_0, const std::vector<double> time_1, std::vector<int>& sensorID){
    std::vector<Sensor> sensorArray;

    for( uint32_t i = 0; i < time_0.size(); i++){
        Sensor data = Sensor(time_0[i],0);
        sensorArray.push_back(data);
    }
    for( uint32_t i = 0; i < time_1.size(); i++){
        Sensor data = Sensor(time_1[i],1);
        sensorArray.push_back(data);
    }
    
    std::sort(sensorArray.begin(), sensorArray.end(), [](const Sensor &a, const Sensor &b){return a.time < b.time;});

    for( uint32_t i = 0; i < sensorArray.size(); i++){
        sensorID.push_back(sensorArray[i].id);
    }
}

/**
 * @brief Timestamp를 정렬하는 함수
 * @details  timestamp 순서로 센서의 ID를 정렬한다.
 * @param time_0 0번 ID를 갖는 센서의 timestamp
 * @param time_1 1번 ID를 갖는 센서의 timestamp
 * @param time_2 2번 ID를 갖는 센서의 timestamp
 * @param sensorID 정렬된 센서 ID 벡터
 * @param sensorArray 정렬된 Sensor struct를 갖는 벡터
 * @author Sangil Lee (sangillee724@gmail.com)
 * @date 24-Dec-2019
 */
void lsi::sortTimestamps(const std::vector<double> time_0, const std::vector<double> time_1, const std::vector<double> time_2, std::vector<int>& sensorID){
    std::vector<Sensor> sensorArray;

    for( uint32_t i = 0; i < time_0.size(); i++){
        Sensor data = Sensor(time_0[i],0);
        sensorArray.push_back(data);
    }
    for( uint32_t i = 0; i < time_1.size(); i++){
        Sensor data = Sensor(time_1[i],1);
        sensorArray.push_back(data);
    }
    for( uint32_t i = 0; i < time_2.size(); i++){
        Sensor data = Sensor(time_2[i],2);
        sensorArray.push_back(data);
    }

    std::sort(sensorArray.begin(), sensorArray.end(), [](const Sensor &a, const Sensor &b){return a.time < b.time;});

    for( uint32_t i = 0; i < sensorArray.size(); i++){
        sensorID.push_back(sensorArray[i].id);
    }
}
