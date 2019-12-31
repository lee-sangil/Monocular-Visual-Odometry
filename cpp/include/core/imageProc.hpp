#ifndef __IMAGE_PROC_HPP__
#define __IMAGE_PROC_HPP__

#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

/**
 * @author Changhyeon Kim (hyun91015@gmail.com)
 * @date 24-Dec-2019
 */

namespace chk{
	void downScaleImg(const cv::Mat&, cv::Mat&); /**< @brief 이미지 크기를 줄여주는 함수 */
	void getImgTUMdataset(const std::string&, cv::Mat&); /**< @brief 파일 경로에서 Gray image 하나를 읽어내는 함수 */
	void getImageFile(const std::string&, std::vector<double>&, std::vector<std::string>&, bool); /**< @brief 디렉토리 경로에서 이미지 파일 경로를 읽어내는 함수 */
	void getIMUFile(const std::string&, std::vector<double>&, std::vector<std::array<double,6>>&); /**< @brief 파일 경로에서 IMU 값을 읽어내는 함수 */
	
	std::string dtos(double x); /** @brief double 값을 string으로 변환하는 함수 */
}

/**
 * @author Sangil Lee (sangillee724@gmail.com)
 * @date 24-Dec-2019
 */

namespace lsi{
	void sortTimestamps(const std::vector<double>, const std::vector<double>, std::vector<int>&); /** @brief Timestamp 및 센서 ID를 정렬하는 함수 */
	void sortTimestamps(const std::vector<double>, const std::vector<double>, const std::vector<double>, std::vector<int>&); /** @brief Timestamp 및 센서 ID를 정렬하는 함수 */
}

#endif //__IMAGE_PROC_HPP__