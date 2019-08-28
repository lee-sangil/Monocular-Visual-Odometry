#ifndef __IMAGE_PROC_HPP__
#define __IMAGE_PROC_HPP__

#include "core/common.hpp"

namespace chk{
	void downScaleImg(const cv::Mat&, cv::Mat&);
	void getImgPairTUMdataset(const std::string&, const std::string&, cv::Mat&, cv::Mat&);
	void getImgTUMdataset(const std::string&, cv::Mat&);
	void getImageFile(const std::string&, std::vector<double>&, std::vector<std::string>&);
	void getIMUFile(const std::string&, std::vector<double>&, std::vector<std::array<double,6>>&);
	
	std::string dtos(double x);
}

namespace lsi{
	void sortImageAndImu(const std::vector<double>, const std::vector<double>, std::vector<int>&);
}

#endif //__IMAGE_PROC_HPP__