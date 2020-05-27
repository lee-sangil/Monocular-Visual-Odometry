#ifndef __RANDOM_HPP__
#define __RANDOM_HPP__

#include <vector>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

/**
 * @author Sangil Lee (sangillee724@gmail.com)
 * @date 27-Dec-2019
 */
namespace lsi{
	void seed(); /**< @brief 랜덤 seed 배정 함수 */
	double rand(); /**< @brief Uniform 랜덤 변수를 추출 */
	double randn(); /**< @brief Gaussian 랜덤 변수를 추출 */
	void idx_randselect(Eigen::MatrixXd weight, bool * mask, int& i, int& j); /**< @brief Matrix의 랜덤 index를 추출 */
	std::vector<uint32_t> randperm(uint32_t ptNum, int minPtNum); /**< @brief 순열 함수 */
	std::vector<uint32_t> randweightedpick(const std::vector<double>& h, int n = 1); /**< @brief weight를 반영하여 n개의 indices를 추출하는 함수 */
}

#endif //__RANDOM_HPP__