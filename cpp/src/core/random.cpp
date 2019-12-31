#include "core/random.hpp"
#include "core/time.hpp"
#include <numeric>
#include <algorithm>

/**
 * @brief 랜덤 seed 배정 함수
 * @details 랜덤 seed로서 알고리즘 동작 시간을 대입한다
 * @author Sangil Lee (sangillee724@gmail.com)
 * @date 24-Dec-2019
 */
void lsi::seed(){std::srand(lsi::toc());}

/**
 * @brief Uniform 랜덤 변수를 추출
 * @details 랜덤 변수의 크기를 0부터 1로 제한한다.
 * @return Uniform 랜덤 변수
 * @author Sangil Lee (sangillee724@gmail.com)
 * @date 24-Dec-2019
 */
double lsi::rand(){return std::rand() / (double)RAND_MAX;}

/**
 * @brief Gaussian 랜덤 변수를 추출
 * @details Gaussian N(0, 1^2)의 랜덤 변수를 생성하여 반환한다.
 * @return Gaussian 랜덤 변수
 * @author Sangil Lee (sangillee724@gmail.com)
 * @date 24-Dec-2019
 */
double lsi::randn(){return std::sqrt(-2.0 * std::log(lsi::rand())) * std::cos(2*M_PI*lsi::rand());}

/**
 * @brief Matrix의 랜덤 index를 추출
 * @details weight matrix의 랜덤 index를 추출한다
 * @param weight 랜덤 변수를 추출할 행렬
 * @param mask weight 변수 중, 랜덤 변수를 추출할 영역
 * @param idx_row 랜덤 index 행
 * @param idx_col 랜덤 index 열
 * @author Sangil Lee (sangillee724@gmail.com)
 * @date 24-Dec-2019
 */
void lsi::idx_randselect(Eigen::MatrixXd weight, Eigen::MatrixXd& mask, int& idx_row, int& idx_col){

    // Calculate weight
	Eigen::VectorXd weightVec(Eigen::Map<Eigen::VectorXd>(weight.data(), weight.rows()*weight.cols()));
	Eigen::VectorXd maskVec(Eigen::Map<Eigen::VectorXd>(mask.data(), mask.rows()*mask.cols()));

	// Invalidate unmasked elements
	weightVec = weightVec.cwiseProduct(maskVec);

	// Normalize weight
	weightVec /= weightVec.sum();

    // Select index
	double rand = lsi::rand();
	double cumsum = 0.0;
	for( int i = 0; i < weightVec.size(); i++ ){
		cumsum += weightVec(i);
		if( rand < cumsum ){
			idx_col = std::floor((double) i / weight.rows());
    		idx_row = i % weight.rows();
			return;
		}
	}
}

/**
 * @brief 순열 함수
 * @details ptNum의 수 중에서 minPtNum의 개수 만큼 추출하여 벡터에 저장
 * @param ptNum 순열의 최대 index
 * @param minPtNum 순열에서 추출할 index의 개수
 * @return minPtNum의 개수를 가지며 0부터 ptNum 까지의 임의의 숫자를 담고 있는 벡터
 * @author Sangil Lee (sangillee724@gmail.com) Haram Kim (rlgkfka614@gmail.com)
 * @date 24-Dec-2019
 */
std::vector<uint32_t> lsi::randperm(uint32_t ptNum, int minPtNum){
    std::vector<uint32_t> vector;
    for (uint32_t i = 0; i < ptNum; i++)
        vector.push_back(i);
    std::random_shuffle(vector.begin(), vector.end());
    std::vector<uint32_t> sample(vector.begin(), vector.begin()+minPtNum);
    return sample;
}

/**
 * @brief weight를 반영하여 n개의 indices를 추출하는 함수
 * @details normalized되지 않은 weight h를 normalize 한뒤, 이를 누적하였을 때 random 값을 넘는 index를 비복원 추출
 * @param h weight 벡터
 * @param n 추출할 변수 개수
 * @return weight h를 반영하여 n개를 추출한 변수 indices
 * @author Sangil Lee (sangillee724@gmail.com) Haram Kim (rlgkfka614@gmail.com)
 * @date 24-Dec-2019
 */
std::vector<uint32_t> lsi::randweightedpick(const std::vector<double> &h, int n /*=1*/){
    int u = h.size();
    int s_under;
    double sum, rand_num;
    std::vector<double> H = h;
    std::vector<double> Hs, Hsc;
    std::vector<uint32_t> result;

    n = std::min(std::max(1, n), u);
    std::vector<int> HI(u, 0);          // vector with #u ints.
    std::iota(HI.begin(), HI.end(), 0); // Fill with 0, ..., u-1.
    
    for (int i = 0; i < n; i++){
        // initial variables
        Hs.clear();
        Hsc.clear();
        // random weight
        sum = std::accumulate(H.begin(), H.end(), 0.0);
        std::transform(H.begin(), H.end(), std::back_inserter(Hs),
                       std::bind(std::multiplies<double>(), std::placeholders::_1, 1 / sum)); // divdie elements in H with the value of sum
        std::partial_sum(Hs.begin(), Hs.end(), std::back_inserter(Hsc), std::plus<double>()); // cummulative sum.

        // generate rand num btw 0 to 1
        rand_num = lsi::rand();
        // increase s_under if Hsc is lower than rand_num
        s_under = std::count_if(Hsc.begin(), Hsc.end(), [&](double elem) { return elem < rand_num; });

        result.push_back(HI[s_under]);
        H.erase(H.begin() + s_under);
        HI.erase(HI.begin() + s_under);
    }
    
    return result;
}
