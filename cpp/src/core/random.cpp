#include "core/random.hpp"
#include "core/time.hpp"
#include <numeric>
#include <algorithm>

void lsi::seed(){std::srand(lsi::toc());}
double lsi::rand(){return std::rand() / (double)RAND_MAX;}
double lsi::randn(){return std::sqrt(-2.0 * std::log(lsi::rand())) * std::cos(2*M_PI*lsi::rand());}

// select random bucket index
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

// generate random permutation
std::vector<uint32_t> lsi::randperm(uint32_t ptNum, int minPtNum){
    std::vector<uint32_t> vector;
    for (uint32_t i = 0; i < ptNum; i++)
        vector.push_back(i);
    std::random_shuffle(vector.begin(), vector.end());
    std::vector<uint32_t> sample(vector.begin(), vector.begin()+minPtNum);
    return sample;
}

// pick random index with weighted probability
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
