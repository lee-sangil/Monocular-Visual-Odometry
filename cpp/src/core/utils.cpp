#include "core/utils.hpp"
#include "core/time.hpp"

double lsi::rand(){
    // Set seed
    std::srand(lsi::toc());

    return std::rand() / (double)RAND_MAX;
}

void lsi::idx_randselect(Eigen::MatrixXd weight, uint32_t& idx_row, uint32_t& idx_col){
    // Calculate weight
	Eigen::VectorXd weightVec(Eigen::Map<Eigen::VectorXd>(weight.data(), weight.rows()*weight.cols()));

    // weightVec = 0.05 / (weightVec + 0.05);
	weightVec.array() += 0.05;
	weightVec = weightVec.cwiseInverse();
	weightVec *= 0.05;
	weightVec /= weightVec.sum();

    // Select index
	double rand = lsi::rand();
	double cumsum = 0.0;
	for( int i = 0; i < weightVec.size(); i++ ){
		cumsum += weightVec(i);
		if( rand < cumsum ){
			idx_row = std::floor((double) i / weight.cols());
    		idx_col = std::remainder((double)i, (double)weight.cols());
		}
	}
}