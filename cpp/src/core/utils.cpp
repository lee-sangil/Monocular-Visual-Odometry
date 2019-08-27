#include "core/utils.hpp"
#include "core/time.hpp"

double lsi::rand(){
    // Set seed
    std::srand(lsi::toc());

    return std::rand() / (double)RAND_MAX;
}

void lsi::idx_randselect(Eigen::MatrixXd weight, uint32_t& idx_row, uint32_t& idx_col){
    // Calculate weight
	Eigen::MatrixXd weightVec = weight;
	weightVec.resize(weight.rows()*weight.cols(),1);

    weightVec = 0.05 / (weightVec + 0.05);
	
    // Select index
	double rand = lsi::rand();
	double cumsum = 0.0;
	for( int i = 0; i < weightVec.size(); i++ ){
		cumsum += weightVec(i);
		if( rand < cumsum ){
			idx_row = std::floor(select / weight.cols());
    		idx_col = std::modf(select, weight.cols());
		}
	}
}