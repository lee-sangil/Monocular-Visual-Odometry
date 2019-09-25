#include "core/utils.hpp"
#include "core/time.hpp"

double lsi::rand(){
    // Set seed
    std::srand(lsi::toc());

    return std::rand() / (double)RAND_MAX;
}

void lsi::idx_randselect(Eigen::MatrixXd weight, Eigen::MatrixXd& mask, int& idx_row, int& idx_col){

	// Assign high weight for ground
	for( int i = 0; i < weight.rows(); i++ ){
		weight.block(i,0,1,weight.cols()) *= std::pow(weight.rows(),2);
	}

    // Calculate weight
	Eigen::VectorXd weightVec(Eigen::Map<Eigen::VectorXd>(weight.data(), weight.rows()*weight.cols()));
	Eigen::VectorXd maskVec(Eigen::Map<Eigen::VectorXd>(mask.data(), mask.rows()*mask.cols()));

    // weightVec = 0.05 / (weightVec + 0.05) * mask;
	weightVec.array() += 0.05;
	weightVec = weightVec.cwiseInverse();
	weightVec *= 0.05;

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
			idx_row = std::floor((double) i / weight.cols());
    		idx_col = i % weight.cols();
			return;
		}
	}
}