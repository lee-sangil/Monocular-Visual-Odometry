#include "core/utils.hpp"
#include "core/time.hpp"

void lsi::seed(){std::srand(lsi::toc());}
double lsi::rand(){return std::rand() / (double)RAND_MAX;}
double lsi::randn(){return std::sqrt(-2.0 * std::log(lsi::rand())) * std::cos(2*M_PI*lsi::rand());}

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