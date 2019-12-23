#include "core/time.hpp"
#include "sys/time.h"
#include "stdio.h"

//auto ti = std::chrono::high_resolution_clock::now();
//
//void lsi::tic(){
//	auto ti = std::chrono::high_resolution_clock::now();
//}
//
//unsigned long lsi::toc(){
//	auto tf = std::chrono::high_resolution_clock::now();
//	auto ticks = std::chrono::duration_cast<std::chrono::microseconds>(tf - ti);
//
//	return (unsigned long)ticks.count();
//}

struct timeval ti;

unsigned long lsi::tic(){

	gettimeofday(&ti, NULL);
	return ti.tv_sec*1e6 + ti.tv_usec;

}

// return elapsed time [sec]
unsigned long lsi::toc(){

	struct timeval tf;
	gettimeofday(&tf, NULL);
	return (tf.tv_sec-ti.tv_sec)*1e6 + tf.tv_usec-ti.tv_usec;

}
