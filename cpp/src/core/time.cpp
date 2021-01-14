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

/**
 * @brief 현재 time을 저장하는 함수
 * @details usec 단위의 시간을 저장한다.
 * @author Sangil Lee (sangillee724@gmail.com)
 * @date 24-Dec-2019
 */
unsigned long lsi::tic(){

	gettimeofday(&ti, NULL);
	return ti.tv_sec*1e6 + ti.tv_usec;

}

/**
 * @brief tic 시간 이후의 시간 경과를 저장하는 함수
 * @details usec 단위의 tic 시간 이후의 시간 경과를 저장한다.
 * @author Sangil Lee (sangillee724@gmail.com)
 * @date 24-Dec-2019
 */
unsigned long lsi::toc(){

	struct timeval tf;
	gettimeofday(&tf, NULL);
	return (tf.tv_sec-ti.tv_sec)*1e6 + tf.tv_usec-ti.tv_usec;

}
