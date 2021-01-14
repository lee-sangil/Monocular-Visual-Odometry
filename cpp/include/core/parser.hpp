#ifndef __PARSER_HPP__
#define __PARSER_HPP__

#include <iostream>
#include <stdio.h>
#include <iterator>
#include <cassert>
#include <map>
#include <string>

/**
 * @brief 명령어 해석 모듈.
 * @details 명령창으로부터 입력된 명령어를 해석하는 모듈.
 * @author Sangil Lee (sangillee724@gmail.com)
 * @date 28-Dec-2019
 */
class Parser{
	public:
		/**
		 * @brief 명령어 해석 모듈 초기화.
		 * @param argc 커맨드라인 입력 수
		 * @param argv 커맨드라인 입력 문자열
		 * @author Sangil Lee (sangillee724@gmail.com)
		 * @date 28-Dec-2019
		 */
		static void init(int argc, char* argv[], bool verbose=false) {
			initialised = true;
			std::string key;
			std::string val;
			for (int i = 0; i < argc; ++i) {
				if(argv[i][0] == '-') {
					if(key!="") parameters[key] = val;
					key = argv[i];
					val = "";
				} else {
					val += argv[i];
				}
			}
			if(key!="") parameters[key] = val;
			if(verbose)
				for(auto p : parameters)
					std::cout << "k: " << p.first << "  v: " << p.second << std::endl;
		}

		/**
		 * @brief 명령어 유무 반환.
		 * @param option 명령어
		 * @return 명령어가 있으면, true
		 * @author Sangil Lee (sangillee724@gmail.com)
		 * @date 28-Dec-2019
		 */
		static bool hasOption(const std::string& option){
			assert(initialised);
			return parameters.find(option) != parameters.end();
		}

		/**
		 * @brief 문자열 명령어 반환.
		 * @param option 명령어
		 * @return 명령어가 있으면, 그에 대응되는 값을 문자열로 반환
		 * @author Sangil Lee (sangillee724@gmail.com)
		 * @date 28-Dec-2019
		 */
		static std::string getOption(const std::string& option){
			assert(initialised);
			std::map<std::string,std::string>::iterator it = parameters.find(option);
			if(it != parameters.end()) return it->second;
			return "";
		}

		/**
		 * @brief 문자열 명령어 반환.
		 * @param option 명령어
		 * @return 명령어가 있으면, 그 다음에 입력된 문자열 주소 반환
		 * @author Sangil Lee (sangillee724@gmail.com)
		 * @date 28-Dec-2019
		 */
		static std::string getPathOption(const std::string& option){
			return getOption(option) + '/';
		}

		/**
		 * @brief 문자열 명령어 반환.
		 * @param option 명령어
		 * @param default_value 기본값
		 * @return 명령어가 있으면, 그 다음에 입력된 float형 값 반환
		 * @author Sangil Lee (sangillee724@gmail.com)
		 * @date 28-Dec-2019
		 */
		static float getFloatOption(const std::string& option, float default_value = 0){
			return hasOption(option) ? stof(getOption(option)) : default_value;
		}

		/**
		 * @brief 문자열 명령어 반환.
		 * @param option 명령어
		 * @param default_value 기본값
		 * @return 명령어가 있으면, 그 다음에 입력된 double형 값 반환
		 * @author Sangil Lee (sangillee724@gmail.com)
		 * @date 28-Dec-2019
		 */
		static float getDoubleOption(const std::string& option, double default_value = 0){
			return hasOption(option) ? stod(getOption(option)) : default_value;
		}

		/**
		 * @brief 문자열 명령어 반환.
		 * @param option 명령어
		 * @param default_value 기본값
		 * @return 명령어가 있으면, 그 다음에 입력된 integer형 숫자 반환
		 * @author Sangil Lee (sangillee724@gmail.com)
		 * @date 28-Dec-2019
		 */
		static int getIntOption(const std::string& option, int default_value = 0){
			return hasOption(option) ? stoi(getOption(option)) : default_value;
		}

		/**
		 * @brief 문자열 명령어 반환.
		 * @param option 명령어
		 * @param default_value 기본값
		 * @return 명령어가 있으면, 그 다음에 입력된 문자열 반환
		 * @author Sangil Lee (sangillee724@gmail.com)
		 * @date 28-Dec-2019
		 */
		static std::string getStringOption(const std::string& option, const std::string& default_value = ""){
			return hasOption(option) ? getOption(option) : default_value;
		}

		/**
		 * @brief 문자열 명령어 반환.
		 * @param option 명령어
		 * @param default_value 기본값
		 * @return 명령어가 있으면, 그 다음에 입력된 char형 문자 반환
		 * @author Sangil Lee (sangillee724@gmail.com)
		 * @date 28-Dec-2019
		 */
		static unsigned char getUCharOption(const std::string& option, unsigned char default_value = 0){
			int val = hasOption(option) ? stoi(getOption(option)) : default_value;
			if(val > 255) throw std::invalid_argument("UChar option is out of bounds.");
			return (unsigned char)val;
		}

	private:
		static bool initialised; /**< @brief 명령어 해석 모듈 초기화 */
		static std::map<std::string,std::string> parameters; /**< @brief 명령어 해석 모듈 분류 */
};

bool Parser::initialised = false;
std::map<std::string,std::string> Parser::parameters = std::map<std::string,std::string>();

#endif //__PARSER_H__