#ifndef __PARSER_HPP__
#define __PARSER_HPP__

#include <iostream>
#include <stdio.h>
#include <iterator>
#include <cassert>
#include <map>
#include <string>

class Parser{
	public:
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

		static bool hasOption(const std::string& option){
			assert(initialised);
			return parameters.find(option) != parameters.end();
		}

		static std::string getOption(const std::string& option){
			assert(initialised);
			std::map<std::string,std::string>::iterator it = parameters.find(option);
			if(it != parameters.end()) return it->second;
			return "";
		}

		static std::string getPathOption(const std::string& option){
			return getOption(option) + '/';
		}

		static float getFloatOption(const std::string& option, float default_value = 0){
			return hasOption(option) ? stof(getOption(option)) : default_value;
		}

		static float getDoubleOption(const std::string& option, double default_value = 0){
			return hasOption(option) ? stod(getOption(option)) : default_value;
		}

		static int getIntOption(const std::string& option, int default_value = 0){
			return hasOption(option) ? stoi(getOption(option)) : default_value;
		}

		static std::string getStringOption(const std::string& option, const std::string& default_value = ""){
			return hasOption(option) ? getOption(option) : default_value;
		}

		static uchar getUCharOption(const std::string& option, uchar default_value = 0){
			int val = hasOption(option) ? stoi(getOption(option)) : default_value;
			if(val > 255) throw std::invalid_argument("UChar option is out of bounds.");
			return (uchar)val;
		}

	private:
		static bool initialised;
		static std::map<std::string,std::string> parameters;
};

bool Parser::initialised = false;
std::map<std::string,std::string> Parser::parameters = std::map<std::string,std::string>();

#endif //__PARSER_H__