#include <cerrno>
#include <cstdio>
#include <cstdarg>
#include <iostream>
#include "../include/check.h"

void Check::check(bool condition, char *format,...){
	if(!condition){
		char buffer[256];
		va_list args;
		va_start (args, format);
		vsprintf (buffer,format, args);
		va_end (args);

		std::cerr << buffer << std::endl;
		exit(-1);
	}
}
