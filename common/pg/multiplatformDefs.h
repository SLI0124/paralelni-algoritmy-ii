#pragma once
#include <cstring>
#include <sys/types.h> // required for stat.h
#include <sys/stat.h> // no clue why required -- man pages say so
#include <algorithm>
#include <iostream>

#ifdef __unix__	// __unix__ is usually defined by compilers targeting Unix systems
    
  	#ifndef OS_WINDOWS
		#define OS_WINDOWS 0

		#define strcpy_s(dst,dstSize,src) strcpy(dst, src)
		#define strcat_s(dst,dstSize,src) strcat(dst,src)
		#define sscanf_s(...) scanf(__VA_ARGS__)
		#define sprintf_s(str, size, format, ...) sprintf(str, format, __VA_ARGS__)
		#define fprintf_s(...) fprintf(__VA_ARGS__)
		#define fopen_s(pFile,filename,mode) ((*(pFile))=fopen((filename),(mode)))==NULL
		#define memcpy_s(dst,dstSize,src,srcSize) memcpy(dst,src,srcSize)
		#define memmove_s(dst,dstSize,src,srcSize) memmove(dst,src,srcSize)
		#define createDir(path, nMode) mkdir(path,nMode)
	#endif

#elif defined(_WIN32) || defined(WIN32)     // _Win32 is usually defined by compilers targeting 32 or   64 bit Windows systems

  	#ifndef OS_WINDOWS
		#define OS_WINDOWS 1

		#define createDir(path, nMode) _mkdir(path)
	#endif

#endif
