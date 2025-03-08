////////////////////////////////////////////////////////////////////////////////////////////////////
// file: benchmark.h
// version: 1.0
// author: Petr Gajdos
////////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <chrono> 
#include <iomanip> 
#include <iostream> 
#include <string> 
#include <algorithm>
#include <functional>

#ifndef GPU_BENCHMARK
	#define GPU_BENCHMARK 1
#endif

namespace cpubenchmark
{
	//typedef void(*timeableFn)(void);
	using  timeableFn = void(*)(void);

	template <typename F>
	inline double single(F fn)
	{
		unsigned long i, count = 1;
		double timePer = 0;
		for (count = 1; count != 0; count *= 2)
		{
			auto start = std::chrono::high_resolution_clock::now();
			for (i = 0; i < count; i++)
			{
				fn();
			}
			auto end = std::chrono::high_resolution_clock::now();
			double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() * 1.0e-9;
			const double time_granularity = 1.0e-3;
			timePer = elapsed / count;
			if (elapsed > time_granularity) /* That took long enough */
			{
				return timePer;
			}
		}
		/* woa-- if we got here, "count" reached integer wraparound before the timer ran long enough.  Return the last known time */
		return timePer;
	}

	template <typename F>
	inline double multiple(F fn, const unsigned int count)
	{
		double* times = new double[count];

		//Init run, the time will be not taken into account
		single(fn);
		for (unsigned int t = 0; t < count; t++)
		{
			//std::cout << t + 1 << "/" << count << std::endl;
			times[t] = cpubenchmark::single(fn);
		}

		std::sort(&times[0], &times[count]);

		double rv = times[count >> 1];
		delete[] times;
		return rv;
	}

	// Print the time taken for some action, in our standard format.
	inline void print_time(const std::string& what, double seconds)
	{
		//std::cout << what << ":\t" << std::setprecision(3) << seconds * 1.0e9 << "\tns" << std::endl;
		std::cout << what << ":\t" << std::setprecision(3) << seconds * 1.0e3 << "\tms" << std::endl;
	}

	// Time a function's execution, and print the time out in nanoseconds.
	template <typename F>
	inline void print_time(const std::string& fnName, F fn, const unsigned int count)
	{
		print_time(fnName, cpubenchmark::multiple(fn, count));
	}

	#define CPUTIME(x, ...) (x==1) ? cpubenchmark::single([&]() { __VA_ARGS__; }) : cpubenchmark::multiple([&]() { __VA_ARGS__; }, x+1)
}




#if GPU_BENCHMARK

#include <cudaDefs.h>

namespace gpubenchmark
{
	//Single pass benchmark
	template <typename F>
	inline float single(F fn)
	{
		cudaEvent_t startEvent, stopEvent;
		float elapsedTime;
		checkCudaErrors(cudaEventCreate(&startEvent));
		checkCudaErrors(cudaEventCreate(&stopEvent));

		checkCudaErrors(cudaEventRecord(startEvent, 0));
		fn();
		checkCudaErrors(cudaEventRecord(stopEvent, 0));
		checkCudaErrors(cudaEventSynchronize(stopEvent));

		float result;
		checkCudaErrors(cudaEventElapsedTime(&result, startEvent, stopEvent));

		checkCudaErrors(cudaEventDestroy(startEvent));
		checkCudaErrors(cudaEventDestroy(stopEvent));

		return result;
	}

	//Multiple pass benchmark
	template <typename F>
	inline double multiple(F fn, const unsigned int count)
	{
		float* times = new float[count];

		cudaEvent_t startEvent, stopEvent;
		checkCudaErrors(cudaEventCreate(&startEvent));
		checkCudaErrors(cudaEventCreate(&stopEvent));

		//Init run, the time will be not taken into account
		fn();
		for (unsigned int t = 0; t < count; t++)
		{
			checkCudaErrors(cudaEventRecord(startEvent, 0));
			fn();
			checkCudaErrors(cudaEventRecord(stopEvent, 0));
			checkCudaErrors(cudaEventSynchronize(stopEvent));
			checkCudaErrors(cudaEventElapsedTime(&times[t], startEvent, stopEvent));
		}

		std::sort(&times[0], &times[count]);

		checkCudaErrors(cudaEventDestroy(startEvent));
		checkCudaErrors(cudaEventDestroy(stopEvent));

		double rv = times[count >> 1];
		delete[] times;
		return rv;
	}

	// Print the time taken for some action, in our standard format.
	inline void print_time(const std::string& what, double ms)
	{
		std::cout << what << ":\t" << std::setprecision(3) << ms << "\tms" << std::endl;
	}

	// Time a function's execution, and print the time out in milliseconds.
	template <typename F>
	inline void print_time(const std::string& fnName, F fn, const unsigned int count)
	{
		print_time(fnName, gpubenchmark::multiple(fn, count));
	}

	//#define GPUTIME(fn) gpubenchmark::single([&](){ fn; })					// OK - but when called, extra parentheses are needed  GPUTIME(( something ));
	//#define GPUTIME(x, ...) gpubenchmark::single([&]() { __VA_ARGS__; })		// OK - variadic is the solution, moreover, the first parameter can be used to define repetition
	#define GPUTIME(x, ...) (x==1) ? gpubenchmark::single([&]() { __VA_ARGS__; }) : gpubenchmark::multiple([&]() { __VA_ARGS__; }, x+1)

	//printf("\x1B[93m[GPU time] %s: %f ms\033[0m\n", "methodName", gpuTime);
}
#endif
