#pragma once

#include <cstdio>
#include <cuda_runtime.h>
// #include <helper_functions.h>
#include <helper_cuda.h>
#include <vector_types.h>
#include <vector_functions.h>

//Intellisence defs for threadIdx, ...
#include <device_launch_parameters.h>

#define __CHECK_DATA_

#define FLOAT_EPSILON 0.0001

//#define SWAP(a, b) {a ^= b; b ^= a; a ^= b;}
#define SWAP(a, b) (a^=b^=a^=b)
#define MINIMUM(a, b) ((a) < (b) ? (a) : (b))
#define MAXIMUM(a, b) ((a) > (b) ? (a) : (b))
#define CLIP(a, b, c) MINIMUM(MAXIMUM((a), (b)), (c))					//a = your value, b = left bound, c = rightbound
#define INTERPOLATE(first,last,x) ((x-first)/(last-first))
#define ISININTERVAL(first,last,x) ((first<=x)&&(x<=last))
#define NOTININTERVAL(first,last,x) ((x<first)||(last<x))
#define CHECK_ZERO(x) ((x < FLOAT_EPSILON) || (-x > FLOAT_EPSILON))

#define SAFE_DELETE(p) if(p){delete (p);(p)=nullptr;}
#define SAFE_DELETE_ARRAY(p) if(p){delete[] (p);(p)=nullptr;}

#define SAFE_DELETE_CUDA(p) if(p){cudaFree(p);(p)=nullptr;}
#define SAFE_DELETE_CUDAARRAY(p) if(p){cudaFreeArray(p);(p)=nullptr;}
#define SAFE_DELETE_CUDAHOST(p) if(p){cudaFreeHost(p);(p)=nullptr;}

#define GET_K_BIT(n, k) ((n >> k) & 1)

#ifdef __DEVICE_EMULATION__
	#define EMUSYNC __syncthreads()
#else
#define EMUSYNC
#endif

#define WARP_SIZE 32							//multiple 32   - do not change !!!
#define WARP_SIZE_MINUS_ONE 31
#define WARP_SIZE_SHIFT 5						//= log_2(WARP_SIZE)

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Defines an alias representing the kernel setting. </summary>
////////////////////////////////////////////////////////////////////////////////////////////////////
typedef struct __align__(8) KernelSetting
{
public:
	dim3 dimBlock;
	dim3 dimGrid;
	unsigned int blockSize;
	unsigned int sharedMemSize;
	unsigned int noChunks;

	KernelSetting()
	{
		dimBlock = dim3(1,1,1);
		dimGrid = dim3(1,1,1);
		blockSize = 0;
		sharedMemSize = 0;
		noChunks = 1;
	}

	inline void print()
	{
		printf("\n------------------------------ KERNEL SETTING\n");
		printf("Block dimensions: %u %u %u\n", dimBlock.x, dimBlock.y, dimBlock.z);
		printf("Grid dimensions:  %u %u %u\n", dimGrid.x, dimGrid.y, dimGrid.z);
		printf("BlockSize: %u\n", blockSize);
		printf("Shared Memory Size: %u\n", sharedMemSize);
		printf("Number of chunks: %u\n", noChunks);
	}
}KernelSetting;


#pragma region TEMPLATE FUNCTIONS
////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Check host matrix. </summary>
/// <typeparam name="T">	Generic type parameter. </typeparam>
/// <param name="m">	  	The const T * to process. </param>
/// <param name="pitchInBytes">  	The pitch. </param>
/// <param name="rows">   	The rows. </param>
/// <param name="cols">   	The cols. </param>
/// <param name="format"> 	(optional) describes the format to use. </param>
/// <param name="message">	(optional) the message. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////
template< class T> __host__ void checkHostMatrix(const T *m, const unsigned int pitchInBytes, const unsigned int rows, const unsigned int cols, const char* format = "%f ", const char* message = "")
{
#ifdef __CHECK_DATA_
	printf("\nHOST MEMORY: %s [%u %u]\n", message, rows, cols);

	T *ptr = (T*)m;
	for (unsigned int i=0; i<rows; i++)
	{
		for (unsigned int j=0; j<cols; j++)
		{
			printf(format, ptr[j]);
		}
		printf("\n");
		ptr = (T*)(((char*)ptr)+pitchInBytes);
	}
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Check device matrix. </summary>
/// <typeparam name="T">	Generic type parameter. </typeparam>
/// <param name="m">		  	The const T * to process. </param>
/// <param name="pitchInBytes">	  	The pitch. </param>
/// <param name="rows">		  	The rows. </param>
/// <param name="cols">		  	The cols. </param>
/// <param name="format">	  	(optional) describes the format to use. </param>
/// <param name="message">	  	(optional) the message. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////
template< class T> __host__ void checkDeviceMatrix(const T *m, const unsigned int pitchInBytes, const unsigned int rows, const unsigned int cols, const char* format = "%f ", const char* message = "")
{
#ifdef __CHECK_DATA_
	printf("\nDEVICE MEMORY: %s [%u %u]\n", message, rows, cols);
	T *ptr;
	checkCudaErrors(cudaHostAlloc((void**)&ptr, pitchInBytes * rows, cudaHostAllocWriteCombined));
	checkCudaErrors(cudaMemcpy(ptr, m, rows * pitchInBytes, cudaMemcpyDeviceToHost));
	T *p = ptr;
	for (unsigned int i=0; i<rows; i++)
	{
		for (unsigned int j=0; j<cols; j++)
		{
			printf(format, p[j]);
		}
		printf("\n");
		p = (T*)(((char*)p)+pitchInBytes);
	}
	cudaFreeHost(ptr);
#endif
}


template< class T> __host__ void checkDeviceArray(const cudaArray *m, const unsigned int pitchInBytes, const unsigned int rows, const unsigned int cols, const char* format = "%f ", const char* message = "")
{
#ifdef __CHECK_DATA_
	printf("\nDEVICE MEMORY: %s [%u %u]\n", message, rows, cols);
	T *ptr;
	checkCudaErrors(cudaHostAlloc((void**)&ptr, pitchInBytes * rows, cudaHostAllocWriteCombined));
	checkCudaErrors(cudaMemcpyFromArray(ptr, m, 0, 0, rows * pitchInBytes, cudaMemcpyDeviceToHost));
	T *p = ptr;
	for (unsigned int i = 0; i<rows; i++)
	{
		for (unsigned int j = 0; j<cols; j++)
		{
			printf(format, p[j]);
		}
		printf("\n");
		p = (T*)(((char*)p) + pitchInBytes);
	}
	cudaFreeHost(ptr);
#endif
}


////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Verify device matrix. </summary>
/// <typeparam name="T">	Generic type parameter. </typeparam>
/// <typeparam name="T minValue">	Type of the minimum value. </typeparam>
/// <typeparam name="T maxValue">	Type of the maximum value. </typeparam>
/// <param name="m">		  	The const T * to process. </param>
/// <param name="pitchInBytes">	  	The pitch. </param>
/// <param name="rows">		  	The rows. </param>
/// <param name="cols">		  	The cols. </param>
/// <param name="format">	  	(optional) describes the format to use. </param>
/// <param name="message">	  	(optional) the message. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////
template< class T> __host__ void verifyDeviceMatrix(const T *m, const unsigned int pitchInBytes, const unsigned int rows, const unsigned int cols, const T minValue, const T maxValue, const char* message = "")
{
#ifdef __CHECK_DATA_
	printf("\nDEVICE MEMORY: %s [%u %u]\n", message, rows, cols);
	T *ptr;
	checkCudaErrors(cudaHostAlloc((void**)&ptr, pitchInBytes * rows, cudaHostAllocWriteCombined));
	checkCudaErrors(cudaMemcpy(ptr, m, rows * pitchInBytes, cudaMemcpyDeviceToHost));
	T *p = ptr;
	for (unsigned int i=0; i<rows; i++)
	{
		for (unsigned int j=0; j<cols; j++)
		{
			printf("%c", ISININTERVAL(minValue, maxValue, p[j]) ? ' ' : 'x');
		}
		printf("\n");
		p = (T*)(((char*)p)+pitchInBytes);
	}
	cudaFreeHost(ptr);
#endif
}


template< class T> struct check_data
{
	static __host__ void checkDeviceMatrix(const T *m, const unsigned int rows, const unsigned int cols, const bool isRowMatrix = true, const char* format = "%f ", const char* message = ""){}
};

template<> struct check_data<float4>
{
	static __host__ void checkDeviceMatrix(const float4 *m, const unsigned int rows, const unsigned int cols, const bool isRowMatrix = true, const char* format = "%f %f %f %f ", const char* message = "")
	{
#ifdef __CHECK_DATA_
		printf("\n------------------------- DEVICE MEMORY: %s [%u %u] %s\n", message, rows, cols, (isRowMatrix) ? "row matrix" : "column matrix " );
		float4 *tmp;
		checkCudaErrors(cudaMallocHost((void**)&tmp, rows * cols * sizeof(float4)));
		checkCudaErrors(cudaMemcpy(tmp, m, rows * cols * sizeof(float4), cudaMemcpyDeviceToHost));
		for (unsigned int i=0; i<rows * cols; i++)
		{
			if ((isRowMatrix)&&((i%cols)==0))
				printf("\nRow: ");
			if ((!isRowMatrix)&&((i%rows)==0))
				printf("\nCol: ");
			printf(format, tmp[i].x, tmp[i].y, tmp[i].z, tmp[i].w);
		}
		printf("\n");
		cudaFreeHost(tmp);
#endif
	}
};

template<> struct check_data<uchar4>
{
	static __host__ void checkDeviceMatrix(const uchar4 *m, const unsigned int rows, const unsigned int cols, const bool isRowMatrix = true, const char* format = "%hhu %hhu %hhu %hhu ", const char* message = "")
	{
#ifdef __CHECK_DATA_
		printf("\n------------------------- DEVICE MEMORY: %s [%u %u] %s\n", message, rows, cols, (isRowMatrix) ? "row matrix" : "column matrix " );
		uchar4 *tmp;
		checkCudaErrors(cudaMallocHost((void**)&tmp, rows * cols * sizeof(uchar4)));
		checkCudaErrors(cudaMemcpy(tmp, m, rows * cols * sizeof(uchar4), cudaMemcpyDeviceToHost));
		for (unsigned int i=0; i<rows * cols; i++)
		{
			if ((isRowMatrix)&&((i%cols)==0))
				printf("\nRow: ");
			if ((!isRowMatrix)&&((i%rows)==0))
				printf("\nCol: ");
			printf(format, tmp[i].x, tmp[i].y, tmp[i].z, tmp[i].w);
		}
		printf("\n");
		cudaFreeHost(tmp);
#endif
	}
};




////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Check device matrix. </summary>
/// <remarks>	Copies the matrix from device to host and prints its elements. </remarks>
///
/// <param name="m">		[in,out] If non-null, the matrix. </param>
/// <param name="mSize">	The size of matrix. </param>
/// <param name="message">	[in,out] If non-null, the message. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////
template< class T> __host__ void checkDeviceMatrixCUBLAS(const T *m, const unsigned int rows, const unsigned int cols, const bool isRowMatrix = true, const char* format = "%f ", const char* message = "")
{
#ifdef __CHECK_DATA_
	printf("\n------------------------- DEVICE MEMORY: %s [%u %u] %s\n", message, rows, cols, (isRowMatrix) ? "row matrix" : "column matrix " );
	unsigned int tmpSize = rows * cols * sizeof(T);
	T *tmp = (T*)malloc(tmpSize);
	cublasGetVector (rows *cols, sizeof(T), m, 1, tmp, 1);
	for (unsigned int i=0; i<rows * cols; i++)
	{
		if ((isRowMatrix)&&((i%cols)==0))
			printf("\nRow %u: ", i/cols);
		if ((!isRowMatrix)&&((i%rows)==0))
			printf("\nCol %u: ", i/rows);
		printf(format, tmp[i]);
	}
	printf("\n");
	free(tmp);
#endif
}

#pragma endregion

#pragma region INLINE FUNCTIONS

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Check device properties. </summary>
/// <param name="deviceProp">	[in,out] the device property. </param>
/// <returns>	true if it succeeds, false if it fails. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////
__forceinline__  __host__ bool checkDeviceProperties()
{	
	cudaDeviceProp deviceProp;
	bool result = true;
	printf("CUDA Device Query (Runtime API) version (CUDART static linking)\n");

	int deviceCount;
	cudaGetDeviceCount(&deviceCount);

	// This function call returns 0 if there are no CUDA capable devices.
	if (deviceCount == 0)
	{	
		printf("There is no device supporting CUDA\n");
		result =  false;
	}

	int dev;
	for (dev = 0; dev < deviceCount; ++dev) 
	{
		cudaGetDeviceProperties(&deviceProp, dev);

		if (dev == 0) 
		{
			// This function call returns 9999 for both major & minor fields, if no CUDA capable devices are present
			if (deviceProp.major == 9999 && deviceProp.minor == 9999)
			{
				printf("There is no device supporting CUDA.\n");
				result = false;
			}
			else if (deviceCount == 1)
				printf("There is 1 device supporting CUDA\n");
			else
				printf("There are %d devices supporting CUDA\n", deviceCount);
		}

		printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);
		int driverVersion = 0, runtimeVersion = 0;
		cudaDriverGetVersion(&driverVersion);
		printf("  %-50s: %d.%d\n", "CUDA Driver Version", driverVersion/1000, driverVersion%100);
		cudaRuntimeGetVersion(&runtimeVersion);
		printf("  %-50s: %d.%d\n", "CUDA Runtime Version", runtimeVersion/1000, runtimeVersion%100);
		printf("  %-50s: %d\n", "CUDA Capability Major revision number",	deviceProp.major);
		printf("  %-50s: %d\n", "CUDA Capability Minor revision number",	deviceProp.minor);

		printf("\n[GPU details]:\n");
		printf("  %-50s: %.2f GHz\n", "Clock rate", deviceProp.clockRate * 1e-6f);
		printf("  %-50s: %d\n", "Number of multiprocessors",	deviceProp.multiProcessorCount);
		printf("  %-50s: %d\n", "Number of cores", _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount);
		printf("  %-50s: %d\n", "Warp size", deviceProp.warpSize);
		printf("  %-50s: %u Mb\n", "Total amount of global memory", static_cast<unsigned int>(deviceProp.totalGlobalMem >> 20));
		printf("  %-50s: %llu bytes\n", "Total amount of constant memory", static_cast<long long unsigned int>(deviceProp.totalConstMem));
		printf("  %-50s: %llu bytes\n", "Maximum memory pitch", static_cast<long long unsigned int>(deviceProp.memPitch));
		printf("  %-50s: %llu bytes\n", "Texture alignment", static_cast<long long unsigned int>(deviceProp.textureAlignment));
		printf("  %-50s: %s\n", "Run time limit on kernels", deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
		printf("  %-50s: %s\n", "Integrated", deviceProp.integrated ? "Yes" : "No");
		printf("  %-50s: %s\n", "Support host page-locked memory mapping", deviceProp.canMapHostMemory ? "Yes" : "No");
		printf("  %-50s: %s\n", "Compute mode", deviceProp.computeMode == cudaComputeModeDefault ?
			"Default (multiple host threads can use this device simultaneously)" :
			deviceProp.computeMode == cudaComputeModeExclusive ?
			"Exclusive (only one host thread at a time can use this device)" :
			deviceProp.computeMode == cudaComputeModeProhibited ?
			"Prohibited (no host thread can use this device)" :
			"Unknown");

		printf("\n[SM details]:\n");
		printf("  %-50s: %d\n", "Number of cores", _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor));
		printf("  \x1B[91m%-50s: %llu bytes\033[0m\n", "Total amount of shared memory per SM", static_cast<long long unsigned int>(deviceProp.sharedMemPerMultiprocessor));
		printf("  \x1B[91m%-50s: %d\033[0m\n", "Total number of registers available per SM", deviceProp.regsPerMultiprocessor);
		printf("  \x1B[93m%-50s: %d\033[0m\n", "Maximum number of resident blocks per SM", deviceProp.maxBlocksPerMultiProcessor);
		printf("  \x1B[93m%-50s: %d\033[0m\n", "Maximum number of resident threads per SM", deviceProp.maxThreadsPerMultiProcessor);
		printf("  \x1B[91m%-50s: %d\033[0m\n", "Maximum number of resident warps per SM", deviceProp.maxThreadsPerMultiProcessor / deviceProp.warpSize);

		printf("\n[BLOCK details]:\n");
		printf("  %-50s: %llu bytes\n", "Total amount of shared memory per block", static_cast<long long unsigned int>(deviceProp.sharedMemPerBlock));
		printf("  %-50s: %d\n", "Total number of registers available for block", deviceProp.regsPerBlock);
		printf("  %-50s: %d\n", "Maximum number of threads per block", deviceProp.maxThreadsPerBlock);
		printf("  %-50s: %d, %d, %d\n", "Maximum sizes of each dimension of a block", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
		printf("  %-50s: %d, %d, %d\n", "Maximum sizes of each dimension of a grid",  deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
	}
	printf("\nDevice Test PASSED -----------------------------------------------------\n\n");
	return result;
}

__forceinline__  __host__ void checkError()
{
	cudaError_t err= cudaGetLastError();
	if (cudaGetLastError() != cudaSuccess)
	{
		printf("%s\n", cudaGetErrorString(err));
	}
}


__forceinline__  __host__ unsigned int getNumberOfParts(const unsigned int totalSize, const unsigned int partSize)
{
	return (totalSize+partSize-1)/partSize;
}

__forceinline__  __host__ void prepareKernelSettings(const cudaDeviceProp &deviceProp, const unsigned int initNoChunks, const unsigned int threadsPerBlock, const unsigned int dataLength, KernelSetting &ks)
{
	ks.blockSize = threadsPerBlock;
	ks.dimBlock = dim3(threadsPerBlock,1,1);
	ks.sharedMemSize = (unsigned int)deviceProp.sharedMemPerBlock;

	ks.noChunks = initNoChunks;		//Initial Number of Chunks

	unsigned int noBlocks = getNumberOfParts(dataLength, threadsPerBlock * ks.noChunks);
	if (noBlocks > (unsigned int)deviceProp.maxGridSize[0])
	{
		unsigned int multiplicator = noBlocks / deviceProp.maxGridSize[0];
		if ((noBlocks % deviceProp.maxGridSize[0]) != 0)
			multiplicator++;
		ks.noChunks *= multiplicator;
		ks.dimGrid = getNumberOfParts(dataLength, threadsPerBlock * ks.noChunks);
	}
	else
	{
		ks.dimGrid = dim3(noBlocks, 1,1);
	}
	//ks.print();
}

__forceinline__  __host__ void  initializeCUDA(cudaDeviceProp &deviceProp, int requiredDid = -1)
{
	if (!checkDeviceProperties()) return exit(-1);
	if (requiredDid!=-1)
	{
		int did = requiredDid;
		checkCudaErrors(cudaSetDevice(did));
		cudaGetDeviceProperties(&deviceProp, did);
		printf("SELECTED GPU Device %d: \"%s\" with compute capability %d.%d\n\n", did, deviceProp.name, deviceProp.major, deviceProp.minor);
		if (cudaGetLastError() != cudaSuccess)
			exit(-1);
	}
	else
	{
		int did = 0;
		//cudaGetDevice(&did);
		did = gpuGetMaxGflopsDeviceId();
		checkCudaErrors(cudaSetDevice(did));
		cudaGetDeviceProperties(&deviceProp, did);
		printf("SELECTED GPU Device %d: \"%s\" with compute capability %d.%d\n\n", did, deviceProp.name, deviceProp.major, deviceProp.minor);
		if (cudaGetLastError() != cudaSuccess)
			exit(-1);
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Next pow 2. </summary>
/// <param name="x">	The number x. </param>
/// <returns>	Returns the number which is close to given number x and its is pow 2. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////
__forceinline__ __host__ __device__ unsigned int nextPow2(unsigned int x)
{
	--x;
	x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x >> 16;
	return ++x;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Tests if 'x' is pow 2. </summary>
/// <param name="x">The number x. </param>
/// <returns>	TUE if it is pow 2, otherwise FALSE. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////
__forceinline__  __host__ __device__ bool isPow2( const unsigned int x)
{
	return ((x&(x-1))==0);
}

/// <summary>	Next multiple of warp. It is supposed that isPow2(multiple)==true </summary>
/// <param name="x">	   	The const unsigned int to process. </param>
/// <param name="multiple">	(Optional) the multiple. </param>
/// <returns>	An int. </returns>
__forceinline__  __host__ __device__ unsigned int nextMultipleOfWarp(const unsigned int x, const unsigned int multiple = WARP_SIZE)
{
	return (x + multiple - 1) & ~(multiple - 1);
	//return ((x + multiple - 1) / multiple) * multiple;
}

/// <summary>	Next multiple of a number.</summary>
/// <param name="x">	   	The const unsigned int to process. </param>
/// <param name="multiple">	(Optional) the multiple. </param>
/// <returns>	An int. </returns>
__forceinline__  __host__ __device__ unsigned int nextMultipleOf(const unsigned int x, const unsigned int multiple)
{
	if (multiple == 0)
	{
		return 0;
	}
	return ((x - 1) / multiple + 1) * multiple;
}

__forceinline__  __host__ void createTimer(cudaEvent_t *startEvent, cudaEvent_t *stopEvent, float *elapsedTime)
{
	cudaEventCreate(startEvent);
	cudaEventCreate(stopEvent);
	*elapsedTime = 0.0f;
}

__forceinline__  __host__ void startTimer(cudaEvent_t &startEvent)
{
	cudaEventRecord(startEvent, 0);
}

__forceinline__  __host__ void stopTimer(cudaEvent_t &startEvent, cudaEvent_t &stopEvent, float &elapsedTime, const bool appendTime = false)
{
	cudaEventRecord(stopEvent, 0);
	cudaEventSynchronize(stopEvent);
	if (!appendTime)
	{
		cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
		return;
	}

	float t = 0.0f;
	cudaEventElapsedTime(&t, startEvent, stopEvent);
	elapsedTime += t;
}

__forceinline__  __host__ void destroyTimer(cudaEvent_t &startEvent, cudaEvent_t &stopEvent)
{
	cudaEventDestroy(startEvent);
	cudaEventDestroy(stopEvent);
}


__forceinline__  __host__ void CUDA_SAFE_CALL(cudaError_t call, int line) 
{
	switch (call) {
	case cudaSuccess:
		break;
	default:
		printf("ERROR at line :%i.%d' ' %s\n", line, call, cudaGetErrorString(call));
		exit(-1);
		break;
	}
}

__forceinline__  __host__ void printSettings(dim3& grid, dim3 block)
{
	printf("grid[%u, %u, %u]\t block[%u, %u, %u]\n", grid.x, grid.y, grid.z, block.x, block.y, block.z);
}
#pragma endregion
