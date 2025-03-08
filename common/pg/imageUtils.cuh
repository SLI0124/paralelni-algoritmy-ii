#pragma once
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <vector_types.h>

#include <cudaDefs.h>
#include <imageManager.h>
#include <benchmark.h>

#define __FN_NAME__  __FUNCSIG__ 	//use __PRETTY_FUNCTION__ with GCC or CLANG

template<typename T>
struct ImageInfo
{
	uint32_t width = 0;
	uint32_t height = 0;
	size_t pitch = 0;
	T* dPtr = nullptr;						// Device pointer
};

//Template function and its specializations
template<uint8_t SRC_ITEM_BYTES, typename T> __forceinline__ __device__ void convert(const uint8_t* __restrict__ src, T* dst) {}
template<> __forceinline__ __device__ void convert<1, uint32_t>(const uint8_t* __restrict__ src, uint32_t* dst) { *dst = src[0]; }
template<> __forceinline__ __device__ void convert<2, uint32_t>(const uint8_t* __restrict__ src, uint32_t* dst) { *dst = src[1] << 8  | src[0]; }
template<> __forceinline__ __device__ void convert<3, uint32_t>(const uint8_t* __restrict__ src, uint32_t* dst) { *dst = src[2] << 16 | src[1] << 8 | src[0]; }
template<> __forceinline__ __device__ void convert<4, uint32_t>(const uint8_t* __restrict__ src, uint32_t* dst) { *dst = src[3] << 24 |src[2] << 16 | src[1] << 8 | src[0]; }

template<> __forceinline__ __device__ void convert<1, float>(const uint8_t* __restrict__ src, float* dst) { *dst = src[0]; }
template<> __forceinline__ __device__ void convert<2, float>(const uint8_t* __restrict__ src, float* dst) { *dst = src[1] << 8 | src[0]; }
template<> __forceinline__ __device__ void convert<3, float>(const uint8_t* __restrict__ src, float* dst) { *dst = src[2] << 16 | src[1] << 8 | src[0]; }
template<> __forceinline__ __device__ void convert<4, float>(const uint8_t* __restrict__ src, float* dst) { *dst = src[3] << 24 | src[2] << 16 | src[1] << 8 | src[0]; }


template<uint8_t SRC_ITEM_BYTES, typename T>
__global__ void convertBytes(const uint8_t* __restrict__ src, const uint32_t srcWidth, const uint32_t srcHeight, const size_t srcPitchInBytes,
							    const size_t dstPitchInBytes, T* __restrict__ dst)
{
	static_assert(SRC_ITEM_BYTES <= sizeof(T), "SRC_ITEM_BYTES must be LEQ than sizeof(T) to prevent data loss.");

	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int skipX = gridDim.x * blockDim.x;
	unsigned int skipY = gridDim.y * blockDim.y;

	const uint8_t* srcPtr = src + y * srcPitchInBytes + x * SRC_ITEM_BYTES;
	T* dstPtr = (T*)((uint8_t * )dst + y * dstPitchInBytes) + x;

	while (y < srcHeight)
	{
		while (x < srcWidth)
		{
			convert<SRC_ITEM_BYTES>(srcPtr, dstPtr);
			//*dstPtr = y * srcWidth + x;
			srcPtr += skipX * SRC_ITEM_BYTES;
			dstPtr += skipX;
			x += skipX;
		}
		x = blockIdx.x * blockDim.x + threadIdx.x;
		y += skipY;
		srcPtr = src + y * srcPitchInBytes + x * SRC_ITEM_BYTES;
		dstPtr = (T*)((uint8_t*)dst + y * dstPitchInBytes) + x;
	}
}

template<bool NEED_PITCH_MEMORY = true, typename T>
__host__ void prepareData(const char* imageFileName, ImageInfo<T>& img)
{
	FIBITMAP* a = ImageManager::GenericLoader(imageFileName, 0);
	auto aPitch = FreeImage_GetPitch(a);		// FREEIMAGE align row data ... You have to use pitch instead of width
	auto aWidth = FreeImage_GetWidth(a);
	auto aHeight = FreeImage_GetHeight(a);
	auto aBPP = FreeImage_GetBPP(a);

	//Create a memory block using UNIFIED MEMORY to store original image. This is a redundant copy, however the data will be ready to use directly by GPU.
	uint8_t* b = nullptr;
	auto aSizeInBytes = aPitch * aHeight;
	checkCudaErrors(cudaMallocManaged(&b, aSizeInBytes));
	checkCudaErrors(cudaMemcpy(b, FreeImage_GetBits(a), aSizeInBytes, cudaMemcpyHostToDevice));

	FreeImage_Unload(a);
	//checkHostMatrix(b, aPitch, aHeight, aWidth, "%d ", "Reference");

	img.width = aWidth;
	img.height = aHeight;

	if constexpr (NEED_PITCH_MEMORY == true)
	{
		checkCudaErrors(cudaMallocPitch((void**)&img.dPtr, &img.pitch, img.width * sizeof(T), img.height));
	}
	else
	{
		img.pitch = img.width * sizeof(T); 
		checkCudaErrors(cudaMalloc((void**)&img.dPtr, img.pitch * img.height));
	}

	dim3 block{8,8,1 };
	dim3 grid{ getNumberOfParts(img.width, 8), getNumberOfParts(img.height, 8), 1 };

	float gpuTime = 0.0f;

	switch (aBPP)
	{
	case 8:  gpuTime += GPUTIME(1, convertBytes<1, T> << <grid, block >> > (b, aWidth, aHeight, aPitch, img.pitch, img.dPtr)); break;
	case 16: gpuTime += GPUTIME(1, convertBytes<2, T> << <grid, block >> > (b, aWidth, aHeight, aPitch, img.pitch, img.dPtr)); break;
	case 24: gpuTime += GPUTIME(1, convertBytes<3, T> << <grid, block >> > (b, aWidth, aHeight, aPitch, img.pitch, img.dPtr)); break;
	case 32: gpuTime += GPUTIME(1, convertBytes<4, T> << <grid, block >> > (b, aWidth, aHeight, aPitch, img.pitch, img.dPtr)); break;
	}
	printf("\x1B[93m[GPU time] %s: %f ms\033[0m\n", "convertBytes", gpuTime);

	//checkDeviceMatrix(img.dPtr, img.pitch, img.height, img.width, "%d ", "Image Data");
	cudaFree(b);

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
		printf("Error in %s: %s\n", __FN_NAME__, cudaGetErrorName(error));
}


struct TextureInfo
{
	cudaExtent              size;
	cudaArray_t				texArrayData;
	cudaChannelFormatDesc	texChannelDesc;

	cudaTextureDesc			texDesc;
	cudaResourceDesc		resDesc;

	cudaTextureObject_t		texObj;

	TextureInfo()
	{
		memset(this, 0, sizeof(TextureInfo));		// DO NOT DELETE THIS !!!
	}
};


