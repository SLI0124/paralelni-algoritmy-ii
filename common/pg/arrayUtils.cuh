#pragma once

#include <cstdint>
#include <cudaDefs.h>

template<typename S, typename T>
__global__ void arrayReshape(const S* src,
						  const size_t srcWidth, const size_t srcHeight, const size_t srcPitchInBytes,
						  const size_t dstWidth, const size_t dstHeight, const size_t dstPitchInBytes,
						  T* dst)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const size_t offset = gridDim.x * blockDim.x;
	
	size_t sx, sy, dx, dy;
	const S* srcPtr = nullptr;
	T* dstPtr = nullptr;
	while (idx < (srcWidth * srcHeight))
	{
		sy = idx / srcWidth;
		sx = idx - sy * srcWidth;
		dy = idx / dstWidth;
		dx = idx - dy * dstWidth;

		srcPtr = (S*)((const char*)src + sy * srcPitchInBytes) + sx;
		dstPtr = (T*)((char*)dst + dy * dstPitchInBytes) + dx;

		*dstPtr = static_cast<T>(*srcPtr);
		idx += offset;
	}
}

template<typename T>
__global__ void arrayInit(T* src, const size_t srcLength, const T value)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const size_t offset = gridDim.x * blockDim.x;

	while (idx < (srcLength))
	{
		src[idx] = value;
		idx += offset;
	}
}