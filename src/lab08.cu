#include <cudaDefs.h>
#include <limits>
#include <benchmark.h>

namespace lab08 {
#define __PRINT__  cout <<  __PRETTY_FUNCTION__ <<  endl

	constexpr unsigned int TPB = 512;
	constexpr unsigned int NO_BLOCKS = 46;
	constexpr unsigned int N = 1 << 20;

	constexpr int numberOfPasses = 1;


	int* a, * b;
	int* da, * db, * dGlobalMax;

	__host__ void fillData(int* data, const int length)
	{
		for (int i = 0; i < length; i++)
		{
			data[i] = i;
		}
		data[static_cast<int>(length * 0.5)] = length;
	}

	__host__ void fillData(int* data, const int length, const unsigned int value)
	{
		for (int i = 0; i < length; i++)
		{
			data[i] = i;
		}
	}

	__host__ void prepareData()
	{
		// paged-locked allocation
		constexpr unsigned int aSize = N * sizeof(int);
		constexpr unsigned int bSize = NO_BLOCKS * sizeof(int);

		cudaHostAlloc((void**)&a, aSize, cudaHostAllocDefault);
		cudaHostAlloc((void**)&b, bSize, cudaHostAllocDefault);

		fillData(a, N);
		fillData(b, NO_BLOCKS, INT_MIN);

		cudaMalloc((void**)&da, aSize);
		cudaMalloc((void**)&db, aSize);
		cudaMalloc((void**)&dGlobalMax, sizeof(int));

		cudaMemcpy(da, a, aSize, cudaMemcpyHostToDevice);
		cudaMemcpy(db, b, bSize, cudaMemcpyHostToDevice);
	}

	__host__ void releaseData()
	{
		cudaFree(da);
		cudaFree(db);
		cudaFree(dGlobalMax);

		cudaFreeHost(a);
		cudaFreeHost(da);
	}

	template<bool MAKE_IF>
	__global__ void kernel0(const int* __restrict__ data, const unsigned int dataLength, int* __restrict__ globalMax)
	{
		unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
		const unsigned int jump = gridDim.x * blockDim.x;

		const int* threadData = (int*)data + idx;

		while (idx < dataLength) {
			if constexpr (MAKE_IF) {
				if (*globalMax < *threadData) {
					atomicMax(globalMax, data[idx]);
				}
			}
			else {
				atomicMax(globalMax, data[idx]);
			}

			threadData += jump;
			idx += jump;
		}
	}

	template<bool MAKE_IF>
	__host__ void testKernel0()
	{
		dim3 blockSize(TPB, 1, 1);
		dim3 gridSize(getNumberOfParts(N, TPB), 1, 1);

		int globalMax = INT_MIN;

		auto test = [&]() {
			cudaMemcpy(dGlobalMax, &globalMax, sizeof(int), cudaMemcpyHostToDevice);
			kernel0<MAKE_IF> << <gridSize, blockSize >> > (da, N, dGlobalMax);
			};

		float gpuTime = GPUTIME(numberOfPasses, test());
		cudaDeviceSynchronize();
		printf("\x1B[93m[GPU time] %s: %f ms\033[0m\n", __PRETTY_FUNCTION__, gpuTime);

		cudaMemcpy(&globalMax, dGlobalMax, sizeof(int), cudaMemcpyDeviceToHost);
		printf("\nMaximum: %d\n", globalMax);
	}

	template<bool MAKE_IF>
	__global__ void kernel1(const int* __restrict__ data, const unsigned int dataLength, int* __restrict__ globalMax)
	{
		__shared__ int blockMax;

		unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (threadIdx.x == 0)
			blockMax = INT32_MIN;
		__syncthreads();

		const unsigned int jump = gridDim.x * blockDim.x;
		const int* threadData = data + idx;

		while (idx < dataLength) {
			if constexpr (MAKE_IF) {
				if (blockMax < *threadData) {
					atomicMax(&blockMax, data[idx]);
				}
			}
			else {
				atomicMax(&blockMax, data[idx]);
			}

			threadData += jump;
			idx += jump;
		}

		if (threadIdx.x == 0)
			atomicMax(globalMax, blockMax);
	}



	template<bool MAKE_IF>
	__host__ void testKernel1()
	{
		dim3 blockSize(TPB, 1, 1);
		dim3 gridSize(getNumberOfParts(N, TPB), 1, 1);

		int globalMax = INT_MIN;

		auto test = [&]() {
			cudaMemcpy(dGlobalMax, &globalMax, sizeof(int), cudaMemcpyHostToDevice);
			kernel1<MAKE_IF> << <gridSize, blockSize >> > (da, N, dGlobalMax);
			};

		float gpuTime = GPUTIME(numberOfPasses, test());
		cudaDeviceSynchronize();
		printf("\x1B[93m[GPU time] %s: %f ms\033[0m\n", __PRETTY_FUNCTION__, gpuTime);

		cudaMemcpy(&globalMax, dGlobalMax, sizeof(int), cudaMemcpyDeviceToHost);
		printf("\nMaximum: %d\n", globalMax);
	}

	void run()
	{
		prepareData();

		// TODO: CALL kernel 0
		// run it in Release mode and run it with Ctrl + F5
		printf("\nKernel 0\n");
		printf("TRUE\n");
		testKernel0<true>();
		printf("FALSE\n");
		testKernel0<false>();

		printf("\nKernel 1\n");
		printf("TRUE\n");
		testKernel1<true>();
		printf("FALSE\n");
		testKernel1<false>();

		releaseData();

		releaseData();
	}
} // namespace lab08