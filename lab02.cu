#include <cudaDefs.h>
#include <benchmark.h>

namespace lab02
{
	constexpr unsigned int THREADS_PER_BLOCK_DIM = 8;				//=64 threads in block

	__global__ void fillData(const unsigned int pitch, const unsigned int rows, const unsigned int cols, float* data)
	{
		//TODO: fill data
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;

		if (row < rows && col < cols)
		{
			float* rowPtr = (float*)((char*)data + row * pitch);
			rowPtr[col] = row * cols + col;
		}
	}


	void run()
	{
		float* devPtr;
		size_t pitch;

		const unsigned int mRows = 5;
		const unsigned int mCols = 10;

		//TODO: Allocate Pitch memory
		cudaMallocPitch(&devPtr, &pitch, mCols * sizeof(float), mRows);

		//TODO: Prepare grid, blocks
		dim3 blockDim = dim3(THREADS_PER_BLOCK_DIM, THREADS_PER_BLOCK_DIM);
		dim3 gridDim = dim3((mCols + blockDim.x - 1) / blockDim.x, (mRows + blockDim.y - 1) / blockDim.y);

		//TODO: Call kernel
		//cudaEvent_t start, stop;
		//float elapsedTime;
		//cudaEventCreate(&start);
		//cudaEventCreate(&stop);
		//cudaEventRecord(start, 0);

		// 52428, 1, 1; 512, 1, 1;
		using gpubenchmark::print_time;
		printSettings(gridDim, blockDim);
		auto test1 = [&]() {fillData << <gridDim, blockDim >> > (pitch, mRows, mCols, devPtr); };
		print_time("fillData", test1, 1000);

		fillData << <gridDim, blockDim >> > (pitch, mRows, mCols, devPtr);

		//cudaEventRecord(stop, 0);
		//cudaEventSynchronize(stop);
		//cudaEventElapsedTime(&elapsedTime, start, stop);

		//printf("Time to get device properties: %f ms\n\n", elapsedTime);
		//cudaEventDestroy(start);
		//cudaEventDestroy(stop);

		//TODO: Allocate Host memory and copy back Device data
		float* data = new float[mRows * mCols];
		cudaMemcpy2D(data, mCols * sizeof(float), devPtr, pitch, mCols * sizeof(float), mRows, cudaMemcpyDeviceToHost);

		//TODO: Check data
		for (unsigned int i = 0; i < mRows; i++)
		{
			for (unsigned int j = 0; j < mCols; j++)
			{
				printf("%5.0f ", data[i * mCols + j]);
			}
			printf("\n");
		}

		//TODO: Free memory
		delete[] data;
		cudaFree(devPtr);
	}
} //namespace lab02