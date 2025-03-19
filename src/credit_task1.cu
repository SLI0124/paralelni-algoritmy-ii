#include <cudaDefs.h>
#include <curand_kernel.h> // curand_init, curand_uniform - initialization and generation of random numbers on GPU

namespace credit_task1
{
	constexpr unsigned int DIMENSIONS = 1 << 20; // rows 
	constexpr unsigned int N = 1024; // columns

	__global__ void generateDataKernel(float* d_data, size_t pitch, unsigned int rows, unsigned int cols, float lower_range, float upper_range, unsigned int seed)
	{
		unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
		if (row >= rows) return;

		// Get row pointer using pitch
		float* row_ptr = (float*)((char*)d_data + row * pitch);

		// Random number generation
		curandState state;
		curand_init(seed + row, 0, 0, &state);

		for (unsigned int col = 0; col < cols; col++)
		{
			row_ptr[col] = lower_range + (upper_range - lower_range) * curand_uniform(&state);
		}
	}

	void run()
	{
		printf("Credit task 1\n");
		printf("Creating data...\n");

		// Define matrix size
		unsigned int rows = DIMENSIONS;
		unsigned int cols = N;
		float lower_range = -100.0f;
		float upper_range = 100.0f;

		// Allocate memory using cudaMallocPitch()
		float* d_data;
		size_t pitch;
		checkCudaErrors(cudaMallocPitch(&d_data, &pitch, cols * sizeof(float), rows));

		// Launch kernel to generate random values on GPU
		const unsigned int threadsPerBlock = 256;
		const unsigned int blocksPerGrid = (rows + threadsPerBlock - 1) / threadsPerBlock;
		generateDataKernel << <blocksPerGrid, threadsPerBlock >> > (d_data, pitch, rows, cols, lower_range, upper_range, time(0));

		// Copy data from device to host
		float* h_data = (float*)malloc(rows * cols * sizeof(float));
		if (h_data == 0)
		{
			printf("Error: Cannot allocate memory\n");
			return;
		}
		checkCudaErrors(cudaMemcpy2D(h_data, cols * sizeof(float), d_data, pitch, cols * sizeof(float), rows, cudaMemcpyDeviceToHost));

		// Free memory after use
		checkCudaErrors(cudaFree(d_data));
		free(h_data);
	}
} // namespace credit_task1
