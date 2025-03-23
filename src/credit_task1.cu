#include <cudaDefs.h>
#include <random>

namespace credit_task1 {

	constexpr unsigned int THREADS_PER_BLOCK_DIM = 32;

	cudaError_t error = cudaSuccess;
	cudaDeviceProp deviceProp = cudaDeviceProp();

	// Function to generate a matrix with random values
	__host__ int generate_matrix(float* data, const unsigned int length, float range_min, float range_max) {
		// Fixed seed for reproducibility
		std::mt19937_64 mt(0);
		std::uniform_real_distribution<float> dist(range_min, range_max);

		// Random seed for random values
		/*std::random_device rd;
		std::mt19937 mt(rd());
		std::uniform_real_distribution<float> dist(range_min, range_max);*/

		if (data == nullptr)
			return -1;

		for (unsigned int i = 0; i < length; i++) {
			data[i] = dist(mt);
		}
		return 0;
	}

	// Kernel to convert float data to uint8_t (or other unsigned integer types) (from data should be int, float is bit misleading)
	template <typename T>
	__global__ void convert_data_type(const unsigned int length, float* data_from, T* data_to) {
		unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index < length) {
			data_to[index] = static_cast<T>(data_from[index]);
		}
	}

	// Kernel to calculate Euclidean distance
	template <typename T>
	__global__ void calculate_euclidean_distance(const unsigned int rows, const unsigned int cols, T* vector, T* data, float* results) {
		unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
		if (row < rows) {
			float sumSquared = 0;
			for (unsigned int i = 0; i < cols; i++) {
				float diff = static_cast<float>(vector[i]) - static_cast<float>(data[row * cols + i]);
				sumSquared += diff * diff;
			}
			results[row] = sqrtf(sumSquared);
		}
	}

	// Kernel to find the farthest object using parallel reduction
	__global__ void find_farthest_object_kernel(float* distances, unsigned int* indexes, float* res_distances, unsigned int* res_indexes, int m_objects) {
		__shared__ float shared_distances[THREADS_PER_BLOCK_DIM];
		__shared__ unsigned int shared_indexes[THREADS_PER_BLOCK_DIM];

		int tid = threadIdx.x;
		int global_id = blockIdx.x * blockDim.x + threadIdx.x;

		if (global_id < m_objects) {
			shared_distances[tid] = distances[global_id];
			shared_indexes[tid] = indexes[global_id];
			__syncthreads();

			for (int s = blockDim.x / 2; s > 0; s >>= 1) {
				if (tid < s) {
					if (shared_distances[tid] < shared_distances[tid + s]) {
						shared_distances[tid] = shared_distances[tid + s];
						shared_indexes[tid] = shared_indexes[tid + s];
					}
				}
				__syncthreads();
			}
		}

		if (tid == 0) {
			res_distances[blockIdx.x] = shared_distances[0];
			res_indexes[blockIdx.x] = shared_indexes[0];
		}
	}

	// Function to find the farthest object
	template <typename T>
	__host__ T* find_farthest_object(T* data, float* distances, const unsigned int m_objects, const unsigned int n_attributes) {
		if ((data == nullptr) || (distances == nullptr))
			return nullptr;

		unsigned int* h_indexes = (unsigned int*)malloc(m_objects * sizeof(unsigned int));
		if (h_indexes == nullptr) {
			perror("malloc returned nullptr");
			return nullptr;
		}

		for (int i = 0; i < m_objects; i++) {
			h_indexes[i] = i;
		}

		unsigned int* d_index;
		float* d_distances;
		unsigned int* d_res_index;
		float* d_res_distances;

		int numBlocks = (m_objects + THREADS_PER_BLOCK_DIM - 1) / THREADS_PER_BLOCK_DIM;
		int size = m_objects;

		checkCudaErrors(cudaMalloc((void**)&d_index, m_objects * sizeof(unsigned int)));
		checkCudaErrors(cudaMalloc((void**)&d_distances, m_objects * sizeof(float)));
		checkCudaErrors(cudaMalloc((void**)&d_res_index, numBlocks * sizeof(unsigned int)));
		checkCudaErrors(cudaMalloc((void**)&d_res_distances, numBlocks * sizeof(float)));

		checkCudaErrors(cudaMemcpy(d_index, h_indexes, m_objects * sizeof(unsigned int), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_distances, distances, m_objects * sizeof(float), cudaMemcpyHostToDevice));

		while (size > 1) { // keep reducing until we have only one element
			find_farthest_object_kernel << <numBlocks, THREADS_PER_BLOCK_DIM >> > (d_distances, d_index, d_res_distances, d_res_index, size);
			checkCudaErrors(cudaDeviceSynchronize());

			cudaMemcpy(d_index, d_res_index, numBlocks * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
			cudaMemcpy(d_distances, d_res_distances, numBlocks * sizeof(float), cudaMemcpyDeviceToDevice);

			size = numBlocks;
			numBlocks = (size + THREADS_PER_BLOCK_DIM - 1) / THREADS_PER_BLOCK_DIM;
		}

		unsigned int h_index;
		cudaMemcpy(&h_index, d_index, sizeof(unsigned int), cudaMemcpyDeviceToHost);

		cudaFree(d_res_index);
		cudaFree(d_res_distances);
		cudaFree(d_distances);
		cudaFree(d_index);

		delete[] h_indexes;
		T* vector = (T*)malloc(n_attributes * sizeof(T));
		if (vector == nullptr) {
			perror("malloc returned nullptr");
			return nullptr;
		}

		// Print the farthest object's distance and index
		printf("=============================================\n");
		printf("Distance Calculation Details:\n");
		printf("---------------------------------------------\n");
		printf("Distance: %f\n", distances[h_index]);
		printf("Index: %d\n", h_index);
		printf("=============================================\n");

		return vector;
	}

	// Function to calculate distances from the origin
	void calculate_object_to_origin_distances(uint8_t* d_mat, uint8_t* d_zero_vec, float* d_results, unsigned int m, unsigned int n, cudaEvent_t& startEvent, cudaEvent_t& stopEvent) {
		dim3 blockDim(THREADS_PER_BLOCK_DIM, 1);
		dim3 gridDim((m + blockDim.x - 1) / blockDim.x);

		cudaEventRecord(startEvent, 0);
		calculate_euclidean_distance << <gridDim, blockDim >> > (m, n, d_zero_vec, d_mat, d_results);
		checkCudaErrors(cudaDeviceSynchronize());
		cudaEventRecord(stopEvent, 0);
		cudaEventSynchronize(stopEvent);
	}

	// Function to calculate distances from the farthest object
	void calculate_object_to_object_distances(uint8_t* d_mat, uint8_t* d_vec, float* d_results, unsigned int m, unsigned int n, cudaEvent_t& startEvent, cudaEvent_t& stopEvent) {
		dim3 blockDim(THREADS_PER_BLOCK_DIM, 1);
		dim3 gridDim((m + blockDim.x - 1) / blockDim.x);

		cudaEventRecord(startEvent, 0);
		calculate_euclidean_distance << <gridDim, blockDim >> > (m, n, d_vec, d_mat, d_results);
		checkCudaErrors(cudaDeviceSynchronize());
		cudaEventRecord(stopEvent, 0);
		cudaEventSynchronize(stopEvent);
	}

	// Main function to run the program
	void run() {
		initializeCUDA(deviceProp);

		cudaEvent_t startEvent, stopEvent;
		float elapsedTime;
		cudaEventCreate(&startEvent);
		cudaEventCreate(&stopEvent);

		const unsigned int m = 2 << 21; // rows
		const unsigned int n = 256;    // cols
		const unsigned int length = n * m;
		float range_min = 0.0f; // misleading, ALWAYS should be 0.0f or more, no time to fix
		float range_max = 255.0f; // max range is whatever the max value of unsigned type you want

		// 1. Generate and discretize data
		float* h_mat = (float*)malloc(length * sizeof(float));
		if (h_mat == nullptr) {
			perror("malloc returned nullptr");
			return;
		}
		uint8_t* h_mat_retyped = new uint8_t[length];
		if (generate_matrix(h_mat, length, range_min, range_max) == -1) {
			perror("GenerateMatrix returned -1");
			return;
		}

		float* d_mat;
		uint8_t* d_mat_retyped;
		checkCudaErrors(cudaMalloc((void**)&d_mat, length * sizeof(float)));
		checkCudaErrors(cudaMalloc((void**)&d_mat_retyped, length * sizeof(uint8_t)));
		checkCudaErrors(cudaMemcpy(d_mat, h_mat, length * sizeof(float), cudaMemcpyHostToDevice));

		dim3 blockDim(THREADS_PER_BLOCK_DIM);
		dim3 gridDim((length + blockDim.x - 1) / blockDim.x);

		cudaEventRecord(startEvent, 0);
		convert_data_type << <gridDim, blockDim >> > (length, d_mat, d_mat_retyped);
		checkCudaErrors(cudaDeviceSynchronize());
		cudaEventRecord(stopEvent, 0);
		cudaEventSynchronize(stopEvent);
		cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
		printf("convert_data_type elapsedTime: %f ms\n", elapsedTime);

		checkCudaErrors(cudaMemcpy(h_mat_retyped, d_mat_retyped, length * sizeof(uint8_t), cudaMemcpyDeviceToHost));

		// 2. Calculate distances from origin
		uint8_t* h_zero_vec = new uint8_t[n];
		memset(h_zero_vec, 0, n * sizeof(uint8_t));
		float* h_results = new float[m];

		uint8_t* d_zero_vec;
		float* d_results;
		checkCudaErrors(cudaMalloc((void**)&d_zero_vec, n * sizeof(uint8_t)));
		checkCudaErrors(cudaMalloc((void**)&d_results, m * sizeof(float)));
		checkCudaErrors(cudaMemcpy(d_zero_vec, h_zero_vec, n * sizeof(uint8_t), cudaMemcpyHostToDevice));

		calculate_object_to_origin_distances(d_mat_retyped, d_zero_vec, d_results, m, n, startEvent, stopEvent);
		cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
		printf("CalculateDistance (Object to Origin) elapsedTime: %f ms\n", elapsedTime);

		checkCudaErrors(cudaMemcpy(h_results, d_results, m * sizeof(float), cudaMemcpyDeviceToHost));

		cudaEventRecord(startEvent, 0);
		uint8_t* h_vec = find_farthest_object(h_mat_retyped, h_results, m, n);
		cudaEventRecord(stopEvent, 0);
		cudaEventSynchronize(stopEvent);
		cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
		printf("Get farthest object (Object to Origin) elapsedTime: %f ms\n", elapsedTime);

		if (h_vec == nullptr) {
			perror("Get farthest object returned nullptr");
			return;
		}

		// 3. Calculate distances from the farthest object
		float* h_results2 = (float*)malloc(m * sizeof(float));
		if (h_results2 == nullptr) {
			perror("malloc returned nullptr");
			return;
		}
		uint8_t* d_vec;
		float* d_results2;
		checkCudaErrors(cudaMalloc((void**)&d_vec, n * sizeof(uint8_t)));
		checkCudaErrors(cudaMalloc((void**)&d_results2, m * sizeof(float)));
		checkCudaErrors(cudaMemcpy(d_vec, h_vec, n * sizeof(uint8_t), cudaMemcpyHostToDevice));

		calculate_object_to_object_distances(d_mat_retyped, d_vec, d_results2, m, n, startEvent, stopEvent);
		cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
		printf("Calculate distance (Object to Object) elapsedTime: %f ms\n", elapsedTime);

		checkCudaErrors(cudaMemcpy(h_results2, d_results2, m * sizeof(float), cudaMemcpyDeviceToHost));

		cudaEventRecord(startEvent, 0);
		uint8_t* h_vec2 = find_farthest_object(h_mat_retyped, h_results2, m, n);
		cudaEventRecord(stopEvent, 0);
		cudaEventSynchronize(stopEvent);
		cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
		printf("Get farthest object (Object to Object) elapsedTime: %f ms\n", elapsedTime);

		// Free memory
		checkCudaErrors(cudaFree(d_results2));
		checkCudaErrors(cudaFree(d_vec));
		checkCudaErrors(cudaFree(d_results));
		checkCudaErrors(cudaFree(d_zero_vec));
		checkCudaErrors(cudaFree(d_mat));
		checkCudaErrors(cudaFree(d_mat_retyped));

		delete[] h_vec2;
		delete[] h_results2;
		delete[] h_vec;
		delete[] h_results;
		delete[] h_zero_vec;
		delete[] h_mat;
		delete[] h_mat_retyped;

		cudaEventDestroy(startEvent);
		cudaEventDestroy(stopEvent);
	}
} // namespace credit_task1