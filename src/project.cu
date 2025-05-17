#include <cudaDefs.h>
#include <time.h>
#include <cmath>
#include <random>
#include <iostream>
#include <curand.h> // Make sure this is included
#include <curand_kernel.h> // Include for curand_init

namespace project {
	using namespace std;

	cudaError_t error = cudaSuccess;
	cudaDeviceProp deviceProp = cudaDeviceProp();

	constexpr unsigned int NP = 1024;    // number of individuals in the population
	constexpr unsigned int DIM = 2 << 11;     // number of dimensions of individual

	//constexpr unsigned int NP = 100;    // number of individuals in the population
	//constexpr unsigned int DIM = 10;     // number of dimensions of individual

	constexpr float F = 0.5;      // mutation constant
	constexpr float CR = 0.5;     // crossover range
	constexpr int G_MAX = 1000;     // number of generation cycles

	constexpr unsigned int TPB = 256;

	typedef float (*TestFunction)(const float*, unsigned int);

	enum TestFunctionEnum { SPHERE = 0, GRIEWANK = 1, RASTRIGIN = 2 };

	struct Indiv
	{
		float values[DIM];
		float fitness;
	};


	// Helper function to check for cuRAND errors
	inline curandStatus_t checkCurandErrors(curandStatus_t err) {
		if (err != CURAND_STATUS_SUCCESS) {
			cerr << "cuRAND Error: " << err << endl;
			exit(-1);
		}
		return err;
	}

	__host__ Indiv* createData(const float lower_bound, const float upper_bound)
	{
		random_device rd;
		mt19937_64 mt(rd());
		uniform_real_distribution<float> dist(lower_bound, upper_bound);
		Indiv* data = new Indiv[NP];

		for (int i = 0; i < NP; i++)
		{
			for (unsigned int j = 0; j < DIM; j++)
			{
				data[i].values[j] = dist(mt);
			}
		}
		return data;
	}

	// Kernel to initialize the cuRAND states
	__global__ void initRand(curandState_t* rand_state, unsigned int seed, unsigned int n) {
		int id = blockIdx.x * blockDim.x + threadIdx.x;
		if (id < n) {
			curand_init(seed, id, 0, &rand_state[id]);
		}
	}

	__device__ float sphere(const float* individual, unsigned int dim) {
		float sum = 0.0f;
		for (unsigned int i = 0; i < dim; ++i) {
			sum += individual[i] * individual[i];
		}
		return sum;
	}

	__device__ float sphere_parallel(const float* individual, unsigned int dim, int tId,
		int localDimStart, int localDimStop) {
		__shared__ float sum[TPB];
		sum[tId] = 0.0f;

		for (unsigned int i = localDimStart; i < localDimStop; ++i) {
			sum[tId] += individual[i] * individual[i];
		}

		__syncthreads();

		int active_threads = (dim + TPB - 1) / TPB;
		int half = active_threads / 2;
		while (half > 0) {
			if (tId < half && tId + half < active_threads) {
				sum[tId] += sum[tId + half];
			}
			__syncthreads();
			half /= 2;
		}
		__syncthreads();

		return sum[0];
	}

	__device__ float griewank(const float* individual, unsigned int dim) {
		float sum = 0.0f;
		float product = 1.0f;
		for (unsigned int i = 0; i < dim; ++i) {
			sum += (individual[i] * individual[i]) / 4000.0f;
			product *= cos(individual[i] / sqrt(static_cast<float>(i + 1)));
		}
		return sum - product + 1.0f;
	}

	__device__ float griewank_parallel(const float* individual, unsigned int dim, int tId,
		int localDimStart, int localDimStop) {

		__shared__ float sum[TPB];
		sum[tId] = 0.0f;
		__shared__ float product[TPB];
		product[tId] = 1.0f;

		for (unsigned int i = localDimStart; i < localDimStop; ++i) {
			sum[tId] += (individual[i] * individual[i]) / 4000.0f;
			product[tId] *= cos(individual[i] / sqrt(static_cast<float>(i + 1)));
		}

		__syncthreads();

		int active_threads = (dim + TPB - 1) / TPB;
		int half = active_threads / 2;
		while (half > 0) {
			if (tId < half && tId + half < active_threads) {
				sum[tId] += sum[tId + half];
				product[tId] *= product[tId + half];
			}
			__syncthreads();
			half /= 2;
		}
		__syncthreads();

		return sum[0] - product[0] + 1.0f;
	}

	__device__ float rastrigin(const float* individual, unsigned int dim) {
		const float PI = 3.14159265358979323846f;
		float sum = 0.0f;
		for (unsigned int i = 0; i < dim; ++i) {
			sum += (individual[i] * individual[i]) - 10.0f * cos(2.0f * PI * individual[i]);
		}
		return 10.0f * dim + sum;
	}

	__device__ float rastrigin_parallel(const float* individual, unsigned int dim, int tId,
		int localDimStart, int localDimStop) {

		const float PI = 3.14159265358979323846f;
		__shared__ float sum[TPB];
		sum[tId] = 0.0f;
		for (unsigned int i = localDimStart; i < localDimStop; ++i) {
			sum[tId] += (individual[i] * individual[i]) - 10.0f * cos(2.0f * PI * individual[i]);
		}

		__syncthreads();

		int active_threads = (dim + TPB - 1) / TPB;
		int half = active_threads / 2;
		while (half > 0) {
			if (tId < half && tId + half < active_threads) {
				sum[tId] += sum[tId + half];
			}
			__syncthreads();
			half /= 2;
		}
		__syncthreads();

		return 10.0f * dim + sum[0];
	}

	__device__ float evaluate(const float* individual, unsigned int dim, int function_id, int tId,
		int localDimStart, int localDimStop) {
		switch (function_id) {
		case SPHERE:
			return sphere_parallel(individual, dim, tId, localDimStart, localDimStop);
			//return sphere(individual, dim);
		case GRIEWANK:
			return griewank_parallel(individual, dim, tId, localDimStart, localDimStop);
			//return griewank(individual, dim);
		case RASTRIGIN:
			return rastrigin_parallel(individual, dim, tId, localDimStart, localDimStop);
			//return rastrigin(individual, dim);
		default:
			return INFINITY;
		}
	}

	__global__ void kernelIndividual(Indiv* pop, curandState_t* global_rand_state,
		const float lower_bound, const float upper_bound,
		int function_id)
	{
		// Declare shared memory arrays dynamically
		extern __shared__ float shared_mem[];
		float* mutated_individual = shared_mem;
		float* new_individual = &shared_mem[DIM];

		int gtId = blockIdx.x * blockDim.x + threadIdx.x; //Global thread id
		int bId = blockIdx.x; //Current block id
		int tId = threadIdx.x; // Current local thread id

		int localDimRange = DIM / blockDim.x;
		int localDimStart = tId * localDimRange;
		int localDimStop = (tId == blockDim.x - 1) || (localDimRange < 1) ? DIM : localDimStart + localDimRange;

		if (localDimRange < 1 && tId != 0)
			return;

		if (bId < NP)
		{
			//Mutation
			curandState_t* local_rand_state = &global_rand_state[bId]; // Use row as the index for the state
			int r1_i = curand(local_rand_state) % NP;
			int r2_i = curand(local_rand_state) % NP;
			int r3_i = curand(local_rand_state) % NP;

			for (unsigned int i = localDimStart; i < localDimStop; i++)
			{
				mutated_individual[i] = (pop[r1_i].values[i] - pop[r2_i].values[i]) * F + pop[r3_i].values[i];
				//bounds check
				if (mutated_individual[i] < lower_bound)
					mutated_individual[i] = lower_bound;
				else if (mutated_individual[i] > upper_bound)
					mutated_individual[i] = upper_bound;
			}

			__syncthreads();

			//Cross-over
			int rand_j = curand(local_rand_state) % DIM;

			for (unsigned int j = localDimStart; j < localDimStop; j++)
			{
				float random = curand_uniform(local_rand_state);

				if (random < CR || j == rand_j)
					new_individual[j] = mutated_individual[j];
				else
					new_individual[j] = pop[bId].values[j];
			}

			__syncthreads();

			//Selection
			float current_fitness = evaluate(pop[bId].values, DIM, function_id, tId, localDimStart, localDimStop);
			float trial_fitness = evaluate(new_individual, DIM, function_id, tId, localDimStart, localDimStop);

			__syncthreads();

			// Make sure thread 0 writes the fitness value
			if (tId == 0) {
				pop[bId].fitness = current_fitness;

				if (trial_fitness < current_fitness) {
					pop[bId].fitness = trial_fitness;
				}
			}

			__syncthreads();

			// Only update if the trial is better
			if (trial_fitness < current_fitness) {
				for (unsigned int i = localDimStart; i < localDimStop; ++i) {
					pop[bId].values[i] = new_individual[i];
				}
			}
		}
	}

	constexpr unsigned int closest_power_of_two(const unsigned int n) {
		unsigned int power = n;
		power--;

		power |= power >> 1;
		power |= power >> 2;
		power |= power >> 4;
		power |= power >> 8;
		power |= power >> 16;

		power++;

		return power;
	}

	constexpr unsigned int NP2 = closest_power_of_two(NP);

	__global__ void parallelReduction(const Indiv* __restrict__ d_pop, unsigned int* __restrict__ d_best_index, float* __restrict__ d_best_value) {
		__shared__ float s_fitnesses[NP2];
		__shared__ unsigned int s_indices[NP2];

		unsigned int tid = threadIdx.x;

		if (tid >= NP2)
			return;

		// Load data from global memory to shared memory
		if (tid >= NP) {
			s_indices[tid] = UINT_MAX;
			s_fitnesses[tid] = INFINITY;
		}
		else {
			s_indices[tid] = tid;
			s_fitnesses[tid] = d_pop[tid].fitness;
		}
		__syncthreads();

		// Hierarchical reduction using ifs
		for (unsigned int stride = NP2 >> 1; stride > 16; stride >>= 1) {
			if (tid < stride) {
				if (s_fitnesses[tid] > s_fitnesses[tid + stride]) {
					s_fitnesses[tid] = s_fitnesses[tid + stride];
					s_indices[tid] = s_indices[tid + stride];
				}
			}
			__syncthreads();
		}

		// Warp-level reduction
		volatile float* v_fitnesses = s_fitnesses;
		volatile unsigned int* v_indices = s_indices;

		if (tid < 16) {
			for (unsigned int stride = 16; stride > 0; stride >>= 1) {
				if (v_fitnesses[tid] > v_fitnesses[tid + stride]) {
					v_fitnesses[tid] = v_fitnesses[tid + stride];
					v_indices[tid] = v_indices[tid + stride];
				}
			}
		}

		// Store the result
		if (tid == 0) {
			*d_best_index = s_indices[0];
			*d_best_value = s_fitnesses[0];
		}
	}

	__host__ Indiv** differentialEvolution(Indiv* pop, curandState_t* d_rand_state,
		const float lower_bound, const float upper_bound,
		TestFunctionEnum test_function)
	{
		Indiv** results = new Indiv * [G_MAX];

		for (unsigned int g = 0; g < G_MAX; g++)
		{
			Indiv* d_pop = nullptr;

			checkCudaErrors(cudaMalloc((void**)&d_pop, sizeof(Indiv) * NP));
			checkCudaErrors(cudaMemcpy(d_pop, pop, sizeof(Indiv) * NP, cudaMemcpyHostToDevice));

			dim3 dimBlock{ TPB,1,1 };
			dim3 dimGrid{ NP,1,1 };

			// Calculate shared memory size needed (2 arrays of DIM floats)
			size_t sharedMemSize = 2 * DIM * sizeof(float);

			kernelIndividual << < dimGrid, dimBlock, sharedMemSize >> > (d_pop, d_rand_state, lower_bound, upper_bound, test_function);

			checkCudaErrors(cudaPeekAtLastError());
			checkCudaErrors(cudaDeviceSynchronize());

			// Copy the data back to host but DO NOT free d_pop yet - we need it for reduction
			checkCudaErrors(cudaMemcpy(pop, d_pop, sizeof(Indiv) * NP, cudaMemcpyDeviceToHost));

			unsigned int* d_best_index = nullptr;
			float* d_best_value = nullptr;
			checkCudaErrors(cudaMalloc((void**)&d_best_index, sizeof(unsigned int)));
			checkCudaErrors(cudaMalloc((void**)&d_best_value, sizeof(float)));

			parallelReduction << <1, NP2 >> > (d_pop, d_best_index, d_best_value);

			checkCudaErrors(cudaPeekAtLastError());
			checkCudaErrors(cudaDeviceSynchronize());

			unsigned int h_best_index;
			float h_best_value;
			checkCudaErrors(cudaMemcpy(&h_best_index, d_best_index, sizeof(unsigned int), cudaMemcpyDeviceToHost));
			checkCudaErrors(cudaMemcpy(&h_best_value, d_best_value, sizeof(float), cudaMemcpyDeviceToHost));

			std::cout << g + 1 << "/" << G_MAX << "> Best index: " << h_best_index << ", Best value: " << h_best_value << std::endl;

			// Free device memory AFTER the reduction is complete
			checkCudaErrors(cudaFree(d_pop));
			checkCudaErrors(cudaFree(d_best_index));
			checkCudaErrors(cudaFree(d_best_value));

			results[g] = new Indiv[NP];
			memcpy(results[g], pop, sizeof(Indiv) * NP);
		}

		return results;
	}

	void run()
	{
		// Initialize CUDA device (you might have a more comprehensive initialization function)
		int deviceId = 0;
		initializeCUDA(deviceProp);
		checkCudaErrors(cudaSetDevice(deviceId));
		checkCudaErrors(cudaGetDeviceProperties(&deviceProp, deviceId));
		cout << "Running on device: " << deviceProp.name << endl;

		const float lower_bound = -50.0f;
		const float upper_bound = 50.0f;

		Indiv* pop = createData(lower_bound, upper_bound);

		// Initialize cuRAND
		curandGenerator_t gen;
		checkCurandErrors(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
		unsigned long long seed = time(NULL);
		checkCurandErrors(curandSetPseudoRandomGeneratorSeed(gen, seed));

		// Allocate device memory for cuRAND states
		curandState_t* d_rand_state;
		checkCudaErrors(cudaMalloc((void**)&d_rand_state, NP * sizeof(curandState_t)));

		// Initialize cuRAND states on the device
		dim3 initBlockDim(TPB);
		dim3 initGridDim((NP + initBlockDim.x - 1) / initBlockDim.x);
		initRand << <initGridDim, initBlockDim >> > (d_rand_state, seed, NP);
		checkCudaErrors(cudaDeviceSynchronize()); // Wait for initialization to complete

		Indiv** results = differentialEvolution(pop, d_rand_state, lower_bound, upper_bound, SPHERE);

		// Cleanup
		checkCurandErrors(curandDestroyGenerator(gen));
		checkCudaErrors(cudaFree(d_rand_state));
		for (int i = 0; i < G_MAX; i++)
		{
			delete[] results[i];
		}
		delete[] results;
		delete[] pop;
		checkCudaErrors(cudaDeviceReset()); // Good practice to reset the device
	}
} // namespace project