#include <cudaDefs.h>
#include <time.h>
#include <math.h>
#include <random>

namespace lab03 {

	//WARNING!!! Do not change TPB and NO_FORCES for this demo !!!
	constexpr unsigned int TPB = 128; // threads per block - double of what we have threads in the block
	constexpr unsigned int NO_FORCES = 256;
	constexpr unsigned int NO_RAIN_DROPS = 1 << 20;

	constexpr unsigned int MEM_BLOCKS_PER_THREAD_BLOCK = 8;

	cudaError_t error = cudaSuccess;
	cudaDeviceProp deviceProp = cudaDeviceProp();

	using namespace std;

	__host__ float3* createData(const unsigned int length)
	{
		//TODO: Generate float3 vectors. You can use 'make_float3' method.
		float3* data = nullptr;
		data = (float3*)malloc(length * sizeof(float3));
		if (data == 0)
		{
			printf("Error: Cannot allocate memory\n");
			return 0;
		}

		//make_float3(0.0f, 0.0f, 0.0f);


		// firstly allocate memory for the data
		// lol, this work
		/*float3* data = static_cast<float*>(::operator new(length * sizeof(float3)));*/
		// open handler 


		// generate random data
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<float> dis(0.0f, 1.0f);

		for (unsigned int i = 0; i < length; i++)
		{
			data[i].x = dis(gen);
			data[i].y = dis(gen);
			data[i].z = dis(gen);
		}

		return data;
	}

	__host__ void printData(const float3* data, const unsigned int length)
	{
		if (data == 0) return;
		const float3* ptr = data;
		for (unsigned int i = 0; i < length; i++, ptr++)
		{
			printf("%5.2f %5.2f %5.2f ", ptr->x, ptr->y, ptr->z);
		}
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// <summary>	Sums the forces to get the final one using parallel reduction. 
	/// 		    WARNING!!! The method was written to meet input requirements of our example, i.e. 128 threads and 256 forces  </summary>
	/// <param name="dForces">	  	The forces. </param>
	/// <param name="noForces">   	The number of forces. </param>
	/// <param name="dFinalForce">	[in,out] If non-null, the final force. </param>
	////////////////////////////////////////////////////////////////////////////////////////////////////
	__global__ void reduce(const float3* __restrict__ dForces, const unsigned int noForces, float3* __restrict__ dFinalForce)
	{
		__shared__ float3 sForces[TPB];					//SEE THE WARNING MESSAGE !!!
		unsigned int tid = threadIdx.x;
		unsigned int next = TPB;						//SEE THE WARNING MESSAGE !!!

		//TODO: Make the reduction
		float3* src1 = &sForces[tid]; // pointer to the current thread's data, taking data from the global memory
		float3* src2 = (float3*)&dForces[tid]; // pointer to the global memory, taking data from the global memory from the half of the block

		*src1 = dForces[tid]; // copy the data from the global memory to the shared memory
		src1->x += src2->x; // add the data from the global memory to the shared memory
		src1->y += src2->y;
		src1->z += src2->z;

		__syncthreads(); // wait for all threads to finish, we will use the shared memory onwards since we don't need the global memory anymore

		// now we have to reduce the data
		next >>= 1; // divide the number of threads by 2, now at 64

		if (tid >= next) return; // if the thread is greater than the number of threads, return, eliminate the second half of the threads

		src2 = src1 + next; // move the pointer to the next element

		src1->x += src2->x; // add the data from the next element to the current element
		src1->y += src2->y;
		src1->z += src2->z;

		__syncthreads(); // wait for all threads to finish

		next >>= 1; // divide the number of threads by 2, now at 32

		if (tid >= next) return; // if the thread is greater than the number of threads, return, eliminate the second half of the threads

		volatile float3* vsrc1 = &sForces[tid]; // pointer to the current thread's data, taking data from the shared memory
		volatile float3* vsrc2 = vsrc1 + next; // pointer to the next element

		vsrc2 = vsrc1 + next; // move the pointer to the next element
		vsrc1->x += vsrc2->x; // add the data from the next element to the current element
		vsrc1->y += vsrc2->y;
		vsrc1->z += vsrc2->z;

		// no need to __syncthreads() here, since some number is at 32 something something

		next >>= 1; // divide the number of threads by 2, now at 16

		if (tid >= next) return; // if the thread is greater than the number of threads, return, eliminate the second half of the threads
		vsrc2 = vsrc1 + next; // move the pointer to the next element
		vsrc1->x += vsrc2->x; // add the data from the next element to the current element
		vsrc1->y += vsrc2->y;
		vsrc1->z += vsrc2->z;

		next >>= 1; // divide the number of threads by 2, now at 8

		if (tid >= next) return; // if the thread is greater than the number of threads, return, eliminate the second half of the threads
		vsrc2 = vsrc1 + next; // move the pointer to the next element
		vsrc1->x += vsrc2->x; // add the data from the next element to the current element
		vsrc1->y += vsrc2->y;
		vsrc1->z += vsrc2->z;

		next >>= 1; // divide the number of threads by 2, now at 4

		if (tid >= next) return; // if the thread is greater than the number of threads, return, eliminate the second half of the threads
		vsrc2 = vsrc1 + next; // move the pointer to the next element
		vsrc1->x += vsrc2->x; // add the data from the next element to the current element
		vsrc1->y += vsrc2->y;
		vsrc1->z += vsrc2->z;

		next >>= 1; // divide the number of threads by 2, now at 1

		if (tid >= next) return; // if the thread is greater than the number of threads, return, eliminate the second half of the threads
		vsrc2 = vsrc1 + next; // move the pointer to the next element
		vsrc1->x += vsrc2->x; // add the data from the next element to the current element
		vsrc1->y += vsrc2->y;
		vsrc1->z += vsrc2->z;

		// now we have the final result 
		if (tid == 0)
		{
			dFinalForce->x = vsrc1->x;
			dFinalForce->y = vsrc1->y;
			dFinalForce->z = vsrc1->z;
		}
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// <summary>	Adds the FinalForce to every Rain drops position. </summary>
	/// <param name="dFinalForce">	The final force. </param>
	/// <param name="noRainDrops">	The number of rain drops. </param>
	/// <param name="dRainDrops"> 	[in,out] If non-null, the rain drops positions. </param>
	////////////////////////////////////////////////////////////////////////////////////////////////////
	__global__ void add(const float3* __restrict__ dFinalForce, const unsigned int noRainDrops, float3* __restrict__ dRainDrops)
	{
		//TODO: Add the FinalForce to every Rain drops position.
		unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
		if (tid >= noRainDrops) return;


	}


	void run()
	{
		initializeCUDA(deviceProp);

		cudaEvent_t startEvent, stopEvent;
		float elapsedTime;

		cudaEventCreate(&startEvent);
		cudaEventCreate(&stopEvent);
		cudaEventRecord(startEvent, 0);

		float3* hForces = createData(NO_FORCES);
		float3* hDrops = createData(NO_RAIN_DROPS);

		float3* dForces = nullptr;
		float3* dDrops = nullptr;
		float3* dFinalForce = nullptr;

		checkCudaErrors(cudaMalloc((void**)&dForces, NO_FORCES * sizeof(float3)));
		checkCudaErrors(cudaMemcpy(dForces, hForces, NO_FORCES * sizeof(float3), cudaMemcpyHostToDevice));

		checkCudaErrors(cudaMalloc((void**)&dDrops, NO_RAIN_DROPS * sizeof(float3)));
		checkCudaErrors(cudaMemcpy(dDrops, hDrops, NO_RAIN_DROPS * sizeof(float3), cudaMemcpyHostToDevice));

		checkCudaErrors(cudaMalloc((void**)&dFinalForce, sizeof(float3)));

		KernelSetting ksReduce;
		//ksReduce.dimBlock = dim3{ TPB,1,1 };
		//ksReduce.dimGrid = dim3{ 1,1,1 };

		//TODO: ... Set ksReduce

		dim3 dimGrid{ 1,1,1 };
		dim3 dimBlock{ TPB,1,1 };

		KernelSetting ksAdd;
		//TODO: ... Set ksAdd

		for (unsigned int i = 0; i < 1000; i++)
		{
			//reduce << <ksReduce.dimGrid, ksReduce.dimBlock >> > (dForces, NO_FORCES, dFinalForce);
			//add << <ksAdd.dimGrid, ksAdd.dimBlock >> > (dFinalForce, NO_RAIN_DROPS, dDrops);
			reduce << <dimGrid, dimBlock >> > (dForces, NO_FORCES, dFinalForce);
			add << <dimGrid, dimBlock >> > (dFinalForce, NO_RAIN_DROPS, dDrops); // other params than for add, maybe check it
		}

		checkDeviceMatrix<float>((float*)dFinalForce, sizeof(float3), 1, 3, "%5.2f ", "Final force");
		// checkDeviceMatrix<float>((float*)dDrops, sizeof(float3), NO_RAIN_DROPS, 3, "%5.2f ", "Final Rain Drops");

		if (hForces)
			free(hForces);
		if (hDrops)
			free(hDrops);

		checkCudaErrors(cudaFree(dForces));
		checkCudaErrors(cudaFree(dDrops));

		cudaEventRecord(stopEvent, 0);
		cudaEventSynchronize(stopEvent);

		cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
		cudaEventDestroy(startEvent);
		cudaEventDestroy(stopEvent);

		printf("Time to get device properties: %f ms", elapsedTime);
	}
} //namespace lab03