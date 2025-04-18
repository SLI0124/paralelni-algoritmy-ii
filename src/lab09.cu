#include <cudaDefs.h>
#include <time.h>
#include <math.h>
#include <benchmark.h>

namespace lab09 {
	constexpr unsigned int N = 1 << 22;
	constexpr unsigned int MEMSIZE = N * sizeof(unsigned int);
	constexpr unsigned int NO_LOOPS = 100;
	constexpr unsigned int TPB = 256;
	constexpr unsigned int GRID_SIZE = (N + TPB - 1) / TPB;

	constexpr unsigned int NO_TEST_PHASES = 10;

	void fillData(unsigned int* data, const unsigned int length)
	{
		for (unsigned int i = 0; i < length; i++)
		{
			data[i] = 1;
		}
	}

	void printData(const unsigned int* data, const unsigned int length)
	{
		if (data == 0) return;
		for (unsigned int i = 0; i < length; i++)
		{
			printf("%u ", data[i]);
		}
	}


	__global__ void kernel(const unsigned int* a, const unsigned int* b, const unsigned int length, unsigned int* c)
	{
		const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
		//TODO:  thread block loop
		if (tid < length)
		{
			c[tid] = a[tid] + b[tid];
		}
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// <summary>	Tests 1. - single stream, async calling </summary>
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void test1()
	{
		unsigned int* a, * b, * c;
		unsigned int* da, * db, * dc;

		// paged-locked allocation
		cudaHostAlloc((void**)&a, NO_LOOPS * MEMSIZE, cudaHostAllocDefault);
		cudaHostAlloc((void**)&b, NO_LOOPS * MEMSIZE, cudaHostAllocDefault);
		cudaHostAlloc((void**)&c, NO_LOOPS * MEMSIZE, cudaHostAllocDefault);

		fillData(a, NO_LOOPS * N);
		fillData(b, NO_LOOPS * N);

		// Data chunks on GPU
		cudaMalloc((void**)&da, MEMSIZE);
		cudaMalloc((void**)&db, MEMSIZE);
		cudaMalloc((void**)&dc, MEMSIZE);

		//TODO: create stream
		cudaStream_t stream;
		cudaStreamCreate(&stream);


		auto lambda = [&]()
			{
				unsigned int dataOffset = 0;
				for (int i = 0; i < NO_LOOPS; i++)
				{
					//TODO:  copy a->da, b->db
					cudaMemcpyAsync(da, &a[dataOffset], MEMSIZE, cudaMemcpyHostToDevice, stream);
					cudaMemcpyAsync(db, &b[dataOffset], MEMSIZE, cudaMemcpyHostToDevice, stream);
					//TODO:  run the kernel in the stream
					kernel << <GRID_SIZE, TPB, 0, stream >> > (da, db, N, dc);
					//TODO:  copy dc->c
					cudaMemcpyAsync(&c[dataOffset], dc, MEMSIZE, cudaMemcpyDeviceToHost, stream);

					dataOffset += N;
				}
			};
		float gpuTime = GPUTIME(NO_TEST_PHASES, lambda());

		cudaStreamSynchronize(stream); // wait for stream to finish
		cudaStreamDestroy(stream);
		cudaDeviceSynchronize();
		printf("\x1B[93m[GPU time] %s: %f ms\033[0m\n", __PRETTY_FUNCTION__, gpuTime);

		//printData(c, 100);

		cudaFree(da);
		cudaFree(db);
		cudaFree(dc);

		cudaFreeHost(a);
		cudaFreeHost(b);
		cudaFreeHost(c);
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// <summary>	Tests 2. - two streams - depth first approach </summary>
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void test2()
	{
		//TODO: reuse the source code of above mentioned method test1()
		unsigned int* a, * b, * c;
		unsigned int* da0, * db0, * dc0;
		unsigned int* da1, * db1, * dc1;

		// paged-locked allocation
		cudaHostAlloc((void**)&a, NO_LOOPS * MEMSIZE, cudaHostAllocDefault);
		cudaHostAlloc((void**)&b, NO_LOOPS * MEMSIZE, cudaHostAllocDefault);
		cudaHostAlloc((void**)&c, NO_LOOPS * MEMSIZE, cudaHostAllocDefault);

		fillData(a, NO_LOOPS * N);
		fillData(b, NO_LOOPS * N);

		// Data chunks on GPU
		cudaMalloc((void**)&da0, MEMSIZE);
		cudaMalloc((void**)&db0, MEMSIZE);
		cudaMalloc((void**)&dc0, MEMSIZE);

		cudaMalloc((void**)&da1, MEMSIZE);
		cudaMalloc((void**)&db1, MEMSIZE);
		cudaMalloc((void**)&dc1, MEMSIZE);

		//TODO: create stream
		cudaStream_t stream0;
		cudaStreamCreate(&stream0);
		cudaStream_t stream1;
		cudaStreamCreate(&stream1);


		auto lambda = [&]()
			{
				unsigned int dataOffset = 0;
				for (int i = 0; i < NO_LOOPS; i += 2)
				{
					//TODO:  copy a->da, b->db
					cudaMemcpyAsync(da0, &a[dataOffset], MEMSIZE, cudaMemcpyHostToDevice, stream0);
					cudaMemcpyAsync(db0, &b[dataOffset], MEMSIZE, cudaMemcpyHostToDevice, stream0);
					//TODO:  run the kernel in the stream
					kernel << <GRID_SIZE, TPB, 0, stream0 >> > (da0, db0, N, dc0);
					//TODO:  copy dc->c
					cudaMemcpyAsync(&c[dataOffset], dc0, MEMSIZE, cudaMemcpyDeviceToHost, stream0);
					dataOffset += N;

					//TODO:  copy a->da, b->db
					cudaMemcpyAsync(da1, &a[dataOffset], MEMSIZE, cudaMemcpyHostToDevice, stream1);
					cudaMemcpyAsync(db1, &b[dataOffset], MEMSIZE, cudaMemcpyHostToDevice, stream1);
					//TODO:  run the kernel in the stream
					kernel << <GRID_SIZE, TPB, 0, stream1 >> > (da1, db1, N, dc1);
					//TODO:  copy dc->c
					cudaMemcpyAsync(&c[dataOffset], dc1, MEMSIZE, cudaMemcpyDeviceToHost, stream1);
					dataOffset += N;
				}
			};
		float gpuTime = GPUTIME(NO_TEST_PHASES, lambda());

		cudaStreamSynchronize(stream0); // wait for stream to finish
		cudaStreamSynchronize(stream1); // wait for stream to finish
		cudaStreamDestroy(stream0);
		cudaStreamDestroy(stream1);
		cudaDeviceSynchronize();
		printf("\x1B[93m[GPU time] %s: %f ms\033[0m\n", __PRETTY_FUNCTION__, gpuTime);

		//printData(c, 100);

		cudaFree(da0);
		cudaFree(db0);
		cudaFree(dc0);

		cudaFree(da1);
		cudaFree(db1);
		cudaFree(dc1);

		cudaFreeHost(a);
		cudaFreeHost(b);
		cudaFreeHost(c);

	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// <summary>	Tests 3. - two streams - breadth first approach</summary>
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void test3()
	{
		//TODO: reuse the source code of above mentioned method test1()
		unsigned int* a, * b, * c;
		unsigned int* da0, * db0, * dc0;
		unsigned int* da1, * db1, * dc1;

		// paged-locked allocation
		cudaHostAlloc((void**)&a, NO_LOOPS * MEMSIZE, cudaHostAllocDefault);
		cudaHostAlloc((void**)&b, NO_LOOPS * MEMSIZE, cudaHostAllocDefault);
		cudaHostAlloc((void**)&c, NO_LOOPS * MEMSIZE, cudaHostAllocDefault);

		fillData(a, NO_LOOPS * N);
		fillData(b, NO_LOOPS * N);

		// Data chunks on GPU
		cudaMalloc((void**)&da0, MEMSIZE);
		cudaMalloc((void**)&db0, MEMSIZE);
		cudaMalloc((void**)&dc0, MEMSIZE);

		cudaMalloc((void**)&da1, MEMSIZE);
		cudaMalloc((void**)&db1, MEMSIZE);
		cudaMalloc((void**)&dc1, MEMSIZE);

		//TODO: create stream
		cudaStream_t stream0;
		cudaStreamCreate(&stream0);
		cudaStream_t stream1;
		cudaStreamCreate(&stream1);


		auto lambda = [&]()
			{
				unsigned int dataOffset0 = 0;
				unsigned int dataOffset1 = N;

				for (int i = 0; i < NO_LOOPS; i += 2)
				{
					//TODO:  copy a->da, b->db
					cudaMemcpyAsync(da0, &a[dataOffset0], MEMSIZE, cudaMemcpyHostToDevice, stream0);
					cudaMemcpyAsync(da1, &a[dataOffset1], MEMSIZE, cudaMemcpyHostToDevice, stream1);
					cudaMemcpyAsync(db0, &b[dataOffset0], MEMSIZE, cudaMemcpyHostToDevice, stream0);
					cudaMemcpyAsync(db1, &b[dataOffset1], MEMSIZE, cudaMemcpyHostToDevice, stream1);
					//TODO:  run the kernel in the stream
					kernel << <GRID_SIZE, TPB, 0, stream0 >> > (da0, db0, N, dc0);
					kernel << <GRID_SIZE, TPB, 0, stream1 >> > (da1, db1, N, dc1);
					//TODO:  copy dc->c
					cudaMemcpyAsync(&c[dataOffset0], dc0, MEMSIZE, cudaMemcpyDeviceToHost, stream0);
					cudaMemcpyAsync(&c[dataOffset1], dc1, MEMSIZE, cudaMemcpyDeviceToHost, stream1);

					dataOffset0 += 2 * N;
					dataOffset1 += 2 * N;
				}
			};
		float gpuTime = GPUTIME(NO_TEST_PHASES, lambda());

		cudaStreamSynchronize(stream0); // wait for stream to finish
		cudaStreamSynchronize(stream1); // wait for stream to finish
		cudaStreamDestroy(stream0);
		cudaStreamDestroy(stream1);
		cudaDeviceSynchronize();
		printf("\x1B[93m[GPU time] %s: %f ms\033[0m\n", __PRETTY_FUNCTION__, gpuTime);

		//printData(c, 100);

		cudaFree(da0);
		cudaFree(db0);
		cudaFree(dc0);

		cudaFree(da1);
		cudaFree(db1);
		cudaFree(dc1);

		cudaFreeHost(a);
		cudaFreeHost(b);
		cudaFreeHost(c);
	}

	void run()
	{
		// run this in release mode and run it with Ctrl+F5
		test1();
		test2();
		test3();
	}
} // namespace lab09