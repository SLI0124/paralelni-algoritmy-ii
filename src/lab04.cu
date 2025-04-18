#include <cudaDefs.h>
#include <math.h>
#include <random>
#include <time.h>

namespace lab04
{

#pragma region CustomStructure
	typedef struct __align__(8) CustomStructure
	{
	public:
		int dim;				//dimension
		int noRecords;			//number of Records

		CustomStructure& operator=(const CustomStructure & other)
		{
			dim = other.dim;
			noRecords = other.noRecords;
			return *this;
		}

		inline void print()
		{
			printf("Dimension: %u\n", dim);
			printf("Number of Records: %u\n", noRecords);
		}
	}CustomStructure;
#pragma endregion

	__constant__ __device__ int dScalarValue;
	__constant__ __device__ struct CustomStructure dCustomStructure;
	__constant__ __device__ int dConstantArray[20];
	__constant__ __device__ float* dPattern;


	__global__ void kernelConstantStruct(int* data, const unsigned int dataLength)
	{
		unsigned int threadOffset = blockIdx.x * blockDim.x + threadIdx.x;

		if (threadOffset < dataLength)
			data[threadOffset] = dCustomStructure.dim;
	}

	__global__ void kernelConstantArray(int* data, const unsigned int dataLength)
	{
		unsigned int threadOffset = blockIdx.x * blockDim.x + threadIdx.x;

		if (threadOffset < dataLength)
			data[threadOffset] = dConstantArray[0];
	}

	float* createData(const unsigned int length)
	{
		float* data = nullptr;
		data = (float*)malloc(length * sizeof(float));
		if (data == 0)
		{
			printf("Error: Cannot allocate memory\n");
			return 0;
		}
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<float> dis(0.0f, 1.0f);
		for (unsigned int i = 0; i < length; i++)
		{
			data[i] = dis(gen);
		}
		return data;
	}

	__global__ void findPattern(const float* __restrict__ reference, const size_t refererenceLenght, const size_t patternLenght, bool* output) {
		const uint32_t offset = blockIdx.x * blockDim.x + threadIdx.x;

		bool match = true;

		for (size_t i = 0; i < patternLenght; i++) {
			match &= (dPattern[i] != reference[offset + i]);

			//if (reference[offset + i] != dPattern[i]) 

			//	//match = false;
			//	// break; // ano i ne, je lepsi nechat, at to bezi, 
			//}
		}
		output[offset] = match;
	}

	// (lenght + size_of_pattern - 1) / size_of_pattern

	void run()
	{
		// pattern
		const unsigned int refereceLength = 1 << 23;
		const unsigned int patterLength = 16;

		float* hPattern = createData(patterLength);
		cudaMemcpyToSymbol(dPattern, hPattern, patterLength * sizeof(float));

		float* dRefence = nullptr;
		checkCudaErrors(cudaMalloc((void**)(&dRefence), patterLength * sizeof(float)));

		float* hReference = createData(refereceLength);
		cudaMemcpy(dRefence, hReference, refereceLength * sizeof(float), cudaMemcpyHostToDevice);



		// tasks
		// 01
		// 10
		// 
		// 0 1 (X2)
		//Test 0 - scalar Value
  //      int temp = 123;
  //      //cudaMemcpyToSymbol(static_cast<const void*>(&dScalarValue), static_cast<const void*>(&temp), 1 * sizeof(int));
		//cudaMemcpyToSymbol(dScalarValue, static_cast<const void*>(&temp), 1 * sizeof(int));
  //      int hScalarValue = -1;
  //      //cudaMemcpyFromSymbol(static_cast<void*>(&hScalarValue), static_cast<const void*>(&dScalarValue), 1 * sizeof(int));
		//cudaMemcpyFromSymbol(&hScalarValue, dScalarValue, 1 * sizeof(int));
		//printf("Scalar Value: %d => %d\n", temp, hScalarValue);
		//printf("\n");

		////Test 1 - structure
		//CustomStructure hCustomStructure = { 10, 20 };
		//cudaMemcpyToSymbol(dCustomStructure, &hCustomStructure, sizeof(CustomStructure));
		//CustomStructure hCustomStructure2;
		//cudaMemcpyFromSymbol(&hCustomStructure2, dCustomStructure, sizeof(CustomStructure));
		//printf("Structure (1): \n");
		//hCustomStructure.print();
		//printf("Structure (2): \n");
		//hCustomStructure2.print();
		//printf("\n");

		////Test2 - array
		//int hConstantArray[20];
		//for (int i = 0; i < 20; i++)
		//	hConstantArray[i] = i;
		//checkCudaErrors(cudaMemcpyToSymbol(dConstantArray, hConstantArray, 20 * sizeof(int)));
		//int hConstantArray2[20];
		//checkCudaErrors(cudaMemcpyFromSymbol(hConstantArray2, dConstantArray, 20 * sizeof(int)));
		//printf("Array (1): \n");
		//for (int i = 0; i < 20; i++)
		//	printf("%d ", hConstantArray[i]);
		//printf("\nArray (2): \n");
		//for (int i = 0; i < 20; i++)
		//	printf("%d ", hConstantArray2[i]);
		//printf("\n");
	}
} // namespace lab04