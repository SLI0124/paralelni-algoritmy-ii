#include <cudaDefs.h>

namespace lab01 {

    constexpr unsigned int THREADS_PER_BLOCK = 256;
    constexpr unsigned int MEMBLOCKS_PER_THREADBLOCK = 2;

    using namespace std;

    __global__ void add1(const int* __restrict__ a, const int* __restrict__ b, const unsigned int length, int* __restrict__ c)
    {
        //TODO: c[i] = a[i] + b[i]
        unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x; // start
        const unsigned skip = gridDim.x * blockDim.x; // jump

        while (offset < length) {
            c[offset] = a[offset] + b[offset];
            offset += skip;
        }
    }

    __global__ void add2(const int* __restrict__ a, const int* __restrict__ b, const unsigned int length, int* __restrict__ c)
    {
        //TODO: c[i] = a[i] + b[i]
    }

	void run()
    {

        constexpr unsigned int length = 1 << 10;
        constexpr unsigned int sizeInBytes = length * sizeof(int);

        //TODO: Allocate Host memory
        int* a = (int*)malloc(length * sizeof(int));
        int* b = new int[length];
        int* c = static_cast<int*>(::operator new(length * sizeof(int)));

        //TODO: Init data
        int* ptrA = a;
        int* ptrB = b;
        for (unsigned int i = 0; i < length; ++i, ptrA++, ptrB++) {
            *ptrA = i;
            *ptrB = i;
        }

        //TODO: Allocate Device memory
        int* da = nullptr;
        int* db = nullptr;
        int* dc = nullptr;
        checkCudaErrors(cudaMalloc((void**)&da, sizeInBytes));
        checkCudaErrors(cudaMalloc((void**)&db, sizeInBytes));
        checkCudaErrors(cudaMalloc((void**)&dc, sizeInBytes));

        //TODO: Copy Data
        checkCudaErrors(cudaMemcpy(da, a, sizeInBytes, cudaMemcpyKind::cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(db, b, sizeInBytes, cudaMemcpyKind::cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(dc, c, sizeInBytes, cudaMemcpyKind::cudaMemcpyHostToDevice));

        //TODO: Prepare grid and blocks

        // thread per block
        constexpr unsigned int TPB = 256;
        constexpr unsigned int MBTPB = 2;


        dim3 dimBlock{ TPB ,1,1 };
        dim3 dimGrid{ 1,1,1 };

        //TODO: Call kernel
        add1 << < dimGrid, dimBlock >> > (da, db, length, dc);

        //TODO: Check results
        checkDeviceMatrix(dc, sizeInBytes, 1, length, "%d", "Device C");

        checkCudaErrors(cudaMemcpy(c, dc, sizeInBytes, cudaMemcpyKind::cudaMemcpyDeviceToHost));

        checkHostMatrix(c, sizeInBytes, 1, length, "%d", "C");

        //TODO: Free memory
        delete[] a;
        delete[] b;
        delete[] c;

        cudaFree(da);
        cudaFree(db);
        cudaFree(dc);

        /*
        SAFE_DELETE_ARRAY(a);
        SAFE_DELETE_ARRAY(b);
        SAFE_DELETE_ARRAY(c);

        SAFE_DELETE_CUDA(da);
        SAFE_DELETE_CUDA(db);
        SAFE_DELETE_CUDA(dc);
        */
    }
} // namespace lab01
