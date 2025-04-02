// includes, cudaimageWidth
#include <cudaDefs.h>

#include <helper_math.h>			// normalize method

#include <benchmark.h>
#include <imageManager.h>
#include <imageUtils.cuh>


namespace lab06 {

#define TPB_1D 8						// ThreadsPerBlock in one dimension
#define TPB_2D TPB_1D*TPB_1D			// ThreadsPerBlock = TPB_1D*TPB_1D (2D block)

	cudaError_t error = cudaSuccess;
	cudaDeviceProp deviceProp = cudaDeviceProp();

	using DT = float;

	__host__ TextureInfo createTextureObjectFrom2DArray(const ImageInfo<DT>& ii)
	{
		TextureInfo ti;

		//TODO

		// Size info
		ti.size = { ii.width, ii.height, 1 };

		//Texture Data settings
		//ti.texChannelDesc = 
		ti.texChannelDesc = cudaCreateChannelDesc<DT>();

		//Allocate cudaArray and copy data into this array
		//...
		checkCudaErrors(cudaMallocArray(&ti.texArrayData, &ti.texChannelDesc, ii.width, ii.height));
		checkCudaErrors(cudaMemcpyToArray(ti.texArrayData, 0, 0, ii.dPtr, ii.pitch * ii.height, cudaMemcpyDeviceToDevice));

		// Specify texture resource
		//ti.resDesc.resType = ... 
		ti.resDesc.resType = cudaResourceTypeArray;
		//ti.resDesc.res.array.array = ... 
		ti.resDesc.res.array.array = ti.texArrayData;

		// Specify texture object parameters
		//ti.texDesc.addressMode[0] = ...
		ti.texDesc.addressMode[0] = cudaAddressModeClamp;
		//ti.texDesc.addressMode[1] = ... 
		ti.texDesc.addressMode[1] = cudaAddressModeClamp;
		//ti.texDesc.filterMode = ... 
		ti.texDesc.filterMode = cudaFilterModePoint;
		//ti.texDesc.readMode = ... 
		ti.texDesc.readMode = cudaReadModeElementType;
		//ti.texDesc.normalizedCoords = false;
		ti.texDesc.normalizedCoords = false;

		// Create texture object
		//checkCudaErrors(cudaCreateTextureObject(...));
		checkCudaErrors(cudaCreateTextureObject(&ti.texObj, &ti.resDesc, &ti.texDesc, nullptr));

		return ti;
	}

	template<bool normalizeTexel>
	__global__ void createNormalmap(const cudaTextureObject_t srcTex, const unsigned int srcWidth, const unsigned int srcHeight, const unsigned int dstPitchInElements, uchar3* dst)
	{
		const int x = blockIdx.x * blockDim.x + threadIdx.x;
		const int y = blockIdx.y * blockDim.y + threadIdx.y;

		if (x >= srcWidth || y >= srcHeight)
		{
			return;
		}

		const int dstOffset = y * dstPitchInElements + x;

		// Extract the neighboring texels
		float top_left = tex2D<float>(srcTex, x - 1, y - 1);
		float top = tex2D<float>(srcTex, x, y - 1);
		float top_right = tex2D<float>(srcTex, x + 1, y - 1);

		float left = tex2D<float>(srcTex, x - 1, y);
		float center = tex2D<float>(srcTex, x, y);
		float right = tex2D<float>(srcTex, x + 1, y);

		float bottom_left = tex2D<float>(srcTex, x - 1, y + 1);
		float bottom = tex2D<float>(srcTex, x, y + 1);
		float bottom_right = tex2D<float>(srcTex, x + 1, y + 1);

		// Compute the normal using the Sobel filter
		float3 normal;
		normal.x = (top_right + 2.0f * right + bottom_right) - (top_left + 2.0f * left + bottom_left);
		normal.y = (bottom_left + 2.0f * bottom + bottom_right) - (top_left + 2.0f * top + top_right);
		normal.z = 1.0f; // Set Z component to 1.0 for a unit vector

		// Normalize the normal vector
		if constexpr (normalizeTexel)
		{
			normal = normalize(normal);
		}

		// Map the normal vector to RGB values
		normal.x = (normal.x + 1.0f) * 127.5f;
		normal.y = (normal.y + 1.0f) * 127.5f;
		normal.z = (normal.z + 1.0f) * 127.5f;

		// RGB - this format uses Linux
		//dst[dstOffset] = make_uchar3(static_cast<uint8_t>(normal.x), static_cast<uint8_t>(normal.y), static_cast<uint8_t>(normal.z));

		// BGR - this format use Windows and MacOS
		dst[dstOffset] = make_uchar3(static_cast<uint8_t>(normal.z), static_cast<uint8_t>(normal.y), static_cast<uint8_t>(normal.x));
	}

	void saveTexImage(const char* imageFileName, const uint32_t dstWidth, const uint32_t dstHeight, const uint32_t dstPitch, const uchar3* dstData)
	{
		FIBITMAP* tmp = FreeImage_Allocate(dstWidth, dstHeight, 24);
		unsigned int tmpPitch = FreeImage_GetPitch(tmp);					// FREEIMAGE align row data ... You have to use pitch instead of width
		checkCudaErrors(cudaMemcpy2D(FreeImage_GetBits(tmp), tmpPitch, dstData, dstPitch, dstWidth * 3, dstHeight, cudaMemcpyDeviceToHost));
		//FreeImage_Save(FIF_BMP, tmp, imageFileName, 0);
		ImageManager::GenericWriter(tmp, imageFileName, FIF_BMP);
		FreeImage_Unload(tmp);
	}

	void run()
	{
		initializeCUDA(deviceProp);
		FreeImage_Initialise();

		// STEP 1 - load raw image data, HOST->DEVICE, with/without pitch
		ImageInfo<DT> src;
		prepareData<false>("./res/terrain3Kx3K.tif", src);

		// STEP 2 - create texture from the raw data
		TextureInfo ti = createTextureObjectFrom2DArray(src);

		// SETP 3 - allocate pitch memory to store output image data
		size_t dstPitch;
		uchar3* dst;
		checkCudaErrors(cudaMallocPitch((void**)&dst, &dstPitch, src.width * sizeof(uchar3), src.height));

		// STEP 4 - create normal map
		dim3 dimBlock{ TPB_1D, TPB_1D, 1 };
		dim3 dimGrid{ (src.width + TPB_1D - 1) / TPB_1D, (src.height + TPB_1D - 1) / TPB_1D, 1 };
		// kernel
		createNormalmap<true> << <dimGrid, dimBlock >> > (ti.texObj, src.width, src.height, dstPitch / sizeof(uchar3), dst);

		// STEP 5 - save the normal map
		saveTexImage("./res/terrain3Kx3K_normalmap.bmp", src.width, src.height, dstPitch, dst);

		// SETP 6 - release unused data
		if (ti.texObj)
			checkCudaErrors(cudaDestroyTextureObject(ti.texObj));

		if (ti.texArrayData)
			checkCudaErrors(cudaFreeArray(ti.texArrayData));

		if (src.dPtr)
			checkCudaErrors(cudaFree(src.dPtr));

		if (dst)
			checkCudaErrors(cudaFree(dst));

		cudaDeviceSynchronize();
		error = cudaGetLastError();

		FreeImage_DeInitialise();
	}
} // namespace lab06