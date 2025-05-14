#include <cudaDefs.h>

#include <helper_math.h>			// normalize method

#include <benchmark.h>
#include <imageManager.h>
#include <imageUtils.cuh>


namespace credit_task4 {

#define TPB_1D 8						// ThreadsPerBlock in one dimension
#define TPB_2D TPB_1D*TPB_1D			// ThreadsPerBlock = TPB_1D*TPB_1D (2D block)

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

	unsigned char* dImageData = nullptr;
	unsigned int imageWidth = 0;
	unsigned int imageHeight = 0;
	unsigned int imagePitch = 0;
	unsigned int imageBPP = 0; // bits per pixel, e.g. 8, 16, 24, 32

	size_t texSourcePitch = 0;
	float* dLinearPitchTextureData = 0;

	size_t texPatternPitch = 0;
	float* dPatternLinearPitchTextureData = 0;

	unsigned char* dPatternData = nullptr;
	unsigned int patternWidth = 0;
	unsigned int patternHeight = 0;
	unsigned int patternPitch = 0;
	unsigned int patternBPP = 0;

	void loadSourceImage(const char* imageFileName)
	{
		FreeImage_Initialise();
		FIBITMAP* tmp = ImageManager::GenericLoader(imageFileName, 0);

		imageWidth = FreeImage_GetWidth(tmp);
		imageHeight = FreeImage_GetHeight(tmp);
		imageBPP = FreeImage_GetBPP(tmp);
		imagePitch = FreeImage_GetPitch(tmp);		// FREEIMAGE align row data ... You have to use pitch instead of width

		std::cout << "Source: " << imageFileName << " (" << imageWidth << "x" << imageHeight << ", " << imageBPP << "bpp)" << std::endl;

		cudaDeviceSynchronize();
		checkCudaErrors(cudaMalloc((void**)&dImageData, imagePitch * imageHeight));
		cudaDeviceSynchronize();

		checkCudaErrors(cudaMemcpy2D(dImageData, imagePitch, FreeImage_GetBits(tmp), imagePitch, imageWidth * imageBPP / 8, imageHeight, cudaMemcpyHostToDevice));
		cudaThreadSynchronize();
		FreeImage_Unload(tmp);
		FreeImage_DeInitialise();
		printf("Source image has been loaded\n");
	}

	void loadPatternImage(const char* patternFileName)
	{
		FreeImage_Initialise();
		FIBITMAP* tmp = ImageManager::GenericLoader(patternFileName, 0);

		patternWidth = FreeImage_GetWidth(tmp);
		patternHeight = FreeImage_GetHeight(tmp);
		patternBPP = FreeImage_GetBPP(tmp);
		patternPitch = FreeImage_GetPitch(tmp);		// FREEIMAGE align row data ... You have to use pitch instead of width

		std::cout << "Pattern: " << patternFileName << " (" << patternWidth << "x" << patternHeight << ", " << patternBPP << "bpp)" << std::endl;

		cudaDeviceSynchronize();
		checkCudaErrors(cudaMalloc((void**)&dPatternData, patternPitch * patternHeight));
		cudaDeviceSynchronize();

		checkCudaErrors(cudaMemcpy2D(dPatternData, patternPitch, FreeImage_GetBits(tmp), patternPitch, patternWidth * patternBPP / 8, patternHeight, cudaMemcpyHostToDevice));
		cudaThreadSynchronize();
		FreeImage_Unload(tmp);
		FreeImage_DeInitialise();
		printf("Pattern image has been loaded\n");
	}

	__device__ float rotated_tex2D(const cudaTextureObject_t pattern, int x, int y, int width, int height, int rotation) {
		switch (rotation) {
		case 0: // No rotation
			return tex2D<float>(pattern, x, y);
		case 1: // 90 degrees
			return tex2D<float>(pattern, height - 1 - y, x);
		case 2: // 180 degrees
			return tex2D<float>(pattern, width - 1 - x, height - 1 - y);
		case 3: // 270 degrees
			return tex2D<float>(pattern, y, width - 1 - x);
		default:
			return 0.0f; // Should not happen
		}
	}

	__global__ void find_pattern(
		const cudaTextureObject_t reference,
		const cudaTextureObject_t pattern,
		const int patternWidth,
		const int patternHeight,
		const int imageWidth,
		const int imageHeight,
		uchar3* dst,
		int* matched_col,
		int* matched_row,
		int* matched_rotation
	)
	{
		unsigned int base_x_idx = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned int base_y_idx = blockIdx.y * blockDim.x + threadIdx.y;

		unsigned int skip_x = gridDim.x * blockDim.x;
		unsigned int skip_y = gridDim.y * blockDim.y;

		unsigned int offset_x = base_x_idx;
		unsigned int offset_y = base_y_idx;

		while (true) {
			for (int rotation = 0; rotation < 4; rotation++) {
				bool diff = false;

				for (int y = 0; y < patternHeight; y++) {
					for (int x = 0; x < patternWidth; x++) {
						float ref_value = tex2D<float>(reference, offset_x + x, offset_y + y);
						float pat_value = rotated_tex2D(pattern, x, y, patternWidth, patternHeight, rotation);

						if (ref_value != pat_value) {
							diff = true;
							break;
						}
					}

					if (diff) {
						break;
					}
				}

				if (!diff) {
					*matched_row = offset_x;
					*matched_col = offset_y;
					*matched_rotation = rotation;
					return;
				}
			}

			offset_x += skip_x;
			if (offset_x + patternWidth > imageWidth) {
				offset_x = base_x_idx;
				offset_y += skip_y;
			}

			if (offset_y + patternHeight > imageHeight) {
				break;
			}
		}
	}

	void run()
	{
		FreeImage_Initialise();

		loadSourceImage("./res/CreditTaskD_src.png");
		loadPatternImage("./res/CreditTaskD_pattern0.png");

		// load source
		ImageInfo<DT> src;
		prepareData<false>("./res/CreditTaskD_src.png", src);
		TextureInfo ti_src = createTextureObjectFrom2DArray(src);
		size_t dst_src_pitch;
		uchar3* dst_src;
		checkCudaErrors(cudaMallocPitch((void**)&dst_src, &dst_src_pitch, src.width * sizeof(uchar3), src.height));

		// load pattern 1
		ImageInfo<DT> pattern_1;
		prepareData<false>("./res/CreditTaskD_pattern0.png", pattern_1);
		TextureInfo ti_pattern_1 = createTextureObjectFrom2DArray(pattern_1);
		size_t dst_pattern_pitch_1;
		uchar3* dst_pattern_1;
		checkCudaErrors(cudaMallocPitch((void**)&dst_pattern_1, &dst_pattern_pitch_1, pattern_1.width * sizeof(uchar3), pattern_1.height));

		int* d_final_row_1 = nullptr;
		int* d_final_col_1 = nullptr;
		int* d_final_rotation_1 = nullptr;

		checkCudaErrors(cudaMalloc((void**)&d_final_row_1, sizeof(int)));
		checkCudaErrors(cudaMalloc((void**)&d_final_col_1, sizeof(int)));
		checkCudaErrors(cudaMalloc((void**)&d_final_rotation_1, sizeof(int)));

		dim3 dimBlock{ TPB_1D, TPB_1D, 1 };
		dim3 dimGrid{ (src.width + TPB_1D - 1) / TPB_1D, (src.height + TPB_1D - 1) / TPB_1D, 1 };

		find_pattern << <dimGrid, dimBlock >> > (
			ti_src.texObj,
			ti_pattern_1.texObj,
			pattern_1.width,
			pattern_1.height,
			src.width,
			src.height,
			dst_src,
			d_final_col_1,
			d_final_row_1,
			d_final_rotation_1);

		checkDeviceMatrix<int>((int*)d_final_row_1, sizeof(int), 1, 1, "%d ", "Row");
		checkDeviceMatrix<int>((int*)d_final_col_1, sizeof(int), 1, 1, "%d ", "Col");
		checkDeviceMatrix<int>((int*)d_final_rotation_1, sizeof(int), 1, 1, "%d ", "Rotation");

		// load pattern 2
		ImageInfo<DT> pattern_2;
		prepareData<false>("./res/CreditTaskD_pattern1.png", pattern_2);
		TextureInfo ti_pattern_2 = createTextureObjectFrom2DArray(pattern_2);
		size_t dst_pattern_pitch_2;
		uchar3* dst_pattern_2;
		checkCudaErrors(cudaMallocPitch((void**)&dst_pattern_2, &dst_pattern_pitch_2, pattern_2.width * sizeof(uchar3), pattern_2.height));

		int* d_final_row_2 = nullptr;
		int* d_final_col_2 = nullptr;
		int* d_final_rotation_2 = nullptr;

		checkCudaErrors(cudaMalloc((void**)&d_final_row_2, sizeof(int)));
		checkCudaErrors(cudaMalloc((void**)&d_final_col_2, sizeof(int)));
		checkCudaErrors(cudaMalloc((void**)&d_final_rotation_2, sizeof(int)));

		find_pattern << <dimGrid, dimBlock >> > (
			ti_src.texObj,
			ti_pattern_2.texObj,
			pattern_2.width,
			pattern_2.height,
			src.width,
			src.height,
			dst_src,
			d_final_col_2,
			d_final_row_2,
			d_final_rotation_2);

		checkDeviceMatrix<int>((int*)d_final_row_2, sizeof(int), 1, 1, "%d ", "Row");
		checkDeviceMatrix<int>((int*)d_final_col_2, sizeof(int), 1, 1, "%d ", "Col");
		checkDeviceMatrix<int>((int*)d_final_rotation_2, sizeof(int), 1, 1, "%d ", "Rotation");

		FreeImage_DeInitialise();
	}
} // namespace credit_task4