#include <glew.h>
#include <freeglut.h>

#include <cudaDefs.h>

#include <cuda_gl_interop.h>
#include <helper_cuda.h>							// normalize method
#include <helper_math.h>							// normalize method

#include <benchmark.h>
#include <imageManager.h>

namespace credit_task3 {
#define TPB_1D 8									// ThreadsPerBlock in one dimension
#define TPB_2D TPB_1D*TPB_1D						// ThreadsPerBlock = TPB_1D*TPB_1D (2D block)

	cudaError_t error = cudaSuccess;
	cudaDeviceProp deviceProp = cudaDeviceProp();

	using DT = uchar4;

	//OpenGL
	struct GLData
	{
		unsigned int imageWidth;
		unsigned int imageHeight;
		unsigned int imageBPP;							//Bits Per Pixel = 8, 16, 24, or 32 bit
		unsigned int imagePitch;

		unsigned int pboID;
		unsigned int textureID;
		unsigned int viewportWidth = 1024;
		unsigned int viewportHeight = 1024;
	};
	GLData gl;

	bool reachedBorders = false;

	//CUDA
	struct CudaData
	{
		cudaTextureDesc			texDesc;				// Texture descriptor used to describe texture parameters

		cudaArray_t				texArrayData;			// Source texture data
		cudaResourceDesc		resDesc;				// A resource descriptor for obtaining the texture data
		cudaChannelFormatDesc	texChannelDesc;			// Texture channel descriptor to define channel bytes
		cudaTextureObject_t		texObj;					// Cuda Texture Object to be produces

		cudaGraphicsResource_t  texResource;
		cudaGraphicsResource_t	pboResource;

		CudaData()
		{
			memset(this, 0, sizeof(CudaData));			// DO NOT DELETE THIS !!!
		}
	};
	CudaData cd;


#pragma region CUDA Routines

	__device__ bool isWithinCircle(int delta_x, int delta_y, float radius) {
		return (delta_x * delta_x + delta_y * delta_y) <= (radius * radius);
	}

	__device__ bool isRed(const uchar4& texel) {
		return texel.x > texel.y && texel.x > texel.z;
	}

	__device__ bool isGreen(const uchar4& texel) {
		return texel.y > texel.x && texel.y > texel.z;
	}

	__device__ bool hasRedNeighbor(const cudaTextureObject_t srcTex, int x, int y, int pboWidth, int pboHeight, float radius) {
		for (int delta_y = -ceil(radius); delta_y <= ceil(radius); delta_y++) {
			for (int delta_x = -ceil(radius); delta_x <= ceil(radius); delta_x++) {
				if (isWithinCircle(delta_x, delta_y, radius)) {
					int new_x = x + delta_x;
					int new_y = y + delta_y;

					// Boundary check
					if (new_x >= 0 && new_y >= 0 && new_x < pboWidth && new_y < pboHeight) {
						const uchar4 neighborTexel = tex2D<DT>(srcTex, new_x + 0.5f, new_y + 0.5f);
						if (isRed(neighborTexel)) {
							return true;
						}
					}
				}
			}
		}
		return false;
	}

	__global__ void applyFilter(const cudaTextureObject_t srcTex, bool* reachedBorders, const unsigned int pboWidth, const unsigned int pboHeight, unsigned char* pbo) {
		const uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
		const uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

		unsigned int elementOffset = (y * pboWidth + x) * 4;

		// Ensure that we're within bounds
		if (x >= pboWidth || y >= pboHeight) return;

		const uchar4 texel = tex2D<DT>(srcTex, x + 0.5f, y + 0.5f); // Use float coordinates
		const float radius = 3.0f; // the radius of the circle, the higer, the better results but higher time to compute

		if (texel.x == 0 && texel.y == 0 && texel.z == 0) { // is it BLACK?
			if (hasRedNeighbor(srcTex, x, y, pboWidth, pboHeight, radius)) {
				pbo[elementOffset++] = 255; // Red
				pbo[elementOffset++] = 0;
				pbo[elementOffset++] = 0;
				return;
			}
		}
		else {
			if (isGreen(texel)) { // is it GREEN?
				if (hasRedNeighbor(srcTex, x, y, pboWidth, pboHeight, radius)) {
					printf("GREEN at (%d, %d): (%d, %d, %d)\n", x, y, texel.x, texel.y, texel.z);
					*reachedBorders = true;
				}
			}

			// Copy original texel to PBO
			pbo[elementOffset++] = texel.x;
			pbo[elementOffset++] = texel.y;
			pbo[elementOffset++] = texel.z;
		}
	}



	void cudaWorker()
	{
		// TODO: Map GL resources (TEXTURE and PBO)
		checkCudaErrors(cudaGraphicsMapResources(1, &cd.texResource, 0));
		checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&cd.texArrayData, cd.texResource, 0, 0));
		checkCudaErrors(cudaGraphicsMapResources(1, &cd.pboResource, 0));

		uint8_t* pboData = nullptr;
		size_t pboSize = 0;

		cudaGraphicsResourceGetMappedPointer((void**)&pboData, &pboSize, cd.pboResource);

		// TODO: Run kernel
		dim3 dimBlock{ TPB_1D , TPB_1D  , 1 };
		dim3 dimGrid{ (gl.imageWidth + TPB_1D - 1) / TPB_1D , (gl.imageHeight + TPB_1D - 1) / TPB_1D , 1 };

		bool* dReachedBorders;
		checkCudaErrors(cudaMalloc(&dReachedBorders, sizeof(bool)));
		checkCudaErrors(cudaMemcpy(dReachedBorders, &reachedBorders, sizeof(bool), cudaMemcpyHostToDevice));

		if (!reachedBorders)
		{
			applyFilter << <dimGrid, dimBlock >> > (cd.texObj, dReachedBorders, gl.imageWidth, gl.imageHeight, pboData);
		}

		checkCudaErrors(cudaMemcpy(&reachedBorders, dReachedBorders, sizeof(bool), cudaMemcpyDeviceToHost));

		// TODO: Unmap GL Resources (TEXTURE + PBO)
		checkCudaErrors(cudaGraphicsUnmapResources(1, &cd.pboResource, 0));
		checkCudaErrors(cudaGraphicsUnmapResources(1, &cd.texResource, 0));


		// This updates GL texture from PBO
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, gl.pboID);
		glBindTexture(GL_TEXTURE_2D, gl.textureID);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, gl.imageWidth, gl.imageHeight, GL_RGBA, GL_UNSIGNED_BYTE, NULL);   //Source parameter is NULL, Data is coming from a PBO, not host memory
	}

	void initCUDAObjects()
	{

		checkCudaErrors(cudaGraphicsGLRegisterImage(&cd.texResource, gl.textureID, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly));

		checkCudaErrors(cudaGraphicsMapResources(1, &cd.texResource));
		checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&cd.texArrayData, cd.texResource, 0, 0));

		cd.resDesc.resType = cudaResourceTypeArray;
		cd.resDesc.res.array.array = cd.texArrayData;

		cd.texDesc.readMode = cudaReadModeElementType;
		cd.texDesc.normalizedCoords = false;
		cd.texDesc.filterMode = cudaFilterModePoint;
		cd.texDesc.addressMode[0] = cudaAddressModeClamp;
		cd.texDesc.addressMode[1] = cudaAddressModeClamp;

		checkCudaErrors(cudaGetChannelDesc(&cd.texChannelDesc, cd.texArrayData));

		checkCudaErrors(cudaCreateTextureObject(&cd.texObj, &cd.resDesc, &cd.texDesc, NULL));

		checkCudaErrors(cudaGraphicsUnmapResources(1, &cd.texResource, 0));

		checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cd.pboResource, gl.pboID, cudaGraphicsRegisterFlagsWriteDiscard));
	}

	void releaseCUDA()
	{
		checkCudaErrors(cudaGraphicsUnregisterResource(cd.texResource));
		checkCudaErrors(cudaGraphicsUnregisterResource(cd.pboResource));
	}
#pragma endregion

#pragma region OpenGL Routines
	void prepareGlObjects(const char* imageFileName)
	{
		FIBITMAP* tmp = ImageManager::GenericLoader(imageFileName, 0);
		gl.imageWidth = FreeImage_GetWidth(tmp);
		gl.imageHeight = FreeImage_GetHeight(tmp);
		gl.imageBPP = FreeImage_GetBPP(tmp);
		gl.imagePitch = FreeImage_GetPitch(tmp);

		//OpenGL Texture
		glEnable(GL_TEXTURE_2D);
		glGenTextures(1, &gl.textureID);
		glBindTexture(GL_TEXTURE_2D, gl.textureID);

		//WARNING: Just some of inner format are supported by CUDA!!!
		// BIGGER WARNING: the map has bit depth only 24 bits, so GL_RGBA should be GL_RGB, GL_BGRA should be GL_BGR
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, gl.imageWidth, gl.imageHeight, 0, GL_BGR, GL_UNSIGNED_BYTE, FreeImage_GetBits(tmp));
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

		FreeImage_Unload(tmp);

		glGenBuffers(1, &gl.pboID);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, gl.pboID);														// Make this the current UNPACK buffer (OpenGL is state-based)
		// also 3 channels, because we are using GL_RGB
		glBufferData(GL_PIXEL_UNPACK_BUFFER, gl.imageWidth * gl.imageHeight * 4, NULL, GL_DYNAMIC_COPY);	// Allocate data for the buffer. 4-channel 8-bit image
	}

	void my_display()
	{
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glEnable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, gl.textureID);

		glBegin(GL_QUADS);

		glTexCoord2d(0, 0);		glVertex2d(0, 0);
		glTexCoord2d(1, 0);		glVertex2d(gl.viewportWidth, 0);
		glTexCoord2d(1, 1);		glVertex2d(gl.viewportWidth, gl.viewportHeight);
		glTexCoord2d(0, 1);		glVertex2d(0, gl.viewportHeight);

		glEnd();

		glDisable(GL_TEXTURE_2D);

		glFlush();
		glutSwapBuffers();
	}

	void my_resize(GLsizei w, GLsizei h)
	{
		gl.viewportWidth = w;
		gl.viewportHeight = h;

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glViewport(0, 0, gl.viewportWidth, gl.viewportHeight);

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		gluOrtho2D(0, gl.viewportWidth, 0, gl.viewportHeight);

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		glutPostRedisplay();
	}

	void my_idle()
	{
		cudaWorker();
		glutPostRedisplay();
	}

	void initGL(int argc, char** argv)
	{
		glutInit(&argc, argv);

		glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE);
		glutInitWindowSize(gl.viewportWidth, gl.viewportHeight);
		glutInitWindowPosition(0, 0);
		glutSetOption(GLUT_RENDERING_CONTEXT, false ? GLUT_USE_CURRENT_CONTEXT : GLUT_CREATE_NEW_CONTEXT);
		glutCreateWindow(0);

		char m_windowsTitle[512];
		sprintf_s(m_windowsTitle, 512, "SimpleView | context %s | renderer %s | vendor %s ",
			(const char*)glGetString(GL_VERSION),
			(const char*)glGetString(GL_RENDERER),
			(const char*)glGetString(GL_VENDOR));
		glutSetWindowTitle(m_windowsTitle);

		glutDisplayFunc(my_display);
		glutReshapeFunc(my_resize);
		glutIdleFunc(my_idle);
		glutSetCursor(GLUT_CURSOR_CROSSHAIR);

		// initialize necessary OpenGL extensions
		glewInit();

		glClearColor(0.0, 0.0, 0.0, 1.0);
		glShadeModel(GL_SMOOTH);
		glViewport(0, 0, gl.viewportWidth, gl.viewportHeight);

		glFlush();
	}

	void releaseOpenGL()
	{
		if (gl.textureID > 0)
			glDeleteTextures(1, &gl.textureID);
		if (gl.pboID > 0)
			glDeleteBuffers(1, &gl.pboID);
	}
#pragma endregion OpenGL Routines

	void releaseResources()
	{
		releaseCUDA();
		releaseOpenGL();
	}

	int run(int argc, char* argv[])
	{
		initializeCUDA(deviceProp);
		FreeImage_Initialise();

		initGL(argc, argv);
		prepareGlObjects("./res/map.png");

		initCUDAObjects();

		//start rendering mainloop
		glutMainLoop();
		FreeImage_DeInitialise();
		atexit(releaseResources);

		return 0;
	}
}