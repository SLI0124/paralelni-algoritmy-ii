#include <cudaDefs.h>
#include "lab01.h"
#include "lab02.h"

cudaError_t error = cudaSuccess;
cudaDeviceProp deviceProp = cudaDeviceProp();


int main(int argc, char** argv)
{
	initializeCUDA(deviceProp);
	//lab01::run();
	lab02::run();
	return 0;
}