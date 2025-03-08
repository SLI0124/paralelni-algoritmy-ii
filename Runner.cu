#include <cudaDefs.h>
#include "lab01.h"
#include "lab02.h"
#include "lab03.h"

cudaError_t error = cudaSuccess;
cudaDeviceProp deviceProp = cudaDeviceProp();

int main(int argc, char** argv)
{
	initializeCUDA(deviceProp);
	//lab01::run();
	//lab02::run();
	lab03::run();
	return 0;
}