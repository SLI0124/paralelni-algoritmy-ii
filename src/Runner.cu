#include <cudaDefs.h>

#include "lab01.h"
#include "lab02.h"
#include "lab03.h"
#include "lab04.h"
#include "lab05.h"

#include "credit_task_1.h"

cudaError_t error = cudaSuccess;
cudaDeviceProp deviceProp = cudaDeviceProp();

int main(int argc, char** argv)
{
	initializeCUDA(deviceProp);
	//lab01::run();
	//lab02::run();
	//lab03::run();
	//lab04::run();
	//lab05::run();

	credit_task1::run();
	return 0;
}