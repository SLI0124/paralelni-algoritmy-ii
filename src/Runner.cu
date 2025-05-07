#include <cudaDefs.h>

#include "lab01.h"
#include "lab02.h"
#include "lab03.h"
#include "lab04.h"
#include "lab05.h"
#include "lab06.h"
#include "lab07.h"
#include "lab08.h"
#include "lab09.h"
#include "lab10.h"

#include "credit_task_1.h"
#include "credit_task3.h"
#include "credit_task4.h"

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
	//lab06::run();
	//lab07::run(argc, argv);
	//lab08::run();
	//lab09::run();
	//lab10::run();

	//credit_task1::run();
	//credit_task3::run(argc, argv);
	credit_task4::run();

	return 0;
}