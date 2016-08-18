#include <stdint.h>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "CudaAlacEncoder.cuh"

#define SIZE 1024
#define DENSHIFT_DEFAULT 9
#define AINIT 38
#define BINIT (-29)
#define CINIT (-2)


__global__ void call_kALACSearch(int16_t * mCoefsU, int16_t * mCoefsV, int32_t kALACMaxCoefs)
{
	int x = blockIdx.x;
	int y = threadIdx.x;

	int index = x * 16 * 16 + y * 16;
	int32_t		k;
	int32_t		den = 1 << DENSHIFT_DEFAULT;

	mCoefsU[index + 0] = (AINIT * den) >> 4;
	mCoefsU[index + 1] = (BINIT * den) >> 4;
	mCoefsU[index + 2] = (CINIT * den) >> 4;

	mCoefsV[index + 0] = (AINIT * den) >> 4;
	mCoefsV[index + 1] = (BINIT * den) >> 4;
	mCoefsV[index + 2] = (CINIT * den) >> 4;

	for (k = 3; k < kALACMaxCoefs; k++)
	{
		mCoefsU[index + k] = 0;
		mCoefsV[index + k] = 0;
	}
}

void kALACSearch(void  *p1, void *p2, int32_t numPairs, int32_t mNumChannels, int32_t kALACMaxSearches){


	int16_t *d_mCoefsU, *d_mCoefsV;

	cudaMalloc(&d_mCoefsU, sizeof(int16_t) * 8 * 16 * 16);
	cudaMalloc(&d_mCoefsV, sizeof(int16_t) * 8 * 16 * 16);

	cudaMemcpy(d_mCoefsU, p1, sizeof(int16_t) * 8 * 16 * 16, cudaMemcpyHostToDevice);
	cudaMemcpy(d_mCoefsV, p2, sizeof(int16_t) * 8 * 16 * 16, cudaMemcpyHostToDevice);

	float elapsedTime;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	call_kALACSearch << < mNumChannels, kALACMaxSearches >> >(d_mCoefsU, d_mCoefsV, numPairs);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("GPU Time elapsed: %f ms\n", elapsedTime);

	cudaMemcpy(p1, d_mCoefsU, sizeof(int16_t) * 8 * 16 * 16, cudaMemcpyDeviceToHost);
	cudaMemcpy(p2, d_mCoefsV, sizeof(int16_t) * 8 * 16 * 16, cudaMemcpyDeviceToHost);

	cudaFree(d_mCoefsU);
	cudaFree(d_mCoefsV);
}
