/*
 * Copyright (c) 2011 Apple Inc. All rights reserved.
 *
 * @APPLE_APACHE_LICENSE_HEADER_START@
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 * @APPLE_APACHE_LICENSE_HEADER_END@
 */

/*
	File:		dp_enc.c

	Contains:	Dynamic Predictor encode routines

	Copyright:	(c) 2001-2011 Apple, Inc.
*/

#include <stdio.h>
#include <stdlib.h>

#include "dplib.h"
#include <string.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#define SIZE 1024

#if __GNUC__
#define ALWAYS_INLINE		__attribute__((always_inline))
#else
#define ALWAYS_INLINE
#endif

#if TARGET_CPU_PPC && (__MWERKS__ >= 0x3200)
// align loops to a 16 byte boundary to make the G5 happy
#pragma function_align 16
#define LOOP_ALIGN			asm { align 16 }
#else
#define LOOP_ALIGN
#endif

__global__ void dynamic_init_coefs(int16_t * mCoefsU, int16_t * mCoefsV, int g){
	int index = threadIdx.x;

	if (index > 2)
	{
		mCoefsU[index + g] = 0;
		mCoefsV[index + g] = 0;
	}
}
__global__ void gpu_init_coefs(int16_t * mCoefsU, int16_t * mCoefsV, int32_t kALACMaxCoefs)
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

	dynamic_init_coefs << <1, kALACMaxCoefs >> >(mCoefsU, mCoefsV, index);

	cudaDeviceSynchronize();

	/*for (k = 3; k < kALACMaxCoefs; k++)
	{
	mCoefsU[index + k] = 0;
	mCoefsV[index + k] = 0;
	}*/
}

void init_coefs(int16_t * mCoefsU, int16_t * mCoefsV, int32_t kALACMaxCoefs, int32_t mNumChannels, int32_t kALACMaxSearches)
{
	gpu_init_coefs << < mNumChannels, kALACMaxSearches >> >(mCoefsU, mCoefsV, kALACMaxCoefs);
}

void copy_coefs( int16_t * srcCoefs, int16_t * dstCoefs, int32_t numPairs )
{
	int32_t k;
//	printf("\ncopy_coefs %d\n", numPairs);
	for ( k = 0; k < numPairs; k++ )
		dstCoefs[k] = srcCoefs[k];
}

static inline int32_t ALWAYS_INLINE sign_of_int( int32_t i )		// <------- use in parallel method
{
    int32_t negishift;
	
    negishift = ((uint32_t)-i) >> 31;
    return negishift | (i >> 31);
}

__global__ void gpu_pc_block_4(int32_t *in, int32_t *pc1, int32_t lim, int32_t num, int32_t sum1, int32_t denhalf,
	uint32_t denshift, int16_t * coefs, int32_t del, int32_t sg, int32_t del0, uint32_t chanshift, int32_t sgn, int g, int limit)
{

	int z = threadIdx.x + g * blockDim.x; 
	if (z >= lim && z < num)
	{
		LOOP_ALIGN

		__shared__ int s1[1024];
		__shared__ int s2[1024];
		__shared__ int s3[1024];
		__shared__ int s4[1024];
		__shared__ int s5[1024];

		s5[threadIdx.x] = in[z - lim];
		int32_t * pin = in + z - 1;

		s1[threadIdx.x] = s5[threadIdx.x] - pin[0];
		s2[threadIdx.x] = s5[threadIdx.x] - pin[-1];
		s3[threadIdx.x] = s5[threadIdx.x] - pin[-2];
		s4[threadIdx.x] = s5[threadIdx.x] - pin[-3];

		__syncthreads();

		if (threadIdx.x == lim){

			register int16_t a0, a1, a2, a3;

			a0 = coefs[0];
			a1 = coefs[1];
			a2 = coefs[2];
			a3 = coefs[3];

			int n;
			if (g != 0)
				lim = 0;
			if (limit < SIZE)
				n = limit;
			else
				n = SIZE;

			for (int j = lim; j < n; j++){
				int index = j + g * blockDim.x;
				sum1 = (denhalf - a0 * s1[j] - a1 * s2[j] - a2 * s3[j] - a3 * s4[j]) >> denshift;
				del = in[index] - s5[j] - sum1;
				del = (del << chanshift) >> chanshift;
				pc1[index] = del;
				del0 = del;
				sg = (del < 0) ? -1 : (del>0) ? 1 : 0;
				if (sg > 0)
				{
					sgn = (s4[j] < 0) ? -1 : (s4[j]>0) ? 1 : 0;
					a3 -= sgn;
					del0 -= (4 - 3) * ((sgn * s4[j]) >> denshift);
					if (del0 <= 0)
						continue;

					sgn = (s3[j] < 0) ? -1 : (s3[j]>0) ? 1 : 0;
					a2 -= sgn;
					del0 -= (4 - 2) * ((sgn * s3[j]) >> denshift);
					if (del0 <= 0)
						continue;

					sgn = (s2[j] < 0) ? -1 : (s2[j]>0) ? 1 : 0;
					a1 -= sgn;
					del0 -= (4 - 1) * ((sgn * s2[j]) >> denshift);
					if (del0 <= 0)
						continue;
					
					sgn = (s1[j] < 0) ? -1 : (s1[j]>0) ? 1 : 0;
					a0 -= sgn;
				}
				else if (sg < 0)
				{
					// note: to avoid unnecessary negations, we flip the value of "sgn"
					sgn = -((s4[j] < 0) ? -1 : (s4[j]>0) ? 1 : 0);
					a3 -= sgn;
					del0 -= (4 - 3) * ((sgn * s4[j]) >> denshift);
					if (del0 >= 0)
						continue;

					sgn = -((s3[j] < 0) ? -1 : (s3[j]>0) ? 1 : 0);
					a2 -= sgn;
					del0 -= (4 - 2) * ((sgn * s3[j]) >> denshift);
					if (del0 >= 0)
						continue;

					sgn = -((s2[j] < 0) ? -1 : (s2[j]>0) ? 1 : 0);
					a1 -= sgn;
					del0 -= (4 - 1) * ((sgn * s2[j]) >> denshift);
					if (del0 >= 0)
						continue;

					sgn = (s1[j] < 0) ? -1 : (s1[j]>0) ? 1 : 0;
					a0 += sgn;
				}
			}
			coefs[0] = a0;
			coefs[1] = a1;
			coefs[2] = a2;
			coefs[3] = a3;
		}
	}
}

__global__ void gpu_pc_block_8(int32_t *in, int32_t *pc1, int32_t lim, int32_t num, int32_t sum1, int32_t denhalf,
	uint32_t denshift, int16_t * coefs, int32_t del, int32_t sg, int32_t del0, uint32_t chanshift, int32_t sgn, int g, int limit)
{
	
	int z = threadIdx.x + g * blockDim.x;
	if (z >= lim && z < num)
	{
		LOOP_ALIGN

		__shared__ int32_t s1[1024];
		__shared__ int32_t s2[1024];
		__shared__ int32_t s3[1024];
		__shared__ int32_t s4[1024];
		__shared__ int32_t s5[1024];
		__shared__ int32_t s6[1024];
		__shared__ int32_t s7[1024];
		__shared__ int32_t s8[1024];
		__shared__ int32_t s9[1024];

		s9[threadIdx.x] = in[z - lim];
		int32_t * pin = in + z - 1;

		s1[threadIdx.x] = s9[threadIdx.x] - (*(pin));
		s2[threadIdx.x] = s9[threadIdx.x] - (*(pin - 1));
		s3[threadIdx.x] = s9[threadIdx.x] - (*(pin - 2));
		s4[threadIdx.x] = s9[threadIdx.x] - (*(pin - 3));
		s5[threadIdx.x] = s9[threadIdx.x] - (*(pin - 4));
		s6[threadIdx.x] = s9[threadIdx.x] - (*(pin - 5));
		s7[threadIdx.x] = s9[threadIdx.x] - (*(pin - 6));
		s8[threadIdx.x] = s9[threadIdx.x] - (*(pin - 7));

		__syncthreads();

		if (threadIdx.x == lim){

			register int16_t a0, a1, a2, a3, a4, a5, a6, a7;

			a0 = coefs[0];
			a1 = coefs[1];
			a2 = coefs[2];
			a3 = coefs[3];
			a4 = coefs[4];
			a5 = coefs[5];
			a6 = coefs[6];
			a7 = coefs[7];

			int n;
			if (g != 0)
				lim = 0;
			if (limit < SIZE)
				n = limit;
			else
				n = SIZE;

			for (int j = lim; j < n; j++){

				int index = j + g * blockDim.x;

				sum1 = (denhalf - a0 * s1[j] - a1 * s2[j] - a2 * s3[j] - a3 * s4[j]
					- a4 * s5[j] - a5 * s6[j] - a6 * s7[j] - a7 * s8[j]) >> denshift;

				del = in[index] - s9[j] - sum1;
				del = (del << chanshift) >> chanshift;
				pc1[index] = del;
				del0 = del;

				sg = (del < 0) ? -1 : (del>0) ? 1 : 0;
				if (sg > 0)
				{
					sgn = (s8[j] < 0) ? -1 : (s8[j]>0) ? 1 : 0;
					a7 -= sgn;
					del0 -= 1 * ((sgn * s8[j]) >> denshift);
					if (del0 <= 0)
						continue;

					sgn = (s7[j] < 0) ? -1 : (s7[j]>0) ? 1 : 0;
					a6 -= sgn;
					del0 -= 2 * ((sgn * s7[j]) >> denshift);
					if (del0 <= 0)
						continue;

					sgn = (s6[j] < 0) ? -1 : (s6[j]>0) ? 1 : 0;
					a5 -= sgn;
					del0 -= 3 * ((sgn * s6[j]) >> denshift);
					if (del0 <= 0)
						continue;

					sgn = (s5[j] < 0) ? -1 : (s5[j]>0) ? 1 : 0;
					a4 -= sgn;
					del0 -= 4 * ((sgn * s5[j]) >> denshift);
					if (del0 <= 0)
						continue;

					sgn = (s4[j] < 0) ? -1 : (s4[j]>0) ? 1 : 0;
					a3 -= sgn;
					del0 -= 5 * ((sgn * s4[j]) >> denshift);
					if (del0 <= 0)
						continue;

					sgn = (s3[j] < 0) ? -1 : (s3[j]>0) ? 1 : 0;
					a2 -= sgn;
					del0 -= 6 * ((sgn * s3[j]) >> denshift);
					if (del0 <= 0)
						continue;

					sgn = (s2[j] < 0) ? -1 : (s2[j]>0) ? 1 : 0;
					a1 -= sgn;
					del0 -= 7 * ((sgn * s2[j]) >> denshift);
					if (del0 <= 0)
						continue;

					a0 -= (s1[j] < 0) ? -1 : (s1[j]>0) ? 1 : 0;
				}
				else if (sg < 0)
				{
					// note: to avoid unnecessary negations, we flip the value of "sgn"
					sgn = -((s8[j] < 0) ? -1 : (s8[j]>0) ? 1 : 0);
					a7 -= sgn;
					del0 -= 1 * ((sgn * s8[j]) >> denshift);
					if (del0 >= 0)
						continue;

					sgn = -((s7[j] < 0) ? -1 : (s7[j]>0) ? 1 : 0);
					a6 -= sgn;
					del0 -= 2 * ((sgn * s7[j]) >> denshift);
					if (del0 >= 0)
						continue;

					sgn = -((s6[j] < 0) ? -1 : (s6[j]>0) ? 1 : 0);
					a5 -= sgn;
					del0 -= 3 * ((sgn * s6[j]) >> denshift);
					if (del0 >= 0)
						continue;

					sgn = -((s5[j] < 0) ? -1 : (s5[j]>0) ? 1 : 0);
					a4 -= sgn;
					del0 -= 4 * ((sgn * s5[j]) >> denshift);
					if (del0 >= 0)
						continue;

					sgn = -((s4[j] < 0) ? -1 : (s4[j]>0) ? 1 : 0);
					a3 -= sgn;
					del0 -= 5 * ((sgn * s4[j]) >> denshift);
					if (del0 >= 0)
						continue;

					sgn = -((s3[j] < 0) ? -1 : (s3[j]>0) ? 1 : 0);
					a2 -= sgn;
					del0 -= 6 * ((sgn * s3[j]) >> denshift);
					if (del0 >= 0)
						continue;

					sgn = -((s2[j] < 0) ? -1 : (s2[j]>0) ? 1 : 0);
					a1 -= sgn;
					del0 -= 7 * ((sgn * s2[j]) >> denshift);
					if (del0 >= 0)
						continue;

					a0 += (s1[j] < 0) ? -1 : (s1[j]>0) ? 1 : 0;
				}
			}

			coefs[0] = a0;
			coefs[1] = a1;
			coefs[2] = a2;
			coefs[3] = a3;
			coefs[4] = a4;
			coefs[5] = a5;
			coefs[6] = a6;
			coefs[7] = a7;
		}
	}
}

__global__ void gpu_pc_block(int32_t * in, int32_t * pc1, int32_t num, int16_t * coefs, int32_t numactive, uint32_t chanbits, uint32_t denshift)
{
	/*register int16_t	a0, a1, a2, a3;
	register int32_t	b0, b1, b2, b3;*/		//<-----used in parallel
	int32_t					j, k, lim;
	int32_t *			pin;		//<-----used in parallel
	int32_t				sum1, dd;
	int32_t				sg, sgn;
	int32_t				top;		//<-----used in parallel
	int32_t				del, del0;
	uint32_t			chanshift = 32 - chanbits;
	int32_t				denhalf = 1 << (denshift - 1);		//<-----used in parallel

	pc1[0] = in[0];
	if ( numactive == 0 )
	{
		// just copy if numactive == 0 (but don't bother if in/out pointers the same)
		if ( (num > 1) && (in != pc1) )
			memcpy( &pc1[1], &in[1], (num - 1) * sizeof(int32_t) );
		return;
	}
	if ( numactive == 31 )
	{
		// short-circuit if numactive == 31
		for( j = 1; j < num; j++ )
		{
			del = in[j] - in[j-1];
			pc1[j] = (del << chanshift) >> chanshift;
		}
		return;
	}
	for ( j = 1; j <= numactive; j++ )
	{
		del = in[j] - in[j-1];
		pc1[j] = (del << chanshift) >> chanshift;
	}

	lim = numactive + 1;

	if ( numactive == 4 )
	{
		// optimization for numactive == 4
		/*a0 = coefs[0];
		a1 = coefs[1];
		a2 = coefs[2];
		a3 = coefs[3];*/

	/*	int32_t *d_in, *d_pc1;*/
		/*int16_t	*d_a0, *d_a1, *d_a2, *d_a3;*/

		/*cudaMalloc(&d_in, num * sizeof(int32_t));
		cudaMalloc(&d_pc1, num * sizeof(int32_t));*/

		/*cudaMalloc((void**)&d_a0, sizeof(int16_t));
		cudaMalloc((void**)&d_a1, sizeof(int16_t));
		cudaMalloc((void**)&d_a2, sizeof(int16_t));
		cudaMalloc((void**)&d_a3, sizeof(int16_t));*/

		/*cudaMemcpy(d_in, in, num * sizeof(int32_t), cudaMemcpyHostToDevice);
		cudaMemcpy(d_pc1, pc1, num * sizeof(int32_t), cudaMemcpyHostToDevice);*/

		/*cudaMemcpy(d_a0, &a0, sizeof(int16_t), cudaMemcpyHostToDevice);
		cudaMemcpy(d_a1, &a1, sizeof(int16_t), cudaMemcpyHostToDevice);
		cudaMemcpy(d_a2, &a2, sizeof(int16_t), cudaMemcpyHostToDevice);
		cudaMemcpy(d_a3, &a3, sizeof(int16_t), cudaMemcpyHostToDevice);*/
		
		int n = num;
		for (int i = 0; i < (num + SIZE - 1) / SIZE; i++){
			gpu_pc_block_4 << <1, SIZE >> >(in, pc1, lim, num, sum1, denhalf, denshift, coefs, del, sg, del0, chanshift, sgn, i, n);
			n -= 1024;
		}

		/*cudaMemcpy(&a0, d_a0, sizeof(int16_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(&a1, d_a1, sizeof(int16_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(&a2, d_a2, sizeof(int16_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(&a3, d_a3, sizeof(int16_t), cudaMemcpyDeviceToHost);*/

		/*cudaMemcpy(in, d_in, num * sizeof(int32_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(pc1, d_pc1, num * sizeof(int32_t), cudaMemcpyDeviceToHost);

		cudaFree(d_in);
		cudaFree(d_pc1);*/

		/*cudaFree(d_a0);
		cudaFree(d_a1);
		cudaFree(d_a2);
		cudaFree(d_a3);*/

		/*coefs[0] = a0;
		coefs[1] = a1;
		coefs[2] = a2;
		coefs[3] = a3;*/
	}
	else if ( numactive == 8 )
	{

		// optimization for numactive == 8
		/*register int16_t	a4, a5, a6, a7;
		register int32_t	b4, b5, b6, b7;*/

		/*a0 = coefs[0];
		a1 = coefs[1];
		a2 = coefs[2];
		a3 = coefs[3];
		a4 = coefs[4];
		a5 = coefs[5];
		a6 = coefs[6];
		a7 = coefs[7];*/


		/*int32_t *d_pin, *d_in, *d_pc1;
		int16_t	*d_a0, *d_a1, *d_a2, *d_a3, *d_a4, *d_a5, *d_a6, *d_a7;

		cudaMalloc(&d_in, num * sizeof(int32_t));
		cudaMalloc(&d_pc1, num * sizeof(int32_t));

		cudaMalloc((void**)&d_a0, sizeof(int16_t));
		cudaMalloc((void**)&d_a1, sizeof(int16_t));
		cudaMalloc((void**)&d_a2, sizeof(int16_t));
		cudaMalloc((void**)&d_a3, sizeof(int16_t));
		cudaMalloc((void**)&d_a4, sizeof(int16_t));
		cudaMalloc((void**)&d_a5, sizeof(int16_t));
		cudaMalloc((void**)&d_a6, sizeof(int16_t));
		cudaMalloc((void**)&d_a7, sizeof(int16_t));

		cudaMemcpy(d_in, in, num * sizeof(int32_t), cudaMemcpyHostToDevice);
		cudaMemcpy(d_pc1, pc1, num * sizeof(int32_t), cudaMemcpyHostToDevice);

		cudaMemcpy(d_a0, &a0, sizeof(int16_t), cudaMemcpyHostToDevice);
		cudaMemcpy(d_a1, &a1, sizeof(int16_t), cudaMemcpyHostToDevice);
		cudaMemcpy(d_a2, &a2, sizeof(int16_t), cudaMemcpyHostToDevice);
		cudaMemcpy(d_a3, &a3, sizeof(int16_t), cudaMemcpyHostToDevice);
		cudaMemcpy(d_a4, &a4, sizeof(int16_t), cudaMemcpyHostToDevice);
		cudaMemcpy(d_a5, &a5, sizeof(int16_t), cudaMemcpyHostToDevice);
		cudaMemcpy(d_a6, &a6, sizeof(int16_t), cudaMemcpyHostToDevice);
		cudaMemcpy(d_a7, &a7, sizeof(int16_t), cudaMemcpyHostToDevice);*/

		int n = num;
		for (int i = 0; i < (num + SIZE - 1) / SIZE; i++){
			gpu_pc_block_8 << <1, SIZE >> >(in, pc1, lim, num, sum1, denhalf, denshift, coefs, del, sg, del0, chanshift, sgn, i, n);
			n -= 1024;
		}

		/*cudaMemcpy(&a0, d_a0, sizeof(int16_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(&a1, d_a1, sizeof(int16_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(&a2, d_a2, sizeof(int16_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(&a3, d_a3, sizeof(int16_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(&a4, d_a4, sizeof(int16_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(&a5, d_a5, sizeof(int16_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(&a6, d_a6, sizeof(int16_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(&a7, d_a7, sizeof(int16_t), cudaMemcpyDeviceToHost);

		cudaMemcpy(in, d_in, num * sizeof(int32_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(pc1, d_pc1, num * sizeof(int32_t), cudaMemcpyDeviceToHost);

		cudaFree(d_in);
		cudaFree(d_pc1);

		cudaFree(d_a0);
		cudaFree(d_a1);
		cudaFree(d_a2);
		cudaFree(d_a3);
		cudaFree(d_a4);
		cudaFree(d_a5);
		cudaFree(d_a6);
		cudaFree(d_a7);*/

		/*coefs[0] = a0;
		coefs[1] = a1;
		coefs[2] = a2;
		coefs[3] = a3;
		coefs[4] = a4;
		coefs[5] = a5;
		coefs[6] = a6;
		coefs[7] = a7;*/
	}
	else
	{
//pc_block_general:
		// general case
		printf("\nENTERS pc_block else\n");
		return;
		for ( j = lim; j < num; j++ )
		{
			LOOP_ALIGN

			top = in[j - lim];
			pin = in + j - 1;

			sum1 = 0;
			for ( k = 0; k < numactive; k++ )
				sum1 -= coefs[k] * (top - pin[-k]);
		
			del = in[j] - top - ((sum1 + denhalf) >> denshift);
			del = (del << chanshift) >> chanshift;
			pc1[j] = del;
			del0 = del;

			sg = (del < 0) ? -1 : (del>0) ? 1 : 0;
			if ( sg > 0 )
			{
				for ( k = (numactive - 1); k >= 0; k-- )
				{
					dd = top - pin[-k];
					sgn = (dd < 0) ? -1 : (dd>0) ? 1 : 0;
					coefs[k] -= sgn;
					del0 -= (numactive - k) * ((sgn * dd) >> denshift);
					if ( del0 <= 0 )
						break;
				}
			}
			else if ( sg < 0 )
			{
				for ( k = (numactive - 1); k >= 0; k-- )
				{
					dd = top - pin[-k];
					sgn = (dd < 0) ? -1 : (dd>0) ? 1 : 0;
					coefs[k] += sgn;
					del0 -= (numactive - k) * ((-sgn * dd) >> denshift);
					if ( del0 >= 0 )
						break;
				}
			}
		}
	}
	cudaDeviceSynchronize();
}

void pc_block(int32_t * in, int32_t * pc1, int32_t num, int16_t * coefs, int32_t numactive, uint32_t chanbits, uint32_t denshift){
	gpu_pc_block << <1, 1 >> >(in, pc1, num, coefs, numactive, chanbits, denshift);
}