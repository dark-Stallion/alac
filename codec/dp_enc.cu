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

#define SIZE 512

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

__global__ void gpu_init_coefs(int16_t * coefs, int32_t numPairs)
{
	int i = threadIdx.x;
	if (i > 2 && i < numPairs)
		coefs[i] = 0;
}

void init_coefs( int16_t * coefs, uint32_t denshift, int32_t numPairs )
{
	int32_t		k;
	int32_t		den = 1 << denshift;

	coefs[0] = (AINIT * den) >> 4;
	coefs[1] = (BINIT * den) >> 4;
	coefs[2] = (CINIT * den) >> 4;

/*	int16_t *d_coefs;

	cudaMalloc(&d_coefs, numPairs * 4);

	cudaMemcpy(d_coefs, coefs, numPairs * 4, cudaMemcpyHostToDevice);

	gpu_init_coefs<<<1, numPairs>>>(d_coefs, numPairs);

	cudaMemcpy(coefs, d_coefs, numPairs * 4, cudaMemcpyDeviceToHost);

	cudaFree(d_coefs);*/

	for ( k = 3; k < numPairs; k++ )
		coefs[k]  = 0;
}

//void copy_coefs( int16_t * srcCoefs, int16_t * dstCoefs, int32_t numPairs )
//{
//	int32_t k;
////	printf("\ncopy_coefs %d\n", numPairs);
//	for ( k = 0; k < numPairs; k++ )
//		dstCoefs[k] = srcCoefs[k];
//}

static inline int32_t ALWAYS_INLINE sign_of_int( int32_t i )		// <------- use in parallel method
{
    int32_t negishift;
	
    negishift = ((uint32_t)-i) >> 31;
    return negishift | (i >> 31);
}

/*__global__ void gpu_pc_block(int32_t *pin, int32_t *in, int32_t *pc1, int32_t lim, int32_t num, int32_t top, int32_t b0, int32_t b1, int32_t b2,
	int32_t b3, int32_t sum1, int32_t denhalf, uint32_t denshift, int16_t *a0, int16_t *a1, int16_t *a2, int16_t *a3, int32_t del, int32_t sg, int32_t del0, uint32_t chanshift, int32_t sgn)
{

	int z = threadIdx.x + blockIdx.x * blockDim.x; 
	if (z >= lim && z < num)
	{
		top = in[z - lim];
		pin = in + z - 1;

		b0 = top - pin[0];
		b1 = top - pin[-1];
		b2 = top - pin[-2];
		b3 = top - pin[-3];

		sum1 = (denhalf - *a0 * b0 - *a1 * b1 - *a2 * b2 - *a3 * b3) >> denshift;

		del = in[z] - top - sum1;
		del = (del << chanshift) >> chanshift;
		pc1[z] = del;
		del0 = del;

		sg = (del<0)?-1:(del>0)?1:0;
		if (sg > 0)
		{
			sgn = (b3<0) ? -1 : (b3>0) ? 1 : 0;
			atomicSub(a3, sgn);
			del0 -= (4 - 3) * ((sgn * b3) >> denshift);
			if (del0 <= 0)
				return;

			sgn = (b2<0) ? -1 : (b2>0) ? 1 : 0;
			atomicSub(a2, sgn);
			del0 -= (4 - 2) * ((sgn * b2) >> denshift);
			if (del0 <= 0)
				return;

			sgn = (b1<0) ? -1 : (b1>0) ? 1 : 0;
			atomicSub(a3, sgn);
			del0 -= (4 - 1) * ((sgn * b1) >> denshift);
			if (del0 <= 0)
				return;

			sgn = (b0<0) ? -1 : (b0>0) ? 1 : 0;
			atomicSub(a0, sgn);
		}
		else if (sg < 0)
		{
			// note: to avoid unnecessary negations, we flip the value of "sgn"
			sgn = -((b3<0) ? -1 : (b3>0) ? 1 : 0);
			atomicSub(a3, sgn);
			del0 -= (4 - 3) * ((sgn * b3) >> denshift);
			if (del0 >= 0)
				return;

			sgn = -((b2<0) ? -1 : (b2>0) ? 1 : 0);
			atomicSub(a2, sgn);
			del0 -= (4 - 2) * ((sgn * b2) >> denshift);
			if (del0 >= 0)
				return;

			sgn = -((b1<0) ? -1 : (b1>0) ? 1 : 0);
			atomicSub(a1, sgn);
			del0 -= (4 - 1) * ((sgn * b1) >> denshift);
			if (del0 >= 0)
				return;

			sgn = (b0<0) ? -1 : (b0>0) ? 1 : 0;
			atomicAdd(a0, sgn);
		}
	}
}*/

void pc_block(int32_t * in, int32_t * pc1, int32_t num, int16_t * coefs, int32_t numactive, uint32_t chanbits, uint32_t denshift)
{
	register int16_t	a0, a1, a2, a3;
	register int32_t	b0, b1, b2, b3;		//<-----used in parallel
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
//		printf("\npc_block (numactive == 31) %d\n", num);
		// short-circuit if numactive == 31
		for( j = 1; j < num; j++ )
		{
			del = in[j] - in[j-1];
			pc1[j] = (del << chanshift) >> chanshift;
		}
		return;
	}
//	printf("\npc_block %d\n", numactive);
	for ( j = 1; j <= numactive; j++ )
	{
		del = in[j] - in[j-1];
		pc1[j] = (del << chanshift) >> chanshift;
	}

	lim = numactive + 1;

	if ( numactive == 4 )
	{
		// optimization for numactive == 4
		a0 = coefs[0];
		a1 = coefs[1];
		a2 = coefs[2];
		a3 = coefs[3];
////		printf("\npc_block (numactive == 4) %d %d\n", lim, num);
//
//
//		int32_t *d_pin, *d_in, *d_pc1;
//		int16_t	*d_a0, *d_a1, *d_a2, *d_a3;
//
//		cudaMalloc(&d_pin, sizeof(int32_t));
//		cudaMalloc(&d_in, num * sizeof(int32_t));
//		cudaMalloc(&d_pc1, num * sizeof(int32_t));
//
//		cudaMalloc((void**)&d_a0, sizeof(int16_t));
//		cudaMalloc((void**)&d_a1, sizeof(int16_t));
//		cudaMalloc((void**)&d_a2, sizeof(int16_t));
//		cudaMalloc((void**)&d_a3, sizeof(int16_t));
//
//		cudaMemcpy(d_in, in, num * sizeof(int32_t), cudaMemcpyHostToDevice);
//		cudaMemcpy(d_pc1, pc1, num * sizeof(int32_t), cudaMemcpyHostToDevice);
//
//		cudaMemcpy(d_a0, &a0, sizeof(int16_t), cudaMemcpyHostToDevice);
//		cudaMemcpy(d_a1, &a1, sizeof(int16_t), cudaMemcpyHostToDevice);
//		cudaMemcpy(d_a2, &a2, sizeof(int16_t), cudaMemcpyHostToDevice);
//		cudaMemcpy(d_a3, &a3, sizeof(int16_t), cudaMemcpyHostToDevice);
//
//		gpu_pc_block<<<(num + SIZE - 1) / SIZE, SIZE >>>(d_pin, d_in, d_pc1, lim, num, top, b0, b1, b2, b3, sum1, denhalf, denshift,
//			d_a0, d_a1, d_a2, d_a3, del, sg, del0, chanshift, sgn);

		for ( j = lim; j < num; j++ )			// <---------------------- make parallel
		{
//			LOOP_ALIGN

			top = in[j - lim];
			pin = in + j - 1;

			b0 = top - pin[0];
			b1 = top - pin[-1];
			b2 = top - pin[-2];
			b3 = top - pin[-3];

			sum1 = (denhalf - a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3) >> denshift;

			del = in[j] - top - sum1;
			del = (del << chanshift) >> chanshift;
			pc1[j] = del;	     
			del0 = del;

			sg = sign_of_int(del);
			if ( sg > 0 )
			{
				sgn = sign_of_int( b3 );
				a3 -= sgn;
				del0 -= (4 - 3) * ((sgn * b3) >> denshift);
				if ( del0 <= 0 )
					continue;
				
				sgn = sign_of_int( b2 );
				a2 -= sgn;
				del0 -= (4 - 2) * ((sgn * b2) >> denshift);
				if ( del0 <= 0 )
					continue;
				
				sgn = sign_of_int( b1 );
				a1 -= sgn;
				del0 -= (4 - 1) * ((sgn * b1) >> denshift);
				if ( del0 <= 0 )
					continue;

				a0 -= sign_of_int( b0 );
			}
			else if ( sg < 0 )
			{
				// note: to avoid unnecessary negations, we flip the value of "sgn"
				sgn = -sign_of_int( b3 );
				a3 -= sgn;
				del0 -= (4 - 3) * ((sgn * b3) >> denshift);
				if ( del0 >= 0 )
					continue;
				
				sgn = -sign_of_int( b2 );
				a2 -= sgn;
				del0 -= (4 - 2) * ((sgn * b2) >> denshift);
				if ( del0 >= 0 )
					continue;
				
				sgn = -sign_of_int( b1 );
				a1 -= sgn;
				del0 -= (4 - 1) * ((sgn * b1) >> denshift);
				if ( del0 >= 0 )
					continue;

				a0 += sign_of_int( b0 );
			}
		}

		/*cudaMemcpy(&a0, d_a0, sizeof(int16_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(&a1, d_a1, sizeof(int16_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(&a2, d_a2, sizeof(int16_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(&a3, d_a3, sizeof(int16_t), cudaMemcpyDeviceToHost);

		cudaFree(d_pin);
		cudaFree(d_in);
		cudaFree(d_pc1);

		cudaFree(d_a0);
		cudaFree(d_a1);
		cudaFree(d_a2);
		cudaFree(d_a3);*/

		coefs[0] = a0;
		coefs[1] = a1;
		coefs[2] = a2;
		coefs[3] = a3;
	}
	else if ( numactive == 8 )
	{
		// optimization for numactive == 8
		register int16_t	a4, a5, a6, a7;
		register int32_t	b4, b5, b6, b7;

		a0 = coefs[0];
		a1 = coefs[1];
		a2 = coefs[2];
		a3 = coefs[3];
		a4 = coefs[4];
		a5 = coefs[5];
		a6 = coefs[6];
		a7 = coefs[7];
//		printf("\npc_block (numactive == 8) %d %d\n", lim, num);
		for ( j = lim; j < num; j++ )			// <---------------------- make parallel
		{
			LOOP_ALIGN

			top = in[j - lim];
			pin = in + j - 1;

			b0 = top - (*pin--);
			b1 = top - (*pin--);
			b2 = top - (*pin--);
			b3 = top - (*pin--);
			b4 = top - (*pin--);
			b5 = top - (*pin--);
			b6 = top - (*pin--);
			b7 = top - (*pin);
			pin += 8;

			sum1 = (denhalf - a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3
					- a4 * b4 - a5 * b5 - a6 * b6 - a7 * b7) >> denshift;

			del = in[j] - top - sum1;
			del = (del << chanshift) >> chanshift;
			pc1[j] = del;	     
			del0 = del;

			sg = sign_of_int(del);
			if ( sg > 0 )
			{
				sgn = sign_of_int( b7 );
				a7 -= sgn;
				del0 -= 1 * ((sgn * b7) >> denshift);
				if ( del0 <= 0 )
					continue;
				
				sgn = sign_of_int( b6 );
				a6 -= sgn;
				del0 -= 2 * ((sgn * b6) >> denshift);
				if ( del0 <= 0 )
					continue;
				
				sgn = sign_of_int( b5 );
				a5 -= sgn;
				del0 -= 3 * ((sgn * b5) >> denshift);
				if ( del0 <= 0 )
					continue;

				sgn = sign_of_int( b4 );
				a4 -= sgn;
				del0 -= 4 * ((sgn * b4) >> denshift);
				if ( del0 <= 0 )
					continue;
				
				sgn = sign_of_int( b3 );
				a3 -= sgn;
				del0 -= 5 * ((sgn * b3) >> denshift);
				if ( del0 <= 0 )
					continue;
				
				sgn = sign_of_int( b2 );
				a2 -= sgn;
				del0 -= 6 * ((sgn * b2) >> denshift);
				if ( del0 <= 0 )
					continue;
				
				sgn = sign_of_int( b1 );
				a1 -= sgn;
				del0 -= 7 * ((sgn * b1) >> denshift);
				if ( del0 <= 0 )
					continue;

				a0 -= sign_of_int( b0 );
			}
			else if ( sg < 0 )
			{
				// note: to avoid unnecessary negations, we flip the value of "sgn"
				sgn = -sign_of_int( b7 );
				a7 -= sgn;
				del0 -= 1 * ((sgn * b7) >> denshift);
				if ( del0 >= 0 )
					continue;
				
				sgn = -sign_of_int( b6 );
				a6 -= sgn;
				del0 -= 2 * ((sgn * b6) >> denshift);
				if ( del0 >= 0 )
					continue;
				
				sgn = -sign_of_int( b5 );
				a5 -= sgn;
				del0 -= 3 * ((sgn * b5) >> denshift);
				if ( del0 >= 0 )
					continue;

				sgn = -sign_of_int( b4 );
				a4 -= sgn;
				del0 -= 4 * ((sgn * b4) >> denshift);
				if ( del0 >= 0 )
					continue;
				
				sgn = -sign_of_int( b3 );
				a3 -= sgn;
				del0 -= 5 * ((sgn * b3) >> denshift);
				if ( del0 >= 0 )
					continue;
				
				sgn = -sign_of_int( b2 );
				a2 -= sgn;
				del0 -= 6 * ((sgn * b2) >> denshift);
				if ( del0 >= 0 )
					continue;
				
				sgn = -sign_of_int( b1 );
				a1 -= sgn;
				del0 -= 7 * ((sgn * b1) >> denshift);
				if ( del0 >= 0 )
					continue;

				a0 += sign_of_int( b0 );
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
	else
	{
//pc_block_general:
		// general case
//		printf("\npc_block general %d %d\n", lim, num);
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

			sg = sign_of_int( del );
			if ( sg > 0 )
			{
				for ( k = (numactive - 1); k >= 0; k-- )
				{
					dd = top - pin[-k];
					sgn = sign_of_int( dd );
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
					sgn = sign_of_int( dd );
					coefs[k] += sgn;
					del0 -= (numactive - k) * ((-sgn * dd) >> denshift);
					if ( del0 >= 0 )
						break;
				}
			}
		}
	}
}
