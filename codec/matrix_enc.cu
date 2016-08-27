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
File:		matrix_enc.c

Contains:	ALAC mixing/matrixing encode routines.

Copyright:	(c) 2004-2011 Apple, Inc.
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <cuda_profiler_api.h>

#include "matrixlib.h"
#include "ALACAudioTypes.h"

#define SIZE 1024
// up to 24-bit "offset" macros for the individual bytes of a 20/24-bit word
#if TARGET_RT_BIG_ENDIAN
#define LBYTE	2
#define MBYTE	1
#define HBYTE	0
#else
#define LBYTE	0
#define MBYTE	1
#define HBYTE	2
#endif

/*
There is no plain middle-side option; instead there are various mixing
modes including middle-side, each lossless, as embodied in the mix()
and unmix() functions.  These functions exploit a generalized middle-side
transformation:

u := [(rL + (m-r)R)/m];
v := L - R;

where [ ] denotes integer floor.  The (lossless) inverse is

L = u + v - [rV/m];
R = L - v;
*/

// 16-bit routines

__global__ void gpu_mix16_1(int16_t * ip, uint32_t stride, int32_t * u, int32_t * v, int32_t numSamples, int32_t m2, int32_t mixbits, int32_t mixres)
{
	int z = threadIdx.x + blockIdx.x * blockDim.x; 
	if (z < numSamples)
	{
		int32_t		l, r;
		ip += stride * z;
		l = (int32_t)ip[0];
		r = (int32_t)ip[1];
		u[z] = (mixres * l + m2 * r) >> mixbits;
		v[z] = l - r;
	}
}

__global__ void gpu_mix16_2(int16_t * ip, uint32_t stride, int32_t * u, int32_t * v, int32_t numSamples)
{
	int z = threadIdx.x + blockIdx.x * blockDim.x;
	if (z < numSamples)
	{
		ip += stride * z;
		u[z] = (int32_t)ip[0];
		v[z] = (int32_t)ip[1];
	}
}

void mix16(int16_t * in, uint32_t stride, int32_t * u, int32_t * v, int32_t numSamples, int32_t mixbits, int32_t mixres)
{

	if (mixres != 0)
	{
		int32_t		mod = 1 << mixbits;
		int32_t		m2;

		/* matrixed stereo */
		m2 = mod - mixres;

		gpu_mix16_1 << < (numSamples + SIZE - 1) / SIZE, SIZE >> >(in, stride, u, v, numSamples, m2, mixbits, mixres);

	
	}
	else
	{
		/* Conventional separated stereo. */
		gpu_mix16_2<<< (numSamples + SIZE - 1) / SIZE, SIZE >>>(in, stride, u, v, numSamples);

	}
}

__global__ void gpu_mix20_1(uint8_t * ip, uint32_t stride, int32_t * u, int32_t * v, int32_t numSamples, int32_t mixres, int32_t m2, int32_t mixbits)
{
	int z = threadIdx.x + blockIdx.x * blockDim.x;
	if (z < numSamples)
	{
		int32_t		l, r;

		ip += 3 * z;
		ip += (stride - 1) * 3 * z;
		l = (int32_t)(((uint32_t)ip[HBYTE] << 16) | ((uint32_t)ip[MBYTE] << 8) | (uint32_t)ip[LBYTE]);
		l = (l << 8) >> 12;

		ip += 3;
		r = (int32_t)(((uint32_t)ip[HBYTE] << 16) | ((uint32_t)ip[MBYTE] << 8) | (uint32_t)ip[LBYTE]);
		r = (r << 8) >> 12;

		u[z] = (mixres * l + m2 * r) >> mixbits;
		v[z] = l - r;
	}
}


__global__ void gpu_mix20_2(uint8_t * ip, uint32_t stride, int32_t * u, int32_t * v, int32_t numSamples)
{
	int z = threadIdx.x + blockIdx.x * blockDim.x;
	if (z < numSamples)
	{
		int32_t		l, r;


		ip += 3 * z;
		ip += (stride - 1) * 3 * z;
		l = (int32_t)(((uint32_t)ip[HBYTE] << 16) | ((uint32_t)ip[MBYTE] << 8) | (uint32_t)ip[LBYTE]);
		u[z] = (l << 8) >> 12;

		ip += 3;
		r = (int32_t)(((uint32_t)ip[HBYTE] << 16) | ((uint32_t)ip[MBYTE] << 8) | (uint32_t)ip[LBYTE]);
		v[z] = (r << 8) >> 12;
	}
}

// 20-bit routines
// - the 20 bits of data are left-justified in 3 bytes of storage but right-aligned for input/output predictor buffers

void mix20(uint8_t * in, uint32_t stride, int32_t * u, int32_t * v, int32_t numSamples, int32_t mixbits, int32_t mixres)
{
	printf("\nENTERS mix20\n");
	return;

	/*int32_t		l, r;
	uint8_t *	ip = in;
	int32_t			j;

	printf("Enters mix20");

	int32_t *d_u, *d_v;
	uint8_t *d_ip;

	cudaMalloc(&d_u, numSamples * sizeof(int32_t));
	cudaMalloc(&d_v, numSamples * sizeof(int32_t));
	cudaMalloc(&d_ip, stride * 3 * numSamples * sizeof(uint8_t));


	cudaMemcpy(d_u, u, numSamples * sizeof(int32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_v, v, numSamples * sizeof(int32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_ip, ip, stride * 3 * numSamples * sizeof(uint8_t), cudaMemcpyHostToDevice);*/

	if (mixres != 0)
	{
		/* matrixed stereo */
		int32_t		mod = 1 << mixbits;
		int32_t		m2 = mod - mixres;

		gpu_mix20_1 << < (numSamples + SIZE - 1) / SIZE, SIZE >> >(in, stride, u, v, numSamples, mixres, m2, mixbits);

	}
	else
	{

		gpu_mix20_2 << < (numSamples + SIZE - 1) / SIZE, SIZE >> >(in, stride, u, v, numSamples);

		/* Conventional separated stereo. */
	}

	/*cudaMemcpy(u, d_u, numSamples * sizeof(int32_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(v, d_v, numSamples * sizeof(int32_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(ip, d_ip, stride * 3 * numSamples * sizeof(uint8_t), cudaMemcpyDeviceToHost);

	cudaFree(d_u);
	cudaFree(d_v);
	cudaFree(d_ip);*/

}

// 24-bit routines
// - the 24 bits of data are right-justified in the input/output predictor buffers

__global__ void gpu_mix24_1_1(uint8_t * ip, uint32_t stride, int32_t * u, int32_t * v, int32_t numSamples, uint16_t * shiftUV, int32_t mixres, uint32_t mask, int32_t m2, int32_t mixbits, int32_t shift)
{
	int z = threadIdx.x + blockIdx.x * blockDim.x;
	if (z < numSamples)
	{
		int32_t		l, r;
		int32_t k = z * 2;

		
		ip += 3 * z;
		ip += (stride - 1) * 3 * z;
		l = (int32_t)(((uint32_t)ip[HBYTE] << 16) | ((uint32_t)ip[MBYTE] << 8) | (uint32_t)ip[LBYTE]);
		l = (l << 8) >> 8;

		ip += 3;
		r = (int32_t)(((uint32_t)ip[HBYTE] << 16) | ((uint32_t)ip[MBYTE] << 8) | (uint32_t)ip[LBYTE]);
		r = (r << 8) >> 8;

		shiftUV[k + 0] = (uint16_t)(l & mask);
		shiftUV[k + 1] = (uint16_t)(r & mask);

		l >>= shift;
		r >>= shift;

		u[z] = (mixres * l + m2 * r) >> mixbits;
		v[z] = l - r;
	}
}

__global__ void gpu_mix24_1_2(uint8_t * ip, uint32_t stride, int32_t * u, int32_t * v, int32_t numSamples, int32_t mixres, int32_t m2, int32_t mixbits)
{
	int z = threadIdx.x + blockIdx.x * blockDim.x;
	if (z < numSamples)
	{
		int32_t		l, r;

		ip += 3 * z;
		ip += (stride - 1) * 3 * z;
		l = (int32_t)(((uint32_t)ip[HBYTE] << 16) | ((uint32_t)ip[MBYTE] << 8) | (uint32_t)ip[LBYTE]);
		l = (l << 8) >> 8;

		ip += 3;
		r = (int32_t)(((uint32_t)ip[HBYTE] << 16) | ((uint32_t)ip[MBYTE] << 8) | (uint32_t)ip[LBYTE]);
		r = (r << 8) >> 8;

		u[z] = (mixres * l + m2 * r) >> mixbits;
		v[z] = l - r;
	}
}

__global__ void gpu_mix24_2_1(uint8_t * ip, uint32_t stride, int32_t * u, int32_t * v, int32_t numSamples, uint16_t * shiftUV, uint32_t mask, int32_t shift)
{
	int z = threadIdx.x + blockIdx.x * blockDim.x;
	if (z < numSamples)
	{
		int32_t		l, r;
		int32_t k = z * 2;


		ip += 3 * z;
		ip += (stride - 1) * 3 * z;
		l = (int32_t)(((uint32_t)ip[HBYTE] << 16) | ((uint32_t)ip[MBYTE] << 8) | (uint32_t)ip[LBYTE]);
		l = (l << 8) >> 8;

		ip += 3;
		r = (int32_t)(((uint32_t)ip[HBYTE] << 16) | ((uint32_t)ip[MBYTE] << 8) | (uint32_t)ip[LBYTE]);
		r = (r << 8) >> 8;

		shiftUV[k + 0] = (uint16_t)(l & mask);
		shiftUV[k + 1] = (uint16_t)(r & mask);

		l >>= shift;
		r >>= shift;

		u[z] = l;
		v[z] = r;
	}
}

__global__ void gpu_mix24_2_2(uint8_t * ip, uint32_t stride, int32_t * u, int32_t * v, int32_t numSamples)
{
	int z = threadIdx.x + blockIdx.x * blockDim.x;
	if (z < numSamples)
	{
		int32_t		l, r;


		ip += 3 * z;
		ip += (stride - 1) * 3 * z;
		l = (int32_t)(((uint32_t)ip[HBYTE] << 16) | ((uint32_t)ip[MBYTE] << 8) | (uint32_t)ip[LBYTE]);
		u[z] = (l << 8) >> 8;

		ip += 3;
		r = (int32_t)(((uint32_t)ip[HBYTE] << 16) | ((uint32_t)ip[MBYTE] << 8) | (uint32_t)ip[LBYTE]);
		v[z] = (r << 8) >> 8;
	}
}

void mix24(uint8_t * in, uint32_t stride, int32_t * u, int32_t * v, int32_t numSamples,
	int32_t mixbits, int32_t mixres, uint16_t * shiftUV, int32_t bytesShifted)
{
//	printf("%d\t%d\t%d\t%d\n", *(uint8_t*)in, (uint8_t*)in, *((uint8_t*)in + 1), (uint8_t*)in + 1);

	uint8_t *	ip = in;
	int32_t			shift = bytesShifted * 8;
	uint32_t	mask = (1ul << shift) - 1;

	if (mixres != 0)
	{
		/* matrixed stereo */
		int32_t		mod = 1 << mixbits;
		int32_t		m2 = mod - mixres;

		if (bytesShifted != 0)
		{
			gpu_mix24_1_1 << < (numSamples + SIZE - 1) / SIZE, SIZE >> >(ip, stride, u, v, numSamples, shiftUV, mixres, mask, m2, mixbits, shift);
		}
		else
		{
			gpu_mix24_1_2 << < (numSamples + SIZE - 1) / SIZE, SIZE >> >(ip, stride, u, v, numSamples, mixres, m2, mixbits);
		}
	}
	else
	{
		/* Conventional separated stereo. */
		if (bytesShifted != 0)
		{
			gpu_mix24_2_1 << < (numSamples + SIZE - 1) / SIZE, SIZE >> >(ip, stride, u, v, numSamples, shiftUV, mask, shift);
		}
		else
		{
			gpu_mix24_2_2 << < (numSamples + SIZE - 1) / SIZE, SIZE >> >(ip, stride, u, v, numSamples);
		}
	}

}

// 32-bit routines
// - note that these really expect the internal data width to be < 32 but the arrays are 32-bit
// - otherwise, the calculations might overflow into the 33rd bit and be lost
// - therefore, these routines deal with the specified "unused lower" bytes in the "shift" buffers

__global__ void gpu_mix32_1(int32_t * ip, uint32_t stride, int32_t * u, int32_t * v, int32_t numSamples, uint16_t * shiftUV, int32_t mixres, uint32_t mask, int32_t m2, int32_t mixbits, int32_t shift)
{
	int z = threadIdx.x + blockIdx.x * blockDim.x;
	if (z < numSamples)
	{
		int32_t		l, r;
		int32_t k = z * 2;


		ip += stride * z;
		l = ip[0];
		r = ip[1];

		shiftUV[k + 0] = (uint16_t)(l & mask);
		shiftUV[k + 1] = (uint16_t)(r & mask);

		l >>= shift;
		r >>= shift;

		u[z] = (mixres * l + m2 * r) >> mixbits;
		v[z] = l - r;
	}
}

__global__ void gpu_mix32_2_1(int32_t * ip, uint32_t stride, int32_t * u, int32_t * v, int32_t numSamples)
{
	int z = threadIdx.x + blockIdx.x * blockDim.x;
	if (z < numSamples)
	{

		ip += stride * z;
		u[z] = ip[0];
		v[z] = ip[1];
	}
}

__global__ void gpu_mix32_2_2(int32_t * ip, uint32_t stride, int32_t * u, int32_t * v, int32_t numSamples, uint16_t * shiftUV, uint32_t mask, int32_t shift)
{
	int z = threadIdx.x + blockIdx.x * blockDim.x;
	if (z < numSamples)
	{
		int32_t		l, r;
		int32_t k = z * 2;


		ip += stride * z;
		l = ip[0];
		r = ip[1];

		shiftUV[k + 0] = (uint16_t)(l & mask);
		shiftUV[k + 1] = (uint16_t)(r & mask);

		l >>= shift;
		r >>= shift;

		u[z] = l;
		v[z] = r;
	}
}

void mix32(int32_t * in, uint32_t stride, int32_t * u, int32_t * v, int32_t numSamples,
	int32_t mixbits, int32_t mixres, uint16_t * shiftUV, int32_t bytesShifted)
{
	int32_t	*	ip = in;
	int32_t			shift = bytesShifted * 8;
	uint32_t	mask = (1ul << shift) - 1;

	if (mixres != 0)
	{
		int32_t		mod = 1 << mixbits;
		int32_t		m2;

		//Assert( bytesShifted != 0 );

		/* matrixed stereo with shift */
		m2 = mod - mixres;

		gpu_mix32_1 << < (numSamples + SIZE - 1) / SIZE, SIZE >> >(ip, stride, u, v, numSamples, shiftUV, mixres, mask, m2, mixbits, shift);
	}
	else
	{
		if (bytesShifted == 0)
		{
			/* de-interleaving w/o shift */
			gpu_mix32_2_1 << < (numSamples + SIZE - 1) / SIZE, SIZE >> >(ip, stride, u, v, numSamples);
		}
		else
		{
			gpu_mix32_2_2 << < (numSamples + SIZE - 1) / SIZE, SIZE >> >(ip, stride, u, v, numSamples, shiftUV, mask, shift);
		}
	}

}

// 20/24-bit <-> 32-bit helper routines (not really matrixing but convenient to put here)

void copy20ToPredictor(uint8_t * in, uint32_t stride, int32_t * out, int32_t numSamples)
{
	printf("\nENTERS copy20ToPredictor\n");
	return;
	uint8_t *	ip = in;
	int32_t			j;
	for (j = 0; j < numSamples; j++)
	{
		int32_t			val;

		// 20-bit values are left-aligned in the 24-bit input buffer but right-aligned in the 32-bit output buffer
		val = (int32_t)(((uint32_t)ip[HBYTE] << 16) | ((uint32_t)ip[MBYTE] << 8) | (uint32_t)ip[LBYTE]);
		out[j] = (val << 8) >> 12;
		ip += stride * 3;
	}
}

void copy24ToPredictor(uint8_t * in, uint32_t stride, int32_t * out, int32_t numSamples)
{
	printf("\nENTERS copy24ToPredictor\n");
	return;
	uint8_t *	ip = in;
	int32_t			j;
	for (j = 0; j < numSamples; j++)
	{
		int32_t			val;

		val = (int32_t)(((uint32_t)ip[HBYTE] << 16) | ((uint32_t)ip[MBYTE] << 8) | (uint32_t)ip[LBYTE]);
		out[j] = (val << 8) >> 8;
		ip += stride * 3;
	}
}
