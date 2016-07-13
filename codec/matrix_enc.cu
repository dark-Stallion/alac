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

#include "matrixlib.h"
#include "ALACAudioTypes.h"

#define SIZE 512

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

__global__ void gpu_mix20_2(int16_t * ip, uint32_t stride, int32_t * u, int32_t * v, int32_t numSamples)
{
	int z = threadIdx.x + blockIdx.x * blockDim.x;
	if (z < numSamples)
	{
		int32_t	l, r;

		l = (int32_t)(((uint32_t)ip[HBYTE] << 16) | ((uint32_t)ip[MBYTE] << 8) | (uint32_t)ip[LBYTE]);
		u[z] = (l << 8) >> 12;
		ip += 3 * z;

		r = (int32_t)(((uint32_t)ip[HBYTE] << 16) | ((uint32_t)ip[MBYTE] << 8) | (uint32_t)ip[LBYTE]);
		v[z] = (r << 8) >> 12;
		ip += (stride - 1) * 3 * z;
	}
}

void mix16(int16_t * in, uint32_t stride, int32_t * u, int32_t * v, int32_t numSamples, int32_t mixbits, int32_t mixres)
{
	int16_t	*	ip = in;
	//	int32_t			j;

	int32_t *d_u, *d_v;
	int16_t *d_ip;

	cudaMalloc(&d_u, numSamples * sizeof(int32_t));
	cudaMalloc(&d_v, numSamples * sizeof(int32_t));
	cudaMalloc(&d_ip, stride * numSamples * sizeof(int16_t));

	cudaMemcpy(d_u, u, numSamples * sizeof(int32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_v, v, numSamples * sizeof(int32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_ip, ip, stride * numSamples * sizeof(int16_t), cudaMemcpyHostToDevice);

	if (mixres != 0)
	{
//		printf("\nENTERS mix16 1\n");
		int32_t		mod = 1 << mixbits;
		int32_t		m2;

		/* matrixed stereo */
		m2 = mod - mixres;
		/*		for (j = 0; j < numSamples; j++)
		{
		int32_t		l, r;

		l = (int32_t)ip[0];
		r = (int32_t)ip[1];
		ip += stride;
		u[j] = (mixres * l + m2 * r) >> mixbits;
		v[j] = l - r;
		}*/
		gpu_mix16_1 << < (numSamples + SIZE - 1) / SIZE, SIZE >> >(d_ip, stride, d_u, d_v, numSamples, m2, mixbits, mixres);
	}
	else
	{
		/* Conventional separated stereo. */

		/*		printf("\n\n---------NEW---------\n\nNumber of Samples: %d\n", numSamples);

		for (int i = 0; i < 10; i++){
		printf("%x\t\t%d\t\t%d\t\t\t%d\t\t%d\n", ip + stride*i, ip[0], ip[1], u[i], v[i]);
		}

		printf("\n\n---------AFTER---------\n\n");*/


		gpu_mix16_2 << < (numSamples + SIZE - 1) / SIZE, SIZE >> >(d_ip, stride, d_u, d_v, numSamples);



		/*		for (int i = 0; i < 10; i++){
		printf("%x\t\t%d\t\t%d\t\t\t%d\t\t%d\n", ip + stride*i, ip[0], ip[1], u[i], v[i]);
		}*/




		/*		printf("\n\n---------NEW---------\n\nNumber of Samples: %d\n", numSamples);

		for (int i = 0; i < 10; i++){
		printf("%x\t\t%d\t\t%d\t\t\t%d\t\t%d\n", ip + stride*i, ip[0], ip[1], u[i], v[i]);
		}

		printf("\n\n---------AFTER---------\n\n");*/

		/*int32_t *du = (int32_t *)malloc(numSamples * sizeof(int32_t));
		int32_t *dv = (int32_t *)malloc(numSamples * sizeof(int32_t));
		memcpy(du, u, numSamples * sizeof(int32_t));
		memcpy(dv, v, numSamples * sizeof(int32_t));
		printf("%x \t\t %x \t\t %d \t\t %d \t\t %d \t\t %d \n", u, v, u[0], v[0], u[numSamples], v[numSamples]);
		printf("%x \t\t %x \t\t %d \t\t %d \t\t %d \t\t %d \n\n", du, dv, du[0], dv[0], du[numSamples], dv[numSamples]);
		free(du);
		free(dv);*/

		/*int16_t *dip = (int16_t *)malloc(numSamples * sizeof(int16_t) * 2);
		memcpy(dip, ip, 2 * numSamples * sizeof(int16_t));
		printf("%x \t\t %x \t\t %d \t\t %d \n", ip, ip + 2, ip[0], (ip + 2)[0]);
		printf("%x \t\t %x \t\t %d \t\t %d \n\n", dip, dip + 2, dip[0], (dip + 2)[0]);

		printf("%x \t\t %x \t\t %d \t\t %d \n", ip + (numSamples - 1) * stride, (ip + 2) + (numSamples - 1) * stride, (ip + (numSamples - 1) * stride)[0], (ip + 2 + (numSamples - 1) * stride)[0]);
		printf("%x \t\t %x \t\t %d \t\t %d \n\n", dip + (numSamples - 1) * stride, (dip + 2) + (numSamples - 1) * stride, (dip + (numSamples - 1) * stride)[0], (dip + 2 + (numSamples - 1) * stride)[0]);
		free(dip);*/

		/*for ( j = 0; j < numSamples; j++ )
		{
		u[j] = (int32_t) ip[0];
		v[j] = (int32_t) ip[1];
		ip += stride;
		}*/
		//		printf("%x \t\t %x \t\t %d \t\t %d \n\n\n", ip - 2, ip, (ip-2)[0], ip[0]);
		//		printf("%x \t\t %x \t\t %d \t\t %d \t\t %d \t\t %d \n\n\n", u, v, u[0], v[0], u[numSamples], v[numSamples]);
		/*		for (int i = 0; i < 10; i++){
		printf("%x\t\t%d\t\t%d\t\t\t%d\t\t%d\n", ip + stride*i, ip[0], ip[1], u[i], v[i]);
		}*/
	}

	cudaMemcpy(u, d_u, numSamples * sizeof(int32_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(v, d_v, numSamples * sizeof(int32_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(ip, d_ip, stride * numSamples * sizeof(int16_t), cudaMemcpyDeviceToHost);

	cudaFree(d_u);
	cudaFree(d_v);
	cudaFree(d_ip);
}

// 20-bit routines
// - the 20 bits of data are left-justified in 3 bytes of storage but right-aligned for input/output predictor buffers

void mix20(uint8_t * in, uint32_t stride, int32_t * u, int32_t * v, int32_t numSamples, int32_t mixbits, int32_t mixres)
{
	int32_t		l, r;
	uint8_t *	ip = in;
	int32_t			j;


	if (mixres != 0)
	{
		printf("\nENTERS mix20 1\n");
		/* matrixed stereo */
		int32_t		mod = 1 << mixbits;
		int32_t		m2 = mod - mixres;
		printf("\nENTERS mix20 1\n");
		for (j = 0; j < numSamples; j++)
		{
			l = (int32_t)(((uint32_t)ip[HBYTE] << 16) | ((uint32_t)ip[MBYTE] << 8) | (uint32_t)ip[LBYTE]);
			l = (l << 8) >> 12;
			ip += 3;

			r = (int32_t)(((uint32_t)ip[HBYTE] << 16) | ((uint32_t)ip[MBYTE] << 8) | (uint32_t)ip[LBYTE]);
			r = (r << 8) >> 12;
			ip += (stride - 1) * 3;

			u[j] = (mixres * l + m2 * r) >> mixbits;
			v[j] = l - r;
		}
	}
	else
	{

		/*int32_t *d_u, *d_v;
		int16_t *d_ip;

		cudaMalloc(&d_u, numSamples * sizeof(int32_t));
		cudaMalloc(&d_v, numSamples * sizeof(int32_t));
		cudaMalloc(&d_ip, stride * numSamples * sizeof(int16_t));

		cudaMemcpy(d_u, u, numSamples * sizeof(int32_t), cudaMemcpyHostToDevice);
		cudaMemcpy(d_v, v, numSamples * sizeof(int32_t), cudaMemcpyHostToDevice);
		cudaMemcpy(d_ip, ip, (stride-1) * 3 * numSamples * sizeof(int16_t), cudaMemcpyHostToDevice);

		gpu_mix20_2 << < (numSamples + SIZE - 1) / SIZE, SIZE >> >(d_ip, stride, d_u, d_v, numSamples);*/

//		printf("\nENTERS mix20 2\n");

		/* Conventional separated stereo. */
		for (j = 0; j < numSamples; j++)
		{
			l = (int32_t)(((uint32_t)ip[HBYTE] << 16) | ((uint32_t)ip[MBYTE] << 8) | (uint32_t)ip[LBYTE]);
			u[j] = (l << 8) >> 12;
			ip += 3;

			r = (int32_t)(((uint32_t)ip[HBYTE] << 16) | ((uint32_t)ip[MBYTE] << 8) | (uint32_t)ip[LBYTE]);
			v[j] = (r << 8) >> 12;
			ip += (stride - 1) * 3;
		}

		/*cudaMemcpy(u, d_u, numSamples * sizeof(int32_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(v, d_v, numSamples * sizeof(int32_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(ip, d_ip, stride * numSamples * sizeof(int16_t), cudaMemcpyDeviceToHost);

		cudaFree(d_u);
		cudaFree(d_v);
		cudaFree(d_ip);*/

	}
}

// 24-bit routines
// - the 24 bits of data are right-justified in the input/output predictor buffers

void mix24(uint8_t * in, uint32_t stride, int32_t * u, int32_t * v, int32_t numSamples,
	int32_t mixbits, int32_t mixres, uint16_t * shiftUV, int32_t bytesShifted)
{
	int32_t		l, r;
	uint8_t *	ip = in;
	int32_t			shift = bytesShifted * 8;
	uint32_t	mask = (1ul << shift) - 1;
	int32_t			j, k;

	if (mixres != 0)
	{
		/* matrixed stereo */
		int32_t		mod = 1 << mixbits;
		int32_t		m2 = mod - mixres;

		if (bytesShifted != 0)
		{
//			printf("\nENTERS mix24 1\n");
			for (j = 0, k = 0; j < numSamples; j++, k += 2)
			{
				l = (int32_t)(((uint32_t)ip[HBYTE] << 16) | ((uint32_t)ip[MBYTE] << 8) | (uint32_t)ip[LBYTE]);
				l = (l << 8) >> 8;
				ip += 3;

				r = (int32_t)(((uint32_t)ip[HBYTE] << 16) | ((uint32_t)ip[MBYTE] << 8) | (uint32_t)ip[LBYTE]);
				r = (r << 8) >> 8;
				ip += (stride - 1) * 3;

				shiftUV[k + 0] = (uint16_t)(l & mask);
				shiftUV[k + 1] = (uint16_t)(r & mask);

				l >>= shift;
				r >>= shift;

				u[j] = (mixres * l + m2 * r) >> mixbits;
				v[j] = l - r;
			}
		}
		else
		{
			
			for (j = 0; j < numSamples; j++)
			{
				l = (int32_t)(((uint32_t)ip[HBYTE] << 16) | ((uint32_t)ip[MBYTE] << 8) | (uint32_t)ip[LBYTE]);
				l = (l << 8) >> 8;
				ip += 3;

				r = (int32_t)(((uint32_t)ip[HBYTE] << 16) | ((uint32_t)ip[MBYTE] << 8) | (uint32_t)ip[LBYTE]);
				r = (r << 8) >> 8;
				ip += (stride - 1) * 3;

				u[j] = (mixres * l + m2 * r) >> mixbits;
				v[j] = l - r;
			}
		}
	}
	else
	{
		/* Conventional separated stereo. */
//		printf("\nENTERS mix24 2\n");
		if (bytesShifted != 0)
		{
			for (j = 0, k = 0; j < numSamples; j++, k += 2)
			{
				l = (int32_t)(((uint32_t)ip[HBYTE] << 16) | ((uint32_t)ip[MBYTE] << 8) | (uint32_t)ip[LBYTE]);
				l = (l << 8) >> 8;
				ip += 3;

				r = (int32_t)(((uint32_t)ip[HBYTE] << 16) | ((uint32_t)ip[MBYTE] << 8) | (uint32_t)ip[LBYTE]);
				r = (r << 8) >> 8;
				ip += (stride - 1) * 3;

				shiftUV[k + 0] = (uint16_t)(l & mask);
				shiftUV[k + 1] = (uint16_t)(r & mask);

				l >>= shift;
				r >>= shift;

				u[j] = l;
				v[j] = r;
			}
		}
		else
		{
			for (j = 0; j < numSamples; j++)
			{
				l = (int32_t)(((uint32_t)ip[HBYTE] << 16) | ((uint32_t)ip[MBYTE] << 8) | (uint32_t)ip[LBYTE]);
				u[j] = (l << 8) >> 8;
				ip += 3;

				r = (int32_t)(((uint32_t)ip[HBYTE] << 16) | ((uint32_t)ip[MBYTE] << 8) | (uint32_t)ip[LBYTE]);
				v[j] = (r << 8) >> 8;
				ip += (stride - 1) * 3;
			}
		}
	}
}

// 32-bit routines
// - note that these really expect the internal data width to be < 32 but the arrays are 32-bit
// - otherwise, the calculations might overflow into the 33rd bit and be lost
// - therefore, these routines deal with the specified "unused lower" bytes in the "shift" buffers

void mix32(int32_t * in, uint32_t stride, int32_t * u, int32_t * v, int32_t numSamples,
	int32_t mixbits, int32_t mixres, uint16_t * shiftUV, int32_t bytesShifted)
{
	int32_t	*	ip = in;
	int32_t			shift = bytesShifted * 8;
	uint32_t	mask = (1ul << shift) - 1;
	int32_t		l, r;
	int32_t			j, k;

	if (mixres != 0)
	{
//		printf("\nENTERS mix32 1\n");
		int32_t		mod = 1 << mixbits;
		int32_t		m2;

		//Assert( bytesShifted != 0 );

		/* matrixed stereo with shift */
		m2 = mod - mixres;
		for (j = 0, k = 0; j < numSamples; j++, k += 2)
		{
			l = ip[0];
			r = ip[1];
			ip += stride;

			shiftUV[k + 0] = (uint16_t)(l & mask);
			shiftUV[k + 1] = (uint16_t)(r & mask);

			l >>= shift;
			r >>= shift;

			u[j] = (mixres * l + m2 * r) >> mixbits;
			v[j] = l - r;
		}
	}
	else
	{
//		printf("\nENTERS mix32 2\n");
		if (bytesShifted == 0)
		{
			/* de-interleaving w/o shift */
			for (j = 0; j < numSamples; j++)
			{
				u[j] = ip[0];
				v[j] = ip[1];
				ip += stride;
			}
		}
		else
		{
			/* de-interleaving with shift */
			for (j = 0, k = 0; j < numSamples; j++, k += 2)
			{
				l = ip[0];
				r = ip[1];
				ip += stride;

				shiftUV[k + 0] = (uint16_t)(l & mask);
				shiftUV[k + 1] = (uint16_t)(r & mask);

				l >>= shift;
				r >>= shift;

				u[j] = l;
				v[j] = r;
			}
		}
	}
}

// 20/24-bit <-> 32-bit helper routines (not really matrixing but convenient to put here)

void copy20ToPredictor(uint8_t * in, uint32_t stride, int32_t * out, int32_t numSamples)
{
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
