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
	File:		ag_enc.c
	
	Contains:   Adaptive Golomb encode routines.

	Copyright:	(c) 2001-2011 Apple, Inc.
*/

#include "aglib.h"
#include "ALACBitUtilities.h"
#include "EndianPortable.h"
#include "ALACAudioTypes.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <cuda_profiler_api.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>



#if __GNUC__ && TARGET_OS_MAC
	#if __POWERPC__
		#include <ppc_intrinsics.h>
	#else
		#include <libkern/OSByteOrder.h>
	#endif
#endif

#define CODE_TO_LONG_MAXBITS	32
#define N_MAX_MEAN_CLAMP		0xffff
#define N_MEAN_CLAMP_VAL		0xffff
#define REPORT_VAL  40

#if __GNUC__
#define ALWAYS_INLINE		__attribute__((always_inline))
#else
#define ALWAYS_INLINE
#endif


/*	And on the subject of the CodeWarrior x86 compiler and inlining, I reworked a lot of this
	to help the compiler out.   In many cases this required manual inlining or a macro.  Sorry
	if it is ugly but the performance gains are well worth it.
	- WSK 5/19/04
*/

#define BSWAP32(x) (((x << 24) | ((x << 8) & 0x00ff0000) | ((x >> 8) & 0x0000ff00) | ((x >> 24) & 0x000000ff)))

#if defined(__i386__)
#define TARGET_RT_LITTLE_ENDIAN 1
#elif defined(__x86_64__)
#define TARGET_RT_LITTLE_ENDIAN 1
#elif defined (TARGET_OS_WIN32)
#define TARGET_RT_LITTLE_ENDIAN 1
#endif

__device__ uint32_t d_Swap32NtoB(uint32_t inUInt32)
{
#if TARGET_RT_LITTLE_ENDIAN
	return BSWAP32(inUInt32);
#else
	return inUInt32;
#endif
}

__device__ uint32_t d_Swap32BtoN(uint32_t inUInt32)
{
#if TARGET_RT_LITTLE_ENDIAN
	return BSWAP32(inUInt32);
#else
	return inUInt32;
#endif
}


// note: implementing this with some kind of "count leading zeros" assembly is a big performance win

__device__ int32_t d_lead(int32_t m)
{
	long j;
	unsigned long c = (1ul << 31);

	for (j = 0; j < 32; j++)
	{
		if ((c & m) != 0)
			break;
		c >>= 1;
	}
	return (j);
}

#define arithmin(a, b) ((a) < (b) ? (a) : (b))


__device__ int32_t d_lg3a(int32_t x)
{

	int32_t result;

	x += 3;
	result = d_lead(x);

	return 31 - result;
}


__device__ int32_t d_abs_func(int32_t a)
{
	// note: the CW PPC intrinsic __abs() turns into these instructions so no need to try and use it
	int32_t isneg = a >> 31;
	int32_t xorval = a ^ isneg;
	int32_t result = xorval - isneg;

	return result;
}

static inline uint32_t ALWAYS_INLINE read32bit( uint8_t * buffer )
{
	// embedded CPUs typically can't read unaligned 32-bit words so just read the bytes
	uint32_t		value;
	
	value = ((uint32_t)buffer[0] << 24) | ((uint32_t)buffer[1] << 16) |
			 ((uint32_t)buffer[2] << 8) | (uint32_t)buffer[3];
	return value;
}

#if PRAGMA_MARK
#pragma mark -
#endif

__device__ int32_t d_dyn_code(int32_t m, int32_t k, int32_t n, uint32_t *outNumBits)
{
	uint32_t 	div, mod, de;
	uint32_t	numBits;
	uint32_t	value;

	//Assert( n >= 0 );

	div = n / m;

	if (div >= MAX_PREFIX_16)
	{
		numBits = MAX_PREFIX_16 + MAX_DATATYPE_BITS_16;
		value = (((1 << MAX_PREFIX_16) - 1) << MAX_DATATYPE_BITS_16) + n;
	}
	else
	{
		mod = n%m;
		de = (mod == 0);
		numBits = div + k + 1 - de;
		value = (((1 << div) - 1) << (numBits - div)) + mod + 1 - de;

		// if coding this way is bigger than doing escape, then do escape
		if (numBits > MAX_PREFIX_16 + MAX_DATATYPE_BITS_16)
		{
			numBits = MAX_PREFIX_16 + MAX_DATATYPE_BITS_16;
			value = (((1 << MAX_PREFIX_16) - 1) << MAX_DATATYPE_BITS_16) + n;
		}
	}

	*outNumBits = numBits;

	return (int32_t)value;
}


__device__ int32_t d_dyn_code_32bit(int32_t maxbits, uint32_t m, uint32_t k, uint32_t n, uint32_t *outNumBits, uint32_t *outValue, uint32_t *overflow, uint32_t *overflowbits)
{
	uint32_t 	div, mod, de;
	uint32_t	numBits;
	uint32_t	value;
	int32_t			didOverflow = 0;

	div = n / m;
	if (div < MAX_PREFIX_32)
	{
		mod = n - (m * div);

		de = (mod == 0);
		numBits = div + k + 1 - de;
		value = (((1 << div) - 1) << (numBits - div)) + mod + 1 - de;
		if (numBits > 25)
			goto codeasescape;
	}
	else
	{
	codeasescape:
		numBits = MAX_PREFIX_32;
		value = (((1 << MAX_PREFIX_32) - 1));
		*overflow = n;
		*overflowbits = maxbits;
		didOverflow = 1;
	}

	*outNumBits = numBits;
	*outValue = value;
	return didOverflow;
}



__device__ void d_dyn_jam_noDeref(unsigned char *out, uint32_t bitPos, uint32_t numBits, uint32_t value)
{
//	printf("%d\n", bitPos);
//	printf("%d\t%d\n", *out, out);

	uint32_t i = 0;
	i |= ((out + (bitPos >> 3))[3] << 24);
	i |= ((out + (bitPos >> 3))[2] << 16);
	i |= ((out + (bitPos >> 3))[1] << 8);
	i |= ((out + (bitPos >> 3))[0]);

//	uint32_t	*i = (uint32_t*)(out + (bitPos >> 3));

	uint32_t	mask;
	uint32_t	curr;
	uint32_t	shift;

//	printf("%d\n", i);

	curr = i;
	curr = d_Swap32NtoB(curr);
//	printf("%d\t",curr);
	shift = 32 - (bitPos & 7) - numBits;
//	printf("%d\t", shift);
	mask = ~0u >> (32 - numBits);		// mask must be created in two steps to avoid compiler sequencing ambiguity
	mask <<= shift;
//	printf("%d\t", mask);
	value = (value << shift) & mask;
	value |= curr & ~mask;
//	printf("%d\n", value);
	i = d_Swap32BtoN(value);

	(out + (bitPos >> 3))[3] = (i >> 24) & 0xFF;
	(out + (bitPos >> 3))[2] = (i >> 16) & 0xFF;
	(out + (bitPos >> 3))[1] = (i >> 8) & 0xFF;
	(out + (bitPos >> 3))[0] = i & 0xFF;

//	printf("%d\n", i);
}


__device__ void d_dyn_jam_noDeref_large(unsigned char *out, uint32_t bitPos, uint32_t numBits, uint32_t value)
{

	uint32_t i = 0;
	i |= ((out + (bitPos >> 3))[3] << 24);
	i |= ((out + (bitPos >> 3))[2] << 16);
	i |= ((out + (bitPos >> 3))[1] << 8);
	i |= ((out + (bitPos >> 3))[0]);

	uint32_t	w;
	uint32_t	curr;
	uint32_t	mask;
	int32_t			shiftvalue = (32 - (bitPos & 7) - numBits);
	//Assert(numBits <= 32);

	curr = i;
	curr = d_Swap32NtoB(curr);
	if (shiftvalue < 0)
	{
		printf("MARK");

		uint8_t 	tailbyte;
		uint8_t 	*tailptr;

		w = value >> -shiftvalue;
		mask = ~0u >> -shiftvalue;
		w |= (curr & ~mask);

		tailptr = ((uint8_t *)i) + 4;
		tailbyte = (value << ((8 + shiftvalue))) & 0xff;
		*tailptr = (uint8_t)tailbyte;
	}
	else
	{
		mask = ~0u >> (32 - numBits);
		mask <<= shiftvalue;			// mask must be created in two steps to avoid compiler sequencing ambiguity

		w = (value << shiftvalue) & mask;
		w |= curr & ~mask;
	}

	i = d_Swap32BtoN(w);

	(out + (bitPos >> 3))[3] = (i >> 24) & 0xFF;
	(out + (bitPos >> 3))[2] = (i >> 16) & 0xFF;
	(out + (bitPos >> 3))[1] = (i >> 8) & 0xFF;
	(out + (bitPos >> 3))[0] = i & 0xFF;
}


__global__ void gpu_dyn_comp(int32_t bitSize, uint32_t *mb, uint32_t	pb, uint32_t kb, uint32_t wb, int32_t *inPtr, int32_t numSamples, unsigned char * out,
	uint32_t	*bitPos, int32_t	rowSize, int32_t rowJump, int32_t *status)
{
	uint32_t numBits, value, value2, overflow, overflowbits, n, m, k, mz, nz, c = 0;
	int32_t del, zmode = 0, rowPos = 0;
	int32_t result, didOverflow;

	while (c < numSamples)
	{
		m = *mb >> QBSHIFT;
		k = d_lg3a(m);
		if (k > kb)
		{
			k = kb;
		}
		m = (1 << k) - 1;

		del = *inPtr++;			
		rowPos++;

		n = (d_abs_func(del) << 1) - ((del >> 31) & 1) - zmode;
		//Assert( 32-lead(n) <= bitSize );
//		printf("%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n\n", bitSize, m, k, n, numBits, value, overflow, overflowbits);
		if (d_dyn_code_32bit(bitSize, m, k, n, &numBits, &value, &overflow, &overflowbits))
		{ 
			d_dyn_jam_noDeref(out, *bitPos, numBits, value);
			*bitPos += numBits;
			d_dyn_jam_noDeref_large(out, *bitPos, overflowbits, overflow);
			*bitPos += overflowbits;
		}
		else
		{
//			printf("--->%d\t%d\n", *(out + 4), (out + 4));
			d_dyn_jam_noDeref(out, *bitPos, numBits, value);
//			printf("<---%d\t%d\n\n", *(out + 4), (out + 4));
			*bitPos += numBits;
		}

//		printf("First- %d\n", *out);
		c++;		// no need
		if (rowPos >= rowSize)
		{
			rowPos = 0;
			inPtr += rowJump;
		}
		*mb = pb * (n + zmode) + *mb - ((pb * *mb) >> QBSHIFT);

		// update mean tracking if it's overflowed
		if (n > N_MAX_MEAN_CLAMP)
			*mb = N_MEAN_CLAMP_VAL;


		zmode = 0;

		RequireAction(c <= numSamples, *status = kALAC_ParamError; return;);			//<---------handle this
		if (((*mb << MMULSHIFT) < QB) && (c < numSamples))
		{
			zmode = 1;
			nz = 0;

			while (c<numSamples && *inPtr == 0) //workable in cuda need to check -s
			{
				/* Take care of wrap-around globals. */
				++inPtr;
				++nz;
				++c;
				if (++rowPos >= rowSize)
				{
					rowPos = 0;
					inPtr += rowJump;
				}

				if (nz >= 65535)
				{
					zmode = 0;
					break;
				}
			}


			k = d_lead(*mb) - BITOFF + ((*mb + MOFF) >> MDENSHIFT);
			mz = ((1 << k) - 1) & wb;

			value = d_dyn_code(mz, k, nz, &numBits);
			d_dyn_jam_noDeref(out, *bitPos, numBits, value);
			*bitPos += numBits;

//			printf("First- %d\n", *out);

			*mb = 0;
		}
//		printf("%d\t%d\t%d\t%d\t%d\t%d\t%d\n\n", *mb, pb, kb, wb, *inPtr, *out, *bitPos);
	}
	
}



int32_t dyn_comp( AGParamRecPtr params, int32_t * pc, BitBuffer * bitstream, int32_t numSamples, int32_t bitSize, uint32_t * outNumBits )
{

    unsigned char *		out;
    uint32_t		bitPos, startPos;
    uint32_t			m, k, n, c, mz, nz;
    uint32_t		numBits;
    uint32_t			value;
    int32_t				del, zmode;
	uint32_t		overflow, overflowbits;
    int32_t					status;

    // shadow the variables in params so there's not the dereferencing overhead
    uint32_t		mb, pb, kb, wb;
    int32_t					rowPos = 0;
    int32_t					rowSize = params->sw;
    int32_t					rowJump = (params->fw) - rowSize;
//    int32_t *			inPtr = pc;
//	int32_t				result; //-s
	*outNumBits = 0;
	RequireAction( (bitSize >= 1) && (bitSize <= 32), return kALAC_ParamError; );

	out = bitstream->cur;
	//printf("length- %d\n", sizeof(out));
	startPos = bitstream->bitIndex;
    bitPos = startPos;

    mb = params->mb = params->mb0;
    pb = params->pb;
    kb = params->kb;
    wb = params->wb;
    zmode = 0;

    c=0;
	status = ALAC_noErr;

/*	int32_t rowpos, int32_t numSamples, uint32_t m, uint32_t mb, uint32_t k, uint32_t kb, int32_t del, int32_t *inPtr int32_t result*/ //-s
	

//	printf("%d\t%d\t%d\t%d\t%d\t%d\t%d\n", mb, pb, kb, wb, *(pc + 1), out, startPos);


	int32_t *d_pc, *d_status;
	uint32_t *d_bitPos, *d_mb;
	unsigned char *d_out;

	cudaMalloc(&d_pc, numSamples * sizeof(int32_t));
	cudaMalloc(&d_status, sizeof(int32_t));
	cudaMalloc(&d_mb, sizeof(uint32_t));
	cudaMalloc(&d_bitPos, sizeof(uint32_t));
	cudaMalloc(&d_out, numSamples * 2 * sizeof(unsigned char));

	cudaMemcpy(d_pc, pc, numSamples * sizeof(int32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_status, &status, sizeof(int32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_bitPos, &bitPos, sizeof(uint32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_mb, &mb, sizeof(uint32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_out, out, numSamples* 2 * sizeof(unsigned char), cudaMemcpyHostToDevice);

	gpu_dyn_comp <<<1, 1>>>(bitSize, d_mb, pb, kb, wb, d_pc, numSamples, d_out,
		d_bitPos, rowSize, rowJump, d_status);

	cudaMemcpy(pc, d_pc, numSamples * sizeof(int32_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(&status, d_status, sizeof(int32_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(&bitPos, d_bitPos, sizeof(uint32_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(&mb, d_mb, sizeof(uint32_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(out, d_out, numSamples* 2 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

	cudaFree(d_pc);
	cudaFree(d_status);
	cudaFree(d_bitPos);
	cudaFree(d_mb);
	cudaFree(d_out);

//	printf("%d\t%d\t%d\t%d\t%d\t%d\t%d\n\n", mb, pb, kb, wb, *pc, *out, bitPos);
	if( status == kALAC_ParamError)
		goto Exit;


    *outNumBits = (bitPos - startPos);
	BitBufferAdvance( bitstream, *outNumBits );

Exit:
//	printf("%d\t%d\t%d\t%d\t%d\t%d\t%d\n\n", mb, pb, kb, wb, *(pc), *out, bitPos);
	return status;
}
