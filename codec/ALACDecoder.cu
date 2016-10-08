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
	File:		ALACDecoder.cpp
*/

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "ALACDecoder.h"

#include "dplib.h"
#include "aglib.h"
#include "matrixlib.h"

#include "ALACBitUtilities.h"
#include "EndianPortable.h"

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

// constants/data
const uint32_t kMaxBitDepth = 32;			// max allowed bit depth is 32


// prototypes
static void Zero16( int16_t * buffer, uint32_t numItems, uint32_t stride );
static void Zero24( uint8_t * buffer, uint32_t numItems, uint32_t stride );
static void Zero32( int32_t * buffer, uint32_t numItems, uint32_t stride );

/*
	Constructor
*/
ALACDecoder::ALACDecoder() :
	mMixBufferU( nil ),
	mMixBufferV( nil ),
	mPredictor( nil ),
	mShiftBuffer( nil )
{
	memset( &mConfig, 0, sizeof(mConfig) );
}

/*
	Destructor
*/
ALACDecoder::~ALACDecoder()
{
	// delete the matrix mixing buffers
	if ( mMixBufferU )
    {
		free(mMixBufferU);
		cudaFree(d_u);
        mMixBufferU = NULL;
    }
	if ( mMixBufferV )
    {
		free(mMixBufferV);
		cudaFree(d_v);
        mMixBufferV = NULL;
    }
	
	// delete the dynamic predictor's "corrector" buffer
	// - note: mShiftBuffer shares memory with this buffer
	if ( mPredictor )
    {
		free(mPredictor);
		cudaFree(d_shiftUV);
        mPredictor = NULL;
    }
}

/*
	Init()
	- initialize the decoder with the given configuration
*/
int32_t ALACDecoder::Init( void * inMagicCookie, uint32_t inMagicCookieSize, int X )
{
	int32_t		status = ALAC_noErr;
    ALACSpecificConfig theConfig;
    uint8_t * theActualCookie = (uint8_t *)inMagicCookie;
    uint32_t theCookieBytesRemaining = inMagicCookieSize;

    // For historical reasons the decoder needs to be resilient to magic cookies vended by older encoders.
    // As specified in the ALACMagicCookieDescription.txt document, there may be additional data encapsulating 
    // the ALACSpecificConfig. This would consist of format ('frma') and 'alac' atoms which precede the
    // ALACSpecificConfig. 
    // See ALACMagicCookieDescription.txt for additional documentation concerning the 'magic cookie'
    
    // skip format ('frma') atom if present
    if (theActualCookie[4] == 'f' && theActualCookie[5] == 'r' && theActualCookie[6] == 'm' && theActualCookie[7] == 'a')
    {
        theActualCookie += 12;
        theCookieBytesRemaining -= 12;
    }
    
    // skip 'alac' atom header if present
    if (theActualCookie[4] == 'a' && theActualCookie[5] == 'l' && theActualCookie[6] == 'a' && theActualCookie[7] == 'c')
    {
        theActualCookie += 12;
        theCookieBytesRemaining -= 12;
    }

    // read the ALACSpecificConfig
    if (theCookieBytesRemaining >= sizeof(ALACSpecificConfig))
    {
        theConfig.frameLength = Swap32BtoN(((ALACSpecificConfig *)theActualCookie)->frameLength);
        theConfig.compatibleVersion = ((ALACSpecificConfig *)theActualCookie)->compatibleVersion;
        theConfig.bitDepth = ((ALACSpecificConfig *)theActualCookie)->bitDepth;
        theConfig.pb = ((ALACSpecificConfig *)theActualCookie)->pb;
        theConfig.mb = ((ALACSpecificConfig *)theActualCookie)->mb;
        theConfig.kb = ((ALACSpecificConfig *)theActualCookie)->kb;
        theConfig.numChannels = ((ALACSpecificConfig *)theActualCookie)->numChannels;
        theConfig.maxRun = Swap16BtoN(((ALACSpecificConfig *)theActualCookie)->maxRun);
        theConfig.maxFrameBytes = Swap32BtoN(((ALACSpecificConfig *)theActualCookie)->maxFrameBytes);
        theConfig.avgBitRate = Swap32BtoN(((ALACSpecificConfig *)theActualCookie)->avgBitRate);
        theConfig.sampleRate = Swap32BtoN(((ALACSpecificConfig *)theActualCookie)->sampleRate);

        mConfig = theConfig;
        
        RequireAction( mConfig.compatibleVersion <= kALACVersion, return kALAC_ParamError; );

		total_numSamples = 0;

        // allocate mix buffers
        mMixBufferU = (int32_t *) calloc( mConfig.frameLength * sizeof(int32_t), 1 );
        mMixBufferV = (int32_t *) calloc( mConfig.frameLength * sizeof(int32_t), 1 );

        // allocate dynamic predictor buffer
        mPredictor = (int32_t *) calloc( mConfig.frameLength * sizeof(int32_t), 1 );

        // "shift off" buffer shares memory with predictor buffer
        mShiftBuffer = (uint16_t *) mPredictor;
        
		cudaMalloc(&d_u, X * mConfig.frameLength * sizeof(int32_t));
		cudaMalloc(&d_v, X * mConfig.frameLength * sizeof(int32_t));
		cudaMalloc(&d_shiftUV, X * mConfig.frameLength * sizeof(int32_t));
		cudaMalloc(&d_numSamples, X * sizeof(uint32_t));
		cudaMalloc(&d_mixRes, X * sizeof(int32_t));
		cudaMalloc(&d_mixBits, X * sizeof(int32_t));

        RequireAction( (mMixBufferU != nil) && (mMixBufferV != nil) && (mPredictor != nil),
                        status = kALAC_MemFullError; goto Exit; );
     }
    else
    {
        status = kALAC_ParamError;
    }

    // skip to Channel Layout Info
    // theActualCookie += sizeof(ALACSpecificConfig);
    
    // Currently, the Channel Layout Info portion of the magic cookie (as defined in the 
    // ALACMagicCookieDescription.txt document) is unused by the decoder. 
    
Exit:
	return status;
}


__global__ void gpu_unmix16(int32_t * u, int32_t * v, int16_t * out, uint32_t stride, uint32_t * numSamples, int32_t * mixbits, int32_t * mixres, int32_t theOutputPacketBytes, uint32_t frameLength)
{
	int block = blockIdx.x % 4;
	int index = blockIdx.x / 4;
	int z = threadIdx.x + block * blockDim.x;

	if (z < numSamples[index])
	{

		int32_t		l, r;
		int16_t * op = out + (index * theOutputPacketBytes) / 2;

		if (mixres[index] != 0)
		{
			/* matrixed stereo */
			l = (u + index * frameLength)[z] + (v + index * frameLength)[z] - ((mixres[index] * (v + index * frameLength)[z]) >> mixbits[index]);
			r = l - (v + index * frameLength)[z];

			op += stride* z;
			op[0] = (int16_t)l;
			op[1] = (int16_t)r;
		}
		else
		{
			/* Conventional separated stereo. */
			op += stride * z;
			op[0] = (int16_t)(u + index * frameLength)[z];
			op[1] = (int16_t)(v + index * frameLength)[z];
		}
	}
	/*if (block == 0 && index == 0 && z == 0)
		printf("%d\n", mixbits[8]);*/
}


__global__ void gpu_unmix20(int32_t * u, int32_t * v, uint8_t * out, uint32_t stride, uint32_t * numSamples, int32_t * mixbits, int32_t * mixres, int32_t theOutputPacketBytes, uint32_t frameLength)
{
	int block = blockIdx.x % 4;
	int index = blockIdx.x / 4;
	int z = threadIdx.x + block * blockDim.x;

	if (z < numSamples[index])
	{

		int32_t		l, r;
		uint8_t * op = out + (index * theOutputPacketBytes) / 2;

		if (mixres[index] != 0)
		{
			/* matrixed stereo */
			l = (u + index * frameLength)[z] + (v + index * frameLength)[z] - ((mixres[index] * (v + index * frameLength)[z]) >> mixbits[index]);
			r = l - (v + index * frameLength)[z];

			l <<= 4;
			r <<= 4;

			op += 3 * z;
			op += (stride - 1) * 3 * z;
			op[HBYTE] = (uint8_t)((l >> 16) & 0xffu);
			op[MBYTE] = (uint8_t)((l >> 8) & 0xffu);
			op[LBYTE] = (uint8_t)((l >> 0) & 0xffu);
			op += 3;

			op[HBYTE] = (uint8_t)((r >> 16) & 0xffu);
			op[MBYTE] = (uint8_t)((r >> 8) & 0xffu);
			op[LBYTE] = (uint8_t)((r >> 0) & 0xffu);
		}
		else
		{
			/* Conventional separated stereo. */
			int32_t		val;

			val = (u + index * frameLength)[z] << 4;
			op += 3 * z;
			op += (stride - 1) * 3 * z;
			op[HBYTE] = (uint8_t)((val >> 16) & 0xffu);
			op[MBYTE] = (uint8_t)((val >> 8) & 0xffu);
			op[LBYTE] = (uint8_t)((val >> 0) & 0xffu);
			op += 3;

			val = (v + index * frameLength)[z] << 4;
			op[HBYTE] = (uint8_t)((val >> 16) & 0xffu);
			op[MBYTE] = (uint8_t)((val >> 8) & 0xffu);
			op[LBYTE] = (uint8_t)((val >> 0) & 0xffu);
		}
	}
}

// 24-bit routines
// - the 24 bits of data are right-justified in the input/output predictor buffers

__global__ void gpu_unmix24(int32_t * u, int32_t * v, uint8_t * out, uint32_t stride, uint32_t * numSamples,
	int32_t * mixbits, int32_t * mixres, uint16_t * shiftUV, int32_t bytesShifted, int32_t theOutputPacketBytes, uint32_t frameLength)
{
	int block = blockIdx.x % 4;
	int index = blockIdx.x / 4;
	int z = threadIdx.x + block * blockDim.x;

	if (z < numSamples[index])
	{

		int32_t			shift = bytesShifted * 8;
		int32_t		l, r;
		int32_t k = z * 2;
		uint8_t * op = out + (index * theOutputPacketBytes);

		if (mixres[index] != 0)
		{
			/* matrixed stereo */
			l = (u + index * frameLength)[z] + (v + index * frameLength)[z] - ((mixres[index] * (v + index * frameLength)[z]) >> mixbits[index]);
			r = l - (v + index * frameLength)[z];

			l = (l << shift) | (uint32_t)(shiftUV + index * frameLength * 2)[k + 0];
			r = (r << shift) | (uint32_t)(shiftUV + index * frameLength * 2)[k + 1];

			op += 3 * z;
			op += (stride - 1) * 3 * z;
			op[HBYTE] = (uint8_t)((l >> 16) & 0xffu);
			op[MBYTE] = (uint8_t)((l >> 8) & 0xffu);
			op[LBYTE] = (uint8_t)((l >> 0) & 0xffu);
			op += 3;

			op[HBYTE] = (uint8_t)((r >> 16) & 0xffu);
			op[MBYTE] = (uint8_t)((r >> 8) & 0xffu);
			op[LBYTE] = (uint8_t)((r >> 0) & 0xffu);
		}
		else
		{
			/* Conventional separated stereo. */
			l = (u + index * frameLength)[z];
			r = (v + index * frameLength)[z];

			l = (l << shift) | (uint32_t)(shiftUV + index * frameLength * 2)[k + 0];
			r = (r << shift) | (uint32_t)(shiftUV + index * frameLength * 2)[k + 1];

			op += 3 * z;
			op += (stride - 1) * 3 * z;
			op[HBYTE] = (uint8_t)((l >> 16) & 0xffu);
			op[MBYTE] = (uint8_t)((l >> 8) & 0xffu);
			op[LBYTE] = (uint8_t)((l >> 0) & 0xffu);
			op += 3;

			op[HBYTE] = (uint8_t)((r >> 16) & 0xffu);
			op[MBYTE] = (uint8_t)((r >> 8) & 0xffu);
			op[LBYTE] = (uint8_t)((r >> 0) & 0xffu);
		}
	}
}


__global__ void gpu_unmix32(int32_t * u, int32_t * v, int32_t * out, uint32_t stride, uint32_t * numSamples,
	int32_t * mixbits, int32_t * mixres, uint16_t * shiftUV, int32_t bytesShifted, int32_t theOutputPacketBytes, uint32_t frameLength)
{
	int block = blockIdx.x % 4;
	int index = blockIdx.x / 4;
	int z = threadIdx.x + block * blockDim.x;
	if (z < numSamples[index])
	{
		int32_t			shift = bytesShifted * 8;
		int32_t		l, r;
		int32_t k = z * 2;
		int32_t * op = out + (index * theOutputPacketBytes)/4;

		if (mixres[index] != 0)
		{
			//Assert( bytesShifted != 0 );

			/* matrixed stereo with shift */
			int32_t		lt, rt;

			lt = (u + index * frameLength)[z];
			rt = (v + index * frameLength)[z];

			l = lt + rt - ((mixres[index] * rt) >> mixbits[index]);
			r = l - rt;

			op += stride * z;
			op[0] = (l << shift) | (uint32_t)(shiftUV + index * frameLength * 2)[k + 0];
			op[1] = (r << shift) | (uint32_t)(shiftUV + index * frameLength * 2)[k + 1];
		}
		else
		{
			/* interleaving with shift */
			op += stride * z;
			op[0] = ((u + index * frameLength)[z] << shift) | (uint32_t)(shiftUV + index * frameLength * 2)[k + 0];
			op[1] = ((v + index * frameLength)[z] << shift) | (uint32_t)(shiftUV + index * frameLength * 2)[k + 1];

		}
	}
}

__global__ void gpu_copyPredictorTo16(int32_t * in, int16_t * out, uint32_t stride, uint32_t * numSamples, int32_t theOutputPacketBytes, uint32_t frameLength)
{
	int block = blockIdx.x % 4;
	int index = blockIdx.x / 4;
	int z = threadIdx.x + block * blockDim.x;

	if (z < numSamples[index])
	{
		int16_t * op = out + (index * theOutputPacketBytes)/2;
		
		op[z] = (int16_t)(in + index * frameLength)[z];
	}
}

__global__ void gpu_copyPredictorTo20(int32_t * in, uint8_t * out, uint32_t stride, uint32_t * numSamples, int32_t theOutputPacketBytes, uint32_t frameLength)
{
	int block = blockIdx.x % 4;
	int index = blockIdx.x / 4;
	int z = threadIdx.x + block * blockDim.x;

	if (z < numSamples[index])
	{
		uint8_t * op = out + (index * theOutputPacketBytes);
		int32_t	val = (in + index * frameLength)[z];
		op += (stride * 3 * z);

		op[HBYTE] = (uint8_t)((val >> 12) & 0xffu);
		op[MBYTE] = (uint8_t)((val >> 4) & 0xffu);
		op[LBYTE] = (uint8_t)((val << 4) & 0xffu);
	}
}

__global__ void gpu_copyPredictorTo24(int32_t * in, uint8_t * out, uint32_t stride, uint32_t * numSamples, int32_t theOutputPacketBytes, uint32_t frameLength)
{
	int block = blockIdx.x % 4;
	int index = blockIdx.x / 4;
	int z = threadIdx.x + block * blockDim.x;

	if (z < numSamples[index])
	{
		uint8_t * op = out + (index * theOutputPacketBytes);
		int32_t	val = (in + index * frameLength)[z];
		op += (stride * 3 * z);

		op[HBYTE] = (uint8_t)((val >> 16) & 0xffu);
		op[MBYTE] = (uint8_t)((val >> 8) & 0xffu);
		op[LBYTE] = (uint8_t)((val >> 0) & 0xffu);
	}
}

__global__ void gpu_copyPredictorTo24Shift(int32_t * in, uint16_t * shift, uint8_t * out, uint32_t stride, uint32_t * numSamples, int32_t bytesShifted, int32_t theOutputPacketBytes, uint32_t frameLength)
{

	int block = blockIdx.x % 4;
	int index = blockIdx.x / 4;
	int z = threadIdx.x + block * blockDim.x;

	if (z < numSamples[index])
	{
		uint8_t * op = out + (index * theOutputPacketBytes);
		int32_t	shiftVal = bytesShifted * 8;

		//Assert( bytesShifted != 0 );

		int32_t	val = (in + index * frameLength)[z];

		val = (val << shiftVal) | (uint32_t)(shift + index * frameLength * 2 )[z];


		op += (stride * 3 * z);
		op[HBYTE] = (uint8_t)((val >> 16) & 0xffu);
		op[MBYTE] = (uint8_t)((val >> 8) & 0xffu);
		op[LBYTE] = (uint8_t)((val >> 0) & 0xffu);
	}
}

__global__ void gpu_copyPredictorTo32(int32_t * in, int32_t * out, uint32_t stride, uint32_t * numSamples, int32_t theOutputPacketBytes, uint32_t frameLength)
{
	int block = blockIdx.x % 4;
	int index = blockIdx.x / 4;
	int z = threadIdx.x + block * blockDim.x;

	if (z < numSamples[index])
	{
		int32_t * op = out + (index * theOutputPacketBytes)/4;
		op[stride * z] = (in + index * frameLength)[z];

	}
}

__global__ void gpu_copyPredictorTo32Shift(int32_t * in, uint16_t * shift, int32_t * out, uint32_t stride, uint32_t * numSamples, int32_t bytesShifted, int32_t theOutputPacketBytes, uint32_t frameLength)
{

	int block = blockIdx.x % 4;
	int index = blockIdx.x / 4;
	int z = threadIdx.x + block * blockDim.x;

	if (z < numSamples[index])
	{
		int32_t * op = out + (index * theOutputPacketBytes)/4;
		int32_t	shiftVal = bytesShifted * 8;

		//Assert( bytesShifted != 0 );

		op += stride * z;
		op[0] = ((in + index * frameLength)[z] << shiftVal) | (uint32_t)(shift + index * frameLength * 2)[z];

	}
}

void ALACDecoder::fillWriteBuffer(void * sampleBuffer, uint32_t numChannels, int32_t theOutputPacketBytes, int X){

	switch (tag){

		case ID_SCE:
		case ID_LFE:
		{
			switch (mConfig.bitDepth)
			{
				case 16:
					gpu_copyPredictorTo16 << <((total_numSamples + SIZE - 1) / SIZE), SIZE >> >(d_u, (int16_t *)sampleBuffer,
						numChannels, d_numSamples, theOutputPacketBytes, mConfig.frameLength);
					break;
				case 20:
					gpu_copyPredictorTo20 << <((total_numSamples + SIZE - 1) / SIZE), SIZE >> >(d_u, (uint8_t *)sampleBuffer,
						numChannels, d_numSamples, theOutputPacketBytes, mConfig.frameLength);
					break;
				case 24:
					if (bytesShifted != 0){
						gpu_copyPredictorTo24Shift << <((total_numSamples + SIZE - 1) / SIZE), SIZE >> >(d_u, d_shiftUV, (uint8_t *)sampleBuffer,
							numChannels, d_numSamples, (int32_t)bytesShifted, theOutputPacketBytes, mConfig.frameLength);
					}
					else{
						gpu_copyPredictorTo24 << <((total_numSamples + SIZE - 1) / SIZE), SIZE >> >(d_u, (uint8_t *)sampleBuffer,
							numChannels, d_numSamples, theOutputPacketBytes, mConfig.frameLength);
					}
					break;
				case 32:
					if (bytesShifted != 0){
						gpu_copyPredictorTo32Shift << <((total_numSamples + SIZE - 1) / SIZE), SIZE >> >(d_u, d_shiftUV, (int32_t *)sampleBuffer,
						numChannels, d_numSamples, (int32_t)bytesShifted, theOutputPacketBytes, mConfig.frameLength);
					}
					else{
						gpu_copyPredictorTo32 << <((total_numSamples + SIZE - 1) / SIZE), SIZE >> >(d_u, (int32_t *)sampleBuffer,
						numChannels, d_numSamples, theOutputPacketBytes, mConfig.frameLength);
					}
					break;
			}
			break;
		}
		case ID_CPE:
		{
			switch (mConfig.bitDepth)
			{
				case 16:
					gpu_unmix16 << < ((total_numSamples + SIZE - 1) / SIZE), SIZE >> >(d_u, d_v, (int16_t *)sampleBuffer,
						numChannels, d_numSamples, d_mixBits, d_mixRes, theOutputPacketBytes, mConfig.frameLength);
					break;
				case 20:
					gpu_unmix20 << < ((total_numSamples + SIZE - 1) / SIZE), SIZE >> >(d_u, d_v, (uint8_t *)sampleBuffer,
						numChannels, d_numSamples, d_mixBits, d_mixRes, theOutputPacketBytes, mConfig.frameLength);
					break;
				case 24:
					gpu_unmix24 << <((total_numSamples + SIZE - 1) / SIZE), SIZE >> >(d_u, d_v, (uint8_t *)sampleBuffer,
						numChannels, d_numSamples, d_mixBits, d_mixRes, d_shiftUV, (int32_t)bytesShifted, theOutputPacketBytes, mConfig.frameLength);
					//cudaMemcpy(out24, d_out, numChannels * 3 * numSamples * sizeof(uint8_t), cudaMemcpyDeviceToHost);
					break;
				case 32:
					gpu_unmix32 << <((total_numSamples + SIZE - 1) / SIZE), SIZE >> >(d_u, d_v, (int32_t *)sampleBuffer,
						numChannels, d_numSamples, d_mixBits, d_mixRes, d_shiftUV, (int32_t)bytesShifted, theOutputPacketBytes, mConfig.frameLength);
					break;
			}
			break;
		}

	}

}


/*
	Decode()
	- the decoded samples are interleaved into the output buffer in the order they arrive in
	  the bitstream
*/
int32_t ALACDecoder::Decode(BitBuffer * bits, uint32_t numSamples, uint32_t numChannels, uint32_t * outNumSamples, uint32_t mBytesPerFrame, int X)
{
	BitBuffer			shiftBits;
	uint32_t            bits1, bits2;
	uint8_t				elementInstanceTag;
	AGParamRec			agParams;
	uint32_t				channelIndex;
	int16_t				coefsU[32];		// max possible size is 32 although NUMCOEPAIRS is the current limit
	int16_t				coefsV[32];
	uint8_t				numU, numV;
	uint8_t				mixBits;
	int8_t				mixRes;
	uint16_t			unusedHeader;
	uint8_t				escapeFlag;
	uint32_t			chanBits;
	//uint8_t				bytesShifted;
	uint32_t			shift;
	uint8_t				modeU, modeV;
	uint32_t			denShiftU, denShiftV;
	uint16_t			pbFactorU, pbFactorV;
	uint16_t			pb;
	int16_t *			out16;
	uint8_t *			out20;
	uint8_t *			out24;
	int32_t *			out32;
	uint8_t				headerByte;
	uint8_t				partialFrame;
	uint32_t			extraBits;
	int32_t				val;
	uint32_t			i, j;
	int32_t             status;
	
	//RequireAction( (bits != nil) && (sampleBuffer != nil) && (outNumSamples != nil), return kALAC_ParamError; );
	RequireAction( numChannels > 0, return kALAC_ParamError; );

	mActiveElements = 0;
	channelIndex	= 0;

	status = ALAC_noErr;
	*outNumSamples = numSamples;

	while ( status == ALAC_noErr )
	{
		// bail if we ran off the end of the buffer
    	RequireAction( bits->cur < bits->end, status = kALAC_ParamError; goto Exit; );

		// copy global decode params for this element
		pb = mConfig.pb;

		// read element tag
		tag = BitBufferReadSmall( bits, 3 );
		switch ( tag )
		{
			case ID_SCE:
			case ID_LFE:
			{
				// mono/LFE channel
				elementInstanceTag = BitBufferReadSmall( bits, 4 );
				mActiveElements |= (1u << elementInstanceTag);

				// read the 12 unused header bits
				unusedHeader = (uint16_t) BitBufferRead( bits, 12 );
				RequireAction( unusedHeader == 0, status = kALAC_ParamError; goto Exit; );

				// read the 1-bit "partial frame" flag, 2-bit "shift-off" flag & 1-bit "escape" flag
				headerByte = (uint8_t) BitBufferRead( bits, 4 );
				
				partialFrame = headerByte >> 3;
				
				bytesShifted = (headerByte >> 1) & 0x3u;
				RequireAction( bytesShifted != 3, status = kALAC_ParamError; goto Exit; );

				shift = bytesShifted * 8;

				escapeFlag = headerByte & 0x1;

				chanBits = mConfig.bitDepth - (bytesShifted * 8);
				
				// check for partial frame to override requested numSamples
				if ( partialFrame != 0 )
				{
					numSamples  = BitBufferRead( bits, 16 ) << 16;
					numSamples |= BitBufferRead( bits, 16 );
				}

				if ( escapeFlag == 0 )
				{
					// compressed frame, read rest of parameters
					mixBits	= (uint8_t) BitBufferRead( bits, 8 );
					mixRes	= (int8_t) BitBufferRead( bits, 8 );
					//Assert( (mixBits == 0) && (mixRes == 0) );		// no mixing for mono

					headerByte	= (uint8_t) BitBufferRead( bits, 8 );
					modeU		= headerByte >> 4;
					denShiftU	= headerByte & 0xfu;
					
					headerByte	= (uint8_t) BitBufferRead( bits, 8 );
					pbFactorU	= headerByte >> 5;
					numU		= headerByte & 0x1fu;

					for ( i = 0; i < numU; i++ )
						coefsU[i] = (int16_t) BitBufferRead( bits, 16 );
					
					// if shift active, skip the the shift buffer but remember where it starts
					if ( bytesShifted != 0 )
					{
						shiftBits = *bits;
						BitBufferAdvance( bits, (bytesShifted * 8) * numSamples ); 
					}

					// decompress
					set_ag_params( &agParams, mConfig.mb, (pb * pbFactorU) / 4, mConfig.kb, numSamples, numSamples, mConfig.maxRun );
					status = dyn_decomp( &agParams, bits, mPredictor, numSamples, chanBits, &bits1 );
					RequireNoErr( status, goto Exit; );

					if ( modeU == 0 )
					{
						unpc_block( mPredictor, mMixBufferU, numSamples, &coefsU[0], numU, chanBits, denShiftU );
					}
					else
					{
						// the special "numActive == 31" mode can be done in-place
						unpc_block( mPredictor, mPredictor, numSamples, nil, 31, chanBits, 0 );
						unpc_block( mPredictor, mMixBufferU, numSamples, &coefsU[0], numU, chanBits, denShiftU );
					}
				}
				else
				{
					//Assert( bytesShifted == 0 );

					// uncompressed frame, copy data into the mix buffer to use common output code
					shift = 32 - chanBits;
					if ( chanBits <= 16 )
					{
						for ( i = 0; i < numSamples; i++ )
						{
							val = (int32_t) BitBufferRead( bits, (uint8_t) chanBits );
							val = (val << shift) >> shift;
							mMixBufferU[i] = val;
						}
					}
					else
					{
						// BitBufferRead() can't read more than 16 bits at a time so break up the reads
						extraBits = chanBits - 16;
						for ( i = 0; i < numSamples; i++ )
						{
							val = (int32_t) BitBufferRead( bits, 16 );
							val = (val << 16) >> shift;
							mMixBufferU[i] = val | BitBufferRead( bits, (uint8_t) extraBits );
						}
					}

					mixBits = mixRes = 0;
					bits1 = chanBits * numSamples;
					bytesShifted = 0;
				}

				// now read the shifted values into the shift buffer
				if ( bytesShifted != 0 )
				{
					shift = bytesShifted * 8;
					//Assert( shift <= 16 );

					for ( i = 0; i < numSamples; i++ )
						mShiftBuffer[i] = (uint16_t) BitBufferRead( &shiftBits, (uint8_t) shift );
				}

				// convert 32-bit integers into output buffer

				cudaMemcpy(d_u + (X * mConfig.frameLength), mMixBufferU, mConfig.frameLength * sizeof(int32_t), cudaMemcpyHostToDevice);
				cudaMemcpy(d_shiftUV + (X * mConfig.frameLength * 2), mShiftBuffer, mConfig.frameLength * sizeof(int32_t), cudaMemcpyHostToDevice);
				cudaMemcpy(d_numSamples + X, &numSamples, sizeof(int32_t), cudaMemcpyHostToDevice);

				total_numSamples += numSamples;

				channelIndex += 1;
				*outNumSamples = numSamples;
				break;
			}

			case ID_CPE:
			{
				// if decoding this pair would take us over the max channels limit, bail
				if ( (channelIndex + 2) > numChannels )
					goto NoMoreChannels;

				// stereo channel pair
				elementInstanceTag = BitBufferReadSmall( bits, 4 );
				mActiveElements |= (1u << elementInstanceTag);

				// read the 12 unused header bits
				unusedHeader = (uint16_t) BitBufferRead( bits, 12 );
				RequireAction( unusedHeader == 0, status = kALAC_ParamError; goto Exit; );

				// read the 1-bit "partial frame" flag, 2-bit "shift-off" flag & 1-bit "escape" flag
				headerByte = (uint8_t) BitBufferRead( bits, 4 );
				
				partialFrame = headerByte >> 3;
				
				bytesShifted = (headerByte >> 1) & 0x3u;
				RequireAction( bytesShifted != 3, status = kALAC_ParamError; goto Exit; );

				shift = bytesShifted * 8;

				escapeFlag = headerByte & 0x1;

				chanBits = mConfig.bitDepth - (bytesShifted * 8) + 1;
				
				// check for partial frame length to override requested numSamples
				if ( partialFrame != 0 )
				{
					numSamples  = BitBufferRead( bits, 16 ) << 16;
					numSamples |= BitBufferRead( bits, 16 );
				}

				if ( escapeFlag == 0 )
				{
					// compressed frame, read rest of parameters
					mixBits		= (uint8_t) BitBufferRead( bits, 8 );
					mixRes		= (int8_t) BitBufferRead( bits, 8 );

					headerByte	= (uint8_t) BitBufferRead( bits, 8 );
					modeU		= headerByte >> 4;
					denShiftU	= headerByte & 0xfu;
					
					headerByte	= (uint8_t) BitBufferRead( bits, 8 );
					pbFactorU	= headerByte >> 5;
					numU		= headerByte & 0x1fu;
					for ( i = 0; i < numU; i++ )
						coefsU[i] = (int16_t) BitBufferRead( bits, 16 );

					headerByte	= (uint8_t) BitBufferRead( bits, 8 );
					modeV		= headerByte >> 4;
					denShiftV	= headerByte & 0xfu;
					
					headerByte	= (uint8_t) BitBufferRead( bits, 8 );
					pbFactorV	= headerByte >> 5;
					numV		= headerByte & 0x1fu;
					for ( i = 0; i < numV; i++ )
						coefsV[i] = (int16_t) BitBufferRead( bits, 16 );

					// if shift active, skip the interleaved shifted values but remember where they start
					if ( bytesShifted != 0 )
					{
						shiftBits = *bits;
						BitBufferAdvance( bits, (bytesShifted * 8) * 2 * numSamples );
					}

					// decompress and run predictor for "left" channel
					set_ag_params( &agParams, mConfig.mb, (pb * pbFactorU) / 4, mConfig.kb, numSamples, numSamples, mConfig.maxRun );
					status = dyn_decomp( &agParams, bits, mPredictor, numSamples, chanBits, &bits1 );
					RequireNoErr( status, goto Exit; );

					if ( modeU == 0 )
					{
						unpc_block( mPredictor, mMixBufferU, numSamples, &coefsU[0], numU, chanBits, denShiftU );
					}
					else
					{
						// the special "numActive == 31" mode can be done in-place
						unpc_block( mPredictor, mPredictor, numSamples, nil, 31, chanBits, 0 );
						unpc_block( mPredictor, mMixBufferU, numSamples, &coefsU[0], numU, chanBits, denShiftU );
					}

					// decompress and run predictor for "right" channel
					set_ag_params( &agParams, mConfig.mb, (pb * pbFactorV) / 4, mConfig.kb, numSamples, numSamples, mConfig.maxRun );
					status = dyn_decomp( &agParams, bits, mPredictor, numSamples, chanBits, &bits2 );
					RequireNoErr( status, goto Exit; );

					if ( modeV == 0 )
					{
						unpc_block( mPredictor, mMixBufferV, numSamples, &coefsV[0], numV, chanBits, denShiftV );
					}
					else
					{
						// the special "numActive == 31" mode can be done in-place
						unpc_block( mPredictor, mPredictor, numSamples, nil, 31, chanBits, 0 );
						unpc_block( mPredictor, mMixBufferV, numSamples, &coefsV[0], numV, chanBits, denShiftV );
					}
				}
				else
				{
					//Assert( bytesShifted == 0 );

					// uncompressed frame, copy data into the mix buffers to use common output code
					chanBits = mConfig.bitDepth;
					shift = 32 - chanBits;
					if ( chanBits <= 16 )
					{
						for ( i = 0; i < numSamples; i++ )
						{
							val = (int32_t) BitBufferRead( bits, (uint8_t) chanBits );
							val = (val << shift) >> shift;
							mMixBufferU[i] = val;

							val = (int32_t) BitBufferRead( bits, (uint8_t) chanBits );
							val = (val << shift) >> shift;
							mMixBufferV[i] = val;
						}
					}
					else
					{
						// BitBufferRead() can't read more than 16 bits at a time so break up the reads
						extraBits = chanBits - 16;
						for ( i = 0; i < numSamples; i++ )
						{
							val = (int32_t) BitBufferRead( bits, 16 );
							val = (val << 16) >> shift;
							mMixBufferU[i] = val | BitBufferRead( bits, (uint8_t)extraBits );

							val = (int32_t) BitBufferRead( bits, 16 );
							val = (val << 16) >> shift;
							mMixBufferV[i] = val | BitBufferRead( bits, (uint8_t)extraBits );
						}
					}

					bits1 = chanBits * numSamples;
					bits2 = chanBits * numSamples;
					mixBits = mixRes = 0;
					bytesShifted = 0;
				}

				// now read the shifted values into the shift buffer
				if ( bytesShifted != 0 )
				{
					shift = bytesShifted * 8;
					//Assert( shift <= 16 );

					for ( i = 0; i < (numSamples * 2); i += 2 )
					{
						mShiftBuffer[i + 0] = (uint16_t) BitBufferRead( &shiftBits, (uint8_t) shift );
						mShiftBuffer[i + 1] = (uint16_t) BitBufferRead( &shiftBits, (uint8_t) shift );
					}
				}

				// un-mix the data and convert to output format
				// - note that mixRes = 0 means just interleave so we use that path for uncompressed frames

				//printf("%d\n", bytesShifted);

				cudaMemcpy(d_u + (X * mConfig.frameLength), mMixBufferU, mConfig.frameLength * sizeof(int32_t), cudaMemcpyHostToDevice);
				cudaMemcpy(d_v + (X * mConfig.frameLength), mMixBufferV, mConfig.frameLength * sizeof(int32_t), cudaMemcpyHostToDevice);
				cudaMemcpy(d_shiftUV + (X * mConfig.frameLength * 2), mShiftBuffer, mConfig.frameLength * sizeof(int32_t), cudaMemcpyHostToDevice);
				cudaMemcpy(d_numSamples + X, &numSamples, sizeof(int32_t), cudaMemcpyHostToDevice);
				cudaMemcpy(d_mixRes + X, &mixRes, sizeof(int8_t), cudaMemcpyHostToDevice);
				cudaMemcpy(d_mixBits + X, &mixBits, sizeof(uint8_t), cudaMemcpyHostToDevice);

				total_numSamples += numSamples;

				channelIndex += 2;
				*outNumSamples = numSamples;
				break;
			}

			case ID_CCE:
			case ID_PCE:
			{
				// unsupported element, bail
				//AssertNoErr( tag );
				status = kALAC_ParamError;
				break;
			}

			case ID_DSE:
			{
				// data stream element -- parse but ignore
				status = this->DataStreamElement( bits );
				break;
			}
			
			case ID_FIL:
			{
				// fill element -- parse but ignore
				status = this->FillElement( bits );
				break;
			}

			case ID_END:
			{
				// frame end, all done so byte align the frame and check for overruns
				BitBufferByteAlign( bits, false );
				//Assert( bits->cur == bits->end );
				goto Exit;
			}
		}

#if ! DEBUG
		// if we've decoded all of our channels, bail (but not in debug b/c we want to know if we're seeing bad bits)
		// - this also protects us if the config does not match the bitstream or crap data bits follow the audio bits
		if ( channelIndex >= numChannels )
			break;
#endif
	}

NoMoreChannels:

	// if we get here and haven't decoded all of the requested channels, fill the remaining channels with zeros
	for ( ; channelIndex < numChannels; channelIndex++ )
	{
		printf("comes here also!");
		/*switch ( mConfig.bitDepth )
		{
			case 16:
			{
				int16_t *	fill16 = &((int16_t *)sampleBuffer)[channelIndex];
				Zero16( fill16, numSamples, numChannels );
				break;
			}
			case 24:
			{
				uint8_t *	fill24 = (uint8_t *)sampleBuffer + (channelIndex * 3);
				Zero24( fill24, numSamples, numChannels );
				break;
			}
			case 32:
			{
				int32_t *	fill32 = &((int32_t *)sampleBuffer)[channelIndex];
				Zero32( fill32, numSamples, numChannels );
				break;
			}
		}*/
	}

Exit:
	return status;
}

#if PRAGMA_MARK
#pragma mark -
#endif

/*
	FillElement()
	- they're just filler so we don't need 'em
*/
int32_t ALACDecoder::FillElement( BitBuffer * bits )
{
	int16_t		count;
	
	// 4-bit count or (4-bit + 8-bit count) if 4-bit count == 15
	// - plus this weird -1 thing I still don't fully understand
	count = BitBufferReadSmall( bits, 4 );
	if ( count == 15 )
		count += (int16_t) BitBufferReadSmall( bits, 8 ) - 1;

	BitBufferAdvance( bits, count * 8 );

	RequireAction( bits->cur <= bits->end, return kALAC_ParamError; );

	return ALAC_noErr;	
}

/*
	DataStreamElement()
	- we don't care about data stream elements so just skip them
*/
int32_t ALACDecoder::DataStreamElement( BitBuffer * bits )
{
	uint8_t		element_instance_tag;
	int32_t		data_byte_align_flag;
	uint16_t		count;
	
	// the tag associates this data stream element with a given audio element
	element_instance_tag = BitBufferReadSmall( bits, 4 );
	
	data_byte_align_flag = BitBufferReadOne( bits );

	// 8-bit count or (8-bit + 8-bit count) if 8-bit count == 255
	count = BitBufferReadSmall( bits, 8 );
	if ( count == 255 )
		count += BitBufferReadSmall( bits, 8 );

	// the align flag means the bitstream should be byte-aligned before reading the following data bytes
	if ( data_byte_align_flag )
		BitBufferByteAlign( bits, false );

	// skip the data bytes
	BitBufferAdvance( bits, count * 8 );

	RequireAction( bits->cur <= bits->end, return kALAC_ParamError; );

	return ALAC_noErr;
}

/*
	ZeroN()
	- helper routines to clear out output channel buffers when decoding fewer channels than requested
*/
static void Zero16( int16_t * buffer, uint32_t numItems, uint32_t stride )
{
	if ( stride == 1 )
	{
		memset( buffer, 0, numItems * sizeof(int16_t) );
	}
	else
	{
		for ( uint32_t index = 0; index < (numItems * stride); index += stride )
			buffer[index] = 0;
	}
}

static void Zero24( uint8_t * buffer, uint32_t numItems, uint32_t stride )
{
	if ( stride == 1 )
	{
		memset( buffer, 0, numItems * 3 );
	}
	else
	{
		for ( uint32_t index = 0; index < (numItems * stride * 3); index += (stride * 3) )
		{
			buffer[index + 0] = 0;
			buffer[index + 1] = 0;
			buffer[index + 2] = 0;
		}
	}
}

static void Zero32( int32_t * buffer, uint32_t numItems, uint32_t stride )
{
	if ( stride == 1 )
	{
		memset( buffer, 0, numItems * sizeof(int32_t) );
	}
	else
	{
		for ( uint32_t index = 0; index < (numItems * stride); index += stride )
			buffer[index] = 0;
	}
}
