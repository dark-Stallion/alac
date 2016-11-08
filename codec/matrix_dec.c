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
File:		matrix_dec.c

Contains:	ALAC mixing/matrixing decode routines.

Copyright:	(c) 2004-2011 Apple, Inc.
*/

#include "matrixlib.h"
#include "ALACAudioTypes.h"

#include <stdio.h>
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

//ALL METHODS HAVE BEEN MOVED TO ALACDecoder.cu FOR PARALLEL PURPOSE

// 16-bit routines

// 20-bit routines
// - the 20 bits of data are left-justified in 3 bytes of storage but right-aligned for input/output predictor buffers


// 24-bit routines
// - the 24 bits of data are right-justified in the input/output predictor buffers


// 32-bit routines
// - note that these really expect the internal data width to be < 32 but the arrays are 32-bit
// - otherwise, the calculations might overflow into the 33rd bit and be lost
// - therefore, these routines deal with the specified "unused lower" bytes in the "shift" buffers


// 20/24-bit <-> 32-bit helper routines (not really matrixing but convenient to put here)

