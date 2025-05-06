/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2024 Yermalayeu Ihar.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/
#ifndef __SimdAmxBf16_h__
#define __SimdAmxBf16_h__

#include "Simd/SimdDefs.h"

namespace Simd
{
#ifdef SIMD_AMXBF16_ENABLE    
    namespace AmxBf16
    {
        void DeinterleaveUv(const uint8_t* uv, size_t uvStride, size_t width, size_t height, uint8_t* u, size_t uStride, uint8_t* v, size_t vStride);

        void DeinterleaveBgr(const uint8_t* bgr, size_t bgrStride, size_t width, size_t height, uint8_t* b, size_t bStride, uint8_t* g, size_t gStride, uint8_t* r, size_t rStride);

        void DeinterleaveBgra(const uint8_t* bgra, size_t bgraStride, size_t width, size_t height, uint8_t* b, size_t bStride, uint8_t* g, size_t gStride, uint8_t* r, size_t rStride, uint8_t* a, size_t aStride);

        void Float32ToBFloat16(const float* src, size_t size, uint16_t* dst);

        void ChangeColors(const uint8_t* src, size_t srcStride, size_t width, size_t height, const uint8_t* colors, uint8_t* dst, size_t dstStride);

        void NormalizeHistogram(const uint8_t* src, size_t srcStride, size_t width, size_t height, uint8_t* dst, size_t dstStride);
    }
#endif
}
#endif
