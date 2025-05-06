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
#include "Simd/SimdSynet.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdSse41.h"
#include "Simd/SimdAvx2.h"
#include "Simd/SimdArray.h"
#include "Simd/SimdPow.h"
#include "Simd/SimdExp.h"
#include "Simd/SimdErf.h"
#include "Simd/SimdPerformance.h"
#include "Simd/SimdGather.h"
#include "Simd/SimdTrigonometric.h"

namespace Simd
{
#if defined(SIMD_AVX2_ENABLE) && defined(SIMD_SYNET_ENABLE)    
    namespace Avx2
    {
        template<SimdSynetUnaryOperation32fType type> __m256 SynetUnaryOperation32f(__m256 value);

        template<> SIMD_INLINE __m256 SynetUnaryOperation32f<SimdSynetUnaryOperation32fAbs>(__m256 value)
        {
            return _mm256_andnot_ps(_mm256_set1_ps(-0.0f), value);
        }

        template<> SIMD_INLINE __m256 SynetUnaryOperation32f<SimdSynetUnaryOperation32fCeil>(__m256 value)
        {
            return _mm256_ceil_ps(value);
        }

        template<> SIMD_INLINE __m256 SynetUnaryOperation32f<SimdSynetUnaryOperation32fCos>(__m256 value)
        {
            return Cos(value);
        }

        template<> SIMD_INLINE __m256 SynetUnaryOperation32f<SimdSynetUnaryOperation32fErf>(__m256 value)
        {
            return Erf(value);
        }

        template<> SIMD_INLINE __m256 SynetUnaryOperation32f<SimdSynetUnaryOperation32fExp>(__m256 value)
        {
            return Exponent(value);
        }

        template<> SIMD_INLINE __m256 SynetUnaryOperation32f<SimdSynetUnaryOperation32fFloor>(__m256 value)
        {
            return _mm256_floor_ps(value);
        }

        template<> SIMD_INLINE __m256 SynetUnaryOperation32f<SimdSynetUnaryOperation32fLog>(__m256 value)
        {
            return Logarithm(value);
        }

        template<> SIMD_INLINE __m256 SynetUnaryOperation32f<SimdSynetUnaryOperation32fNeg>(__m256 value)
        {
            return _mm256_sub_ps(_mm256_setzero_ps(), value);
        }

        template<> SIMD_INLINE __m256 SynetUnaryOperation32f<SimdSynetUnaryOperation32fNot>(__m256 value)
        {
            return Not(value);
        }

        template<> SIMD_INLINE __m256 SynetUnaryOperation32f<SimdSynetUnaryOperation32fRcp>(__m256 value)
        {
            return Rcp<false>(value);
        }

        template<> SIMD_INLINE __m256 SynetUnaryOperation32f<SimdSynetUnaryOperation32fRsqrt>(__m256 value)
        {
            return _mm256_rsqrt_ps(value);
        }

        template<> SIMD_INLINE __m256 SynetUnaryOperation32f<SimdSynetUnaryOperation32fSin>(__m256 value)
        {
            return Sin(value);
        }

        template<> SIMD_INLINE __m256 SynetUnaryOperation32f<SimdSynetUnaryOperation32fSqrt>(__m256 value)
        {
            return _mm256_sqrt_ps(value);
        }

        template<> SIMD_INLINE __m256 SynetUnaryOperation32f<SimdSynetUnaryOperation32fTanh>(__m256 value)
        {
            return Tanh(value);
        }

        template<> SIMD_INLINE __m256 SynetUnaryOperation32f<SimdSynetUnaryOperation32fZero>(__m256 value)
        {
            return _mm256_setzero_ps();
        }

        template<SimdSynetUnaryOperation32fType type, bool align> void SynetUnaryOperation32f(const float* src, size_t size, float* dst)
        {
            size_t sizeF = AlignLo(size, F);
            size_t sizeQF = AlignLo(size, QF);
            size_t i = 0;
            for (; i < sizeQF; i += QF)
            {
                Store<align>(dst + i + 0 * F, SynetUnaryOperation32f<type>(Load<align>(src + i + 0 * F)));
                Store<align>(dst + i + 1 * F, SynetUnaryOperation32f<type>(Load<align>(src + i + 1 * F)));
                Store<align>(dst + i + 2 * F, SynetUnaryOperation32f<type>(Load<align>(src + i + 2 * F)));
                Store<align>(dst + i + 3 * F, SynetUnaryOperation32f<type>(Load<align>(src + i + 3 * F)));
            }
            for (; i < sizeF; i += F)
                Store<align>(dst + i, SynetUnaryOperation32f<type>(Load<align>(src + i)));
            for (; i < size; ++i)
                dst[i] = Base::SynetUnaryOperation32f<type>(src[i]);
        }

        template<bool align> void SynetUnaryOperation32f(const float* src, size_t size, SimdSynetUnaryOperation32fType type, float* dst)
        {
            switch (type)
            {
            case SimdSynetUnaryOperation32fAbs: SynetUnaryOperation32f<SimdSynetUnaryOperation32fAbs, align>(src, size, dst); break;
            case SimdSynetUnaryOperation32fCeil: SynetUnaryOperation32f<SimdSynetUnaryOperation32fCeil, align>(src, size, dst); break;
            case SimdSynetUnaryOperation32fCos: SynetUnaryOperation32f<SimdSynetUnaryOperation32fCos, align>(src, size, dst); break;
            case SimdSynetUnaryOperation32fExp: SynetUnaryOperation32f<SimdSynetUnaryOperation32fExp, align>(src, size, dst); break;
            case SimdSynetUnaryOperation32fErf: SynetUnaryOperation32f<SimdSynetUnaryOperation32fErf, align>(src, size, dst); break;
            case SimdSynetUnaryOperation32fFloor: SynetUnaryOperation32f<SimdSynetUnaryOperation32fFloor, align>(src, size, dst); break;
            case SimdSynetUnaryOperation32fLog: SynetUnaryOperation32f<SimdSynetUnaryOperation32fLog, align>(src, size, dst); break;
            case SimdSynetUnaryOperation32fNeg: SynetUnaryOperation32f<SimdSynetUnaryOperation32fNeg, align>(src, size, dst); break;
            case SimdSynetUnaryOperation32fNot: SynetUnaryOperation32f<SimdSynetUnaryOperation32fNot, align>(src, size, dst); break;
            case SimdSynetUnaryOperation32fRcp: SynetUnaryOperation32f<SimdSynetUnaryOperation32fRcp, align>(src, size, dst); break;
            case SimdSynetUnaryOperation32fRsqrt: SynetUnaryOperation32f<SimdSynetUnaryOperation32fRsqrt, align>(src, size, dst); break;
            case SimdSynetUnaryOperation32fSin: SynetUnaryOperation32f<SimdSynetUnaryOperation32fSin, align>(src, size, dst); break;
            case SimdSynetUnaryOperation32fSqrt: SynetUnaryOperation32f<SimdSynetUnaryOperation32fSqrt, align>(src, size, dst); break;
            case SimdSynetUnaryOperation32fTanh: SynetUnaryOperation32f<SimdSynetUnaryOperation32fTanh, align>(src, size, dst); break;
            case SimdSynetUnaryOperation32fZero: SynetUnaryOperation32f<SimdSynetUnaryOperation32fZero, align>(src, size, dst); break;
            default:
                Sse41::SynetUnaryOperation32f(src, size, type, dst);
            }
        }

        void SynetUnaryOperation32f(const float* src, size_t size, SimdSynetUnaryOperation32fType type, float* dst)
        {
            if (Aligned(src) && Aligned(dst))
                SynetUnaryOperation32f<true>(src, size, type, dst);
            else
                SynetUnaryOperation32f<false>(src, size, type, dst);
        }
    }
#endif// SIMD_AVX2_ENABLE
}
