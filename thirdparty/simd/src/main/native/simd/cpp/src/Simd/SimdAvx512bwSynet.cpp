/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2025 Yermalayeu Ihar.
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
#include "Simd/SimdMemory.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdSse41.h"
#include "Simd/SimdAvx2.h"
#include "Simd/SimdAvx512bw.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdExp.h"
#include "Simd/SimdPow.h"
#include "Simd/SimdInterleave.h"
#include "Simd/SimdDeinterleave.h"
#include "Simd/SimdBFloat16.h"

namespace Simd
{
#if defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_SYNET_ENABLE)     
    namespace Avx512bw
    {
        void SynetChannelSum16b(const uint16_t* src, size_t channels, size_t spatial, SimdTensorFormatType format, float* sum)
        {
            if (format == SimdTensorFormatNhwc)
            {
                size_t channels32 = AlignLo(channels, 32), spatial4 = AlignLo(spatial, 4);
                __mmask32 tail = TailMask32(channels - channels32);
                __mmask16 tail0 = TailMask16(channels - channels32), tail1 = TailMask16(channels - channels32 - 16);
                size_t c = 0;
                for (; c < channels32; c += 32)
                {
                    _mm512_storeu_ps(sum + c + 0x00, _mm512_setzero_ps());
                    _mm512_storeu_ps(sum + c + 0x10, _mm512_setzero_ps());
                }
                if (c < channels)
                {
                    _mm512_mask_storeu_ps(sum + c + 0x00, tail0, _mm512_setzero_ps());
                    _mm512_mask_storeu_ps(sum + c + 0x10, tail1, _mm512_setzero_ps());
                }
                size_t s = 0;
                for (; s < spatial4; s += 4)
                {
                    const uint16_t* src0 = src + 0 * channels;
                    const uint16_t* src1 = src + 1 * channels;
                    const uint16_t* src2 = src + 2 * channels;
                    const uint16_t* src3 = src + 3 * channels;
                    size_t c = 0;
                    for (; c < channels32; c += 32)
                    {
                        __m512 sum0 = _mm512_loadu_ps(sum + c + 0x00);
                        __m512 sum1 = _mm512_loadu_ps(sum + c + 0x10);
                        __m512i _src = _mm512_loadu_si512((__m512i*)(src0 + c));
                        sum0 = _mm512_add_ps(sum0, BFloat16ToFloat32<0>(_src));
                        sum1 = _mm512_add_ps(sum1, BFloat16ToFloat32<1>(_src));
                        _src = _mm512_loadu_si512((__m512i*)(src1 + c));
                        sum0 = _mm512_add_ps(sum0, BFloat16ToFloat32<0>(_src));
                        sum1 = _mm512_add_ps(sum1, BFloat16ToFloat32<1>(_src));
                        _src = _mm512_loadu_si512((__m512i*)(src2 + c));
                        sum0 = _mm512_add_ps(sum0, BFloat16ToFloat32<0>(_src));
                        sum1 = _mm512_add_ps(sum1, BFloat16ToFloat32<1>(_src));
                        _src = _mm512_loadu_si512((__m512i*)(src3 + c));
                        sum0 = _mm512_add_ps(sum0, BFloat16ToFloat32<0>(_src));
                        sum1 = _mm512_add_ps(sum1, BFloat16ToFloat32<1>(_src));
                        _mm512_storeu_ps(sum + c + 0x00, sum0);
                        _mm512_storeu_ps(sum + c + 0x10, sum1);
                    }
                    if (c < channels)
                    {
                        __m512 sum0 = _mm512_maskz_loadu_ps(tail0, sum + c + 0x00);
                        __m512 sum1 = _mm512_maskz_loadu_ps(tail1, sum + c + 0x10);
                        __m512i _src = _mm512_maskz_loadu_epi16(tail, src0 + c);
                        sum0 = _mm512_add_ps(sum0, BFloat16ToFloat32<0>(_src));
                        sum1 = _mm512_add_ps(sum1, BFloat16ToFloat32<1>(_src));
                        _src = _mm512_maskz_loadu_epi16(tail, src1 + c);
                        sum0 = _mm512_add_ps(sum0, BFloat16ToFloat32<0>(_src));
                        sum1 = _mm512_add_ps(sum1, BFloat16ToFloat32<1>(_src));
                        _src = _mm512_maskz_loadu_epi16(tail, src2 + c);
                        sum0 = _mm512_add_ps(sum0, BFloat16ToFloat32<0>(_src));
                        sum1 = _mm512_add_ps(sum1, BFloat16ToFloat32<1>(_src));
                        _src = _mm512_maskz_loadu_epi16(tail, src3 + c);
                        sum0 = _mm512_add_ps(sum0, BFloat16ToFloat32<0>(_src));
                        sum1 = _mm512_add_ps(sum1, BFloat16ToFloat32<1>(_src));
                        _mm512_mask_storeu_ps(sum + c + 0x00, tail0, sum0);
                        _mm512_mask_storeu_ps(sum + c + 0x10, tail1, sum1);
                    }
                    src += channels * 4;
                }
                for (; s < spatial; ++s)
                {
                    c = 0;
                    for (; c < channels32; c += 32)
                    {
                        __m512 sum0 = _mm512_loadu_ps(sum + c + 0x00);
                        __m512 sum1 = _mm512_loadu_ps(sum + c + 0x10);
                        __m512i _src = _mm512_loadu_si512((__m512i*)(src + c));
                        _mm512_storeu_ps(sum + c + 0x00, _mm512_add_ps(sum0, BFloat16ToFloat32<0>(_src)));
                        _mm512_storeu_ps(sum + c + 0x10, _mm512_add_ps(sum1, BFloat16ToFloat32<1>(_src)));
                    }
                    if (c < channels)
                    {
                        __m512 sum0 = _mm512_maskz_loadu_ps(tail0, sum + c + 0x00);
                        __m512 sum1 = _mm512_maskz_loadu_ps(tail1, sum + c + 0x10);
                        __m512i _src = _mm512_maskz_loadu_epi16(tail, src + c);
                        _mm512_mask_storeu_ps(sum + c + 0x00, tail0, _mm512_add_ps(sum0, BFloat16ToFloat32<0>(_src)));
                        _mm512_mask_storeu_ps(sum + c + 0x10, tail1, _mm512_add_ps(sum1, BFloat16ToFloat32<1>(_src)));
                    }
                    src += channels;
                }
            }
            else if (format == SimdTensorFormatNchw)
            {
                size_t channels4 = AlignLo(spatial, 4), spatial32 = AlignLo(spatial, 32);
                __mmask32 tail = TailMask32(spatial - spatial32);
                size_t c = 0;
                for (; c < channels4; c += 4)
                {
                    const uint16_t* src0 = src + 0 * spatial;
                    const uint16_t* src1 = src + 1 * spatial;
                    const uint16_t* src2 = src + 2 * spatial;
                    const uint16_t* src3 = src + 3 * spatial;
                    __m512 sum00 = _mm512_setzero_ps(), sum01 = _mm512_setzero_ps();
                    __m512 sum10 = _mm512_setzero_ps(), sum11 = _mm512_setzero_ps();
                    __m512 sum20 = _mm512_setzero_ps(), sum21 = _mm512_setzero_ps();
                    __m512 sum30 = _mm512_setzero_ps(), sum31 = _mm512_setzero_ps();
                    size_t s = 0;
                    for (; s < spatial32; s += 32)
                    {
                        __m512i _src = _mm512_loadu_si512((__m512i*)(src0 + s));
                        sum00 = _mm512_add_ps(sum00, BFloat16ToFloat32Even(_src));
                        sum01 = _mm512_add_ps(sum01, BFloat16ToFloat32Odd(_src));
                        _src = _mm512_loadu_si512((__m512i*)(src1 + s));
                        sum10 = _mm512_add_ps(sum10, BFloat16ToFloat32Even(_src));
                        sum11 = _mm512_add_ps(sum11, BFloat16ToFloat32Odd(_src));
                        _src = _mm512_loadu_si512((__m512i*)(src2 + s));
                        sum20 = _mm512_add_ps(sum20, BFloat16ToFloat32Even(_src));
                        sum21 = _mm512_add_ps(sum21, BFloat16ToFloat32Odd(_src));
                        _src = _mm512_loadu_si512((__m512i*)(src3 + s));
                        sum30 = _mm512_add_ps(sum30, BFloat16ToFloat32Even(_src));
                        sum31 = _mm512_add_ps(sum31, BFloat16ToFloat32Odd(_src));
                    }
                    if (s < spatial)
                    {
                        __m512i _src = _mm512_maskz_loadu_epi16(tail, src0 + s);
                        sum00 = _mm512_add_ps(sum00, BFloat16ToFloat32Even(_src));
                        sum01 = _mm512_add_ps(sum01, BFloat16ToFloat32Odd(_src));
                        _src = _mm512_maskz_loadu_epi16(tail, src1 + s);
                        sum10 = _mm512_add_ps(sum10, BFloat16ToFloat32Even(_src));
                        sum11 = _mm512_add_ps(sum11, BFloat16ToFloat32Odd(_src));
                        _src = _mm512_maskz_loadu_epi16(tail, src2 + s);
                        sum20 = _mm512_add_ps(sum20, BFloat16ToFloat32Even(_src));
                        sum21 = _mm512_add_ps(sum21, BFloat16ToFloat32Odd(_src));
                        _src = _mm512_maskz_loadu_epi16(tail, src3 + s);
                        sum30 = _mm512_add_ps(sum30, BFloat16ToFloat32Even(_src));
                        sum31 = _mm512_add_ps(sum31, BFloat16ToFloat32Odd(_src));
                    }
                    sum00 = _mm512_add_ps(sum00, sum01);
                    sum10 = _mm512_add_ps(sum10, sum11);
                    sum20 = _mm512_add_ps(sum20, sum21);
                    sum30 = _mm512_add_ps(sum30, sum31);
                    _mm_storeu_ps(sum + c, Extract4Sums(sum00, sum10, sum20, sum30));
                    src += 4 * spatial;
                }
                for (; c < channels; ++c)
                {
                    __m512 sum0 = _mm512_setzero_ps(), sum1 = _mm512_setzero_ps();
                    size_t s = 0;
                    for (; s < spatial32; s += 32)
                    {
                        __m512i _src = _mm512_loadu_si512((__m512i*)(src + s));
                        sum0 = _mm512_add_ps(sum0, BFloat16ToFloat32Even(_src));
                        sum1 = _mm512_add_ps(sum1, BFloat16ToFloat32Odd(_src));
                    }
                    if(s < spatial)
                    {
                        __m512i _src = _mm512_maskz_loadu_epi16(tail, src + s);
                        sum0 = _mm512_add_ps(sum0, BFloat16ToFloat32Even(_src));
                        sum1 = _mm512_add_ps(sum1, BFloat16ToFloat32Odd(_src));
                    }
                    sum[c] = ExtractSum(_mm512_add_ps(sum0, sum1));
                    src += spatial;
                }
            }
            else
                assert(0);
        }

        //-------------------------------------------------------------------------------------------------

        template <SimdSynetEltwiseOperationType type> __m512 SynetEltwiseLayerForward(__m512 src0, __m512 src1);

        template <> SIMD_INLINE __m512 SynetEltwiseLayerForward<SimdSynetEltwiseOperationProduct>(__m512 src0, __m512 src1)
        {
            return _mm512_mul_ps(src0, src1);
        }

        template <> SIMD_INLINE __m512 SynetEltwiseLayerForward<SimdSynetEltwiseOperationMax>(__m512 src0, __m512 src1)
        {
            return _mm512_max_ps(src0, src1);
        }

        template <> SIMD_INLINE __m512 SynetEltwiseLayerForward<SimdSynetEltwiseOperationMin>(__m512 src0, __m512 src1)
        {
            return _mm512_min_ps(src0, src1);
        }

        template <SimdSynetEltwiseOperationType type, bool align, bool mask > SIMD_INLINE void SynetEltwiseLayerForward(const float* src0, const float* src1, float* dst, size_t offset, __mmask16 tail = -1)
        {
            Store<align, mask>(dst + offset, SynetEltwiseLayerForward<type>((Load<align, mask>(src0 + offset, tail)), (Load<align, mask>(src1 + offset, tail))), tail);
        }

        template <SimdSynetEltwiseOperationType type, bool align> void SynetEltwiseLayerForward(float const* const* src, size_t count, size_t size, float* dst)
        {
            size_t aligned = AlignLo(size, QF);
            size_t partial = AlignLo(size, F);
            __mmask16 tail = __mmask16(-1) >> (F + partial - size);
            const float* src0 = src[0];
            const float* src1 = src[1];
            size_t j = 0;
            for (; j < aligned; j += QF)
            {
                SynetEltwiseLayerForward<type, align, false>(src0, src1, dst, j + F * 0);
                SynetEltwiseLayerForward<type, align, false>(src0, src1, dst, j + F * 1);
                SynetEltwiseLayerForward<type, align, false>(src0, src1, dst, j + F * 2);
                SynetEltwiseLayerForward<type, align, false>(src0, src1, dst, j + F * 3);
            }
            for (; j < partial; j += F)
                SynetEltwiseLayerForward<type, align, false>(src0, src1, dst, j);
            if (j < size)
                SynetEltwiseLayerForward<type, align, true>(src0, src1, dst, j, tail);
            for (size_t i = 2; i < count; ++i)
            {
                const float* srci = src[i];
                for (j = 0; j < aligned; j += QF)
                {
                    SynetEltwiseLayerForward<type, align, false>(dst, srci, dst, j + F * 0);
                    SynetEltwiseLayerForward<type, align, false>(dst, srci, dst, j + F * 1);
                    SynetEltwiseLayerForward<type, align, false>(dst, srci, dst, j + F * 2);
                    SynetEltwiseLayerForward<type, align, false>(dst, srci, dst, j + F * 3);
                }
                for (; j < partial; j += F)
                    SynetEltwiseLayerForward<type, align, false>(dst, srci, dst, j);
                if (j < size)
                    SynetEltwiseLayerForward<type, align, true>(dst, srci, dst, j, tail);
            }
        }

        template <bool align, bool mask> void SynetEltwiseLayerForwardSum(const float* src0, const __m512& weight0, const float* src1, const __m512& weight1, float* dst, size_t offset, __mmask16 tail = -1)
        {
            Store<align, mask>(dst + offset, _mm512_fmadd_ps((Load<align, mask>(src0 + offset, tail)), weight0, _mm512_mul_ps((Load<align, mask>(src1 + offset, tail)), weight1)), tail);
        }

        template <bool align, bool mask> void SynetEltwiseLayerForwardSum(const float* src, const __m512& weight, float* dst, size_t offset, __mmask16 tail = -1)
        {
            Store<align, mask>(dst + offset, _mm512_fmadd_ps((Load<align, mask>(src + offset, tail)), weight, (Load<align, mask>(dst + offset, tail))), tail);
        }

        template <bool align> void SynetEltwiseLayerForwardSum(float const* const* src, const float* weight, size_t count, size_t size, float* dst)
        {
            size_t aligned = AlignLo(size, QF);
            size_t partial = AlignLo(size, F);
            __mmask16 tail = __mmask16(-1) >> (F + partial - size);
            const float* src0 = src[0];
            const float* src1 = src[1];
            __m512 weight0 = _mm512_set1_ps(weight[0]);
            __m512 weight1 = _mm512_set1_ps(weight[1]);
            size_t j = 0;
            for (; j < aligned; j += QF)
            {
                SynetEltwiseLayerForwardSum<align, false>(src0, weight0, src1, weight1, dst, j + F * 0);
                SynetEltwiseLayerForwardSum<align, false>(src0, weight0, src1, weight1, dst, j + F * 1);
                SynetEltwiseLayerForwardSum<align, false>(src0, weight0, src1, weight1, dst, j + F * 2);
                SynetEltwiseLayerForwardSum<align, false>(src0, weight0, src1, weight1, dst, j + F * 3);
            }
            for (; j < partial; j += F)
                SynetEltwiseLayerForwardSum<align, false>(src0, weight0, src1, weight1, dst, j);
            if (j < size)
                SynetEltwiseLayerForwardSum<align, true>(src0, weight0, src1, weight1, dst, j, tail);
            for (size_t i = 2; i < count; ++i)
            {
                const float* srci = src[i];
                __m512 weighti = _mm512_set1_ps(weight[i]);
                for (j = 0; j < aligned; j += QF)
                {
                    SynetEltwiseLayerForwardSum<align, false>(srci, weighti, dst, j + F * 0);
                    SynetEltwiseLayerForwardSum<align, false>(srci, weighti, dst, j + F * 1);
                    SynetEltwiseLayerForwardSum<align, false>(srci, weighti, dst, j + F * 2);
                    SynetEltwiseLayerForwardSum<align, false>(srci, weighti, dst, j + F * 3);
                }
                for (; j < partial; j += F)
                    SynetEltwiseLayerForwardSum<align, false>(srci, weighti, dst, j);
                if (j < size)
                    SynetEltwiseLayerForwardSum<align, true>(srci, weighti, dst, j, tail);
            }
        }

        template <bool align> void SynetEltwiseLayerForward(float const* const* src, const float* weight, size_t count, size_t size, SimdSynetEltwiseOperationType type, float* dst)
        {
            switch (type)
            {
            case SimdSynetEltwiseOperationProduct:
                SynetEltwiseLayerForward<SimdSynetEltwiseOperationProduct, align>(src, count, size, dst);
                break;
            case SimdSynetEltwiseOperationSum:
                SynetEltwiseLayerForwardSum<align>(src, weight, count, size, dst);
                break;
            case SimdSynetEltwiseOperationMax:
                SynetEltwiseLayerForward<SimdSynetEltwiseOperationMax, align>(src, count, size, dst);
                break;
            case SimdSynetEltwiseOperationMin:
                SynetEltwiseLayerForward<SimdSynetEltwiseOperationMin, align>(src, count, size, dst);
                break;
            default:
                assert(0);
            }
        }

        void SynetEltwiseLayerForward(float const* const* src, const float* weight, size_t count, size_t size, SimdSynetEltwiseOperationType type, float* dst)
        {
            assert(count >= 2);
            bool aligned = Aligned(dst) && Aligned(src[0]) && Aligned(src[1]);
            for (size_t i = 2; i < count; ++i)
                aligned = aligned && Aligned(src[i]);
            if (aligned)
                SynetEltwiseLayerForward<true>(src, weight, count, size, type, dst);
            else
                SynetEltwiseLayerForward<false>(src, weight, count, size, type, dst);
        }

        //-------------------------------------------------------------------------------------------------

        static SIMD_INLINE void Save4Sums(const __m512i& sum0, const __m512i sum1, const __m512i& sum2, const __m512i& sum3, int32_t* dst)
        {
            __m512i sum02 = _mm512_add_epi32(_mm512_unpacklo_epi32(sum0, sum2), _mm512_unpackhi_epi32(sum0, sum2));
            __m512i sum13 = _mm512_add_epi32(_mm512_unpacklo_epi32(sum1, sum3), _mm512_unpackhi_epi32(sum1, sum3));
            __m512i sum512 = _mm512_add_epi32(_mm512_unpacklo_epi32(sum02, sum13), _mm512_unpackhi_epi32(sum02, sum13));
            _mm_storeu_si128((__m128i*)dst, _mm_add_epi32(_mm_add_epi32(_mm512_extracti32x4_epi32(sum512, 0), _mm512_extracti32x4_epi32(sum512, 1)),
                _mm_add_epi32(_mm512_extracti32x4_epi32(sum512, 2), _mm512_extracti32x4_epi32(sum512, 3))));
        }

        template<bool overflow> static void SynetInnerProduct8i1x1(size_t K, const uint8_t* S, size_t lds, const int8_t* W, size_t ldw, int32_t* D, size_t ldd)
        {
            size_t KA = AlignLo(K, A);
            const uint8_t* S0 = S + 0 * lds;
            const int8_t* W0 = W + 0 * ldw;
            __m512i d00 = _mm512_setzero_si512();
            __m512i s0, w0;
            for (size_t k = 0; k < KA; k += A)
            {
                s0 = _mm512_loadu_si512((__m512i*)(S0 + k));
                w0 = _mm512_loadu_si512((__m512i*)(W0 + k));
                Madd4<overflow>(d00, s0, w0);
            }
            if (KA < K)
            {
                __mmask64 tail = TailMask64(K - KA);
                s0 = Load<false, true>(S0 + KA, tail);
                w0 = Load<false, true>(W0 + KA, tail);
                Madd4<overflow>(d00, s0, w0);
            }
            D[0] = ExtractSum<uint32_t>(d00);
        }

        template<bool overflow> static void SynetInnerProduct8i1x4(size_t K, const uint8_t* S, size_t lds, const int8_t* W, size_t ldw, int32_t* D, size_t ldd)
        {
            size_t KA = AlignLo(K, A);
            const uint8_t* S0 = S + 0 * lds;
            const int8_t* W0 = W + 0 * ldw;
            const int8_t* W1 = W + 1 * ldw;
            const int8_t* W2 = W + 2 * ldw;
            const int8_t* W3 = W + 3 * ldw;
            __m512i d00 = _mm512_setzero_si512();
            __m512i d01 = _mm512_setzero_si512();
            __m512i d02 = _mm512_setzero_si512();
            __m512i d03 = _mm512_setzero_si512();
            __m512i s0, w0;
            for (size_t k = 0; k < KA; k += A)
            {
                s0 = _mm512_loadu_si512((__m512i*)(S0 + k));
                w0 = _mm512_loadu_si512((__m512i*)(W0 + k));
                Madd4<overflow>(d00, s0, w0);
                w0 = _mm512_loadu_si512((__m512i*)(W1 + k));
                Madd4<overflow>(d01, s0, w0);
                w0 = _mm512_loadu_si512((__m512i*)(W2 + k));
                Madd4<overflow>(d02, s0, w0);
                w0 = _mm512_loadu_si512((__m512i*)(W3 + k));
                Madd4<overflow>(d03, s0, w0);
            }
            if (KA < K)
            {
                __mmask64 tail = TailMask64(K - KA);
                s0 = Load<false, true>(S0 + KA, tail);
                w0 = Load<false, true>(W0 + KA, tail);
                Madd4<overflow>(d00, s0, w0);
                w0 = Load<false, true>(W1 + KA, tail);
                Madd4<overflow>(d01, s0, w0);
                w0 = Load<false, true>(W2 + KA, tail);
                Madd4<overflow>(d02, s0, w0);
                w0 = Load<false, true>(W3 + KA, tail);
                Madd4<overflow>(d03, s0, w0);
            }
            Save4Sums(d00, d01, d02, d03, D);
        }

        template<bool overflow> static void SynetInnerProduct8i2x1(size_t K, const uint8_t* S, size_t lds, const int8_t* W, size_t ldw, int32_t* D, size_t ldd)
        {
            size_t KA = AlignLo(K, A);
            const uint8_t* S0 = S + 0 * lds;
            const uint8_t* S1 = S + 1 * lds;
            const int8_t* W0 = W + 0 * ldw;
            __m512i d00 = _mm512_setzero_si512();
            __m512i d10 = _mm512_setzero_si512();
            __m512i s0, s1, w0;
            for (size_t k = 0; k < KA; k += A)
            {
                s0 = _mm512_loadu_si512((__m512i*)(S0 + k));
                s1 = _mm512_loadu_si512((__m512i*)(S1 + k));
                w0 = _mm512_loadu_si512((__m512i*)(W0 + k));
                Madd4<overflow>(d00, s0, w0);
                Madd4<overflow>(d10, s1, w0);
            }
            if (KA < K)
            {
                __mmask64 tail = TailMask64(K - KA);
                s0 = Load<false, true>(S0 + KA, tail);
                s1 = Load<false, true>(S1 + KA, tail);
                w0 = Load<false, true>(W0 + KA, tail);
                Madd4<overflow>(d00, s0, w0);
                Madd4<overflow>(d10, s1, w0);
            }
            D[0 * ldd] = ExtractSum<uint32_t>(d00);
            D[1 * ldd] = ExtractSum<uint32_t>(d10);
        }

        template<bool overflow> static void SynetInnerProduct8i2x4(size_t K, const uint8_t* S, size_t lds, const int8_t* W, size_t ldw, int32_t* D, size_t ldd)
        {
            size_t KA = AlignLo(K, A);
            const uint8_t* S0 = S + 0 * lds;
            const uint8_t* S1 = S + 1 * lds;
            const int8_t* W0 = W + 0 * ldw;
            const int8_t* W1 = W + 1 * ldw;
            const int8_t* W2 = W + 2 * ldw;
            const int8_t* W3 = W + 3 * ldw;
            __m512i d00 = _mm512_setzero_si512();
            __m512i d01 = _mm512_setzero_si512();
            __m512i d02 = _mm512_setzero_si512();
            __m512i d03 = _mm512_setzero_si512();
            __m512i d10 = _mm512_setzero_si512();
            __m512i d11 = _mm512_setzero_si512();
            __m512i d12 = _mm512_setzero_si512();
            __m512i d13 = _mm512_setzero_si512();
            __m512i s0, s1, w0;
            for (size_t k = 0; k < KA; k += A)
            {
                s0 = _mm512_loadu_si512((__m512i*)(S0 + k));
                s1 = _mm512_loadu_si512((__m512i*)(S1 + k));
                w0 = _mm512_loadu_si512((__m512i*)(W0 + k));
                Madd4<overflow>(d00, s0, w0);
                Madd4<overflow>(d10, s1, w0);
                w0 = _mm512_loadu_si512((__m512i*)(W1 + k));
                Madd4<overflow>(d01, s0, w0);
                Madd4<overflow>(d11, s1, w0);
                w0 = _mm512_loadu_si512((__m512i*)(W2 + k));
                Madd4<overflow>(d02, s0, w0);
                Madd4<overflow>(d12, s1, w0);
                w0 = _mm512_loadu_si512((__m512i*)(W3 + k));
                Madd4<overflow>(d03, s0, w0);
                Madd4<overflow>(d13, s1, w0);
            }
            if (KA < K)
            {
                __mmask64 tail = TailMask64(K - KA);
                s0 = Load<false, true>(S0 + KA, tail);
                s1 = Load<false, true>(S1 + KA, tail);
                w0 = Load<false, true>(W0 + KA, tail);
                Madd4<overflow>(d00, s0, w0);
                Madd4<overflow>(d10, s1, w0);
                w0 = Load<false, true>(W1 + KA, tail);
                Madd4<overflow>(d01, s0, w0);
                Madd4<overflow>(d11, s1, w0);
                w0 = Load<false, true>(W2 + KA, tail);
                Madd4<overflow>(d02, s0, w0);
                Madd4<overflow>(d12, s1, w0);
                w0 = Load<false, true>(W3 + KA, tail);
                Madd4<overflow>(d03, s0, w0);
                Madd4<overflow>(d13, s1, w0);
            }
            Save4Sums(d00, d01, d02, d03, D + 0 * ldd);
            Save4Sums(d10, d11, d12, d13, D + 1 * ldd);
        }

        template<bool overflow> static void SynetInnerProduct8i4x1(size_t K, const uint8_t* S, size_t lds, const int8_t* W, size_t ldw, int32_t* D, size_t ldd)
        {
            size_t KA = AlignLo(K, A);
            const uint8_t* S0 = S + 0 * lds;
            const uint8_t* S1 = S + 1 * lds;
            const uint8_t* S2 = S + 2 * lds;
            const uint8_t* S3 = S + 3 * lds;
            const int8_t* W0 = W + 0 * ldw;
            __m512i d00 = _mm512_setzero_si512();
            __m512i d10 = _mm512_setzero_si512();
            __m512i d20 = _mm512_setzero_si512();
            __m512i d30 = _mm512_setzero_si512();
            __m512i s0, s1, s2, s3, w0;
            for (size_t k = 0; k < KA; k += A)
            {
                s0 = _mm512_loadu_si512((__m512i*)(S0 + k));
                s1 = _mm512_loadu_si512((__m512i*)(S1 + k));
                s2 = _mm512_loadu_si512((__m512i*)(S2 + k));
                s3 = _mm512_loadu_si512((__m512i*)(S3 + k));
                w0 = _mm512_loadu_si512((__m512i*)(W0 + k));
                Madd4<overflow>(d00, s0, w0);
                Madd4<overflow>(d10, s1, w0);
                Madd4<overflow>(d20, s2, w0);
                Madd4<overflow>(d30, s3, w0);
            }
            if (KA < K)
            {
                __mmask64 tail = TailMask64(K - KA);
                s0 = Load<false, true>(S0 + KA, tail);
                s1 = Load<false, true>(S1 + KA, tail);
                s2 = Load<false, true>(S2 + KA, tail);
                s3 = Load<false, true>(S3 + KA, tail);
                w0 = Load<false, true>(W0 + KA, tail);
                Madd4<overflow>(d00, s0, w0);
                Madd4<overflow>(d10, s1, w0);
                Madd4<overflow>(d20, s2, w0);
                Madd4<overflow>(d30, s3, w0);
            }
            D[0 * ldd] = ExtractSum<uint32_t>(d00);
            D[1 * ldd] = ExtractSum<uint32_t>(d10);
            D[2 * ldd] = ExtractSum<uint32_t>(d20);
            D[3 * ldd] = ExtractSum<uint32_t>(d30);
        }

        template<bool overflow> static void SynetInnerProduct8i4x4(size_t K, const uint8_t* S, size_t lds, const int8_t* W, size_t ldw, int32_t* D, size_t ldd)
        {
            size_t KA = AlignLo(K, A);
            const uint8_t* S0 = S + 0 * lds;
            const uint8_t* S1 = S + 1 * lds;
            const uint8_t* S2 = S + 2 * lds;
            const uint8_t* S3 = S + 3 * lds;
            const int8_t* W0 = W + 0 * ldw;
            const int8_t* W1 = W + 1 * ldw;
            const int8_t* W2 = W + 2 * ldw;
            const int8_t* W3 = W + 3 * ldw;
            __m512i d00 = _mm512_setzero_si512();
            __m512i d01 = _mm512_setzero_si512();
            __m512i d02 = _mm512_setzero_si512();
            __m512i d03 = _mm512_setzero_si512();
            __m512i d10 = _mm512_setzero_si512();
            __m512i d11 = _mm512_setzero_si512();
            __m512i d12 = _mm512_setzero_si512();
            __m512i d13 = _mm512_setzero_si512();
            __m512i d20 = _mm512_setzero_si512();
            __m512i d21 = _mm512_setzero_si512();
            __m512i d22 = _mm512_setzero_si512();
            __m512i d23 = _mm512_setzero_si512();
            __m512i d30 = _mm512_setzero_si512();
            __m512i d31 = _mm512_setzero_si512();
            __m512i d32 = _mm512_setzero_si512();
            __m512i d33 = _mm512_setzero_si512();
            __m512i s0, s1, s2, s3, w0;
            for (size_t k = 0; k < KA; k += A)
            {
                s0 = _mm512_loadu_si512((__m512i*)(S0 + k));
                s1 = _mm512_loadu_si512((__m512i*)(S1 + k));
                s2 = _mm512_loadu_si512((__m512i*)(S2 + k));
                s3 = _mm512_loadu_si512((__m512i*)(S3 + k));
                w0 = _mm512_loadu_si512((__m512i*)(W0 + k));
                Madd4<overflow>(d00, s0, w0);
                Madd4<overflow>(d10, s1, w0);
                Madd4<overflow>(d20, s2, w0);
                Madd4<overflow>(d30, s3, w0);
                w0 = _mm512_loadu_si512((__m512i*)(W1 + k));
                Madd4<overflow>(d01, s0, w0);
                Madd4<overflow>(d11, s1, w0);
                Madd4<overflow>(d21, s2, w0);
                Madd4<overflow>(d31, s3, w0);
                w0 = _mm512_loadu_si512((__m512i*)(W2 + k));
                Madd4<overflow>(d02, s0, w0);
                Madd4<overflow>(d12, s1, w0);
                Madd4<overflow>(d22, s2, w0);
                Madd4<overflow>(d32, s3, w0);
                w0 = _mm512_loadu_si512((__m512i*)(W3 + k));
                Madd4<overflow>(d03, s0, w0);
                Madd4<overflow>(d13, s1, w0);
                Madd4<overflow>(d23, s2, w0);
                Madd4<overflow>(d33, s3, w0);
            }
            if (KA < K)
            {
                __mmask64 tail = TailMask64(K - KA);
                s0 = Load<false, true>(S0 + KA, tail);
                s1 = Load<false, true>(S1 + KA, tail);
                s2 = Load<false, true>(S2 + KA, tail);
                s3 = Load<false, true>(S3 + KA, tail);
                w0 = Load<false, true>(W0 + KA, tail);
                Madd4<overflow>(d00, s0, w0);
                Madd4<overflow>(d10, s1, w0);
                Madd4<overflow>(d20, s2, w0);
                Madd4<overflow>(d30, s3, w0);
                w0 = Load<false, true>(W1 + KA, tail);
                Madd4<overflow>(d01, s0, w0);
                Madd4<overflow>(d11, s1, w0);
                Madd4<overflow>(d21, s2, w0);
                Madd4<overflow>(d31, s3, w0);
                w0 = Load<false, true>(W2 + KA, tail);
                Madd4<overflow>(d02, s0, w0);
                Madd4<overflow>(d12, s1, w0);
                Madd4<overflow>(d22, s2, w0);
                Madd4<overflow>(d32, s3, w0);
                w0 = Load<false, true>(W3 + KA, tail);
                Madd4<overflow>(d03, s0, w0);
                Madd4<overflow>(d13, s1, w0);
                Madd4<overflow>(d23, s2, w0);
                Madd4<overflow>(d33, s3, w0);
            }
            Save4Sums(d00, d01, d02, d03, D + 0 * ldd);
            Save4Sums(d10, d11, d12, d13, D + 1 * ldd);
            Save4Sums(d20, d21, d22, d23, D + 2 * ldd);
            Save4Sums(d30, d31, d32, d33, D + 3 * ldd);
        }

        template<bool overflow> void SynetInnerProduct8i(size_t M, size_t N, size_t K, const uint8_t* src, const int8_t* weight, int32_t* dst)
        {
            size_t M2 = AlignLoAny(M, 2);
            size_t M4 = AlignLoAny(M, 4);
            size_t N4 = AlignLoAny(N, 4);
            size_t i = 0;
            for (; i < M4; i += 4)
            {
                size_t j = 0;
                for (; j < N4; j += 4)
                    SynetInnerProduct8i4x4<overflow>(K, src, K, weight + j * K, K, dst + j, N);
                for (; j < N; j += 1)
                    SynetInnerProduct8i4x1<overflow>(K, src, K, weight + j * K, K, dst + j, N);
                src += K * 4;
                dst += N * 4;
            }
            for (; i < M2; i += 2)
            {
                size_t j = 0;
                for (; j < N4; j += 4)
                    SynetInnerProduct8i2x4<overflow>(K, src, K, weight + j * K, K, dst + j, N);
                for (; j < N; j += 1)
                    SynetInnerProduct8i2x1<overflow>(K, src, K, weight + j * K, K, dst + j, N);
                src += K * 2;
                dst += N * 2;
            }
            for (; i < M; i += 1)
            {
                size_t j = 0;
                for (; j < N4; j += 4)
                    SynetInnerProduct8i1x4<overflow>(K, src, K, weight + j * K, K, dst + j, N);
                for (; j < N; j += 1)
                    SynetInnerProduct8i1x1<overflow>(K, src, K, weight + j * K, K, dst + j, N);
                src += K;
                dst += N;
            }
        }

        void SynetInnerProduct8i(size_t M, size_t N, size_t K, const uint8_t* src, const int8_t* weight, int32_t* dst, SimdSynetCompatibilityType compatibility)
        {
            if (Base::Precise(compatibility))
                SynetInnerProduct8i<false>(M, N, K, src, weight, dst);
            else
                SynetInnerProduct8i<true>(M, N, K, src, weight, dst);
        }

        //-------------------------------------------------------------------------------------------------

        void SynetInnerProductLayerForward1(const float* S0, const float* W, const float* B, size_t K, float* D)
        {
            size_t K16 = K & (~15);
            size_t K64 = K & (~63);
            const float* W0 = W + 0 * K;
            __m512 d00, d01, d02, d03;
            __m512 s0, s1, s2, s3, w0, w1, w2, w3;
            size_t k = 0;
            d00 = _mm512_setzero_ps();
            if (K64)
            {
                d01 = _mm512_setzero_ps();
                d02 = _mm512_setzero_ps();
                d03 = _mm512_setzero_ps();
                for (; k < K64; k += 64)
                {
                    s0 = _mm512_loadu_ps(S0 + k + 0 * F);
                    s1 = _mm512_loadu_ps(S0 + k + 1 * F);
                    w0 = _mm512_loadu_ps(W0 + k + 0 * F);
                    w1 = _mm512_loadu_ps(W0 + k + 1 * F);
                    d00 = _mm512_fmadd_ps(s0, w0, d00);
                    d01 = _mm512_fmadd_ps(s1, w1, d01);
                    s2 = _mm512_loadu_ps(S0 + k + 2 * F);
                    s3 = _mm512_loadu_ps(S0 + k + 3 * F);
                    w2 = _mm512_loadu_ps(W0 + k + 2 * F);
                    w3 = _mm512_loadu_ps(W0 + k + 3 * F);
                    d02 = _mm512_fmadd_ps(s2, w2, d02);
                    d03 = _mm512_fmadd_ps(s3, w3, d03);
                }
                d00 = _mm512_add_ps(_mm512_add_ps(d00, d01), _mm512_add_ps(d02, d03));
            }
            for (; k < K16; k += 16)
            {
                s0 = _mm512_loadu_ps(S0 + k);
                w0 = _mm512_loadu_ps(W0 + k);
                d00 = _mm512_fmadd_ps(s0, w0, d00);
            }
            if (k < K)
            {
                __mmask16 tail = __mmask16(-1) >> (16 + k - K);
                s0 = _mm512_maskz_loadu_ps(tail, S0 + k);
                w0 = _mm512_maskz_loadu_ps(tail, W0 + k);
                d00 = _mm512_fmadd_ps(s0, w0, d00);
            }
            D[0] = ExtractSum(d00) + B[0];
        }

        void SynetInnerProductLayerForward4(const float* S0, const float* W, const float* B, size_t K, float* D)
        {
            size_t K16 = K & (~15);
            size_t K32 = K & (~31);
            const float* W0 = W + 0 * K;
            const float* W1 = W + 1 * K;
            const float* W2 = W + 2 * K;
            const float* W3 = W + 3 * K;
            __m512 d00, d01, d10, d11, d20, d21, d30, d31;
            __m512 s0, s1, w0, w1;
            size_t k = 0;
            d00 = _mm512_setzero_ps();
            d10 = _mm512_setzero_ps();
            d20 = _mm512_setzero_ps();
            d30 = _mm512_setzero_ps();
            if (K32)
            {
                d01 = _mm512_setzero_ps();
                d11 = _mm512_setzero_ps();
                d21 = _mm512_setzero_ps();
                d31 = _mm512_setzero_ps();
                for (; k < K32; k += 32)
                {
                    s0 = _mm512_loadu_ps(S0 + k + 0 * F);
                    s1 = _mm512_loadu_ps(S0 + k + 1 * F);
                    w0 = _mm512_loadu_ps(W0 + k + 0 * F);
                    w1 = _mm512_loadu_ps(W0 + k + 1 * F);
                    d00 = _mm512_fmadd_ps(s0, w0, d00);
                    d01 = _mm512_fmadd_ps(s1, w1, d01);
                    w0 = _mm512_loadu_ps(W1 + k + 0 * F);
                    w1 = _mm512_loadu_ps(W1 + k + 1 * F);
                    d10 = _mm512_fmadd_ps(s0, w0, d10);
                    d11 = _mm512_fmadd_ps(s1, w1, d11);
                    w0 = _mm512_loadu_ps(W2 + k + 0 * F);
                    w1 = _mm512_loadu_ps(W2 + k + 1 * F);
                    d20 = _mm512_fmadd_ps(s0, w0, d20);
                    d21 = _mm512_fmadd_ps(s1, w1, d21);
                    w0 = _mm512_loadu_ps(W3 + k + 0 * F);
                    w1 = _mm512_loadu_ps(W3 + k + 1 * F);
                    d30 = _mm512_fmadd_ps(s0, w0, d30);
                    d31 = _mm512_fmadd_ps(s1, w1, d31);
                }
                d00 = _mm512_add_ps(d00, d01);
                d10 = _mm512_add_ps(d10, d11);
                d20 = _mm512_add_ps(d20, d21);
                d30 = _mm512_add_ps(d30, d31);
            }
            for (; k < K16; k += 16)
            {
                s0 = _mm512_loadu_ps(S0 + k + 0 * F);
                w0 = _mm512_loadu_ps(W0 + k + 0 * F);
                d00 = _mm512_fmadd_ps(s0, w0, d00);
                w0 = _mm512_loadu_ps(W1 + k + 0 * F);
                d10 = _mm512_fmadd_ps(s0, w0, d10);
                w0 = _mm512_loadu_ps(W2 + k + 0 * F);
                d20 = _mm512_fmadd_ps(s0, w0, d20);
                w0 = _mm512_loadu_ps(W3 + k + 0 * F);
                d30 = _mm512_fmadd_ps(s0, w0, d30);
            }
            if (k < K)
            {
                __mmask16 tail = __mmask16(-1) >> (16 + k - K);
                s0 = _mm512_maskz_loadu_ps(tail, S0 + k);
                w0 = _mm512_maskz_loadu_ps(tail, W0 + k);
                d00 = _mm512_fmadd_ps(s0, w0, d00);
                w0 = _mm512_maskz_loadu_ps(tail, W1 + k);
                d10 = _mm512_fmadd_ps(s0, w0, d10);
                w0 = _mm512_maskz_loadu_ps(tail, W2 + k);
                d20 = _mm512_fmadd_ps(s0, w0, d20);
                w0 = _mm512_maskz_loadu_ps(tail, W3 + k);
                d30 = _mm512_fmadd_ps(s0, w0, d30);
            }
            _mm_storeu_ps(D, _mm_add_ps(Extract4Sums(d00, d10, d20, d30), _mm_loadu_ps(B)));
        }

        void SynetInnerProductLayerForward(const float* src, const float* weight, const float* bias, size_t count, size_t size, float* dst)
        {
            float _bias[4] = { 0, 0, 0, 0 };
            size_t count4 = AlignLo(count, 4);
            size_t i = 0;
            for (; i < count4; i += 4)
                SynetInnerProductLayerForward4(src, weight + i * size, (bias ? bias + i : _bias), size, dst + i);
            for (; i < count; ++i)
                SynetInnerProductLayerForward1(src, weight + i * size, (bias ? bias + i : _bias), size, dst + i);
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE __m512 NoseSquareSum(const float* src)
        {
            __m512 s0 = _mm512_maskz_loadu_ps(0xFFFC, src - 2);
            __m512 s1 = _mm512_maskz_loadu_ps(0xFFFE, src - 1);
            __m512 s2 = _mm512_loadu_ps(src);
            __m512 s3 = _mm512_loadu_ps(src + 1);
            __m512 s4 = _mm512_loadu_ps(src + 2);
            return _mm512_add_ps(_mm512_fmadd_ps(s0, s0, _mm512_mul_ps(s1, s1)), _mm512_fmadd_ps(s2, s2, _mm512_fmadd_ps(s3, s3, _mm512_mul_ps(s4, s4))));
        }

        SIMD_INLINE __m512 BodySquareSum(const float* src)
        {
            __m512 s0 = _mm512_loadu_ps(src - 2);
            __m512 s1 = _mm512_loadu_ps(src - 1);
            __m512 s2 = _mm512_loadu_ps(src);
            __m512 s3 = _mm512_loadu_ps(src + 1);
            __m512 s4 = _mm512_loadu_ps(src + 2);
            return _mm512_add_ps(_mm512_fmadd_ps(s0, s0, _mm512_mul_ps(s1, s1)), _mm512_fmadd_ps(s2, s2, _mm512_fmadd_ps(s3, s3, _mm512_mul_ps(s4, s4))));
        }

        SIMD_INLINE __m512 TailSquareSum(const float* src)
        {
            __m512 s0 = _mm512_loadu_ps(src - 2);
            __m512 s1 = _mm512_loadu_ps(src - 1);
            __m512 s2 = _mm512_loadu_ps(src);
            __m512 s3 = _mm512_maskz_loadu_ps(0x7FFF, src + 1);
            __m512 s4 = _mm512_maskz_loadu_ps(0x3FFF, src + 2);
            return _mm512_add_ps(_mm512_fmadd_ps(s0, s0, _mm512_mul_ps(s1, s1)), _mm512_fmadd_ps(s2, s2, _mm512_fmadd_ps(s3, s3, _mm512_mul_ps(s4, s4))));
        }

        template<bool align> void SynetLrnLayerCrossChannelsNchw(const float* src, size_t half, size_t channels, size_t spatial, const float* k, float* dst)
        {
            __m512 k0 = _mm512_set1_ps(k[0]);
            __m512 k1 = _mm512_set1_ps(k[1]);
            __m512 k2 = _mm512_set1_ps(k[2]);
            Pow pow;
            Array32f sum(spatial, true), zero(spatial, true);
            size_t aligned = AlignLo(spatial, F);
            __mmask16 tail = TailMask16(spatial - aligned);
            for (size_t c = 0; c < half; ++c)
            {
                const float* pos = src + c * spatial;
                size_t s = 0;
                for (; s < aligned; s += F)
                {
                    __m512 _pos = Load<align>(pos + s);
                    Store<true>(sum.data + s, _mm512_fmadd_ps(_pos, _pos, Load<true>(sum.data + s)));
                }
                if (s < spatial)
                {
                    __m512 _pos = Load<align, true>(pos + s, tail);
                    __m512 _sum = Load<true, true>(sum.data + s, tail);
                    Store<true, true>(sum.data + s, _mm512_fmadd_ps(_pos, _pos, _sum), tail);
                }
            }
            for (size_t c = 0; c < channels; ++c)
            {
                const float* pos = (c < channels - half) ? src + half * spatial : zero.data;
                const float* neg = (c > half) ? src - (half + 1) * spatial : zero.data;
                size_t s = 0;
                for (; s < aligned; s += F)
                {
                    __m512 _pos = Load<align>(pos + s);
                    __m512 _neg = Load<align>(neg + s);
                    __m512 _sum = Load<true>(sum.data + s);
                    _sum = _mm512_fmadd_ps(_pos, _pos, _mm512_fnmadd_ps(_neg, _neg, _sum));
                    __m512 _src = Load<align>(src + s);
                    Store<true>(sum.data + s, _sum);
                    Store<align>(dst + s, _mm512_mul_ps(_src, pow(_mm512_fmadd_ps(k1, _sum, k0), k2)));
                }
                if (s < spatial)
                {
                    __m512 _pos = Load<align, true>(pos + s, tail);
                    __m512 _neg = Load<align, true>(neg + s, tail);
                    __m512 _sum = Load<true, true>(sum.data + s, tail);
                    _sum = _mm512_fmadd_ps(_pos, _pos, _mm512_fnmadd_ps(_neg, _neg, _sum));
                    __m512 _src = Load<align, true>(src + s, tail);
                    Store<true, true>(sum.data + s, _sum, tail);
                    Store<align, true>(dst + s, _mm512_mul_ps(_src, pow(_mm512_fmadd_ps(k1, _sum, k0), k2)), tail);
                }
                src += spatial;
                dst += spatial;
            }
        }

        SIMD_INLINE void SynetLrnLayerCrossChannelsNchw(const float* src, size_t half, size_t channels, size_t spatial, const float* k, float* dst)
        {
            if (Aligned(src) && Aligned(dst) && Aligned(spatial, F))
                SynetLrnLayerCrossChannelsNchw<true>(src, half, channels, spatial, k, dst);
            else
                SynetLrnLayerCrossChannelsNchw<false>(src, half, channels, spatial, k, dst);
        }

        template<bool align> void SynetLrnLayerCrossChannelsNhwc2h(const float* src, size_t half, size_t channels, size_t spatial, const float* k, float* dst)
        {
            __m512 k0 = _mm512_set1_ps(k[0]);
            __m512 k1 = _mm512_set1_ps(k[1]);
            __m512 k2 = _mm512_set1_ps(k[2]);
            Pow pow;
            size_t aligned = AlignLo(channels - half, F);
            for (size_t s = 0; s < spatial; ++s)
            {
                Store<align>(dst + 0, _mm512_mul_ps(Load<align>(src + 0), pow(_mm512_add_ps(k0, _mm512_mul_ps(k1, NoseSquareSum(src + 0))), k2)));
                for (size_t c = F; c < aligned; c += F)
                    Store<align>(dst + c, _mm512_mul_ps(Load<align>(src + c), pow(_mm512_add_ps(k0, _mm512_mul_ps(k1, BodySquareSum(src + c))), k2)));
                if (aligned != channels - half)
                {
                    size_t c = channels - half - F;
                    Store<false>(dst + c, _mm512_mul_ps(Load<false>(src + c), pow(_mm512_add_ps(k0, _mm512_mul_ps(k1, BodySquareSum(src + c))), k2)));
                }
                size_t c = channels - F;
                Store<false>(dst + c, _mm512_mul_ps(Load<false>(src + c), pow(_mm512_add_ps(k0, _mm512_mul_ps(k1, TailSquareSum(src + c))), k2)));
                src += channels;
                dst += channels;
            }
        }

        SIMD_INLINE void SynetLrnLayerCrossChannelsNhwc(const float* src, size_t half, size_t channels, size_t spatial, const float* k, float* dst)
        {
            if (half == 2 && channels >= F + half)
            {
                if (Aligned(src) && Aligned(dst) && Aligned(channels, F))
                    SynetLrnLayerCrossChannelsNhwc2h<true>(src, half, channels, spatial, k, dst);
                else
                    SynetLrnLayerCrossChannelsNhwc2h<false>(src, half, channels, spatial, k, dst);
            }
            else
                Avx2::SynetLrnLayerCrossChannels(src, half, channels, spatial, k, dst, SimdTensorFormatNhwc);
        }

        void SynetLrnLayerCrossChannels(const float* src, size_t half, size_t channels, size_t spatial, const float* k, float* dst, SimdTensorFormatType format)
        {
            if (format == SimdTensorFormatNchw)
                SynetLrnLayerCrossChannelsNchw(src, half, channels, spatial, k, dst);
            else if (format == SimdTensorFormatNhwc)
                SynetLrnLayerCrossChannelsNhwc(src, half, channels, spatial, k, dst);
            else
                assert(0);
        }

        //-------------------------------------------------------------------------------------------------

        void SynetShuffleLayerForward(const float* src0, const float* src1, size_t channels0, size_t channels1, size_t spatial, float* dst0, float* dst1, SimdTensorFormatType format, int type)
        {
            if (format == SimdTensorFormatNchw)
                Base::SynetShuffleLayerForward(src0, src1, channels0, channels1, spatial, dst0, dst1, format, type);
            else if (format == SimdTensorFormatNhwc)
            {
                size_t channels = (channels0 + channels1) / 2;
                size_t channels0DF = AlignLo(channels0, DF);
                __mmask16 tail00 = TailMask16(channels0 - channels0DF);
                __mmask16 tail0F = TailMask16(channels0 - channels0DF - F);
                size_t channels0t = (channels0 - channels0DF) / 2;
                __mmask16 tail0 = TailMask16(channels0t);
                size_t channels1DF = AlignLo(channels1, DF);
                __mmask16 tail10 = TailMask16(channels1 - channels1DF);
                __mmask16 tail1F = TailMask16(channels1 - channels1DF - F);
                size_t channels1t = (channels1 - channels1DF) / 2;
                __mmask16 tail1 = TailMask16(channels1t);
                if (type == 0)
                {
                    for (size_t s = 0; s < spatial; ++s)
                    {
                        size_t cd = 0, cs0 = 0, cs1 = 0;
                        for (; cs0 < channels0DF; cs0 += DF, cd += F)
                        {
                            __m512 s0 = _mm512_loadu_ps(src0 + cs0 + 0);
                            __m512 s1 = _mm512_loadu_ps(src0 + cs0 + F);
                            _mm512_storeu_ps(dst0 + cd, Deinterleave<0>(s0, s1));
                            _mm512_storeu_ps(dst1 + cd, Deinterleave<1>(s0, s1));
                        }
                        if (channels0DF < channels0)
                        {
                            __m512 s0 = _mm512_maskz_loadu_ps(tail00, src0 + cs0 + 0);
                            __m512 s1 = _mm512_maskz_loadu_ps(tail0F, src0 + cs0 + F);
                            _mm512_mask_storeu_ps(dst0 + cd, tail0, Deinterleave<0>(s0, s1));
                            _mm512_mask_storeu_ps(dst1 + cd, tail0, Deinterleave<1>(s0, s1));
                            cd += channels0t;
                        }
                        for (; cs1 < channels1DF; cs1 += DF, cd += F)
                        {
                            __m512 s0 = _mm512_loadu_ps(src1 + cs1 + 0);
                            __m512 s1 = _mm512_loadu_ps(src1 + cs1 + F);
                            _mm512_storeu_ps(dst0 + cd, Deinterleave<0>(s0, s1));
                            _mm512_storeu_ps(dst1 + cd, Deinterleave<1>(s0, s1));
                        }
                        if (channels1DF < channels1)
                        {
                            __m512 s0 = _mm512_maskz_loadu_ps(tail10, src1 + cs1 + 0);
                            __m512 s1 = _mm512_maskz_loadu_ps(tail1F, src1 + cs1 + F);
                            _mm512_mask_storeu_ps(dst0 + cd, tail1, Deinterleave<0>(s0, s1));
                            _mm512_mask_storeu_ps(dst1 + cd, tail1, Deinterleave<1>(s0, s1));
                            cd += channels1t;
                        }
                        src0 += channels0;
                        src1 += channels1;
                        dst0 += channels;
                        dst1 += channels;
                    }
                }
                else if (type == 1)
                {
                    for (size_t s = 0; s < spatial; ++s)
                    {
                        size_t cs = 0, cd0 = 0, cd1 = 0;
                        for (; cd0 < channels0DF; cd0 += DF, cs += F)
                        {
                            __m512 s0 = _mm512_loadu_ps(src0 + cs);
                            __m512 s1 = _mm512_loadu_ps(src1 + cs);
                            _mm512_storeu_ps(dst0 + cd0 + 0, Interleave<0>(s0, s1));
                            _mm512_storeu_ps(dst0 + cd0 + F, Interleave<1>(s0, s1));
                        }
                        if (channels0DF < channels0)
                        {
                            __m512 s0 = _mm512_maskz_loadu_ps(tail0, src0 + cs);
                            __m512 s1 = _mm512_maskz_loadu_ps(tail0, src1 + cs);
                            _mm512_mask_storeu_ps(dst0 + cd0 + 0, tail00, Interleave<0>(s0, s1));
                            _mm512_mask_storeu_ps(dst0 + cd0 + F, tail0F, Interleave<1>(s0, s1));
                            cs += channels0t;
                        }
                        for (; cd1 < channels1DF; cd1 += DF, cs += F)
                        {
                            __m512 s0 = _mm512_loadu_ps(src0 + cs);
                            __m512 s1 = _mm512_loadu_ps(src1 + cs);
                            _mm512_storeu_ps(dst1 + cd1 + 0, Interleave<0>(s0, s1));
                            _mm512_storeu_ps(dst1 + cd1 + F, Interleave<1>(s0, s1));
                        }
                        if (channels1DF < channels1)
                        {
                            __m512 s0 = _mm512_maskz_loadu_ps(tail1, src0 + cs);
                            __m512 s1 = _mm512_maskz_loadu_ps(tail1, src1 + cs);
                            _mm512_mask_storeu_ps(dst1 + cd1 + 0, tail10, Interleave<0>(s0, s1));
                            _mm512_mask_storeu_ps(dst1 + cd1 + F, tail1F, Interleave<1>(s0, s1));
                            cs += channels1t;
                        }
                        src0 += channels;
                        src1 += channels;
                        dst0 += channels0;
                        dst1 += channels1;
                    }
                }
                else
                    assert(0);
            }
            else
                assert(0);
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE void SynetTiledScale2D32fNchw(const float* src, const float* ver, __m512 hor, float* dst, size_t offset, __mmask16 mask = -1)
        {
            _mm512_mask_storeu_ps(dst + offset, mask, _mm512_mul_ps(_mm512_maskz_loadu_ps(mask, src + offset), _mm512_mul_ps(_mm512_maskz_loadu_ps(mask, ver + offset), hor)));
        }

        SIMD_INLINE void SynetTiledScale2D32fNhwc(const float* src, const float* ver, const float* hor, float* dst, size_t offset, __mmask16 mask = -1)
        {
            _mm512_mask_storeu_ps(dst + offset, mask, _mm512_mul_ps(_mm512_maskz_loadu_ps(mask, src + offset), _mm512_mul_ps(_mm512_maskz_loadu_ps(mask, ver + offset), _mm512_maskz_loadu_ps(mask, hor + offset))));
        }

        void SynetTiledScale2D32f(const float* src, size_t channels, size_t height, size_t width, SimdTensorFormatType format, const float* ver, const float* hor, float* dst)
        {
            if (format == SimdTensorFormatNchw)
            {
                size_t widthF = AlignLo(width, F);
                size_t width4F = AlignLo(width, 4 * F);
                __mmask16 tail = TailMask16(width - widthF);
                for (size_t c = 0; c < channels; ++c)
                {
                    for (size_t y = 0; y < height; ++y)
                    {
                        __m512 _hor = _mm512_set1_ps(hor[y]);
                        size_t x = 0;
                        for (; x < width4F; x += 4 * F)
                        {
                            SynetTiledScale2D32fNchw(src, ver, _hor, dst, x + 0 * F);
                            SynetTiledScale2D32fNchw(src, ver, _hor, dst, x + 1 * F);
                            SynetTiledScale2D32fNchw(src, ver, _hor, dst, x + 2 * F);
                            SynetTiledScale2D32fNchw(src, ver, _hor, dst, x + 3 * F);
                        }
                        for (; x < widthF; x += F)
                            SynetTiledScale2D32fNchw(src, ver, _hor, dst, x);
                        if  (x < width)
                            SynetTiledScale2D32fNchw(src, ver, _hor, dst, x, tail);
                        src += width, dst += width;
                    }
                    hor += height;
                    ver += width;
                }
            }
            else if (format == SimdTensorFormatNhwc)
            {
                size_t channelsF = AlignLo(channels, F);
                size_t channels4F = AlignLo(channels, 4 * F);
                __mmask16 tail = TailMask16(channels - channelsF);
                for (size_t y = 0; y < height; ++y)
                {
                    const float* pVer = ver;
                    for (size_t x = 0; x < width; ++x)
                    {
                        size_t c = 0;
                        for (; c < channels4F; c += 4 * F)
                        {
                            SynetTiledScale2D32fNhwc(src, pVer, hor, dst, c + 0 * F);
                            SynetTiledScale2D32fNhwc(src, pVer, hor, dst, c + 1 * F);
                            SynetTiledScale2D32fNhwc(src, pVer, hor, dst, c + 2 * F);
                            SynetTiledScale2D32fNhwc(src, pVer, hor, dst, c + 3 * F);
                        }
                        for (; c < channelsF; c += F)
                            SynetTiledScale2D32fNhwc(src, pVer, hor, dst, c);
                        if(c < channels)
                            SynetTiledScale2D32fNhwc(src, pVer, hor, dst, c, tail);
                        src += channels, dst += channels, pVer += channels;
                    }
                    hor += channels;
                }
            }
            else
                assert(0);
        }
    }
#endif
}
