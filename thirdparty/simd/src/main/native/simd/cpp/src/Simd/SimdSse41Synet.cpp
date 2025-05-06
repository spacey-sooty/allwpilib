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
#include "Simd/SimdArray.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdSse41.h"
#include "Simd/SimdExp.h"
#include "Simd/SimdGather.h"
#include "Simd/SimdPow.h"
#include "Simd/SimdBFloat16.h"

namespace Simd
{
#if defined(SIMD_SSE41_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Sse41
    {
        void SynetChannelSum16b(const uint16_t* src, size_t channels, size_t spatial, SimdTensorFormatType format, float* sum)
        {
            if (format == SimdTensorFormatNhwc)
            {
                size_t channels4 = AlignLo(channels, 4), channels8 = AlignLo(channels, 8);
                size_t spatial4 = AlignLo(spatial, 4);
                size_t c = 0;
                for (; c < channels4; c += 4)
                    _mm_storeu_ps(sum + c, _mm_setzero_ps());
                for (; c < channels; ++c)
                    sum[c] = 0.0f;

                size_t s = 0;
                for (; s < spatial4; s += 4)
                {
                    const uint16_t* src0 = src + 0 * channels;
                    const uint16_t* src1 = src + 1 * channels;
                    const uint16_t* src2 = src + 2 * channels;
                    const uint16_t* src3 = src + 3 * channels;
                    size_t c = 0;
                    for (; c < channels8; c += 8)
                    {
                        __m128 sum0 = _mm_loadu_ps(sum + c + 0);
                        __m128 sum1 = _mm_loadu_ps(sum + c + 4);
                        __m128i _src = _mm_loadu_si128((__m128i*)(src0 + c));
                        sum0 = _mm_add_ps(sum0, BFloat16ToFloat32<0>(_src));
                        sum1 = _mm_add_ps(sum1, BFloat16ToFloat32<1>(_src));
                        _src = _mm_loadu_si128((__m128i*)(src1 + c));
                        sum0 = _mm_add_ps(sum0, BFloat16ToFloat32<0>(_src));
                        sum1 = _mm_add_ps(sum1, BFloat16ToFloat32<1>(_src));
                        _src = _mm_loadu_si128((__m128i*)(src2 + c));
                        sum0 = _mm_add_ps(sum0, BFloat16ToFloat32<0>(_src));
                        sum1 = _mm_add_ps(sum1, BFloat16ToFloat32<1>(_src));
                        _src = _mm_loadu_si128((__m128i*)(src3 + c));
                        sum0 = _mm_add_ps(sum0, BFloat16ToFloat32<0>(_src));
                        sum1 = _mm_add_ps(sum1, BFloat16ToFloat32<1>(_src));
                        _mm_storeu_ps(sum + c + 0, sum0);
                        _mm_storeu_ps(sum + c + 4, sum1);
                    }
                    for (; c < channels4; c += 4)
                    {
                        __m128 sum0 = _mm_loadu_ps(sum + c + 0);
                        __m128i _src = _mm_loadl_epi64((__m128i*)(src0 + c));
                        sum0 = _mm_add_ps(sum0, BFloat16ToFloat32<0>(_src));
                        _src = _mm_loadl_epi64((__m128i*)(src1 + c));
                        sum0 = _mm_add_ps(sum0, BFloat16ToFloat32<0>(_src));
                        _src = _mm_loadl_epi64((__m128i*)(src2 + c));
                        sum0 = _mm_add_ps(sum0, BFloat16ToFloat32<0>(_src));
                        _src = _mm_loadl_epi64((__m128i*)(src3 + c));
                        sum0 = _mm_add_ps(sum0, BFloat16ToFloat32<0>(_src));
                        _mm_storeu_ps(sum + c + 0, sum0);
                    }
                    for (; c < channels; ++c)
                    {
                        sum[c] += Base::BFloat16ToFloat32(src0[c]);
                        sum[c] += Base::BFloat16ToFloat32(src1[c]);
                        sum[c] += Base::BFloat16ToFloat32(src2[c]);
                        sum[c] += Base::BFloat16ToFloat32(src3[c]);
                    }
                    src += channels * 4;
                }
                for (; s < spatial; ++s)
                {
                    c = 0;
                    for (; c < channels8; c += 8)
                    {
                        __m128i _src = _mm_loadu_si128((__m128i*)(src + c));
                        __m128 sum0 = _mm_loadu_ps(sum + c + 0);
                        __m128 sum1 = _mm_loadu_ps(sum + c + 4);
                        _mm_storeu_ps(sum + c + 0, _mm_add_ps(sum0, BFloat16ToFloat32<0>(_src)));
                        _mm_storeu_ps(sum + c + 4, _mm_add_ps(sum1, BFloat16ToFloat32<1>(_src)));
                    }
                    for (; c < channels4; c += 4)
                    {
                        __m128 _sum = _mm_loadu_ps(sum + c);
                        __m128i _src = _mm_loadl_epi64((__m128i*)(src + c));
                        _mm_storeu_ps(sum + c, _mm_add_ps(_sum, BFloat16ToFloat32<0>(_src)));
                    }
                    for (; c < channels; ++c)
                        sum[c] += Base::BFloat16ToFloat32(src[c]);
                    src += channels;
                }
            }
            else if (format == SimdTensorFormatNchw)
            {
                size_t spatial8 = AlignLo(spatial, 8);
                for (size_t c = 0; c < channels; ++c)
                {
                    __m128 sum0 = _mm_setzero_ps(), sum1 = _mm_setzero_ps();
                    size_t s = 0;
                    for (; s < spatial8; s += 8)
                    {
                        __m128i _src = _mm_loadu_si128((__m128i*)(src + s));
                        sum0 = _mm_add_ps(sum0, BFloat16ToFloat32Even(_src));
                        sum1 = _mm_add_ps(sum1, BFloat16ToFloat32Odd(_src));
                    }
                    sum[c] = ExtractSum(_mm_add_ps(sum0, sum1));
                    for (; s < spatial; ++s)
                        sum[c] += Base::BFloat16ToFloat32(src[s]);
                    src += spatial;
                }
            }
            else
                assert(0);
        }

        //-------------------------------------------------------------------------------------------------

        template <SimdSynetEltwiseOperationType type> __m128 SynetEltwiseLayerForward(__m128 src0, __m128 src1);

        template <> SIMD_INLINE __m128 SynetEltwiseLayerForward<SimdSynetEltwiseOperationProduct>(__m128 src0, __m128 src1)
        {
            return _mm_mul_ps(src0, src1);
        }

        template <> SIMD_INLINE __m128 SynetEltwiseLayerForward<SimdSynetEltwiseOperationMax>(__m128 src0, __m128 src1)
        {
            return _mm_max_ps(src0, src1);
        }

        template <> SIMD_INLINE __m128 SynetEltwiseLayerForward<SimdSynetEltwiseOperationMin>(__m128 src0, __m128 src1)
        {
            return _mm_min_ps(src0, src1);
        }

        template <SimdSynetEltwiseOperationType type, bool align> SIMD_INLINE void SynetEltwiseLayerForward(const float* src0, const float* src1, float* dst, size_t offset)
        {
            Store<align>(dst + offset, SynetEltwiseLayerForward<type>(Load<align>(src0 + offset), Load<align>(src1 + offset)));
        }

        template <SimdSynetEltwiseOperationType type, bool align> void SynetEltwiseLayerForward(float const* const* src, size_t count, size_t size, float* dst)
        {
            size_t aligned = AlignLo(size, QF);
            size_t partial = AlignLo(size, F);
            const float* src0 = src[0];
            const float* src1 = src[1];
            size_t j = 0;
            if (partial)
            {
                for (; j < aligned; j += QF)
                {
                    SynetEltwiseLayerForward<type, align>(src0, src1, dst, j + F * 0);
                    SynetEltwiseLayerForward<type, align>(src0, src1, dst, j + F * 1);
                    SynetEltwiseLayerForward<type, align>(src0, src1, dst, j + F * 2);
                    SynetEltwiseLayerForward<type, align>(src0, src1, dst, j + F * 3);
                }
                for (; j < partial; j += F)
                    SynetEltwiseLayerForward<type, align>(src0, src1, dst, j);
            }
            for (; j < size; ++j)
                dst[j] = Base::SynetEltwiseLayerForward<type>(src0[j], src1[j]);
            for (size_t i = 2; i < count; ++i)
            {
                const float* srci = src[i];
                size_t j = 0;
                if (partial)
                {
                    for (; j < aligned; j += QF)
                    {
                        SynetEltwiseLayerForward<type, align>(dst, srci, dst, j + F * 0);
                        SynetEltwiseLayerForward<type, align>(dst, srci, dst, j + F * 1);
                        SynetEltwiseLayerForward<type, align>(dst, srci, dst, j + F * 2);
                        SynetEltwiseLayerForward<type, align>(dst, srci, dst, j + F * 3);
                    }
                    for (; j < partial; j += F)
                        SynetEltwiseLayerForward<type, align>(dst, srci, dst, j);
                }
                for (; j < size; ++j)
                    dst[j] = Base::SynetEltwiseLayerForward<type>(dst[j], srci[j]);
            }
        }

        template <bool align> SIMD_INLINE void SynetEltwiseLayerForwardSum(const float* src0, const __m128& weight0, const float* src1, const __m128& weight1, float* dst, size_t offset)
        {
            Store<align>(dst + offset, _mm_add_ps(_mm_mul_ps(Load<align>(src0 + offset), weight0), _mm_mul_ps(Load<align>(src1 + offset), weight1)));
        }

        template <bool align> SIMD_INLINE void SynetEltwiseLayerForwardSum(const float* src, const __m128& weight, float* dst, size_t offset)
        {
            Store<align>(dst + offset, _mm_add_ps(_mm_mul_ps(Load<align>(src + offset), weight), Load<align>(dst + offset)));
        }

        template <bool align> void SynetEltwiseLayerForwardSum(float const* const* src, const float* weight, size_t count, size_t size, float* dst)
        {
            size_t aligned = AlignLo(size, QF);
            size_t partial = AlignLo(size, F);
            const float* src0 = src[0];
            const float* src1 = src[1];
            __m128 weight0 = _mm_set1_ps(weight[0]);
            __m128 weight1 = _mm_set1_ps(weight[1]);
            size_t j = 0;
            if (partial)
            {
                for (; j < aligned; j += QF)
                {
                    SynetEltwiseLayerForwardSum<align>(src0, weight0, src1, weight1, dst, j + F * 0);
                    SynetEltwiseLayerForwardSum<align>(src0, weight0, src1, weight1, dst, j + F * 1);
                    SynetEltwiseLayerForwardSum<align>(src0, weight0, src1, weight1, dst, j + F * 2);
                    SynetEltwiseLayerForwardSum<align>(src0, weight0, src1, weight1, dst, j + F * 3);
                }
                for (; j < partial; j += F)
                    SynetEltwiseLayerForwardSum<align>(src0, weight0, src1, weight1, dst, j);
            }
            for (; j < size; ++j)
                dst[j] = src0[j] * weight[0] + src1[j] * weight[1];
            for (size_t i = 2; i < count; ++i)
            {
                const float* srci = src[i];
                __m128 weighti = _mm_set1_ps(weight[i]);
                size_t j = 0;
                if (partial)
                {
                    for (; j < aligned; j += QF)
                    {
                        SynetEltwiseLayerForwardSum<align>(srci, weighti, dst, j + F * 0);
                        SynetEltwiseLayerForwardSum<align>(srci, weighti, dst, j + F * 1);
                        SynetEltwiseLayerForwardSum<align>(srci, weighti, dst, j + F * 2);
                        SynetEltwiseLayerForwardSum<align>(srci, weighti, dst, j + F * 3);
                    }
                    for (; j < partial; j += F)
                        SynetEltwiseLayerForwardSum<align>(srci, weighti, dst, j);
                }
                for (; j < size; ++j)
                    dst[j] += srci[j] * weight[i];
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

        template <bool align> SIMD_INLINE void SynetInnerProductLayerForward(const float* src, const float* weight, size_t offset, __m128& sum)
        {
            __m128 s = Load<align>(src + offset);
            __m128 w = Load<align>(weight + offset);
            sum = _mm_add_ps(_mm_mul_ps(s, w), sum);
        }

        template<bool align> void SynetInnerProductLayerForward(const float* src, const float* weight, const float* bias, size_t count, size_t size, float* dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(weight) && Aligned(size) && Aligned(dst));
            size_t partial = AlignLo(size, F);
            size_t aligned = AlignLo(size, QF);
            for (size_t i = 0; i < count; ++i)
            {
                size_t j = 0;
                float sum = 0;
                if (partial)
                {
                    __m128 sums[4] = { _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps() };
                    if (aligned)
                    {
                        for (; j < aligned; j += QF)
                        {
                            SynetInnerProductLayerForward<align>(src, weight, j + 0 * F, sums[0]);
                            SynetInnerProductLayerForward<align>(src, weight, j + 1 * F, sums[1]);
                            SynetInnerProductLayerForward<align>(src, weight, j + 2 * F, sums[2]);
                            SynetInnerProductLayerForward<align>(src, weight, j + 3 * F, sums[3]);
                        }
                        sums[0] = _mm_add_ps(_mm_add_ps(sums[0], sums[1]), _mm_add_ps(sums[2], sums[3]));
                    }
                    for (; j < partial; j += F)
                        SynetInnerProductLayerForward<align>(src, weight, j, sums[0]);
                    sum = ExtractSum(sums[0]);
                }
                for (; j < size; ++j)
                    sum += src[j] * weight[j];
                dst[i] = sum + (bias ? bias[i] : 0);
                weight += size;
            }
        }

        void SynetInnerProductLayerForward(const float* src, const float* weight, const float* bias, size_t count, size_t size, float* dst)
        {
            if (Aligned(src) && Aligned(weight) && Aligned(size) && Aligned(dst))
                SynetInnerProductLayerForward<true>(src, weight, bias, count, size, dst);
            else
                SynetInnerProductLayerForward<false>(src, weight, bias, count, size, dst);
        }

        //-------------------------------------------------------------------------------------------------

        template<int shift> SIMD_INLINE __m128 LoadAtEdge(const float* src)
        {
            static const int32_t mask[3 * F] = { 0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0, 0 };
            return _mm_and_ps(_mm_loadu_ps(src + shift), _mm_loadu_ps((float*)mask + F + shift));
        }

        SIMD_INLINE __m128 NoseSquareSum(const float* src)
        {
            return _mm_add_ps(_mm_add_ps(Square(LoadAtEdge<-2>(src)), Square(LoadAtEdge<-1>(src))),
                _mm_add_ps(Square(_mm_loadu_ps(src)), _mm_add_ps(Square(_mm_loadu_ps(src + 1)), Square(_mm_loadu_ps(src + 2)))));
        }

        SIMD_INLINE __m128 BodySquareSum(const float* src)
        {
            return _mm_add_ps(_mm_add_ps(Square(_mm_loadu_ps(src - 2)), Square(_mm_loadu_ps(src - 1))),
                _mm_add_ps(Square(_mm_loadu_ps(src)), _mm_add_ps(Square(_mm_loadu_ps(src + 1)), Square(_mm_loadu_ps(src + 2)))));
        }

        SIMD_INLINE __m128 TailSquareSum(const float* src)
        {
            return _mm_add_ps(_mm_add_ps(Square(LoadAtEdge<2>(src)), Square(LoadAtEdge<1>(src))),
                _mm_add_ps(Square(_mm_loadu_ps(src)), _mm_add_ps(Square(_mm_loadu_ps(src - 1)), Square(_mm_loadu_ps(src - 2)))));
        }

        template<bool align> void SynetLrnLayerCrossChannelsNchw(const float* src, size_t half, size_t channels, size_t spatial, const float* k, float* dst)
        {
            __m128 k0 = _mm_set1_ps(k[0]);
            __m128 k1 = _mm_set1_ps(k[1]);
            __m128 k2 = _mm_set1_ps(k[2]);
            Pow pow;
            Array32f sum(spatial, true), zero(spatial, true);
            size_t aligned = AlignLo(spatial, F);
            for (size_t c = 0; c < half; ++c)
            {
                const float* pos = src + c * spatial;
                size_t s = 0;
                for (; s < aligned; s += F)
                {
                    __m128 _pos = Load<align>(pos + s);
                    Store<true>(sum.data + s, _mm_add_ps(Load<true>(sum.data + s), _mm_mul_ps(_pos, _pos)));
                }
                for (; s < spatial; ++s)
                    sum[s] += Simd::Square(pos[s]);
            }
            for (size_t c = 0; c < channels; ++c)
            {
                const float* pos = (c < channels - half) ? src + half * spatial : zero.data;
                const float* neg = (c > half) ? src - (half + 1) * spatial : zero.data;
                size_t s = 0;
                for (; s < aligned; s += F)
                {
                    __m128 _pos = Load<align>(pos + s);
                    __m128 _neg = Load<align>(neg + s);
                    __m128 _sum = Load<true>(sum.data + s);
                    _sum = _mm_add_ps(_sum, _mm_sub_ps(_mm_mul_ps(_pos, _pos), _mm_mul_ps(_neg, _neg)));
                    __m128 _src = Load<align>(src + s);
                    Store<true>(sum.data + s, _sum);
                    Store<align>(dst + s, _mm_mul_ps(_src, pow(_mm_add_ps(k0, _mm_mul_ps(k1, _sum)), k2)));
                }
                for (; s < spatial; ++s)
                {
                    sum[s] += Simd::Square(pos[s]);
                    sum[s] -= Simd::Square(neg[s]);
                    dst[s] = src[s] * Base::Pow(k[0] + k[1] * sum[s], k[2]);
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
            __m128 k0 = _mm_set1_ps(k[0]);
            __m128 k1 = _mm_set1_ps(k[1]);
            __m128 k2 = _mm_set1_ps(k[2]);
            Pow pow;
            size_t aligned = AlignLo(channels - half, F);
            for (size_t s = 0; s < spatial; ++s)
            {
                Store<align>(dst + 0, _mm_mul_ps(Load<align>(src + 0), pow(_mm_add_ps(k0, _mm_mul_ps(k1, NoseSquareSum(src + 0))), k2)));
                for (size_t c = F; c < aligned; c += F)
                    Store<align>(dst + c, _mm_mul_ps(Load<align>(src + c), pow(_mm_add_ps(k0, _mm_mul_ps(k1, BodySquareSum(src + c))), k2)));
                if (aligned != channels - half)
                {
                    size_t c = channels - half - F;
                    Store<false>(dst + c, _mm_mul_ps(Load<false>(src + c), pow(_mm_add_ps(k0, _mm_mul_ps(k1, BodySquareSum(src + c))), k2)));
                }
                size_t c = channels - F;
                Store<false>(dst + c, _mm_mul_ps(Load<false>(src + c), pow(_mm_add_ps(k0, _mm_mul_ps(k1, TailSquareSum(src + c))), k2)));
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
                Base::SynetLrnLayerCrossChannels(src, half, channels, spatial, k, dst, SimdTensorFormatNhwc);
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
                size_t channels1DF = AlignLo(channels1, DF);
                if (type == 0)
                {
                    for (size_t s = 0; s < spatial; ++s)
                    {
                        size_t cd = 0, cs0 = 0, cs1 = 0;
                        for (; cs0 < channels0DF; cs0 += DF, cd += F)
                        {
                            __m128 s0 = _mm_loadu_ps(src0 + cs0 + 0);
                            __m128 s1 = _mm_loadu_ps(src0 + cs0 + F);
                            _mm_storeu_ps(dst0 + cd, _mm_shuffle_ps(s0, s1, 0x88));
                            _mm_storeu_ps(dst1 + cd, _mm_shuffle_ps(s0, s1, 0xDD));
                        }
                        for (; cs0 < channels0; cs0 += 2, cd += 1)
                        {
                            dst0[cd] = src0[cs0 + 0];
                            dst1[cd] = src0[cs0 + 1];
                        }
                        for (; cs1 < channels1DF; cs1 += DF, cd += F)
                        {
                            __m128 s0 = _mm_loadu_ps(src1 + cs1 + 0);
                            __m128 s1 = _mm_loadu_ps(src1 + cs1 + F);
                            _mm_storeu_ps(dst0 + cd, _mm_shuffle_ps(s0, s1, 0x88));
                            _mm_storeu_ps(dst1 + cd, _mm_shuffle_ps(s0, s1, 0xDD));
                        }
                        for (; cs1 < channels1; cs1 += 2, cd += 1)
                        {
                            dst0[cd] = src1[cs1 + 0];
                            dst1[cd] = src1[cs1 + 1];
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
                            __m128 s0 = _mm_loadu_ps(src0 + cs);
                            __m128 s1 = _mm_loadu_ps(src1 + cs);
                            _mm_storeu_ps(dst0 + cd0 + 0, _mm_unpacklo_ps(s0, s1));
                            _mm_storeu_ps(dst0 + cd0 + F, _mm_unpackhi_ps(s0, s1));
                        }
                        for (; cd0 < channels0; cd0 += 2, cs += 1)
                        {
                            dst0[cd0 + 0] = src0[cs];
                            dst0[cd0 + 1] = src1[cs];
                        }
                        for (; cd1 < channels1DF; cd1 += DF, cs += F)
                        {
                            __m128 s0 = _mm_loadu_ps(src0 + cs);
                            __m128 s1 = _mm_loadu_ps(src1 + cs);
                            _mm_storeu_ps(dst1 + cd1 + 0, _mm_unpacklo_ps(s0, s1));
                            _mm_storeu_ps(dst1 + cd1 + F, _mm_unpackhi_ps(s0, s1));
                        }
                        for (; cd1 < channels1; cd1 += 2, cs += 1)
                        {
                            dst1[cd1 + 0] = src0[cs];
                            dst1[cd1 + 1] = src1[cs];
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

        SIMD_INLINE void SynetTiledScale2D32fNchw(const float* src, const float* ver, __m128 hor, float* dst, size_t offset)
        {
            _mm_storeu_ps(dst + offset, _mm_mul_ps(_mm_loadu_ps(src + offset), _mm_mul_ps(_mm_loadu_ps(ver + offset), hor)));
        }

        SIMD_INLINE void SynetTiledScale2D32fNhwc(const float* src, const float* ver, const float* hor, float* dst, size_t offset)
        {
            _mm_storeu_ps(dst + offset, _mm_mul_ps(_mm_loadu_ps(src + offset), _mm_mul_ps(_mm_loadu_ps(ver + offset), _mm_loadu_ps(hor + offset))));
        }

        void SynetTiledScale2D32f(const float* src, size_t channels, size_t height, size_t width, SimdTensorFormatType format, const float* ver, const float* hor, float* dst)
        {
            if (format == SimdTensorFormatNchw)
            {
                size_t widthF = AlignLo(width, F);
                size_t width4F = AlignLo(width, 4 * F);
                for (size_t c = 0; c < channels; ++c)
                {
                    for (size_t y = 0; y < height; ++y)
                    {
                        __m128 _hor = _mm_set1_ps(hor[y]);
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
                        for (; x < width; x += 1)
                            _mm_store_ss(dst + x, _mm_mul_ss(_mm_load_ss(src + x), _mm_mul_ss(_mm_load_ss(ver + x), _hor)));
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
                        for (; c < channels; c += 1)
                            _mm_store_ss(dst + c, _mm_mul_ss(_mm_load_ss(src + c), _mm_mul_ss(_mm_load_ss(pVer + c), _mm_load_ss(hor + c))));
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
