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
#include "Simd/SimdResizer.h"
#include "Simd/SimdResizerCommon.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdSet.h"
#include "Simd/SimdUpdate.h"
#include "Simd/SimdEnable.h"
#include "Simd/SimdBFloat16.h"

namespace Simd
{
#ifdef SIMD_AVX2_ENABLE 
    namespace Avx2
    {
        ResizerByteBilinear::ResizerByteBilinear(const ResParam & param)
            : Sse41::ResizerByteBilinear(param)
        {
        }

        void ResizerByteBilinear::EstimateParams()
        {
            if (_ax.data)
                return;
            if (_param.channels == 1 && _param.srcW < 4 * _param.dstW)
                _blocks = BlockCountMax(A);
            float scale = (float)_param.srcW / _param.dstW;
            _ax.Resize(AlignHi(_param.dstW, A) * _param.channels * 2, false, _param.align);
            uint8_t * alphas = _ax.data;
            if (_blocks)
            {
                _ixg.Resize(_blocks);
                int block = 0;
                _ixg[0].src = 0;
                _ixg[0].dst = 0;
                for (int dstIndex = 0; dstIndex < (int)_param.dstW; ++dstIndex)
                {
                    float alpha = (float)((dstIndex + 0.5)*scale - 0.5);
                    int srcIndex = (int)::floor(alpha);
                    alpha -= srcIndex;

                    if (srcIndex < 0)
                    {
                        srcIndex = 0;
                        alpha = 0;
                    }

                    if (srcIndex > (int)_param.srcW - 2)
                    {
                        srcIndex = (int)_param.srcW - 2;
                        alpha = 1;
                    }

                    int dst = 2 * dstIndex - _ixg[block].dst;
                    int src = srcIndex - _ixg[block].src;
                    if (src >= A - 1 || dst >= A)
                    {
                        block++;
                        _ixg[block].src = Simd::Min(srcIndex, int(_param.srcW - A));
                        _ixg[block].dst = 2 * dstIndex;
                        dst = 0;
                        src = srcIndex - _ixg[block].src;
                    }
                    _ixg[block].shuffle[dst] = src;
                    _ixg[block].shuffle[dst + 1] = src + 1;

                    alphas[1] = (uint8_t)(alpha * Base::FRACTION_RANGE + 0.5);
                    alphas[0] = (uint8_t)(Base::FRACTION_RANGE - alphas[1]);
                    alphas += 2;
                }
                _blocks = block + 1;
            }
            else
            {
                _ix.Resize(AlignHi(_param.dstW, _param.align/4), true, _param.align);
                for (size_t i = 0; i < _param.dstW; ++i)
                {
                    float alpha = (float)((i + 0.5)*scale - 0.5);
                    ptrdiff_t index = (ptrdiff_t)::floor(alpha);
                    alpha -= index;

                    if (index < 0)
                    {
                        index = 0;
                        alpha = 0;
                    }

                    if (index >(ptrdiff_t)_param.srcW - 2)
                    {
                        index = _param.srcW - 2;
                        alpha = 1;
                    }

                    _ix[i] = (int)index;
                    alphas[1] = (uint8_t)(alpha * Base::FRACTION_RANGE + 0.5);
                    alphas[0] = (uint8_t)(Base::FRACTION_RANGE - alphas[1]);
                    for (size_t channel = 1; channel < _param.channels; channel++)
                        ((uint16_t*)alphas)[channel] = *(uint16_t*)alphas;
                    alphas += 2 * _param.channels;
                }
            }
            size_t size = AlignHi(_param.dstW, _param.align)*_param.channels * 2 + SIMD_ALIGN;
            _bx[0].Resize(size, false, _param.align);
            _bx[1].Resize(size, false, _param.align);
        }

        template <size_t channelCount> void ResizerByteBilinearInterpolateX(const __m256i * alpha, __m256i * buffer);

        template <> SIMD_INLINE void ResizerByteBilinearInterpolateX<1>(const __m256i * alpha, __m256i * buffer)
        {
#ifdef SIMD_MADDUBS_ERROR
            __m256i _buffer = _mm256_or_si256(K_ZERO, _mm256u_load_si256(buffer));
#else
            __m256i _buffer = _mm256_loadu_si256(buffer);
#endif
            _mm256_storeu_si256(buffer, _mm256_maddubs_epi16(_buffer, _mm256_loadu_si256(alpha)));
        }

        const __m256i K8_SHUFFLE_X2 = SIMD_MM256_SETR_EPI8(0x0, 0x2, 0x1, 0x3, 0x4, 0x6, 0x5, 0x7, 0x8, 0xA, 0x9, 0xB, 0xC, 0xE, 0xD, 0xF,
            0x0, 0x2, 0x1, 0x3, 0x4, 0x6, 0x5, 0x7, 0x8, 0xA, 0x9, 0xB, 0xC, 0xE, 0xD, 0xF);

        SIMD_INLINE void ResizerByteBilinearInterpolateX2(const __m256i * alpha, __m256i * buffer)
        {
            __m256i src = _mm256_shuffle_epi8(_mm256_loadu_si256(buffer), K8_SHUFFLE_X2);
            _mm256_storeu_si256(buffer, _mm256_maddubs_epi16(src, _mm256_loadu_si256(alpha)));
        }

        template <> SIMD_INLINE void ResizerByteBilinearInterpolateX<2>(const __m256i * alpha, __m256i * buffer)
        {
            ResizerByteBilinearInterpolateX2(alpha + 0, buffer + 0);
            ResizerByteBilinearInterpolateX2(alpha + 1, buffer + 1);
        }

        const __m256i K8_SHUFFLE_X3_00 = SIMD_MM256_SETR_EPI8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            0xE, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        const __m256i K8_SHUFFLE_X3_01 = SIMD_MM256_SETR_EPI8(0x0, 0x3, 0x1, 0x4, 0x2, 0x5, 0x6, 0x9, 0x7, 0xA, 0x8, 0xB, 0xC, 0xF, 0xD, -1,
            -1, 0x1, 0x2, 0x5, 0x3, 0x6, 0x4, 0x7, 0x8, 0xB, 0x9, 0xC, 0xA, 0xD, 0xE, -1);
        const __m256i K8_SHUFFLE_X3_02 = SIMD_MM256_SETR_EPI8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0x0,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0x1);

        const __m256i K8_SHUFFLE_X3_10 = SIMD_MM256_SETR_EPI8(0xF, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        const __m256i K8_SHUFFLE_X3_11 = SIMD_MM256_SETR_EPI8(-1, 0x2, 0x0, 0x3, 0x4, 0x7, 0x5, 0x8, 0x6, 0x9, 0xA, 0xD, 0xB, 0xE, 0xC, 0xF,
            0x0, 0x3, 0x1, 0x4, 0x2, 0x5, 0x6, 0x9, 0x7, 0xA, 0x8, 0xB, 0xC, 0xF, 0xD, -1);
        const __m256i K8_SHUFFLE_X3_12 = SIMD_MM256_SETR_EPI8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0x0);

        const __m256i K8_SHUFFLE_X3_20 = SIMD_MM256_SETR_EPI8(0xE, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            0xF, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        const __m256i K8_SHUFFLE_X3_21 = SIMD_MM256_SETR_EPI8(-1, 0x1, 0x2, 0x5, 0x3, 0x6, 0x4, 0x7, 0x8, 0xB, 0x9, 0xC, 0xA, 0xD, 0xE, -1,
            -1, 0x2, 0x0, 0x3, 0x4, 0x7, 0x5, 0x8, 0x6, 0x9, 0xA, 0xD, 0xB, 0xE, 0xC, 0xF);
        const __m256i K8_SHUFFLE_X3_22 = SIMD_MM256_SETR_EPI8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0x1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);

        template <> SIMD_INLINE void ResizerByteBilinearInterpolateX<3>(const __m256i * alpha, __m256i * buffer)
        {
            __m256i src[3], shuffled;
            src[0] = _mm256_loadu_si256(buffer + 0);
            src[1] = _mm256_loadu_si256(buffer + 1);
            src[2] = _mm256_loadu_si256(buffer + 2);

            shuffled = _mm256_shuffle_epi8(_mm256_permute2x128_si256(src[0], src[0], 0x21), K8_SHUFFLE_X3_00);
            shuffled = _mm256_or_si256(shuffled, _mm256_shuffle_epi8(src[0], K8_SHUFFLE_X3_01));
            shuffled = _mm256_or_si256(shuffled, _mm256_shuffle_epi8(_mm256_permute2x128_si256(src[0], src[1], 0x21), K8_SHUFFLE_X3_02));
            _mm256_storeu_si256(buffer + 0, _mm256_maddubs_epi16(shuffled, _mm256_loadu_si256(alpha + 0)));

            shuffled = _mm256_shuffle_epi8(_mm256_permute2x128_si256(src[0], src[1], 0x21), K8_SHUFFLE_X3_10);
            shuffled = _mm256_or_si256(shuffled, _mm256_shuffle_epi8(src[1], K8_SHUFFLE_X3_11));
            shuffled = _mm256_or_si256(shuffled, _mm256_shuffle_epi8(_mm256_permute2x128_si256(src[1], src[2], 0x21), K8_SHUFFLE_X3_12));
            _mm256_storeu_si256(buffer + 1, _mm256_maddubs_epi16(shuffled, _mm256_loadu_si256(alpha + 1)));

            shuffled = _mm256_shuffle_epi8(_mm256_permute2x128_si256(src[1], src[2], 0x21), K8_SHUFFLE_X3_20);
            shuffled = _mm256_or_si256(shuffled, _mm256_shuffle_epi8(src[2], K8_SHUFFLE_X3_21));
            shuffled = _mm256_or_si256(shuffled, _mm256_shuffle_epi8(_mm256_permute2x128_si256(src[2], src[2], 0x21), K8_SHUFFLE_X3_22));
            _mm256_storeu_si256(buffer + 2, _mm256_maddubs_epi16(shuffled, _mm256_loadu_si256(alpha + 2)));
        }

        const __m256i K8_SHUFFLE_X4 = SIMD_MM256_SETR_EPI8(0x0, 0x4, 0x1, 0x5, 0x2, 0x6, 0x3, 0x7, 0x8, 0xC, 0x9, 0xD, 0xA, 0xE, 0xB, 0xF,
            0x0, 0x4, 0x1, 0x5, 0x2, 0x6, 0x3, 0x7, 0x8, 0xC, 0x9, 0xD, 0xA, 0xE, 0xB, 0xF);

        SIMD_INLINE void ResizerByteBilinearInterpolateX4(const __m256i * alpha, __m256i * buffer)
        {
            __m256i src = _mm256_shuffle_epi8(_mm256_loadu_si256(buffer), K8_SHUFFLE_X4);
            _mm256_storeu_si256(buffer, _mm256_maddubs_epi16(src, _mm256_loadu_si256(alpha)));
        }

        template <> SIMD_INLINE void ResizerByteBilinearInterpolateX<4>(const __m256i * alpha, __m256i * buffer)
        {
            ResizerByteBilinearInterpolateX4(alpha + 0, buffer + 0);
            ResizerByteBilinearInterpolateX4(alpha + 1, buffer + 1);
            ResizerByteBilinearInterpolateX4(alpha + 2, buffer + 2);
            ResizerByteBilinearInterpolateX4(alpha + 3, buffer + 3);
        }

        const __m256i K16_FRACTION_ROUND_TERM = SIMD_MM256_SET1_EPI16(Base::BILINEAR_ROUND_TERM);

        template<bool align> SIMD_INLINE __m256i ResizerByteBilinearInterpolateY(const __m256i * pbx0, const __m256i * pbx1, __m256i alpha[2])
        {
            __m256i sum = _mm256_add_epi16(_mm256_mullo_epi16(Load<align>(pbx0), alpha[0]), _mm256_mullo_epi16(Load<align>(pbx1), alpha[1]));
            return _mm256_srli_epi16(_mm256_add_epi16(sum, K16_FRACTION_ROUND_TERM), Base::BILINEAR_SHIFT);
        }

        template<bool align> SIMD_INLINE void ResizerByteBilinearInterpolateY(const uint8_t * bx0, const uint8_t * bx1, __m256i alpha[2], uint8_t * dst)
        {
            __m256i lo = ResizerByteBilinearInterpolateY<align>((__m256i*)bx0 + 0, (__m256i*)bx1 + 0, alpha);
            __m256i hi = ResizerByteBilinearInterpolateY<align>((__m256i*)bx0 + 1, (__m256i*)bx1 + 1, alpha);
            Store<false>((__m256i*)dst, PackI16ToU8(lo, hi));
        }

        template<size_t N> void ResizerByteBilinear::Run(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride)
        {
            struct One { uint8_t val[N * 1]; };
            struct Two { uint8_t val[N * 2]; };

            size_t size = 2 * _param.dstW*N;
            size_t aligned = AlignHi(size, DA) - DA;
            const size_t step = A * N;
            size_t dstW = _param.dstW;
            ptrdiff_t previous = -2;
            __m256i a[2];
            uint8_t * bx[2] = { _bx[0].data, _bx[1].data };
            const uint8_t * ax = _ax.data;
            const int32_t * ix = _ix.data;

            for (size_t yDst = 0; yDst < _param.dstH; yDst++, dst += dstStride)
            {
                a[0] = _mm256_set1_epi16(int16_t(Base::FRACTION_RANGE - _ay[yDst]));
                a[1] = _mm256_set1_epi16(int16_t(_ay[yDst]));

                ptrdiff_t sy = _iy[yDst];
                int k = 0;

                if (sy == previous)
                    k = 2;
                else if (sy == previous + 1)
                {
                    Swap(bx[0], bx[1]);
                    k = 1;
                }

                previous = sy;

                for (; k < 2; k++)
                {
                    Two * pb = (Two *)bx[k];
                    const One * psrc = (const One *)(src + (sy + k)*srcStride);
                    for (size_t x = 0; x < dstW; x++)
                        pb[x] = *(Two *)(psrc + ix[x]);

                    uint8_t * pbx = bx[k];
                    for (size_t i = 0; i < size; i += step)
                        ResizerByteBilinearInterpolateX<N>((__m256i*)(ax + i), (__m256i*)(pbx + i));
                }

                for (size_t ib = 0, id = 0; ib < aligned; ib += DA, id += A)
                    ResizerByteBilinearInterpolateY<true>(bx[0] + ib, bx[1] + ib, a, dst + id);
                size_t i = size - DA;
                ResizerByteBilinearInterpolateY<false>(bx[0] + i, bx[1] + i, a, dst + i / 2);
            }
        }

        void ResizerByteBilinear::RunG(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride)
        {
            size_t bufW = AlignHi(_param.dstW, A) * 2;
            size_t size = 2 * _param.dstW;
            size_t aligned = AlignHi(size, DA) - DA;
            size_t blocks = _blocks;
            ptrdiff_t previous = -2;
            __m256i a[2];
            uint8_t * bx[2] = { _bx[0].data, _bx[1].data };
            const uint8_t * ax = _ax.data;
            const Idx * ixg = _ixg.data;

            for (size_t yDst = 0; yDst < _param.dstH; yDst++, dst += dstStride)
            {
                a[0] = _mm256_set1_epi16(int16_t(Base::FRACTION_RANGE - _ay[yDst]));
                a[1] = _mm256_set1_epi16(int16_t(_ay[yDst]));

                ptrdiff_t sy = _iy[yDst];
                int k = 0;

                if (sy == previous)
                    k = 2;
                else if (sy == previous + 1)
                {
                    Swap(bx[0], bx[1]);
                    k = 1;
                }

                previous = sy;

                for (; k < 2; k++)
                {
                    const uint8_t * psrc = src + (sy + k)*srcStride;
                    uint8_t * pdst = bx[k];
                    for (size_t i = 0; i < blocks; ++i)
                        ResizerByteBilinearLoadGrayInterpolated(psrc, ixg[i], ax, pdst);
                }

                for (size_t ib = 0, id = 0; ib < aligned; ib += DA, id += A)
                    ResizerByteBilinearInterpolateY<true>(bx[0] + ib, bx[1] + ib, a, dst + id);
                size_t i = size - DA;
                ResizerByteBilinearInterpolateY<false>(bx[0] + i, bx[1] + i, a, dst + i / 2);
            }
        }

        void ResizerByteBilinear::Run(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride)
        {
            assert(_param.dstW >= A);

            EstimateParams();
            switch (_param.channels)
            {
            case 1:
                if (_blocks)
                    RunG(src, srcStride, dst, dstStride);
                else
                    Run<1>(src, srcStride, dst, dstStride);
                break;
            case 2: Run<2>(src, srcStride, dst, dstStride); break;
            case 3: Run<3>(src, srcStride, dst, dstStride); break;
            case 4: Run<4>(src, srcStride, dst, dstStride); break;
            default:
                assert(0);
            }
        }

        //---------------------------------------------------------------------

        ResizerShortBilinear::ResizerShortBilinear(const ResParam& param)
            : Sse41::ResizerShortBilinear(param)
        {
        }

        const __m256i RSB_1_0 = SIMD_MM256_SETR_EPI8(
            0x0, 0x1, -1, -1, 0x4, 0x5, -1, -1, 0x8, 0x9, -1, -1, 0xC, 0xD, -1, -1,
            0x0, 0x1, -1, -1, 0x4, 0x5, -1, -1, 0x8, 0x9, -1, -1, 0xC, 0xD, -1, -1);
        const __m256i RSB_1_1 = SIMD_MM256_SETR_EPI8(
            0x2, 0x3, -1, -1, 0x6, 0x7, -1, -1, 0xA, 0xB, -1, -1, 0xE, 0xF, -1, -1,
            0x2, 0x3, -1, -1, 0x6, 0x7, -1, -1, 0xA, 0xB, -1, -1, 0xE, 0xF, -1, -1);

        SIMD_INLINE __m256 BilColS1(const uint16_t* src, const int32_t* idx, __m256 fx0, __m256 fx1)
        {
            __m256i s = _mm256_i32gather_epi32((int32_t*)src, _mm256_loadu_si256((__m256i*)idx), 2);
            __m256 m0 = _mm256_mul_ps(fx0, _mm256_cvtepi32_ps(_mm256_shuffle_epi8(s, RSB_1_0)));
            __m256 m1 = _mm256_mul_ps(fx1, _mm256_cvtepi32_ps(_mm256_shuffle_epi8(s, RSB_1_1)));
            return _mm256_add_ps(m0, m1);
        }

        SIMD_INLINE __m256 BilColS1NoGather(const uint16_t* src, const int32_t* idx, __m256 fx0, __m256 fx1)
        {
            __m256i s = _mm256_setr_epi32(
                *(uint32_t*)(src + idx[0]), *(uint32_t*)(src + idx[1]),
                *(uint32_t*)(src + idx[2]), *(uint32_t*)(src + idx[3]),
                *(uint32_t*)(src + idx[4]), *(uint32_t*)(src + idx[5]),
                *(uint32_t*)(src + idx[6]), *(uint32_t*)(src + idx[7]));
            __m256 m0 = _mm256_mul_ps(fx0, _mm256_cvtepi32_ps(_mm256_shuffle_epi8(s, RSB_1_0)));
            __m256 m1 = _mm256_mul_ps(fx1, _mm256_cvtepi32_ps(_mm256_shuffle_epi8(s, RSB_1_1)));
            return _mm256_add_ps(m0, m1);
        }

        const __m256i RSB_2_0 = SIMD_MM256_SETR_EPI8(
            0x0, 0x1, -1, -1, 0x2, 0x3, -1, -1, 0x8, 0x9, -1, -1, 0xA, 0xB, -1, -1,
            0x0, 0x1, -1, -1, 0x2, 0x3, -1, -1, 0x8, 0x9, -1, -1, 0xA, 0xB, -1, -1);
        const __m256i RSB_2_1 = SIMD_MM256_SETR_EPI8(
            0x4, 0x5, -1, -1, 0x6, 0x7, -1, -1, 0xC, 0xD, -1, -1, 0xE, 0xF, -1, -1,
            0x4, 0x5, -1, -1, 0x6, 0x7, -1, -1, 0xC, 0xD, -1, -1, 0xE, 0xF, -1, -1);

        SIMD_INLINE __m256 BilColS2(const uint16_t* src, const int32_t* idx, __m256 fx0, __m256 fx1)
        {
            __m256i s = _mm256_setr_epi64x(
                *(uint64_t*)(src + idx[0]), *(uint64_t*)(src + idx[2]),
                *(uint64_t*)(src + idx[4]), *(uint64_t*)(src + idx[6]));
            //__m256i s = _mm256_i64gather_epi64((long long*)src, _mm256_and_si256(_mm256_loadu_si256((__m256i*)idx), K64_00000000FFFFFFFF), 2);
            __m256 m0 = _mm256_mul_ps(fx0, _mm256_cvtepi32_ps(_mm256_shuffle_epi8(s, RSB_2_0)));
            __m256 m1 = _mm256_mul_ps(fx1, _mm256_cvtepi32_ps(_mm256_shuffle_epi8(s, RSB_2_1)));
            return _mm256_add_ps(m0, m1);
        }

        const __m256i RSB_3_0 = SIMD_MM256_SETR_EPI8(
            0x0, 0x1, -1, -1, 0x2, 0x3, -1, -1, 0x4, 0x5, -1, -1, -1, -1, -1, -1,
            0x0, 0x1, -1, -1, 0x2, 0x3, -1, -1, 0x4, 0x5, -1, -1, -1, -1, -1, -1);
        const __m256i RSB_3_1 = SIMD_MM256_SETR_EPI8(
            0x6, 0x7, -1, -1, 0x8, 0x9, -1, -1, 0xA, 0xB, -1, -1, -1, -1, -1, -1,
            0x6, 0x7, -1, -1, 0x8, 0x9, -1, -1, 0xA, 0xB, -1, -1, -1, -1, -1, -1);
        const __m256i RSB_3_P1 = SIMD_MM256_SETR_EPI32(0, 1, 2, 4, 5, 6, 7, 7);

        SIMD_INLINE __m256 BilColS3(const uint16_t* src, const int32_t* idx, __m256 fx0, __m256 fx1)
        {
            __m256i s = Load<false>((__m128i*)(src + idx[0]), (__m128i*)(src + idx[3]));
            __m256 m0 = _mm256_mul_ps(fx0, _mm256_cvtepi32_ps(_mm256_shuffle_epi8(s, RSB_3_0)));
            __m256 m1 = _mm256_mul_ps(fx1, _mm256_cvtepi32_ps(_mm256_shuffle_epi8(s, RSB_3_1)));
            return _mm256_permutevar8x32_ps(_mm256_add_ps(m0, m1), RSB_3_P1);
        }

        const __m256i RSB_4_0 = SIMD_MM256_SETR_EPI8(
            0x0, 0x1, -1, -1, 0x2, 0x3, -1, -1, 0x4, 0x5, -1, -1, 0x6, 0x7, -1, -1,
            0x0, 0x1, -1, -1, 0x2, 0x3, -1, -1, 0x4, 0x5, -1, -1, 0x6, 0x7, -1, -1);
        const __m256i RSB_4_1 = SIMD_MM256_SETR_EPI8(
            0x8, 0x9, -1, -1, 0xA, 0xB, -1, -1, 0xC, 0xD, -1, -1, 0xE, 0xF, -1, -1,
            0x8, 0x9, -1, -1, 0xA, 0xB, -1, -1, 0xC, 0xD, -1, -1, 0xE, 0xF, -1, -1);

        SIMD_INLINE __m256 BilColS4(const uint16_t* src, const int32_t* idx, __m256 fx0, __m256 fx1)
        {
            __m256i s = Load<false>((__m128i*)(src + idx[0]), (__m128i*)(src + idx[4]));
            __m256 m0 = _mm256_mul_ps(fx0, _mm256_cvtepi32_ps(_mm256_shuffle_epi8(s, RSB_4_0)));
            __m256 m1 = _mm256_mul_ps(fx1, _mm256_cvtepi32_ps(_mm256_shuffle_epi8(s, RSB_4_1)));
            return _mm256_add_ps(m0, m1);
        }

        template<size_t N> void ResizerShortBilinear::RunB(const uint16_t* src, size_t srcStride, uint16_t* dst, size_t dstStride)
        {
            size_t rs = _param.dstW * N;
            float* pbx[2] = { _bx[0].data, _bx[1].data };
            int32_t prev = -2;
            size_t rs3 = AlignLoAny(rs - 1, 3);
            size_t rs6 = AlignLoAny(rs - 1, 6);
            size_t rs4 = AlignLo(rs, 4);
            size_t rs8 = AlignLo(rs, 8);
            size_t rs16 = AlignLo(rs, 16);
            __m256 _1 = _mm256_set1_ps(1.0f);
            for (size_t dy = 0; dy < _param.dstH; dy++, dst += dstStride)
            {
                float fy1 = _ay[dy];
                float fy0 = 1.0f - fy1;
                int32_t sy = _iy[dy];
                int32_t k = 0;

                if (sy == prev)
                    k = 2;
                else if (sy == prev + 1)
                {
                    Swap(pbx[0], pbx[1]);
                    k = 1;
                }

                prev = sy;

                for (; k < 2; k++)
                {
                    float* pb = pbx[k];
                    const uint16_t* ps = src + (sy + k) * srcStride;
                    size_t dx = 0;
                    if (N == 1)
                    {
                        if (Avx2::SlowGather)
                        {
                            for (; dx < rs8; dx += 8)
                            {
                                __m256 fx1 = _mm256_loadu_ps(_ax.data + dx);
                                __m256 fx0 = _mm256_sub_ps(_1, fx1);
                                _mm256_storeu_ps(pb + dx, BilColS1NoGather(ps, _ix.data + dx, fx0, fx1));
                            }
                        }
                        else
                        {
                            for (; dx < rs8; dx += 8)
                            {
                                __m256 fx1 = _mm256_loadu_ps(_ax.data + dx);
                                __m256 fx0 = _mm256_sub_ps(_1, fx1);
                                _mm256_storeu_ps(pb + dx, BilColS1(ps, _ix.data + dx, fx0, fx1));
                            }
                        }
                        for (; dx < rs4; dx += 4)
                        {
                            __m128 fx1 = _mm_loadu_ps(_ax.data + dx);
                            __m128 fx0 = _mm_sub_ps(_mm256_castps256_ps128(_1), fx1);
                            _mm_storeu_ps(pb + dx, Sse41::BilColS1(ps, _ix.data + dx, fx0, fx1));
                        }
                    }
                    if (N == 2)
                    {
                        for (; dx < rs8; dx += 8)
                        {
                            __m256 fx1 = _mm256_loadu_ps(_ax.data + dx);
                            __m256 fx0 = _mm256_sub_ps(_1, fx1);
                            _mm256_storeu_ps(pb + dx, BilColS2(ps, _ix.data + dx, fx0, fx1));
                        }
                        for (; dx < rs4; dx += 4)
                        {
                            __m128 fx1 = _mm_loadu_ps(_ax.data + dx);
                            __m128 fx0 = _mm_sub_ps(_mm256_castps256_ps128(_1), fx1);
                            _mm_storeu_ps(pb + dx, Sse41::BilColS2(ps, _ix.data + dx, fx0, fx1));
                        }
                    }
                    if (N == 3)
                    {
                        for (; dx < rs6; dx += 6)
                        {
                            __m256 fx1 = Load<false>(_ax.data + dx, _ax.data + dx + 3);
                            __m256 fx0 = _mm256_sub_ps(_1, fx1);
                            _mm256_storeu_ps(pb + dx, BilColS3(ps, _ix.data + dx, fx0, fx1));
                        }
                        for (; dx < rs3; dx += 3)
                        {
                            __m128 fx1 = _mm_loadu_ps(_ax.data + dx);
                            __m128 fx0 = _mm_sub_ps(_mm256_castps256_ps128(_1), fx1);
                            _mm_storeu_ps(pb + dx, Sse41::BilColS3(ps + _ix[dx], fx0, fx1));
                        }
                    }
                    if (N == 4)
                    {
                        for (; dx < rs8; dx += 8)
                        {
                            __m256 fx1 = _mm256_loadu_ps(_ax.data + dx);
                            __m256 fx0 = _mm256_sub_ps(_1, fx1);
                            _mm256_storeu_ps(pb + dx, BilColS4(ps, _ix.data + dx, fx0, fx1));
                        }
                        for (; dx < rs4; dx += 4)
                        {
                            __m128 fx1 = _mm_loadu_ps(_ax.data + dx);
                            __m128 fx0 = _mm_sub_ps(_mm256_castps256_ps128(_1), fx1);
                            _mm_storeu_ps(pb + dx, Sse41::BilColS4(ps + _ix[dx], fx0, fx1));
                        }
                    }
                    for (; dx < rs; dx++)
                    {
                        int32_t sx = _ix[dx];
                        float fx = _ax[dx];
                        pb[dx] = ps[sx] * (1.0f - fx) + ps[sx + N] * fx;
                    }
                }

                size_t dx = 0;
                __m256 _fy0 = _mm256_set1_ps(fy0);
                __m256 _fy1 = _mm256_set1_ps(fy1);
                for (; dx < rs16; dx += 16)
                {
                    __m256 m00 = _mm256_mul_ps(_mm256_loadu_ps(pbx[0] + dx + 0), _fy0);
                    __m256 m01 = _mm256_mul_ps(_mm256_loadu_ps(pbx[1] + dx + 0), _fy1);
                    __m256i i0 = _mm256_cvttps_epi32(_mm256_add_ps(m00, m01));
                    __m256 m10 = _mm256_mul_ps(_mm256_loadu_ps(pbx[0] + dx + 8), _fy0);
                    __m256 m11 = _mm256_mul_ps(_mm256_loadu_ps(pbx[1] + dx + 8), _fy1);
                    __m256i i1 = _mm256_cvttps_epi32(_mm256_add_ps(m10, m11));
                    _mm256_storeu_si256((__m256i*)(dst + dx), PackU32ToI16(i0, i1));
                }
                for (; dx < rs8; dx += 8)
                {
                    __m256 m0 = _mm256_mul_ps(_mm256_loadu_ps(pbx[0] + dx), _fy0);
                    __m256 m1 = _mm256_mul_ps(_mm256_loadu_ps(pbx[1] + dx), _fy1);
                    __m256i i0 = _mm256_cvttps_epi32(_mm256_add_ps(m0, m1));
                    _mm_storeu_si128((__m128i*)(dst + dx), _mm256_castsi256_si128(PackU32ToI16(i0, K_ZERO)));
                }
                for (; dx < rs4; dx += 4)
                {
                    __m128 m0 = _mm_mul_ps(_mm_loadu_ps(pbx[0] + dx), _mm256_castps256_ps128(_fy0));
                    __m128 m1 = _mm_mul_ps(_mm_loadu_ps(pbx[1] + dx), _mm256_castps256_ps128(_fy1));
                    __m128i i0 = _mm_cvttps_epi32(_mm_add_ps(m0, m1));
                    _mm_storel_epi64((__m128i*)(dst + dx), _mm_packus_epi32(i0, Sse41::K_ZERO));
                }
                for (; dx < rs; dx++)
                    dst[dx] = Round(pbx[0][dx] * fy0 + pbx[1][dx] * fy1);
            }
        }

        const __m256i RSB_3_P2 = SIMD_MM256_SETR_EPI32(0, 1, 4, 2, 3, 6, 5, 7);

        SIMD_INLINE __m256i PackU32ToI16Rsb3(__m256i lo, __m256i hi)
        {
            return _mm256_permutevar8x32_epi32(_mm256_packus_epi32(lo, hi), RSB_3_P2);
        }

        template<size_t N> void ResizerShortBilinear::RunS(const uint16_t* src, size_t srcStride, uint16_t* dst, size_t dstStride)
        {
            size_t rs = _param.dstW * N;
            size_t rs3 = AlignLoAny(rs - 1, 3);
            size_t rs6 = AlignLoAny(rs - 1, 6);
            size_t rs12 = AlignLoAny(rs - 1, 12);
            size_t rs4 = AlignLo(rs, 4);
            size_t rs8 = AlignLo(rs, 8);
            size_t rs16 = AlignLo(rs, 16);
            __m256 _1 = _mm256_set1_ps(1.0f);
            for (size_t dy = 0; dy < _param.dstH; dy++, dst += dstStride)
            {
                float fy1 = _ay[dy];
                float fy0 = 1.0f - fy1;
                int32_t sy = _iy[dy];
                const uint16_t* ps0 = src + (sy + 0) * srcStride;
                const uint16_t* ps1 = src + (sy + 1) * srcStride;
                size_t dx = 0;
                __m256 _fy0 = _mm256_set1_ps(fy0);
                __m256 _fy1 = _mm256_set1_ps(fy1);
                if (N == 1)
                {
                    if (Avx2::SlowGather)
                    {
                        for (; dx < rs16; dx += 16)
                        {
                            __m256 fx01 = _mm256_loadu_ps(_ax.data + dx + 0);
                            __m256 fx00 = _mm256_sub_ps(_1, fx01);
                            __m256 m00 = _mm256_mul_ps(BilColS1NoGather(ps0, _ix.data + dx + 0, fx00, fx01), _fy0);
                            __m256 m01 = _mm256_mul_ps(BilColS1NoGather(ps1, _ix.data + dx + 0, fx00, fx01), _fy1);
                            __m256i i0 = _mm256_cvttps_epi32(_mm256_add_ps(m00, m01));
                            __m256 fx11 = _mm256_loadu_ps(_ax.data + dx + 8);
                            __m256 fx10 = _mm256_sub_ps(_1, fx11);
                            __m256 m10 = _mm256_mul_ps(BilColS1NoGather(ps0, _ix.data + dx + 8, fx10, fx11), _fy0);
                            __m256 m11 = _mm256_mul_ps(BilColS1NoGather(ps1, _ix.data + dx + 8, fx10, fx11), _fy1);
                            __m256i i1 = _mm256_cvttps_epi32(_mm256_add_ps(m10, m11));
                            _mm256_storeu_si256((__m256i*)(dst + dx), PackU32ToI16(i0, i1));
                        }
                        for (; dx < rs8; dx += 8)
                        {
                            __m256 fx1 = _mm256_loadu_ps(_ax.data + dx);
                            __m256 fx0 = _mm256_sub_ps(_1, fx1);
                            __m256 m0 = _mm256_mul_ps(BilColS1NoGather(ps0, _ix.data + dx, fx0, fx1), _fy0);
                            __m256 m1 = _mm256_mul_ps(BilColS1NoGather(ps1, _ix.data + dx, fx0, fx1), _fy1);
                            __m256i i0 = _mm256_cvttps_epi32(_mm256_add_ps(m0, m1));
                            _mm_storeu_si128((__m128i*)(dst + dx), _mm256_castsi256_si128(PackU32ToI16(i0, K_ZERO)));
                        }
                    }
                    else
                    {
                        for (; dx < rs16; dx += 16)
                        {
                            __m256 fx01 = _mm256_loadu_ps(_ax.data + dx + 0);
                            __m256 fx00 = _mm256_sub_ps(_1, fx01);
                            __m256 m00 = _mm256_mul_ps(BilColS1(ps0, _ix.data + dx + 0, fx00, fx01), _fy0);
                            __m256 m01 = _mm256_mul_ps(BilColS1(ps1, _ix.data + dx + 0, fx00, fx01), _fy1);
                            __m256i i0 = _mm256_cvttps_epi32(_mm256_add_ps(m00, m01));
                            __m256 fx11 = _mm256_loadu_ps(_ax.data + dx + 8);
                            __m256 fx10 = _mm256_sub_ps(_1, fx11);
                            __m256 m10 = _mm256_mul_ps(BilColS1(ps0, _ix.data + dx + 8, fx10, fx11), _fy0);
                            __m256 m11 = _mm256_mul_ps(BilColS1(ps1, _ix.data + dx + 8, fx10, fx11), _fy1);
                            __m256i i1 = _mm256_cvttps_epi32(_mm256_add_ps(m10, m11));
                            _mm256_storeu_si256((__m256i*)(dst + dx), PackU32ToI16(i0, i1));
                        }
                        for (; dx < rs8; dx += 8)
                        {
                            __m256 fx1 = _mm256_loadu_ps(_ax.data + dx);
                            __m256 fx0 = _mm256_sub_ps(_1, fx1);
                            __m256 m0 = _mm256_mul_ps(BilColS1(ps0, _ix.data + dx, fx0, fx1), _fy0);
                            __m256 m1 = _mm256_mul_ps(BilColS1(ps1, _ix.data + dx, fx0, fx1), _fy1);
                            __m256i i0 = _mm256_cvttps_epi32(_mm256_add_ps(m0, m1));
                            _mm_storeu_si128((__m128i*)(dst + dx), _mm256_castsi256_si128(PackU32ToI16(i0, K_ZERO)));
                        }
                    }
                    for (; dx < rs4; dx += 4)
                    {
                        __m128 fx1 = _mm_loadu_ps(_ax.data + dx);
                        __m128 fx0 = _mm_sub_ps(_mm256_castps256_ps128(_1), fx1);
                        __m128 m0 = _mm_mul_ps(Sse41::BilColS1(ps0, _ix.data + dx, fx0, fx1), _mm256_castps256_ps128(_fy0));
                        __m128 m1 = _mm_mul_ps(Sse41::BilColS1(ps1, _ix.data + dx, fx0, fx1), _mm256_castps256_ps128(_fy1));
                        __m128i i0 = _mm_cvttps_epi32(_mm_add_ps(m0, m1));
                        _mm_storel_epi64((__m128i*)(dst + dx), _mm_packus_epi32(i0, Sse41::K_ZERO));
                    }
                }
                if (N == 2)
                {
                    for (; dx < rs16; dx += 16)
                    {
                        __m256 fx01 = _mm256_loadu_ps(_ax.data + dx + 0);
                        __m256 fx00 = _mm256_sub_ps(_1, fx01);
                        __m256 m00 = _mm256_mul_ps(BilColS2(ps0, _ix.data + dx + 0, fx00, fx01), _fy0);
                        __m256 m01 = _mm256_mul_ps(BilColS2(ps1, _ix.data + dx + 0, fx00, fx01), _fy1);
                        __m256i i0 = _mm256_cvttps_epi32(_mm256_add_ps(m00, m01));
                        __m256 fx11 = _mm256_loadu_ps(_ax.data + dx + 8);
                        __m256 fx10 = _mm256_sub_ps(_1, fx11);
                        __m256 m10 = _mm256_mul_ps(BilColS2(ps0, _ix.data + dx + 8, fx10, fx11), _fy0);
                        __m256 m11 = _mm256_mul_ps(BilColS2(ps1, _ix.data + dx + 8, fx10, fx11), _fy1);
                        __m256i i1 = _mm256_cvttps_epi32(_mm256_add_ps(m10, m11));
                        _mm256_storeu_si256((__m256i*)(dst + dx), PackU32ToI16(i0, i1));
                    }
                    for (; dx < rs8; dx += 8)
                    {
                        __m256 fx1 = _mm256_loadu_ps(_ax.data + dx);
                        __m256 fx0 = _mm256_sub_ps(_1, fx1);
                        __m256 m0 = _mm256_mul_ps(BilColS2(ps0, _ix.data + dx, fx0, fx1), _fy0);
                        __m256 m1 = _mm256_mul_ps(BilColS2(ps1, _ix.data + dx, fx0, fx1), _fy1);
                        __m256i i0 = _mm256_cvttps_epi32(_mm256_add_ps(m0, m1));
                        _mm_storeu_si128((__m128i*)(dst + dx), _mm256_castsi256_si128(PackU32ToI16(i0, K_ZERO)));
                    }
                    for (; dx < rs4; dx += 4)
                    {
                        __m128 fx1 = _mm_loadu_ps(_ax.data + dx);
                        __m128 fx0 = _mm_sub_ps(_mm256_castps256_ps128(_1), fx1);
                        __m128 m0 = _mm_mul_ps(Sse41::BilColS2(ps0, _ix.data + dx, fx0, fx1), _mm256_castps256_ps128(_fy0));
                        __m128 m1 = _mm_mul_ps(Sse41::BilColS2(ps1, _ix.data + dx, fx0, fx1), _mm256_castps256_ps128(_fy1));
                        __m128i i0 = _mm_cvttps_epi32(_mm_add_ps(m0, m1));
                        _mm_storel_epi64((__m128i*)(dst + dx), _mm_packus_epi32(i0, Sse41::K_ZERO));
                    }
                }
                if (N == 3)
                {
                    for (; dx < rs12; dx += 12)
                    {
                        __m256 fx01 = Load<false>(_ax.data + dx + 0, _ax.data + dx + 3);
                        __m256 fx00 = _mm256_sub_ps(_1, fx01);
                        __m256 m00 = _mm256_mul_ps(BilColS3(ps0, _ix.data + dx, fx00, fx01), _fy0);
                        __m256 m01 = _mm256_mul_ps(BilColS3(ps1, _ix.data + dx, fx00, fx01), _fy1);
                        __m256i i0 = _mm256_cvttps_epi32(_mm256_add_ps(m00, m01));
                        __m256 fx11 = Load<false>(_ax.data + dx + 6, _ax.data + dx + 9);
                        __m256 fx10 = _mm256_sub_ps(_1, fx11);
                        __m256 m10 = _mm256_mul_ps(BilColS3(ps0, _ix.data + dx + 6, fx10, fx11), _fy0);
                        __m256 m11 = _mm256_mul_ps(BilColS3(ps1, _ix.data + dx + 6, fx10, fx11), _fy1);
                        __m256i i1 = _mm256_cvttps_epi32(_mm256_add_ps(m10, m11));
                        _mm256_storeu_si256((__m256i*)(dst + dx), PackU32ToI16Rsb3(i0, i1));
                    }
                    for (; dx < rs6; dx += 6)
                    {
                        __m256 fx1 = Load<false>(_ax.data + dx, _ax.data + dx + 3);
                        __m256 fx0 = _mm256_sub_ps(_1, fx1);
                        __m256 m0 = _mm256_mul_ps(BilColS3(ps0, _ix.data + dx, fx0, fx1), _fy0);
                        __m256 m1 = _mm256_mul_ps(BilColS3(ps1, _ix.data + dx, fx0, fx1), _fy1);
                        __m256i i0 = _mm256_cvttps_epi32(_mm256_add_ps(m0, m1));
                        _mm_storeu_si128((__m128i*)(dst + dx), _mm256_castsi256_si128(PackU32ToI16(i0, K_ZERO)));
                    }
                    for (; dx < rs3; dx += 3)
                    {
                        __m128 fx1 = _mm_loadu_ps(_ax.data + dx);
                        __m128 fx0 = _mm_sub_ps(_mm256_castps256_ps128(_1), fx1);
                        __m128 m0 = _mm_mul_ps(Sse41::BilColS3(ps0 + _ix[dx], fx0, fx1), _mm256_castps256_ps128(_fy0));
                        __m128 m1 = _mm_mul_ps(Sse41::BilColS3(ps1 + _ix[dx], fx0, fx1), _mm256_castps256_ps128(_fy1));
                        __m128i i0 = _mm_cvttps_epi32(_mm_add_ps(m0, m1));
                        _mm_storel_epi64((__m128i*)(dst + dx), _mm_packus_epi32(i0, Sse41::K_ZERO));
                    }
                }
                if (N == 4)
                {
                    for (; dx < rs16; dx += 16)
                    {
                        __m256 fx01 = _mm256_loadu_ps(_ax.data + dx + 0);
                        __m256 fx00 = _mm256_sub_ps(_1, fx01);
                        __m256 m00 = _mm256_mul_ps(BilColS4(ps0, _ix.data + dx + 0, fx00, fx01), _fy0);
                        __m256 m01 = _mm256_mul_ps(BilColS4(ps1, _ix.data + dx + 0, fx00, fx01), _fy1);
                        __m256i i0 = _mm256_cvttps_epi32(_mm256_add_ps(m00, m01));
                        __m256 fx11 = _mm256_loadu_ps(_ax.data + dx + 8);
                        __m256 fx10 = _mm256_sub_ps(_1, fx11);
                        __m256 m10 = _mm256_mul_ps(BilColS4(ps0, _ix.data + dx + 8, fx10, fx11), _fy0);
                        __m256 m11 = _mm256_mul_ps(BilColS4(ps1, _ix.data + dx + 8, fx10, fx11), _fy1);
                        __m256i i1 = _mm256_cvttps_epi32(_mm256_add_ps(m10, m11));
                        _mm256_storeu_si256((__m256i*)(dst + dx), PackU32ToI16(i0, i1));
                    }
                    for (; dx < rs8; dx += 8)
                    {
                        __m256 fx1 = _mm256_loadu_ps(_ax.data + dx);
                        __m256 fx0 = _mm256_sub_ps(_1, fx1);
                        __m256 m0 = _mm256_mul_ps(BilColS4(ps0, _ix.data + dx, fx0, fx1), _fy0);
                        __m256 m1 = _mm256_mul_ps(BilColS4(ps1, _ix.data + dx, fx0, fx1), _fy1);
                        __m256i i0 = _mm256_cvttps_epi32(_mm256_add_ps(m0, m1));
                        _mm_storeu_si128((__m128i*)(dst + dx), _mm256_castsi256_si128(PackU32ToI16(i0, K_ZERO)));
                    }
                    for (; dx < rs4; dx += 4)
                    {
                        __m128 fx1 = _mm_loadu_ps(_ax.data + dx);
                        __m128 fx0 = _mm_sub_ps(_mm256_castps256_ps128(_1), fx1);
                        __m128 m0 = _mm_mul_ps(Sse41::BilColS4(ps0 + _ix[dx], fx0, fx1), _mm256_castps256_ps128(_fy0));
                        __m128 m1 = _mm_mul_ps(Sse41::BilColS4(ps1 + _ix[dx], fx0, fx1), _mm256_castps256_ps128(_fy1));
                        __m128i i0 = _mm_cvttps_epi32(_mm_add_ps(m0, m1));
                        _mm_storel_epi64((__m128i*)(dst + dx), _mm_packus_epi32(i0, Sse41::K_ZERO));
                    }
                }
                for (; dx < rs; dx++)
                {
                    int32_t sx = _ix[dx];
                    float fx1 = _ax[dx];
                    float fx0 = 1.0f - fx1;
                    float r0 = ps0[sx] * fx0 + ps0[sx + N] * fx1;
                    float r1 = ps1[sx] * fx0 + ps1[sx + N] * fx1;
                    dst[dx] = Round(r0 * fy0 + r1 * fy1);
                }
            }
        }

        void ResizerShortBilinear::Run(const uint16_t* src, size_t srcStride, uint16_t* dst, size_t dstStride)
        {
            bool sparse = _param.dstH * 2.0 <= _param.srcH;
            switch (_param.channels)
            {
            case 1: sparse ? RunS<1>(src, srcStride, dst, dstStride) : RunB<1>(src, srcStride, dst, dstStride); return;
            case 2: sparse ? RunS<2>(src, srcStride, dst, dstStride) : RunB<2>(src, srcStride, dst, dstStride); return;
            case 3: sparse ? RunS<3>(src, srcStride, dst, dstStride) : RunB<3>(src, srcStride, dst, dstStride); return;
            case 4: sparse ? RunS<4>(src, srcStride, dst, dstStride) : RunB<4>(src, srcStride, dst, dstStride); return;
            default:
                assert(0);
            }
        }

        //-------------------------------------------------------------------------------------------------

        ResizerFloatBilinear::ResizerFloatBilinear(const ResParam & param)
            : Sse41::ResizerFloatBilinear(param)
        {
        }

        const __m256i RFB_2_WU = SIMD_MM256_SETR_EPI32(0, 0, 1, 1, 2, 2, 3, 3);
        const __m256i RFB_4_WU = SIMD_MM256_SETR_EPI32(0, 0, 0, 0, 1, 1, 1, 1);

        void ResizerFloatBilinear::Run(const float * src, size_t srcStride, float * dst, size_t dstStride)
        {
            size_t cn = _param.channels,
                cnH = AlignLo(cn, HF), cnTH = cn - cnH, cnLH = cnTH - HF,
                cnF = AlignLo(cn, F), cnTF = cn - cnF, cnLF = cnTF - F;
            size_t dw = _param.dstW, dw2 = AlignLo(dw, 2), dw4 = AlignLo(dw, 4), dw8 = AlignLo(dw, 8), dw1 = dw - 1;
            __m256 _1 = _mm256_set1_ps(1.0f);
            if (_rowBuf)
            {
                size_t rs = _param.dstW * cn, rsH = AlignLo(rs, HF), rsF = AlignLo(rs, F);
                size_t rs3 = rs - 3, rs6 = AlignLoAny(rs3, 6), rscn = rs - cn, cnHF = cn - HF;
                float* pbx[2] = { _bx[0].data, _bx[1].data };
                int32_t prev = -2;
                for (size_t dy = 0; dy < _param.dstH; dy++, dst += dstStride)
                {
                    float fy1 = _ay[dy];
                    float fy0 = 1.0f - fy1;
                    int32_t sy = _iy[dy];
                    int32_t k = 0;

                    if (sy == prev)
                        k = 2;
                    else if (sy == prev + 1)
                    {
                        Swap(pbx[0], pbx[1]);
                        k = 1;
                    }

                    prev = sy;

                    for (; k < 2; k++)
                    {
                        float* pb = pbx[k];
                        const float* ps = src + (sy + k) * srcStride;
                        size_t dx = 0;
                        if (cn == 1)
                        {
                            if (Avx2::SlowGather)
                            {
                                for (; dx < rsF; dx += F)
                                {
                                    __m256 s0145 = Load(ps + _ix[dx + 0], ps + _ix[dx + 1], ps + _ix[dx + 4], ps + _ix[dx + 5]);
                                    __m256 s2367 = Load(ps + _ix[dx + 2], ps + _ix[dx + 3], ps + _ix[dx + 6], ps + _ix[dx + 7]);
                                    __m256 fx1 = _mm256_loadu_ps(_ax.data + dx);
                                    __m256 fx0 = _mm256_sub_ps(_1, fx1);
                                    __m256 m0 = _mm256_mul_ps(fx0, _mm256_shuffle_ps(s0145, s2367, 0x88));
                                    __m256 m1 = _mm256_mul_ps(fx1, _mm256_shuffle_ps(s0145, s2367, 0xDD));
                                    _mm256_storeu_ps(pb + dx, _mm256_add_ps(m0, m1));
                                }
                            }
                            else
                            {
                                for (; dx < rsF; dx += F)
                                {
                                    __m256i idx = Avx2::LoadPermuted<true>((__m256i*)(_ix.data + dx));
                                    __m256 s0145 = _mm256_castpd_ps(_mm256_i32gather_pd((double*)ps, _mm256_extracti128_si256(idx, 0), 4));
                                    __m256 s2367 = _mm256_castpd_ps(_mm256_i32gather_pd((double*)ps, _mm256_extracti128_si256(idx, 1), 4));
                                    __m256 fx1 = _mm256_loadu_ps(_ax.data + dx);
                                    __m256 fx0 = _mm256_sub_ps(_1, fx1);
                                    __m256 s0 = _mm256_shuffle_ps(s0145, s2367, 0x88);
                                    __m256 s1 = _mm256_shuffle_ps(s0145, s2367, 0xDD);
                                    _mm256_storeu_ps(pb + dx, _mm256_fmadd_ps(s0, fx0, _mm256_mul_ps(s1, fx1)));
                                }
                            }
                            for (; dx < rsH; dx += HF)
                            {
                                __m128 s01 = Sse41::Load(ps + _ix[dx + 0], ps + _ix[dx + 1]);
                                __m128 s23 = Sse41::Load(ps + _ix[dx + 2], ps + _ix[dx + 3]);
                                __m128 fx1 = _mm_loadu_ps(_ax.data + dx);
                                __m128 fx0 = _mm_sub_ps(_mm256_castps256_ps128(_1), fx1);
                                __m128 m0 = _mm_mul_ps(fx0, _mm_shuffle_ps(s01, s23, 0x88));
                                __m128 m1 = _mm_mul_ps(fx1, _mm_shuffle_ps(s01, s23, 0xDD));
                                _mm_storeu_ps(pb + dx, _mm_add_ps(m0, m1));
                            }
                        }
                        else if(cn == 2)
                        {
                            for (; dx < rsF; dx += F)
                            {
                                __m256 s02 = Load<false>(ps + _ix[dx + 0], ps + _ix[dx + 4]);
                                __m256 s13 = Load<false>(ps + _ix[dx + 2], ps + _ix[dx + 6]);
                                __m256 fx1 = _mm256_loadu_ps(_ax.data + dx);
                                __m256 fx0 = _mm256_sub_ps(_1, fx1);
                                __m256 m0 = _mm256_mul_ps(fx0, _mm256_shuffle_ps(s02, s13, 0x44));
                                __m256 m1 = _mm256_mul_ps(fx1, _mm256_shuffle_ps(s02, s13, 0xEE));
                                _mm256_storeu_ps(pb + dx, _mm256_add_ps(m0, m1));
                            }
                            for (; dx < rsH; dx += HF)
                            {
                                __m128 s0 = _mm_loadu_ps(ps + _ix[dx + 0]);
                                __m128 s1 = _mm_loadu_ps(ps + _ix[dx + 2]);
                                __m128 fx1 = _mm_loadu_ps(_ax.data + dx);
                                __m128 fx0 = _mm_sub_ps(_mm256_castps256_ps128(_1), fx1);
                                __m128 m0 = _mm_mul_ps(fx0, _mm_shuffle_ps(s0, s1, 0x44));
                                __m128 m1 = _mm_mul_ps(fx1, _mm_shuffle_ps(s0, s1, 0xEE));
                                _mm_storeu_ps(pb + dx, _mm_add_ps(m0, m1));
                            }
                        }
                        else if (cn == 3)
                        {
                            for (; dx < rs6; dx += 6)
                            {
                                __m256 s0 = Load<false>(ps + _ix[dx + 0] + 0, ps + _ix[dx + 3] + 0);
                                __m256 s1 = Load<false>(ps + _ix[dx + 0] + 3, ps + _ix[dx + 3] + 3);
                                __m256 fx1 = Load<false>(_ax.data + dx + 0, _ax.data + dx + 3);
                                __m256 fx0 = _mm256_sub_ps(_1, fx1);
                                Store<false>(pb + dx + 0, pb + dx + 3, _mm256_fmadd_ps(fx0, s0, _mm256_mul_ps(fx1, s1)));
                            }
                            for (; dx < rs3; dx += 3)
                            {
                                __m128 s0 = _mm_loadu_ps(ps + _ix[dx] + 0);
                                __m128 s1 = _mm_loadu_ps(ps + _ix[dx] + 3);
                                __m128 fx1 = _mm_set1_ps(_ax.data[dx]);
                                __m128 fx0 = _mm_sub_ps(_mm256_castps256_ps128(_1), fx1);
                                _mm_storeu_ps(pb + dx, _mm_add_ps(_mm_mul_ps(fx0, s0), _mm_mul_ps(fx1, s1)));
                            }
                        }
                        else if (cn == 4)
                        {
                            for (; dx < rsF; dx += F)
                            {
                                __m256 s0 = Load<false>(ps + _ix[dx + 0] + 0, ps + _ix[dx + 4] + 0);
                                __m256 s1 = Load<false>(ps + _ix[dx + 0] + 4, ps + _ix[dx + 4] + 4);
                                __m256 fx1 = _mm256_loadu_ps(_ax.data + dx);
                                __m256 fx0 = _mm256_sub_ps(_1, fx1);
                                _mm256_storeu_ps(pb + dx, _mm256_fmadd_ps(fx0, s0, _mm256_mul_ps(fx1, s1)));
                            }
                            for (; dx < rsH; dx += HF)
                            {
                                __m128 s0 = _mm_loadu_ps(ps + _ix[dx] + 0);
                                __m128 s1 = _mm_loadu_ps(ps + _ix[dx] + 4);
                                __m128 fx1 = _mm_set1_ps(_ax.data[dx]);
                                __m128 fx0 = _mm_sub_ps(_mm256_castps256_ps128(_1), fx1);
                                _mm_storeu_ps(pb + dx, _mm_add_ps(_mm_mul_ps(fx0, s0), _mm_mul_ps(fx1, s1)));
                            }
                        }
                        else if (cn < 8)
                        {
                            for (; dx < rscn; dx += cn)
                            {
                                const float* ps0 = ps + _ix[dx];
                                __m256 s0 = _mm256_loadu_ps(ps0);
                                __m256 s1 = _mm256_loadu_ps(ps0 + cn);
                                __m256 fx1 = _mm256_loadu_ps(_ax.data + dx);
                                __m256 fx0 = _mm256_sub_ps(_1, fx1);
                                _mm256_storeu_ps(pb + dx, _mm256_fmadd_ps(fx0, s0, _mm256_mul_ps(fx1, s1)));
                            }
                            for (; dx < rs; dx += cn)
                            {
                                const float* ps0 = ps + _ix[dx], * ps1 = ps0 + cn;
                                __m256 s0 = Load<false>(ps0, ps0 + cnHF);
                                __m256 s1 = Load<false>(ps1, ps1 + cnHF);
                                __m256 fx1 = Load<false>(_ax.data + dx, _ax.data + dx + cnHF);
                                __m256 fx0 = _mm256_sub_ps(_1, fx1);
                                Store<false>(pb + dx, pb + dx + cnHF, _mm256_fmadd_ps(fx0, s0, _mm256_mul_ps(fx1, s1)));
                            }
                        }
                        else
                        {
                            for (; dx < rs;)
                            {
                                const float* ps0 = ps + _ix[dx];
                                __m256 fx1 = _mm256_set1_ps(_ax[dx]);
                                __m256 fx0 = _mm256_sub_ps(_1, fx1);
                                for (size_t eF = dx + cnF; dx < eF; dx += F, ps0 += F)
                                    _mm256_storeu_ps(pb + dx, _mm256_fmadd_ps(fx0, _mm256_loadu_ps(ps0), _mm256_mul_ps(fx1, _mm256_loadu_ps(ps0 + cn))));
                                if (cnTF)
                                {
                                    _mm256_storeu_ps(pb + dx + cnLF, _mm256_fmadd_ps(fx0, _mm256_loadu_ps(ps0 + cnLF), _mm256_mul_ps(fx1, _mm256_loadu_ps(ps0 + cn + cnLF))));
                                    dx += cnTF;
                                }
                            }
                        }
                        for (; dx < rs; dx++)
                        {
                            int32_t sx = _ix[dx];
                            float fx = _ax[dx];
                            pb[dx] = ps[sx] * (1.0f - fx) + ps[sx + cn] * fx;
                        }
                    }

                    size_t dx = 0;
                    __m256 _fy0 = _mm256_set1_ps(fy0);
                    __m256 _fy1 = _mm256_set1_ps(fy1);
                    for (; dx < rsF; dx += F)
                    {
                        __m256 b0 = _mm256_loadu_ps(pbx[0] + dx);
                        __m256 b1 = _mm256_loadu_ps(pbx[1] + dx);
                        _mm256_storeu_ps(dst + dx, _mm256_fmadd_ps(b0, _fy0, _mm256_mul_ps(b1, _fy1)));
                    }
                    for (; dx < rsH; dx += HF)
                    {
                        __m128 m0 = _mm_mul_ps(_mm_loadu_ps(pbx[0] + dx), _mm256_castps256_ps128(_fy0));
                        __m128 m1 = _mm_mul_ps(_mm_loadu_ps(pbx[1] + dx), _mm256_castps256_ps128(_fy1));
                        _mm_storeu_ps(dst + dx, _mm_add_ps(m0, m1));
                    }
                    for (; dx < rs; dx++)
                        dst[dx] = pbx[0][dx] * fy0 + pbx[1][dx] * fy1;
                }
            }
            else
            {
                for (size_t dy = 0; dy < _param.dstH; dy++, dst += dstStride)
                {
                    __m256 fy1 = _mm256_set1_ps(_ay[dy]);
                    __m256 fy0 = _mm256_sub_ps(_1, fy1);
                    const float* src0 = src + _iy[dy] * srcStride, * src1 = src0 + srcStride;
                    if (cn == 1)
                    {
                        size_t dx = 0;
                        if (Avx2::SlowGather)
                        {
                            for (; dx < dw8; dx += 8)
                            {
                                __m256 fx1 = _mm256_loadu_ps(_ax.data + dx);
                                __m256 fx0 = _mm256_sub_ps(_1, fx1);
                                __m256 s00 = Load(src0 + _ix[dx + 0], src0 + _ix[dx + 1], src0 + _ix[dx + 4], src0 + _ix[dx + 5]);
                                __m256 s01 = Load(src0 + _ix[dx + 2], src0 + _ix[dx + 3], src0 + _ix[dx + 6], src0 + _ix[dx + 7]);
                                __m256 r0 = _mm256_fmadd_ps(_mm256_shuffle_ps(s00, s01, 0x88), fx0, _mm256_mul_ps(_mm256_shuffle_ps(s00, s01, 0xDD), fx1));
                                __m256 s10 = Load(src1 + _ix[dx + 0], src1 + _ix[dx + 1], src1 + _ix[dx + 4], src1 + _ix[dx + 5]);
                                __m256 s11 = Load(src1 + _ix[dx + 2], src1 + _ix[dx + 3], src1 + _ix[dx + 6], src1 + _ix[dx + 7]);
                                __m256 r1 = _mm256_fmadd_ps(_mm256_shuffle_ps(s10, s11, 0x88), fx0, _mm256_mul_ps(_mm256_shuffle_ps(s10, s11, 0xDD), fx1));
                                _mm256_storeu_ps(dst + dx, _mm256_fmadd_ps(r0, fy0, _mm256_mul_ps(r1, fy1)));
                            }
                        }
                        else
                        {
                            for (; dx < dw8; dx += 8)
                            {
                                __m256 fx1 = _mm256_loadu_ps(_ax.data + dx);
                                __m256 fx0 = _mm256_sub_ps(_1, fx1);
                                __m256i idx = Avx2::LoadPermuted<true>((__m256i*)(_ix.data + dx));
                                __m256 s00 = _mm256_castpd_ps(_mm256_i32gather_pd((double*)src0, _mm256_extracti128_si256(idx, 0), 4));
                                __m256 s01 = _mm256_castpd_ps(_mm256_i32gather_pd((double*)src0, _mm256_extracti128_si256(idx, 1), 4));
                                __m256 r0 = _mm256_fmadd_ps(_mm256_shuffle_ps(s00, s01, 0x88), fx0, _mm256_mul_ps(_mm256_shuffle_ps(s00, s01, 0xDD), fx1));
                                __m256 s10 = _mm256_castpd_ps(_mm256_i32gather_pd((double*)src1, _mm256_extracti128_si256(idx, 0), 4));
                                __m256 s11 = _mm256_castpd_ps(_mm256_i32gather_pd((double*)src1, _mm256_extracti128_si256(idx, 1), 4));
                                __m256 r1 = _mm256_fmadd_ps(_mm256_shuffle_ps(s10, s11, 0x88), fx0, _mm256_mul_ps(_mm256_shuffle_ps(s10, s11, 0xDD), fx1));
                                _mm256_storeu_ps(dst + dx, _mm256_fmadd_ps(r0, fy0, _mm256_mul_ps(r1, fy1)));
                            }
                        }
                        for (; dx < dw4; dx += 4)
                        {
                            __m128 fx1 = _mm_loadu_ps(_ax.data + dx);
                            __m128 fx0 = _mm_sub_ps(_mm256_castps256_ps128(_1), fx1);
                            __m128 s00 = Sse41::Load(src0 + _ix[dx + 0], src0 + _ix[dx + 1]);
                            __m128 s01 = Sse41::Load(src0 + _ix[dx + 2], src0 + _ix[dx + 3]);
                            __m128 r0 = _mm_add_ps(_mm_mul_ps(_mm_shuffle_ps(s00, s01, 0x88), fx0), _mm_mul_ps(_mm_shuffle_ps(s00, s01, 0xDD), fx1));
                            __m128 s10 = Sse41::Load(src1 + _ix[dx + 0], src1 + _ix[dx + 1]);
                            __m128 s11 = Sse41::Load(src1 + _ix[dx + 2], src1 + _ix[dx + 3]);
                            __m128 r1 = _mm_add_ps(_mm_mul_ps(_mm_shuffle_ps(s10, s11, 0x88), fx0), _mm_mul_ps(_mm_shuffle_ps(s10, s11, 0xDD), fx1));
                            _mm_storeu_ps(dst + dx, _mm_add_ps(_mm_mul_ps(r0, _mm256_castps256_ps128(fy0)), _mm_mul_ps(r1, _mm256_castps256_ps128(fy1))));
                        }
                        for (; dx < dw; dx++)
                        {
                            size_t os = _ix[dx];
                            __m128 fx1 = _mm_set1_ps(_ax[dx]);
                            __m128 fx0 = _mm_sub_ps(_mm256_castps256_ps128(_1), fx1);
                            __m128 r0 = _mm_add_ps(_mm_mul_ps(_mm_load_ss(src0 + os), fx0), _mm_mul_ps(_mm_load_ss(src0 + os + 1), fx1));
                            __m128 r1 = _mm_add_ps(_mm_mul_ps(_mm_load_ss(src1 + os), fx0), _mm_mul_ps(_mm_load_ss(src1 + os + 1), fx1));
                            _mm_store_ss(dst + dx, _mm_add_ps(_mm_mul_ps(r0, _mm256_castps256_ps128(fy0)), _mm_mul_ps(r1, _mm256_castps256_ps128(fy1))));
                        }
                    }
                    else if (cn == 2)
                    {
                        size_t dx = 0, od = 0;
                        for (; dx < dw8; dx += 4, od += 8)
                        {
                            __m256 fx1 = _mm256_permutevar8x32_ps(_mm256_castps128_ps256(_mm_loadu_ps(_ax.data + dx)), RFB_2_WU);
                            __m256 fx0 = _mm256_sub_ps(_1, fx1);
                            __m256 s00 = Load<false>(src0 + _ix[dx + 0], src0 + _ix[dx + 2]);
                            __m256 s01 = Load<false>(src0 + _ix[dx + 1], src0 + _ix[dx + 3]);
                            __m256 r0 = _mm256_fmadd_ps(_mm256_shuffle_ps(s00, s01, 0x44), fx0, _mm256_mul_ps(_mm256_shuffle_ps(s00, s01, 0xEE), fx1));
                            __m256 s10 = Load<false>(src1 + _ix[dx + 0], src1 + _ix[dx + 2]);
                            __m256 s11 = Load<false>(src1 + _ix[dx + 1], src1 + _ix[dx + 3]);
                            __m256 r1 = _mm256_fmadd_ps(_mm256_shuffle_ps(s10, s11, 0x44), fx0, _mm256_mul_ps(_mm256_shuffle_ps(s10, s11, 0xEE), fx1));
                            _mm256_storeu_ps(dst + od, _mm256_fmadd_ps(r0, fy0, _mm256_mul_ps(r1, fy1)));
                        }
                        for (; dx < dw4; dx += 2, od += 4)
                        {
                            __m128 fx = Sse41::LoadHalf(_ax.data + dx);
                            __m128 fx1 = _mm_unpacklo_ps(fx, fx);
                            __m128 fx0 = _mm_sub_ps(_mm256_castps256_ps128(_1), fx1);
                            __m128 s00 = _mm_loadu_ps(src0 + _ix[dx + 0]);
                            __m128 s01 = _mm_loadu_ps(src0 + _ix[dx + 1]);
                            __m128 r0 = _mm_add_ps(_mm_mul_ps(_mm_shuffle_ps(s00, s01, 0x44), fx0), _mm_mul_ps(_mm_shuffle_ps(s00, s01, 0xEE), fx1));
                            __m128 s10 = _mm_loadu_ps(src1 + _ix[dx + 0]);
                            __m128 s11 = _mm_loadu_ps(src1 + _ix[dx + 1]);
                            __m128 r1 = _mm_add_ps(_mm_mul_ps(_mm_shuffle_ps(s10, s11, 0x44), fx0), _mm_mul_ps(_mm_shuffle_ps(s10, s11, 0xEE), fx1));
                            _mm_storeu_ps(dst + od, _mm_add_ps(_mm_mul_ps(r0, _mm256_castps256_ps128(fy0)), _mm_mul_ps(r1, _mm256_castps256_ps128(fy1))));
                        }
                        for (; dx < dw; dx++, od += 2)
                        {
                            size_t os = _ix[dx];
                            __m128 fx1 = _mm_set1_ps(_ax[dx]);
                            __m128 fx0 = _mm_sub_ps(_mm256_castps256_ps128(_1), fx1);
                            __m128 s0 = _mm_loadu_ps(src0 + os);
                            __m128 r0 = _mm_add_ps(_mm_mul_ps(s0, fx0), _mm_mul_ps(_mm_shuffle_ps(s0, s0, 0xEE), fx1));
                            __m128 s1 = _mm_loadu_ps(src1 + os);
                            __m128 r1 = _mm_add_ps(_mm_mul_ps(s1, fx0), _mm_mul_ps(_mm_shuffle_ps(s1, s1, 0xEE), fx1));
                            Sse41::StoreHalf<0>(dst + od, _mm_add_ps(_mm_mul_ps(r0, _mm256_castps256_ps128(fy0)), _mm_mul_ps(r1, _mm256_castps256_ps128(fy1))));
                        }
                    }
                    else if (cn == 3)
                    {
                        size_t dx = 0, od = 0;
                        for (; dx < dw2; dx += 2, od += 6)
                        {
                            size_t os = _ix[dx];
                            __m256 fx1 = _mm256_permutevar8x32_ps(_mm256_castps128_ps256(_mm_loadu_ps(_ax.data + dx)), RFB_4_WU);
                            __m256 fx0 = _mm256_sub_ps(_1, fx1);
                            __m256 s00 = Load<false>(src0 + _ix[dx + 0] + 0, src0 + _ix[dx + 1] + 0);
                            __m256 s01 = Load<false>(src0 + _ix[dx + 0] + 3, src0 + _ix[dx + 1] + 3);
                            __m256 r0 = _mm256_fmadd_ps(fx0, s00, _mm256_mul_ps(fx1, s01));
                            __m256 s10 = Load<false>(src1 + _ix[dx + 0] + 0, src1 + _ix[dx + 1] + 0);
                            __m256 s11 = Load<false>(src1 + _ix[dx + 0] + 3, src1 + _ix[dx + 1] + 3);
                            __m256 r1 = _mm256_fmadd_ps(fx0, s10, _mm256_mul_ps(fx1, s11));
                            _mm256_storeu_ps(dst + od, _mm256_permutevar8x32_ps(_mm256_fmadd_ps(r0, fy0, _mm256_mul_ps(r1, fy1)), RSB_3_P1));
                        }
                        for (; dx < dw1; dx += 1, od += 3)
                        {
                            size_t os = _ix[dx];
                            __m128 fx1 = _mm_set1_ps(_ax[dx]);
                            __m128 fx0 = _mm_sub_ps(_mm256_castps256_ps128(_1), fx1);
                            __m128 r0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src0 + os), fx0), _mm_mul_ps(_mm_loadu_ps(src0 + os + 3), fx1));
                            __m128 r1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src1 + os), fx0), _mm_mul_ps(_mm_loadu_ps(src1 + os + 3), fx1));
                            _mm_storeu_ps(dst + od, _mm_add_ps(_mm_mul_ps(r0, _mm256_castps256_ps128(fy0)), _mm_mul_ps(r1, _mm256_castps256_ps128(fy1))));
                        }
                        if (dx < dw)
                        {
                            size_t os = _ix[dx];
                            __m128 fx1 = _mm_set1_ps(_ax[dx]);
                            __m128 fx0 = _mm_sub_ps(_mm256_castps256_ps128(_1), fx1);
                            for (size_t ed = od + 3; od < ed; od++, os++)
                            {
                                __m128 r0 = _mm_add_ps(_mm_mul_ps(_mm_load_ss(src0 + os), fx0), _mm_mul_ps(_mm_load_ss(src0 + os + 3), fx1));
                                __m128 r1 = _mm_add_ps(_mm_mul_ps(_mm_load_ss(src1 + os), fx0), _mm_mul_ps(_mm_load_ss(src1 + os + 3), fx1));
                                _mm_store_ss(dst + od, _mm_add_ps(_mm_mul_ps(r0, _mm256_castps256_ps128(fy0)), _mm_mul_ps(r1, _mm256_castps256_ps128(fy1))));
                            }
                        }
                    }
                    else if (cn == 4)
                    {
                        size_t dx = 0, od = 0;
                        for (; dx < dw2; dx += 2, od += 8)
                        {
                            size_t os = _ix[dx];
                            __m256 fx1 = _mm256_permutevar8x32_ps(_mm256_castps128_ps256(_mm_loadu_ps(_ax.data + dx)), RFB_4_WU);
                            __m256 fx0 = _mm256_sub_ps(_1, fx1);
                            __m256 s00 = Load<false>(src0 + _ix[dx + 0] + 0, src0 + _ix[dx + 1] + 0);
                            __m256 s01 = Load<false>(src0 + _ix[dx + 0] + 4, src0 + _ix[dx + 1] + 4);
                            __m256 r0 = _mm256_fmadd_ps(fx0, s00, _mm256_mul_ps(fx1, s01));
                            __m256 s10 = Load<false>(src1 + _ix[dx + 0] + 0, src1 + _ix[dx + 1] + 0);
                            __m256 s11 = Load<false>(src1 + _ix[dx + 0] + 4, src1 + _ix[dx + 1] + 4);
                            __m256 r1 = _mm256_fmadd_ps(fx0, s10, _mm256_mul_ps(fx1, s11));
                            _mm256_storeu_ps(dst + od, _mm256_fmadd_ps(r0, fy0, _mm256_mul_ps(r1, fy1)));
                        }
                        for (; dx < dw; dx += 1, od += 4)
                        {
                            size_t os = _ix[dx];
                            __m128 fx1 = _mm_set1_ps(_ax[dx]);
                            __m128 fx0 = _mm_sub_ps(_mm256_castps256_ps128(_1), fx1);
                            __m128 r0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src0 + os), fx0), _mm_mul_ps(_mm_loadu_ps(src0 + os + 4), fx1));
                            __m128 r1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src1 + os), fx0), _mm_mul_ps(_mm_loadu_ps(src1 + os + 4), fx1));
                            _mm_storeu_ps(dst + od, _mm_add_ps(_mm_mul_ps(r0, _mm256_castps256_ps128(fy0)), _mm_mul_ps(r1, _mm256_castps256_ps128(fy1))));
                        }
                    }
                    else if (cn < 8)
                    {
                        size_t dx = 0;
                        for (; dx < dw1; dx++)
                        {
                            size_t os = _ix[dx], od = dx * cn;
                            __m256 fx1 = _mm256_set1_ps(_ax[dx]);
                            __m256 fx0 = _mm256_sub_ps(_1, fx1);
                            __m256 r0 = _mm256_fmadd_ps(fx0, _mm256_loadu_ps(src0 + os), _mm256_mul_ps(fx1, _mm256_loadu_ps(src0 + os + cn)));
                            __m256 r1 = _mm256_fmadd_ps(fx0, _mm256_loadu_ps(src1 + os), _mm256_mul_ps(fx1, _mm256_loadu_ps(src1 + os + cn)));
                            _mm256_storeu_ps(dst + od, _mm256_fmadd_ps(r0, fy0, _mm256_mul_ps(r1, fy1)));
                        }
                        for (; dx < dw; dx++)
                        {
                            size_t os = _ix[dx], eH = os + cnH, od = dx * cn;
                            __m128 fx1 = _mm_set1_ps(_ax[dx]);
                            __m128 fx0 = _mm_sub_ps(_mm256_castps256_ps128(_1), fx1);
                            for (; os < eH; os += HF, od += HF)
                            {
                                __m128 r0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src0 + os), fx0), _mm_mul_ps(_mm_loadu_ps(src0 + os + cn), fx1));
                                __m128 r1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src1 + os), fx0), _mm_mul_ps(_mm_loadu_ps(src1 + os + cn), fx1));
                                _mm_storeu_ps(dst + od, _mm_add_ps(_mm_mul_ps(r0, _mm256_castps256_ps128(fy0)), _mm_mul_ps(r1, _mm256_castps256_ps128(fy1))));
                            }
                            if (cnTH)
                            {
                                os += cnLH;
                                od += cnLH;
                                __m128 r0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src0 + os), fx0), _mm_mul_ps(_mm_loadu_ps(src0 + os + cn), fx1));
                                __m128 r1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src1 + os), fx0), _mm_mul_ps(_mm_loadu_ps(src1 + os + cn), fx1));
                                _mm_storeu_ps(dst + od, _mm_add_ps(_mm_mul_ps(r0, _mm256_castps256_ps128(fy0)), _mm_mul_ps(r1, _mm256_castps256_ps128(fy1))));
                            }
                        }
                    }
                    else
                    {
                        for (size_t dx = 0; dx < dw; dx++)
                        {
                            size_t os = _ix[dx], eF = os + cnF, od = dx * cn;
                            __m256 fx1 = _mm256_set1_ps(_ax[dx]);
                            __m256 fx0 = _mm256_sub_ps(_1, fx1);
                            for (; os < eF; os += F, od += F)
                            {
                                __m256 r0 = _mm256_fmadd_ps(_mm256_loadu_ps(src0 + os), fx0, _mm256_mul_ps(_mm256_loadu_ps(src0 + os + cn), fx1));
                                __m256 r1 = _mm256_fmadd_ps(_mm256_loadu_ps(src1 + os), fx0, _mm256_mul_ps(_mm256_loadu_ps(src1 + os + cn), fx1));
                                _mm256_storeu_ps(dst + od, _mm256_fmadd_ps(r0, fy0, _mm256_mul_ps(r1, fy1)));
                            }
                            if (cnTF)
                            {
                                os += cnLF;
                                od += cnLF;
                                __m256 r0 = _mm256_fmadd_ps(_mm256_loadu_ps(src0 + os), fx0, _mm256_mul_ps(_mm256_loadu_ps(src0 + os + cn), fx1));
                                __m256 r1 = _mm256_fmadd_ps(_mm256_loadu_ps(src1 + os), fx0, _mm256_mul_ps(_mm256_loadu_ps(src1 + os + cn), fx1));
                                _mm256_storeu_ps(dst + od, _mm256_fmadd_ps(r0, fy0, _mm256_mul_ps(r1, fy1)));
                            }
                        }
                    }
                }
            }
        }

        //-------------------------------------------------------------------------------------------------

        ResizerBf16Bilinear::ResizerBf16Bilinear(const ResParam& param)
            : Sse41::ResizerBf16Bilinear(param)
        {
        }

        __m256i K8_IDX_20 = SIMD_MM256_SETR_EPI8(
            -1, -1, 0x0, 0x1, -1, -1, 0x2, 0x3, -1, -1, 0x8, 0x9, -1, -1, 0xA, 0xB,
            -1, -1, 0x0, 0x1, -1, -1, 0x2, 0x3, -1, -1, 0x8, 0x9, -1, -1, 0xA, 0xB);
        __m256i K8_IDX_21 = SIMD_MM256_SETR_EPI8(
            -1, -1, 0x4, 0x5, -1, -1, 0x6, 0x7, -1, -1, 0xC, 0xD, -1, -1, 0xE, 0xF,
            -1, -1, 0x4, 0x5, -1, -1, 0x6, 0x7, -1, -1, 0xC, 0xD, -1, -1, 0xE, 0xF);

        const __m256i K8_IDX_30 = SIMD_MM256_SETR_EPI8(
            -1, -1, 0x0, 0x1, -1, -1, 0x2, 0x3, -1, -1, 0x4, 0x5, -1, -1, -1, -1, 
            -1, -1, 0x0, 0x1, -1, -1, 0x2, 0x3, -1, -1, 0x4, 0x5, -1, -1, -1, -1);
        const __m256i K8_IDX_31 = SIMD_MM256_SETR_EPI8(
            -1, -1, 0x6, 0x7, -1, -1, 0x8, 0x9, -1, -1, 0xA, 0xB, -1, -1, -1, -1, 
            -1, -1, 0x6, 0x7, -1, -1, 0x8, 0x9, -1, -1, 0xA, 0xB, -1, -1, -1, -1);

        SIMD_INLINE __m128 BilinearRowSumBf16(const uint16_t* src, size_t channels, __m128 fx0, __m128 fx1)
        {
            __m128 s0 = Sse41::BFloat16ToFloat32(Sse41::UnpackU16<0>(_mm_loadl_epi64((__m128i*)src)));
            __m128 s1 = Sse41::BFloat16ToFloat32(Sse41::UnpackU16<0>(_mm_loadl_epi64((__m128i*)(src + channels))));
            return _mm_fmadd_ps(fx0, s0, _mm_mul_ps(fx1, s1));
        }

        SIMD_INLINE __m256 BilinearRowSumBf16(const uint16_t* src, size_t channels, __m256 fx0, __m256 fx1)
        {
            __m256 s0 = BFloat16ToFloat32(_mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i*)src)));
            __m256 s1 = BFloat16ToFloat32(_mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i*)(src + channels))));
            return _mm256_fmadd_ps(fx0, s0, _mm256_mul_ps(fx1, s1));
        }

        void ResizerBf16Bilinear::Run(const uint16_t* src, size_t srcStride, uint16_t* dst, size_t dstStride)
        {
            size_t cn = _param.channels, 
                cnH = AlignLo(cn, Sse41::F), cnTH = cn - cnH, cnLH = cnTH - Sse41::F,
                cnD = AlignLo(cn, DF), cnF = AlignLo(cn, F), cnTF = cn - cnF, cnLF = cnTF - F;
            __m256 _1 = _mm256_set1_ps(1.0f);
            if (_rowBuf)
            {
                size_t rs = _param.dstW * cn, rsH = AlignLo(rs, Sse41::F), rsF = AlignLo(rs, F), rsD = AlignLo(rs, DF);
                size_t rs3 = AlignLoAny(rs - 1, 3), rs6 = AlignLoAny(rs - 1, 6);
                float* pbx[2] = { _bx[0].data, _bx[1].data };
                int32_t prev = -2;
                for (size_t dy = 0; dy < _param.dstH; dy++, dst += dstStride)
                {
                    float fy1 = _ay[dy];
                    float fy0 = 1.0f - fy1;
                    int32_t sy = _iy[dy];
                    int32_t k = 0;

                    if (sy == prev)
                        k = 2;
                    else if (sy == prev + 1)
                    {
                        Swap(pbx[0], pbx[1]);
                        k = 1;
                    }

                    prev = sy;

                    for (; k < 2; k++)
                    {
                        float* pb = pbx[k];
                        const uint16_t* ps = src + (sy + k) * srcStride;
                        size_t dx = 0;
                        if (cn == 1)
                        {
                            if (Avx2::SlowGather)
                            {
                                for (; dx < rsF; dx += F)
                                {
                                    SIMD_ALIGNED(32) uint32_t buf[8];
                                    buf[0] = *(uint32_t*)(ps + _ix[dx + 0]);
                                    buf[1] = *(uint32_t*)(ps + _ix[dx + 1]);
                                    buf[2] = *(uint32_t*)(ps + _ix[dx + 2]);
                                    buf[3] = *(uint32_t*)(ps + _ix[dx + 3]);
                                    buf[4] = *(uint32_t*)(ps + _ix[dx + 4]);
                                    buf[5] = *(uint32_t*)(ps + _ix[dx + 5]);
                                    buf[6] = *(uint32_t*)(ps + _ix[dx + 6]);
                                    buf[7] = *(uint32_t*)(ps + _ix[dx + 7]);
                                    __m256i _src = _mm256_loadu_si256((__m256i*)buf);
                                    __m256 s0 = BFloat16ToFloat32Even(_src);
                                    __m256 s1 = BFloat16ToFloat32Odd(_src);
                                    __m256 fx1 = _mm256_loadu_ps(_ax.data + dx);
                                    __m256 fx0 = _mm256_sub_ps(_1, fx1);
                                    _mm256_storeu_ps(pb + dx, _mm256_fmadd_ps(fx0, s0, _mm256_mul_ps(fx1, s1)));
                                }
                            }
                            else
                            {
                                for (; dx < rsF; dx += F)
                                {
                                    __m256i _src = _mm256_i32gather_epi32((int*)ps, _mm256_loadu_si256((__m256i*)(_ix.data + dx)), 2);
                                    __m256 s0 = BFloat16ToFloat32Even(_src);
                                    __m256 s1 = BFloat16ToFloat32Odd(_src);
                                    __m256 fx1 = _mm256_loadu_ps(_ax.data + dx);
                                    __m256 fx0 = _mm256_sub_ps(_1, fx1);
                                    _mm256_storeu_ps(pb + dx, _mm256_fmadd_ps(fx0, s0, _mm256_mul_ps(fx1, s1)));
                                }
                            }
                            for (; dx < rsH; dx += Sse41::F)
                            {
                                SIMD_ALIGNED(16) uint32_t buf[4];
                                buf[0] = *(uint32_t*)(ps + _ix[dx + 0]);
                                buf[1] = *(uint32_t*)(ps + _ix[dx + 1]);
                                buf[2] = *(uint32_t*)(ps + _ix[dx + 2]);
                                buf[3] = *(uint32_t*)(ps + _ix[dx + 3]);
                                __m128i _src = _mm_loadu_si128((__m128i*)buf);
                                __m128 s0 = Sse41::BFloat16ToFloat32Even(_src);
                                __m128 s1 = Sse41::BFloat16ToFloat32Odd(_src);
                                __m128 fx1 = _mm_loadu_ps(_ax.data + dx);
                                __m128 fx0 = _mm_sub_ps(_mm256_castps256_ps128(_1), fx1);
                                _mm_storeu_ps(pb + dx, _mm_fmadd_ps(fx0, s0, _mm_mul_ps(fx1, s1)));
                            }
                        }
                        else if (cn == 2)
                        {
                            for (; dx < rsF; dx += F)
                            {
                                __m256i _src = Load((__m128i*)(ps + _ix[dx + 0]), (__m128i*)(ps + _ix[dx + 2]), (__m128i*)(ps + _ix[dx + 4]), (__m128i*)(ps + _ix[dx + 6]));
                                __m256 s0 = _mm256_castsi256_ps(_mm256_shuffle_epi8(_src, K8_IDX_20));
                                __m256 s1 = _mm256_castsi256_ps(_mm256_shuffle_epi8(_src, K8_IDX_21));
                                __m256 fx1 = _mm256_loadu_ps(_ax.data + dx);
                                __m256 fx0 = _mm256_sub_ps(_1, fx1);
                                _mm256_storeu_ps(pb + dx, _mm256_fmadd_ps(fx0, s0, _mm256_mul_ps(fx1, s1)));
                            }
                            for (; dx < rsH; dx += Sse41::F)
                            {
                                __m128i _src = Sse41::Load((__m128i*)(ps + _ix[dx + 0]), (__m128i*)(ps + _ix[dx + 2]));
                                __m128 s0 = _mm_castsi128_ps(_mm_shuffle_epi8(_src, _mm256_castsi256_si128(K8_IDX_20)));
                                __m128 s1 = _mm_castsi128_ps(_mm_shuffle_epi8(_src, _mm256_castsi256_si128(K8_IDX_21)));
                                __m128 fx1 = _mm_loadu_ps(_ax.data + dx);
                                __m128 fx0 = _mm_sub_ps(_mm256_castps256_ps128(_1), fx1);
                                _mm_storeu_ps(pb + dx, _mm_fmadd_ps(fx0, s0, _mm_mul_ps(fx1, s1)));
                            }
                        }
                        else if (cn == 3 && rs >= 3)
                        {
                            for (; dx < rs6; dx += 6)
                            {
                                __m256 fx1 = Load<false>(_ax.data + dx, _ax.data + dx + 3);
                                __m256 fx0 = _mm256_sub_ps(_1, fx1);
                                __m256i _src = Load<false>((__m128i*)(ps + _ix[dx + 0]), (__m128i*)(ps + _ix[dx + 3]));
                                __m256 s0 = _mm256_castsi256_ps(_mm256_shuffle_epi8(_src, K8_IDX_30));
                                __m256 s1 = _mm256_castsi256_ps(_mm256_shuffle_epi8(_src, K8_IDX_31));
                                _mm256_storeu_ps(pb + dx, _mm256_permutevar8x32_ps(_mm256_fmadd_ps(fx0, s0, _mm256_mul_ps(fx1, s1)), RSB_3_P1));
                            }
                            for (; dx < rs3; dx += 3)
                            {
                                __m128 fx1 = _mm_set1_ps(_ax.data[dx]);
                                __m128 fx0 = _mm_sub_ps(_mm256_castps256_ps128(_1), fx1);
                                _mm_storeu_ps(pb + dx, BilinearRowSumBf16(ps + _ix[dx], cn, fx0, fx1));
                            }
                        }
                        else if (cn == 4)
                        {
                            for (; dx < rsF; dx += F)
                            {
                                __m256 fx1 = Load<false>(_ax.data + dx, _ax.data + dx + 4);
                                __m256 fx0 = _mm256_sub_ps(_1, fx1);
                                __m256i _src = Load<false>((__m128i*)(ps + _ix[dx + 0]), (__m128i*)(ps + _ix[dx + 4]));
                                _mm256_storeu_ps(pb + dx, _mm256_fmadd_ps(fx0, BFloat16ToFloat32<0>(_src), _mm256_mul_ps(fx1, BFloat16ToFloat32<1>(_src))));
                            }
                            for (; dx < rs; dx += 4)
                            {
                                __m128 fx1 = _mm_set1_ps(_ax[dx]);
                                __m128 fx0 = _mm_sub_ps(_mm256_castps256_ps128(_1), fx1);
                                __m128i _src = _mm_loadu_si128((__m128i*)(ps + _ix[dx]));
                                _mm_storeu_ps(pb + dx, _mm_add_ps(_mm_mul_ps(fx0, Sse41::BFloat16ToFloat32<0>(_src)), _mm_mul_ps(fx1, Sse41::BFloat16ToFloat32<1>(_src))));
                            }
                        }
                        else if (cn < 8)
                        {
                            for (; dx < rs;)
                            {
                                const uint16_t* ps0 = ps + _ix[dx];
                                __m128 fx1 = _mm_set1_ps(_ax[dx]);
                                __m128 fx0 = _mm_sub_ps(_mm256_castps256_ps128(_1), fx1);
                                for (size_t end = dx + cnH; dx < end; dx += Sse41::F, ps0 += Sse41::F)
                                    _mm_storeu_ps(pb + dx, BilinearRowSumBf16(ps0, cn, fx0, fx1));
                                if (cnTH)
                                    _mm_storeu_ps(pb + dx + cnLH, BilinearRowSumBf16(ps0 + cnLH, cn, fx0, fx1)), dx += cnTH;
                            }
                        }                          
                        else
                        {
                            for (; dx < rs;)
                            {
                                const uint16_t* ps0 = ps + _ix[dx];
                                __m256 fx1 = _mm256_set1_ps(_ax[dx]);
                                __m256 fx0 = _mm256_sub_ps(_1, fx1);
                                for (size_t end = dx + cnF; dx < end; dx +=F, ps0 += F)
                                    _mm256_storeu_ps(pb + dx, BilinearRowSumBf16(ps0, cn, fx0, fx1));
                                if (cnTF)
                                    _mm256_storeu_ps(pb + dx + cnLF, BilinearRowSumBf16(ps0 + cnLF, cn, fx0, fx1)), dx += cnTF;
                            }
                        }
                      
                        for (; dx < rs; dx++)
                        {
                            int32_t sx = _ix[dx];
                            float fx = _ax[dx];
                            pb[dx] = Base::BFloat16ToFloat32(ps[sx]) * (1.0f - fx) + Base::BFloat16ToFloat32(ps[sx + cn]) * fx;
                        }
                    }

                    size_t dx = 0;
                    __m256 _fy0 = _mm256_set1_ps(fy0);
                    __m256 _fy1 = _mm256_set1_ps(fy1);
                    for (; dx < rsD; dx += DF)
                    {
                        __m256i d0 = Float32ToBFloat16(_mm256_fmadd_ps(_mm256_loadu_ps(pbx[0] + dx + 0), _fy0, _mm256_mul_ps(_mm256_loadu_ps(pbx[1] + dx + 0), _fy1)));
                        __m256i d1 = Float32ToBFloat16(_mm256_fmadd_ps(_mm256_loadu_ps(pbx[0] + dx + F), _fy0, _mm256_mul_ps(_mm256_loadu_ps(pbx[1] + dx + F), _fy1)));
                        _mm256_storeu_si256((__m256i*)(dst + dx), _mm256_permute4x64_epi64(_mm256_packus_epi32(d0, d1), 0xD8));
                    }
                    for (; dx < rsF; dx += F)
                    {
                        __m256i d0 = Float32ToBFloat16(_mm256_fmadd_ps(_mm256_loadu_ps(pbx[0] + dx), _fy0, _mm256_mul_ps(_mm256_loadu_ps(pbx[1] + dx), _fy1)));
                        _mm_storeu_si128((__m128i*)(dst + dx), _mm256_castsi256_si128(_mm256_permute4x64_epi64(_mm256_packus_epi32(d0, K_ZERO), 0xD8)));
                    }
                    for (; dx < rsH; dx += Sse41::F)
                    {
                        __m128i d0 = Sse41::Float32ToBFloat16(_mm_fmadd_ps(_mm_loadu_ps(pbx[0] + dx), _mm256_castps256_ps128(_fy0), 
                            _mm_mul_ps(_mm_loadu_ps(pbx[1] + dx), _mm256_castps256_ps128(_fy1))));
                        _mm_storel_epi64((__m128i*)(dst + dx), _mm_packus_epi32(d0, Sse41::K_ZERO));
                    }
                    for (; dx < rs; dx++)
                        dst[dx] = Base::Float32ToBFloat16(pbx[0][dx] * fy0 + pbx[1][dx] * fy1);
                }
            }
            else
            {
                if (cnD)
                {
                    for (size_t dy = 0; dy < _param.dstH; dy++, dst += dstStride)
                    {
                        __m256 fy1 = _mm256_set1_ps(_ay[dy]);
                        __m256 fy0 = _mm256_sub_ps(_1, fy1);
                        const uint16_t* src0 = src + _iy[dy] * srcStride, * src1 = src0 + srcStride;
                        for (size_t dx = 0; dx < _param.dstW; dx++)
                        {
                            size_t os = _ix[dx], eD = os + cnD, eF = os + cnF, od = dx * cn;
                            __m256 fx1 = _mm256_set1_ps(_ax[dx]);
                            __m256 fx0 = _mm256_sub_ps(_1, fx1);
                            for (; os < eD; os += DF, od += DF)
                            {
                                __m256i s00 = _mm256_loadu_si256((__m256i*)(src0 + os));
                                __m256i s01 = _mm256_loadu_si256((__m256i*)(src0 + os + cn));
                                __m256i s10 = _mm256_loadu_si256((__m256i*)(src1 + os));
                                __m256i s11 = _mm256_loadu_si256((__m256i*)(src1 + os + cn));

                                __m256 r0e = _mm256_fmadd_ps(fx0, BFloat16ToFloat32Even(s00), _mm256_mul_ps(fx1, BFloat16ToFloat32Even(s01)));
                                __m256 r1e = _mm256_fmadd_ps(fx0, BFloat16ToFloat32Even(s10), _mm256_mul_ps(fx1, BFloat16ToFloat32Even(s11)));
                                __m256 even = _mm256_fmadd_ps(r0e, fy0, _mm256_mul_ps(r1e, fy1));

                                __m256 r0o = _mm256_fmadd_ps(fx0, BFloat16ToFloat32Odd(s00), _mm256_mul_ps(fx1, BFloat16ToFloat32Odd(s01)));
                                __m256 r1o = _mm256_fmadd_ps(fx0, BFloat16ToFloat32Odd(s10), _mm256_mul_ps(fx1, BFloat16ToFloat32Odd(s11)));
                                __m256 odd = _mm256_fmadd_ps(r0o, fy0, _mm256_mul_ps(r1o, fy1));

                                _mm256_storeu_si256((__m256i*)(dst + od), Float32ToBFloat16Interlived(even, odd));
                            }
                            for (; os < eF; os += F, od += F)
                            {
                                __m256 r0 = BilinearRowSumBf16(src0 + os, cn, fx0, fx1);
                                __m256 r1 = BilinearRowSumBf16(src1 + os, cn, fx0, fx1);
                                __m256i d0 = Float32ToBFloat16(_mm256_fmadd_ps(r0, fy0, _mm256_mul_ps(r1, fy1)));
                                _mm_storeu_si128((__m128i*)(dst + od), _mm256_castsi256_si128(_mm256_permute4x64_epi64(_mm256_packus_epi32(d0, K_ZERO), 0xD8)));
                            }
                            if (cnTF)
                            {
                                os += cnLF;
                                od += cnLF;
                                __m256 r0 = BilinearRowSumBf16(src0 + os, cn, fx0, fx1);
                                __m256 r1 = BilinearRowSumBf16(src1 + os, cn, fx0, fx1);
                                __m256i d0 = Float32ToBFloat16(_mm256_fmadd_ps(r0, fy0, _mm256_mul_ps(r1, fy1)));
                                _mm_storeu_si128((__m128i*)(dst + od), _mm256_castsi256_si128(_mm256_permute4x64_epi64(_mm256_packus_epi32(d0, K_ZERO), 0xD8)));
                            }
                        }
                    }
                }
                else if (cnF)
                {
                    for (size_t dy = 0; dy < _param.dstH; dy++, dst += dstStride)
                    {
                        __m256 fy1 = _mm256_set1_ps(_ay[dy]);
                        __m256 fy0 = _mm256_sub_ps(_1, fy1);
                        const uint16_t* src0 = src + _iy[dy] * srcStride, * src1 = src0 + srcStride;
                        for (size_t dx = 0; dx < _param.dstW; dx++)
                        {
                            size_t os = _ix[dx], eF = os + cnF, od = dx * cn;
                            __m256 fx1 = _mm256_set1_ps(_ax[dx]);
                            __m256 fx0 = _mm256_sub_ps(_1, fx1);
                            for (; os < eF; os += F, od += F)
                            {
                                __m256 r0 = BilinearRowSumBf16(src0 + os, cn, fx0, fx1);
                                __m256 r1 = BilinearRowSumBf16(src1 + os, cn, fx0, fx1);
                                __m256i d0 = Float32ToBFloat16(_mm256_fmadd_ps(r0, fy0, _mm256_mul_ps(r1, fy1)));
                                _mm_storeu_si128((__m128i*)(dst + od), _mm256_castsi256_si128(_mm256_permute4x64_epi64(_mm256_packus_epi32(d0, K_ZERO), 0xD8)));
                            }
                            if (cnTF)
                            {
                                os += cnLF;
                                od += cnLF;
                                __m256 r0 = BilinearRowSumBf16(src0 + os, cn, fx0, fx1);
                                __m256 r1 = BilinearRowSumBf16(src1 + os, cn, fx0, fx1);
                                __m256i d0 = Float32ToBFloat16(_mm256_fmadd_ps(r0, fy0, _mm256_mul_ps(r1, fy1)));
                                _mm_storeu_si128((__m128i*)(dst + od), _mm256_castsi256_si128(_mm256_permute4x64_epi64(_mm256_packus_epi32(d0, K_ZERO), 0xD8)));
                            }
                        }
                    }
                }
                else
                {
                    for (size_t dy = 0; dy < _param.dstH; dy++, dst += dstStride)
                    {
                        __m128 fy1 = _mm_set1_ps(_ay[dy]);
                        __m128 fy0 = _mm_sub_ps(_mm256_castps256_ps128(_1), fy1);
                        const uint16_t* src0 = src + _iy[dy] * srcStride, * src1 = src0 + srcStride;
                        for (size_t dx = 0; dx < _param.dstW; dx++)
                        {
                            size_t os = _ix[dx], eF = os + cnH, od = dx * cn;
                            __m128 fx1 = _mm_set1_ps(_ax[dx]);
                            __m128 fx0 = _mm_sub_ps(_mm256_castps256_ps128(_1), fx1);
                            for (; os < eF; os += Sse41::F, od += Sse41::F)
                            {
                                __m128 r0 = BilinearRowSumBf16(src0 + os, cn, fx0, fx1);
                                __m128 r1 = BilinearRowSumBf16(src1 + os, cn, fx0, fx1);
                                __m128i d0 = Sse41::Float32ToBFloat16(_mm_fmadd_ps(r0, fy0, _mm_mul_ps(r1, fy1)));
                                _mm_storel_epi64((__m128i*)(dst + od), _mm_packus_epi32(d0, Sse41::K_ZERO));
                            }
                            if (cnTH)
                            {
                                os += cnLH;
                                od += cnLH;
                                __m128 r0 = BilinearRowSumBf16(src0 + os, cn, fx0, fx1);
                                __m128 r1 = BilinearRowSumBf16(src1 + os, cn, fx0, fx1);
                                __m128i d0 = Sse41::Float32ToBFloat16(_mm_fmadd_ps(r0, fy0, _mm_mul_ps(r1, fy1)));
                                _mm_storel_epi64((__m128i*)(dst + od), _mm_packus_epi32(d0, Sse41::K_ZERO));
                            }
                        }
                    }
                }
            }
        }
    }
#endif 
}

