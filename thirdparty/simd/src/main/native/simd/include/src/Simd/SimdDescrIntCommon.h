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
#ifndef __SimdDescrIntCommon_h__
#define __SimdDescrIntCommon_h__

#include "Simd/SimdDefs.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdLoad.h"
#include "Simd/SimdSet.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdShuffle.h"

namespace Simd
{
    namespace Base
    {
        SIMD_INLINE void DecodeCosineDistance(const uint8_t* a, const uint8_t* b, float abSum, float* distance)
        {
            float aScale = ((float*)a)[0];
            float aShift = ((float*)a)[1];
            float aMean = ((float*)a)[2];
            float aNorm = ((float*)a)[3];
            float bScale = ((float*)b)[0];
            float bShift = ((float*)b)[1];
            float bMean = ((float*)b)[2];
            float bNorm = ((float*)b)[3];
            float ab = abSum * aScale * bScale + aMean * bShift + bMean * aShift;
            distance[0] = Simd::RestrictRange(1.0f - ab / (aNorm * bNorm), 0.0f, 2.0f);
        }
    }

#ifdef SIMD_SSE41_ENABLE
    namespace Sse41
    {
        const __m128i E4_MULLO = SIMD_MM_SETR_EPI16(4096, 1, 4096, 1, 4096, 1, 4096, 1);

        const __m128i E5_MULLO = SIMD_MM_SETR_EPI16(256, 32, 4, 128, 16, 2, 64, 8);
        const __m128i E5_SHFL0 = SIMD_MM_SETR_EPI8(0x1, 0x3, 0x7, 0x9, 0xD, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        const __m128i E5_SHFL1 = SIMD_MM_SETR_EPI8(0x2, 0x4, 0x8, 0xA, 0xE, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        const __m128i E5_SHFL2 = SIMD_MM_SETR_EPI8( -1, 0x6,  -1, 0xC,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);

        const __m128i E6_MULLO = SIMD_MM_SETR_EPI16(256, 64, 16, 4, 256, 64, 16, 4);
        const __m128i E6_SHFL0 = SIMD_MM_SETR_EPI8(0x1, 0x3, 0x5, 0x9, 0xB, 0xD, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        const __m128i E6_SHFL1 = SIMD_MM_SETR_EPI8(0x2, 0x4, 0x6, 0xA, 0xC, 0xE, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);

        const __m128i E7_MULLO = SIMD_MM_SETR_EPI16(256, 128, 64, 32, 16, 8, 4, 2);
        const __m128i E7_SHFL0 = SIMD_MM_SETR_EPI8(0x1, 0x3, 0x5, 0x7, 0x9, 0xB, 0xD, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        const __m128i E7_SHFL1 = SIMD_MM_SETR_EPI8(0x2, 0x4, 0x6, 0x8, 0xA, 0xC, 0xE, -1, -1, -1, -1, -1, -1, -1, -1, -1);

        const __m128i C4_MULLO = SIMD_MM_SETR_EPI16(4096, 256, 4096, 256, 4096, 256, 4096, 256);
        const __m128i C4_SHFL0 = SIMD_MM_SETR_EPI8(0x0, 0x0, 0x0, 0x0, 0x1, 0x1, 0x1, 0x1, 0x2, 0x2, 0x2, 0x2, 0x3, 0x3, 0x3, 0x3);
        const __m128i C4_SHFL1 = SIMD_MM_SETR_EPI8(0x4, 0x4, 0x4, 0x4, 0x5, 0x5, 0x5, 0x5, 0x6, 0x6, 0x6, 0x6, 0x7, 0x7, 0x7, 0x7);

        const __m128i C5_SHFL0 = SIMD_MM_SETR_EPI8(0x0, 0x0, 0x0, 0x1, 0x1, 0x1, 0x1, 0x2, 0x2, 0x3, 0x3, 0x3, 0x3, 0x4, 0x4, 0x4);
        const __m128i C5_SHFL1 = SIMD_MM_SETR_EPI8(0x5, 0x5, 0x5, 0x6, 0x6, 0x6, 0x6, 0x7, 0x7, 0x8, 0x8, 0x8, 0x8, 0x9, 0x9, 0x9);
        const __m128i C5_MULLO = SIMD_MM_SETR_EPI16(8, 64, 2, 16, 128, 4, 32, 256);

        const __m128i C6_SHFL0 = SIMD_MM_SETR_EPI8(0x0, 0x0, 0x0, 0x1, 0x1, 0x2, 0x2, 0x2, 0x3, 0x3, 0x3, 0x4, 0x4, 0x5, 0x5, 0x5);
        const __m128i C6_SHFL1 = SIMD_MM_SETR_EPI8(0x6, 0x6, 0x6, 0x7, 0x7, 0x8, 0x8, 0x8, 0x9, 0x9, 0x9, 0xA, 0xA, 0xB, 0xB, 0xB);
        const __m128i C6_MULLO = SIMD_MM_SETR_EPI16(4, 16, 64, 256, 4, 16, 64, 256);

        const __m128i C7_SHFL0 = SIMD_MM_SETR_EPI8(0x0, 0x0, 0x0, 0x1, 0x1, 0x2, 0x2, 0x3, 0x3, 0x4, 0x4, 0x5, 0x5, 0x6, 0x6, 0x6);
        const __m128i C7_SHFL1 = SIMD_MM_SETR_EPI8(0x7, 0x7, 0x7, 0x8, 0x8, 0x9, 0x9, 0xA, 0xA, 0xB, 0xB, 0xC, 0xC, 0xD, 0xD, 0xD);
        const __m128i C7_MULLO = SIMD_MM_SETR_EPI16(2, 4, 8, 16, 32, 64, 128, 256);

        //-------------------------------------------------------------------------------------------------

        template<int bits> __m128i LoadLast8(const uint8_t* src)
        {
            return _mm_srli_si128(_mm_loadl_epi64((__m128i*)(src + bits - 8)), 8 - bits);
        }

        template<int bits> __m128i LoadLast16(const uint8_t* src)
        {
            return _mm_srli_si128(_mm_loadu_si128((__m128i*)(src + bits * 2 - 16)), 16 - bits * 2);
        }

        //-------------------------------------------------------------------------------------------------

        template<int bits> __m128i UnpackData8(const uint8_t* src);

        template<> SIMD_INLINE __m128i UnpackData8<4>(const uint8_t* src)
        {
            __m128i _src = LoadLast8<4>(src);
            __m128i lo = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_src, C4_SHFL0), C4_MULLO), 12);
            return _mm_packus_epi16(lo, K_ZERO);
        }

        template<> SIMD_INLINE __m128i UnpackData8<5>(const uint8_t* src)
        {
            __m128i _src = LoadLast8<5>(src);
            __m128i lo = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_src, C5_SHFL0), C5_MULLO), 11);
            return _mm_packus_epi16(lo, K_ZERO);
        }

        template<> SIMD_INLINE __m128i UnpackData8<6>(const uint8_t* src)
        {
            __m128i _src = LoadLast8<6>(src);
            __m128i lo = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_src, C6_SHFL0), C6_MULLO), 10);
            return _mm_packus_epi16(lo, K_ZERO);
        }

        template<> SIMD_INLINE __m128i UnpackData8<7>(const uint8_t* src)
        {
            __m128i _src = LoadLast8<7>(src);
            __m128i lo = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_src, C7_SHFL0), C7_MULLO), 9);
            return _mm_packus_epi16(lo, K_ZERO);
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE void DecodeCosineDistances1x4(const uint8_t* a, const uint8_t* const* B, __m128 abSum, float* distances)
        {
            __m128 aScale, aShift, aMean, aNorm, bScale, bShift, bMean, bNorm;
            bScale = _mm_loadu_ps((float*)B[0]);
            bShift = _mm_loadu_ps((float*)B[1]);
            bMean = _mm_loadu_ps((float*)B[2]);
            bNorm = _mm_loadu_ps((float*)B[3]);
            aScale = _mm_unpacklo_ps(bScale, bMean);
            aShift = _mm_unpacklo_ps(bShift, bNorm);
            aMean = _mm_unpackhi_ps(bScale, bMean);
            aNorm = _mm_unpackhi_ps(bShift, bNorm);
            bScale = _mm_unpacklo_ps(aScale, aShift);
            bShift = _mm_unpackhi_ps(aScale, aShift);
            bMean = _mm_unpacklo_ps(aMean, aNorm);
            bNorm = _mm_unpackhi_ps(aMean, aNorm);

            aScale = _mm_set1_ps(((float*)a)[0]);
            aShift = _mm_set1_ps(((float*)a)[1]);
            aMean = _mm_set1_ps(((float*)a)[2]);
            aNorm = _mm_set1_ps(((float*)a)[3]);

            __m128 ab = _mm_mul_ps(abSum, _mm_mul_ps(aScale, bScale));
            ab = _mm_add_ps(_mm_mul_ps(aMean, bShift), ab);
            ab = _mm_add_ps(_mm_mul_ps(bMean, aShift), ab);

            _mm_storeu_ps(distances, _mm_min_ps(_mm_max_ps(_mm_sub_ps(_mm_set1_ps(1.0f), _mm_div_ps(ab, _mm_mul_ps(aNorm, bNorm))), _mm_setzero_ps()), _mm_set1_ps(2.0f)));
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE void DecodeCosineDistances1xF(const float* a, const float *b, size_t stride, __m128i abSum, float* distances)
        {
            __m128 aScale = _mm_set1_ps(a[0]);
            __m128 aShift = _mm_set1_ps(a[1]);
            __m128 aMean = _mm_set1_ps(a[2]);
            __m128 aNorm = _mm_set1_ps(a[3]);
            __m128 bScale = _mm_loadu_ps(b + 0 * stride);
            __m128 bShift = _mm_loadu_ps(b + 1 * stride);
            __m128 bMean = _mm_loadu_ps(b + 2 * stride);
            __m128 bNorm = _mm_loadu_ps(b + 3 * stride);
            __m128 ab = _mm_mul_ps(_mm_cvtepi32_ps(abSum), _mm_mul_ps(aScale, bScale));
            ab = _mm_add_ps(_mm_mul_ps(aMean, bShift), ab);
            ab = _mm_add_ps(_mm_mul_ps(bMean, aShift), ab);
            _mm_storeu_ps(distances, _mm_min_ps(_mm_max_ps(_mm_sub_ps(_mm_set1_ps(1.0f), _mm_div_ps(ab, _mm_mul_ps(aNorm, bNorm))), _mm_setzero_ps()), _mm_set1_ps(2.0f)));
        }

        SIMD_INLINE void DecodeCosineDistances1xF(const float* a, const float* b, size_t stride, __m128i abSum, float* distances, size_t N)
        {
            float d[F];
            DecodeCosineDistances1xF(a, b, stride, abSum, d);
            for (size_t i = 0; i < N; ++i)
                distances[i] = d[i];
        }
    }
#endif

#ifdef SIMD_AVX2_ENABLE
    namespace Avx2
    {
        const __m256i E4_MULLO = SIMD_MM256_SETR_EPI16(4096, 1, 4096, 1, 4096, 1, 4096, 1, 4096, 1, 4096, 1, 4096, 1, 4096, 1);

        const __m256i E5_MULLO = SIMD_MM256_SETR_EPI16(256, 32, 4, 128, 16, 2, 64, 8, 256, 32, 4, 128, 16, 2, 64, 8);
        const __m256i E5_SHFL0 = SIMD_MM256_SETR_EPI8(
            0x1, 0x3, 0x7, 0x9, 0xD, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, 0x1, 0x3, 0x7, 0x9, 0xD, -1, -1, -1, -1, -1, -1);
        const __m256i E5_SHFL1 = SIMD_MM256_SETR_EPI8(
            0x2, 0x4, 0x8, 0xA, 0xE, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, 0x2, 0x4, 0x8, 0xA, 0xE, -1, -1, -1, -1, -1, -1);
        const __m256i E5_SHFL2 = SIMD_MM256_SETR_EPI8(
            -1, 0x6, -1, 0xC, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, 0x6, -1, 0xC, -1, -1, -1, -1, -1, -1, -1);

        const __m256i E6_MULLO = SIMD_MM256_SETR_EPI16(256, 64, 16, 4, 256, 64, 16, 4, 256, 64, 16, 4, 256, 64, 16, 4);
        const __m256i E6_SHFL0 = SIMD_MM256_SETR_EPI8(
            0x1, 0x3, 0x5, 0x9, 0xB, 0xD, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, 0x1, 0x3, 0x5, 0x9, 0xB, 0xD, -1, -1, -1, -1);
        const __m256i E6_SHFL1 = SIMD_MM256_SETR_EPI8(
            0x2, 0x4, 0x6, 0xA, 0xC, 0xE, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, 0x2, 0x4, 0x6, 0xA, 0xC, 0xE, -1, -1, -1, -1);

        const __m256i E7_MULLO = SIMD_MM256_SETR_EPI16(256, 128, 64, 32, 16, 8, 4, 2, 256, 128, 64, 32, 16, 8, 4, 2);
        const __m256i E7_SHFL0 = SIMD_MM256_SETR_EPI8(
            0x1, 0x3, 0x5, 0x7, 0x9, 0xB, 0xD, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, 0x1, 0x3, 0x5, 0x7, 0x9, 0xB, 0xD, -1, -1);
        const __m256i E7_SHFL1 = SIMD_MM256_SETR_EPI8(
            0x2, 0x4, 0x6, 0x8, 0xA, 0xC, 0xE, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, 0x2, 0x4, 0x6, 0x8, 0xA, 0xC, 0xE, -1, -1);

        const __m256i C4_SHFL = SIMD_MM256_SETR_EPI8(
            0x0, 0x0, 0x0, 0x0, 0x1, 0x1, 0x1, 0x1, 0x2, 0x2, 0x2, 0x2, 0x3, 0x3, 0x3, 0x3,
            0x4, 0x4, 0x4, 0x4, 0x5, 0x5, 0x5, 0x5, 0x6, 0x6, 0x6, 0x6, 0x7, 0x7, 0x7, 0x7);
        const __m256i C4_MULLO = SIMD_MM256_SETR_EPI16(4096, 256, 4096, 256, 4096, 256, 4096, 256, 4096, 256, 4096, 256, 4096, 256, 4096, 256);

        const __m256i C5_SHFL = SIMD_MM256_SETR_EPI8(
            0x0, 0x0, 0x0, 0x1, 0x1, 0x1, 0x1, 0x2, 0x2, 0x3, 0x3, 0x3, 0x3, 0x4, 0x4, 0x4,
            0x5, 0x5, 0x5, 0x6, 0x6, 0x6, 0x6, 0x7, 0x7, 0x8, 0x8, 0x8, 0x8, 0x9, 0x9, 0x9);
        const __m256i C5_MULLO = SIMD_MM256_SETR_EPI16(8, 64, 2, 16, 128, 4, 32, 256, 8, 64, 2, 16, 128, 4, 32, 256);

        const __m256i C6_SHFL = SIMD_MM256_SETR_EPI8(
            0x0, 0x0, 0x0, 0x1, 0x1, 0x2, 0x2, 0x2, 0x3, 0x3, 0x3, 0x4, 0x4, 0x5, 0x5, 0x5,
            0x6, 0x6, 0x6, 0x7, 0x7, 0x8, 0x8, 0x8, 0x9, 0x9, 0x9, 0xA, 0xA, 0xB, 0xB, 0xB);
        const __m256i C6_MULLO = SIMD_MM256_SETR_EPI16(4, 16, 64, 256, 4, 16, 64, 256, 4, 16, 64, 256, 4, 16, 64, 256);

        const __m256i C7_SHFL = SIMD_MM256_SETR_EPI8(
            0x0, 0x0, 0x0, 0x1, 0x1, 0x2, 0x2, 0x3, 0x3, 0x4, 0x4, 0x5, 0x5, 0x6, 0x6, 0x6,
            0x7, 0x7, 0x7, 0x8, 0x8, 0x9, 0x9, 0xA, 0xA, 0xB, 0xB, 0xC, 0xC, 0xD, 0xD, 0xD);
        const __m256i C7_MULLO = SIMD_MM256_SETR_EPI16(2, 4, 8, 16, 32, 64, 128, 256, 2, 4, 8, 16, 32, 64, 128, 256);

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE void DecodeCosineDistances2x4(const uint8_t* const* A, const uint8_t* const* B, __m256 abSum, float* distances, size_t stride)
        {
            __m256 aScale, aShift, aMean, aNorm, bScale, bShift, bMean, bNorm;
            bScale = _mm256_broadcast_ps((__m128*)B[0]);
            bShift = _mm256_broadcast_ps((__m128*)B[1]);
            bMean = _mm256_broadcast_ps((__m128*)B[2]);
            bNorm = _mm256_broadcast_ps((__m128*)B[3]);
            aScale = _mm256_unpacklo_ps(bScale, bMean);
            aShift = _mm256_unpacklo_ps(bShift, bNorm);
            aMean = _mm256_unpackhi_ps(bScale, bMean);
            aNorm = _mm256_unpackhi_ps(bShift, bNorm);
            bScale = _mm256_unpacklo_ps(aScale, aShift);
            bShift = _mm256_unpackhi_ps(aScale, aShift);
            bMean = _mm256_unpacklo_ps(aMean, aNorm);
            bNorm = _mm256_unpackhi_ps(aMean, aNorm);

            aNorm = Load<false>((float*)A[0], (float*)A[1]);
            aScale = Broadcast<0>(aNorm);
            aShift = Broadcast<1>(aNorm);
            aMean = Broadcast<2>(aNorm);
            aNorm = Broadcast<3>(aNorm);

            __m256 ab = _mm256_mul_ps(abSum, _mm256_mul_ps(aScale, bScale));
            ab = _mm256_fmadd_ps(aMean, bShift, ab);
            ab = _mm256_fmadd_ps(bMean, aShift, ab);

            Store<false>(distances + 0 * stride, distances + 1 * stride,
                _mm256_min_ps(_mm256_max_ps(_mm256_sub_ps(_mm256_set1_ps(1.0f), _mm256_div_ps(ab, _mm256_mul_ps(aNorm, bNorm))), _mm256_setzero_ps()), _mm256_set1_ps(2.0f)));
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE void DecodeCosineDistances1xF(const float* a, const float* b, size_t stride, __m256i abSum, float* distances)
        {
            __m256 aScale = _mm256_set1_ps(a[0]);
            __m256 aShift = _mm256_set1_ps(a[1]);
            __m256 aMean = _mm256_set1_ps(a[2]);
            __m256 aNorm = _mm256_set1_ps(a[3]);
            __m256 bScale = _mm256_loadu_ps(b + 0 * stride);
            __m256 bShift = _mm256_loadu_ps(b + 1 * stride);
            __m256 bMean = _mm256_loadu_ps(b + 2 * stride);
            __m256 bNorm = _mm256_loadu_ps(b + 3 * stride);
            __m256 ab = _mm256_mul_ps(_mm256_cvtepi32_ps(abSum), _mm256_mul_ps(aScale, bScale));
            ab = _mm256_add_ps(_mm256_mul_ps(aMean, bShift), ab);
            ab = _mm256_add_ps(_mm256_mul_ps(bMean, aShift), ab);
            _mm256_storeu_ps(distances, _mm256_min_ps(_mm256_max_ps(_mm256_sub_ps(_mm256_set1_ps(1.0f), _mm256_div_ps(ab, _mm256_mul_ps(aNorm, bNorm))), _mm256_setzero_ps()), _mm256_set1_ps(2.0f)));
        }

        SIMD_INLINE void DecodeCosineDistances1xF(const float* a, const float* b, size_t stride, __m256i abSum, float* distances, size_t N)
        {
            float d[F];
            DecodeCosineDistances1xF(a, b, stride, abSum, d);
            for (size_t i = 0; i < N; ++i)
                distances[i] = d[i];
        }
    }
#endif

#ifdef SIMD_AVX512BW_ENABLE
    namespace Avx512bw
    {
        const __m512i EX_PERM = SIMD_MM512_SETR_EPI64(0, 2, 1, 3, 4, 6, 5, 7);

        const __m512i E4_MULLO = SIMD_MM512_SETR_EPI16(
            4096, 1, 4096, 1, 4096, 1, 4096, 1, 4096, 1, 4096, 1, 4096, 1, 4096, 1,
            4096, 1, 4096, 1, 4096, 1, 4096, 1, 4096, 1, 4096, 1, 4096, 1, 4096, 1);

        const __m512i E5_MULLO = SIMD_MM512_SETR_EPI16(
            256, 32, 4, 128, 16, 2, 64, 8, 256, 32, 4, 128, 16, 2, 64, 8,
            256, 32, 4, 128, 16, 2, 64, 8, 256, 32, 4, 128, 16, 2, 64, 8);
        const __m512i E5_SHFL0 = SIMD_MM512_SETR_EPI8(
            -1, -1, -1, -1, -1, -1, 0x1, 0x3, 0x7, 0x9, 0xD, -1, -1, -1, -1, -1,
            0x1, 0x3, 0x7, 0x9, 0xD, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0x1, 0x3, 0x7, 0x9, 0xD,
            -1, -1, -1, -1, -1, 0x1, 0x3, 0x7, 0x9, 0xD, -1, -1, -1, -1, -1, -1);
        const __m512i E5_SHFL1 = SIMD_MM512_SETR_EPI8(
            -1, -1, -1, -1, -1, -1, 0x2, 0x4, 0x8, 0xA, 0xE, -1, -1, -1, -1, -1,
            0x2, 0x4, 0x8, 0xA, 0xE, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0x2, 0x4, 0x8, 0xA, 0xE,
            -1, -1, -1, -1, -1, 0x2, 0x4, 0x8, 0xA, 0xE, -1, -1, -1, -1, -1, -1);
        const __m512i E5_SHFL2 = SIMD_MM512_SETR_EPI8(
            -1, -1, -1, -1, -1, -1, -1, 0x6, -1, 0xC, -1, -1, -1, -1, -1, -1,
            -1, 0x6, -1, 0xC, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0x6, -1, 0xC, -1,
            -1, -1, -1, -1, -1, -1, 0x6, -1, 0xC, -1, -1, -1, -1, -1, -1, -1);

        const __m512i E6_MULLO = SIMD_MM512_SETR_EPI16(
            256, 64, 16, 4, 256, 64, 16, 4, 256, 64, 16, 4, 256, 64, 16, 4,
            256, 64, 16, 4, 256, 64, 16, 4, 256, 64, 16, 4, 256, 64, 16, 4);
        const __m512i E6_SHFL0 = SIMD_MM512_SETR_EPI8(
            -1, -1, -1, -1, 0x1, 0x3, 0x5, 0x9, 0xB, 0xD, -1, -1, -1, -1, -1, -1,
            0x1, 0x3, 0x5, 0x9, 0xB, 0xD, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0x1, 0x3, 0x5, 0x9, 0xB, 0xD,
            -1, -1, -1, -1, -1, -1, 0x1, 0x3, 0x5, 0x9, 0xB, 0xD, -1, -1, -1, -1);
        const __m512i E6_SHFL1 = SIMD_MM512_SETR_EPI8(
            -1, -1, -1, -1, 0x2, 0x4, 0x6, 0xA, 0xC, 0xE, -1, -1, -1, -1, -1, -1,
            0x2, 0x4, 0x6, 0xA, 0xC, 0xE, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0x2, 0x4, 0x6, 0xA, 0xC, 0xE,
            -1, -1, -1, -1, -1, -1, 0x2, 0x4, 0x6, 0xA, 0xC, 0xE, -1, -1, -1, -1);

        const __m512i E7_MULLO = SIMD_MM512_SETR_EPI16(
            256, 128, 64, 32, 16, 8, 4, 2, 256, 128, 64, 32, 16, 8, 4, 2,
            256, 128, 64, 32, 16, 8, 4, 2, 256, 128, 64, 32, 16, 8, 4, 2);
        const __m512i E7_SHFL0 = SIMD_MM512_SETR_EPI8(
            -1, -1, 0x1, 0x3, 0x5, 0x7, 0x9, 0xB, 0xD, -1, -1, -1, -1, -1, -1, -1,
            0x1, 0x3, 0x5, 0x7, 0x9, 0xB, 0xD, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, 0x1, 0x3, 0x5, 0x7, 0x9, 0xB, 0xD,
            -1, -1, -1, -1, -1, -1, -1, 0x1, 0x3, 0x5, 0x7, 0x9, 0xB, 0xD, -1, -1);
        const __m512i E7_SHFL1 = SIMD_MM512_SETR_EPI8(
            -1, -1, 0x2, 0x4, 0x6, 0x8, 0xA, 0xC, 0xE, -1, -1, -1, -1, -1, -1, -1,
            0x2, 0x4, 0x6, 0x8, 0xA, 0xC, 0xE, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, 0x2, 0x4, 0x6, 0x8, 0xA, 0xC, 0xE,
            -1, -1, -1, -1, -1, -1, -1, 0x2, 0x4, 0x6, 0x8, 0xA, 0xC, 0xE, -1, -1);

        const __m512i C4_MULLO = SIMD_MM512_SETR_EPI16(
            4096, 256, 4096, 256, 4096, 256, 4096, 256, 4096, 256, 4096, 256, 4096, 256, 4096, 256,
            4096, 256, 4096, 256, 4096, 256, 4096, 256, 4096, 256, 4096, 256, 4096, 256, 4096, 256);

        const __m512i C5_PERM = SIMD_MM512_SETR_EPI32(
            0x0, 0x1, 0x0, 0x0, 0x1, 0x2, 0x0, 0x0, 0x2, 0x3, 0x0, 0x0, 0x3, 0x4, 0x0, 0x0);
        const __m512i C5_SHFL = SIMD_MM512_SETR_EPI8(
            0x0, 0x0, 0x0, 0x1, 0x1, 0x1, 0x1, 0x2, 0x2, 0x3, 0x3, 0x3, 0x3, 0x4, 0x4, 0x4,
            0x1, 0x1, 0x1, 0x2, 0x2, 0x2, 0x2, 0x3, 0x3, 0x4, 0x4, 0x4, 0x4, 0x5, 0x5, 0x5,
            0x2, 0x2, 0x2, 0x3, 0x3, 0x3, 0x3, 0x4, 0x4, 0x5, 0x5, 0x5, 0x5, 0x6, 0x6, 0x6,
            0x3, 0x3, 0x3, 0x4, 0x4, 0x4, 0x4, 0x5, 0x5, 0x6, 0x6, 0x6, 0x6, 0x7, 0x7, 0x7);
        const __m512i C5_MULLO = SIMD_MM512_SETR_EPI16(
            8, 64, 2, 16, 128, 4, 32, 256, 8, 64, 2, 16, 128, 4, 32, 256,
            8, 64, 2, 16, 128, 4, 32, 256, 8, 64, 2, 16, 128, 4, 32, 256);

        const __m512i C6_PERM = SIMD_MM512_SETR_EPI32(
            0x0, 0x1, 0x0, 0x0, 0x1, 0x2, 0x0, 0x0, 0x3, 0x4, 0x0, 0x0, 0x4, 0x5, 0x0, 0x0);
        const __m512i C6_SHFL = SIMD_MM512_SETR_EPI8(
            0x0, 0x0, 0x0, 0x1, 0x1, 0x2, 0x2, 0x2, 0x3, 0x3, 0x3, 0x4, 0x4, 0x5, 0x5, 0x5,
            0x2, 0x2, 0x2, 0x3, 0x3, 0x4, 0x4, 0x4, 0x5, 0x5, 0x5, 0x6, 0x6, 0x7, 0x7, 0x7,
            0x0, 0x0, 0x0, 0x1, 0x1, 0x2, 0x2, 0x2, 0x3, 0x3, 0x3, 0x4, 0x4, 0x5, 0x5, 0x5,
            0x2, 0x2, 0x2, 0x3, 0x3, 0x4, 0x4, 0x4, 0x5, 0x5, 0x5, 0x6, 0x6, 0x7, 0x7, 0x7);
        const __m512i C6_MULLO = SIMD_MM512_SETR_EPI16(
            4, 16, 64, 256, 4, 16, 64, 256, 4, 16, 64, 256, 4, 16, 64, 256,
            4, 16, 64, 256, 4, 16, 64, 256, 4, 16, 64, 256, 4, 16, 64, 256);

        const __m512i C7_PERM = SIMD_MM512_SETR_EPI32(
            0x0, 0x1, 0x0, 0x0, 0x1, 0x2, 0x3, 0x0, 0x3, 0x4, 0x5, 0x0, 0x5, 0x6, 0x0, 0x0);
        const __m512i C7_SHFL = SIMD_MM512_SETR_EPI8(
            0x0, 0x0, 0x0, 0x1, 0x1, 0x2, 0x2, 0x3, 0x3, 0x4, 0x4, 0x5, 0x5, 0x6, 0x6, 0x6,
            0x3, 0x3, 0x3, 0x4, 0x4, 0x5, 0x5, 0x6, 0x6, 0x7, 0x7, 0x8, 0x8, 0x9, 0x9, 0x9,
            0x2, 0x2, 0x2, 0x3, 0x3, 0x4, 0x4, 0x5, 0x5, 0x6, 0x6, 0x7, 0x7, 0x8, 0x8, 0x8,
            0x1, 0x1, 0x1, 0x2, 0x2, 0x3, 0x3, 0x4, 0x4, 0x5, 0x5, 0x6, 0x6, 0x7, 0x7, 0x7);
        const __m512i C7_MULLO = SIMD_MM512_SETR_EPI16(
            2, 4, 8, 16, 32, 64, 128, 256, 2, 4, 8, 16, 32, 64, 128, 256,
            2, 4, 8, 16, 32, 64, 128, 256, 2, 4, 8, 16, 32, 64, 128, 256);

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE void DecodeCosineDistances1xF(const float* a, const float* b, size_t stride, __m512i abSum, float* distances, __mmask16 mask = -1)
        {
            __m512 aScale = _mm512_set1_ps(a[0]);
            __m512 aShift = _mm512_set1_ps(a[1]);
            __m512 aMean = _mm512_set1_ps(a[2]);
            __m512 aNorm = _mm512_set1_ps(a[3]);
            __m512 bScale = _mm512_maskz_loadu_ps(mask, b + 0 * stride);
            __m512 bShift = _mm512_maskz_loadu_ps(mask, b + 1 * stride);
            __m512 bMean = _mm512_maskz_loadu_ps(mask, b + 2 * stride);
            __m512 bNorm = _mm512_maskz_loadu_ps(mask, b + 3 * stride);
            __m512 ab = _mm512_mul_ps(_mm512_cvtepi32_ps(abSum), _mm512_mul_ps(aScale, bScale));
            ab = _mm512_add_ps(_mm512_mul_ps(aMean, bShift), ab);
            ab = _mm512_add_ps(_mm512_mul_ps(bMean, aShift), ab);
            _mm512_mask_storeu_ps(distances, mask, _mm512_min_ps(_mm512_max_ps(_mm512_sub_ps(_mm512_set1_ps(1.0f), _mm512_div_ps(ab, _mm512_mul_ps(aNorm, bNorm))), _mm512_setzero_ps()), _mm512_set1_ps(2.0f)));
        }
    }
#endif

#if defined(SIMD_AMXBF16_ENABLE)
    namespace AmxBf16
    {
        const __m512i C4_PERM = SIMD_MM512_SETR_EPI32(0x0, 0x0, 0x1, 0x0, 0x2, 0x0, 0x03, 0x0, 0x4, 0x0, 0x5, 0x0, 0x06, 0x0, 0x7, 0x0);

        const __m512i C4_MUSH = SIMD_MM512_SETR_EPI8(
            0x00, 0x04, 0x08, 0x0C, 0x10, 0x14, 0x18, 0x1C, 0x00, 0x04, 0x08, 0x0C, 0x10, 0x14, 0x18, 0x1C,
            0x00, 0x04, 0x08, 0x0C, 0x10, 0x14, 0x18, 0x1C, 0x00, 0x04, 0x08, 0x0C, 0x10, 0x14, 0x18, 0x1C,
            0x00, 0x04, 0x08, 0x0C, 0x10, 0x14, 0x18, 0x1C, 0x00, 0x04, 0x08, 0x0C, 0x10, 0x14, 0x18, 0x1C,
            0x00, 0x04, 0x08, 0x0C, 0x10, 0x14, 0x18, 0x1C, 0x00, 0x04, 0x08, 0x0C, 0x10, 0x14, 0x18, 0x1C);

        const __m512i C4_MASK = SIMD_MM512_SET1_EPI8(0x0F);

        const __m512i C5_PERM = SIMD_MM512_SETR_EPI8(
            0x00, 0x01, 0x02, 0x03, 0x04, 0x00, 0x00, 0x00, 0x05, 0x06, 0x07, 0x08, 0x09, 0x00, 0x00, 0x00,
            0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x00, 0x00, 0x00, 0x0F, 0x10, 0x11, 0x12, 0x13, 0x00, 0x00, 0x00,
            0x14, 0x15, 0x16, 0x17, 0x18, 0x00, 0x00, 0x00, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x00, 0x00, 0x00,
            0x1E, 0x1F, 0x20, 0x21, 0x22, 0x00, 0x00, 0x00, 0x23, 0x24, 0x25, 0x26, 0x27, 0x00, 0x00, 0x00);

        const __m512i C5_MUSH = SIMD_MM512_SETR_EPI8(
            0x00, 0x05, 0x0A, 0x0F, 0x14, 0x19, 0x1E, 0x23, 0x00, 0x05, 0x0A, 0x0F, 0x14, 0x19, 0x1E, 0x23,
            0x00, 0x05, 0x0A, 0x0F, 0x14, 0x19, 0x1E, 0x23, 0x00, 0x05, 0x0A, 0x0F, 0x14, 0x19, 0x1E, 0x23,
            0x00, 0x05, 0x0A, 0x0F, 0x14, 0x19, 0x1E, 0x23, 0x00, 0x05, 0x0A, 0x0F, 0x14, 0x19, 0x1E, 0x23,
            0x00, 0x05, 0x0A, 0x0F, 0x14, 0x19, 0x1E, 0x23, 0x00, 0x05, 0x0A, 0x0F, 0x14, 0x19, 0x1E, 0x23);

        const __m512i C5_MASK = SIMD_MM512_SET1_EPI8(0x1F);

        const __m512i C6_PERM = SIMD_MM512_SETR_EPI8(
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x00, 0x00, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x00, 0x00,
            0x0C, 0x0D, 0x0E, 0x0F, 0x10, 0x11, 0x00, 0x00, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x00, 0x00,
            0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x00, 0x00, 0x1E, 0x1F, 0x20, 0x21, 0x22, 0x23, 0x00, 0x00,
            0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x00, 0x00, 0x2A, 0x2B, 0x2C, 0x2D, 0x2E, 0x2F, 0x00, 0x00);

        const __m512i C6_MUSH = SIMD_MM512_SETR_EPI8(
            0x00, 0x06, 0x0C, 0x12, 0x18, 0x1E, 0x24, 0x2A, 0x00, 0x06, 0x0C, 0x12, 0x18, 0x1E, 0x24, 0x2A,
            0x00, 0x06, 0x0C, 0x12, 0x18, 0x1E, 0x24, 0x2A, 0x00, 0x06, 0x0C, 0x12, 0x18, 0x1E, 0x24, 0x2A,
            0x00, 0x06, 0x0C, 0x12, 0x18, 0x1E, 0x24, 0x2A, 0x00, 0x06, 0x0C, 0x12, 0x18, 0x1E, 0x24, 0x2A,
            0x00, 0x06, 0x0C, 0x12, 0x18, 0x1E, 0x24, 0x2A, 0x00, 0x06, 0x0C, 0x12, 0x18, 0x1E, 0x24, 0x2A);

        const __m512i C6_MASK = SIMD_MM512_SET1_EPI8(0x3F);

        const __m512i C7_PERM = SIMD_MM512_SETR_EPI8(
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x00, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x00,
            0x0E, 0x0F, 0x10, 0x11, 0x12, 0x13, 0x14, 0x00, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x1B, 0x00,
            0x1C, 0x1D, 0x1E, 0x1F, 0x20, 0x21, 0x22, 0x00, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x00,
            0x2A, 0x2B, 0x2C, 0x2D, 0x2E, 0x2F, 0x30, 0x00, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x00);

        const __m512i C7_MUSH = SIMD_MM512_SETR_EPI8(
            0x00, 0x07, 0x0E, 0x15, 0x1C, 0x23, 0x2A, 0x31, 0x00, 0x07, 0x0E, 0x15, 0x1C, 0x23, 0x2A, 0x31,
            0x00, 0x07, 0x0E, 0x15, 0x1C, 0x23, 0x2A, 0x31, 0x00, 0x07, 0x0E, 0x15, 0x1C, 0x23, 0x2A, 0x31,
            0x00, 0x07, 0x0E, 0x15, 0x1C, 0x23, 0x2A, 0x31, 0x00, 0x07, 0x0E, 0x15, 0x1C, 0x23, 0x2A, 0x31,
            0x00, 0x07, 0x0E, 0x15, 0x1C, 0x23, 0x2A, 0x31, 0x00, 0x07, 0x0E, 0x15, 0x1C, 0x23, 0x2A, 0x31);

        const __m512i C7_MASK = SIMD_MM512_SET1_EPI8(0x7F);

        //-------------------------------------------------------------------------------------------------

        template<int bits> __m512i Load8u(const uint8_t* src, __mmask64 mask);

        template<> SIMD_INLINE __m512i Load8u<4>(const uint8_t* src, __mmask64 mask)
        {
            return _mm512_and_si512(C4_MASK, _mm512_multishift_epi64_epi8(C4_MUSH, _mm512_permutexvar_epi32(C4_PERM, _mm512_maskz_loadu_epi8(mask, src))));
        }

        template<> SIMD_INLINE __m512i Load8u<5>(const uint8_t* src, __mmask64 mask)
        {
            return _mm512_and_si512(C5_MASK, _mm512_multishift_epi64_epi8(C5_MUSH, _mm512_permutexvar_epi8(C5_PERM, _mm512_maskz_loadu_epi8(mask, src))));
        }

        template<> SIMD_INLINE __m512i Load8u<6>(const uint8_t* src, __mmask64 mask)
        {
            return _mm512_and_si512(C6_MASK, _mm512_multishift_epi64_epi8(C6_MUSH, _mm512_permutexvar_epi8(C6_PERM, _mm512_maskz_loadu_epi8(mask, src))));
        }

        template<> SIMD_INLINE __m512i Load8u<7>(const uint8_t* src, __mmask64 mask)
        {
            return _mm512_and_si512(C7_MASK, _mm512_multishift_epi64_epi8(C7_MUSH, _mm512_permutexvar_epi8(C7_PERM, _mm512_maskz_loadu_epi8(mask, src))));
        }

        template<> SIMD_INLINE __m512i Load8u<8>(const uint8_t* src, __mmask64 mask)
        {
            return _mm512_maskz_loadu_epi8(mask, src);
        }
    }
#endif

#ifdef SIMD_NEON_ENABLE
    namespace Neon
    {
        const uint16x8_t E4_MULLO = SIMD_VEC_SETR_EPI16(4096, 1, 4096, 1, 4096, 1, 4096, 1);

        const uint16x8_t E5_MULLO = SIMD_VEC_SETR_EPI16(256, 32, 4, 128, 16, 2, 64, 8);
        const uint8x8_t E5_SHFL0 = SIMD_VEC_SETR_PI8(0x1, 0x3, 0x7, 0x9, 0xD, 0, 0, 0);
        const uint8x8_t E5_SHFL1 = SIMD_VEC_SETR_PI8(0x2, 0x4, 0x8, 0xA, 0xE, 0, 0, 0);
        const uint8x8_t E5_SHFL2 = SIMD_VEC_SETR_PI8(0, 0x6, 0, 0xC, 0, 0, 0, 0);

        const uint16x8_t E6_MULLO = SIMD_VEC_SETR_EPI16(256, 64, 16, 4, 256, 64, 16, 4);
        const uint8x8_t E6_SHFL0 = SIMD_VEC_SETR_PI8(0x1, 0x3, 0x5, 0x9, 0xB, 0xD, 0, 0);
        const uint8x8_t E6_SHFL1 = SIMD_VEC_SETR_PI8(0x2, 0x4, 0x6, 0xA, 0xC, 0xE, 0, 0);

        const uint16x8_t E7_MULLO = SIMD_VEC_SETR_EPI16(256, 128, 64, 32, 16, 8, 4, 2);
        const uint8x8_t E7_SHFL0 = SIMD_VEC_SETR_PI8(0x1, 0x3, 0x5, 0x7, 0x9, 0xB, 0xD, 0);
        const uint8x8_t E7_SHFL1 = SIMD_VEC_SETR_PI8(0x2, 0x4, 0x6, 0x8, 0xA, 0xC, 0xE, 0);

        const int32x4_t C4_SHL0 = SIMD_VEC_SETR_EPI32(0, -4, -8, -12);
        const int32x4_t C4_SHL1 = SIMD_VEC_SETR_EPI32(-16, -20, -24, -28);
        const uint32x4_t C4_AND = SIMD_VEC_SET1_EPI32(0x0F);
        const int8x16_t C4_8SHL = SIMD_VEC_SETR_EPI8(0, -4, 0, -4, 0, -4, 0, -4, 0, -4, 0, -4, 0, -4, 0, -4);
        const uint8x16_t C4_8AND = SIMD_VEC_SET1_EPI8(0x0F);

        const int32x4_t C5_SHL0 = SIMD_VEC_SETR_EPI32(0, -5, -10, -15);
        const int32x4_t C5_SHL1 = SIMD_VEC_SETR_EPI32(-12, -17, -22, -27);
        const uint32x4_t C5_AND = SIMD_VEC_SET1_EPI32(0x1F);
        const uint8x8_t C5_TBL0 = SIMD_VEC_SETR_PI8(0x0, 0x0, 0x0, 0x1, 0x1, 0x1, 0x1, 0x2);
        const uint8x8_t C5_TBL1 = SIMD_VEC_SETR_PI8(0x2, 0x3, 0x3, 0x3, 0x3, 0x4, 0x4, 0x4);
        const int16x8_t C5_16SHL = SIMD_VEC_SETR_EPI16(-8, -5, -10, -7, -4, -9, -6, -11);
        const uint16x8_t C5_16AND = SIMD_VEC_SET1_EPI16(0x1F);

        const int32x4_t C6_SHL0 = SIMD_VEC_SETR_EPI32(0, -6, -12, -18);
        const int32x4_t C6_SHL1 = SIMD_VEC_SETR_EPI32(-8, -14, -20, -26);
        const uint32x4_t C6_AND = SIMD_VEC_SET1_EPI32(0x3F);
        const uint8x8_t C6_TBL0 = SIMD_VEC_SETR_PI8(0x0, 0x0, 0x0, 0x1, 0x1, 0x2, 0x2, 0x2);
        const uint8x8_t C6_TBL1 = SIMD_VEC_SETR_PI8(0x3, 0x3, 0x3, 0x4, 0x4, 0x5, 0x5, 0x5);
        const int16x8_t C6_16SHL = SIMD_VEC_SETR_EPI16(-8, -6, -4, -2, -8, -6, -4, -2);
        const uint16x8_t C6_16AND = SIMD_VEC_SET1_EPI16(0x3F);

        const int32x4_t C7_SHL0 = SIMD_VEC_SETR_EPI32(0, -7, -14, -21);
        const int32x4_t C7_SHL1 = SIMD_VEC_SETR_EPI32(-4, -11, -18, -25);
        const uint32x4_t C7_AND = SIMD_VEC_SET1_EPI32(0x7F);
        const uint8x8_t C7_TBL0 = SIMD_VEC_SETR_PI8(0x0, 0x0, 0x0, 0x1, 0x1, 0x2, 0x2, 0x3);
        const uint8x8_t C7_TBL1 = SIMD_VEC_SETR_PI8(0x3, 0x4, 0x4, 0x5, 0x5, 0x6, 0x6, 0x6);
        const int16x8_t C7_16SHL = SIMD_VEC_SETR_EPI16(-8, -7, -6, -5, -4, -3, -2, -1);
        const uint16x8_t C7_16AND = SIMD_VEC_SET1_EPI16(0x7F);

        //-------------------------------------------------------------------------------------------------

        template<int bits> SIMD_INLINE uint8x8_t LoadLast8(const uint8_t* src)
        {
            uint8x8_t val = LoadHalf<false>(src + bits - 8);
            return vext_u8(val, val, 8 - bits);
        }

        template<int bits> SIMD_INLINE uint8x16_t LoadLast16(const uint8_t* src)
        {
            uint8x16_t val = Load<false>(src + bits * 2 - 16);
            return vextq_u8(val, val, 16 - bits * 2);
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE uint8x16_t Cvt4To8(uint8x8_t a)
        {
            uint8x8x2_t aa = vzip_u8(a, a);
            return vandq_u8(vshlq_u8(*(uint8x16_t*)&aa, C4_8SHL), C4_8AND);
        }

        SIMD_INLINE uint16x8_t Cvt5To16(uint8x8_t src)
        {
            return vandq_u16(vshlq_u16((uint16x8_t)Shuffle(src, C5_TBL0, C5_TBL1), C5_16SHL), C5_16AND);
        }

        SIMD_INLINE uint8x8_t Cvt5To8(uint8x8_t src)
        {
            return vmovn_u16(Cvt5To16(src));
        }

        SIMD_INLINE uint16x8_t Cvt6To16(uint8x8_t src)
        {
            return vandq_u16(vshlq_u16((uint16x8_t)Shuffle(src, C6_TBL0, C6_TBL1), C6_16SHL), C6_16AND);
        }

        SIMD_INLINE uint8x8_t Cvt6To8(uint8x8_t src)
        {
            return vmovn_u16(Cvt6To16(src));
        }

        SIMD_INLINE uint16x8_t Cvt7To16(uint8x8_t src)
        {
            return vandq_u16(vshlq_u16((uint16x8_t)Shuffle(src, C7_TBL0, C7_TBL1), C7_16SHL), C7_16AND);
        }

        SIMD_INLINE uint8x8_t Cvt7To8(uint8x8_t src)
        {
            return vmovn_u16(Cvt7To16(src));
        }

        template<int bits> SIMD_INLINE uint16x8_t CvtTo16(uint8x8_t src);

        template<> SIMD_INLINE uint16x8_t CvtTo16<5>(uint8x8_t src)
        {
            return vandq_u16(vshlq_u16((uint16x8_t)Shuffle(src, C5_TBL0, C5_TBL1), C5_16SHL), C5_16AND);
        }

        template<> SIMD_INLINE uint16x8_t CvtTo16<6>(uint8x8_t src)
        {
            return vandq_u16(vshlq_u16((uint16x8_t)Shuffle(src, C6_TBL0, C6_TBL1), C6_16SHL), C6_16AND);
        }

        template<> SIMD_INLINE uint16x8_t CvtTo16<7>(uint8x8_t src)
        {
            return vandq_u16(vshlq_u16((uint16x8_t)Shuffle(src, C7_TBL0, C7_TBL1), C7_16SHL), C7_16AND);
        }
        
        template<int bits> SIMD_INLINE uint8x8_t CvtTo8(uint8x8_t src)
        {
            return vmovn_u16(CvtTo16<bits>(src));
        }

        template<> SIMD_INLINE uint8x8_t CvtTo8<4>(uint8x8_t src)
        {
            return Half<0>(Cvt4To8(src));
        }

        template<> SIMD_INLINE uint8x8_t CvtTo8<8>(uint8x8_t src)
        {
            return src;
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE void DecodeCosineDistances1x4(const uint8_t* a, const uint8_t* const* B, float32x4_t abSum, float* distances)
        {
            float32x4x2_t a0, a1, b0, b1;
            b0.val[0] = Load<false>((float*)B[0]);
            b0.val[1] = Load<false>((float*)B[1]);
            b1.val[0] = Load<false>((float*)B[2]);
            b1.val[1] = Load<false>((float*)B[3]);
            a0 = vzipq_f32(b0.val[0], b1.val[0]);
            a1 = vzipq_f32(b0.val[1], b1.val[1]);
            b0 = vzipq_f32(a0.val[0], a1.val[0]);
            b1 = vzipq_f32(a0.val[1], a1.val[1]);
            a0.val[0] = vdupq_n_f32(((float*)a)[0]);
            a0.val[1] = vdupq_n_f32(((float*)a)[1]);
            a1.val[0] = vdupq_n_f32(((float*)a)[2]);
            a1.val[1] = vdupq_n_f32(((float*)a)[3]);
            float32x4_t ab = vmulq_f32(abSum, vmulq_f32(a0.val[0], b0.val[0]));
            ab = vmlaq_f32(ab, a1.val[0], b0.val[1]);
            ab = vmlaq_f32(ab, b1.val[0], a0.val[1]);
            Store<false>(distances, vminq_f32(vmaxq_f32(vsubq_f32(vdupq_n_f32(1.0f), Div<2>(ab, vmulq_f32(a1.val[1], b1.val[1]))), vdupq_n_f32(0.0f)), vdupq_n_f32(2.0f)));
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE void DecodeCosineDistances1x4(const float* a, const float* b, size_t stride, uint32x4_t abSum, float* distances)
        {
            float32x4_t aScale = vdupq_n_f32(a[0]);
            float32x4_t aShift = vdupq_n_f32(a[1]);
            float32x4_t aMean = vdupq_n_f32(a[2]);
            float32x4_t aNorm = vdupq_n_f32(a[3]);
            float32x4_t bScale = Load<false>(b + 0 * stride);
            float32x4_t bShift = Load<false>(b + 1 * stride);
            float32x4_t bMean = Load<false>(b + 2 * stride);
            float32x4_t bNorm = Load<false>(b + 3 * stride);
            float32x4_t ab = vmulq_f32(vcvtq_f32_u32(abSum), vmulq_f32(aScale, bScale));
            ab = vmlaq_f32(ab, aMean, bShift);
            ab = vmlaq_f32(ab, bMean, aShift);
            Store<false>(distances, vminq_f32(vmaxq_f32(vsubq_f32(vdupq_n_f32(1.0f), Div<2>(ab, vmulq_f32(aNorm, bNorm))), vdupq_n_f32(0.0f)), vdupq_n_f32(2.0f)));
        }

        SIMD_INLINE void DecodeCosineDistances1x4(const float* a, const float* b, size_t stride, uint32x4_t abSum, float* distances, size_t N)
        {
            float d[F];
            DecodeCosineDistances1x4(a, b, stride, abSum, d);
            for (size_t i = 0; i < N; ++i)
                distances[i] = d[i];
        }

        SIMD_INLINE void DecodeCosineDistances1x8(const float* a, const float* b, size_t stride, uint32x4_t ab0, uint32x4_t ab1, float* distances)
        {
            DecodeCosineDistances1x4(a, b + 0 * 4, stride, ab0, distances + 0 * 4);
            DecodeCosineDistances1x4(a, b + 1 * 4, stride, ab1, distances + 1 * 4);
        }

        SIMD_INLINE void DecodeCosineDistances1x8(const float* a, const float* b, size_t stride, uint32x4_t ab0, uint32x4_t ab1, float* distances, size_t N)
        {
            DecodeCosineDistances1x4(a, b + 0 * 4, stride, ab0, distances + 0 * 4);
            DecodeCosineDistances1x4(a, b + 1 * 4, stride, ab1, distances + 1 * 4, N - 4);
        }

        SIMD_INLINE void DecodeCosineDistances1x12(const float* a, const float* b, size_t stride, uint32x4_t ab0, uint32x4_t ab1, uint32x4_t ab2, float* distances)
        {
            DecodeCosineDistances1x4(a, b + 0 * 4, stride, ab0, distances + 0 * 4);
            DecodeCosineDistances1x4(a, b + 1 * 4, stride, ab1, distances + 1 * 4);
            DecodeCosineDistances1x4(a, b + 2 * 4, stride, ab2, distances + 2 * 4);
        }

        SIMD_INLINE void DecodeCosineDistances1x12(const float* a, const float* b, size_t stride, uint32x4_t ab0, uint32x4_t ab1, uint32x4_t ab2, float* distances, size_t N)
        {
            DecodeCosineDistances1x4(a, b + 0 * 4, stride, ab0, distances + 0 * 4);
            DecodeCosineDistances1x4(a, b + 1 * 4, stride, ab1, distances + 1 * 4);
            DecodeCosineDistances1x4(a, b + 2 * 4, stride, ab2, distances + 2 * 4, N - 8);
        }

        SIMD_INLINE void DecodeCosineDistances1x16(const float* a, const float* b, size_t stride, uint32x4_t ab0, uint32x4_t ab1, uint32x4_t ab2, uint32x4_t ab3, float* distances)
        {
            DecodeCosineDistances1x4(a, b + 0 * 4, stride, ab0, distances + 0 * 4);
            DecodeCosineDistances1x4(a, b + 1 * 4, stride, ab1, distances + 1 * 4);
            DecodeCosineDistances1x4(a, b + 2 * 4, stride, ab2, distances + 2 * 4);
            DecodeCosineDistances1x4(a, b + 3 * 4, stride, ab3, distances + 3 * 4);
        }

        SIMD_INLINE void DecodeCosineDistances1x16(const float* a, const float* b, size_t stride, uint32x4_t ab0, uint32x4_t ab1, uint32x4_t ab2, uint32x4_t ab3, float* distances, size_t N)
        {
            DecodeCosineDistances1x4(a, b + 0 * 4, stride, ab0, distances + 0 * 4);
            DecodeCosineDistances1x4(a, b + 1 * 4, stride, ab1, distances + 1 * 4);
            DecodeCosineDistances1x4(a, b + 2 * 4, stride, ab2, distances + 2 * 4);
            DecodeCosineDistances1x4(a, b + 3 * 4, stride, ab3, distances + 3 * 4, N - 12);
        }
    }
#endif
}
#endif//__SimdDescrIntCommon_h__
