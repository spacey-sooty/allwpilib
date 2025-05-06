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
#include "Simd/SimdSynetInnerProduct16b.h"
#include "Simd/SimdSynetConvolution16bCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdAvx2.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdBFloat16.h"
#include "Simd/SimdCopy.h"

namespace Simd
{
#if defined(SIMD_AVX2_ENABLE) && defined(SIMD_SYNET_ENABLE)      
    namespace Avx2
    {
        typedef Base::SynetInnerProduct16bGemmNN::AlgParam AlgParam;
        typedef Base::SynetInnerProduct16bGemmNN::GemmPtr GemmPtr;

        //-----------------------------------------------------------------------------------------

        static void InnerProduct16bGemmNN_ConvertA(const uint8_t* src8, const InnerProductParam16b& p, const AlgParam& a, size_t M, size_t K, uint16_t* dst)
        {
            const float* src = (float*)src8;
            if (p.K == a.aK)
            {
                Float32ToBFloat16(src, K * M, dst);
            }
            else
            {
                size_t KDF = Simd::AlignLo(p.K, DF);
                size_t KF = Simd::AlignLo(p.K, F);
                for (size_t i = 0; i < M; ++i)
                {
                    size_t k = 0;
                    for (; k < KDF; k += DF)
                    {
                        __m256i d0 = Float32ToBFloat16(_mm256_loadu_ps(src + k + 0));
                        __m256i d1 = Float32ToBFloat16(_mm256_loadu_ps(src + k + F));
                        _mm256_storeu_si256((__m256i*)(dst + k), _mm256_permute4x64_epi64(_mm256_packus_epi32(d0, d1), 0xD8));
                    }
                    for (; k < KF; k += F)
                    {
                        __m128i d0 = Sse41::Float32ToBFloat16(_mm_loadu_ps(src + k + 0));
                        __m128i d1 = Sse41::Float32ToBFloat16(_mm_loadu_ps(src + k + 4));
                        _mm_storeu_si128((__m128i*)(dst + k), _mm_packus_epi32(d0, d1));
                    }
                    for (; k < p.K; ++k)
                        dst[k] = Base::Float32ToBFloat16(src[k]);
                    for (; k < a.aK; ++k)
                        dst[k] = 0;
                    src += p.K;
                    dst += a.aK;
                }
            }
        }

        //-----------------------------------------------------------------------------------------

        static void InnerProduct16bGemmNN_ReorderA(const uint8_t* src8, const InnerProductParam16b& p, const AlgParam& a, size_t M, size_t K, uint16_t* dst)
        {
            const uint16_t* src = (uint16_t*)src8;
            size_t KDF = Simd::AlignLo(p.K, DF);
            for (size_t i = 0; i < M; ++i)
            {
                size_t k = 0;
                for (; k < KDF; k += DF)
                    Copy(src + k, dst + k);
                for (; k < p.K; ++k)
                    dst[k] = src[k];
                for (; k < a.aK; ++k)
                    dst[k] = 0;
                src += p.K;
                dst += a.aK;
            }
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE void ConvertBn(const float* src, size_t stride, uint16_t* dst)
        {
            __m256i d0 = _mm256_srli_epi32(_mm256_add_epi32(_mm256_castps_si256(_mm256_loadu_ps(src + 0 * stride)), Bf16::ROUND), Base::Bf16::SHIFT);
            __m256i d1 = _mm256_and_si256(_mm256_add_epi32(_mm256_castps_si256(_mm256_loadu_ps(src + 1 * stride)), Bf16::ROUND), Bf16::MASK);
            _mm256_storeu_si256((__m256i*)dst, _mm256_or_si256(d0, d1));
        }

        static void InnerProduct16bGemmNN_ConvertBn(const uint8_t* src8, const InnerProductParam16b& p, const AlgParam& a, size_t N, size_t K, uint16_t* dst)
        {
            const float* src = (float*)src8;
            size_t Kl = AlignLo(K, a.microK), Kh = AlignHi(K, a.microK), Nf = AlignLo(N, a.F), j = 0, gap = (a.bK - Kh) * a.F;
            for (; j < Nf; j += a.F)
            {
                size_t k = 0;
                for (; k < Kl; k += 2)
                {
                    const float* ps = src + k * p.N + j;
                    for (size_t f = 0; f < a.F; f += F, dst += DF)
                        ConvertBn(ps + f, p.N, dst);
                }
                for (; k < Kh; k += 2)
                {
                    const float* ps = src + k * p.N + j;
                    for (size_t f = 0; f < a.F; ++f)
                    {
                        for (size_t i = 0; i < 2; ++i)
                        {
                            if (j + f < N && k + i < K)
                                *(dst++) = Base::Float32ToBFloat16(ps[i * p.N + f]);
                            else
                                *(dst++) = 0;
                        }
                    }
                }
                dst += gap;
            }
            for (; j < N; j += a.F)
            {
                for (size_t k = 0; k < Kh; k += 2)
                {
                    const float* ps = src + k * p.N + j;
                    for (size_t f = 0; f < a.F; ++f)
                    {
                        for (size_t i = 0; i < 2; ++i)
                        {
                            if (j + f < N && k + i < K)
                                *(dst++) = Base::Float32ToBFloat16(ps[i * p.N + f]);
                            else
                                *(dst++) = 0;
                        }
                    }
                }
            }
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE void ConvertBt(const float* src, size_t stride, uint16_t* dst)
        {
            __m256i d01 = Float32ToBFloat16(Load(src + 0 * stride, src + 1 * stride, src + 4 * stride, src + 5 * stride));
            __m256i d23 = Float32ToBFloat16(Load(src + 2 * stride, src + 3 * stride, src + 6 * stride, src + 7 * stride));
            _mm256_storeu_si256((__m256i*)dst, _mm256_packus_epi32(d01, d23));
        }

        static void InnerProduct16bGemmNN_ConvertBt(const uint8_t* src8, const InnerProductParam16b& p, const AlgParam& a, size_t N, size_t K, uint16_t* dst)
        {
            const float* src = (float*)src8;
            size_t Kl = AlignLo(K, a.microK), Kh = AlignHi(K, a.microK), Nf = AlignLo(N, a.F), j = 0, gap = (a.bK - Kh) * a.F;
            for (; j < Nf; j += a.F)
            {
                size_t k = 0;
                for (; k < Kl; k += 2)
                {
                    const float* ps = src + j * p.K + k;
                    for (size_t f = 0; f < a.F; f += F, dst += DF)
                        ConvertBt(ps + f * p.K, p.K, dst);
                }
                for (; k < Kh; k += 2)
                {
                    const float* ps = src + j * p.K + k;
                    for (size_t f = 0; f < a.F; ++f)
                    {
                        for (size_t i = 0; i < 2; ++i)
                        {
                            if (j + f < N && k + i < K)
                                *(dst++) = Base::Float32ToBFloat16(ps[f * p.K + i]);
                            else
                                *(dst++) = 0;
                        }
                    }
                }
                dst += gap;
            }
            for (; j < N; j += a.F)
            {
                for (size_t k = 0; k < Kh; k += 2)
                {
                    const float* ps = src + j * p.K + k;
                    for (size_t f = 0; f < a.F; ++f)
                    {
                        for (size_t i = 0; i < 2; ++i)
                        {
                            if (j + f < N && k + i < K)
                                *(dst++) = Base::Float32ToBFloat16(ps[f * p.K + i]);
                            else
                                *(dst++) = 0;
                        }
                    }
                }
            }
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE void ReorderBn(const uint16_t* src, size_t stride, uint16_t* dst)
        {
            __m128i d0 = _mm_loadu_si128((__m128i*)(src + 0 * stride));
            __m128i d1 = _mm_loadu_si128((__m128i*)(src + 1 * stride));
            _mm_storeu_si128((__m128i*)dst + 0, _mm_unpacklo_epi16(d0, d1));
            _mm_storeu_si128((__m128i*)dst + 1, _mm_unpackhi_epi16(d0, d1));
        }

        static void InnerProduct16bGemmNN_ReorderBn(const uint8_t* src8, const InnerProductParam16b& p, const AlgParam& a, size_t N, size_t K, uint16_t* dst)
        {
            const uint16_t* src = (uint16_t*)src8;
            size_t Kl = AlignLo(K, a.microK), Kh = AlignHi(K, a.microK), Nf = AlignLo(N, a.F), j = 0, gap = (a.bK - Kh) * a.F;
            for (; j < Nf; j += a.F)
            {
                size_t k = 0;
                for (; k < Kl; k += 2)
                {
                    const uint16_t* ps = src + k * p.N + j;
                    for (size_t f = 0; f < a.F; f += F, dst += DF)
                        ReorderBn(ps + f, p.N, dst);
                }
                for (; k < Kh; k += 2)
                {
                    const uint16_t* ps = src + k * p.N + j;
                    for (size_t f = 0; f < a.F; ++f)
                    {
                        for (size_t i = 0; i < 2; ++i)
                        {
                            if (j + f < N && k + i < K)
                                *(dst++) = ps[i * p.N + f];
                            else
                                *(dst++) = 0;
                        }
                    }
                }
                dst += gap;
            }
            for (; j < N; j += a.F)
            {
                for (size_t k = 0; k < Kh; k += 2)
                {
                    const uint16_t* ps = src + k * p.N + j;
                    for (size_t f = 0; f < a.F; ++f)
                    {
                        for (size_t i = 0; i < 2; ++i)
                        {
                            if (j + f < N && k + i < K)
                                *(dst++) = ps[i * p.N + f];
                            else
                                *(dst++) = 0;
                        }
                    }
                }
            }
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE void ReorderBt(const uint16_t* src, size_t stride, uint16_t* dst)
        {
            *(uint32_t*)(dst + 0x0) = *(uint32_t*)(src + 0 * stride);
            *(uint32_t*)(dst + 0x2) = *(uint32_t*)(src + 1 * stride);
            *(uint32_t*)(dst + 0x4) = *(uint32_t*)(src + 2 * stride);
            *(uint32_t*)(dst + 0x6) = *(uint32_t*)(src + 3 * stride);
            *(uint32_t*)(dst + 0x8) = *(uint32_t*)(src + 4 * stride);
            *(uint32_t*)(dst + 0xa) = *(uint32_t*)(src + 5 * stride);
            *(uint32_t*)(dst + 0xc) = *(uint32_t*)(src + 6 * stride);
            *(uint32_t*)(dst + 0xe) = *(uint32_t*)(src + 7 * stride);
        }

        static void InnerProduct16bGemmNN_ReorderBt(const uint8_t* src8, const InnerProductParam16b& p, const AlgParam& a, size_t N, size_t K, uint16_t* dst)
        {
            const uint16_t* src = (uint16_t*)src8;
            size_t Kl = AlignLo(K, a.microK), Kh = AlignHi(K, a.microK), Nf = AlignLo(N, a.F), j = 0, gap = (a.bK - Kh) * a.F;
            for (; j < Nf; j += a.F)
            {
                size_t k = 0;
                for (; k < Kl; k += 2)
                {
                    const uint16_t* ps = src + j * p.K + k;
                    for (size_t f = 0; f < a.F; f += F, dst += DF)
                        ReorderBt(ps + f * p.K, p.K, dst);
                }
                for (; k < Kh; k += 2)
                {
                    const uint16_t* ps = src + j * p.K + k;
                    for (size_t f = 0; f < a.F; ++f)
                    {
                        for (size_t i = 0; i < 2; ++i)
                        {
                            if (j + f < N && k + i < K)
                                *(dst++) = ps[f * p.K + i];
                            else
                                *(dst++) = 0;
                        }
                    }
                }
                dst += gap;
            }
            for (; j < N; j += a.F)
            {
                for (size_t k = 0; k < Kh; k += 2)
                {
                    const uint16_t* ps = src + j * p.K + k;
                    for (size_t f = 0; f < a.F; ++f)
                    {
                        for (size_t i = 0; i < 2; ++i)
                        {
                            if (j + f < N && k + i < K)
                                *(dst++) = ps[f * p.K + i];
                            else
                                *(dst++) = 0;
                        }
                    }
                }
            }
        }

        //-----------------------------------------------------------------------------------------

        template<Term16bType term, int M> void InnerProduct16bGemmNN_2xM(const uint16_t* A0, const InnerProductParam16b& p, const AlgParam& a,
            size_t N, size_t K, int update, const uint16_t* B0, float* C, const __m256* bias, uint8_t* dst)
        {
            __m256 c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, a0, b00, b01, b10, b11, m = _mm256_castsi256_ps(Bf16::MASK);
            size_t dC = a.cN, dA = a.aK, dD = p.N * a.eC;
            const uint16_t* B1 = B0 + a.bK * F;
            const uint16_t* A1 = A0 + 1 * dA;
            const uint16_t* A2 = A0 + 2 * dA;
            const uint16_t* A3 = A0 + 3 * dA;
            const uint16_t* A4 = A0 + 4 * dA;
            if (N > F)
            {
                if (update)
                {
                    if (M > 0) c00 = _mm256_loadu_ps(C + 0 * dC + 0), c01 = _mm256_loadu_ps(C + 0 * dC + F);
                    if (M > 1) c10 = _mm256_loadu_ps(C + 1 * dC + 0), c11 = _mm256_loadu_ps(C + 1 * dC + F);
                    if (M > 2) c20 = _mm256_loadu_ps(C + 2 * dC + 0), c21 = _mm256_loadu_ps(C + 2 * dC + F);
                    if (M > 3) c30 = _mm256_loadu_ps(C + 3 * dC + 0), c31 = _mm256_loadu_ps(C + 3 * dC + F);
                    if (M > 4) c40 = _mm256_loadu_ps(C + 4 * dC + 0), c41 = _mm256_loadu_ps(C + 4 * dC + F);
                }                
                else
                {
                    if (M > 0) c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();
                    if (M > 1) c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();
                    if (M > 2) c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps();
                    if (M > 3) c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps();
                    if (M > 4) c40 = _mm256_setzero_ps(), c41 = _mm256_setzero_ps();
                }
                for (size_t k = 0; k < K; k += 2)
                {
                    b01 = _mm256_loadu_ps((float*)B0);
                    b00 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(b01), Base::Bf16::SHIFT));
                    b01 = _mm256_and_ps(b01, m);
                    b11 = _mm256_loadu_ps((float*)B1);
                    b10 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(b11), Base::Bf16::SHIFT));
                    b11 = _mm256_and_ps(b11, m);
                    if (M > 0)
                    {
                        a0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(A0 + k - 1)), m);
                        c00 = _mm256_fmadd_ps(a0, b00, c00);
                        c01 = _mm256_fmadd_ps(a0, b10, c01);
                        a0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(A0 + k - 0)), m);
                        c00 = _mm256_fmadd_ps(a0, b01, c00);
                        c01 = _mm256_fmadd_ps(a0, b11, c01);
                    }
                    if (M > 1)
                    {
                        a0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(A1 + k - 1)), m);
                        c10 = _mm256_fmadd_ps(a0, b00, c10);
                        c11 = _mm256_fmadd_ps(a0, b10, c11);
                        a0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(A1 + k - 0)), m);
                        c10 = _mm256_fmadd_ps(a0, b01, c10);
                        c11 = _mm256_fmadd_ps(a0, b11, c11);
                    }
                    if (M > 2)
                    {
                        a0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(A2 + k - 1)), m);
                        c20 = _mm256_fmadd_ps(a0, b00, c20);
                        c21 = _mm256_fmadd_ps(a0, b10, c21);
                        a0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(A2 + k - 0)), m);
                        c20 = _mm256_fmadd_ps(a0, b01, c20);
                        c21 = _mm256_fmadd_ps(a0, b11, c21);
                    }
                    if (M > 3)
                    {
                        a0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(A3 + k - 1)), m);
                        c30 = _mm256_fmadd_ps(a0, b00, c30);
                        c31 = _mm256_fmadd_ps(a0, b10, c31);
                        a0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(A3 + k - 0)), m);
                        c30 = _mm256_fmadd_ps(a0, b01, c30);
                        c31 = _mm256_fmadd_ps(a0, b11, c31);
                    }
                    if (M > 4)
                    {
                        a0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(A4 + k - 1)), m);
                        c40 = _mm256_fmadd_ps(a0, b00, c40);
                        c41 = _mm256_fmadd_ps(a0, b10, c41);
                        a0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(A4 + k - 0)), m);
                        c40 = _mm256_fmadd_ps(a0, b01, c40);
                        c41 = _mm256_fmadd_ps(a0, b11, c41);
                    }
                    B0 += DF;
                    B1 += DF;
                }
                if (N == DF)
                {
                    if (M > 0) Save2<term>(dst, C, c00, c01, bias), C += dC, dst += dD;
                    if (M > 1) Save2<term>(dst, C, c10, c11, bias), C += dC, dst += dD;
                    if (M > 2) Save2<term>(dst, C, c20, c21, bias), C += dC, dst += dD;
                    if (M > 3) Save2<term>(dst, C, c30, c31, bias), C += dC, dst += dD;
                    if (M > 4) Save2<term>(dst, C, c40, c41, bias), C += dC, dst += dD;
                }
                else
                {
                    size_t tail = N - F;
                    if (M > 0) Save2<term>(dst, C, c00, c01, bias, tail), C += dC, dst += dD;
                    if (M > 1) Save2<term>(dst, C, c10, c11, bias, tail), C += dC, dst += dD;
                    if (M > 2) Save2<term>(dst, C, c20, c21, bias, tail), C += dC, dst += dD;
                    if (M > 3) Save2<term>(dst, C, c30, c31, bias, tail), C += dC, dst += dD;
                    if (M > 4) Save2<term>(dst, C, c40, c41, bias, tail), C += dC, dst += dD;
                }
            }
            else
            {
                if (update)
                {
                    if (M > 0) c00 = _mm256_loadu_ps(C + 0 * dC + 0);
                    if (M > 1) c10 = _mm256_loadu_ps(C + 1 * dC + 0);
                    if (M > 2) c20 = _mm256_loadu_ps(C + 2 * dC + 0);
                    if (M > 3) c30 = _mm256_loadu_ps(C + 3 * dC + 0);
                    if (M > 4) c40 = _mm256_loadu_ps(C + 4 * dC + 0);
                }
                else
                {
                    if (M > 0) c00 = _mm256_setzero_ps();
                    if (M > 1) c10 = _mm256_setzero_ps();
                    if (M > 2) c20 = _mm256_setzero_ps();
                    if (M > 3) c30 = _mm256_setzero_ps();
                    if (M > 4) c40 = _mm256_setzero_ps();
                }
                for (size_t k = 0; k < K; k += 2)
                {
                    b01 = _mm256_loadu_ps((float*)B0);
                    b00 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(b01), Base::Bf16::SHIFT));
                    b01 = _mm256_and_ps(b01, m);
                    if (M > 0)
                    {
                        a0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(A0 + k - 1)), m);
                        c00 = _mm256_fmadd_ps(a0, b00, c00);
                        a0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(A0 + k - 0)), m);
                        c00 = _mm256_fmadd_ps(a0, b01, c00);
                    }
                    if (M > 1)
                    {
                        a0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(A1 + k - 1)), m);
                        c10 = _mm256_fmadd_ps(a0, b00, c10);
                        a0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(A1 + k - 0)), m);
                        c10 = _mm256_fmadd_ps(a0, b01, c10);
                    }
                    if (M > 2)
                    {
                        a0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(A2 + k - 1)), m);
                        c20 = _mm256_fmadd_ps(a0, b00, c20);
                        a0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(A2 + k - 0)), m);
                        c20 = _mm256_fmadd_ps(a0, b01, c20);
                    }
                    if (M > 3)
                    {
                        a0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(A3 + k - 1)), m);
                        c30 = _mm256_fmadd_ps(a0, b00, c30);
                        a0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(A3 + k - 0)), m);
                        c30 = _mm256_fmadd_ps(a0, b01, c30);
                    }
                    if (M > 4)
                    {
                        a0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(A4 + k - 1)), m);
                        c40 = _mm256_fmadd_ps(a0, b00, c40);
                        a0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(A4 + k - 0)), m);
                        c40 = _mm256_fmadd_ps(a0, b01, c40);
                    }
                    B0 += DF;
                }
                if (N == F)
                {
                    if (M > 0) Save1<term>(dst, C, c00, bias), C += dC, dst += dD;
                    if (M > 1) Save1<term>(dst, C, c10, bias), C += dC, dst += dD;
                    if (M > 2) Save1<term>(dst, C, c20, bias), C += dC, dst += dD;
                    if (M > 3) Save1<term>(dst, C, c30, bias), C += dC, dst += dD;
                    if (M > 4) Save1<term>(dst, C, c40, bias), C += dC, dst += dD;
                }
                else
                {
                    size_t tail = N;
                    if (M > 0) Save1<term>(dst, C, c00, bias, tail), C += dC, dst += dD;
                    if (M > 1) Save1<term>(dst, C, c10, bias, tail), C += dC, dst += dD;
                    if (M > 2) Save1<term>(dst, C, c20, bias, tail), C += dC, dst += dD;
                    if (M > 3) Save1<term>(dst, C, c30, bias, tail), C += dC, dst += dD;
                    if (M > 4) Save1<term>(dst, C, c40, bias, tail), C += dC, dst += dD;
                }
            }
        }

        typedef void(*GemmNN_2xM_Ptr)(const uint16_t* A0, const InnerProductParam16b& p, const AlgParam& a, size_t N, size_t K, int update, const uint16_t* B0, float* C, const __m256* bias, uint8_t* dst);

        template<Term16bType term> GemmNN_2xM_Ptr GetGemmNN_2xM(size_t M)
        {
            switch (M)
            {
            case 0: return NULL;
            case 1: return InnerProduct16bGemmNN_2xM<term, 1>;
            case 2: return InnerProduct16bGemmNN_2xM<term, 2>;
            case 3: return InnerProduct16bGemmNN_2xM<term, 3>;
            case 4: return InnerProduct16bGemmNN_2xM<term, 4>;
            case 5: return InnerProduct16bGemmNN_2xM<term, 5>;
            }
            assert(0);
            return NULL;
        }

        template<Term16bType term> void InnerProduct16bGemmNN_Gemm2(const uint16_t* A, const InnerProductParam16b& p, const AlgParam& a,
            size_t M, size_t N, size_t K, int update, const uint16_t* B, float* C, int post, const float* bias, uint8_t* dst)
        {
            size_t m1 = M, m = 5;
            size_t mm = AlignLoAny(m1, m), t = m1 - mm;
            size_t dA = a.aK, dB = a.bK * DF, dC = a.cN, dD = p.N * a.eC;
            GemmNN_2xM_Ptr gemm_2xM = post ? GetGemmNN_2xM<term>(m) : GetGemmNN_2xM<Term16bInterim>(m);
            GemmNN_2xM_Ptr gemm_2xT = post ? GetGemmNN_2xM<term>(t) : GetGemmNN_2xM<Term16bInterim>(t);

            __m256 _bias[2];
            for (size_t j = 0; j < N; j += DF)
            {
                size_t dN = Simd::Min(DF, N - j);
                _bias[0] = _mm256_loadu_ps(bias + j + 0);
                _bias[1] = _mm256_loadu_ps(bias + j + F);
                size_t i = 0;
                for (; i < mm; i += m)
                    gemm_2xM(A + i * dA, p, a, dN, K, update, B, C + i * dC, _bias, dst + i * dD);
                for (; i < m1; i += t)
                    gemm_2xT(A + i * dA, p, a, dN, K, update, B, C + i * dC, _bias, dst + i * dD);
                B += dB;
                C += dN;
                dst += DF * a.eC;
            }
        }

        //-------------------------------------------------------------------------------------------------

        SynetInnerProduct16bGemmNN::SynetInnerProduct16bGemmNN(const InnerProductParam16b& p)
            : Sse41::SynetInnerProduct16bGemmNN(p)
        {
            SetAlgParam(F, 5, F * 2, 2, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3());
            if (_sizeA)
            {
                if (p.typeA == SimdTensorData16b)
                    _prepA = InnerProduct16bGemmNN_ReorderA;
                else
                    _prepA = InnerProduct16bGemmNN_ConvertA;
            }
            if (p.typeB == SimdTensorData32f || p.constB)
            {
                if (p.transB)
                    _prepB = InnerProduct16bGemmNN_ConvertBt;
                else
                    _prepB = InnerProduct16bGemmNN_ConvertBn;
            }
            else
            {
                if (p.transB)
                    _prepB = InnerProduct16bGemmNN_ReorderBt;
                else
                    _prepB = InnerProduct16bGemmNN_ReorderBn;
            }
            if (p.typeC == SimdTensorData16b)
                _gemm = InnerProduct16bGemmNN_Gemm2<Term16bLast16b>;
            else
                _gemm = InnerProduct16bGemmNN_Gemm2<Term16bLast32f>;
        }
    }
#endif
}
