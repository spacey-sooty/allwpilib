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
#include "Simd/SimdSynetConvolution32f.h"
#include "Simd/SimdSynetConvolution32fCommon.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdPrefetch.h"

namespace Simd
{
#if defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_SYNET_ENABLE)  
    namespace Avx512bw
    {
        template<TermType term, SimdConvolutionActivationType type> void ConvolutionNhwcDirect_2x1(const float* src0, const ConvParam& p,
            size_t dy, size_t dx, size_t srcC, const float* weight0, const __m512* bias, const __m512* params, float* dst, const __mmask16* tails, int first)
        {
            __m512 d00, d01, s0, w0, w1;
            size_t srcH = p.srcH, srcW = p.srcW, dilY = p.dilationY, dilX = p.dilationX;
            size_t dY = p.srcW * p.srcC, dX = p.srcC, dS = p.srcC * p.strideX, dW = srcC * DF;
            size_t sy = dy * p.strideY - p.padY, sx = dx * p.strideX - p.padX;
            size_t kY = p.kernelY * p.dilationY, kX = p.kernelX * p.dilationX;
            const float* weight1 = weight0 + F;
            if (tails[1])
            {
                if (first)
                    d00 = _mm512_setzero_ps(), d01 = _mm512_setzero_ps();
                else
                    d00 = _mm512_loadu_ps(dst + 0), d01 = _mm512_maskz_loadu_ps(tails[1], dst + F);
                for (size_t ky = 0; ky < kY; ky += dilY)
                {
                    size_t beg = (sy + ky) * dY + sx * dX;
                    for (size_t kx = 0; kx < kX; kx += dilX)
                    {
                        if (sy + ky < srcH && sx + kx < srcW)
                        {
                            size_t offs = beg + kx * dX, end = offs + srcC, offw = 0;
                            for (; offs < end; ++offs, offw += DF)
                            {
                                w0 = _mm512_loadu_ps(weight0 + offw);
                                w1 = _mm512_loadu_ps(weight1 + offw);
                                s0 = _mm512_set1_ps(src0[offs]), d00 = _mm512_fmadd_ps(s0, w0, d00), d01 = _mm512_fmadd_ps(s0, w1, d01);
                            }
                        }
                        weight0 += dW, weight1 += dW;
                    }
                }
                Save2<term, type>(dst, d00, d01, bias, params, tails);
            }
            else
            {
                if (first)
                    d00 = _mm512_setzero_ps();
                else
                    d00 = _mm512_maskz_loadu_ps(tails[0], dst + 0);
                for (size_t ky = 0; ky < kY; ky += dilY)
                {
                    size_t beg = (sy + ky) * dY + sx * dX;
                    for (size_t kx = 0; kx < kX; kx += dilX)
                    {
                        if (sy + ky < srcH && sx + kx < srcW)
                        {
                            size_t offs = beg + kx * dX, end = offs + srcC, offw = 0;
                            for (; offs < end; ++offs, offw += DF)
                            {
                                w0 = _mm512_loadu_ps(weight0 + offw);
                                s0 = _mm512_set1_ps(src0[offs]), d00 = _mm512_fmadd_ps(s0, w0, d00);
                            }
                        }
                        weight0 += dW;
                    }
                }
                Save1<term, type>(dst, d00, bias, params, tails);
            }
        }

        template<TermType term, SimdConvolutionActivationType type, int M> void ConvolutionNhwcDirect_2xM(const float* src0, const ConvParam& p,
            size_t dy, size_t dx, size_t srcC, const float* weight0, const __m512* bias, const __m512* params, float* dst, const __mmask16* tails, int first)
        {
            __m512 d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51, d60, d61, d70, d71, d80, d81, d90, d91, da0, da1, db0, db1, dc0, dc1, dd0, dd1, s0, w0, w1;
            size_t srcH = p.srcH, srcW = p.srcW, dilY = p.dilationY, dilX = p.dilationX;
            size_t dY = p.srcW * p.srcC, dX = p.srcC, dS = p.srcC * p.strideX, dW = srcC * DF, dWz = p.kernelX * srcC * DF, dD = p.dstC;
            size_t sy = dy * p.strideY - p.padY, sx = dx * p.strideX - p.padX;
            size_t kY = p.kernelY * p.dilationY, kX = p.kernelX * p.dilationX;
            const float* weight1 = weight0 + F;
            const float* src1 = src0 + 1 * dS;
            const float* src2 = src0 + 2 * dS;
            const float* src3 = src0 + 3 * dS;
            const float* src4 = src0 + 4 * dS;
            const float* src5 = src0 + 5 * dS;
            const float* src6 = src0 + 6 * dS;
            if (tails[1])
            {
                if (first)
                {
                    if (M > 0x0) d00 = _mm512_setzero_ps(), d01 = _mm512_setzero_ps();
                    if (M > 0x1) d10 = _mm512_setzero_ps(), d11 = _mm512_setzero_ps();
                    if (M > 0x2) d20 = _mm512_setzero_ps(), d21 = _mm512_setzero_ps();
                    if (M > 0x3) d30 = _mm512_setzero_ps(), d31 = _mm512_setzero_ps();
                    if (M > 0x4) d40 = _mm512_setzero_ps(), d41 = _mm512_setzero_ps();
                    if (M > 0x5) d50 = _mm512_setzero_ps(), d51 = _mm512_setzero_ps();
                    if (M > 0x6) d60 = _mm512_setzero_ps(), d61 = _mm512_setzero_ps();
                    if (M > 0x7) d70 = _mm512_setzero_ps(), d71 = _mm512_setzero_ps();
                    if (M > 0x8) d80 = _mm512_setzero_ps(), d81 = _mm512_setzero_ps();
                    if (M > 0x9) d90 = _mm512_setzero_ps(), d91 = _mm512_setzero_ps();
                    if (M > 0xa) da0 = _mm512_setzero_ps(), da1 = _mm512_setzero_ps();
                    if (M > 0xb) db0 = _mm512_setzero_ps(), db1 = _mm512_setzero_ps();
                    if (M > 0xc) dc0 = _mm512_setzero_ps(), dc1 = _mm512_setzero_ps();
                    if (M > 0xd) dd0 = _mm512_setzero_ps(), dd1 = _mm512_setzero_ps();
                }
                else
                {
                    if (M > 0x0) d00 = _mm512_loadu_ps(dst + 0x0 * dD + 0), d01 = _mm512_maskz_loadu_ps(tails[1], dst + 0x0 * dD + F);
                    if (M > 0x1) d10 = _mm512_loadu_ps(dst + 0x1 * dD + 0), d11 = _mm512_maskz_loadu_ps(tails[1], dst + 0x1 * dD + F);
                    if (M > 0x2) d20 = _mm512_loadu_ps(dst + 0x2 * dD + 0), d21 = _mm512_maskz_loadu_ps(tails[1], dst + 0x2 * dD + F);
                    if (M > 0x3) d30 = _mm512_loadu_ps(dst + 0x3 * dD + 0), d31 = _mm512_maskz_loadu_ps(tails[1], dst + 0x3 * dD + F);
                    if (M > 0x4) d40 = _mm512_loadu_ps(dst + 0x4 * dD + 0), d41 = _mm512_maskz_loadu_ps(tails[1], dst + 0x4 * dD + F);
                    if (M > 0x5) d50 = _mm512_loadu_ps(dst + 0x5 * dD + 0), d51 = _mm512_maskz_loadu_ps(tails[1], dst + 0x5 * dD + F);
                    if (M > 0x6) d60 = _mm512_loadu_ps(dst + 0x6 * dD + 0), d61 = _mm512_maskz_loadu_ps(tails[1], dst + 0x6 * dD + F);
                    if (M > 0x7) d70 = _mm512_loadu_ps(dst + 0x7 * dD + 0), d71 = _mm512_maskz_loadu_ps(tails[1], dst + 0x7 * dD + F);
                    if (M > 0x8) d80 = _mm512_loadu_ps(dst + 0x8 * dD + 0), d81 = _mm512_maskz_loadu_ps(tails[1], dst + 0x8 * dD + F);
                    if (M > 0x9) d90 = _mm512_loadu_ps(dst + 0x9 * dD + 0), d91 = _mm512_maskz_loadu_ps(tails[1], dst + 0x9 * dD + F);
                    if (M > 0xa) da0 = _mm512_loadu_ps(dst + 0xa * dD + 0), da1 = _mm512_maskz_loadu_ps(tails[1], dst + 0xa * dD + F);
                    if (M > 0xb) db0 = _mm512_loadu_ps(dst + 0xb * dD + 0), db1 = _mm512_maskz_loadu_ps(tails[1], dst + 0xb * dD + F);
                    if (M > 0xc) dc0 = _mm512_loadu_ps(dst + 0xc * dD + 0), dc1 = _mm512_maskz_loadu_ps(tails[1], dst + 0xc * dD + F);
                    if (M > 0xd) dd0 = _mm512_loadu_ps(dst + 0xd * dD + 0), dd1 = _mm512_maskz_loadu_ps(tails[1], dst + 0xd * dD + F);
                }
                if (p.kernelY * p.kernelX * srcC * F * sizeof(float) > PREFETCH_SIZE)
                {
                    for (size_t ky = 0; ky < kY; ky += dilY)
                    {
                        if (sy + ky < srcH)
                        {
                            size_t beg = (sy + ky) * dY + sx * dX;
                            for (size_t kx = 0; kx < kX; kx += dilX)
                            {
                                assert(sx + kx < srcW&& sx + kx + M <= srcW);
                                size_t off0 = beg + kx * dX, end = off0 + srcC, off7 = off0 + 7 * dS, offw = 0;
                                for (; off0 < end; ++off0, ++off7, offw += DF)
                                {
                                    PrefetchL1(weight0 + offw);
                                    PrefetchL1(weight1 + offw);
                                    w0 = _mm512_loadu_ps(weight0 + offw);
                                    w1 = _mm512_loadu_ps(weight1 + offw);
                                    if (M > 0x0) s0 = _mm512_set1_ps(src0[off0]), d00 = _mm512_fmadd_ps(s0, w0, d00), d01 = _mm512_fmadd_ps(s0, w1, d01);
                                    if (M > 0x1) s0 = _mm512_set1_ps(src1[off0]), d10 = _mm512_fmadd_ps(s0, w0, d10), d11 = _mm512_fmadd_ps(s0, w1, d11);
                                    if (M > 0x2) s0 = _mm512_set1_ps(src2[off0]), d20 = _mm512_fmadd_ps(s0, w0, d20), d21 = _mm512_fmadd_ps(s0, w1, d21);
                                    if (M > 0x3) s0 = _mm512_set1_ps(src3[off0]), d30 = _mm512_fmadd_ps(s0, w0, d30), d31 = _mm512_fmadd_ps(s0, w1, d31);
                                    if (M > 0x4) s0 = _mm512_set1_ps(src4[off0]), d40 = _mm512_fmadd_ps(s0, w0, d40), d41 = _mm512_fmadd_ps(s0, w1, d41);
                                    if (M > 0x5) s0 = _mm512_set1_ps(src5[off0]), d50 = _mm512_fmadd_ps(s0, w0, d50), d51 = _mm512_fmadd_ps(s0, w1, d51);
                                    if (M > 0x6) s0 = _mm512_set1_ps(src6[off0]), d60 = _mm512_fmadd_ps(s0, w0, d60), d61 = _mm512_fmadd_ps(s0, w1, d61);
                                    if (M > 0x7) s0 = _mm512_set1_ps(src0[off7]), d70 = _mm512_fmadd_ps(s0, w0, d70), d71 = _mm512_fmadd_ps(s0, w1, d71);
                                    if (M > 0x8) s0 = _mm512_set1_ps(src1[off7]), d80 = _mm512_fmadd_ps(s0, w0, d80), d81 = _mm512_fmadd_ps(s0, w1, d81);
                                    if (M > 0x9) s0 = _mm512_set1_ps(src2[off7]), d90 = _mm512_fmadd_ps(s0, w0, d90), d91 = _mm512_fmadd_ps(s0, w1, d91);
                                    if (M > 0xa) s0 = _mm512_set1_ps(src3[off7]), da0 = _mm512_fmadd_ps(s0, w0, da0), da1 = _mm512_fmadd_ps(s0, w1, da1);
                                    if (M > 0xb) s0 = _mm512_set1_ps(src4[off7]), db0 = _mm512_fmadd_ps(s0, w0, db0), db1 = _mm512_fmadd_ps(s0, w1, db1);
                                    if (M > 0xc) s0 = _mm512_set1_ps(src5[off7]), dc0 = _mm512_fmadd_ps(s0, w0, dc0), dc1 = _mm512_fmadd_ps(s0, w1, dc1);
                                    if (M > 0xd) s0 = _mm512_set1_ps(src6[off7]), dd0 = _mm512_fmadd_ps(s0, w0, dd0), dd1 = _mm512_fmadd_ps(s0, w1, dd1);
                                }
                                weight0 += dW, weight1 += dW;
                            }
                        }
                        else
                            weight0 += dWz, weight1 += dWz;
                    }
                }
                else
                {
                    for (size_t ky = 0; ky < kY; ky += dilY)
                    {
                        if (sy + ky < srcH)
                        {
                            size_t beg = (sy + ky) * dY + sx * dX;
                            for (size_t kx = 0; kx < kX; kx += dilX)
                            {
                                assert(sx + kx < srcW&& sx + kx + M <= srcW);
                                size_t off0 = beg + kx * dX, end = off0 + srcC, off7 = off0 + 7 * dS, offw = 0;
                                for (; off0 < end; ++off0, ++off7, offw += DF)
                                {
                                    w0 = _mm512_loadu_ps(weight0 + offw);
                                    w1 = _mm512_loadu_ps(weight1 + offw);
                                    if (M > 0x0) s0 = _mm512_set1_ps(src0[off0]), d00 = _mm512_fmadd_ps(s0, w0, d00), d01 = _mm512_fmadd_ps(s0, w1, d01);
                                    if (M > 0x1) s0 = _mm512_set1_ps(src1[off0]), d10 = _mm512_fmadd_ps(s0, w0, d10), d11 = _mm512_fmadd_ps(s0, w1, d11);
                                    if (M > 0x2) s0 = _mm512_set1_ps(src2[off0]), d20 = _mm512_fmadd_ps(s0, w0, d20), d21 = _mm512_fmadd_ps(s0, w1, d21);
                                    if (M > 0x3) s0 = _mm512_set1_ps(src3[off0]), d30 = _mm512_fmadd_ps(s0, w0, d30), d31 = _mm512_fmadd_ps(s0, w1, d31);
                                    if (M > 0x4) s0 = _mm512_set1_ps(src4[off0]), d40 = _mm512_fmadd_ps(s0, w0, d40), d41 = _mm512_fmadd_ps(s0, w1, d41);
                                    if (M > 0x5) s0 = _mm512_set1_ps(src5[off0]), d50 = _mm512_fmadd_ps(s0, w0, d50), d51 = _mm512_fmadd_ps(s0, w1, d51);
                                    if (M > 0x6) s0 = _mm512_set1_ps(src6[off0]), d60 = _mm512_fmadd_ps(s0, w0, d60), d61 = _mm512_fmadd_ps(s0, w1, d61);
                                    if (M > 0x7) s0 = _mm512_set1_ps(src0[off7]), d70 = _mm512_fmadd_ps(s0, w0, d70), d71 = _mm512_fmadd_ps(s0, w1, d71);
                                    if (M > 0x8) s0 = _mm512_set1_ps(src1[off7]), d80 = _mm512_fmadd_ps(s0, w0, d80), d81 = _mm512_fmadd_ps(s0, w1, d81);
                                    if (M > 0x9) s0 = _mm512_set1_ps(src2[off7]), d90 = _mm512_fmadd_ps(s0, w0, d90), d91 = _mm512_fmadd_ps(s0, w1, d91);
                                    if (M > 0xa) s0 = _mm512_set1_ps(src3[off7]), da0 = _mm512_fmadd_ps(s0, w0, da0), da1 = _mm512_fmadd_ps(s0, w1, da1);
                                    if (M > 0xb) s0 = _mm512_set1_ps(src4[off7]), db0 = _mm512_fmadd_ps(s0, w0, db0), db1 = _mm512_fmadd_ps(s0, w1, db1);
                                    if (M > 0xc) s0 = _mm512_set1_ps(src5[off7]), dc0 = _mm512_fmadd_ps(s0, w0, dc0), dc1 = _mm512_fmadd_ps(s0, w1, dc1);
                                    if (M > 0xd) s0 = _mm512_set1_ps(src6[off7]), dd0 = _mm512_fmadd_ps(s0, w0, dd0), dd1 = _mm512_fmadd_ps(s0, w1, dd1);
                                }
                                weight0 += dW, weight1 += dW;
                            }
                        }
                        else
                            weight0 += dWz, weight1 += dWz;
                    }
                }
                if (M > 0x0) Save2<term, type>(dst, d00, d01, bias, params, tails), dst += dD;
                if (M > 0x1) Save2<term, type>(dst, d10, d11, bias, params, tails), dst += dD;
                if (M > 0x2) Save2<term, type>(dst, d20, d21, bias, params, tails), dst += dD;
                if (M > 0x3) Save2<term, type>(dst, d30, d31, bias, params, tails), dst += dD;
                if (M > 0x4) Save2<term, type>(dst, d40, d41, bias, params, tails), dst += dD;
                if (M > 0x5) Save2<term, type>(dst, d50, d51, bias, params, tails), dst += dD;
                if (M > 0x6) Save2<term, type>(dst, d60, d61, bias, params, tails), dst += dD;
                if (M > 0x7) Save2<term, type>(dst, d70, d71, bias, params, tails), dst += dD;
                if (M > 0x8) Save2<term, type>(dst, d80, d81, bias, params, tails), dst += dD;
                if (M > 0x9) Save2<term, type>(dst, d90, d91, bias, params, tails), dst += dD;
                if (M > 0xa) Save2<term, type>(dst, da0, da1, bias, params, tails), dst += dD;
                if (M > 0xb) Save2<term, type>(dst, db0, db1, bias, params, tails), dst += dD;
                if (M > 0xc) Save2<term, type>(dst, dc0, dc1, bias, params, tails), dst += dD;
                if (M > 0xd) Save2<term, type>(dst, dd0, dd1, bias, params, tails), dst += dD;
            }
            else
            {
                if (first)
                {
                    if (M > 0x0) d00 = _mm512_setzero_ps();
                    if (M > 0x1) d10 = _mm512_setzero_ps();
                    if (M > 0x2) d20 = _mm512_setzero_ps();
                    if (M > 0x3) d30 = _mm512_setzero_ps();
                    if (M > 0x4) d40 = _mm512_setzero_ps();
                    if (M > 0x5) d50 = _mm512_setzero_ps();
                    if (M > 0x6) d60 = _mm512_setzero_ps();
                    if (M > 0x7) d70 = _mm512_setzero_ps();
                    if (M > 0x8) d80 = _mm512_setzero_ps();
                    if (M > 0x9) d90 = _mm512_setzero_ps();
                    if (M > 0xa) da0 = _mm512_setzero_ps();
                    if (M > 0xb) db0 = _mm512_setzero_ps();
                    if (M > 0xc) dc0 = _mm512_setzero_ps();
                    if (M > 0xd) dd0 = _mm512_setzero_ps();
                }
                else
                {
                    if (M > 0x0) d00 = _mm512_maskz_loadu_ps(tails[0], dst + 0x0 * dD + 0);
                    if (M > 0x1) d10 = _mm512_maskz_loadu_ps(tails[0], dst + 0x1 * dD + 0);
                    if (M > 0x2) d20 = _mm512_maskz_loadu_ps(tails[0], dst + 0x2 * dD + 0);
                    if (M > 0x3) d30 = _mm512_maskz_loadu_ps(tails[0], dst + 0x3 * dD + 0);
                    if (M > 0x4) d40 = _mm512_maskz_loadu_ps(tails[0], dst + 0x4 * dD + 0);
                    if (M > 0x5) d50 = _mm512_maskz_loadu_ps(tails[0], dst + 0x5 * dD + 0);
                    if (M > 0x6) d60 = _mm512_maskz_loadu_ps(tails[0], dst + 0x6 * dD + 0);
                    if (M > 0x7) d70 = _mm512_maskz_loadu_ps(tails[0], dst + 0x7 * dD + 0);
                    if (M > 0x8) d80 = _mm512_maskz_loadu_ps(tails[0], dst + 0x8 * dD + 0);
                    if (M > 0x9) d90 = _mm512_maskz_loadu_ps(tails[0], dst + 0x9 * dD + 0);
                    if (M > 0xa) da0 = _mm512_maskz_loadu_ps(tails[0], dst + 0xa * dD + 0);
                    if (M > 0xb) db0 = _mm512_maskz_loadu_ps(tails[0], dst + 0xb * dD + 0);
                    if (M > 0xc) dc0 = _mm512_maskz_loadu_ps(tails[0], dst + 0xc * dD + 0);
                    if (M > 0xd) dd0 = _mm512_maskz_loadu_ps(tails[0], dst + 0xd * dD + 0);
                }
                for (size_t ky = 0; ky < kY; ky += dilY)
                {
                    if (sy + ky < srcH)
                    {
                        size_t beg = (sy + ky) * dY + sx * dX;
                        for (size_t kx = 0; kx < kX; kx += dilX)
                        {
                            assert(sx + kx < srcW&& sx + kx + M <= srcW);
                            size_t off0 = beg + kx * dX, end = off0 + srcC, off7 = off0 + 7 * dS, offw = 0;
                            for (; off0 < end; ++off0, ++off7, offw += DF)
                            {
                                w0 = _mm512_loadu_ps(weight0 + offw);
                                if (M > 0x0) s0 = _mm512_set1_ps(src0[off0]), d00 = _mm512_fmadd_ps(s0, w0, d00);
                                if (M > 0x1) s0 = _mm512_set1_ps(src1[off0]), d10 = _mm512_fmadd_ps(s0, w0, d10);
                                if (M > 0x2) s0 = _mm512_set1_ps(src2[off0]), d20 = _mm512_fmadd_ps(s0, w0, d20);
                                if (M > 0x3) s0 = _mm512_set1_ps(src3[off0]), d30 = _mm512_fmadd_ps(s0, w0, d30);
                                if (M > 0x4) s0 = _mm512_set1_ps(src4[off0]), d40 = _mm512_fmadd_ps(s0, w0, d40);
                                if (M > 0x5) s0 = _mm512_set1_ps(src5[off0]), d50 = _mm512_fmadd_ps(s0, w0, d50);
                                if (M > 0x6) s0 = _mm512_set1_ps(src6[off0]), d60 = _mm512_fmadd_ps(s0, w0, d60);
                                if (M > 0x7) s0 = _mm512_set1_ps(src0[off7]), d70 = _mm512_fmadd_ps(s0, w0, d70);
                                if (M > 0x8) s0 = _mm512_set1_ps(src1[off7]), d80 = _mm512_fmadd_ps(s0, w0, d80);
                                if (M > 0x9) s0 = _mm512_set1_ps(src2[off7]), d90 = _mm512_fmadd_ps(s0, w0, d90);
                                if (M > 0xa) s0 = _mm512_set1_ps(src3[off7]), da0 = _mm512_fmadd_ps(s0, w0, da0);
                                if (M > 0xb) s0 = _mm512_set1_ps(src4[off7]), db0 = _mm512_fmadd_ps(s0, w0, db0);
                                if (M > 0xc) s0 = _mm512_set1_ps(src5[off7]), dc0 = _mm512_fmadd_ps(s0, w0, dc0);
                                if (M > 0xd) s0 = _mm512_set1_ps(src6[off7]), dd0 = _mm512_fmadd_ps(s0, w0, dd0);
                            }
                            weight0 += dW;
                        }
                    }
                    else
                        weight0 += dWz;
                }
                if (M > 0x0) Save1<term, type>(dst, d00, bias, params, tails), dst += dD;
                if (M > 0x1) Save1<term, type>(dst, d10, bias, params, tails), dst += dD;
                if (M > 0x2) Save1<term, type>(dst, d20, bias, params, tails), dst += dD;
                if (M > 0x3) Save1<term, type>(dst, d30, bias, params, tails), dst += dD;
                if (M > 0x4) Save1<term, type>(dst, d40, bias, params, tails), dst += dD;
                if (M > 0x5) Save1<term, type>(dst, d50, bias, params, tails), dst += dD;
                if (M > 0x6) Save1<term, type>(dst, d60, bias, params, tails), dst += dD;
                if (M > 0x7) Save1<term, type>(dst, d70, bias, params, tails), dst += dD;
                if (M > 0x8) Save1<term, type>(dst, d80, bias, params, tails), dst += dD;
                if (M > 0x9) Save1<term, type>(dst, d90, bias, params, tails), dst += dD;
                if (M > 0xa) Save1<term, type>(dst, da0, bias, params, tails), dst += dD;
                if (M > 0xb) Save1<term, type>(dst, db0, bias, params, tails), dst += dD;
                if (M > 0xc) Save1<term, type>(dst, dc0, bias, params, tails), dst += dD;
                if (M > 0xd) Save1<term, type>(dst, dd0, bias, params, tails), dst += dD;
            }
        }

        typedef void(*ConvolutionNhwcDirect_NxM_Ptr)(const float* src0, const ConvParam& p, size_t dy, size_t dx, size_t srcC, const float* weight0, const __m512* bias, const __m512* params, float* dst, const __mmask16* tails, int first);

        template<TermType term, SimdConvolutionActivationType type> ConvolutionNhwcDirect_NxM_Ptr GetConvolutionNhwcDirect_2xM(size_t M)
        {
            switch (M)
            {
            case 0x0: return NULL;
            case 0x1: return ConvolutionNhwcDirect_2xM<term, type, 0x1>;
            case 0x2: return ConvolutionNhwcDirect_2xM<term, type, 0x2>;
            case 0x3: return ConvolutionNhwcDirect_2xM<term, type, 0x3>;
            case 0x4: return ConvolutionNhwcDirect_2xM<term, type, 0x4>;
            case 0x5: return ConvolutionNhwcDirect_2xM<term, type, 0x5>;
            case 0x6: return ConvolutionNhwcDirect_2xM<term, type, 0x6>;
            case 0x7: return ConvolutionNhwcDirect_2xM<term, type, 0x7>;
            case 0x8: return ConvolutionNhwcDirect_2xM<term, type, 0x8>;
            case 0x9: return ConvolutionNhwcDirect_2xM<term, type, 0x9>;
            case 0xa: return ConvolutionNhwcDirect_2xM<term, type, 0xa>;
            case 0xb: return ConvolutionNhwcDirect_2xM<term, type, 0xb>;
            case 0xc: return ConvolutionNhwcDirect_2xM<term, type, 0xc>;
            case 0xd: return ConvolutionNhwcDirect_2xM<term, type, 0xd>;
            case 0xe: return ConvolutionNhwcDirect_2xM<term, type, 0xe>;
            }
            assert(0);
            return NULL;
        }

        template<TermType term, SimdConvolutionActivationType type> void ConvolutionNhwcDirect_2(const float* src, const ConvParam& p,
            size_t dstC, size_t yBeg, size_t yEnd, size_t srcC, const float* weight, const float* bias, const float* params, float* dst, int first)
        {
            size_t noseH = p.NoseH(), noseW = p.NoseW(), bodyH = p.BodyH(), bodyW = p.BodyW();
            size_t n = 14, bodyWn = AlignLoAny(bodyW - noseW, n) + noseW, m = bodyW - bodyWn;
            ConvolutionNhwcDirect_NxM_Ptr convolutionNhwcDirect_2x1 = ConvolutionNhwcDirect_2x1<term, type>;
            ConvolutionNhwcDirect_NxM_Ptr convolutionNhwcDirect_2xN = GetConvolutionNhwcDirect_2xM<term, type>(n);
            ConvolutionNhwcDirect_NxM_Ptr convolutionNhwcDirect_2xM = GetConvolutionNhwcDirect_2xM<term, type>(m);
            size_t tailH = p.dstH, tailW = p.dstW;
            size_t kY = p.kernelY - noseH, kX = p.kernelX - noseW, kH = bodyH + p.kernelY - 1, kW = bodyW + p.kernelX - 1;

            __m512 _params[2], _bias[2];
            _params[0] = _mm512_set1_ps(params[0]);
            if (type == SimdConvolutionActivationRestrictRange ||
                type == SimdConvolutionActivationHswish ||
                type == SimdConvolutionActivationHardSigmoid)
                _params[1] = _mm512_set1_ps(params[1]);

            for (size_t dc = 0; dc < dstC; dc += DF)
            {
                size_t dC = Simd::Min(DF, dstC - dc);
                __mmask16 tails[2] = { TailMask16(dC), TailMask16(dC - F) };
                if (dC > 0 * F) _bias[0] = _mm512_loadu_ps(bias + dc + 0 * F);
                if (dC > 1 * F) _bias[1] = _mm512_loadu_ps(bias + dc + 1 * F);
                if (type == ::SimdConvolutionActivationPrelu)
                {
                    if (dC > 0 * F) _params[0] = _mm512_loadu_ps(params + dc + 0 * F);
                    if (dC > 1 * F) _params[1] = _mm512_loadu_ps(params + dc + 1 * F);
                }
                float* d = dst + dc + yBeg * p.dstW * p.dstC;
                for (size_t dy = yBeg; dy < yEnd; dy++)
                {
                    size_t dx = 0;
                    for (; dx < noseW; dx++, d += p.dstC)
                        convolutionNhwcDirect_2x1(src, p, dy, dx, srcC, weight, _bias, _params, d, tails, first);
                    for (; dx < bodyWn; dx += n, d += p.dstC * n)
                        convolutionNhwcDirect_2xN(src, p, dy, dx, srcC, weight, _bias, _params, d, tails, first);
                    for (; dx < bodyW; dx += m, d += p.dstC * m)
                        convolutionNhwcDirect_2xM(src, p, dy, dx, srcC, weight, _bias, _params, d, tails, first);
                    for (; dx < tailW; dx++, d += p.dstC)
                        convolutionNhwcDirect_2x1(src, p, dy, dx, srcC, weight, _bias, _params, d, tails, first);
                }
                weight += p.kernelY * p.kernelX * srcC * DF;
            }
        }

        template<SimdConvolutionActivationType type> void ConvolutionNhwcDirect_2(const float* src, const ConvParam& p,
            const SynetConvolution32fNhwcDirect::AlgParam& a, const float* weight, const float* bias, const float* params, float* dst)
        {
            for (size_t dc = 0; dc < p.dstC; dc += a.macroD)
            {
                size_t macroD = Simd::Min(p.dstC, dc + a.macroD) - dc;
                for (size_t sc = 0; sc < p.srcC; sc += a.macroC)
                {
                    size_t macroC = Simd::Min(p.srcC, sc + a.macroC) - sc;
                    size_t macroK = p.kernelY * p.kernelX * macroC;
                    for (size_t yBeg = 0; yBeg < p.dstH;)
                    {
                        size_t yEnd = Simd::Min(yBeg + a.macroH, p.dstH);
                        if (sc + macroC == p.srcC)
                            ConvolutionNhwcDirect_2<TermLast, type>(src + sc, p, macroD, yBeg, yEnd, macroC, weight, bias + dc, params, dst + dc, macroC == p.srcC ? 1 : 0);
                        else
                            ConvolutionNhwcDirect_2<TermInterim, SimdConvolutionActivationIdentity>(src + sc, p, macroD, yBeg, yEnd, macroC, weight, bias + dc, params, dst + dc, sc == 0 ? 1 : 0);
                        yBeg = yEnd;
                    }
                    weight += AlignHiAny(macroD, a.microD) * macroK;
                }
                if (type == ::SimdConvolutionActivationPrelu)
                    params += macroD;
            }
        }

        //---------------------------------------------------------------------

        template<TermType term, SimdConvolutionActivationType type, int M> void ConvolutionNhwcDirect1x1_2xM(const float* src0, const ConvParam& p,
            size_t srcC, const float* weight, const __m512* bias, const __m512* params, float* dst, const __mmask16 tails[2], int first)
        {
            __m512 d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51, d60, d61, d70, d71, d80, d81, d90, d91, da0, da1, db0, db1, dc0, dc1, dd0, dd1, s0, w0, w1;
            size_t dS = p.srcC, dD = p.dstC;
            const float* src1 = src0 + 1 * dS;
            const float* src2 = src0 + 2 * dS;
            const float* src3 = src0 + 3 * dS;
            const float* src4 = src0 + 4 * dS;
            const float* src5 = src0 + 5 * dS;
            const float* src6 = src0 + 6 * dS;
            if (tails[1])
            {
                if (first)
                {
                    if (M > 0x0) d00 = _mm512_setzero_ps(), d01 = _mm512_setzero_ps();
                    if (M > 0x1) d10 = _mm512_setzero_ps(), d11 = _mm512_setzero_ps();
                    if (M > 0x2) d20 = _mm512_setzero_ps(), d21 = _mm512_setzero_ps();
                    if (M > 0x3) d30 = _mm512_setzero_ps(), d31 = _mm512_setzero_ps();
                    if (M > 0x4) d40 = _mm512_setzero_ps(), d41 = _mm512_setzero_ps();
                    if (M > 0x5) d50 = _mm512_setzero_ps(), d51 = _mm512_setzero_ps();
                    if (M > 0x6) d60 = _mm512_setzero_ps(), d61 = _mm512_setzero_ps();
                    if (M > 0x7) d70 = _mm512_setzero_ps(), d71 = _mm512_setzero_ps();
                    if (M > 0x8) d80 = _mm512_setzero_ps(), d81 = _mm512_setzero_ps();
                    if (M > 0x9) d90 = _mm512_setzero_ps(), d91 = _mm512_setzero_ps();
                    if (M > 0xa) da0 = _mm512_setzero_ps(), da1 = _mm512_setzero_ps();
                    if (M > 0xb) db0 = _mm512_setzero_ps(), db1 = _mm512_setzero_ps();
                    if (M > 0xc) dc0 = _mm512_setzero_ps(), dc1 = _mm512_setzero_ps();
                    if (M > 0xd) dd0 = _mm512_setzero_ps(), dd1 = _mm512_setzero_ps();
                }
                else
                {
                    if (M > 0x0) d00 = _mm512_loadu_ps(dst + 0x0 * dD + 0), d01 = _mm512_maskz_loadu_ps(tails[1], dst + 0x0 * dD + F);
                    if (M > 0x1) d10 = _mm512_loadu_ps(dst + 0x1 * dD + 0), d11 = _mm512_maskz_loadu_ps(tails[1], dst + 0x1 * dD + F);
                    if (M > 0x2) d20 = _mm512_loadu_ps(dst + 0x2 * dD + 0), d21 = _mm512_maskz_loadu_ps(tails[1], dst + 0x2 * dD + F);
                    if (M > 0x3) d30 = _mm512_loadu_ps(dst + 0x3 * dD + 0), d31 = _mm512_maskz_loadu_ps(tails[1], dst + 0x3 * dD + F);
                    if (M > 0x4) d40 = _mm512_loadu_ps(dst + 0x4 * dD + 0), d41 = _mm512_maskz_loadu_ps(tails[1], dst + 0x4 * dD + F);
                    if (M > 0x5) d50 = _mm512_loadu_ps(dst + 0x5 * dD + 0), d51 = _mm512_maskz_loadu_ps(tails[1], dst + 0x5 * dD + F);
                    if (M > 0x6) d60 = _mm512_loadu_ps(dst + 0x6 * dD + 0), d61 = _mm512_maskz_loadu_ps(tails[1], dst + 0x6 * dD + F);
                    if (M > 0x7) d70 = _mm512_loadu_ps(dst + 0x7 * dD + 0), d71 = _mm512_maskz_loadu_ps(tails[1], dst + 0x7 * dD + F);
                    if (M > 0x8) d80 = _mm512_loadu_ps(dst + 0x8 * dD + 0), d81 = _mm512_maskz_loadu_ps(tails[1], dst + 0x8 * dD + F);
                    if (M > 0x9) d90 = _mm512_loadu_ps(dst + 0x9 * dD + 0), d91 = _mm512_maskz_loadu_ps(tails[1], dst + 0x9 * dD + F);
                    if (M > 0xa) da0 = _mm512_loadu_ps(dst + 0xa * dD + 0), da1 = _mm512_maskz_loadu_ps(tails[1], dst + 0xa * dD + F);
                    if (M > 0xb) db0 = _mm512_loadu_ps(dst + 0xb * dD + 0), db1 = _mm512_maskz_loadu_ps(tails[1], dst + 0xb * dD + F);
                    if (M > 0xc) dc0 = _mm512_loadu_ps(dst + 0xc * dD + 0), dc1 = _mm512_maskz_loadu_ps(tails[1], dst + 0xc * dD + F);
                    if (M > 0xd) dd0 = _mm512_loadu_ps(dst + 0xd * dD + 0), dd1 = _mm512_maskz_loadu_ps(tails[1], dst + 0xd * dD + F);
                }
                if (srcC * 2 * F * sizeof(float) > PREFETCH_SIZE)
                {
                    for (size_t offset0 = 0, offset7 = 7 * dS; offset0 < srcC; ++offset0, ++offset7)
                    {
                        PrefetchL1(weight + 0);
                        PrefetchL1(weight + F);
                        w0 = _mm512_loadu_ps(weight + 0);
                        w1 = _mm512_loadu_ps(weight + F);
                        if (M > 0x0) s0 = _mm512_set1_ps(src0[offset0]), d00 = _mm512_fmadd_ps(s0, w0, d00), d01 = _mm512_fmadd_ps(s0, w1, d01);
                        if (M > 0x1) s0 = _mm512_set1_ps(src1[offset0]), d10 = _mm512_fmadd_ps(s0, w0, d10), d11 = _mm512_fmadd_ps(s0, w1, d11);
                        if (M > 0x2) s0 = _mm512_set1_ps(src2[offset0]), d20 = _mm512_fmadd_ps(s0, w0, d20), d21 = _mm512_fmadd_ps(s0, w1, d21);
                        if (M > 0x3) s0 = _mm512_set1_ps(src3[offset0]), d30 = _mm512_fmadd_ps(s0, w0, d30), d31 = _mm512_fmadd_ps(s0, w1, d31);
                        if (M > 0x4) s0 = _mm512_set1_ps(src4[offset0]), d40 = _mm512_fmadd_ps(s0, w0, d40), d41 = _mm512_fmadd_ps(s0, w1, d41);
                        if (M > 0x5) s0 = _mm512_set1_ps(src5[offset0]), d50 = _mm512_fmadd_ps(s0, w0, d50), d51 = _mm512_fmadd_ps(s0, w1, d51);
                        if (M > 0x6) s0 = _mm512_set1_ps(src6[offset0]), d60 = _mm512_fmadd_ps(s0, w0, d60), d61 = _mm512_fmadd_ps(s0, w1, d61);
                        if (M > 0x7) s0 = _mm512_set1_ps(src0[offset7]), d70 = _mm512_fmadd_ps(s0, w0, d70), d71 = _mm512_fmadd_ps(s0, w1, d71);
                        if (M > 0x8) s0 = _mm512_set1_ps(src1[offset7]), d80 = _mm512_fmadd_ps(s0, w0, d80), d81 = _mm512_fmadd_ps(s0, w1, d81);
                        if (M > 0x9) s0 = _mm512_set1_ps(src2[offset7]), d90 = _mm512_fmadd_ps(s0, w0, d90), d91 = _mm512_fmadd_ps(s0, w1, d91);
                        if (M > 0xa) s0 = _mm512_set1_ps(src3[offset7]), da0 = _mm512_fmadd_ps(s0, w0, da0), da1 = _mm512_fmadd_ps(s0, w1, da1);
                        if (M > 0xb) s0 = _mm512_set1_ps(src4[offset7]), db0 = _mm512_fmadd_ps(s0, w0, db0), db1 = _mm512_fmadd_ps(s0, w1, db1);
                        if (M > 0xc) s0 = _mm512_set1_ps(src5[offset7]), dc0 = _mm512_fmadd_ps(s0, w0, dc0), dc1 = _mm512_fmadd_ps(s0, w1, dc1);
                        if (M > 0xd) s0 = _mm512_set1_ps(src6[offset7]), dd0 = _mm512_fmadd_ps(s0, w0, dd0), dd1 = _mm512_fmadd_ps(s0, w1, dd1);
                        weight += DF;
                    }
                }
                else
                {
                    for (size_t offset0 = 0, offset7 = 7 * dS; offset0 < srcC; ++offset0, ++offset7)
                    {
                        w0 = _mm512_loadu_ps(weight + 0);
                        w1 = _mm512_loadu_ps(weight + F);
                        if (M > 0x0) s0 = _mm512_set1_ps(src0[offset0]), d00 = _mm512_fmadd_ps(s0, w0, d00), d01 = _mm512_fmadd_ps(s0, w1, d01);
                        if (M > 0x1) s0 = _mm512_set1_ps(src1[offset0]), d10 = _mm512_fmadd_ps(s0, w0, d10), d11 = _mm512_fmadd_ps(s0, w1, d11);
                        if (M > 0x2) s0 = _mm512_set1_ps(src2[offset0]), d20 = _mm512_fmadd_ps(s0, w0, d20), d21 = _mm512_fmadd_ps(s0, w1, d21);
                        if (M > 0x3) s0 = _mm512_set1_ps(src3[offset0]), d30 = _mm512_fmadd_ps(s0, w0, d30), d31 = _mm512_fmadd_ps(s0, w1, d31);
                        if (M > 0x4) s0 = _mm512_set1_ps(src4[offset0]), d40 = _mm512_fmadd_ps(s0, w0, d40), d41 = _mm512_fmadd_ps(s0, w1, d41);
                        if (M > 0x5) s0 = _mm512_set1_ps(src5[offset0]), d50 = _mm512_fmadd_ps(s0, w0, d50), d51 = _mm512_fmadd_ps(s0, w1, d51);
                        if (M > 0x6) s0 = _mm512_set1_ps(src6[offset0]), d60 = _mm512_fmadd_ps(s0, w0, d60), d61 = _mm512_fmadd_ps(s0, w1, d61);
                        if (M > 0x7) s0 = _mm512_set1_ps(src0[offset7]), d70 = _mm512_fmadd_ps(s0, w0, d70), d71 = _mm512_fmadd_ps(s0, w1, d71);
                        if (M > 0x8) s0 = _mm512_set1_ps(src1[offset7]), d80 = _mm512_fmadd_ps(s0, w0, d80), d81 = _mm512_fmadd_ps(s0, w1, d81);
                        if (M > 0x9) s0 = _mm512_set1_ps(src2[offset7]), d90 = _mm512_fmadd_ps(s0, w0, d90), d91 = _mm512_fmadd_ps(s0, w1, d91);
                        if (M > 0xa) s0 = _mm512_set1_ps(src3[offset7]), da0 = _mm512_fmadd_ps(s0, w0, da0), da1 = _mm512_fmadd_ps(s0, w1, da1);
                        if (M > 0xb) s0 = _mm512_set1_ps(src4[offset7]), db0 = _mm512_fmadd_ps(s0, w0, db0), db1 = _mm512_fmadd_ps(s0, w1, db1);
                        if (M > 0xc) s0 = _mm512_set1_ps(src5[offset7]), dc0 = _mm512_fmadd_ps(s0, w0, dc0), dc1 = _mm512_fmadd_ps(s0, w1, dc1);
                        if (M > 0xd) s0 = _mm512_set1_ps(src6[offset7]), dd0 = _mm512_fmadd_ps(s0, w0, dd0), dd1 = _mm512_fmadd_ps(s0, w1, dd1);
                        weight += DF;
                    }
                }
                if (M > 0x0) Save2<term, type>(dst, d00, d01, bias, params, tails), dst += dD;
                if (M > 0x1) Save2<term, type>(dst, d10, d11, bias, params, tails), dst += dD;
                if (M > 0x2) Save2<term, type>(dst, d20, d21, bias, params, tails), dst += dD;
                if (M > 0x3) Save2<term, type>(dst, d30, d31, bias, params, tails), dst += dD;
                if (M > 0x4) Save2<term, type>(dst, d40, d41, bias, params, tails), dst += dD;
                if (M > 0x5) Save2<term, type>(dst, d50, d51, bias, params, tails), dst += dD;
                if (M > 0x6) Save2<term, type>(dst, d60, d61, bias, params, tails), dst += dD;
                if (M > 0x7) Save2<term, type>(dst, d70, d71, bias, params, tails), dst += dD;
                if (M > 0x8) Save2<term, type>(dst, d80, d81, bias, params, tails), dst += dD;
                if (M > 0x9) Save2<term, type>(dst, d90, d91, bias, params, tails), dst += dD;
                if (M > 0xa) Save2<term, type>(dst, da0, da1, bias, params, tails), dst += dD;
                if (M > 0xb) Save2<term, type>(dst, db0, db1, bias, params, tails), dst += dD;
                if (M > 0xc) Save2<term, type>(dst, dc0, dc1, bias, params, tails), dst += dD;
                if (M > 0xd) Save2<term, type>(dst, dd0, dd1, bias, params, tails), dst += dD;
            }
            else
            {
                if (first)
                {
                    if (M > 0x0) d00 = _mm512_setzero_ps();
                    if (M > 0x1) d10 = _mm512_setzero_ps();
                    if (M > 0x2) d20 = _mm512_setzero_ps();
                    if (M > 0x3) d30 = _mm512_setzero_ps();
                    if (M > 0x4) d40 = _mm512_setzero_ps();
                    if (M > 0x5) d50 = _mm512_setzero_ps();
                    if (M > 0x6) d60 = _mm512_setzero_ps();
                    if (M > 0x7) d70 = _mm512_setzero_ps();
                    if (M > 0x8) d80 = _mm512_setzero_ps();
                    if (M > 0x9) d90 = _mm512_setzero_ps();
                    if (M > 0xa) da0 = _mm512_setzero_ps();
                    if (M > 0xb) db0 = _mm512_setzero_ps();
                    if (M > 0xc) dc0 = _mm512_setzero_ps();
                    if (M > 0xd) dd0 = _mm512_setzero_ps();
                }
                else
                {
                    if (M > 0x0) d00 = _mm512_maskz_loadu_ps(tails[0], dst + 0x0 * dD + 0);
                    if (M > 0x1) d10 = _mm512_maskz_loadu_ps(tails[0], dst + 0x1 * dD + 0);
                    if (M > 0x2) d20 = _mm512_maskz_loadu_ps(tails[0], dst + 0x2 * dD + 0);
                    if (M > 0x3) d30 = _mm512_maskz_loadu_ps(tails[0], dst + 0x3 * dD + 0);
                    if (M > 0x4) d40 = _mm512_maskz_loadu_ps(tails[0], dst + 0x4 * dD + 0);
                    if (M > 0x5) d50 = _mm512_maskz_loadu_ps(tails[0], dst + 0x5 * dD + 0);
                    if (M > 0x6) d60 = _mm512_maskz_loadu_ps(tails[0], dst + 0x6 * dD + 0);
                    if (M > 0x7) d70 = _mm512_maskz_loadu_ps(tails[0], dst + 0x7 * dD + 0);
                    if (M > 0x8) d80 = _mm512_maskz_loadu_ps(tails[0], dst + 0x8 * dD + 0);
                    if (M > 0x9) d90 = _mm512_maskz_loadu_ps(tails[0], dst + 0x9 * dD + 0);
                    if (M > 0xa) da0 = _mm512_maskz_loadu_ps(tails[0], dst + 0xa * dD + 0);
                    if (M > 0xb) db0 = _mm512_maskz_loadu_ps(tails[0], dst + 0xb * dD + 0);
                    if (M > 0xc) dc0 = _mm512_maskz_loadu_ps(tails[0], dst + 0xc * dD + 0);
                    if (M > 0xd) dd0 = _mm512_maskz_loadu_ps(tails[0], dst + 0xd * dD + 0);
                }
                for (size_t offset0 = 0, offset7 = 7 * dS; offset0 < srcC; ++offset0, ++offset7)
                {
                    w0 = _mm512_loadu_ps(weight + 0);
                    if (M > 0x0) s0 = _mm512_set1_ps(src0[offset0]), d00 = _mm512_fmadd_ps(s0, w0, d00);
                    if (M > 0x1) s0 = _mm512_set1_ps(src1[offset0]), d10 = _mm512_fmadd_ps(s0, w0, d10);
                    if (M > 0x2) s0 = _mm512_set1_ps(src2[offset0]), d20 = _mm512_fmadd_ps(s0, w0, d20);
                    if (M > 0x3) s0 = _mm512_set1_ps(src3[offset0]), d30 = _mm512_fmadd_ps(s0, w0, d30);
                    if (M > 0x4) s0 = _mm512_set1_ps(src4[offset0]), d40 = _mm512_fmadd_ps(s0, w0, d40);
                    if (M > 0x5) s0 = _mm512_set1_ps(src5[offset0]), d50 = _mm512_fmadd_ps(s0, w0, d50);
                    if (M > 0x6) s0 = _mm512_set1_ps(src6[offset0]), d60 = _mm512_fmadd_ps(s0, w0, d60);
                    if (M > 0x7) s0 = _mm512_set1_ps(src0[offset7]), d70 = _mm512_fmadd_ps(s0, w0, d70);
                    if (M > 0x8) s0 = _mm512_set1_ps(src1[offset7]), d80 = _mm512_fmadd_ps(s0, w0, d80);
                    if (M > 0x9) s0 = _mm512_set1_ps(src2[offset7]), d90 = _mm512_fmadd_ps(s0, w0, d90);
                    if (M > 0xa) s0 = _mm512_set1_ps(src3[offset7]), da0 = _mm512_fmadd_ps(s0, w0, da0);
                    if (M > 0xb) s0 = _mm512_set1_ps(src4[offset7]), db0 = _mm512_fmadd_ps(s0, w0, db0);
                    if (M > 0xc) s0 = _mm512_set1_ps(src5[offset7]), dc0 = _mm512_fmadd_ps(s0, w0, dc0);
                    if (M > 0xd) s0 = _mm512_set1_ps(src6[offset7]), dd0 = _mm512_fmadd_ps(s0, w0, dd0);
                    weight += DF;
                }
                if (M > 0x0) Save1<term, type>(dst, d00, bias, params, tails), dst += dD;
                if (M > 0x1) Save1<term, type>(dst, d10, bias, params, tails), dst += dD;
                if (M > 0x2) Save1<term, type>(dst, d20, bias, params, tails), dst += dD;
                if (M > 0x3) Save1<term, type>(dst, d30, bias, params, tails), dst += dD;
                if (M > 0x4) Save1<term, type>(dst, d40, bias, params, tails), dst += dD;
                if (M > 0x5) Save1<term, type>(dst, d50, bias, params, tails), dst += dD;
                if (M > 0x6) Save1<term, type>(dst, d60, bias, params, tails), dst += dD;
                if (M > 0x7) Save1<term, type>(dst, d70, bias, params, tails), dst += dD;
                if (M > 0x8) Save1<term, type>(dst, d80, bias, params, tails), dst += dD;
                if (M > 0x9) Save1<term, type>(dst, d90, bias, params, tails), dst += dD;
                if (M > 0xa) Save1<term, type>(dst, da0, bias, params, tails), dst += dD;
                if (M > 0xb) Save1<term, type>(dst, db0, bias, params, tails), dst += dD;
                if (M > 0xc) Save1<term, type>(dst, dc0, bias, params, tails), dst += dD;
                if (M > 0xd) Save1<term, type>(dst, dd0, bias, params, tails), dst += dD;
            }
        }

        typedef void(*ConvolutionNhwcDirect1x1_2xM_Ptr)(const float* src0, const ConvParam& p, size_t srcC, const float* weight, const __m512* bias, const __m512* params, float* dst, const __mmask16 tails[2], int first);

        template<TermType term, SimdConvolutionActivationType type> ConvolutionNhwcDirect1x1_2xM_Ptr GetConvolutionNhwcDirect1x1_2xM(size_t M)
        {
            switch (M)
            {
            case 0x0: return NULL;
            case 0x1: return ConvolutionNhwcDirect1x1_2xM<term, type, 0x1>;
            case 0x2: return ConvolutionNhwcDirect1x1_2xM<term, type, 0x2>;
            case 0x3: return ConvolutionNhwcDirect1x1_2xM<term, type, 0x3>;
            case 0x4: return ConvolutionNhwcDirect1x1_2xM<term, type, 0x4>;
            case 0x5: return ConvolutionNhwcDirect1x1_2xM<term, type, 0x5>;
            case 0x6: return ConvolutionNhwcDirect1x1_2xM<term, type, 0x6>;
            case 0x7: return ConvolutionNhwcDirect1x1_2xM<term, type, 0x7>;
            case 0x8: return ConvolutionNhwcDirect1x1_2xM<term, type, 0x8>;
            case 0x9: return ConvolutionNhwcDirect1x1_2xM<term, type, 0x9>;
            case 0xa: return ConvolutionNhwcDirect1x1_2xM<term, type, 0xa>;
            case 0xb: return ConvolutionNhwcDirect1x1_2xM<term, type, 0xb>;
            case 0xc: return ConvolutionNhwcDirect1x1_2xM<term, type, 0xc>;
            case 0xd: return ConvolutionNhwcDirect1x1_2xM<term, type, 0xd>;
            case 0xe: return ConvolutionNhwcDirect1x1_2xM<term, type, 0xe>;
            }
            assert(0);
            return NULL;
        }

        template<TermType term, SimdConvolutionActivationType type> void ConvolutionNhwcDirect1x1_2(const float* src, const ConvParam& p,
            size_t dstC, size_t yBeg, size_t yEnd, size_t srcC, const float* weight, const float* bias, const float* params, float* dst, int first)
        {
            const size_t n = 14, n1 = (yEnd - yBeg) * p.dstW;
            size_t nn = AlignLoAny(n1, n), m = n1 - nn;
            ConvolutionNhwcDirect1x1_2xM_Ptr convolutionNhwcDirect1x1_2xN = GetConvolutionNhwcDirect1x1_2xM<term, type>(n);
            ConvolutionNhwcDirect1x1_2xM_Ptr convolutionNhwcDirect1x1_2xM = GetConvolutionNhwcDirect1x1_2xM<term, type>(m);

            __m512 _params[2], _bias[2];
            _params[0] = _mm512_set1_ps(params[0]);
            if (type == SimdConvolutionActivationRestrictRange ||
                type == SimdConvolutionActivationHswish ||
                type == SimdConvolutionActivationHardSigmoid)
                _params[1] = _mm512_set1_ps(params[1]);

            for (size_t dc = 0; dc < dstC; dc += DF)
            {
                size_t tail = Simd::Min(DF, dstC - dc);
                __mmask16 tails[2] = { TailMask16(tail), TailMask16(tail - F) };
                _bias[0] = _mm512_loadu_ps(bias + dc + 0);
                _bias[1] = _mm512_loadu_ps(bias + dc + F);
                if (type == ::SimdConvolutionActivationPrelu)
                {
                    _params[0] = _mm512_loadu_ps(params + dc + 0);
                    _params[1] = _mm512_loadu_ps(params + dc + F);
                }
                const float* ps = src + yBeg * p.srcW * p.srcC;
                float* pd = dst + dc + yBeg * p.dstW * p.dstC;
                size_t i = 0;
                for (; i < nn; i += n, ps += n * p.srcC, pd += n * p.dstC)
                    convolutionNhwcDirect1x1_2xN(ps, p, srcC, weight, _bias, _params, pd, tails, first);
                for (; i < n1; i += m, ps += m * p.srcC, pd += m * p.dstC)
                    convolutionNhwcDirect1x1_2xM(ps, p, srcC, weight, _bias, _params, pd, tails, first);
                weight += srcC * DF;
            }
        }

        template<SimdConvolutionActivationType type> void ConvolutionNhwcDirect1x1_2(const float* src, const ConvParam& p,
            const SynetConvolution32fNhwcDirect::AlgParam& a, const float* weight, const float* bias, const float* params, float* dst)
        {
            for (size_t dc = 0; dc < p.dstC; dc += a.macroD)
            {
                size_t macroD = Simd::Min(p.dstC, dc + a.macroD) - dc;
                for (size_t sc = 0; sc < p.srcC; sc += a.macroC)
                {
                    size_t macroC = Simd::Min(p.srcC, sc + a.macroC) - sc;
                    for (size_t yBeg = 0; yBeg < p.dstH;)
                    {
                        size_t yEnd = Simd::Min(yBeg + a.macroH, p.dstH);
                        if (sc + macroC == p.srcC)
                            ConvolutionNhwcDirect1x1_2<TermLast, type>(src + sc, p, macroD, yBeg, yEnd, macroC, weight, bias + dc, params, dst + dc, macroC == p.srcC ? 1 : 0);
                        else
                            ConvolutionNhwcDirect1x1_2<TermInterim, SimdConvolutionActivationIdentity>(src + sc, p, macroD, yBeg, yEnd, macroC, weight, bias + dc, params, dst + dc, sc == 0 ? 1 : 0);
                        yBeg = yEnd;
                    }
                    weight += AlignHiAny(macroD, a.microD) * macroC;
                }
                if (type == ::SimdConvolutionActivationPrelu)
                    params += macroD;
            }
        }

        //---------------------------------------------------------------------

        template <SimdConvolutionActivationType type> SIMD_INLINE void Set(const ConvParam& p, SynetConvolution32fNhwcDirect::OldConvolutionPtr& convolution)
        {
            if (p.Is1x1())
                convolution = ConvolutionNhwcDirect1x1_2<type>;
            else
                convolution = ConvolutionNhwcDirect_2<type>;
        }

        bool SynetConvolution32fNhwcDirect::Set2f(const ConvParam& p, OldConvolutionPtr& convolution)
        {
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: Set<SimdConvolutionActivationRestrictRange>(p, convolution); break;
            case SimdConvolutionActivationRelu: Set<SimdConvolutionActivationRestrictRange>(p, convolution); break;
            case SimdConvolutionActivationLeakyRelu: Set<SimdConvolutionActivationPrelu>(p, convolution); break;
            case SimdConvolutionActivationRestrictRange: Set<SimdConvolutionActivationRestrictRange>(p, convolution); break;
            case SimdConvolutionActivationPrelu: Set<SimdConvolutionActivationPrelu>(p, convolution); break;
            case SimdConvolutionActivationElu: Set<SimdConvolutionActivationElu>(p, convolution); break;
            case SimdConvolutionActivationHswish: Set<SimdConvolutionActivationHswish>(p, convolution); break;
            case SimdConvolutionActivationMish: Set<SimdConvolutionActivationMish>(p, convolution); break;
            case SimdConvolutionActivationHardSigmoid: Set<SimdConvolutionActivationHardSigmoid>(p, convolution); break;
            case SimdConvolutionActivationSwish: Set<SimdConvolutionActivationSwish>(p, convolution); break;
            case SimdConvolutionActivationGelu: Set<SimdConvolutionActivationGelu>(p, convolution); break;
            default: assert(0);
            }
            return true;
        }
    }
#endif
}
