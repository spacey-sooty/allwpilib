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
#include "Simd/SimdSynetConvolution16b.h"
#include "Simd/SimdSynetConvolution32f.h"
#include "Simd/SimdSynetConvolution32fCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdBFloat16.h"
#include "Simd/SimdAlignment.h"

namespace Simd
{
#if defined(SIMD_SYNET_ENABLE)
    namespace Base
    {
        SynetConvolution16bNhwcSpecV0::SynetConvolution16bNhwcSpecV0(const ConvParam& p)
            : SynetConvolution16b(p)
        {
            _preprocess = 0;
            _convolution = 0;
            _postprocess = 0;
        }

        String SynetConvolution16bNhwcSpecV0::Desc() const
        {
            std::stringstream desc;
            desc << Ext() << "::NhwcSpecV0";
            if (_alg.batch > 1)
                desc << "-" << _alg.batch;
            if (_alg.inv)
                desc << "-i";
            return desc.str();
        }

        void SynetConvolution16bNhwcSpecV0::SetAlgParam(size_t F, size_t microD, size_t microS, size_t microC, size_t L1, size_t L2, size_t L3)
        {
            const ConvParam& p = _param;
            AlgParam& a = _alg;

            a.F = F;
            a.microD = microD;
            a.microS = microS;
            a.microC = microC;
            a.srcC = AlignHi(p.srcC, a.microC);
            a.padV = Simd::Max(p.padY, p.padH);
            a.padH = Simd::Max(p.padX, p.padW);
            a.srcH = p.srcH + a.padV;
            a.srcW = p.srcW + a.padH;
            a.gapV = a.srcH - p.dstH;
            a.gapH = a.srcW - p.dstW;
            a.dstC = AlignHi(p.dstC, a.F);
            a.kA = p.kernelX * p.kernelY;
            a.K = a.srcC * a.kA;
            a.padE = a.srcW * a.padV + a.padH * Simd::Max<size_t>(1, a.padV) + a.microC;

            if (InvertedOrder())
            {
                a.inv = 1;
                a.macroC = Simd::RestrictRange<size_t>(AlignLo(L1 / a.microS / p.kernelY / 2, a.microC), a.microC, a.srcC);
                a.batch = 1;
                size_t bufSize = a.srcC * a.srcH * a.srcW * 2;
                if (bufSize * 2 <= L3 && p.batch > 1)
                {
                    for (size_t batch = 1; batch <= p.batch; ++batch)
                        if (p.batch % batch == 0 && batch * bufSize <= L3)
                            a.batch = batch;
                }
                a.macroH = Simd::RestrictRange(L3 / a.macroC / a.srcW / 2, size_t(1), p.dstH * a.batch);
                a.macroD = Simd::RestrictRange(AlignLo(L2 / a.macroC / a.kA / 2, a.microD), a.microD, AlignHi(p.dstC, a.microD));

                a.bufD = ((a.macroH + a.batch - 1) * a.srcW + a.F) * AlignHiAny(a.dstC, a.macroD);
            }
            else
            {
                a.inv = 0;                
                a.macroC = Simd::RestrictRange(AlignLo(L1 / a.microD / a.kA / 2, a.microC), a.microC, a.srcC);
                a.batch = 1;
                size_t bufSize = a.srcC * a.srcH * a.srcW * 2;
                if (bufSize * 2 <= L2 && p.batch > 1)
                {
                    for (size_t batch = 1; batch <= p.batch; ++batch)
                        if (p.batch % batch == 0 && batch * bufSize <= L2)
                            a.batch = batch;
                }
                a.macroH = Simd::RestrictRange(L2 / a.macroC / a.srcW / 2, size_t(1), p.dstH * a.batch);
                a.macroD = Simd::RestrictRange(AlignLoAny(L3 / a.macroC / a.kA / 2, a.microD), a.microD, AlignHiAny(p.dstC, a.microD));

                a.numH = DivHi(p.dstH * a.batch, a.macroH);
                a.bufD = (a.batch * a.srcH * a.srcW + a.numH * a.F) * a.macroD;            
            }
            a.macroO = DivHi(a.macroC, a.microC) * a.kA;

            a.elem = _elemD;
            a.bufS = (a.batch * a.srcH * a.srcW + a.padE) * a.srcC + a.microC * a.F;

            _stepS = p.srcH * p.srcW * p.srcC * a.batch * _elemS;
            _stepD = p.dstH * p.dstW * p.dstC * a.batch * _elemD;

            int dX = (int)a.microC, dY = (int)a.srcW * dX, dC = int(a.batch * a.srcH * a.srcW + a.padE) * dX;
            _offset.Resize(DivHi(a.K, a.microC));
            for (size_t c = 0, offsS = 0, i = 0; c < a.srcC; c += dX, offsS += dC)
                for (size_t y = 0, offsY = offsS; y < p.kernelY; y += 1, offsY += dY)
                    for (size_t offsX = offsY, endX = offsY + p.kernelX * dX; offsX < endX; offsX += dX, i++)
                        _offset[i] = (int)offsX;
        }

        bool SynetConvolution16bNhwcSpecV0::InvertedOrder() const
        {
            static int choise = 0;
            const ConvParam& p = _param;
            const AlgParam& a = _alg;
            if (a.microS != 32)
                return false;
            if (!p.IsKernel(3))
                return false;
            //if (p.srcC <= 64 || p.dstC <= 64)
            //    return false; 
            return false;// (choise++) & 1;
        }

        size_t SynetConvolution16bNhwcSpecV0::ExternalBufferSize() const
        {
            const AlgParam& a = _alg;
            size_t size = 0;
            size += a.bufS * sizeof(uint16_t);
            size += a.bufD * sizeof(float);
            return size;
        }

        void SynetConvolution16bNhwcSpecV0::SetParams(const float* weight, const float* bias, const float* params)
        {
            SetWeight(weight);
            SynetConvolution16b::SetBias(bias, _alg.microD);
            SynetConvolution16b::SetParams(params, _alg.microD);
        }

        void SynetConvolution16bNhwcSpecV0::SetWeight(const float* weight)
        {
            const ConvParam& p = _param;
            const AlgParam& a = _alg;
            _weight.Resize(a.K * a.dstC, true);
            uint16_t* dst = _weight.data;
            for (size_t mad = 0; mad < p.dstC; mad += _alg.F)
            {
                for (size_t mac = 0; mac < p.srcC; mac += a.microC)
                {
                    for (size_t k = 0; k < a.kA; k++)
                    {
                        for (size_t c = 0; c < a.microC; c += 2)
                        {
                            const float* src = weight + (k * p.srcC + mac + c) * p.dstC + mad;
                            for (size_t d = 0; d < a.F; ++d)
                            {
                                for (size_t i = 0; i < 2; ++i)
                                {
                                    if (mad + d < p.dstC && mac + c + i < p.srcC)
                                        *(dst++) = Float32ToBFloat16(src[i * p.dstC]);
                                    else
                                        *(dst++) = 0;
                                }
                                src++;
                            }
                         }
                    }
                }
            }
        }

        void SynetConvolution16bNhwcSpecV0::Forward(const uint8_t* src, uint8_t* buf8, uint8_t* dst)
        {
            const ConvParam& p = _param;
            const AlgParam& a = _alg;
            buf8 = Buffer(buf8);
            uint16_t* bufS = a.bufS ? Allocate<uint16_t>(buf8, a.bufS) : NULL;
            float* bufD = a.bufD ? Allocate<float>(buf8, a.bufD) : NULL;
            for (size_t b = 0; b < p.batch; b += a.batch)
            {
                uint16_t* buf = bufS ? bufS : (uint16_t*)src;
                float* sum = bufD ? bufD : (float*)dst;
                if(a.inv)
                    ForwardInverse(src, buf, sum, dst);
                else
                    ForwardDirect(src, buf, sum, dst);
                src += _stepS;
                dst += _stepD;
            }
        }

        void SynetConvolution16bNhwcSpecV0::ForwardDirect(const uint8_t* src, uint16_t* buf, float* sum, uint8_t* dst)
        {
            const ConvParam& p = _param;
            const AlgParam& a = _alg;
            const float* bias = _bias.data, * params = _params.data;
            const int* offs = _offset.data;
            size_t dstH = p.dstH * a.batch, dstHb = a.srcH * a.batch - a.gapV;
            size_t bufOffs = ((a.padV - p.padY) * a.srcW + (a.padH - p.padX)) * a.microC;
            for (size_t mad = 0; mad < p.dstC; mad += a.macroD)
            {
                size_t macroD = Simd::Min(p.dstC, mad + a.macroD) - mad;
                const uint16_t* weight = _weight.data + mad * a.K;
                for (size_t mac = 0, mao = 0; mac < a.srcC; mac += a.macroC, mao += a.macroO)
                {
                    size_t macroC = Simd::Min(a.srcC, mac + a.macroC) - mac;
                    size_t nK = DivHi(macroC, a.microC) * a.kA;
                    for (size_t dyBeg = 0, dyN = 0; dyBeg < dstH; dyN++)
                    {
                        size_t dyEnd = Simd::Min(dyBeg + a.macroH, dstH);
                        if (mad == 0 && mac == 0)
                        {
                            if (a.batch > 1)
                            {
                                size_t dS = p.srcH * p.srcW * p.srcC * _elemS;
                                size_t dB = a.srcH * a.srcW * a.microC;
                                for (size_t b = 0; b < a.batch; ++b)
                                    _preprocess(src + b * dS, p, a, 0, p.dstH, b == a.batch - 1 ? 1 : 0, buf + b * dB);
                            }
                            else
                                _preprocess(src, p, a, dyBeg, dyEnd, dyEnd == dstH ? 1 : 0, buf);
                        }
                        if (a.batch > 1)
                        {
                            _convolution(buf + bufOffs, p, a, offs + mao, macroD, dstHb, nK, mac == 0 ? 1 : 0, weight, sum);
                        }
                        else
                        {
                            _convolution(buf + bufOffs + dyBeg * a.srcW * a.microC, p, a, offs + mao, macroD, dyEnd - dyBeg,
                                nK, mac == 0 ? 1 : 0, weight, sum + (dyBeg * a.srcW + dyN * a.F) * a.macroD);
                        }
                        if (mac + macroC == a.srcC)
                        {
                            if (a.batch > 1)
                            {
                                size_t dS = a.srcH * a.srcW * a.macroD;
                                size_t dD = p.dstH * p.dstW * p.dstC * a.elem;
                                for (size_t b = 0; b < a.batch; ++b)
                                    _postprocess(sum + b * dS, p, a, macroD, 0, p.dstH, bias, params, dst + b * dD);
                            }
                            else
                                _postprocess(sum + dyN * a.F * a.macroD, p, a, macroD, dyBeg, dyEnd, bias, params, dst);
                        }
                        dyBeg = dyEnd;
                    }
                    weight += macroC * a.kA * a.F;
                }
                bias += macroD;
                if (p.activation == ::SimdConvolutionActivationPrelu)
                    params += macroD;
                dst += macroD * _elemD;
            }
        }

        void SynetConvolution16bNhwcSpecV0::ForwardInverse(const uint8_t* src, uint16_t* buf, float* sum, uint8_t* dst)
        {
            const ConvParam& p = _param;
            const AlgParam& a = _alg;
            const int* offs = _offset.data;
            size_t dstH = p.dstH * a.batch, dstHb = a.srcH * a.batch - a.gapV;
            size_t bufOffs = ((a.padV - p.padY) * a.srcW + (a.padH - p.padX)) * a.microC;
            for (size_t dyBeg = 0; dyBeg < dstH;)
            {
                size_t dyEnd = Simd::Min(dyBeg + a.macroH, dstH);
                if (a.batch > 1)
                {
                    size_t dS = p.srcH * p.srcW * p.srcC * _elemS;
                    size_t dB = a.srcH * a.srcW * a.microC;
                    for (size_t b = 0; b < a.batch; ++b)
                        _preprocess(src + b * dS, p, a, 0, p.dstH, b == a.batch - 1 ? 1 : 0, buf + b * dB);
                }
                else
                    _preprocess(src, p, a, dyBeg, dyEnd, dyEnd == dstH ? 1 : 0, buf);
                for (size_t mac = 0, mao = 0; mac < a.srcC; mac += a.macroC, mao += a.macroO)
                {
                    size_t macroC = Simd::Min(a.srcC, mac + a.macroC) - mac;
                    size_t nK = DivHi(macroC, a.microC) * a.kA;
                    const float* bias = _bias.data, * params = _params.data;
                    for (size_t mad = 0; mad < p.dstC; mad += a.macroD)
                    {
                        size_t macroD = Simd::Min(p.dstC, mad + a.macroD) - mad;
                        size_t sumOffs = ((a.macroH + a.batch - 1) * a.srcW + a.F) * mad;
                        const uint16_t* weight = _weight.data + mad * a.K + mac * a.kA * a.F;
                        if (a.batch > 1)
                        {
                            _convolution(buf + bufOffs, p, a, offs + mao, macroD, dstHb, nK, mac == 0 ? 1 : 0, weight, sum + sumOffs);
                        }
                        else
                        {
                            _convolution(buf + bufOffs + dyBeg * a.srcW * a.microC, p, a, offs + mao, macroD, dyEnd - dyBeg,
                                nK, mac == 0 ? 1 : 0, weight, sum + sumOffs);
                        }
                        if (mac + macroC == a.srcC)
                        {
                            if (a.batch > 1)
                            {
                                size_t dS = a.srcH * a.srcW * a.macroD;
                                size_t dD = p.dstH * p.dstW * p.dstC * a.elem;
                                for (size_t b = 0; b < a.batch; ++b)
                                    _postprocess(sum + sumOffs + b * dS, p, a, macroD, 0, p.dstH, bias, params, dst + b * dD + mad * _elemD);
                            }
                            else
                                _postprocess(sum + sumOffs, p, a, macroD, dyBeg, dyEnd, bias, params, dst + mad * _elemD);
                        }
                        bias += macroD;
                        if (p.activation == ::SimdConvolutionActivationPrelu)
                            params += macroD;
                    }
                }
                dyBeg = dyEnd;
            }
        }

        bool SynetConvolution16bNhwcSpecV0::Preferable(const ConvParam& p)
        {
            return p.trans != 0 && p.group == 1 && p.IsDilation(1) && p.IsStride(1) && !p.IsKernel(1) && p.dstC >= 4;
        }
    }
#endif
}
