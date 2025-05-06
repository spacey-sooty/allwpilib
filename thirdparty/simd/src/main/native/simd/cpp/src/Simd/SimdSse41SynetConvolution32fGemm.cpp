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
#include "Simd/SimdSynet.h"
#include "Simd/SimdGemm.h"
#include "Simd/SimdExp.h"
#include "Simd/SimdErf.h"
#include "Simd/SimdSse41.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdStore.h"

namespace Simd
{
#if defined(SIMD_SSE41_ENABLE) && defined(SIMD_SYNET_ENABLE) 
    namespace Sse41
    {
        void ConvolutionBiasAndActivation(const float* bias, size_t count, size_t size, ::SimdConvolutionActivationType activation, const float* params, ::SimdBool trans, float* dst)
        {
            size_t aligned = trans ? AlignLo(count, F) : AlignLo(size, F);
            if (activation == ::SimdConvolutionActivationIdentity)
            {
                if (bias)
                    Sse41::SynetAddBias(bias, count, size, dst, (SimdTensorFormatType)trans);
            }
            else if (activation == ::SimdConvolutionActivationRelu)
            {
                if (bias)
                {
                    __m128 _0 = _mm_set1_ps(0.0f);
                    if (trans)
                    {
                        for (size_t j = 0; j < size; ++j)
                        {
                            size_t i = 0;
                            for (; i < aligned; i += F)
                            {
                                __m128 _dst = _mm_loadu_ps(dst + i);
                                __m128 _bias = _mm_loadu_ps(bias + i);
                                _mm_storeu_ps(dst + i, _mm_max_ps(_0, _mm_add_ps(_dst, _bias)));
                            }
                            for (; i < count; ++i)
                                dst[i] = Simd::Max(0.0f, dst[i] + bias[i]);
                            dst += count;
                        }
                    }
                    else
                    {
                        for (size_t i = 0; i < count; ++i)
                        {
                            __m128 _bias = _mm_set1_ps(bias[i]);
                            size_t j = 0;
                            for (; j < aligned; j += F)
                            {
                                __m128 _dst = _mm_loadu_ps(dst + j);
                                _mm_storeu_ps(dst + j, _mm_max_ps(_0, _mm_add_ps(_dst, _bias)));
                            }
                            for (; j < size; ++j)
                                dst[j] = Simd::Max(0.0f, dst[j] + bias[i]);
                            dst += size;
                        }
                    }
                }
                else
                {
                    float slope = 0;
                    SynetRelu32f(dst, size * count, &slope, dst);
                }
            }
            else if (activation == ::SimdConvolutionActivationLeakyRelu)
            {
                float slope = params[0];
                if (bias)
                {
                    __m128 _slope = _mm_set1_ps(slope);
                    if (trans)
                    {
                        for (size_t j = 0; j < size; ++j)
                        {
                            size_t i = 0;
                            for (; i < aligned; i += F)
                            {
                                __m128 value = _mm_add_ps(_mm_loadu_ps(dst + i), _mm_loadu_ps(bias + i));
                                _mm_storeu_ps(dst + i, SynetRelu32f(value, _slope));
                            }
                            for (; i < count; ++i)
                                dst[i] = Base::SynetRelu32f(dst[i] + bias[i], slope);
                            dst += count;
                        }
                    }
                    else
                    {
                        for (size_t i = 0; i < count; ++i)
                        {
                            __m128 _bias = _mm_set1_ps(bias[i]);
                            size_t j = 0;
                            for (; j < aligned; j += F)
                            {
                                __m128 value = _mm_add_ps(_mm_loadu_ps(dst + j), _bias);
                                _mm_storeu_ps(dst + j, SynetRelu32f(value, _slope));
                            }
                            for (; j < size; ++j)
                                dst[j] = Base::SynetRelu32f(dst[j] + bias[i], slope);
                            dst += size;
                        }
                    }
                }
                else
                    SynetRelu32f(dst, size * count, &slope, dst);
            }
            else if (activation == ::SimdConvolutionActivationRestrictRange)
            {
                float lower = params[0];
                float upper = params[1];
                if (bias)
                {
                    __m128 _lower = _mm_set1_ps(lower);
                    __m128 _upper = _mm_set1_ps(upper);
                    if (trans)
                    {
                        for (size_t j = 0; j < size; ++j)
                        {
                            size_t i = 0;
                            for (; i < aligned; i += F)
                            {
                                __m128 value = _mm_add_ps(_mm_loadu_ps(dst + i), _mm_loadu_ps(bias + i));
                                _mm_storeu_ps(dst + i, _mm_min_ps(_mm_max_ps(_lower, value), _upper));
                            }
                            for (; i < count; ++i)
                                dst[i] = Simd::RestrictRange(dst[i] + bias[i], lower, upper);
                            dst += count;
                        }
                    }
                    else
                    {
                        for (size_t i = 0; i < count; ++i)
                        {
                            __m128 _bias = _mm_set1_ps(bias[i]);
                            size_t j = 0;
                            for (; j < aligned; j += F)
                            {
                                __m128 value = _mm_add_ps(_mm_loadu_ps(dst + j), _bias);
                                _mm_storeu_ps(dst + j, _mm_min_ps(_mm_max_ps(_lower, value), _upper));
                            }
                            for (; j < size; ++j)
                                dst[j] = Simd::RestrictRange(dst[j] + bias[i], lower, upper);
                            dst += size;
                        }
                    }
                }
                else
                    SynetRestrictRange32f(dst, size * count, &lower, &upper, dst);
            }
            else if (activation == ::SimdConvolutionActivationPrelu)
            {
                if (bias)
                {
                    if (trans)
                    {
                        for (size_t j = 0; j < size; ++j)
                        {
                            size_t i = 0;
                            for (; i < aligned; i += F)
                            {
                                __m128 value = _mm_add_ps(_mm_loadu_ps(dst + i), _mm_loadu_ps(bias + i));
                                _mm_storeu_ps(dst + i, SynetRelu32f(value, _mm_loadu_ps(params + i)));
                            }
                            for (; i < count; ++i)
                                dst[i] = Base::SynetRelu32f(dst[i] + bias[i], params[i]);
                            dst += count;
                        }
                    }
                    else
                    {
                        for (size_t i = 0; i < count; ++i)
                        {
                            __m128 _bias = _mm_set1_ps(bias[i]);
                            __m128 _slope = _mm_set1_ps(params[i]);
                            size_t j = 0;
                            for (; j < aligned; j += F)
                            {
                                __m128 value = _mm_add_ps(_mm_loadu_ps(dst + j), _bias);
                                _mm_storeu_ps(dst + j, SynetRelu32f(value, _slope));
                            }
                            for (; j < size; ++j)
                                dst[j] = Base::SynetRelu32f(dst[j] + bias[i], params[i]);
                            dst += size;
                        }
                    }
                }
                else
                    SynetPreluLayerForward(dst, params, count, size, dst, (SimdTensorFormatType)trans);
            }
            else if (activation == ::SimdConvolutionActivationElu)
            {
                float alpha = params[0];
                if (bias)
                {
                    __m128 _alpha = _mm_set1_ps(alpha);
                    if (trans)
                    {
                        for (size_t j = 0; j < size; ++j)
                        {
                            size_t i = 0;
                            for (; i < aligned; i += F)
                            {
                                __m128 value = _mm_add_ps(_mm_loadu_ps(dst + i), _mm_loadu_ps(bias + i));
                                _mm_storeu_ps(dst + i, Elu(value, _alpha));
                            }
                            for (; i < count; ++i)
                                dst[i] = Base::SynetElu32f(dst[i] + bias[i], alpha);
                            dst += count;
                        }
                    }
                    else
                    {
                        for (size_t i = 0; i < count; ++i)
                        {
                            __m128 _bias = _mm_set1_ps(bias[i]);
                            size_t j = 0;
                            for (; j < aligned; j += F)
                            {
                                __m128 value = _mm_add_ps(_mm_loadu_ps(dst + j), _bias);
                                _mm_storeu_ps(dst + j, Elu(value, _alpha));
                            }
                            for (; j < size; ++j)
                                dst[j] = Base::SynetElu32f(dst[j] + bias[i], alpha);
                            dst += size;
                        }
                    }
                }
                else
                    SynetElu32f(dst, size * count, &alpha, dst);
            }
            else if (activation == ::SimdConvolutionActivationHswish)
            {
                float shift = params[0];
                float scale = params[1];
                if (bias)
                {
                    __m128 _shift = _mm_set1_ps(shift);
                    __m128 _scale = _mm_set1_ps(scale);
                    if (trans)
                    {
                        for (size_t j = 0; j < size; ++j)
                        {
                            size_t i = 0;
                            for (; i < aligned; i += F)
                            {
                                __m128 value = _mm_add_ps(Load<false>(dst + i), Load<false>(bias + i));
                                Store<false>(dst + i, SynetHswish32f(value, _shift, _scale));
                            }
                            for (; i < count; ++i)
                                dst[i] = Base::SynetHswish32f(dst[i] + bias[i], shift, scale);
                            dst += count;
                        }
                    }
                    else
                    {
                        for (size_t i = 0; i < count; ++i)
                        {
                            __m128 _bias = _mm_set1_ps(bias[i]);
                            size_t j = 0;
                            for (; j < aligned; j += F)
                            {
                                __m128 value = _mm_add_ps(Load<false>(dst + j), _bias);
                                Store<false>(dst + j, SynetHswish32f(value, _shift, _scale));
                            }
                            for (; j < size; ++j)
                                dst[j] = Base::SynetHswish32f(dst[j] + bias[i], shift, scale);
                            dst += size;
                        }
                    }
                }
                else
                    SynetHswish32f(dst, count * size, &shift, &scale, dst);
            }
            else if (activation == ::SimdConvolutionActivationMish)
            {
                float threshold = params[0];
                if (bias)
                {
                    __m128 _threshold = _mm_set1_ps(threshold);
                    if (trans)
                    {
                        for (size_t j = 0; j < size; ++j)
                        {
                            size_t i = 0;
                            for (; i < aligned; i += F)
                            {
                                __m128 value = _mm_add_ps(Load<false>(dst + i), Load<false>(bias + i));
                                Store<false>(dst + i, Mish(value, _threshold));
                            }
                            for (; i < count; ++i)
                                dst[i] = Base::SynetMish32f(dst[i] + bias[i], threshold);
                            dst += count;
                        }
                    }
                    else
                    {
                        for (size_t i = 0; i < count; ++i)
                        {
                            __m128 _bias = _mm_set1_ps(bias[i]);
                            size_t j = 0;
                            for (; j < aligned; j += F)
                            {
                                __m128 value = _mm_add_ps(Load<false>(dst + j), _bias);
                                Store<false>(dst + j, Mish(value, _threshold));
                            }
                            for (; j < size; ++j)
                                dst[j] = Base::SynetMish32f(dst[j] + bias[i], threshold);
                            dst += size;
                        }
                    }
                }
                else
                    SynetMish32f(dst, size * count, &threshold, dst);
            }
            else if (activation == ::SimdConvolutionActivationHardSigmoid)
            {
                float scale = params[0];
                float shift = params[1];
                if (bias)
                {
                    __m128 _scale = _mm_set1_ps(scale);
                    __m128 _shift = _mm_set1_ps(shift);
                    if (trans)
                    {
                        for (size_t j = 0; j < size; ++j)
                        {
                            size_t i = 0;
                            for (; i < aligned; i += F)
                            {
                                __m128 value = _mm_add_ps(Load<false>(dst + i), Load<false>(bias + i));
                                Store<false>(dst + i, SynetHardSigmoid32f(value, _scale, _shift));
                            }
                            for (; i < count; ++i)
                                dst[i] = Base::SynetHardSigmoid32f(dst[i] + bias[i], scale, shift);
                            dst += count;
                        }
                    }
                    else
                    {
                        for (size_t i = 0; i < count; ++i)
                        {
                            __m128 _bias = _mm_set1_ps(bias[i]);
                            size_t j = 0;
                            for (; j < aligned; j += F)
                            {
                                __m128 value = _mm_add_ps(Load<false>(dst + j), _bias);
                                Store<false>(dst + j, SynetHardSigmoid32f(value, _scale, _shift));
                            }
                            for (; j < size; ++j)
                                dst[j] = Base::SynetHardSigmoid32f(dst[j] + bias[i], scale, shift);
                            dst += size;
                        }
                    }
                }
                else
                    SynetHardSigmoid32f(dst, count * size, &scale, &shift, dst);
            }
            else if (activation == ::SimdConvolutionActivationSwish)
            {
                float slope = params[0];
                if (bias)
                {
                    __m128 _slope = _mm_set1_ps(slope);
                    if (trans)
                    {
                        for (size_t j = 0; j < size; ++j)
                        {
                            size_t i = 0;
                            for (; i < aligned; i += F)
                            {
                                __m128 value = _mm_add_ps(Load<false>(dst + i), Load<false>(bias + i));
                                Store<false>(dst + i, Swish(value, _slope));
                            }
                            for (; i < count; ++i)
                                dst[i] = Base::SynetSwish32f(dst[i] + bias[i], slope);
                            dst += count;
                        }
                    }
                    else
                    {
                        for (size_t i = 0; i < count; ++i)
                        {
                            __m128 _bias = _mm_set1_ps(bias[i]);
                            size_t j = 0;
                            for (; j < aligned; j += F)
                            {
                                __m128 value = _mm_add_ps(Load<false>(dst + j), _bias);
                                Store<false>(dst + j, Swish(value, _slope));
                            }
                            for (; j < size; ++j)
                                dst[j] = Base::SynetSwish32f(dst[j] + bias[i], slope);
                            dst += size;
                        }
                    }
                }
                else
                    SynetSwish32f(dst, count * size, &slope, dst);
            }
            else if (activation == ::SimdConvolutionActivationGelu)
            {
                if (bias)
                {
                    if (trans)
                    {
                        for (size_t j = 0; j < size; ++j)
                        {
                            size_t i = 0;
                            for (; i < aligned; i += F)
                            {
                                __m128 value = _mm_add_ps(Load<false>(dst + i), Load<false>(bias + i));
                                Store<false>(dst + i, Gelu(value));
                            }
                            for (; i < count; ++i)
                                dst[i] = Base::Gelu(dst[i] + bias[i]);
                            dst += count;
                        }
                    }
                    else
                    {
                        for (size_t i = 0; i < count; ++i)
                        {
                            __m128 _bias = _mm_set1_ps(bias[i]);
                            size_t j = 0;
                            for (; j < aligned; j += F)
                            {
                                __m128 value = _mm_add_ps(Load<false>(dst + j), _bias);
                                Store<false>(dst + j, Gelu(value));
                            }
                            for (; j < size; ++j)
                                dst[j] = Base::Gelu(dst[j] + bias[i]);
                            dst += size;
                        }
                    }
                }
                else
                    SynetGelu32f(dst, count * size, dst);
            }
            else
            {
                Base::ConvolutionBiasAndActivation(bias, count, size, activation, params, trans, dst);
            }
        }

        //-------------------------------------------------------------------------------------------------

        SynetConvolution32fGemmNN::SynetConvolution32fGemmNN(const ConvParam& p)
            : Base::SynetConvolution32fGemmNN(p)
        {
            _gemm.Init(InitGemmFuncs(Sse41::Gemm32fNN, "Sse41"));
            if (_param.trans && _param.group == 1)
            {
                if (GemmRuntime())
                {
                    _gemmCb.Init(InitGemmCbFuncs(Sse41::Gemm32fNNcbBufferSize, Sse41::Gemm32fNNcbReorderB, Sse41::Gemm32fNNcbRun, "Sse41", GemmKernelF2, GemmKernelF3));
                    _nhwcWeight.Resize(_gemmCb.At(0).BufferSize(_M * _merge, _N, _K));
                }
                else
                    _nhwcWeight.Resize(Sse41::Gemm32fNNcbBufferSize(_M * _merge, _N, _K, GemmKernelAny, NHWC_GEMM_COMPATIBLE));
                _nhwcRun = Sse41::Gemm32fNNcbRun;
                _nhwcReorderB = Sse41::Gemm32fNNcbReorderB;
            }
            _biasAndActivation = Sse41::ConvolutionBiasAndActivation;
        }

        //-------------------------------------------------------------------------------------------------

        SynetConvolution32fGemmNT::SynetConvolution32fGemmNT(const ConvParam& p)
            : Base::SynetConvolution32fGemmNT(p)
        {
            _gemm.Init(InitGemmFuncs(Sse41::Gemm32fNT, "Sse41"));
            _biasAndActivation = Sse41::ConvolutionBiasAndActivation;
        }

        //-------------------------------------------------------------------------------------------------

        SynetConvolution32fWinograd::SynetConvolution32fWinograd(const ConvParam& p)
            : Base::SynetConvolution32fWinograd(p)
        {
            if (p.kernelY == 1 && p.kernelX == 3)
            {
                {
                    SetBlock(1, 4);
                    _setFilter = Sse41::WinogradKernel1x3Block1x4SetFilter;
                    _setInput = Sse41::WinogradKernel1x3Block1x4SetInput;
                    _setOutput = Sse41::WinogradKernel1x3Block1x4SetOutput;
                }
            }
            else if (p.kernelY == 1 && p.kernelX == 5)
            {
                {
                    SetBlock(1, 4);
                    _setFilter = Sse41::WinogradKernel1x5Block1x4SetFilter;
                    _setInput = Sse41::WinogradKernel1x5Block1x4SetInput;
                    _setOutput = Sse41::WinogradKernel1x5Block1x4SetOutput;
                }
            }
            else if (p.kernelY == 2 && p.kernelX == 2)
            {
                if (_blockY == 4 && _blockX == 4)
                {
                    SetBlock(4, 4);
                    _setFilter = Sse41::WinogradKernel2x2Block4x4SetFilter;
                    _setInput = Sse41::WinogradKernel2x2Block4x4SetInput;
                    _setOutput = Sse41::WinogradKernel2x2Block4x4SetOutput;
                }
                else if (_blockY == 2 && _blockX == 2)
                {
                    SetBlock(2, 2);
                    _setFilter = Sse41::WinogradKernel2x2Block2x2SetFilter;
                    _setInput = Sse41::WinogradKernel2x2Block2x2SetInput;
                    _setOutput = Sse41::WinogradKernel2x2Block2x2SetOutput;
                }
                else
                    assert(0);
            }
            else if (p.kernelY == 3 && p.kernelX == 3)
            {
                if (_blockY == 4 && _blockX == 4)
                {
                    _setFilter = Sse41::WinogradKernel3x3Block4x4SetFilter;
                    _setInput = Sse41::WinogradKernel3x3Block4x4SetInput;
                    _setOutput = Sse41::WinogradKernel3x3Block4x4SetOutput;
                }
                else if (_blockY == 3 && _blockX == 3)
                {
                    _setFilter = Sse41::WinogradKernel3x3Block3x3SetFilter;
                    _setInput = Sse41::WinogradKernel3x3Block3x3SetInput;
                    _setOutput = Sse41::WinogradKernel3x3Block3x3SetOutput;
                }
                else if (_blockY == 2 && _blockX == 2)
                {
                    _setFilter = Sse41::WinogradKernel3x3Block2x2SetFilter;
                    _setInput = Sse41::WinogradKernel3x3Block2x2SetInput;
                    _setOutput = Sse41::WinogradKernel3x3Block2x2SetOutput;
                }
                else
                    assert(0);
            }
            else
                assert(0);
            _gemm.Init(InitGemmFuncs(Sse41::Gemm32fNN, "Sse41"));
            if (_param.trans)
            {
                if (NHWC_GEMM_RUNTIME)
                {
                    _gemmCb.Init(InitGemmCbFuncs(Sse41::Gemm32fNNcbBufferSize, Sse41::Gemm32fNNcbReorderB, Sse41::Gemm32fNNcbRun, "Sse41", GemmKernelF2, GemmKernelF3));
                    _nhwcStrideW = _gemmCb.At(0).BufferSize(_M * _merge, _N, _K);
                }
                else
                    _nhwcStrideW = Sse41::Gemm32fNNcbBufferSize(_M * _merge, _N, _K, GemmKernelAny, NHWC_GEMM_COMPATIBLE);
                _nhwcWeight.Resize(_nhwcStrideW * _count);
                _nhwcRun = Sse41::Gemm32fNNcbRun;
                _nhwcReorderB = Sse41::Gemm32fNNcbReorderB;
            }
            _biasAndActivation = Sse41::ConvolutionBiasAndActivation;
        }
    }
#endif
}
