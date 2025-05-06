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
#include "Simd/SimdSynetConvolution8i.h"
#include "Simd/SimdSynetConvolution8iCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdAvx512bw.h"
#include "Simd/SimdAmxBf16.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#if (defined(SIMD_AMXBF16_ENABLE) || (defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_AMX_EMULATE))) && defined(SIMD_SYNET_ENABLE)  
    namespace AmxBf16
    {
        SynetConvolution8iNhwcDirect::SynetConvolution8iNhwcDirect(const ConvParam& p)
#if defined(SIMD_AMX_EMULATE)
            : Avx512bw::SynetConvolution8iNhwcDirect(p)
#else
            : Avx512vnni::SynetConvolution8iNhwcDirect(p)
#endif
        {
            const size_t microC = 16 * 4, microHW = 2 * F;
#if !defined(SIMD_AMX_EMULATE)
            if(p.srcC < 2 * microC)
                return;
#endif
            SetAlgParam(F, 2 * F, microHW, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3());
            if (p.Is1x1())
                SetDirect1x1(p, _alg, _convolutions);
            else
                SetDirectAny(p, _alg, _convolutions);
            _convertSrc = Avx512bw::SynetConvert32fTo8u;
        }

        //---------------------------------------------------------------------

        void * SynetConvolution8iInit(size_t batch, const SimdConvolutionParameters * conv, SimdSynetCompatibilityType compatibility)
        {
            ConvParam param(batch, conv, compatibility);
            if (!param.Valid(SimdTensorData32f, SimdTensorData8u))
                return NULL;
#if defined(SIMD_INT8_DEBUG_ENABLE)
            else if (Avx512vnni::SynetConvolution8iNhwcDepthwise::Preferable(param))
                return new Avx512vnni::SynetConvolution8iNhwcDepthwise(param);
#endif
            else if (SynetConvolution8iNhwcDirect::Preferable(param))
                return new SynetConvolution8iNhwcDirect(param);
            else
                return new Base::SynetConvolution8iGemmNN(param);
        }
    }
#endif
}
