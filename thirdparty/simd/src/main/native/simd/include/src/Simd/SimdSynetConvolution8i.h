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
#ifndef __SimdSynetConvolution8i_h__
#define __SimdSynetConvolution8i_h__

#include "Simd/SimdSynetConvParam.h"
#include "Simd/SimdArray.h"
#include "Simd/SimdPerformance.h"

#ifdef _N
#undef _N
#endif

namespace Simd
{
    struct CvtParam
    {
        Array8u zero;
        Array32f scale, shift, iScale, iShift;
        bool neg;
        int iMin, iMax, uMin, uMax;

        CvtParam() 
            : neg(false) 
        {
        }

        void Init(const float* min, const float* max, size_t size, SimdSynetCompatibilityType compatibility);

        size_t Size() const
        {
            return (zero.size) * sizeof(uint8_t) + (scale.size + shift.size + iScale.size + iShift.size) * sizeof(float);
        }
    };

    class SynetConvolution8i : public Deletable
    {
    public:
        SynetConvolution8i(const ConvParam& p);

        const ConvParam & Param() const { return _param; }

        virtual String Ext() const = 0;
        virtual String Desc() const = 0;

        virtual size_t ExternalBufferSize() const;
        virtual size_t InternalBufferSize() const;

        virtual void SetParams(const float* weight, const float* bias, const float* params, const float* const* stats);

        virtual void Forward(const uint8_t * src, uint8_t * buf, uint8_t * dst);

#if defined(SIMD_PERFORMANCE_STATISTIC) && (defined(NDEBUG) || defined(SIMD_PERF_STAT_IN_DEBUG))
        Base::PerformanceMeasurer* Perf(const char* func);
#endif

        const char* Info() const
        {
            _info = Desc();
            return _info.c_str();
        }

    protected:
        virtual void Forward8u(const uint8_t* src, uint8_t* buf, uint8_t* dst) = 0;

        typedef void(*Convert32fTo8u)(const float* src, size_t batch, size_t channels, size_t height, size_t width, SimdTensorFormatType format, const float* scale, const float* shift, uint8_t* dst, SimdSynetCompatibilityType compatibility);

        ConvParam _param;
        Array8u _buffer;
#if defined(SIMD_PERFORMANCE_STATISTIC) && (defined(NDEBUG) || defined(SIMD_PERF_STAT_IN_DEBUG))
        Base::PerformanceMeasurer * _perf;
#endif
        mutable String _info;
        Convert32fTo8u _convertSrc;
        CvtParam _srcCvt, _dstCvt;
        Array8i _weight;
        Array32f _norm, _bias, _params; 
        bool _src8u, _dst8u;
        size_t _merge, _sizeS, _sizeD;
    };

    namespace Base
    {
        class SynetConvolution8iGemmNN : public SynetConvolution8i
        {
        public:
            SynetConvolution8iGemmNN(const ConvParam & p);
            virtual String Ext() const { return "Base"; }
            virtual String Desc() const { return Ext() + "::GemmNN"; }
            virtual size_t ExternalBufferSize() const;

        protected:
            virtual void Forward8u(const uint8_t* src, uint8_t* buf, uint8_t* dst);

            bool _skipConv;
            size_t _ldW, _ldS, _ldD, _grW, _grS, _grD, _siC, _siK, _siS, _siD, _sizeB;
        };

        class SynetConvolution8iNhwcDirect : public SynetConvolution8i
        {
        public:
            SynetConvolution8iNhwcDirect(const ConvParam& p);
            virtual String Ext() const { return "Base"; }
            virtual String Desc() const;
            virtual size_t InternalBufferSize() const;
            virtual size_t ExternalBufferSize() const;
            virtual void SetParams(const float* weight, const float* bias, const float* params, const float* const* stats);

            static bool Preferable(const ConvParam& p);

            struct AlgParam
            {
                size_t F, microD, macroH, macroC, macroD;
                int32_t zero, size, upper;
            };

            typedef void(*ConvolutionPtr)(const uint8_t* src, const ConvParam& p, const AlgParam& a, size_t dstC, size_t yBeg, size_t yEnd, size_t srcC, 
                const int8_t* weight, const float* norm, const float* bias, const float* params, const float* scale, const float* shift, int32_t* buf, uint8_t* dst, int first);

        protected:
            void SetAlgParam(size_t F, size_t microD, size_t microHW, size_t L1, size_t L2, size_t L3);
            void ReorderWeight();
            bool PadEnable(size_t microHW);
            void PadInput(const uint8_t* src, uint8_t* dst);

            virtual void Forward8u(const uint8_t* src, uint8_t* buf, uint8_t* dst);
            void Forward8u(const uint8_t* src, const ConvParam & p, int32_t* buf, uint8_t* dst);

            AlgParam _alg;
            size_t _sizeP, _sizeB;
            ConvParam _paramP;
            ConvolutionPtr _convolutions[3];
        };

        class SynetConvolution8iNhwcDepthwise : public SynetConvolution8i
        {
        public:
            SynetConvolution8iNhwcDepthwise(const ConvParam& p);
            virtual String Ext() const { return "Base"; }
            virtual String Desc() const;
            virtual void SetParams(const float* weight, const float* bias, const float* params, const float* const* stats);

            static bool Preferable(const ConvParam& p);

            struct AlgParam
            {
                int32_t zero, size, upper;
            };

            typedef void(*ConvolutionPtr)(const uint8_t* src, const ConvParam& p, const AlgParam& a, const int8_t* weight, 
                const float* norm, const float* bias, const float* params, const float* scale, const float* shift, uint8_t* dst);

        protected:

            virtual void Forward8u(const uint8_t* src, uint8_t* buf, uint8_t* dst);

            AlgParam _alg;
            ConvolutionPtr _convolution;
        };

        void * SynetConvolution8iInit(size_t batch, const SimdConvolutionParameters * conv, SimdSynetCompatibilityType compatibility);
    }

#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        class SynetConvolution8iNhwcDirect : public Base::SynetConvolution8iNhwcDirect
        {
        public:
            SynetConvolution8iNhwcDirect(const ConvParam& p);

            virtual String Ext() const { return "Sse41"; }

            static bool Preferable(const ConvParam& p);
        };

        void SetDirectAny(const ConvParam& p, const SynetConvolution8iNhwcDirect::AlgParam& a, SynetConvolution8iNhwcDirect::ConvolutionPtr* d);

        void SetDirect1x1(const ConvParam& p, const SynetConvolution8iNhwcDirect::AlgParam& a, SynetConvolution8iNhwcDirect::ConvolutionPtr* d);

#if defined(SIMD_INT8_DEBUG_ENABLE)
        class SynetConvolution8iNhwcDepthwise : public Base::SynetConvolution8iNhwcDepthwise
        {
        public:
            SynetConvolution8iNhwcDepthwise(const ConvParam& p);

            virtual String Ext() const { return "Sse41"; }

            static bool Preferable(const ConvParam& p);
        };
#endif

        void* SynetConvolution8iInit(size_t batch, const SimdConvolutionParameters* conv, SimdSynetCompatibilityType compatibility);
    }
#endif

#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        class SynetConvolution8iNhwcDirect : public Sse41::SynetConvolution8iNhwcDirect
        {
        public:
            SynetConvolution8iNhwcDirect(const ConvParam& p);

            virtual String Ext() const { return "Avx2"; }
        };

        void SetDirectAny(const ConvParam& p, const SynetConvolution8iNhwcDirect::AlgParam& a, SynetConvolution8iNhwcDirect::ConvolutionPtr* d);

        void SetDirect1x1(const ConvParam& p, const SynetConvolution8iNhwcDirect::AlgParam& a, SynetConvolution8iNhwcDirect::ConvolutionPtr* d);

#if defined(SIMD_INT8_DEBUG_ENABLE)
        class SynetConvolution8iNhwcDepthwise : public Sse41::SynetConvolution8iNhwcDepthwise
        {
        public:
            SynetConvolution8iNhwcDepthwise(const ConvParam& p);

            virtual String Ext() const { return "Avx2"; }
        };
#endif

        void* SynetConvolution8iInit(size_t batch, const SimdConvolutionParameters* conv, SimdSynetCompatibilityType compatibility);
    }
#endif

#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        class SynetConvolution8iNhwcDirect : public Avx2::SynetConvolution8iNhwcDirect
        {
        public:
            SynetConvolution8iNhwcDirect(const ConvParam& p);

            virtual String Ext() const { return "Avx512bw"; }
        };

        void SetDirectAny(const ConvParam& p, const SynetConvolution8iNhwcDirect::AlgParam& a, SynetConvolution8iNhwcDirect::ConvolutionPtr* d);

        void SetDirect1x1(const ConvParam& p, const SynetConvolution8iNhwcDirect::AlgParam& a, SynetConvolution8iNhwcDirect::ConvolutionPtr* d);

#if defined(SIMD_INT8_DEBUG_ENABLE)
        class SynetConvolution8iNhwcDepthwise : public Avx2::SynetConvolution8iNhwcDepthwise
        {
        public:
            SynetConvolution8iNhwcDepthwise(const ConvParam& p);

            virtual String Ext() const { return "Avx512bw"; }
        };
#endif

        void* SynetConvolution8iInit(size_t batch, const SimdConvolutionParameters* conv, SimdSynetCompatibilityType compatibility);
    }
#endif

#ifdef SIMD_AVX512VNNI_ENABLE    
    namespace Avx512vnni
    {
        class SynetConvolution8iNhwcDirect : public Avx512bw::SynetConvolution8iNhwcDirect
        {
        public:
            SynetConvolution8iNhwcDirect(const ConvParam& p);

            virtual String Ext() const { return "Avx512vnni"; }
        };

        void SetDirectAny(const ConvParam& p, const SynetConvolution8iNhwcDirect::AlgParam& a, SynetConvolution8iNhwcDirect::ConvolutionPtr* d);

        void SetDirect1x1(const ConvParam& p, const SynetConvolution8iNhwcDirect::AlgParam& a, SynetConvolution8iNhwcDirect::ConvolutionPtr* d);

#if defined(SIMD_INT8_DEBUG_ENABLE)
        class SynetConvolution8iNhwcDepthwise : public Avx512bw::SynetConvolution8iNhwcDepthwise
        {
        public:
            SynetConvolution8iNhwcDepthwise(const ConvParam& p);

            virtual String Ext() const { return "Avx512vnni"; }
        };
#endif

        void* SynetConvolution8iInit(size_t batch, const SimdConvolutionParameters* conv, SimdSynetCompatibilityType compatibility);
    }
#endif

#if defined(SIMD_AMXBF16_ENABLE) || (defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_AMX_EMULATE))    
    namespace AmxBf16
    {
#if defined(SIMD_AMX_EMULATE)
        class SynetConvolution8iNhwcDirect : public Avx512bw::SynetConvolution8iNhwcDirect
#else
        class SynetConvolution8iNhwcDirect : public Avx512vnni::SynetConvolution8iNhwcDirect
#endif
        {
        public:
            SynetConvolution8iNhwcDirect(const ConvParam& p);

            virtual String Ext() const { return "AmxBf16"; }
        };

        void SetDirectAny(const ConvParam& p, const SynetConvolution8iNhwcDirect::AlgParam& a, SynetConvolution8iNhwcDirect::ConvolutionPtr* d);

        void SetDirect1x1(const ConvParam& p, const SynetConvolution8iNhwcDirect::AlgParam& a, SynetConvolution8iNhwcDirect::ConvolutionPtr* d);

        void* SynetConvolution8iInit(size_t batch, const SimdConvolutionParameters* conv, SimdSynetCompatibilityType compatibility);
    }
#endif

#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
        class SynetConvolution8iNhwcDirect : public Base::SynetConvolution8iNhwcDirect
        {
        public:
            SynetConvolution8iNhwcDirect(const ConvParam& p);

            virtual String Ext() const { return "Neon"; }

            static bool Preferable(const ConvParam& p);
        };

        void* SynetConvolution8iInit(size_t batch, const SimdConvolutionParameters* conv, SimdSynetCompatibilityType compatibility);
    }
#endif
}

#endif//__SimdSynetConvolution8i_h__
