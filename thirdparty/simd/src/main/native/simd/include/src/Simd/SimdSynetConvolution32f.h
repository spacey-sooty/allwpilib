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
#ifndef __SimdSynetConvolution32f_h__
#define __SimdSynetConvolution32f_h__

#include "Simd/SimdArray.h"
#include "Simd/SimdRuntime.h"
#include "Simd/SimdGemm.h"
#include "Simd/SimdSynetConvParam.h"

#ifdef _N
#undef _N
#endif

namespace Simd
{
    const bool NHWC_GEMM_COMPATIBLE = false;
#if defined(SIMD_RUNTIME_DISABLE)
    const bool NHWC_GEMM_RUNTIME = false;
#else
    const bool NHWC_GEMM_RUNTIME = true;
#endif

    //-------------------------------------------------------------------------------------------------

    class SynetConvolution32f : public Deletable
    {
    public:
        SynetConvolution32f(const ConvParam & p) 
            : _param(p)
            , _0(0.0f)
            , _1(1.0f)
            , _nhwcRun(0)
            , _nhwcReorderB(0)
            , _biasAndActivation(0)
#if defined(SIMD_PERFORMANCE_STATISTIC) && (defined(NDEBUG) || defined(SIMD_PERF_STAT_IN_DEBUG))
            , _perf(NULL)
#endif
        {
        }

        const ConvParam & Param() const 
        {
            return _param;
        }

        virtual String Ext() const = 0;
        virtual String Desc() const = 0;

        virtual size_t ExternalBufferSize() const
        {
            return 1;
        }

        virtual size_t InternalBufferSize() const
        {
            return _buffer.size + _nhwcWeight.size;
        }

        virtual void SetParams(const float * weight, SimdBool * internal, const float * bias, const float * params)
        {
            _weight = weight;
            if (internal)
                *internal = SimdFalse;
            _bias = bias;
            _params = params;
        }

        virtual void Forward(const float * src, float * buf, float * dst) = 0;

        float * Buffer(float * buffer)
        {
            if (buffer)
                return buffer;
            else
            {
                _buffer.Resize(ExternalBufferSize());
                return _buffer.data;
            }
        }

#if defined(SIMD_PERFORMANCE_STATISTIC) && (defined(NDEBUG) || defined(SIMD_PERF_STAT_IN_DEBUG))
        Base::PerformanceMeasurer* Perf(const char* func)
        {
            if (_perf == NULL)
                _perf = Simd::Base::PerformanceMeasurerStorage::s_storage.Get(func, Param().Info() + " " + Desc(), Param().Flop());
            return _perf;
        }
#endif

        const char* Info() const
        {
            _info = Desc();
            return _info.c_str();
        }

    protected:
        typedef void(*NhwcReorderB)(size_t M, size_t N, size_t K, const float * B, float * pB, GemmKernelType type, bool compatibility);
        typedef void(*NhwcRun)(size_t M, size_t N, size_t K, const float * A, const float * B, float * C, GemmKernelType type, bool compatibility);
        typedef void(*BiasAndActivation)(const float * bias, size_t count, size_t size, ::SimdConvolutionActivationType activation, const float * params, SimdBool trans, float * dst);

        ConvParam _param;
        Array32f _buffer;
        float _0, _1;
        const float * _weight, * _bias, * _params;
        RuntimeGemm _gemm;
        RuntimeGemmCb _gemmCb;
        Array32f _nhwcWeight;
        NhwcRun _nhwcRun;
        NhwcReorderB _nhwcReorderB;
        BiasAndActivation _biasAndActivation;
#if defined(SIMD_PERFORMANCE_STATISTIC) && (defined(NDEBUG) || defined(SIMD_PERF_STAT_IN_DEBUG))
        Base::PerformanceMeasurer * _perf;
#endif
        mutable String _info;
    };

    //-------------------------------------------------------------------------------------------------

    namespace Base
    {
        void ConvolutionBiasAndActivation(const float * bias, size_t count, size_t size, ::SimdConvolutionActivationType activation, const float * params, SimdBool trans, float * dst);

        //-------------------------------------------------------------------------------------------------

        class SynetConvolution32fGemmNN : public SynetConvolution32f
        {
        public:
            SynetConvolution32fGemmNN(const ConvParam & p);
            virtual String Ext() const { return "Base"; }
            virtual String Desc() const { return Ext() + "::GemmNN" + (_merge > 1 ? "-" + ToStr(_merge) : ""); }
            virtual size_t ExternalBufferSize() const;
            virtual void SetParams(const float * weight, SimdBool * internal, const float * bias, const float * params);
            virtual void Forward(const float * src, float * buf, float * dst);

        protected:
            virtual void ImgToCol(const float * src, float * dst);
            virtual void ImgToRow(const float * src, float * dst);
            bool GemmRuntime() const;

            bool _skipConv;
            size_t _M, _N, _K, _ldW, _ldS, _ldD, _grW, _grS, _grD, _batch, _sizeS, _sizeB, _sizeD, _merge;
        };

        //-------------------------------------------------------------------------------------------------

        class SynetConvolution32fGemmNT : public SynetConvolution32f
        {
        public:
            SynetConvolution32fGemmNT(const ConvParam & p);
            virtual String Ext() const { return "Base"; }
            virtual String Desc() const { return Ext() + "::GemmNT"; }
            virtual size_t ExternalBufferSize() const;
            virtual void Forward(const float * src, float * buf, float * dst);

            static bool Preferable(const ConvParam & p);

        protected:
            static void ImgToRow(const float * src, const ConvParam & p, float * dst);

            size_t _M, _N, _K, _batch, _sizeS, _sizeB, _sizeD;
        };

        //-------------------------------------------------------------------------------------------------

        class SynetConvolution32fWinograd : public SynetConvolution32f
        {
        public:
            SynetConvolution32fWinograd(const ConvParam & p);
            virtual String Ext() const { return "Base"; }
            virtual String Desc() const;
            virtual size_t ExternalBufferSize() const;
            virtual size_t InternalBufferSize() const;
            virtual void SetParams(const float * weight, SimdBool * internal, const float * bias, const float * params);
            virtual void Forward(const float * src, float * buf, float * dst);

            static bool Preferable(const ConvParam & p);

        protected:
            typedef void(*SetFilter)(const float * src, size_t size, float * dst, SimdBool trans);
            typedef void(*SetInput)(const float* src, size_t srcChannels, size_t srcHeight, size_t srcWidth, size_t padY, size_t padX, size_t padH, size_t padW, float* dst, size_t dstStride, SimdBool trans);
            typedef void(*SetOutput)(const float * src, size_t srcStride, float * dst, size_t dstChannels, size_t dstHeight, size_t dstWidth, SimdBool trans);

            void SetBlock(size_t blockY, size_t blockX);
            void ForwardMerged(const float * src, float * bufS, float * bufD, float * dst);
            void ForwardSplitted(const float * src, float * bufS, float * bufD, float * dst);
#ifdef SIMD_PERFORMANCE_STATISTIC
            long long RealFlop() const
            {
                const ConvParam & p = _param;
                return p.batch * _count * p.srcC * _tileH * _tileW * p.dstC * 2 ;
            }
#endif
            size_t _count, _blockY, _blockX, _tileH, _tileW, _strideW, _strideS, _strideD, _M, _N, _K, _batch, _sizeS, _sizeD, _nhwcStrideW, _merge, _split, _tileHs;
            Array32f _winogradWeight;
            SetFilter _setFilter;
            SetInput _setInput;
            SetOutput _setOutput;
        };

        //-------------------------------------------------------------------------------------------------

        class SynetConvolution32fDirectNchw : public SynetConvolution32f
        {
        public:
            SynetConvolution32fDirectNchw(const ConvParam & p);
            virtual String Ext() const { return "Base"; }
            virtual String Desc() const { return Ext() + "::DirectNchw"; }
            virtual size_t ExternalBufferSize() const;
            virtual void Forward(const float * src, float * buf, float * dst);

            static bool Preferable(const ConvParam & p);

            typedef void(*ConvolutionBiasActivationPtr)(const float * src, size_t srcC, size_t srcH, size_t srcW, const float * weight, const float * bias, const float * params, float * dst, size_t dstC, size_t dstH, size_t dstW);
        protected:
            void Pad(const float * src, float * dst) const;
            virtual ConvolutionBiasActivationPtr SetConvolutionBiasActivation();

            size_t _grW, _grS, _grD, _srcC, _srcH, _srcW, _dstC;
            int _pad;
            ConvolutionBiasActivationPtr _convolutionBiasActivation;
        };

        //-------------------------------------------------------------------------------------------------

        class SynetConvolution32fNhwcDepthwise : public SynetConvolution32f
        {
        public:
            SynetConvolution32fNhwcDepthwise(const ConvParam & p);
            virtual String Ext() const { return "Base"; }
            virtual String Desc() const { return Ext() + "::NhwcDepthwise"; }
            virtual void Forward(const float * src, float * buf, float * dst);

            static bool Preferable(const ConvParam & p);

            typedef void(*ConvolutionPtr)(const float * src, const ConvParam & p, const float * weight, const float * bias, const float * params, float * dst);
        protected:

            size_t _batch, _sizeS, _sizeD;
            ConvolutionPtr _convolution;
        };

        //-------------------------------------------------------------------------------------------------

        class SynetConvolution32fDepthwiseDotProduct : public SynetConvolution32f
        {
        public:
            SynetConvolution32fDepthwiseDotProduct(const ConvParam & p);
            virtual String Ext() const { return "Base"; }
            virtual String Desc() const { return Ext() + "::DepthwiseDotProduct"; }
            virtual void Forward(const float * src, float * buf, float * dst);

            static bool Preferable(const ConvParam & p);

        protected:
            size_t _count, _size, _batch, _sizeS, _sizeD;
        };

        //-------------------------------------------------------------------------------------------------

        class SynetConvolution32fNhwcGroupedBlock1x2 : public SynetConvolution32f
        {
        public:
            SynetConvolution32fNhwcGroupedBlock1x2(const ConvParam& p);
            virtual String Ext() const { return "Base"; }
            virtual String Desc() const { return Ext() + "::NhwcGroupedBlock1x2"; }
            virtual void SetParams(const float* weight, SimdBool* internal, const float* bias, const float* params);
            virtual void Forward(const float* src, float* buf, float* dst);

            static bool Preferable(const ConvParam& p);

            typedef void(*ConvolutionPtr)(const float* src, const ConvParam& p, const float* weight, const float* bias, const float* params, float* dst);

        protected:
            size_t _batch, _sizeS, _sizeD;
            Array32f _rWeight, _rBias;
            ConvolutionPtr _convolution;
        };

        //-------------------------------------------------------------------------------------------------

        class SynetConvolution32fNhwcDirect : public SynetConvolution32f
        {
        public:
            SynetConvolution32fNhwcDirect(const ConvParam & p);
            virtual String Ext() const { return "Base"; }
            virtual String Desc() const { return Ext() + "::NhwcDirect" + (_old.enable ? "-f" : "-r"); }
            virtual size_t InternalBufferSize() const;
            virtual void SetParams(const float * weight, SimdBool * internal, const float * bias, const float * params);
            virtual void Forward(const float * src, float * buf, float * dst);

            static bool Preferable(const ConvParam & p);

            struct AlgParam;

            typedef void(*ConvolutionPtr)(const float * src, const ConvParam & p, const AlgParam & a, size_t dstC, size_t yBeg, size_t yEnd, size_t srcC, 
                const float * weight, const float * bias, const float * params, float * dst, int first);

            struct AlgParam
            {
                size_t F, microD, macroH, macroC, macroD;
                ConvolutionPtr convolutions[2];
                size_t stepW;
            };

            typedef void(*OldConvolutionPtr)(const float* src, const ConvParam& p, const AlgParam& a, const float* weight, const float* bias, const float* params, float* dst);

        protected:
            size_t _sizeS, _sizeD;
            Array32f _rWeight, _rBias, _rParams;

            static void Forward(const float* src, const ConvParam& p, const AlgParam& a, const float* weight, const float* bias, const float* params, float* dst);

            struct RunArgs
            {
                const float* src; const ConvParam& p; const float* weight; const float* bias; const float* params; float* dst;
                SIMD_INLINE RunArgs(const float* src_, const ConvParam& p_, const float* weight_, const float* bias_, const float* params_, float* dst_)
                    :src(src_), p(p_), weight(weight_), bias(bias_), params(params_), dst(dst_)
                {}
            };

            struct RunFunc
            {
                SIMD_INLINE RunFunc(const String& name)
                    : _name(name)
                {
                }

                SIMD_INLINE const String & Name() const { return _name; }

                SIMD_INLINE void Run(const RunArgs& args)
                {
                    Forward(args.src, args.p, alg, args.weight, args.bias, args.params, args.dst);
                }

#ifdef SIMD_RUNTIME_STATISTIC
                SIMD_INLINE String Info(const RunArgs& args) const
                {
                    std::stringstream ss;
                    ss << "NhwcDirect [" << args.p.Info() << "]";
                    return ss.str();
                }
#endif

                AlgParam alg;
            private:
                String _name;
            };
            typedef std::vector<RunFunc> RunFuncs;
            typedef Runtime<RunFunc, RunArgs> RuntimeRun;
            RuntimeRun _run;

            struct Old
            {
                bool enable;
                OldConvolutionPtr convolution;
                AlgParam alg;
                Array32f weight;
            } _old;
            void OldSetAlgParam(size_t F);
            void OldReorderWeight(const float* src, float* dst);

            void SetAlgParam(size_t F, size_t N, AlgParam & alg);
            void ReorderWeight(const float* src, float* dst);
        };

        //-------------------------------------------------------------------------------------------------

        void * SynetConvolution32fInit(size_t batch, const SimdConvolutionParameters * conv);
    }

#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        void ConvolutionBiasAndActivation(const float* bias, size_t count, size_t size, ::SimdConvolutionActivationType activation, const float* params, ::SimdBool trans, float* dst);

        class SynetConvolution32fGemmNN : public Base::SynetConvolution32fGemmNN
        {
        public:
            SynetConvolution32fGemmNN(const ConvParam& p);
            virtual String Ext() const { return "Sse41"; }
        };

        class SynetConvolution32fGemmNT : public Base::SynetConvolution32fGemmNT
        {
        public:
            SynetConvolution32fGemmNT(const ConvParam& p);
            virtual String Ext() const { return "Sse41"; }
        };

        class SynetConvolution32fWinograd : public Base::SynetConvolution32fWinograd
        {
        public:
            SynetConvolution32fWinograd(const ConvParam& p);
            virtual String Ext() const { return "Sse41"; }
        };

        class SynetConvolution32fDirectNchw : public Base::SynetConvolution32fDirectNchw
        {
        public:
            SynetConvolution32fDirectNchw(const ConvParam& p);
            virtual String Ext() const { return "Sse41"; }

            static bool Preferable(const ConvParam& p);

        protected:
            virtual ConvolutionBiasActivationPtr SetConvolutionBiasActivation();
        };

        class SynetConvolution32fNhwcDepthwise : public Base::SynetConvolution32fNhwcDepthwise
        {
        public:
            SynetConvolution32fNhwcDepthwise(const ConvParam& p);
            virtual String Ext() const { return "Sse41"; }
        };

        class SynetConvolution32fDepthwiseDotProduct : public Base::SynetConvolution32fDepthwiseDotProduct
        {
        public:
            SynetConvolution32fDepthwiseDotProduct(const ConvParam& p);
            virtual String Ext() const { return "Sse41"; }
            virtual void Forward(const float* src, float* buf, float* dst);
        };

        class SynetConvolution32fNhwcGroupedBlock1x2 : public Base::SynetConvolution32fNhwcGroupedBlock1x2
        {
        public:
            SynetConvolution32fNhwcGroupedBlock1x2(const ConvParam& p);
            virtual String Ext() const { return "Sse41"; }
        };

        class SynetConvolution32fNhwcDirect : public Base::SynetConvolution32fNhwcDirect
        {
        public:
            SynetConvolution32fNhwcDirect(const ConvParam& p);
            virtual String Ext() const { return "Sse41"; }

            static bool Preferable(const ConvParam& p);
        private:
            static bool Set2f(const ConvParam& p, OldConvolutionPtr& convolution);
            static bool SetRt(const ConvParam& p, AlgParam& a);
            static bool Set2r(const ConvParam& p, AlgParam& a);
            static bool Set3r(const ConvParam& p, AlgParam& a);
        };

        //-------------------------------------------------------------------------------------------------

        void * SynetConvolution32fInit(size_t batch, const SimdConvolutionParameters * conv);
    }
#endif

#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        void NhwcRun(size_t M, size_t N, size_t K, const float * A, const float * B, float * C);
        void NhwcReorderB(size_t M, size_t N, size_t K, const float * B, float * pB);
        size_t NhwcBufferSize(size_t M, size_t N, size_t K);

        void ConvolutionBiasAndActivation(const float* bias, size_t count, size_t size, ::SimdConvolutionActivationType activation, const float* params, ::SimdBool trans, float* dst);

        //-----------------------------------------------------------------------------------------

        class SynetConvolution32fGemmNN : public Sse41::SynetConvolution32fGemmNN
        {
        public:
            SynetConvolution32fGemmNN(const ConvParam & p);
            virtual String Ext() const { return "Avx2"; }
        protected:
            virtual void ImgToCol(const float * src, float * dst);
        private:
            Array32i _index, _nose, _tail, _start;
        };

        class SynetConvolution32fGemmNT : public Sse41::SynetConvolution32fGemmNT
        {
        public:
            SynetConvolution32fGemmNT(const ConvParam & p);
            virtual String Ext() const { return "Avx2"; }
        };

        class SynetConvolution32fWinograd : public Sse41::SynetConvolution32fWinograd
        {
        public:
            SynetConvolution32fWinograd(const ConvParam & p);
            virtual String Ext() const { return "Avx2"; }
        };

        class SynetConvolution32fDirectNchw : public Sse41::SynetConvolution32fDirectNchw
        {
        public:
            SynetConvolution32fDirectNchw(const ConvParam & p);
            virtual String Ext() const { return "Avx2"; }
        protected:
            virtual ConvolutionBiasActivationPtr SetConvolutionBiasActivation();
        };

        class SynetConvolution32fNhwcDepthwise : public Sse41::SynetConvolution32fNhwcDepthwise
        {
        public:
            SynetConvolution32fNhwcDepthwise(const ConvParam & p);
            virtual String Ext() const { return "Avx2"; }
        };

        class SynetConvolution32fDepthwiseDotProduct : public Sse41::SynetConvolution32fDepthwiseDotProduct
        {
        public:
            SynetConvolution32fDepthwiseDotProduct(const ConvParam& p);
            virtual String Ext() const { return "Avx2"; }
            virtual void Forward(const float* src, float* buf, float* dst);
        };

        class SynetConvolution32fNhwcGroupedBlock1x2 : public Sse41::SynetConvolution32fNhwcGroupedBlock1x2
        {
        public:
            SynetConvolution32fNhwcGroupedBlock1x2(const ConvParam& p);
            virtual String Ext() const { return "Avx2"; }
        };

        class SynetConvolution32fNhwcDirect : public Sse41::SynetConvolution32fNhwcDirect
        {
        public:
            SynetConvolution32fNhwcDirect(const ConvParam & p);
            virtual String Ext() const { return "Avx2"; }
        private:
            static bool Set2f(const ConvParam& p, OldConvolutionPtr& convolution);
            static bool SetRt(const ConvParam& p, AlgParam& a);
            static bool Set2r(const ConvParam& p, AlgParam& a);
            static bool Set3r(const ConvParam& p, AlgParam& a);
        };

        //-----------------------------------------------------------------------------------------

        void * SynetConvolution32fInit(size_t batch, const SimdConvolutionParameters * conv);
    }
#endif

#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        void ConvolutionBiasAndActivation(const float* bias, size_t count, size_t size, ::SimdConvolutionActivationType activation, const float* params, ::SimdBool trans, float* dst);

        //-----------------------------------------------------------------------------------------

        class SynetConvolution32fGemmNN : public Avx2::SynetConvolution32fGemmNN
        {
        public:
            SynetConvolution32fGemmNN(const ConvParam& p);
            virtual String Ext() const { return "Avx512bw"; }
        protected:
            virtual void ImgToCol(const float* src, float* dst);
        private:
            Array32i _index;
            Array16u _nose, _tail;
        };

        class SynetConvolution32fGemmNT : public Avx2::SynetConvolution32fGemmNT
        {
        public:
            SynetConvolution32fGemmNT(const ConvParam& p);
            virtual String Ext() const { return "Avx512bw"; }
        };

        class SynetConvolution32fWinograd : public Avx2::SynetConvolution32fWinograd
        {
        public:
            SynetConvolution32fWinograd(const ConvParam& p);
            virtual String Ext() const { return "Avx512bw"; }
        };

        class SynetConvolution32fDirectNchw : public Avx2::SynetConvolution32fDirectNchw
        {
        public:
            SynetConvolution32fDirectNchw(const ConvParam& p);
            virtual String Ext() const { return "Avx512bw"; }

            static bool Preferable(const ConvParam& p);

        protected:
            virtual ConvolutionBiasActivationPtr SetConvolutionBiasActivation();
        };

        class SynetConvolution32fNhwcDepthwise : public Avx2::SynetConvolution32fNhwcDepthwise
        {
        public:
            SynetConvolution32fNhwcDepthwise(const ConvParam& p);
            virtual String Ext() const { return "Avx512bw"; }
        };

        class SynetConvolution32fNhwcGroupedBlock1x2 : public Avx2::SynetConvolution32fNhwcGroupedBlock1x2
        {
        public:
            SynetConvolution32fNhwcGroupedBlock1x2(const ConvParam& p);
            virtual String Ext() const { return "Avx512bw"; }
        };

        class SynetConvolution32fNhwcDirect : public Avx2::SynetConvolution32fNhwcDirect
        {
        public:
            SynetConvolution32fNhwcDirect(const ConvParam& p);
            virtual String Ext() const { return "Avx512bw"; }
        private:
            static bool Set2f(const ConvParam& p, OldConvolutionPtr& convolution);
            static bool SetRt(const ConvParam& p, AlgParam& a);
            static bool Set2r(const ConvParam& p, AlgParam& a);
            static bool Set3r(const ConvParam& p, AlgParam& a);
        };

        //-----------------------------------------------------------------------------------------

        void* SynetConvolution32fInit(size_t batch, const SimdConvolutionParameters* conv);
    }
#endif

#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
        void ConvolutionBiasAndActivation(const float * bias, size_t count, size_t size, ::SimdConvolutionActivationType activation, const float * params, ::SimdBool trans, float * dst);

        //-----------------------------------------------------------------------------------------

        class SynetConvolution32fGemmNN : public Base::SynetConvolution32fGemmNN
        {
        public:
            SynetConvolution32fGemmNN(const ConvParam & p);
            virtual String Ext() const { return "Neon"; }
        };

        class SynetConvolution32fGemmNT : public Base::SynetConvolution32fGemmNT
        {
        public:
            SynetConvolution32fGemmNT(const ConvParam & p);
            virtual String Ext() const { return "Neon"; }

            static bool Preferable(const ConvParam & p);
        };

        class SynetConvolution32fWinograd : public Base::SynetConvolution32fWinograd
        {
        public:
            SynetConvolution32fWinograd(const ConvParam & p);
            virtual String Ext() const { return "Neon"; }

            static bool Preferable(const ConvParam & p);
        };

        class SynetConvolution32fDirectNchw : public Base::SynetConvolution32fDirectNchw
        {
        public:
            SynetConvolution32fDirectNchw(const ConvParam & p);
            virtual String Ext() const { return "Neon"; }

            static bool Preferable(const ConvParam & p);

        protected:
            virtual ConvolutionBiasActivationPtr SetConvolutionBiasActivation();
        };

        class SynetConvolution32fNhwcDepthwise : public Base::SynetConvolution32fNhwcDepthwise
        {
        public:
            SynetConvolution32fNhwcDepthwise(const ConvParam & p);
            virtual String Ext() const { return "Neon"; }
        };

        class SynetConvolution32fDepthwiseDotProduct : public Base::SynetConvolution32fDepthwiseDotProduct
        {
        public:
            SynetConvolution32fDepthwiseDotProduct(const ConvParam & p);
            virtual String Ext() const { return "Neon"; }
            virtual void Forward(const float * src, float * buf, float * dst);
        };

        class SynetConvolution32fNhwcDirect : public Base::SynetConvolution32fNhwcDirect
        {
        public:
            SynetConvolution32fNhwcDirect(const ConvParam & p);
            virtual String Ext() const { return "Neon"; }

            static bool Preferable(const ConvParam & p);
        private:
            static bool Set2f(const ConvParam& p, OldConvolutionPtr& convolution);
            static bool SetRt(const ConvParam& p, AlgParam& a);
            static bool Set2r(const ConvParam& p, AlgParam& a);
            static bool Set3r(const ConvParam& p, AlgParam& a);
            static bool Set4r(const ConvParam& p, AlgParam& a);
        };

        //-----------------------------------------------------------------------------------------

        void * SynetConvolution32fInit(size_t batch, const SimdConvolutionParameters * conv);
    }
#endif
}

#endif
