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
#include "Simd/SimdArray.h"
#include "Simd/SimdPow.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdAlignment.h"
#include "Simd/SimdExp.h"
#include "Simd/SimdBFloat16.h"

namespace Simd
{
#if defined(SIMD_SYNET_ENABLE)
    namespace Base
    {
        void SynetChannelSum16b(const uint16_t* src, size_t channels, size_t spatial, SimdTensorFormatType format, float* sum)
        {
            if (format == SimdTensorFormatNhwc)
            {
                for (size_t c = 0; c < channels; ++c)
                    sum[c] = 0.0f;
                for (size_t s = 0; s < spatial; ++s)
                {
                    for (size_t c = 0; c < channels; ++c)
                        sum[c] += BFloat16ToFloat32(src[c]);
                    src += channels;
                }
            }
            else if (format == SimdTensorFormatNchw)
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    sum[c] = 0.0f;
                    for (size_t s = 0; s < spatial; ++s)
                        sum[c] += BFloat16ToFloat32(src[s]);
                    src += spatial;
                }
            }
            else
                assert(0);
        }

        //-------------------------------------------------------------------------------------------------

        template <SimdSynetEltwiseOperationType type> void SynetEltwiseLayerForward(float const * const * src, size_t count, size_t size, float * dst)
        {
            size_t aligned = Simd::AlignLo(size, 4);
            const float * src0 = src[0];
            const float * src1 = src[1];
            size_t j = 0;
            for (; j < aligned; j += 4)
            {
                dst[j + 0] = SynetEltwiseLayerForward<type>(src0[j + 0], src1[j + 0]);
                dst[j + 1] = SynetEltwiseLayerForward<type>(src0[j + 1], src1[j + 1]);
                dst[j + 2] = SynetEltwiseLayerForward<type>(src0[j + 2], src1[j + 2]);
                dst[j + 3] = SynetEltwiseLayerForward<type>(src0[j + 3], src1[j + 3]);
            }
            for (; j < size; ++j)
                dst[j] = SynetEltwiseLayerForward<type>(src0[j], src1[j]);
            for (size_t i = 2; i < count; ++i)
            {
                const float * srci = src[i];
                for (j = 0; j < aligned; j += 4)
                {
                    dst[j + 0] = SynetEltwiseLayerForward<type>(dst[j + 0], srci[j + 0]);
                    dst[j + 1] = SynetEltwiseLayerForward<type>(dst[j + 1], srci[j + 1]);
                    dst[j + 2] = SynetEltwiseLayerForward<type>(dst[j + 2], srci[j + 2]);
                    dst[j + 3] = SynetEltwiseLayerForward<type>(dst[j + 3], srci[j + 3]);
                }
                for (; j < size; ++j)
                    dst[j] = SynetEltwiseLayerForward<type>(dst[j], srci[j]);
            }
        }

        void SynetEltwiseLayerForwardSum(float const * const * src, const float * weight, size_t count, size_t size, float * dst)
        {
            size_t aligned = Simd::AlignLo(size, 4);
            const float * src0 = src[0];
            const float * src1 = src[1];
            float weight0 = weight[0], weight1 = weight[1];
            size_t j = 0;
            for (; j < aligned; j += 4)
            {
                dst[j + 0] = src0[j + 0] * weight0 + src1[j + 0] * weight1;
                dst[j + 1] = src0[j + 1] * weight0 + src1[j + 1] * weight1;
                dst[j + 2] = src0[j + 2] * weight0 + src1[j + 2] * weight1;
                dst[j + 3] = src0[j + 3] * weight0 + src1[j + 3] * weight1;
            }
            for (; j < size; ++j)
                dst[j] = src0[j] * weight0 + src1[j] * weight1;
            for (size_t i = 2; i < count; ++i)
            {
                const float * srci = src[i];
                float weighti = weight[i];
                for (j = 0; j < aligned; j += 4)
                {
                    dst[j + 0] += srci[j + 0] * weighti;
                    dst[j + 1] += srci[j + 1] * weighti;
                    dst[j + 2] += srci[j + 2] * weighti;
                    dst[j + 3] += srci[j + 3] * weighti;
                }
                for (; j < size; ++j)
                    dst[j] += srci[j] * weighti;
            }
        }

        void SynetEltwiseLayerForward(float const * const * src, const float * weight, size_t count, size_t size, SimdSynetEltwiseOperationType type, float * dst)
        {
            switch (type)
            {
            case SimdSynetEltwiseOperationProduct:
                SynetEltwiseLayerForward<SimdSynetEltwiseOperationProduct>(src, count, size, dst);
                break;
            case SimdSynetEltwiseOperationSum:
                SynetEltwiseLayerForwardSum(src, weight, count, size, dst);
                break;
            case SimdSynetEltwiseOperationMax:
                SynetEltwiseLayerForward<SimdSynetEltwiseOperationMax>(src, count, size, dst);
                break;
            case SimdSynetEltwiseOperationMin:
                SynetEltwiseLayerForward<SimdSynetEltwiseOperationMin>(src, count, size, dst);
                break;
            default:
                assert(0);
            }
        }

        //-------------------------------------------------------------------------------------------------

        void SynetInnerProduct8i(size_t M, size_t N, size_t K, const uint8_t* src, const int8_t* weight, int32_t* dst, SimdSynetCompatibilityType compatibility)
        {
            const size_t K2 = Base::Precise(compatibility) ? 0 : K / 2 * 2;
            for (size_t i = 0; i < M; ++i)
            {
                for (size_t j = 0; j < N; ++j)
                {
                    const int8_t* w = weight + j * K;
                    size_t k = 0;
                    int32_t sum = 0;
                    for (; k < K2; k += 2)
                        sum += RestrictRange(int(src[k + 0]) * int(w[k + 0]) + int(src[k + 1]) * int(w[k + 1]), SHRT_MIN, SHRT_MAX);
                    for (; k < K; ++k)
                        sum += int(src[k + 0]) * int(w[k + 0]);
                    dst[j] = sum;
                }
                src += K;
                dst += N;
            }
        }

        //-------------------------------------------------------------------------------------------------

        void SynetLrnLayerCrossChannelsNchw(const float * src, size_t half, size_t channels, size_t spatial, const float * k, float * dst)
        {
            float k0 = k[0], k1 = k[1], k2 = k[2];
            Array32f sum(spatial, true), zero(spatial, true);
            for (size_t c = 0; c < half; ++c)
            {
                const float * pos = src + c * spatial;
                for (size_t s = 0; s < spatial; ++s)
                    sum[s] += Simd::Square(pos[s]);
            }
            for (size_t c = 0; c < channels; ++c)
            {
                const float * pos = (c < channels - half) ? src + half * spatial : zero.data;
                const float * neg = (c > half) ? src - (half + 1) * spatial : zero.data;
                for (size_t s = 0; s < spatial; ++s)
                {
                    sum[s] += Simd::Square(pos[s]);
                    sum[s] -= Simd::Square(neg[s]);
                    dst[s] = src[s] * Pow(k0 + k1 * sum[s], k2);
                }
                src += spatial;
                dst += spatial;
            }
        }

        void SynetLrnLayerCrossChannelsNhwc(const float * src, size_t half, size_t channels, size_t spatial, const float * k, float * dst)
        {
            float k0 = k[0], k1 = k[1], k2 = k[2];
            size_t beg = half + 1;
            size_t end = channels - half;
            for (size_t s = 0; s < spatial; ++s)
            {
                float sum = 0;
                for (size_t c = 0; c < half; ++c)
                    sum += Simd::Square(src[c]);
                for (size_t c = 0; c < beg; ++c)
                {
                    sum += Simd::Square(src[c + half]);
                    dst[c] = src[c] * Pow(k0 + k1 * sum, k2);
                }
                for (size_t c = beg; c < end; ++c)
                {
                    sum += Simd::Square(src[c + half]);
                    sum -= Simd::Square(src[c - half - 1]);
                    dst[c] = src[c] * Pow(k0 + k1 * sum, k2);
                }
                for (size_t c = end; c < channels; ++c)
                {
                    sum -= Simd::Square(src[c - half - 1]);
                    dst[c] = src[c] * Pow(k0 + k1 * sum, k2);
                }
                src += channels;
                dst += channels;
            }
        }

        void SynetLrnLayerCrossChannels(const float * src, size_t half, size_t channels, size_t spatial, const float * k, float * dst, SimdTensorFormatType format)
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
            size_t channels = (channels0 + channels1) / 2, size = sizeof(float) * spatial;
            switch (type)
            {
            case 0:
                if (format == SimdTensorFormatNchw)
                {
                    size_t cd = 0;
                    for (size_t cs = 0; cs < channels0; cs += 2, cd += 1)
                    {
                        memcpy(dst0, src0 + 0 * spatial, size);
                        memcpy(dst1, src0 + 1 * spatial, size);
                        src0 += 2 * spatial;
                        dst0 += spatial;
                        dst1 += spatial;
                    }
                    for (size_t cs = 0; cs < channels1; cs += 2, cd += 1)
                    {
                        memcpy(dst0, src1 + 0 * spatial, size);
                        memcpy(dst1, src1 + 1 * spatial, size);
                        src1 += 2 * spatial;
                        dst0 += spatial;
                        dst1 += spatial;
                    }
                }
                else if (format == SimdTensorFormatNhwc)
                {
                    for (size_t s = 0; s < spatial; ++s)
                    {
                        size_t cd = 0;
                        for (size_t cs = 0; cs < channels0; cs += 2, cd += 1)
                        {
                            dst0[cd] = src0[cs + 0];
                            dst1[cd] = src0[cs + 1];
                        }
                        for (size_t cs = 0; cs < channels1; cs += 2, cd += 1)
                        {
                            dst0[cd] = src1[cs + 0];
                            dst1[cd] = src1[cs + 1];
                        }
                        src0 += channels0;
                        src1 += channels1;
                        dst0 += channels;
                        dst1 += channels;
                    }
                }
                else
                    assert(0);                
                break;
            case 1:
                if (format == SimdTensorFormatNchw)
                {
                    size_t cs = 0;
                    for (size_t cd = 0; cd < channels0; cs += 1, cd += 2)
                    {
                        memcpy(dst0 + 0 * spatial, src0, size);
                        memcpy(dst0 + 1 * spatial, src1, size);
                        src0 += spatial;
                        src1 += spatial;
                        dst0 += 2 * spatial;
                    }
                    for (size_t cd = 0; cd < channels1; cs += 1, cd += 2)
                    {
                        memcpy(dst1 + 0 * spatial, src0, size);
                        memcpy(dst1 + 1 * spatial, src1, size);
                        src0 += spatial;
                        src1 += spatial;
                        dst1 += 2 * spatial;
                    }
                }                
                else if (format == SimdTensorFormatNhwc)
                {
                    for (size_t s = 0; s < spatial; ++s)
                    {
                        size_t cs = 0;
                        for (size_t cd = 0; cd < channels0; cd += 2, cs += 1)
                        {
                            dst0[cd + 0] = src0[cs];
                            dst0[cd + 1] = src1[cs];
                        }
                        for (size_t cd = 0; cd < channels1; cd += 2, cs += 1)
                        {
                            dst1[cd + 0] = src0[cs];
                            dst1[cd + 1] = src1[cs];
                        }
                        src0 += channels;
                        src1 += channels;
                        dst0 += channels0;
                        dst1 += channels1;
                    }
                }
                else
                    assert(0);
                break;
            default:
                assert(0);
            }
        }

        //-------------------------------------------------------------------------------------------------

        void SynetTiledScale2D32f(const float* src, size_t channels, size_t height, size_t width, SimdTensorFormatType format, const float* ver, const float* hor, float* dst)
        {
            if (format == SimdTensorFormatNchw)
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    for (size_t y = 0; y < height; ++y)
                    {
                        for (size_t x = 0; x < width; ++x)
                            dst[x] = src[x] * ver[x] * hor[0];
                        src += width, dst += width;
                        hor += 1;
                    }
                    ver += width;
                }
            }
            else if (format == SimdTensorFormatNhwc)
            {
                for (size_t y = 0; y < height; ++y)
                {
                    const float* pVer = ver;
                    for (size_t x = 0; x < width; ++x)
                    {
                        for (size_t c = 0; c < channels; ++c)
                            dst[c] = src[c] * pVer[c] * hor[c];
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
