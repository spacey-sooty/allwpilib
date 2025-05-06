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
#include "Simd/SimdSynet.h"
#include "Simd/SimdArray.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdExtract.h"

namespace Simd
{
#if defined(SIMD_AVX2_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Avx2
    {
        void NormalizeNchw1(const float* src, size_t batch, size_t channels, size_t spatial, const float* scale, float eps, float* dst)
        {
            size_t size = channels * spatial;
            size_t sizeF = AlignLo(size, F);
            size_t spatialF = AlignLo(spatial, F);
            for (size_t b = 0; b < batch; ++b)
            {
                __m256 _sum = _mm256_setzero_ps();
                size_t i = 0;
                for (; i < sizeF; i += F)
                {
                    __m256 _src = _mm256_loadu_ps(src + i);
                    _sum = _mm256_fmadd_ps(_src, _src, _sum);
                }
                float sum = ExtractSum(_sum);
                for (; i < size; ++i)
                    sum += Simd::Square(src[i]);
                float k0 = 1.0f / ::sqrt(sum + eps);
                for (size_t c = 0; c < channels; ++c)
                {
                    __m256 _k = _mm256_set1_ps(scale[c] * k0);
                    size_t s = 0;
                    for (; s < spatialF; s += F)
                        _mm256_storeu_ps(dst + s, _mm256_mul_ps(_mm256_loadu_ps(src + s), _k));
                    for (; s < spatial; ++s)
                        _mm_store_ss(dst + s, _mm_mul_ss(_mm_load_ss(src + s), _mm256_castps256_ps128(_k)));
                    dst += spatial;
                    src += spatial;
                }
            }
        }

        void NormalizeNchw0(const float* src, size_t batch, size_t channels, size_t spatial, const float* scale, float eps, float* buf, float* dst)
        {
            Array32f _buf;
            if (buf == NULL)
            {
                _buf.Resize(spatial);
                buf = _buf.data;
            }
            size_t spatialF = AlignLo(spatial, F);
            __m256 _eps = _mm256_set1_ps(eps), _1 = _mm256_set1_ps(1.0f);
            for (size_t b = 0; b < batch; ++b)
            {
                size_t s = 0;
                for (; s < spatialF; s += F)
                    _mm256_storeu_ps(buf + s, _mm256_setzero_ps());
                for (; s < spatial; ++s)
                    _mm_store_ss(buf + s, _mm_setzero_ps());
                for (size_t c = 0; c < channels; ++c)
                {
                    const float* ps = src + c * spatial;
                    for (s = 0; s < spatialF; s += F)
                    {
                        __m256 _src = _mm256_loadu_ps(ps + s);
                        _mm256_storeu_ps(buf + s, _mm256_fmadd_ps(_src, _src, _mm256_loadu_ps(buf + s)));
                    }
                    for (; s < spatial; ++s)
                    {
                        __m128 _src = _mm_load_ss(ps + s);
                        __m128 _sum = _mm_load_ss(buf + s);
                        _mm_store_ss(buf + s, _mm_add_ss(_sum, _mm_mul_ps(_src, _src)));
                    }
                }
                for (s = 0; s < spatialF; s += F)
                    _mm256_storeu_ps(buf + s, _mm256_div_ps(_1, _mm256_sqrt_ps(_mm256_add_ps(_mm256_loadu_ps(buf + s), _eps))));
                for (; s < spatial; ++s)
                    _mm_store_ss(buf + s, _mm_div_ss(_mm256_castps256_ps128(_1), _mm_sqrt_ss(_mm_add_ss(_mm_load_ss(buf + s), _mm256_castps256_ps128(_eps)))));
                for (size_t c = 0; c < channels; ++c)
                {
                    float k = scale[c];
                    __m256 _k = _mm256_set1_ps(k);
                    for (s = 0; s < spatialF; s += F)
                        _mm256_storeu_ps(dst + s, _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(src + s), _mm256_loadu_ps(buf + s)), _k));
                    for (; s < spatial; ++s)
                        _mm_store_ss(dst + s, _mm_mul_ss(_mm_mul_ps(_mm_load_ss(src + s), _mm_load_ss(buf + s)), _mm256_castps256_ps128(_k)));
                    dst += spatial;
                    src += spatial;
                }
            }
        }

        void NormalizeNhwc1(const float* src, size_t batch, size_t channels, size_t spatial, const float* scale, float eps, float* dst)
        {
            size_t size = channels * spatial;
            size_t sizeF = AlignLo(size, F);
            size_t channelsF = AlignLo(channels, F);
            for (size_t b = 0; b < batch; ++b)
            {
                __m256 _sum = _mm256_setzero_ps();
                size_t i = 0;
                for (; i < sizeF; i += F)
                {
                    __m256 _src = _mm256_loadu_ps(src + i);
                    _sum = _mm256_fmadd_ps(_src, _src, _sum);
                }
                float sum = ExtractSum(_sum);
                for (; i < size; ++i)
                    sum += Simd::Square(src[i]);
                __m256 _k = _mm256_set1_ps(1.0f / ::sqrt(sum + eps));
                for (size_t s = 0; s < spatial; ++s)
                {
                    size_t c = 0;
                    for (; c < channelsF; c += F)
                        _mm256_storeu_ps(dst + c, _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(src + c), _mm256_loadu_ps(scale + c)), _k));
                    for (; c < channels; ++c)
                        _mm_store_ss(dst + c, _mm_mul_ss(_mm_mul_ps(_mm_load_ss(src + c), _mm_load_ss(scale + c)), _mm256_castps256_ps128(_k)));
                    dst += channels;
                    src += channels;
                }
            }
        }

        void NormalizeNhwc0(const float* src, size_t batch, size_t channels, size_t spatial, const float* scale, float eps, float* dst)
        {
            size_t channelsF = AlignLo(channels, F);
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    __m256 _sum = _mm256_setzero_ps();
                    size_t c = 0;
                    for (; c < channelsF; c += F)
                    {
                        __m256 _src = _mm256_loadu_ps(src + c);
                        _sum = _mm256_fmadd_ps(_src, _src, _sum);
                    }
                    float sum = ExtractSum(_sum);
                    for (; c < channels; ++c)
                        sum += Simd::Square(src[c]);
                    __m256 _k = _mm256_set1_ps(1.0f / ::sqrt(sum + eps));
                    for (c = 0; c < channelsF; c += F)
                        _mm256_storeu_ps(dst + c, _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(src + c), _mm256_loadu_ps(scale + c)), _k));
                    for (; c < channels; ++c)
                        _mm_store_ss(dst + c, _mm_mul_ss(_mm_mul_ps(_mm_load_ss(src + c), _mm_load_ss(scale + c)), _mm256_castps256_ps128(_k)));
                    dst += channels;
                    src += channels;
                }
            }
        }

        void SynetNormalizeLayerForward(const float* src, size_t batch, size_t channels, size_t spatial, const float* scale,
            const float* eps, SimdBool acrossSpatial, SimdTensorFormatType format, float* buf, float* dst)
        {
            if (format == SimdTensorFormatNchw)
            {
                if (acrossSpatial)
                    NormalizeNchw1(src, batch, channels, spatial, scale, eps[0], dst);
                else
                    NormalizeNchw0(src, batch, channels, spatial, scale, eps[0], buf, dst);
            }
            else if (format == SimdTensorFormatNhwc)
            {
                if (acrossSpatial)
                    NormalizeNhwc1(src, batch, channels, spatial, scale, eps[0], dst);
                else
                    NormalizeNhwc0(src, batch, channels, spatial, scale, eps[0], dst);
            }
            else
                assert(0);
        }

        //-------------------------------------------------------------------------------------------------

        void NormalizeNchwV2(const float* src, size_t batch, size_t channels, size_t spatial, const float* scale, const float* shift, float eps, float* buf, float* dst)
        {
            float k = 1.0f / float(channels);
            Array32f _buf;
            if (buf == NULL)
            {
                _buf.Resize(spatial);
                buf = _buf.data;
            }
            size_t spatialF = AlignLo(spatial, F);
            __m256 _eps = _mm256_set1_ps(eps), _k = _mm256_set1_ps(k), _1 = _mm256_set1_ps(1.0f);
            for (size_t b = 0, s; b < batch; ++b)
            {
                for (s = 0; s < spatialF; s += F)
                    _mm256_storeu_ps(buf + s, _mm256_setzero_ps());
                for (; s < spatial; ++s)
                    _mm_store_ss(buf + s, _mm_setzero_ps());
                for (size_t c = 0; c < channels; ++c)
                {
                    const float* ps = src + c * spatial;
                    for (s = 0; s < spatialF; s += F)
                    {
                        __m256 _src = _mm256_loadu_ps(ps + s);
                        __m256 _sum = _mm256_loadu_ps(buf + s);
                        _mm256_storeu_ps(buf + s, _mm256_add_ps(_sum, _src));
                    }
                    for (; s < spatial; ++s)
                    {
                        __m128 _src = _mm_load_ss(ps + s);
                        __m128 _sum = _mm_load_ss(buf + s);
                        _mm_store_ss(buf + s, _mm_add_ss(_sum, _src));
                    }
                }
                for (s = 0; s < spatialF; s += F)
                    _mm256_storeu_ps(buf + s, _mm256_mul_ps(_mm256_loadu_ps(buf + s), _k));
                for (; s < spatial; ++s)
                    _mm_store_ss(buf + s, _mm_mul_ss(_mm_load_ss(buf + s), _mm256_castps256_ps128(_k)));
                for (size_t c = 0; c < channels; ++c)
                {
                    const float* ps = src + c * spatial;
                    float* pd = dst + c * spatial;
                    for (s = 0; s < spatialF; s += F)
                    {
                        __m256 _src = _mm256_loadu_ps(ps + s);
                        __m256 mean = _mm256_loadu_ps(buf + s);
                        _mm256_storeu_ps(pd + s, _mm256_sub_ps(_src, mean));
                    }
                    for (; s < spatial; ++s)
                    {
                        __m128 _src = _mm_load_ss(ps + s);
                        __m128 mean = _mm_load_ss(buf + s);
                        _mm_store_ss(pd + s, _mm_sub_ps(_src, mean));
                    }
                }

                for (s = 0; s < spatialF; s += F)
                    _mm256_storeu_ps(buf + s, _mm256_setzero_ps());
                for (; s < spatial; ++s)
                    _mm_store_ss(buf + s, _mm_setzero_ps());
                for (size_t c = 0; c < channels; ++c)
                {
                    const float* pd = dst + c * spatial;
                    for (s = 0; s < spatialF; s += F)
                    {
                        __m256 _dst = _mm256_loadu_ps(pd + s);
                        __m256 _sum = _mm256_loadu_ps(buf + s);
                        _mm256_storeu_ps(buf + s, _mm256_fmadd_ps(_dst, _dst, _sum));
                    }
                    for (; s < spatial; ++s)
                    {
                        __m128 _dst = _mm_load_ss(pd + s);
                        __m128 _sum = _mm_load_ss(buf + s);
                        _mm_store_ss(buf + s, _mm_fmadd_ss(_dst, _dst, _sum));
                    }
                }
                for (s = 0; s < spatialF; s += F)
                    _mm256_storeu_ps(buf + s, _mm256_div_ps(_1, _mm256_sqrt_ps(_mm256_fmadd_ps(_mm256_loadu_ps(buf + s), _k, _eps))));
                for (; s < spatial; ++s)
                    _mm_store_ss(buf + s, _mm_div_ss(_mm256_castps256_ps128(_1), _mm_sqrt_ss(_mm_fmadd_ss(_mm_load_ss(buf + s), _mm256_castps256_ps128(_k), _mm256_castps256_ps128(_eps)))));
                for (size_t c = 0; c < channels; ++c)
                {
                    __m256 _scale = _mm256_set1_ps(scale[c]);
                    __m256 _shift = _mm256_set1_ps(shift[c]);
                    float* pd = dst + c * spatial;
                    for (s = 0; s < spatialF; s += F)
                    {
                        __m256 _dst = _mm256_loadu_ps(pd + s);
                        __m256 norm = _mm256_loadu_ps(buf + s);
                        _mm256_storeu_ps(pd + s, _mm256_fmadd_ps(_mm256_mul_ps(_dst, norm), _scale, _shift));
                    }
                    for (; s < spatial; ++s)
                    {
                        __m128 _dst = _mm_load_ss(pd + s);
                        __m128 norm = _mm_load_ss(buf + s);
                        _mm_store_ss(pd + s, _mm_fmadd_ss(_mm_mul_ss(_dst, norm), _mm256_castps256_ps128(_scale), _mm256_castps256_ps128(_shift)));
                    }
                }

                src += channels * spatial;
                dst += channels * spatial;
            }
        }

        void NormalizeNhwcV2(const float* src, size_t batch, size_t channels, size_t spatial, const float* scale, const float* shift, float eps, float* dst)
        {
            float k = 1.0f / float(channels);
            size_t channelsF = AlignLo(channels, F), c;
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    __m256 _sum = _mm256_setzero_ps();
                    for (c = 0; c < channelsF; c += F)
                        _sum = _mm256_add_ps(_mm256_loadu_ps(src + c), _sum);
                    float sum = ExtractSum(_sum);
                    for (; c < channels; ++c)
                        sum += src[c];
                    __m256 mean = _mm256_set1_ps(sum * k);
                    for (c = 0; c < channelsF; c += F)
                        _mm256_storeu_ps(dst + c, _mm256_sub_ps(_mm256_loadu_ps(src + c), mean));
                    for (; c < channels; ++c)
                        _mm_store_ss(dst + c, _mm_sub_ss(_mm_load_ss(src + c), _mm256_castps256_ps128(mean)));

                    __m256 _sqsum = _mm256_setzero_ps();
                    for (c = 0; c < channelsF; c += F)
                    {
                        __m256 _dst = _mm256_loadu_ps(dst + c);
                        _sqsum = _mm256_fmadd_ps(_dst, _dst, _sqsum);
                    }
                    float sqsum = ExtractSum(_sqsum);
                    for (; c < channels; ++c)
                        sqsum += Simd::Square(dst[c]);
                    __m256 norm = _mm256_set1_ps(1.0f / ::sqrt(sqsum * k + eps));
                    for (c = 0; c < channelsF; c += F)
                        _mm256_storeu_ps(dst + c, _mm256_fmadd_ps(_mm256_mul_ps(_mm256_loadu_ps(dst + c), norm), _mm256_loadu_ps(scale + c), _mm256_loadu_ps(shift + c)));
                    for (; c < channels; ++c)
                        _mm_store_ss(dst + c, _mm_fmadd_ss(_mm_mul_ss(_mm_load_ss(dst + c), _mm256_castps256_ps128(norm)), _mm_load_ss(scale + c), _mm_load_ss(shift + c)));

                    dst += channels;
                    src += channels;
                }
            }
        }

        void SynetNormalizeLayerForwardV2(const float* src, size_t batch, size_t channels, size_t spatial,
            const float* scale, const float* shift, const float* eps, SimdTensorFormatType format, float* buf, float* dst)
        {
            if (format == SimdTensorFormatNchw)
                NormalizeNchwV2(src, batch, channels, spatial, scale, shift, *eps, buf, dst);
            else if (format == SimdTensorFormatNhwc)
                NormalizeNhwcV2(src, batch, channels, spatial, scale, shift, *eps, dst);
            else
                assert(0);
        }

        //-------------------------------------------------------------------------------------------------

        void NormalizeNchwV3(const float* src, size_t batch, size_t channels, size_t spatial, const float* scale, const float* shift, float eps, float* dst)
        {
            float k = 1.0f / float(spatial);
            size_t spatialF = AlignLo(spatial, F), s;
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    __m256 _sum = _mm256_setzero_ps();
                    for (s = 0; s < spatialF; s += F)
                        _sum = _mm256_add_ps(_mm256_loadu_ps(src + s), _sum);
                    float sum = ExtractSum(_sum);
                    for (; s < spatial; ++s)
                        sum += src[s];
                    __m256 mean = _mm256_set1_ps(sum * k);
                    for (s = 0; s < spatialF; s += F)
                        _mm256_storeu_ps(dst + s, _mm256_sub_ps(_mm256_loadu_ps(src + s), mean));
                    for (; s < spatial; ++s)
                        _mm_store_ss(dst + s, _mm_sub_ss(_mm_load_ss(src + s), _mm256_castps256_ps128(mean)));

                    __m256 _sqsum = _mm256_setzero_ps();
                    for (s = 0; s < spatialF; s += F)
                        _sqsum = _mm256_add_ps(Square(_mm256_loadu_ps(dst + s)), _sqsum);
                    float sqsum = ExtractSum(_sqsum);
                    for (; s < spatial; ++s)
                        sqsum += Simd::Square(dst[s]);
                    __m256 norm = _mm256_set1_ps(1.0f / ::sqrt(sqsum * k + eps));
                    __m256 _scale = _mm256_set1_ps(scale[c]);
                    __m256 _shift = _mm256_set1_ps(shift[c]);
                    for (s = 0; s < spatialF; s += F)
                        _mm256_storeu_ps(dst + s, _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(dst + s), norm), _scale), _shift));
                    for (; s < spatial; ++s)
                        _mm_store_ss(dst + s, _mm_add_ss(_mm_mul_ss(_mm_mul_ss(_mm_load_ss(dst + s), _mm256_castps256_ps128(norm)), _mm256_castps256_ps128(_scale)), _mm256_castps256_ps128(_shift)));

                    dst += spatial;
                    src += spatial;
                }
            }
        }

        void NormalizeNhwcV3(const float* src, size_t batch, size_t channels, size_t spatial, const float* scale, const float* shift, float eps, float* buf, float* dst)
        {
            float k = 1.0f / float(spatial);
            Array32f _buf;
            if (buf == NULL)
            {
                _buf.Resize(spatial);
                buf = _buf.data;
            }
            size_t channelsF = AlignLo(channels, F);
            __m256 _eps = _mm256_set1_ps(eps), _k = _mm256_set1_ps(k), _1 = _mm256_set1_ps(1.0f);
            for (size_t b = 0, c; b < batch; ++b)
            {
                for (c = 0; c < channelsF; c += F)
                    _mm256_storeu_ps(buf + c, _mm256_setzero_ps());
                for (; c < channels; ++c)
                    _mm_store_ss(buf + c, _mm_setzero_ps());
                for (size_t s = 0; s < spatial; ++s)
                {
                    const float* ps = src + s * channels;
                    for (c = 0; c < channelsF; c += F)
                    {
                        __m256 _src = _mm256_loadu_ps(ps + c);
                        __m256 _sum = _mm256_loadu_ps(buf + c);
                        _mm256_storeu_ps(buf + c, _mm256_add_ps(_sum, _src));
                    }
                    for (; c < channels; ++c)
                    {
                        __m128 _src = _mm_load_ss(ps + c);
                        __m128 _sum = _mm_load_ss(buf + c);
                        _mm_store_ss(buf + c, _mm_add_ss(_sum, _src));
                    }
                }
                for (c = 0; c < channelsF; c += F)
                    _mm256_storeu_ps(buf + c, _mm256_mul_ps(_mm256_loadu_ps(buf + c), _k));
                for (; c < channels; ++c)
                    _mm_store_ss(buf + c, _mm_mul_ss(_mm_load_ss(buf + c), _mm256_castps256_ps128(_k)));
                for (size_t s = 0; s < spatial; ++s)
                {
                    const float* ps = src + s * channels;
                    float* pd = dst + s * channels;
                    for (c = 0; c < channelsF; c += F)
                    {
                        __m256 _src = _mm256_loadu_ps(ps + c);
                        __m256 mean = _mm256_loadu_ps(buf + c);
                        _mm256_storeu_ps(pd + c, _mm256_sub_ps(_src, mean));
                    }
                    for (; c < channels; ++c)
                    {
                        __m128 _src = _mm_load_ss(ps + c);
                        __m128 mean = _mm_load_ss(buf + c);
                        _mm_store_ss(pd + c, _mm_sub_ps(_src, mean));
                    }
                }

                for (c = 0; c < channelsF; c += F)
                    _mm256_storeu_ps(buf + c, _mm256_setzero_ps());
                for (; c < channels; ++c)
                    _mm_store_ss(buf + c, _mm_setzero_ps());
                for (size_t s = 0; s < spatial; ++s)
                {
                    const float* pd = dst + s * channels;
                    for (c = 0; c < channelsF; c += F)
                    {
                        __m256 _dst = _mm256_loadu_ps(pd + c);
                        __m256 _sum = _mm256_loadu_ps(buf + c);
                        _mm256_storeu_ps(buf + c, _mm256_add_ps(_sum, _mm256_mul_ps(_dst, _dst)));
                    }
                    for (; c < channels; ++c)
                    {
                        __m128 _dst = _mm_load_ss(pd + c);
                        __m128 _sum = _mm_load_ss(buf + c);
                        _mm_store_ss(buf + c, _mm_add_ss(_sum, _mm_mul_ss(_dst, _dst)));
                    }
                }
                for (c = 0; c < channelsF; c += F)
                    _mm256_storeu_ps(buf + c, _mm256_div_ps(_1, _mm256_sqrt_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(buf + c), _k), _eps))));
                for (; c < channels; ++c)
                    _mm_store_ss(buf + c, _mm_div_ss(_mm256_castps256_ps128(_1), _mm_sqrt_ss(_mm_add_ss(_mm_mul_ss(_mm_load_ss(buf + c), _mm256_castps256_ps128(_k)), _mm256_castps256_ps128(_eps)))));
                for (size_t s = 0; s < spatial; ++s)
                {
                    float* pd = dst + s * channels;
                    for (c = 0; c < channelsF; c += F)
                    {
                        __m256 _dst = _mm256_loadu_ps(pd + c);
                        __m256 norm = _mm256_loadu_ps(buf + c);
                        _mm256_storeu_ps(pd + c, _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(_dst, norm), _mm256_loadu_ps(scale + c)), _mm256_loadu_ps(shift + c)));
                    }
                    for (; c < channels; ++c)
                    {
                        __m128 _dst = _mm_load_ss(pd + c);
                        __m128 norm = _mm_load_ss(buf + c);
                        _mm_store_ss(pd + c, _mm_add_ss(_mm_mul_ss(_mm_mul_ss(_dst, norm), _mm_load_ss(scale + c)), _mm_load_ss(shift + c)));
                    }
                }

                src += channels * spatial;
                dst += channels * spatial;
            }
        }

        void SynetNormalizeLayerForwardV3(const float* src, size_t batch, size_t channels, size_t spatial,
            const float* scale, const float* shift, const float* eps, SimdTensorFormatType format, float* buf, float* dst)
        {
            if (format == SimdTensorFormatNchw)
                NormalizeNchwV3(src, batch, channels, spatial, scale, shift, *eps, dst);
            else if (format == SimdTensorFormatNhwc)
                NormalizeNhwcV3(src, batch, channels, spatial, scale, shift, *eps, buf, dst);
            else
                assert(0);
        }

        //-------------------------------------------------------------------------------------------------

        void NormalizeNchwV4(const float* src, size_t batch, size_t channels, size_t spatial, const float* scale, const float* shift, float eps, float* buf, float* dst)
        {
            float k = 1.0f / float(channels);
            size_t spatialF = AlignLo(spatial, F), s;
            for (size_t b = 0; b < batch; ++b)
            {
                float sum = 0;
                for (size_t c = 0, o = 0; c < channels; ++c)
                {
                    __m256 _sqsum = _mm256_setzero_ps();
                    for (s = 0; s < spatialF; s += F, o += F)
                        _sqsum = _mm256_add_ps(Square(_mm256_loadu_ps(src + o)), _sqsum);
                    float sqsum = ExtractSum(_sqsum);
                    for (; s < spatial; ++s, ++o)
                        sqsum += Simd::Square(src[o]);
                    buf[c] = sqrt(sqsum);
                    sum += buf[c];
                }
                float norm = 1.0f / (sum * k + eps);
                for (size_t c = 0; c < channels; ++c)
                {
                    __m256 _alpha = _mm256_set1_ps(1.0f + scale[c] * buf[c] * norm);
                    __m256 _shift = _mm256_set1_ps(shift[c]);
                    for (s = 0; s < spatialF; s += F)
                        _mm256_storeu_ps(dst + s, _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(src + s), _alpha), _shift));
                    for (; s < spatial; ++s)
                        _mm_store_ss(dst + s, _mm_add_ss(_mm_mul_ss(_mm_load_ss(src + s), _mm256_castps256_ps128(_alpha)), _mm256_castps256_ps128(_shift)));
                    dst += spatial;
                    src += spatial;
                }
            }
        }

        void NormalizeNhwcV4(const float* src, size_t batch, size_t channels, size_t spatial, const float* scale, const float* shift, float eps, float* buf, float* dst)
        {
            float k = 1.0f / float(channels);
            size_t channelsF = AlignLo(channels, F), c;
            __m256 _eps = _mm256_set1_ps(eps), _k = _mm256_set1_ps(k), _1 = _mm256_set1_ps(1.0f);
            for (size_t b = 0; b < batch; ++b)
            {
                for (c = 0; c < channelsF; c += F)
                    _mm256_storeu_ps(buf + c, _mm256_setzero_ps());
                for (; c < channels; ++c)
                    _mm_store_ss(buf + c, _mm_setzero_ps());
                for (size_t s = 0, o = 0; s < spatial; ++s)
                {
                    for (c = 0; c < channelsF; c += F, o += F)
                        _mm256_storeu_ps(buf + c, _mm256_add_ps(Square(_mm256_loadu_ps(src + o)), _mm256_loadu_ps(buf + c)));
                    for (; c < channels; c += 1, o += 1)
                        _mm_store_ss(buf + c, _mm_add_ss(Sse41::Square(_mm_load_ss(src + o)), _mm_load_ss(buf + c)));
                }
                float sum = 0;
                for (size_t c = 0; c < channels; ++c)
                {
                    buf[c] = sqrt(buf[c]);
                    sum += buf[c];
                }
                float norm = 1.0f / (sum * k + eps);
                for (size_t c = 0; c < channels; ++c)
                    buf[c] = 1.0f + scale[c] * buf[c] * norm;
                for (size_t s = 0, o = 0; s < spatial; ++s)
                {
                    for (c = 0; c < channelsF; c += F)
                        _mm256_storeu_ps(dst + c, _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(src + c), _mm256_loadu_ps(buf + c)), _mm256_loadu_ps(shift + c)));
                    for (; c < channels; c += 1, o += 1)
                        _mm_store_ss(dst + c, _mm_add_ss(_mm_mul_ss(_mm_load_ss(src + c), _mm_load_ss(buf + c)), _mm_load_ss(shift + c)));
                    src += channels;
                    dst += channels;

                }
            }
        }

        void SynetNormalizeLayerForwardV4(const float* src, size_t batch, size_t channels, size_t spatial,
            const float* scale, const float* shift, const float* eps, SimdTensorFormatType format, float* buf, float* dst)
        {
            Array32f _buf;
            if (buf == NULL)
            {
                _buf.Resize(channels);
                buf = _buf.data;
            }
            if (format == SimdTensorFormatNchw)
                NormalizeNchwV4(src, batch, channels, spatial, scale, shift, *eps, buf, dst);
            else if (format == SimdTensorFormatNhwc)
                NormalizeNhwcV4(src, batch, channels, spatial, scale, shift, *eps, buf, dst);
            else
                assert(0);
        }
    }
#endif
}
