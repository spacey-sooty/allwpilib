/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2023 Yermalayeu Ihar.
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
#include "Simd/SimdStore.h"
#include "Simd/SimdMemory.h"
#include "Simd/SimdConversion.h"
#include "Simd/SimdSet.h"
#include "Simd/SimdAlphaBlending.h"
#include "Simd/SimdUnpack.h"
#include "Simd/SimdYuvToBgr.h"
#include "Simd/SimdSse41.h"

namespace Simd
{
#if defined( SIMD_AVX2_ENABLE)   
    namespace Avx2
    {
        template <bool align, size_t channelCount> struct AlphaBlender
        {
            void operator()(const __m256i * src, __m256i * dst, __m256i alpha);
        };

        template <bool align> struct AlphaBlender<align, 1>
        {
            SIMD_INLINE void operator()(const __m256i * src, __m256i * dst, __m256i alpha)
            {
                AlphaBlending<align>(src, dst, alpha);
            }
        };

        template <bool align> struct AlphaBlender<align, 2>
        {
            SIMD_INLINE void operator()(const __m256i * src, __m256i * dst, __m256i alpha)
            {
                alpha = _mm256_permute4x64_epi64(alpha, 0xD8);
                AlphaBlending<align>(src + 0, dst + 0, _mm256_unpacklo_epi8(alpha, alpha));
                AlphaBlending<align>(src + 1, dst + 1, _mm256_unpackhi_epi8(alpha, alpha));
            }
        };

        template <bool align> struct AlphaBlender<align, 3>
        {
            SIMD_INLINE void operator()(const __m256i * src, __m256i * dst, __m256i alpha)
            {
                AlphaBlending<align>(src + 0, dst + 0, GrayToBgr<0>(alpha));
                AlphaBlending<align>(src + 1, dst + 1, GrayToBgr<1>(alpha));
                AlphaBlending<align>(src + 2, dst + 2, GrayToBgr<2>(alpha));
            }
        };

        template <bool align> struct AlphaBlender<align, 4>
        {
            SIMD_INLINE void operator()(const __m256i * src, __m256i * dst, __m256i alpha)
            {
                alpha = _mm256_permute4x64_epi64(alpha, 0xD8);
                __m256i lo = _mm256_permute4x64_epi64(_mm256_unpacklo_epi8(alpha, alpha), 0xD8);
                AlphaBlending<align>(src + 0, dst + 0, _mm256_unpacklo_epi8(lo, lo));
                AlphaBlending<align>(src + 1, dst + 1, _mm256_unpackhi_epi8(lo, lo));
                __m256i hi = _mm256_permute4x64_epi64(_mm256_unpackhi_epi8(alpha, alpha), 0xD8);
                AlphaBlending<align>(src + 2, dst + 2, _mm256_unpacklo_epi8(hi, hi));
                AlphaBlending<align>(src + 3, dst + 3, _mm256_unpackhi_epi8(hi, hi));
            }
        };

        template <bool align, size_t channelCount> void AlphaBlending(const uint8_t *src, size_t srcStride, size_t width, size_t height,
            const uint8_t *alpha, size_t alphaStride, uint8_t *dst, size_t dstStride)
        {
            size_t alignedWidth = AlignLo(width, A);
            __m256i tailMask = SetMask<uint8_t>(0, A - width + alignedWidth, 0xFF);
            size_t step = channelCount * A;
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0, offset = 0; col < alignedWidth; col += A, offset += step)
                {
                    __m256i _alpha = Load<align>((__m256i*)(alpha + col));
                    AlphaBlender<align, channelCount>()((__m256i*)(src + offset), (__m256i*)(dst + offset), _alpha);
                }
                if (alignedWidth != width)
                {
                    __m256i _alpha = _mm256_and_si256(Load<false>((__m256i*)(alpha + width - A)), tailMask);
                    AlphaBlender<false, channelCount>()((__m256i*)(src + (width - A)*channelCount), (__m256i*)(dst + (width - A)*channelCount), _alpha);
                }
                src += srcStride;
                alpha += alphaStride;
                dst += dstStride;
            }
        }

        template <bool align> void AlphaBlending(const uint8_t *src, size_t srcStride, size_t width, size_t height, size_t channelCount,
            const uint8_t *alpha, size_t alphaStride, uint8_t *dst, size_t dstStride)
        {
            assert(width >= A);
            if (align)
            {
                assert(Aligned(src) && Aligned(srcStride));
                assert(Aligned(alpha) && Aligned(alphaStride));
                assert(Aligned(dst) && Aligned(dstStride));
            }

            switch (channelCount)
            {
            case 1: AlphaBlending<align, 1>(src, srcStride, width, height, alpha, alphaStride, dst, dstStride); break;
            case 2: AlphaBlending<align, 2>(src, srcStride, width, height, alpha, alphaStride, dst, dstStride); break;
            case 3: AlphaBlending<align, 3>(src, srcStride, width, height, alpha, alphaStride, dst, dstStride); break;
            case 4: AlphaBlending<align, 4>(src, srcStride, width, height, alpha, alphaStride, dst, dstStride); break;
            default:
                assert(0);
            }
        }

        void AlphaBlending(const uint8_t *src, size_t srcStride, size_t width, size_t height, size_t channelCount,
            const uint8_t *alpha, size_t alphaStride, uint8_t *dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride) && Aligned(alpha) && Aligned(alphaStride) && Aligned(dst) && Aligned(dstStride))
                AlphaBlending<true>(src, srcStride, width, height, channelCount, alpha, alphaStride, dst, dstStride);
            else
                AlphaBlending<false>(src, srcStride, width, height, channelCount, alpha, alphaStride, dst, dstStride);
        }

        //-------------------------------------------------------------------------------------------------

        template <bool align, size_t channelCount> struct AlphaBlender2x
        {
            void operator()(const __m256i* src0, __m256i alpha0, const __m256i* src1, __m256i alpha1, __m256i* dst);
        };

        template <bool align> struct AlphaBlender2x<align, 1>
        {
            SIMD_INLINE void operator()(const __m256i* src0, __m256i alpha0, const __m256i* src1, __m256i alpha1, __m256i* dst)
            {
                AlphaBlending2x<align>(src0, alpha0, src1, alpha1, dst);
            }
        };

        template <bool align> struct AlphaBlender2x<align, 2>
        {
            SIMD_INLINE void operator()(const __m256i* src0, __m256i alpha0, const __m256i* src1, __m256i alpha1, __m256i* dst)
            {
                alpha0 = _mm256_permute4x64_epi64(alpha0, 0xD8);
                alpha1 = _mm256_permute4x64_epi64(alpha1, 0xD8);
                AlphaBlending2x<align>(src0 + 0, _mm256_unpacklo_epi8(alpha0, alpha0), src1 + 0, _mm256_unpacklo_epi8(alpha1, alpha1), dst + 0);
                AlphaBlending2x<align>(src0 + 1, _mm256_unpackhi_epi8(alpha0, alpha0), src1 + 1, _mm256_unpackhi_epi8(alpha1, alpha1), dst + 1);
            }
        };

        template <bool align> struct AlphaBlender2x<align, 3>
        {
            SIMD_INLINE void operator()(const __m256i* src0, __m256i alpha0, const __m256i* src1, __m256i alpha1, __m256i* dst)
            {
                AlphaBlending2x<align>(src0 + 0, GrayToBgr<0>(alpha0), src1 + 0, GrayToBgr<0>(alpha1), dst + 0);
                AlphaBlending2x<align>(src0 + 1, GrayToBgr<1>(alpha0), src1 + 1, GrayToBgr<1>(alpha1), dst + 1);
                AlphaBlending2x<align>(src0 + 2, GrayToBgr<2>(alpha0), src1 + 2, GrayToBgr<2>(alpha1), dst + 2);
            }
        };

        template <bool align> struct AlphaBlender2x<align, 4>
        {
            SIMD_INLINE void operator()(const __m256i* src0, __m256i alpha0, const __m256i* src1, __m256i alpha1, __m256i* dst)
            {
                alpha0 = _mm256_permute4x64_epi64(alpha0, 0xD8);
                alpha1 = _mm256_permute4x64_epi64(alpha1, 0xD8);
                __m256i lo0 = _mm256_permute4x64_epi64(_mm256_unpacklo_epi8(alpha0, alpha0), 0xD8);
                __m256i lo1 = _mm256_permute4x64_epi64(_mm256_unpacklo_epi8(alpha1, alpha1), 0xD8);
                AlphaBlending2x<align>(src0 + 0, _mm256_unpacklo_epi8(lo0, lo0), src1 + 0, _mm256_unpacklo_epi8(lo1, lo1), dst + 0);
                AlphaBlending2x<align>(src0 + 1, _mm256_unpackhi_epi8(lo0, lo0), src1 + 1, _mm256_unpackhi_epi8(lo1, lo1), dst + 1);
                __m256i hi0 = _mm256_permute4x64_epi64(_mm256_unpackhi_epi8(alpha0, alpha0), 0xD8);
                __m256i hi1 = _mm256_permute4x64_epi64(_mm256_unpackhi_epi8(alpha1, alpha1), 0xD8);
                AlphaBlending2x<align>(src0 + 2, _mm256_unpacklo_epi8(hi0, hi0), src1 + 2, _mm256_unpacklo_epi8(hi1, hi1), dst + 2);
                AlphaBlending2x<align>(src0 + 3, _mm256_unpackhi_epi8(hi0, hi0), src1 + 3, _mm256_unpackhi_epi8(hi1, hi1), dst + 3);
            }
        };

        template <bool align, size_t channelCount> void AlphaBlending2x(const uint8_t* src0, size_t src0Stride, const uint8_t* alpha0, size_t alpha0Stride,
            const uint8_t* src1, size_t src1Stride, const uint8_t* alpha1, size_t alpha1Stride, size_t width, size_t height, uint8_t* dst, size_t dstStride)
        {
            size_t alignedWidth = AlignLo(width, A);
            __m256i tailMask = SetMask<uint8_t>(0, A - width + alignedWidth, 0xFF);
            size_t step = channelCount * A;
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0, offset = 0; col < alignedWidth; col += A, offset += step)
                {
                    __m256i _alpha0 = Load<align>((__m256i*)(alpha0 + col));
                    __m256i _alpha1 = Load<align>((__m256i*)(alpha1 + col));
                    AlphaBlender2x<align, channelCount>()((__m256i*)(src0 + offset), _alpha0,
                        (__m256i*)(src1 + offset), _alpha1, (__m256i*)(dst + offset));
                }
                if (alignedWidth != width)
                {
                    size_t col = width - A, offset = col * channelCount;
                    __m256i _alpha0 = _mm256_and_si256(Load<false>((__m256i*)(alpha0 + col)), tailMask);
                    __m256i _alpha1 = _mm256_and_si256(Load<false>((__m256i*)(alpha1 + col)), tailMask);
                    AlphaBlender2x<false, channelCount>()((__m256i*)(src0 + offset), _alpha0,
                        (__m256i*)(src1 + offset), _alpha1, (__m256i*)(dst + offset));
                }
                src0 += src0Stride;
                alpha0 += alpha0Stride;
                src1 += src1Stride;
                alpha1 += alpha1Stride;
                dst += dstStride;
            }
        }

        template <bool align> void AlphaBlending2x(const uint8_t* src0, size_t src0Stride, const uint8_t* alpha0, size_t alpha0Stride,
            const uint8_t* src1, size_t src1Stride, const uint8_t* alpha1, size_t alpha1Stride,
            size_t width, size_t height, size_t channelCount, uint8_t* dst, size_t dstStride)
        {
            assert(width >= A);
            if (align)
            {
                assert(Aligned(src0) && Aligned(src0Stride));
                assert(Aligned(alpha0) && Aligned(alpha0Stride));
                assert(Aligned(src1) && Aligned(src1Stride));
                assert(Aligned(alpha1) && Aligned(alpha1Stride));
                assert(Aligned(dst) && Aligned(dstStride));
            }

            switch (channelCount)
            {
            case 1: AlphaBlending2x<align, 1>(src0, src0Stride, alpha0, alpha0Stride, src1, src1Stride, alpha1, alpha1Stride, width, height, dst, dstStride); break;
            case 2: AlphaBlending2x<align, 2>(src0, src0Stride, alpha0, alpha0Stride, src1, src1Stride, alpha1, alpha1Stride, width, height, dst, dstStride); break;
            case 3: AlphaBlending2x<align, 3>(src0, src0Stride, alpha0, alpha0Stride, src1, src1Stride, alpha1, alpha1Stride, width, height, dst, dstStride); break;
            case 4: AlphaBlending2x<align, 4>(src0, src0Stride, alpha0, alpha0Stride, src1, src1Stride, alpha1, alpha1Stride, width, height, dst, dstStride); break;
            default:
                assert(0);
            }
        }

        void AlphaBlending2x(const uint8_t* src0, size_t src0Stride, const uint8_t* alpha0, size_t alpha0Stride,
            const uint8_t* src1, size_t src1Stride, const uint8_t* alpha1, size_t alpha1Stride,
            size_t width, size_t height, size_t channelCount, uint8_t* dst, size_t dstStride)
        {
            if (Aligned(src0) && Aligned(src0Stride) && Aligned(alpha0) && Aligned(alpha0Stride) &&
                Aligned(src1) && Aligned(src1Stride) && Aligned(alpha1) && Aligned(alpha1Stride) && Aligned(dst) && Aligned(dstStride))
                AlphaBlending2x<true>(src0, src0Stride, alpha0, alpha0Stride, src1, src1Stride, alpha1, alpha1Stride, width, height, channelCount, dst, dstStride);
            else
                AlphaBlending2x<false>(src0, src0Stride, alpha0, alpha0Stride, src1, src1Stride, alpha1, alpha1Stride, width, height, channelCount, dst, dstStride);
        }
        
        //-------------------------------------------------------------------------------------------------

        template <class T, int part, bool tail> SIMD_INLINE __m256i LoadAndBgrToY16(const __m256i* bgra, const __m256i& y8, const __m256i& m8, __m256i& b16_r16, __m256i& g16_1, __m256i& a16)
        {
            static const __m256i Y_LO = SIMD_MM256_SET1_EPI16(T::Y_LO);

            __m256i _b16_r16[2], _g16_1[2], a32[2];
            LoadPreparedBgra16<false>(bgra + 0, _b16_r16[0], _g16_1[0], a32[0]);
            LoadPreparedBgra16<false>(bgra + 1, _b16_r16[1], _g16_1[1], a32[1]);
            b16_r16 = PermutedHadd32i(_b16_r16[0], _b16_r16[1]);
            g16_1 = PermutedHadd32i(_g16_1[0], _g16_1[1]);
            a16 = PackI32ToI16(a32[0], a32[1]);
            if (tail)
                a16 = _mm256_and_si256(UnpackU8<part>(m8), a16);
            __m256i y16 = SaturateI16ToU8(_mm256_add_epi16(Y_LO, PackI32ToI16(BgrToY32<T>(_b16_r16[0], _g16_1[0]), BgrToY32<T>(_b16_r16[1], _g16_1[1]))));
            return AlphaBlending16i(y16, UnpackU8<part>(y8), a16);
        }

        template <class T, bool tail> SIMD_INLINE void AlphaBlendingBgraToYuv420p(const uint8_t* bgra0, size_t bgraStride, uint8_t* y0, size_t yStride, uint8_t* u, uint8_t* v, __m256i mask = K_INV_ZERO)
        {
            static const __m256i UV_Z = SIMD_MM256_SET1_EPI16(T::UV_Z);
            const uint8_t* bgra1 = bgra0 + bgraStride;
            uint8_t* y1 = y0 + yStride;

            __m256i b16_r16[2][2], g16_1[2][2], a16[2][2];
            __m256i _y0 = LoadPermuted<false>((__m256i*)y0);
            __m256i y00 = LoadAndBgrToY16<T, 0, tail>((__m256i*)bgra0 + 0, _y0, mask, b16_r16[0][0], g16_1[0][0], a16[0][0]);
            __m256i y01 = LoadAndBgrToY16<T, 1, tail>((__m256i*)bgra0 + 2, _y0, mask, b16_r16[0][1], g16_1[0][1], a16[0][1]);
            Store<false>((__m256i*)y0, PackI16ToU8(y00, y01));

            __m256i _y1 = LoadPermuted<false>((__m256i*)y1);
            __m256i y10 = LoadAndBgrToY16<T, 0, tail>((__m256i*)bgra1 + 0, _y1, mask, b16_r16[1][0], g16_1[1][0], a16[1][0]);
            __m256i y11 = LoadAndBgrToY16<T, 1, tail>((__m256i*)bgra1 + 2, _y1, mask, b16_r16[1][1], g16_1[1][1], a16[1][1]);
            Store<false>((__m256i*)y1, PackI16ToU8(y10, y11));

            b16_r16[0][0] = _mm256_srli_epi16(_mm256_add_epi16(_mm256_add_epi16(b16_r16[0][0], b16_r16[1][0]), K16_0002), 2);
            b16_r16[0][1] = _mm256_srli_epi16(_mm256_add_epi16(_mm256_add_epi16(b16_r16[0][1], b16_r16[1][1]), K16_0002), 2);
            g16_1[0][0] = _mm256_srli_epi16(_mm256_add_epi16(_mm256_add_epi16(g16_1[0][0], g16_1[1][0]), K16_0002), 2);
            g16_1[0][1] = _mm256_srli_epi16(_mm256_add_epi16(_mm256_add_epi16(g16_1[0][1], g16_1[1][1]), K16_0002), 2);
            a16[0][0] = _mm256_srli_epi16(_mm256_add_epi16(_mm256_add_epi16(PermutedHadd16i(a16[0][0], a16[0][1]), PermutedHadd16i(a16[1][0], a16[1][1])), K16_0002), 2);

            __m256i u16 = SaturateI16ToU8(_mm256_add_epi16(UV_Z, PackI32ToI16(BgrToU32<T>(b16_r16[0][0], g16_1[0][0]), BgrToU32<T>(b16_r16[0][1], g16_1[0][1]))));
            u16 = AlphaBlending16i(u16, _mm256_cvtepu8_epi16(Sse41::Load<false>((__m128i*)u)), a16[0][0]);
            StoreHalf<false, 0>((__m128i*)u, PackI16ToU8(u16, K_ZERO));

            __m256i v16 = SaturateI16ToU8(_mm256_add_epi16(UV_Z, PackI32ToI16(BgrToV32<T>(b16_r16[0][0], g16_1[0][0]), BgrToV32<T>(b16_r16[0][1], g16_1[0][1]))));
            v16 = AlphaBlending16i(v16, _mm256_cvtepu8_epi16(Sse41::Load<false>((__m128i*)v)), a16[0][0]);
            StoreHalf<false, 0>((__m128i*)v, PackI16ToU8(v16, K_ZERO));
        }

        template <class T> void AlphaBlendingBgraToYuv420p(const uint8_t* bgra, size_t bgraStride, size_t width, size_t height,
            uint8_t* y, size_t yStride, uint8_t* u, size_t uStride, uint8_t* v, size_t vStride)
        {
            assert((width % 2 == 0) && (height % 2 == 0) && (width >= 2) && (height >= 2));

            size_t widthA = AlignLo(width, A);
            __m256i tailMask = _mm256_permute4x64_epi64(SetMask<uint8_t>(0, A - width + widthA, 0xFF), 0xD8);
            for (size_t row = 0; row < height; row += 2)
            {
                for (size_t colY = 0, colUV = 0, colBgra = 0; colY < widthA; colY += A, colUV += HA, colBgra += QA)
                    AlphaBlendingBgraToYuv420p<T, false>(bgra + colBgra, bgraStride, y + colY, yStride, u + colUV, v + colUV);
                if (widthA != width)
                {
                    size_t colY = width - A, colUV = colY / 2, colBgra = colY * 4;
                    AlphaBlendingBgraToYuv420p<T, true>(bgra + colBgra, bgraStride, y + colY, yStride, u + colUV, v + colUV, tailMask);
                }
                bgra += 2 * bgraStride;
                y += 2 * yStride;
                u += uStride;
                v += vStride;
            }
        }

        void AlphaBlendingBgraToYuv420p(const uint8_t* bgra, size_t bgraStride, size_t width, size_t height,
            uint8_t* y, size_t yStride, uint8_t* u, size_t uStride, uint8_t* v, size_t vStride, SimdYuvType yuvType)
        {
#if defined(SIMD_X86_ENABLE) && defined(NDEBUG) && defined(_MSC_VER) && _MSC_VER <= 1900
            Sse41::AlphaBlendingBgraToYuv420p(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride, yuvType);
#else
            switch (yuvType)
            {
            case SimdYuvBt601: AlphaBlendingBgraToYuv420p<Base::Bt601>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvBt709: AlphaBlendingBgraToYuv420p<Base::Bt709>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvBt2020: AlphaBlendingBgraToYuv420p<Base::Bt2020>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvTrect871: AlphaBlendingBgraToYuv420p<Base::Trect871>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            default:
                assert(0);
            }
#endif
        }

        //-------------------------------------------------------------------------------------------------

        template <bool align> void AlphaBlendingUniform(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            size_t channelCount, uint8_t alpha, uint8_t* dst, size_t dstStride)
        {
            assert(width >= A);
            if (align)
            {
                assert(Aligned(src) && Aligned(srcStride));
                assert(Aligned(dst) && Aligned(dstStride));
            }
            size_t size = width * channelCount;
            size_t sizeA = AlignLo(size, A);
            __m256i _alpha = _mm256_set1_epi8(alpha);
            __m256i tail = _mm256_and_si256(SetMask<uint8_t>(0, A - size + sizeA, 0xFF), _alpha);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t offs = 0; offs < sizeA; offs += A)
                    AlphaBlending<align>((__m256i*)(src + offs), (__m256i*)(dst + offs), _alpha);
                if (sizeA != size)
                    AlphaBlending<false>((__m256i*)(src + size - A), (__m256i*)(dst + size - A), tail);
                src += srcStride;
                dst += dstStride;
            }
        }

        void AlphaBlendingUniform(const uint8_t* src, size_t srcStride, size_t width, size_t height, size_t channelCount,
            uint8_t alpha, uint8_t* dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride))
                AlphaBlendingUniform<true>(src, srcStride, width, height, channelCount, alpha, dst, dstStride);
            else
                AlphaBlendingUniform<false>(src, srcStride, width, height, channelCount, alpha, dst, dstStride);
        }

        //-----------------------------------------------------------------------------------------

        template <bool align> SIMD_INLINE void AlphaFilling(__m256i * dst, __m256i channelLo, __m256i channelHi, __m256i alpha)
        {
            __m256i _dst = Load<align>(dst);
            __m256i lo = AlphaBlending16i(channelLo, _mm256_unpacklo_epi8(_dst, K_ZERO), _mm256_unpacklo_epi8(alpha, K_ZERO));
            __m256i hi = AlphaBlending16i(channelHi, _mm256_unpackhi_epi8(_dst, K_ZERO), _mm256_unpackhi_epi8(alpha, K_ZERO));
            Store<align>(dst, _mm256_packus_epi16(lo, hi));
        }

        template <bool align, size_t channelCount> struct AlphaFiller
        {
            void operator() (__m256i * dst, const __m256i * channel, __m256i alpha);
        };

        template <bool align> struct AlphaFiller<align, 1>
        {
            SIMD_INLINE void operator()(__m256i * dst, const __m256i * channel, __m256i alpha)
            {
                AlphaFilling<align>(dst, channel[0], channel[0], alpha);
            }
        };

        template <bool align> struct AlphaFiller<align, 2>
        {
            SIMD_INLINE void operator()(__m256i * dst, const __m256i * channel, __m256i alpha)
            {
                alpha = _mm256_permute4x64_epi64(alpha, 0xD8);
                AlphaFilling<align>(dst + 0, channel[0], channel[0], UnpackU8<0>(alpha, alpha));
                AlphaFilling<align>(dst + 1, channel[0], channel[0], UnpackU8<1>(alpha, alpha));
            }
        };

        template <bool align> struct AlphaFiller<align, 3>
        {
            SIMD_INLINE void operator()(__m256i * dst, const __m256i * channel, __m256i alpha)
            {
                AlphaFilling<align>(dst + 0, channel[0], channel[1], GrayToBgr<0>(alpha));
                AlphaFilling<align>(dst + 1, channel[1], channel[2], GrayToBgr<1>(alpha));
                AlphaFilling<align>(dst + 2, channel[2], channel[0], GrayToBgr<2>(alpha));
            }
        };

        template <bool align> struct AlphaFiller<align, 4>
        {
            SIMD_INLINE void operator()(__m256i * dst, const __m256i * channel, __m256i alpha)
            {
                alpha = _mm256_permute4x64_epi64(alpha, 0xD8);
                __m256i lo = _mm256_permute4x64_epi64(_mm256_unpacklo_epi8(alpha, alpha), 0xD8);
                AlphaFilling<align>(dst + 0, channel[0], channel[0], UnpackU8<0>(lo, lo));
                AlphaFilling<align>(dst + 1, channel[0], channel[0], UnpackU8<1>(lo, lo));
                __m256i hi = _mm256_permute4x64_epi64(_mm256_unpackhi_epi8(alpha, alpha), 0xD8);
                AlphaFilling<align>(dst + 2, channel[0], channel[0], UnpackU8<0>(hi, hi));
                AlphaFilling<align>(dst + 3, channel[0], channel[0], UnpackU8<1>(hi, hi));
            }
        };

        template <bool align, size_t channelCount> void AlphaFilling(uint8_t * dst, size_t dstStride, size_t width, size_t height, const __m256i * channel, const uint8_t * alpha, size_t alphaStride)
        {
            size_t alignedWidth = AlignLo(width, A);
            __m256i tailMask = SetMask<uint8_t>(0, A - width + alignedWidth, 0xFF);
            size_t step = channelCount * A;
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0, offset = 0; col < alignedWidth; col += A, offset += step)
                {
                    __m256i _alpha = Load<align>((__m256i*)(alpha + col));
                    AlphaFiller<align, channelCount>()((__m256i*)(dst + offset), channel, _alpha);
                }
                if (alignedWidth != width)
                {
                    __m256i _alpha = _mm256_and_si256(Load<false>((__m256i*)(alpha + width - A)), tailMask);
                    AlphaFiller<false, channelCount>()((__m256i*)(dst + (width - A)*channelCount), channel, _alpha);
                }
                alpha += alphaStride;
                dst += dstStride;
            }
        }

        template <bool align> void AlphaFilling(uint8_t * dst, size_t dstStride, size_t width, size_t height, const uint8_t * channel, size_t channelCount, const uint8_t * alpha, size_t alphaStride)
        {
            assert(width >= A);
            if (align)
            {
                assert(Aligned(dst) && Aligned(dstStride));
                assert(Aligned(alpha) && Aligned(alphaStride));
            }

            __m256i _channel[3];
            switch (channelCount)
            {
            case 1:
                _channel[0] = UnpackU8<0>(_mm256_set1_epi8(*(uint8_t*)channel));
                AlphaFilling<align, 1>(dst, dstStride, width, height, _channel, alpha, alphaStride);
                break;
            case 2:
                _channel[0] = UnpackU8<0>(_mm256_set1_epi16(*(uint16_t*)channel));
                AlphaFilling<align, 2>(dst, dstStride, width, height, _channel, alpha, alphaStride);
                break;
            case 3:
                _channel[0] = _mm256_setr_epi16(
                    channel[0], channel[1], channel[2], channel[0], channel[1], channel[2], channel[0], channel[1],
                    channel[1], channel[2], channel[0], channel[1], channel[2], channel[0], channel[1], channel[2]);
                _channel[1] = _mm256_setr_epi16(
                    channel[2], channel[0], channel[1], channel[2], channel[0], channel[1], channel[2], channel[0],
                    channel[0], channel[1], channel[2], channel[0], channel[1], channel[2], channel[0], channel[1]);
                _channel[2] = _mm256_setr_epi16(
                    channel[1], channel[2], channel[0], channel[1], channel[2], channel[0], channel[1], channel[2],
                    channel[2], channel[0], channel[1], channel[2], channel[0], channel[1], channel[2], channel[0]);
                AlphaFilling<align, 3>(dst, dstStride, width, height, _channel, alpha, alphaStride);
                break;
            case 4:
                _channel[0] = UnpackU8<0>(_mm256_set1_epi32(*(uint32_t*)channel));
                AlphaFilling<align, 4>(dst, dstStride, width, height, _channel, alpha, alphaStride);
                break;
            default:
                assert(0);
            }
        }

        void AlphaFilling(uint8_t * dst, size_t dstStride, size_t width, size_t height, const uint8_t * channel, size_t channelCount, const uint8_t * alpha, size_t alphaStride)
        {
            if (Aligned(dst) && Aligned(dstStride) && Aligned(alpha) && Aligned(alphaStride))
                AlphaFilling<true>(dst, dstStride, width, height, channel, channelCount, alpha, alphaStride);
            else
                AlphaFilling<false>(dst, dstStride, width, height, channel, channelCount, alpha, alphaStride);
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE __m256i AlphaPremultiply16i(__m256i value, __m256i alpha)
        {
            return Divide16uBy255(_mm256_mullo_epi16(value, alpha));
        }

        template<bool argb> void AlphaPremultiply(const uint8_t* src, uint8_t* dst);

        template<> SIMD_INLINE void AlphaPremultiply<false>(const uint8_t* src, uint8_t* dst)
        {
            static const __m256i K8_SHUFFLE_BGRA_TO_A0A0 = SIMD_MM256_SETR_EPI8(
                0x3, -1, 0x3, -1, 0x7, -1, 0x7, -1, 0xB, -1, 0xB, -1, 0xF, -1, 0xF, -1,
                0x3, -1, 0x3, -1, 0x7, -1, 0x7, -1, 0xB, -1, 0xB, -1, 0xF, -1, 0xF, -1);
            __m256i bgra = _mm256_loadu_si256((__m256i*)src);
            __m256i a0a0 = _mm256_shuffle_epi8(bgra, K8_SHUFFLE_BGRA_TO_A0A0);
            __m256i b0r0 = _mm256_and_si256(bgra, K16_00FF);
            __m256i g0f0 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_si256(bgra, 1), K32_000000FF), K32_00FF0000);
            __m256i B0R0 = AlphaPremultiply16i(b0r0, a0a0);
            __m256i G0A0 = AlphaPremultiply16i(g0f0, a0a0);
            _mm256_storeu_si256((__m256i*)dst, _mm256_or_si256(B0R0, _mm256_slli_si256(G0A0, 1)));
        }

        template<> SIMD_INLINE void AlphaPremultiply<true>(const uint8_t* src, uint8_t* dst)
        {
            static const __m256i K8_SHUFFLE_ARGB_TO_A0A0 = SIMD_MM256_SETR_EPI8(
                0x0, -1, 0x0, -1, 0x4, -1, 0x4, -1, 0x8, -1, 0x8, -1, 0xC, -1, 0xC, -1,
                0x0, -1, 0x0, -1, 0x4, -1, 0x4, -1, 0x8, -1, 0x8, -1, 0xC, -1, 0xC, -1);
            __m256i argb = _mm256_loadu_si256((__m256i*)src);
            __m256i a0a0 = _mm256_shuffle_epi8(argb, K8_SHUFFLE_ARGB_TO_A0A0);
            __m256i f0g0 = _mm256_or_si256(_mm256_and_si256(argb, K32_00FF0000), K32_000000FF);
            __m256i r0b0 = _mm256_and_si256(_mm256_srli_si256(argb, 1), K16_00FF);
            __m256i F0A0 = AlphaPremultiply16i(f0g0, a0a0);
            __m256i R0B0 = AlphaPremultiply16i(r0b0, a0a0);
            _mm256_storeu_si256((__m256i*)dst, _mm256_or_si256(F0A0, _mm256_slli_si256(R0B0, 1)));
        }

        template<bool argb> void AlphaPremultiply(const uint8_t* src, size_t srcStride, size_t width, size_t height, uint8_t* dst, size_t dstStride)
        {
            size_t size = width * 4;
            size_t sizeA = AlignLo(size, A);
            for (size_t row = 0; row < height; ++row)
            {
                size_t i = 0;
                for (; i < sizeA; i += A)
                    AlphaPremultiply<argb>(src + i, dst + i);
                for (; i < size; i += 4)
                    Base::AlphaPremultiply<argb>(src + i, dst + i);
                src += srcStride;
                dst += dstStride;
            }
        }

        void AlphaPremultiply(const uint8_t* src, size_t srcStride, size_t width, size_t height, uint8_t* dst, size_t dstStride, SimdBool argb)
        {
            if (argb)
                AlphaPremultiply<true>(src, srcStride, width, height, dst, dstStride);
            else
                AlphaPremultiply<false>(src, srcStride, width, height, dst, dstStride);
        }

        //-----------------------------------------------------------------------------------------

        const __m256i K8_SHUFFLE_0123_TO_0 = SIMD_MM256_SETR_EPI8(
            0x0, -1, -1, -1, 0x4, -1, -1, -1, 0x8, -1, -1, -1, 0xC, -1, -1, -1,
            0x0, -1, -1, -1, 0x4, -1, -1, -1, 0x8, -1, -1, -1, 0xC, -1, -1, -1);
        const __m256i K8_SHUFFLE_0123_TO_1 = SIMD_MM256_SETR_EPI8(
            0x1, -1, -1, -1, 0x5, -1, -1, -1, 0x9, -1, -1, -1, 0xD, -1, -1, -1,
            0x1, -1, -1, -1, 0x5, -1, -1, -1, 0x9, -1, -1, -1, 0xD, -1, -1, -1);
        const __m256i K8_SHUFFLE_0123_TO_2 = SIMD_MM256_SETR_EPI8(
            0x2, -1, -1, -1, 0x6, -1, -1, -1, 0xA, -1, -1, -1, 0xE, -1, -1, -1,
            0x2, -1, -1, -1, 0x6, -1, -1, -1, 0xA, -1, -1, -1, 0xE, -1, -1, -1);
        const __m256i K8_SHUFFLE_0123_TO_3 = SIMD_MM256_SETR_EPI8(
            0x3, -1, -1, -1, 0x7, -1, -1, -1, 0xB, -1, -1, -1, 0xF, -1, -1, -1,
            0x3, -1, -1, -1, 0x7, -1, -1, -1, 0xB, -1, -1, -1, 0xF, -1, -1, -1);

        template<bool argb> void AlphaUnpremultiply(const uint8_t* src, uint8_t* dst, __m256 _255);

        template<> SIMD_INLINE void AlphaUnpremultiply<false>(const uint8_t* src, uint8_t* dst, __m256 _255)
        {
            __m256i _src = _mm256_loadu_si256((__m256i*)src);
            __m256i b = _mm256_shuffle_epi8(_src, K8_SHUFFLE_0123_TO_0);
            __m256i g = _mm256_shuffle_epi8(_src, K8_SHUFFLE_0123_TO_1);
            __m256i r = _mm256_shuffle_epi8(_src, K8_SHUFFLE_0123_TO_2);
            __m256i a = _mm256_shuffle_epi8(_src, K8_SHUFFLE_0123_TO_3);
            __m256 k = _mm256_cvtepi32_ps(a);
            k = _mm256_blendv_ps(_mm256_div_ps(_255, k), k, _mm256_cmp_ps(k, _mm256_setzero_ps(), _CMP_EQ_OQ));
            b = _mm256_cvtps_epi32(_mm256_min_ps(_mm256_floor_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(b), k)), _255));
            g = _mm256_cvtps_epi32(_mm256_min_ps(_mm256_floor_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(g), k)), _255));
            r = _mm256_cvtps_epi32(_mm256_min_ps(_mm256_floor_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(r), k)), _255));
            __m256i _dst = _mm256_or_si256(b, _mm256_slli_si256(g, 1));
            _dst = _mm256_or_si256(_dst, _mm256_slli_si256(r, 2));
            _dst = _mm256_or_si256(_dst, _mm256_slli_si256(a, 3));
            _mm256_storeu_si256((__m256i*)dst, _dst);
        }

        template<> SIMD_INLINE void AlphaUnpremultiply<true>(const uint8_t* src, uint8_t* dst, __m256 _255)
        {
            __m256i _src = _mm256_loadu_si256((__m256i*)src);
            __m256i a = _mm256_shuffle_epi8(_src, K8_SHUFFLE_0123_TO_0);
            __m256i r = _mm256_shuffle_epi8(_src, K8_SHUFFLE_0123_TO_1);
            __m256i g = _mm256_shuffle_epi8(_src, K8_SHUFFLE_0123_TO_2);
            __m256i b = _mm256_shuffle_epi8(_src, K8_SHUFFLE_0123_TO_3);
            __m256 k = _mm256_cvtepi32_ps(a);
            k = _mm256_blendv_ps(_mm256_div_ps(_255, k), k, _mm256_cmp_ps(k, _mm256_setzero_ps(), _CMP_EQ_OQ));
            b = _mm256_cvtps_epi32(_mm256_min_ps(_mm256_floor_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(b), k)), _255));
            g = _mm256_cvtps_epi32(_mm256_min_ps(_mm256_floor_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(g), k)), _255));
            r = _mm256_cvtps_epi32(_mm256_min_ps(_mm256_floor_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(r), k)), _255));
            __m256i _dst = _mm256_or_si256(a, _mm256_slli_si256(r, 1));
            _dst = _mm256_or_si256(_dst, _mm256_slli_si256(g, 2));
            _dst = _mm256_or_si256(_dst, _mm256_slli_si256(b, 3));
            _mm256_storeu_si256((__m256i*)dst, _dst);
        }

        template<bool argb> void AlphaUnpremultiply(const uint8_t* src, size_t srcStride, size_t width, size_t height, uint8_t* dst, size_t dstStride)
        {
            __m256 _255 = _mm256_set1_ps(255.00001f);
            size_t size = width * 4;
            size_t sizeA = AlignLo(size, A);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < sizeA; col += A)
                    AlphaUnpremultiply<argb>(src + col, dst + col, _255);
                for (; col < size; col += 4)
                    Base::AlphaUnpremultiply<argb>(src + col, dst + col);
                src += srcStride;
                dst += dstStride;
            }
        }

        void AlphaUnpremultiply(const uint8_t* src, size_t srcStride, size_t width, size_t height, uint8_t* dst, size_t dstStride, SimdBool argb)
        {
            if (argb)
                AlphaUnpremultiply<true>(src, srcStride, width, height, dst, dstStride);
            else
                AlphaUnpremultiply<false>(src, srcStride, width, height, dst, dstStride);
        }
    }
#endif
}
