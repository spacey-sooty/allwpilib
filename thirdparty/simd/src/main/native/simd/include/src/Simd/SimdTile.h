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
#ifndef __SimdTile_h__
#define __SimdTile_h__

#include "Simd/SimdDefs.h"

namespace Simd
{
    struct TileConf
    {
        uint8_t paletteId;
        uint8_t startRow;
        uint8_t reserved[14];
        uint16_t colsb[16];
        uint8_t rows[16];

        SIMD_INLINE TileConf(bool zero = true)
        {
            uint64_t* dst = (uint64_t*)this;
            dst[0] = 0x0000000000000001;
            if (zero)
            {
                dst[1] = 0x0000000000000000;
                dst[2] = 0x0000000000000000;
                dst[3] = 0x0000000000000000;
                dst[4] = 0x0000000000000000;
                dst[5] = 0x0000000000000000;
                dst[6] = 0x0000000000000000;
                dst[7] = 0x0000000000000000;
            }
            else
            {
                dst[1] = 0x0000000000000000;
                dst[2] = 0x0040004000400040;
                dst[3] = 0x0040004000400040;
                dst[4] = 0x0000000000000000;
                dst[5] = 0x0000000000000000;
                dst[6] = 0x1010101010101010;
                dst[7] = 0x0000000000000000;
            }
        }
    };
}

//-------------------------------------------------------------------------------------------------

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE  
    namespace Avx512bw
    {
#if defined (SIMD_AMX_EMULATE)
        union SIMD_ALIGNED(64) TileReg
        {
            int8_t i8[16][64];
            uint8_t u8[16][64];
            int16_t i16[16][32];
            uint16_t u16[16][32];
            int32_t i32[16][16];
            uint32_t u32[16][16];
            float f32[16][16];
        };

        struct Tile1024
        {
            uint16_t row;
            uint16_t col;
            TileReg tile;
        };

        const size_t TileRegCount = 8;

        //-------------------------------------------------------------------------------------------------

        void TileLoadConfig(const TileConf* tileConf);

        void TileStoreConfig(TileConf* tileConf);

        void TileRelease();

        void TileZero(int dst);

        void TileZero(Tile1024* dst);

        void TileLoad(int dst, const void* base, int stride);

        void TileLoad(Tile1024* dst, const void* base, int stride);

        void TileStore(int src, void* base, int stride);

        void TileStore(void* base, int stride, const Tile1024& src);

        void TileMatMul8u8i(int dst, int a, int b);

        void TileMatMul8u8i(Tile1024* dst, const Tile1024& a, const Tile1024& b);

        void TileMatMulBf16(int dst, int a, int b);

        void TileMatMulBf16(Tile1024* dst, const Tile1024& a, const Tile1024& b);

        void TileMatMulFp16(int dst, int a, int b);

        void TileMatMulFp16(Tile1024* dst, const Tile1024& a, const Tile1024& b);
#endif
    }
#endif

#ifdef SIMD_AMXBF16_ENABLE
    namespace AmxBf16
    {
        SIMD_INLINE bool TileConfChanged(size_t rows, size_t cols)
        {
            static thread_local size_t _size = 0;
            size_t size = (rows << 16) | cols;
            if (size != _size)
            {
                _size = size;
                return true;
            }
            return false;
        }

        SIMD_INLINE void SetTileConfFull(bool force = false)
        {
            if (TileConfChanged(32, 32) || force)
            {
                TileConf conf = TileConf(false);
                _tile_loadconfig(&conf);
            }
        }

        SIMD_INLINE void SetTileConf2x2(size_t rows, size_t cols)
        {
            if (TileConfChanged(rows, cols))
            {
                TileConf conf = TileConf(false);
                uint8_t tailR = uint8_t(rows - 16);
                conf.rows[2] = tailR;
                conf.rows[3] = tailR;
                conf.rows[5] = tailR;
                uint16_t tailC = uint16_t((cols - 16) * 4);
                conf.colsb[1] = tailC;
                conf.colsb[3] = tailC;
                conf.colsb[7] = tailC;
                _tile_loadconfig(&conf);
            }
        }

        SIMD_INLINE void SetTileConf2x1(size_t rows, size_t cols)
        {
            if (TileConfChanged(rows, cols))
            {
                TileConf conf = TileConf(false);
                uint8_t tailR = uint8_t(rows - 16);
                conf.rows[2] = tailR;
                conf.rows[5] = tailR;
                uint16_t tailC = uint16_t(cols * 4);
                conf.colsb[0] = tailC;
                conf.colsb[2] = tailC;
                conf.colsb[6] = tailC;
                _tile_loadconfig(&conf);
            }
        }

        SIMD_INLINE void SetTileConf1x2(size_t rows, size_t cols)
        {
            if (TileConfChanged(rows, cols))
            {
                TileConf conf = TileConf(false);
                uint8_t tailR = uint8_t(rows);
                conf.rows[0] = tailR;
                conf.rows[1] = tailR;
                conf.rows[4] = tailR;
                uint16_t tailC = uint16_t((cols - 16) * 4);
                conf.colsb[1] = tailC;
                conf.colsb[7] = tailC;
                _tile_loadconfig(&conf);
            }
        }

        SIMD_INLINE void SetTileConf1x1(size_t rows, size_t cols)
        {
            if (TileConfChanged(rows, cols))
            {
                TileConf conf = TileConf(false);
                uint8_t tailR = uint8_t(rows);
                conf.rows[0] = tailR;
                conf.rows[4] = tailR;
                uint16_t tailC = uint16_t(cols * 4);
                conf.colsb[0] = tailC;
                conf.colsb[6] = tailC;
                _tile_loadconfig(&conf);
            }
        }

        //-------------------------------------------------------------------------------------------------

        template<class T> SIMD_INLINE void TileMoveToMemory(const T* ptr, size_t stride, size_t count = 16)
        {
            for (const T* end = ptr + stride * count; ptr < end; ptr += 4 * stride)
            {
                _mm_prefetch((const char*)(ptr + 0 * stride), _MM_HINT_NTA);
                _mm_prefetch((const char*)(ptr + 1 * stride), _MM_HINT_NTA);
                _mm_prefetch((const char*)(ptr + 2 * stride), _MM_HINT_NTA);
                _mm_prefetch((const char*)(ptr + 3 * stride), _MM_HINT_NTA);
            }
        }
    }
#endif
}

#if defined(SIMD_AMX_EMULATE)  

#ifdef _tile_loadconfig
#undef _tile_loadconfig
#endif
#define _tile_loadconfig Simd::Avx512bw::TileLoadConfig

#ifdef _tile_storeconfig
#undef _tile_storeconfig
#endif
#define _tile_storeconfig Simd::Avx512bw::TileStoreConfig

#ifdef _tile_release
#undef _tile_release
#endif
#define _tile_release Simd::Avx512bw::TileRelease

#ifdef _tile_loadd
#undef _tile_loadd
#endif
#define _tile_loadd Simd::Avx512bw::TileLoad

#ifdef _tile_stream_loadd
#undef _tile_stream_loadd
#endif
#define _tile_stream_loadd Simd::Avx512bw::TileLoad

#ifdef _tile_stored
#undef _tile_stored
#endif
#define _tile_stored Simd::Avx512bw::TileStore

#ifdef _tile_zero
#undef _tile_zero
#endif
#define _tile_zero Simd::Avx512bw::TileZero

#ifdef _tile_dpbusd
#undef _tile_dpbusd
#endif
#define _tile_dpbusd Simd::Avx512bw::TileMatMul8u8i

#ifdef _tile_dpbf16ps
#undef _tile_dpbf16ps
#endif
#define _tile_dpbf16ps Simd::Avx512bw::TileMatMulBf16

#ifdef _tile_dpfp16ps
#undef _tile_dpfp16ps
#endif
#define _tile_dpfp16ps Simd::Avx512bw::TileMatMulFp16

#endif

#endif
