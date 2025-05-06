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
#include "Simd/SimdWarpAffine.h"
#include "Simd/SimdWarpAffineCommon.h"
#include "Simd/SimdCopy.h"
#include "Simd/SimdBase.h"

#include "Simd/SimdPoint.hpp"
#include "Simd/SimdParallel.hpp"

namespace Simd
{
    static SIMD_INLINE void SetInv(const float * mat, float * inv)
    {
        double D = mat[0] * mat[4] - mat[1] * mat[3];
        D = D != 0.0 ? 1.0 / D : 0.0;
        double A11 = mat[4] * D;
        double A22 = mat[0] * D;
        double A12 = -mat[1] * D;
        double A21 = -mat[3] * D;
        double b1 = -A11 * mat[2] - A12 * mat[5];
        double b2 = -A21 * mat[2] - A22 * mat[5];
        inv[0] = (float)A11;
        inv[1] = (float)A12;
        inv[2] = (float)b1;
        inv[3] = (float)A21;
        inv[4] = (float)A22;
        inv[5] = (float)b2;
    }

    WarpAffParam::WarpAffParam(size_t srcW, size_t srcH, size_t srcS, size_t dstW, size_t dstH, size_t dstS, size_t channels, const float* mat, SimdWarpAffineFlags flags, const uint8_t* border, size_t align)
    {
        this->srcW = srcW;
        this->srcH = srcH;
        this->srcS = srcS;
        this->dstW = dstW;
        this->dstH = dstH;
        this->dstS = dstS;
        this->channels = channels;
        memcpy(this->mat, mat, 6 * sizeof(float));
        this->flags = flags;
        memset(this->border, 0, BorderSizeMax);
        if (border && (flags & SimdWarpAffineBorderMask) == SimdWarpAffineBorderConstant)
            memcpy(this->border, border, this->PixelSize());
        this->align = align;
        SetInv(this->mat, this->inv);
    }

    //---------------------------------------------------------------------------------------------

    WarpAffine::WarpAffine(const WarpAffParam& param)
        : _param(param)
        , _first(true)
        , _threads(Base::GetThreadNumber())
    {
    }

    //---------------------------------------------------------------------------------------------

    namespace Base
    {
        template<int N> SIMD_INLINE void FillBorder(uint8_t* dst, int count, const uint8_t* bs)
        {
            int i = 0, size = count * N;
            for (int i = 0; i < size; i += N)
                Base::CopyPixel<N>(bs, dst + i);
        }

        template<> SIMD_INLINE void FillBorder<3>(uint8_t* dst, int count, const uint8_t* bs)
        {
            int i = 0, size = count * 3, size3 = size - 3;
            for (; i < size3; i += 3)
                Base::CopyPixel<4>(bs, dst + i);
            for (; i < size; i += 3)
                Base::CopyPixel<3>(bs, dst + i);
        }

        //---------------------------------------------------------------------------------------------

        template<int N> void NearestRun(const WarpAffParam& p, int yBeg, int yEnd, const int32_t* beg, const int32_t* end, const uint8_t* src, uint8_t* dst, uint32_t* buf)
        {
            bool fill = p.NeedFill();
            int width = (int)p.dstW, s = (int)p.srcS, w = (int)p.srcW - 1, h = (int)p.srcH - 1;
            dst += yBeg * p.dstS;
            for (int y = yBeg; y < yEnd; ++y)
            {
                int nose = beg[y], tail = end[y];
                if (fill)
                    FillBorder<N>(dst, nose, p.border);
                if (N == 3)
                {
                    int x = nose, tail1 = tail - 1;
                    for (; x < tail1; ++x)
                        Base::CopyPixel<4>(src + NearestOffset<3>(x, y, p.inv, w, h, s), dst + x * 3);
                    for (; x < tail; ++x)
                        Base::CopyPixel<3>(src + NearestOffset<3>(x, y, p.inv, w, h, s), dst + x * 3);
                }
                else
                {
                    for (int x = nose; x < tail; ++x)
                        CopyPixel<N>(src + NearestOffset<N>(x, y, p.inv, w, h, s), dst + x * N);
                }
                if (fill)
                    FillBorder<N>(dst + tail * N, width - tail, p.border);
                dst += p.dstS;
            }
        }

        //---------------------------------------------------------------------------------------------

        WarpAffineNearest::WarpAffineNearest(const WarpAffParam& param)
            : WarpAffine(param)
        {
            switch (_param.channels)
            {
            case 1: _run = NearestRun<1>; break;
            case 2: _run = NearestRun<2>; break;
            case 3: _run = NearestRun<3>; break;
            case 4: _run = NearestRun<4>; break;
            }
        }

        void WarpAffineNearest::Run(const uint8_t* src, uint8_t* dst)
        {
            if(_first)
                Init();

            Simd::Parallel(0, _param.dstH, [&](size_t thread, size_t begin, size_t end)
            {
                _run(_param, (int)begin, (int)end, _beg.data, _end.data, src, dst, (uint32_t*)(_buf.data + thread * _size));
            }, _threads, 1);

            _first = false;
        }

        SIMD_INLINE Point Conv(float x, float y, const float* m)
        {
            return Point(x * m[0] + y * m[1] + m[2], x * m[3] + y * m[4] + m[5]);
        }

        void WarpAffineNearest::Init()
        {
            const WarpAffParam& p = _param;
            _beg.Resize(p.dstH);
            _end.Resize(p.dstH);
            _size = (AlignHi(p.dstW, p.align) + p.align) * 4;
            _buf.Resize(_size * _threads);
            float w = (float)(p.srcW - 1), h = (float)(p.srcH - 1);
            Point points[4];
            points[0] = Conv(0, 0, p.mat);
            points[1] = Conv(w, 0, p.mat);
            points[2] = Conv(w, h, p.mat);
            points[3] = Conv(0, h, p.mat);
            SetRange(points);
        }

        void WarpAffineNearest::SetRange(const Base::Point* points)
        {
            const WarpAffParam& p = _param;
            int w = (int)p.dstW;
            for (size_t y = 0; y < p.dstH; ++y)
            {
                _beg[y] = w;
                _end[y] = 0;
            }
            for (int v = 0; v < 4; ++v)
            {
                const Point& curr = points[v];
                const Point& next = points[(v + 1) & 3];
                float yMin = Simd::Max(Simd::Min(curr.y, next.y), 0.0f);
                float yMax = Simd::Min(Simd::Max(curr.y, next.y), (float)p.dstH);
                int yBeg = Round(yMin);
                int yEnd = Round(yMax);
                if (next.y == curr.y)
                    continue;
                float a = (next.x - curr.x) / (next.y - curr.y);
                float b = curr.x - curr.y * a;
                if (abs(a) <= 1.0f)
                {
                    for (int y = yBeg; y < yEnd; ++y)
                    {
                        int x = Round(y * a + b);
                        _beg[y] = Simd::Min(_beg[y], Simd::Max(x, 0));
                        _end[y] = Simd::Max(_end[y], Simd::Min(x + 1, w));
                    }
                }
                else
                {
                    for (int y = yBeg; y < yEnd; ++y)
                    {
                        float xM = b + Simd::RestrictRange(float(y) - 0.5f, yMin, yMax) * a;
                        float xP = b + Simd::RestrictRange(float(y) + 0.5f, yMin, yMax) * a;
                        int xBeg = Round(Simd::Min(xM, xP));
                        int xEnd = Round(Simd::Max(xM, xP));
                        _beg[y] = Simd::Min(_beg[y], Simd::Max(xBeg, 0));
                        _end[y] = Simd::Max(_end[y], Simd::Min(xEnd + 1, w));
                    }
                }
            }
        }

        //-----------------------------------------------------------------------------------------

        template<int N> void ByteBilinearRun(const WarpAffParam& p, int yBeg, int yEnd, const int* ib, const int* ie, const int* ob, const int* oe, const uint8_t* src, uint8_t* dst, uint8_t* buf)
        {
            constexpr int M = (N == 3 ? 4 : N);
            bool fill = p.NeedFill();
            int width = (int)p.dstW, s = (int)p.srcS, w = (int)p.srcW - 2, h = (int)p.srcH - 2;
            size_t wa = AlignHi(p.dstW, p.align);
            uint32_t* offs = (uint32_t*)buf;
            uint8_t* fx = (uint8_t*)(offs + wa);
            uint16_t* fy = (uint16_t*)(fx + wa * 2);
            uint8_t* rb0 = (uint8_t*)(fy + wa * 2);
            uint8_t* rb1 = (uint8_t*)(rb0 + wa * M * 2);
            dst += yBeg * p.dstS;
            for (int y = yBeg; y < yEnd; ++y)
            {
                int iB = ib[y], iE = ie[y], oB = ob[y], oE = oe[y];
                if (fill)
                {
                    FillBorder<N>(dst, oB, p.border);
                    for (int x = oB; x < iB; ++x)
                        ByteBilinearInterpEdge<N>(x, y, p.inv, w, h, s, src, p.border, dst + x * N);
                }
                else
                {
                    for (int x = oB; x < iB; ++x)
                        ByteBilinearInterpEdge<N>(x, y, p.inv, w, h, s, src, dst + x * N, dst + x * N);
                }
                {
                    for (int x = iB; x < iE; ++x)
                        Base::ByteBilinearPrepMain(x, y, p.inv, N, s, offs + x, fx + 2 * x, fy + 2 * x);
                    ByteBilinearGather<M>(src, src + s, offs + iB, iE - iB, rb0 + 2 * M * iB, rb1 + 2 * M * iB);
                    for (int x = iB; x < iE; ++x)
                        Base::ByteBilinearInterpMain<N>(rb0 + x * M * 2, rb1 + x * M * 2, fx + 2 * x, fy + 2 * x, dst + x * N);
                }
                if (fill)
                {
                    for (int x = iE; x < oE; ++x)
                        ByteBilinearInterpEdge<N>(x, y, p.inv, w, h, s, src, p.border, dst + x * N);
                    FillBorder<N>(dst + oE * N, width - oE, p.border);
                }
                else
                {
                    for (int x = iE; x < oE; ++x)
                        ByteBilinearInterpEdge<N>(x, y, p.inv, w, h, s, src, dst + x * N, dst + x * N);
                }
                dst += p.dstS;
            }
        }

        //---------------------------------------------------------------------------------------------

        WarpAffineByteBilinear::WarpAffineByteBilinear(const WarpAffParam& param)
            : WarpAffine(param)
        {
            switch (_param.channels)
            {
            case 1: _run = ByteBilinearRun<1>; break;
            case 2: _run = ByteBilinearRun<2>; break;
            case 3: _run = ByteBilinearRun<3>; break;
            case 4: _run = ByteBilinearRun<4>; break;
            }
        }

        void WarpAffineByteBilinear::Run(const uint8_t* src, uint8_t* dst)
        {
            if (_first)
                Init();

            Simd::Parallel(0, _param.dstH, [&](size_t thread, size_t begin, size_t end)
            {
                _run(_param, (int)begin, (int)end, _ib, _ie, _ob, _oe, src, dst, _buf.data + thread * _size);
            }, _threads, 1);

            _first = false;
        }

        void WarpAffineByteBilinear::Init()
        {
            const WarpAffParam& p = _param;
            _range.Resize(p.dstH * 4);
            _ib = _range.data + 0 * p.dstH;
            _ie = _range.data + 1 * p.dstH;
            _ob = _range.data + 2 * p.dstH;
            _oe = _range.data + 3 * p.dstH;
            size_t na = (p.channels == 3 ? 4 : p.channels), wa = AlignHi(p.dstW, p.align) + p.align;
            _size = Simd::Max(wa * 10 + wa * na * 4, p.dstH * 8);
            _buf.Resize(_size * _threads);
            float z, h, w, e = 0.0001f;
            Point rect[4];
            z = -1.0f + e, w = (float)(p.srcW + 0) - e, h = (float)(p.srcH + 0) - e;
            rect[0] = Conv(z, z, p.mat);
            rect[1] = Conv(w, z, p.mat);
            rect[2] = Conv(w, h, p.mat);
            rect[3] = Conv(z, h, p.mat);
            SetRange(rect, _ob, _oe, NULL, NULL);
            z = 0.0f + e, w = (float)(p.srcW - 1) - e, h = (float)(p.srcH - 1) - e;
            rect[0] = Conv(z, z, p.mat);
            rect[1] = Conv(w, z, p.mat);
            rect[2] = Conv(w, h, p.mat);
            rect[3] = Conv(z, h, p.mat);
            SetRange(rect, _ib, _ie, _ob, _oe);
        }

        void WarpAffineByteBilinear::SetRange(const Base::Point* rect, int* beg, int* end, const int* lo, const int* hi)
        {
            const WarpAffParam& p = _param;
            float* min = (float*)_buf.data;
            float* max = min + p.dstH;
            float w = (float)p.dstW, h = (float)p.dstH, z = 0.0f;
            for (size_t y = 0; y < p.dstH; ++y)
            {
                min[y] = w;
                max[y] = z;
            }
            for (int v = 0; v < 4; ++v)
            {
                const Point& curr = rect[v];
                const Point& next = rect[(v + 1) & 3];
                float yMin = Simd::Max(Simd::Min(curr.y, next.y), z);
                float yMax = Simd::Min(Simd::Max(curr.y, next.y), h);
                int yBeg = (int)ceil(yMin);
                int yEnd = (int)ceil(yMax);
                if (next.y == curr.y)
                    continue;
                float a = (next.x - curr.x) / (next.y - curr.y);
                float b = curr.x - curr.y * a;
                for (int y = yBeg; y < yEnd; ++y)
                {
                    float x = Simd::RestrictRange(float(y), yMin, yMax) * a + b;
                    min[y] = Simd::Min(min[y], Simd::Max(x, z));
                    max[y] = Simd::Max(max[y], Simd::Min(x, w));
                }
            }
            for (size_t y = 0; y < p.dstH; ++y)
            {
                beg[y] = (int)ceil(min[y]);
                end[y] = (int)ceil(max[y]);
                end[y] = Simd::Max(beg[y], end[y]);
            }
            if (hi)
            {
                for (size_t y = 0; y < p.dstH; ++y)
                {
                    beg[y] = Simd::Min(beg[y], hi[y]);
                    end[y] = Simd::Min(end[y], hi[y]);
                }
            }
        }

        //-----------------------------------------------------------------------------------------

        void* WarpAffineInit(size_t srcW, size_t srcH, size_t srcS, size_t dstW, size_t dstH, size_t dstS, size_t channels, const float* mat, SimdWarpAffineFlags flags, const uint8_t* border)
        {
            WarpAffParam param(srcW, srcH, srcS, dstW, dstH, dstS, channels, mat, flags, border, 1);
            if (!param.Valid())
                return NULL;
            if (param.IsNearest())
                return new WarpAffineNearest(param);
            else if (param.IsByteBilinear())
                return new WarpAffineByteBilinear(param);
            else
                return NULL;
        }
    }
}
