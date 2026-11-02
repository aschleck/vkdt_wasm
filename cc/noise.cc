// Estimates the noise profile vkdt's denoise needs from the raw itself. vkdt does this on the GPU
// with the rawhist and nprof modules (see bin/vkdt-noise-profile), but rawhist accumulates with
// image atomics, which WebGPU lacks. This is the same computation on the CPU: rawhist's coldumb
// kernel builds the histogram and nprof's write_sink fits the line.

#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

#include "RawSpeed-API.h"

namespace {

// rawhist's output width, see modify_roi_out in src/pipe/modules/rawhist/main.c
constexpr int kBins = 1000;

std::unique_ptr<rawspeed::CameraMetaData> meta;

struct Histogram {
  std::vector<double> count, m1, m2;
  Histogram() : count(kBins), m1(kBins), m2(kBins) {}
  void add(double mean, double mom2) {
    const int x = static_cast<int>(kBins * std::log2(mean + 1.0));
    if(x < 0 || x >= kBins) return;
    count[x] += 1.0;
    m1[x] += mean;
    m2[x] += mom2;
  }
};

// from write_sink in src/pipe/modules/nprof/main.c
void fit(const Histogram &h, float *a_out, float *b_out) {
  std::vector<int> valid;
  float a = 0, b = 0;
  for(int f = 0; f < 7 && valid.empty(); f++) {
    const double fk = 1.0 / std::pow(2, f);
    for(int i = 0; i < kBins; i++) {
      int score = 0;
      for(int j = i + 1; j < kBins * (0.8 - fk); j++) {
        double c = h.count[i];
        if(c < fk * 64) continue;
        const double x1 = h.m1[i] / c;
        const double y1 = h.m2[i] / c - x1 * x1;
        c = h.count[j];
        if(c < fk * 64) continue;
        const double x2 = h.m1[j] / c;
        const double y2 = h.m2[j] / c - x2 * x2;
        if(y1 <= 0 || y2 <= 0) continue;
        const double eb = (y2 - y1) / (x2 - x1);
        const double ea = y1 - x1 * eb;
        if(!(eb > 0.0) || !(ea > 0.0)) continue;
        if(++score > fk * 32) {
          a = ea;
          b = eb;
          valid.push_back(i);
          break;
        }
      }
    }
  }

  double sx = 0, sx2 = 0, sy = 0, sxy = 0, cnt = 0;
  for(const int i : valid) {
    const double c = h.count[i];
    const double x = h.m1[i] / c;
    const double y = std::max(0.0, h.m2[i] / c - x * x);
    cnt += c;
    sx += x * c;
    sx2 += x * x * c;
    sy += y * c;
    sxy += x * y * c;
  }
  const double denom = cnt * sx2 - sx * sx;
  if(std::fabs(denom) > 1e-10) {
    b = std::max(0.0, (cnt * sxy - sx * sy) / denom);
    a = std::max(0.0, (sy * sx2 - sx * sxy) / denom);
  }
  if(valid.empty()) {
    // nprof's fallback
    a = 100.0f;
    b = 1.0f;
  }
  *a_out = a;
  *b_out = b;
}

}  // namespace

extern "C" {

/** Estimates vkdt's noise model parameters for the raw at path. Returns 0 on success. */
int wasm_noise_profile(const char *path, float *a, float *b) {
  try {
    if(!meta) meta = std::make_unique<rawspeed::CameraMetaData>("/data/cameras.xml");
    FILE *f = fopen(path, "rb");
    if(!f) return 1;
    fseek(f, 0, SEEK_END);
    const long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> data(size);
    const size_t read = fread(data.data(), 1, size, f);
    fclose(f);
    if(read != static_cast<size_t>(size)) return 1;

    rawspeed::Buffer buffer(data.data(), static_cast<rawspeed::Buffer::size_type>(size));
    rawspeed::RawParser parser(buffer);
    auto decoder = parser.getDecoder(meta.get());
    if(!decoder) return 1;
    decoder->failOnUnknown = false;
    decoder->checkSupport(meta.get());
    decoder->decodeRaw();
    decoder->decodeMetaData(meta.get());
    rawspeed::RawImage r = decoder->mRaw;
    if(!r->isCFA || r->getDataType() != rawspeed::RawImageType::UINT16) return 1;
    if(!r->blackLevelSeparate) r->calculateBlackAreas();

    // rawhist reads the 16-bit raw through a unorm texture but scales black and white by 2^16.
    const auto bl = *(r->blackLevelSeparate->getAsArray1DRef());
    const double black = bl(1) / 65536.0;
    const double white = r->whitePoint.value_or(65535) / 65536.0;
    const auto img = r->getU16DataAsCroppedArray2DRef();
    auto at = [&](int x, int y) {
      const double v = (img(y, x) / 65535.0 - black) / (white - black);
      return std::min(1.0, std::max(0.0, v));
    };

    Histogram h;
    const int wd = r->dim.x, ht = r->dim.y;
    const bool xtrans = r->cfa.getSize().x == 6;
    if(xtrans) {
      // the cross of green pixels in each 3x3 block
      for(int y = 0; y + 2 < ht; y += 3)
        for(int x = 0; x + 2 < wd; x += 3) {
          const double v[5] = {at(x, y), at(x + 2, y), at(x + 1, y + 1), at(x, y + 2),
                               at(x + 2, y + 2)};
          double mean = 0, mom2 = 0;
          for(const double s : v) {
            mean += s / 5.0;
            mom2 += s * s / 5.0;
          }
          h.add(mean, mom2);
        }
    } else {
      // rawhist assumes the greens sit at (1, 0) and (0, 1), but we look the pattern up.
      int gx[2], gy[2], n = 0;
      for(int y = 0; y < 2; y++)
        for(int x = 0; x < 2; x++)
          if(n < 2 && r->cfa.getColorAt(x, y) == rawspeed::CFAColor::GREEN) {
            gx[n] = x;
            gy[n++] = y;
          }
      if(n < 2) return 1;
      for(int y = 0; y + 1 < ht; y += 2)
        for(int x = 0; x + 1 < wd; x += 2) {
          const double v0 = at(x + gx[0], y + gy[0]), v1 = at(x + gx[1], y + gy[1]);
          h.add((v0 + v1) / 2.0, (v0 * v0 + v1 * v1) / 2.0);
        }
    }
    fit(h, a, b);
    return 0;
  } catch(const std::exception &e) {
    fprintf(stderr, "[vkdt_wasm] noise profile: %s\n", e.what());
    return 1;
  }
}

}  // extern "C"
