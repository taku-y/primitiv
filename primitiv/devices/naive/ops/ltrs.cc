#include <primitiv/config.h>

#include <primitiv/devices/naive/device.h>
#include <primitiv/devices/naive/ops/common.h>

namespace primitiv {
namespace devices {

void solve_lower(const float* a, const float* b, float* y, std::uint32_t di,
    std::uint32_t dk) {
  for (std::uint32_t k = 0; k < dk; k++) {
    for (std::uint32_t i = 0; i < di; i++) {
      float b_ik = b[i + di * k];
      for (std::uint32_t j = 0; j < i; j++) {
        b_ik -= a[i + j * di] * y[j + k * di];
      }
      y[i + k * di] = b_ik / a[i + i * di];
    }
  }
}

void solve_upper_tr(const float* a, const float* b, float* y,
    std::uint32_t di, std::uint32_t dj, std::uint32_t dk) {
  for (std::uint32_t k = 0; k < dk; k++) {
    for (std::uint32_t i = di - 1; i != UINT32_MAX; i--) {
      float b_ik = b[i + di * k];
      for (std::uint32_t j = i + 1; j < di; j++) {
        // a is transposed
        b_ik -= a[i * dj + j] * y[j + k * di];
      }
      y[i + k * di] = b_ik / a[i * dj + i];
    }
  }
}

void inplace_add_raw(float* a, const float* b, size_t size) {
  for (size_t i=0; i < size; i++) {
    a[i] += b[i];
  }
}

void Naive::ltrs_fw_impl(const Tensor &a, const Tensor &b, Tensor &y) {
  const std::uint32_t di = a.shape()[0];
  const std::uint32_t dj = a.shape()[1]; // dj should be equal to di
  const std::uint32_t dk = b.shape()[1];

  const float *src_a = CDATA(a);
  const float *src_b = CDATA(b);
  float *dest = MDATA(y);

  if (a.shape().has_batch()) {
    // Apply solver multiple times.
    const std::uint32_t a_skip = di * dj;
    const std::uint32_t b_skip = b.shape().has_batch() * dj * dk;
    const std::uint32_t y_skip = di * dk;
    const std::uint32_t bs = a.shape().batch();
    for (std::uint32_t n = 0; n < bs; ++n) {
      solve_lower(src_a + n * a_skip, src_b + n * b_skip, dest + n * y_skip,
                  di, dk);
    }
  } else {
    // Apply solver only once using a combined matrix.
    const std::uint32_t dk_batch = dk * b.shape().batch();
    solve_lower(src_a, src_b, dest, di, dk_batch);
  }
}

void Naive::ltrs_bw_impl(
    const Tensor &a, const Tensor &b, const Tensor &y, const Tensor &gy,
    Tensor &ga, Tensor &gb) {
  // Gradient of upper part of a is forced to be 0, 
  // based on implementation of torch
  const std::uint32_t di = a.shape()[0];
  const std::uint32_t dj = a.shape()[1]; // dj should be equal to di
  const std::uint32_t dk = b.shape()[1];

  const float *src_a = CDATA(a);
  const float *src_y = CDATA(y);
  const float *src_gy = CDATA(gy);
  float *dest_ga = MDATA(ga);
  float *dest_gb = MDATA(gb);

  if (a.shape().has_batch()) {
    // Do multiplication multiple times.
    const std::uint32_t a_skip = di * dj;
    const std::uint32_t b_skip = b.shape().has_batch() * dj * dk;
    const std::uint32_t y_skip = di * dk;
    const std::uint32_t bs = a.shape().batch();
    std::vector<float> gb_(gy.shape().volume());
    for (std::uint32_t n = 0; n < bs; ++n) {
      // gb
      solve_upper_tr(
        src_a + n * a_skip, src_gy + n * y_skip, gb_.data(),
        di, dj, dk
      );
      inplace_add_raw(dest_gb + n * b_skip, gb_.data(), dj * dk);

      // ga
      for (std::uint32_t i=0; i < di; i++) {
        for (std::uint32_t j=0; j < i + 1; j++) {
          for (std::uint32_t k=0; k < dk; k++) {
            dest_ga[i + j * di + n * a_skip] -= \
              gb_.data()[i + k * di] * \
              src_y[j + k * dj + n * y_skip];
          }
        }
      }
    }
  } else {
    std::vector<float> gb_(gy.shape().size());
    // gb
    solve_upper_tr(
      src_a, src_gy, gb_.data(),
      di, dj, dk * gb.shape().batch()
    );

    // ga
    inplace_add_raw(dest_gb, gb_.data(), dj * dk * gb.shape().batch());
    for (std::uint32_t i=0; i < di; i++) {
      for (std::uint32_t j=0; j < i + 1; j++) {
        for (std::uint32_t k=0; k < dk * gb.shape().batch(); k++) {
          dest_ga[i + j * di] -= \
            gb_.data()[i + k * di] * \
            src_y[j + k * dj];
        }
      }
    }
  }
}

}  // namespace devices
}  // namespace primitiv
