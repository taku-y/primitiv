#include <primitiv/config.h>

#include <primitiv/devices/eigen/device.h>
#include <primitiv/devices/eigen/ops/common.h>

namespace primitiv {
namespace devices {

void Eigen::ltrs_fw_impl(const Tensor &a, const Tensor &b, Tensor &y) {
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
      EMap<const EMatrixXf> aa(src_a + n * a_skip, di, dj);
      EMap<const EMatrixXf> bb(src_b + n * b_skip, dj, dk);
      EMap<EMatrixXf> yy(dest + n * y_skip, di, dk);
      yy.noalias() = aa.triangularView<::Eigen::Lower>().solve(bb);
    }
  } else {
    // Apply solver only once using a combined matrix.
    const std::uint32_t dk_batch = dk * b.shape().batch();
    EMap<const EMatrixXf> aa(src_a, di, dj);
    EMap<const EMatrixXf> bb(src_b, dj, dk_batch);
    EMap<EMatrixXf> yy(dest, di, dk_batch);
    yy.noalias() = aa.triangularView<::Eigen::Lower>().solve(bb);
  }
}

void Eigen::ltrs_bw_impl(
const Tensor &a, const Tensor &b, const Tensor &y, const Tensor &gy,
    Tensor &ga, Tensor &gb) {
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
    for (std::uint32_t n = 0; n < bs; ++n) {
      EMap<const EMatrixXf> aa(src_a + n * a_skip, di, dj);
      EMap<const EMatrixXf> yy(src_y + n * b_skip, dj, dk);
      EMap<const EMatrixXf> gyy(src_gy + n * y_skip, di, dk);
      EMap<EMatrixXf> gaa(dest_ga + n * a_skip, di, dj);
      EMap<EMatrixXf> gbb(dest_gb + n * b_skip, dj, dk);
      gbb.noalias() += aa.transpose().triangularView<::Eigen::Upper>().solve(gyy);
      gaa.noalias() -= aa.transpose().triangularView<::Eigen::Upper>().solve(gyy) * yy.transpose();
    }
  }
  else {
    // Do multiplication only once using a combined matrix.
    const std::uint32_t dk_batch = dk * b.shape().batch();
    EMap<const EMatrixXf> aa(src_a, di, dj);
    EMap<const EMatrixXf> yy(src_y, dj, dk_batch);
    EMap<const EMatrixXf> gyy(src_gy, di, dk_batch);
    EMap<EMatrixXf> gaa(dest_ga, di, dj);
    EMap<EMatrixXf> gbb(dest_gb, dj, dk_batch);
    gbb.noalias() += aa.transpose().triangularView<::Eigen::Upper>().solve(gyy);
    gaa.noalias() -= aa.transpose().triangularView<::Eigen::Upper>().solve(gyy) * yy.transpose();
  }
}

}  // namespace devices
}  // namespace primitiv
