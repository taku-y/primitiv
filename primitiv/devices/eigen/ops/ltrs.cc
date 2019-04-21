#include <primitiv/config.h>

#include <primitiv/devices/eigen/device.h>
#include <primitiv/devices/eigen/ops/common.h>

namespace primitiv {
namespace devices {

void Eigen::ltrs_fw_impl(const Tensor &, const Tensor &, Tensor &) {
  PRIMITIV_THROW_NOT_IMPLEMENTED;
}

void Eigen::ltrs_bw_impl(
    const Tensor &, const Tensor &, const Tensor &, const Tensor &,
    Tensor &, Tensor &) {
  PRIMITIV_THROW_NOT_IMPLEMENTED;
}

}  // namespace devices
}  // namespace primitiv
