#include <primitiv/config.h>

#include <primitiv/devices/naive/device.h>
#include <primitiv/devices/naive/ops/common.h>

namespace primitiv {
namespace devices {

void Naive::ltrs_fw_impl(const Tensor &, const Tensor &, Tensor &) {
  PRIMITIV_THROW_NOT_IMPLEMENTED;
}

void Naive::ltrs_bw_impl(
    const Tensor &, const Tensor &, const Tensor &, const Tensor &,
    Tensor &, Tensor &) {
  PRIMITIV_THROW_NOT_IMPLEMENTED;
}

}  // namespace devices
}  // namespace primitiv
