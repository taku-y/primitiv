#include <primitiv/config.h>

#include <primitiv/eigen_device.h>
#include <primitiv/device_ops/eigen/common.h>

namespace primitiv {
namespace devices {

EIGEN_DEV_FW_X(abs, x.abs());
EIGEN_DEV_BW_X(abs, x.sign() * gy);

}  // namespace devices
}  // namespace primitiv
