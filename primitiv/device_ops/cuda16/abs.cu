#include <primitiv/config.h>

#include <primitiv/cuda16_device.h>
#include <primitiv/internal/cuda_utils.h>
#include <primitiv/device_ops/cuda16/common.h>

namespace {

CUDA16_KERNEL_FW_X(abs, ::fabsf(X_VAL));
CUDA16_KERNEL_BW_X(abs, ((X_VAL > .0f) - (X_VAL < .0f)) * GY_VAL);

}  // namespace

namespace primitiv {
namespace devices {

CUDA16_DEV_FW_X(abs);
CUDA16_DEV_BW_X(abs);

}  // namespace devices
}  // namespace primitiv
