#include <config.h>

#include <primitiv/device.h>
#include <primitiv/shape_ops.h>
#include <primitiv/tensor.h>

namespace primitiv {

float Tensor::to_float() const {
  std::lock_guard<RecursiveSpinlock> lock(spinlock_);

  if (!valid()) THROW_ERROR("Invalid tensor.");
  if (shape_.size() != 1) {
    THROW_ERROR(
        "Tensor has more than 1 values. shape = " << shape_.to_string());
  }
  return device_->tensor_to_vector(*this)[0];
}

std::vector<float> Tensor::to_vector() const {
  std::lock_guard<RecursiveSpinlock> lock(spinlock_);

  if (!valid()) THROW_ERROR("Invalid tensor.");
  return device_->tensor_to_vector(*this);
}

std::vector<std::uint32_t> Tensor::argmax(std::uint32_t dim) const {
  std::lock_guard<RecursiveSpinlock> lock(spinlock_);

  if (!valid()) THROW_ERROR("Invalid tensor.");
  return device_->argmax(*this, dim);
}

std::vector<std::uint32_t> Tensor::argmin(std::uint32_t dim) const {
  std::lock_guard<RecursiveSpinlock> lock(spinlock_);

  if (!valid()) THROW_ERROR("Invalid tensor.");
  return device_->argmin(*this, dim);
}

void *Tensor::data() {
  std::lock_guard<RecursiveSpinlock> lock(spinlock_);

  if (!valid()) THROW_ERROR("Invalid tensor.");
  // If the internal memory is shared with other objects, the memory will be
  // duplicated to maintain the safety of other objects.
  if (data_.use_count() > 1) {
    *this = device_->copy_tensor(*this);
  }
  return data_.get();
}

void Tensor::reset(float k) {
  std::lock_guard<RecursiveSpinlock> lock(spinlock_);

  if (!valid()) THROW_ERROR("Invalid tensor.");
  device_->reset_tensor(k, *this);
}

void Tensor::reset_by_array(const float *values) {
  std::lock_guard<RecursiveSpinlock> lock(spinlock_);

  if (!valid()) THROW_ERROR("Invalid tensor.");
  device_->reset_tensor_by_array(values, *this);
}

void Tensor::reset_by_vector(const std::vector<float> &values) {
  std::lock_guard<RecursiveSpinlock> lock(spinlock_);

  if (!valid()) THROW_ERROR("Invalid tensor.");
  device_->reset_tensor_by_vector(values, *this);
}

Tensor Tensor::reshape(const Shape &new_shape) const {
  std::lock_guard<RecursiveSpinlock> lock(spinlock_);

  if (!valid()) THROW_ERROR("Invalid tensor.");
  return Tensor(shape_ops::reshape(shape_, new_shape), *device_, data_);
}

Tensor Tensor::flatten() const {
  std::lock_guard<RecursiveSpinlock> lock(spinlock_);

  if (!valid()) THROW_ERROR("Invalid tensor.");
  return Tensor(shape_ops::flatten(shape_), *device_, data_);
}

Tensor &Tensor::inplace_multiply_const(float k) {
  std::lock_guard<RecursiveSpinlock> lock(spinlock_);

  if (!valid()) THROW_ERROR("Invalid tensor.");
  device_->inplace_multiply_const(k, *this);
  return *this;
}

Tensor &Tensor::inplace_add(const Tensor &x) {
  std::lock_guard<RecursiveSpinlock> lock(spinlock_);

  if (!valid()) THROW_ERROR("Invalid tensor.");
  device_->inplace_add(x, *this);
  return *this;
}

Tensor &Tensor::inplace_subtract(const Tensor &x) {
  std::lock_guard<RecursiveSpinlock> lock(spinlock_);

  if (!valid()) THROW_ERROR("Invalid tensor.");
  device_->inplace_subtract(x, *this);
  return *this;
}

}  // namepsace primitiv
