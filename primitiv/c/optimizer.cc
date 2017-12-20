/* Copyright 2017 The primitiv Authors. All Rights Reserved. */
#include <primitiv/config.h>

#include <primitiv/optimizer.h>
#include <primitiv/c/internal.h>
#include <primitiv/c/optimizer.h>

using primitiv::Optimizer;
using primitiv::c::internal::to_c_ptr;
using primitiv::c::internal::to_cpp_ptr;

extern "C" {

void primitiv_Optimizer_delete(primitiv_Optimizer *optimizer) {
  delete to_cpp_ptr(optimizer);
}

primitiv_Status primitiv_Optimizer_load(primitiv_Optimizer *optimizer,
                                        const char *path) {
  try {
    to_cpp_ptr(optimizer)->load(path);
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Optimizer_save(const primitiv_Optimizer *optimizer,
                                        const char *path) {
  try {
    to_cpp_ptr(optimizer)->save(path);
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

uint32_t primitiv_Optimizer_get_epoch(const primitiv_Optimizer *optimizer) {
  return to_cpp_ptr(optimizer)->get_epoch();
}

void primitiv_Optimizer_set_epoch(primitiv_Optimizer *optimizer,
                                  uint32_t epoch) {
  to_cpp_ptr(optimizer)->set_epoch(epoch);
}

float primitiv_Optimizer_get_learning_rate_scaling(
    const primitiv_Optimizer *optimizer) {
  return to_cpp_ptr(optimizer)->get_learning_rate_scaling();
}

primitiv_Status primitiv_Optimizer_set_learning_rate_scaling(
    primitiv_Optimizer *optimizer, float scale) {
  try {
    to_cpp_ptr(optimizer)->set_epoch(scale);
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

float primitiv_Optimizer_get_weight_decay(const primitiv_Optimizer *optimizer) {
  return to_cpp_ptr(optimizer)->get_weight_decay();
}

primitiv_Status primitiv_Optimizer_set_weight_decay(
    primitiv_Optimizer *optimizer, float strength) {
  try {
    to_cpp_ptr(optimizer)->set_weight_decay(strength);
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

float primitiv_Optimizer_get_gradient_clipping(
    const primitiv_Optimizer *optimizer) {
  return to_cpp_ptr(optimizer)->get_gradient_clipping();
}

primitiv_Status primitiv_Optimizer_set_gradient_clipping(
    primitiv_Optimizer *optimizer, float threshold) {
  try {
    to_cpp_ptr(optimizer)->set_gradient_clipping(threshold);
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Optimizer_add_parameter(primitiv_Optimizer *optimizer,
                                                 primitiv_Parameter *param) {
  try {
    to_cpp_ptr(optimizer)->add(*to_cpp_ptr(param));
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Optimizer_add_parameters(
    primitiv_Optimizer *optimizer, primitiv_Parameter **params, size_t n) {
  try {
    Optimizer *cc_optimizer = to_cpp_ptr(optimizer);
    for (size_t i = 0; i < n; ++i) {
      cc_optimizer->add(*to_cpp_ptr(params[i]));
    }
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Optimizer_add_model(primitiv_Optimizer *optimizer,
                                             primitiv_Model *model) {
  try {
    to_cpp_ptr(optimizer)->add(*to_cpp_ptr(model));
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Optimizer_add_models(
    primitiv_Optimizer *optimizer, primitiv_Model **models, size_t n) {
  try {
    Optimizer *cc_optimizer = to_cpp_ptr(optimizer);
    for (size_t i = 0; i < n; ++i) {
      cc_optimizer->add(*to_cpp_ptr(models[i]));
    }
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Optimizer_reset_gradients(
    primitiv_Optimizer *optimizer) {
  try {
    to_cpp_ptr(optimizer)->reset_gradients();
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Optimizer_update(primitiv_Optimizer *optimizer) {
  try {
    to_cpp_ptr(optimizer)->update();
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

}  // end extern "C"
