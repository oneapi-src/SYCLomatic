//==---- dnnl_utils.hpp ---------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_DNNL_UTILS_HPP__
#define __DPCT_DNNL_UTILS_HPP__

#include <CL/sycl.hpp>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include <unordered_map>

#include "device.hpp"
#include "lib_common_utils.hpp"

namespace dpct {
namespace dnnl {
/// An enum class representing memory layout. Used by
/// memory_desc_ext to create a memory with pre-defined layout.
enum class memory_format_tag { nchw, nhwc, nchw_blocked };

/// A class holding the description of an N-dimensions memory.
class memory_desc_ext {
  ::dnnl::memory::desc _desc;
  /// Convert dpct::library_data_t to dnnl::memory::data_type.
  static ::dnnl::memory::data_type to_dnnl_data_type(dpct::library_data_t dt);
  /// Convert dnnl::memory::data_type to dpct::library_data_t.
  static dpct::library_data_t to_dpct_library_data_t(::dnnl::memory::data_type dt,
                                   unsigned block_size);
  /// Convert dpct::dnnl::memory_format_tag to dnnl::memory::format_tag.
  static ::dnnl::memory::format_tag to_dnnl_format_tag(dpct::library_data_t dt,
                                         memory_format_tag tag);

public:
  memory_desc_ext() = default;
  memory_desc_ext(::dnnl::memory::desc &desc) : _desc(desc) {}
  memory_desc_ext(::dnnl::memory::desc &&desc) : _desc(std::move(desc)) {}
/// Setting a 4D memory with given parameters.
/// \param [in] tag Format tag.
/// \param [in] dt Data type.
/// \param [in] n Number of images.
/// \param [in] c Number of channels.
/// \param [in] h Height of images.
/// \param [in] w Width of images.
  void set(memory_format_tag tag, dpct::library_data_t dt, int n, int c, int h,
           int w);
/// Setting a 4D memory with given parameters.
/// \param [in] dt Data type.
/// \param [in] n Number of images.
/// \param [in] c Number of channels.
/// \param [in] h Height of images.
/// \param [in] w Width of images.
/// \param [in] n_stride Stride between two continuous images.
/// \param [in] c_stride Stride between two continuous channels.
/// \param [in] h_stride Stride between two continuous rows.
/// \param [in] w_stride Stride between two continuous columns.
  void set(dpct::library_data_t dt, int n, int c, int h, int w, int n_stride,
           int c_stride, int h_stride, int w_stride);
/// Setting a ND memory with given parameters.
/// \param [in] dt Data type.
/// \param [in] ndims Dimension of the memory.
/// \param [in] dims  Array of dimension ndims that contain the size of each memory dimension.
/// \param [in] strides Array of dimension ndims that contain the stride of each memory dimension.
  void set(dpct::library_data_t dt, int ndims, const int dims[],
           const int strides[]);
/// Setting a ND memory with given parameters.
/// \param [in] tag Format tag.
/// \param [in] dt Data type.
/// \param [in] ndims Dimension of the memory.
/// \param [in] dims  Array of dimension ndims that contain the size of each memory dimension.
  void set(memory_format_tag tag, dpct::library_data_t dt, int ndims,
           const int dims[]);
/// Getting a ::dnnl::memory::desc from a memory_desc_ext.
/// \returns The ::dnnl::memory::desc.
  const ::dnnl::memory::desc &get_desc() const { return _desc; }
/// Getting a size of a memory_desc_ext in bytes.
/// \returns The size.
  size_t get_size() const { return _desc.get_size(); }
/// Getting parameters from a 4D memory.
/// \param [out] dt Data type.
/// \param [out] n Number of images.
/// \param [out] c Number of channels.
/// \param [out] h Height of images.
/// \param [out] w Width of images.
/// \param [out] n_stride Stride between two continuous images.
/// \param [out] c_stride Stride between two continuous channels.
/// \param [out] h_stride Stride between two continuous rows.
/// \param [out] w_stride Stride between two continuous columns.
  void get(dpct::library_data_t *dt, int *n, int *c, int *h, int *w, int *n_stride,
           int *c_stride, int *h_stride, int *w_stride) const;
/// Getting parameters from a ND memory.
/// \param [in] requested_ndims Requested number of dimensions to get from a given memory descriptor.
/// \param [out] dt Data type.
/// \param [out] ndims Dimension of the memory.
/// \param [out] dims  Array of dimension requested_ndims that contain the size of each memory dimension.
/// \param [out] strides Array of dimension requested_ndims that contain the stride of each memory dimension.
  void get(int requested_ndims, dpct::library_data_t *dt, int *ndims, int dims[],
           int strides[]) const;
};

/// A class holding description for an activation operation.
class activation_desc {
  ::dnnl::algorithm _alg;
  float _alpha;
  float _beta;

public:
/// Setting an activation descriptor with given parameters.
/// \param [in] alg Activation algorithm.
/// \param [in] alpha Value of alpha parameter.
  void set(::dnnl::algorithm alg, float alpha) {
    _alg = alg;
    _alpha = alpha;
  }
/// Getting parameters form an activation descriptor.
/// \param [out] alg Activation algorithm.
/// \param [out] alpha Value of alpha parameter.
  void get(::dnnl::algorithm *alg, float *alpha) const {
    *alg = _alg;
    *alpha = _alpha;
  }
/// Setting the alpha parameter of an activation descriptor.
/// \param [in] alpha Value of alpha parameter.
  void set_alpha(float alpha) { _alpha = alpha; }
/// Setting the beta parameter of an activation descriptor.
/// \param [in] beta Value of beta parameter.
  void set_beta(float beta) { _beta = beta; }
/// Setting the algorithm parameter of an activation descriptor.
/// \param [in] alg Activation algorithm.
  void set_algorithm(::dnnl::algorithm alg) { _alg = alg; }
/// Getting the alpha parameter from an activation descriptor.
/// \param [out] alpha Value of alpha parameter.
  float get_alpha() const { return _alpha; }
/// Getting the beta parameter from an activation descriptor.
/// \param [out] beta Value of beta parameter.
  float get_beta() const { return _beta; }
/// Getting the algorithm parameter from an activation descriptor.
/// \param [out] alg Activation algorithm.
  ::dnnl::algorithm get_algorithm() const { return _alg; }
};

/// A class holding description for a local response normalization operation.
class lrn_desc {
  unsigned int _local_size;
  float _alpha;
  float _beta;
  float _k;

public:
/// Setting a local response normalization descriptor with given parameters.
/// \param [in] local_size Value of local_size parameter.
/// \param [in] alpha Value of alpha parameter.
/// \param [in] beta Value of beta parameter.
/// \param [in] k Value of k parameter.
  void set(unsigned int local_size, float alpha, float beta, float k) {
    _local_size = local_size;
    _alpha = alpha;
    _beta = beta;
    _k = k;
  }
/// Getting parameters form a local response normalization descriptor.
/// \param [out] local_size Value of local_size parameter.
/// \param [out] alpha Value of alpha parameter.
/// \param [out] beta Value of beta parameter.
/// \param [out] k Value of k parameter.
  void get(unsigned int *local_size, float *alpha, float *beta,
           float *k) const {
    *local_size = _local_size;
    *alpha = _alpha;
    *beta = _beta;
    *k = _k;
  }
/// Setting the local size parameter of a local response normalization descriptor.
/// \param [in] local_size Value of local_size parameter.
  void set_local_size(unsigned int local_size) { _local_size = local_size; }
/// Setting the alpha parameter of a local response normalization descriptor.
/// \param [in] alpha Value of alpha parameter.
  void set_alpha(float alpha) { _alpha = alpha; }
/// Setting the beta parameter of a local response normalization descriptor.
/// \param [in] beta Value of beta parameter.
  void set_beta(float beta) { _beta = beta; }
/// Setting the k parameter of a local response normalization descriptor.
/// \param [in] k Value of k parameter.
  void set_k(float k) { _k = k; }
/// Getting the local size parameter from a local response normalization descriptor.
/// \param [out] local_size Value of local_size parameter.
  unsigned int get_local_size() const { return _local_size; }
/// Getting the alpha parameter from a local response normalization descriptor.
/// \param [out] alpha Value of alpha parameter.
  float get_alpha() const { return _alpha; }
/// Getting the beta parameter from a local response normalization descriptor.
/// \param [out] beta Value of beta parameter.
  float get_beta() const { return _beta; }
/// Getting the k parameter from a local response normalization descriptor.
/// \param [out] k Value of k parameter.
  float get_k() const { return _k; }
};

/// An enum class representing softmax algorithm.
enum class softmax_algorithm { normal, log };
/// An enum class representing softmax mode.
enum class softmax_mode { instance, channel };

/// A class holding description for a pooling operation.
class pooling_desc {
  ::dnnl::algorithm _alg;
  std::vector<int64_t> _stride;
  std::vector<int64_t> _kernel;
  std::vector<int64_t> _padding;

public:
/// Setting a 2D pooling descriptor with given parameters.
/// \param [in] alg Pooling algorithm.
/// \param [in] kernel_h Value of height of kernel.
/// \param [in] kernel_w Value of width of kernel.
/// \param [in] padding_h Value of height of padding.
/// \param [in] padding_w Value of width of padding.
/// \param [in] stride_h Value of height of stride.
/// \param [in] stride_w Value of width of stride.
  void set(::dnnl::algorithm alg, int kernel_h, int kernel_w, int padding_h,
           int padding_w, int stride_h, int stride_w) {
    _alg = alg;
    _stride = {stride_h, stride_w};
    _kernel = {kernel_h, kernel_w};
    _padding = {padding_h, padding_w};
  }
/// Setting a ND pooling descriptor with given parameters.
/// \param [in] alg Pooling algorithm.
/// \param [in] ndims Dimension of the pooling operation.
/// \param [in] kernel Array of dimension ndims containing the kernel size of each dimension.
/// \param [in] padding Array of dimension ndims containing the padding size of each dimension.
/// \param [in] stride Array of dimension ndims containing the stride size of each dimension.
  void set(::dnnl::algorithm alg, int ndims, int kernel[], int padding[],
           int stride[]) {
    _alg = alg;
    _stride = std::vector<int64_t>(stride, stride + ndims);
    _kernel = std::vector<int64_t>(kernel, kernel + ndims);
    _padding = std::vector<int64_t>(padding, padding + ndims);
  }
/// Getting parameters from a 2D pooling descriptor.
/// \param [out] alg Pooling algorithm.
/// \param [out] kernel_h Value of height of kernel.
/// \param [out] kernel_w Value of width of kernel.
/// \param [out] padding_h Value of height of padding.
/// \param [out] padding_w Value of width of padding.
/// \param [out] stride_h Value of height of stride.
/// \param [out] stride_w Value of width of stride.
  void get(::dnnl::algorithm *alg, int *kernel_h, int *kernel_w, int *padding_h,
           int *padding_w, int *stride_h, int *stride_w) const {
    *alg = _alg;
    *kernel_h = _kernel[0];
    *kernel_w = _kernel[1];
    *padding_h = _padding[0];
    *padding_w = _padding[1];
    *stride_h = _stride[0];
    *stride_w = _stride[1];
  }
/// Getting parameters from a ND pooling descriptor.
/// \param [in] requested_ndims Requested number of dimensions to get from a given pooling descriptor.
/// \param [out] alg Pooling algorithm.
/// \param [out] ndims Dimension of the pooling operation.
/// \param [out] kernel Array of dimension ndims containing the kernel size of each dimension.
/// \param [out] padding Array of dimension ndims containing the padding size of each dimension.
/// \param [out] stride Array of dimension ndims containing the stride size of each dimension.
  void get(int requested_ndims, ::dnnl::algorithm *alg, int *ndims, int kernel[],
           int padding[], int stride[]) const {
    *alg = _alg;
    for (int i = 0; i < requested_ndims; i++) {
      kernel[i] = _kernel[i];
      padding[i] = _padding[i];
      stride[i] = _stride[i];
    }
  }
/// Setting the algorithm parameter of a pooling descriptor.
/// \param [in] alg Pooling algorithm.
  void set_algorithm(::dnnl::algorithm alg) { _alg = alg; }
/// Setting the stride parameter of a pooling descriptor.
/// \param [in] stride Array of dimension ndims containing the stride size of each dimension.
  void set_stride(std::vector<int64_t> stride) { _stride = stride; }
/// Setting the kernel parameter of a pooling descriptor.
/// \param [in] kernel Array of dimension ndims containing the kernel size of each dimension.
  void set_kernel(std::vector<int64_t> kernel) { _kernel = kernel; }
/// Setting the padding parameter of a pooling descriptor.
/// \param [in] padding Array of dimension ndims containing the padding size of each dimension.
  void set_padding(std::vector<int64_t> padding) { _padding = padding; }

/// Getting the algorithm parameter from a pooling descriptor.
/// \param [out] alg Pooling algorithm.
  ::dnnl::algorithm get_algorithm() const { return _alg; }
/// Getting the stride parameter from a pooling descriptor.
/// \returns Array of dimension ndims containing the stride size of each dimension.
  const std::vector<int64_t> &get_stride() const { return _stride; }
/// Getting the stride parameter from a pooling descriptor.
/// \returns Array of dimension ndims containing the stride size of each dimension.
  std::vector<int64_t> &get_stride() { return _stride; }
/// Getting the kernel parameter from a pooling descriptor.
/// \returns Array of dimension ndims containing the kernel size of each dimension.
  const std::vector<int64_t> &get_kernel() const { return _kernel; }
/// Getting the kernel parameter from a pooling descriptor.
/// \returns Array of dimension ndims containing the kernel size of each dimension.
  std::vector<int64_t> &get_kernel() { return _kernel; }
/// Getting the padding parameter from a pooling descriptor.
/// \returns Array of dimension ndims containing the padding size of each dimension.
  const std::vector<int64_t> &get_padding() const { return _padding; }
/// Getting the padding parameter from a pooling descriptor.
/// \returns Array of dimension ndims containing the padding size of each dimension.
  std::vector<int64_t> &get_padding() { return _padding; }
/// Getting the output dimensions of a memory after 2D pooling has been applied.
/// \param [in] desc Input memory descriptor.
/// \param [out] out_n Number of images.
/// \param [out] out_c Number of channels.
/// \param [out] out_h Height of images.
/// \param [out] out_w Width of images.
  void get_forward_output_dim(const memory_desc_ext &desc, int *out_n, int *out_c,
                              int *out_h, int *out_w) const {
    auto dims = desc.get_desc().data.dims;
    *out_n = dims[0];
    *out_c = dims[1];
    *out_h = 1 + (dims[2] + 2 * _padding[0] - _kernel[0]) / _stride[0];
    *out_w = 1 + (dims[3] + 2 * _padding[1] - _kernel[1]) / _stride[1];
  }
/// Getting the output dimensions of a memory after ND pooling has been applied.
/// \param [in] desc Input memory descriptor.
/// \param [out] ndims Dimension of the memory.
/// \param [out] out_dims  Array of dimension requested_ndims that contain the size of each memory dimension.
  void get_forward_output_dim(const memory_desc_ext &desc, int ndims,
                              int out_dims[]) const {
    assert(ndims >= 4 && "ndims is at least 4.");
    auto dims = desc.get_desc().data.dims;
    out_dims[0] = dims[0];
    out_dims[1] = dims[1];
    for (int i = 2; i < ndims; i++) {
      out_dims[i] =
          1 + (dims[i] + 2 * _padding[i - 2] - _kernel[i - 2]) / _stride[i - 2];
    }
  }
};

/// A class holding the oneDNN engine.
class engine_ext {
  ::dnnl::engine _eng;
  ::dnnl::stream _s;
  sycl::queue *_q;
  std::map<void *, ::dnnl::memory> workspace_map;
  struct device_pointer_deleter {
    sycl::queue _que;
    explicit device_pointer_deleter(sycl::queue q) : _que(q) {}
    void operator()(void *ptr) const noexcept {
      if (ptr)
        sycl::free(ptr, _que);
    }
  };
  ::dnnl::memory &get_workspace(void *key) { return workspace_map[key]; }
  void insert_workspace(void *key, ::dnnl::memory workspace) {
    workspace_map[key] = workspace;
  }
  const ::dnnl::stream &get_stream() const { return _s; }
  const ::dnnl::engine &get_engine() const { return _eng; }
  std::unique_ptr<void, device_pointer_deleter>
  allocate(const memory_desc_ext &desc) const;
  std::vector<int64_t>
  compress_spatial_dimensions_to_channel(::dnnl::memory::desc desc);
  template <typename primitive_type, typename... args_type>
  primitive_type create_forward_primitive(args_type &&...args);

  template <typename primitive_type, typename... args_type>
  typename primitive_type::primitive_desc
  create_forward_primitive_desc(args_type &&...args);

  template <typename primitive_type, typename forward_primitive_type,
            typename... args_type>
  primitive_type create_backward_primitive(forward_primitive_type &&fp,
                                           args_type &&...args);

  template <bool is_forward, typename primitive_type, typename args_type>
  void execute_primitive(float alpha, float beta, primitive_type &&primitive,
                         args_type &&args, const memory_desc_ext &mem_desc,
                         void *mem);
  static ::dnnl::primitive_attr generate_scaling_attr(float alpha, float beta);

public:
  engine_ext() {}
/// Creating oneDNN engine.
  void create_engine() {
    _eng = ::dnnl::sycl_interop::make_engine(
        dpct::get_current_device(), dpct::get_current_device().get_context());
    _s = ::dnnl::sycl_interop::make_stream(
        _eng, dpct::get_current_device().default_queue());
    _q = &dpct::get_current_device().default_queue();
  }
/// Setting the user's SYCL queue for an oneDNN engine.
/// \param [in] q Pointer to the SYCL queue.
  void set_queue(sycl::queue* q) {
    if(!q) {
      throw std::runtime_error("set_queue: pointer must not be nullptr.");
    }
    if (!_eng) {
      throw std::runtime_error("set_queue: current engine is invalid.");
    }
    if (q->get_context() != ::dnnl::sycl_interop::get_context(_eng)) {
      throw std::runtime_error(
          "set_queue: queue is mismatch with current engine context.");
    }
    _q = q;
    _s = ::dnnl::sycl_interop::make_stream(_eng, *q);
  }
/// Retrieving the user's SYCL queue set in the oneDNN engine.
/// \returns Pointer to the SYCL queue.
  sycl::queue* get_queue() const { return _q; }
/// Setting all elements of a memory to a given value.
/// \param [in] src_desc Source memory descriptor.
/// \param [in] src Pointer to source data.
/// \param [in] valuePtr Pointer to a signle value.
  void fill(const memory_desc_ext &src_desc, void *src, const void *valuePtr);
/// Coping the scaled data from a memory to another memory with a different description.
/// \param [in] alpha Pointer to scaling factors used to scale the computed value.
/// \param [in] src_desc Source memory descriptor.
/// \param [in] src Pointer to source data.
/// \param [in] beta Pointer to scaling factors used to scale the prior value in the destination memory.
/// \param [in] dst_desc Destination memory descriptor.
/// \param [out] dst Pointer to destination data.
  void reorder(float alpha, const memory_desc_ext &src_desc, void *src, float beta,
               const memory_desc_ext &dst_desc, void *dst);
/// Scaling all the elements of a memory by a given factor.
/// \param [in] alpha Pointer to scaling factors.
/// \param [in] src_desc Source memory descriptor.
/// \param [out] src Pointer to source data.
  void scale(float alpha, const memory_desc_ext &src_desc, void *src);
/// Adding the scaled values of a memory to another memory.
/// \param [in] alpha Pointer to scaling factors used to scale the computed value.
/// \param [in] src_desc Source memory descriptor.
/// \param [in] src Pointer to source data.
/// \param [in] beta Pointer to scaling factors used to scale the prior value in the destination memory.
/// \param [in] dst_desc Destination memory descriptor.
/// \param [out] dst Pointer to destination data.
  void sum(float alpha, const memory_desc_ext &src_desc, void *src, float beta,
           const memory_desc_ext &dst_desc, void *dst);
/// Computing a specified activation function value.
/// \param [in] desc Activation descriptor.
/// \param [in] alpha Pointer to scaling factors used to scale the computed value.
/// \param [in] src_desc Source memory descriptor.
/// \param [in] src Pointer to source data.
/// \param [in] beta Pointer to scaling factors used to scale the prior value in the destination memory.
/// \param [in] dst_desc Destination memory descriptor.
/// \param [out] dst Pointer to destination data.
  void activation_forward(activation_desc &desc, float alpha,
                          const memory_desc_ext &src_desc, void *src, float beta,
                          const memory_desc_ext &dst_desc, void *dst);
/// Computing the gradient of a specified activation function.
/// \param [in] desc Activation descriptor.
/// \param [in] alpha Pointer to scaling factors used to scale the computed value.
/// \param [in] dst_desc Destination memory descriptor.
/// \param [in] dst Pointer to destination data.
/// \param [in] diff_dst_desc Differential destination memory descriptor.
/// \param [in] diff_dst Pointer to differential destination data.
/// \param [in] src_desc Source memory descriptor.
/// \param [in] src Pointer to source data.
/// \param [in] beta Pointer to scaling factors used to scale the prior value in the differential destination memory.
/// \param [in] diff_src_desc Differential source memory descriptor.
/// \param [out] diff_src Pointer to differential source data.
  void activation_backward(activation_desc &desc, float alpha,
                           const memory_desc_ext &dst_desc, void *dst,
                           const memory_desc_ext &diff_dst_desc, void *diff_dst,
                           const memory_desc_ext &src_desc, void *src, float beta,
                           const memory_desc_ext &diff_src_desc, void *diff_src);
/// Computing a specified pooling function value.
/// \param [in] desc Pooling descriptor.
/// \param [in] alpha Pointer to scaling factors used to scale the computed value.
/// \param [in] src_desc Source memory descriptor.
/// \param [in] src Pointer to source data.
/// \param [in] beta Pointer to scaling factors used to scale the prior value in the destination memory.
/// \param [in] dst_desc Destination memory descriptor.
/// \param [out] dst Pointer to destination data.
/// \param [out] workspace Pointer to workspace generated from forward propagation.
  void pooling_forward(pooling_desc &desc, float alpha,
                       const memory_desc_ext &src_desc, void *src, float beta,
                       const memory_desc_ext &dst_desc, void *dst,
                       ::dnnl::memory *workspace = nullptr);
/// Computing the gradient of a specified pooling function.
/// \param [in] desc Activation descriptor.
/// \param [in] alpha Pointer to scaling factors used to scale the computed value.
/// \param [in] dst_desc Destination memory descriptor.
/// \param [in] dst Pointer to destination data.
/// \param [in] diff_dst_desc Differential destination memory descriptor.
/// \param [in] diff_dst Pointer to differential destination data.
/// \param [in] src_desc Source memory descriptor.
/// \param [in] src Pointer to source data.
/// \param [in] beta Pointer to scaling factors used to scale the prior value in the differential destination memory.
/// \param [in] diff_src_desc Differential source memory descriptor.
/// \param [out] diff_src Pointer to differential source data.
/// \param [in] workspace Pointer to workspace used for backward propagation.
  void pooling_backward(pooling_desc &desc, float alpha,
                        const memory_desc_ext &dst_desc, void *dst,
                        const memory_desc_ext &diff_dst_desc, void *diff_dst,
                        const memory_desc_ext &src_desc, void *src, float beta,
                        const memory_desc_ext &diff_src_desc, void *diff_src,
                        ::dnnl::memory *workspace = nullptr);
/// Computing a specified softmax function value.
/// \param [in] alg Softmax algorithm.
/// \param [in] mode Softmax mode.
/// \param [in] alpha Pointer to scaling factors used to scale the computed value.
/// \param [in] src_desc Source memory descriptor.
/// \param [in] src Pointer to source data.
/// \param [in] beta Pointer to scaling factors used to scale the prior value in the destination memory.
/// \param [in] dst_desc Destination memory descriptor.
/// \param [out] dst Pointer to destination data.
  void softmax_forward(softmax_algorithm alg, softmax_mode mode, float alpha,
                       const memory_desc_ext &src_desc, void *src, float beta,
                       const memory_desc_ext &dst_desc, void *dst);
/// Computing the gradient of a specified softmax function.
/// \param [in] alg Softmax algorithm.
/// \param [in] mode Softmax mode.
/// \param [in] alpha Pointer to scaling factors used to scale the computed value.
/// \param [in] dst_desc Destination memory descriptor.
/// \param [in] dst Pointer to destination data.
/// \param [in] diff_dst_desc Differential destination memory descriptor.
/// \param [in] diff_dst Pointer to differential destination data.
/// \param [in] beta Pointer to scaling factors used to scale the prior value in the differential destination memory.
/// \param [in] diff_src_desc Differential source memory descriptor.
/// \param [out] diff_src Pointer to differential source data.
  void softmax_backward(softmax_algorithm alg, softmax_mode mode, float alpha,
                        const memory_desc_ext &dst_desc, void *dst,
                        const memory_desc_ext &diff_dst_desc, void *diff_dst,
                        float beta, const memory_desc_ext &diff_src_desc,
                        void *diff_src);
/// Computing a specified local response normalization function value.
/// \param [in] desc Local response normalization descriptor.
/// \param [in] alpha Pointer to scaling factors used to scale the computed value.
/// \param [in] src_desc Source memory descriptor.
/// \param [in] src Pointer to source data.
/// \param [in] beta Pointer to scaling factors used to scale the prior value in the destination memory.
/// \param [in] dst_desc Destination memory descriptor.
/// \param [out] dst Pointer to destination data.
/// \param [out] workspace Pointer to workspace generated from forward propagation.
  void lrn_forward(lrn_desc &desc, float alpha, const memory_desc_ext &src_desc,
                   void *src, float beta, const memory_desc_ext &dst_desc,
                   void *dst, ::dnnl::memory *workspace = nullptr);
/// Computing the gradient of a specified local response normalization function.
/// \param [in] desc Local response normalization descriptor.
/// \param [in] alpha Pointer to scaling factors used to scale the computed value.
/// \param [in] dst_desc Destination memory descriptor.
/// \param [in] dst Pointer to destination data.
/// \param [in] diff_dst_desc Differential destination memory descriptor.
/// \param [in] diff_dst Pointer to differential destination data.
/// \param [in] src_desc Source memory descriptor.
/// \param [in] src Pointer to source data.
/// \param [in] beta Pointer to scaling factors used to scale the prior value in the differential destination memory.
/// \param [in] diff_src_desc Differential source memory descriptor.
/// \param [out] diff_src Pointer to differential source data.
/// \param [in] workspace Pointer to workspace used for backward propagation.
  void lrn_backward(lrn_desc &desc, float alpha, const memory_desc_ext &dst_desc,
                    void *dst, const memory_desc_ext &diff_dst_desc,
                    void *diff_dst, const memory_desc_ext &src_desc, void *src,
                    float beta, const memory_desc_ext &diff_src_desc,
                    void *diff_src, ::dnnl::memory *workspace = nullptr);
};

::dnnl::memory::data_type memory_desc_ext::to_dnnl_data_type(dpct::library_data_t dt) {
  using dnnl_dt = ::dnnl::memory::data_type;
  switch (dt) {
  case dpct::library_data_t::real_half:
    return dnnl_dt::f16;
  case dpct::library_data_t::real_bfloat16:
    return dnnl_dt::bf16;
  case dpct::library_data_t::real_float:
    return dnnl_dt::f32;
  case dpct::library_data_t::real_int32:
    return dnnl_dt::s32;
  case dpct::library_data_t::real_int8:
    return dnnl_dt::s8;
  case dpct::library_data_t::real_uint8:
    return dnnl_dt::u8;
  case dpct::library_data_t::real_int8_4:
    return dnnl_dt::s8;
  case dpct::library_data_t::real_int8_32:
    return dnnl_dt::s8;
  case dpct::library_data_t::real_uint8_4:
    return dnnl_dt::u8;
  default:
    throw std::runtime_error("to_dnnl_data_type: unsupported data type.");
  }
}

dpct::library_data_t memory_desc_ext::to_dpct_library_data_t(::dnnl::memory::data_type dt,
                                        unsigned block_size) {
  using dpct_dt = dpct::library_data_t;
  using dnnl_dt = ::dnnl::memory::data_type;
  switch (dt) {
  case dnnl_dt::f16:
    return dpct_dt::real_half;
  case dnnl_dt::bf16:
    return dpct_dt::real_bfloat16;
  case dnnl_dt::f32:
    return dpct_dt::real_float;
  case dnnl_dt::s32:
    return dpct_dt::real_int32;
  case dnnl_dt::s8:
    if (block_size == 4) {
      return dpct_dt::real_int8_4;
    } else if (block_size == 32) {
      return dpct_dt::real_int8_32;
    } else {
      return dpct_dt::real_int8;
    }
  case dnnl_dt::u8:
    if (block_size == 4) {
      return dpct_dt::real_uint8_4;
    } else {
      return dpct_dt::real_uint8;
    }
  default:
    throw std::runtime_error(
        "to_dpct_library_data_t: unsupported data type dnnl::memory::data_type::undef.");
  }
}

::dnnl::memory::format_tag memory_desc_ext::to_dnnl_format_tag(dpct::library_data_t dt,
                                              memory_format_tag tag) {
  using dpct_dt = dpct::library_data_t;
  using dpct_tag = memory_format_tag;
  using dnnl_tag = ::dnnl::memory::format_tag;
  switch (tag) {
  case dpct_tag::nchw:
    return dnnl_tag::nchw;
  case dpct_tag::nhwc:
    return dnnl_tag::nhwc;
  default:
    if (dt == dpct_dt::real_int8_32) {
      return dnnl_tag::nChw32c;
    } else {
      return dnnl_tag::nChw4c;
    }
  }
}

void memory_desc_ext::set(memory_format_tag tag, dpct::library_data_t dt, int n, int c,
                          int h, int w) {
  _desc = ::dnnl::memory::desc({n, c, h, w}, to_dnnl_data_type(dt),
                            to_dnnl_format_tag(dt, tag));
}

void memory_desc_ext::set(dpct::library_data_t dt, int n, int c, int h, int w,
                       int n_stride, int c_stride, int h_stride, int w_stride) {
  _desc = ::dnnl::memory::desc({n, c, h, w}, to_dnnl_data_type(dt),
                             {n_stride, c_stride, h_stride, w_stride});
}

void memory_desc_ext::set(dpct::library_data_t dt, int ndims, const int dims[],
                       const int strides[]) {
  _desc = ::dnnl::memory::desc({dims, dims + ndims}, to_dnnl_data_type(dt),
                             {strides, strides + ndims});
}

void memory_desc_ext::set(memory_format_tag tag, dpct::library_data_t dt, int ndims,
                       const int dims[]) {
  _desc = ::dnnl::memory::desc({dims, dims + ndims}, to_dnnl_data_type(dt),
                             to_dnnl_format_tag(dt, tag));
}

void memory_desc_ext::get(dpct::library_data_t *dt, int *n, int *c, int *h, int *w,
                       int *n_stride, int *c_stride, int *h_stride,
                       int *w_stride) const {
  unsigned block_size = 1;
  if (_desc.data.format_desc.blocking.inner_blks[0]) {
    block_size = _desc.data.format_desc.blocking.inner_blks[0];
  }

  *dt = to_dpct_library_data_t(_desc.data_type(), block_size);
  *n = _desc.data.dims[0];
  *c = _desc.data.dims[1];
  *h = _desc.data.dims[2];
  *w = _desc.data.dims[3];
  *n_stride = _desc.data.format_desc.blocking.strides[0] / block_size;
  *c_stride = _desc.data.format_desc.blocking.strides[1] / block_size;
  *h_stride = _desc.data.format_desc.blocking.strides[2] / block_size;
  *w_stride = _desc.data.format_desc.blocking.strides[3] / block_size;
}

void memory_desc_ext::get(int requested_ndims, dpct::library_data_t *dt, int *ndims,
                       int dims[], int strides[]) const {
  unsigned block_size = 1;
  if (_desc.data.format_desc.blocking.inner_blks[0]) {
    block_size = _desc.data.format_desc.blocking.inner_blks[0];
  }
  *dt = to_dpct_library_data_t(_desc.data_type(), block_size);
  *ndims = _desc.data.ndims;
  for (int index = 0; index < requested_ndims; index++) {
    dims[index] = _desc.data.dims[index];
    strides[index] =
        _desc.data.format_desc.blocking.strides[index] / block_size;
  }
}

std::unique_ptr<void, engine_ext::device_pointer_deleter>
engine_ext::allocate(const memory_desc_ext &data_desc) const {
  size_t mem_size = data_desc.get_size();
  void *mem = cl::sycl::malloc_device(mem_size, *_q);
  return std::unique_ptr<void, device_pointer_deleter>(
      mem, device_pointer_deleter(*_q));
}

std::vector<int64_t>
engine_ext::compress_spatial_dimensions_to_channel(::dnnl::memory::desc desc) {
  int ndims = desc.data.ndims;
  assert(ndims >= 4 && "ndims is at least 4.");
  std::vector<int64_t> compressed_dims(ndims);
  compressed_dims[0] = desc.data.dims[0];
  compressed_dims[1] = desc.data.dims[1];
  for (int index = 2; index < ndims; index++) {
    compressed_dims[1] = compressed_dims[1] * desc.data.dims[index];
    compressed_dims[index] = 1;
  }
  return compressed_dims;
}

::dnnl::primitive_attr engine_ext::generate_scaling_attr(float alpha,
                                                       float beta) {
  ::dnnl::primitive_attr attr;
  if (alpha != 1.f) {
    attr.set_output_scales(0, {alpha});
  }
  if (beta != 0.f) {
    ::dnnl::post_ops po;
    po.append_sum(beta);
    attr.set_post_ops(po);
  }
  return attr;
}

template <bool is_forward = true, typename primitive_type, typename args_type>
void engine_ext::execute_primitive(float alpha, float beta,
                                   primitive_type &&primitive, args_type &&args,
                                   const memory_desc_ext &mem_desc, void *mem) {
  if (beta != 0.f) {
    auto cache = allocate(mem_desc);
    args.insert({is_forward ? DNNL_ARG_DST : DNNL_ARG_DIFF_SRC,
                 ::dnnl::memory(mem_desc.get_desc(), _eng, cache.get())});
    ::dnnl::sycl_interop::execute(primitive, _s, args);
    sum(alpha, mem_desc, cache.get(), beta, mem_desc, mem);
  } else {
    args.insert({is_forward ? DNNL_ARG_DST : DNNL_ARG_DIFF_SRC,
                 ::dnnl::memory(mem_desc.get_desc(), _eng, mem)});
    ::dnnl::sycl_interop::execute(primitive, _s, args);
    scale(alpha, mem_desc, mem);
  }
}

template <typename primitive_type, typename... args_type>
primitive_type engine_ext::create_forward_primitive(args_type &&...args) {
  return primitive_type(create_forward_primitive_desc<primitive_type>(
      std::forward<args_type>(args)...));
}

template <typename primitive_type, typename... args_type>
typename primitive_type::primitive_desc
engine_ext::create_forward_primitive_desc(args_type &&...args) {
  return typename primitive_type::primitive_desc(
      {std::forward<args_type>(args)...}, _eng);
}

template <typename primitive_type, typename forward_primitive_type,
          typename... args_type>
primitive_type
engine_ext::create_backward_primitive(forward_primitive_type &&fp,
                                      args_type &&...args) {
  return primitive_type(typename primitive_type::primitive_desc(
      {std::forward<args_type>(args)...}, _eng,
      std::forward<forward_primitive_type>(fp)));
}

void engine_ext::fill(const memory_desc_ext &src_desc, void *src,
                      const void *valuePtr) {
  ::dnnl::memory::data_type dt = src_desc.get_desc().data_type();
  unsigned mem_size = src_desc.get_size();
  switch (dt) {
  case ::dnnl::memory::data_type::f32:
    _q->fill<float>(src, *reinterpret_cast<const float *>(valuePtr),
                  mem_size / sizeof(float));
    break;
  case ::dnnl::memory::data_type::f16:
    _q->fill<sycl::half>(src, *reinterpret_cast<const sycl::half *>(valuePtr),
                       mem_size / sizeof(sycl::half));
    break;
  case ::dnnl::memory::data_type::s32:
    _q->fill<int32_t>(src, *reinterpret_cast<const int32_t *>(valuePtr),
                    mem_size / sizeof(int32_t));
    break;
  case ::dnnl::memory::data_type::s8:
    _q->fill<int8_t>(src, *reinterpret_cast<const int8_t *>(valuePtr),
                   mem_size / sizeof(int8_t));
    break;
  case ::dnnl::memory::data_type::u8:
    _q->fill<uint8_t>(src, *reinterpret_cast<const uint8_t *>(valuePtr),
                    mem_size / sizeof(uint8_t));
    break;
  default:
    throw std::runtime_error("fill: unsupported data type sycl::bfloat16.");
  }
  _q->wait();
}

void engine_ext::reorder(float alpha, const memory_desc_ext &src_desc, void *src,
                         float beta, const memory_desc_ext &dst_desc, void *dst) {
  ::dnnl::reorder reorder_primitive({_eng, src_desc.get_desc(), _eng,
                                   dst_desc.get_desc(),
                                   generate_scaling_attr(alpha, beta)});
  std::unordered_map<int, ::dnnl::memory> args = {
      {DNNL_ARG_DST, ::dnnl::memory(dst_desc.get_desc(), _eng, dst)},
      {DNNL_ARG_SRC, ::dnnl::memory(src_desc.get_desc(), _eng, src)}};

  ::dnnl::sycl_interop::execute(reorder_primitive, _s, args);
  _s.wait();
}

void engine_ext::scale(float alpha, const memory_desc_ext &src_desc, void *src) {
  if (alpha == 1.f) {
    return;
  }
  auto primitive = create_forward_primitive<::dnnl::eltwise_forward>(
      ::dnnl::prop_kind::forward_inference, ::dnnl::algorithm::eltwise_linear,
      src_desc.get_desc(), alpha, 0.f);

  std::unordered_map<int, ::dnnl::memory> args = {
      {DNNL_ARG_DST, ::dnnl::memory(src_desc.get_desc(), _eng, src)},
      {DNNL_ARG_SRC, ::dnnl::memory(src_desc.get_desc(), _eng, src)}};

  ::dnnl::sycl_interop::execute(primitive, _s, args);
  _s.wait();
}

void engine_ext::sum(float alpha, const memory_desc_ext &src_desc, void *src,
                     float beta, const memory_desc_ext &dst_desc, void *dst) {
  if (alpha == 0.f && beta == 1.f) {
    return;
  }
  ::dnnl::sum::primitive_desc sum_primitive_desc(
      {beta, alpha}, {src_desc.get_desc(), dst_desc.get_desc()}, _eng);

  std::unordered_map<int, ::dnnl::memory> args = {
      {DNNL_ARG_DST, ::dnnl::memory(dst_desc.get_desc(), _eng, dst)},
      {DNNL_ARG_MULTIPLE_SRC + 1, ::dnnl::memory(src_desc.get_desc(), _eng, src)},
      {DNNL_ARG_MULTIPLE_SRC,
       ::dnnl::memory(dst_desc.get_desc(), _eng, dst)}};

  ::dnnl::sycl_interop::execute(::dnnl::sum(sum_primitive_desc), _s, args);
  _s.wait();
}

#define SCALE_PARAMETER_PREPROCESS(alpha, beta, mem_desc, mem)            \
  if (alpha == 0.f) {                                                          \
    if (beta == 1.f) {                                                         \
      return;                                                                  \
    }                                                                          \
    scale(beta, mem_desc, mem);                                                \
    return;                                                                    \
  }

void engine_ext::activation_forward(activation_desc &desc, float alpha,
                                    const memory_desc_ext &src_desc, void *src,
                                    float beta, const memory_desc_ext &dst_desc,
                                    void *dst) {
  SCALE_PARAMETER_PREPROCESS(alpha, beta, dst_desc, dst)

  auto primitive = create_forward_primitive<::dnnl::eltwise_forward>(
      ::dnnl::prop_kind::forward, desc.get_algorithm(), src_desc.get_desc(),
      desc.get_alpha(), desc.get_beta());

  std::unordered_map<int, ::dnnl::memory> execution_args = {
      {DNNL_ARG_SRC, {::dnnl::memory(src_desc.get_desc(), _eng, src)}}};

  execute_primitive(alpha, beta, primitive, execution_args, dst_desc, dst);
}

void engine_ext::activation_backward(
    activation_desc &desc, float alpha, const memory_desc_ext &dst_desc, void *dst,
    const memory_desc_ext &diff_dst_desc, void *diff_dst,
    const memory_desc_ext &src_desc, void *src, float beta,
    const memory_desc_ext &diff_src_desc, void *diff_src) {

  SCALE_PARAMETER_PREPROCESS(alpha, beta, diff_src_desc, diff_src)

  auto primitive = create_backward_primitive<::dnnl::eltwise_backward>(
      create_forward_primitive_desc<::dnnl::eltwise_forward>(
          ::dnnl::prop_kind::forward, desc.get_algorithm(), src_desc.get_desc(),
          desc.get_alpha(), desc.get_beta()),
      desc.get_algorithm(), diff_src_desc.get_desc(), src_desc.get_desc(),
      desc.get_alpha(), desc.get_beta());

  std::unordered_map<int, ::dnnl::memory> execution_args = {
      {DNNL_ARG_DST, {::dnnl::memory(dst_desc.get_desc(), _eng, dst)}},
      {DNNL_ARG_SRC, {::dnnl::memory(src_desc.get_desc(), _eng, src)}},
      {DNNL_ARG_DIFF_DST,
       {::dnnl::memory(diff_dst_desc.get_desc(), _eng, diff_dst)}}};

  execute_primitive<false>(alpha, beta, primitive, execution_args,
                           diff_src_desc, diff_src);
}

void engine_ext::pooling_forward(pooling_desc &desc, float alpha,
                                 const memory_desc_ext &src_desc, void *src,
                                 float beta, const memory_desc_ext &dst_desc,
                                 void *dst, ::dnnl::memory *workspace) {
  SCALE_PARAMETER_PREPROCESS(alpha, beta, dst_desc, dst)

  auto primitive_desc = create_forward_primitive_desc<::dnnl::pooling_forward>(
      ::dnnl::prop_kind::forward_training, desc.get_algorithm(),
      src_desc.get_desc(), dst_desc.get_desc(), desc.get_stride(),
      desc.get_kernel(), desc.get_padding(), desc.get_padding());

  std::unordered_map<int, ::dnnl::memory> execution_args = {
      {DNNL_ARG_SRC, {::dnnl::memory(src_desc.get_desc(), _eng, src)}}};

  ::dnnl::memory ws_mem(primitive_desc.workspace_desc(), _eng);
  execution_args.insert({DNNL_ARG_WORKSPACE, ws_mem});
  if (workspace) {
    *workspace = ws_mem;
  } else {
    insert_workspace(src, ws_mem);
  }

  execute_primitive(alpha, beta, ::dnnl::pooling_forward(primitive_desc),
                    execution_args, dst_desc, dst);
}

void engine_ext::pooling_backward(pooling_desc &desc, float alpha,
                                  const memory_desc_ext &dst_desc, void *dst,
                                  const memory_desc_ext &diff_dst_desc,
                                  void *diff_dst, const memory_desc_ext &src_desc,
                                  void *src, float beta,
                                  const memory_desc_ext &diff_src_desc,
                                  void *diff_src, ::dnnl::memory *workspace) {
  SCALE_PARAMETER_PREPROCESS(alpha, beta, diff_src_desc, diff_src)

  auto primitive = create_backward_primitive<::dnnl::pooling_backward>(
      create_forward_primitive_desc<::dnnl::pooling_forward>(
          ::dnnl::prop_kind::forward_training, desc.get_algorithm(),
          src_desc.get_desc(), dst_desc.get_desc(), desc.get_stride(),
          desc.get_kernel(), desc.get_padding(), desc.get_padding()),
      desc.get_algorithm(), diff_src_desc.get_desc(), diff_dst_desc.get_desc(),
      desc.get_stride(), desc.get_kernel(), desc.get_padding(),
      desc.get_padding());

  std::unordered_map<int, ::dnnl::memory> execution_args = {
      {DNNL_ARG_DST, {::dnnl::memory(dst_desc.get_desc(), _eng, dst)}},
      {DNNL_ARG_SRC, {::dnnl::memory(src_desc.get_desc(), _eng, src)}},
      {DNNL_ARG_DIFF_DST,
       {::dnnl::memory(diff_dst_desc.get_desc(), _eng, diff_dst)}}};

  if (workspace) {
    execution_args.insert({DNNL_ARG_WORKSPACE, *workspace});
  } else {
    execution_args.insert({DNNL_ARG_WORKSPACE, get_workspace(src)});
  }

  execute_primitive<false>(alpha, beta, primitive, execution_args,
                           diff_src_desc, diff_src);
}

void engine_ext::softmax_forward(softmax_algorithm alg, softmax_mode mode,
                                 float alpha, const memory_desc_ext &src_desc,
                                 void *src, float beta,
                                 const memory_desc_ext &dst_desc, void *dst) {
  SCALE_PARAMETER_PREPROCESS(alpha, beta, dst_desc, dst)

  ::dnnl::memory::desc help_src_desc = src_desc.get_desc();
  ::dnnl::memory::desc help_dst_desc = dst_desc.get_desc();
  if (mode == softmax_mode::instance) {
    std::vector<int64_t> help_dims =
        compress_spatial_dimensions_to_channel(help_src_desc);
    help_src_desc = help_src_desc.reshape(help_dims);
    help_dst_desc = help_dst_desc.reshape(help_dims);
  }

  std::unordered_map<int, ::dnnl::memory> execution_args = {
      {DNNL_ARG_SRC, {::dnnl::memory(help_src_desc, _eng, src)}}};

  if (alg == softmax_algorithm::normal) {
    auto primitive = create_forward_primitive<::dnnl::softmax_forward>(
        ::dnnl::prop_kind::forward, help_src_desc, 1);

    execute_primitive(alpha, beta, primitive, execution_args,
                      memory_desc_ext(help_dst_desc), dst);
  } else {
    auto primitive = create_forward_primitive<::dnnl::logsoftmax_forward>(
        ::dnnl::prop_kind::forward, help_src_desc, 1);

    execute_primitive(alpha, beta, primitive, execution_args,
                      memory_desc_ext(help_dst_desc), dst);
  }
}

void engine_ext::softmax_backward(softmax_algorithm alg, softmax_mode mode,
                                  float alpha, const memory_desc_ext &dst_desc,
                                  void *dst, const memory_desc_ext &diff_dst_desc,
                                  void *diff_dst, float beta,
                                  const memory_desc_ext &diff_src_desc,
                                  void *diff_src) {
  SCALE_PARAMETER_PREPROCESS(alpha, beta, diff_src_desc, diff_src)

  ::dnnl::memory::desc help_diff_src_desc = diff_src_desc.get_desc();
  ::dnnl::memory::desc help_dst_desc = dst_desc.get_desc();
  ::dnnl::memory::desc help_diff_dst_desc = diff_dst_desc.get_desc();
  if (mode == softmax_mode::instance) {
    std::vector<int64_t> help_dims =
        compress_spatial_dimensions_to_channel(help_diff_src_desc);
    help_diff_src_desc = help_diff_src_desc.reshape(help_dims);
    help_dst_desc = help_dst_desc.reshape(help_dims);
    help_diff_dst_desc = help_diff_dst_desc.reshape(help_dims);
  }

  std::unordered_map<int, ::dnnl::memory> execution_args = {
      {DNNL_ARG_DST, {::dnnl::memory(help_dst_desc, _eng, dst)}},
      {DNNL_ARG_DIFF_DST, {::dnnl::memory(help_diff_dst_desc, _eng, diff_dst)}}};

  if (alg == softmax_algorithm::normal) {
    auto primitive = create_backward_primitive<::dnnl::softmax_backward>(
        create_forward_primitive_desc<::dnnl::softmax_forward>(
            ::dnnl::prop_kind::forward, help_diff_src_desc, 1),
        help_diff_src_desc, help_diff_src_desc, 1);
    execute_primitive<false>(alpha, beta, primitive, execution_args,
                             memory_desc_ext(help_diff_src_desc), diff_src);

  } else {
    auto primitive = create_backward_primitive<::dnnl::logsoftmax_backward>(
        create_forward_primitive_desc<::dnnl::logsoftmax_forward>(
            ::dnnl::prop_kind::forward, help_diff_src_desc, 1),
        help_diff_src_desc, help_diff_src_desc, 1);
    execute_primitive<false>(alpha, beta, primitive, execution_args,
                             memory_desc_ext(help_diff_src_desc), diff_src);
  }
}

void engine_ext::lrn_forward(lrn_desc &desc, float alpha,
                             const memory_desc_ext &src_desc, void *src,
                             float beta, const memory_desc_ext &dst_desc,
                             void *dst, ::dnnl::memory *workspace) {

  SCALE_PARAMETER_PREPROCESS(alpha, beta, dst_desc, dst)

  auto primitive_desc = create_forward_primitive_desc<::dnnl::lrn_forward>(
      ::dnnl::prop_kind::forward_training, ::dnnl::algorithm::lrn_across_channels,
      src_desc.get_desc(), desc.get_local_size(), desc.get_alpha(),
      desc.get_beta(), desc.get_k());

  std::unordered_map<int, ::dnnl::memory> execution_args = {
      {DNNL_ARG_SRC, {::dnnl::memory(src_desc.get_desc(), _eng, src)}}};

  ::dnnl::memory ws_mem(primitive_desc.workspace_desc(), _eng);
  execution_args.insert({DNNL_ARG_WORKSPACE, ws_mem});
  if (workspace) {
    *workspace = ws_mem;
  } else {
    insert_workspace(src, ws_mem);
  }

  execute_primitive(alpha, beta, ::dnnl::lrn_forward(primitive_desc),
                    execution_args, dst_desc, dst);
}

void engine_ext::lrn_backward(lrn_desc &desc, float alpha,
                              const memory_desc_ext &dst_desc, void *dst,
                              const memory_desc_ext &diff_dst_desc, void *diff_dst,
                              const memory_desc_ext &src_desc, void *src,
                              float beta, const memory_desc_ext &diff_src_desc,
                              void *diff_src,
                              ::dnnl::memory *workspace) {

  SCALE_PARAMETER_PREPROCESS(alpha, beta, diff_src_desc, diff_src)

  auto primitive = create_backward_primitive<::dnnl::lrn_backward>(
      create_forward_primitive_desc<::dnnl::lrn_forward>(
          ::dnnl::prop_kind::forward_training,
          ::dnnl::algorithm::lrn_across_channels, src_desc.get_desc(),
          desc.get_local_size(), desc.get_alpha(), desc.get_beta(),
          desc.get_k()),
      ::dnnl::algorithm::lrn_across_channels, src_desc.get_desc(),
      diff_src_desc.get_desc(), desc.get_local_size(), desc.get_alpha(),
      desc.get_beta(), desc.get_k());

  std::unordered_map<int, ::dnnl::memory> execution_args = {
      {DNNL_ARG_DST, {::dnnl::memory(dst_desc.get_desc(), _eng, dst)}},
      {DNNL_ARG_SRC, {::dnnl::memory(src_desc.get_desc(), _eng, src)}},
      {DNNL_ARG_DIFF_DST,
       {::dnnl::memory(diff_dst_desc.get_desc(), _eng, diff_dst)}}};

  if (workspace) {
    execution_args.insert({DNNL_ARG_WORKSPACE, *workspace});
  } else {
    execution_args.insert({DNNL_ARG_WORKSPACE, get_workspace(src)});
  }

  execute_primitive<false>(alpha, beta, primitive, execution_args,
                           diff_src_desc, diff_src);
}
} // namespace dnnl
} // namespace dpct

#endif // __DPCT_DNNL_UTILS_HPP__