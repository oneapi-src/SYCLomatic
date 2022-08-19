// DPCT_LABEL_BEGIN|License|
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
//==---- fft_utils.hpp ----------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
// DPCT_LABEL_END

#ifndef __DPCT_FFT_UTILS_HPP__
#define __DPCT_FFT_UTILS_HPP__
// DPCT_COMMENT
// DPCT_COMMENT Example1:
// DPCT_COMMENT // DPCT_LABEL_BEGIN|FeatureNameDef|[Namespace]
// DPCT_COMMENT // DPCT_DEPENDENCY_EMPTY
// DPCT_COMMENT // DPCT_CODE
// DPCT_COMMENT some code
// DPCT_COMMENT // DPCT_LABEL_END
// DPCT_COMMENT
// DPCT_COMMENT Example2:
// DPCT_COMMENT // DPCT_LABEL_BEGIN|FeatureNameDef|[Namespace]
// DPCT_COMMENT // DPCT_DEPENDENCY_BEGIN
// DPCT_COMMENT // FileID|FeatureNameRef
// DPCT_COMMENT [// FileID|FeatureNameRef]
// DPCT_COMMENT ...
// DPCT_COMMENT // DPCT_DEPENDENCY_END
// DPCT_COMMENT // DPCT_CODE
// DPCT_COMMENT some code
// DPCT_COMMENT // DPCT_LABEL_END
// DPCT_COMMENT
// DPCT_COMMENT For header file including dependency, please use predefined feature name:
// DPCT_COMMENT   local_include_dependency: dpct helper files
// DPCT_COMMENT   non_local_include_dependency: other header files

// DPCT_LABEL_BEGIN|non_local_include_dependency|
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|local_include_dependency|
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
// DPCT_LABEL_END
#include "lib_common_utils.hpp"

namespace dpct {
namespace fft {
// DPCT_LABEL_BEGIN|fft_direction|dpct::fft
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
/// An enumeration type to describe the FFT direction is forward or backward.
enum fft_direction : int {
  forward = 0,
  backward
};
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|fft_type|dpct::fft
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
/// An enumeration type to describe the types of FFT input and output data.
enum fft_type : int {
  real_float_to_complex_float = 0,
  complex_float_to_real_float,
  real_double_to_complex_double,
  complex_double_to_real_double,
  complex_float_to_complex_float,
  complex_double_to_complex_double,
};
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|fft_engine|dpct::fft
// DPCT_DEPENDENCY_BEGIN
// LibCommonUtils|library_data_t
// FftUtils|fft_direction
// FftUtils|fft_type
// LibCommonUtils|get_memory
// Device|get_default_queue
// DPCT_DEPENDENCY_END
// DPCT_CODE
/// A class to perform FFT calculation.
class fft_engine {
public:
  /// Construct the class for calculate n-D FFT.
  /// \param [in] dim Dimension number of the data.
  /// \param [in] n Pointer to an array containing each dimension's size.
  /// \param [in] inembed Pointer to an array containing each dimension's size
  /// of the embedded input data.
  /// \param [in] istride Stride size of the input data.
  /// \param [in] idist Distance between the two batches of the input data.
  /// \param [in] input_type Input data type.
  /// \param [in] onembed Pointer to an array containing each dimension's size
  /// of the embedded output data.
  /// \param [in] ostride Stride size of the output data.
  /// \param [in] odist Distance between the two batches of the output data.
  /// \param [in] output_type Output data type.
  /// \param [in] batch The number of FFT operations to perform.
  fft_engine(int dim, long long *n, long long *inembed, long long istride,
             long long idist, library_data_t input_type, long long *onembed,
             long long ostride, long long odist, library_data_t output_type,
             long long batch) {
    init<long long>(dim, n, inembed, istride, idist, input_type, onembed,
                    ostride, odist, output_type, batch);
  }
  /// Construct the class for calculate n-D FFT.
  /// \param [in] dim Dimension number of the data.
  /// \param [in] n Pointer to an array containing each dimension's size.
  /// \param [in] inembed Pointer to an array containing each dimension's size
  /// of the embedded input data.
  /// \param [in] istride Stride size of the input data.
  /// \param [in] idist Distance between the two batches of the input data.
  /// \param [in] input_type Input data type.
  /// \param [in] onembed Pointer to an array containing each dimension's size
  /// of the embedded output data.
  /// \param [in] ostride Stride size of the output data.
  /// \param [in] odist Distance between the two batches of the output data.
  /// \param [in] output_type Output data type.
  /// \param [in] batch The number of FFT operations to perform.
  fft_engine(int dim, int *n, int *inembed, int istride, int idist,
             library_data_t input_type, int *onembed, int ostride, int odist,
             library_data_t output_type, int batch) {
    init<int>(dim, n, inembed, istride, idist, input_type, onembed, ostride,
              odist, output_type, batch);
  }
  /// Construct the class for calculate n-D FFT.
  /// \param [in] dim Dimension number of the data.
  /// \param [in] n Pointer to an array containing each dimension's size.
  /// \param [in] inembed Pointer to an array containing each dimension's size
  /// of the embedded input data.
  /// \param [in] istride Stride size of the input data.
  /// \param [in] idist Distance between the two batches of the input data.
  /// \param [in] onembed Pointer to an array containing each dimension's size
  /// of the embedded output data.
  /// \param [in] ostride Stride size of the output data.
  /// \param [in] odist Distance between the two batches of the output data.
  /// \param [in] type The FFT type.
  /// \param [in] batch The number of FFT operations to perform.
  fft_engine(int dim, long long *n, long long *inembed, long long istride,
             long long idist, long long *onembed, long long ostride,
             long long odist, fft_type type, long long batch)
      : fft_engine(dim, n, inembed, istride, idist,
                   fft_type_to_data_type(type).first, onembed, ostride, odist,
                   fft_type_to_data_type(type).second, batch) {}
  /// Construct the class for calculate n-D FFT.
  /// \param [in] dim Dimension number of the data.
  /// \param [in] n Pointer to an array containing each dimension's size.
  /// \param [in] inembed Pointer to an array containing each dimension's size
  /// of the embedded input data.
  /// \param [in] istride Stride size of the input data.
  /// \param [in] idist Distance between the two batches of the input data.
  /// \param [in] onembed Pointer to an array containing each dimension's size
  /// of the embedded output data.
  /// \param [in] ostride Stride size of the output data.
  /// \param [in] odist Distance between the two batches of the output data.
  /// \param [in] type The FFT type.
  /// \param [in] batch The number of FFT operations to perform.
  fft_engine(int dim, int *n, int *inembed, int istride, int idist,
             int *onembed, int ostride, int odist, fft_type type, int batch)
      : fft_engine(dim, n, inembed, istride, idist,
                   fft_type_to_data_type(type).first, onembed, ostride, odist,
                   fft_type_to_data_type(type).second, batch) {}
  /// Construct the class for calculate 1-D FFT.
  /// \param [in] n1 The size of the dimension of the data.
  /// \param [in] type The FFT type.
  /// \param [in] batch The number of FFT operations to perform.
  fft_engine(long long n1, fft_type type, long long batch) {
    _n.resize(1);
    _n[0] = n1;
    std::tie(_input_type, _output_type) = fft_type_to_data_type(type);
    _dim = 1;
    _batch = batch;
    _is_basic = true;
  }
  /// Construct the class for calculate 2-D FFT.
  /// \param [in] n2 The size of the 2nd dimension (outermost) of the data.
  /// \param [in] n1 The size of the 1st dimension (innermost) of the data.
  /// \param [in] type The FFT type.
  fft_engine(long long n2, long long n1, fft_type type) {
    _n.resize(2);
    _n[0] = n2;
    _n[1] = n1;
    std::tie(_input_type, _output_type) = fft_type_to_data_type(type);
    _dim = 2;
    _is_basic = true;
  }
  /// Construct the class for calculate 3-D FFT.
  /// \param [in] n3 The size of the 3rd dimension (outermost) of the data.
  /// \param [in] n2 The size of the 2nd dimension of the data.
  /// \param [in] n1 The size of the 1st dimension (innermost) of the data.
  /// \param [in] type The FFT type.
  fft_engine(long long n3, long long n2, long long n1, fft_type type) {
    _n.resize(3);
    _n[0] = n3;
    _n[1] = n2;
    _n[2] = n1;
    std::tie(_input_type, _output_type) = fft_type_to_data_type(type);
    _dim = 3;
    _is_basic = true;
  }
  /// Execute the FFT calculation.
  /// \param [in] input Pointer to the input data.
  /// \param [out] output Pointer to the output data.
  /// \param [in] direction The FFT direction.
  template<typename input_t, typename output_t>
  void compute(input_t *input, output_t *output, fft_direction direction) {
    _direction = direction;
    if (_input_type == library_data_t::complex_float &&
        _output_type == library_data_t::complex_float) {
      compute_complex<float, oneapi::mkl::dft::precision::SINGLE>(
          _idist, _odist, (float *)input, (float *)output);
    } else if (_input_type == library_data_t::complex_double &&
               _output_type == library_data_t::complex_double) {
      compute_complex<double, oneapi::mkl::dft::precision::DOUBLE>(
          _idist, _odist, (double *)input, (double *)output);
    } else if (_input_type == library_data_t::real_float &&
               _output_type == library_data_t::complex_float) {
      compute_real<float, oneapi::mkl::dft::precision::SINGLE>(
          _idist, _odist, (float *)input, (float *)output);
    } else if (_input_type == library_data_t::complex_float &&
               _output_type == library_data_t::real_float) {
      compute_real<float, oneapi::mkl::dft::precision::SINGLE>(
          _odist, _idist, (float *)input, (float *)output);
    } else if (_input_type == library_data_t::real_double &&
               _output_type == library_data_t::complex_double) {
      compute_real<double, oneapi::mkl::dft::precision::DOUBLE>(
          _idist, _odist, (double *)input, (double *)output);
    } else if (_input_type == library_data_t::complex_double &&
               _output_type == library_data_t::real_double) {
      compute_real<double, oneapi::mkl::dft::precision::DOUBLE>(
          _odist, _idist, (double *)input, (double *)output);
    }
  }
  template<>
  void compute(float *input, sycl::float2 *output, fft_direction direction) {
    _direction = direction;
    compute_real<float, oneapi::mkl::dft::precision::SINGLE>(
        _idist, _odist, (float *)input, (float *)output);
  }
  template<>
  void compute(sycl::float2 *input, float *output, fft_direction direction) {
    _direction = direction;
    compute_real<float, oneapi::mkl::dft::precision::SINGLE>(
        _odist, _idist, (float *)input, (float *)output);
  }
  template<>
  void compute(double *input, sycl::double2 *output, fft_direction direction) {
    _direction = direction;
    compute_real<double, oneapi::mkl::dft::precision::DOUBLE>(
        _idist, _odist, (double *)input, (double *)output);
  }
  template<>
  void compute(sycl::double2 *input, double *output, fft_direction direction) {
    _direction = direction;
    compute_real<double, oneapi::mkl::dft::precision::DOUBLE>(
        _odist, _idist, (double *)input, (double *)output);
  }
  template<>
  void compute(sycl::float2 *input, sycl::float2 *output, fft_direction direction) {
    _direction = direction;
    compute_complex<float, oneapi::mkl::dft::precision::SINGLE>(
        _idist, _odist, (float *)input, (float *)output);
  }
  template<>
  void compute(sycl::double2 *input, sycl::double2 *output, fft_direction direction) {
    _direction = direction;
    compute_complex<double, oneapi::mkl::dft::precision::DOUBLE>(
        _idist, _odist, (double *)input, (double *)output);
  }
  /// Setting the user's SYCL queue for calculation.
  /// \param [in] q Pointer to the SYCL queue.
  void set_queue(sycl::queue *q) { _q = q; }

private:
  static std::pair<library_data_t, library_data_t>
  fft_type_to_data_type(fft_type type) {
    switch (type) {
    case fft_type::real_float_to_complex_float: {
      return std::make_pair(library_data_t::real_float,
                            library_data_t::complex_float);
    }
    case fft_type::complex_float_to_real_float: {
      return std::make_pair(library_data_t::complex_float,
                            library_data_t::real_float);
    }
    case fft_type::real_double_to_complex_double: {
      return std::make_pair(library_data_t::real_double,
                            library_data_t::complex_double);
    }
    case fft_type::complex_double_to_real_double: {
      return std::make_pair(library_data_t::complex_double,
                            library_data_t::real_double);
    }
    case fft_type::complex_float_to_complex_float: {
      return std::make_pair(library_data_t::complex_float,
                            library_data_t::complex_float);
    }
    case fft_type::complex_double_to_complex_double: {
      return std::make_pair(library_data_t::complex_double,
                            library_data_t::complex_double);
    }
    }
  }

  template <typename T>
  void init(int dim, T *n, T *inembed, T istride, T idist,
            library_data_t input_type, T *onembed, T ostride, T odist,
            library_data_t output_type, T batch) {
    _n.resize(dim);
    _inembed.resize(dim);
    _onembed.resize(dim);
    _input_type = input_type;
    _output_type = output_type;
    for (int i = 0; i < dim; i++) {
      _n[i] = n[i];
    }
    if (inembed && onembed) {
      for (int i = 0; i < dim; i++) {
        _inembed[i] = inembed[i];
        _onembed[i] = onembed[i];
      }
      _istride = istride;
      _idist = idist;
      _ostride = ostride;
      _odist = odist;
    } else {
      _is_basic = true;
    }
    _batch = batch;
    _dim = dim;
  }
  template <class Desc_t>
  void set_stride_advance(Desc_t &desc) {
    if (_dim == 1) {
      std::int64_t input_stride[2] = {0, _istride};
      std::int64_t output_stride[2] = {0, _ostride};
      desc.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES,
                     input_stride);
      desc.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES,
                     output_stride);
    } else if (_dim == 2) {
      std::int64_t input_stride[3] = {0, _inembed[1] * _istride, _istride};
      std::int64_t output_stride[3] = {0, _onembed[1] * _ostride, _ostride};
      desc.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES,
                     input_stride);
      desc.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES,
                     output_stride);
    } else if (_dim == 3) {
      std::int64_t input_stride[4] = {0, _inembed[2] * _inembed[1] * _istride,
                                      _inembed[2] * _istride, _istride};
      std::int64_t output_stride[4] = {0, _onembed[2] * _onembed[1] * _ostride,
                                       _onembed[2] * _ostride, _ostride};
      desc.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES,
                     input_stride);
      desc.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES,
                     output_stride);
    }
  }

  template <class T, oneapi::mkl::dft::precision Precision>
  void compute_complex(long long idist, long long odist, T *input, T *output) {
    if (input == output) {
      _is_inplace = true;
    }
    long long fwd_dist = idist;
    long long bwd_dist = odist;
    if (_direction == fft_direction::backward) {
      fwd_dist = odist;
      bwd_dist = idist;
    }
    oneapi::mkl::dft::descriptor<Precision, oneapi::mkl::dft::domain::COMPLEX>
        desc(_n);
    if (!_is_inplace)
      desc.set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                     DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
    desc.set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS,
                   _batch);
    if (!_is_basic) {
      set_stride_advance(desc);
      desc.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, fwd_dist);
      desc.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, bwd_dist);
    } else {
      std::int64_t distance = 1;
      for (auto i : _n)
        distance = distance * i;
      desc.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, distance);
      desc.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, distance);
    }
    desc.commit(*_q);
    if (_is_inplace) {
      auto data_input = dpct::detail::get_memory(reinterpret_cast<T *>(input));
      if (_direction == fft_direction::forward) {
        oneapi::mkl::dft::compute_forward(desc, data_input);
      } else {
        oneapi::mkl::dft::compute_backward(desc, data_input);
      }
    } else {
      auto data_input = dpct::detail::get_memory(reinterpret_cast<T *>(input));
      auto data_output = dpct::detail::get_memory(reinterpret_cast<T *>(output));
      if (_direction == fft_direction::forward) {
        oneapi::mkl::dft::compute_forward(desc, data_input, data_output);
      } else {
        oneapi::mkl::dft::compute_backward(desc, data_input, data_output);
      }
    }
  }

  template <class Desc_t> void set_stride_and_distance_basic(Desc_t &desc) {
    std::int64_t forward_distance = 0;
    std::int64_t backward_distance = 0;
    if (_dim == 1) {
      if (_is_inplace) {
        std::int64_t stride[2] = {0, 1};
        desc.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, stride);
        desc.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, stride);
        forward_distance = 2 * (_n[0] / 2 + 1);
        backward_distance = _n[0] / 2 + 1;
      } else {
        std::int64_t stride[2] = {0, 1};
        if (_direction == fft_direction::forward) {
          desc.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES,
                         stride);
        } else {
          desc.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, stride);
        }
        forward_distance = _n[0];
        backward_distance = _n[0] / 2 + 1;
      }
    } else if (_dim == 2) {
      if (_is_inplace) {
        std::int64_t complex_stride[3] = {0, _n[1] / 2 + 1, 1};
        std::int64_t real_stride[3] = {0, 2 * (_n[1] / 2 + 1), 1};
        if (_direction == fft_direction::forward) {
          desc.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES,
                         real_stride);
          desc.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES,
                         complex_stride);
        } else {
          desc.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES,
                         complex_stride);
          desc.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES,
                         real_stride);
        }
        forward_distance = _n[0] * 2 * (_n[1] / 2 + 1);
        backward_distance = _n[0] * (_n[1] / 2 + 1);
      } else {
        std::int64_t stride[3] = {0, _n[1] / 2 + 1, 1};
        if (_direction == fft_direction::forward) {
          desc.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES,
                         stride);
        } else {
          desc.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, stride);
        }
        forward_distance = _n[0] * _n[1];
        backward_distance = _n[0] * (_n[1] / 2 + 1);
      }
    } else if (_dim == 3) {
      if (_is_inplace) {
        std::int64_t complex_stride[4] = {0, _n[1] * (_n[2] / 2 + 1),
                                          _n[2] / 2 + 1, 1};
        std::int64_t real_stride[4] = {0, _n[1] * 2 * (_n[2] / 2 + 1),
                                       2 * (_n[2] / 2 + 1), 1};

        if (_direction == fft_direction::forward) {
          desc.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES,
                         real_stride);
          desc.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES,
                         complex_stride);
        } else {
          desc.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES,
                         complex_stride);
          desc.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES,
                         real_stride);
        }
        forward_distance = _n[0] * _n[1] * 2 * (_n[2] / 2 + 1);
        backward_distance = _n[0] * _n[1] * (_n[2] / 2 + 1);
      } else {
        std::int64_t stride[4] = {0, _n[1] * (_n[2] / 2 + 1), _n[2] / 2 + 1, 1};
        if (_direction == fft_direction::forward) {
          desc.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES,
                         stride);
        } else {
          desc.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, stride);
        }
        forward_distance = _n[0] * _n[1] * _n[2];
        backward_distance = _n[0] * _n[1] * (_n[2] / 2 + 1);
      }
    }
    desc.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE,
                   forward_distance);
    desc.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE,
                   backward_distance);
  }

  template <class T, oneapi::mkl::dft::precision Precision>
  void compute_real(int fwd_dist, int bwd_dist, T *input, T *output) {
    if (input == output) {
      _is_inplace = true;
    }
    oneapi::mkl::dft::descriptor<Precision, oneapi::mkl::dft::domain::REAL>
        desc(_n);
    if (!_is_inplace)
      desc.set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                     DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
    desc.set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS,
                   _batch);
    if (!_is_basic) {
      set_stride_advance(desc);
      desc.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, fwd_dist);
      desc.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, bwd_dist);
    } else {
      set_stride_and_distance_basic(desc);
    }
    desc.commit(*_q);
    if (_is_inplace) {
      auto data_input = dpct::detail::get_memory(reinterpret_cast<T *>(input));
      if (_direction == fft_direction::forward) {
        oneapi::mkl::dft::compute_forward(desc, data_input);
      } else {
        oneapi::mkl::dft::compute_backward(desc, data_input);
      }
    } else {
      auto data_input = dpct::detail::get_memory(reinterpret_cast<T *>(input));
      auto data_output = dpct::detail::get_memory(reinterpret_cast<T *>(output));
      if (_direction == fft_direction::forward) {
        oneapi::mkl::dft::compute_forward(desc, data_input, data_output);
      } else {
        oneapi::mkl::dft::compute_backward(desc, data_input, data_output);
      }
    }
  }

private:
  sycl::queue *_q = &dpct::get_default_queue();
  int _dim;
  std::vector<std::int64_t> _n;
  std::vector<std::int64_t> _inembed;
  std::int64_t _istride;
  std::int64_t _idist;
  library_data_t _input_type;
  std::vector<std::int64_t> _onembed;
  std::int64_t _ostride;
  std::int64_t _odist;
  library_data_t _output_type;
  std::int64_t _batch = 1;
  bool _is_basic = false;
  bool _is_inplace = false;
  fft_direction _direction;
};
// DPCT_LABEL_END
} // namespace fft
} // namespace dpct

#endif // __DPCT_FFT_UTILS_HPP__