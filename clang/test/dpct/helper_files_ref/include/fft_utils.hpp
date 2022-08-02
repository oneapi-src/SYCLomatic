//==---- fft_utils.hpp ----------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_FFT_UTILS_HPP__
#define __DPCT_FFT_UTILS_HPP__

#include <CL/sycl.hpp>
#include <oneapi/mkl.hpp>
#include "memory.hpp"
#include "lib_common_utils.hpp"

namespace dpct {
namespace fft {
enum class fft_dir { forward, backward };
enum class fft_type {
  real_float_to_complex_float,
  complex_float_to_real_float,
  real_double_to_complex_double,
  complex_double_to_real_double,
  complex_float_to_complex_float,
  complex_double_to_complex_double,
};

class fft_solver {
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

public:
  fft_solver(int rank, long long *n, long long *inembed, long long istride,
             long long idist, library_data_t inputtype, long long *onembed,
             long long ostride, long long odist, library_data_t outputtype,
             long long batch) {
    _n.resize(rank);
    _inembed.resize(rank);
    _onembed.resize(rank);
    _inputtype = inputtype;
    _outputtype = outputtype;
    for (int i = 0; i < rank; i++) {
      _n[i] = n[i];
    }
    if (inembed && onembed) {
      for (int i = 0; i < rank; i++) {
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
    _rank = rank;
  }
  fft_solver(int rank, long long *n, long long *inembed, long long istride,
             long long idist, long long *onembed, long long ostride,
             long long odist, fft_type type, long long batch)
      : fft_solver(rank, n, inembed, istride, idist,
                   fft_type_to_data_type(type).first, onembed, ostride, odist,
                   fft_type_to_data_type(type).second, batch) {}
  fft_solver(long long n, fft_type type, long long batch) {
    _n.resize(1);
    _n[0] = n;
    std::tie(_inputtype, _outputtype) = fft_type_to_data_type(type);
    _rank = 1;
    _batch = batch;
    _is_basic = true;
  }
  fft_solver(long long n2, long long n1, fft_type type) {
    _n.resize(2);
    _n[0] = n2;
    _n[1] = n1;
    std::tie(_inputtype, _outputtype) = fft_type_to_data_type(type);
    _rank = 2;
    _is_basic = true;
  }
  fft_solver(long long n3, long long n2, long long n1, fft_type type) {
    _n.resize(3);
    _n[0] = n3;
    _n[1] = n2;
    _n[2] = n1;
    std::tie(_inputtype, _outputtype) = fft_type_to_data_type(type);
    _rank = 3;
    _is_basic = true;
  }

  void set_queue(sycl::queue *q) { _q = q; }

private:
  template <class DESC_T>
  void set_stride_and_distance_advance(DESC_T &plan, int FWD_DIS, int BWD_DIS) {
    if (_rank == 1) {
      std::int64_t input_stride[2] = {0, _istride};
      std::int64_t output_stride[2] = {0, _ostride};
      plan.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES,
                     input_stride);
      plan.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES,
                     output_stride);
    } else if (_rank == 2) {
      std::int64_t input_stride[3] = {0, _inembed[1] * _istride, _istride};
      std::int64_t output_stride[3] = {0, _onembed[1] * _ostride, _ostride};
      plan.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES,
                     input_stride);
      plan.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES,
                     output_stride);
    } else if (_rank == 3) {
      std::int64_t input_stride[4] = {0, _inembed[2] * _inembed[1] * _istride,
                                      _inembed[2] * _istride, _istride};
      std::int64_t output_stride[4] = {0, _onembed[2] * _onembed[1] * _ostride,
                                       _onembed[2] * _ostride, _ostride};
      plan.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES,
                     input_stride);
      plan.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES,
                     output_stride);
    }
    plan.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, FWD_DIS);
    plan.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, BWD_DIS);
  }

  template <class T, oneapi::mkl::dft::precision PRECISION>
  void compute_complex(int FWD_DIS, int BWD_DIS, T *input, T *output) {
    oneapi::mkl::dft::descriptor<PRECISION, oneapi::mkl::dft::domain::COMPLEX>
        plan(_n);
    if (!_is_inplace)
      plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                     DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
    plan.set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS,
                   _batch);
    if (!_is_basic) {
      set_stride_and_distance_advance(plan, FWD_DIS, BWD_DIS);
    } else {
      std::int64_t distance = 1;
      for (const auto &i : _n)
        distance = distance * i;
      plan.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, distance);
      plan.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, distance);
    }
    plan.commit(*_q);
    if (_is_inplace) {
#ifdef DPCT_USM_LEVEL_NONE
      auto input_buffer = dpct::get_buffer<T>(input);
      if (_direction == fft_dir::forward) {
        oneapi::mkl::dft::compute_forward(plan, input_buffer);
      } else {
        oneapi::mkl::dft::compute_backward(plan, input_buffer);
      }
#else
      if (_direction == fft_dir::forward) {
        oneapi::mkl::dft::compute_forward(plan, input);
      } else {
        oneapi::mkl::dft::compute_backward(plan, input);
      }
#endif
    } else {
#ifdef DPCT_USM_LEVEL_NONE
      auto input_buffer = dpct::get_buffer<T>(input);
      auto output_buffer = dpct::get_buffer<T>(output);
      if (_direction == fft_dir::forward) {
        oneapi::mkl::dft::compute_forward(plan, input_buffer, output_buffer);
      } else {
        oneapi::mkl::dft::compute_backward(plan, input_buffer, output_buffer);
      }
#else
      if (_direction == fft_dir::forward) {
        oneapi::mkl::dft::compute_forward(plan, input, output);
      } else {
        oneapi::mkl::dft::compute_backward(plan, input, output);
      }
#endif
    }
  }

  template <class DESC_T> void set_stride_and_distance_basic(DESC_T &plan) {
    std::int64_t forward_distance = 0;
    std::int64_t backward_distance = 0;
    if (_rank == 1) {
      if (_is_inplace) {
        std::int64_t stride[2] = {0, 1};
        plan.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, stride);
        plan.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, stride);
        forward_distance = 2 * (_n[0] / 2 + 1);
        backward_distance = _n[0] / 2 + 1;
      } else {
        std::int64_t stride[2] = {0, 1};
        if (_direction == fft_dir::forward) {
          plan.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES,
                         stride);
        } else {
          plan.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, stride);
        }
        forward_distance = _n[0];
        backward_distance = _n[0] / 2 + 1;
      }
    } else if (_rank == 2) {
      if (_is_inplace) {
        std::int64_t complex_stride[3] = {0, _n[1] / 2 + 1, 1};
        std::int64_t real_stride[3] = {0, 2 * (_n[1] / 2 + 1), 1};
        if (_direction == fft_dir::forward) {
          plan.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES,
                         real_stride);
          plan.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES,
                         complex_stride);
        } else {
          plan.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES,
                         complex_stride);
          plan.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES,
                         real_stride);
        }
        forward_distance = _n[0] * 2 * (_n[1] / 2 + 1);
        backward_distance = _n[0] * (_n[1] / 2 + 1);
      } else {
        std::int64_t stride[3] = {0, _n[1] / 2 + 1, 1};
        if (_direction == fft_dir::forward) {
          plan.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES,
                         stride);
        } else {
          plan.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, stride);
        }
        forward_distance = _n[0] * _n[1];
        backward_distance = _n[0] * (_n[1] / 2 + 1);
      }
    } else if (_rank == 3) {
      if (_is_inplace) {
        std::int64_t complex_stride[4] = {0, _n[1] * (_n[2] / 2 + 1),
                                          _n[2] / 2 + 1, 1};
        std::int64_t real_stride[4] = {0, _n[1] * 2 * (_n[2] / 2 + 1),
                                       2 * (_n[2] / 2 + 1), 1};

        if (_direction == fft_dir::forward) {
          plan.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES,
                         real_stride);
          plan.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES,
                         complex_stride);
        } else {
          plan.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES,
                         complex_stride);
          plan.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES,
                         real_stride);
        }
        forward_distance = _n[0] * _n[1] * 2 * (_n[2] / 2 + 1);
        backward_distance = _n[0] * _n[1] * (_n[2] / 2 + 1);
      } else {
        std::int64_t stride[4] = {0, _n[1] * (_n[2] / 2 + 1), _n[2] / 2 + 1, 1};
        if (_direction == fft_dir::forward) {
          plan.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES,
                         stride);
        } else {
          plan.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, stride);
        }
        forward_distance = _n[0] * _n[1] * _n[2];
        backward_distance = _n[0] * _n[1] * (_n[2] / 2 + 1);
      }
    }
    plan.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE,
                   forward_distance);
    plan.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE,
                   backward_distance);
  }

  template <class T, oneapi::mkl::dft::precision PRECISION>
  void compute_real(int FWD_DIS, int BWD_DIS, T *input, T *output) {
    oneapi::mkl::dft::descriptor<PRECISION, oneapi::mkl::dft::domain::REAL>
        plan(_n);
    if (!_is_inplace)
      plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                     DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
    plan.set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS,
                   _batch);
    if (!_is_basic) {
      set_stride_and_distance_advance(plan, FWD_DIS, BWD_DIS);
    } else {
      set_stride_and_distance_basic(plan);
    }
    plan.commit(*_q);
    if (_is_inplace) {
#ifdef DPCT_USM_LEVEL_NONE
      auto input_buffer = dpct::get_buffer<T>(input);
      if (_direction == fft_dir::forward) {
        oneapi::mkl::dft::compute_forward(plan, input_buffer);
      } else {
        oneapi::mkl::dft::compute_backward(plan, input_buffer);
      }
#else
      if (_direction == fft_dir::forward) {
        oneapi::mkl::dft::compute_forward(plan, input);
      } else {
        oneapi::mkl::dft::compute_backward(plan, input);
      }
#endif
    } else {
#ifdef DPCT_USM_LEVEL_NONE
      auto input_buffer = dpct::get_buffer<T>(input);
      auto output_buffer = dpct::get_buffer<T>(output);
      if (_direction == fft_dir::forward) {
        oneapi::mkl::dft::compute_forward(plan, input_buffer, output_buffer);
      } else {
        oneapi::mkl::dft::compute_backward(plan, input_buffer, output_buffer);
      }
#else
      if (_direction == fft_dir::forward) {
        oneapi::mkl::dft::compute_forward(plan, input, output);
      } else {
        oneapi::mkl::dft::compute_backward(plan, input, output);
      }
#endif
    }
  }

public:
  void compute(void *input, void *output, fft_dir direction) {
    _direction = direction;
    if (input == output) {
      _is_inplace = true;
    }
    if (_inputtype == library_data_t::complex_float &&
        _outputtype == library_data_t::complex_float &&
        _direction == fft_dir::forward) {
      // c2c forward
      compute_complex<float, oneapi::mkl::dft::precision::SINGLE>(
          _idist, _odist, (float *)input, (float *)output);
    } else if (_inputtype == library_data_t::complex_float &&
               _outputtype == library_data_t::complex_float &&
               _direction == fft_dir::backward) {
      // c2c backward
      compute_complex<float, oneapi::mkl::dft::precision::SINGLE>(
          _odist, _idist, (float *)input, (float *)output);
    } else if (_inputtype == library_data_t::complex_double &&
               _outputtype == library_data_t::complex_double &&
               _direction == fft_dir::forward) {
      // z2z forward
      compute_complex<double, oneapi::mkl::dft::precision::DOUBLE>(
          _idist, _odist, (double *)input, (double *)output);
    } else if (_inputtype == library_data_t::complex_double &&
               _outputtype == library_data_t::complex_double &&
               _direction == fft_dir::backward) {
      // z2z backward
      compute_complex<double, oneapi::mkl::dft::precision::DOUBLE>(
          _odist, _idist, (double *)input, (double *)output);
    } else if (_inputtype == library_data_t::real_float &&
               _outputtype == library_data_t::complex_float) {
      // r2c
      compute_real<float, oneapi::mkl::dft::precision::SINGLE>(
          _idist, _odist, (float *)input, (float *)output);
    } else if (_inputtype == library_data_t::complex_float &&
               _outputtype == library_data_t::real_float) {
      // c2r
      compute_real<float, oneapi::mkl::dft::precision::SINGLE>(
          _odist, _idist, (float *)input, (float *)output);
    } else if (_inputtype == library_data_t::real_double &&
               _outputtype == library_data_t::complex_double) {
      // d2z
      compute_real<double, oneapi::mkl::dft::precision::DOUBLE>(
          _idist, _odist, (double *)input, (double *)output);
    } else if (_inputtype == library_data_t::complex_double &&
               _outputtype == library_data_t::real_double) {
      // z2d
      compute_real<double, oneapi::mkl::dft::precision::DOUBLE>(
          _odist, _idist, (double *)input, (double *)output);
    }
  }

private:
  sycl::queue *_q = &dpct::get_default_queue();
  int _rank;
  std::vector<std::int64_t> _n;
  std::vector<std::int64_t> _inembed;
  std::int64_t _istride;
  std::int64_t _idist;
  library_data_t _inputtype;
  std::vector<std::int64_t> _onembed;
  std::int64_t _ostride;
  std::int64_t _odist;
  library_data_t _outputtype;
  std::int64_t _batch = 1;
  bool _is_basic = false;
  bool _is_inplace = false;
  fft_dir _direction;
};
} // namespace fft
} // namespace dpct

#endif // __DPCT_FFT_UTILS_HPP__