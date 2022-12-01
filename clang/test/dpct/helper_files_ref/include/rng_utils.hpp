//==---- rng_utils.hpp ----------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_RNG_UTILS_HPP__
#define __DPCT_RNG_UTILS_HPP__

#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>
#include <oneapi/mkl/rng/device.hpp>

namespace dpct {
namespace rng {
namespace device {
/// The random number generator on device.
/// \tparam engine_t The device random number generator engine. It can only be
/// oneapi::mkl::rng::device::mrg32k3a<1> or
/// oneapi::mkl::rng::device::mcg59<1> or
/// oneapi::mkl::rng::device::philox4x32x10<1>.
template <typename engine_t> class rng_generator {
  static_assert(
      std::is_same_v<engine_t, oneapi::mkl::rng::device::mrg32k3a<1>> ||
          std::is_same_v<engine_t, oneapi::mkl::rng::device::mcg59<1> ||
          std::is_same_v<engine_t, oneapi::mkl::rng::device::philox4x32x10<1>>,
      "engine_t can only be oneapi::mkl::rng::device::mrg32k3a<1> or "
      "oneapi::mkl::rng::device::mcg59<1> or "
      "oneapi::mkl::rng::device::philox4x32x10<1>.");
  static constexpr std::uint64_t default_seed = 0;
  oneapi::mkl::rng::device::bits<std::uint32_t> _distr_bits;
  oneapi::mkl::rng::device::gaussian<float> _distr_gaussian_float;
  oneapi::mkl::rng::device::gaussian<double> _distr_gaussian_double;
  oneapi::mkl::rng::device::lognormal<float> _distr_lognormal_float;
  oneapi::mkl::rng::device::lognormal<double> _distr_lognormal_double;
  oneapi::mkl::rng::device::poisson<std::uint32_t> _distr_poisson;
  oneapi::mkl::rng::device::uniform<float> _distr_uniform_float;
  oneapi::mkl::rng::device::uniform<double> _distr_uniform_double;
  engine_t _engine;

public:
  /// Default constructor of rng_generator
  rng_generator() { _engine = engine_t(default_seed); }
  /// Constructor of rng_generator
  /// \param [in] seed The seed to initialize the engine state.
  /// \param [in] num_to_skip Set the number of elements need to be skipped.
  /// The number is calculated as: num_to_skip[0] + num_to_skip[1] * 2^64 +
  /// num_to_skip[2] * 2^128 + ... + num_to_skip[n-1] * 2^(64*(n-1))
  rng_generator(std::uint64_t seed,
                std::initializer_list<std::uint64_t> num_to_skip) {
    _engine = engine_t(seed, num_to_skip);
  }

  /// Generate random number(s) obeys distribution \tparam distr_t.
  /// \tparam T The distribution of the random number. It can only be
  /// oneapi::mkl::rng::device::bits<std::uint32_t>,
  /// oneapi::mkl::rng::device::gaussian<float>,
  /// oneapi::mkl::rng::device::gaussian<double>,
  /// oneapi::mkl::rng::device::lognormal<float>,
  /// oneapi::mkl::rng::device::lognormal<double>,
  /// oneapi::mkl::rng::device::poisson<std::uint32_t>,
  /// oneapi::mkl::rng::device::uniform<float> or
  /// oneapi::mkl::rng::device::uniform<double>
  /// \tparam vec_size The length of the return vector. It can only be 1, 2
  /// or 4.
  /// \param distr_params The parameter(s) for lognormal or poisson
  /// distribution.
  /// \return The vector of the random number(s).
  template <typename distr_t, int vec_size, class... distr_params_t>
  auto generate(distr_params_t... distr_params) {
    static_assert(vec_size == 1 || vec_size == 2 || vec_size == 4,
                  "vec_size is not supported.");
    static_assert(
        std::is_same_v<distr_t,
                       oneapi::mkl::rng::device::bits<std::uint32_t>> ||
            std::is_same_v<distr_t,
                           oneapi::mkl::rng::device::gaussian<float>> ||
            std::is_same_v<distr_t,
                           oneapi::mkl::rng::device::gaussian<double>> ||
            std::is_same_v<distr_t,
                           oneapi::mkl::rng::device::lognormal<float>> ||
            std::is_same_v<distr_t,
                           oneapi::mkl::rng::device::lognormal<double>> ||
            std::is_same_v<distr_t,
                           oneapi::mkl::rng::device::poisson<std::uint32_t>> ||
            std::is_same_v<distr_t, oneapi::mkl::rng::device::uniform<float>> ||
            std::is_same_v<distr_t, oneapi::mkl::rng::device::uniform<double>>,
        "distribution is not supported.");

    if constexpr (std::is_same_v<
                      distr_t, oneapi::mkl::rng::device::bits<std::uint32_t>>) {
      return generate_vec<vec_size>(_distr_bits);
    }
    if constexpr (std::is_same_v<distr_t,
                                 oneapi::mkl::rng::device::gaussian<float>>) {
      return generate_vec<vec_size>(_distr_gaussian_float);
    }
    if constexpr (std::is_same_v<distr_t,
                                 oneapi::mkl::rng::device::gaussian<double>>) {
      return generate_vec<vec_size>(_distr_gaussian_double);
    }
    if constexpr (std::is_same_v<distr_t,
                                 oneapi::mkl::rng::device::lognormal<float>>) {
      return generate_vec<vec_size>(_distr_lognormal_float, distr_params...,
                                    0.0f, 1.0f);
    }
    if constexpr (std::is_same_v<distr_t,
                                 oneapi::mkl::rng::device::lognormal<double>>) {
      return generate_vec<vec_size>(_distr_lognormal_double, distr_params...,
                                    0.0, 1.0);
    }
    if constexpr (std::is_same_v<distr_t, oneapi::mkl::rng::device::poisson<
                                              std::uint32_t>>) {
      return generate_vec<vec_size>(_distr_poisson, distr_params...);
    }
    if constexpr (std::is_same_v<distr_t,
                                 oneapi::mkl::rng::device::uniform<float>>) {
      return generate_vec<vec_size>(_distr_uniform_float);
    }
    if constexpr (std::is_same_v<distr_t,
                                 oneapi::mkl::rng::device::uniform<double>>) {
      return generate_vec<vec_size>(_distr_uniform_double);
    }
  }

  /// Get the random number generator engine.
  /// \return The reference of the internal random number generator engine.
  engine_t &get_engine() { return _engine; }

private:
  template <int vec_size, typename distr_t, class... distr_params_t>
  auto generate_vec(distr_t &distr, distr_params_t... distr_params) {
    if constexpr (sizeof...(distr_params_t)) {
      typename distr_t::param_type pt(distr_params...);
      distr.param(pt);
    }
    if constexpr (vec_size == 4) {
      return oneapi::mkl::rng::device::generate(distr, _engine);
    } else if constexpr (vec_size == 1) {
      return oneapi::mkl::rng::device::generate_single(distr, _engine);
    } else if constexpr (vec_size == 2) {
      sycl::vec<typename distr_t::result_type, 2> res;
      res.x() = oneapi::mkl::rng::device::generate_single(distr, _engine);
      res.y() = oneapi::mkl::rng::device::generate_single(distr, _engine);
      return res;
    }
  }
};

} // namespace device
} // namespace rng
} // namespace dpct

#endif // __DPCT_RNG_UTILS_HPP__
