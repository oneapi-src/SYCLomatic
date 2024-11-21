//===--------------- MapNamesRandom.cpp-----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MapNamesRandom.h"
#include "ASTTraversal.h"
#include "FileGenerator/GenFiles.h"
#include "RuleInfra/CallExprRewriter.h"
#include "RuleInfra/MapNames.h"
#include "RulesDNN/DNNAPIMigration.h"
#include "RulesLang/RulesLang.h"
#include <map>

using namespace clang;
using namespace clang::dpct;

namespace clang {
namespace dpct {

MapNamesRandom::MapTy MapNamesRandom::RandomEngineTypeMap;
MapNamesRandom::MapTy MapNamesRandom::RandomOrderingTypeMap;
MapNamesRandom::MapTy MapNamesRandom::DeviceRandomGeneratorTypeMap;

void MapNamesRandom::setExplicitNamespaceMap(
    const std::set<ExplicitNamespace> &ExplicitNamespaces) {
  // Host Random Engine Type mapping
  RandomEngineTypeMap = {
      {"CURAND_RNG_PSEUDO_DEFAULT", MapNames::getLibraryHelperNamespace() +
                                        "rng::random_engine_type::mcg59"},
      {"CURAND_RNG_PSEUDO_XORWOW", MapNames::getLibraryHelperNamespace() +
                                       "rng::random_engine_type::mcg59"},
      {"CURAND_RNG_PSEUDO_MRG32K3A", MapNames::getLibraryHelperNamespace() +
                                         "rng::random_engine_type::mrg32k3a"},
      {"CURAND_RNG_PSEUDO_MTGP32", MapNames::getLibraryHelperNamespace() +
                                       "rng::random_engine_type::mt2203"},
      {"CURAND_RNG_PSEUDO_MT19937", MapNames::getLibraryHelperNamespace() +
                                        "rng::random_engine_type::mt19937"},
      {"CURAND_RNG_PSEUDO_PHILOX4_32_10",
       MapNames::getLibraryHelperNamespace() +
           "rng::random_engine_type::philox4x32x10"},
      {"CURAND_RNG_QUASI_DEFAULT", MapNames::getLibraryHelperNamespace() +
                                       "rng::random_engine_type::sobol"},
      {"CURAND_RNG_QUASI_SOBOL32", MapNames::getLibraryHelperNamespace() +
                                       "rng::random_engine_type::sobol"},
      {"CURAND_RNG_QUASI_SCRAMBLED_SOBOL32",
       MapNames::getLibraryHelperNamespace() +
           "rng::random_engine_type::sobol"},
      {"CURAND_RNG_QUASI_SOBOL64", MapNames::getLibraryHelperNamespace() +
                                       "rng::random_engine_type::sobol"},
      {"CURAND_RNG_QUASI_SCRAMBLED_SOBOL64",
       MapNames::getLibraryHelperNamespace() +
           "rng::random_engine_type::sobol"},
  };

  // Random Ordering Type mapping
  RandomOrderingTypeMap = {
      {"CURAND_ORDERING_PSEUDO_DEFAULT",
       MapNames::getLibraryHelperNamespace() + "rng::random_mode::best"},
      {"CURAND_ORDERING_PSEUDO_BEST",
       MapNames::getLibraryHelperNamespace() + "rng::random_mode::best"},
      // CURAND_ORDERING_PSEUDO_SEEDED not support now.
      {"CURAND_ORDERING_PSEUDO_LEGACY",
       MapNames::getLibraryHelperNamespace() + "rng::random_mode::legacy"},
      {"CURAND_ORDERING_PSEUDO_DYNAMIC",
       MapNames::getLibraryHelperNamespace() + "rng::random_mode::optimal"},
      // CURAND_ORDERING_QUASI_DEFAULT not support now.
  };

  // Device Random Generator Type mapping
  DeviceRandomGeneratorTypeMap = {
      {"curandStateXORWOW_t", MapNames::getLibraryHelperNamespace() +
                                  "rng::device::rng_generator<oneapi::"
                                  "mkl::rng::device::mcg59<1>>"},
      {"curandStateXORWOW", MapNames::getLibraryHelperNamespace() +
                                "rng::device::rng_generator<oneapi::"
                                "mkl::rng::device::mcg59<1>>"},
      {"curandState_t", MapNames::getLibraryHelperNamespace() +
                            "rng::device::rng_generator<oneapi::mkl::"
                            "rng::device::mcg59<1>>"},
      {"curandState", MapNames::getLibraryHelperNamespace() +
                          "rng::device::rng_generator<oneapi::mkl::"
                          "rng::device::mcg59<1>>"},
      {"curandStatePhilox4_32_10_t",
       MapNames::getLibraryHelperNamespace() +
           "rng::device::rng_generator<oneapi::mkl::rng::device::"
           "philox4x32x10<1>>"},
      {"curandStatePhilox4_32_10",
       MapNames::getLibraryHelperNamespace() +
           "rng::device::rng_generator<"
           "oneapi::mkl::rng::device::philox4x32x10<1>>"},
      {"curandStateMRG32k3a_t", MapNames::getLibraryHelperNamespace() +
                                    "rng::device::rng_generator<"
                                    "oneapi::mkl::rng::device::mrg32k3a<1>>"},
      {"curandStateMRG32k3a", MapNames::getLibraryHelperNamespace() +
                                  "rng::device::rng_generator<oneapi::"
                                  "mkl::rng::device::mrg32k3a<1>>"},
  };
}

const std::map<std::string, std::string> MapNamesRandom::RandomGenerateFuncMap{
    {"curandGenerate", {"generate_uniform_bits"}},
    {"curandGenerateLongLong", {"generate_uniform_bits"}},
    {"curandGenerateLogNormal", {"generate_lognormal"}},
    {"curandGenerateLogNormalDouble", {"generate_lognormal"}},
    {"curandGenerateNormal", {"generate_gaussian"}},
    {"curandGenerateNormalDouble", {"generate_gaussian"}},
    {"curandGeneratePoisson", {"generate_poisson"}},
    {"curandGenerateUniform", {"generate_uniform"}},
    {"curandGenerateUniformDouble", {"generate_uniform"}},
};

} // namespace dpct
} // namespace clang