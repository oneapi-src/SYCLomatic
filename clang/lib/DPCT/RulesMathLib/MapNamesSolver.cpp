//===--------------- MapNamesSolver.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MapNamesSolver.h"
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
std::unordered_set<std::string> MapNamesSolver::SOLVERAPIWithRewriter = {
    "cusolverDnSetAdvOptions",
    "cusolverDnSetStream",
    "cusolverDnGetStream",
    "cusolverDnCreateParams",
    "cusolverDnDestroyParams",
    "cusolverDnSpotrfBatched",
    "cusolverDnDpotrfBatched",
    "cusolverDnCpotrfBatched",
    "cusolverDnZpotrfBatched",
    "cusolverDnSpotrsBatched",
    "cusolverDnDpotrsBatched",
    "cusolverDnCpotrsBatched",
    "cusolverDnZpotrsBatched",
    "cusolverDnSsygvd",
    "cusolverDnDsygvd",
    "cusolverDnSsygvd_bufferSize",
    "cusolverDnDsygvd_bufferSize",
    "cusolverDnChegvd",
    "cusolverDnZhegvd",
    "cusolverDnChegvd_bufferSize",
    "cusolverDnZhegvd_bufferSize",
    "cusolverDnXgetrf",
    "cusolverDnXgetrf_bufferSize",
    "cusolverDnXgetrs",
    "cusolverDnXgeqrf",
    "cusolverDnXgeqrf_bufferSize",
    "cusolverDnGetrf",
    "cusolverDnGetrf_bufferSize",
    "cusolverDnGetrs",
    "cusolverDnGeqrf",
    "cusolverDnGeqrf_bufferSize",
    "cusolverDnCreateGesvdjInfo",
    "cusolverDnDestroyGesvdjInfo",
    "cusolverDnSgesvdj_bufferSize",
    "cusolverDnDgesvdj_bufferSize",
    "cusolverDnCgesvdj_bufferSize",
    "cusolverDnZgesvdj_bufferSize",
    "cusolverDnXgesvd_bufferSize",
    "cusolverDnGesvd_bufferSize",
    "cusolverDnSgesvdj",
    "cusolverDnDgesvdj",
    "cusolverDnCgesvdj",
    "cusolverDnZgesvdj",
    "cusolverDnXgesvd",
    "cusolverDnGesvd",
    "cusolverDnXpotrf_bufferSize",
    "cusolverDnPotrf_bufferSize",
    "cusolverDnXpotrf",
    "cusolverDnPotrf",
    "cusolverDnXpotrs",
    "cusolverDnPotrs",
    "cusolverDnSgeqrf_bufferSize",
    "cusolverDnDgeqrf_bufferSize",
    "cusolverDnCgeqrf_bufferSize",
    "cusolverDnZgeqrf_bufferSize",
    "cusolverDnSsyevdx",
    "cusolverDnDsyevdx",
    "cusolverDnSsyevdx_bufferSize",
    "cusolverDnDsyevdx_bufferSize",
    "cusolverDnCheevdx",
    "cusolverDnZheevdx",
    "cusolverDnCheevdx_bufferSize",
    "cusolverDnZheevdx_bufferSize",
    "cusolverDnSsygvdx",
    "cusolverDnDsygvdx",
    "cusolverDnSsygvdx_bufferSize",
    "cusolverDnDsygvdx_bufferSize",
    "cusolverDnChegvdx",
    "cusolverDnZhegvdx",
    "cusolverDnChegvdx_bufferSize",
    "cusolverDnZhegvdx_bufferSize",
    "cusolverDnSsygvj",
    "cusolverDnDsygvj",
    "cusolverDnSsygvj_bufferSize",
    "cusolverDnDsygvj_bufferSize",
    "cusolverDnChegvj",
    "cusolverDnZhegvj",
    "cusolverDnChegvj_bufferSize",
    "cusolverDnZhegvj_bufferSize",
    "cusolverDnXsyevdx",
    "cusolverDnXsyevdx_bufferSize",
    "cusolverDnSyevdx",
    "cusolverDnSyevdx_bufferSize",
    "cusolverDnCreateSyevjInfo",
    "cusolverDnDestroySyevjInfo",
    "cusolverDnSsyevj",
    "cusolverDnDsyevj",
    "cusolverDnSsyevj_bufferSize",
    "cusolverDnDsyevj_bufferSize",
    "cusolverDnCheevj",
    "cusolverDnZheevj",
    "cusolverDnCheevj_bufferSize",
    "cusolverDnZheevj_bufferSize",
    "cusolverDnXsyevd",
    "cusolverDnXsyevd_bufferSize",
    "cusolverDnSyevd",
    "cusolverDnSyevd_bufferSize",
    "cusolverDnXtrtri",
    "cusolverDnXtrtri_bufferSize",
    "cusolverDnSsyevd_bufferSize",
    "cusolverDnDsyevd_bufferSize",
    "cusolverDnCheevd_bufferSize",
    "cusolverDnZheevd_bufferSize",
    "cusolverDnSsyevd",
    "cusolverDnDsyevd",
    "cusolverDnCheevd",
    "cusolverDnZheevd"};

// SOLVER enums mapping
const MapNamesSolver::MapTy MapNamesSolver::SOLVEREnumsMap{
    {"CUSOLVER_EIG_TYPE_1", "1"},
    {"CUSOLVER_EIG_TYPE_2", "2"},
    {"CUSOLVER_EIG_TYPE_3", "3"},
    {"CUSOLVER_EIG_MODE_NOVECTOR", "oneapi::mkl::job::novec"},
    {"CUSOLVER_EIG_MODE_VECTOR", "oneapi::mkl::job::vec"},
};

// SOLVER functions names and parameters replacements information mapping
const std::map<std::string, MapNamesSolver::SOLVERFuncReplInfo>
    MapNamesSolver::SOLVERFuncReplInfoMap{
        {"cusolverDnSpotrf_bufferSize",
         MapNamesSolver::SOLVERFuncReplInfo::migrateReturnAndRedundant(
             true, std::vector<int>{3, 5},
             "oneapi::mkl::lapack::potrf_scratchpad_size<float>")},
        {"cusolverDnDpotrf_bufferSize",
         MapNamesSolver::SOLVERFuncReplInfo::migrateReturnAndRedundant(
             true, std::vector<int>{3, 5},
             "oneapi::mkl::lapack::potrf_scratchpad_size<double>")},
        {"cusolverDnCpotrf_bufferSize",
         MapNamesSolver::SOLVERFuncReplInfo::migrateReturnAndRedundant(
             true, std::vector<int>{3, 5},
             "oneapi::mkl::lapack::potrf_scratchpad_size<std::complex<float>"
             ">")},
        {"cusolverDnZpotrf_bufferSize",
         MapNamesSolver::SOLVERFuncReplInfo::migrateReturnAndRedundant(
             true, std::vector<int>{3, 5},
             "oneapi::mkl::lapack::potrf_scratchpad_size<std::complex<double>"
             ">")},
        {"cusolverDnSgetrf_bufferSize",
         MapNamesSolver::SOLVERFuncReplInfo::migrateReturnAndRedundant(
             true, std::vector<int>{3, 5},
             "oneapi::mkl::lapack::getrf_scratchpad_size<float>")},
        {"cusolverDnDgetrf_bufferSize",
         MapNamesSolver::SOLVERFuncReplInfo::migrateReturnAndRedundant(
             true, std::vector<int>{3, 5},
             "oneapi::mkl::lapack::getrf_scratchpad_size<double>")},
        {"cusolverDnCgetrf_bufferSize",
         MapNamesSolver::SOLVERFuncReplInfo::migrateReturnAndRedundant(
             true, std::vector<int>{3, 5},
             "oneapi::mkl::lapack::getrf_scratchpad_size<std::complex<float>"
             ">")},
        {"cusolverDnZgetrf_bufferSize",
         MapNamesSolver::SOLVERFuncReplInfo::migrateReturnAndRedundant(
             true, std::vector<int>{3, 5},
             "oneapi::mkl::lapack::getrf_scratchpad_size<std::complex<double>"
             ">")},
        {"cusolverDnSgeqrf_bufferSize",
         MapNamesSolver::SOLVERFuncReplInfo::migrateReturnAndRedundant(
             true, std::vector<int>{3, 5},
             "oneapi::mkl::lapack::geqrf_scratchpad_size<float>")},
        {"cusolverDnDgeqrf_bufferSize",
         MapNamesSolver::SOLVERFuncReplInfo::migrateReturnAndRedundant(
             true, std::vector<int>{3, 5},
             "oneapi::mkl::lapack::geqrf_scratchpad_size<double>")},
        {"cusolverDnCgeqrf_bufferSize",
         MapNamesSolver::SOLVERFuncReplInfo::migrateReturnAndRedundant(
             true, std::vector<int>{3, 5},
             "oneapi::mkl::lapack::geqrf_scratchpad_size<std::complex<float>"
             ">")},
        {"cusolverDnZgeqrf_bufferSize",
         MapNamesSolver::SOLVERFuncReplInfo::migrateReturnAndRedundant(
             true, std::vector<int>{3, 5},
             "oneapi::mkl::lapack::geqrf_scratchpad_size<std::complex<double>"
             ">")},
        {"cusolverDnSormqr_bufferSize",
         MapNamesSolver::SOLVERFuncReplInfo::migrateReturnAndRedundant(
             true, std::vector<int>{6, 8, 9, 11},
             "oneapi::mkl::lapack::ormqr_scratchpad_size<float>")},
        {"cusolverDnDormqr_bufferSize",
         MapNamesSolver::SOLVERFuncReplInfo::migrateReturnAndRedundant(
             true, std::vector<int>{6, 8, 9, 11},
             "oneapi::mkl::lapack::ormqr_scratchpad_size<double>")},
        {"cusolverDnCunmqr_bufferSize",
         MapNamesSolver::SOLVERFuncReplInfo::migrateReturnAndRedundant(
             true, std::vector<int>{6, 8, 9, 11},
             "oneapi::mkl::lapack::unmqr_scratchpad_size<std::complex<float>"
             ">")},
        {"cusolverDnZunmqr_bufferSize",
         MapNamesSolver::SOLVERFuncReplInfo::migrateReturnAndRedundant(
             true, std::vector<int>{6, 8, 9, 11},
             "oneapi::mkl::lapack::unmqr_scratchpad_size<std::complex<double>"
             ">")},
        {"cusolverDnSorgqr_bufferSize",
         MapNamesSolver::SOLVERFuncReplInfo::migrateReturnAndRedundant(
             true, std::vector<int>{4, 6, 7},
             "oneapi::mkl::lapack::orgqr_scratchpad_size<float>")},
        {"cusolverDnDorgqr_bufferSize",
         MapNamesSolver::SOLVERFuncReplInfo::migrateReturnAndRedundant(
             true, std::vector<int>{4, 6, 7},
             "oneapi::mkl::lapack::orgqr_scratchpad_size<double>")},
        {"cusolverDnCungqr_bufferSize",
         MapNamesSolver::SOLVERFuncReplInfo::migrateReturnAndRedundant(
             true, std::vector<int>{4, 6, 7},
             "oneapi::mkl::lapack::ungqr_scratchpad_size<std::complex<float>"
             ">")},
        {"cusolverDnZungqr_bufferSize",
         MapNamesSolver::SOLVERFuncReplInfo::migrateReturnAndRedundant(
             true, std::vector<int>{4, 6, 7},
             "oneapi::mkl::lapack::ungqr_scratchpad_size<std::complex<double>"
             ">")},
        {"cusolverDnSsytrd_bufferSize",
         MapNamesSolver::SOLVERFuncReplInfo::migrateReturnAndRedundant(
             true, std::vector<int>{3, 5, 6, 7, 8},
             "oneapi::mkl::lapack::sytrd_scratchpad_size<float>")},
        {"cusolverDnDsytrd_bufferSize",
         MapNamesSolver::SOLVERFuncReplInfo::migrateReturnAndRedundant(
             true, std::vector<int>{3, 5, 6, 7, 8},
             "oneapi::mkl::lapack::sytrd_scratchpad_size<double>")},
        {"cusolverDnChetrd_bufferSize",
         MapNamesSolver::SOLVERFuncReplInfo::migrateReturnAndRedundant(
             true, std::vector<int>{3, 5, 6, 7, 8},
             "oneapi::mkl::lapack::hetrd_scratchpad_size<std::complex<float>"
             ">")},
        {"cusolverDnZhetrd_bufferSize",
         MapNamesSolver::SOLVERFuncReplInfo::migrateReturnAndRedundant(
             true, std::vector<int>{3, 5, 6, 7, 8},
             "oneapi::mkl::lapack::hetrd_scratchpad_size<std::complex<double>"
             ">")},
        {"cusolverDnSsytrf_bufferSize",
         MapNamesSolver::SOLVERFuncReplInfo::migrateReturnRedundantAndMissed(
             true, std::vector<int>{2, 4}, std::vector<int>{1},
             std::vector<int>{1}, std::vector<bool>{false},
             std::vector<std::string>{"oneapi::mkl::uplo"},
             std::vector<std::string>{"uplo_ct_mkl_upper_lower"},
             "oneapi::mkl::lapack::sytrf_scratchpad_size<float>")},
        {"cusolverDnDsytrf_bufferSize",
         MapNamesSolver::SOLVERFuncReplInfo::migrateReturnRedundantAndMissed(
             true, std::vector<int>{2, 4}, std::vector<int>{1},
             std::vector<int>{1}, std::vector<bool>{false},
             std::vector<std::string>{"oneapi::mkl::uplo"},
             std::vector<std::string>{"uplo_ct_mkl_upper_lower"},
             "oneapi::mkl::lapack::sytrf_scratchpad_size<double>")},
        {"cusolverDnCsytrf_bufferSize",
         MapNamesSolver::SOLVERFuncReplInfo::migrateReturnRedundantAndMissed(
             true, std::vector<int>{2, 4}, std::vector<int>{1},
             std::vector<int>{1}, std::vector<bool>{false},
             std::vector<std::string>{"oneapi::mkl::uplo"},
             std::vector<std::string>{"uplo_ct_mkl_upper_lower"},
             "oneapi::mkl::lapack::sytrf_scratchpad_size<std::complex<float>"
             ">")},
        {"cusolverDnZsytrf_bufferSize",
         MapNamesSolver::SOLVERFuncReplInfo::migrateReturnRedundantAndMissed(
             true, std::vector<int>{2, 4}, std::vector<int>{1},
             std::vector<int>{1}, std::vector<bool>{false},
             std::vector<std::string>{"oneapi::mkl::uplo"},
             std::vector<std::string>{"uplo_ct_mkl_upper_lower"},
             "oneapi::mkl::lapack::sytrf_scratchpad_size<std::complex<double>"
             ">")},
        {"cusolverDnSgebrd_bufferSize",
         MapNamesSolver::SOLVERFuncReplInfo::migrateReturnRedundantAndMissed(
             true, std::vector<int>{3}, std::vector<int>{3},
             std::vector<int>{3}, std::vector<bool>{false},
             std::vector<std::string>{"std::int64_t"},
             std::vector<std::string>{"lda_ct"},
             "oneapi::mkl::lapack::gebrd_scratchpad_size<float>")},
        {"cusolverDnDgebrd_bufferSize",
         MapNamesSolver::SOLVERFuncReplInfo::migrateReturnRedundantAndMissed(
             true, std::vector<int>{3}, std::vector<int>{3},
             std::vector<int>{3}, std::vector<bool>{false},
             std::vector<std::string>{"std::int64_t"},
             std::vector<std::string>{"lda_ct"},
             "oneapi::mkl::lapack::gebrd_scratchpad_size<double>")},
        {"cusolverDnCgebrd_bufferSize",
         MapNamesSolver::SOLVERFuncReplInfo::migrateReturnRedundantAndMissed(
             true, std::vector<int>{3}, std::vector<int>{3},
             std::vector<int>{3}, std::vector<bool>{false},
             std::vector<std::string>{"std::int64_t"},
             std::vector<std::string>{"lda_ct"},
             "oneapi::mkl::lapack::gebrd_scratchpad_size<std::complex<float>"
             ">")},
        {"cusolverDnZgebrd_bufferSize",
         MapNamesSolver::SOLVERFuncReplInfo::migrateReturnRedundantAndMissed(
             true, std::vector<int>{3}, std::vector<int>{3},
             std::vector<int>{3}, std::vector<bool>{false},
             std::vector<std::string>{"std::int64_t"},
             std::vector<std::string>{"lda_ct"},
             "oneapi::mkl::lapack::gebrd_scratchpad_size<std::complex<double>"
             ">")},
        {"cusolverDnSorgbr_bufferSize",
         MapNamesSolver::SOLVERFuncReplInfo::migrateReturnRedundantAndCast(
             true, std::vector<int>{5, 7, 8}, std::vector<int>{1},
             std::vector<std::string>{"oneapi::mkl::generate"},
             "oneapi::mkl::lapack::orgbr_scratchpad_size<float>")},
        {"cusolverDnDorgbr_bufferSize",
         MapNamesSolver::SOLVERFuncReplInfo::migrateReturnRedundantAndCast(
             true, std::vector<int>{5, 7, 8}, std::vector<int>{1},
             std::vector<std::string>{"oneapi::mkl::generate"},
             "oneapi::mkl::lapack::orgbr_scratchpad_size<double>")},
        {"cusolverDnCungbr_bufferSize",
         MapNamesSolver::SOLVERFuncReplInfo::migrateReturnRedundantAndCast(
             true, std::vector<int>{5, 7, 8}, std::vector<int>{1},
             std::vector<std::string>{"oneapi::mkl::generate"},
             "oneapi::mkl::lapack::ungbr_scratchpad_size<std::complex<float>"
             ">")},
        {"cusolverDnZungbr_bufferSize",
         MapNamesSolver::SOLVERFuncReplInfo::migrateReturnRedundantAndCast(
             true, std::vector<int>{5, 7, 8}, std::vector<int>{1},
             std::vector<std::string>{"oneapi::mkl::generate"},
             "oneapi::mkl::lapack::ungbr_scratchpad_size<std::complex<double>"
             ">")},
        {"cusolverDnSormtr_bufferSize",
         MapNamesSolver::SOLVERFuncReplInfo::migrateReturnAndRedundant(
             true, std::vector<int>{6, 8, 9, 11},
             "oneapi::mkl::lapack::ormtr_scratchpad_size<float>")},
        {"cusolverDnDormtr_bufferSize",
         MapNamesSolver::SOLVERFuncReplInfo::migrateReturnAndRedundant(
             true, std::vector<int>{6, 8, 9, 11},
             "oneapi::mkl::lapack::ormtr_scratchpad_size<double>")},
        {"cusolverDnCunmtr_bufferSize",
         MapNamesSolver::SOLVERFuncReplInfo::migrateReturnAndRedundant(
             true, std::vector<int>{6, 8, 9, 11},
             "oneapi::mkl::lapack::unmtr_scratchpad_size<std::complex<float>"
             ">")},
        {"cusolverDnZunmtr_bufferSize",
         MapNamesSolver::SOLVERFuncReplInfo::migrateReturnAndRedundant(
             true, std::vector<int>{6, 8, 9, 11},
             "oneapi::mkl::lapack::unmtr_scratchpad_size<std::complex<double>"
             ">")},
        {"cusolverDnSorgtr_bufferSize",
         MapNamesSolver::SOLVERFuncReplInfo::migrateReturnAndRedundant(
             true, std::vector<int>{3, 5, 6},
             "oneapi::mkl::lapack::orgtr_scratchpad_size<float>")},
        {"cusolverDnDorgtr_bufferSize",
         MapNamesSolver::SOLVERFuncReplInfo::migrateReturnAndRedundant(
             true, std::vector<int>{3, 5, 6},
             "oneapi::mkl::lapack::orgtr_scratchpad_size<double>")},
        {"cusolverDnCungtr_bufferSize",
         MapNamesSolver::SOLVERFuncReplInfo::migrateReturnAndRedundant(
             true, std::vector<int>{3, 5, 6},
             "oneapi::mkl::lapack::ungtr_scratchpad_size<std::complex<float>"
             ">")},
        {"cusolverDnZungtr_bufferSize",
         MapNamesSolver::SOLVERFuncReplInfo::migrateReturnAndRedundant(
             true, std::vector<int>{3, 5, 6},
             "oneapi::mkl::lapack::ungtr_scratchpad_size<std::complex<double>"
             ">")},
        {"cusolverDnSgesvd_bufferSize",
         MapNamesSolver::SOLVERFuncReplInfo::
             migrateReturnCopyRedundantAndMissed(
                 true, std::vector<int>{3}, std::vector<int>{1, 1, 2},
                 std::vector<int>{3, 3, 3}, std::vector<int>{1, 2},
                 std::vector<int>{1, 1}, std::vector<bool>{false, false},
                 std::vector<std::string>{"oneapi::mkl::jobsvd",
                                          "oneapi::mkl::jobsvd"},
                 std::vector<std::string>{"job_ct_mkl_jobu",
                                          "job_ct_mkl_jobvt"},
                 "oneapi::mkl::lapack::gesvd_scratchpad_size<float>")},
        {"cusolverDnDgesvd_bufferSize",
         MapNamesSolver::SOLVERFuncReplInfo::
             migrateReturnCopyRedundantAndMissed(
                 true, std::vector<int>{3}, std::vector<int>{1, 1, 2},
                 std::vector<int>{3, 3, 3}, std::vector<int>{1, 2},
                 std::vector<int>{1, 1}, std::vector<bool>{false, false},
                 std::vector<std::string>{"oneapi::mkl::jobsvd",
                                          "oneapi::mkl::jobsvd"},
                 std::vector<std::string>{"job_ct_mkl_jobu",
                                          "job_ct_mkl_jobvt"},
                 "oneapi::mkl::lapack::gesvd_scratchpad_size<double>")},
        {"cusolverDnCgesvd_bufferSize",
         MapNamesSolver::SOLVERFuncReplInfo::
             migrateReturnCopyRedundantAndMissed(
                 true, std::vector<int>{3}, std::vector<int>{1, 1, 2},
                 std::vector<int>{3, 3, 3}, std::vector<int>{1, 2},
                 std::vector<int>{1, 1}, std::vector<bool>{false, false},
                 std::vector<std::string>{"oneapi::mkl::jobsvd",
                                          "oneapi::mkl::jobsvd"},
                 std::vector<std::string>{"job_ct_mkl_jobu",
                                          "job_ct_mkl_jobvt"},
                 "oneapi::mkl::lapack::gesvd_scratchpad_size<std::complex<"
                 "float>"
                 ">")},
        {"cusolverDnZgesvd_bufferSize",
         MapNamesSolver::SOLVERFuncReplInfo::
             migrateReturnCopyRedundantAndMissed(
                 true, std::vector<int>{3}, std::vector<int>{1, 1, 2},
                 std::vector<int>{3, 3, 3}, std::vector<int>{1, 2},
                 std::vector<int>{1, 1}, std::vector<bool>{false, false},
                 std::vector<std::string>{"oneapi::mkl::jobsvd",
                                          "oneapi::mkl::jobsvd"},
                 std::vector<std::string>{"job_ct_mkl_jobu",
                                          "job_ct_mkl_jobvt"},
                 "oneapi::mkl::lapack::gesvd_scratchpad_size<std::complex<"
                 "double>"
                 ">")},
        {"cusolverDnSpotri_bufferSize",
         MapNamesSolver::SOLVERFuncReplInfo::migrateReturnAndRedundant(
             true, std::vector<int>{3, 5},
             "oneapi::mkl::lapack::potri_scratchpad_size<float>")},
        {"cusolverDnDpotri_bufferSize",
         MapNamesSolver::SOLVERFuncReplInfo::migrateReturnAndRedundant(
             true, std::vector<int>{3, 5},
             "oneapi::mkl::lapack::potri_scratchpad_size<double>")},
        {"cusolverDnCpotri_bufferSize",
         MapNamesSolver::SOLVERFuncReplInfo::migrateReturnAndRedundant(
             true, std::vector<int>{3, 5},
             "oneapi::mkl::lapack::potri_scratchpad_size<std::complex<float>"
             ">")},
        {"cusolverDnZpotri_bufferSize",
         MapNamesSolver::SOLVERFuncReplInfo::migrateReturnAndRedundant(
             true, std::vector<int>{3, 5},
             "oneapi::mkl::lapack::potri_scratchpad_size<std::complex<double>"
             ">")},
        {"cusolverDnSpotrf",
         MapNamesSolver::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{3, 5}, std::vector<std::string>{"float", "float"},
             std::vector<int>{7}, "oneapi::mkl::lapack::potrf")},
        {"cusolverDnDpotrf",
         MapNamesSolver::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{3, 5},
             std::vector<std::string>{"double", "double"}, std::vector<int>{7},
             "oneapi::mkl::lapack::potrf")},
        {"cusolverDnCpotrf",
         MapNamesSolver::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{3, 5},
             std::vector<std::string>{"std::complex<float>",
                                      "std::complex<float>"},
             std::vector<int>{7}, "oneapi::mkl::lapack::potrf")},
        {"cusolverDnZpotrf",
         MapNamesSolver::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{3, 5},
             std::vector<std::string>{"std::complex<double>",
                                      "std::complex<double>"},
             std::vector<int>{7}, "oneapi::mkl::lapack::potrf")},
        {"cusolverDnSpotrs",
         MapNamesSolver::SOLVERFuncReplInfo::migrateBufferRedundantAndWS(
             std::vector<int>{4, 6}, std::vector<std::string>{"float", "float"},
             std::vector<int>{8}, std::vector<int>{7},
             std::vector<int>{0, 1, 2, 3, 5, 7},
             "oneapi::mkl::lapack::potrs_scratchpad_size<float>",
             "oneapi::mkl::lapack::potrs")},
        {"cusolverDnDpotrs",
         MapNamesSolver::SOLVERFuncReplInfo::migrateBufferRedundantAndWS(
             std::vector<int>{4, 6},
             std::vector<std::string>{"double", "double"}, std::vector<int>{8},
             std::vector<int>{7}, std::vector<int>{0, 1, 2, 3, 5, 7},
             "oneapi::mkl::lapack::potrs_scratchpad_size<double>",
             "oneapi::mkl::lapack::potrs")},
        {"cusolverDnCpotrs",
         MapNamesSolver::SOLVERFuncReplInfo::migrateBufferRedundantAndWS(
             std::vector<int>{4, 6},
             std::vector<std::string>{"std::complex<float>",
                                      "std::complex<float>"},
             std::vector<int>{8}, std::vector<int>{7},
             std::vector<int>{0, 1, 2, 3, 5, 7},
             "oneapi::mkl::lapack::potrs_scratchpad_size<std::complex<float>>",
             "oneapi::mkl::lapack::potrs")},
        {"cusolverDnZpotrs",
         MapNamesSolver::SOLVERFuncReplInfo::migrateBufferRedundantAndWS(
             std::vector<int>{4, 6},
             std::vector<std::string>{"std::complex<double>",
                                      "std::complex<double>"},
             std::vector<int>{8}, std::vector<int>{7},
             std::vector<int>{0, 1, 2, 3, 5, 7},
             "oneapi::mkl::lapack::potrs_scratchpad_size<std::complex<double>>",
             "oneapi::mkl::lapack::potrs")},
        {"cusolverDnSpotri",
         MapNamesSolver::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{3, 5}, std::vector<std::string>{"float", "float"},
             std::vector<int>{7}, "oneapi::mkl::lapack::potri")},
        {"cusolverDnDpotri",
         MapNamesSolver::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{3, 5},
             std::vector<std::string>{"double", "double"}, std::vector<int>{7},
             "oneapi::mkl::lapack::potri")},
        {"cusolverDnCpotri",
         MapNamesSolver::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{3, 5},
             std::vector<std::string>{"std::complex<float>",
                                      "std::complex<float>"},
             std::vector<int>{7}, "oneapi::mkl::lapack::potri")},
        {"cusolverDnZpotri",
         MapNamesSolver::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{3, 5},
             std::vector<std::string>{"std::complex<double>",
                                      "std::complex<double>"},
             std::vector<int>{7}, "oneapi::mkl::lapack::potri")},
        {"cusolverDnSgetrf",
         MapNamesSolver::SOLVERFuncReplInfo::migrateBufferMoveRedundantAndWSS(
             std::vector<int>{3, 5, 6},
             std::vector<std::string>{"float", "float", "int"},
             std::vector<int>{7}, std::vector<int>{6}, std::vector<int>{5},
             std::vector<int>{5}, std::vector<int>{0, 1, 2, 4},
             "oneapi::mkl::lapack::getrf_scratchpad_size<float>",
             "oneapi::mkl::lapack::getrf")},
        {"cusolverDnDgetrf",
         MapNamesSolver::SOLVERFuncReplInfo::migrateBufferMoveRedundantAndWSS(
             std::vector<int>{3, 5, 6},
             std::vector<std::string>{"double", "double", "int"},
             std::vector<int>{7}, std::vector<int>{6}, std::vector<int>{5},
             std::vector<int>{5}, std::vector<int>{0, 1, 2, 4},
             "oneapi::mkl::lapack::getrf_scratchpad_size<double>",
             "oneapi::mkl::lapack::getrf")},
        {"cusolverDnCgetrf",
         MapNamesSolver::SOLVERFuncReplInfo::migrateBufferMoveRedundantAndWSS(
             std::vector<int>{3, 5, 6},
             std::vector<std::string>{"std::complex<float>",
                                      "std::complex<float>", "int"},
             std::vector<int>{7}, std::vector<int>{6}, std::vector<int>{5},
             std::vector<int>{5}, std::vector<int>{0, 1, 2, 4},
             "oneapi::mkl::lapack::getrf_scratchpad_size<std::complex<float>>",
             "oneapi::mkl::lapack::getrf")},
        {"cusolverDnZgetrf",
         MapNamesSolver::SOLVERFuncReplInfo::migrateBufferMoveRedundantAndWSS(
             std::vector<int>{3, 5, 6},
             std::vector<std::string>{"std::complex<double>",
                                      "std::complex<double>", "int"},
             std::vector<int>{7}, std::vector<int>{6}, std::vector<int>{5},
             std::vector<int>{5}, std::vector<int>{0, 1, 2, 4},
             "oneapi::mkl::lapack::getrf_scratchpad_size<std::complex<double>>",
             "oneapi::mkl::lapack::getrf")},
        {"cusolverDnSgetrs",
         MapNamesSolver::SOLVERFuncReplInfo::migrateBufferRedundantAndWS(
             std::vector<int>{4, 6, 7},
             std::vector<std::string>{"float", "int", "float"},
             std::vector<int>{9}, std::vector<int>{8},
             std::vector<int>{0, 1, 2, 3, 5, 8},
             "oneapi::mkl::lapack::getrs_scratchpad_size<float>",
             "oneapi::mkl::lapack::getrs")},
        {"cusolverDnDgetrs",
         MapNamesSolver::SOLVERFuncReplInfo::migrateBufferRedundantAndWS(
             std::vector<int>{4, 6, 7},
             std::vector<std::string>{"double", "int", "double"},
             std::vector<int>{9}, std::vector<int>{8},
             std::vector<int>{0, 1, 2, 3, 5, 8},
             "oneapi::mkl::lapack::getrs_scratchpad_size<double>",
             "oneapi::mkl::lapack::getrs")},
        {"cusolverDnCgetrs",
         MapNamesSolver::SOLVERFuncReplInfo::migrateBufferRedundantAndWS(
             std::vector<int>{4, 6, 7},
             std::vector<std::string>{"std::complex<float>", "int",
                                      "std::complex<float>"},
             std::vector<int>{9}, std::vector<int>{8},
             std::vector<int>{0, 1, 2, 3, 5, 8},
             "oneapi::mkl::lapack::getrs_scratchpad_size<std::complex<float>>",
             "oneapi::mkl::lapack::getrs")},
        {"cusolverDnZgetrs",
         MapNamesSolver::SOLVERFuncReplInfo::migrateBufferRedundantAndWS(
             std::vector<int>{4, 6, 7},
             std::vector<std::string>{"std::complex<double>", "int",
                                      "std::complex<double>"},
             std::vector<int>{9}, std::vector<int>{8},
             std::vector<int>{0, 1, 2, 3, 5, 8},
             "oneapi::mkl::lapack::getrs_scratchpad_size<std::complex<double>>",
             "oneapi::mkl::lapack::getrs")},
        {"cusolverDnSgeqrf",
         MapNamesSolver::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{3, 5, 6},
             std::vector<std::string>{"float", "float", "float", "int"},
             std::vector<int>{8}, "oneapi::mkl::lapack::geqrf")},
        {"cusolverDnDgeqrf",
         MapNamesSolver::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{3, 5, 6},
             std::vector<std::string>{"double", "double", "double", "int"},
             std::vector<int>{8}, "oneapi::mkl::lapack::geqrf")},
        {"cusolverDnCgeqrf",
         MapNamesSolver::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{3, 5, 6},
             std::vector<std::string>{"std::complex<float>",
                                      "std::complex<float>",
                                      "std::complex<float>", "int"},
             std::vector<int>{8}, "oneapi::mkl::lapack::geqrf")},
        {"cusolverDnZgeqrf",
         MapNamesSolver::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{3, 5, 6},
             std::vector<std::string>{"std::complex<double>",
                                      "std::complex<double>",
                                      "std::complex<double>", "int"},
             std::vector<int>{8}, "oneapi::mkl::lapack::geqrf")},
        {"cusolverDnSormqr",
         MapNamesSolver::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{6, 8, 9, 11},
             std::vector<std::string>{"float", "float", "float", "float"},
             std::vector<int>{13}, "oneapi::mkl::lapack::ormqr")},
        {"cusolverDnDormqr",
         MapNamesSolver::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{6, 8, 9, 11},
             std::vector<std::string>{
                 "double",
                 "double",
                 "double",
                 "double",
             },
             std::vector<int>{13}, "oneapi::mkl::lapack::ormqr")},
        {"cusolverDnCunmqr",
         MapNamesSolver::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{6, 8, 9, 11},
             std::vector<std::string>{
                 "std::complex<float>", "std::complex<float>",
                 "std::complex<float>", "std::complex<float>"},
             std::vector<int>{13}, "oneapi::mkl::lapack::unmqr")},
        {"cusolverDnZunmqr",
         MapNamesSolver::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{6, 8, 9, 11},
             std::vector<std::string>{
                 "std::complex<double>", "std::complex<double>",
                 "std::complex<double>", "std::complex<double>"},
             std::vector<int>{13}, "oneapi::mkl::lapack::unmqr")},
        {"cusolverDnSorgqr",
         MapNamesSolver::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{4, 6, 7},
             std::vector<std::string>{"float", "float", "float"},
             std::vector<int>{9}, "oneapi::mkl::lapack::orgqr")},
        {"cusolverDnDorgqr",
         MapNamesSolver::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{4, 6, 7},
             std::vector<std::string>{"double", "double", "double"},
             std::vector<int>{9}, "oneapi::mkl::lapack::orgqr")},
        {"cusolverDnCungqr",
         MapNamesSolver::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{4, 6, 7},
             std::vector<std::string>{"std::complex<float>",
                                      "std::complex<float>",
                                      "std::complex<float>"},
             std::vector<int>{9}, "oneapi::mkl::lapack::ungqr")},
        {"cusolverDnZungqr",
         MapNamesSolver::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{4, 6, 7},
             std::vector<std::string>{"std::complex<double>",
                                      "std::complex<double>",
                                      "std::complex<double>"},
             std::vector<int>{9}, "oneapi::mkl::lapack::ungqr")},
        {"cusolverDnSsytrf",
         MapNamesSolver::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{3, 5, 6},
             std::vector<std::string>{"float", "int", "float"},
             std::vector<int>{8}, "oneapi::mkl::lapack::sytrf")},
        {"cusolverDnDsytrf",
         MapNamesSolver::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{3, 5, 6},
             std::vector<std::string>{"double", "int", "double"},
             std::vector<int>{8}, "oneapi::mkl::lapack::sytrf")},
        {"cusolverDnCsytrf",
         MapNamesSolver::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{3, 5, 6},
             std::vector<std::string>{"std::complex<float>", "int",
                                      "std::complex<float>"},
             std::vector<int>{8}, "oneapi::mkl::lapack::sytrf")},
        {"cusolverDnZsytrf",
         MapNamesSolver::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{3, 5, 6},
             std::vector<std::string>{"std::complex<double>", "int",
                                      "std::complex<double>"},
             std::vector<int>{8}, "oneapi::mkl::lapack::sytrf")},
        {"cusolverDnSgebrd",
         MapNamesSolver::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{3, 5, 6, 7, 8, 9},
             std::vector<std::string>{"float", "float", "float", "float",
                                      "float", "float"},
             std::vector<int>{11}, "oneapi::mkl::lapack::gebrd")},
        {"cusolverDnDgebrd",
         MapNamesSolver::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{3, 5, 6, 7, 8, 9},
             std::vector<std::string>{"double", "double", "double", "double",
                                      "double", "double"},
             std::vector<int>{11}, "oneapi::mkl::lapack::gebrd")},
        {"cusolverDnCgebrd",
         MapNamesSolver::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{3, 5, 6, 7, 8, 9},
             std::vector<std::string>{
                 "std::complex<float>", "float", "float", "std::complex<float>",
                 "std::complex<float>", "std::complex<float>"},
             std::vector<int>{11}, "oneapi::mkl::lapack::gebrd")},
        {"cusolverDnZgebrd",
         MapNamesSolver::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{3, 5, 6, 7, 8, 9},
             std::vector<std::string>{"std::complex<double>", "double",
                                      "double", "std::complex<double>",
                                      "std::complex<double>",
                                      "std::complex<double>"},
             std::vector<int>{11}, "oneapi::mkl::lapack::gebrd")},
        {"cusolverDnSorgbr",
         MapNamesSolver::SOLVERFuncReplInfo::migrateBufferRedundantAndCast(
             std::vector<int>{5, 7, 8},
             std::vector<std::string>{"float", "float", "float"},
             std::vector<int>{10}, std::vector<int>{1},
             std::vector<std::string>{"oneapi::mkl::generate"},
             "oneapi::mkl::lapack::orgbr")},
        {"cusolverDnDorgbr",
         MapNamesSolver::SOLVERFuncReplInfo::migrateBufferRedundantAndCast(
             std::vector<int>{5, 7, 8},
             std::vector<std::string>{"double", "double", "double"},
             std::vector<int>{10}, std::vector<int>{1},
             std::vector<std::string>{"oneapi::mkl::generate"},
             "oneapi::mkl::lapack::orgbr")},
        {"cusolverDnCungbr",
         MapNamesSolver::SOLVERFuncReplInfo::migrateBufferRedundantAndCast(
             std::vector<int>{5, 7, 8},
             std::vector<std::string>{"std::complex<float>",
                                      "std::complex<float>",
                                      "std::complex<float>"},
             std::vector<int>{10}, std::vector<int>{1},
             std::vector<std::string>{"oneapi::mkl::generate"},
             "oneapi::mkl::lapack::ungbr")},
        {"cusolverDnZungbr",
         MapNamesSolver::SOLVERFuncReplInfo::migrateBufferRedundantAndCast(
             std::vector<int>{5, 7, 8},
             std::vector<std::string>{"std::complex<double>",
                                      "std::complex<double>",
                                      "std::complex<double>"},
             std::vector<int>{10}, std::vector<int>{1},
             std::vector<std::string>{"oneapi::mkl::generate"},
             "oneapi::mkl::lapack::ungbr")},
        {"cusolverDnSsytrd",
         MapNamesSolver::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{3, 5, 6, 7, 8},
             std::vector<std::string>{"float", "float", "float", "float",
                                      "float"},
             std::vector<int>{10}, "oneapi::mkl::lapack::sytrd")},
        {"cusolverDnDsytrd",
         MapNamesSolver::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{3, 5, 6, 7, 8},
             std::vector<std::string>{"double", "double", "double", "double",
                                      "double"},
             std::vector<int>{10}, "oneapi::mkl::lapack::sytrd")},
        {"cusolverDnChetrd",
         MapNamesSolver::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{3, 5, 6, 7, 8},
             std::vector<std::string>{"std::complex<float>", "float", "float",
                                      "std::complex<float>",
                                      "std::complex<float>"},
             std::vector<int>{10}, "oneapi::mkl::lapack::hetrd")},
        {"cusolverDnZhetrd",
         MapNamesSolver::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{3, 5, 6, 7, 8},
             std::vector<std::string>{"std::complex<double>", "double",
                                      "double", "std::complex<double>",
                                      "std::complex<double>"},
             std::vector<int>{10}, "oneapi::mkl::lapack::hetrd")},
        {"cusolverDnSormtr",
         MapNamesSolver::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{6, 8, 9, 11},
             std::vector<std::string>{"float", "float", "float", "float"},
             std::vector<int>{13}, "oneapi::mkl::lapack::ormtr")},
        {"cusolverDnDormtr",
         MapNamesSolver::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{6, 8, 9, 11},
             std::vector<std::string>{"double", "double", "double", "double"},
             std::vector<int>{13}, "oneapi::mkl::lapack::ormtr")},
        {"cusolverDnCunmtr",
         MapNamesSolver::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{6, 8, 9, 11},
             std::vector<std::string>{
                 "std::complex<float>", "std::complex<float>",
                 "std::complex<float>", "std::complex<float>"},
             std::vector<int>{13}, "oneapi::mkl::lapack::unmtr")},
        {"cusolverDnZunmtr",
         MapNamesSolver::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{6, 8, 9, 11},
             std::vector<std::string>{
                 "std::complex<double>", "std::complex<double>",
                 "std::complex<double>", "std::complex<double>"},
             std::vector<int>{13}, "oneapi::mkl::lapack::unmtr")},
        {"cusolverDnSorgtr",
         MapNamesSolver::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{3, 5, 6},
             std::vector<std::string>{"float", "float", "float"},
             std::vector<int>{8}, "oneapi::mkl::lapack::orgtr")},
        {"cusolverDnDorgtr",
         MapNamesSolver::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{3, 5, 6},
             std::vector<std::string>{"double", "double", "double"},
             std::vector<int>{8}, "oneapi::mkl::lapack::orgtr")},
        {"cusolverDnCungtr",
         MapNamesSolver::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{3, 5, 6},
             std::vector<std::string>{"std::complex<float>",
                                      "std::complex<float>",
                                      "std::complex<float>"},
             std::vector<int>{8}, "oneapi::mkl::lapack::ungtr")},
        {"cusolverDnZungtr",
         MapNamesSolver::SOLVERFuncReplInfo::migrateBufferAndRedundant(
             std::vector<int>{3, 5, 6},
             std::vector<std::string>{"std::complex<double>",
                                      "std::complex<double>",
                                      "std::complex<double>"},
             std::vector<int>{8}, "oneapi::mkl::lapack::ungtr")},
        {"cusolverDnSgesvd",
         MapNamesSolver::SOLVERFuncReplInfo::migrateBufferRedundantAndCast(
             std::vector<int>{5, 7, 8, 10, 12},
             std::vector<std::string>{"float", "float", "float", "float",
                                      "float"},
             std::vector<int>{14, 15}, std::vector<int>{1, 2},
             std::vector<std::string>{"oneapi::mkl::jobsvd",
                                      "oneapi::mkl::jobsvd"},
             "oneapi::mkl::lapack::gesvd")},
        {"cusolverDnDgesvd",
         MapNamesSolver::SOLVERFuncReplInfo::migrateBufferRedundantAndCast(
             std::vector<int>{5, 7, 8, 10, 12},
             std::vector<std::string>{"double", "double", "double", "double",
                                      "double"},
             std::vector<int>{14, 15}, std::vector<int>{1, 2},
             std::vector<std::string>{"oneapi::mkl::jobsvd",
                                      "oneapi::mkl::jobsvd"},
             "oneapi::mkl::lapack::gesvd")},
        {"cusolverDnCgesvd",
         MapNamesSolver::SOLVERFuncReplInfo::migrateBufferRedundantAndCast(
             std::vector<int>{5, 7, 8, 10, 12},
             std::vector<std::string>{
                 "std::complex<float>", "float", "std::complex<float>",
                 "std::complex<float>", "std::complex<float>"},
             std::vector<int>{14, 15}, std::vector<int>{1, 2},
             std::vector<std::string>{"oneapi::mkl::jobsvd",
                                      "oneapi::mkl::jobsvd"},
             "oneapi::mkl::lapack::gesvd")},
        {"cusolverDnZgesvd",
         MapNamesSolver::SOLVERFuncReplInfo::migrateBufferRedundantAndCast(
             std::vector<int>{5, 7, 8, 10, 12},
             std::vector<std::string>{
                 "std::complex<double>", "double", "std::complex<double>",
                 "std::complex<double>", "std::complex<double>"},
             std::vector<int>{14, 15}, std::vector<int>{1, 2},
             std::vector<std::string>{"oneapi::mkl::jobsvd",
                                      "oneapi::mkl::jobsvd"},
             "oneapi::mkl::lapack::gesvd")},
    };

} // namespace dpct
} // namespace clang