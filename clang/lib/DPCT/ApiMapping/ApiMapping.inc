//===--------------- ApiMapping.inc ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

REGISTER_ENTRY("cudaStreamSynchronize", "sycl::queue::wait",
"CUDA API\n"
"========\n"
"\n"
"   cudaError_t cudaStreamSynchronize(...)\n"
"\n"
"\n"
"SYCL API\n"
"============\n"
"\n"
"   void sycl::queue::wait(...)\n")

REGISTER_ENTRY("cudaEventSynchronize", "sycl::event::wait",
"CUDA API\n"
"========\n"
"\n"
"   cudaError_t cudaEventSynchronize(...)\n"
"\n"
"\n"
"SYCL API\n"
"============\n"
"\n"
"   void sycl::event::wait(...)\n")
