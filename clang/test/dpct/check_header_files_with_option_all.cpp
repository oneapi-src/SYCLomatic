// RUN: dpct -out-root %T/check_header_files_with_option_all %s --cuda-include-path="%cuda-path/include" --use-custom-helper=all -- -x cuda --cuda-host-only

// RUN: echo "begin" > %T/check_header_files_with_option_all/diff_res.txt
// RUN: diff %S/helper_files_ref/include/atomic.hpp  %T/check_header_files_with_option_all/include/dpct/atomic.hpp >> %T/check_header_files_with_option_all/diff_res.txt
// RUN: echo "end" >> %T/check_header_files_with_option_all/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files_with_option_all/diff_res.txt

// RUN: echo "begin" > %T/check_header_files_with_option_all/diff_res.txt
// RUN: diff %S/helper_files_ref/include/blas_utils.hpp  %T/check_header_files_with_option_all/include/dpct/blas_utils.hpp >> %T/check_header_files_with_option_all/diff_res.txt
// RUN: echo "end" >> %T/check_header_files_with_option_all/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files_with_option_all/diff_res.txt

// RUN: echo "begin" > %T/check_header_files_with_option_all/diff_res.txt
// RUN: diff %S/helper_files_ref/include/device.hpp  %T/check_header_files_with_option_all/include/dpct/device.hpp >> %T/check_header_files_with_option_all/diff_res.txt
// RUN: echo "end" >> %T/check_header_files_with_option_all/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files_with_option_all/diff_res.txt

// RUN: echo "begin" > %T/check_header_files_with_option_all/diff_res.txt
// RUN: diff %S/helper_files_ref/include/dpct.hpp  %T/check_header_files_with_option_all/include/dpct/dpct.hpp >> %T/check_header_files_with_option_all/diff_res.txt
// RUN: echo "end" >> %T/check_header_files_with_option_all/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files_with_option_all/diff_res.txt

// RUN: echo "begin" > %T/check_header_files_with_option_all/diff_res.txt
// RUN: diff %S/helper_files_ref/include/dpl_utils.hpp  %T/check_header_files_with_option_all/include/dpct/dpl_utils.hpp >> %T/check_header_files_with_option_all/diff_res.txt
// RUN: echo "end" >> %T/check_header_files_with_option_all/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files_with_option_all/diff_res.txt

// RUN: echo "begin" > %T/check_header_files_with_option_all/diff_res.txt
// RUN: diff %S/helper_files_ref/include/image.hpp  %T/check_header_files_with_option_all/include/dpct/image.hpp >> %T/check_header_files_with_option_all/diff_res.txt
// RUN: echo "end" >> %T/check_header_files_with_option_all/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files_with_option_all/diff_res.txt

// RUN: echo "begin" > %T/check_header_files_with_option_all/diff_res.txt
// RUN: diff %S/helper_files_ref/include/kernel.hpp  %T/check_header_files_with_option_all/include/dpct/kernel.hpp >> %T/check_header_files_with_option_all/diff_res.txt
// RUN: echo "end" >> %T/check_header_files_with_option_all/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files_with_option_all/diff_res.txt

// RUN: echo "begin" > %T/check_header_files_with_option_all/diff_res.txt
// RUN: diff %S/helper_files_ref/include/memory.hpp  %T/check_header_files_with_option_all/include/dpct/memory.hpp >> %T/check_header_files_with_option_all/diff_res.txt
// RUN: echo "end" >> %T/check_header_files_with_option_all/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files_with_option_all/diff_res.txt

// RUN: echo "begin" > %T/check_header_files_with_option_all/diff_res.txt
// RUN: diff %S/helper_files_ref/include/util.hpp  %T/check_header_files_with_option_all/include/dpct/util.hpp >> %T/check_header_files_with_option_all/diff_res.txt
// RUN: echo "end" >> %T/check_header_files_with_option_all/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files_with_option_all/diff_res.txt

// RUN: echo "begin" > %T/check_header_files_with_option_all/diff_res.txt
// RUN: diff %S/helper_files_ref/include/rng_utils.hpp  %T/check_header_files_with_option_all/include/dpct/rng_utils.hpp >> %T/check_header_files_with_option_all/diff_res.txt
// RUN: echo "end" >> %T/check_header_files_with_option_all/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files_with_option_all/diff_res.txt

// RUN: echo "begin" > %T/check_header_files_with_option_all/diff_res.txt
// RUN: diff %S/helper_files_ref/include/lib_common_utils.hpp  %T/check_header_files_with_option_all/include/dpct/lib_common_utils.hpp >> %T/check_header_files_with_option_all/diff_res.txt
// RUN: echo "end" >> %T/check_header_files_with_option_all/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files_with_option_all/diff_res.txt

// RUN: echo "begin" > %T/check_header_files_with_option_all/diff_res.txt
// RUN: diff %S/helper_files_ref/include/ccl_utils.hpp  %T/check_header_files_with_option_all/include/dpct/ccl_utils.hpp >> %T/check_header_files_with_option_all/diff_res.txt
// RUN: echo "end" >> %T/check_header_files_with_option_all/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files_with_option_all/diff_res.txt

// RUN: echo "begin" > %T/check_header_files_with_option_all/diff_res.txt
// RUN: diff %S/helper_files_ref/include/dnnl_utils.hpp  %T/check_header_files_with_option_all/include/dpct/dnnl_utils.hpp >> %T/check_header_files_with_option_all/diff_res.txt
// RUN: echo "end" >> %T/check_header_files_with_option_all/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files_with_option_all/diff_res.txt

// RUN: echo "begin" > %T/check_header_files_with_option_all/diff_res.txt
// RUN: diff %S/helper_files_ref/include/fft_utils.hpp  %T/check_header_files_with_option_all/include/dpct/fft_utils.hpp >> %T/check_header_files_with_option_all/diff_res.txt
// RUN: echo "end" >> %T/check_header_files_with_option_all/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files_with_option_all/diff_res.txt

// RUN: echo "begin" > %T/check_header_files_with_option_all/diff_res.txt
// RUN: diff %S/helper_files_ref/include/lapack_utils.hpp  %T/check_header_files_with_option_all/include/dpct/lapack_utils.hpp >> %T/check_header_files_with_option_all/diff_res.txt
// RUN: echo "end" >> %T/check_header_files_with_option_all/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files_with_option_all/diff_res.txt

// RUN: echo "begin" > %T/check_header_files_with_option_all/diff_res.txt
// RUN: diff %S/helper_files_ref/include/dpl_extras/algorithm.h  %T/check_header_files_with_option_all/include/dpct/dpl_extras/algorithm.h >> %T/check_header_files_with_option_all/diff_res.txt
// RUN: echo "end" >> %T/check_header_files_with_option_all/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files_with_option_all/diff_res.txt

// RUN: echo "begin" > %T/check_header_files_with_option_all/diff_res.txt
// RUN: diff %S/helper_files_ref/include/dpl_extras/functional.h  %T/check_header_files_with_option_all/include/dpct/dpl_extras/functional.h >> %T/check_header_files_with_option_all/diff_res.txt
// RUN: echo "end" >> %T/check_header_files_with_option_all/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files_with_option_all/diff_res.txt

// RUN: echo "begin" > %T/check_header_files_with_option_all/diff_res.txt
// RUN: diff %S/helper_files_ref/include/dpl_extras/iterators.h  %T/check_header_files_with_option_all/include/dpct/dpl_extras/iterators.h >> %T/check_header_files_with_option_all/diff_res.txt
// RUN: echo "end" >> %T/check_header_files_with_option_all/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files_with_option_all/diff_res.txt

// RUN: echo "begin" > %T/check_header_files_with_option_all/diff_res.txt
// RUN: diff %S/helper_files_ref/include/dpl_extras/memory.h  %T/check_header_files_with_option_all/include/dpct/dpl_extras/memory.h >> %T/check_header_files_with_option_all/diff_res.txt
// RUN: echo "end" >> %T/check_header_files_with_option_all/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files_with_option_all/diff_res.txt

// RUN: echo "begin" > %T/check_header_files_with_option_all/diff_res.txt
// RUN: diff %S/helper_files_ref/include/dpl_extras/numeric.h  %T/check_header_files_with_option_all/include/dpct/dpl_extras/numeric.h >> %T/check_header_files_with_option_all/diff_res.txt
// RUN: echo "end" >> %T/check_header_files_with_option_all/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files_with_option_all/diff_res.txt

// RUN: echo "begin" > %T/check_header_files_with_option_all/diff_res.txt
// RUN: diff %S/helper_files_ref/include/dpl_extras/vector.h  %T/check_header_files_with_option_all/include/dpct/dpl_extras/vector.h >> %T/check_header_files_with_option_all/diff_res.txt
// RUN: echo "end" >> %T/check_header_files_with_option_all/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files_with_option_all/diff_res.txt

// RUN: echo "begin" > %T/check_header_files_with_option_all/diff_res.txt
// RUN: diff %S/helper_files_ref/include/dpl_extras/dpcpp_extensions.h  %T/check_header_files_with_option_all/include/dpct/dpl_extras/dpcpp_extensions.h >> %T/check_header_files_with_option_all/diff_res.txt
// RUN: echo "end" >> %T/check_header_files_with_option_all/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files_with_option_all/diff_res.txt

// RUN: rm -rf %T/check_header_files_with_option_all

// CHECK: begin
// CHECK-NEXT: end

int main() {
  float2 f2;
  return 0;
}
