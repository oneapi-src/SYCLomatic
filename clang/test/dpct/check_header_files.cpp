// RUN: mkdir %T/check_header_files
// RUN: dpct --out-root %T/check_header_files/out --gen-helper-function --cuda-include-path="%cuda-path/include" || true

// RUN: echo "begin" > %T/check_header_files/diff_res.txt
// RUN: diff %T/check_header_files/out/include/dpct/atomic.hpp  %S/../../runtime/dpct-rt/include/atomic.hpp >> %T/check_header_files/diff_res.txt
// RUN: echo "end" >> %T/check_header_files/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files/diff_res.txt

// RUN: echo "begin" > %T/check_header_files/diff_res.txt
// RUN: diff %T/check_header_files/out/include/dpct/blas_utils.hpp  %S/../../runtime/dpct-rt/include/blas_utils.hpp >> %T/check_header_files/diff_res.txt
// RUN: echo "end" >> %T/check_header_files/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files/diff_res.txt

// RUN: echo "begin" > %T/check_header_files/diff_res.txt
// RUN: diff %T/check_header_files/out/include/dpct/device.hpp  %S/../../runtime/dpct-rt/include/device.hpp >> %T/check_header_files/diff_res.txt
// RUN: echo "end" >> %T/check_header_files/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files/diff_res.txt

// RUN: echo "begin" > %T/check_header_files/diff_res.txt
// RUN: diff %T/check_header_files/out/include/dpct/dpct.hpp  %S/../../runtime/dpct-rt/include/dpct.hpp >> %T/check_header_files/diff_res.txt
// RUN: echo "end" >> %T/check_header_files/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files/diff_res.txt

// RUN: echo "begin" > %T/check_header_files/diff_res.txt
// RUN: diff %T/check_header_files/out/include/dpct/dpl_utils.hpp  %S/../../runtime/dpct-rt/include/dpl_utils.hpp >> %T/check_header_files/diff_res.txt
// RUN: echo "end" >> %T/check_header_files/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files/diff_res.txt

// RUN: echo "begin" > %T/check_header_files/diff_res.txt
// RUN: diff %T/check_header_files/out/include/dpct/image.hpp  %S/../../runtime/dpct-rt/include/image.hpp >> %T/check_header_files/diff_res.txt
// RUN: echo "end" >> %T/check_header_files/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files/diff_res.txt

// RUN: echo "begin" > %T/check_header_files/diff_res.txt
// RUN: diff %T/check_header_files/out/include/dpct/kernel.hpp  %S/../../runtime/dpct-rt/include/kernel.hpp >> %T/check_header_files/diff_res.txt
// RUN: echo "end" >> %T/check_header_files/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files/diff_res.txt

// RUN: echo "begin" > %T/check_header_files/diff_res.txt
// RUN: diff %T/check_header_files/out/include/dpct/memory.hpp  %S/../../runtime/dpct-rt/include/memory.hpp >> %T/check_header_files/diff_res.txt
// RUN: echo "end" >> %T/check_header_files/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files/diff_res.txt

// RUN: echo "begin" > %T/check_header_files/diff_res.txt
// RUN: diff %T/check_header_files/out/include/dpct/util.hpp  %S/../../runtime/dpct-rt/include/util.hpp >> %T/check_header_files/diff_res.txt
// RUN: echo "end" >> %T/check_header_files/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files/diff_res.txt

// RUN: echo "begin" > %T/check_header_files/diff_res.txt
// RUN: diff %T/check_header_files/out/include/dpct/rng_utils.hpp  %S/../../runtime/dpct-rt/include/rng_utils.hpp >> %T/check_header_files/diff_res.txt
// RUN: echo "end" >> %T/check_header_files/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files/diff_res.txt

// RUN: echo "begin" > %T/check_header_files/diff_res.txt
// RUN: diff %T/check_header_files/out/include/dpct/lib_common_utils.hpp  %S/../../runtime/dpct-rt/include/lib_common_utils.hpp >> %T/check_header_files/diff_res.txt
// RUN: echo "end" >> %T/check_header_files/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files/diff_res.txt

// RUN: echo "begin" > %T/check_header_files/diff_res.txt
// RUN: diff %T/check_header_files/out/include/dpct/ccl_utils.hpp  %S/../../runtime/dpct-rt/include/ccl_utils.hpp >> %T/check_header_files/diff_res.txt
// RUN: echo "end" >> %T/check_header_files/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files/diff_res.txt

// RUN: echo "begin" > %T/check_header_files/diff_res.txt
// RUN: diff %T/check_header_files/out/include/dpct/dnnl_utils.hpp  %S/../../runtime/dpct-rt/include/dnnl_utils.hpp >> %T/check_header_files/diff_res.txt
// RUN: echo "end" >> %T/check_header_files/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files/diff_res.txt

// RUN: echo "begin" > %T/check_header_files/diff_res.txt
// RUN: diff %T/check_header_files/out/include/dpct/sparse_utils.hpp  %S/../../runtime/dpct-rt/include/sparse_utils.hpp >> %T/check_header_files/diff_res.txt
// RUN: echo "end" >> %T/check_header_files/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files/diff_res.txt

// RUN: echo "begin" > %T/check_header_files/diff_res.txt
// RUN: diff %T/check_header_files/out/include/dpct/fft_utils.hpp  %S/../../runtime/dpct-rt/include/fft_utils.hpp >> %T/check_header_files/diff_res.txt
// RUN: echo "end" >> %T/check_header_files/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files/diff_res.txt

// RUN: echo "begin" > %T/check_header_files/diff_res.txt
// RUN: diff %T/check_header_files/out/include/dpct/lapack_utils.hpp  %S/../../runtime/dpct-rt/include/lapack_utils.hpp >> %T/check_header_files/diff_res.txt
// RUN: echo "end" >> %T/check_header_files/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files/diff_res.txt

// RUN: echo "begin" > %T/check_header_files/diff_res.txt
// RUN: diff %T/check_header_files/out/include/dpct/dpl_extras/algorithm.h  %S/../../runtime/dpct-rt/include/dpl_extras/algorithm.h >> %T/check_header_files/diff_res.txt
// RUN: echo "end" >> %T/check_header_files/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files/diff_res.txt

// RUN: echo "begin" > %T/check_header_files/diff_res.txt
// RUN: diff %T/check_header_files/out/include/dpct/dpl_extras/functional.h  %S/../../runtime/dpct-rt/include/dpl_extras/functional.h >> %T/check_header_files/diff_res.txt
// RUN: echo "end" >> %T/check_header_files/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files/diff_res.txt

// RUN: echo "begin" > %T/check_header_files/diff_res.txt
// RUN: diff %T/check_header_files/out/include/dpct/dpl_extras/iterators.h  %S/../../runtime/dpct-rt/include/dpl_extras/iterators.h >> %T/check_header_files/diff_res.txt
// RUN: echo "end" >> %T/check_header_files/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files/diff_res.txt

// RUN: echo "begin" > %T/check_header_files/diff_res.txt
// RUN: diff %T/check_header_files/out/include/dpct/dpl_extras/memory.h  %S/../../runtime/dpct-rt/include/dpl_extras/memory.h >> %T/check_header_files/diff_res.txt
// RUN: echo "end" >> %T/check_header_files/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files/diff_res.txt

// RUN: echo "begin" > %T/check_header_files/diff_res.txt
// RUN: diff %T/check_header_files/out/include/dpct/dpl_extras/numeric.h  %S/../../runtime/dpct-rt/include/dpl_extras/numeric.h >> %T/check_header_files/diff_res.txt
// RUN: echo "end" >> %T/check_header_files/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files/diff_res.txt

// RUN: echo "begin" > %T/check_header_files/diff_res.txt
// RUN: diff %T/check_header_files/out/include/dpct/dpl_extras/vector.h  %S/../../runtime/dpct-rt/include/dpl_extras/vector.h >> %T/check_header_files/diff_res.txt
// RUN: echo "end" >> %T/check_header_files/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files/diff_res.txt

// RUN: echo "begin" > %T/check_header_files/diff_res.txt
// RUN: diff %T/check_header_files/out/include/dpct/dpl_extras/dpcpp_extensions.h  %S/../../runtime/dpct-rt/include/dpl_extras/dpcpp_extensions.h >> %T/check_header_files/diff_res.txt
// RUN: echo "end" >> %T/check_header_files/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files/diff_res.txt

// RUN: rm -rf %T/check_header_files

// CHECK: begin
// CHECK-NEXT: end

