// RUN: mkdir %T/check_header_files
// RUN: echo "begin" > %T/check_header_files/diff_res.txt
// RUN: diff %S/helper_files_ref/include/atomic.hpp  %T/../../../runtime/dpct-rt/include/atomic.hpp >> %T/check_header_files/diff_res.txt
// RUN: echo "end" >> %T/check_header_files/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files/diff_res.txt

// RUN: echo "begin" > %T/check_header_files/diff_res.txt
// RUN: diff %S/helper_files_ref/include/blas_utils.hpp  %T/../../../runtime/dpct-rt/include/blas_utils.hpp >> %T/check_header_files/diff_res.txt
// RUN: echo "end" >> %T/check_header_files/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files/diff_res.txt

// RUN: echo "begin" > %T/check_header_files/diff_res.txt
// RUN: diff %S/helper_files_ref/include/device.hpp  %T/../../../runtime/dpct-rt/include/device.hpp >> %T/check_header_files/diff_res.txt
// RUN: echo "end" >> %T/check_header_files/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files/diff_res.txt

// RUN: echo "begin" > %T/check_header_files/diff_res.txt
// RUN: diff %S/helper_files_ref/include/dpct.hpp  %T/../../../runtime/dpct-rt/include/dpct.hpp >> %T/check_header_files/diff_res.txt
// RUN: echo "end" >> %T/check_header_files/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files/diff_res.txt

// RUN: echo "begin" > %T/check_header_files/diff_res.txt
// RUN: diff %S/helper_files_ref/include/dpl_utils.hpp  %T/../../../runtime/dpct-rt/include/dpl_utils.hpp >> %T/check_header_files/diff_res.txt
// RUN: echo "end" >> %T/check_header_files/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files/diff_res.txt

// RUN: echo "begin" > %T/check_header_files/diff_res.txt
// RUN: diff %S/helper_files_ref/include/image.hpp  %T/../../../runtime/dpct-rt/include/image.hpp >> %T/check_header_files/diff_res.txt
// RUN: echo "end" >> %T/check_header_files/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files/diff_res.txt

// RUN: echo "begin" > %T/check_header_files/diff_res.txt
// RUN: diff %S/helper_files_ref/include/kernel.hpp  %T/../../../runtime/dpct-rt/include/kernel.hpp >> %T/check_header_files/diff_res.txt
// RUN: echo "end" >> %T/check_header_files/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files/diff_res.txt

// RUN: echo "begin" > %T/check_header_files/diff_res.txt
// RUN: diff %S/helper_files_ref/include/memory.hpp  %T/../../../runtime/dpct-rt/include/memory.hpp >> %T/check_header_files/diff_res.txt
// RUN: echo "end" >> %T/check_header_files/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files/diff_res.txt

// RUN: echo "begin" > %T/check_header_files/diff_res.txt
// RUN: diff %S/helper_files_ref/include/util.hpp  %T/../../../runtime/dpct-rt/include/util.hpp >> %T/check_header_files/diff_res.txt
// RUN: echo "end" >> %T/check_header_files/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files/diff_res.txt

// RUN: echo "begin" > %T/check_header_files/diff_res.txt
// RUN: diff %S/helper_files_ref/include/rng_utils.hpp  %T/../../../runtime/dpct-rt/include/rng_utils.hpp >> %T/check_header_files/diff_res.txt
// RUN: echo "end" >> %T/check_header_files/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files/diff_res.txt

// RUN: echo "begin" > %T/check_header_files/diff_res.txt
// RUN: diff %S/helper_files_ref/include/lib_common_utils.hpp  %T/../../../runtime/dpct-rt/include/lib_common_utils.hpp >> %T/check_header_files/diff_res.txt
// RUN: echo "end" >> %T/check_header_files/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files/diff_res.txt

// RUN: echo "begin" > %T/check_header_files/diff_res.txt
// RUN: diff %S/helper_files_ref/include/ccl_utils.hpp  %T/../../../runtime/dpct-rt/include/ccl_utils.hpp >> %T/check_header_files/diff_res.txt
// RUN: echo "end" >> %T/check_header_files/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files/diff_res.txt

// RUN: echo "begin" > %T/check_header_files/diff_res.txt
// RUN: diff %S/helper_files_ref/include/dnnl_utils.hpp  %T/../../../runtime/dpct-rt/include/dnnl_utils.hpp >> %T/check_header_files/diff_res.txt
// RUN: echo "end" >> %T/check_header_files/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files/diff_res.txt

// RUN: echo "begin" > %T/check_header_files/diff_res.txt
// RUN: diff %S/helper_files_ref/include/fft_utils.hpp  %T/../../../runtime/dpct-rt/include/fft_utils.hpp >> %T/check_header_files/diff_res.txt
// RUN: echo "end" >> %T/check_header_files/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files/diff_res.txt

// RUN: echo "begin" > %T/check_header_files/diff_res.txt
// RUN: diff %S/helper_files_ref/include/lapack_utils.hpp  %T/../../../runtime/dpct-rt/include/lapack_utils.hpp >> %T/check_header_files/diff_res.txt
// RUN: echo "end" >> %T/check_header_files/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files/diff_res.txt

// RUN: echo "begin" > %T/check_header_files/diff_res.txt
// RUN: diff %S/helper_files_ref/include/dpl_extras/algorithm.h  %T/../../../runtime/dpct-rt/include/dpl_extras/algorithm.h >> %T/check_header_files/diff_res.txt
// RUN: echo "end" >> %T/check_header_files/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files/diff_res.txt

// RUN: echo "begin" > %T/check_header_files/diff_res.txt
// RUN: diff %S/helper_files_ref/include/dpl_extras/functional.h  %T/../../../runtime/dpct-rt/include/dpl_extras/functional.h >> %T/check_header_files/diff_res.txt
// RUN: echo "end" >> %T/check_header_files/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files/diff_res.txt

// RUN: echo "begin" > %T/check_header_files/diff_res.txt
// RUN: diff %S/helper_files_ref/include/dpl_extras/iterators.h  %T/../../../runtime/dpct-rt/include/dpl_extras/iterators.h >> %T/check_header_files/diff_res.txt
// RUN: echo "end" >> %T/check_header_files/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files/diff_res.txt

// RUN: echo "begin" > %T/check_header_files/diff_res.txt
// RUN: diff %S/helper_files_ref/include/dpl_extras/memory.h  %T/../../../runtime/dpct-rt/include/dpl_extras/memory.h >> %T/check_header_files/diff_res.txt
// RUN: echo "end" >> %T/check_header_files/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files/diff_res.txt

// RUN: echo "begin" > %T/check_header_files/diff_res.txt
// RUN: diff %S/helper_files_ref/include/dpl_extras/numeric.h  %T/../../../runtime/dpct-rt/include/dpl_extras/numeric.h >> %T/check_header_files/diff_res.txt
// RUN: echo "end" >> %T/check_header_files/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files/diff_res.txt

// RUN: echo "begin" > %T/check_header_files/diff_res.txt
// RUN: diff %S/helper_files_ref/include/dpl_extras/vector.h  %T/../../../runtime/dpct-rt/include/dpl_extras/vector.h >> %T/check_header_files/diff_res.txt
// RUN: echo "end" >> %T/check_header_files/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files/diff_res.txt

// RUN: echo "begin" > %T/check_header_files/diff_res.txt
// RUN: diff %S/helper_files_ref/include/dpl_extras/dpcpp_extensions.h  %T/../../../runtime/dpct-rt/include/dpl_extras/dpcpp_extensions.h >> %T/check_header_files/diff_res.txt
// RUN: echo "end" >> %T/check_header_files/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files/diff_res.txt

// RUN: rm -rf %T/check_header_files

// CHECK: begin
// CHECK-NEXT: end

