// RUN: mkdir %T/check_header_files
// RUN: echo "begin" > %T/check_header_files/diff_res.txt
// RUN: diff %S/helper_files_ref/include/atomic.hpp  %T/../../../runtime/c2s-rt/include/atomic.hpp >> %T/check_header_files/diff_res.txt
// RUN: echo "end" >> %T/check_header_files/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files/diff_res.txt

// RUN: echo "begin" > %T/check_header_files/diff_res.txt
// RUN: diff %S/helper_files_ref/include/blas_utils.hpp  %T/../../../runtime/c2s-rt/include/blas_utils.hpp >> %T/check_header_files/diff_res.txt
// RUN: echo "end" >> %T/check_header_files/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files/diff_res.txt

// RUN: echo "begin" > %T/check_header_files/diff_res.txt
// RUN: diff %S/helper_files_ref/include/device.hpp  %T/../../../runtime/c2s-rt/include/device.hpp >> %T/check_header_files/diff_res.txt
// RUN: echo "end" >> %T/check_header_files/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files/diff_res.txt

// RUN: echo "begin" > %T/check_header_files/diff_res.txt
// RUN: diff %S/helper_files_ref/include/c2s.hpp  %T/../../../runtime/c2s-rt/include/c2s.hpp >> %T/check_header_files/diff_res.txt
// RUN: echo "end" >> %T/check_header_files/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files/diff_res.txt

// RUN: echo "begin" > %T/check_header_files/diff_res.txt
// RUN: diff %S/helper_files_ref/include/dpl_utils.hpp  %T/../../../runtime/c2s-rt/include/dpl_utils.hpp >> %T/check_header_files/diff_res.txt
// RUN: echo "end" >> %T/check_header_files/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files/diff_res.txt

// RUN: echo "begin" > %T/check_header_files/diff_res.txt
// RUN: diff %S/helper_files_ref/include/image.hpp  %T/../../../runtime/c2s-rt/include/image.hpp >> %T/check_header_files/diff_res.txt
// RUN: echo "end" >> %T/check_header_files/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files/diff_res.txt

// RUN: echo "begin" > %T/check_header_files/diff_res.txt
// RUN: diff %S/helper_files_ref/include/kernel.hpp  %T/../../../runtime/c2s-rt/include/kernel.hpp >> %T/check_header_files/diff_res.txt
// RUN: echo "end" >> %T/check_header_files/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files/diff_res.txt

// RUN: echo "begin" > %T/check_header_files/diff_res.txt
// RUN: diff %S/helper_files_ref/include/memory.hpp  %T/../../../runtime/c2s-rt/include/memory.hpp >> %T/check_header_files/diff_res.txt
// RUN: echo "end" >> %T/check_header_files/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files/diff_res.txt

// RUN: echo "begin" > %T/check_header_files/diff_res.txt
// RUN: diff %S/helper_files_ref/include/util.hpp  %T/../../../runtime/c2s-rt/include/util.hpp >> %T/check_header_files/diff_res.txt
// RUN: echo "end" >> %T/check_header_files/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files/diff_res.txt

// RUN: echo "begin" > %T/check_header_files/diff_res.txt
// RUN: diff %S/helper_files_ref/include/rng_utils.hpp  %T/../../../runtime/c2s-rt/include/rng_utils.hpp >> %T/check_header_files/diff_res.txt
// RUN: echo "end" >> %T/check_header_files/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files/diff_res.txt

// RUN: echo "begin" > %T/check_header_files/diff_res.txt
// RUN: diff %S/helper_files_ref/include/lib_common_utils.hpp  %T/../../../runtime/c2s-rt/include/lib_common_utils.hpp >> %T/check_header_files/diff_res.txt
// RUN: echo "end" >> %T/check_header_files/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files/diff_res.txt

// RUN: echo "begin" > %T/check_header_files/diff_res.txt
// RUN: diff %S/helper_files_ref/include/dpl_extras/algorithm.h  %T/../../../runtime/c2s-rt/include/dpl_extras/algorithm.h >> %T/check_header_files/diff_res.txt
// RUN: echo "end" >> %T/check_header_files/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files/diff_res.txt

// RUN: echo "begin" > %T/check_header_files/diff_res.txt
// RUN: diff %S/helper_files_ref/include/dpl_extras/functional.h  %T/../../../runtime/c2s-rt/include/dpl_extras/functional.h >> %T/check_header_files/diff_res.txt
// RUN: echo "end" >> %T/check_header_files/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files/diff_res.txt

// RUN: echo "begin" > %T/check_header_files/diff_res.txt
// RUN: diff %S/helper_files_ref/include/dpl_extras/iterators.h  %T/../../../runtime/c2s-rt/include/dpl_extras/iterators.h >> %T/check_header_files/diff_res.txt
// RUN: echo "end" >> %T/check_header_files/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files/diff_res.txt

// RUN: echo "begin" > %T/check_header_files/diff_res.txt
// RUN: diff %S/helper_files_ref/include/dpl_extras/memory.h  %T/../../../runtime/c2s-rt/include/dpl_extras/memory.h >> %T/check_header_files/diff_res.txt
// RUN: echo "end" >> %T/check_header_files/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files/diff_res.txt

// RUN: echo "begin" > %T/check_header_files/diff_res.txt
// RUN: diff %S/helper_files_ref/include/dpl_extras/numeric.h  %T/../../../runtime/c2s-rt/include/dpl_extras/numeric.h >> %T/check_header_files/diff_res.txt
// RUN: echo "end" >> %T/check_header_files/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files/diff_res.txt

// RUN: echo "begin" > %T/check_header_files/diff_res.txt
// RUN: diff %S/helper_files_ref/include/dpl_extras/vector.h  %T/../../../runtime/c2s-rt/include/dpl_extras/vector.h >> %T/check_header_files/diff_res.txt
// RUN: echo "end" >> %T/check_header_files/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files/diff_res.txt

// RUN: echo "begin" > %T/check_header_files/diff_res.txt
// RUN: diff %S/helper_files_ref/include/dpl_extras/dpcpp_extensions.h  %T/../../../runtime/c2s-rt/include/dpl_extras/dpcpp_extensions.h >> %T/check_header_files/diff_res.txt
// RUN: echo "end" >> %T/check_header_files/diff_res.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/check_header_files/diff_res.txt

// RUN: rm -rf %T/check_header_files

// CHECK: begin
// CHECK-NEXT: end

