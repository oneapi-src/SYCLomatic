// UNSUPPORTED: -linux-
// RUN: cat %S/DemoCudaProj.vcxproj > %T/DemoCudaProj.vcxproj
// RUN: dpct  --vcxprojfile=%T/DemoCudaProj.vcxproj  -in-root=%S -out-root=%T  %s  -- -x cuda --cuda-host-only --cuda-path="%cuda-path"
// RUN: echo "// CHECK: [" >%T/check_compilation_db.txt
// RUN: echo "// CHECK:     {" >>%T/check_compilation_db.txt
// RUN: echo "// CHECK:         \"file\":\"a_kernel.cu\"," >>%T/check_compilation_db.txt
// RUN: echo "// CHECK:         \"command\":\"compile -m64 -DNDEBUG -DWIN32 -DWIN64 -D_CONSOLE -D_DEBUG  {{.*}}">>%T/check_compilation_db.txt
// RUN: echo "// CHECK:         \"directory\":\"{{.*}}/a_vcxproj_test/Output\"" >>%T/check_compilation_db.txt
// RUN: echo "// CHECK:     }" >>%T/check_compilation_db.txt
// RUN: echo "// CHECK: ]" >>%T/check_compilation_db.txt
// RUN: cat %T/compile_commands.json >>%T/check_compilation_db.txt
// RUN: FileCheck --match-full-lines --input-file %T/check_compilation_db.txt %T/check_compilation_db.txt

#include "cuda_runtime.h"
#include <stdio.h>

// CHECK: void addKernel(int *c, const int *a, const int *b, cl::sycl::nd_item<3> item_ct1)
__global__ void addKernel(int *c, const int *a, const int *b)
{
    // CHECK: int i = item_ct1.get_local_id(0);
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

