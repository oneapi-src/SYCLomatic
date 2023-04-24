// RUN: dpct --format-range=none -out-root %T/minmax_cast %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/minmax_cast/minmax_cast.dp.cpp --match-full-lines %s

#include <algorithm>
#include <cmath>
#include <cstdio>

__global__ void func() {

  // case1: int min(arg0:int, arg1:int)
  // no cast required.
  int CHUNK_SIZE2 = 64;
  int numz2;
  //CHECK: const int zlength2 = sycl::min(CHUNK_SIZE2, numz2);
  const int zlength2 = min(CHUNK_SIZE2, numz2);
  // case2: unsigned int min(arg0:unsigned int, arg1:int)
  // cast for arg1: in AST tree, unsigned int version min is used.
  unsigned int CHUNK_SIZE3 = 64;
  int numz3;
  //CHECK: const int zlength3 = dpct::min(CHUNK_SIZE3, numz3);
  const int zlength3 = min(CHUNK_SIZE3, numz3);

  // case3: unsigned int min(arg0(int),arg1(unsigned int))
  // cast for arg0: in AST tree, unsigned int version min is used.
  // cast for arg1: arg1 is expr which require tranlation.
  int CHUNK_SIZE0 = 64;
  int numz0;
  //CHECK: const int zlength0 = dpct::min(CHUNK_SIZE0, numz0 - item_ct1.get_group(2) * CHUNK_SIZE0);
  const int zlength0 = min(CHUNK_SIZE0, numz0 - blockIdx.x * CHUNK_SIZE0);

  // case4: unsigned int min(arg0:unsigned int, arg1:unsigned int)
  // cast for arg1: arg1 is expr which require tranlation.
  unsigned int CHUNK_SIZE1 = 64;
  int numz1;
  //CHECK: const int zlength1 = sycl::min(CHUNK_SIZE1, numz1 - item_ct1.get_group(2) * CHUNK_SIZE1);
  const int zlength1 = min(CHUNK_SIZE1, numz1 - blockIdx.x * CHUNK_SIZE1);

  // case5: compare return type and argument tpye using canonical type to avoid unexpected type-cast
  using T = float;
  T x;
  const T y(1.0f);
  //CHECK: dpct::min(dpct::max(x + y, y), y);
  std::min(std::max(x + y, y), y);
}

