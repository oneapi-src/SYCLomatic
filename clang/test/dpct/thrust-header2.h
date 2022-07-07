

//CHECK:#include <oneapi/dpl/execution>
//CHECK-NEXT:#include <oneapi/dpl/algorithm>
//CHECK-NEXT:#include <CL/sycl.hpp>
//CHECK-NEXT:#include <dpct/dpct.hpp>
//CHECK-NEXT:#include <cstdio>
//CHECK-NEXT:#include <dpct/dpl_utils.hpp>
//CHECK-EMPTY:
//CHECK-NEXT:#include <algorithm>
#include <cstdio>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <algorithm>
