// RUN: dpct --format-range=none -out-root %T/qualified_template_func %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -ptx
// RUN: FileCheck %s --match-full-lines --input-file %T/qualified_template_func/qualified_template_func.dp.cpp

#include<cuda.h>
//CHECK: #define DEFINE_CHECK_FUNC(name, op)                               \
//CHECK-NEXT:  template <typename X, typename Y>                               \
//CHECK-NEXT:  inline void LogCheck##name(const X& x, const Y& y) {   \
//CHECK-NEXT:																  \
//CHECK-NEXT:  }
#define DEFINE_CHECK_FUNC(name, op)                               \
  template <typename X, typename Y>                               \
  inline void LogCheck##name(const X& x, const Y& y) {            \
								  \
  }

#define CHECK_BINARY_OP(name, op, x, y) LogCheck##name(x, y)

#define CHECK_EQ(x, y) CHECK_BINARY_OP(_EQ, ==, x, y)

DEFINE_CHECK_FUNC(_EQ, ==)

int main() {
	cudaError_t error ;
	CHECK_EQ(error, 0);
	return 0;
}

