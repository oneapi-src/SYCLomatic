// RUN: dpct --format-range=none -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++14
// RUN: FileCheck --input-file %T/unnamed-struct.dp.cpp --match-full-lines %s

#include <vector>
#include <algorithm>
#include <cctype>

int main()
{
  std::vector<unsigned char> s;
  // CHECK: std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) -> unsigned char { return std::toupper(c);  });
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) -> unsigned char { return std::toupper(c);  });
  return 0;
}

//CHECK: struct __dpct_align__(4)
//CHECK-NEXT: {
//CHECK-NEXT:     unsigned i;
//CHECK-NEXT: } A;
struct __align__(4)
{
    unsigned i;
} A;

//CHECK: typedef class dpct_type_{{[a-f0-9]+}}{
typedef class{
    unsigned i;
} T1;


//CHECK: typedef struct dpct_type_{{[a-f0-9]+}}
typedef struct
	: public T1
{
    unsigned j;
} T2;


//CHECK:class: public T2 {
//CHECK-NEXT:    unsigned k;
//CHECK-NEXT:} B;
class: public T2 {
    unsigned k;
} B;

//CHECK:struct patch_pair {
//CHECK-NEXT:  union {
//CHECK-NEXT:    bool done[2];
//CHECK-NEXT:    struct {
//CHECK-NEXT:      int t1;
//CHECK-NEXT:      int t2;
//CHECK-NEXT:    };
//CHECK-NEXT:  };
//CHECK-NEXT:};
struct patch_pair {
  union {
    bool done[2];
    struct {
      int t1;
      int t2;
    };
  };
};

//CHECK:typedef struct tt {
//CHECK-NEXT:    int t1;
//CHECK-NEXT:    union {
//CHECK-NEXT:        int cut;
//CHECK-NEXT:        int *t2;
//CHECK-NEXT:    };
//CHECK-NEXT:    struct {
//CHECK-NEXT:        int a;
//CHECK-NEXT:    };
//CHECK-NEXT:} tt;
typedef struct tt {
    int t1;
    union {
        int cut;
        int *t2;
    };
    struct {
        int a;
    };
} tt;
