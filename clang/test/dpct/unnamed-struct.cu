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

//CHECK: struct __dpct_align__(4) dpct_type_{{[a-f0-9]+}}
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

//CHECK: class dpct_type_{{[a-f0-9]+}}: public T2 {
class: public T2 {
    unsigned k;
} B;
