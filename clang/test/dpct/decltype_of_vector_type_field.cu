// RUN: dpct -out-root %T/decltype_of_vector_type_field %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/decltype_of_vector_type_field/decltype_of_vector_type_field.dp.cpp

void f() {
  // CHECK: using dim3_x_type = size_t;
  using dim3_x_type = decltype(dim3::x);
  // CHECK: using dim3_y_type = size_t;
  using dim3_y_type = decltype(dim3::y);
  // CHECK: using dim3_z_type = size_t;
  using dim3_z_type = decltype(dim3::z);
  // CHECK: using int1_x_type = int;
  using int1_x_type = decltype(int1::x);
  // CHECK: using uint3_x_type = sycl::uint3::element_type;
  using uint3_x_type = decltype(uint3::x);
  // CHECK: using uint3_y_type = sycl::uint3::element_type;
  using uint3_y_type = decltype(uint3::y);
  // CHECK: using uint3_z_type = sycl::uint3::element_type;
  using uint3_z_type = decltype(uint3::z);
  // CHECK: using char1_x_type = char;
  using char1_x_type = decltype(char1::x);
  // CHECK: using char4_x_type = sycl::char4::element_type;
  using char4_x_type = decltype(char4::x);
  // CHECK: using char4_y_type = sycl::char4::element_type;
  using char4_y_type = decltype(char4::y);
  // CHECK: using char4_z_type = sycl::char4::element_type;
  using char4_z_type = decltype(char4::z);
  // CHECK: using char4_w_type = sycl::char4::element_type;
  using char4_w_type = decltype(char4::w);
}
