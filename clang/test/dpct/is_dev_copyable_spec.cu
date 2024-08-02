// RUN: dpct --format-range=none --out-root %T/is_dev_copyable_spec %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck --input-file %T/is_dev_copyable_spec/is_dev_copyable_spec.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/is_dev_copyable_spec/is_dev_copyable_spec.dp.cpp -o %T/is_dev_copyable_spec/is_dev_copyable_spec.dp.o %}

// Case1: template struct with <<<>>>
//      CHECK: template<class T1, class T2>
// CHECK-NEXT: /*
// CHECK-NEXT: DPCT1128:{{[0-9]+}}: The type "UserStruct1<float, int>" is not device copyable for copy constructor breaking the device copyable requirement. It is used in the SYCL kernel, please rewrite the code.
// CHECK-NEXT: */
// CHECK-NEXT: struct UserStruct1 {
// CHECK-NEXT:   UserStruct1() {}
// CHECK-NEXT:   UserStruct1(const UserStruct1&) {}
// CHECK-NEXT: };
// CHECK-NEXT: template <class T1, class T2>
// CHECK-NEXT: struct sycl::is_device_copyable<UserStruct1<T1, T2>> : std::true_type {};
// CHECK-NEXT: /*
// CHECK-NEXT: DPCT1128:{{[0-9]+}}: The type "UserStruct2" is not device copyable for copy constructor breaking the device copyable requirement. It is used in the SYCL kernel, please rewrite the code.
// CHECK-NEXT: */
// CHECK-NEXT: struct UserStruct2 {
// CHECK-NEXT:   UserStruct2() {}
// CHECK-NEXT:   UserStruct2(const UserStruct2&) {}
// CHECK-NEXT: };
// CHECK-NEXT: template <>
// CHECK-NEXT: struct sycl::is_device_copyable<UserStruct2> : std::true_type {};
template<class T1, class T2>
struct UserStruct1 {
  UserStruct1() {}
  UserStruct1(UserStruct1&) {}
};
struct UserStruct2 {
  UserStruct2() {}
  UserStruct2(UserStruct2&) {}
};

template<class V1, class V2>
__global__ void k1(UserStruct1<V1, V2>, UserStruct2) {}

template<class V1, class V2>
void foo1() {
  UserStruct1<V1, V2> us1;
  UserStruct2 us2;
  k1<<<1, 1>>>(us1, us2);
}

template void foo1<float, int>();

// Case2: template struct with cudaLaunchKernel
//      CHECK: template<class T1, class T2>
// CHECK-NEXT: /*
// CHECK-NEXT: DPCT1128:{{[0-9]+}}: The type "UserStruct3<float, int>" is not device copyable for copy constructor breaking the device copyable requirement. It is used in the SYCL kernel, please rewrite the code.
// CHECK-NEXT: */
// CHECK-NEXT: struct UserStruct3 {
// CHECK-NEXT:   UserStruct3() {}
// CHECK-NEXT:   UserStruct3(const UserStruct3&) {}
// CHECK-NEXT: };
// CHECK-NEXT: template <class T1, class T2>
// CHECK-NEXT: struct sycl::is_device_copyable<UserStruct3<T1, T2>> : std::true_type {};
// CHECK-NEXT: /*
// CHECK-NEXT: DPCT1128:{{[0-9]+}}: The type "UserStruct4" is not device copyable for copy constructor breaking the device copyable requirement. It is used in the SYCL kernel, please rewrite the code.
// CHECK-NEXT: */
// CHECK-NEXT: struct UserStruct4 {
// CHECK-NEXT:   UserStruct4() {}
// CHECK-NEXT:   UserStruct4(const UserStruct4&) {}
// CHECK-NEXT: };
// CHECK-NEXT: template <>
// CHECK-NEXT: struct sycl::is_device_copyable<UserStruct4> : std::true_type {};
template<class T1, class T2>
struct UserStruct3 {
  UserStruct3() {}
  UserStruct3(UserStruct3&) {}
};
struct UserStruct4 {
  UserStruct4() {}
  UserStruct4(UserStruct4&) {}
};

template<class V1, class V2>
__global__ void k2(UserStruct3<V1, V2>, UserStruct4) {}

void foo2() {
  UserStruct3<float, int> us3;
  UserStruct4 us4;

  void *args[2] = { &us3, &us4 };
  cudaLaunchKernel((void *)&k2<float, int>, dim3(1), dim3(1), args, 0, 0);
}

// Case3: template struct as a template argument
//      CHECK: struct UserStruct5 {
// CHECK-NEXT:   UserStruct5() {}
// CHECK-NEXT:   UserStruct5(UserStruct5 const &) {}
// CHECK-NEXT: };
// CHECK-NEXT: template<class T>
// CHECK-NEXT: /*
// CHECK-NEXT: DPCT1128:{{[0-9]+}}: The type "UserStruct6<UserStruct5>" is not device copyable for copy constructor breaking the device copyable requirement. It is used in the SYCL kernel, please rewrite the code.
// CHECK-NEXT: */
// CHECK-NEXT: struct UserStruct6 {
// CHECK-NEXT:   UserStruct6() {}
// CHECK-NEXT:   UserStruct6(UserStruct6 const &) {}
// CHECK-NEXT: };
// CHECK-NEXT: template <class T>
// CHECK-NEXT: struct sycl::is_device_copyable<UserStruct6<T>> : std::true_type {};
struct UserStruct5 {
  UserStruct5() {}
  UserStruct5(UserStruct5 const &) {}
};
template<class T>
struct UserStruct6 {
  UserStruct6() {}
  UserStruct6(UserStruct6 const &) {}
};

template<class V>
__global__ void k3(V) {}

void foo3() {
  UserStruct6<UserStruct5> us6;
  k3<<<1, 1>>>(us6);
}

// Case4: Forward declaraion of template struct
//      CHECK: template<class V4>
// CHECK-NEXT: struct UserStruct7;
// CHECK-NEXT: template <class V4> 
// CHECK-NEXT: struct sycl::is_device_copyable<UserStruct7<V4>> : std::true_type {};
template<class V4>
struct UserStruct7;

template<class V>
__global__ void k4(V) {}

template<class V>
void foo4() {
  UserStruct7<V> us7;
  k4<<<1, 1>>>(us7);
}

//      CHECK: template<class T>
// CHECK-NEXT: /*
// CHECK-NEXT: DPCT1128:{{[0-9]+}}: The type "UserStruct7<float>" is not device copyable for copy constructor breaking the device copyable requirement. It is used in the SYCL kernel, please rewrite the code.
// CHECK-NEXT: */
// CHECK-NEXT: struct UserStruct7 {
// CHECK-NEXT:   UserStruct7() {}
// CHECK-NEXT:   UserStruct7(UserStruct7 const &) {}
// CHECK-NEXT: };
// CHECK-EMPTY:
// CHECK-NEXT: template void foo4<float>();
template<class T>
struct UserStruct7 {
  UserStruct7() {}
  UserStruct7(UserStruct7 const &) {}
};

template void foo4<float>();

// Case5: Full specialization of template struct
//      CHECK: template<class T1, class T2>
// CHECK-NEXT: struct UserStruct8 {
// CHECK-NEXT:   UserStruct8() {}
// CHECK-NEXT:   UserStruct8(UserStruct8 const &) {}
// CHECK-NEXT: };
// CHECK-NEXT: template<>
// CHECK-NEXT: /*
// CHECK-NEXT: DPCT1128:{{[0-9]+}}: The type "UserStruct8<double, float>" is not device copyable for copy constructor breaking the device copyable requirement. It is used in the SYCL kernel, please rewrite the code.
// CHECK-NEXT: */
// CHECK-NEXT: struct UserStruct8<double, float> {
// CHECK-NEXT:   UserStruct8() {}
// CHECK-NEXT:   UserStruct8(UserStruct8 const &) {}
// CHECK-NEXT: };
// CHECK-NEXT: template <class T1, class T2> 
// CHECK-NEXT: struct sycl::is_device_copyable<UserStruct8<T1, T2>> : std::true_type {};
template<class T1, class T2>
struct UserStruct8 {
  UserStruct8() {}
  UserStruct8(UserStruct8 const &) {}
};
template<>
struct UserStruct8<double, float> {
  UserStruct8() {}
  UserStruct8(UserStruct8 const &) {}
};

template<class V>
__global__ void k5(V) {}

void foo5() {
  UserStruct8<double, float> us8;
  k5<<<1, 1>>>(us8);
}

// Case6: Partial specialization of template struct
//      CHECK: template<class T1, class T2>
// CHECK-NEXT: struct UserStruct9 {
// CHECK-NEXT:   UserStruct9() {}
// CHECK-NEXT:   UserStruct9(UserStruct9 const &) {}
// CHECK-NEXT: };
// CHECK-NEXT: template<class TT>
// CHECK-NEXT: /*
// CHECK-NEXT: DPCT1128:{{[0-9]+}}: The type "UserStruct9<float, int>" is not device copyable for copy constructor breaking the device copyable requirement. It is used in the SYCL kernel, please rewrite the code.
// CHECK-NEXT: */
// CHECK-NEXT: struct UserStruct9<TT, int> {
// CHECK-NEXT:   UserStruct9() {}
// CHECK-NEXT:   UserStruct9(UserStruct9 const &) {}
// CHECK-NEXT: };
// CHECK-NEXT: template <class T1, class T2> 
// CHECK-NEXT: struct sycl::is_device_copyable<UserStruct9<T1, T2>> : std::true_type {};
template<class T1, class T2>
struct UserStruct9 {
  UserStruct9() {}
  UserStruct9(UserStruct9 const &) {}
};
template<class TT>
struct UserStruct9<TT, int> {
  UserStruct9() {}
  UserStruct9(UserStruct9 const &) {}
};

template<class V>
__global__ void k6(V) {}

void foo6() {
  UserStruct9<float, int> us9;
  k6<<<1, 1>>>(us9);
}

// Case7: template struct with nested template struct
//template<class U>
//struct Test {
//template<class T>
//struct UserStruct10 {
//  UserStruct10() {}
//  UserStruct10(UserStruct10 const &) {}
//};
//};
//
//template<class V>
//__global__ void k7(V) {}
//
//void foo7() {
//  Test<int>::UserStruct10<float> us10;
//  k7<<<1, 1>>>(us10);
//}
//
// Case8: locally-defined template struct
//template<class V>
//__global__ void k8(V) {}
//
//void test8() {
//  struct UserStruct11 {
//    UserStruct11() {}
//    UserStruct11(UserStruct11 const &) {}
//  };
//
//  UserStruct11 us11;
//  k8<<<1, 1>>>(us11);
//}

// Case9: template struct with using and template alias
//      CHECK: DPCT1128:{{[0-9]+}}: The type "UserStruct12" is not device copyable for copy constructor breaking the device copyable requirement. It is used in the SYCL kernel, please rewrite the code.
// CHECK-NEXT: */
// CHECK-NEXT: struct UserStruct12 {
// CHECK-NEXT:   UserStruct12() {}
// CHECK-NEXT:   UserStruct12(UserStruct12 const &) {}
// CHECK-NEXT: };
// CHECK-NEXT: template <>
// CHECK-NEXT: struct sycl::is_device_copyable<UserStruct12> : std::true_type {};
// CHECK-NEXT: template<class T>
// CHECK-NEXT: /*
// CHECK-NEXT: DPCT1128:{{[0-9]+}}: The type "UserStruct13<float>" is not device copyable for copy constructor breaking the device copyable requirement. It is used in the SYCL kernel, please rewrite the code.
// CHECK-NEXT: */
// CHECK-NEXT: struct UserStruct13 {
// CHECK-NEXT:   UserStruct13() {}
// CHECK-NEXT:   UserStruct13(UserStruct13 const &) {}
// CHECK-NEXT: };
// CHECK-NEXT: template <class T> 
// CHECK-NEXT: struct sycl::is_device_copyable<UserStruct13<T>> : std::true_type {};
struct UserStruct12 {
  UserStruct12() {}
  UserStruct12(UserStruct12 const &) {}
};
template<class T>
struct UserStruct13 {
  UserStruct13() {}
  UserStruct13(UserStruct13 const &) {}
};

struct UserStruct14 {
  using A = UserStruct12;
  template<class U> using B = UserStruct13<U>;
};

template<class V1, class V2>
__global__ void k9(V1, V2) {}

void test9() {
  UserStruct14::A us13;
  UserStruct14::B<float> us14;
  k9<<<1, 1>>>(us13, us14);
}

// Case10: user-define struct in namespace
//      CHECK: namespace ns {
// CHECK-NEXT: /*
// CHECK-NEXT: DPCT1128:{{[0-9]+}}: The type "ns::UserStruct15" is not device copyable for copy constructor breaking the device copyable requirement. It is used in the SYCL kernel, please rewrite the code.
// CHECK-NEXT: */
// CHECK-NEXT: struct UserStruct15 {
// CHECK-NEXT:   UserStruct15() {}
// CHECK-NEXT:   UserStruct15(UserStruct15 const &) {}
// CHECK-NEXT: };
// CHECK-NEXT: }
// CHECK-NEXT: template <>
// CHECK-NEXT: struct sycl::is_device_copyable<ns::UserStruct15> : std::true_type {}; // namespace ns
namespace ns {
struct UserStruct15 {
  UserStruct15() {}
  UserStruct15(UserStruct15 const &) {}
};
} // namespace ns

template<class V>
__global__ void k10(V) {}

void test10() {
  ns::UserStruct15 us15;
  k10<<<1, 1>>>(us15);
}

// Case11: test warning message
template<class T>
__global__ void k11(T) {}
//      CHECK: struct B1 {};
// CHECK-NEXT: /*
// CHECK-NEXT: DPCT1128:{{[0-9]+}}: The type "B2" is not device copyable for copy constructor breaking the device copyable requirement. It is used in the SYCL kernel, please rewrite the code.
// CHECK-NEXT: */
// CHECK-NEXT: struct B2 {
// CHECK-NEXT:   B2() {}
// CHECK-NEXT:   B2(const B2& other) {}
// CHECK-NEXT: };
// CHECK-NEXT: /*
// CHECK-NEXT: DPCT1128:{{[0-9]+}}: The type "F" is not device copyable for copy constructor breaking the device copyable requirement. It is used in the SYCL kernel, please rewrite the code.
// CHECK-NEXT: */
// CHECK-NEXT: struct F {
// CHECK-NEXT:   F() {}
// CHECK-NEXT:   F(const F& other) {}
// CHECK-NEXT: };
// CHECK-NEXT: /*
// CHECK-NEXT: DPCT1128:{{[0-9]+}}: The type "UserStruct16" is not device copyable for copy constructor, copy assignment, move constructor, move assignment, destructor, virtual method "m", virtual base class "B1", non trivially copyable base class "B2" and non trivially copyable field "f" breaking the device copyable requirement. It is used in the SYCL kernel, please rewrite the code.
// CHECK-NEXT: */
// CHECK-NEXT: struct UserStruct16 : virtual public B1, public B2 {
// CHECK-NEXT:   UserStruct16() {}
// CHECK-NEXT:   UserStruct16(const UserStruct16& other) {}
// CHECK-NEXT:   UserStruct16& operator=(const UserStruct16& other) { return *this; }
// CHECK-NEXT:   UserStruct16(UserStruct16&& other) {}
// CHECK-NEXT:   UserStruct16& operator=(UserStruct16&& other) { return *this; }
// CHECK-NEXT:   ~UserStruct16() {}
// CHECK-NEXT:   virtual void m() {}
// CHECK-NEXT:   F f;
// CHECK-NEXT: };
// CHECK-NEXT: template <>
// CHECK-NEXT: struct sycl::is_device_copyable<UserStruct16> : std::true_type {};
struct B1 {};
struct B2 {
  B2() {}
  B2(B2& other) {}
};
struct F {
  F() {}
  F(F& other) {}
};
struct UserStruct16 : virtual public B1, public B2 {
  UserStruct16() {}
  UserStruct16(const UserStruct16& other) {}
  UserStruct16& operator=(const UserStruct16& other) { return *this; }
  UserStruct16(UserStruct16&& other) {}
  UserStruct16& operator=(UserStruct16&& other) { return *this; }
  ~UserStruct16() {}
  virtual void m() {}
  F f;
};
//      CHECK: /*
// CHECK-NEXT: DPCT1128:{{[0-9]+}}: The type "UserStruct17" is not device copyable for copy constructor breaking the device copyable requirement. It is used in the SYCL kernel, please rewrite the code.
// CHECK-NEXT: */
// CHECK-NEXT: struct UserStruct17 {
// CHECK-NEXT:   UserStruct17() {}
// CHECK-NEXT:   UserStruct17(const UserStruct17& other) {}
// CHECK-NEXT:   UserStruct17& operator=(const UserStruct17& other) = delete;
// CHECK-NEXT: };
// CHECK-NEXT: template <>
// CHECK-NEXT: struct sycl::is_device_copyable<UserStruct17> : std::true_type {};
struct UserStruct17 {
  UserStruct17() {}
  UserStruct17(const UserStruct17& other) {}
  UserStruct17& operator=(const UserStruct17& other) = delete;
};

void test11() {
  UserStruct16 us16;
  //      CHECK: /*
  // CHECK-NEXT: DPCT1129:{{[0-9]+}}: The type "UserStruct16" is used in the SYCL kernel, but it is not device copyable. The sycl::is_device_copyable specialization has been added for this type. Please review the code.
  // CHECK-NEXT: */
  // CHECK-NEXT: q_ct1.parallel_for(
  // CHECK-NEXT:   sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)), 
  // CHECK-NEXT:   [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:     k11(us16);
  // CHECK-NEXT:   });
  k11<<<1, 1>>>(us16);
  UserStruct17 us17;
  //      CHECK: /*
  // CHECK-NEXT: DPCT1129:{{[0-9]+}}: The type "UserStruct17" is used in the SYCL kernel, but it is not device copyable. The sycl::is_device_copyable specialization has been added for this type. Please review the code.
  // CHECK-NEXT: */
  // CHECK-NEXT: q_ct1.parallel_for(
  // CHECK-NEXT:   sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)), 
  // CHECK-NEXT:   [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:     k11(us17);
  // CHECK-NEXT:   });
  k11<<<1, 1>>>(us17);
}

// Case12: test adding const qualifier
// CHECK: struct UserStruct18 {
// CHECK-NEXT:   UserStruct18() {}
// CHECK-NEXT:   UserStruct18(const volatile UserStruct18& other) {}
// CHECK-NEXT:   UserStruct18(const UserStruct18& other) {}
// CHECK-NEXT: };
struct UserStruct18 {
  UserStruct18() {}
  UserStruct18(volatile UserStruct18& other) {}
  UserStruct18(const UserStruct18& other) {}
};

// CHECK: struct UserStruct19 {
// CHECK-NEXT:   UserStruct19() {}
// CHECK-NEXT:   UserStruct19(const volatile UserStruct19& other) {}
// CHECK-NEXT:   UserStruct19(const UserStruct19& other) {}
// CHECK-NEXT: };
// CHECK-NEXT: template <>
// CHECK-NEXT: struct sycl::is_device_copyable<UserStruct19> : std::true_type {};
struct UserStruct19 {
  UserStruct19() {}
  UserStruct19(volatile UserStruct19& other) {}
  UserStruct19(UserStruct19& other) {}
};

void test12() {
  volatile UserStruct18 us18;
  k11<<<1, 1>>>(us18);
  UserStruct19 us19;
  k11<<<1, 1>>>(us19);
}
