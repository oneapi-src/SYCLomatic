// RUN: dpct --format-range=none --out-root %T/is_dev_copyable_spec %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck --input-file %T/is_dev_copyable_spec/is_dev_copyable_spec.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/is_dev_copyable_spec/is_dev_copyable_spec.dp.cpp -o %T/is_dev_copyable_spec/is_dev_copyable_spec.dp.o %}

// Case1: template struct with <<<>>>
//      CHECK: template<class T1, class T2>
// CHECK-NEXT: struct UserStruct1 {
// CHECK-NEXT:   UserStruct1() {}
// CHECK-NEXT:   UserStruct1(UserStruct1 const &) {}
// CHECK-NEXT: };
// CHECK-NEXT: template <class T1, class T2>
// CHECK-NEXT: struct sycl::is_device_copyable<UserStruct1<T1, T2>> : std::true_type {};
// CHECK-NEXT: struct UserStruct2 {
// CHECK-NEXT:   UserStruct2() {}
// CHECK-NEXT:   UserStruct2(UserStruct2 const &) {}
// CHECK-NEXT: };
// CHECK-NEXT: template <>
// CHECK-NEXT: struct sycl::is_device_copyable<UserStruct2> : std::true_type {};
template<class T1, class T2>
struct UserStruct1 {
  UserStruct1() {}
  UserStruct1(UserStruct1 const &) {}
};
struct UserStruct2 {
  UserStruct2() {}
  UserStruct2(UserStruct2 const &) {}
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
// CHECK-NEXT: struct UserStruct3 {
// CHECK-NEXT:   UserStruct3() {}
// CHECK-NEXT:   UserStruct3(UserStruct3 const &) {}
// CHECK-NEXT: };
// CHECK-NEXT: template <class T1, class T2>
// CHECK-NEXT: struct sycl::is_device_copyable<UserStruct3<T1, T2>> : std::true_type {};
// CHECK-NEXT: struct UserStruct4 {
// CHECK-NEXT:   UserStruct4() {}
// CHECK-NEXT:   UserStruct4(UserStruct4 const &) {}
// CHECK-NEXT: };
// CHECK-NEXT: template <>
// CHECK-NEXT: struct sycl::is_device_copyable<UserStruct4> : std::true_type {};
template<class T1, class T2>
struct UserStruct3 {
  UserStruct3() {}
  UserStruct3(UserStruct3 const &) {}
};
struct UserStruct4 {
  UserStruct4() {}
  UserStruct4(UserStruct4 const &) {}
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
template<class V>
struct UserStruct7;

template<class V>
__global__ void k4(V) {}

template<class V>
void foo4() {
  UserStruct7<V> us7;
  k4<<<1, 1>>>(us7);
}

template<class T>
struct UserStruct7 {
  UserStruct7() {}
  UserStruct7(UserStruct7 const &) {}
};

template void foo4<float>();

// Case5: Full specialization of template struct
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
