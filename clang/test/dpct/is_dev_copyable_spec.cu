// RUN: dpct --format-range=none --out-root %T/is_dev_copyable_spec %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck --input-file %T/is_dev_copyable_spec/is_dev_copyable_spec.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/is_dev_copyable_spec/is_dev_copyable_spec.dp.cpp -o %T/is_dev_copyable_spec/is_dev_copyable_spec.dp.o %}

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

void foo1() {
  UserStruct1<float, int> us1;
  UserStruct2 us2;
  k1<<<1, 1>>>(us1, us2);
}

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
