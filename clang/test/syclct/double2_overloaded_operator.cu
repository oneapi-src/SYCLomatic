// RUN: syclct -out-root %T %s -- -std=c++11 -x cuda --cuda-host-only --cuda-path=%cuda-path
// RUN: FileCheck %s --match-full-lines --input-file %T/double2_overloaded_operator.sycl.cpp

#include <cuda_runtime.h>

// CHECK: typedef cl::sycl::double2 ty;
typedef double2 ty;
// CHECK: typedef cl::sycl::double2& ty2;
typedef double2& ty2;
// CHECK: typedef cl::sycl::double2&& ty3;
typedef double2&& ty3;
// CHECK: typedef cl::sycl::double2** ty4;
typedef double2** ty4;
// CHECK: typedef cl::sycl::double2* ty5;
typedef double2* ty5;
// CHECK: typedef cl::sycl::double2*** ty6;
typedef double2*** ty6;
// CHECK: typedef const cl::sycl::double2*** ty7;
typedef const double2*** ty7;

// CHECK: namespace syclct_operator_overloading {
// CHECK: inline cl::sycl::double2 &operator*=(cl::sycl::double2 &v, const cl::sycl::double2 &v2)          ;
// CHECK: }  // namespace syclct_operator_overloading
inline double2 &operator*=(double2 &v, const double2 &v2)          ;

// CHECK: inline cl::sycl::double2 &copy(cl::sycl::double2 &v, const cl::sycl::double2 &v2) {
// CHECK:   v.x() = static_cast<const double>(v2.x());
// CHECK:   v.y() = static_cast<const double>(v2.y());
// CHECK:   return v;
// CHECK: }
__host__ __device__ inline double2 &copy(double2 &v, const double2 &v2) {
  v.x = v2.x;
  v.y = v2.y;
  return v;
}

// CHECK: namespace syclct_operator_overloading {
// CHECK: inline cl::sycl::double2 &operator+=(cl::sycl::double2 &v, const cl::sycl::double2 &v2) {
// CHECK:   v.x() += static_cast<const double>(v2.x());
// CHECK:   v.y() += static_cast<const double>(v2.y());
// CHECK:   return v;
// CHECK: }
// CHECK: }  // namespace syclct_operator_overloading
__host__ __device__ inline double2 &operator+=(double2 &v, const double2 &v2) {
  v.x += v2.x;
  v.y += v2.y;
  return v;
}

// CHECK: namespace syclct_operator_overloading {
// CHECK: inline cl::sycl::double2 &operator-=(cl::sycl::double2 &v, const cl::sycl::double2 &v2) {
// CHECK:   v.x() -= static_cast<const double>(v2.x());
// CHECK:   v.y() -= static_cast<const double>(v2.y());
// CHECK:   return v;
// CHECK: }  // namespace syclct_operator_overloading
__host__ __device__ inline double2 &operator-=(double2 &v, const double2 &v2) {
  v.x -= v2.x;
  v.y -= v2.y;
  return v;
}

// CHECK: namespace syclct_operator_overloading {
// CHECK: inline cl::sycl::double2 &operator*=(cl::sycl::double2 &v, const double &r) {
// CHECK:   v.x() *= r;
// CHECK:   v.y() *= r;
// CHECK:   return v;
// CHECK: }  // namespace syclct_operator_overloading
__host__ __device__ inline double2 &operator*=(double2 &v, const double &r) {
  v.x *= r;
  v.y *= r;
  return v;
}

// CHECK: namespace syclct_operator_overloading {
// CHECK: inline cl::sycl::double2 &operator/=(cl::sycl::double2 &v, const double &r) {
// CHECK:   v.x() /= r;
// CHECK:   v.y() /= r;
// CHECK:   return v;
// CHECK: }  // namespace syclct_operator_overloading
__host__ __device__ inline double2 &operator/=(double2 &v, const double &r) {
  v.x /= r;
  v.y /= r;
  return v;
}

// CHECK: namespace syclct_operator_overloading {
// CHECK: inline bool operator==(const cl::sycl::double2 &v1,
// CHECK:                                            const cl::sycl::double2 &v2) {
// CHECK:   return ((static_cast<const double>(v1.x()) == static_cast<const double>(v2.x())) && (static_cast<const double>(v1.y()) == static_cast<const double>(v2.y())));
// CHECK: }
// CHECK: }  // namespace syclct_operator_overloading
__host__ __device__ inline bool operator==(const double2 &v1,
                                           const double2 &v2) {
  return ((v1.x == v2.x) && (v1.y == v2.y));
}

// CHECK: namespace syclct_operator_overloading {
// CHECK: inline bool operator!=(const cl::sycl::double2 &v1,
// CHECK:                                            const cl::sycl::double2 &v2) {
// CHECK:   return (!(syclct_operator_overloading::operator==(v1 , v2)));
// CHECK: }
// CHECK: }  // namespace syclct_operator_overloading
__host__ __device__ inline bool operator!=(const double2 &v1,
                                           const double2 &v2) {
  return (!(v1 == v2));
}

// CHECK: inline cl::sycl::double2 operator+(const cl::sycl::double2 &v) { return v; }
__host__ __device__ inline double2 operator+(const double2 &v) { return v; }

// CHECK: inline cl::sycl::double2 operator-(const cl::sycl::double2 &v) {
// CHECK:   return cl::sycl::double2(-static_cast<const double>(v.x()), -static_cast<const double>(v.y()));
// CHECK: }
__host__ __device__ inline double2 operator-(const double2 &v) {
  return make_double2(-v.x, -v.y);
}

// CHECK: namespace syclct_operator_overloading {
// CHECK: inline cl::sycl::double2 operator+(const cl::sycl::double2 &v1,
// CHECK:                                              const cl::sycl::double2 &v2) {
// CHECK:   return cl::sycl::double2(static_cast<const double>(v1.x()) + static_cast<const double>(v2.x()), static_cast<const double>(v1.y()) + static_cast<const double>(v2.y()));
// CHECK: }
// CHECK:}  // namespace syclct_operator_overloading
__host__ __device__ inline double2 operator+(const double2 &v1,
                                             const double2 &v2) {
  return make_double2(v1.x + v2.x, v1.y + v2.y);
}

// CHECK: namespace syclct_operator_overloading {
// CHECK: inline cl::sycl::double2 operator-(const cl::sycl::double2 &v1,
// CHECK:                                              const cl::sycl::double2 &v2) {
// CHECK:   return cl::sycl::double2(static_cast<const double>(v1.x()) - static_cast<const double>(v2.x()), static_cast<const double>(v1.y()) - static_cast<const double>(v2.y()));
// CHECK: }
// CHECK: }  // namespace syclct_operator_overloading
__host__ __device__ inline double2 operator-(const double2 &v1,
                                             const double2 &v2) {
  return make_double2(v1.x - v2.x, v1.y - v2.y);
}

// CHECK: namespace syclct_operator_overloading {
// CHECK: inline cl::sycl::double2 operator*(const cl::sycl::double2 &v,
// CHECK:                                              const double &r) {
// CHECK:   return cl::sycl::double2(static_cast<const double>(v.x()) * r, static_cast<const double>(v.y()) * r);
// CHECK: }
// CHECK: }  // namespace syclct_operator_overloading
__host__ __device__ inline double2 operator*(const double2 &v,
                                             const double &r) {
  return make_double2(v.x * r, v.y * r);
}

// CHECK: namespace syclct_operator_overloading {
// CHECK: inline cl::sycl::double2 operator*(const double &r,
// CHECK:                                              const cl::sycl::double2 &v) {
// CHECK:   return cl::sycl::double2(static_cast<const double>(v.x()) * r, static_cast<const double>(v.y()) * r);
// CHECK: }
// CHECK: }  // namespace syclct_operator_overloading
__host__ __device__ inline double2 operator*(const double &r,
                                             const double2 &v) {
  return make_double2(v.x * r, v.y * r);
}

// CHECK: namespace syclct_operator_overloading {
// CHECK: inline cl::sycl::double2 operator/(const cl::sycl::double2 &v,
// CHECK:                                              const double &r) {
// CHECK:   double rinv = (double)1. / r;
// CHECK:   return cl::sycl::double2(static_cast<const double>(v.x()) * rinv, static_cast<const double>(v.y()) * rinv);
// CHECK: }
// CHECK: }  // namespace syclct_operator_overloading
__host__ __device__ inline double2 operator/(const double2 &v,
                                             const double &r) {
  double rinv = (double)1. / r;
  return make_double2(v.x * rinv, v.y * rinv);
}

// CHECK: inline double dot(const cl::sycl::double2 &v1, const cl::sycl::double2 &v2) {
// CHECK:   return (static_cast<const double>(v1.x()) * static_cast<const double>(v2.x()) + static_cast<const double>(v1.y()) * static_cast<const double>(v2.y()));
// CHECK: }
__host__ __device__ inline double dot(const double2 &v1, const double2 &v2) {
  return (v1.x * v2.x + v1.y * v2.y);
}

// CHECK: inline double cross(const cl::sycl::double2 &v1, const cl::sycl::double2 &v2) {
// CHECK:   return (static_cast<const double>(v1.x()) * static_cast<const double>(v2.y()) - static_cast<const double>(v1.y()) * static_cast<const double>(v2.x()));
// CHECK: }
__host__ __device__ inline double cross(const double2 &v1, const double2 &v2) {
  return (v1.x * v2.y - v1.y * v2.x);
}

// CHECK: inline double length(const cl::sycl::double2 &v) {
// CHECK:   return (cl::sycl::sqrt(static_cast<const double>(v.x()) * static_cast<const double>(v.x()) + static_cast<const double>(v.y()) * static_cast<const double>(v.y())));
// CHECK: }
__host__ __device__ inline double length(const double2 &v) {
  return (std::sqrt(v.x * v.x + v.y * v.y));
}

// CHECK: inline double length2(const cl::sycl::double2 &v) {
// CHECK:   return (static_cast<const double>(v.x()) * static_cast<const double>(v.x()) + static_cast<const double>(v.y()) * static_cast<const double>(v.y()));
// CHECK: }
__host__ __device__ inline double length2(const double2 &v) {
  return (v.x * v.x + v.y * v.y);
}

// CHECK: inline cl::sycl::double2 rotateCCW(const cl::sycl::double2 &v) {
// CHECK:   return cl::sycl::double2(-static_cast<const double>(v.y()), static_cast<const double>(v.x()));
// CHECK: }
__host__ __device__ inline double2 rotateCCW(const double2 &v) {
  return make_double2(-v.y, v.x);
}

// CHECK: inline cl::sycl::double2 rotateCW(const cl::sycl::double2 &v) {
// CHECK:   return cl::sycl::double2(static_cast<const double>(v.y()), -static_cast<const double>(v.x()));
// CHECK: }
__host__ __device__ inline double2 rotateCW(const double2 &v) {
  return make_double2(v.y, -v.x);
}

// CHECK: inline cl::sycl::double2 project(cl::sycl::double2 &v, const cl::sycl::double2 &u) {
// CHECK:   return syclct_operator_overloading::operator-(v , syclct_operator_overloading::operator*(dot(v, u) , u));
// CHECK: }
__host__ __device__ inline double2 project(double2 &v, const double2 &u) {
  return v - dot(v, u) * u;
}

// CHECK: void test() try {
// CHECK:   cl::sycl::double2 a;
// CHECK:   cl::sycl::double2 b;
// CHECK:   a  = syclct_operator_overloading::operator+=(a , b);
// CHECK:   syclct_operator_overloading::operator-(a);
// CHECK:   b = a;
// CHECK: }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
void test() {
  double2 a;
  double2 b;
  a += b;
  -a;
  b = a;
}
