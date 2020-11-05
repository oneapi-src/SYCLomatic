// RUN: dpct --format-range=none -out-root %T/macro-nested-cuda-attr %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/macro-nested-cuda-attr/macro-nested-cuda-attr.dp.cpp --match-full-lines %s

#define HAVE_CUDA

#ifdef HAVE_CUDA
  // CHECK: #define HOST_DEVICE
  // CHECK-NEXT: #define HOST_DEVICE_CUDA
  #define HOST_DEVICE __host__ __device__
  #define HOST_DEVICE_CUDA __host__ __device__
  #define HOST_DEVICE_CLASS
  #define HOST_DEVICE_END
  // CHECK: #define DEVICE
  #define DEVICE __device__
  #define DEVICE_END
  //#define HOST __host__
  #define HOST_END
  // CHECK: #define GLOBAL
  #define GLOBAL __global__
#elif HAVE_OPENMP_TARGET
  #define HOST_DEVICE _Pragma("omp declare target")
  #define HOST_DEVICE_CUDA
  #define HOST_DEVICE_CLASS _Pragma("omp declare target")
  #define HOST_DEVICE_END _Pragma("omp end declare target")
  //#define HOST_DEVICE #pragma omp declare target
  //#define HOST_DEVICE_END #pragma omp end declare target
  //#define DEVICE #pragma omp declare target
  //#define DEVICE_END #pragma omp end declare target
  //#define HOST
  #define HOST_END
  #define GLOBAL
#else
  #define HOST_DEVICE
  #define HOST_DEVICE_CUDA
  #define HOST_DEVICE_CLASS
  #define HOST_DEVICE_END
  #define DEVICE
  #define DEVICE_END
  //#define HOST
  #define HOST_END
  #define GLOBAL
#endif

#include <cmath>

HOST_DEVICE_CLASS
class MC_Vector {
public:
  double x;
  double y;
  double z;

  // CHECK: HOST_DEVICE_CUDA
  HOST_DEVICE_CUDA
  MC_Vector() : x(0), y(0), z(0) {}
  // CHECK: HOST_DEVICE_CUDA
  HOST_DEVICE_CUDA
  MC_Vector(double a, double b, double c) : x(a), y(b), z(c) {}

  // CHECK: HOST_DEVICE_CUDA
  HOST_DEVICE_CUDA
  MC_Vector &operator=(const MC_Vector &tmp) {
    if (this == &tmp) {
      return *this;
    }

    x = tmp.x;
    y = tmp.y;
    z = tmp.z;

    return *this;
  }

  // CHECK: HOST_DEVICE_CUDA
  HOST_DEVICE_CUDA
  bool operator==(const MC_Vector &tmp) {
    return tmp.x == x && tmp.y == y && tmp.z == z;
  }

  // CHECK: HOST_DEVICE_CUDA
  HOST_DEVICE_CUDA
  MC_Vector &operator+=(const MC_Vector &tmp) {
    x += tmp.x;
    y += tmp.y;
    z += tmp.z;
    return *this;
  }

  // CHECK: HOST_DEVICE_CUDA
  HOST_DEVICE_CUDA
  MC_Vector &operator-=(const MC_Vector &tmp) {
    x -= tmp.x;
    y -= tmp.y;
    z -= tmp.z;
    return *this;
  }

  // CHECK: HOST_DEVICE_CUDA
  HOST_DEVICE_CUDA
  MC_Vector &operator*=(const double scalar) {
    x *= scalar;
    y *= scalar;
    z *= scalar;
    return *this;
  }

  // CHECK: HOST_DEVICE_CUDA
  HOST_DEVICE_CUDA
  MC_Vector &operator/=(const double scalar) {
    x /= scalar;
    y /= scalar;
    z /= scalar;
    return *this;
  }

  // CHECK: HOST_DEVICE_CUDA
  HOST_DEVICE_CUDA
  const MC_Vector operator+(const MC_Vector &tmp) const {
    return MC_Vector(x + tmp.x, y + tmp.y, z + tmp.z);
  }

  // CHECK: HOST_DEVICE_CUDA
  HOST_DEVICE_CUDA
  const MC_Vector operator-(const MC_Vector &tmp) const {
    return MC_Vector(x - tmp.x, y - tmp.y, z - tmp.z);
  }

  // CHECK: HOST_DEVICE_CUDA
  HOST_DEVICE_CUDA
  const MC_Vector operator*(const double scalar) const {
    return MC_Vector(scalar * x, scalar * y, scalar * z);
  }

  // CHECK: HOST_DEVICE_CUDA
  HOST_DEVICE_CUDA
  inline double Length() const { return std::sqrt(x * x + y * y + z * z); }

  // Distance from this vector to another point.
  HOST_DEVICE_CUDA
  inline double Distance(const MC_Vector &vv) const { return std::sqrt((x - vv.x) * (x - vv.x) + (y - vv.y) * (y - vv.y) + (z - vv.z) * (z - vv.z)); }

  // CHECK: HOST_DEVICE_CUDA
  HOST_DEVICE_CUDA
  inline double Dot(const MC_Vector &tmp) const {
    return this->x * tmp.x + this->y * tmp.y + this->z * tmp.z;
  }

  // CHECK: HOST_DEVICE_CUDA
  HOST_DEVICE_CUDA
  inline MC_Vector Cross(const MC_Vector &v) const {
    return MC_Vector(y * v.z - z * v.y,
                     z * v.x - x * v.z,
                     x * v.y - y * v.x);
  }
};
// CHECK: HOST_DEVICE_END
HOST_DEVICE_END

class MC_Particle;
class MC_Domain;
class MC_Location;
class MC_Vector;
class DirectionCosine;
class MC_Nearest_Facet;
class Subfacet_Adjacency;
class MonteCarlo;

// CHECK: HOST_DEVICE
HOST_DEVICE
MC_Nearest_Facet MCT_Nearest_Facet(
    MC_Particle *mc_particle,
    MC_Location &location,
    MC_Vector &coordinate,
    const DirectionCosine *direction_cosine,
    double distance_threshold,
    double current_best_distance,
    bool new_segment,
    MonteCarlo *monteCarlo);
// CHECK: HOST_DEVICE_END
HOST_DEVICE_END

// CHECK: HOST_DEVICE
HOST_DEVICE
void MCT_Generate_Coordinate_3D_G(
    unsigned long *random_number_seed,
    int domain_num,
    int cell,
    MC_Vector &coordinate,
    MonteCarlo *monteCarlo);
// CHECK: HOST_DEVICE_END
HOST_DEVICE_END

// CHECK: HOST_DEVICE
HOST_DEVICE
MC_Vector MCT_Cell_Position_3D_G(
    const MC_Domain &domain,
    int cell_index);
// CHECK: HOST_DEVICE_END
HOST_DEVICE_END

// CHECK: HOST_DEVICE
HOST_DEVICE
Subfacet_Adjacency &MCT_Adjacent_Facet(const MC_Location &location, MC_Particle &mc_particle, MonteCarlo *monteCarlo);
// CHECK: HOST_DEVICE_END
HOST_DEVICE_END

// CHECK: HOST_DEVICE
HOST_DEVICE
void MCT_Reflect_Particle(MonteCarlo *mcco, MC_Particle &particle);
// CHECK: HOST_DEVICE_END
HOST_DEVICE_END

#if defined(HAVE_CUDA)
void warmup_kernel();
int ThreadBlockLayout(dim3 &grid, dim3 &block, int num_particles);
// CHECK: DEVICE
DEVICE
int getGlobalThreadID();
#endif

// CHECK: GLOBAL
GLOBAL
void globalFunc() {}

