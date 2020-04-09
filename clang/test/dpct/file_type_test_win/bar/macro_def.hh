#ifndef DECLAREMACRO_HH
#define DECLAREMACRO_HH

// CHECK:#define HOST_DEVICE
// CHECK-NEXT:#define HOST_DEVICE_CUDA
// CHECK-NEXT:#define HOST_DEVICE_END
// CHECK-NEXT:#define DEVICE
// CHECK-NEXT:#define DEVICE_END
#define HOST_DEVICE __host__ __device__
#define HOST_DEVICE_CUDA __host__ __device__
#define HOST_DEVICE_END
#define DEVICE __device__
#define DEVICE_END
#endif
