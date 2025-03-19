#include <string>

namespace std {
template <typename T>
class optional {
public:
  optional() {}
};
} // namespace std

namespace c10 {
using DeviceIndex = int8_t;
class Device {
public:
  Device(std::string str) {}
};

namespace cuda {
class OptionalCUDAGuard {
public:
  OptionalCUDAGuard(std::optional<c10::Device> device) {}
};
struct CUDAGuard {
  explicit CUDAGuard() = delete;
  explicit CUDAGuard(DeviceIndex device_index) {}
  explicit CUDAGuard(Device device) {}
  CUDAGuard(const CUDAGuard&) = delete;
  CUDAGuard& operator=(const CUDAGuard&) = delete;
  CUDAGuard(CUDAGuard&& other) = delete;
  CUDAGuard& operator=(CUDAGuard&& other) = delete;
  ~CUDAGuard() = default;
};
} // namespace cuda
} // namespace c10
