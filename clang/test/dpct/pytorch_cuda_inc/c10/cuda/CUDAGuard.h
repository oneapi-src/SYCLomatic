#include <string>

namespace std {
template <typename T>
class optional {
public:
  optional() {}
};
} // namespace std

namespace c10 {
class Device {
public:
  Device(std::string str) {}
};

namespace cuda {
class OptionalCUDAGuard {
public:
  OptionalCUDAGuard(std::optional<c10::Device> device) {}
};
} // namespace cuda
} // namespace c10
