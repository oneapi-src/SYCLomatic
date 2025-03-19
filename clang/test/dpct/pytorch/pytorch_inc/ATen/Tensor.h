#pragma once
namespace at {
class Tensor {
public:
  int get_device() const { return 0; }
};
} // namespace at
