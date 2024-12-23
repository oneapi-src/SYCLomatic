#include <cmath>

namespace c10 {
using DeviceIndex = int8_t;

namespace cuda {
DeviceIndex device_count();
DeviceIndex device_count_ensure_non_zero();
DeviceIndex current_device();
void set_device(DeviceIndex device);
DeviceIndex ExchangeDevice(DeviceIndex device);
DeviceIndex MaybeExchangeDevice(DeviceIndex to_device);
} // namespace cuda
} // namespace c10
