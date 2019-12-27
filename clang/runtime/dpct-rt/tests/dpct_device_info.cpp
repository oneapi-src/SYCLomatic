// This test case is to get device info of current device in the dpct device manager.
// RUN: dpcpp dpct_device_info.cpp -o dpct_device_info -I/path/to/sycl/include -I/path/to/dpct/include
// E.g:
// export SYCL_DEVICE_TYPE=GPU; ./dpct_device_info
// gpu  device: Selected device: Intel(R) Gen9 HD Graphics NEO
//             -> Device vendor: Intel(R) Corporation
//
// export SYCL_DEVICE_TYPE=CPU; ./dpct_device_info
// cpu  device: Selected device: Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz
//              -> Device vendor: Intel(R) Corporation
//
// export SYCL_DEVICE_TYPE=HOST; ./dpct_device_info
// host device: Selected device: SYCL host device
//              -> Device vendor:

#include <CL/sycl.hpp>
#include <iostream>
#include <dpct/dpct.hpp>

using namespace cl::sycl;

void dev_info_output(const device& dev, std::string selector_name) {
    std::cout << selector_name << ": Selected device: " << dev.get_info<info::device::name>() << "\n";
    std::cout << "            -> Device vendor: " << dev.get_info<info::device::vendor>() << "\n";
}

int main() {

    auto device = dpct::get_default_queue().get_device();

    if(device.is_gpu())
        dev_info_output(device, "gpu  device");
    if(device.is_cpu())
        dev_info_output(device, "cpu  device");
    if(device.is_host())
        dev_info_output(device, "host device");
    if(device.is_accelerator())
        dev_info_output(device, "accelerator device");
    return 0;
}

