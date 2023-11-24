#ifndef __DEBUG_HELPER__
#define __DEBUG_HELPER__

#include "json.hpp"
#include "schema.hpp"
#include <memory>
#ifdef __NVCC__
#include <cuda_runtime.h>
#else
#include <dpct/dpct.hpp>
#include <sycl/sycl.hpp>
#endif
namespace dpct {
namespace experimental {

#ifdef __NVCC__
inline void synchronize(cudaStream_t stream) {
  cudaStreamSynchronize(stream);
}

template <class... Args>
void gen_prolog_API_CP(const std::string &api_name, cudaStream_t stream,
                       Args... args) {
  synchronize(stream);
  gen_log_API_CP(api_name, args...);
}

template <class... Args>
void gen_epilog_API_CP(const std::string &api_name, cudaStream_t stream,
                       Args... args) {
  gen_prolog_API_CP(api_name, stream, args...);
}
#else
void synchronize(sycl::queue *q) { q->wait(); }

template <class... Args>
void gen_prolog_API_CP(const std::string &api_name, sycl::queue *queue,
                       Args... args) {
  synchronize(queue);
  gen_log_API_CP(api_name, args...);
}

template <class... Args>
void gen_epilog_API_CP(const std::string &api_name, sycl::queue *queue,
                       Args... args) {
  gen_prolog_API_CP(api_name, queue, args...);
}
#endif

void gen_data_CP(const std::string &data_name, std::shared_ptr<Schema> schema,
                 long value, size_t size = 0) {
  std::string detail = "";
  switch (schema->get_val_type()) {
  case ValType::SCALAR:
    get_val_from_addr(detail, schema, (void *)&value, size);
    break;
  case ValType::ARRAY:
  case ValType::POINTER:
    get_val_from_addr(detail, schema, (void *)value, size);
    break;
  case ValType::POINTERTOPOINTER:
    get_val_from_addr(detail, schema, *(void **)value, size);
    break;
  };
  std::cout << "Data name: " << data_name << " \n Data: \n"
            << detail << std::endl;
}

} // namespace experimental
} // namespace dpct
#endif // End of __DEBUG_HELPER__
