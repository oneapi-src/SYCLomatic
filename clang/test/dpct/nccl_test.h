#ifndef NCCL_TEST_H
#define NCCL_TEST_H

int ncclAAA();

enum ncclFooEnum {
  NCCL_FOO_VAL = 0,
};
typedef ncclFooEnum ncclFooType;
class ncclCLASS{};

template<typename T>
class ncclTemplateCLASS{};
#endif