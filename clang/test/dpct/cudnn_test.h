#ifndef CUDNN_TEST_H
#define CUDNN_TEST_H

enum cudnnFooEnum {
  CUDNN_FOO_VAL = 0,
};
typedef cudnnFooEnum cudnnFooType;

cudnnFooType cudnnAAA();

class cudnnCLASS{};

template<typename T>
class cudnnTemplateCLASS{};
#endif
