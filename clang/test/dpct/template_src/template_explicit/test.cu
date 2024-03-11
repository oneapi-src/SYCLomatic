// RUN: echo "Empty command."
#include "test.hpp"

void bar(int parm, bool isTrue) {
  if (isTrue) {
    foo<true><<<1, 1>>>(parm);
  } else {
    foo<false><<<1, 1>>>(parm);
  }
}
