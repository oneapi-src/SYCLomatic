// RUN: dpct --format-range=none  -out-root %T/debug_feature/test %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only --std=c++14
// RUN: FileCheck --input-file %T/debug_feature/test/output.txt --match-full-lines %s

#include <iostream>
#include <vector>
// #include "dpct/dpl_utils.hpp"
namespace test {
class BDebug {
public:
  const static int data_a=5;

private:
  double data_b[5];
  void output() { std::cout << "just output" << std::endl; }
};

class ADebug {
public:
  BDebug***** data_a[5];
  void output() { std::cout << "just output" << std::endl; }

private:
double data_b[5];
};


class CDebug : public ADebug {
public:
  int data_a[5];

private:
  double data_b[5];
};

class DDebug : public CDebug {
public:
  int data_a[5];

private:
  double data_b[5];
};

class EDebug : public CDebug, public BDebug {
public:
  int data_a[5];

private:
  double data_b[5];

};

union UDebug {
  int intValue[4];
  float floatValue[4];
};

class FDebug : virtual public CDebug {
public:
  int data_a[5];

private:
  double data_b[5];
};
}


namespace test_namepsace {
class B {
public:
  B(int b) : b(b) {}
 
private:
  int b;
};
class A {
public:
  A(int a) : b(new B(3)), a(a) {}
 
private:
  B *b;
  int a;
};
 

}; // namespace test_namepsace

void faketest(test::ADebug a){
return;
}

void faketest(std::vector<int> a){
return;
}

void faketest(test::ADebug* a){
return;
}

void faketest(int* a){
return;
}

void faketest(test_namepsace::A a){
return;
}





int main() {
  test::ADebug a[5];
  test::ADebug* a0;
  std::vector<int> avec(5,0);
  // faketest(a[0]);
  // faketest(a0);
  // faketest(avec);
faketest(a0);
  test::BDebug b;
  test::BDebug b2;
  test::CDebug c;
  test::DDebug d;
  test::EDebug e;
  test::UDebug u;
  int *i;
  faketest(i);
  test_namepsace::A ta(1);
  faketest(ta);

  return 1;

}

