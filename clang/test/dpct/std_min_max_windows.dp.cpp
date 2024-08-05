// UNSUPPORTED: system-linux
// RUN: %if build_lit %{icpx -c -fsycl %T/std_min_max_windows.dp.cpp -o %T/std_min_max_windows.o %}

#include <windows.h>
#include <dpct/dpct.hpp>

int main() { 
  return 0;
}
