// RUN: c2s --check-unicode-security --format-range=none -out-root %T/misleading_bidirectional %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/misleading_bidirectional/misleading_bidirectional.dp.cpp --match-full-lines %s

#include <iostream>
#include <string>
__global__ void kernel() {

}

int main() {
    bool isAdmin = false;
    // CHECK: /*
    // CHECK: DPCT1094:{{[0-9]+}}: Content contains misleading bidirectional Unicode characters.
    // CHECK: */
    std::string s = "/* end admins only <202e> { ⁦*/";
    // CHECK: /*
    // CHECK: DPCT1094:{{[0-9]+}}: Content contains misleading bidirectional Unicode characters.
    // CHECK: */
    /*<202e> } ⁦if (isAdmin)⁩ ⁦ begin admins only */
        std::cout << "You are an admin.\n";
    // CHECK: /*
    // CHECK: DPCT1094:{{[0-9]+}}: Content contains misleading bidirectional Unicode characters.
    // CHECK: */
    /* end admins only <202e> { ⁦*/
    return 0;
}
