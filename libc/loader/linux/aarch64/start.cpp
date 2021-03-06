//===-- Implementation of crt for aarch64 ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "config/linux/app.h"
#include "src/__support/OSUtil/syscall.h"

#include <linux/auxvec.h>
#include <linux/elf.h>
#include <stdint.h>
#include <sys/syscall.h>

extern "C" int main(int, char **, char **);

// Source documentation:
// https://github.com/ARM-software/abi-aa/tree/main/sysvabi64

namespace __llvm_libc {

AppProperties app;

} // namespace __llvm_libc

using __llvm_libc::app;

struct Args {
  // In the ARM64 ABI, arguments are usually passed in registers.  x0 is a
  // doubleword register, so this is 64 bit.
  uint64_t argc;

  // C++ Doesn't have flexible arrays: P1039 proposes to fix this, but for now
  // we just fake it.  Even if argc is zero, "argv[argc] shall be a null
  // pointer" (ISO C 5.1.2.2.1) so one is fine.
  uint64_t argv[1];
};

// TODO: Would be nice to use the aux entry structure from elf.h when available.
struct AuxEntry {
  uint64_t type;
  uint64_t value;
};

extern "C" void _start() {
  uintptr_t *frame_ptr =
      reinterpret_cast<uintptr_t *>(__builtin_frame_address(0));

  // Skip the Frame Pointer and the Link Register
  // https://github.com/ARM-software/abi-aa/blob/main/aapcs64/aapcs64.rst
  // Section 6.2.3

  Args *args = reinterpret_cast<Args *>(frame_ptr + 2);

  // After the argv array, is a 8-byte long NULL value before the array of env
  // values. The end of the env values is marked by another 8-byte long NULL
  // value. We step over it (the "+ 1" below) to get to the env values.
  uint64_t *env_ptr = args->argv + args->argc + 1;
  uint64_t *env_end_marker = env_ptr;
  while (*env_end_marker)
    ++env_end_marker;

  // After the env array, is the aux-vector. The end of the aux-vector is
  // denoted by an AT_NULL entry.
  for (AuxEntry *aux_entry = reinterpret_cast<AuxEntry *>(env_end_marker + 1);
       aux_entry->type != AT_NULL; ++aux_entry) {
    switch (aux_entry->type) {
    case AT_PAGESZ:
      app.pageSize = aux_entry->value;
      break;
    default:
      break; // TODO: Read other useful entries from the aux vector.
    }
  }

  // TODO: Init TLS

  __llvm_libc::syscall(SYS_exit,
                       main(args->argc, reinterpret_cast<char **>(args->argv),
                            reinterpret_cast<char **>(env_ptr)));
}
