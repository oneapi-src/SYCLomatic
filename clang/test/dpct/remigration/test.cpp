// UNSUPPORTED: system-windows

// RUN: cd %T
// RUN: mkdir out
// RUN: cp %S/LastMigration.yaml out/MainSourceFiles.yaml
// RUN: sed -i "s|PATH_PLACEHOLDER|$(pwd)|g" out/MainSourceFiles.yaml
// RUN: cp %S/UpstreamChanges.yaml .
// RUN: cp %S/UserChanges.yaml .
// RUN: cp %S/src.txt test.cu

// RUN: dpct --out-root out ./test.cu --format-range=none --remigration
// RUN: diff %S/expect.txt out/test.dp.cpp
