// RUN: dpct --out-root %T/inc_migration_err_msg/out1 %s --cuda-include-path="%cuda-path/include" --use-dpcpp-extensions=c_cxx_standard_library
// RUN: dpct --out-root %T/inc_migration_err_msg/out1 %s --cuda-include-path="%cuda-path/include" 2>%T/inc_migration_err_msg/out1/err.txt  || true
// RUN: grep "\-\-analysis-scope-path=" %T/inc_migration_err_msg/out1/err.txt
// RUN: grep "\-\-use-dpcpp-extensions=c_cxx_standard_library" %T/inc_migration_err_msg/out1/err.txt
// RUN: rm -rf %T/inc_migration_err_msg/out1

// RUN: dpct --out-root %T/inc_migration_err_msg/out2 %s --cuda-include-path="%cuda-path/include" --no-dpcpp-extensions=enqueued_barriers,device_info
// RUN: dpct --out-root %T/inc_migration_err_msg/out2 %s --cuda-include-path="%cuda-path/include" 2>%T/inc_migration_err_msg/out2/err.txt  || true
// RUN: grep "\-\-analysis-scope-path=" %T/inc_migration_err_msg/out2/err.txt
// RUN: grep "\-\-no-dpcpp-extensions=enqueued_barriers,device_info" %T/inc_migration_err_msg/out2/err.txt
// RUN: rm -rf %T/inc_migration_err_msg/out2

// RUN: dpct --out-root %T/inc_migration_err_msg/out3 %s --cuda-include-path="%cuda-path/include" --use-experimental-features=free-function-queries,local-memory-kernel-scope-allocation,logical-group,nd_range_barrier
// RUN: dpct --out-root %T/inc_migration_err_msg/out3 %s --cuda-include-path="%cuda-path/include" 2>%T/inc_migration_err_msg/out3/err.txt  || true
// RUN: grep "\-\-analysis-scope-path=" %T/inc_migration_err_msg/out3/err.txt
// RUN: grep "\-\-use-experimental-features=free-function-queries,nd_range_barrier,local-memory-kernel-scope-allocation,logical-group" %T/inc_migration_err_msg/out3/err.txt
// RUN: rm -rf %T/inc_migration_err_msg/out3

void foo() {
  float2 f2;
}
