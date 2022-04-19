
//CHECK:namespace dpct_operator_overloading {
//CHECK-EMPTY:
//CHECK-NEXT:std::ostream &operator<<(std::ostream &os, const sycl::float2 &x)
//CHECK-NEXT:{
//CHECK-NEXT:    os << x.x();
//CHECK-NEXT:    return os;
//CHECK-NEXT:}
//CHECK-NEXT:} // namespace dpct_operator_overloading
//CHECK-EMPTY:

// Test description:
// This test is to cover the case that no newline character or ";" after "}"
std::ostream &operator<<(std::ostream &os, const cuComplex &x)
{
    os << x.x;
    return os;
} // Do not add code or new lines after this line. EOF must immediately after this line.