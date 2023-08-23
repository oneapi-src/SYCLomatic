__global__ void test(const unsigned long long int ull, const long long int ll,
                     const unsigned long int ul, const long int l,
                     const unsigned int u, const int i, const float f,
                     const double d) {
  // Start
  min(ull /*const unsigned long long int*/, ll /*const long long int*/);
  min(ll /*const long long int*/, ull /*const unsigned long long int*/);
  min(ull /*const unsigned long long int*/,
      ull /*const unsigned long long int*/);
  min(ll /*const long long int*/, ll /*const long long int*/);
  min(ul /*const unsigned long int*/, l /*const long int*/);
  min(l /*const long int*/, ul /*const unsigned long int*/);
  min(ul /*const unsigned long int*/, ul /*const unsigned long int*/);
  min(l /*const long int*/, l /*const long int*/);
  min(u /*const unsigned int*/, i /*const int*/);
  min(i /*const int*/, u /*const unsigned int*/);
  min(u /*const unsigned int*/, u /*const unsigned int*/);
  min(i /*const int*/, i /*const int*/);
  min(f /*const float*/, f /*const float*/);
  min(d /*const double*/, f /*const float*/);
  min(f /*const float*/, d /*const double*/);
  min(d /*const double*/, d /*const double*/);
  // End
}
