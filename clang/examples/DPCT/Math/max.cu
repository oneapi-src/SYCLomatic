__global__ void test(const unsigned long long int ull, const long long int ll,
                     const unsigned long int ul, const long int l,
                     const unsigned int u, const int i, const float f,
                     const double d) {
  // Start
  max(ull /*const unsigned long long int*/, ll /*const long long int*/);
  max(ll /*const long long int*/, ull /*const unsigned long long int*/);
  max(ull /*const unsigned long long int*/,
      ull /*const unsigned long long int*/);
  max(ll /*const long long int*/, ll /*const long long int*/);
  max(ul /*const unsigned long int*/, l /*const long int*/);
  max(l /*const long int*/, ul /*const unsigned long int*/);
  max(ul /*const unsigned long int*/, ul /*const unsigned long int*/);
  max(l /*const long int*/, l /*const long int*/);
  max(u /*const unsigned int*/, i /*const int*/);
  max(i /*const int*/, u /*const unsigned int*/);
  max(u /*const unsigned int*/, u /*const unsigned int*/);
  max(i /*const int*/, i /*const int*/);
  max(f /*const float*/, f /*const float*/);
  max(d /*const double*/, f /*const float*/);
  max(f /*const float*/, d /*const double*/);
  max(d /*const double*/, d /*const double*/);
  // End
}
