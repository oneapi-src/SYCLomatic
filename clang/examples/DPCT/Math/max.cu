__global__ void test(const unsigned long long int ull, const long long int ll,
                     const unsigned long int ul, const long int l,
                     const unsigned int u, const int i, const float f,
                     const double d) {
  // Start
  /* 1 */ max(ull /*const unsigned long long int*/, ll /*const long long int*/);
  /* 2 */ max(ll /*const long long int*/, ull /*const unsigned long long int*/);
  /* 3 */ max(ull /*const unsigned long long int*/,
              ull /*const unsigned long long int*/);
  /* 4 */ max(ll /*const long long int*/, ll /*const long long int*/);
  /* 5 */ max(ul /*const unsigned long int*/, l /*const long int*/);
  /* 6 */ max(l /*const long int*/, ul /*const unsigned long int*/);
  /* 7 */ max(ul /*const unsigned long int*/, ul /*const unsigned long int*/);
  /* 8 */ max(l /*const long int*/, l /*const long int*/);
  /* 9 */ max(u /*const unsigned int*/, i /*const int*/);
  /* 10 */ max(i /*const int*/, u /*const unsigned int*/);
  /* 11 */ max(u /*const unsigned int*/, u /*const unsigned int*/);
  /* 12 */ max(i /*const int*/, i /*const int*/);
  /* 13 */ max(f /*const float*/, f /*const float*/);
  /* 14 */ max(d /*const double*/, f /*const float*/);
  /* 15 */ max(f /*const float*/, d /*const double*/);
  /* 16 */ max(d /*const double*/, d /*const double*/);
  // End
}
