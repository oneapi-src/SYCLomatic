__global__ void test(const unsigned long long int ull, const long long int ll,
                     const unsigned long int ul, const long int l,
                     const unsigned int u, const int i, const float f,
                     const double d) {
  // Start
  /* 1 */ min(ull /*const unsigned long long int*/, ll /*const long long int*/);
  /* 2 */ min(ll /*const long long int*/, ull /*const unsigned long long int*/);
  /* 3 */ min(ull /*const unsigned long long int*/,
              ull /*const unsigned long long int*/);
  /* 4 */ min(ll /*const long long int*/, ll /*const long long int*/);
  /* 5 */ min(ul /*const unsigned long int*/, l /*const long int*/);
  /* 6 */ min(l /*const long int*/, ul /*const unsigned long int*/);
  /* 7 */ min(ul /*const unsigned long int*/, ul /*const unsigned long int*/);
  /* 8 */ min(l /*const long int*/, l /*const long int*/);
  /* 9 */ min(u /*const unsigned int*/, i /*const int*/);
  /* 10 */ min(i /*const int*/, u /*const unsigned int*/);
  /* 11 */ min(u /*const unsigned int*/, u /*const unsigned int*/);
  /* 12 */ min(i /*const int*/, i /*const int*/);
  /* 13 */ min(f /*const float*/, f /*const float*/);
  /* 14 */ min(d /*const double*/, f /*const float*/);
  /* 15 */ min(f /*const float*/, d /*const double*/);
  /* 16 */ min(d /*const double*/, d /*const double*/);
  // End
}
