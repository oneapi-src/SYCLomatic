#include "macro_def.hh"

// CHECK:SYCL_EXTERNAL HOST_DEVICE
HOST_DEVICE
void util_bar(void);
HOST_DEVICE_END

class FooQueue
{
  public:
    // CHECK:    SYCL_EXTERNAL HOST_DEVICE_CUDA
    HOST_DEVICE_CUDA
    void push(int a, int b);
};

template <class T>
class qs_vector
{
 public:

   const T& operator[]( int index ) const
   {
      return _data[index];
   }
   T& operator[]( int index )
   {
      return _data[index];
   }
 private:
   T* _data;
};


class SubFooReaction
{
 public:
   // CHECK:   SYCL_EXTERNAL HOST_DEVICE_CUDA
   HOST_DEVICE_CUDA
   void fooCollision();
};

class SubFooSpecies
{
 public:
   qs_vector<SubFooReaction> _reactions;
};

class SubFooIsotope
{
 public:
   qs_vector<SubFooSpecies> _species;
};

class SubFoo
{
 public:
   qs_vector<SubFooIsotope> _isotopes;
};

class TopFooConf
{
public:
    SubFoo* fooData;
private:
   TopFooConf(const TopFooConf&);
   TopFooConf& operator=(const TopFooConf&);
};
