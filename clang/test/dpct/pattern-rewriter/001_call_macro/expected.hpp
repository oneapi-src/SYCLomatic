#define CALL(func, ...) func(__VA_ARGS__)

int add(int, int);
int sub(int, int);

static int test() {
  int result = add(1, 2) + sub(3, 4);
  return result;
}
