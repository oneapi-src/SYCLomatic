#define CALL(func, ...) func(__VA_ARGS__)

int add(int, int, int);
int sub(int, int);

static int test() {
  int result = add(1, 2, 3) + sub(4, 5);
  return result;
}
