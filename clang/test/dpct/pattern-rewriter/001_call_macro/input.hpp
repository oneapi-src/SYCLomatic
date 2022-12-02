#define CALL(func, ...) func(__VA_ARGS__)

int add(int, int);
int sub(int, int);

static int test() {
  int result = CALL(add, 1, 2) + CALL(sub, 3, 4);
  return result;
}
