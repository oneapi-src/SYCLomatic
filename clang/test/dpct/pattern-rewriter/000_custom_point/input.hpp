struct Point {
  float x;
  float y;
  float z;

  float example() { return 5.0f; }
};

static int f() {
  float other;
  other = 5.0f;
  return static_cast<int>(other);
}
