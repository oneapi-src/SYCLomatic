struct Point {
  float x_;
  float &x() { return x_; }
  float y_;
  float &y() { return y_; }
  float z_;
  float &z() { return z_; }

  float example() { return 5.0f; }
};

static int f() {
  float other;
  other = 5.0f;
  return static_cast<int>(other);
}
