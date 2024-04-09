
namespace dpct {
namespace experimental {

namespace detail {

#define emit_error_msg(msg)                                                    \
  {                                                                            \
    std::cerr << "Failed at:" << __FILE__ << "\nLine number is : " << __LINE__ \
              << "\n" msg << std::endl;                                        \
  }

// The class json has 1 public member functions to validate input json by is_valid function.
class json {
private:
  std::string original_json;
  std::string cur_key = "";
  const char *begin = nullptr;
  const char *cur_p = nullptr;
  const char *end = nullptr;

  bool validate();
  bool parse_str(std::string &ret);
  bool parse_number(char first, std::string &out);
  bool is_number(char c);
  void ignore_space() {
    while (cur_p != end && (*cur_p == ' ' || *cur_p == '\t' || *cur_p == '\r' ||
                            *cur_p == '\n'))
      cur_p++;
  }
  char next() { return cur_p != end ? *cur_p++ : 0; }
  char peek() { return cur_p != end ? *cur_p : 0; }

public:
  json(const std::string &json)
      : original_json(json), begin(original_json.c_str()),
        cur_p(original_json.c_str()),
        end(original_json.c_str() + original_json.size()) {}

  bool is_valid() { return validate(); }
};

inline bool json::parse_str(std::string &str) {
  char prev = peek();
  while (peek() != '"' || (peek() == '"' && prev == '\\')) {
    prev = peek();
    if (cur_p == end) {
      return false;
    }
    str += next();
  }
  next();
  return true;
}
inline bool json::is_number(char c) {
  return (c == '0') || (c == '1') || (c == '2') || (c == '3') || (c == '4') ||
         (c == '5') || (c == '6') || (c == '7') || (c == '8') || (c == '9') ||
         (c == '-') || (c == '+') || (c == '.') || (c == 'e') || (c == 'E');
}

// Parse the char value one by one to generate the number string.
inline bool json::parse_number(char c, std::string &number) {
  number += c;
  while (is_number(peek())) {
    number += next();
  }
  try {
    size_t pos;
    std::stod(number, &pos);
    return pos == number.length();
  } catch (const std::invalid_argument &ia) {
    emit_error_msg(
        "[CODEPIN VALIDATOR]: Parsing number value failed. Value is " + number);
    return false;
  } catch (const std::out_of_range &oor) {
    emit_error_msg(
        "[CODEPIN VALIDATOR]: Parsing number value failed. Value is " + number);
    return false;
  }
}
inline bool json::validate() {
  ignore_space();
  char c = next();
  switch (c) {
  case '[': {
    for (;;) {
      ignore_space();
      if (!validate())
        return false;
      ignore_space();
      switch (next()) {
      case ',':
        continue;
      case ']':
        return true;
      default:
        emit_error_msg(
            "[CODEPIN VALIDATOR]: Parsing JSON value error. The key is " +
            cur_key);
        return false;
      }
    }
  } break;

  case '{': {
    for (;;) {
      ignore_space();
      if (peek() == '"') {
        std::string key = "";
        next();
        if (!parse_str(key)) {
          emit_error_msg(
              "[CODEPIN VALIDATOR]: key value of a JSON need to be wrapped in "
              "\". Please check the JSON file format.");
          return false;
        } else {
          cur_key = "\"" + key + "\"";
        }
      }
      ignore_space();
      if (next() == ':') {
        if (!validate()) {
          emit_error_msg(
              "[CODEPIN VALIDATOR]: Can not parse value, the JSON key is " +
              cur_key + ".\n");
          return false;
        }
      }
      ignore_space();
      switch (next()) {
      case ',': {
        continue;
      }
      case '}': {
        return true;
      }
      default:
        emit_error_msg("[CODEPIN VALIDATOR]: The " + cur_key +
                       " value pair should be end with '}' or ','.\n");
        return false;
      }
    }
  } break;
  case '"': {
    std::string str = "";
    if (!parse_str(str)) {
      emit_error_msg("[CODEPIN VALIDATOR]: The Json is invalid after " +
                     cur_key + "\n");
      return false;
    }
    return true;
  }
  case 't':
    if (next() == 'r' && next() == 'u' && next() == 'e') {
      return true;
    }
    emit_error_msg("[CODEPIN VALIDATOR]: The bool value of " + cur_key +
                   " should be \"true\", please check "
                   "the spelling.");
    return false;
  case 'f':
    if (next() == 'a' && next() == 'l' && next() == 's' && next() == 'e') {
      return true;
    }
    emit_error_msg("[CODEPIN VALIDATOR]: The bool value of " + cur_key +
                   " should be \"false\", please "
                   "check the spelling.");
    return false;
  default:
    if (is_number(c)) { // When the value is not string, bool value, dict and
                        // array. Then it should be literal number
      std::string num = "";
      parse_number(c, num);
      return true;
    }
    emit_error_msg("[CODEPIN VALIDATOR]: Unkown JSON type, the last key is " +
                   cur_key +
                   ". Please check the format JSON "
                   "format.\n");
    return false;
  }
  return true;
}
} // namespace detail
} // namespace experimental
} // namespace dpct