#pragma once

/**
 * @file Config.hpp
 * @brief Simple YAML-like configuration parser (header-only, no dependencies)
 *
 * This is a minimal config parser that reads key-value pairs from a YAML file.
 * Only supports flat structure (no nested objects) and basic types.
 */

#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace unina {

/**
 * @brief Simple configuration container
 *
 * Reads YAML-like config files with key: value pairs.
 * Supports: strings, floats, ints, bools, and string arrays.
 */
class Config {
public:
  Config() = default;

  /**
   * @brief Load configuration from file
   * @param path Path to config file
   * @return true if loaded successfully
   */
  bool load(const std::string &path) {
    std::ifstream file(path);
    if (!file.is_open()) {
      return false;
    }

    std::string line;
    std::string current_array_key;
    bool in_array = false;

    while (std::getline(file, line)) {
      // Skip empty lines and comments
      if (line.empty() || line[0] == '#') {
        continue;
      }

      // Check for array item (starts with "  - ")
      if (in_array && line.size() > 4 && line.substr(0, 4) == "  - ") {
        std::string value = line.substr(4);
        // Remove quotes if present
        if (value.size() >= 2 && value.front() == '"' && value.back() == '"') {
          value = value.substr(1, value.size() - 2);
        }
        arrays_[current_array_key].push_back(value);
        continue;
      }

      // Check for key: value pair
      size_t colon_pos = line.find(':');
      if (colon_pos != std::string::npos) {
        std::string key = trim(line.substr(0, colon_pos));
        std::string value = trim(line.substr(colon_pos + 1));

        // Check if this starts an array
        if (value.empty()) {
          current_array_key = key;
          in_array = true;
          arrays_[key] = {};
          continue;
        }

        in_array = false;

        // Remove quotes if present
        if (value.size() >= 2 && value.front() == '"' && value.back() == '"') {
          value = value.substr(1, value.size() - 2);
        }

        values_[key] = value;
      }
    }

    file.close();
    return true;
  }

  // Getters with defaults
  std::string get_string(const std::string &key,
                         const std::string &default_val = "") const {
    auto it = values_.find(key);
    return (it != values_.end()) ? it->second : default_val;
  }

  float get_float(const std::string &key, float default_val = 0.0f) const {
    auto it = values_.find(key);
    if (it == values_.end())
      return default_val;
    try {
      return std::stof(it->second);
    } catch (...) {
      return default_val;
    }
  }

  int get_int(const std::string &key, int default_val = 0) const {
    auto it = values_.find(key);
    if (it == values_.end())
      return default_val;
    try {
      return std::stoi(it->second);
    } catch (...) {
      return default_val;
    }
  }

  bool get_bool(const std::string &key, bool default_val = false) const {
    auto it = values_.find(key);
    if (it == values_.end())
      return default_val;
    std::string val = it->second;
    return (val == "true" || val == "True" || val == "1" || val == "yes");
  }

  std::vector<std::string> get_array(const std::string &key) const {
    auto it = arrays_.find(key);
    return (it != arrays_.end()) ? it->second : std::vector<std::string>{};
  }

  bool has(const std::string &key) const {
    return values_.find(key) != values_.end() ||
           arrays_.find(key) != arrays_.end();
  }

private:
  std::map<std::string, std::string> values_;
  std::map<std::string, std::vector<std::string>> arrays_;

  static std::string trim(const std::string &s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos)
      return "";
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
  }
};

} // namespace unina
