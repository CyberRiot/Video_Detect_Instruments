#ifndef __DATA_HPP
#define __DATA_HPP

#include <vector>
#include <cstdint>
#include <string>

class data {
    // Store vectors by value
    std::vector<uint8_t> feature_vector;
    std::vector<int> class_vector;
    int label;
    int enum_label;
    double distance;

public:
    data();

    // Setters using const references; no manual memory management needed.
    void set_feature_vector(const std::vector<uint8_t>& vect);
    void append_to_feature_vector(uint8_t val);

    void set_label(int lab);
    void set_distance(double val);
    void set_class_vector(const std::vector<int>& vect);
    void set_enum_label(int lab);

    double get_distance() const;
    int get_label() const;
    int get_enum_label() const;
    const std::vector<int>& get_class_vector() const;
    const std::vector<uint8_t>& get_feature_vector() const;
};

#endif
