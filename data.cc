#include "data.hpp"

data::data() : label(0), enum_label(0), distance(0.0) { }

void data::set_feature_vector(const std::vector<uint8_t>& vect) {
    feature_vector = vect;
}

void data::append_to_feature_vector(uint8_t val) {
    feature_vector.push_back(val);
}

void data::set_label(int lab) {
    label = lab;
}

void data::set_distance(double val) {
    distance = val;
}

void data::set_class_vector(const std::vector<int>& vect) {
    class_vector = vect;
}

void data::set_enum_label(int lab) {
    enum_label = lab;
}

double data::get_distance() const {
    return distance;
}

int data::get_label() const {
    return label;
}

int data::get_enum_label() const {
    return enum_label;
}

const std::vector<int>& data::get_class_vector() const {
    return class_vector;
}

const std::vector<uint8_t>& data::get_feature_vector() const {
    return feature_vector;
}
