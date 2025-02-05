#ifndef __DATA_HANDLER_HPP
#define __DATA_HANDLER_HPP

#include "data.hpp"
#include <vector>
#include <cstdint>
#include <string>
#include <fstream>
#include <iostream>
#include <unordered_map>

class data_handler {
    std::vector<data *> *data_array;
    std::vector<data *> *training_data;
    std::vector<data *> *testing_data;
    std::vector<data *> *validation_data;

    std::unordered_map<std::string, int> class_map;
    int class_counts;
    int num_classes;
    int feature_vector_size;

    const double TRAINING_DATA_SET_PERCENTAGE = 0.60;
    const double TESTING_DATA_SET_PERCENTAGE = 0.20;
    const double VALIDATION_DATA_SET_PERCENTAGE = 0.20;

public:
    data_handler();
    ~data_handler();

    void read_data(const std::string& data_path);
    void split_data();
    void count_classes();

    int get_class_counts();
    std::vector<data *> *get_data_array();
    std::vector<data *> *get_training_data();
    std::vector<data *> *get_testing_data();
    std::vector<data *> *get_validation_data();
};

#endif
