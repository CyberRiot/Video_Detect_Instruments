#include "../include/data_handler.hpp"
#include <algorithm>
#include <filesystem>
#include <random>



data_handler::data_handler() {
    data_array = new std::vector<data *>;
    training_data = new std::vector<data *>;
    testing_data = new std::vector<data *>;
    validation_data = new std::vector<data *>;
    class_counts = 0;
}

data_handler::~data_handler() {
    delete data_array;
    delete training_data;
    delete testing_data;
    delete validation_data;
}

void data_handler::read_data(const std::string& data_path) {
    std::ifstream data_file(data_path, std::ios::binary);
    if (!data_file) {
        std::cerr << "Error: Could not open data file: " << data_path << std::endl;
        exit(1);
    }

    feature_vector_size = 480 * 270;  // Grayscale, known resolution
    int frame_count = 0;

    while (true) {
        auto feature_vector = new std::vector<uint8_t>(feature_vector_size);
        data_file.read(reinterpret_cast<char*>(feature_vector->data()), feature_vector_size);
        
        if (data_file.gcount() != static_cast<std::streamsize>(feature_vector_size)) {
            delete feature_vector;
            break;
        }

        std::string label;
        std::getline(data_file, label);  

        if (!class_map.count(label)) {
            class_map[label] = class_counts++;
        }

        data* d = new data();
        d->set_feature_vector(*feature_vector);
        d->set_label(class_map[label]);

        auto class_vector = new std::vector<int>(class_counts, 0);
        (*class_vector)[class_map[label]] = 1;
        d->set_class_vector(*class_vector);

        data_array->push_back(d);
        frame_count++;

        // Show progress
    }

    data_file.close();
    std::cout << "\nSuccessfully loaded " << data_array->size() << " images." << std::endl;
}

void data_handler::split_data() {
    int train_size = data_array->size() * TRAINING_DATA_SET_PERCENTAGE;
    int test_size = data_array->size() * TESTING_DATA_SET_PERCENTAGE;

    // 🔹 REMOVE `shuffle()` to keep the sequence intact

    training_data->assign(data_array->begin(), data_array->begin() + train_size);
    testing_data->assign(data_array->begin() + train_size, data_array->begin() + train_size + test_size);
    validation_data->assign(data_array->begin() + train_size + test_size, data_array->end());

    std::cout << "Training Data: " << training_data->size() 
              << ", Testing Data: " << testing_data->size()
              << ", Validation Data: " << validation_data->size() << std::endl;
}

std::vector<data *> *data_handler::get_data_array() { return data_array; }
std::vector<data *> *data_handler::get_training_data() { return training_data; }
std::vector<data *> *data_handler::get_testing_data() { return testing_data; }
std::vector<data *> *data_handler::get_validation_data() { return validation_data; }

int data_handler::get_class_counts(){
    return class_counts;
}

std::unordered_map<std::string, int>& data_handler::get_class_map(){
    return class_map;
}