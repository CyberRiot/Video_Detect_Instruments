#include "../include/common.hpp"

void common::initialize_matrix(std::vector<std::vector<double>>& matrix, int rows, int cols) {
    matrix.resize(rows, std::vector<double>(cols));

    std::random_device rd;  // Random seed
    std::default_random_engine generator(rd());  // Seed the generator
    std::normal_distribution<double> distribution(0.0, 1.0 / std::sqrt(cols));  // Xavier initialization

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = distribution(generator);
        }
    }
}

common::common() {
    common_training_data = nullptr;
    common_testing_data = nullptr;
    common_validation_data = nullptr;
    num_chunks = 0;  // Initialize num_chunks
}

common::~common() {
    delete common_training_data;
    delete common_testing_data;
    delete common_validation_data;
}

std::vector<data *> *common::get_common_training_data() {
    return common_training_data;
}

std::vector<data *> *common::get_common_testing_data() {
    return common_testing_data;
}

std::vector<data *> *common::get_common_validation_data() {
    return common_validation_data;
}

void common::set_common_training_data(std::vector<data *> *vect) {
    common_training_data = vect;
}

void common::set_common_testing_data(std::vector<data *> *vect) {
    common_testing_data = vect;
}

void common::set_common_validation_data(std::vector<data *> *vect) {
    common_validation_data = vect;
}

int common::get_num_chunks() const {
    if (num_chunks <= 0) {
        std::cerr << "Warning: num_chunks is currently invalid or uninitialized! Current value: " << num_chunks << std::endl;
        throw std::runtime_error("Invalid or uninitialized num_chunks accessed in get_num_chunks.");
    }
    return num_chunks;
}

void common::set_num_chunks(int* chunk_count_ptr) {
    if (chunk_count_ptr == nullptr) {
        std::cerr << "Error: Null pointer passed to set_num_chunks." << std::endl;
        return;
    }

    if (*chunk_count_ptr < 0) {
        std::cerr << "Error: Attempted to set a negative value for num_chunks: " << *chunk_count_ptr << std::endl;
        return;
    }

    num_chunks = *chunk_count_ptr;
    //std::cout << "num_chunks successfully set to: " << num_chunks << std::endl;
}
