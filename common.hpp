#ifndef __COMMON_HPP
#define __COMMON_HPP

#include "data.hpp"
#include <vector>
#include <iostream>
#include <random>

class common{
    protected:
        std::vector<data *> *common_training_data;
        std::vector<data *> *common_testing_data;
        std::vector<data *> *common_validation_data;

        int num_chunks;

    public:
        common();
        ~common();

        void initialize_matrix(std::vector<std::vector<double>>&matrix, int rows, int cols);

        std::vector<data *> *get_common_training_data();
        std::vector<data *> *get_common_testing_data();
        std::vector<data *> *get_common_validation_data();

        void set_common_training_data(std::vector<data *> *vect);
        void set_common_testing_data(std::vector<data *> *vect);
        void set_common_validation_data(std::vector<data *> *vect);

        int get_num_chunks() const;
        void set_num_chunks(int* chunk_count_ptr);
};

#endif