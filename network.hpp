#ifndef __NETWORK_HPP
#define __NETWORK_HPP

#include "data.hpp"
#include "neuron.hpp"
#include "layer.hpp"
#include "common.hpp"
#include "data_handler.hpp"
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>

class network : public common {
private:
    int input_size;
    int num_classes;
    std::ofstream debug_output;
public:
    std::vector<layer *> *layers;
    double learning_rate;
    double test_performance;
    common *cd = new common();

    network(std::vector<int> *spec, int input_size, int num_classes, double learning_rate);
    ~network();

    std::vector<double>* fprop(data *d);
    double activate(std::vector<double> *weights, std::vector<double> *input);
    double transfer(double activation);
    double transfer_derivative(double output);
    void initialize_chunks(common* cd, std::vector<double>* input);
    void bprop(data *d);
    void update_weights(data *d);
    int predict(data *d);
    // In network.hpp
    void train(int epochs, double validation_threshold);
    double test();
    double validate(data* d);
    void save_model(const std::string &filename);
    void load_model(const std::string &filename);
    void output_predictions(const std::string &filename, data_handler *dh);
    void set_debug_output(const std::string &filename);
    void close_debug_output();
    double calculate_loss(const std::vector<double>& logits, int class_id);
};
#endif
