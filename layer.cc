#include "../include/layer.hpp"
#include <chrono>
#include <algorithm>
#include <vector>
#include <cmath>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <algorithm>

// Corrected version: for each input pixel (i,j), only iterate over output positions
// for which the filter (of size k) actually contributes.
// H_in, W_in: dimensions of the input image
// k: filter size (expected to be 3 in our case)
// dY: gradient for the convolution output (shape: H_out x W_out, where H_out = H_in - k + 1)
std::vector<double> conv_backward_input(const std::vector<double>& dY,
                                          const std::vector<double>& W,
                                          int H_in, int W_in, int k) {
    // Compute output dimensions (as produced by the forward convolution)
    int H_out = H_in - k + 1;
    int W_out = W_in - k + 1;
    
    // Allocate the gradient for the input image
    std::vector<double> dX(H_in * W_in, 0.0);
    
    // Precompute the flipped filter (flip both vertically and horizontally)
    std::vector<double> W_flipped(W.size(), 0.0);
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            W_flipped[i * k + j] = W[(k - 1 - i) * k + (k - 1 - j)];
        }
    }
    
    // For each input pixel (i,j), sum contributions from output pixels that used it.
    // The valid output pixels (p,q) must satisfy:
    //    p in [max(0, i - (k-1)), min(i, H_out-1)]
    //    q in [max(0, j - (k-1)), min(j, W_out-1)]
    for (int i = 0; i < H_in; i++) {
        for (int j = 0; j < W_in; j++) {
            double sum = 0.0;
            int p_start = std::max(0, i - (k - 1));
            int p_end   = std::min(i, H_out - 1);
            int q_start = std::max(0, j - (k - 1));
            int q_end   = std::min(j, W_out - 1);
            for (int p = p_start; p <= p_end; p++) {
                int i_filter = i - p;  // will be in [0, k-1]
                for (int q = q_start; q <= q_end; q++) {
                    int j_filter = j - q;  // in [0, k-1]
                    if (i_filter >= 0 && i_filter < k && j_filter >= 0 && j_filter < k) {
                        sum += dY[p * W_out + q] * W_flipped[i_filter * k + j_filter];
                    }
                }
            }
            dX[i * W_in + j] = sum;
        }
    }
    return dX;
}

double custom_tanh(double x) {
    // Asymptotes for large x
    if (x > 10.0) return 1.0;
    if (x < -10.0) return -1.0;

    // Linear approximation for small x
    if (std::abs(x) < 1e-3) return x;

    // Use standard tanh for moderate x
    return (std::exp(2 * x) - 1) / (std::exp(2 * x) + 1);
}

//Constructor: initializing filters with random values
ConvLayer::ConvLayer(int filter_size, int num_filters, double learning_rate)
    : filter_size(filter_size), num_filters(num_filters), learning_rate(learning_rate), convolution_done(false) {
    filters = new std::vector<std::vector<neuron*>>(num_filters);
    conv_output = nullptr;  // Initialize to null to avoid dangling pointers

    for (int i = 0; i < num_filters; i++) {
        (*filters)[i] = std::vector<neuron*>(filter_size * filter_size);
        for (int j = 0; j < filter_size * filter_size; j++) {
            (*filters)[i][j] = new neuron(filter_size * filter_size);
        }
    }
}

ConvLayer::~ConvLayer() {
    // Clean up the filters
    for (auto& filter : *filters) {
        for (neuron* n : filter) {
            delete n;
        }
    }
    delete filters;
}

// Forward Pass: Apply convolution on the input
std::vector<double>* ConvLayer::forward(std::vector<double>* input) {
    if (input == nullptr || input->empty()) {
        throw std::runtime_error("Error: ConvLayer received an empty input vector!");
    }
    
    // Assume the input is a square image.
    int image_size = static_cast<int>(std::sqrt(input->size())); // Expected: 360
    if (image_size * image_size != input->size()) {
        throw std::runtime_error("Error: ConvLayer received a non-square input vector!");
    }
    
    // Save a copy of the input for use in the backward pass.
    if (this->input != nullptr)
        delete this->input;
    this->input = new std::vector<double>(*input);
    
    // Convolution output dimensions.
    int output_side = image_size - filter_size + 1; // 360 - 3 + 1 = 358
    
    // We want each filter to produce a pooled feature map of 8x8.
    // Therefore, choose pool_size = stride = 44 (since 44*8 = 352, which is close to 358).
    int pool_size = 22;  
    int pooled_side = output_side / pool_size; // floor(358/44) = floor(8.136) = 8
    
    // We'll collect the pooled outputs from each filter.
    std::vector<double> pooled_all;
    
    // Process each filter individually.
    for (int f = 0; f < num_filters; f++) {
        // Allocate convolution output for this filter.
        std::vector<double> conv_out(output_side * output_side, 0.0);
        
        // Perform convolution for filter f (simple nested loops).
        for (int i = 0; i < output_side; i++) {
            for (int j = 0; j < output_side; j++) {
                double sum = 0.0;
                for (int p = 0; p < filter_size; p++) {
                    for (int q = 0; q < filter_size; q++) {
                        int input_row = i + p;
                        int input_col = j + q;
                        double input_val = (*input)[input_row * image_size + input_col];
                        // Here we assume that the weight for this filter is stored in filters[f] as a flat array.
                        // Adjust the index if needed.
                        double weight_val = (*filters)[f][p * filter_size + q]->weights->at(0);
                        sum += input_val * weight_val;
                    }
                }
                conv_out[i * output_side + j] = sum;
            }
        }
        
        // Apply average pooling on conv_out with pool kernel = pool_size.
        std::vector<double> pooled(pooled_side * pooled_side, 0.0);
        for (int i = 0; i < pooled_side; i++) {
            for (int j = 0; j < pooled_side; j++) {
                double pool_sum = 0.0;
                for (int p = 0; p < pool_size; p++) {
                    for (int q = 0; q < pool_size; q++) {
                        int r = i * pool_size + p;
                        int c = j * pool_size + q;
                        if (r < output_side && c < output_side) {
                            pool_sum += conv_out[r * output_side + c];
                        }
                    }
                }
                pooled[i * pooled_side + j] = pool_sum / (pool_size * pool_size);
            }
        }
        
        // Append this filter's pooled output.
        pooled_all.insert(pooled_all.end(), pooled.begin(), pooled.end());
    }
    
    /*std::cout << "Forward Pass: image " << image_size << "x" << image_size 
              << ", conv output " << output_side << "x" << output_side 
              << ", pooled " << pooled_side << "x" << pooled_side
              << ", total pooled vector size: " << pooled_all.size() << std::endl;*/
    // Expected pooled_all size: num_filters * (8*8) = 128 * 64 = 8192
    
    return new std::vector<double>(pooled_all);
}

// Assume these member variables exist in your ConvLayer class:
// int filter_size;            // = 3
// int num_filters;            // = 128
// std::vector<double>* input; // stored during forward pass (size 360x360 = 129600)
// std::vector<std::vector<neuron*>>* filters; // each filter is a vector of (filter_size*filter_size) neurons
// double learning_rate;       // learning rate for weight updates

std::vector<double>* ConvLayer::backward(std::vector<double>* d_out) {
    if (!d_out) {
        std::cerr << "[ConvLayer::backward] Error: Received null gradient." << std::endl;
        return nullptr;
    }
    if (this->input == nullptr || this->input->empty()) {
        std::cerr << "[ConvLayer::backward] Error: No stored input for backward pass." << std::endl;
        return nullptr;
    }
    
    int image_size = static_cast<int>(std::sqrt(this->input->size())); // 360
    int output_side = image_size - filter_size + 1; // 358
    int pool_size = 22;
    int pooled_side = output_side / pool_size; // floor(358/44)= floor(8.136)= 8

    // Check that d_out size is as expected.
    int expected_size = num_filters * pooled_side * pooled_side; // 128 * 8 * 8 = 8192
    if (d_out->size() != expected_size) {
        std::cerr << "[ConvLayer::backward] Error: d_out size (" << d_out->size() 
                  << ") does not match expected (" << expected_size << ")." << std::endl;
        return nullptr;
    }
    
    // Initialize gradient with respect to the input image.
    std::vector<double>* d_input = new std::vector<double>(this->input->size(), 0.0);
    
    // Process each filter.
    for (int f = 0; f < num_filters; f++) {
        // Extract the portion of d_out for filter f.
        std::vector<double> d_out_f(d_out->begin() + f * pooled_side * pooled_side,
                                    d_out->begin() + (f + 1) * pooled_side * pooled_side);
        
        // Unpool d_out_f back to the conv output dimensions (output_side x output_side).
        std::vector<double> d_conv(output_side * output_side, 0.0);
        for (int i = 0; i < pooled_side; i++) {
            for (int j = 0; j < pooled_side; j++) {
                double grad_val = d_out_f[i * pooled_side + j] / (pool_size * pool_size);
                for (int p = 0; p < pool_size; p++) {
                    for (int q = 0; q < pool_size; q++) {
                        int r = i * pool_size + p;
                        int c = j * pool_size + q;
                        if (r < output_side && c < output_side) {
                            d_conv[r * output_side + c] = grad_val;
                        }
                    }
                }
            }
        }
        
        // Retrieve filter f’s weights as a flat vector.
        std::vector<double> W_f(filter_size * filter_size, 0.0);
        for (int i = 0; i < filter_size; i++) {
            for (int j = 0; j < filter_size; j++) {
                // Adjust the index if needed to match your storage.
                W_f[i * filter_size + j] = (*filters)[f][i * filter_size + j]->weights->at(0);
            }
        }
        
        // Compute the gradient with respect to the input for this filter.
        std::vector<double> dX_f = conv_backward_input(d_conv, W_f, image_size, image_size, filter_size);
        
        // Accumulate the gradients from this filter.
        for (int i = 0; i < image_size * image_size; i++) {
            (*d_input)[i] += dX_f[i];
        }
        
        // (Optionally, compute and apply weight updates here.)
    }
    
    return d_input;
}

std::vector<std::vector<double>>* ConvLayer::convolve(std::vector<std::vector<double>>* input, std::vector<neuron*>* filter) {
    int input_size = input->size();
    int stride = 1;  // Define stride (adjust as needed)
    int output_size = ((input_size - filter_size) / stride) + 1;  // Correct output size calculation

    //std::cout << "Convolution Output Size: " << output_size << " x " << output_size << std::endl;

    if (output_size <= 0) {
        std::cerr << "Error: Output size is zero or negative!" << std::endl;
        return nullptr;
    }

    auto result = new std::vector<std::vector<double>>(output_size, std::vector<double>(output_size));

    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < output_size; j++) {
            double convolved_value = 0;
            int filter_idx = 0;
            for (int fi = 0; fi < filter_size; fi++) {
                for (int fj = 0; fj < filter_size; fj++) {
                    if (filter_idx >= filter->size()) {
                        std::cerr << "Error: Filter neuron out of bounds!" << std::endl;
                        return nullptr;
                    }
                    neuron* n = (*filter)[filter_idx++];
                    if (!n) {
                        std::cerr << "Error: Null neuron at filter index " << filter_idx - 1 << std::endl;
                        return nullptr;
                    }
                    double input_value = (*input)[i + fi][j + fj];
                    double weight_value = n->weights->at(fi * filter_size + fj);
                    convolved_value += input_value * weight_value;
                }
            }
            (*result)[i][j] = convolved_value;
        }
    }
    return result;
}

std::vector<double>* ConvLayer::average_pooling(std::vector<double>* input, int pooling_size) {
    int input_size = static_cast<int>(std::sqrt(input->size()));
    int pooling_stride = pooling_size; // usually equal to the pooling size for non-overlapping pooling
    int output_size = ((input_size - pooling_size) / pooling_stride) + 1;
    
    if (output_size <= 0) {
        throw std::runtime_error("Pooling failed: Output size is non-positive.");
    }
    
    std::cout << "Applying Average Pooling | Input: " << input_size << " x " << input_size
              << " -> Output: " << output_size << " x " << output_size << std::endl;
    
    auto output = new std::vector<double>(output_size * output_size, 0.0);
    
    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < output_size; j++) {
            double sum = 0.0;
            for (int x = 0; x < pooling_size; x++) {
                for (int y = 0; y < pooling_size; y++) {
                    int index = (i * pooling_stride + x) * input_size + (j * pooling_stride + y);
                    if (index < input->size()) {
                        sum += (*input)[index];
                    }
                }
            }
            (*output)[i * output_size + j] = sum / (pooling_size * pooling_size);
        }
    }
    
    return output;
}

int ConvLayer::get_pooled_output_size() const {
    int pooling_size = 2;  // Adjust based on actual pooling settings
    return (this->input->size() / (pooling_size * pooling_size));
}

// RNNLayer (LSTM/GRU) Implementation

//Initialize the hidden and cell states for usage
RNNLayer::RNNLayer(int input_size, int hidden_size, double learning_rate)
    : input_size(input_size), hidden_size(hidden_size), learning_rate(learning_rate), chunk_size(0), num_chunks(0) {
    cd->initialize_matrix(W_f, hidden_size, input_size + hidden_size);
    cd->initialize_matrix(W_i, hidden_size, input_size + hidden_size);
    cd->initialize_matrix(W_o, hidden_size, input_size + hidden_size);
    cd->initialize_matrix(W_C, hidden_size, input_size + hidden_size);

    hidden_state = new std::vector<double>(hidden_size, 0.0);
    cell_state = new std::vector<double>(hidden_size, 0.0);
    hidden_neurons = new std::vector<neuron*>();

    for (int i = 0; i < hidden_size; ++i) {
        hidden_neurons->push_back(new neuron(input_size + hidden_size));
    }

    std::cout << "Hidden neurons initialized: " << hidden_neurons->size() << " neurons." << std::endl;
}

RNNLayer::~RNNLayer() {
    delete hidden_state;
    delete cell_state;
    for (neuron* n : *hidden_neurons) {
        delete n;
    }
    delete hidden_neurons;
}

// Inside the RNNLayer's forward pass method
std::vector<double>* RNNLayer::forward(std::vector<double>* input) {
    if (!input || input->empty()) {
        throw std::runtime_error("RNNLayer::forward - Invalid input.");
    }

    //std::cout << "Processing RNN Layer | Input Size: " << input->size() << std::endl;

    // Calculate chunk size and number of chunks
    chunk_size = 8192;  // Example chunk size
    num_chunks = (input->size() + chunk_size - 1) / chunk_size;  // Round up to cover all elements

    //std::cout << "Chunk Size: " << chunk_size << ", Number of Chunks: " << num_chunks << std::endl;

    std::vector<double>* output = new std::vector<double>(hidden_size, 0.0);

    for (int i = 0; i < num_chunks; ++i) {
        int chunk_start = i * chunk_size;
        int chunk_end = std::min(chunk_start + chunk_size, static_cast<int>(input->size()));

        std::vector<double> chunk(input->begin() + chunk_start, input->begin() + chunk_end);
        std::vector<double>* chunk_output = lstm_forward(&chunk);

        for (int j = 0; j < hidden_size; ++j) {
            (*output)[j] += (*chunk_output)[j];  // Aggregate outputs
        }

        delete chunk_output;
    }

    return output;
}


std::vector<double>* RNNLayer::forward_chunk(std::vector<double>* chunk) {
    if (chunk == nullptr || chunk->empty()) {
        throw std::runtime_error("Invalid chunk input. Chunk is null or empty.");
    }

    //std::cout << "Processing RNN chunk of size: " << chunk->size() << std::endl;

    // Placeholder for RNN forward logic
    // You can replace this with the actual computation for your RNN layer
    std::vector<double>* output = new std::vector<double>(chunk->size(), 0.0);  // Dummy output
    for (size_t i = 0; i < chunk->size(); ++i) {
        // Example: Copy input directly to output for now
        (*output)[i] = (*chunk)[i] * 0.5;  // Dummy operation
    }

    std::cout << "RNN Chunk processed. Output size: " << output->size() << std::endl;
    return output;
}

std::vector<double>* RNNLayer::backward(std::vector<double>* d_out) {
    if (!d_out) {
        std::cerr << "[RNNLayer::backward] Error: Received null gradient vector." << std::endl;
        return nullptr;
    }
    
    // Create a gradient vector for the input with the expected size.
    std::vector<double>* d_input = new std::vector<double>(input_size, 0.0);
    
    // Iterate over the hidden neurons in reverse order.
    for (int i = hidden_neurons->size() - 1; i >= 0; --i) {
        neuron* current_neuron = (*hidden_neurons)[i];
        if (!current_neuron) {
            std::cerr << "[RNNLayer::backward] Warning: Null neuron encountered at index " << i << std::endl;
            continue;
        }
        
        // Use the neuron's stored delta (make sure this was computed properly).
        double delta = current_neuron->delta;
        
        // Propagate the gradient for each input component.
        for (int j = 0; j < input_size; ++j) {
            if (j < static_cast<int>(current_neuron->weights->size())) {
                (*d_input)[j] += delta * (*current_neuron->weights)[j];
            } else {
                std::cerr << "[RNNLayer::backward] Warning: Weight index " << j
                          << " out of range for neuron " << i << std::endl;
            }
        }
    }
    
    return d_input;
}

std::vector<double>* RNNLayer::backward_chunked(std::vector<double>* d_output, int chunk_size) {
    if (chunk_size <= 0) {
        throw std::runtime_error("[RNNLayer] Invalid chunk size!");
    }

    std::vector<double>* d_input = new std::vector<double>(input_size, 0.0);

    for (int chunk_start = 0; chunk_start < d_output->size(); chunk_start += chunk_size) {
        int chunk_end = std::min(chunk_start + chunk_size, static_cast<int>(d_output->size()));
        std::vector<double> chunk_output(d_output->begin() + chunk_start, d_output->begin() + chunk_end);

        std::vector<double> dummy_cell_state(hidden_size, 0.0);
        std::vector<double>* d_chunk_input = lstm_backward(&chunk_output, &dummy_cell_state);

        // Aggregate gradients for the input
        for (size_t i = 0; i < d_chunk_input->size(); ++i) {
            (*d_input)[i] += (*d_chunk_input)[i];
        }
        delete d_chunk_input;
    }

    return d_input;
}

std::vector<double>* RNNLayer::lstm_forward(std::vector<double>* input) {
    std::vector<double>* output = new std::vector<double>(hidden_size);

    for (int i = 0; i < hidden_size; ++i) {
        double forget_sum = 0.0, input_sum = 0.0, output_sum = 0.0, cell_candidate_sum = 0.0;

        // Compute contributions from input
        for (int j = 0; j < input->size(); ++j) {
            forget_sum += W_f[i][j] * (*input)[j];
            input_sum += W_i[i][j] * (*input)[j];
            output_sum += W_o[i][j] * (*input)[j];
            cell_candidate_sum += W_C[i][j] * (*input)[j];
        }

        // Compute contributions from hidden state
        for (int j = 0; j < hidden_state->size(); ++j) {
            forget_sum += W_f[i][input->size() + j] * (*hidden_state)[j];
            input_sum += W_i[i][input->size() + j] * (*hidden_state)[j];
            output_sum += W_o[i][input->size() + j] * (*hidden_state)[j];
            cell_candidate_sum += W_C[i][input->size() + j] * (*hidden_state)[j];
        }

        // Debug: Print summation values before activation
        /*std::cout << "Neuron " << i << ": Forget Sum: " << forget_sum
                  << ", Input Sum: " << input_sum
                  << ", Output Sum: " << output_sum
                  << ", Cell Candidate Sum: " << cell_candidate_sum << std::endl;*/

        // Clamp the summations for numerical stability
        forget_sum = std::max(-10.0, std::min(10.0, forget_sum));
        input_sum = std::max(-10.0, std::min(10.0, input_sum));
        output_sum = std::max(-10.0, std::min(10.0, output_sum));
        cell_candidate_sum = std::max(-10.0, std::min(10.0, cell_candidate_sum));

        // Apply activations
        double forget_gate = sigmoid(forget_sum);
        double input_gate = sigmoid(input_sum);
        double output_gate = sigmoid(output_sum);
        double cell_candidate = custom_tanh(cell_candidate_sum);

        // Debug: Print activation outputs
        /*std::cout << "Neuron " << i << ": Forget Gate = " << forget_gate
                  << ", Input Gate = " << input_gate
                  << ", Output Gate = " << output_gate
                  << ", Cell Candidate = " << cell_candidate
                  << std::endl;*/

        // Update cell state and hidden state
        (*cell_state)[i] = forget_gate * (*cell_state)[i] + input_gate * cell_candidate;
        (*hidden_state)[i] = output_gate * std::tanh((*cell_state)[i]);

        // Set output for this neuron
        (*output)[i] = (*hidden_state)[i];
    }

    return output;
}

std::vector<double>* RNNLayer::lstm_backward(std::vector<double>* d_output, std::vector<double>* d_next_cell_state) {
    if (!hidden_neurons || hidden_neurons->empty()) {
        throw std::runtime_error("Error: hidden_neurons is not initialized or empty!");
    }

    // Skip validation for the first neuron if d_output is empty for it
    if ((*d_output)[0] == 0) {
        //std::cout << "Skipping size validation for the first neuron due to zero d_output." << std::endl;
        std::cerr << "Skipping size validation for the first neuron due to zero d_output." << std::endl;
    } else {
        // Ensure all sizes align
        if (d_output->size() != hidden_size || d_next_cell_state->size() != hidden_size) {
            std::cout << "d_output: " << d_output << std::endl;
            std::cerr << "Error: Gradient size mismatch in lstm_backward!" << std::endl;
            std::cerr << "Expected hidden_size: " << hidden_size 
                      << ", d_output size: " << d_output->size() 
                      << ", d_next_cell_state size: " << d_next_cell_state->size() << std::endl;
            throw std::runtime_error("Gradient size mismatch in lstm_backward!");
        }
    }

    std::vector<double>* d_input = new std::vector<double>(input_size, 0.0);  // Gradients w.r.t input
    std::vector<double>* d_hidden_state = new std::vector<double>(hidden_size, 0.0);  // Gradients w.r.t hidden state
    std::vector<double>* d_cell_state = new std::vector<double>(hidden_size, 0.0);  // Gradients w.r.t cell state

    std::vector<double> d_forget_gate(hidden_size, 0.0);
    std::vector<double> d_input_gate(hidden_size, 0.0);
    std::vector<double> d_output_gate(hidden_size, 0.0);
    std::vector<double> d_cell_candidate(hidden_size, 0.0);

    // Loop through each neuron in the hidden layer to compute gradients
    for (int i = 0; i < hidden_size; ++i) {
        if (!(*hidden_neurons)[i]) {
            throw std::runtime_error("Error: Null neuron at index " + std::to_string(i));
        }

        neuron* current_neuron = (*hidden_neurons)[i];

        // Skip the first neuron if d_output is zero
        if (i == 0 && (*d_output)[i] == 0) {
            std::cerr << "Skipping gradient computation for the first neuron due to zero d_output." << std::endl;
            continue;
        }

        // Gradient computations
        d_output_gate[i] = (*d_output)[i] * std::tanh((*cell_state)[i]);
        double d_tanh_cell_state = (*d_output)[i] * (*hidden_state)[i];

        d_forget_gate[i] = d_tanh_cell_state * (*cell_state)[i];
        d_input_gate[i] = d_tanh_cell_state * (*cell_state)[i];
        d_cell_candidate[i] = d_tanh_cell_state * (1 - std::pow(std::tanh((*cell_state)[i]), 2));

        (*d_cell_state)[i] = (*d_next_cell_state)[i] * d_forget_gate[i];

        for (size_t j = 0; j < current_neuron->weights->size(); ++j) {
            current_neuron->weights->at(j) -= learning_rate * d_forget_gate[i];
            (*d_input)[j] += current_neuron->weights->at(j) * d_output_gate[i];
        }
    }

    // Clean up
    delete d_hidden_state;
    delete d_cell_state;

    return d_input;
}


double RNNLayer::sigmoid(double x) {
    // Clip large values to avoid overflow/underflow
    if (x > 20.0) return 1.0;
    if (x < -20.0) return 0.0;
    return 1.0 / (1.0 + std::exp(-x));
}