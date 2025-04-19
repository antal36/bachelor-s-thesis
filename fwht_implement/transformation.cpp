// transformation module
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include <cmath>
#include <vector>

namespace py = pybind11;

// fwht itself
void perform_fwht(std::vector<double> &array) {
    size_t h = 1;
    size_t n = array.size();
    while (h < n) {
        for (size_t i = 0; i < n; i += h * 2) {
            for (size_t j = i; j < i + h; j++) {
                double x = array[j % n];
                double y = array[(j + h) % n];
                array[j % n] = x + y;
                array[(j + h) % n] = x - y;
            }
        }
        // Normalize?
        // for (size_t k = 0; k < n; k++) {
        //         array[k] /= sqrt(2);}
                
        h *= 2;
    }
}

// Function to add two NumPy arrays element-wise
py::array_t<double> fwht(py::array_t<double> input_matrix) {
    // Request buffers
    py::buffer_info input_buf = input_matrix.request();

    // Check if input is matrix
    if (input_buf.ndim != 2) {
        throw std::runtime_error("Input must be a matrix!");
    }
    // Dimensions of matrix
    size_t rows = input_buf.shape[0];
    size_t cols = input_buf.shape[1];

    // Check if number of rows is power of 2
    if ((rows & (rows - 1)) != 0) {
        throw std::runtime_error("Number of rows must be a power of 2!");
    }

    // Create an output array
    auto result = py::array_t<double>(input_buf.shape);
    py::buffer_info buf_result = result.request();

    double *ptr_input = static_cast<double *>(input_buf.ptr);
    double *ptr_result = static_cast<double *>(buf_result.ptr);

    // Perform the transformation on pointers. Please note that pointers are not two dimensional
    for (size_t j = 0; j < cols; j++) {
        std::vector<double> column(rows);
        // extract the column
        for (size_t i = 0; i < rows; i++) {
            column[i] = ptr_input[(i * cols) + j];
        }
        // perform the fwht
        perform_fwht(column);

        for(size_t idx = 0; idx < rows; idx++) {
            ptr_result[(idx * cols) + j] = column[idx];
        }
    }
    return result;
}

PYBIND11_MODULE(transformation, m) {
    m.doc() = "Module for array operations using pybind11";

    m.def("fwht", &fwht, "Perform fast Walsh-Hadard tranformation",
          py::arg("input_matrix"));
}
