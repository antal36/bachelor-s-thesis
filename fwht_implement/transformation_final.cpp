// transformation.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include <cmath>
#include <cstring> // for std::memcpy

namespace py = pybind11;

// Perform FWHT on data with given length and stride.
// data: pointer to the start of the data
// length: number of elements in the column
// stride: distance between consecutive elements of the column in memory
void perform_fwht(double *data, size_t length, size_t stride) {
    size_t h = 1;
    while (h < length) {
        for (size_t i = 0; i < length; i += h * 2) {
            for (size_t j = i; j < i + h; j++) {
                double x = data[j * stride];
                double y = data[(j + h) * stride];
                data[j * stride]       = x + y;
                data[(j + h) * stride] = x - y;
            }
        }
        h *= 2;
    }
}

void fwht(py::array_t<double> input_matrix) {
    py::buffer_info input_buf = input_matrix.request();

    // Ensure that we have a 2D array
    if (input_buf.ndim != 2) {
        throw std::runtime_error("Input must be a 2D matrix!");
    }

    size_t rows = input_buf.shape[0];
    size_t cols = input_buf.shape[1];

    // Check if rows is a power of 2
    if ((rows & (rows - 1)) != 0) {
        throw std::runtime_error("Number of rows must be a power of 2!");
    }

    double *ptr_input = static_cast<double *>(input_buf.ptr);

    // Perform FWHT on each column using strided access
    for (size_t col = 0; col < cols; col++) {
        // Column start pointer
        double *col_ptr = ptr_input + col;
        // For row-major order, stride is `cols`
        perform_fwht(col_ptr, rows, cols);
    }
}

PYBIND11_MODULE(fwht, m) {
    m.doc() = "Module for fast Walsh-Hadamard transform on 2D arrays";
    m.def("fwht", &fwht, "Perform the fast Walsh-Hadamard transform in-place",
          py::arg("input_matrix"));
}
