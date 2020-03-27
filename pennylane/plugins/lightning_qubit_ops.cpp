#include <numeric>                        // Standard library import for std::accumulate
#include "pybind11/pybind11.h"            // Pybind11 import to define Python bindings
#include "xtensor/xmath.hpp"              // xtensor import for the C++ universal functions
#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pyarray.hpp"     // Numpy bindings
#include "xtensor-blas/xlinalg.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xadapt.hpp"
#include "pybind11/stl.h"

xt::pyarray<double> mvp(xt::pyarray<double> op, xt::pyarray<double> state, std::vector<unsigned
long int> op_wires)
{
    auto shape = op.shape();
    unsigned long int length = shape.size() / 2;
    std::vector<unsigned long int> axis(length);

    for (int i=0; i<length; i++){ axis[i] = i + length; }

    auto result = xt::linalg::tensordot(op, state, axis, op_wires);
    return result;
}

double sum_of_sines(xt::pyarray<double>& m)
{
    auto sines = xt::sin(m);  // sines does not actually hold values.
    return std::accumulate(sines.begin(), sines.end(), 0.0);
}

PYBIND11_MODULE(lightning_qubit_ops, m)
{
    xt::import_numpy();
    m.doc() = "Lightning qubit operations using XTensor";

    m.def("sum_of_sines", sum_of_sines, "Sum the sines of the input values");
    m.def("mvp", mvp, "Matrix vector product");
}
