#include "xtensor-blas/xlinalg.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xadapt.hpp"

xt::xarray<double> mvp(xt::xarray<double> op, xt::xarray<double> state, std::vector<unsigned long int> op_wires)
{
    auto shape = op.shape();
    unsigned long int length = shape.size() / 2;
//    unsigned long int axis[length];

//    for (int i=0; i < length; i++)
//    {
//        axis[i] = i + length;
//    }

    std::vector<unsigned long int> axis(length, 2 * length);

    std::cout << xt::adapt(shape) << std::endl;

    auto result = xt::linalg::tensordot(op, state, axis, op_wires);
    return 0;
}

int main()
{
    xt::xarray<double> op, state;
    std::vector<unsigned long int> op_wires;

    op = {{{{1., 0.},
         {0., 0.}},
        {{0., 1.},
         {0., 0.}}},
       {{{0., 0.},
         {1., 0.}},
        {{0., 0.},
         {0., 1.}}}};
    state = {{1, -1}, {1, -1}};
    op_wires = {0, 1};

    std::cout << mvp(op, state, op_wires) << std::endl;
    return 0;
}