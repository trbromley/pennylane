#include "xtensor-blas/xlinalg.hpp"

int main()
{
    xt::xarray<double> a = {{1, 0}, {0, -1}};
    auto d = xt::linalg::det(a);
    std::cout << d << std::endl;  // 6.661338e-16
}