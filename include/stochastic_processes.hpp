// stochastic_processes.hpp
// functions for manipulation of general stochastic processes
//
// Oliver W. Laslett (2016)
// O.Laslett@soton.ac.uk
#ifndef STOC_H
#define STOC_H
#include <cstdlib>

namespace stochastic
{
    void master_equation(double *derivs, const double *transition_matrix,
                         const double *current_state, const size_t dim );
}

#endif
