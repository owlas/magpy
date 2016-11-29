// simulation.hpp
// high level functions for simulating magnetic dynamics
//
// Oliver W. Laslett (2016)
// O.Laslett@soton.ac.uk
#ifndef SIM_H
#define SIM_H
#include <memory>
#include <functional>
#include <array>
#include <random>

using d3 = std::array<double,3>;


namespace simulation
{
    struct results {
        double *magnetisation;
        double *field;
        double *time;
        size_t N;

        results( size_t _N ) {
            N=_N;
            magnetisation = new double[N*3];
            field = new double[N];
            time = new double[N];
        }

        ~results() {
            delete [] magnetisation;
            delete [] field;
            delete [] time;
        }
    };
    struct results full_dynamics(
        const double damping,
        const double thermal_field_strength,
        const std::function<double(double)> applied_field,
        const d3 initial_magnetisation,
        const double time_step,
        const double end_time,
        std::mt19937_64 rng );
}
#endif
