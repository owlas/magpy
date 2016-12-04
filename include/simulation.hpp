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
#include <string>
#include <cstdlib>

using d3 = std::array<double,3>;


namespace simulation
{
    // This struct holds the results from the simulation
    // data is dynamically allocated and handled by the dtor
    // magnetisation - [Nx3] x,y,z compononents of the magnetisation
    // field - [N] the value of the field at time t_n
    // time - [N] the time at n
    // N - the number of steps taken
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

    // Run the full dynamics and return the results
    struct results full_dynamics(
        const double damping,
        const double thermal_field_strength,
        const d3 anis_axis,
        const std::function<double(double)> applied_field,
        const d3 initial_magnetisation,
        const double time_step,
        const double end_time,
        std::mt19937_64 rng );

    // Save a results file
    void save_results( const std::string fname, const struct results& );
}
#endif
