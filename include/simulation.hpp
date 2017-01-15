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
#include "rng.hpp"

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
        double *mx;
        double *my;
        double *mz;
        double *field;
        double *time;
        size_t N;

        results( size_t _N ) {
            N=_N;
            mx = new double[N];
            my = new double[N];
            mz = new double[N];
            field = new double[N];
            time = new double[N];
        }

        ~results() {
            delete [] mx; delete [] my; delete [] mz;
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
        Rng &rng,
        const bool renorm,
        const int max_samples );

    /*
      Simulates the dynamics for a single cycle of the applied
      alternating field. Simulated repeatedly until the magnetisation
      reaches a steady state.
    */
    struct results steady_state_cycle_dynamics(
        const double damping,
        const double thermal_field_strength,
        const d3 anis_axis,
        const std::function<double(double)> applied_field,
        const d3 initial_magnetisation,
        const double time_step,
        const double applied_field_period,
        Rng &rng,
        const bool renorm,
        const int max_samples,
        const double steady_state_condition=1e-6 );

    // Save a results file
    void save_results( const std::string fname, const struct results& );

    // Compute the power loss for a particle from its simulation results
    double power_loss(
        const struct results&, double volume, double anisotropy,
        double magnetisation, double anisotropy_field,
        double field_frequency );

    // Sets all of the arrays in the results struct to zero
    void zero_results( struct results& );
}
#endif
