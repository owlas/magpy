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
#include <vector>
#include "rng.hpp"

using d3 = std::array<double,3>;
using rng_vec=std::vector<std::shared_ptr<Rng>>;


namespace simulation
{
    // This struct holds the results from the simulation
    // data is dynamically allocated using unique_ptrs
    // magnetisation - [Nx3] x,y,z compononents of the magnetisation
    // field - [N] the value of the field at time t_n
    // time - [N] the time at n
    // N - the number of steps taken
    struct results {
        std::unique_ptr<double[]> mx;
        std::unique_ptr<double[]> my;
        std::unique_ptr<double[]> mz;
        std::unique_ptr<double[]> field;
        std::unique_ptr<double[]> time;
        size_t N;

        results( size_t _N ) {
            N=_N;
            mx = std::unique_ptr<double[]>( new double[N] );
            my = std::unique_ptr<double[]>( new double[N] );
            mz = std::unique_ptr<double[]>( new double[N] );
            field = std::unique_ptr<double[]>( new double[N] );
            time = std::unique_ptr<double[]>( new double[N] );
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
      Simulate an ensemble of identical systems and average
      their results.
      Initial condition can be specified once for all particles
      or a vector of initial conditions can be specified.
    */
    struct results ensemble_dynamics(
        const double damping,
        const double thermal_field_strength,
        const d3 anis_axis,
        const std::function<double(double)> applied_field,
        const std::vector<d3> initial_mags,
        const double time_step,
        const double end_time,
        const rng_vec rngs,
        const bool renorm,
        const int max_samples,
        const size_t ensemble_size );
    struct results ensemble_dynamics(
        const double damping,
        const double thermal_field_strength,
        const d3 anis_axis,
        const std::function<double(double)> applied_field,
        const d3 initial_mag,
        const double time_step,
        const double end_time,
        const rng_vec rngs,
        const bool renorm,
        const int max_samples,
        const size_t ensemble_size );

    /*
      Runs a full simulation and returns the final state of each of the systems
      in the ensemble.
    */
    std::vector<d3> ensemble_final_state(
        const double damping,
        const double thermal_field_strength,
        const d3 anis_axis,
        const std::function<double(double)> applied_field,
        const std::vector<d3> initial_mags,
        const double time_step,
        const double end_time,
        const rng_vec rngs,
        const bool renorm,
        const int max_samples,
        const size_t ensemble_size );

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
        const rng_vec rngs,
        const bool renorm,
        const int max_samples,
        const size_t ensemble_size,
        const double steady_state_condition=1e-3 );

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
