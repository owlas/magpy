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
using rng_vec=std::vector<std::shared_ptr<Rng>, std::allocator<std::shared_ptr<Rng> > >;


namespace simulation
{
    // This struct holds the results from the simulation
    // data is dynamically allocated using unique_ptrs
    // magnetisation - [Nx3] x,y,z compononents of the magnetisation
    // field - [N] the value of the field at time t_n
    // time - [N] the time at n
    // N - the number of steps taken
    // energyloss - the energy lost in the simulation
    struct results {
        std::unique_ptr<double[]> mx;
        std::unique_ptr<double[]> my;
        std::unique_ptr<double[]> mz;
        std::unique_ptr<double[]> field;
        std::unique_ptr<double[]> time;
        size_t N;
        double energy_loss;

        results( size_t _N ) {
            N=_N;
            mx = std::unique_ptr<double[]>( new double[N] );
            my = std::unique_ptr<double[]>( new double[N] );
            mz = std::unique_ptr<double[]>( new double[N] );
            field = std::unique_ptr<double[]>( new double[N] );
            time = std::unique_ptr<double[]>( new double[N] );
            energy_loss=0;
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
      Computes the probability trajectories based on a master equation
      approximation
    */
    struct results dom_ensemble_dynamics(
        const double volume,
        const double anisotropy,
        const double temperature,
        const double magnetisation,
        const double alpha,
        const std::function<double(double)> applied_field,
        const std::array<double,2> initial_mags,
        const double time_step,
        const double end_time,
        const int max_samples );

    /*
      Simulate an ensemble of identical systems and average
      their results.
      Initial condition can be specified once for all particles
      or a vector of initial conditions can be specified.
    */
    template <typename... T>
    struct results ensemble_run(
        const size_t max_samples,
        std::function<results(T...)> run_function,
        std::vector<T>... varying_arguments );

    /*
      Runs a full simulation and returns the final state of each of the systems
      in the ensemble.
    */
    template <typename... T>
    std::vector<d3> ensemble_run_final_state(
        std::function<results(T...)> run_function,
        std::vector<T>... varing_arguments );

    /*
      Simulates the dynamics for a single cycle of the applied
      alternating field. Simulated repeatedly until the magnetisation
      reaches a steady state.
    */
    template <typename... T>
    struct results steady_state_cycle_dynamics(
        std::function<results(d3, T...)> run_function,
        const int max_samples,
        const double steady_state_condition,
        const std::vector<d3> initial_magnetisations,
        std::vector<T >... varying_arguments );

    // Save a results file
    void save_results( const std::string fname, const struct results& );

    // Compute the energy loss for a particle from its simulation results
    double energy_loss(
        const struct results&, const double ms, const double hk );

    // Compute the energy loss for a particle from the probability flow
    double energy_loss(
        const std::unique_ptr<double[]> &transition_energy,
        const std::unique_ptr<double[]> &probability_flow,
        const std::unique_ptr<double[]> &time,
        const size_t N );

    // Sets all of the arrays in the results struct to zero
    void zero_results( struct results& );

    void reduce_to_system_magnetisation(
        double *mag, const double *particle_mags, const size_t N_particles );


}
#include "simulation.tpp"
#endif
