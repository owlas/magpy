// simulation.cpp
// implementation of functions for simulating magnetic dynamics
//
// Oliver W. Laslett (2016)
// O.Laslett@soton.ac.uk
#include "llg.hpp"
#include "integrators.hpp"
#include "io.hpp"
#include "field.hpp"
#include "trap.hpp"
#include "easylogging++.h"
#include "optimisation.hpp"
#include "dom.hpp"
#include "curry.hpp"
#include <exception>
#ifdef USEMKL
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif

using namespace std::placeholders;
using sde_jac = std::function<void(double*,double*,double*,double*,
                                   const double*,const double, const double)>;



template <typename... T>
struct simulation::results simulation::ensemble_run(
    const size_t max_samples,
    std::function<simulation::results(T...)> run_function,
    std::vector<T>... varying_arguments )
{
    // Create a vector of run functions
    auto run_functions = curry::vector_curry(
        run_function, varying_arguments... );

    // Initialise space for the ensemble results
    struct simulation::results ensemble( max_samples );
    simulation::zero_results( ensemble );

    size_t ensemble_size = run_functions.size();

    // MONTE CARLO RUNS
    // Embarrassingly parallel - simulation per thread
    #pragma omp parallel for schedule(dynamic, 1) default(none) shared(ensemble) firstprivate(ensemble_size, run_functions)
    for( unsigned int run_id=0; run_id<ensemble_size; run_id++ )
    {
        // Simulate a single realisation of the system
        auto results = run_functions[run_id]();

        // Copy the results into the ensemble
        #pragma omp critical
        {
            ensemble.energy_loss += results.energy_loss;
            for( unsigned int j=0; j<results.N; j++ )
            {
                ensemble.mx[j] += results.mx[j];
                ensemble.my[j] += results.my[j];
                ensemble.mz[j] += results.mz[j];

            } // end copying ensemble
        }
        // Copy in the field and time values
        if( run_id == 0 )
            for( unsigned int j=0; j<results.N; j++ )
            {

                ensemble.time[j] = results.time[j];
                ensemble.field[j] = results.field[j];
            }

    }
    for( unsigned int i=0; i<ensemble.N; i++ )
    {
        ensemble.mx[i] /= ensemble_size;
        ensemble.my[i] /= ensemble_size;
        ensemble.mz[i] /= ensemble_size;
    }
    ensemble.energy_loss /= ensemble_size;
    return ensemble;
}
template <typename... T>
std::vector<d3> simulation::ensemble_run_final_state(
    std::function<simulation::results(T...)> run_function,
    std::vector<T>... varying_arguments )
{
    // Create a vector of run functions
    auto run_functions = curry::vector_curry(
        run_function, varying_arguments... );

    // Run the simulation
    size_t ensemble_size = run_functions.size();
    std::vector<d3> states;
    states.resize( ensemble_size );
    #pragma omp parallel for schedule(dynamic, 1) default(none) shared(states) firstprivate(run_functions, ensemble_size)
    for( unsigned int run_id=0; run_id<ensemble_size; run_id++ )
    {
        auto results = run_functions[run_id]();
        // Copy in the final state
        #pragma omp critical
        {
            states[run_id][0] = results.mx[results.N-1];
            states[run_id][1] = results.my[results.N-1];
            states[run_id][2] = results.mz[results.N-1];
        }
    } // end Monte-Carlo loop
    return states;
}

template<typename... T>
struct simulation::results simulation::steady_state_cycle_dynamics(
    std::function<simulation::results(d3, T...)> run_function,
    const int max_samples,
    const double steady_state_condition,
    const std::vector<d3> initial_mags,
    std::vector<T>... varying_arguments )
{
    // Keep track of the previous magnetisation
    std::vector<d3> prev_mags{ initial_mags };

    unsigned int n_field_cycles=0;
    // Keep simulating until the steady state condition is met
    while ( true )
    {
        n_field_cycles++;
        size_t ensemble_size = initial_mags.size();

        auto mags = simulation::ensemble_run_final_state(
            run_function,
            prev_mags,
            varying_arguments... );

        // Compute the ensemble magnetisation before and after
        // Assume magnetisation is taken in the z-direction
        double mag_before=0, mag_after=0;
        for( unsigned int i=0; i<ensemble_size; i++ )
        {
            mag_before += prev_mags[i][2];
            mag_after += mags[i][2];
        }
        mag_before /= ensemble_size;
        mag_after /= ensemble_size;

        // Check the steady state condition
        if( std::abs( mag_before - mag_after ) < steady_state_condition )
        {
            // If the steady state is matched, then we rerun with a
            // full run
            auto full_res = simulation::ensemble_run(
                max_samples,
                run_function,
                mags,
                varying_arguments... );
            LOG(INFO) << "Steady state reached after " << n_field_cycles
                      << " field cycles";
            return full_res;
        }
        // If not reached - run another cycle from the current state
        prev_mags = mags;
        LOG(INFO) << "Error after cycle " << n_field_cycles
                  << ": " << std::abs( mag_before - mag_after );
    }
}
