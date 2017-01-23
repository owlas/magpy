// simulation.cpp
// implementation of functions for simulating magnetic dynamics
//
// Oliver W. Laslett (2016)
// O.Laslett@soton.ac.uk
#include "../include/simulation.hpp"
#include "../include/llg.hpp"
#include "../include/integrators.hpp"
#include "../include/io.hpp"
#include "../include/field.hpp"
#include "../include/trap.hpp"
#include "../include/easylogging++.h"
#include "../include/optimisation.hpp"
#include <exception>
#ifdef USEMKL
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif

using namespace std::placeholders;
using sde_function = std::function<void(double*,const double*,const double)>;
using sde_jac = std::function<void(double*,double*,double*,double*,
                                   const double*,const double, const double)>;

struct simulation::results simulation::full_dynamics(
    const double damping,
    const double thermal_field_strength,
    const d3 anis_axis,
    const std::function<double(double)> applied_field,
    const d3 initial_magnetisation,
    const double time_step,
    const double end_time,
    Rng &rng,
    const bool renorm,
    const int max_samples )
{
    size_t dims = 3;

    /*
      To ease memory requirements - can specify the maximum number of
      samples to store in memory. This is used to compute the number
      of integration steps per write to the in-memory results arrays.

      max_samples=-1 is equivalent to max_samples=N_steps

      The sampling interval is taken to be regularly spaced and is interpolated
      from the integration steps using a zero-order-hold technique.
     */
    const size_t N_samples = max_samples==-1 ?
        int( end_time / time_step ) + 1
        : max_samples;
    const double sampling_time = end_time / ( N_samples-1 );

    // allocate memory for results
    simulation::results res( N_samples );


    // Allocate matrices needed for the midpoint method
    double *state = new double[dims*2]; // stores old and new state
                                        // i.e 2 *dims
    double *dwm = new double[dims];
    double *a_work = new double[dims];
    double *b_work = new double[dims*dims];
    double *adash_work = new double[dims*dims];
    double *bdash_work = new double[dims*dims*dims];
    double *x_guess = new double[dims];
    double *x_opt_tmp = new double[dims];
    double *x_opt_jac = new double[dims*dims];
    lapack_int *x_opt_ipiv = new lapack_int[dims];

    // Limits for the implicit solver
    const double eps=1e-9;
    const size_t max_iter=1000;

    // Copy in the initial state
    res.time[0] = 0;
    res.field[0] = applied_field( 0 );
    res.mx[0] = initial_magnetisation[0];
    res.my[0] = initial_magnetisation[1];
    res.mz[0] = initial_magnetisation[2];

    // The wiener paths
    double wiener[3];

    // The effective field and its Jacobian is updated at each time step
    double heff[3];
    double heffjac[9];
    double happ[3];

    // Vars for loops
    unsigned int step = 0;
    double t = 0;
    double hz = 0;
    double pstate[3], nstate[3];
    nstate[0] = initial_magnetisation[0];
    nstate[1] = initial_magnetisation[1];
    nstate[2] = initial_magnetisation[2];

    /*
      The time for each point in the regularly spaced grid is
      known. We want to obtain the state at each step
     */
    for( unsigned int sample=1; sample<N_samples; sample++ )
    {
        // Perform a simulation step until we breach the next sampling point
        while ( t <= sample*sampling_time )
        {
            // take a step
            pstate[0] = nstate[0];
            pstate[1] = nstate[1];
            pstate[2] = nstate[2];
            step++;

            // Compute current time
            // When max_samples=-1 uses sampling_time directly to avoid rounding
            // errors
            t = max_samples==-1 ? sample*sampling_time : step*time_step;

            // Compute the applied field - always in the z-direction
            happ[2] = applied_field( t );

            // Assumes that applied field is constant over the period
            // Bind the parameters to create the required SDE function
            sde_jac sde = std::bind(
                llg::jacobians_with_update, _1, _2, _3, _4, heff, heffjac,
                _5, _6, _7, happ, anis_axis.data(), damping,
                thermal_field_strength );

            // Generate the wiener increments
            for( unsigned int i=0; i<3; i++ )
                wiener[i] = rng.get();

            // perform integration step
            int errcode = integrator::implicit_midpoint(
                nstate, dwm, a_work, b_work, adash_work, bdash_work, x_guess,
                x_opt_tmp, x_opt_jac, x_opt_ipiv, pstate, wiener, sde,
                dims, dims, t, time_step, eps, max_iter );
            if( errcode != optimisation::SUCCESS )
                LOG(FATAL) << "integration error code: " << errcode;

            // Renormalise the length of the magnetisation
            if( renorm  )
            {
                double norm = cblas_dnrm2( 3, nstate, 1 );
                for( unsigned int i=0; i<dims; i++ )
                    nstate[i] = nstate[i]/norm;
            } // end renormalisation

        } // end integration stepping loop

        /*
          Once this point is reached, we are currently one step beyond
          the desired sampling point. Use a zero-order-hold:
          i.e. take the previous state before the sampling time as the
          state at the sampling time.
         */
        res.time[sample] = sample*sampling_time; // sampling time
        res.mx[sample] = pstate[0];
        res.my[sample] = pstate[1];
        res.mz[sample] = pstate[2];
        res.field[sample] = applied_field( sample*sampling_time );
    } // end sampling loop

    delete[] state;
    delete[] a_work;
    delete[] b_work;
    delete[] dwm;
    delete[] adash_work;
    delete[] bdash_work;
    delete[] x_guess;
    delete[] x_opt_tmp;
    delete[] x_opt_jac;
    delete[] x_opt_ipiv;

    return res; // Ensure elison else copy is made and dtor is called!
}

struct simulation::results simulation::ensemble_dynamics(
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
    const size_t ensemble_size )
{
    // Initialise space for the ensemble results
    struct simulation::results ensemble( max_samples );
    simulation::zero_results( ensemble ); // initialise all values to zero

    // MONTE CARLO RUNS
    // Embarrassingly parallel - simulation per thread
#pragma omp parallel for schedule(dynamic, 1) shared(ensemble) firstprivate(ensemble_size, damping, thermal_field_strength, anis_axis, applied_field, initial_mags, time_step, end_time, rngs, renorm, max_samples) default(none)
    for( unsigned int run_id=0; run_id<ensemble_size; run_id++ )
    {
        // Simulate a single realisation of the system
        auto results = simulation::full_dynamics(
            damping, thermal_field_strength, anis_axis, applied_field,
            initial_mags[run_id], time_step, end_time, *(rngs[run_id].get()),
            renorm, max_samples );

        // Copy the results into the ensemble
	#pragma omp critical
        for( unsigned int j=0; j<results.N; j++ )
        {
            ensemble.mx[j] += results.mx[j];
            ensemble.my[j] += results.my[j];
            ensemble.mz[j] += results.mz[j];
        } // end copying to ensemble

        // Copy in the field and time values
        if( run_id==0 )
            for( unsigned int j=0; j<results.N; j++ )
            {
                ensemble.time[j] = results.time[j];
                ensemble.field[j] = results.field[j];
            }
    } // end Monte-Carlo loop

    for( unsigned int i=0; i<ensemble.N; i++ )
    {
        ensemble.mx[i] /= ensemble_size;
        ensemble.my[i] /= ensemble_size;
        ensemble.mz[i] /= ensemble_size;
    }
    return ensemble;
}

struct simulation::results simulation::ensemble_dynamics(
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
    const size_t ensemble_size )
{
    std::vector<d3> init_mags;
    for( unsigned int i=0; i<ensemble_size; i++ )
        init_mags.push_back( initial_mag );
    return simulation::ensemble_dynamics(
        damping, thermal_field_strength, anis_axis, applied_field, init_mags,
        time_step, end_time, rngs, renorm, max_samples, ensemble_size );
}

std::vector<d3> simulation::ensemble_final_state(
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
    const size_t ensemble_size )
{
    // Initialise the total states
    std::vector<d3> states( initial_mags );

    // MONTE CARLO RUNS
    // Embarrassingly parallel - simulation per thread
#pragma omp parallel for schedule(dynamic, 1) shared(states) firstprivate(ensemble_size, damping, thermal_field_strength, anis_axis, applied_field, initial_mags, time_step, end_time, rngs, renorm, max_samples) default(none)
    for( unsigned int run_id=0; run_id<ensemble_size; run_id++ )
    {
        // Simulate a single realisation of the system
        auto results = simulation::full_dynamics(
            damping, thermal_field_strength, anis_axis, applied_field,
            initial_mags[run_id], time_step, end_time, *(rngs[run_id].get()),
            renorm, max_samples );
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

struct simulation::results simulation::steady_state_cycle_dynamics(
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
    const double steady_state_condition )
{
    std::vector<d3> prev_mags;
    for( unsigned int i=0; i<ensemble_size; i++ )
        prev_mags.push_back( initial_magnetisation );

    unsigned int n_field_cycles=0;
    // Keep simulating until the steady state condition is met
    while ( true )
    {
        n_field_cycles++;
        auto mags = simulation::ensemble_final_state(
            damping, thermal_field_strength, anis_axis, applied_field,
            prev_mags, time_step, applied_field_period, rngs,
            renorm, max_samples, ensemble_size );

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
            auto full_res = simulation::ensemble_dynamics(
                damping, thermal_field_strength, anis_axis, applied_field,
                mags, time_step, applied_field_period, rngs, renorm,
                max_samples, ensemble_size );
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

double simulation::power_loss(
    const struct results &res,
    double K, double Ms, double Hk, double f )
{
    double area = trap::trapezoidal( res.field.get(), res.mz.get(), res.N );
    return 2*K*Ms*Hk*area*f;
}

void simulation::save_results( const std::string fname, const struct results &res )
{
    std::stringstream magx_fname, magy_fname, magz_fname, field_fname, time_fname;
    magx_fname << fname << ".mx";
    magy_fname << fname << ".my";
    magz_fname << fname << ".mz";
    field_fname << fname << ".field";
    time_fname << fname << ".time";
    int err;
    err = io::write_array( magx_fname.str(), res.mx.get(), res.N );
    if( err != 0 )
        throw std::runtime_error( "failed to write file" );
    err = io::write_array( magy_fname.str(), res.my.get(), res.N );
    if( err != 0 )
        throw std::runtime_error( "failed to write file" );
    err = io::write_array( magz_fname.str(), res.mz.get(), res.N );
    if( err != 0 )
        throw std::runtime_error( "failed to write file" );
    err = io::write_array( field_fname.str(), res.field.get(), res.N );
    if( err != 0 )
        throw std::runtime_error( "failed to write file" );
    err = io::write_array( time_fname.str(), res.time.get(), res.N );
    if( err != 0 )
        throw std::runtime_error( "failed to write file" );
}

void simulation::zero_results( struct simulation::results &res )
{
    for( unsigned int i=0; i<res.N; i++ )
        res.mx[i] = res.my[i] = res.mz[i] = res.field[i] = res.time[i] = 0;
}


// struct simulation::results simulation::dom_ensemble_dynamics(
//     const double damping,
//     const double radius,
//     const d3 anis_axis,
//     const std::function<double(double)> applied_field,
//     const std::vector<d3> initial_mags,
//     const double time_step,
//     const double sim_time,
//     const rng_vec rngs,
//     const int max_samples,
//     const size_t ensemble_size )
// {

// }
