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
#include <exception>

using namespace std::placeholders;
using sde_function = std::function<void(double*,const double*,const double)>;

struct simulation::results simulation::full_dynamics(
    const double damping,
    const double thermal_field_strength,
    const d3 anis_axis,
    const std::function<double(double)> applied_field,
    const d3 initial_magnetisation,
    const double time_step,
    const double end_time,
    std::mt19937_64 rng,
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
    const size_t N_steps = int( end_time/time_step ) + 1;
    const size_t N_samples = max_samples==-1 ?
        int( end_time / time_step ) + 1
        : max_samples;
    const double sampling_time = end_time / N_samples;

    // allocate memory for results
    simulation::results res( N_samples );


    // Allocate matrices needed for Heun scheme
    double *state = new double[dims*2]; // stores old and new state
                                        // i.e 2 *dims
    double *drift_arr = new double[dims];
    double *trial_drift_arr = new double[dims];
    double *diffusion_mat = new double[dims*dims];
    double *trial_diffusion_mat = new double[dims*dims];

    // Copy in the initial state
    res.time[0] = 0;
    res.field[0] = applied_field( 0 );
    res.mx[0] = initial_magnetisation[0];
    res.my[0] = initial_magnetisation[1];
    res.mz[0] = initial_magnetisation[2];

    // Generate the wiener paths needed for simulation
    std::normal_distribution<double> dist( 0, thermal_field_strength );
    size_t wiener_size = dims*(N_steps-1); // not needed for initial state
    double *wiener = new double[wiener_size];
    for( unsigned int i=0; i<wiener_size; i++ )
        wiener[i] = dist( rng );


    // The effective field is updated at each time step
    double heff[3];

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

            // Get pointers for the previous and next states and
            // compute current time
            // prev_state = state+(step-1)%2;
            // next_state = state+step%2;
            t = step*time_step;

            // Compute the anisotropy field
            field::uniaxial_anisotropy( heff, pstate, anis_axis.data() );

            // Compute the applied field - always in the z-direction
            hz = applied_field( t );
            heff[2] += hz;

            // bind parameters to the LLG functions
            sde_function drift = std::bind(
                llg::drift, _1, _2, _3, damping, heff );
            sde_function diffusion = std::bind(
                llg::diffusion, _1, _2, _3, thermal_field_strength, damping );

            // perform integration step
            integrator::heun(
                nstate, drift_arr, trial_drift_arr, diffusion_mat,
                trial_diffusion_mat, pstate, &wiener[dims*(step-1)], drift,
                diffusion, dims, dims, t, time_step );

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

    delete[] drift_arr; delete[] trial_drift_arr;
    delete[] diffusion_mat; delete[] trial_diffusion_mat;
    delete[] wiener; delete[] state;

    return res; // Ensure elison else copy is made and dtor is called!
}

double simulation::power_loss(
    const struct results &res,
    double v, double K, double Ms, double Hk, double f )
{
    double area = trap::trapezoidal( res.mz, res.field, res.N );
    return 2*K*Ms*Hk*area*f/v;
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
    err = io::write_array( magx_fname.str(), res.mx, res.N );
    if( err != 0 )
        throw std::runtime_error( "failed to write file" );
    err = io::write_array( magy_fname.str(), res.my, res.N );
    if( err != 0 )
        throw std::runtime_error( "failed to write file" );
    err = io::write_array( magz_fname.str(), res.mz, res.N );
    if( err != 0 )
        throw std::runtime_error( "failed to write file" );
    err = io::write_array( field_fname.str(), res.field, res.N );
    if( err != 0 )
        throw std::runtime_error( "failed to write file" );
    err = io::write_array( time_fname.str(), res.time, res.N );
    if( err != 0 )
        throw std::runtime_error( "failed to write file" );
}
