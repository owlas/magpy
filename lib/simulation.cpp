// simulation.cpp
// implementation of functions for simulating magnetic dynamics
//
// Oliver W. Laslett (2016)
// O.Laslett@soton.ac.uk
#include "../include/simulation.hpp"
#include "../include/llg.hpp"
#include "../include/integrators.hpp"
#include "../include/easylogging++.h"

using namespace std::placeholders;
using sde_function = std::function<void(double*,const double*,const double)>;

struct simulation::results simulation::full_dynamics(
    const double damping,
    const double thermal_field_strength,
    const std::function<double(double)> applied_field,
    const d3 initial_magnetisation,
    const double time_step,
    const double end_time,
    std::mt19937_64 rng )
{
    // compute the number of steps to take
    size_t N_steps = int( end_time / time_step ) + 1;
    size_t dims = 3;

    LOG(INFO) << "Simulating " << N_steps << " steps of full dynamics.";

    // allocate memory for results
    simulation::results res( N_steps );

    // Copy in the initial state
    res.time[0] = 0;
    res.field[0] = applied_field( 0 );
    for( unsigned int i=0; i<dims; i++ )
        res.magnetisation[i] = initial_magnetisation[i];

    // Generate the wiener paths needed for simulation
    std::normal_distribution<double> dist( 0, thermal_field_strength );
    size_t wiener_size = dims*(N_steps-1); // not needed for initial state
    double *wiener = new double[wiener_size];
    for( unsigned int i=0; i<wiener_size; i++ )
        wiener[i] = dist( rng );


    // Allocate matrices needed for Heun scheme
    double *drift_arr = new double[dims];
    double *trial_drift_arr = new double[dims];
    double *diffusion_mat = new double[dims*dims];
    double *trial_diffusion_mat = new double[dims*dims];

    // Run the simulation
    for( unsigned int step=1; step<N_steps; step++ )
    {
        // update time
        double t = step*time_step;
        res.time[step] = t;

        // compute the applied field
        double hz[3] = { 0, 0, applied_field( t ) };

        // bind parameters to the LLG function
        sde_function drift = std::bind( llg::drift, _1, _2, _3, damping, hz );
        sde_function diffusion = std::bind(
            llg::diffusion, _1, _2, _3, thermal_field_strength, damping );

        integrator::heun(
            &res.magnetisation[dims*step], drift_arr, trial_drift_arr, diffusion_mat,
            trial_diffusion_mat, &res.magnetisation[dims*(step-1)],
            &wiener[dims*(step-1)], drift, diffusion, dims,
            dims, t, time_step );
    }
    delete[] drift_arr; delete[] trial_drift_arr;
    delete[] diffusion_mat; delete[] trial_diffusion_mat;
    delete[] wiener;

    return res; // Ensure elison else copy is made and dtor is called!
}
