/////////////////////////////////////////
// CONVERGENCE TESTS                   //
//                                     //
// TASK 4                              //
// Deterministic LLG                   //
// Single particle, applied field only //
/////////////////////////////////////////

#include "../../include/integrators.hpp"
#include "../../include/rng.hpp"
#include "../../include/io.hpp"
#include "../../include/llg.hpp"
#include "../../include/field.hpp"
#include "../../include/constants.hpp"
#include <functional>
#include <cmath>
#include <stdio.h>
#include <iostream>

void task4()
{

    //                                  aout    bout    j_a     j_b       x_in           t_a           t_b
    using sde_jac = std::function<void(double*,double*,double*,double*,const double*,const double,const double)>;
    //                              drift   diff       state           t
    using sde = std::function<void(double*,double*,const double*,const double)>;

    ///////////////////////////////////////
    // LAUNDAU-LIFSHITZ-GILBERT EQUATION //
    ///////////////////////////////////////

    // LLG parameters
    double H=0.01;
    double alpha=0.1; double alpha_arr[1] = {alpha};
    double thermal_strength=0.0; double thermal_strength_arr[1] = {thermal_strength};
    size_t n_particles=1;
    double *heff = new double[3]; // allocate work arrays
    double *jeff = new double[9]; // allocate work arrays


    // Define the effective field (constant in z_direction [index 2])
    std::function<double(const double)> happ = [H](const double) {return H;};

    std::function<void(double*,const double*,const double)> heff_func =
        [happ](double *out, const double*, const double t)
        {
            field::zero_all_field_terms( out, 3 );
            field::multi_add_applied_Z_field_function( out, happ, t, 1 );
        };

    std::function<void(double*, const double*, const double)> heff_jac =
        [](double *out, const double*, const double)
        {
            field::zero_all_field_terms( out, 9 );
        };


    // Define the SDE representing the LLG
    sde_jac llg_sde_jac = [heff_func, heff_jac, heff, jeff,
                           n_particles, alpha_arr, thermal_strength_arr]
        (double *drift, double *diffusion, double *jdrift, double *jdiffusion,
         const double *state, const double a_t, const double )
        {
            llg::multi_stochastic_llg_jacobians_field_update(
                drift, diffusion,
                jdrift, jdiffusion,
                heff, jeff,
                state,
                a_t,
                alpha_arr,
                thermal_strength_arr,
                n_particles,
                heff_func,
                heff_jac );
        };

    // Very inefficient hack for Heun scheme (need to change interface)
    std::function<void(double*,double*,const double*,const double)> llg_sde =
        [llg_sde_jac] (double *drift, double*diffusion,
                       const double*state, const double time)
    {
        double jdrift[9];
        double jdiffusion[27];
        llg_sde_jac( drift, diffusion, jdrift, jdiffusion, state, time, time );
    };


    // Initial condition
    double m0[3] = {1.0, 0.0, 0.0};
    double t0 = 0.0;


    // Solution to the LLG
    auto llg_solution = [alpha,H](double *out, double t)
        {
            out[0] = 1.0/std::cosh(alpha*H*t) * std::cos(H*t);
            out[1] = 1.0/std::cosh(alpha*H*t) * std::sin(H*t);
            out[2] = std::tanh(alpha*H*t);
        };


    // STEP SIZE
    size_t n_steps=70000;
    double dt = 1e-1;
    size_t n_dt=4;
    int dt_multipliers[4] = {1, 10, 100, 1000};
    size_t dt_mult;
    double dts[4];
    for( unsigned int i=0; i<n_dt; i++ )
        dts[i] = dt * dt_multipliers[i];
    io::write_array( "output/task4/dt", dts, n_dt );


    // IMPLICIT SCHEME PARAMS
    double eps = 1e-9;
    double max_iter = 100;
    size_t n_dim=3;
    size_t w_dim=3;


    // ALLOCATE MEM FOR WIENER PROCESS
    double *dw = new double[w_dim*n_steps];
    for( size_t n=0; n<w_dim*n_steps; n++ )
        dw[n] = 0.0; // no stochastic needed


    // ALLOCATE MEM FOR RESULTS
    double *x = new double[n_dim*(n_steps+1)]; // +1 for the initial condition
    char fname[100];

    // SIMULATE SOLUTION
    std::cout << std::endl;
    std::cout << "Executing task 4" << std::endl;
    for( unsigned int i=0; i<n_dt; i++ )
    {
        dt_mult = dt_multipliers[i];
        std::cout << "Simulating with time step: " << dt * dt_mult << std::endl;
        std::cout << "For number of steps: " << n_steps / dt_mult << std::endl;

        // IMPLICIT MIDPOINT
        driver::implicit_midpoint( x, m0, dw, llg_sde_jac, n_dim, w_dim,
                                   n_steps/dt_mult, t0, dt*dt_mult, eps, max_iter );
        sprintf( fname, "output/task4/implicit_slow%d", i );
        io::write_array( fname, x, n_dim*(n_steps/dt_mult + 1) );

        driver::implicit_midpoint( x, m0, dw, llg_sde_jac, n_dim, w_dim,
                                   n_steps/dt_mult, t0, dt*dt_mult, eps*100, max_iter );
        sprintf( fname, "output/task4/implicit_mid%d", i );
        io::write_array( fname, x, n_dim*(n_steps/dt_mult + 1) );

        driver::implicit_midpoint( x, m0, dw, llg_sde_jac, n_dim, w_dim,
                                   n_steps/dt_mult, t0, dt*dt_mult, eps*100*100, max_iter );
        sprintf( fname, "output/task4/implicit_fast%d", i );
        io::write_array( fname, x, n_dim*(n_steps/dt_mult + 1) );

        // HEUN SCHEME
        driver::heun( x, m0, dw, llg_sde, n_steps/dt_mult,
                      n_dim, w_dim, dt*dt_mult );
        sprintf( fname, "output/task4/heun%d", i );
        io::write_array( fname, x, n_dim*(n_steps/dt_mult + 1));
    }

    // ANALYTIC SOLUTION
    x[0] = m0[0]; x[1] = m0[1]; x[2] = m0[2];
    for( unsigned int n=1; n<n_steps+1; n++ )
        llg_solution( x + n_dim*n, dt*n );
    io::write_array( "output/task4/true", x, n_dim*(n_steps + 1) );

    delete[] x; delete[] dw; delete[] heff; delete[] jeff;
}
