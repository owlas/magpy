/////////////////////////////////////////
// CONVERGENCE TESTS                   //
//                                     //
// TASK 5                              //
// Stochastic LLG (Heun vs. implicit)  //
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

void task5()
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
    double thermal_strength=0.01; double thermal_strength_arr[1] = {thermal_strength};
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


    // STEP SIZE
    size_t n_steps=6400;
    double dt = 1e-2;
    size_t n_dt=7;
    int dt_multipliers[7] = {1, 2, 4, 8, 16, 32, 64};
    size_t dt_mult;
    double dts[7];
    for( unsigned int i=0; i<n_dt; i++ )
        dts[i] = dt * dt_multipliers[i];
    io::write_array( "output/task5/dt", dts, n_dt );


    // RANDOM NUMBER GENERATOR
    size_t n_runs = 5000;
    long *seeds = new long[n_runs];
    for ( size_t i=0; i<n_runs; i++ )
        seeds[i] = i*13*( i%5 ); // allocate some seeds
    RngMtNorm rng( 1001, std::sqrt(dt) ); // preallocate, seed is changed later
    double *dw;


    // IMPLICIT SCHEME PARAMS
    double eps = 1e-9;
    double max_iter = 100;
    size_t n_dim=3;
    size_t w_dim=3;


    // ALLOCATE MEM FOR RESULTS
    // store final state for each run for each dt
    double *implicit_errs = new double[n_runs*3];
    double *heun_errs = new double[n_runs*3];
    double *x = new double[3*(n_steps+1)]; // +1 for the initial condition
    char fname[100];


    // SIMULATE
    std::cout << std::endl;
    std::cout << "Executing task 5..." << std::endl;


    for( unsigned int i=0; i<n_dt; i++ )
    {
        std::cout << "Simulating dt " << i << " of " << n_dt << std::endl;
        std::cout << "----------" << std::endl;

        dt_mult = dt_multipliers[i];
        dw = new double[3*n_steps/dt_mult];

        // Do multiple runs
        for( unsigned int run=0; run<n_runs; run++ )
        {

            if( (run%100) == 0)
                std::cout << "Simulating run " << run << " of " << n_runs << std::endl;

            // Set up 3-dimensional Wiener process
            rng = RngMtNorm(seeds[run], 1.0/std::sqrt((double) dt_mult) );
            for( size_t n=0; n<n_steps/dt_mult; n++ )
            {
                for( size_t i=0; i<3; i++ )
                    dw[n*3 + i] = 0.0;
                for( unsigned int i=0; i<dt_mult; i++)
                    for( size_t j=0; j<3; j++ )
                        dw[n*3 + j] += rng.get();
            }


            // IMPLICIT MIDPOINT
            driver::implicit_midpoint( x, m0, dw, llg_sde_jac, n_dim, w_dim,
                                       n_steps/dt_mult, t0, dt*dt_mult, eps, max_iter );
            implicit_errs[run*3 + 0] = x[n_steps/dt_mult*3 + 0];
            implicit_errs[run*3 + 1] = x[n_steps/dt_mult*3 + 1];
            implicit_errs[run*3 + 2] = x[n_steps/dt_mult*3 + 2];

            if( (i==0) && (run==0) )
                io::write_array( "output/task5/example_sol", x, 3*(n_steps+1));

            // HEUN SCHEME
            driver::heun( x, m0, dw, llg_sde, n_steps/dt_mult,
                          n_dim, w_dim, dt*dt_mult );
            heun_errs[run*3 + 0] = x[n_steps/dt_mult*3 + 0];
            heun_errs[run*3 + 1] = x[n_steps/dt_mult*3 + 1];
            heun_errs[run*3 + 2] = x[n_steps/dt_mult*3 + 2];
        }
        sprintf( fname, "output/task5/implicit%d", i );
        io::write_array( fname, implicit_errs, n_runs*3);

        sprintf( fname, "output/task5/heun%d", i );
        io::write_array( fname, heun_errs, n_runs*3);

        delete[] dw;
    }

    delete[] x; delete[] heff; delete[] jeff; delete[] seeds;
    delete[] implicit_errs; delete[] heun_errs;
}
