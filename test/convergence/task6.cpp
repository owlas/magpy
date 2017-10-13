/////////////////////////////////////////
// CONVERGENCE TESTS                   //
//                                     //
// TASK 6                              //
// Stochastic LLG (implicit tol.)      //
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

void task6()
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


    // Initial condition
    double m0[3] = {1.0, 0.0, 0.0};
    double t0 = 0.0;


    // STEP SIZE
    size_t n_steps=6400;
    double dt = 2e-3;
    size_t n_dt=7;
    int dt_multipliers[7] = {1, 2, 4, 8, 16, 32, 64};
    size_t dt_mult;
    double dts[7];
    for( unsigned int i=0; i<n_dt; i++ )
        dts[i] = dt * dt_multipliers[i];
    io::write_array( "output/task6/dt", dts, n_dt );


    // RANDOM NUMBER GENERATOR
    size_t n_runs = 5000;
    long *seeds = new long[n_runs];
    for ( size_t i=0; i<n_runs; i++ )
        seeds[i] = i*13*( i%5 ); // allocate some seeds
    RngMtNorm rng( 1001, std::sqrt(dt) ); // preallocate, seed is changed later
    double *dw;


    // IMPLICIT SCHEME PARAMS
    size_t n_eps = 3;
    double epss[3] = {1e-4, 1e-3, 1e-2};
    double eps;
    double *errs[3];
    double max_iter = 100;
    size_t n_dim=3;
    size_t w_dim=3;
    io::write_array( "output/task6/eps", epss, n_eps );

    // ALLOCATE MEM FOR RESULTS
    // store final state for each run for each dt
    for( size_t i=0; i<n_eps; i++ )
        errs[i] = new double[n_runs*3];
    double *x = new double[3*(n_steps+1)]; // +1 for the initial condition
    char fname[100];


    // SIMULATE
    std::cout << std::endl;
    std::cout << "Executing task 6..." << std::endl;
    for( unsigned int i=0; i<n_dt; i++ )
    {
        std::cout << "Simulating dt " << i << " of " << n_dt << std::endl;
        std::cout << "**********" << std::endl;

        dt_mult = dt_multipliers[i];
        dw = new double[3*n_steps/dt_mult];

        // Simulate different tolerances
        for( unsigned int j=0; j<n_eps; j++ )
        {
            std::cout << "Simulating eps " << j << " of " << n_eps << std::endl;
            std::cout << "----------" << std::endl;

            eps = epss[j];

            // Do multiple runs
            for( unsigned int run=0; run<n_runs; run++ )
            {

                if( (run%500) == 0)
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
                errs[j][run*3 + 0] = x[n_steps/dt_mult*3 + 0];
                errs[j][run*3 + 1] = x[n_steps/dt_mult*3 + 1];
                errs[j][run*3 + 2] = x[n_steps/dt_mult*3 + 2];

                if( (i+run)==0 )
                {
                    sprintf( fname, "output/task6/example_sol_%d", j );
                    io::write_array( fname, x, 3*(n_steps+1));
                }

            } // end for each run
            sprintf( fname, "output/task6/implicit%d_%d", j, i );
            io::write_array( fname, errs[j], n_runs*3);
        } // end for each eps value
        delete[] dw;
    } // end for each dt value

    delete[] x; delete[] heff; delete[] jeff; delete[] seeds;
    for( size_t i=0; i<n_eps; i++ )
        delete[] errs[i];
}
