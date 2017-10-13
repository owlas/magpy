/// Test convergence
/**
 * @deprecated
 *
 */

#include "../include/integrators.hpp"
#include "../include/rng.hpp"
#include "../include/io.hpp"
#include "../include/llg.hpp"
#include "../include/field.hpp"
#include "../include/constants.hpp"
#include <functional>
#include <vector>
#include <cmath>
#include <iostream>
#include <stdio.h>

//                                  aout    bout    j_a     j_b       x_in           t_a           t_b
using sde_jac = std::function<void(double*,double*,double*,double*,const double*,const double,const double)>;
//                              drift   diff       state           t
using sde = std::function<void(double*,double*,const double*,const double)>;

int main()
{
    // // STOCHASTIC DIFFERENTIAL EQUATION
    // // dX(t) = aX(t) dW(t)  [stratonovich]
    // double a=3.0;
    // size_t n_dim=1, w_dim=1;

    // // Individual drift and diffusion terms
    // sde test_drift = [](double *out, const double*, const double)
    //     { out[0] = 0.0; };
    // sde test_diffusion = [a](double *out, const double *in, const double)
    //     { out[0] = a*in[0]; };

    // // Combined equation with jacobian
    // sde_jac test_equation = [a](double *aout, double *bout, double *ja,
    //                             double *jb, const double *xin,
    //                             const double, const double)
    //     {
    //         aout[0] = 0.0;
    //         bout[0] = a * xin[0];
    //         ja[0] = 0.0;
    //         jb[0] = a;
    //     };

    // STOCHASTIC DIFFERENTIAL EQUATION
    // dX(t) = aX(t) dt + bX(t) dW(t)  [stratonovich]
    double a=1.0, b=1.0; // not stiff
    // double a=-20.0, b=5.0 // stiff
    size_t n_dim=1, w_dim=1;

    // Individual drift and diffusion terms
    sde test_sde = [a,b](double *drift, double *diff, const double*in, const double)
        {
            drift[0] = a * in[0];
            diff[0] = b * in[0];
        };

    // Combined equation with jacobian
    sde_jac test_sde_jac = [a,b](double *aout, double *bout, double *ja,
                                 double *jb, const double *xin,
                                 const double, const double)
        {
            aout[0] = a * xin[0];
            bout[0] = b * xin[0];
            ja[0] = a;
            jb[0] = b;
        };


    // INITIAL CONDITION
    double x0[1] = { 1.0 };
    double t0 = 0.0;


    // SDE SOLUTION
    auto solution = [a, b, x0]( double t, double Wt )
        { return x0[0] * std::exp( a * t + b * Wt ); };



    ////////////////////////////////////////////
    // TASK 1 - Single trajectory (non stiff) //
    ////////////////////////////////////////////

    // STEP SIZE
    size_t n_steps=10000;
    double dt = 1e-5;
    size_t n_dt=4;
    int dt_multipliers[4] = {1, 10, 100, 1000};
    size_t dt_mult;
    double dts[4];
    for( unsigned int i=0; i<n_dt; i++ )
        dts[i] = dt * dt_multipliers[i];
    io::write_array( "output/task1/dt", dts, n_dt );


    // RANDOM NUMBER GENERATOR
    const long seed = 1001;
    RngMtNorm rng( seed, std::sqrt(dt) );
    double *dw;

    // IMPLICIT SCHEME PARAMS
    double eps = 1e-8;
    size_t max_iter = 1000;


    // ALLOCATE MEM FOR RESULTS
    double *x = new double[n_steps+1]; // +1 for the initial condition
    char fname[100];


    // SIMULATE SOLUTION
    std::cout << std::endl;
    std::cout << "Executing task 1" << std::endl;
    for( unsigned int i=0; i<n_dt; i++ )
    {
        dt_mult = dt_multipliers[i];
        std::cout << "Simulating with time step: " << dt * dt_mult << std::endl;
        std::cout << "For number of steps: " << n_steps / dt_mult << std::endl;

        dw = new double[n_steps/dt_mult];
        rng = RngMtNorm( seed, 1.0/std::sqrt((double) dt_mult) );
        for( size_t n=0; n<n_steps/dt_mult; n++ )
        {
            dw[n] = 0.0;
            for( unsigned int i=0; i<dt_mult; i++)
                dw[n] += rng.get();
        }

        // IMPLICIT MIDPOINT
        driver::implicit_midpoint( x, x0, dw, test_sde_jac, n_dim, w_dim,
                                   n_steps/dt_mult, t0, dt*dt_mult, eps, max_iter );
        sprintf( fname, "output/task1/implicit%d", i );
        io::write_array( fname, x, n_steps/dt_mult + 1 );
        // HEUN SCHEME
        driver::heun( x, x0, dw, test_sde, n_steps/dt_mult,
                      n_dim, w_dim, dt*dt_mult );
        sprintf( fname, "output/task1/heun%d", i );
        io::write_array( fname, x, n_steps/dt_mult + 1);
        delete[] dw;
    }

    // ANALYTIC SOLUTION
    double Wt=0;
    rng = RngMtNorm( seed, std::sqrt( dt ) );
    for( unsigned int n=0; n<n_steps+1; n++ )
    {
        Wt += rng.get();
        x[n] = solution( dt*n, Wt );
    }
    io::write_array( "output/task1/true", x, n_steps + 1 );

    delete[] x;

    //*************************************************************//
    /////////////////////////////////////////////////////////////////



    /////////////////////////////////////////////////////////////////
    // Task 2 - Single particle, applied field only, deterministic //
    /////////////////////////////////////////////////////////////////


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
    t0 = 0.0;


    // Solution to the LLG
    auto llg_solution = [alpha,H](double *out, double t)
        {
            out[0] = 1.0/std::cosh(alpha*H*t) * std::cos(H*t);
            out[1] = 1.0/std::cosh(alpha*H*t) * std::sin(H*t);
            out[2] = std::tanh(alpha*H*t);
        };


    // STEP SIZE
    n_steps=70000;
    dt = 1e-1;
    for( unsigned int i=0; i<n_dt; i++ )
        dts[i] = dt * dt_multipliers[i];
    io::write_array( "output/task2/dt", dts, n_dt );


    // IMPLICIT SCHEME PARAMS
    eps = 1e-9;
    max_iter = 100;
    n_dim=3;
    w_dim=3;


    // ALLOCATE MEM FOR WIENER PROCESS
    dw = new double[w_dim*n_steps];
    for( size_t n=0; n<w_dim*n_steps; n++ )
        dw[n] = 0.0; // no stochastic needed


    // ALLOCATE MEM FOR RESULTS
    x = new double[n_dim*(n_steps+1)]; // +1 for the initial condition

    // SIMULATE SOLUTION
    std::cout << std::endl;
    std::cout << "Executing task 2" << std::endl;
    for( unsigned int i=0; i<n_dt; i++ )
    {
        dt_mult = dt_multipliers[i];
        std::cout << "Simulating with time step: " << dt * dt_mult << std::endl;
        std::cout << "For number of steps: " << n_steps / dt_mult << std::endl;

        // IMPLICIT MIDPOINT
        driver::implicit_midpoint( x, m0, dw, llg_sde_jac, n_dim, w_dim,
                                   n_steps/dt_mult, t0, dt*dt_mult, eps, max_iter );
        sprintf( fname, "output/task2/implicit_slow%d", i );
        io::write_array( fname, x, n_dim*(n_steps/dt_mult + 1) );

        driver::implicit_midpoint( x, m0, dw, llg_sde_jac, n_dim, w_dim,
                                   n_steps/dt_mult, t0, dt*dt_mult, eps*100, max_iter );
        sprintf( fname, "output/task2/implicit_mid%d", i );
        io::write_array( fname, x, n_dim*(n_steps/dt_mult + 1) );

        driver::implicit_midpoint( x, m0, dw, llg_sde_jac, n_dim, w_dim,
                                   n_steps/dt_mult, t0, dt*dt_mult, eps*100*100, max_iter );
        sprintf( fname, "output/task2/implicit_fast%d", i );
        io::write_array( fname, x, n_dim*(n_steps/dt_mult + 1) );

        // HEUN SCHEME
        driver::heun( x, m0, dw, llg_sde, n_steps/dt_mult,
                      n_dim, w_dim, dt*dt_mult );
        sprintf( fname, "output/task2/heun%d", i );
        io::write_array( fname, x, n_dim*(n_steps/dt_mult + 1));
    }

    // ANALYTIC SOLUTION
    for( unsigned int n=0; n<n_steps+1; n++ )
        llg_solution( x + n_dim*n, dt*n );
    io::write_array( "output/task2/true", x, n_dim*(n_steps + 1) );

    delete[] x; delete[] dw; delete[] heff; delete[] jeff;

    // std::vector<double> imid_errors;
    // std::vector<double> heun_errors;


    // // SEEDS - we run the simulation 500 times
    // int n_seeds = 500;
    // std::vector<int> seeds;
    // for( int i=0; i<n_seeds; i++ )
    //     seeds.push_back( i );


    // // RUN INTEGRATOR FOR EACH TIME STEP
    // double Wt, imid_err, heun_err;
    // for( auto &dt_mult : dt_multipliers )
    // {
    //     std::cout << "Simulating with time step: " << dt * dt_mult << std::endl;
    //     std::cout << "For number of steps: " << n_steps / dt_mult << std::endl;

    //     // Alloc memory for Wiener process
    //     dw = new double[n_steps/dt_mult];
    //     imid_err = 0.0;
    //     heun_err = 0.0;
    //     for( auto &seed : seeds )
    //     {
    //         // Scale Wiener process to be identical path for all timesteps
    //         rng = RngMtNorm( seed, 1.0/std::sqrt(dt_mult) );
    //         Wt = 0.0;
    //         for( size_t n=0; n<n_steps/dt_mult; n++ )
    //         {
    //             dw[n] = 0.0;
    //             for( unsigned int i=0; i<dt_mult; i++)
    //                 dw[n] += rng.get();
    //             Wt += dw[n] * std::sqrt(dt * dt_mult);
    //         }

    //         // IMPLICIT MIDPOINT
    //         driver::implicit_midpoint( x, x0, dw, test_equation, n_dim, w_dim,
    //                                    n_steps/dt_mult, t0, dt*dt_mult, eps, max_iter );
    //         imid_err += std::pow( std::abs( x[n_steps/dt_mult-1] - solution( Wt ) ), 2 );

    //         // HEUN SCHEME
    //         driver::heun( x, x0, dw, test_drift, test_diffusion, n_steps/dt_mult,
    //                       n_dim, w_dim, dt*dt_mult );
    //         heun_err += std::pow( std::abs( x[n_steps/dt_mult-1] - solution( Wt ) ), 2);
    //     }
    //     imid_err = std::sqrt( imid_err / seeds.size() );
    //     heun_err = std::sqrt( heun_err / seeds.size() );
    //     imid_errors.push_back( imid_err );
    //     heun_errors.push_back( heun_err );
    //     delete[] dw;
    // }
    // delete[] x;

    // std::cout << "IMID ERR:" << std::endl;
    // for( auto &i : imid_errors )
    //     std::cout << i << ",";
    // std::cout << std::endl;
    // std::cout << "HEUN ERR:" << std::endl;
    // for( auto &i : heun_errors )
    //     std::cout << i << ",";
    // std::cout << std::endl;

    return 0;
}
