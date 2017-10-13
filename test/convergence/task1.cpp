///////////////////////////////////
// CONVERGENCE TESTS             //
//                               //
// TASK 1                        //
// Single trajectory (non-stiff) //
///////////////////////////////////

#include "../../include/integrators.hpp"
#include "../../include/rng.hpp"
#include "../../include/io.hpp"
#include <functional>
#include <iostream>
#include <stdio.h>
#include <cmath>

void task1() {

    //                                  aout    bout    j_a     j_b       x_in           t_a           t_b
    using sde_jac = std::function<void(double*,double*,double*,double*,const double*,const double,const double)>;
    //                              drift   diff       state           t
    using sde = std::function<void(double*,double*,const double*,const double)>;

    ////////////////////////////////////////////////////
    // STOCHASTIC DIFFERENTIAL EQUATION               //
    // dX(t) = aX(t) dt + bX(t) dW(t)  [stratonovich] //
    ////////////////////////////////////////////////////

    double a=-1.0, b=1.0; // not stiff
    size_t n_dim=1, w_dim=1;


    // SDE FUNCTION - drift and diffusion
    sde test_sde = [a,b](double *drift, double *diff, const double*in, const double)
        {
            drift[0] = a * in[0];
            diff[0] = b * in[0];
        };


    // SDE AND JACOBIANS - needed for implicit solver
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


    // SDE SOLUTION - analytic solution
    auto solution = [a, b, x0]( double t, double Wt )
        { return x0[0] * std::exp( a * t + b * Wt ); };


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
    x[0] = x0[0];
    for( unsigned int n=1; n<n_steps+1; n++ )
    {
        Wt += rng.get();
        x[n] = solution( dt*n, Wt );
    }
    io::write_array( "output/task1/true", x, n_steps + 1 );

    delete[] x;
}
