/// llg convergence tests
/**
 * @deprecated
 */

#include "../include/integrators.hpp"
#include "../include/llg.hpp"
#include "../include/constants.hpp"
#include "../include/rng.hpp"
#include "../include/io.hpp"
#include <functional>
#include <cmath>
#include <iostream>

using sde_jac = std::function<void(double*,double*,double*,double*,const double*,const double,const double)>;

int main()
{
    /////////////////////////////////////////////////////////////////
    // Single particle, applied field only, deterministic (task 2) //
    /////////////////////////////////////////////////////////////////


    // LLG parameters
    double H=0.1, alpha=0.1, thermal_strength=0.0;
    size_t n_particles=1;
    double heff[1], jeff[1]; // allocate work arrays


    // Define the effective field (constant in z_direction [index 2])
    std::function<void(double*,const double*,const double)> heff_func =
        [H](double *out, const double*, const double)
        { out[0]=0.0; out[1]=0.0; out[2]=H; };

    std::function<void(double*, const double*, const double)> heff_jac =
        [](double *out, const double*, const double)
        { out[0]=0.0; out[1]=0.0; out[2]=0.0; };


    // Define the SDE representing the LLG
    sde_jac llg_sde = [heff_func, heff_jac, heff, jeff, n_particles, alpha, thermal_strength]
        (double *drift, double *diffusion, double *jdrift, double *jdiffusion,
         const double *state, const double a_t, const double )
        {
            llg::multi_stochastic_llg_jacobians_field_update(
                drift, diffusion,
                jdrift, jdiffusion,
                heff, jeff,
                state,
                a_t,
                alpha,
                thermal_strength,
                n_particles,
                heff_func,
                heff_jac );
        };


    // Initial condition
    double x0[3] = {1.0, 0.0, 0.0};
    double t0 = 0.0;


    // Solution to the LLG
    auto solution = [alpha,H](double *out, double t)
        {
            out[0] = 1.0/std::cosh(alpha*constants::GYROMAG*H / (1 + alpha*alpha) * t)
                * std::cos(alpha*H/(1+alpha*alpha) * t);
            out[1] = 1.0/std::cosh(alpha*constants::GYROMAG*H / (1 + alpha*alpha) * t)
                * std::sin(alpha*H/(1+alpha*alpha) * t);
            out[2] = std::tanh(alpha*constants::GYROMAG*H / (1 + alpha*alpha) * t );
        };


    // STEP SIZE
    size_t n_steps=10000;
    double dt = 1e-8;
    size_t n_dt=4;
    int dt_multipliers[4] = {1, 10, 100, 1000};
    size_t dt_mult;
    double dts[4];
    for( unsigned int i=0; i<n_dt; i++ )
        dts[i] = dt * dt_multipliers[i];
    io::write_array( "output/task2/dt", dts, n_dt );


    // IMPLICIT SCHEME PARAMS
    double eps = 1e-8;
    size_t max_iter = 1000;
    size_t n_dim=3, w_dim=3;


    // ALLOCATE MEM FOR WIENER PROCESS
    double *dw = new double[w_dim*n_steps];
    for( size_t n=0; n<w_dim*n_steps; n++ )
        dw[n] = 0.0; // no stochastic needed


    // ALLOCATE MEM FOR RESULTS
    double *x = new double[n_dim*(n_steps+1)]; // +1 for the initial condition
    char fname[100];

    // SIMULATE SOLUTION
    std::cout << std::endl;
    std::cout << "Executing task 2" << std::endl;
    for( unsigned int i=0; i<n_dt; i++ )
    {
        dt_mult = dt_multipliers[i];
        std::cout << "Simulating with time step: " << dt * dt_mult << std::endl;
        std::cout << "For number of steps: " << n_steps / dt_mult << std::endl;

        // IMPLICIT MIDPOINT
        driver::implicit_midpoint( x, x0, dw, llg_sde, n_dim, w_dim,
                                   n_steps/dt_mult, t0, dt*dt_mult, eps, max_iter );
        sprintf( fname, "output/task2/implicit%d", i );
        io::write_array( fname, x, n_dim*(n_steps/dt_mult + 1) );
        // // HEUN SCHEME
        // driver::heun( x, x0, dw, test_drift, test_diffusion, n_steps/dt_mult,
        //               n_dim, w_dim, dt*dt_mult );
        // sprintf( fname, "output/task2/heun%d", i );
        // io::write_array( fname, x, n_steps/dt_mult + 1);
    }

    // ANALYTIC SOLUTION
    for( unsigned int n=0; n<n_steps+1; n++ )
        solution( x + n_dim*n, dt*n );
    io::write_array( "output/task2/true", x, n_steps + 1 );

    delete[] x; delete[] dw;

}
