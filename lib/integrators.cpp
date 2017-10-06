/** @file integrators.cpp
 * Integrator implementation
 */

/**
 * @namespace integrator
 * @brief Numerical methods for differential equations
 * @details Numerical methods for simulating the time evolution
 * of deterministic and stochastic ordinary nonlinear differential
 * equations. All integrators compute a single step of the solution.
 * @author Oliver Laslett
 * @date 2017
 */

/** @namespace driver
 * @brief High level interface to differential equation integrators
 * @details Drivers wrap the single step integrators and provide an
 * interface for simulating differential equations for multiple steps
 * from the initial condition. Drivers also handle memory management
 * of the work arrays needed by the integrators.
 * @author Oliver Laslett
 * @date 2017
 */
#include "../include/integrators.hpp"
#include "../include/optimisation.hpp"
#include <cmath>
#include <iostream>

using sde_function = std::function<void(double*,const double*,const double)>;
using sde_func = std::function<void(double*,double*,const double*,const double)>;
using sde_jac = std::function<void(double*,double*,double*,double*,const double*,const double,const double)>;
namespace ck = integrator::ck_butcher_table;

/// Runge-Kutta 4 integration step
/**
 * Takes one step of the Runge-Kutta 4 method; an explicit solver for
 * deterministic differential equations.
 * @param[out] next_state the state of the system after time step
 * (length \p n_dims)
 * @param[out] k1 memory needed for work (length \p n_dims)
 * @param[out] k2 memory needed for work (length \p n_dims)
 * @param[out] k3 memory needed for work (length \p n_dims)
 * @param[out] k4 memory needed for work (length \p n_dims)
 * @param[in] current_state the initial state of the system (length \p n_dims)
 * @param[in] derivs the ordinary differential equation. A function
 * that takes the current state (length \p n_dims) and time and returns
 * derivatives (length \p n_dims).
 * @param[in] n_dims dimensionality of the system
 * @param[in] t current time
 * @param[in] h time step size
 */
void integrator::rk4(
    double *next_state,
    double *k1,
    double *k2,
    double *k3,
    double *k4,
    const double *current_state,
    const sde_function derivs,
    const size_t n_dims,
    const double t,
    const double h )
{
    derivs( k1, current_state, t);
    for( unsigned int i=0; i<n_dims; i++ )
        next_state[i] = k1[i] * h/2.0
            + current_state[i];

    derivs( k2, next_state, t + h/2.0 );
    for( unsigned int i=0; i<n_dims; i++ )
        next_state[i] = k2[i] * h/2.0
            + current_state[i];

    derivs( k3, next_state, t + h/2.0 );
    for( unsigned int i=0; i<n_dims; i++ )
        next_state[i] = k3[i] * h
            + current_state[i];

    derivs( k4, next_state, t + h );
    for( unsigned int i=0; i<n_dims; i++ )
        next_state[i] = current_state[i]
            + ( k1[i] + 2*k2[i] + 2*k3[i] + k4[i] ) * h/6.0;
}

/// Runge-Kutta 4 driver
/**
 * Takes multiple steps of the Runge-Kutta 4 method; an explicit
 * solver for deterministic differential equations.
 * @param[out] states the state of the system at each time step
 (length \p n_steps*\p n_dims). Each state is a row stored in a row-major
 fashion.
 * @param[in] initial_state initial state of the system (length
 \p n_dims)
 * @param[in] derivs the ordinary differential equation. A function
 * that takes the current state (length \p n_dims) and time and returns
 * derivatives (length \p n_dims).
 * @param[in] n_steps number of integration steps to take
 * @param[in] n_dims dimensionality of the system
 * @param[in] step_size size of time step
 */
void driver::rk4(
    double *states,
    const double* initial_state,
    const sde_function derivs,
    const size_t n_steps,
    const size_t n_dims,
    const double step_size )
{
    for( unsigned int i=0; i<n_dims; i++ )
        states[i] = initial_state[i];

    double *k1 = new double[n_dims];
    double *k2 = new double[n_dims];
    double *k3 = new double[n_dims];
    double *k4 = new double[n_dims];

    for( unsigned int i=1; i<n_steps; i++ )
        integrator::rk4( states+i*n_dims, k1, k2, k3, k4,
                         states+(i-1)*n_dims, derivs, n_dims,
                         i*step_size, step_size );

    delete[] k1; delete[] k2; delete[] k3; delete[] k4;
}

/// Runge-Kutta 45 Cash-Karp integration step
/**
 * Takes one step of the Runge-Kutta 45 adaptive step method; an
 * explicit solver for deterministic differential equations. Uses the
 * Cash-Karp Butcher tableau.
 * @param[out] next_state the state of the system after time step
 * (length \p n_dims)
 * @param[out] temp_state memory needed for work (length \p n_dims)
 * @param[out] k1 memory needed for work (length \p n_dims)
 * @param[out] k2 memory needed for work (length \p n_dims)
 * @param[out] k3 memory needed for work (length \p n_dims)
 * @param[out] k4 memory needed for work (length \p n_dims)
 * @param[out] k5 memory needed for work (length \p n_dims)
 * @param[out] k6 memory needed for work (length \p n_dims)
 * @param[in,out] h_ptr pointer to initial step size to take. Returns
 * the time step size to take on the next step.
 * @param[in,out] t_ptr pointer to the initial time. Returns the time
 * after one adaptive time step.
 * @param[in] current_state the initial state of the system (length \p n_dims)
 * @param[in] derivs the ordinary differential equation. A function
 * that takes the current state (length \p n_dims) and time and returns
 * derivatives (length \p n_dims).
 * @param[in] n_dims dimensionality of the system
 * @param[in] tol tolerance of the solver. Step size will be reduced
 * in order to keep error within tolerance. The maximum allowable
 * error is \p tol + state*\p tol.
 */
void integrator::rk45(
    double *next_state,
    double *temp_state,
    double *k1,
    double *k2,
    double *k3,
    double *k4,
    double *k5,
    double *k6,
    double *h_ptr,
    double *t_ptr,
    const double *current_state,
    const sde_function derivs,
    const size_t n_dims,
    const double tol )
{
    bool step_success = false;
    double err;
    double h=*h_ptr;
    double t=*t_ptr;
    while( step_success == false )
    {
        derivs( k1, current_state, t);
        for( unsigned int i=0; i<n_dims; i++ )
            next_state[i] = k1[i] * h * ck::c11
                + current_state[i];

        derivs( k2, next_state, t + h*ck::hc1 );
        for( unsigned int i=0; i<n_dims; i++ )
            next_state[i] = current_state[i]
                + h*( ck::c21 * k1[i] + ck::c22 * k2[i] );

        derivs( k3, next_state, t + h*ck::hc2 );
        for( unsigned int i=0; i<n_dims; i++ )
            next_state[i] = current_state[i]
                + h*( ck::c31*k1[i] + ck::c32*k2[i]
                      + ck::c33*k3[i] );

        derivs( k4, next_state, t + h*ck::hc3 );
        for( unsigned int i=0; i<n_dims; i++ )
            next_state[i] = current_state[i]
                + h*( ck::c41*k1[i] + ck::c42*k2[i]
                      + ck::c43*k3[i] + ck::c44*k4[i] );

        derivs( k5, next_state, t + h*ck::hc4 );
        for( unsigned int i=0; i<n_dims; i++ )
            next_state[i] = current_state[i]
                + h*( ck::c51*k1[i] + ck::c52*k2[i]
                      + ck::c53*k3[i] + ck::c54*k4[i]
                      + ck::c55*k5[i] );

        derivs( k6, next_state, t + h*ck::hc5 );

        // Compute order 5 estimate
        for( unsigned int i=0; i<n_dims; i++ )
            temp_state[i] = current_state[i]
                + h*( ck::x11 * k1[i]
                      + ck::x13 * k3[i]
                      + ck::x14 * k4[i]
                      + ck::x16 * k6[i] );

        // Compute order 4 estimate
        for( unsigned int i=0; i<n_dims; i++)
            next_state[i] = current_state[i]
                + h*( ck::x21 * k1[i]
                      + ck::x23 * k3[i]
                      + ck::x24 * k4[i]
                      + ck::x25 * k5[i]
                      + ck::x26 * k6[i] );

        // Compute the error and scale according to (tol+tol*|state|)
        err=0;
        double mag=0;
        for( unsigned int i=0; i<n_dims; i++ )
            mag += current_state[i] * current_state[i];
        mag = std::pow( mag, 0.5 );
        for( unsigned int i=0; i<n_dims; i++ )
            err += std::pow( std::abs( temp_state[i] - next_state[i] ), 2 );
        err = pow( err, 0.5 );
        err /= ( n_dims*tol*( 1 + mag ) );

        // If relative error is below 1 then step was successful
        // otherwise reduce the step size (max 10x reduction)
        if( err < 1.0 )
            step_success = true;
        else
        {
            double hfactor = 0.84*pow( err, -0.2 );
            hfactor = std::abs( hfactor ) < 0.1 ? 0.1 : hfactor;
            h *= hfactor;
        }
    }
    // Set the new time
    *t_ptr = t+h;

    // Set the next step size
    double hfactor = err==0.0 ? 5.0 : 0.84*std::pow( err, -0.2 );
    hfactor = hfactor > 5 ? 5.0 : hfactor;
    *h_ptr = hfactor*h;
}

/// Euler-Maruyama integration step
/**
 * Takes one step of the Euler-Maruyama scheme; an explicit solver for
 * Ito stochastic differential equations.
 * @param[out] next_state the state of the system after time step
 * (length \p n_dims)
 * @param[out] diffusion_matrix the diffusion matrix computed at the
 * initial time \p t (length \p n_dims*\p wiener_dims). Row-major.
 * @param[in] current_state the initial state of the system (length
 * \p n_dims)
 * @param[in] wiener_steps incremental step in the Wiener processes
 * over the time step \p step_size (length \p wiener_dims)
 * @param[in] drift the stochastic differential equation drift
 * component. A function that takes the current state and time and
 * returns the deterministic drift component of length \p n_dims
 * @param[in] diffusion the stochastic differential equation diffusion
 * component. A function that takes the current state and time and
 * returns the stochastic diffusion component of length
 * \p n_dims*\p wiener_dims (row-major)
 * @param[in] n_dims dimension of the system state
 * @param[in] wiener_dims dimension of the Wiener process
 * @param[in] t current time
 * @param[in] step_size size of time step to take
 */
void integrator::eulerm(
    double *next_state,
    double *diffusion_matrix,
    const double *current_state,
    const double *wiener_steps,
    const sde_function drift,
    const sde_function diffusion,
    const size_t n_dims,
    const size_t wiener_dims,
    const double t,
    const double step_size )
{
    drift( next_state, current_state, t );
    diffusion( diffusion_matrix, current_state, t );

    for( unsigned int i=0; i<n_dims; i++ ){
        next_state[i] = current_state[i] * step_size*next_state[i];
        for( unsigned int j=0; j<wiener_dims; j++ )
            next_state[i] = diffusion_matrix[j+i*n_dims]*wiener_steps[j];
    }
}
/// Euler-Maruyama driver
/**
 * Takes multiple steps of the Euler-Maruyama scheme; an explicit
 * solver for Ito stochastic differential equations. The first
 * solution step is always the initial condition followed by \p
 * n_steps -1 steps of the integrator.xo
xo * @param[out] states the state of the system at each time step
 (length \p n_steps*\p n_dims). Each state is a row stored in a row-major
 fashion.
 * @param[in] initial_state initial state of the system (length
 * \p n_dims)
 * @param[in] wiener_process the Wiener process increments for each
 * time step (length \p n_steps*\p n_wiener). Row-major where the ith row is
 * the increments in the \p n_wiener-dimensional Wiener process at the
 * ith step.
 * @param[in] drift the stochastic differential equation drift
 * component. A function that takes the current state and time and
 * returns the deterministic drift component of length \p n_dims
 * @param[in] diffusion the stochastic differential equation diffusion
 * component. A function that takes the current state and time and
 * returns the stochastic diffusion component of length
 * \p n_dims*\p n_wiener (row-major)
 * @param[in] n_steps  number of integration steps to take
 * @param[in] n_dims dimension of the system state
 * @param[in] n_wiener dimension of the Wiener process
 * @param[in] step_size size of the time step to take
 */
void driver::eulerm(
    double *states,
    const double* initial_state,
    const double *wiener_process,
    const sde_function drift,
    const sde_function diffusion,
    const size_t n_steps,
    const size_t n_dims,
    const size_t n_wiener,
    const double step_size )
{
    for( unsigned int i=0; i<n_dims; i++ )
        states[i] = initial_state[i];

    double *diffusion_mat = new double[n_dims*n_wiener];
    for( unsigned int i=1; i<n_steps; i++ )
        integrator::eulerm( states+i*n_dims, diffusion_mat, states+(i-1)*n_dims,
                            wiener_process+(i-1)*n_wiener, drift, diffusion,
                            n_dims, n_wiener, i*step_size, step_size );
    delete[] diffusion_mat;
}

/// Heun integration step
/**
 * Takes one step of the Heun scheme; an explicit solver for
 * Stratonovich stochastic differential equations.
 * @param[out] next_state the state of the system after time step
 * (length \p n_dims)
 * @param[out] drift_arr memory needed for work (length \p n_dims)
 * @param[out] trial_drift_arr memory needed for work (length \p n_dims)
 * @param[out] diffusion_matrix memory needed for work (length \p n_dims
 * * \p wiener_dims)
 * @param[out] trial_diffusion_matrix memory needed for work (length
 * \p n_dims * \p wiener_dims)
 * @param[in] current_state the initial state of the system (length
 * \p n_dims)
 * @param[in] wiener_steps incremental step in the Wiener processes
 * over the time step \p step_size (length \p wiener_dims)
 * @param[in] sde the stochastic differential equation drift and
 * diffusion. Function that takes the current state and time and
 * returns the drift component of length \p n_dims and the diffusion
 * component of length \p n_dims * \p wiener_dims
 * @param[in] n_dims dimension of the system state
 * @param[in] wiener_dims dimension of the Wiener process
 * @param[in] t initial time before taking the step
 * @param[in] step_size size of the time step to take
 */
void integrator::heun(
    double *next_state,
    double *drift_arr,
    double *trial_drift_arr,
    double *diffusion_matrix,
    double *trial_diffusion_matrix,
    const double *current_state,
    const double *wiener_steps,
    const sde_func sde,
    const size_t n_dims,
    const size_t wiener_dims,
    const double t,
    const double step_size )
{
    sde( drift_arr, diffusion_matrix, current_state, t );

    for( unsigned int i=0; i<n_dims; i++ ){
        next_state[i] = current_state[i] + step_size*drift_arr[i];
        for( unsigned int j=0; j<wiener_dims; j++ )
            next_state[i] += diffusion_matrix[j+i*wiener_dims]*wiener_steps[j] * std::sqrt( step_size );
    }

    sde( trial_drift_arr, trial_diffusion_matrix, next_state, t +step_size );

    for( unsigned int i=0; i<n_dims; i++ )
    {
        next_state[i] = current_state[i]
            + 0.5*step_size*( trial_drift_arr[i] + drift_arr[i] );
        for( unsigned int j=0; j<wiener_dims; j++ )
            next_state[i] += 0.5*wiener_steps[j] * std::sqrt( step_size )
                *( trial_diffusion_matrix[j+i*wiener_dims]
                   + diffusion_matrix[j+i*wiener_dims] );
    }
}

/// Heun driver
/**
 * Takes multiple steps of the Heun scheme; an explicit
 * predictor-corrector solver for Stratonvich stochastic differential
 * equations. The first solution step is always the initial condition
 * followed by \p n_steps-1 steps of the integrator.
 * @param[out] states the state of the system at each time step
 (length \p n_steps*\p n_dims). Each state is a row stored in a row-major
 fashion.
 * @param[in] wiener_process the Wiener process increments for each
 * time step (length \p n_steps*\p n_wiener). Row-major where the ith row is
 * the increments in the \p n_wiener-dimensional Wiener process at the
 * ith step.
 * @param[in] sde the stochastic differential equation drift and
 * diffusion. Function that takes the current state and time and
 * returns the drift component of length \p n_dims and the diffusion
 * component of length \p n_dims * \p n_wiener
 * @param[in] n_steps number of integration steps to take
 * @param[in] n_dims dimension of the system state
 * @param[in] n_wiener dimension of the Wiener process
 * @param[in] step_size size of the time step to take
 */
void driver::heun(
    double *states,
    const double* initial_state,
    const double *wiener_process,
    const sde_func sde,
    const size_t n_steps,
    const size_t n_dims,
    const size_t n_wiener,
    const double step_size )
{
    double *drift_arr = new double[n_dims];
    double *trial_drift_arr = new double[n_dims];
    double *diffusion_mat = new double[n_dims*n_wiener];
    double *trial_diffusion_mat = new double[n_dims*n_wiener];

    // First step
    for( unsigned int i=0; i<n_dims; i++ )
        states[i] = initial_state[i];

    // More steps
    for( unsigned int i=0; i<n_steps; i++ )
        integrator::heun(
            states+(i+1)*n_dims, drift_arr, trial_drift_arr, diffusion_mat,
            trial_diffusion_mat, states+i*n_dims, wiener_process+i*n_wiener,
            sde, n_dims, n_wiener, i*step_size, step_size );

    delete[] drift_arr; delete[] trial_drift_arr;
    delete[] diffusion_mat; delete[] trial_diffusion_mat;
}

/// Implicit midpoint driver
/**
 * Takes multiple steps of the implicit midpoint scheme; an implicit
 * solver for Stratonovich stochastic differential equations. The
 * first step is always the initial condition followed by \p n_steps
 * -1 steps of the integrator.
 * @param[out] x the state of the system at each time step
 (length \p n_steps*\p n_dim). Each state is a row stored in a row-major
 fashion.
 * @param[in] dw the Wiener process increments for each
 * time step (length \p n_steps*\p w_dim). Row-major where the ith row is
 * the increments in the \p w_dim-dimensional Wiener process at the
 * ith step.xo
 * @param[in] sde the stochastic differential equation drift and
 * diffusion. Function that takes the current state and time and
 * returns the drift component of length \p n_dims and the diffusion
 * component of length \p n_dim * \p w_dim
 * @param[in] t0 initial time of the system
 * @param[in] dt size of time step to take
 * @param[in] eps the tolerance of the internal quasi-Newton method
 * solver. Sensible values of the the tolerance are problem
 * dependent.
 * @param[in] max_iter maximum number of iterations for internal
 * quasi-Newton method. Limits the number of iterations to reach
 * required tolerance. Error will be raised if maximum number is
 * reached.
 */
void driver::implicit_midpoint(
    double *x,
    const double *x0,
    const double *dw,
    const sde_jac sde,
    const size_t n_dim,
    const size_t w_dim,
    const size_t n_steps,
    const double t0,
    const double dt,
    const double eps,
    const size_t max_iter )
{
    double *dwm = new double[w_dim];
    double *a_work = new double[n_dim];
    double *b_work = new double[n_dim*w_dim];
    double *adash_work = new double[n_dim*n_dim];
    double *bdash_work = new double[n_dim*w_dim*n_dim];
    double *x_guess = new double[n_dim];
    double *x_opt_tmp = new double [n_dim];
    double *x_opt_jac = new double[n_dim*n_dim];
    lapack_int *x_opt_ipiv = new lapack_int[n_dim];

    // First step
    double t = t0;
    int err_code;
    for( unsigned int i=0; i<n_dim; i++ )
        x[i] = x0[i];

    // Take N steps
    for( unsigned int n=0; n<n_steps; n++ )
    {
        t+=dt;
        err_code = integrator::implicit_midpoint(
            x+((n+1)*n_dim), dwm, a_work, b_work, adash_work, bdash_work,
            x_guess, x_opt_tmp, x_opt_jac, x_opt_ipiv, x+(n*n_dim),
            dw+(n*w_dim), sde, n_dim, w_dim, t, dt, eps, max_iter );
        //std::cout << x[n] << " " << dw[n] << " " << x[n+1] << std::endl;
        if( err_code != 0 )
            std::cout << "At time step " << n << " implicit solver errcode:" << err_code << std::endl;
    }

    delete[] dwm;
    delete[] a_work;
    delete[] b_work;
    delete[] adash_work;
    delete[] bdash_work;
    delete[] x_guess;
    delete[] x_opt_tmp;
    delete[] x_opt_jac;
    delete[] x_opt_ipiv;
}

/// Implicit midpoint integration step
/**
 * Takes on step of the implicit midpoint scheme; an implicit solver
 * for Stratonovich stochastic differential equations.
 * @param[out] x the state of the system after stepping (length \p
 * n_dim)
 * @param[out] dwm memory needed for work (length \p wdim)
 * @param[out] a_work memory needed for work (length \p n_dim)
 * @param[out] b_work memory needed for work (length \p n_dim * \p w_dim)
 * @param[out] adash_work memory needed for work (length \p n_dim * \p
 * n_dim)
 * @param[out] b_dash_work memory needed for work (length \p n_dim *
 * \p w_dim * \p n_dim)
 * @param[out] x_guess memory needed for work (length \p n_dim)
 * @param[out] x_opt_tmp memory needed for work (length \p n_dim)
 * @param[out] x_opt_jac memory needed for work (length \p n_dim * \p
 * n_dim)
 * @param[out] x_opt_ipiv memory needed for work (length \p n_dim)
 * @param[in] x0 initial state of the system (length \p n_dim)
 * @param[in] dw incremental step in the Wiener processes
 * over the time step \p dt (length \p w_dim)
 * @param[in] sde the stochastic differential equation drift and
 * diffusion. Function that takes the current state and time and
 * returns the drift component of length \p n_dim and the diffusion
 * component of length \p n_dim * \p w_dim
 * @param[in] n_dim dimension of the system state
 * @param[in] w_dim dimension of the Wiener process
 * @param[in] t initial time before taking a step
 * @param[in] dt size of the time step to use
 * @param[in] eps the tolerance of the internal quasi-Newton method
 * solver. Sensible values of the the tolerance are problem
 * dependent.
 * @param[in] max_iter maximum number of iterations for internal
 * quasi-Newton method. Limits the number of iterations to reach
 * required tolerance. Error will be raised if maximum number is
 * reached.
 */
int integrator::implicit_midpoint(
    double *x,
    double *dwm,
    double *a_work,
    double *b_work,
    double *adash_work,
    double *bdash_work,
    double *x_guess,
    double *x_opt_tmp,
    double *x_opt_jac,
    lapack_int *x_opt_ipiv,
    const double *x0,
    const double *dw,
    const sde_jac sde,
    const size_t n_dim,
    const size_t w_dim,
    const double t,
    const double dt,
    const double eps,
    const size_t max_iter )
{
    // Implicit method depends on a modified random variable
    // See http://epubs.siam.org/doi/pdf/10.1137/S0036142901395588 for details
    double Ah = std::sqrt( 2*std::abs( std::log( dt ) ) );
    for( unsigned int i=0; i<w_dim; i++ )
        dwm[i] = std::max( -Ah, std::min( Ah, dw[i] ) ) * std::sqrt( dt );

    // The initial guess will be an Euler step
    sde( a_work, b_work, adash_work, bdash_work, x0, t, t );
    for( unsigned int i=0; i<n_dim; i++ )
    {
        x_guess[i] = a_work[i]*dt;
        for( unsigned int j=0; j<w_dim; j++ )
            x_guess[i] += b_work[i*w_dim+j]*dwm[j];
    }
    // Convert this into (x1+x2)/2
    for( unsigned int i=0; i<n_dim; i++ )
        x_guess[i] = ( x_guess[i]+x0[i] ) / 2;

    // construct F(x) and J(x) - Jacobian of F
    auto FJ = [sde, a_work,b_work,t,dt,n_dim,x0,w_dim,dwm,adash_work,bdash_work]
        (double *fout, double *jacout, const double *in)->void
        {
            // F(X)
            sde(a_work,b_work,adash_work,bdash_work,in,t+dt/2,t);
            for( unsigned int i=0; i<n_dim; i++ )
            {
                fout[i] = in[i] - 0.5*a_work[i]*dt - x0[i];
                for( unsigned int j=0; j<w_dim; j++ )
                    fout[i] -= 0.5*b_work[i*w_dim+j]*dwm[j];
            }

            // J(X)
            for( unsigned int i=0; i<n_dim; i++ )
                for( unsigned int j=0; j<n_dim; j++ )
                {
                    jacout[i*n_dim+j] = (i==j) - 0.5*adash_work[i*n_dim+j];
                    for( unsigned int k=0; k<w_dim; k++ )
                        jacout[i*n_dim+j] -= 0.5*bdash_work[i*w_dim*n_dim+k*n_dim+j]*dwm[k];
                }
        };

    // Solve the nonlinear equations to get the next step
    int err_code;
    auto flag = optimisation::newton_raphson_noinv(
        x, x_opt_tmp, x_opt_jac, x_opt_ipiv, &err_code, FJ,
        x_guess, n_dim, eps, max_iter );

    // Recover the next state from the current
    for( unsigned int i=0; i<n_dim; i++ )
        x[i] = 2*x[i] - x0[i];

    // Return error code
    return flag + err_code;
}
