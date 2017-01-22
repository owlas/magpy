// integrators.hpp
// Header includes various numerical methods for estimating solutions
// to stochastic and deterministic ODEs. Methods included in this are:
//   -ODEs: RK4
//   -SDEs: Euler, Heun
//
// Oliver W. Laslett 2016
// O.Laslett@soton.ac.uk
#ifndef INTEGRATORS_H
#define INTEGRATORS_H

#ifdef USEMKL
#include <mkl_lapacke.h>
#else
#include <lapacke.h>
#endif
#include <functional>

// The drivers can be used to solve ODEs and SDEs over a grid. The
// drivers take an array (states) of length NxM where N is the number
// of states and M is the grid. Grids are regularly spaced with
// step_size
namespace driver
{
    // ODEs
    void rk4(
        double *states, const double *initial_state,
        const std::function<void(double*,const double*,const double)> derivs,
        const size_t n_steps, const size_t n_dims,
        const double step_size );

    // SDEs
    void eulerm(
        double *states, const double* initial_state,
        const double *wiener_process,
        const std::function<void(double*,const double*,const double)> drift,
        const std::function<void(double*,const double*,const double)> diffusion,
        const size_t n_steps, const size_t n_dims, const size_t n_wiener,
        const double step_size );

    void heun(
        double *states, const double* initial_state,
        const double *wiener_process,
        const std::function<void(double*,const double*,const double)> drift,
        const std::function<void(double*,const double*,const double)> diffusion,
        const size_t n_steps, const size_t n_dims, const size_t n_wiener,
        const double step_size );
}

// The integrators contain the core numerical methods to compute the
// next step based on the current step. Use drivers for an easier to
// use interface.
namespace integrator
{
    // For ODEs
    // The following integrators work as follows
    // Returns the next states after step_size given the current
    // states and their derivatives.
    // Runge-Kutta 4
    void rk4(
        double *next_state, double *k1, double *k2, double *k3,
        double *k4, const double *current_state,
        const std::function<void(double*,const double*,const double)> derivs,
        const size_t n_dims,
        const double t, const double step_size );

    // FOR SDEs
    // The following integrators work as follows
    // Returns the next states after step_size given the current
    // states, drift, and diffusion.
    // Heun
    void heun(
        double *next_state, double *drift_arr, double *trial_drift_arr,
        double *diffusion_matrix, double *trial_diffusion_matrix,
        const double *current_state, const double *wiener_steps,
        const std::function<void(double*,const double*,const double)> drift,
        const std::function<void(double*,const double*,const double)> diffusion,
        const size_t n_dims, const size_t wiener_dims, const double t,
        const double step_size );

    // Euler-Maruyama
    void eulerm(
        double *states, double *diffusion_matrix,
        const double* initial_state, const double *wiener_process,
        const std::function<void(double*,const double*,const double)> drift,
        const std::function<void(double*,const double*,const double)> diffusion,
        const size_t n_dims, const size_t n_wiener,
        const double t, const double step_size );

    // Milstein
    template <class CSTATES, class CDIFF>
    void milstein( CSTATES &next_state, const CSTATES &current_state,
                   const CSTATES &drift, const CDIFF &diffusion,
                   const CSTATES &wiener_increments, const double step_size );

    // Fully implicit midpoint method
    int implicit_midpoint(
        double *x, double *dwm, double *a_work, double *b_work,
        double *adash_work, double *bdash_work, double *x_guess,
        double *x_opt_tmp, double *x_opt_jac, lapack_int *x_opt_ipiv,
        const double *x0, const double *dw,
        const std::function<void(double*,double*,double*,double*,const double*,const double,const double)> sde,
        const size_t n_dim, const size_t w_dim, const double t, const double dt,
        const double eps, const size_t max_iter );

}
#endif
