/** @file integrators.hpp
 * Header includes various numerical methods for estimating solutions
 * to stochastic and deterministic ODEs. Methods included in this are:
 *   -ODEs: RK4
 *   -SDEs: Euler, Heun
 */
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
        const std::function<void(double*,double*,const double*,const double)> sde,
        const size_t n_steps, const size_t n_dims, const size_t n_wiener,
        const double step_size );

    void heun(
        double *states, const double* initial_state,
        const double *wiener_process,
        const std::function<void(double*,double*,const double*,const double)> sde,
        const size_t n_steps, const size_t n_dims, const size_t n_wiener,
        const double step_size );

    void implicit_midpoint(
        double *x,
        const double *x0,
        const double *dw,
        const std::function<void(double*,double*,double*,double*,const double*,const double,const double)> sde,
        const size_t n_dim,
        const size_t w_dim,
        const size_t n_steps,
        const double t0,
        const double dt,
        const double eps,
        const size_t max_iter );
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
        const double t, const double h );

    // FOR SDEs
    // The following integrators work as follows
    // Returns the next states after step_size given the current
    // states, drift, and diffusion.
    // Heun
    void heun(
        double *next_state, double *drift_arr, double *trial_drift_arr,
        double *diffusion_matrix, double *trial_diffusion_matrix,
        const double *current_state, const double *wiener_steps,
        const std::function<void(double*,double*,const double*,const double)> sde,
        const size_t n_dims, const size_t wiener_dims, const double t,
        const double step_size );

    // Euler-Maruyama
    void eulerm(
        double *states, double *diffusion_matrix,
        const double* initial_state, const double *wiener_process,
        const std::function<void(double*,double*,const double*,const double)> sde,
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

    /// RK45 Cash-Karp adaptive step deterministic ODE solver
    /**
     * Solves ODEs :)
     * returns the next state and the time step to be used in the next instance
     * @TODO what about the actual time step that it used? That's also important!
     */
    void rk45(
        double *next_state, double *temp_state, double *k1, double *k2,
        double *k3, double *k4, double *k5, double *k6, double *h_ptr,
        double *t_ptr, const double *current_state,
        const std::function<void(double*,const double*,const double)> ode,
        const size_t n_dims, const double eps );


    /// Cash-Karp parameter table for RK45
    namespace ck_butcher_table {
        static constexpr double c11 = 0.2;
        static constexpr double c21 = 3.0/40.0;
        static constexpr double c22 = 9.0/40.0;
        static constexpr double c31 = 3.0/10.0;
        static constexpr double c32 = -9.0/10.0;
        static constexpr double c33 = 6.0/5.0;
        static constexpr double c41 = -11.0/54.0;
        static constexpr double c42 = 2.5;
        static constexpr double c43 = -70.0/27.0;
        static constexpr double c44 = 35.0/27.0;
        static constexpr double c51 = 1631.0/55296.0;
        static constexpr double c52 = 175.0/512.0;
        static constexpr double c53 = 575.0/13824.0;
        static constexpr double c54 = 44275.0/110592.0;
        static constexpr double c55 = 253.0/4096.0;

        static constexpr double hc1 = 0.2;
        static constexpr double hc2 = 0.3;
        static constexpr double hc3 = 0.6;
        static constexpr double hc4 = 1.0;
        static constexpr double hc5 = 7.0/8.0;

        static constexpr double x11 = 37.0/378.0;     // 5th order params
        static constexpr double x13 = 250.0/621.0;
        static constexpr double x14 = 125.0/594.0;
        static constexpr double x16 = 512.0/1771.0;
        static constexpr double x21 = 2825.0/27648.0; // 4th order params
        static constexpr double x23 = 18575.0/48384.0;
        static constexpr double x24 = 13525.0/55296.0;
        static constexpr double x25 = 277.0/14336.0;
        static constexpr double x26 = 0.25;
    }

}
#endif
