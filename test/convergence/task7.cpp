///////////////////////////////
// CONVERGENCE TESTS	     //
// 			     //
// TASK 7		     //
// SINGLE TRAJECTORY (STIFF) //
// Effect of QNR tolerance   //
///////////////////////////////

#include "../../include/integrators.hpp"
#include "../../include/rng.hpp"
#include "../../include/io.hpp"
#include <functional>
#include <iostream>
#include <stdio.h>
#include <cmath>

void task7() {

  //                                  aout    bout    j_a     j_b       x_in           t_a           t_b
  using sde_jac = std::function<void(double*,double*,double*,double*,const double*,const double,const double)>;
  //                              drift   diff       state           t
  using sde = std::function<void(double*,double*,const double*,const double)>;

  ////////////////////////////////////////////////////
  // STOCHASTIC DIFFERENTIAL EQUATION               //
  // dX(t) = aX(t) dt + bX(t) dW(t)  [stratonovich] //
  ////////////////////////////////////////////////////

  double a=-20.0, b=5.0; // stiff
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
  size_t n_steps=6400;
  double dt = 1e-2;
  size_t n_dt=5;
  int dt_multipliers[5] = {1, 2, 4, 8, 16};
  size_t dt_mult;
  double dts[5];
  for( unsigned int i=0;i<n_dt; i++ )
    dts[i] = dt * dt_multipliers[i];
  io::write_array( "output/task7/dt", dts, n_dt );

  // RANDOM NUMBER GENERATOR
  size_t n_runs = 5000;
  long *seeds = new long[n_runs];
  for( size_t i=0; i<n_runs; i++ )
    seeds[i] = i*13*(i%5); // allocate some seeds
  RngMtNorm rng( 1001, std::sqrt( dt ) );
  double *dw;

  // IMPLICIT SCHEME PARAMS
  size_t n_eps = 3;
  double epss[3] = {1e-3, 1e-4, 1e-5};
  double eps;
  double *errs[3];
  double max_iter = 100;
  size_t n_dim=1;
  size_t w_dim=1;

  // ALLOCATE MEM FOR RESULTS
  for( size_t i=0; i<n_eps; i++ )
    errs[i] = new double[n_runs];
  double *x = new 
}
