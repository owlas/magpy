#include "../googletest/include/gtest/gtest.h"
#include "../include/llg.hpp"
#include "../include/integrators.hpp"
#include "../include/io.hpp"
#include "../include/simulation.hpp"
#include "../include/trap.hpp"
#include "../include/optimisation.hpp"
#include "../include/rng.hpp"
#include "../include/stochastic_processes.hpp"
#include "../include/field.hpp"
#include "../include/dom.hpp"
#include "../include/constants.hpp"
#include "../include/distances.hpp"
#include <cmath>
#include <random>
#ifdef USEMKL
#include <mkl_lapacke.h>
#else
#include <lapacke.h>
#endif
#include <stdexcept>

TEST(llg, drift)
{
    double deriv[3];
    const double state[3] = {2, 3, 4};
    const double field[3] = {1, 2, 3};
    const double time=0, damping=3;
    llg::drift( deriv, state, time, damping, field );

    EXPECT_EQ( -34, deriv[0] );
    EXPECT_EQ(  -4, deriv[1] );
    EXPECT_EQ(  20, deriv[2] );
}

TEST(llg, diffusion)
{
    double deriv[3*3];
    const double state[3] = { 2, 3, 4 };
    const double time=0,  sr=2, alpha=3;
    llg::diffusion( deriv, state, time, sr, alpha );

    EXPECT_EQ( 150, deriv[0] );
    EXPECT_EQ( -28, deriv[1] );
    EXPECT_EQ ( -54, deriv[2] );
    EXPECT_EQ ( -44, deriv[3] );
    EXPECT_EQ ( 120, deriv[4] );
    EXPECT_EQ ( -68, deriv[5] );
    EXPECT_EQ ( -42, deriv[6] );
    EXPECT_EQ ( -76, deriv[7] );
    EXPECT_EQ ( 78, deriv[8] );
}

TEST(llg, multi_diffusion)
{
    double deriv[6*6];
    const double state[6] = { 2, 3, 4, 2, 3, 4 };
    const double sr[2] = { 2, 2 };
    const double alpha[2]  = { 3, 3 };
    llg::multi_diffusion( deriv, state, sr, alpha, 2 );

    for( unsigned int i=0; i<6; i++ )
        for( unsigned int j=0; j<6; j++ )
            if( (i/3) != (j/3) )
	      {
                EXPECT_EQ( 0, deriv[i*6+j] );
	      }

    EXPECT_EQ( 150, deriv[0*6 + 0] );
    EXPECT_EQ( -28, deriv[0*6 + 1] );
    EXPECT_EQ ( -54, deriv[0*6 + 2] );
    EXPECT_EQ ( -44, deriv[1*6 + 0] );
    EXPECT_EQ ( 120, deriv[1*6 + 1] );
    EXPECT_EQ ( -68, deriv[1*6 + 2] );
    EXPECT_EQ ( -42, deriv[2*6 + 0] );
    EXPECT_EQ ( -76, deriv[2*6 + 1] );
    EXPECT_EQ ( 78, deriv[2*6 + 2] );

    EXPECT_EQ( 150, deriv[3*6 + 3] );
    EXPECT_EQ( -28, deriv[3*6 + 4] );
    EXPECT_EQ ( -54, deriv[3*6 + 5] );
    EXPECT_EQ ( -44, deriv[4*6 + 3] );
    EXPECT_EQ ( 120, deriv[4*6 + 4] );
    EXPECT_EQ ( -68, deriv[4*6 + 5] );
    EXPECT_EQ ( -42, deriv[5*6 + 3] );
    EXPECT_EQ ( -76, deriv[5*6 + 4] );
    EXPECT_EQ ( 78, deriv[5*6 + 5] );
}

/*
  Equations evaluated symbolically with Mathematica then evaluated
  with test values.
*/
TEST(llg, drift_jacobian )
{
    double jac[3*3];
    const double state[3] = { 1.0, 1.2, 2.2 };
    const double heff[3] = { 6.7, 13.4, 10.05 };
    const double heff_jac[9] = {1.0, 2.0, 1.5, 2.0, 4.0, 3.0, 1.5, 3.0, 2.25 };
    const double time=0, alpha=0.1;
    llg::drift_jacobian( jac, state, time, alpha, heff, heff_jac );
    EXPECT_DOUBLE_EQ( - 1.161, jac[0] );
    EXPECT_DOUBLE_EQ( - 4.466, jac[1] );
    EXPECT_DOUBLE_EQ(  19.33 , jac[2] );
    EXPECT_DOUBLE_EQ(  11.878, jac[3] );
    EXPECT_DOUBLE_EQ( - 2.977, jac[4] );
    EXPECT_NEAR( - 2.082, jac[5], 1e-10 );
    EXPECT_DOUBLE_EQ( -14.046, jac[6] );
    EXPECT_DOUBLE_EQ(   3.8  , jac[7] );
    EXPECT_DOUBLE_EQ( - 4.051, jac[8] );
}

/*
  Equations evaluated symbolically with Mathematica then evaluated
  with test values.
*/
TEST( field, uniaxial )
{
    double h[3];
    const double state[3] = { 1.0, 1.2, 2.2 };
    const double aaxis[3] = { 1.0, 2.0, 1.5 };
    field::uniaxial_anisotropy( h, state, aaxis );
    EXPECT_DOUBLE_EQ( 6.7, h[0] );
    EXPECT_DOUBLE_EQ( 13.4, h[1] );
    EXPECT_DOUBLE_EQ( 10.05, h[2] );
}

/*
  Equations evaluated symbolically with Mathematica then evaluated
  with test values.
*/
TEST( field, uniaxial_jacobian )
{
    double jac[3*3];
    const double aaxis[3] = { 1.0, 2.0, 1.5 };
    field::uniaxial_anisotropy_jacobian( jac, aaxis );
    EXPECT_DOUBLE_EQ( 1.0, jac[0] );
    EXPECT_DOUBLE_EQ( 2.0, jac[1] );
    EXPECT_DOUBLE_EQ( 1.5, jac[2] );
    EXPECT_DOUBLE_EQ( 2.0, jac[3] );
    EXPECT_DOUBLE_EQ( 4.0, jac[4] );
    EXPECT_DOUBLE_EQ( 3.0, jac[5] );
    EXPECT_DOUBLE_EQ( 1.5, jac[6] );
    EXPECT_DOUBLE_EQ( 3.0, jac[7] );
    EXPECT_DOUBLE_EQ( 2.25, jac[8] );
}

TEST(heun, multiplicative )
{
    double next_state[2];
    double drift_arr[2];
    double trial_drift_arr[2];
    double diffusion_matrix[6];
    double trial_diffusion_matrix[6];
    double wiener_steps[3] = { 0.1, 0.01, 0.001 };
    const double current_state[2] = { 1, 2 };

    const std::function<void(double*,double*,const double*,const double)> sde =
        [](double *drift, double*diff, const double *in, const double t)
        {
            drift[0]=in[0]*in[1]*t; drift[1]=3*in[0];
            diff[0]=t; diff[1]=in[0]; diff[2]=in[1]; diff[3]=in[1]; diff[4]=4.0;
            diff[5]=in[0]*in[1];

        };

    const size_t n_dims=2, wiener_dims=3;
    const double t=0, step_size=0.1;;

    for( size_t i=0; i<wiener_dims; i++ )
        wiener_steps[i] = wiener_steps[i] / std::sqrt( step_size );

    integrator::heun(
        next_state, drift_arr, trial_drift_arr, diffusion_matrix, trial_diffusion_matrix,
        current_state, wiener_steps, sde, n_dims, wiener_dims,
        t, step_size );


    EXPECT_DOUBLE_EQ( 1.03019352, next_state[0] );
    EXPECT_DOUBLE_EQ( 2.571186252, next_state[1] );
}

TEST( heun_driver, ou )
{
    const size_t n_steps = 5000;
    const size_t n_dims = 1;
    const size_t n_wiener = 1;
    const double step_size = 1e-5;
    const double initial_state[1]={-3};

    // define the Ornstein-Uhlenbeck process
    const double ou_theta = 10;
    const double ou_mu = -1;
    const double ou_sigma = 0.8;
    const std::function<void(double*,double*,const double*,const double)> sde =
        [ou_theta,ou_mu,ou_sigma](double *drift, double *diff,
                         const double *in, const double )
        {
            drift[0]=ou_theta*(ou_mu - in[0]);
            diff[0] = ou_sigma;
        };


    // Create a wiener path
    double wiener_process[n_steps];
    RngMtNorm rng( 1001, 1.0 );
    for( unsigned int i=0; i<n_steps; i++ )
        wiener_process[i] = rng.get();

    // Generate the states
    double states[n_steps+1];
    driver::heun(
        states, initial_state, wiener_process, sde,
        n_steps, n_dims, n_wiener, step_size );

    // Compute the analytic solution and compare pathwise similarity
    double truesol = initial_state[0];
    for( unsigned int i=1; i<n_steps+1; i++ )
    {
        truesol = truesol*std::exp( -ou_theta*step_size )
            + ou_mu*( 1-std::exp( -ou_theta*step_size ) )
            + ou_sigma*std::sqrt( ( 1- std::exp( -2*ou_theta*step_size ) )/( 2*ou_theta ) )
            * wiener_process[i-1];

        ASSERT_NEAR( truesol, states[i], 1e-8 );
    }
}

TEST( trapezoidal_method, triangle )
{
    double x[4] = { 0.0, 0.1, 0.3, 0.7 };
    double y[4] = { 5.0, 10.0, 10.0, 20.0 };
    double area = trap::trapezoidal( x, y, 4 );
    ASSERT_DOUBLE_EQ( 8.75, area );
}


TEST( newton_raphson, 1d_function )
{
    auto f = [](const double x)->double
        { return -(x-3.2)*(x-3.2); };
    auto fdash = [](const double x)->double
        { return -2*(x-3.2); };

    double x0=0.0, eps=1e-6, x_root;
    const size_t max_iter=100;

    int res = optimisation::newton_raphson_1(
        &x_root, f, fdash, x0, eps, max_iter );
    ASSERT_EQ( optimisation::SUCCESS, res );
    ASSERT_LE( std::abs( x_root - 3.2 ), eps );
    }

TEST( newton_raphson, 1d_funtion_max_iter )
{
    auto f = [](const double x)->double
        { return -(x-3.2)*(x-3.2); };
    auto fdash = [](const double x)->double
        { return -2*(x-3.2); };

    double x0=0.0, eps=1e-6, x_root;
    const size_t max_iter=2;

    int res = optimisation::newton_raphson_1(
        &x_root, f, fdash, x0, eps, max_iter );

    ASSERT_EQ( optimisation::MAX_ITERATIONS_ERR, res );
}

TEST( newton_raphson_noinv, 2d_function_2_sols )
{
    double x_root[2], x_tmp[2], jac_out[4], x0[2], eps=1e-10;
    lapack_int dim=2, ipiv[2];
    size_t max_iter=100;
    int lapack_err;

    /*
      F = [ -(x-2)^2 - 2xy,
            -y(x+1) - 4y ]
      F(x,y)==0 has 2 solutions:
        x,y=-5,4.9    x,y=2,0
    */
    auto fj = []( double*fout,double*jacout,const double*in )->void
    {
        fout[0] = -( in[0]-2 )*( in[0]-2 ) - 2*in[0]*in[1];
        fout[1] = -in[1]*( in[0]+1 ) - 4*in[1];

        jacout[0] = -2*in[0] - 2*in[1] + 4;
        jacout[1] = -2*in[0];
        jacout[2] = -in[1];
        jacout[3] = -in[0]-5;
    };

    // Find the first solution
    x0[0] = x0[1] = -5.2;
    double tolerance = std::sqrt(2 * 5.2 * 5.2) * eps;
    auto flag = optimisation::newton_raphson_noinv(
        x_root, x_tmp, jac_out, ipiv, &lapack_err,
        fj, x0, dim, eps, max_iter );
    ASSERT_EQ( optimisation::SUCCESS, flag );
    EXPECT_LE( std::abs( x_root[0] + 5 ), tolerance );
    EXPECT_LE( std::abs( x_root[1] - 4.9 ), tolerance );

    // Find the second solution
    x0[0] = x0[1] = 1.0;
    tolerance = std::sqrt(2.0) * eps;
    flag = optimisation::newton_raphson_noinv(
        x_root, x_tmp, jac_out, ipiv, &lapack_err,
        fj, x0, dim, eps, max_iter );
    ASSERT_EQ( optimisation::SUCCESS, flag );
    EXPECT_LE( std::abs( x_root[0] - 2 ), tolerance );
    EXPECT_LE( std::abs( x_root[1] ), tolerance );
}

TEST( newton_raphson_noinv, 2d_function_singular )
{
    double x_root[2], x_tmp[2], jac_out[4], x0[2], eps=1e-8;
    lapack_int dim=2, ipiv[2];
    size_t max_iter=100;
    int lapack_err;

    /*
      This function has infinite solutions which will create
      a singular matrix in the LU factorisation in lapack routine
    */
    auto fj = []( double*fout,double*jacout,const double*in )->void
        {
            fout[0] = -( in[0]-2 )*( in[0]-2 ) - 2*in[0]*in[1];
            fout[1] = 2.0*fout[0];

            jacout[0] = -2*in[0] - 2*in[1] + 4;
            jacout[1] = -2*in[0];
            jacout[2] = 2.0*jacout[0];
            jacout[3] = 2.0*jacout[1];
        };

    x0[0] = x0[1] = -5.2;
    auto flag = optimisation::newton_raphson_noinv(
        x_root, x_tmp, jac_out, ipiv, &lapack_err,
        fj, x0, dim, eps, max_iter );

    ASSERT_EQ( optimisation::LAPACK_ERR, flag );
    ASSERT_GE( lapack_err, 0 );
}

TEST( rng, mt_norm )
{
    // Test the mt algorithm
    RngMtNorm rng( 999, 0.2 ); // seed, standard deviation
    EXPECT_DOUBLE_EQ(  0.27100401702487437, rng.get() );
    EXPECT_DOUBLE_EQ( -0.18950297511996847, rng.get() );
    EXPECT_DOUBLE_EQ( -0.27620532277321042, rng.get() );
    EXPECT_DOUBLE_EQ( -0.10214651921310165, rng.get() );
    EXPECT_DOUBLE_EQ( -0.19125423381292103, rng.get() );
}

TEST( rng, array )
{
    // Test a pre-allocated array
    double arr[5] = {0.01, 0.2, 1, 5, -0.1};
    RngArray rng( arr, 5 );
    EXPECT_DOUBLE_EQ( 0.01, rng.get() );
    EXPECT_DOUBLE_EQ(  0.2, rng.get() );
    EXPECT_DOUBLE_EQ(    1, rng.get() );
    EXPECT_DOUBLE_EQ(    5, rng.get() );
    EXPECT_DOUBLE_EQ( -0.1, rng.get() );

    // Check that error is thrown if too many calls
    try
    {
        rng.get();
        FAIL() << "Expected std::out_of_range";
    }
    catch( std::out_of_range const & err )
    {
        ASSERT_EQ( err.what(),
                   std::string("Exceeded allocated random number array size"));
    }
    catch( ... )
    {
        FAIL() << "Expected std::out_of_range";
    }
}

TEST( rng, array_stride_3 )
{
    // Test a pre-allocated array
    double arr[5] = {0.01, 0.2, 1, 5, -0.1};
    RngArray rng( arr, 5, 3 );
    EXPECT_DOUBLE_EQ( 0.01, rng.get() ); // 0th index
    EXPECT_DOUBLE_EQ(    5, rng.get() ); // 3rd index

    // Next call will be 6th index which exceeds array length
    try
    {
        rng.get();
        FAIL() << "Expected std::out_of_range";
    }
    catch( std::out_of_range const & err )
    {
        ASSERT_EQ( err.what(),
                   std::string("Exceeded allocated random number array size"));
    }
    catch( ... )
    {
        FAIL() << "Expected std::out_of_range";
    }
}

TEST( rng, rng_mt_downsample )
{
    RngMtDownsample rng( 1, 0.1, 2, 3 );
    RngMtNorm rng_ref( 1, 0.1 );

    double ref_1 = rng_ref.get();
    double ref_2 = rng_ref.get();
    ref_1 += rng_ref.get();
    ref_2 += rng_ref.get();
    ref_1 += rng_ref.get();
    ref_2 += rng_ref.get();
    double ref_3 = rng_ref.get();
    double ref_4 = rng_ref.get();
    ref_3 += rng_ref.get();
    ref_4 += rng_ref.get();
    ref_3 += rng_ref.get();
    ref_4 += rng_ref.get();

    EXPECT_DOUBLE_EQ( ref_1, rng.get() );
    EXPECT_DOUBLE_EQ( ref_2, rng.get() );
    EXPECT_DOUBLE_EQ( ref_3, rng.get() );
    EXPECT_DOUBLE_EQ( ref_4, rng.get() );
}

TEST( implicit_integrator_midpoint, 2d_stiff )
{
    // Stiff 2d system with 1d wiener
    double x[2], dwm[1], a_work[2], b_work[2], adash_work[4], bdash_work[4];
    double x_guess[2], x_opt_tmp[2], x_opt_jac[4], x0[2]={1.0, 2.0}, dw[1]={0.25};
    lapack_int x_opt_ipiv[2];
    const size_t n_dim=2;
    const size_t w_dim=1;
    const double t=0;
    const double dt=0.0001;
    const double eps=1e-6;
    const size_t max_iter=100;

    const double a=0.1, b=5.0;
    auto sde = [a,b]
        (double*aout,double*bout,double*adashout,double*bdashout,
         const double*in,const double, const double )
        {
            aout[0]=a*(in[1]-in[0])-0.5*b*b*in[0];
            aout[1]=a*(in[0]-in[1])-0.5*b*b*in[1];

            adashout[0]=-a-0.5*b*b;
            adashout[1]=a;
            adashout[2]=a;
            adashout[3]=-a-0.5*b*b;

            bout[0]=b*in[0];
            bout[1]=b*in[1];

            bdashout[0]=b;
            bdashout[1]=0;
            bdashout[2]=0;
            bdashout[3]=b;
        };

    int ans = integrator::implicit_midpoint(
        x, dwm, a_work, b_work, adash_work, bdash_work,
        x_guess, x_opt_tmp, x_opt_jac, x_opt_ipiv,
        x0, dw, sde, n_dim, w_dim,
        t, dt, eps, max_iter );

    // Assert the integrator was successful
    ASSERT_EQ( optimisation::SUCCESS, ans );

    // Solutions from Kloeden & Platen (1992) pp.397
    EXPECT_NEAR( 1.01132363, x[0], 0.0005 );
    EXPECT_NEAR( 2.02261693, x[1], 0.0005 );
}

TEST( implicit_driver, stiff_2d )
{
    size_t N = 100;
    double *x = new double[2*N];
    double x0[2] = {1.0, 2.0};
    double *dw = new double[N];
    double *Wt = new double[N];

    RngMtNorm rng( 1001, 1.0 );
    for( unsigned int i=0; i<N; i++ )
        dw[i] = rng.get();

    Wt[0] = 0;
    for( unsigned int i=1; i<N; i++ )
        Wt[i] = Wt[i-1] + dw[i-1];

    const double a=0.1, b=5.0;
    auto sde = [a,b]
        (double*aout,double*bout,double*adashout,double*bdashout,
         const double*in,const double, const double )
        {
            aout[0]=a*(in[1]-in[0])-0.5*b*b*in[0];
            aout[1]=a*(in[0]-in[1])-0.5*b*b*in[1];

            adashout[0]=-a-0.5*b*b;
            adashout[1]=a;
            adashout[2]=a;
            adashout[3]=-a-0.5*b*b;

            bout[0]=b*in[0];
            bout[1]=b*in[1];

            bdashout[0]=b;
            bdashout[1]=0;
            bdashout[2]=0;
            bdashout[3]=b;
        };

    size_t n_dim=2, w_dim=1, n_steps=N-1, max_iter=500;
    double t0=0.0, dt=0.0001, eps=1e-6;

    driver::implicit_midpoint(x, x0, dw, sde, n_dim, w_dim, n_steps, t0, dt, eps, max_iter);

    double rho_p, rho_m, ondiag, offdiag, x0_true, x1_true;
    for( unsigned int i=1; i<N; i++ )
    {
        rho_p = -0.5*b*b*i*dt + b*Wt[i] * std::sqrt(dt);
        rho_m = (-2*a - 0.5*b*b)*i*dt + b*Wt[i] * std::sqrt(dt);
        ondiag = std::exp(rho_p) + std::exp(rho_m);
        offdiag = std::exp(rho_p) - std::exp(rho_m);
        x0_true = 0.5*( ondiag*x0[0] + offdiag*x0[1] );
        x1_true = 0.5*( offdiag*x0[0] + ondiag*x0[1] );
        EXPECT_NEAR( x0_true, x[i*2+0], 0.002);
        EXPECT_NEAR( x1_true, x[i*2+1], 0.002);
    }
    delete[] x; delete[] dw; delete[] Wt;
}

TEST( rk4, time_dependent_step )
{
    auto ode = [](double*dx,const double*x,const double t)
    {
        dx[0] =  x[1] * t;
        dx[1] = -x[0] - 5;
    };

    double x2[2], k1[2], k2[2], k3[2], k4[4], x1[2]={1,2};
    size_t dims=2;
    double t=0.5, dt=0.01;
    integrator::rk4( x2, k1, k2, k3, k4, x1, ode, dims, t, dt );

    ASSERT_DOUBLE_EQ( 1.0, k1[0] );
    ASSERT_DOUBLE_EQ( -6.0, k1[1] );
    ASSERT_DOUBLE_EQ( 0.99485, k2[0] );
    ASSERT_DOUBLE_EQ( -6.005, k2[1] );
    ASSERT_DOUBLE_EQ( 0.994837375, k3[0] );
    ASSERT_DOUBLE_EQ( -6.00497425, k3[1] );
    ASSERT_DOUBLE_EQ( 0.989374631325, k4[0] );
    ASSERT_DOUBLE_EQ( -6.0099483737499995, k4[1] );
    ASSERT_DOUBLE_EQ( 1.0099479156355418, x2[0] );
    ASSERT_DOUBLE_EQ( 1.9399501718770833, x2[1] );
}

TEST( rk45, time_dependent_step )
{
    auto ode = [](double*dx,const double*x,const double t)
        {
            dx[0] =  x[1] * t;
            dx[1] = -x[0] - 5;
        };

    double x2[2], tmp[2], k1[2], k2[2], k3[2], k4[2], k5[2], k6[2], x1[2]={1,2};
    size_t dims=2;
    double t=0.5, dt=0.01;
    double eps = 1e-4;
    integrator::rk45( x2, tmp, k1, k2, k3, k4, k5, k6, &dt, &t, x1, ode, dims, eps );

    ASSERT_DOUBLE_EQ( 1.0, k1[0] );
    ASSERT_DOUBLE_EQ( -6.0, k1[1] );
    ASSERT_DOUBLE_EQ( 0.997976, k2[0] );
    ASSERT_DOUBLE_EQ( -6.002, k2[1] );
    ASSERT_DOUBLE_EQ( 0.9969437365, k3[0] );
    ASSERT_DOUBLE_EQ( -6.002995446, k3[1] );
    ASSERT_DOUBLE_EQ( 0.993774919651888, k4[0] );
    ASSERT_DOUBLE_EQ( -6.005981540838, k4[1] );
    ASSERT_DOUBLE_EQ( 0.9893745618215710, k5[0] );
    ASSERT_DOUBLE_EQ( -6.009947940975117, k5[1] );
    ASSERT_DOUBLE_EQ( 0.99077120433373344621, k6[0] );
    ASSERT_DOUBLE_EQ( -6.00871032589945830438, k6[1] );
    ASSERT_DOUBLE_EQ( 1.00994791563354557873, x2[0] );
    ASSERT_DOUBLE_EQ( 1.93995017187702467609, x2[1] );
}

TEST( master_equation, 3d_system )
{
    const double W[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    const double in[3] = {3, 2, 1};
    double out[3];
    const size_t dim = 3;
    stochastic::master_equation( out, W, in, dim );

    EXPECT_DOUBLE_EQ( 10.0, out[0] );
    EXPECT_DOUBLE_EQ( 28.0, out[1] );
    EXPECT_DOUBLE_EQ( 46.0, out[2] );
}

TEST( dom, uniaxial_transition_matrix )
{
    const double k=1, v=2e-21, h=0.5, T=10, ms=2.7, alpha=0.1;
    double W[4];


    dom::transition_matrix( W, k, v, T, h, ms, alpha );
    EXPECT_DOUBLE_EQ( -0.00021827108536028533, W[0] );
    EXPECT_DOUBLE_EQ( 278104386.1896516, W[1]);
    EXPECT_DOUBLE_EQ( 0.00021827108536028533, W[2] );
    EXPECT_DOUBLE_EQ( -278104386.1896516, W[3] );
}

TEST( dipolar_field, two_particles )
{
    // Two particles
    double field[6] = {0, 0, 0, 0, 0, 0};

    double k_av = 0.5;
    double ms = 1./std::sqrt( constants::MU0 );
    double v_red[2] = {15./16, 17./16};
    double mag[6] = {1, 0, 0, 0, 0, 1};
    double dists[12] = {0, 0, 0, -0.5, 0, std::sqrt(3.)/2., 0.5, 0, -std::sqrt(3.)/2., 0, 0, 0};
    double dist_cubes[4] = {0, 1, 1, 0};
    size_t N = 2;

    field::multi_add_dipolar( field, ms, k_av, v_red, mag, dists, dist_cubes, N );

    EXPECT_DOUBLE_EQ( -0.10983505338481014402, field[0] );
    EXPECT_DOUBLE_EQ( 0, field[1] );
    EXPECT_DOUBLE_EQ( 0.10568882939696175316, field[2] );
    EXPECT_DOUBLE_EQ( -0.01865096989358148646, field[3] );
    EXPECT_DOUBLE_EQ( 0, field[4] );
    EXPECT_DOUBLE_EQ( -0.09691328239836190239, field[5] );
}

TEST( dipolar_field, two_identical_aligned_particles )
{
    // Two identical particles aligned along their z-axis
    // location_1 = [0, 0, 0]
    // location_2 = [0, 0, R]

    double r=2.5e-9;
    double v=4./3 * M_PI * r * r * r;

    double R=2.917e-9;
    double R_red = R / std::pow(v,1./3);

    double field[6] = {0, 0, 0, 0, 0, 0};
    double k_av = 6.33e4;
    double ms = 400e3;
    double v_red[2] = {1.0, 1.0};
    double mag[6] = {1, 0, 0, 1, 0, 0};
    double dists[12] = {0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0};
    double dist_cubes[4] = {0, R_red*R_red*R_red, R_red*R_red*R_red, 0};
    size_t N=2;

    field::multi_add_dipolar( field, ms, k_av, v_red, mag, dists, dist_cubes, N );

    // Reference calculation
    // H1 = V*Ms/(4*pi*R**3) * (3*dot(m2,r12) - m2)
    // H2 = V*Ms/(4*pi*R**3) * (3*dot(m1,r21) - m1)
    // h1 = H1/Hk
    // h2 = H2/Hk
    double Hk = 2 * k_av / constants::MU0 / ms;
    double H1x = v * ms / (4.0*M_PI*R*R*R) * (-mag[3]);
    double H1y = v * ms / (4.0*M_PI*R*R*R) * (-mag[4]);
    double H1z = v * ms / (4.0*M_PI*R*R*R) * (2*mag[5]);
    double H2x = v * ms / (4.0*M_PI*R*R*R) * (-mag[0]);
    double H2y = v * ms / (4.0*M_PI*R*R*R) * (-mag[1]);
    double H2z = v * ms / (4.0*M_PI*R*R*R) * (2*mag[2]);

    EXPECT_NEAR( H1x/Hk, field[0], 1e-14);
    EXPECT_NEAR( H1y/Hk, field[1], 1e-14);
    EXPECT_NEAR( H1z/Hk, field[2], 1e-14);
    EXPECT_NEAR( H2x/Hk, field[3], 1e-14);
    EXPECT_NEAR( H2y/Hk, field[4], 1e-14);
    EXPECT_NEAR( H2z/Hk, field[5], 1e-14);


    mag[0] = mag[3] = 0.0;
    mag[2] = mag[5] = 1.0;
    field::zero_all_field_terms( field, 6 );
    field::multi_add_dipolar( field, ms, k_av, v_red, mag, dists, dist_cubes, N );
    H1x = v * ms / (4.0*M_PI*R*R*R) * (-mag[3]);
    H1y = v * ms / (4.0*M_PI*R*R*R) * (-mag[4]);
    H1z = v * ms / (4.0*M_PI*R*R*R) * (2*mag[5]);
    H2x = v * ms / (4.0*M_PI*R*R*R) * (-mag[0]);
    H2y = v * ms / (4.0*M_PI*R*R*R) * (-mag[1]);
    H2z = v * ms / (4.0*M_PI*R*R*R) * (2*mag[2]);

    EXPECT_NEAR( H1x/Hk, field[0], 1e-14);
    EXPECT_NEAR( H1y/Hk, field[1], 1e-14);
    EXPECT_NEAR( H1z/Hk, field[2], 1e-14);
    EXPECT_NEAR( H2x/Hk, field[3], 1e-14);
    EXPECT_NEAR( H2y/Hk, field[4], 1e-14);
    EXPECT_NEAR( H2z/Hk, field[5], 1e-14);

    mag[2] = mag[5] = 0.0;
    double norm = std::sqrt(0.1*0.1 + 0.2*0.2 + 0.3*0.3);
    mag[0] = 0.1/norm; mag[1]=0.2/norm; mag[2]=0.3/norm;
    norm = std::sqrt(0.8*0.8 + 0.5*0.5 + 0.1*0.1);
    mag[3] = -0.8/norm; mag[4] = 0.5/norm; mag[5] = -0.1/norm;
    field::zero_all_field_terms( field, 6 );
    field::multi_add_dipolar( field, ms, k_av, v_red, mag, dists, dist_cubes, N );
    H1x = v * ms / (4.0*M_PI*R*R*R) * (-mag[3]);
    H1y = v * ms / (4.0*M_PI*R*R*R) * (-mag[4]);
    H1z = v * ms / (4.0*M_PI*R*R*R) * (2*mag[5]);
    H2x = v * ms / (4.0*M_PI*R*R*R) * (-mag[0]);
    H2y = v * ms / (4.0*M_PI*R*R*R) * (-mag[1]);
    H2z = v * ms / (4.0*M_PI*R*R*R) * (2*mag[2]);

    EXPECT_NEAR( H1x/Hk, field[0], 1e-14);
    EXPECT_NEAR( H1y/Hk, field[1], 1e-14);
    EXPECT_NEAR( H1z/Hk, field[2], 1e-14);
    EXPECT_NEAR( H2x/Hk, field[3], 1e-14);
    EXPECT_NEAR( H2y/Hk, field[4], 1e-14);
    EXPECT_NEAR( H2z/Hk, field[5], 1e-14);
}

TEST( dipolar_field, two_different_aligned_particles )
{
    // Two identical particles aligned along their z-axis
    // location_1 = [0, 0, 0]
    // location_2 = [0, 0, R]

    double r1=2.5e-9;
    double r2=3.0e-9;
    double v1=4./3 * M_PI * r1 * r1 * r1;
    double v2=4./3 * M_PI * r2 * r2 * r2;

    double k1=5e4;
    double k2=7e4;

    double vav = 0.5*(v1+v2);
    double kav = 0.5*(k1+k2);

    double R=2.917e-9;
    double R_red = R / std::pow(vav,1./3);

    double field[6] = {0, 0, 0, 0, 0, 0};
    double ms = 400e3;
    double v_red[2] = {v1/vav, v2/vav};
    double mag[6] = {1, 0, 0, 1, 0, 0};
    double dists[12] = {0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0};
    double dist_cubes[4] = {0, R_red*R_red*R_red, R_red*R_red*R_red, 0};
    size_t N=2;

    field::multi_add_dipolar( field, ms, kav, v_red, mag, dists, dist_cubes, N );

    // Reference calculation
    // H1 = V1*Ms/(4*pi*R**3) * (3*dot(m2,r12) - m2)
    // H2 = V2*Ms/(4*pi*R**3) * (3*dot(m1,r21) - m1)
    // h1 = H1/Hk
    // h2 = H2/Hk
    double Hk = 2 * kav / constants::MU0 / ms;
    double H1x = v2 * ms / (4.0*M_PI*R*R*R) * (-mag[3]); // r12[0]=0
    double H1y = v2 * ms / (4.0*M_PI*R*R*R) * (-mag[4]); // r12[1]=0
    double H1z = v2 * ms / (4.0*M_PI*R*R*R) * (2*mag[5]);// r12[2]=1
    double H2x = v1 * ms / (4.0*M_PI*R*R*R) * (-mag[0]); // r21[0]=0
    double H2y = v1 * ms / (4.0*M_PI*R*R*R) * (-mag[1]); // r21[1]=0
    double H2z = v1 * ms / (4.0*M_PI*R*R*R) * (2*mag[2]);// r21[2]=-1

    EXPECT_NEAR( H1x/Hk, field[0], 1e-14);
    EXPECT_NEAR( H1y/Hk, field[1], 1e-14);
    EXPECT_NEAR( H1z/Hk, field[2], 1e-14);
    EXPECT_NEAR( H2x/Hk, field[3], 1e-14);
    EXPECT_NEAR( H2y/Hk, field[4], 1e-14);
    EXPECT_NEAR( H2z/Hk, field[5], 1e-14);


    mag[0] = mag[3] = 0.0;
    double norm = std::sqrt(0.1*0.1 + 0.2*0.2 + 0.3*0.3);
    mag[0] = 0.1/norm; mag[1]=0.2/norm; mag[2]=0.3/norm;
    norm = std::sqrt(0.8*0.8 + 0.5*0.5 + 0.1*0.1);
    mag[3] = -0.8/norm; mag[4] = 0.5/norm; mag[5] = -0.1/norm;

    field::zero_all_field_terms( field, 6 );
    field::multi_add_dipolar( field, ms, kav, v_red, mag, dists, dist_cubes, N );

    H1x = v2 * ms / (4.0*M_PI*R*R*R) * (-mag[3]);
    H1y = v2 * ms / (4.0*M_PI*R*R*R) * (-mag[4]);
    H1z = v2 * ms / (4.0*M_PI*R*R*R) * (2*mag[5]);
    H2x = v1 * ms / (4.0*M_PI*R*R*R) * (-mag[0]);
    H2y = v1 * ms / (4.0*M_PI*R*R*R) * (-mag[1]);
    H2z = v1 * ms / (4.0*M_PI*R*R*R) * (2*mag[2]);

    EXPECT_NEAR( H1x/Hk, field[0], 1e-14);
    EXPECT_NEAR( H1y/Hk, field[1], 1e-14);
    EXPECT_NEAR( H1z/Hk, field[2], 1e-14);
    EXPECT_NEAR( H2x/Hk, field[3], 1e-14);
    EXPECT_NEAR( H2y/Hk, field[4], 1e-14);
    EXPECT_NEAR( H2z/Hk, field[5], 1e-14);
}

TEST( dipolar_field, prefactor )
{
    double ms = 1 / std::sqrt(constants::MU0), k_av = 0.5;
    double result = field::dipolar_prefactor( ms, k_av );
    EXPECT_DOUBLE_EQ( 0.07957747154594767280, result );
}

TEST( dipolar_field, p2p_term )
{
    double out[3] = {1.0, 1.5, 2.0};
    double vj = 2.3;
    double rij3 = 1.2;
    double mj[3] = {0.7, 0.2, 0.4};
    double dist[3] = {1.2, 1.1, 1.3};
    double prefactor = 3.7;

    field::dipolar_add_p2p_term( out, vj, rij3, mj, dist, prefactor );
    EXPECT_DOUBLE_EQ( 36.37323333333333333333, out[0] );
    EXPECT_DOUBLE_EQ( 37.05761666666666666667, out[1] );
    EXPECT_DOUBLE_EQ( 42.86218333333333333333, out[2] );
}

TEST( applied_field, multi_add )
{
    std::function<double(double)> func = []( const double t) -> double { return 2.0*t; };
    double h[6];
    for( unsigned int i=0; i<6; i++ )
        h[i] = 2.0;
    field::multi_add_applied_Z_field_function( h, func, 3.0, 2 );

    EXPECT_DOUBLE_EQ( 2.0, h[0] );
    EXPECT_DOUBLE_EQ( 2.0, h[1] );
    EXPECT_DOUBLE_EQ( 8.0, h[2] );
    EXPECT_DOUBLE_EQ( 2.0, h[3] );
    EXPECT_DOUBLE_EQ( 2.0, h[4] );
    EXPECT_DOUBLE_EQ( 8.0, h[5] );
}

TEST( anisotropy_field, multi_uniaxial_jacobian )
{
    // Test the addition of the uniaxial jacobian
    // for many particles.
    double axes[9] = {1, 2, 3, 4, 5, 6, 4, 2, 3};
    double k_red[3] = {3.2, 5.2, 1.2};
    constexpr int jaclen = 9;
    constexpr int jaclen2 = jaclen * jaclen;
    double jac[jaclen2];

    // Init the field as all 0.0
    for( unsigned int i=0; i<jaclen2; i++ )
        jac[i] = 0.0;

    // Add the jacobian on top of the field
    field::multi_add_uniaxial_anisotropy_jacobian( jac, axes, k_red, 3 );

    // Make results more readable
    double res[jaclen][jaclen];
    for( unsigned int i=0; i<jaclen; i++ )
        for( unsigned int j=0; j<jaclen; j++ )
            res[i][j] = jac[i*jaclen+j];

    // Check that anything not on the block diag is still 10
    for( unsigned int i=0; i<jaclen; i++ )
        for( unsigned int j=0; j<jaclen; j++ )
            if ( (i/3) != (j/3) )
	      {
                EXPECT_DOUBLE_EQ( 0.0, res[i][j] );
	      }

    // Check the blocks
    EXPECT_DOUBLE_EQ( 3.2, res[0][0] );
    EXPECT_DOUBLE_EQ( 6.4, res[0][1] );
    EXPECT_DOUBLE_EQ( 9.6, res[0][2] );
    EXPECT_DOUBLE_EQ( 6.4, res[1][0] );
    EXPECT_DOUBLE_EQ( 12.8, res[1][1] );
    EXPECT_DOUBLE_EQ( 19.2, res[1][2] );
    EXPECT_DOUBLE_EQ( 9.6, res[2][0] );
    EXPECT_DOUBLE_EQ( 19.2, res[2][1] );
    EXPECT_DOUBLE_EQ( 28.8, res[2][2] );

    EXPECT_DOUBLE_EQ( 83.2, res[0+3][0+3] );
    EXPECT_DOUBLE_EQ( 104, res[0+3][1+3] );
    EXPECT_DOUBLE_EQ( 124.8, res[0+3][2+3] );
    EXPECT_DOUBLE_EQ( 104, res[1+3][0+3] );
    EXPECT_DOUBLE_EQ( 130, res[1+3][1+3] );
    EXPECT_DOUBLE_EQ( 156, res[1+3][2+3] );
    EXPECT_DOUBLE_EQ( 124.8, res[2+3][0+3] );
    EXPECT_DOUBLE_EQ( 156, res[2+3][1+3] );
    EXPECT_DOUBLE_EQ( 187.2, res[2+3][2+3] );

    EXPECT_DOUBLE_EQ( 19.2, res[0+6][0+6] );
    EXPECT_DOUBLE_EQ( 9.6, res[0+6][1+6] );
    EXPECT_DOUBLE_EQ( 14.4, res[0+6][2+6] );
    EXPECT_DOUBLE_EQ( 9.6, res[1+6][0+6] );
    EXPECT_DOUBLE_EQ( 4.8, res[1+6][1+6] );
    EXPECT_DOUBLE_EQ( 7.2, res[1+6][2+6] );
    EXPECT_DOUBLE_EQ( 14.4, res[2+6][0+6] );
    EXPECT_DOUBLE_EQ( 7.2, res[2+6][1+6] );
    EXPECT_DOUBLE_EQ( 10.8, res[2+6][2+6] );

}

TEST( distances, pair_wise_distances )
{
    std::array<double,3> p1 = {0, 0, 1};
    std::array<double,3> p2 = {3, 1, 0};
    std::array<double,3> p3 = {0.5, 0.5, 0.5};

    std::vector<std::array<double,3> > points = {p1, p2, p3};
    auto dists = distances::pair_wise_distance_vectors( points );

    EXPECT_DOUBLE_EQ( 0, dists[0][0][0] );
    EXPECT_DOUBLE_EQ( 0, dists[0][0][1] );
    EXPECT_DOUBLE_EQ( 0, dists[0][0][2] );
    EXPECT_DOUBLE_EQ( 0, dists[1][1][0] );
    EXPECT_DOUBLE_EQ( 0, dists[1][1][1] );
    EXPECT_DOUBLE_EQ( 0, dists[1][1][2] );
    EXPECT_DOUBLE_EQ( 0, dists[2][2][0] );
    EXPECT_DOUBLE_EQ( 0, dists[2][2][1] );
    EXPECT_DOUBLE_EQ( 0, dists[2][2][2] );


    EXPECT_DOUBLE_EQ( 3, dists[0][1][0] );
    EXPECT_DOUBLE_EQ( 1, dists[0][1][1] );
    EXPECT_DOUBLE_EQ( -1, dists[0][1][2] );
    EXPECT_DOUBLE_EQ( -3, dists[1][0][0] );
    EXPECT_DOUBLE_EQ( -1, dists[1][0][1] );
    EXPECT_DOUBLE_EQ( 1, dists[1][0][2] );

    EXPECT_DOUBLE_EQ( 0.5, dists[0][2][0] );
    EXPECT_DOUBLE_EQ( 0.5, dists[0][2][1] );
    EXPECT_DOUBLE_EQ( -0.5, dists[0][2][2] );
    EXPECT_DOUBLE_EQ( -0.5, dists[2][0][0] );
    EXPECT_DOUBLE_EQ( -0.5, dists[2][0][1] );
    EXPECT_DOUBLE_EQ( 0.5, dists[2][0][2] );

    EXPECT_DOUBLE_EQ( -2.5, dists[1][2][0] );
    EXPECT_DOUBLE_EQ( -0.5, dists[1][2][1] );
    EXPECT_DOUBLE_EQ( 0.5, dists[1][2][2] );
    EXPECT_DOUBLE_EQ( 2.5, dists[2][1][0] );
    EXPECT_DOUBLE_EQ( 0.5, dists[2][1][1] );
    EXPECT_DOUBLE_EQ( -0.5, dists[2][1][2] );
}

TEST( distances, pair_wise_magnitudes )
{
    std::array<double,3> d00 = {0, 0, 0};
    std::array<double,3> d01 = {0, 0, 1};
    std::array<double,3> d02 = {3, 1, 0};
    std::array<double,3> d12 = {0.5, 0.5, 0.5};

    std::vector<std::vector<std::array<double, 3> > > dists( 3, std::vector<std::array<double,3> >( 3 ) );
    dists[0][0] = d00;
    dists[0][1] = d01;
    dists[0][2] = d02;
    dists[1][0] = d01;
    dists[1][1] = d00;
    dists[1][2] = d12;
    dists[2][0] = d02;
    dists[2][1] = d12;
    dists[2][2] = d00;

    auto mags = distances::pair_wise_distance_magnitude( dists );

    EXPECT_DOUBLE_EQ( 0, mags[0][0] );
    EXPECT_DOUBLE_EQ( 0, mags[1][1] );
    EXPECT_DOUBLE_EQ( 0, mags[2][2] );

    EXPECT_DOUBLE_EQ( 1, mags[0][1] );
    EXPECT_DOUBLE_EQ( 1, mags[1][0] );

    EXPECT_DOUBLE_EQ( 3.1622776601683795, mags[0][2] );
    EXPECT_DOUBLE_EQ( 3.1622776601683795, mags[2][0] );

    EXPECT_DOUBLE_EQ( 0.8660254037844386, mags[1][2] );
    EXPECT_DOUBLE_EQ( 0.8660254037844386, mags[2][1] );
}
