#include "../googletest/include/gtest/gtest.h"
#include "../include/easylogging++.h"
#include "../include/llg.hpp"
#include "../include/integrators.hpp"
#include "../include/io.hpp"
#include "../include/simulation.hpp"
#include "../include/json.hpp"
#include "../include/moma_config.hpp"
#include "../include/trap.hpp"
#include "../include/optimisation.hpp"
#include "../include/rng.hpp"
#include "../include/stochastic_processes.hpp"
#include "../include/field.hpp"
#include "../include/dom.hpp"
#include "../include/constants.hpp"
#include <cmath>
#include <random>
#ifdef USEMKL
#include <mkl_lapacke.h>
#else
#include <lapacke.h>
#endif
#include <stdexcept>

INITIALIZE_EASYLOGGINGPP

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
    const double current_state[2] = { 1, 2 };
    const double wiener_steps[3] = { 0.1, 0.01, 0.001 };

    const std::function<void(double*,const double*,const double)> drift =
        [](double *out, const double *in, const double t) {
        out[0]=in[0]*in[1]*t; out[1]=3*in[0];
    };
    const std::function<void(double*,const double*,const double)> diffusion =
        [](double *out, const double *in, const double t) {
        out[0]=t; out[1]=in[0]; out[2]=in[1]; out[3]=in[1]; out[4]=4.0;
        out[5]=in[0]*in[1];
    };

    const size_t n_dims=2, wiener_dims=3;
    const double t=0, step_size=0.1;;

    integrator::heun(
        next_state, drift_arr, trial_drift_arr, diffusion_matrix, trial_diffusion_matrix,
        current_state, wiener_steps, drift, diffusion, n_dims, wiener_dims,
        t, step_size );

    EXPECT_DOUBLE_EQ( 1.03019352, next_state[0] );
    EXPECT_DOUBLE_EQ( 2.571186252, next_state[1] );
}

TEST(io, write_array)
{
    double arr[3] = {1, 2, 3}, arrback[3];
    int fail, nread;
    fail = io::write_array( "./output/test.out", arr, 3 );
    ASSERT_EQ( 0, fail );

    // Read back the data
    FILE *in;
    in = fopen( "./output/test.out", "rb" );
    nread = fread( arrback, sizeof(double), 3, in );
    fclose( in );

    ASSERT_EQ( 3, nread );
    EXPECT_DOUBLE_EQ( 1, arrback[0] );
    EXPECT_DOUBLE_EQ( 2, arrback[1] );
    EXPECT_DOUBLE_EQ( 3, arrback[2] );

}

TEST( simulation, save_results )
{
    simulation::results res( 2 );

    res.mx[0] = 2;
    res.mx[1] = 3;
    res.field[0] = 4;
    res.field[1] = 5;
    res.time[0] = 6;
    res.time[1] = 7;
    res.energy_loss = 10;


    simulation::save_results( "output/test.out", res );

    int nread;
    double arr[2];
    FILE *in;
    in=fopen( "output/test.out.mx", "rb" );
    nread = fread( arr, sizeof(double), 2, in );
    ASSERT_EQ( 2, nread );
    EXPECT_DOUBLE_EQ( 2, arr[0] );
    EXPECT_DOUBLE_EQ( 3, arr[1] );

    in=fopen( "output/test.out.field", "rb" );
    nread = fread( arr, sizeof(double), 2, in );
    ASSERT_EQ( 2, nread );
    EXPECT_DOUBLE_EQ( 4, arr[0] );
    EXPECT_DOUBLE_EQ( 5, arr[1] );

    in=fopen( "output/test.out.time", "rb" );
    nread = fread( arr, sizeof(double), 2, in );
    ASSERT_EQ( 2, nread );
    EXPECT_DOUBLE_EQ( 6, arr[0] );
    EXPECT_DOUBLE_EQ( 7, arr[1] );

    in=fopen( "output/test.out.energy", "rb" );
    nread = fread( arr, sizeof(double), 1, in );
    ASSERT_EQ( 1, nread );
    EXPECT_DOUBLE_EQ( 10, arr[0] );
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
    const std::function<void(double*,const double*,const double)> drift =
        [ou_theta,ou_mu](double *out, const double *in, const double t)
        { out[0]=ou_theta*(ou_mu - in[0]); };
    const std::function<void(double*,const double*,const double)> diffusion =
        [ou_sigma](double *out, const double *in, const double t) { out[0]=ou_sigma; };


    // Create a wiener path
    double wiener_process[n_steps];
    std::normal_distribution<double> dist( 0, std::sqrt( step_size ) );
    std::mt19937_64 rng;
    for( unsigned int i=0; i<n_steps; i++ )
        wiener_process[i] = dist( rng );

    // Generate the states
    double states[n_steps];
    driver::heun(
        states, initial_state, wiener_process, drift, diffusion,
        n_steps, n_dims, n_wiener, step_size );

    // Compute the analytic solution and compare pathwise similarity
    double truesol = initial_state[0];
    for( unsigned int i=1; i<n_steps; i++ )
    {
        truesol = truesol*std::exp( -ou_theta*step_size )
            + ou_mu*( 1-std::exp( -ou_theta*step_size ) )
            + ou_sigma*std::sqrt( ( 1- std::exp( -2*ou_theta*step_size ) )/( 2*ou_theta ) )
            * wiener_process[i-1]/std::sqrt( step_size );

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

json test_llg_config()
{
    nlohmann::json j;
    j = {
        {"simulation-mode", "llg"},
        {"simulation", {
                {"ensemble-size", 1},
                {"simulation-time", 1e-9},
                {"time-step", 1e-14},
                {"renormalisation", true},
                {"seeds", {100, 200}}
            }},
        {"output", {
                {"directory", "output"},
                {"max-samples", 1000}
            }},
        {"global", {
                {"temperature", 300},
                {"applied-field", {
                        {"shape", "square-fourier"},
                        {"frequency", 300e3},
                        {"amplitude", 5e5},
                        {"components", 10}
                    }}
            }},
        {"particle", {
                {"damping", 0.1},
                {"radius", 7e-9},
                {"anisotropy", 23e3},
                {"anisotropy-axis", {0, 0, 1}},
                {"magnetisation", 446e3},
                {"magnetisation-direction", {1, 0, 0}}
            }}
    };
    return j;
}

json test_dom_config()
{
    nlohmann::json j;
    j = {
        {"simulation-mode", "dom"},
        {"simulation", {
                {"simulation-time", 1e-9},
                {"time-step", 1e-14},
            }},
        {"output", {
                {"directory", "output"},
                {"max-samples", 1000}
            }},
        {"global", {
                {"temperature", 300},
                {"applied-field", {
                        {"shape", "square-fourier"},
                        {"frequency", 300e3},
                        {"amplitude", 5e5},
                        {"components", 10}
                    }},
                {"tau0", 1e-10}
            }},
        {"particles", {{
                {"radius", 7e-9},
                {"anisotropy", 23e3},
                {"magnetisation", 446e3},
                {"initial-probs", {1, 0}},
                {"damping", 0.01}
            }}}
    };
    return j;
}

TEST( moma_config, valid_llg_input )
{
    nlohmann::json in = test_llg_config();
    // This should be a valid input

    moma_config::validate_for_llg( in );
}

TEST( moma_config, valid_dom_input )
{
    nlohmann::json in = test_dom_config();
    moma_config::validate_for_dom( in );
}

TEST( moma_config, transform_llg )
{
    nlohmann::json in = test_llg_config();
    auto out = moma_config::transform_input_parameters_for_llg( in );

    EXPECT_EQ( 1,
               out.at("simulation").at("ensemble-size").get<int>() );
    EXPECT_NEAR( 0.000179815211117,
                 out.at("simulation").at("time-step").get<double>(), 1e-14);
    EXPECT_DOUBLE_EQ( 17.981521111752436,
                      out.at("simulation").at("simulation-time").get<double>() );
    EXPECT_DOUBLE_EQ( 17981521111.752434,
                      out.at("simulation").at("time-factor").get<double>() );
    EXPECT_EQ( true,
               out.at("simulation").at("renormalisation").get<bool>() );
    EXPECT_EQ( false,
               out.at("simulation").at("steady-cycle-activated").get<bool>() );
    EXPECT_EQ( "output",
               out.at("output").at("directory").get<std::string>() );
    EXPECT_EQ( 1000,
               out.at("output").at("max-samples").get<int>() );
    EXPECT_DOUBLE_EQ( 300,
                      out.at("global").at("temperature").get<double>() );
    EXPECT_DOUBLE_EQ( 1.6683794331722291e-05,
                      out.at("global").at("applied-field").at("frequency").get<double>() );
    EXPECT_DOUBLE_EQ( 6.0919579213043464,
                      out.at("global").at("applied-field").at("amplitude").get<double>() );
    EXPECT_DOUBLE_EQ( 82075.419177049276,
                      out.at("global").at("anisotropy-field"));
    EXPECT_DOUBLE_EQ( 0.1,
                      out.at("particle").at("damping") );
    EXPECT_DOUBLE_EQ( 1.436755040241732e-24,
                      out.at("particle").at("volume"));
    EXPECT_DOUBLE_EQ( 7e-9,
                      out.at("particle").at("radius"));
    EXPECT_DOUBLE_EQ( 23e3,
                      out.at("particle").at("anisotropy"));
    EXPECT_DOUBLE_EQ( 0.11140026492035397,
                      out.at("particle").at("thermal-field-strength"));
    EXPECT_DOUBLE_EQ( 7.97822314341568,
                      out.at("particle").at("stability-ratio"));
}

TEST( moma_config, transform_dom )
{
    nlohmann::json in = test_dom_config();
    auto out = moma_config::transform_input_parameters_for_dom( in );

    EXPECT_EQ( false,
               out.at("simulation").at("steady-cycle-activated").get<bool>() );
    EXPECT_DOUBLE_EQ( 1.436755040241732e-24,
                      out.at("particles")[0].at("volume").get<double>() );
    EXPECT_DOUBLE_EQ( 7.97822314341568,
                      out.at("particles")[0].at("stability-ratio").get<double>() );
    EXPECT_DOUBLE_EQ( 82075.419177049276,
                      out.at("particles")[0].at("anisotropy-field").get<double>() );
    EXPECT_DOUBLE_EQ( 6.0919579213043464,
                      out.at("particles")[0].at("reduced-field-amplitude").get<double>() );
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
    double x_root[2], x_tmp[2], jac_out[4], x0[2], eps=1e-8;
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
    auto flag = optimisation::newton_raphson_noinv(
        x_root, x_tmp, jac_out, ipiv, &lapack_err,
        fj, x0, dim, eps, max_iter );
    ASSERT_EQ( optimisation::SUCCESS, flag );
    EXPECT_LE( std::abs( x_root[0] + 5 ), eps );
    EXPECT_LE( std::abs( x_root[1] - 4.9 ), eps );

    // Find the second solution
    x0[0] = x0[1] = 1.0;
    flag = optimisation::newton_raphson_noinv(
        x_root, x_tmp, jac_out, ipiv, &lapack_err,
        fj, x0, dim, eps, max_iter );
    ASSERT_EQ( optimisation::SUCCESS, flag );
    EXPECT_LE( std::abs( x_root[0] - 2 ), eps );
    EXPECT_LE( std::abs( x_root[1] ), eps );
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

TEST( implicit_integrator_midpoint, atest )
{
    // Stiff 2d system with 1d wiener
    double x[2], dwm[1], a_work[2], b_work[2], adash_work[4], bdash_work[4];
    double x_guess[2], x_opt_tmp[2], x_opt_jac[4], x0[2]={1.0, 2.0}, dw[1]={0.07};
    lapack_int x_opt_ipiv[2];
    const size_t n_dim=2;
    const size_t w_dim=1;
    const double t=0;
    const double dt=1e-3;
    const double eps=1e-10;
    const size_t max_iter=200;

    const double a=5, b=0.1; //try b=10.0 too
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
    EXPECT_NEAR( 1.01202953, x[0], 1e-7 );
    EXPECT_NEAR( 2.00902904, x[1], 1e-7 );
}

TEST( simulation, energy_loss )
{
    struct simulation::results res( 5 );
    res.field[0] = 0;
    res.field[1] = 0.5;
    res.field[2] = 1;
    res.field[3] = 0.5;
    res.field[4] = 0;
    res.mz[0] = 0;
    res.mz[1] = 1;
    res.mz[2] = 1;
    res.mz[3] = 0;
    res.mz[4] = 0;

    double area = trap::trapezoidal( res.field.get(), res.mz.get(), 5 );
    ASSERT_DOUBLE_EQ( 0.5, area );

    double power = simulation::energy_loss(
        res, 5, 2 );
    EXPECT_DOUBLE_EQ( 5*2*constants::MU0*area, power );
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
