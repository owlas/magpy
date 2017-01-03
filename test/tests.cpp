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
#include <cmath>
#include <random>
#include <lapacke.h>
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
    fail = io::write_array( "test.out", arr, 3 );
    ASSERT_EQ( 0, fail );

    // Read back the data
    FILE *in;
    in = fopen( "test.out", "rb" );
    nread = fread( arrback, sizeof(double), 3, in );
    fclose( in );

    ASSERT_EQ( 3, nread );
    ASSERT_DOUBLE_EQ( 1, arrback[0] );
    ASSERT_DOUBLE_EQ( 2, arrback[1] );
    ASSERT_DOUBLE_EQ( 3, arrback[2] );

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

    simulation::save_results( "test.out", res );

    int nread;
    double arr[2];
    FILE *in;
    in=fopen( "test.out.mx", "rb" );
    nread = fread( arr, sizeof(double), 2, in );
    ASSERT_EQ( 2, nread );
    ASSERT_DOUBLE_EQ( 2, arr[0] );
    ASSERT_DOUBLE_EQ( 3, arr[1] );

    in=fopen( "test.out.field", "rb" );
    nread = fread( arr, sizeof(double), 2, in );
    ASSERT_EQ( 2, nread );
    ASSERT_DOUBLE_EQ( 4, arr[0] );
    ASSERT_DOUBLE_EQ( 5, arr[1] );

    in=fopen( "test.out.time", "rb" );
    nread = fread( arr, sizeof(double), 2, in );
    ASSERT_EQ( 2, nread );
    ASSERT_DOUBLE_EQ( 6, arr[0] );
    ASSERT_DOUBLE_EQ( 7, arr[1] );
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

TEST( moma_config, normalise_json )
{
    nlohmann::json in;
    in = {
        {"simulation", {
                {"ensemble-size", 1},
                {"simulation-time", 1e-9},
                {"time-step", 1e-14}
            }},
        {"output", {
                {"directory", "output"}
            }},
        {"global", {
                {"temperature", 300},
                {"applied-field", {
                        {"shape", "sine"},
                        {"frequency", 300e3},
                        {"amplitude", 5e5}
                    }}
            }},
        {"particle", {
                {"damping", 0.1},
                {"radius", 7e-9},
                {"anisotropy", 23e3},
                {"anisotropy-axis", {0, 0, 1}},
                {"initial-magnetisation", {446e3, 0, 0}}
            }}
    };

    nlohmann::json ref_out = {
        {"simulation", {
                {"ensemble-size", 1},
                {"simulation-time", 17.981521111752436},
                {"time-step", 0.000179815211117},
                {"time-factor", 17981521111.752434}
            }},
        {"output", {
                {"directory", "output"}
            }},
        {"global", {
                {"temperature", 300},
                {"applied-field", {
                        {"frequency", 1.6683794331722291e-05},
                        {"shape", "sine"},
                        {"amplitude", 6.0919579213043464}
                    }},
                {"anisotropy-field", 82075.419177049276}
            }},
        {"particle", {
                {"damping", 0.1},
                {"volume", 1.436755040241732e-24},
                {"radius", 7e-9},
                {"anisotropy", 23e3},
                {"anisotropy-axis", {0, 0, 1}},
                {"initial-magnetisation", {1, 0, 0}},
                {"thermal-field-strength", 0.11140026492035397},
                {"stability-ratio", 7.97822314341568}
            }}
    };
    nlohmann::json out = moma_config::normalise( in );
    ASSERT_EQ( ref_out["simulation"]["ensemble-size"], out["simulation"]["ensemble-size"]);
    ASSERT_NEAR( ref_out["simulation"]["time-step"].get<double>(), out["simulation"]["time-step"].get<double>(), 1e-14);
    ASSERT_DOUBLE_EQ( ref_out["simulation"]["simulation-time"], out["simulation"]["simulation-time"]);
    ASSERT_DOUBLE_EQ( ref_out["simulation"]["time-factor"], out["simulation"]["time-factor"]);
    ASSERT_EQ( ref_out["output"]["directory"], out["output"]["directory"]);
    ASSERT_DOUBLE_EQ( ref_out["global"]["temperature"], out["global"]["temperature"]);
    ASSERT_DOUBLE_EQ( ref_out["global"]["applied-field"]["frequency"], out["global"]["applied-field"]["frequency"]);
    ASSERT_EQ( ref_out["global"]["applied-field"]["shape"], out["global"]["applied-field"]["shape"]);
    ASSERT_DOUBLE_EQ( ref_out["global"]["applied-field"]["amplitude"], out["global"]["applied-field"]["amplitude"]);
    ASSERT_DOUBLE_EQ( ref_out["global"]["anisotropy-field"], out["global"]["anisotropy-field"]);
    ASSERT_DOUBLE_EQ( ref_out["particle"]["damping"], out["particle"]["damping"]);
    ASSERT_DOUBLE_EQ( ref_out["particle"]["volume"], out["particle"]["volume"]);
    ASSERT_DOUBLE_EQ( ref_out["particle"]["radius"], out["particle"]["radius"]);
    ASSERT_DOUBLE_EQ( ref_out["particle"]["anisotropy"], out["particle"]["anisotropy"]);
    ASSERT_EQ( ref_out["particle"]["anisotropy-axis"], out["particle"]["anisotropy-axis"]);
    ASSERT_EQ( ref_out["particle"]["initial-magnetisation"], out["particle"]["initial-magnetisation"]);
    ASSERT_DOUBLE_EQ( ref_out["particle"]["thermal-field-strength"], out["particle"]["thermal-field-strength"]);
    ASSERT_DOUBLE_EQ( ref_out["particle"]["stability-ratio"], out["particle"]["stability-ratio"]);
}

TEST( moma_config, validate_compute_options )
{
    // removed test for validate
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
    ASSERT_LE( std::abs( x_root - 3.2 ), eps );
    ASSERT_EQ( optimisation::SUCCESS, res );
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
    auto f = []( double*out,const double*in )->void
    {
        out[0] = -( in[0]-2 )*( in[0]-2 ) - 2*in[0]*in[1];
        out[1] = -in[1]*( in[0]+1 ) - 4*in[1];
    };
    auto jac = []( double*out,const double*in )->void
    {
        out[0] = -2*in[0] - 2*in[1] + 4;
        out[1] = -2*in[0];
        out[2] = -in[1];
        out[3] = -in[0]-5;
    };

    // Find the first solution
    x0[0] = x0[1] = -5.2;
    auto flag = optimisation::newton_raphson_noinv(
        x_root, x_tmp, jac_out, ipiv, &lapack_err,
        f, jac, x0, dim, eps, max_iter );
    ASSERT_EQ( optimisation::SUCCESS, flag );
    ASSERT_LE( std::abs( x_root[0] + 5 ), eps );
    ASSERT_LE( std::abs( x_root[1] - 4.9 ), eps );

    // Find the second solution
    x0[0] = x0[1] = 1.0;
    flag = optimisation::newton_raphson_noinv(
        x_root, x_tmp, jac_out, ipiv, &lapack_err,
        f, jac, x0, dim, eps, max_iter );
    ASSERT_EQ( optimisation::SUCCESS, flag );
    ASSERT_LE( std::abs( x_root[0] - 2 ), eps );
    ASSERT_LE( std::abs( x_root[1] ), eps );
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
    auto f = []( double*out,const double*in )->void
    {
        out[0] = -( in[0]-2 )*( in[0]-2 ) - 2*in[0]*in[1];
        out[1] = out[0]*5.9;
    };
    auto jac = []( double*out,const double*in )->void
    {
        out[0] = -2*in[0] - 2*in[1] + 4;
        out[1] = -2*in[0];
        out[2] = out[0]*5.9;
        out[3] = out[1]*5.9;
    };

    // Find the first solution
    x0[0] = x0[1] = -5.2;
    auto flag = optimisation::newton_raphson_noinv(
        x_root, x_tmp, jac_out, ipiv, &lapack_err,
        f, jac, x0, dim, eps, max_iter );
    ASSERT_EQ( optimisation::LAPACK_ERR, flag );
    ASSERT_GE( lapack_err, 0 );
}

TEST( rng, mt_norm )
{
    // Test the mt algorithm
    RngMtNorm rng( 999, 0.2 ); // seed, standard deviation
    ASSERT_DOUBLE_EQ(  0.27100401702487437, rng.get() );
    ASSERT_DOUBLE_EQ( -0.18950297511996847, rng.get() );
    ASSERT_DOUBLE_EQ( -0.27620532277321042, rng.get() );
    ASSERT_DOUBLE_EQ( -0.10214651921310165, rng.get() );
    ASSERT_DOUBLE_EQ( -0.19125423381292103, rng.get() );
}

TEST( rng, array )
{
    // Test a pre-allocated array
    double arr[5] = {0.01, 0.2, 1, 5, -0.1};
    RngArray rng( arr, 5 );
    ASSERT_DOUBLE_EQ( 0.01, rng.get() );
    ASSERT_DOUBLE_EQ(  0.2, rng.get() );
    ASSERT_DOUBLE_EQ(    1, rng.get() );
    ASSERT_DOUBLE_EQ(    5, rng.get() );
    ASSERT_DOUBLE_EQ( -0.1, rng.get() );

    // Check that error is thrown if too many calls
    try
    {
        rng.get();
        FAIL() << "Expected std::out_of_range";
    }
    catch( std::out_of_range const & err )
    {
        EXPECT_EQ( err.what(),
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
    ASSERT_DOUBLE_EQ( 0.01, rng.get() ); // 0th index
    ASSERT_DOUBLE_EQ(    5, rng.get() ); // 3rd index

    // Next call will be 6th index which exceeds array length
    try
    {
        rng.get();
        FAIL() << "Expected std::out_of_range";
    }
    catch( std::out_of_range const & err )
    {
        EXPECT_EQ( err.what(),
                   std::string("Exceeded allocated random number array size"));
    }
    catch( ... )
    {
        FAIL() << "Expected std::out_of_range";
    }
}

TEST( stochastic, reduce_wiener_increments )
{
    double arr[6] = {0.1, 0.4, 0.4, 0.2, 0.1, 0.9};
    auto len = stochastic::reduce_wiener_increments( arr, 6, 2 );
    ASSERT_DOUBLE_EQ( 0.5, arr[0] ); // arr[0]+arr[1]
    ASSERT_DOUBLE_EQ( 0.6, arr[1] ); // arr[2]+arr[3]
    ASSERT_DOUBLE_EQ( 1.0, arr[2] ); // arr[4]+arr[5]
    ASSERT_DOUBLE_EQ( 0.2, arr[3] ); // unchanged
    ASSERT_EQ( len, 3 );

    double arr4[6] = {0.1, 0.4, 0.4, 0.2, 0.1, 0.9};
    auto len4 = stochastic::reduce_wiener_increments( arr4, 6, 4 );
    ASSERT_DOUBLE_EQ( 1.1, arr4[0] ); // arr[0]+arr[1]+arr[2]+arr[3]
    ASSERT_DOUBLE_EQ( 1.0, arr4[1] ); // unchanged
    ASSERT_EQ( len4, 1 );
}
