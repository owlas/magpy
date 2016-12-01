#include "../googletest/include/gtest/gtest.h"
#include "../include/easylogging++.h"
#include "../include/llg.hpp"
#include "../include/integrators.hpp"
#include "../include/io.hpp"
#include "../include/simulation.hpp"
#include <cmath>
#include <random>

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

    EXPECT_FLOAT_EQ( 1.03019352, next_state[0] );
    EXPECT_FLOAT_EQ( 2.57118625, next_state[1] );
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

    res.magnetisation[0] = 2;
    res.magnetisation[1] = 3;
    res.field[0] = 4;
    res.field[1] = 5;
    res.time[0] = 6;
    res.time[1] = 7;

    simulation::save_results( "test.out", res );

    int nread;
    double arr[2];
    FILE *in;
    in=fopen( "test.out.mag", "rb" );
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
