#include "../googletest/include/gtest/gtest.h"
#include "../include/easylogging++.h"
#include "../include/llg.hpp"
#include "../include/integrators.hpp"
#include "../include/io.hpp"
#include "../include/simulation.hpp"

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

TEST(heun, 2d_wiener)
{
    double next_state[2];
    double drift_arr[2];
    double trial_drift_arr[2];
    double diffusion_matrix[4];
    double trial_diffusion_matrix[4];
    const double current_state[2] = { 0, 0 };
    const double wiener_steps[2] = { 0.1, 0.01 };

    const std::function<void(double*,const double*,const double)> drift =
        [](double *out, const double *in, const double t) { out[0]=0;out[1]=0; };
    const std::function<void(double*,const double*,const double)> diffusion =
        [](double *out, const double *in, const double t) { out[0]=1;out[1]=0;out[2]=0;out[3]=1; };

    const size_t n_dims=2, wiener_dims=2;
    const double t=0, step_size=1.44;

    integrator::heun(
        next_state, drift_arr, trial_drift_arr, diffusion_matrix, trial_diffusion_matrix,
        current_state, wiener_steps, drift, diffusion, n_dims, wiener_dims,
        t, step_size );

    EXPECT_FLOAT_EQ( 0.1, next_state[0] );
    EXPECT_FLOAT_EQ( 0.01, next_state[1] );
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
    ASSERT_FLOAT_EQ( 1, arrback[0] );
    ASSERT_FLOAT_EQ( 2, arrback[1] );
    ASSERT_FLOAT_EQ( 3, arrback[2] );

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
    ASSERT_FLOAT_EQ( 2, arr[0] );
    ASSERT_FLOAT_EQ( 3, arr[1] );

    in=fopen( "test.out.field", "rb" );
    nread = fread( arr, sizeof(double), 2, in );
    ASSERT_EQ( 2, nread );
    ASSERT_FLOAT_EQ( 4, arr[0] );
    ASSERT_FLOAT_EQ( 5, arr[1] );

    in=fopen( "test.out.time", "rb" );
    nread = fread( arr, sizeof(double), 2, in );
    ASSERT_EQ( 2, nread );
    ASSERT_FLOAT_EQ( 6, arr[0] );
    ASSERT_FLOAT_EQ( 7, arr[1] );
}
