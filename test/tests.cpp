#include "../googletest/include/gtest/gtest.h"
#include "../include/easylogging++.h"
#include "../include/llg.hpp"

INITIALIZE_EASYLOGGINGPP

TEST(llg, drift)
{
    double deriv[3];
    double state[3] = {2, 3, 4};
    double field[3] = {1, 2, 3};
    double time=0, damping=3;
    llg::drift( deriv, state, time, damping, field );

    EXPECT_EQ( -34, deriv[0] );
    EXPECT_EQ(  -4, deriv[1] );
    EXPECT_EQ(  20, deriv[2] );
}

TEST(llg, diffusion)
{
    double deriv[3*3];
    double state[3] = { 2, 3, 4 };
    double time=0; double sr=2; double alpha=3;
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
