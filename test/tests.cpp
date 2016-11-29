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
