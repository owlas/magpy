// llg.cpp
// llg implementation
#include "../include/llg.hpp"

void llg::drift( double *deriv, const double *state, const double time,
                 const double alpha, const double *heff )
{
    deriv[0] = state[2]*heff[1] - state[1]*heff[2]
        + alpha*(heff[0]*(state[1]*state[1] + state[2]*state[2])
                 - state[0]*(state[1]*heff[1]
                             + state[2]*heff[2]));
    deriv[1] = state[0]*heff[2] - state[2]*heff[0]
        + alpha*(heff[1]*(state[0]*state[0] + state[2]*state[2])
                 - state[1]*(state[0]*heff[0]
                             + state[2]*heff[2]));
    deriv[2] = state[1]*heff[0] - state[0]*heff[1]
        + alpha*(heff[2]*(state[0]*state[0] + state[1]*state[1])
                 - state[2]*(state[0]*heff[0]
                             + state[1]*heff[1]));
}

void llg::diffusion( double *deriv, const double *state, const double time,
                     const double sr, const double alpha )
{
    deriv[0] = alpha*sr*(state[1]*state[1]+state[2]*state[2]); // 1st state
    deriv[1] = sr*(state[2]-alpha*state[0]*state[1]);
    deriv[2] = -sr*(state[1]+alpha*state[0]*state[2]);

    deriv[3] = -sr*(state[2]+alpha*state[0]*state[1]);         // 2nd state
    deriv[4] = alpha*sr*(state[0]*state[0]+state[2]*state[2]);
    deriv[5] = sr*(state[0]-alpha*state[1]*state[2]);

    deriv[6] = sr*(state[1]-alpha*state[0]*state[2]);          // 3rd state
    deriv[7] = -sr*(state[0]+alpha*state[1]*state[2]);
    deriv[8] = alpha*sr*(state[0]*state[0]+state[1]*state[1]);
}

void llg::diffusion_jacobian( double *jacobian, const double *state,
                              const double time,
                              const double sr, const double alpha )
{
    // 1st state, 1st wiener process, state 1,2,3 partial derivatives
    jacobian[0] = 0;
    jacobian[1] = 2*alpha*sr*state[1];
    jacobian[2] = 2*alpha*sr*state[2];

    jacobian[3] = -alpha*sr*state[1];
    jacobian[4] = -alpha*sr*state[2];
    jacobian[5] = sr;

    jacobian[6] = -alpha*sr*state[2];
    jacobian[7] = -sr;
    jacobian[8] = -alpha*sr*state[0];

    jacobian[9] = -alpha*sr*state[1];
    jacobian[10] = -alpha*sr*state[0];
    jacobian[11] = -sr;

    jacobian[12] = 2*alpha*sr*state[0];
    jacobian[13] = 0;
    jacobian[14] = 2*alpha*sr*state[2];

    jacobian[15] = sr;
    jacobian[16] = -alpha*sr*state[2];
    jacobian[17] = -alpha*sr*state[1];

    jacobian[18] = -alpha*sr*state[2];
    jacobian[19] = sr;
    jacobian[20] = -alpha*sr*state[0];

    jacobian[21] = -sr;
    jacobian[22] = -alpha*sr*state[2];
    jacobian[23] = -alpha*sr*state[1];

    jacobian[24] = 2*alpha*sr*state[0];
    jacobian[25] = 2*alpha*sr*state[2];
    jacobian[26] = 0;
}
