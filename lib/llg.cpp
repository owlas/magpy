#include "../include/llg.hpp"
#include "../include/field.hpp"

/// Deterministic drift component of the stochastic LLG
/**
 * @param[out] deriv  drift derivative of the deterministic part
 * of the stochastic llg [length 3]
 * @param[in] state current state of the magnetisation vector [length
 * 3]
 * @param[in] t time (has no effect)
 * @param[in] alpha damping ratio
 * @param[in] the effective field on the magnetisation [length 3]
 */
void llg::drift( double *deriv, const double *state, const double,
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

/// Deterministic drift component of the stochastic Ito form of the LLG
/**
 * @param[out] deriv  drift derivative of the deterministic part
 * of the stochastic llg [length 3]
 * @param[in] state current state of the magnetisation vector [length
 * 3]
 * @param[in] t time (has no effect)
 * @param[in] alpha damping ratio
 * @param[in] sig thermal noise strength
 * @param[in] the effective field on the magnetisation [length 3]
 */
void llg::ito_drift( double *deriv, const double *state, const double,
                     const double alpha, const double sig, const double *heff )
{
    llg::drift( deriv, state, 0, alpha, heff );

    // ITO CORRECTION
    deriv[0] -= 0;//sig*state[0];
    deriv[1] -= 0;//sig*state[1];
    deriv[2] -= 0;//sig*state[2];
}

/// Jacobian of the deterministic drift component of the stochastic
/// LLG
/**
 * Since, in the general case, the effective field is a function of
 * the magnetisation, the Jacobian of the effective field must be
 * known in order to compute the Jacobian of the drift component.
 * @param[out] jac Jacobian of the drift [length 3x3]
 * @param[in]  m state of the magnetisation vector [length 3]
 * @param[in]  t time (has no effect)
 * @param[in]  a damping ratio \f$\alpha\f$
 * @param[in]  h effective field acting on the magnetisation [length
 * 3]
 * @param[in]  hj Jacobian of the effective field evaluated at the
 * current value of `m` [length 3x3]
 */
void llg::drift_jacobian( double *jac, const double *m, const double,
                          const double a, const double *h,
                          const double *hj )
{
    jac[0] =  m[2]*hj[3] - m[1]*hj[6] + a*( -m[1]*h[1] - m[2]*h[2] + ( m[1]*m[1] + m[2]*m[2] )*hj[0] - m[0]*( m[1]*hj[3] + m[2]*hj[6] ) );
    jac[1] = -h[2] + m[2]*hj[4] - m[1]*hj[7] + a*( 2*m[1]*h[0] + ( m[1]*m[1] + m[2]*m[2] )*hj[1] - m[0]*( h[1] + m[1]*hj[4] + m[2]*hj[7] ) );
    jac[2] =  h[1] + m[2]*hj[5] - m[1]*hj[8] + a*( 2*m[2]*h[0] + ( m[1]*m[1] + m[2]*m[2] )*hj[2] - m[0]*( h[2] + m[1]*hj[5] + m[2]*hj[8] ) );
    jac[3] =  h[2] - m[2]*hj[0] + m[0]*hj[6] + a*( 2*m[0]*h[1] + ( m[0]*m[0] + m[2]*m[2] )*hj[3] - m[1]*( h[0] + m[0]*hj[0] + m[2]*hj[6] ) );
    jac[4] = -m[2]*hj[1] + m[0]*hj[7] + a*( -m[0]*h[0] - m[2]*h[2] + ( m[0]*m[0] + m[2]*m[2] )*hj[4] - m[1]*( m[0]*hj[1] + m[2]*hj[7] ) );
    jac[5] = -h[0] - m[2]*hj[2] + m[0]*hj[8] + a*( 2*m[2]*h[1] + ( m[0]*m[0] + m[2]*m[2] )*hj[5] - m[1]*( h[2] + m[0]*hj[2] + m[2]*hj[8] ) );
    jac[6] = -h[1] + m[1]*hj[0] - m[0]*hj[3] + a*( 2*m[0]*h[2] + ( m[0]*m[0] + m[1]*m[1] )*hj[6] - m[2]*( h[0] + m[0]*hj[0] + m[1]*hj[3] ) );
    jac[7] =  h[0] + m[1]*hj[1] - m[0]*hj[4] + a*( 2*m[1]*h[2] + ( m[0]*m[0] + m[1]*m[1] )*hj[7] - m[2]*( h[1] + m[0]*hj[1] + m[1]*hj[4] ) );
    jac[8] =  m[1]*hj[2] - m[0]*hj[5] + a*( -m[0]*h[0] - m[1]*h[1] + ( m[0]*m[0] + m[1]*m[1] )*hj[8] - m[2]*( m[0]*hj[2] + m[1]*hj[5] ) );
}

/// The stochastic diffusion component of the stochastic LLG
/**
 * @param[out] deriv diffusion derivatives [length 3x3]
 * @param[in] state current state of the magnetisation [length 3]
 * @param[in] t time (has no effect)
 * @param[in] sr normalised noise power of the thermal field (see notes
 * on LLG normalisation for details)
 * @param[in] alpha damping ratio
 */
void llg::diffusion( double *deriv, const double *state, const double,
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

/// Jacobian of the stochastic diffusion component of the LLG
/**
 * @param[out] jacobian Jacobian of the LLG diffusion [length 3x3]
 * @param[in]  state current state of the magnetisation vector [length
 * 3]
 * @param[in] t time (has no effect)
 * @param[in] sr normalised noise power of the thermal field (see notes
 * on LLG normalisation for details)
 * @param[in] alpha damping ratio
 */
void llg::diffusion_jacobian( double *jacobian, const double *state,
                              const double,
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

/// Computes drift and diffusion of LLG after updating the field
/**
 * The effective field is first computed based on the applied field
 * and current state of the magnetisation. This is then used to
 * compute the current drift and diffusion components of the LLG.
 * Assumes uniaxial anisotropy.
 * @param[out] drift deterministic component of the LLG [length 3]
 * @param[out] diffusion stochastic component of the LLG [length 3x3]
 * @param[out] heff effective field including the applied field
 * contribution [length 3]
 * @param[in] state current state of the magnetisation [length 3]
 * @param[in] a_t time at which to evaluate the drift
 * @param[in] b_t time at which to evaluate the diffusion
 * @param[in] happ the applied field at time `a_t` [length 3]
 * @param[in] aaxis the anisotropy axis of the particle [length 3]
 * @param[in] alpha damping ratio
 * @param[in] sr normalised noise power of the thermal field (see notes
 * on LLG normalisation for details)
 */
void llg::sde_with_update( double *drift,
                           double *diffusion,
                           double *heff,
                           const double *state,
                           const double a_t,
                           const double b_t,
                           const double *happ,
                           const double *aaxis,
                           const double alpha,
                           const double sr )
{
    // Compute the effective field
    field::uniaxial_anisotropy( heff, state, aaxis );
    heff[0] += happ[0];
    heff[1] += happ[1];
    heff[2] += happ[2];

    // Compute the drift and diffusion
    llg::drift( drift, state, a_t, alpha, heff );
    llg::diffusion( diffusion, state, b_t, sr, alpha );
}

/// Computes effective field, drift, diffusion and Jacobians of LLG
/**
 * The effective field is first computed based on the applied field
 * and current state of the magnetisation. This is then used to
 * compute the drift, diffusion, and their respective Jacobians.
 * Assumes uniaxial anisotropy.
 * @param[out] drift deterministic component of the LLG [length 3]
 * @param[out] diffusion stochastic component of the LLG [length 3x3]
 * @param[out] drift_jac Jacobian of the deterministic component
 * [length 3x3]
 * @param[out] diffusion_jac Jacobian of the diffusion component
 * [length 3x3x3]
 * @param[out] heff effective field including the applied field
 * contribution [length 3]
 * @param[out] heff_jac Jacobian of the effective field [length 3x3]
 * @param[in] state current state of the magnetisation [length 3]
 * @param[in] a_t time at which to evaluate the drift
 * @param[in] b_t time at which to evaluate the diffusion
 * @param[in] happ the applied field at time `a_t` [length 3]
 * @param[in] aaxis the anisotropy axis of the particle [length 3]
 * @param[in] alpha damping ratio
 * @param[in] s normalised noise power of the thermal field (see notes
 * on LLG normalisation for details)
 */
void llg::jacobians_with_update( double *drift,
                                 double *diffusion,
                                 double *drift_jac,
                                 double *diffusion_jac,
                                 double *heff,
                                 double *heff_jac,
                                 const double *state,
                                 const double a_t,
                                 const double b_t,
                                 const double *happ,
                                 const double *aaxis,
                                 const double alpha,
                                 const double s )
{
    llg::sde_with_update( drift, diffusion, heff, state, a_t, b_t,
                          happ, aaxis, alpha, s );
    field::uniaxial_anisotropy_jacobian( heff_jac, aaxis );
    llg::drift_jacobian( drift_jac, state, a_t, alpha, heff, heff_jac );
    llg::diffusion_jacobian( diffusion_jac, state, b_t, s, alpha );
}

/// Deterministic drift component of the stochastic LLG for many particles
/**
 * @param[out] deriv  drift derivative of the deterministic part
 * of the stochastic llg for each particle [length 3xN]
 * @param[in] state current state of the magnetisation vectors [length
 * 3xN]
 * @param[in] t time (has no effect)
 * @param[in] alpha damping ratio
 * @param[in] heff the effective field on each particle [length 3xN]
 * @param[in] N_particles the number of particles
 */
void llg::multi_drift( double *deriv, const double *state,
                       const double *alphas, const double *heff,
                       const size_t N_particles )
{
    for( unsigned int n=0; n<N_particles; n++ )
    {
        unsigned int n3 = 3*n;
        llg::drift(deriv+n3, state+n3, 0, alphas[n], heff+n3 );
    }
}

/// Deterministic drift component of the stochastic Ito LLG for many
/// particles
/**
 * @param[out] deriv  drift derivative of the deterministic part
 * of the stochastic llg for each particle [length 3xN]
 * @param[in] state current state of the magnetisation vectors [length
 * 3xN]
 * @param[in] t time (has no effect)
 * @param[in] alphas damping ratio of each particle
 * @param[in] sigs the thermal noise strength for each particle
 * @param[in] heff the effective field on each particle [length 3xN]
 * @param[in] N_particles the number of particles
 */
void llg::ito_multi_drift( double *deriv, const double *state,
                           const double *alphas, const double *sigs,
                           const double *heff, const size_t N_particles )
{
    for( unsigned int n=0; n<N_particles; n++ )
    {
        unsigned int n3 = 3*n;
        llg::ito_drift(deriv+n3, state+n3, 0, alphas[n], sigs[n], heff+n3 );
    }
}

/// Compute 3x3 block diagonal multi diffusion
/**
 * Note zero terms are not written.
 */
void llg::multi_diffusion( double *deriv, const double *state,
                           const double *field_strengths, const double *alphas,
                           const size_t N_particles )
{
    size_t row_offset, N3 = 3*N_particles;
    double diffusion_work[9];

    // Each 3-row-block
    for( unsigned int bidx=0; bidx<N_particles; bidx++ )
    {
        llg::diffusion(diffusion_work, state+3*bidx,
                       0.0, field_strengths[bidx], alphas[bidx] );

        // Each row in 3-row-block
        for( unsigned int row=0; row<3; row++ )
        {
            row_offset = (bidx*3+row)*N3;
            // Before 3x3 block diagonal
            for( unsigned int col=0; col<bidx*3; col++ )
                deriv[row_offset + col] = 0.0;
            // Inside 3x3 block diagonal
            for( unsigned int col=0; col<3; col++ )
                deriv[row_offset + bidx*3 + col] = diffusion_work[3*row + col];
            // After 3x3 block diagonal
            for( unsigned int col=(bidx+1)*3; col<N3; col++ )
                deriv[row_offset + col] = 0.0;
        }
    }
}

/// Updates field and computes LLG for N interacting particles
/**
 * heff_fuc is a function that returns the effective field given
 * the current state and the current time. This can be whatever you
 * want e.g. cubic anisotropy terms and interactions. EZEEE.
 */
void llg::multi_stochastic_llg_field_update(
    double *drift,
    double *diffusion,
    double *heff,
    const std::function<void(double*,const double*,const double)> heff_func,
    const double *state,
    const double t,
    const double *alphas,
    const double *field_strengths,
    const size_t N_particles )
{
    heff_func( heff, state, t );
    llg::multi_drift(
        drift, state, alphas, heff, N_particles );
    llg::multi_diffusion(
        diffusion, state, field_strengths, alphas, N_particles );
}

/// Updates field and computes Ito LLG for N interacting particles
/**
 * heff_fuc is a function that returns the effective field given
 * the current state and the current time. This can be whatever you
 * want e.g. cubic anisotropy terms and interactions. EZEEE.
 */
void llg::multi_stochastic_ito_llg_field_update(
    double *drift,
    double *diffusion,
    double *heff,
    const std::function<void(double*,const double*,const double)> heff_func,
    const double *state,
    const double t,
    const double *alphas,
    const double *field_strengths,
    const size_t N_particles )
{
    heff_func( heff, state, t );
    llg::ito_multi_drift(
        drift, state, alphas, field_strengths, heff, N_particles );
    llg::multi_diffusion(
        diffusion, state, field_strengths, alphas, N_particles );
}

/// Computes the Jacobian of the drift for N interacting particles
/**
 * Assumes that jac is zero'd (i.e. function will not fill in 0 entries)
 */
void llg::multi_drift_quasijacobian(double *jac, const double *m,
                                    const double *alphas, const double *h,
                                    const double *hj, size_t N_particles )
{
    double jacwork[9];
    // jac is 3xNx3xN = 9*N2
    for( unsigned int n=0; n<N_particles; n++ )
    {
        // Compute the Jacobian for the particle
        llg::drift_jacobian(jacwork, m+(3*n), 0.0, alphas[n], h+(3*n), hj+(3*n) );

        // Place on block diagonal
        jac[3*n+0 + (3*n+0)*3*N_particles] = jacwork[0 + 0*3];
        jac[3*n+1 + (3*n+0)*3*N_particles] = jacwork[1 + 0*3];
        jac[3*n+2 + (3*n+0)*3*N_particles] = jacwork[2 + 0*3];

        jac[3*n+0 + (3*n+1)*3*N_particles] = jacwork[0 + 1*3];
        jac[3*n+1 + (3*n+1)*3*N_particles] = jacwork[1 + 1*3];
        jac[3*n+2 + (3*n+1)*3*N_particles] = jacwork[2 + 1*3];

        jac[3*n+0 + (3*n+2)*3*N_particles] = jacwork[0 + 2*3];
        jac[3*n+1 + (3*n+2)*3*N_particles] = jacwork[1 + 2*3];
        jac[3*n+2 + (3*n+2)*3*N_particles] = jacwork[2 + 2*3];
    }
}

/// Computes the Jacobian of the diffusion for N interacting particles
/**
 * Only computes non-zero values. jacobian must be 0 initialised
 * before function call.
 */
void llg::multi_diffusion_jacobian(
    double *jac, const double *state,
    const double *therm_field_strengths, const double *alphas,
    const size_t N_particles )
{
    double jacwork[27];
    for( unsigned int n=0; n<N_particles; n++ )
    {
        llg::diffusion_jacobian(
            jacwork, state+(3*n), 0.0, therm_field_strengths[n], alphas[n] );

        /// Place on 3dimensional block diagonal
        for( unsigned int x=0; x<3; x++ )
            for( unsigned int y=0; y<3; y++ )
                for( unsigned int z=0; z<3; z++ )
                    jac[3*n+z + (3*n+y)*3*N_particles + (3*n+x)*9*N_particles*N_particles]
                        = jacwork[z + 3*y + 9*x];
    }
}

/// Computes all fields, drift/diffusion, jacobians for N particles
/**
 * The effective field is first computed based on the applied field
 * and current state of the magnetisation. This is then used to
 * compute the drift, diffusion, and their respective Jacobians.
 * Assumes uniaxial anisotropy.
 * @param[out] drift deterministic component of the LLG [length 3]
 * @param[out] diffusion stochastic component of the LLG [length 3x3]
 * @param[out] drift_jac Jacobian of the deterministic component
 * [length 3x3]
 * @param[out] diffusion_jac Jacobian of the diffusion component
 * [length 3x3x3]
 * @param[out] heff effective field including the applied field
 * contribution [length 3]
 * @param[out] heff_jac Jacobian of the effective field [length 3x3]
 * @param[in] state current state of the magnetisation [length 3]
 * @param[in] a_t time at which to evaluate the drift
 * @param[in] b_t time at which to evaluate the diffusion
 * @param[in] happ the applied field at time `a_t` [length 3]
 * @param[in] aaxis the anisotropy axis of the particle [length 3]
 * @param[in] alpha damping ratio
 * @param[in] s normalised noise power of the thermal field (see notes
 * on LLG normalisation for details)
 */
void llg::multi_stochastic_llg_jacobians_field_update(
    double *drift,
    double *diffusion,
    double *drift_jac,
    double *diffusion_jac,
    double *heff,
    double *heff_jac,
    const double *state,
    const double t,
    const double *alphas,
    const double *field_strengths,
    const size_t N_particles,
    const std::function<void(double*,const double*,const double)> heff_func,
    const std::function<void(double*,const double*,const double)> heff_jac_func )
{
    // Update field and compute stochastic llg
    llg::multi_stochastic_llg_field_update(
        drift, diffusion, heff, heff_func, state, t,
        alphas, field_strengths , N_particles);

    // Compute field jacobian for all particles
    heff_jac_func( heff_jac, state, t );

    // Compute the stochastic llg jacobians
    llg::multi_drift_quasijacobian(
        drift_jac, state, alphas, heff, heff_jac, N_particles );
    llg::multi_diffusion_jacobian(
        diffusion_jac, state, field_strengths, alphas, N_particles );
}
