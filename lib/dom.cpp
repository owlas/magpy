#include "../include/dom.hpp"
#include "../include/constants.hpp"
#include "../include/stochastic_processes.hpp"
#include <cmath>

/// Compute the 2x2 transition matrix for a single particle
/**
 * Assumes uniaxial anisotropy and an applied field h<1
 * Only valid for large energy barriers: $\sigma(1-h)^2>>1$
 * @param[out] W transition matrix [2x2]
 * @param[in]  k anisotropy strength constant for the uniaxial
 * anisotropy
 * @param[in]  v volume of the particle in meter^3
 * @param[in]  T temperature of environment in Kelvin
 * @param[in]  h dimensionless applied field - normalised by
 *               \f$H_k=\frac{2K}{\mu_0M_s}\f$
 * @param[in]  tau0 \f$\tau_0=1/f_0\f$ where f0 is the attempt
 * frequency - this constant is often chosen as
 * \f$10^{-10}-10^{-12}\f$
 */
void dom::transition_matrix(
    double *W,
    const double k,
    const double v,
    const double T,
    const double h,
    const double tau0)
{
    double sigma = k*v/constants::KB/T;
    double norm_ebar_1 = sigma*( 1-h )*( 1-h );
    double norm_ebar_2 = sigma*( 1+h )*( 1+h );
    double rate1 = 1/( tau0*std::exp( norm_ebar_1 ) );
    double rate2 = 1/( tau0*std::exp( norm_ebar_2 ) );
    W[0] = -rate2; W[1] =  rate1;
    W[2] =  rate2; W[3] = -rate1;
}

/// Computes master equation for particle in time-dependent field
/**
 * Computes the applied field in the z-direction at time t.
 * Uses the field value to compute the transition matrix and
 * corresponding master equation derivatives for the single particle.
 * @param[out] derivs master equation derivatives [length 2]
 * @param[out] work vector [length 4]
 * @param[in]  k anisotropy strength constant for the uniaxial
 * anisotropy
 * @param[in]  v volume of the particle in meter^3
 * @param[in]  T temperature of environment in Kelvin
 * @param[in]  tau0 \f$\tau_0=1/f_0\f$ where f0 is the attempt
 * @param[in]  t time at which to evaluate the external field
 * @param[in]  state_probabilities the current state probabilities for
 * each of the 2 states (up and down) [length 2]
 * @param[in]  applied_field a scalar function takes a double and
 * returns a double. Given the current time should return normalised
 * field \f$h=(t)\f$ where the field is normalised by \f$H_k\f$
 */
void dom::master_equation_with_update(
    double *derivs,
    double *work,
    const double k,
    const double v,
    const double T,
    const double tau0,
    const double t,
    const double *state_probabilities,
    const std::function<double(double)> applied_field )
{
    dom::transition_matrix(
        work, k, v, T, applied_field( t ), tau0 );
    stochastic::master_equation(
        derivs, work, state_probabilities, 2 );
}
