/**
 * @namespace field
 * @brief Contains functions for computing magnetic fields.
 * @details Functions for computing effective fields from anistoropy
 * and a range of time-varying applied fields.
 * @author Oliver Laslett
 * @date 2017
 */
#include "../include/field.hpp"
#define _USE_MATH_DEFINES
#include<cmath>

/// A constant applied field.
/**
 * A simple placeholder function representing a constant
 * field. Always returns the same value.
 * @param[in] h applied field amplitude
 * @param[in] t time (parameter has no effect)
 * @returns the constant field amplitude at all values of time
 */
double field::constant( const double h, const double )
{
    return h;
}

/// A sinusoidal alternating applied field
/**
 * Returns the value of a sinusoidally varying field at any given
 * time.
 * @param[in] h applied field amplitude
 * @param[in] f applied field frequency
 * @param[in] t time
 * @returns the value of the varying applied field at time `t`
 */
double field::sinusoidal( const double h, const double f, const double t )
{
    return h*std::sin( 2*M_PI*f*t );
}

/// A square wave switching applied field
/**
 * An alternating applied field with a square shape centred around
 * zero. i.e. it alternates between values `-h` and `h`
 * @param[in] h applied field amplitude
 * @param[in] f applied field frequency
 * @param[in] t time
 * @returns the value of the square wave applied field at time `t`
 */
double field::square( const double h, const double f, const double t )
{
    return h*( int( t*f*2 )%2 ? -1 : 1 );
}

/// A square wave applied field of finite Fourier components.
/**
 * An approximate square wave is computed from a finite number of
 * Fourier components. The square wave alternates between `-h` and
 * `h`.
 * @param[in] h            applied field amplitude
 * @param[in] f            applied field frequency
 * @param[in] n_components number of Fourier components to compute
 * @param[in] t            time
 * @returns the value of the square wave applied field at time `t`
 */
double field::square_fourier( const double h,
                              const double f,
                              const size_t n_components,
                              double t )
{
    double field=0;
    for( unsigned int k=1; k<n_components+1; k++ )
        field += std::sin( 2*M_PI*( 2*k - 1 )*f*t ) / ( 2*k-1 );
    field *= 4/M_PI * h;
    return field;
}

/// Effective field contribution from uniaxial anisotropy
/**
 * The effective field experienced by a single particle with a
 * uniaxial anisotropy.
 * @param[out] h_anis effective field [length 3]
 * @param[in]  mag    the magnetisation of the particle of [length 3]
 * @param[in]  axis   the anisotropy axis of the particle [length 3]
 */
void field::uniaxial_anisotropy(
    double *h_anis, const double *mag, const double *axis )
{
    // compute the dot product
    double dot = mag[0]*axis[0]+mag[1]*axis[1]+mag[2]*axis[2];
    h_anis[0] = dot*axis[0];
    h_anis[1] = dot*axis[1];
    h_anis[2] = dot*axis[2];
}

/// Jacobian of the uniaxial anisotropy effective field term
/**
 * The Jacobian of a particle's uniaxial anisotropy with respect to
 * it's magnetisation value.
 * \f$J_h(m) = \frac{\partial h(m)}{\partial m}\f$
 * @param[out] jac the jacobian of the effective field [length 3*3]
 * @param[in]  axis the anisotropy axis of the particle [length 3]
 */
void field::uniaxial_anisotropy_jacobian(
    double *jac, const double *axis )
{
    for( unsigned int i=0; i<3; i++ )
        for( unsigned int j=0; j<3; j++ )
            jac[i*3+j] = axis[i]*axis[j];
}
