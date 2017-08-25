/**
 * @namespace field
 * @brief Contains functions for computing magnetic fields.
 * @details Functions for computing effective fields from anistoropy
 * and a range of time-varying applied fields.
 * @author Oliver Laslett
 * @date 2017
 */
#include "../include/field.hpp"
#include "../include/constants.hpp"
#define _USE_MATH_DEFINES
#include<cmath>
#include<functional>

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

/// Add the applied field in the z direction
void field::multi_add_applied_Z_field_function(
    double *heff, const std::function<double(const double)> &hfunc,
    const double t, const size_t N )
{
    double happ = hfunc( t );
    for( unsigned int i=0; i<N; i++ )
        heff[i*3+2] += happ;
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

/// Add the uniaxial anisotropy term term to the field of N particles
/**
 * @param[out] h effective field incremented with the anisotropy term
 * @param[in] states magnetic state of the particles (length 3N)
 * @param[in] axes anisotropy axis of each particle (length 3N)
 * @param[in] k_reduced the reduced anisotropy constant for each particle (length N)
 * @param[in] N number of particles
 */
void field::multi_add_uniaxial_anisotropy(
    double *h, const double *states,
    const double *axes, const double *k_reduced, const size_t N )
{
    for( unsigned int n=0; n<N; n++ )
    {
        double k_red = k_reduced[n];
        const double *axis = axes+3*n;
        const double *state = states+3*n;

        double dot = state[0]*axis[0]+state[1]*axis[1]+state[2]*axis[2];
        dot *= k_red;
        h[3*n+0] += dot*axis[0];
        h[3*n+1] += dot*axis[1];
        h[3*n+2] += dot*axis[2];
    }
}

/// Jacobian of the uniaxial anisotropy effective field term
/**
 * The Jacobian of a particle's uniaxial anisotropy with respect to
 * it's magnetisation value.
 * \f$J_h(m) = \frac{\partial h(m)}{\partial m}\f$
 * @param[out] jac the jacobian of the effective field [length 3x3]
 * @param[in]  axis the anisotropy axis of the particle [length 3]
 */
void field::uniaxial_anisotropy_jacobian(
    double *jac, const double *axis )
{
    for( unsigned int i=0; i<3; i++ )
        for( unsigned int j=0; j<3; j++ )
            jac[i*3+j] = axis[i]*axis[j];
}

/// Jacobian of the uniaxial anisotropy effective field term for many particles
/**
 * The Jacobian of a particle's uniaxial anisotropy with respect to
 * it's magnetisation value.
 * \f$J_h(m) = \frac{\partial h(m)}{\partial m}\f$
 * @param[out] jac the jacobian of the effective field [length 3Nx3N]
 * @param[in]  axis the anisotropy axis of the particle [length 3N]
 * @param[in] N number of particles
 */
void field::multi_add_uniaxial_anisotropy_jacobian( double *jac, const double *axes,
                                                    const double *k_reduced, const size_t N )
{
    for( unsigned int i=0; i<3*N; i++ )
        for( unsigned int j=0; j<3*N; j++ )
            if( (i/3) == (j/3) )
                jac[i*3*N + j] = k_reduced[i/3] * axes[i] * axes[j];

    // for( unsigned int i=0; i<N; i++ )
    // {
    //     // place on diagonal
    //     for( unsigned int x=0; x<3; x++ )
    //         for( unsigned int y=0; y<3; y++ )
    //             jac[(3*i+x)*3*N + 3*i+y] += k_reduced[i] * axes[3*i+x] * axes[3*i+y];
    // }
}

/**
 * field is Nx3 and is the effective field on each particle
 * ms is the same for all particles
 * k_av is the average anisotropy constant for all particles
 * v_reduced is the reduced volume of each particle
 * v_av is the average volume for all particles
 * mag is Nx3 long and is the magnetic state of each of the N particles
 * dists is NxNx3 long and is the distance between each pair of particles
 * N is the number of particles
 * dist_cubes is NxN and is the reduced cubed distance between each pair
 */
void field::multi_add_dipolar(double *field, const double ms, const double k_av,
                              const double *v_reduced, const double *mag,
                              const double *dists, const double *dist_cubes,
                              const size_t N  )
{
    double prefactor = field::dipolar_prefactor( ms, k_av );
    for( unsigned int i=0; i<N; i++ )
    {
        for( unsigned int j=0; j<i; j++ )
            field::dipolar_add_p2p_term( field + i*3, v_reduced[j], dist_cubes[i*N + j],
                                         mag + j*3, dists + i*N*3 + j*3, prefactor );
        for( unsigned int j=i+1; j<N; j++ )
            field::dipolar_add_p2p_term( field + i*3, v_reduced[j], dist_cubes[i*N + j],
                                         mag + j*3, dists + i*N*3 + j*3, prefactor );
    }

}

/// Prefactor term (helper for field::dipolar)
/**
 * prefactor is \f$\frac{\mu0 M_s^2}{8\pi\bar{K}}\f$
 * @param[in] ms saturation magnetisation
 * @param[in] k_av average anisotropy constant for system
 * @returns prefactor term
 */
double field::dipolar_prefactor( const double ms, const double k_av )
{
    return constants::MU0 * ms * ms / 8.0 / M_PI / k_av;
}

void field::dipolar_add_p2p_term( double *out, const double vj, const double rij3,
                                  const double *mj, const double *dist,
                                  const double prefactor )
{
    double term_dotprod = mj[0]*dist[0] + mj[1]*dist[1] + mj[2]*dist[2];
    double term_1 = vj / rij3;
    for( unsigned int i=0; i<3; i++ )
        out[i] += prefactor * term_1 * ( 3 * term_dotprod * dist[i] - mj[i] );
}

/// Set all N values of the field to zero
/**
 * @param[out] h pointer to array of N doubles will all be set to zero
 * @param[in] N number of elements in array
 */
void field::zero_all_field_terms( double *h, const size_t N)
{
    for( unsigned int i=0; i<N; i++ )
        h[i] = 0.0;
}

/// Binds amplitude and frequency to the sinusoidal function
/**
 * Used by python interface
 */
std::function<double(double)> field_py::sinusoidal( const double h, const double f )
{
    return [h, f](double t) { return field::sinusoidal(h, f, t ); };
}

/// Binds amplitude to the constant field function
/**
 * Used by python interface
 */
std::function<double(double)> field_py::constant( const double h )
{
    return [h](double t) { return field::constant(h, t ); };
}

/// Binds amplitude and frequency to the square function
/**
 * Used by pyhton interface
 */
std::function<double(double)> field_py::square( const double h, const double f )
{
    return [h, f](double t) { return field::square(h, f, t ); };
}
