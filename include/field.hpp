// field.hpp
//
// functions to compute time-varying applied field waveforms
// Oliver W. Laslett (2016)
// O.Laslett@soton.ac.uk
#ifndef FIELD_H
#define FIELD_H
#include <cstddef>
#include <functional>

namespace field
{
    /// Constant field term
    double constant( const double h, const double t);

    // field strength h, frequency f, at time t
    double sinusoidal( const double h, const double f, const double t );
    double square( const double h, const double f, const double t );

    // Use this to approximate a square wave with its fourier components
    double square_fourier( const double h, const double f,
                           const size_t n_compononents, double t );

    void multi_add_applied_Z_field_function(
        double *heff,
        const std::function<double(const double)> &hfunc,
        const double t,
        const size_t N );

    // Computes the anisotropy field
    void uniaxial_anisotropy( double *h_anis, const double *magnetisation,
                              const double *anis_axis );

    // Computes the anisotropy field for multiple particles
    void multi_add_uniaxial_anisotropy(
        double *h,
        const double *states,
        const double *axes,
        const double *k_reduced,
        const size_t N );

    void uniaxial_anisotropy_jacobian(
        double *jac,
        const double *axis );

    void multi_add_uniaxial_anisotropy_jacobian(
        double *jac,
        const double *axes,
        const double *anis,
        const size_t N );


    void multi_add_dipolar(
        double *field,
        const double ms,
        const double k_av,
        const double *v_reduced,
        const double *mag,
        const double *dists,
        const double *dist_cubes,
        const size_t N );

    double dipolar_prefactor(
        const double ms,
        const double k_av );

    void dipolar_add_p2p_term(
        double *out,
        const double vj,
        const double rij3,
        const double *mj,
        const double *dist,
        const double prefactor );

    void zero_all_field_terms(
        double *h,
        const size_t N);

}
#endif
