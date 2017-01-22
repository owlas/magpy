// field.hpp
//
// functions to compute time-varying applied field waveforms
// Oliver W. Laslett (2016)
// O.Laslett@soton.ac.uk
#ifndef FIELD_H
#define FIELD_H
#include <cstddef>

namespace field
{
    // constant field term
    double constant( const double h, const double t);

    // field strength h, frequency f, at time t
    double sinusoidal( const double h, const double f, const double t );
    double square( const double h, const double f, const double t );

    // Use this to approximate a square wave with its fourier components
    double square_fourier( const double h, const double f,
                           const size_t n_compononents, double t );

    // Computes the anisotropy field
    void uniaxial_anisotropy( double *h_anis, const double *magnetisation,
                              const double *anis_axis );

    /*
      Compute the Jacobian of the uniaxial anisotropy field with respect
      to the magnetisation.
      >> J = ∂h(m)/∂m
      axis: IN is the 3 dimensional anisotropy axis
      jac: OUT is the 3x3 dimensional Jacobian
    */
    void uniaxial_anisotropy_jacobian( double *jac, const double *axis );

}
#endif
