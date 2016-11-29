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
    // field strength h, frequency f, at time t
    double sinusoidal( const double h, const double f, const double t );
    double square( const double h, const double f, const double t );

    // Use this to approximate a square wave with its fourier components
    double square_fourier( const double h, const double f,
                           const size_t n_compononents, double t );
}
#endif
