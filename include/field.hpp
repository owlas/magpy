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
    /// Bind field parameters into a field function
    /**
     * Field functions are of the form `std::function(double<double>)`
     * This function allows additional arguments to be bound to functions
     * and returns the function as a field function.
     *
     * @param[in] func function to bind
     * @param[in] bind_args arguments to bind
     * @returns field function
     */
    template <typename... T>
    std::function<double(double)> bind_field_function(
        std::function<double(double, T...)> func, T... bind_args)
    {
        std::function<double(double)> field_function =
            [func, bind_args...](double t){ return func( t, bind_args...); };
        return field_function;
    }


    // Time-varying externally applied field shapes
    double constant( const double t, const double h);
    double sinusoidal( const double t, const double h, const double f);
    double square( const double t, const double h, const double f );
    double square_fourier( const double t, const double h, const double f,
                           const size_t n_compononents );

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

    enum options {
        SINE, SQUARE, CONSTANT
    };
}
#endif
