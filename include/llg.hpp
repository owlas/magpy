// llg.hpp
// Landau-Lifshitz-Gilbert stochastic differential equations
// in standard (Stratonovich) and Ito formulations
#ifndef LLG_H
#define LLG_H
namespace llg
{
    void drift( double *deriv, const double *state, const double time,
                const double alpha, const double *heff );

    void drift_jacobian( double *deriv, const double *state, const double time,
                         const double alpha, const double *heff,
                         const double *heff_jac );

    void diffusion( double *deriv, const double *state, const double time,
                    const double sr, const double alpha );

    void diffusion_jacobian( double *jacobian, const double *state,
                             const double time, const double sr,
                             const double alpha );
}
#endif
