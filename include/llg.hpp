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

    /*
      Computes the effective field, llg drift, and llg diffusion
      Non-const are outputs, const are inputs.

      Assumes single particle with uniaxial anisotropy
    */
    void sde_with_update( double *drift,
                          double *diffusion,
                          double *heff,
                          const double *current_state,
                          const double drift_time,
                          const double diffusion_time,
                          const double *happ,
                          const double *anisotropy_axis,
                          const double alpha,
                          const double noise_power );

    /*
      Computes the effective field, llg drift, llg diffusion and their
      respective derivatives.

      Assumes a single particle with uniaxial anisotropy
    */
    void jacobians_with_update( double *drift,
                                double *diffusion,
                                double *drift_jac,
                                double *diffusion_jac,
                                double *heff,
                                double *heff_jac,
                                const double *current_state,
                                const double drift_time,
                                const double diffusion_time,
                                const double *happ,
                                const double *anisotropy_axis,
                                const double alpha,
                                const double noise_power );
}
#endif
