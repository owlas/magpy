// llg.hpp
// Landau-Lifshitz-Gilbert stochastic differential equations
// in standard (Stratonovich) and Ito formulations
#ifndef LLG_H
#define LLG_H
#include <cstdlib>
#include <functional>

/**
 * @namespace llg
 * @brief Functions for evaluating the Landau-Lifshitz-Gilbert equation
 * @details Includes the basic equation as well as Jacobians and
 * combined functions to update fields during integration.
 * @author Oliver Laslett
 * @date 2017
 */

namespace llg
{
    void drift(
        double *deriv,
        const double *state,
        const double time,
        const double alpha,
        const double *heff );

    void ito_drift(
        double *deriv,
        const double *state,
        const double time,
        const double alpha,
        const double sig,
        const double *heff );

    void drift_jacobian(
        double *deriv,
        const double *state,
        const double time,
        const double alpha,
        const double *heff,
        const double *heff_jac );

    void diffusion(
        double *deriv,
        const double *state,
        const double time,
        const double sr,
        const double alpha );

    void diffusion_jacobian(
        double *jacobian,
        const double *state,
        const double time,
        const double sr,
        const double alpha );

    /*
      Computes the effective field, llg drift, and llg diffusion
      Non-const are outputs, const are inputs.

      Assumes single particle with uniaxial anisotropy
    */
    void sde_with_update(
        double *drift,
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
    void jacobians_with_update(
        double *drift,
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

    void multi_drift(
        double *deriv,
        const double *state,
        const double *alphas,
        const double *heff,
        const size_t N_particles );

    void ito_multi_drift(
        double *deriv,
        const double *state,
        const double *alphas,
        const double *sigs,
        const double *heff,
        const size_t N_particles );

    void multi_diffusion(
        double *deriv,
        const double *state,
        const double *field_strengths,
        const double *alphas,
        const size_t N_particles );

    void multi_stochastic_llg_field_update(
        double *drift,
        double *diffusion,
        double *heff,
        const std::function<void(double*,const double*,const double)> heff_func,
        const double *state,
        const double t,
        const double *alphas,
        const double *field_strengths,
        const size_t N_particles );

    void multi_stochastic_ito_llg_field_update(
        double *drift,
        double *diffusion,
        double *heff,
        const std::function<void(double*,const double*,const double)> heff_func,
        const double *state,
        const double t,
        const double *alphas,
        const double *field_strengths,
        const size_t N_particles );

    void multi_drift_quasijacobian(
        double *jac,
        const double *m,
        const double *alphas,
        const double *h,
        const double *hj,
        size_t N_particles );

    void multi_diffusion_jacobian(
        double *jacobian,
        const double *state,
        const double *therm_field_strengths,
        const double *alphas,
        const size_t N_particles );

    void multi_stochastic_llg_jacobians_field_update(
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
        const std::function<void(double*,const double*,const double)> heff_jac_func );

}
#endif
