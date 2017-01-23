#ifndef DOM_H
#define DOM_H
#include <functional>

/**
 * @namespace dom
 * @brief discrete orientation model for magnetic particles
 */
namespace dom
{
    void transition_matrix(
        double *W,
        const double k, const double v, const double T, const double h,
        const double tau0 );

    void master_equation_with_update(
        double *derivs, double *work,
        const double k, const double v, const double T,
        const double tau0, const double t, const double *state_probabilities,
        const std::function<double(double)> applied_field );
}
#endif
