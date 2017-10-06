#include "../include/optimisation.hpp"
#include <cmath>
#ifdef USEMKL
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif


/// Newton-Raphson method for scalar root finding
/**
 * Find a root \f$x\f$ of a scalar function such that
 * \f$f(x)=0\f$. Starting with an initial guess at \p x0 the method
 * attempts to find the nearest root. Requires function
 * gradient. Convergence criteria is the abs error between two
 * sucessive steps.
 * @param[out] x_root estimate of the root ofter final iteration
 * @param[in] f scalar function \f$f(.)\f$
 * @param[in] fdash the gradient of the function \f$\partial f /
 * \partial x\f$
 * @param[in] x0 initial guess of the root
 * @param[in] eps error tolerance for the iterative solution
 * @param[in] max_iter maximum allowed number of iterations
 * @returns SUCCESS on success or MAX_ITERATIONS_ERR if \p max_iter is
 * reached
 */
int optimisation::newton_raphson_1(
    double *x_root,
    const std::function<double(const double )> f,
    const std::function<double(const double )> fdash,
    const double x0,
    const double eps,
    const size_t max_iter )
{
    // Initialise the error (ensure at least one iteration)
    double err = 2*eps;
    double x1, x2=x0;

    // Repeat until convergence
    int iter=max_iter;
    while( ( err > eps ) && ( iter --> 0 ) )
    {
        x1 = x2;
        x2 = x1 - f( x1 )/fdash( x1 );
        err = std::abs( x2 - x1 );
    }
    *x_root=x2;
    return iter==-1 ? optimisation::MAX_ITERATIONS_ERR: optimisation::SUCCESS;
}

/// Newton-Raphson method for vector valued functions
/**
 * Newton-Raphson method iteratively computes the root \f$\vec{x}\f$
 * of the vector valued function \f$f(\vec{x})\f$, starting from an
 * initial guess at \f$\vec{x}_0\f$. Requires the Jacobian of the
 * function \f$J(\vec{x})=\partial f / \partial \vec{x}$. Avoids
 * the matrix inversion: \f$\vec{x}_n=x-J^{-1}(\vec{x})/f(\vec{x})\f$
 *  by solving a linear system of equations
 * \f$J(\vec{x})(\vec{x}_n-\vec{x}) = -f(\vec{x})\f$
 * using LAPACK DGESV.
 * @param[out] final estimate of the root (length \p dim)
 * @param[out] allocated memory needed for work (length \p dim)
 * @param[out] jac_out memory needed for work (length \p dim * \p dim)
 * @param[out] ipiv memory needed for work (length \p dim)
 * @param[out] lapack_err_code error code from the DGESV
 * solver. There is no requirement to check this code unless the
 * function returns the LAPACK_ERR flag. 0=ok -i=ith value is illegal,
 * i=ith value is exactly 0 (i.e. singular)
 * @param[in] fj std::function that computes the function value
 * \f$f(\vec{x})\f$ and the Jacobian \f$J(\vec{x})\f$ given the
 * current state \f$\vec{x}\f$
 * @param[in] x0 initial guess for the root
 * @param[in] dim dimension of the function input \f$\vec{x}\f$
 * @param[in] eps tolerance on the solution error. Error is computed
 * as the 2-norm of the residual vector \f$\vec{x}_n - \vec{x}_{n-1}\f$
 * @param[in] maximum allowable number of iterations
 * @returns SUCCESS if success, MAX_ITERATIONS_ERR if \p max_iter
 * reached, LAPACK_ERR if error with DGESV (check \p lapack_err_code
 * for more info)
 */
int optimisation::newton_raphson_noinv (
    double *x_root,
    double *x_tmp,
    double *jac_out,
    lapack_int *ipiv,
    int *lapack_err_code,
    const std::function<void(double*,double*,const double* )> fj,
    const double *x0,
    const lapack_int dim,
    const double eps,
    const size_t max_iter )
{
    // Copy in the initial condition
    for( int i=0; i<dim; i++ )
        x_root[i] = x0[i];

    // Relative tolerance
    double tol = eps * cblas_dnrm2( dim, x_root, 1 );

    // Initialise the error
    double err=2*tol;

    // Repeat until convergence
    int iter = max_iter;
    while( ( err > tol ) && ( iter --> 0 ) )
    {
        // Get the previous state
        for( int i=0; i<dim; i++ )
            x_tmp[i] = x_root[i];

        fj( x_root, jac_out, x_tmp );

        // Arrange into form J(xprev)(xnext-xprev)=-F(xprev)
        for( int i=0; i<dim; i++ )
            x_root[i] *= -1;

        /*
          Solve for the next state
          Use LAPACKE_dgesv to solve Ax=B for x
          lapack_int dgesv( matrix_order, N, NRHS, A, LDA, IPIV, B, LDB )
          order - LAPACK_ROW_MAJOR or LAPACK_COL_MAJOR
              N - number of equations
           NRHS - number of right hand sides (columns of B)
              A - NxN left matrix
            LDA - leading dimension of A (i.e length of first dimension)
           IPIV - [out] the pivot indices
              B - [in] NxNHRS input B [out] the NxNHRS solution for x
            LDB - leading dimension of array B

          suffix _work indicates user will alloc required memory for the routine
          but in this case no additional allocations are required
          Returns error flag - 0=ok -i=ith val is illegal i=i is exactly 0 (singular)
        */
        *lapack_err_code = LAPACKE_dgesv_work(
            LAPACK_ROW_MAJOR, dim, 1, jac_out, dim, ipiv, x_root, 1 );
        if( *lapack_err_code!=0 )
            return optimisation::LAPACK_ERR;


        // The returned value is in x_root and is (xnext-xprev)
        // The error is the 2-norm of (xnext-xprev)
        err = cblas_dnrm2( dim, x_root, 1 );

        // Add xprev to solution to get the actual result
        for( int i=0; i<dim; i++ )
            x_root[i] += x_tmp[i];
    }
    return iter==-1 ? optimisation::MAX_ITERATIONS_ERR : optimisation::SUCCESS;
}
