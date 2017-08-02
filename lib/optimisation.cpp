// optimisation.cpp
//
// Oliver W. Laslett (2016)
// O.Laslett@soton.ac.uk
#include "../include/optimisation.hpp"
#include <cmath>
#ifdef USEMKL
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif

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
