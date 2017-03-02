#include "../include/simulation.hpp"
#include "../include/constants.hpp"
double simulation::power_loss(
    const struct results &res,
    double Ms, double Hk, double f )
{
    double area = trap::trapezoidal( res.field.get(), res.mz.get(), res.N );
    return f*constants::MU0*Hk*Ms*area;
}

void simulation::save_results( const std::string fname, const struct results &res )
{
    std::stringstream magx_fname, magy_fname, magz_fname, field_fname, time_fname;
    magx_fname << fname << ".mx";
    magy_fname << fname << ".my";
    magz_fname << fname << ".mz";
    field_fname << fname << ".field";
    time_fname << fname << ".time";
    int err;
    err = io::write_array( magx_fname.str(), res.mx.get(), res.N );
    if( err != 0 )
        throw std::runtime_error( "failed to write file" );
    err = io::write_array( magy_fname.str(), res.my.get(), res.N );
    if( err != 0 )
        throw std::runtime_error( "failed to write file" );
    err = io::write_array( magz_fname.str(), res.mz.get(), res.N );
    if( err != 0 )
        throw std::runtime_error( "failed to write file" );
    err = io::write_array( field_fname.str(), res.field.get(), res.N );
    if( err != 0 )
        throw std::runtime_error( "failed to write file" );
    err = io::write_array( time_fname.str(), res.time.get(), res.N );
    if( err != 0 )
        throw std::runtime_error( "failed to write file" );
}

void simulation::zero_results( struct simulation::results &res )
{
    for( unsigned int i=0; i<res.N; i++ )
        res.mx[i] = res.my[i] = res.mz[i] = res.field[i] = res.time[i] = 0;
}

struct simulation::results simulation::full_dynamics(
    const double damping,
    const double thermal_field_strength,
    const d3 anis_axis,
    const std::function<double(double)> applied_field,
    const d3 initial_magnetisation,
    const double time_step,
    const double end_time,
    Rng &rng,
    const bool renorm,
    const int max_samples )
{
    size_t dims = 3;

    /*
      To ease memory requirements - can specify the maximum number of
      samples to store in memory. This is used to compute the number
      of integration steps per write to the in-memory results arrays.

      max_samples=-1 is equivalent to max_samples=N_steps

      The sampling interval is taken to be regularly spaced and is interpolated
      from the integration steps using a zero-order-hold technique.
     */
    const size_t N_samples = max_samples==-1 ?
        int( end_time / time_step ) + 1
        : max_samples;
    const double sampling_time = end_time / ( N_samples-1 );

    // allocate memory for results
    simulation::results res( N_samples );


    // Allocate matrices needed for the midpoint method
    double *state = new double[dims*2]; // stores old and new state
                                        // i.e 2 *dims
    double *dwm = new double[dims];
    double *a_work = new double[dims];
    double *b_work = new double[dims*dims];
    double *adash_work = new double[dims*dims];
    double *bdash_work = new double[dims*dims*dims];
    double *x_guess = new double[dims];
    double *x_opt_tmp = new double[dims];
    double *x_opt_jac = new double[dims*dims];
    lapack_int *x_opt_ipiv = new lapack_int[dims];

    // Limits for the implicit solver
    const double eps=1e-9;
    const size_t max_iter=1000;

    // Copy in the initial state
    res.time[0] = 0;
    res.field[0] = applied_field( 0 );
    res.mx[0] = initial_magnetisation[0];
    res.my[0] = initial_magnetisation[1];
    res.mz[0] = initial_magnetisation[2];

    // The wiener paths
    double wiener[3];

    // The effective field and its Jacobian is updated at each time step
    double heff[3];
    double heffjac[9];
    double happ[3];

    // Vars for loops
    unsigned int step = 0;
    double t = 0;
    double hz = 0;
    double pstate[3], nstate[3];
    nstate[0] = initial_magnetisation[0];
    nstate[1] = initial_magnetisation[1];
    nstate[2] = initial_magnetisation[2];

    /*
      The time for each point in the regularly spaced grid is
      known. We want to obtain the state at each step
     */
    for( unsigned int sample=1; sample<N_samples; sample++ )
    {
        // Perform a simulation step until we breach the next sampling point
        while ( t <= sample*sampling_time )
        {
            // take a step
            pstate[0] = nstate[0];
            pstate[1] = nstate[1];
            pstate[2] = nstate[2];
            step++;

            // Compute current time
            // When max_samples=-1 uses sampling_time directly to avoid rounding
            // errors
            t = max_samples==-1 ? sample*sampling_time : step*time_step;

            // Compute the applied field - always in the z-direction
            happ[2] = applied_field( t );

            // Assumes that applied field is constant over the period
            // Bind the parameters to create the required SDE function
            sde_jac sde = std::bind(
                llg::jacobians_with_update, _1, _2, _3, _4, heff, heffjac,
                _5, _6, _7, happ, anis_axis.data(), damping,
                thermal_field_strength );

            // Generate the wiener increments
            for( unsigned int i=0; i<3; i++ )
                wiener[i] = rng.get();

            // perform integration step
            int errcode = integrator::implicit_midpoint(
                nstate, dwm, a_work, b_work, adash_work, bdash_work, x_guess,
                x_opt_tmp, x_opt_jac, x_opt_ipiv, pstate, wiener, sde,
                dims, dims, t, time_step, eps, max_iter );
            if( errcode != optimisation::SUCCESS )
                LOG(FATAL) << "integration error code: " << errcode;

            // Renormalise the length of the magnetisation
            if( renorm  )
            {
                double norm = cblas_dnrm2( 3, nstate, 1 );
                for( unsigned int i=0; i<dims; i++ )
                    nstate[i] = nstate[i]/norm;
            } // end renormalisation

        } // end integration stepping loop

        /*
          Once this point is reached, we are currently one step beyond
          the desired sampling point. Use a zero-order-hold:
          i.e. take the previous state before the sampling time as the
          state at the sampling time.
         */
        res.time[sample] = sample*sampling_time; // sampling time
        res.mx[sample] = pstate[0];
        res.my[sample] = pstate[1];
        res.mz[sample] = pstate[2];
        res.field[sample] = applied_field( sample*sampling_time );
    } // end sampling loop

    delete[] state;
    delete[] a_work;
    delete[] b_work;
    delete[] dwm;
    delete[] adash_work;
    delete[] bdash_work;
    delete[] x_guess;
    delete[] x_opt_tmp;
    delete[] x_opt_jac;
    delete[] x_opt_ipiv;

    return res; // Ensure elison else copy is made and dtor is called!
}

/// Simulates a single particle under the discrete orientation model
/**
 * Simulates a single uniaxial magnetic nanoparticle using the master
 * equation. Assumes that the anisotropy axis is aligned with the
 * external field in the $z$ direction.
 *
 * The system is modelled as having two possible states (up and
 * down). Given the initial probability that the system is in these
 * two states, the master equation simulates the time-evolution of the
 * probability of states. The magnetisation in the $z$ direction is
 * computed as a function of the states.
 * @param[in] damping damping ratio - dimensionless
 * @param[in] anisotropy anisotropy constant  - Kgm-3 ?
 * @param[in] temperature temperature - K
 * @param[in] tau0 reciprocal of the attempt frequency \f$1/f\f$ - s-1
 * @param[in] applied_field a scalar in-out function that returns the
 * value of the applied field in the z-direction at time t
 * @param[in] initial_prbs length 2 array of doubles with the initial
 * probability of each state of the system.
 * @param[in] time_step time_step for the integrator
 * @param[in] end_time total length of the simulation
 * @param[in] max_samples integer number of times to sample the
 * output. Setting to -1 will sample the output at every time step of
 * the integrator.
 * @returns simulation::results struct containing the results of the
 * simulation. The x and y components of the magnetisation are always
 * zero.
 */
struct simulation::results simulation::dom_ensemble_dynamics(
    const double volume,
    const double anisotropy,
    const double temperature,
    const double tau0,
    const std::function<double(double)> applied_field,
    const std::array<double,2> initial_probs,
    const double time_step,
    const double end_time,
    const int max_samples )
{
    // Allocate array for work dims*dims
    const size_t n_dims = 2;
    double work[4];

    // Allocate array for state and copy in initial condition
    double last_state[2], next_state[2];
    next_state[0] = initial_probs[0];
    next_state[1] = initial_probs[1];

    // Allocate arrays for the RK4 integrator
    double k1[2], k2[2], k3[2], k4[2], k5[2], k6[2], tmpstate[2];

    // Construct the time dependent master equation
    std::function<void(double*,const double*,const double)> master_equation =
        std::bind(dom::master_equation_with_update, _1, work, anisotropy,
                  volume, temperature, tau0, _3, _2, applied_field );

    /*
      To ease memory requirements - can specify the maximum number of
      samples to store in memory. This is used to compute the number
      of integration steps per write to the in-memory results arrays.

      max_samples=-1 is equivalent to max_samples=N_steps

      The sampling interval is taken to be regularly spaced and is interpolated
      from the integration steps using a zero-order-hold technique.
    */
    const size_t N_samples = max_samples==-1 ?
        int( end_time / time_step ) + 1
        : max_samples;
    const double sampling_time = end_time / ( N_samples-1 );

    // allocate memory for results and copy in initial state
    simulation::results res( N_samples );
    simulation::zero_results( res );
    res.mz[0] = next_state[0] - next_state[1];

    // Variables needed in the loop
    double t=0;
    double dt = time_step;
    unsigned int step=0;
    double eps=1e-8; // tolerance of the rk45 integrator
    for( unsigned int sample=1; sample<N_samples; sample++ )
    {
        // Perform a simulation step until we breach the next sampling point
        while ( t <= sample*sampling_time )
        {
            // take a step
            last_state[0] = next_state[0];
            last_state[1] = next_state[1];
            step++;

            // perform integration step
            integrator::rk45( next_state, tmpstate, k1, k2, k3, k4, k5, k6,
                              &dt, &t, last_state, master_equation, n_dims,
                              eps );

        } // end integration stepping loop
        /*
          Once this point is reached, we are currently one step beyond
          the desired sampling point. Use a zero-order-hold:
          i.e. take the previous state before the sampling time as the
          state at the sampling time.
        */
        /**
           The magnetisation is computed as the magnetisation of
           each state multiplied by it's probability.
           \f$M(t) = M_1p_1(t) + M_2p_2(t)\f$
           For a uniaxial particle we take \f$M_1=1,M_2=-1\f$
           Thus the magnetisation is the difference between the two
           state probabilities.
        */
        res.mz[sample] = next_state[0] - next_state[1];
        res.time[sample] = t;
        res.field[sample] = applied_field(t);
    }
    return res;
}
