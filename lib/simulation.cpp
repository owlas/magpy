#include "../include/simulation.hpp"
#include "../include/constants.hpp"
#include <cmath>
#include "../include/field.hpp"

double simulation::energy_loss(
    const struct results &res,
    const double ms,
    const double hk )
{
    double area = trap::trapezoidal( res.mz.get(), res.field.get(), res.N );
    return -constants::MU0*ms*hk*area;
}

double simulation::energy_loss(
    const std::unique_ptr<double[]> &transition_energy,
    const std::unique_ptr<double[]> &probability_flow,
    const std::unique_ptr<double[]> &time,
    const double volume,
    const size_t N )
{
    double *mult = new double[N];
    for( unsigned int i; i<N; i++ )
        mult[i] = transition_energy[i] * probability_flow[i];
    delete[] mult;
    return trap::trapezoidal( mult, time.get(), N ) / volume;
}

double simulation::one_step_energy_loss(
    const double et1, const  double et2, const double pflow1, const double pflow2,
    const double t1, const double t2, const double volume
    )
{
    return trap::one_trapezoid( t1, t2, et1*pflow1, et2*pflow2 ) / volume;
}

void simulation::reduce_to_system_magnetisation(
    double *mag, const double *particle_mags, const size_t N_particles )
{
    mag[0] = mag[1] = mag[2] = 0;
    for( unsigned int n=0; n<N_particles; n++ )
    {
        mag[0] += particle_mags[3*n];
        mag[1] += particle_mags[3*n + 1];
        mag[2] += particle_mags[3*n + 2];
    }
}

void simulation::save_results( const std::string fname, const struct results &res )
{
    std::stringstream magx_fname, magy_fname, magz_fname, field_fname, time_fname, energy_fname;
    magx_fname << fname << ".mx";
    magy_fname << fname << ".my";
    magz_fname << fname << ".mz";
    field_fname << fname << ".field";
    time_fname << fname << ".time";
    energy_fname << fname << ".energy";

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
    err = io::write_array( energy_fname.str(), &res.energy_loss, 1 );
}

void simulation::zero_results( struct simulation::results &res )
{
    for( unsigned int i=0; i<res.N; i++ )
        res.mx[i] = res.my[i] = res.mz[i] = res.field[i] = res.time[i] = 0;
    res.energy_loss = 0;
}

// Run a simulation of the full dynamics of an ensemble of particles
std::vector<struct simulation::results> simulation::full_dynamics(
    const std::vector<double> thermal_field_strengths,
    const std::vector<double> reduced_anisotropy_constants,
    const std::vector<double> reduced_particle_volumes,
    const std::vector<d3> anisotropy_unit_axes,
    const std::vector<d3> initial_magnetisations,
    const std::vector<std::vector<d3> > interparticle_unit_distances,
    const std::vector<std::vector<double> > interparticle_reduced_distance_magnitudes,
    const std::function<double(const double)> applied_field,
    const double average_anisotropy,
    const double average_volume,
    const double damping_constant,
    const double saturation_magnetisation,
    const double time_step,
    const double end_time,
    Rng &rng,
    const bool renorm,
    const int max_samples )
{
    // Dimensions
    constexpr size_t dims = 3;
    const size_t n_particles = initial_magnetisations.size();
    const size_t state_size = dims*n_particles;
    const size_t state_square_size = state_size * state_size;
    const size_t state_cube_size = state_square_size * state_size;

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
    std::vector<simulation::results> results;
    for( unsigned int i=0; i<n_particles; i++ )
        results.push_back( simulation::results( N_samples ) );

    // Allocate matrices needed for the midpoint method
    double *state = new double[state_size];

    double *dwm = new double[state_size];
    double *a_work = new double[state_size];
    for( size_t i=0; i<state_size; i++ ) a_work[i] = 0.0;

    double *b_work = new double[state_square_size];
    for( size_t i=0; i<state_square_size; i++ ) b_work[i] = 0.0;

    double *adash_work = new double[state_square_size];
    for( size_t i=0; i<state_square_size; i++ ) adash_work[i] = 0.0;

    double *bdash_work = new double[state_cube_size];
    for( size_t i=0; i<state_cube_size; i++ ) bdash_work[i] = 0.0;

    double *x_guess = new double[state_size];
    double *x_opt_tmp = new double[state_size];
    double *x_opt_jac = new double[state_square_size];
    lapack_int *x_opt_ipiv = new lapack_int[state_size];

    /// @TODO optimise the implicit solver tolerance
    // Limits for the implicit solver
    const double eps=1e-9;
    const size_t max_iter=1000;

    // Copy in the initial state
    for( auto &res : results )
    {
        res.time[0] = 0;
        res.field[0] = applied_field( 0 );
    }

    // The wiener paths
    double *wiener = new double[state_size];

    // The effective field and its Jacobian is updated at each time step
    double *heff = new double[state_size];
    double *heffjac = new double[state_square_size];

    // The anisotropy axes
    double *anis = new double [state_size];
    for( unsigned int n=0; n<n_particles; n++ )
        for( unsigned int i=0; i<dims; i++ )
            anis[i+n*dims] = anisotropy_unit_axes[n][i];

    // The damping ratios
    std::vector<double> damping_ratios( n_particles );
    for( auto &a : damping_ratios )
        a = damping_constant;

    // Vars for loops
    unsigned int step = 0;
    double t = 0;
    double *pstate = new double[state_size];
    double *nstate = new double[state_size];

    /// Initialise the system state from the initial mags vector
    for( unsigned int n=0; n<n_particles; n++ )
        for( unsigned int i=0; i<dims; i++ )
            nstate[i+dims*n] = initial_magnetisations[n][i];


    // Get the unit distances
    double *distances = new double[n_particles*n_particles*3];
    for( unsigned int i=0; i<n_particles; i++ )
        for( unsigned int j=0; j<n_particles; j++ )
            for( unsigned int k=0; k<3; k++ )
                distances[i*n_particles*3 + j*3 + k] = interparticle_unit_distances[i][j][k];

    // Get the cubed distance magnitudes
    double *cubed_distance_magnitudes = new double[n_particles*n_particles];
    for( unsigned int i=0; i<n_particles; i++ )
        for( unsigned int j=0; j<n_particles; j++ )
            cubed_distance_magnitudes[i*n_particles + j] = std::pow(
                interparticle_reduced_distance_magnitudes[i][j], 3 );

    /// Craft the effective field function
    /**
     * Use uniaxial anisotropy for each particle
     * The zeeman term (i.e applied field function)
     * And finally dipole-dipole interactions
     */
    std::function<void(double*,const double*,const double)> heff_func =
        [state_size, anis, reduced_anisotropy_constants, n_particles,
         applied_field, saturation_magnetisation, average_anisotropy, average_volume,
         distances, reduced_particle_volumes, cubed_distance_magnitudes]
        ( double *heff, const double *state, const double t )
        {
            field::zero_all_field_terms( heff, state_size );

            field::multi_add_uniaxial_anisotropy(
                heff, state, anis, reduced_anisotropy_constants.data(), n_particles );

            field::multi_add_applied_Z_field_function( heff, applied_field,  t, n_particles );

            field::multi_add_dipolar(
                heff, saturation_magnetisation, average_anisotropy,
                reduced_particle_volumes.data(), state,
                distances, cubed_distance_magnitudes, n_particles );

        };

    /// Craft the jacobian of the field function
    /**
     * Here we ignore interactions
     */
    std::function<void(double*,const double*,const double)> heff_jac_func =
        [state_square_size, anis, n_particles, reduced_anisotropy_constants]
        ( double *heff_jac, const double *, const double )
        {
            field::zero_all_field_terms( heff_jac, state_square_size );
            field::multi_add_uniaxial_anisotropy_jacobian(
                heff_jac, anis, reduced_anisotropy_constants.data(), n_particles );
        };

    /// Craft the stochastic differential equation interface for the integrator
    sde_jac sde = [heff, heffjac, damping_ratios, thermal_field_strengths,
                   n_particles, heff_func, heff_jac_func]
        ( double *drift, double *diffusion, double *jdrift,
          double *jdiffusion, const double *state,
          const double a_t,
          const double // b_t unused
            )
        {
            llg::multi_stochastic_llg_jacobians_field_update(
                drift, diffusion,
                jdrift, jdiffusion,
                heff, heffjac,
                state,
                a_t,
                damping_ratios.data(),
                thermal_field_strengths.data(),
                n_particles,
                heff_func,
                heff_jac_func );
        };

    /*
      The time for each point in the regularly spaced grid is
      known. We want to obtain the state at each step
     */
    for( unsigned int sample=1; sample<N_samples; sample++ )
    {
        // Perform a simulation step until we breach the next sampling point
        while ( t <= sample*sampling_time )
        {
            // Copy in the previous state
            for( unsigned int i=0; i<state_size; i++ )
                pstate[i] = nstate[i];
            step++; // Take a step

            // Compute current time
            // When max_samples=-1 uses sampling_time directly to avoid rounding
            // errors
            t = max_samples==-1 ? sample*sampling_time : step*time_step;

            // Generate the wiener increments
            for( unsigned int i=0; i<state_size; i++ )
                wiener[i] = rng.get();

            // perform integration step
            int errcode = integrator::implicit_midpoint(
                nstate, dwm, a_work, b_work, adash_work, bdash_work, x_guess,
                x_opt_tmp, x_opt_jac, x_opt_ipiv, pstate, wiener, sde,
                state_size, state_size, t, time_step, eps, max_iter );
            if( errcode != optimisation::SUCCESS )
                LOG(FATAL) << "integration error code: " << errcode;

            // Renormalise the length of the magnetisation for each particle
            if( renorm  )
            {
                for( unsigned int offset=0; offset<state_size; offset+=dims )
                {
                    double norm = cblas_dnrm2( dims, nstate+offset, 1 );
                    for( unsigned int i=0; i<dims; i++ )
                        nstate[offset+i] = nstate[offset+i]/norm;
                }
            } // end renormalisation

        } // end integration stepping loop

        /*
          Once this point is reached, we are currently one step beyond
          the desired sampling point. Use a zero-order-hold:
          i.e. take the previous state before the sampling time as the
          state at the sampling time.
         */
        /// @TODO implement first-order hold for sampling
        for( size_t i=0; i<results.size(); i++ )
        {
            results[i].time[sample] = sample*sampling_time; // sampling time
            results[i].field[sample] = applied_field( sample * sampling_time );
            results[i].mx[sample] = pstate[3*i];
            results[i].my[sample] = pstate[3*i + 1];
            results[i].mz[sample] = pstate[3*i + 2];
        }
    } // end sampling loop

    /// @TODO compute energy loss for llg
    for( auto &res : results )
        res.energy_loss = 0;

    /// Free memory
    delete[] state;
    delete[] dwm;
    delete[] a_work;
    delete[] b_work;
    delete[] adash_work;
    delete[] bdash_work;
    delete[] x_guess;
    delete[] x_opt_tmp;
    delete[] x_opt_jac;
    delete[] x_opt_ipiv;
    delete[] wiener;
    delete[] heff;
    delete[] heffjac;
    delete[] anis;
    delete[] pstate;
    delete[] nstate;
    delete[] distances;
    delete[] cubed_distance_magnitudes;

    return results;
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
 * @param[in] tau0 reciprocal of the attempt frequency \f$1/f_0\f$ - s-1
 * @param[in] magnetisation the saturation magnetisation of the
 *            particle
 * @param[in] alpha the dimensionless damping parameter of the particle
 * @param[in] applied_field a scalar in-out function that returns the
 * value of the reduced applied field in the z-direction at time
 * t. Reduced field is field value /f$h=H/H_k/f$ where /f$H_k/f$ is
 * the anisotropy field
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
    const double magnetisation,
    const double alpha,
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
    // _1 output - probability derivatives
    // _2 input - current probabilities
    // _3 input - current time
    std::function<void(double*,const double*,const double)> master_equation =
        std::bind(dom::master_equation_with_update, _1, work, anisotropy,
                  volume, temperature, magnetisation, alpha, _3, _2, applied_field );

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
    res.time[0] = 0;
    res.field[0] = applied_field( 0 );

    // allocate memory needed for computing the power loss
    double prob_derivs[2];
    double energy_sum = 0;
    master_equation( prob_derivs, next_state, 0 );
    double last_pflow=0, next_pflow=prob_derivs[0];
    double last_et=0, next_et = dom::single_transition_energy(
        anisotropy, volume, applied_field( 0 ) );


    // Variables needed in the loop
    double last_t, t=0;
    double max_dt = end_time / 1000.0; // never step more than 1000th of the simulation time
    double dt = 0.01*time_step;
    unsigned int step=0;
    double eps=time_step; // tolerance of the rk45 integrator
    for( unsigned int sample=1; sample<N_samples; sample++ )
    {
        // Perform a simulation step until we breach the next sampling point
        while ( t <= sample*sampling_time )
        {
            // take a step
            last_state[0] = next_state[0];
            last_state[1] = next_state[1];
            last_pflow = next_pflow;
            last_et = next_et;
            last_t = t;
            step++;

            // perform integration step
            integrator::rk45( next_state, tmpstate, k1, k2, k3, k4, k5, k6,
                              &dt, &t, last_state, master_equation, n_dims,
                              eps );

            // dt should never be greater than 1/1000 of the simulation time
            dt = dt>max_dt? max_dt : dt;

            // Compute the energy loss for this step and add to total energy of cycle
            master_equation( prob_derivs, next_state, t );
            next_pflow = prob_derivs[0];
            next_et = dom::single_transition_energy(
                anisotropy, volume, applied_field( t ) );
            energy_sum += one_step_energy_loss(
                last_et, next_et, last_pflow, next_pflow, last_t, t, volume);


        } // end integration stepping loop
        /*
          Once this point is reached, we are currently one step beyond
          the desired sampling point. Use a first-order-hold:
        */
        /**
           The magnetisation is computed as the magnetisation of
           each state multiplied by it's probability.
           \f$M(t) = M_1p_1(t) + M_2p_2(t)\f$
           For a uniaxial particle we take \f$M_1=1,M_2=-1\f$
           Thus the magnetisation is the difference between the two
           state probabilities.
        */
        double mz_next = next_state[0] - next_state[1];
        double t_next = t;
        double t_this = sample*sampling_time;

        // First-order-hold
        double beta = (mz_next - res.mz[sample-1])/(t_next-res.time[sample-1]);
        res.mz[sample] = res.mz[sample-1] + beta*(t_this - res.time[sample-1]);

        res.time[sample] = t_this;
        res.field[sample] = applied_field(t_this);
    }

    // @deprecated
    // Compute the energy loss from the hysteresis area
    // double hk = 2*anisotropy / constants::MU0 / magnetisation;
    // res.energy_loss = simulation::energy_loss( res, magnetisation, hk );

    // Store the total energy dissipated
    // Sign is not deterministic and depends on where the simulation begins
    // the steady state cycle. So we take the absolute value.
    res.energy_loss = std::abs(energy_sum);

    // free memory
    return res;
}
