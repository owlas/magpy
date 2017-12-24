#include "../include/simulation.hpp"
#include "../include/constants.hpp"
#include <cmath>
#include "../include/field.hpp"
#include "../include/distances.hpp"
#include "../include/io.hpp"
#include "../include/dom.hpp"
#include "../include/integrators.hpp"
#include "../include/optimisation.hpp"
#include "../include/llg.hpp"
#include <numeric>
#include <exception>
#include <sstream>
#include <cblas.h>

using namespace std::placeholders;

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

/// Save results to disk
/**
 * Saves the contents of a results struct to disk. Given a file name
 * 'foo' the following files are written to disk 'foo.mx', 'foo.my',
 * 'foo.mz', 'foo.field', 'foo.time', 'foo.energy'.
 * @param[in] fname /path/to/filename prefix for files
 * @param[in] res results struct to save
 */
void simulation::save_results( const std::string fname, const struct results &res )
{
    std::stringstream magx_fname, magy_fname, magz_fname, field_fname, time_fname, energy_fname;
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

/// Initialise results memory to zero
/**
 * @param[in,out] res results struct to zero
 */
void simulation::zero_results( struct simulation::results &res )
{
    for( unsigned int i=0; i<res.N; i++ )
        res.mx[i] = res.my[i] = res.mz[i] = res.field[i] = res.time[i] = 0;
}

/// Simulate the dynamics of interacting magnetic particles (reduced input)
/**
 * Simulates the stochastic Laundau-Lifshitz-Gilbert equation for a
 * system of single domain magnetic nanoparticles interacting through
 * the dipolar interaction term.
 * @param[in] thermal_field_strengths intensity of the thermal field
 * experienced by each particle \f$=\frac{\alpha K_BT}{\bar{K} V_i
 * (1+\alpha^2)}$\f for the ith particle.
 * @param[in] reduced_anisotropy_constants strength of the uniaxial
 * anisotropy for each particle divided by the average anisotropy
 * constant of all particles \f$=K_i/\bar{K}$\f
 * @param[in] reduced_particle_volumes volume of each particle divided
 * by the average volume of all particles \f$=V_i/\bar{V}$\f
 * @param[in] anisotropy_unit_axes the unit vector (x,y,z) indicating
 * the direction in 3D space of each particle's uniaxial anisotropy
 * axis
 * @param[in] initial_magnetisations initial state of the system. A
 * unit vector (x,y,z) of the initial direction of the magnetisation
 * for each particle
 * @param[in] interparticle_unit_distances a matrix of unit distance
 * vectors (x,y,z) where element i,j is the unit distance vector
 * between particle i and particle j
 * @param[in] interparticle_reduced_distance_magnitudes a matrix of
 * reduced distance magnitudes which for element i,j is the Euclidean
 * distance between particle i and particle j divided by a normalising
 * factor \f$\cbrt{\bar{V}}
 * @param[in] applied_field time-varying externally applied magnetic
 * field, which is assumed to be applied along the z-axis. Returns the
 * value of the reduced field
 * \f$h(t)=H(t)/H_k,H_k=\frac{2*\bar{K}}{\mu_0 M_s}$\f given the time
 * t
 * @param[in] average_anisotropy average of the particle
 * anisotropies \f$\bar{K}=\frac{1}{N}\sum_i K_i
 * @param[in] average_volume average of the particle volumes
 * \f$bar{V}=\frac{1}{N}\sum_i V_i
 * @param[in] damping_constant dimensionless damping ratio. Fixed for
 * all particles
 * @param[in] saturation_magnetisation the magnitude of the saturated
 * magnetisation. Fixed for all particles.
 * @param[in] time_step size of the time step for each step of the
 * simulation (in reduced time units)
 * @param[in] end_time total time of the simulation (in reduced time
 * units)
 * @param[in] rng initialised random number generator (keep the seed
 * same for reproducible simulations)
 * @param[in] renorm set to True to artificially scale the
 * magnetisation vectors back to unity at each time step
 * @param[in] interactions set to True to enable dipolar interactions
 * @param[in] use_implicit set to true to use the implicit midpoint
 * scheme
 * for time integration. Set to false to use the explicit Heun scheme
 * @param[in] eps (only needed if \p use_implicit =true) the tolerance
 * of the implicit scheme
 * @param[in] max_samples number of times to sample the solution. Uses
 * a first-order hold approach to interpolate between time steps.
 * @returns simulation results struct for each particle
 */
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
    const bool interactions,
    const bool use_implicit,
    const double eps,
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
    const size_t max_iter=1000;

    // Copy in the initial state
    for( auto &res : results )
    {
        res.time[0] = 0;
        res.field[0] = applied_field( 0 );
    }
    for( size_t i=0; i<n_particles; i++ )
    {
        results[i].time[0] = 0;
        results[i].field[0] = applied_field( 0 );
        results[i].mx[0] = initial_magnetisations[i][0];
        results[i].my[0] = initial_magnetisations[i][1];
        results[i].mz[0] = initial_magnetisations[i][2];
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
         distances, reduced_particle_volumes, cubed_distance_magnitudes, interactions]
        ( double *heff, const double *state, const double t )
        {
            field::zero_all_field_terms( heff, state_size );

            field::multi_add_uniaxial_anisotropy(
                heff, state, anis, reduced_anisotropy_constants.data(), n_particles );

            field::multi_add_applied_Z_field_function( heff, applied_field,  t, n_particles );

            if( interactions )
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

    /// Create a version without Jacobians if implicit is not requested
    std::function<void(double*,double*,const double*,double)> sde_no_jacs =
        [heff, heff_func, damping_ratios, thermal_field_strengths, n_particles]
        (double *drift, double *diffusion, const double *state, const double t)
        {
            llg::multi_stochastic_llg_field_update(
                drift, diffusion, heff, heff_func, state, t,
                damping_ratios.data(), thermal_field_strengths.data(),
                n_particles );
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
            if( use_implicit )
            {
                int errcode = integrator::implicit_midpoint(
                    nstate, dwm, a_work, b_work, adash_work, bdash_work, x_guess,
                    x_opt_tmp, x_opt_jac, x_opt_ipiv, pstate, wiener, sde,
                    state_size, state_size, t, time_step, eps, max_iter );
                if( errcode != optimisation::SUCCESS )
                    std::runtime_error( "implicit integration error code" );
            }
            else
                integrator::heun(
                    nstate, a_work, adash_work, b_work, bdash_work,
                    pstate, wiener, sde_no_jacs, state_size, state_size,
                    t, time_step );


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


/// Simulate the dynamics of interacting magnetic particles
/**
 * Simulates the stochastic Laundau-Lifshitz-Gilbert equation for a
 * system of single domain magnetic nanoparticles interacting through
 * the dipolar interaction term.
 * @param[in] radius radius of each nanoparticle [m]
 * @param[in] anisotropy anisotropy strength constant for each
 * particle's uniaxial anisotropy axis [J/m3]
 * @param[in] anisotropy_axis unit vector of the uniaxial anisotropy
 * axis direction (x,y,z) for each particle [dimensionless]
 * @param[in] magnetisation_direction unit vector of the initial
 * direction of the magnetisation (x,y,z) for each particle [dimensionless]
 * @param[in] location coordinate vector (x,y,z) of the location in 3D
 * space of each particle [m]
 * @param[in] magnetisation saturation magnetisation for all particles
 * [A/m]
 * @param[in] damping damping constant for all particles
 * [dimensionless]
 * @param[in] temperature temperature of the coupled heat bath
 * (uniform temperature environment for all particles) [K]
 * @param[in] renorm set to True to artificially scale the
 * magnetisation vectors back to unity at each time step
 * @param[in] interactions set to True to enable dipolar interactions
 * @param[in] use_implicit set to true to use the implicit midpoint
 * scheme
 * for time integration. Set to false to use the explicit Heun scheme
 * @param[in] eps (only needed if \p use_implicit =true) the tolerance
 * of the implicit scheme
 * @param[in] time_step size of the time step for each step of the
 * simulation [s]
 * @param[in] end_time total time of the simulation [s]
 * @param[in] max_samples number of times to sample the solution. Uses
 * a first-order hold approach to interpolate between time steps.
 * @param[in] seed seed for the random number generator (reuse same
 * seed for reproducible simulations)
 * @param[in] field_shape time-varying externally applied field shape
 * see field::options
 * @param[in] field_amplitude peak amplitude of the time-varying
 * externally appplied field [A/m]
 * @param[in] field_frequency frequency of the time-varying externally
 * applied field [Hz]
 * @returns simulation results struct for each particle
 * @todo refactor to accept field function (see dom simulation)
 */
std::vector<simulation::results> simulation::full_dynamics(
    const std::vector<double> radius,
    const std::vector<double> anisotropy,
    const std::vector<d3> anisotropy_axis,
    const std::vector<d3> magnetisation_direction,
    const std::vector<d3> location,
    const double magnetisation,
    const double damping,
    const double temperature,
    const bool renorm,
    const bool interactions,
    const bool use_implicit,
    const double eps,
    const double time_step,
    const double end_time,
    const size_t max_samples,
    const long seed,
    const field::options field_shape,
    const double field_amplitude,
    const double field_frequency
    )
{
    size_t n_particles = radius.size();

    // VOLUME
    std::vector<double> volume;
    for( auto &r : radius )
        volume.push_back( 4.0 / 3.0 * M_PI * r * r * r );

    double average_volume = std::accumulate(
        volume.begin(), volume.end(), 0.0 ) / n_particles;

    std::vector<double> reduced_volume;
    for( auto &v : volume )
        reduced_volume.push_back( v / average_volume );

    // STABILITY RATIO
    std::vector<double> stability_ratio;
    for( size_t p=0; p<n_particles; p++ )
        stability_ratio.push_back(
            anisotropy[p] * volume[p] / constants::KB / temperature );

    // ANISOTROPY
    double average_anisotropy = std::accumulate(
        anisotropy.begin(), anisotropy.end(), 0.0 ) / n_particles;

    std::vector<double> reduced_anisotropy;
    for( auto &a : anisotropy )
        reduced_anisotropy.push_back( a / average_anisotropy );

    // ANISOTROPY FIELD
    double anisotropy_field = 2 * average_anisotropy / constants::MU0 / magnetisation;

    // TIME
    double time_factor = constants::GYROMAG * constants::MU0 * anisotropy_field
        / ( 1+damping*damping );
    double reduced_time_step = time_step * time_factor;
    double reduced_end_time = end_time * time_factor;

    // THERMAL FIELD
    std::vector<double> thermal_field_strength;
    for( auto &v : volume )
        thermal_field_strength.push_back(
            std::sqrt(
                damping * constants::KB * temperature
                / ( average_anisotropy * v) / ( 1 + damping*damping ) ) );

    ///////////////////////
    // Compute the field //
    ///////////////////////

    // Reduced field amplitude and frequency
    double happ = field_amplitude / anisotropy_field;
    double fapp = field_frequency / time_factor;

    // The field function
    std::function<double(const double)> field_function;
    switch( field_shape )
    {
    case field::CONSTANT :
        field_function=
            [happ](const double)
            { return happ; };
        break;
    case field::SINE :
        field_function =
            [happ, fapp](const double t)
            { return field::sinusoidal( t, happ, fapp ); };
        break;
    case field::SQUARE :
        field_function =
            [happ, fapp](const double t)
            { return field::square( t, happ, fapp ); };
        break;
    default :
        throw std::invalid_argument( "Must specify valid field::options enum" );
        break;
    }


    // DISTANCES
    auto distance = distances::pair_wise_distance_vectors( location );
    auto distance_mag = distances::pair_wise_distance_magnitude( distance );
    auto distance_unit = distances::pair_wise_distance_unit_vectors( location );
    auto reduced_distance_mag = distance_mag;
    for( auto &i : reduced_distance_mag )
        for( auto &j : i )
            j /= std::pow( average_volume, 1./3 );

    // RANDOM NUMBER GENERATOR
    RngMtNorm rng( seed, 1.0 );

    auto results = simulation::full_dynamics(
        thermal_field_strength,
        reduced_anisotropy,
        reduced_volume,
        anisotropy_axis,
        magnetisation_direction,
        distance_unit,
        reduced_distance_mag,
        field_function,
        average_anisotropy,
        average_volume,
        damping,
        magnetisation,
        reduced_time_step,
        reduced_end_time,
        rng,
        renorm,
        interactions,
        use_implicit,
        eps,
        max_samples );

    for( size_t p=0; p<n_particles; p++ )
        for( size_t i=0; i<max_samples; i++ )
        {
            // Convert reduced time back to real time
            results[p].time[i] /= time_factor;
            // Convert reduced field back to real field
            results[p].field[i] *= anisotropy_field;
            // Convert magnetisation
            results[p].mx[i] *= magnetisation;
            results[p].my[i] *= magnetisation;
            results[p].mz[i] *= magnetisation;
        }

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

    // Allocate arrays for the RK45 integrator
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
    

    // Variables needed in the loop
    double t=0;
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
            step++;

            // perform integration step
            integrator::rk45( next_state, tmpstate, k1, k2, k3, k4, k5, k6,
                              &dt, &t, last_state, master_equation, n_dims,
                              eps );

            // dt should never be greater than 1/1000 of the simulation time
            dt = dt>max_dt? max_dt : dt;


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

    // free memory
    return res;
}
