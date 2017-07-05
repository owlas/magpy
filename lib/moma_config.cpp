// moma_config.cpp
// Implementation
//
// Oliver W. Laslett (2016)
// O.Laslett@soton.ac.uk
#include "../include/moma_config.hpp"

#define _USE_MATH_DEFINES
#include <cmath>
#include <array>
#include <string>
#include <omp.h>
#include <exception>
#include <fstream>
#include "../include/constants.hpp"
#include "../include/field.hpp"
#include "../include/simulation.hpp"
#include "../include/rng.hpp"
#include "../include/distances.hpp"

using d3=std::array<double,3>;
using rng_vec=std::vector<std::shared_ptr<Rng>>;

/// Validate that fields for llg simulation are in json
/**
 * Inspects a json file to check that it has the right structure and
 * that parameters match assumptions required for simulation
 * (e.g. temperature is not negative).
 * Fatal logs on failure.
 * @param[in] input json config object
 */
void moma_config::validate_for_llg( const json input )
{
    // Check simulation parameters
    bool renorm = input.at("simulation").at("renormalisation");
    bool steady_cycle;
    double sim_time;
    if( input.at("simulation").at("simulation-time").is_string() )
        steady_cycle = true;
    else {
        steady_cycle = false;
        sim_time = input.at("simulation").at("simulation-time").get<double>();
    }
    double time_step = input.at("simulation").at("time-step");
    if( time_step < 0 )
        throw std::invalid_argument( "Time_Step must be greater than 0" );
    int ensemble_size = input.at("simulation").at("ensemble-size");
    if ( ensemble_size < 1 )
        throw std::invalid_argument( "Ensemble size must be interger value at least 1" );
    std::vector<long> seeds = input.at("simulation").at("seeds");
    if ( seeds.size() < ensemble_size )
        throw std::invalid_argument( "Not enough seeds. Must have as many seeds as ensemble size." );

    // Check output
    std::string dir = input.at("output").at("directory");
    int max_samples = input.at("output").at("max-samples");
    if ( ( max_samples < 1 ) && ( max_samples != -1 ) )
        throw std::invalid_argument( "Valid values for max-samples are -1 or integers greater than 0" );

    // Check global
    double temp = input.at("global").at("temperature");
    if( temp < 0 )
        throw std::invalid_argument( "Temperature must be greater than 0" );
    std::string shape = input.at("global").at("applied-field").at("shape");
    if( ( shape.compare( "sine" ) != 0 )
        && ( shape.compare("square") != 0 )
        && ( shape.compare("square-fourier") != 0 ) )
        throw std::invalid_argument( "Valid values for shape are 'sine' or 'square'" );
    if( shape.compare( "square-fourier" ) == 0 )
        if( input.at("global").at("applied-field").count("components") != 1 )
            throw std::invalid_argument( "If specifying Fourier series field, "
                                         "must specify number of components" );
    double freq = input.at("global").at("applied-field").at("frequency");
    if( freq < 0 )
        throw std::invalid_argument( "Frequency must be greater than 0" );
    double amp = input.at("global").at("applied-field").at("amplitude");

    // Check particles
    if( !input.at("particles").is_array() )
        throw std::invalid_argument( "particles should be an array of particles!" );
    std::vector<json> particles = input.at("particles");

    double ms_ref = input.at("particles")[0].at("magnetisation");
    double alpha_ref = input.at("particles")[0].at("damping");
    for( auto &p : particles )
    {
        double damping = p.at("damping");
        if( damping < 0 )
            throw std::invalid_argument( "Damping must be greater than 0" );
        if( damping != alpha_ref )
            throw std::invalid_argument( "All particles must have the same damping value" );
        double radius = p.at("radius");
        if( radius < 0 )
            throw std::invalid_argument( "Radius must be greater than 0" );
        double anisotropy = p.at("anisotropy");
        if( anisotropy < 0 )
            throw std::invalid_argument( "Anisotropy must be greater than 0" );
        std::array<double,3> aaxis = {
            p.at("anisotropy-axis")[0],
            p.at("anisotropy-axis")[1],
            p.at("anisotropy-axis")[2]
        };
        double aaxis_mag = aaxis[0]*aaxis[0] + aaxis[1]*aaxis[1] + aaxis[2]*aaxis[2];
        if( std::abs( aaxis_mag - 1 ) > 0.00001 )
            throw std::invalid_argument( "Anisotropy axis must be unit vector" );
        double magnetisation = p.at("magnetisation");
        if( magnetisation < 0 )
            throw std::invalid_argument( "Magnetisation must be greater than 0" );
        if( magnetisation != ms_ref )
            throw std::invalid_argument( "All particles must have the same saturation magnetisation" );
        std::array<double,3> magaxis = {
            p.at("magnetisation-direction")[0],
            p.at("magnetisation-direction")[1],
            p.at("magnetisation-direction")[2]
        };
        double magaxis_mag = magaxis[0]*magaxis[0] + magaxis[1]*magaxis[1] + magaxis[2]*magaxis[2];
        if( std::abs( magaxis_mag - 1 ) > 0.00001 )
            throw std::invalid_argument( "Magnetisation direction axis must be unit vector" );
    }
}

/// Validate that fields for discerete-orientation-model simulation are in json
/**
 * Inspects a json file to check that it has the right structure and
 * that parameters match assumptions required for simulation
 * (e.g. temperature is not negative).
 * Fatal logs on failure.
 * @param[in] input json config object
 */
void moma_config::validate_for_dom( const json input )
{
    // Check simulation parameters
    bool steady_cycle;
    double sim_time;
    if( input.at("simulation").at("simulation-time").is_string() )
        steady_cycle = true;
    else {
        steady_cycle = false;
        sim_time = input.at("simulation").at("simulation-time").get<double>();
    }
    double time_step = input.at("simulation").at("time-step");
    if( time_step < 0 )
        throw std::invalid_argument( "Time_Step must be greater than 0" );

    // Check output
    std::string dir = input.at("output").at("directory");
    int max_samples = input.at("output").at("max-samples");
    if ( ( max_samples < 1 ) && ( max_samples != -1 ) )
        throw std::invalid_argument( "Valid values for max-samples are -1 or integers greater than 0" );

    // Check global
    double temp = input.at("global").at("temperature");
    if( temp < 0 )
        throw std::invalid_argument( "Temperature must be greater than 0" );
    std::string shape = input.at("global").at("applied-field").at("shape");
    if( ( shape.compare( "sine" ) != 0 )
        && ( shape.compare("square") != 0 )
        && ( shape.compare("square-fourier") !=0 ) )
        throw std::invalid_argument( "Valid values for shape are 'sine' or 'square'" );
    if( shape.compare( "square-fourier" ) == 0 )
        if( input.at("global").at("applied-field").count("components") != 1 )
            throw std::invalid_argument( "If specifying Fourier series field, "
                                         "must specify number of components" );
    double freq = input.at("global").at("applied-field").at("frequency");
    if( freq < 0 )
        throw std::invalid_argument( "Frequency must be greater than 0" );
    double amp = input.at("global").at("applied-field").at("amplitude");
    double tau0 = input.at("global").at("tau0");

    std::vector<json> particles = input.at("particles");
    for( auto &p : particles )
    {
        double radius = p.at("radius");
        if( radius < 0 )
            throw std::invalid_argument( "Radius must be greater than 0" );
        double anisotropy = p.at("anisotropy");
        if( anisotropy < 0 )
            throw std::invalid_argument( "Anisotropy must be greater than 0" );
        std::array<double,2> probs = {
            p.at("initial-probs")[0],
            p.at("initial-probs")[1]
        };
        double probs_sum = probs[0] + probs[1];
        if( std::abs( probs_sum - 1 ) > 0.00001 )
            throw std::invalid_argument( "Initial probabilities must sum to 1" );
        double magnetisation = p.at("magnetisation");
        if( magnetisation < 0 )
            throw std::invalid_argument( "Magnetisation must be greater than 0" );

        if( p.count("damping") != 1 )
            throw std::invalid_argument( "Missing damping parameter for particle" );
    }
}


json moma_config::transform_input_parameters_for_dom( const json in )
{
    /**
     * Output json is initialised with the input json
     */
    json out = in;

    /**
     * Convert the sim_time input
     */
    bool steady_cycle;
    if( in.at("simulation").at("simulation-time").is_string() )
        steady_cycle = true;
    else {
        steady_cycle = false;
    }
    out["simulation"]["steady-cycle-activated"] = steady_cycle;

    std::vector<json> particles = in["particles"];
    for( auto &p : particles )
    {
        /**
         * The volume of the particle is computed from its radius.
         */
        double rad = p["radius"];
        double volume = 4.0 / 3.0 * M_PI * rad * rad * rad;
        p["volume"] = volume;

        /**
         * The stability ratio \f$\frac{KV}{k_BT}\f$ is computed.
         */
        double k = p["anisotropy"];
        double T = out["global"]["temperature"];
        double stability = k*volume/constants::KB/T;
        p["stability-ratio"] = stability;

        /**
         * The anisotropy field associated with each particle is
         * computed \f$h_k=\frac{2K}{\mu_0Ms}\f$
         */
        double ms = p["magnetisation"];
        double H = out["global"]["applied-field"]["amplitude"];
        double Hk = 2*k/constants::MU0/ms;
        double h = H/Hk;
        p["anisotropy-field"] = Hk;
        p["reduced-field-amplitude"] = h;
        p["reduced-field-frequency"] = out["global"]["applied-field"]["frequency"];
    }
    /**
     * Copy the updated particles list into the transformed paramaters
     */
    out["particles"] = particles;

    return out;
}


json moma_config::transform_input_parameters_for_llg( const json in )
{
    /**
     * Output json is initialised with the input json
     */
    json out = in;

    // Resolve the simulation time
    bool steady_cycle; double sim_time;
    if( in.at("simulation").at("simulation-time").is_string() )
        steady_cycle = true;
    else {
        steady_cycle = false;
        sim_time = in.at("simulation").at("simulation-time").get<double>();
    }
    out["simulation"]["steady-cycle-activated"] = steady_cycle;


    // Get the particles
    std::vector<json> particles = in.at("particles");
    const size_t N_particles = particles.size();

    // Compute particle volumes
    double average_volume = 0.0;
    for( auto &p : particles )
    {
        p["volume"] = 4.0 / 3.0 * M_PI * std::pow( p["radius"].get<double>(), 3 );
        average_volume += p["volume"].get<double>();
    }
    average_volume /= N_particles;
    out["global"]["average-volume"] = average_volume;

    // Compute particle reduced volumes
    for( auto &p : particles )
        p["reduced-volume"] = p["volume"].get<double>() / average_volume;

    // Compute particle stability ratios
    for( auto &p : particles )
        p["stability-ratio"] = p.at("anisotropy").get<double>() * p.at("volume").get<double>()
            / constants::KB / in.at("global").at("temperature").get<double>();

    // Compute particle reduced anisotropy constants
    double average_anisotropy = 0.0;
    for( auto &p : particles )
        average_anisotropy += p.at("anisotropy").get<double>();
    average_anisotropy /= N_particles;
    out["global"]["average-anisotropy"] = average_anisotropy;
    for( auto &p : particles )
        p["reduced-anisotropy"] = p.at("anisotropy").get<double>() / average_anisotropy;

    // Compute the anisotropy field and reduced field
    double hk= 2.0 * average_anisotropy / constants::MU0
        / particles[0].at("magnetisation").get<double>();
    out.at("global")["anisotropy-field"] = hk;
    out.at("global").at("applied-field")["reduced-amplitude"] =
        out.at("global").at("applied-field").at("amplitude").get<double>() / hk;

    // Compute the time rescaling
    double damping = particles[0].at("damping");
    double time_factor = constants::GYROMAG * constants::MU0 * hk
        / ( 1+damping*damping );
    out["simulation"]["time-factor"] = time_factor;

    // Rescale time dependent parameters
    double time_step = in["simulation"]["time-step"];
    double tau_step = time_step * time_factor;
    out["simulation"]["time-step"] = tau_step;
    double f = in["global"]["applied-field"]["frequency"];
    double f_in_tau = f / time_factor;
    out["global"]["applied-field"]["reduced-frequency"] = f_in_tau;
    if( !steady_cycle )
    {
        double sim_tau = sim_time * time_factor;
        out["simulation"]["simulation-time"] = sim_tau;
    }

    /// Compute the thermal field strength for each particle
    /**
     * Computed as \f$D=\frac{\alpha k_BT}{2V_i\bar{K}(1+\alpha^2)}\f$
     */
    for( auto &p : particles )
        p["thermal-field-strength"] = std::sqrt(
            damping * constants::KB * in.at("global").at("temperature").get<double>()
            / ( average_anisotropy * p.at("volume").get<double>()
                * ( 1 + damping * damping ) ) );
    out["particles"] = particles;
    return out;
}

int moma_config::write( const std::string fname, const json output )
{
    std::ofstream os( fname );
    if( os.is_open() )
    {
        os << std::setw(4) << output;
        os.close();
        return 0;
    }
    else
    {

        throw std::runtime_error( "Error opening file to write json: " );
        return -1;
    }
}

/// Main entry point for launching simulations from json config
/**
 * Runs simulations specified by the json configuration. The config
 * must be normalised `moma_config::normalise`. Function does not
 * return anything but will write results to disk (see logs and config
 * options).
 * @param[in] norm_config json object of the normalised configuration
 * file
 */
void moma_config::launch_simulation( const json config )
{
    // Get the simulation mode
    std::string sim_mode = config.at("simulation-mode");

    // Transform the input for simulation
    if( !sim_mode.compare( "dom" ) )
        moma_config::launch_dom_simulation( config );
    else if( !sim_mode.compare( "llg" ) )
        moma_config::launch_llg_simulation( config );
    else
        throw std::invalid_argument( "Simulation-mode must be 'llg' or 'dom'" );
}

void moma_config::launch_dom_simulation( const json in )
{
    /**
     * Validates and transforms the input config to get additional
     * simulation parameters and normalised parameters.
     */
    moma_config::validate_for_dom( in );
    auto params = moma_config::transform_input_parameters_for_dom( in );

    /**
     * Writes the transformed json to file
     */
    std::ostringstream config_out_fname;
    std::string dir = params["output"]["directory"];
    config_out_fname << dir << "/config_norm.json";
    moma_config::write( config_out_fname.str(), params );

    // Number of particles in the ensemble
    std::vector<json> particles = params.at("particles");
    size_t n_particles = particles.size();

    /**
     * Bind the applied field function for each particle
     */
    std::vector<std::function<double(double)> > happs;
    std::string field_shape = params["global"]["applied-field"]["shape"];

    for( unsigned int i=0; i<n_particles; i++ )
    {
        std::function<double(double)> happ;
        double field_amp = particles[i].at("reduced-field-amplitude");
        if( field_shape.compare("sine") == 0 )
            happ = std::bind(
                field::sinusoidal,
                field_amp,
                params["global"]["applied-field"]["frequency"],
                std::placeholders::_1 );
        else if( field_shape.compare("square") == 0 )
            happ = std::bind(
                field::square,
                field_amp,
                params["global"]["applied-field"]["frequency"],
                std::placeholders::_1 );
        else if( field_shape.compare("square-fourier") == 0 )
            happ = std::bind(
                field::square_fourier,
                field_amp,
                params["global"]["applied-field"]["frequency"],
                params["global"]["applied-field"]["components"],
                std::placeholders::_1 );
        else
            throw std::invalid_argument( "field is not a valid field shape." );

        happs.push_back( happ );
    }

    // Get the temperature
    double temperature = params.at("global").at("temperature");

    // Get the particle properties
    std::vector<double> volumes;
    std::vector<double> anisotropies;
    std::vector<double> magnetisations;
    std::vector<double> dampings;
    for( unsigned int i=0; i<n_particles; i++ )
    {
        volumes.push_back(particles[i].at("volume"));
        anisotropies.push_back(particles[i].at("anisotropy"));
        magnetisations.push_back(particles[i].at("magnetisation"));
        dampings.push_back(particles[i].at("damping"));
    }

    // Get the initial conditions for each particle
    std::vector<std::array<double,2> > init_probs;
    std::vector<d3> init_mags;
    for( unsigned int i=0; i<n_particles; i++ )
    {
        std::array<double,2> init = {
            particles[i].at("initial-probs")[0],
            particles[i].at("initial-probs")[1]
        };
        init_probs.push_back( init );

        d3 initm = { 0, 0, init[0]-init[1] };
        init_mags.push_back( initm );
    }

    // Get simulation properties
    double time_step = params.at("simulation").at("time-step");
    int max_samples = params.at("output").at("max-samples");
    double tau0 = params.at("global").at("tau0");

    // Initialise results
    simulation::results results( max_samples );

    // Check if we should run until steady state
    bool steady_cycle = params.at("simulation").at("steady-cycle-activated");

    // Compute the simulation time
    double sim_time = steady_cycle ?
        1/ params.at("global").at("applied-field").at("frequency").get<double>()
        : params.at("simulation").at("simulation-time").get<double>();

    /**
     * TRANSITION STATE DYNAMICS
     * The run function is created. This function should return a
     * simulation::results struct. For arguments that are constants
     * across all simulations, the values are bound to the run
     * function. The run function arguments are the parameters that
     * are variable between simulations.
     *
     * Transition state simulation does not require random number
     * generators. Only the initial state of the system.
     */
    std::function<simulation::results(d3,double,double,double,double,std::function<double(double)>)> run_function=
        [temperature, time_step, sim_time, max_samples]
        (d3 initial_mag, double volume, double anisotropy, double ms, double damping,
         std::function<double(double)> happ)
        {
            // Convert initial magnetisation into states
            std::array<double,2> initial_probs = {
                0.5*( initial_mag[2] + 1 ),
                0.5*( 1 - initial_mag[2] )
            };

            return simulation::dom_ensemble_dynamics(
                volume, anisotropy, temperature, ms, damping, happ,
                initial_probs, time_step, sim_time, max_samples );
        };

    // Run the simulation
    if( steady_cycle )
    {
        double steady_state_condition = 1e-3;
        results = simulation::steady_state_cycle_dynamics(
            run_function,
            max_samples,
            steady_state_condition,
            init_mags,
            volumes,
            anisotropies,
            magnetisations,
            dampings,
            happs );
    }
    else
    {
        results = simulation::ensemble_run(
            max_samples,
            run_function,
            init_mags,
            volumes,
            anisotropies,
            magnetisations,
            dampings,
            happs );
    }

    // save the results
    std::stringstream fname;
    fname << params.at("output").at("directory").get<std::string>()
          << "/results";
    simulation::save_results( fname.str(), results );


    // Compute the power emitted by the particle ensemble
    std::cout << "ensemble energy loss: " << results.energy_loss << std::endl;
    double f = params["global"]["applied-field"]["frequency"];
    double power = results.energy_loss * f;

    json output;
    std::cout << "ensemble power: " << power << std::endl;
    output["power"] = power;

    /**
     * The current git commit version is written to the file. Made
     * available through a compiler flag (see makefile)
     */
    output["git-version"] = VERSION;

    // Write the normalised parameters to file
    std::ostringstream output_fname;
    output_fname << params.at("output").at("directory").get<std::string>()
                 << "/output.json";
    moma_config::write( output_fname.str(), output );
}


void moma_config::launch_llg_simulation( const json in )
{
    // Validate and transform the input
    moma_config::validate_for_llg( in );
    auto params = moma_config::transform_input_parameters_for_llg( in );

    /**
     * Writes the transformed json to file
     */
    std::ostringstream config_out_fname;
    std::string dir = params["output"]["directory"];
    config_out_fname << dir << "/config_norm.json";
    moma_config::write( config_out_fname.str(), params );

    // Bind the applied field waveform
    std::string field_shape = in["global"]["applied-field"]["shape"];
    std::function<double(double)> happ;
    if( field_shape.compare("sine") == 0 )
        happ = std::bind(
            field::sinusoidal,
            params["global"]["applied-field"]["reduced-amplitude"],
            params["global"]["applied-field"]["reduced-frequency"],
            std::placeholders::_1 );
    else if( field_shape.compare("square") == 0 )
        happ = std::bind(
            field::square,
            params["global"]["applied-field"]["reduced-amplitude"],
            params["global"]["applied-field"]["reduced-frequency"],
            std::placeholders::_1 );
    else if( field_shape.compare("square-fourier") == 0 )
        happ = std::bind(
            field::square_fourier,
            params["global"]["applied-field"]["reduced-amplitude"],
            params["global"]["applied-field"]["reduced-frequency"],
            params["global"]["applied-field"]["components"],
            std::placeholders::_1 );
    else
        throw std::invalid_argument( "field is not a valid field shape." );

    // Get the uniaxial anisotropy axes
    std::vector<json> particles = params.at("particles");
    std::vector<std::array<double,3> > axes;
    for( auto &p : particles )
    {
        std::array<double,3> aaxis{
            p.at("anisotropy-axis")[0],
            p.at("anisotropy-axis")[1],
            p.at("anisotropy-axis")[2] };
        axes.push_back( aaxis );
    }

    // Run the simulation N times for a full ensemble
    size_t n_runs = params["simulation"]["ensemble-size"];

    // Get the particle properties
    double damping = particles[0].at("damping");
    double saturation_magnetisation = particles[0].at("magnetisation");
    double average_volume = params.at("global").at("average-volume");
    double average_anisotropy = params.at("global").at("average-anisotropy");
    std::vector<double> thermal_field_strengths;
    std::vector<double> reduced_anisotropy_constants;
    std::vector<double> reduced_particle_volumes;
    for( auto &p : particles )
    {
        thermal_field_strengths.push_back(
            p.at("thermal-field-strength").get<double>() );
        reduced_anisotropy_constants.push_back(
            p.at("reduced-anisotropy").get<double>() );
        reduced_particle_volumes.push_back(
            p.at("reduced-volume").get<double>() );
    }

    // compute the distances
    std::vector<std::array<double, 3> > locations;
    for( auto &p : particles )
    {
        std::array<double, 3> loc{
            p.at("location")[0],
            p.at("location")[1],
            p.at("location")[2]
        };
        locations.push_back( loc );
    }
    auto interparticle_distances = distances::pair_wise_distance_vectors( locations );
    auto interparticle_distance_magnitudes = distances::pair_wise_distance_magnitude(
        interparticle_distances
        );
    auto interparticle_unit_distances = distances::pair_wise_distance_unit_vectors( locations );
    auto interparticle_reduced_distance_magnitudes = interparticle_distance_magnitudes;
    for( auto &i : interparticle_reduced_distance_magnitudes )
        for( auto &j : i )
            j /= std::pow( average_volume, 1./3 );

    // Get simulation properties
    double time_step = params.at("simulation").at("time-step");
    bool renorm = params.at("simulation").at("renormalisation");
    int max_samples = params.at("output").at("max-samples");
    size_t ensemble_size = params.at("simulation").at("ensemble-size");
    bool steady_cycle = params.at("simulation").at("steady-cycle-activated");

    /**
     * LANGEVIN DYNAMICS
     * The initial state of each system is assumed to be the same
     */
    std::vector<std::array<double,3> > system_state;
    for( auto &p: particles )
    {
        std::array<double,3> init{
            p.at("magnetisation-direction")[0],
            p.at("magnetisation-direction")[1],
            p.at("magnetisation-direction")[2] };
        system_state.push_back( init );
    }

    std::vector<std::vector<std::array<double,3> > > initial_system_state;
    for( unsigned int i=0; i<ensemble_size; i++ )
        initial_system_state.push_back( system_state );

    /**
     * LANGEVIN DYNAMICS
     * Each run of the simulation within the ensemble uses a different
     * seed value. The list of seeds in the config json are used to
     * create a vector of random number generators. One for each
     * simulation.
     */
    rng_vec rngs;
    for( unsigned int i=0; i<n_runs; i++ )
        rngs.push_back( std::make_shared<RngMtNorm>(
                            params.at("simulation").at("seeds")[i],
                            std::sqrt(
                                params.at("simulation").at("time-step")
                                .get<double>() ) ) );

    /**
     * In steady state mode the simulation time is ignored and it is
     * taken to be a single cycle of the applied magnetic field
     * \f$1/f\f$
     */
    double simulation_time = steady_cycle
        ? 1 / params.at("global").at("applied-field").at("reduced-frequency").get<double>()
        : params.at("simulation").at("simulation-time").get<double>();

    /**
     * Langevin dynamics run function accepts two arguments. The
     * initial state of the system for each run and a random number
     * generator
     * The run function must always return the system state
     */
    std::function<std::vector<simulation::results>(std::vector<d3>, std::shared_ptr<Rng>)> run_function =
        [thermal_field_strengths, reduced_anisotropy_constants, reduced_particle_volumes,
         axes, interparticle_unit_distances, interparticle_reduced_distance_magnitudes,
         happ, average_anisotropy, average_volume, damping, saturation_magnetisation,
         time_step, simulation_time, renorm, max_samples]
        (std::vector<d3> initial_system_state, std::shared_ptr<Rng> rng)
        {
            return simulation::full_dynamics(
                thermal_field_strengths,
                reduced_anisotropy_constants,
                reduced_particle_volumes,
                axes,
                initial_system_state,
                interparticle_unit_distances,
                interparticle_reduced_distance_magnitudes,
                happ,
                average_anisotropy,
                average_volume,
                damping,
                saturation_magnetisation,
                time_step,
                simulation_time,
                *(rng.get()),
                renorm,
                max_samples );
        };

    // No cycle mode for now
    // simulation::results results( max_samples );
    // if( steady_cycle )
    // {
    //     double steady_state_condition = 1e-3;
    //     results = simulation::steady_state_cycle_dynamics(
    //         run_function,
    //         max_samples,
    //         steady_state_condition,
    //         initial_system_state,
    //         rngs );
    // }
    // else
    // auto results = simulation::ensemble_run(
    //     max_samples,
    //     run_function,
    //     initial_system_state,
    //     rngs );
    auto results = simulation::full_dynamics(
        thermal_field_strengths,
        reduced_anisotropy_constants,
        reduced_particle_volumes,
        axes,
        initial_system_state[0],
        interparticle_unit_distances,
        interparticle_reduced_distance_magnitudes,
        happ,
        average_anisotropy,
        average_volume,
        damping,
        saturation_magnetisation,
        time_step,
        simulation_time,
        *(rngs[0].get()),
        renorm,
        max_samples );
    // auto results2 = simulation::results( max_samples );
    // auto run_funcs = curry::vector_curry( run_function, initial_system_state, rngs );
    // for( unsigned int i=0; i<initial_system_state.size(); i++ )
    // {
    //     auto results = run_funcs[i]();
    //     for( int j=0; j<max_samples; j++ )
    //     {
    //         results2.mx[j] = results.mx[j];
    //         results2.my[j] = results.my[j];
    //         results2.mz[j] = results.mz[j];
    //         results2.time[j] = results.time[j];
    //         results2.field[j] = results.field[j];
    //     }
    // }
    // results2.energy_loss = 0;


    // save the results
    std::stringstream fname;
    for( unsigned int i=0; i<results.size(); i++ )
    {
        fname.str("");
        fname << params.at("output").at("directory").get<std::string>()
              << "/results" << i;
        simulation::save_results( fname.str(), results[i] );
    }



    // Compute the power emitted by the particle ensemble
    // double f = params["global"]["applied-field"]["frequency"];
    // double Ms = params["particle"]["magnetisation"];
    // double Hk = params["global"]["anisotropy-field"];
    // double power = results.energy_loss * f * Ms * Hk;

    // json output;
    // output["power"] = power;

    /**
     * The current git commit version is written to the file. Made
     * available through a compiler flag (see makefile)
     */
    json output;
    output["git-version"] = VERSION;

    // Write the normalised parameters to file
    std::ostringstream output_fname;
    output_fname << params.at("output").at("directory").get<std::string>()
                 << "/output.json";
    moma_config::write( output_fname.str(), output );
}
