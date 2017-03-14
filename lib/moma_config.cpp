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
#include "../include/constants.hpp"
#include "../include/easylogging++.h"
#include "../include/field.hpp"
#include "../include/simulation.hpp"
#include "../include/rng.hpp"

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
        LOG( FATAL ) << "Time_Step must be greater than 0";
    int ensemble_size = input.at("simulation").at("ensemble-size");
    if ( ensemble_size < 1 )
        LOG( FATAL ) << "Ensemble size must be interger value at least 1";
    std::vector<long> seeds = input.at("simulation").at("seeds");
    if ( seeds.size() < ensemble_size )
        LOG( FATAL ) << "Not enough seeds. Must have as many seeds as ensemble size.";

    // Check output
    std::string dir = input.at("output").at("directory");
    int max_samples = input.at("output").at("max-samples");
    if ( ( max_samples < 1 ) && ( max_samples != -1 ) )
        LOG( FATAL ) << "Valid values for max-samples are -1 or integers greater than 0";

    // Check global
    double temp = input.at("global").at("temperature");
    if( temp < 0 )
        LOG( FATAL ) << "Temperature must be greater than 0";
    std::string shape = input.at("global").at("applied-field").at("shape");
    if( ( shape.compare( "sine" ) != 0 )
        && ( shape.compare("square") != 0 )
        && ( shape.compare("square-fourier") != 0 ) )
        LOG( FATAL ) << "Valid values for shape are 'sine' or 'square'";
    if( shape.compare( "square-fourier" ) == 0 )
        if( input.at("global").at("applied-field").count("components") != 1 )
            LOG( FATAL ) << "If specifying Fourier series field, "
                         << "must specify number of components";
    double freq = input.at("global").at("applied-field").at("frequency");
    if( freq < 0 )
        LOG( FATAL ) << "Frequency must be greater than 0";
    double amp = input.at("global").at("applied-field").at("amplitude");

    // Check particle
    double damping = input.at("particle").at("damping");
    if( damping < 0 )
        LOG( FATAL ) << "Damping must be greater than 0";
    double radius = input.at("particle").at("radius");
    if( radius < 0 )
        LOG( FATAL ) << "Radius must be greater than 0";
    double anisotropy = input.at("particle").at("anisotropy");
    if( anisotropy < 0 )
        LOG( FATAL ) << "Anisotropy must be greater than 0";
    std::array<double,3> aaxis = {
        input.at("particle").at("anisotropy-axis")[0],
        input.at("particle").at("anisotropy-axis")[1],
        input.at("particle").at("anisotropy-axis")[2]
    };
    double aaxis_mag = aaxis[0]*aaxis[0] + aaxis[1]*aaxis[1] + aaxis[2]*aaxis[2];
    if( std::abs( aaxis_mag - 1 ) > 0.00001 )
        LOG( FATAL ) << "Anisotropy axis must be unit vector but has magnitude " << aaxis_mag;
    double magnetisation = input.at("particle").at("magnetisation");
    if( magnetisation < 0 )
        LOG( FATAL ) << "Magnetisation must be greater than 0";
    std::array<double,3> magaxis = {
        input.at("particle").at("magnetisation-direction")[0],
        input.at("particle").at("magnetisation-direction")[1],
        input.at("particle").at("magnetisation-direction")[2]
    };
    double magaxis_mag = magaxis[0]*magaxis[0] + magaxis[1]*magaxis[1] + magaxis[2]*magaxis[2];
    if( std::abs( magaxis_mag - 1 ) > 0.00001 )
        LOG( FATAL ) << "Magnetisation direction axis must be unit vector but has magnitude" << magaxis_mag;
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
        LOG( FATAL ) << "Time_Step must be greater than 0";

    // Check output
    std::string dir = input.at("output").at("directory");
    int max_samples = input.at("output").at("max-samples");
    if ( ( max_samples < 1 ) && ( max_samples != -1 ) )
        LOG( FATAL ) << "Valid values for max-samples are -1 or integers greater than 0";

    // Check global
    double temp = input.at("global").at("temperature");
    if( temp < 0 )
        LOG( FATAL ) << "Temperature must be greater than 0";
    std::string shape = input.at("global").at("applied-field").at("shape");
    if( ( shape.compare( "sine" ) != 0 )
        && ( shape.compare("square") != 0 )
        && ( shape.compare("square-fourier") !=0 ) )
        LOG( FATAL ) << "Valid values for shape are 'sine' or 'square'";
    if( shape.compare( "square-fourier" ) == 0 )
        if( input.at("global").at("applied-field").count("components") != 1 )
            LOG( FATAL ) << "If specifying Fourier series field, "
                         << "must specify number of components";
    double freq = input.at("global").at("applied-field").at("frequency");
    if( freq < 0 )
        LOG( FATAL ) << "Frequency must be greater than 0";
    double amp = input.at("global").at("applied-field").at("amplitude");
    double tau0 = input.at("global").at("tau0");

    // Check particles
    if( !input.at("particles").is_array() )
        LOG( FATAL ) << "particles should be an array of particles!";

    std::vector<json> particles = input.at("particles");
    for( auto &p : particles )
    {
        double radius = p.at("radius");
        if( radius < 0 )
            LOG( FATAL ) << "Radius must be greater than 0";
        double anisotropy = p.at("anisotropy");
        if( anisotropy < 0 )
            LOG( FATAL ) << "Anisotropy must be greater than 0";
        std::array<double,2> probs = {
            p.at("initial-probs")[0],
            p.at("initial-probs")[1]
        };
        double probs_sum = probs[0] + probs[1];
        if( std::abs( probs_sum - 1 ) > 0.00001 )
            LOG( FATAL ) << "Initial probabilities must sum to 1 but sum to " << probs_sum;
        double magnetisation = p.at("magnetisation");
        if( magnetisation < 0 )
            LOG( FATAL ) << "Magnetisation must be greater than 0";
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
    bool steady_cycle; double sim_time;
    if( in.at("simulation").at("simulation-time").is_string() )
        steady_cycle = true;
    else {
        steady_cycle = false;
        sim_time = in.at("simulation").at("simulation-time").get<double>();
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


    // Compute particle volume
    double rad = in["particle"]["radius"];
    double volume = 4.0 / 3.0 * M_PI * rad * rad * rad;
    out["particle"]["volume"] = volume;

    // Compute particle stability ratio
    double k = in["particle"]["anisotropy"];
    double T = in["global"]["temperature"];
    double stability = k*volume/constants::KB/T;
    out["particle"]["stability-ratio"] = stability;

    // anisotropy field and reduced field
    double ms = in["particle"]["magnetisation"];
    double H = in["global"]["applied-field"]["amplitude"];
    double Hk = 2*k/constants::MU0/ms;
    double h = H/Hk;
    out["global"]["anisotropy-field"] = Hk;
    out["global"]["applied-field"]["amplitude"] = h;

    // Compute the time rescaling
    double damping = in["particle"]["damping"];
    double time_factor = constants::GYROMAG * constants::MU0 * Hk
        / ( 1+damping*damping );
    out["simulation"]["time-factor"] = time_factor;

    // Rescale time dependent parameters
    double time_step = in["simulation"]["time-step"];
    double tau_step = time_step * time_factor;
    out["simulation"]["time-step"] = tau_step;
    double f = in["global"]["applied-field"]["frequency"];
    double f_in_tau = f / time_factor;
    out["global"]["applied-field"]["frequency"] = f_in_tau;
    if( !steady_cycle )
    {
        double sim_tau = sim_time * time_factor;
        out["simulation"]["simulation-time"] = sim_tau;
    }

    // Compute the thermal field strength
    double therm_strength = std::sqrt(
        damping * constants::KB * T
        / ( k * volume * ( 1 + damping * damping ) ) );
    out["particle"]["thermal-field-strength"] = therm_strength;

    return out;
}

int moma_config::write( const std::string fname, const json output )
{
    std::ofstream os( fname );
    if( os.is_open() )
    {
        os << std::setw(4) << output;
        os.close();
        LOG(INFO) << "Successfully wrote json file: " << fname;
        return 0;
    }
    else
    {
        LOG(ERROR) << "Error opening file to write json: " << fname;
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
        LOG( FATAL ) << "Simulation-mode must be 'llg' or 'dom'";
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
            LOG(FATAL) << field_shape << " is not a valid field shape.";

        happs.push_back( happ );
    }

    // Get the temperature
    double temperature = params.at("global").at("temperature");

    // Get the particle properties
    std::vector<double> volumes;
    std::vector<double> anisotropies;
    std::vector<double> magnetisations;
    for( unsigned int i=0; i<n_particles; i++ )
    {
        volumes.push_back(particles[i].at("volume"));
        anisotropies.push_back(particles[i].at("anisotropy"));
        magnetisations.push_back(particles[i].at("magnetisation"));
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
    std::function<simulation::results(d3,double,double,double,std::function<double(double)>)> run_function=
        [temperature, tau0, time_step, sim_time, max_samples]
        (d3 initial_mag, double volume, double anisotropy, double ms, std::function<double(double)> happ)
        {
            // Convert initial magnetisation into states
            std::array<double,2> initial_probs = {
                0.5*( initial_mag[2] + 1 ),
                0.5*( 1 - initial_mag[2] )
            };

            return simulation::dom_ensemble_dynamics(
                volume, anisotropy, temperature, tau0, ms, happ,
                initial_probs, time_step, sim_time, max_samples );
        };

    // Run the simulation
    if( steady_cycle )
    {
        double steady_state_condition = 1e-4;
        results = simulation::steady_state_cycle_dynamics(
            run_function,
            max_samples,
            steady_state_condition,
            init_mags,
            volumes,
            anisotropies,
            magnetisations,
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
            params["global"]["applied-field"]["amplitude"],
            params["global"]["applied-field"]["frequency"],
            std::placeholders::_1 );
    else if( field_shape.compare("square") == 0 )
        happ = std::bind(
            field::square,
            params["global"]["applied-field"]["amplitude"],
            params["global"]["applied-field"]["frequency"],
            std::placeholders::_1 );
    else if( field_shape.compare("square-fourier") == 0 )
        happ = std::bind(
            field::square_fourier,
            params["global"]["applied-field"]["amplitude"],
            params["global"]["applied-field"]["frequency"],
            params["global"]["applied-field"]["components"],
            std::placeholders::_1 );
    else
        LOG(FATAL) << field_shape << " is not a valid field shape.";

    // Get the uniaxial anisotropy axis
    std::array<double,3> aaxis{
        params["particle"]["anisotropy-axis"][0],
            params["particle"]["anisotropy-axis"][1],
            params["particle"]["anisotropy-axis"][2]
            };

    // Run the simulation N times for a full ensemble
    size_t n_runs = params["simulation"]["ensemble-size"];

    // Get the particle properties
    double damping = params.at("particle").at("damping");
    double thermal_field = params.at("particle").at("thermal-field-strength");

    // Get simulation properties
    double time_step = params.at("simulation").at("time-step");
    bool renorm = params.at("simulation").at("renormalisation");
    int max_samples = params.at("output").at("max-samples");
    size_t ensemble_size = params.at("simulation").at("ensemble-size");
    bool steady_cycle = params.at("simulation").at("steady-cycle-activated");

    /**
     * LANGEVIN DYNAMICS
     * The initial state of the system is assumed to be the same for
     * all particles.
     */
    std::array<double,3> init{
        params["particle"]["magnetisation-direction"][0],
            params["particle"]["magnetisation-direction"][1],
            params["particle"]["magnetisation-direction"][2] };
    std::vector<d3> init_mags;
    for( unsigned int i=0; i<ensemble_size; i++ )
        init_mags.push_back( init );

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
        ? 1 / params.at("global").at("applied-field").at("frequency").get<double>()
        : params.at("simulation").at("simulation-time").get<double>();

    /**
     * Langevin dynamics run function accepts two arguments. The
     * initial state of the system for each run and a random number
     * generator
     */
    std::function<simulation::results(d3, std::shared_ptr<Rng>)> run_function =
        [damping,thermal_field,aaxis,happ,time_step,simulation_time,renorm,max_samples,ensemble_size]
        (d3 initial_mag, std::shared_ptr<Rng> rng)
        {
            return simulation::full_dynamics(
                damping, thermal_field, aaxis, happ, initial_mag, time_step,
                simulation_time, *(rng.get()), renorm, max_samples );
        };

    simulation::results results( max_samples );
    if( steady_cycle )
    {
        double steady_state_condition = 1e-3;
        results = simulation::steady_state_cycle_dynamics(
            run_function,
            max_samples,
            steady_state_condition,
            init_mags,
            rngs );
    }
    else
        results = simulation::ensemble_run(
            max_samples,
            run_function,
            init_mags,
            rngs );

    // save the results
    std::stringstream fname;
    fname << params.at("output").at("directory").get<std::string>()
          << "/results";
    simulation::save_results( fname.str(), results );


    // Compute the power emitted by the particle ensemble
    double f = params["global"]["applied-field"]["frequency"];
    double Ms = params["particle"]["magnetisation"];
    double Hk = params["global"]["anisotropy-field"];
    double power = results.energy_loss * f * Ms * Hk;

    json output;
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
