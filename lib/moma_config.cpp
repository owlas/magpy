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
#include "../include/constants.hpp"
#include "../include/easylogging++.h"
#include "../include/field.hpp"
#include "../include/simulation.hpp"
#include "../include/rng.hpp"

using d3=std::array<double,3>;
using rng_vec=std::vector<std::shared_ptr<Rng>>;

json moma_config::normalise( const json in )
{
    // Simulation parameters
    double sim_time = in["simulation"]["simulation-time"];
    double time_step = in["simulation"]["time-step"];
    bool renorm = in.at("simulation").at("renormalisation");
    bool steady_state = in.at("simulation").at("enable-steady-state-condition");

    // Get output parameters
    std::string dir = in["output"]["directory"];

    // Global system parameters
    double T = in["global"]["temperature"];
    double H = in["global"]["applied-field"]["amplitude"];
    double f = in["global"]["applied-field"]["frequency"];

    // Get particle parameters
    double damping = in["particle"]["damping"];
    double rad = in["particle"]["radius"];
    double k = in["particle"]["anisotropy"];
    d3 k_vec = { in["particle"]["anisotropy-axis"][0],
                 in["particle"]["anisotropy-axis"][1],
                 in["particle"]["anisotropy-axis"][2] };
    d3 mag = { in["particle"]["initial-magnetisation"][0],
               in["particle"]["initial-magnetisation"][1],
               in["particle"]["initial-magnetisation"][2] };
    double ms = std::sqrt(
        mag[0]*mag[0] + mag[1]*mag[1] + mag[2]*mag[2] );

    // stability ratio of particle
    double volume = 4.0 / 3.0 * M_PI * rad * rad * rad;
    double stability = k*volume/constants::KB/T;

    // anisotropy field and reduced field
    double Hk = 2*k/constants::MU0/ms;
    double h = H/Hk;

    // rescale time
    double time_factor = constants::GYROMAG * constants::MU0 * Hk
        / ( 1+damping*damping );
    double sim_tau = sim_time * time_factor;
    double tau_step = time_step * time_factor;
    double f_in_tau = f / time_factor;

    // normalised thermal field strength
    double therm_strength = std::sqrt(
        damping * constants::KB * T
        / ( k * volume * ( 1 + damping * damping ) ) );

    // reduced magnetisation
    d3 unit_mag{ mag[0]/ms, mag[1]/ms, mag[2]/ms };

    // write the reduced simulation parameters
    json out = {
        {"simulation", {
                {"ensemble-size", in["simulation"]["ensemble-size"]},
                {"simulation-time", sim_tau},
                {"time-step", tau_step},
                {"time-factor", time_factor},
                {"renormalisation", renorm},
                {"enable-steady-state-condition", steady_state},
                {"seeds", in.at("simulation").at("seeds")},
                {"max-samples", in.at("simulation").at("max-samples")}
            }},
        {"output", {
                {"directory", dir}
            }},
        {"global", {
                {"temperature", T},
                {"applied-field", {
                        {"frequency", f_in_tau},
                        {"shape", in["global"]["applied-field"]["shape"]},
                        {"amplitude", h},
                    }},
                 {"anisotropy-field", Hk}
            }},
        {"particle", {
                {"damping", damping},
                {"volume", volume},
                {"radius", rad},
                {"anisotropy", k},
                {"anisotropy-axis", {k_vec[0], k_vec[1], k_vec[2]}},
                {"initial-magnetisation", {unit_mag[0], unit_mag[1], unit_mag[2]}},
                {"thermal-field-strength", therm_strength},
                {"stability-ratio", stability},
                {"saturation-magnetisation", ms}
        }}
    };

    LOG(INFO) << "Normalised simulation parameters. New time-step->"
              << tau_step << " simulation-time->" << sim_tau;
    return out;
}

int moma_config::validate( const json input )
{
    return 0;
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

void moma_config::launch_simulation( const json norm_config )
{
    // Bind the applied field waveform
    // assumes sinusoidal for now
    std::function<double(double)> happ = std::bind(
        field::sinusoidal,
        norm_config["global"]["applied-field"]["amplitude"],
        norm_config["global"]["applied-field"]["frequency"],
        std::placeholders::_1 );

    // Get the initial magnetisation
    std::array<double,3> init{
        norm_config["particle"]["initial-magnetisation"][0],
            norm_config["particle"]["initial-magnetisation"][1],
            norm_config["particle"]["initial-magnetisation"][2] };

    // Get the uniaxial anisotropy axis
    std::array<double,3> aaxis{
        norm_config["particle"]["anisotropy-axis"][0],
            norm_config["particle"]["anisotropy-axis"][1],
            norm_config["particle"]["anisotropy-axis"][2]
            };

    // Run the simulation N times for a full ensemble
    size_t n_runs = norm_config["simulation"]["ensemble-size"];

    // Create the random number generators
    rng_vec rngs;
    for( unsigned int i=0; i<n_runs; i++ )
        rngs.push_back( std::make_shared<RngMtNorm>(
                            norm_config.at("simulation").at("seeds")[i],
                            std::sqrt(
                                norm_config.at("simulation").at("time-step")
                                .get<double>() ) ) );

    LOG(INFO) << "Running " << n_runs << " simulations on "
              << omp_get_max_threads() << " threads";

    if( norm_config.at("simulation").at("enable-steady-state-condition") )
    {
        double cycle_time = 1 /
            norm_config.at("global").at("applied-field").at("frequency").get<double>();

        auto results = simulation::steady_state_cycle_dynamics(
            norm_config.at("particle").at("damping"),
            norm_config.at("particle").at("thermal-field-strength"),
            aaxis, happ, init,
            norm_config.at("simulation").at("time-step"),
            cycle_time, rngs,
            norm_config.at("simulation").at("renormalisation"),
            norm_config.at("simulation").at("max-samples"),
            norm_config.at("simulation").at("ensemble-size") );

        // save the results
        std::stringstream fname;
        fname << norm_config.at("output").at("directory").get<std::string>()
              << "/results";
        simulation::save_results( fname.str(), results );

        // Compute the power emitted by the particle ensemble
        double power = simulation::power_loss(
            results,
            norm_config["particle"]["volume"],
            norm_config["particle"]["anisotropy"],
            norm_config["particle"]["saturation-magnetisation"],
            norm_config["global"]["anisotropy-field"],
            norm_config["global"]["applied-field"]["frequency"] );

        json output;
        output["power"] = power;
        output["git-version"] = VERSION; // from compiler flag variable (see makefile)

        // Write the normalised parameters to file
        std::ostringstream output_fname;
        output_fname << norm_config.at("output").at("directory").get<std::string>()
                     << "/output.json";
        moma_config::write( output_fname.str(), output );

    } // end steady state ensemble dynamics sim
    else
    {
        auto results = simulation::ensemble_dynamics(
            norm_config.at("particle").at("damping"),
            norm_config.at("particle").at("thermal-field-strength"),
            aaxis, happ, init,
            norm_config.at("simulation").at("time-step"),
            norm_config.at("simulation").at("simulation-time"),
            rngs,
            norm_config.at("simulation").at("renormalisation"),
            norm_config.at("simulation").at("max-samples"),
            n_runs );

        // save the results
        std::stringstream fname;
        fname << norm_config.at("output").at("directory").get<std::string>()
              << "/results";
        simulation::save_results( fname.str(), results );
        LOG(INFO) << "Successfully wrote results file to: " << fname.str();

        json output;
        output["git-version"] = VERSION; // from compiler flag variable (see makefile)

        // Write the normalised parameters to file
        std::ostringstream output_fname;
        output_fname << norm_config.at("output").at("directory").get<std::string>()
                     << "/output.json";
        moma_config::write( output_fname.str(), output );
    } // end ensemble dynamics sim
}
