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
#include "../include/constants.hpp"
#include "../include/easylogging++.h"

using d3=std::array<double,3>;

json moma_config::normalise( const json in )
{
    // Simulation parameters
    double sim_time = in["simulation"]["simulation-time"];
    double time_step = in["simulation"]["time-step"];

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
                {"time-factor", time_factor}
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
                {"stability-ratio", stability}
        }}
    };

    LOG(INFO) << "Normalised simulation parameters. New time-step->"
              << tau_step << " simulation-time->" << sim_tau;
    return out;
}

std::map<std::string, moma_config::ComputeOptions>
moma_config::map_compute_options = {
    {"full", Full},
    {"power", Power}
};

int moma_config::validate( const json input )
{
    // Confirm a valid compute option
    if ( map_compute_options.count( input["simulation"]["compute"] ) == 0)
    {
        LOG(ERROR) << "Simulation.compute value of "
                   << input["simulation"]["compute"]
                   << " is not valid.";
        return -1;
    }
    LOG(INFO) << "Json config validated";
    return 0;
}
