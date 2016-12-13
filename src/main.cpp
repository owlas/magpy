// main.cpp
// Main executable for MOMA - a modern, open-source, magnetic
// materials simulator... again.
//
// Oliver W. Laslett (2016)
// O.Laslett@soton.ac.uk

#include <fstream>
#include <array>
#include <iostream>
#include <random>
#include <fenv.h>
#include <omp.h>
#include "../include/normalisation.hpp"
#include "../include/field.hpp"
#include "../include/simulation.hpp"
#include "../include/json.hpp"
#include "../include/easylogging++.h"

using json = nlohmann::json;

INITIALIZE_EASYLOGGINGPP
void print_help()
{
    std::cout << "MOMA - Modern, open-source, magnetics... again" << std::endl;
    std::cout << "Usage:" << std::endl;
    std::cout << "$ ./moma INPUT" << std::endl;
    std::cout << "    - INPUT: json config file" << std::endl;
}

// specify the json on the command line (add args)
int main( int argc, char *argv[] )
{
    START_EASYLOGGINGPP( argc, argv );

    feraiseexcept(FE_ALL_EXCEPT & ~FE_INEXACT );

    // Check that enough arguments have been supplied
    if( argc < 2 )
    {
        print_help();
        return 1;
    }

    // Read in the json config
    json config;
    std::ifstream filestream( argv[1] );
    config << filestream;
    filestream.close();

    // get the normalised system parameters write them to a file
    json norm_config = normalisation::normalise( config );

    // Write the normalised parameters to file
    std::ostringstream config_out_fname;
    std::string dir = norm_config["output"]["directory"];
    config_out_fname << dir << "/config_norm.json";
    std::ofstream out_config_stream( config_out_fname.str() );
    if( out_config_stream.is_open() )
    {
        out_config_stream << std::setw(4) << norm_config;
        out_config_stream.close();
        LOG(INFO) << "Wrote normalised simulation parameters to: "
                  << config_out_fname.str();
    }
    else
        LOG(ERROR) << "Error opening file to write normalised simulation parameters: "
                   << config_out_fname.str();


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

    LOG(INFO) << "Running " << n_runs << " simulations on "
              << omp_get_max_threads() << " threads";
#pragma omp parallel for schedule(dynamic,1)
    for( unsigned int i=0; i<n_runs; i++ )
    {
        // Create the random number generator
        unsigned long seed = config["simulation"]["seeds"][i];
        std::mt19937_64 rng( seed );

        auto results = simulation::full_dynamics(
            norm_config["particle"]["damping"],
            norm_config["particle"]["thermal-field-strength"],
            aaxis, happ, init,
            norm_config["simulation"]["time-step"],
            norm_config["simulation"]["simulation-time"],
            rng,
            config["simulation"]["max-samples"]);

        // Write the results to disk
        std::stringstream fname;
        fname << "output/results" << i;
        simulation::save_results( fname.str(), results );

        // Log progress
        #pragma omp critical
        LOG(INFO) << "Simulation " << i << " - saved " << results.N
                  << " steps to: " << fname.str();
    }
    return 0;
}
