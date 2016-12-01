// main.cpp
// Main executable for MOMA - a modern, open-source, magnetic
// materials simulator... again.
//
// Oliver W. Laslett (2016)
// O.Laslett@soton.ac.uk

#include <fstream>
#include <iostream>
#include <array>
#include <iostream>
#include <random>
#include <fenv.h>
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
    config_out_fname << norm_config["output"]["directory"] << "/config_norm.json";
    std::ofstream out_config_stream( config_out_fname.str() );
    out_config_stream << norm_config;
    out_config_stream.close();

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

    // Create the random number generator
    std::mt19937_64 rng( 1001 );

    // Run the simulation
    auto results = simulation::full_dynamics(
        norm_config["particle"]["damping"],
        norm_config["particle"]["thermal-field-strength"],
        happ, init,
        norm_config["simulation"]["time-step"],
        norm_config["simulation"]["simulation-time"],
        rng );





    // start with single particle
    // define the struct from the config

    // normalisation of all parameters

    // simulate the llg for N steps

    // Write to file
    return 0;
}
