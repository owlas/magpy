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
#include "../include/moma_config.hpp"
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

    // Check the config
    moma_config::validate( config );

    // get the normalised system parameters write them to a file
    json norm_config = moma_config::normalise( config );

    // Write the normalised parameters to file
    std::ostringstream config_out_fname;
    std::string dir = norm_config["output"]["directory"];
    config_out_fname << dir << "/config_norm.json";
    moma_config::write( config_out_fname.str(), norm_config );

    // Run the simulation from the config file
    moma_config::launch_simulation( norm_config );

    return 0;
}
