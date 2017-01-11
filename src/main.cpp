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
#include "../include/moma_config.hpp"
#include "../include/field.hpp"
#include "../include/simulation.hpp"
#include "../include/json.hpp"
#include "../include/easylogging++.h"
#include "../include/io.hpp"
#include "../include/rng.hpp"

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

    // We will average the magnetisation directions into an ensemble
    struct simulation::results ensemble(
        config.at("simulation").at("max-samples").get<int>() );
    simulation::zero_results( ensemble );


    #pragma omp parallel for schedule(dynamic,1) shared(ensemble)
    for( unsigned int i=0; i<n_runs; i++ )
    {
        // Create the random number generator
        unsigned long seed = config["simulation"]["seeds"][i];
        RngMtNorm rng(
            seed, std::sqrt(
                norm_config.at("simulation").at("time-step").get<double>() ) );

        auto results = simulation::full_dynamics(
            norm_config["particle"]["damping"],
            norm_config["particle"]["thermal-field-strength"],
            aaxis, happ, init,
            norm_config["simulation"]["time-step"],
            norm_config["simulation"]["simulation-time"],
            rng,
            config["simulation"]["renormalisation"],
            config["simulation"]["max-samples"]);

        // Copy into the reduced results
        for( unsigned int j=0; j<results.N; j++ )
        {
            #pragma omp atomic
            ensemble.mx[j] += results.mx[j]; // maybe try with a critical
            #pragma omp atomic
            ensemble.my[j] += results.my[j];
            #pragma omp atomic
            ensemble.mz[j] += results.mz[j];
        }

        if( i==0 )
            for( unsigned int j=0; j<results.N; j++ )
            {
                ensemble.time[j] = results.time[j];
                ensemble.field[j] = results.field[j];
            }

        #pragma omp critical
        LOG(INFO) << "Completed simulation " << i << "/" << n_runs;

    } // end monte-carlo loop

    // Average the ensemble magnetisation
    for( unsigned int j=0; j<ensemble.N; j++ )
    {
        ensemble.mx[j] /= n_runs;
        ensemble.my[j] /= n_runs;
        ensemble.mz[j] /= n_runs;
    }

    // Write the full results to disk
    std::stringstream fname;
    fname << "output/results";
    simulation::save_results( fname.str(), ensemble );

    LOG(INFO) << "Ensemble trajectory of " << ensemble.N
              << " steps written to: " << fname.str();

    // Compute the power emitted by the particle ensemble
    double power = simulation::power_loss(
        ensemble,
        norm_config["particle"]["volume"],
        norm_config["particle"]["anisotropy"],
        norm_config["particle"]["saturation-magnetisation"],
        norm_config["global"]["anisotropy-field"],
        norm_config["global"]["applied-field"]["frequency"]);

    json output;
    output["power"] = power;
    output["git-version"] = VERSION; // from compiler flag variable (see makefile)

    // Write the normalised parameters to file
    std::ostringstream output_fname;
    output_fname << dir << "/output.json";
    moma_config::write( output_fname.str(), output );

    return 0;
}
