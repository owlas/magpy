// convergence.cpp
// Convergence tests for solvers in moma package
//
// Oliver W. Laslett
// O.Laslett@soton.ac.uk

#include <random>
#include <functional>
#include <fstream>
#include <cstdlib>
#include "../include/easylogging++.h"
#include "../include/moma_config.hpp"
#include "../include/simulation.hpp"
#include "../include/stochastic_processes.hpp"
#include "../include/io.hpp"

INITIALIZE_EASYLOGGINGPP
int main( int argc, char *argv[] )
{
    LOG(INFO) << "Running equilibrium tests...";
    LOG(INFO) << "Reading in json...";
    // Read in the json config
    json config;
    std::ifstream filestream( argv[1] );
    if( filestream.is_open() )
    {
        config << filestream;
        filestream.close();
    }
    else
        LOG(FATAL) << "Could not open config json file: " << argv[1];

    LOG(INFO) << "Initialising simulation parameters";

    // Validate the config and transform params
    json norm_config = moma_config::transform_input_parameters_for_llg( config );

    // Write the normalised parameters to file
    std::ostringstream config_out_fname;
    std::string dir = norm_config["output"]["directory"];
    config_out_fname << dir << "/equilibrium_normalised.json";
    moma_config::write( config_out_fname.str(), norm_config );

    // Get the initial magnetisation
    std::array<double,3> init{
        norm_config["particle"]["magnetisation-direction"][0],
            norm_config["particle"]["magnetisation-direction"][1],
            norm_config["particle"]["magnetisation-direction"][2] };

    // Get the uniaxial anisotropy axis
    std::array<double,3> aaxis{
        norm_config["particle"]["anisotropy-axis"][0],
            norm_config["particle"]["anisotropy-axis"][1],
            norm_config["particle"]["anisotropy-axis"][2]
            };

    // Run the simulation N times for a full ensemble
    size_t n_runs = norm_config["simulation"]["ensemble-size"];

    // No applied field
    auto happ = [](double) { return 0; };

    // Store the final result of the z-axis for each simulation for
    // each time step
    auto sols = new double[n_runs];

    // Create a vector of RNGs
    std::vector<std::shared_ptr<Rng> > rngs;
    for( unsigned int i=0; i<n_runs; i++ )
    {
        unsigned int seed = i;
        double time_step = norm_config.at("simulation").at("time-step");
        rngs.push_back(
            std::make_shared<RngMtNorm>( seed, std::sqrt(time_step) ) );
    }

    // Run the same simulation many times with a different random number generator.
    // Create run function here
    std::function<simulation::results(std::shared_ptr<Rng>)> run_function =
                  [=]( std::shared_ptr<Rng> rng )
        {
            return simulation::full_dynamics(
                norm_config["particle"]["damping"],
                norm_config["particle"]["thermal-field-strength"],
                aaxis, happ, init,
                norm_config.at("simulation").at("time-step").get<double>(),
                norm_config["simulation"]["simulation-time"],
                *rng,
                config["simulation"]["renormalisation"],
                config["output"]["max-samples"] );
        };

    // Execute the run_function many times with different rngs
    auto final_states = simulation::ensemble_run_final_state( run_function, rngs );
    for( unsigned int i=0; i<n_runs; i++ )
        sols[i] = final_states[i][2];

    // Write results to disk
    std::ostringstream outfilename;
    outfilename << norm_config.at("output").at("directory").get<std::string>()
                << "/equilibrium.out";
    std::ofstream outputfile( outfilename.str() );
    if ( outputfile.is_open() )
    {
        for( unsigned int i=0; i<n_runs; i++ )
            outputfile << std::setprecision(15) << sols[i] << " ";
        outputfile.close();
        LOG(INFO) << "Successfully wrote results to disk: " << outfilename.str();
    }
    else
        LOG(FATAL) << "Failed to write results to disk: " << outfilename.str();

    delete[] sols;
    LOG(INFO) << "Equilibrium tests completed!";
} // end main
