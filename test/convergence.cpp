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
    LOG(INFO) << "Running convergence tests...";
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
    //moma_config::validate_for_llg( config );
    json norm_config = moma_config::transform_input_parameters_for_llg( config );

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

    // Generate an ensemble of solutions for each time step
    const size_t N_time_steps = 6;

    // No applied field
    auto happ = [](double) { return 0; };

    // Store the final result of the z-axis for each simulation for
    // each time step
    auto sols = new double[n_runs*N_time_steps*3];

    // Run the simulation for each particle in the ensemble
    // For each particle we solve the system for the same wiener paths
    // with different sampling rates.
    LOG(INFO) << "Running test body...";
    for( unsigned i=0; i<n_runs; i++ )
    {
        // Each time step is double the previous
        for( unsigned int ts_factor=0; ts_factor<N_time_steps; ts_factor++ )
        {
            // Calculate the time step multiplier and time step
            int N_mul = std::pow( 2, ts_factor );
            double base_time_step = norm_config["simulation"]["time-step"];
            double time_step = base_time_step * N_mul;

            // Uses the same seed for each run and down samples to
            // create samples of the same Wiener path
            size_t dim=3;
            RngMtDownsample rng( 888+i, std::sqrt( base_time_step ), dim, N_mul );

            auto results = simulation::full_dynamics(
                norm_config["particle"]["damping"],
                norm_config["particle"]["thermal-field-strength"],
                aaxis, happ, init, time_step,
                norm_config["simulation"]["simulation-time"],
                rng,
                config["simulation"]["renormalisation"],
                config["output"]["max-samples"] );
            // Store the results
            sols[i*N_time_steps*3 + ts_factor*3 + 0] = results.mx[results.N-1];
            sols[i*N_time_steps*3 + ts_factor*3 + 1] = results.my[results.N-1];
            sols[i*N_time_steps*3 + ts_factor*3 + 2] = results.mz[results.N-1];
        } // end for each time step multiplier

    } // end for each path

    // Write results to disk
    std::ofstream outputfile( "output/convergence.out" );
    if ( outputfile.is_open() )
    {
        for( unsigned int i=0; i<n_runs*N_time_steps*3; i++ )
            outputfile << sols[i] << " ";
        outputfile.close();
        LOG(INFO) << "Successfully wrote results to disk: ./output/convergence.out";
    }
    else
        LOG(FATAL) << "Failed to write results to disk: ./output/convergence.out";

    std::ofstream dtfile( "output/convergence.dt" );
    if( dtfile.is_open() )
    {
        for( unsigned int i=0; i<N_time_steps; i++ )
            dtfile << std::pow( 2, i )*norm_config["simulation"]["time-step"].get<double>()
                   << " ";
        dtfile.close();
        LOG(INFO) << "Successfully wrote results to disk: ./output/convergence.dt";
    }
    else
        LOG(FATAL) << "Failed to write results to disk: ./output/convergence.dt";

    delete[] sols;
    LOG(INFO) << "Convergence tests completed!";
} // end main
