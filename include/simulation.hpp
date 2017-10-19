// simulation.hpp
// high level functions for simulating magnetic dynamics
//
// Oliver W. Laslett (2016)
// O.Laslett@soton.ac.uk
#ifndef SIM_H
#define SIM_H
#include <memory>
#include <functional>
#include <array>
#include <random>
#include <string>
#include <cstdlib>
#include <vector>
#include "rng.hpp"
#include "field.hpp"

using d3 = std::array<double,3>;
using rng_vec=std::vector<std::shared_ptr<Rng>, std::allocator<std::shared_ptr<Rng> > >;
using sde_jac = std::function<void(double*,double*,double*,double*,const double*, const double, const double)>;


namespace simulation
{
    // This struct holds the results from the simulation
    // data is dynamically allocated using unique_ptrs
    // magnetisation - [Nx3] x,y,z compononents of the magnetisation
    // field - [N] the value of the field at time t_n
    // time - [N] the time at n
    // N - the number of steps taken
    // energyloss - the energy lost in the simulation
    struct results {
        std::unique_ptr<double[]> mx;
        std::unique_ptr<double[]> my;
        std::unique_ptr<double[]> mz;
        std::unique_ptr<double[]> field;
        std::unique_ptr<double[]> time;
        size_t N;

        results( size_t _N ) {
            N=_N;
            mx = std::unique_ptr<double[]>( new double[N] );
            my = std::unique_ptr<double[]>( new double[N] );
            mz = std::unique_ptr<double[]>( new double[N] );
            field = std::unique_ptr<double[]>( new double[N] );
            time = std::unique_ptr<double[]>( new double[N] );
        }
    };

    // Run the full dynamics and return the results
    std::vector<struct results> full_dynamics(
        const std::vector<double> thermal_field_strengths,
        const std::vector<double> reduced_anisotropy_constants,
        const std::vector<double> reduced_particle_volumes,
        const std::vector<d3> anisotropy_unit_axes,
        const std::vector<d3> initial_magnetisations,
        const std::vector<std::vector<d3> > interparticle_unit_distances,
        const std::vector<std::vector<double> > interparticle_reduced_distance_magnitudes,
        const std::function<double(const double)> applied_field,
        const double average_anisotropy,
        const double average_volume,
        const double damping_constant,
        const double saturation_magnetisation,
        const double time_step,
        const double end_time,
        Rng &rng,
        const bool renorm,
        const bool interactions,
        const bool use_implicit,
        const double eps,
        const int max_samples
    );

    std::vector<results> full_dynamics(
        const std::vector<double> radius,
        const std::vector<double> anisotropy,
        const std::vector<d3> anisotropy_axes,
        const std::vector<d3> magnetisation_direction,
        const std::vector<d3> location,
        const double magnetisation,
        const double damping,
        const double temperature,
        const bool renorm,
        const bool interactions,
        const bool use_implicit,
        const double eps,
        const double time_step,
        const double end_time,
        const size_t max_samples,
        const long seed,
        const field::options field_option=field::CONSTANT,
        const double field_amplitude=0.0,
        const double field_frequency=0.0
    );


    /*
      Computes the probability trajectories based on a master equation
      approximation
    */
    struct results dom_ensemble_dynamics(
        const double volume,
        const double anisotropy,
        const double temperature,
        const double magnetisation,
        const double alpha,
        const std::function<double(double)> applied_field,
        const std::array<double,2> initial_mags,
        const double time_step,
        const double end_time,
        const int max_samples );

    // Save a results file
    void save_results( const std::string fname, const struct results& );

    // Sets all of the arrays in the results struct to zero
    void zero_results( struct results& );

    void reduce_to_system_magnetisation(
        double *mag, const double *particle_mags, const size_t N_particles );


}
#endif
