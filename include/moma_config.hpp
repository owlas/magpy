// moma_config.hpp
// Definitions of json schemas and functions for output and config
//
// Oliver W. Laslett (2016)
// O.Laslett@soton.ac.uk
#ifndef JSONS_H
#define JSONS_H
#include <map>
#include <string>
#include "json.hpp"
#include "simulation.hpp"
using json = nlohmann::json;

namespace moma_config
{
    /*
      Validate that the config is correctly formatted for the
      simulation type.
    */
    void validate_for_dom( const json input );

    /*
      Takes a json config file and normalises the parameters for
      simulation. Returns the normalised params.
    */
    json transform_input_parameters_for_dom( const json input );

    /*
      Takes a json and pretty prints it to a file.
      Returns 0 if success else -1
    */
    int write( const std::string, const json );

    /*
      Provides an interface between normalised json config files and
      simulation functions. Main entry point for launching simulations
      from the CLI.
    */
    void launch_simulation( const json input );
    void launch_dom_simulation( const json input );
}
#endif
