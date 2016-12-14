// moma_config.hpp
// Definitions of json schemas and functions for output and config
//
// Oliver W. Laslett (2016)
// O.Laslett@soton.ac.uk
#ifndef JSONS_H
#define JSONS_H
#include <map>
#include "json.hpp"
using json = nlohmann::json;

namespace moma_config
{
    /*
      Takes a json config file and normalises the parameters for
      simulation. Returns the normalised params.
    */
    json normalise( const json input );

    /*
      Validate that the config is correctly formatted.
      Returns 0 if ok. Returns -1 otherwise.
    */
    int validate( const json input );


    /*
      Enumerators for the different options in the json config.
    */
    enum ComputeOptions {
        Power,
        Full
    };
    extern std::map<std::string, ComputeOptions> map_compute_options;
}
#endif
