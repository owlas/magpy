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
      Takes a json and pretty prints it to a file.
      Returns 0 if success else -1
    */
    int write( std::string, const json );
}
#endif
