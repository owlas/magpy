// normalisation.hpp
// Normalise system parameters for simulation
//
// Oliver W. Laslett (2016)
// O.Laslett@soton.ac.uk
#ifndef NORM_H
#define NORM_H
#include "json.hpp"
using json = nlohmann::json;

namespace normalisation
{
    json normalise( const json input );
}
#endif
