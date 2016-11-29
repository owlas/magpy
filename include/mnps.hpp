// mnps.hpp
//
// structs for magnetic nanoparticles
#ifndef MNP_H
#define MNP_H

#include <array>

namespace mnp
{
    using axis = std::array<double,3>;
    struct params {
        double gamma;
        double alpha;
        double saturation_mag;
        double diameter;
        double anisotropy;
        axis anisotropy_axis;
    };

    struct norm_params {
        double gamma;
        double alpha;
        double stability;
        double volume;
        double temperature;
        axis anisotropy_axis;
    };
}
#endif
