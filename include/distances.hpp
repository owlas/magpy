#ifndef DIST_H
#define DIST_H

#include<vector>
#include<array>

namespace distances
{
    std::vector<std::vector<std::array<double,3> > > pair_wise_distance_vectors(
        std::vector<std::array<double,3> > points);

    std::vector<std::vector<double> > pair_wise_distance_magnitude(
        std::vector<std::vector<std::array<double,3> > > distance_vectors );

    std::vector<std::vector<double> > pair_wise_distance_magnitude(
        std::vector<std::array<double,3> > points );
}
#endif
