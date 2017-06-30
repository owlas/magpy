#include "../include/distances.hpp"
#include <cstdlib>
#include <cmath>

/// Computes the distance vector between points in 3d space
std::vector<std::vector<std::array<double,3> > > distances::pair_wise_distance_vectors(
    std::vector<std::array<double,3> > points)
{
    size_t n_points = points.size();
    std::vector<std::vector<std::array<double,3> > > distances;

    distances.resize( n_points );
    for( auto &vec : distances )
        vec.resize( n_points );

    for( unsigned int i=0; i<n_points; i++ )
        for( unsigned int j=0; j<n_points; j++ )
            for( unsigned int k=0; k<3; k++ )
                distances[i][j][k] = points[j][k] - points[i][k] ;
    return distances;
}

std::vector<std::vector<double> > distances::pair_wise_distance_magnitude(
    std::vector<std::vector<std::array<double,3> > > distance_vectors )
{
    size_t n_points = distance_vectors.size();
    std::vector<std::vector<double> > distances;
    distances.resize( n_points );
    for( auto &vec : distances )
        vec.resize( n_points );

    for( unsigned int i=0; i<n_points; i++ )
        for( unsigned int j=0; j<n_points; j++ )
            distances[i][j] = distances[j][i] = std::sqrt(
                distance_vectors[i][j][0] * distance_vectors[i][j][0]
                + distance_vectors[i][j][1] * distance_vectors[i][j][1]
                + distance_vectors[i][j][2] * distance_vectors[i][j][2] );
    return distances;
}

std::vector<std::vector<std::array<double,3> > > distances::pair_wise_distance_unit_vectors(
    std::vector<std::array<double,3> > points )
{
    auto distance_vectors = pair_wise_distance_vectors( points );
    auto distance_magnitudes = pair_wise_distance_magnitude( distance_vectors );
    return pair_wise_distance_unit_vectors( distance_vectors, distance_magnitudes );
}

std::vector<std::vector<std::array<double,3> > > distances::pair_wise_distance_unit_vectors(
    std::vector<std::vector<std::array<double, 3> > > distance_vectors,
    std::vector<std::vector<double> > distance_magnitudes
    )
{
    std::vector<std::vector<std::array<double,3> > > unit_vectors(
        distance_vectors.size(), std::vector<std::array<double,3> >( distance_vectors.size() ) );
    for( unsigned int i=0; i<unit_vectors.size(); i++ )
        for( unsigned int j=0; j<unit_vectors.size(); j++ )
            for( unsigned int k=0; k<3; k++ )
                unit_vectors[i][j][k] = distance_vectors[i][j][k] / distance_magnitudes[i][j];
    return unit_vectors;
}

std::vector<std::vector<double> > pair_wise_distance_magnitude(
    std::vector<std::array<double,3> > points
    )
{
    auto dis_vecs = distances::pair_wise_distance_vectors( points );
    return distances::pair_wise_distance_magnitude( dis_vecs );
}
