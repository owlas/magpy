/**
 * @namespace distances
 * @brief Compute distances between points.
 * @details Functions for computing distances between points
 * in 3D space.
 * @author Oliver Laslett
 * @date 2017
 */
#include "../include/distances.hpp"
#include <cstdlib>
#include <cmath>

/// Computes the distance vector between points in 3d space
/**
 * Given a list of x,y,z coordinates, compute the distance vector between
 * every pair of x,y,z coordinates in the list.
 * @param[in] points length N vector of arrays of x,y,z coordinates
 * @returns NxN std::vector matrix of distances between each pair of points
 */
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

/// Computes the Euclidean distance between pairs of points
/*
 * Given a vector of x,y,z coordinate arrays, compute the Euclidean distance
 * between every pair of points.
 * @param[in] points length N vector of arrays of x,y,z coordinates
 * @returns NxN matrix where element i,j is the Euclidean distance between
 *    points[i] and points[j]
 */
std::vector<std::vector<double> > pair_wise_distance_magnitude(
    std::vector<std::array<double,3> > points
    )
{
    auto dis_vecs = distances::pair_wise_distance_vectors( points );
    return distances::pair_wise_distance_magnitude( dis_vecs );
}

/// Computes the Euclidean distance from a matrix of distance vectors
/**
 * Given a matrix of distance vectors, compute the 2norm of the vector
 * corresponding to the Euclidean distance.
 * @param[in] distance_vectors NxN symmetrical matrix of distance vectors (x,y,z coordinates)
 * @returns NxN matrix of Euclidean distance values
 */
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

/// Computes the unit distance vector between a list of points
/*
 * Computes the unit distance vector between every pair of points
 * in a list of (x,y,z) coordinates.
 * @param[in] points length N vector of arrays of x,y,z coordinates
 * @returns NxN std::vector matrix of unit distances between each pair of points
 */
std::vector<std::vector<std::array<double,3> > > distances::pair_wise_distance_unit_vectors(
    std::vector<std::array<double,3> > points )
{
    auto distance_vectors = pair_wise_distance_vectors( points );
    auto distance_magnitudes = pair_wise_distance_magnitude( distance_vectors );
    return pair_wise_distance_unit_vectors( distance_vectors, distance_magnitudes );
}

/// Computes the unit distance vectors from distance vectors and their magnitudes
/*
 * Given a matrix of distance vectors and their magnitudes, compute the
 * corresponding unit distance vectors.
 * @param[in] distance_vectors NxN symmetrical matrix of distance vectors
 * @param[in] distance_magnitudes NxN symmetrical matrix of corresponding vector magnitudes
 * @returns NxN matrix of unit vectors
 */
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
