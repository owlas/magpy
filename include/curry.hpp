#ifndef CURRY_H
#define CURRY_H
#include <functional>
#include <vector>
/**
 * @namespace: curry
 * Contains higher order functions for currying.
 */
namespace curry {

    /// Curry single argument function
    /**
     * Binds a function's single argument and returns the
     * callable function without arguments.
     * @param[in] f std::function with a single argument
     * @param[in] x value of argument to bind to function
     * @returns std::function with no arguments
     */
    template <typename RET, typename FIRST>
    std::function<RET()> curry( std::function<RET(FIRST)> f, FIRST x );

    /// Curry multi-argument function
    /**
     * Binds the multiple arguments to a std::function's signature and
     * returns the call without function call with no arguments
     * @param[in] f std::function with multiple arguments
     * @param[in] x value of first argument to bind to function
     * @param[in] rest... variable number of arguments to bind (can
     * have any type)
     * @returns std::function with no arguments
     */
    template <typename RET, typename FIRST, typename... REST>
    std::function<RET()> curry( std::function<RET(FIRST, REST...)> f, FIRST x, REST... rest );

    /// Curry multi-argument function many times with lists of arguments
    /**
     * Maps multiple vectors of function arguments to a vector of
     * callable functions with all parameters bound.
     * For example given the add function `returns x+y` and two
     * std::vector<double> instances with `xs={1,2},ys={3,4}` then the
     * vector_curry will return a `std::vector<std::function<double()>
     * >` with two items. The first call will evaluate to `1+3=4` and
     * a call to the second item will evaluate to `2+4=6`.
     * @param[in] f std::function with multiple arguments
     * @param[in] first std::vector of different values for first
     * argument
     * @param[in] args... variable number of std::vectors with
     * different values of function arguments (all vectors must be
     * same length)
     * @returns std::vector of callable functions
     */
    template<typename RET, typename FIRST, typename... ARGS>
    std::vector<std::function<RET()> > vector_curry(
        std::function<RET(FIRST, ARGS... )> f,
        std::vector<FIRST> first,
        std::vector<ARGS>... args );
}

#include "curry.tpp"
#endif // CURRY_H
