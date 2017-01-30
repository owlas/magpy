template <typename RET, typename FIRST>
std::function<RET()> curry::curry(
    std::function<RET(FIRST)> f, FIRST x )
{
    return [f,x](){ return f( x ); };
}

template <typename RET, typename FIRST, typename... REST>
std::function<RET()> curry::curry(
    std::function<RET(FIRST, REST...)> f, FIRST x, REST... rest )
{
    std::function<RET(REST...)> g = [f,x](REST... r){ return f( x, r... ); };
    return curry( g, rest... );
}



template<typename RET, typename FIRST, typename... ARGS>
std::vector<std::function<RET()> > curry::vector_curry(
    std::function<RET(FIRST, ARGS... )> f,
    std::vector<FIRST> first,
    std::vector<ARGS>... args )
{
    std::vector<std::function<RET()> > funcs;
    for( unsigned int i=0; i<first.size(); i++ )
        funcs.push_back( curry( f, first[i], args[i]... ) );
    return funcs;
}
