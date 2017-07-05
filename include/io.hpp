// io.hpp
// reading and writing from disk
#ifndef IO_H
#define IO_H
#include <cstdlib>
#include <string>
#include <exception>

namespace io
{
    // returns 0 if written successfully else 1
    template<typename T>
    int write_array( const std::string fname, T const *arr, const size_t len )
    {
        FILE *out;
        out = fopen( fname.c_str(), "wb" );
        if( out==NULL )
        {
            std::runtime_error( "Failed to write array to disk" );
            return 1;
        }

        // Could improve this with a specific write batch size
        fwrite( arr, sizeof(T), len, out );
        fclose( out );
        return 0;
    }
}
#endif
