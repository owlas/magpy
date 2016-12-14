// io.hpp
// reading and writing from disk
#ifndef IO_H
#define IO_H
#include "../include/easylogging++.h"
#include <cstdlib>
#include <string>

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
            LOG(ERROR) << "Failed to write array to file: " << fname;
            return 1;
        }

        // Could improve this with a specific write batch size
        fwrite( arr, sizeof(T), len, out );
        fclose( out );
        return 0;
    }
}
#endif
