#include <Exceptions.h>

namespace npp {
    /// Output stream inserter for Exception.
    /// \param rOutputStream The stream the exception information is written to.
    /// \param rException The exception that's being written.
    /// \return Reference to the output stream being used.
    std::ostream &
    operator<<(std::ostream &rOutputStream, const Exception &rException)
    {
        rOutputStream << rException.toString();
        return rOutputStream;
    }
}