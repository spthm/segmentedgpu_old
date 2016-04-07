#pragma once

#include "sgpucontext.h"

namespace sgpu {

// Create a context on the default stream (0).
ContextPtr CreateCudaDeviceFromArgv(int argc, char** argv,
	bool printInfo = false);

// Create a context on a new stream.
ContextPtr CreateCudaDeviceStreamFromArgv(int argc, char** argv,
	bool printInfo = false);

} // namespace sgpu
