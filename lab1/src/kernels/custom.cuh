#pragma once
// =============================================================================
//  custom.cuh  --  Your own optimized kernels (Part 4)
// =============================================================================
//  Row-major, FP32:  C (MxN) = alpha * A (MxK) @ B (KxN) + beta * C
//  YOU EDIT THIS FILE.
// =============================================================================
#include "../common.cuh"

// e.g. a vectorized / double-buffered / swizzled variant of the reg2d kernel,
// a shape-dispatching launcher, ... Register each one in src/kernels.cuh.
