#pragma once
// Part 4 Optimization 1: shared-memory A tile padding.
// Based on the student's verified Part 3 reg2d implementation. Apart from
// As[BK][BM + 1], the loads, math, stores, block geometry and launch settings
// intentionally remain the same for a controlled A/B experiment.
#include "../common.cuh"

template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM / TM) * (BN / TN))
    sgemm_custom_padA(int M, int N, int K, float alpha, const float* __restrict__ A,
                const float* __restrict__ B, float beta,
                float* __restrict__ C) {
  constexpr int NT = (BM / TM) * (BN / TN);  // threads per block
  // As is stored TRANSPOSED: As[k][m] holds A[m][k] of the current tile.
  __shared__ float As[BK][BM + 1];  // +1 breaks 32-bank transpose store aliasing
  __shared__ float Bs[BK][BN];
  const int tid = threadIdx.x;
  // This thread computes rows tr*TM .. tr*TM+TM-1 and
  // columns tc*TN .. tc*TN+TN-1 of the block's BM x BN tile.
  const int tr = tid / (BN / TN), tc = tid % (BN / TN);
  float acc[TM][TN] = {};
  float ra[TM], rb[TN];
  for (int k0 = 0; k0 < K; k0 += BK) {
    // Load the same A/B tiles as the Part 3 baseline.
    
    const int rowBase = blockIdx.y * BM;
    const int colBase = blockIdx.x * BN;

    // Load A tile into shared memory (transposed).
    for (int idx = tid; idx < BM * BK; idx += NT) {
        const int m = idx / BK;
        const int k = idx % BK;

        As[k][m] =
            A[(size_t)(rowBase + m) * K + (k0 + k)];
    }

    // Load B tile into shared memory.
    for (int idx = tid; idx < BK * BN; idx += NT) {
        const int k = idx / BN;
        const int n = idx % BN;

        Bs[k][n] =
            B[(size_t)(k0 + k) * N + (colBase + n)];
    }

    __syncthreads();
    // GIVEN: outer product. Per k, each thread loads TM values of A and TN
    // values of B into registers and does TM*TN FMAs with them.
    for (int k = 0; k < BK; ++k) {
      for (int i = 0; i < TM; ++i) ra[i] = As[k][tr * TM + i];
      for (int j = 0; j < TN; ++j) rb[j] = Bs[k][tc * TN + j];
      for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j) acc[i][j] += ra[i] * rb[j];
    }
    __syncthreads();
  }
  // Write the output tile to C.  
    const int outRow = blockIdx.y * BM + tr * TM;
    const int outCol = blockIdx.x * BN + tc * TN;

    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
            const size_t idx =
                (size_t)(outRow + i) * N + (outCol + j);

            C[idx] = alpha * acc[i][j] + beta * C[idx];
        }
    }

}

template <int BM, int BN, int BK, int TM, int TN>
bool launch_custom_padA(int M, int N, int K, float alpha, const float* A,
                        const float* B, float beta, float* C, cudaStream_t s) {
  static_assert(BM % TM == 0 && BN % TN == 0, "tile must divide block tile");
  constexpr int kThreads = (BM / TM) * (BN / TN);
  static_assert(kThreads <= 1024, "too many threads per block");
  if (M % BM || N % BN || K % BK) return false;
  dim3 grid(N / BN, M / BM);
  sgemm_custom_padA<BM, BN, BK, TM, TN>
      <<<grid, kThreads, 0, s>>>(M, N, K, alpha, A, B, beta, C);
  return true;
}

// Part 4 Optimization 2: retain padding while restoring a 16-byte-aligned A row stride.
// Only As row pitch changes versus custom_padA; calculation and access mapping stay unchanged.
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM / TM) * (BN / TN))
    sgemm_custom_padA_align16(int M, int N, int K, float alpha, const float* __restrict__ A,
                const float* __restrict__ B, float beta,
                float* __restrict__ C) {
  constexpr int NT = (BM / TM) * (BN / TN);  // threads per block
  // As is stored TRANSPOSED: As[k][m] holds A[m][k] of the current tile.
  __shared__ float As[BK][BM + 4];  // +4 floats = 16-byte pitch alignment for vectorized shared loads
  __shared__ float Bs[BK][BN];
  const int tid = threadIdx.x;
  // This thread computes rows tr*TM .. tr*TM+TM-1 and
  // columns tc*TN .. tc*TN+TN-1 of the block's BM x BN tile.
  const int tr = tid / (BN / TN), tc = tid % (BN / TN);
  float acc[TM][TN] = {};
  float ra[TM], rb[TN];
  for (int k0 = 0; k0 < K; k0 += BK) {
    // Load the same A/B tiles as the Part 3 baseline.
    
    const int rowBase = blockIdx.y * BM;
    const int colBase = blockIdx.x * BN;

    // Load A tile into shared memory (transposed).
    for (int idx = tid; idx < BM * BK; idx += NT) {
        const int m = idx / BK;
        const int k = idx % BK;

        As[k][m] =
            A[(size_t)(rowBase + m) * K + (k0 + k)];
    }

    // Load B tile into shared memory.
    for (int idx = tid; idx < BK * BN; idx += NT) {
        const int k = idx / BN;
        const int n = idx % BN;

        Bs[k][n] =
            B[(size_t)(k0 + k) * N + (colBase + n)];
    }

    __syncthreads();
    // GIVEN: outer product. Per k, each thread loads TM values of A and TN
    // values of B into registers and does TM*TN FMAs with them.
    for (int k = 0; k < BK; ++k) {
      for (int i = 0; i < TM; ++i) ra[i] = As[k][tr * TM + i];
      for (int j = 0; j < TN; ++j) rb[j] = Bs[k][tc * TN + j];
      for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j) acc[i][j] += ra[i] * rb[j];
    }
    __syncthreads();
  }
  // Write the output tile to C.  
    const int outRow = blockIdx.y * BM + tr * TM;
    const int outCol = blockIdx.x * BN + tc * TN;

    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
            const size_t idx =
                (size_t)(outRow + i) * N + (outCol + j);

            C[idx] = alpha * acc[i][j] + beta * C[idx];
        }
    }

}

template <int BM, int BN, int BK, int TM, int TN>
bool launch_custom_padA_align16(int M, int N, int K, float alpha, const float* A,
                        const float* B, float beta, float* C, cudaStream_t s) {
  static_assert(BM % TM == 0 && BN % TN == 0, "tile must divide block tile");
  constexpr int kThreads = (BM / TM) * (BN / TN);
  static_assert(kThreads <= 1024, "too many threads per block");
  if (M % BM || N % BN || K % BK) return false;
  dim3 grid(N / BN, M / BM);
  sgemm_custom_padA_align16<BM, BN, BK, TM, TN>
      <<<grid, kThreads, 0, s>>>(M, N, K, alpha, A, B, beta, C);
  return true;
}
