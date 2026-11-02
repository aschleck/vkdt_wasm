#pragma once

// Solves full-rank linear systems by LU decomposition with partial pivoting. Matrices are n by n
// and row major.

#include <math.h>

/**
 * Factors A in place into L below the diagonal, whose own diagonal is an implicit 1, and U on and
 * above it, so that LU is A with its rows swapped as p records: row k with row p[k], for k in
 * increasing order. Returns 0 if A is singular.
 */
static inline int gauss_make_triangular(double *A, int *p, int n)
{
  for(int k = 0; k < n; k++)
  {
    int m = k;
    for(int i = k + 1; i < n; i++)
      if(fabs(A[i*n + k]) > fabs(A[m*n + k])) m = i;
    p[k] = m;
    if(A[m*n + k] == 0.0) return 0;
    // Whole rows swap, including the multipliers already in L, so that L and U factor the same
    // permutation of A.
    if(m != k)
      for(int j = 0; j < n; j++)
      {
        const double t = A[k*n + j];
        A[k*n + j] = A[m*n + j];
        A[m*n + j] = t;
      }
    for(int i = k + 1; i < n; i++)
    {
      const double l = A[i*n + k] / A[k*n + k];
      A[i*n + k] = l;
      for(int j = k + 1; j < n; j++) A[i*n + j] -= l * A[k*n + j];
    }
  }
  return 1;
}

/** Solves Ax = b into b, given the A and p that gauss_make_triangular left. */
static inline void gauss_solve_triangular(const double *A, const int *p, double *b, int n)
{
  for(int k = 0; k < n; k++)
  {
    const double t = b[k];
    b[k] = b[p[k]];
    b[p[k]] = t;
  }
  for(int i = 1; i < n; i++)
    for(int j = 0; j < i; j++) b[i] -= A[i*n + j] * b[j];
  for(int i = n - 1; i >= 0; i--)
  {
    for(int j = i + 1; j < n; j++) b[i] -= A[i*n + j] * b[j];
    b[i] /= A[i*n + i];
  }
}

/** Solves Ax = b into b, destroying A. Returns 0 if A is singular. */
static inline int gauss_solve(double *A, double *b, int n)
{
  int p[n];
  if(!gauss_make_triangular(A, p, n)) return 0;
  gauss_solve_triangular(A, p, b, n);
  return 1;
}
