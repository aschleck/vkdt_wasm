#pragma once
// vkdt's raw loader configures OpenMP, which we run without.
static inline void omp_set_max_active_levels(int levels) {}
static inline int omp_get_thread_num(void) { return 0; }
static inline int omp_get_num_threads(void) { return 1; }
static inline int omp_get_max_threads(void) { return 1; }
