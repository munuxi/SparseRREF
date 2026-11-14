# Rational multiplication scheduling and exactness tests

`sparse_mat_mul` uses a rational crossover of 256 estimated products per worker
and a width guard: `products / workers >= max(256, ceil(output_columns / 16))`.
This bounds the parallel dense-cache slot count by 16 times the estimated
products. Big-number limbs are additional and have no fixed memory cap. One-row
and one-worker products stay serial. The word-size finite-field policy and
arithmetic within a row are unchanged. Constants are plain `inline constexpr`
values in `sparse_rref.h`, following the existing threshold configuration.

For rational rows, the requested block count is
`min(rows, max(rows / 10, 64 * workers))`, evaluated without multiplication
overflow. Small products keep per-row balance; tall products approach ten rows
per task rather than allocating one task per row. Both crossover constants
are empirical. The pre-existing average-density estimate still truncates to
zero when the right operand averages fewer than one nonzero per row.

Build and run from the repository root; add include/library paths and `-lmpfr`
if your FLINT installation needs them:

```sh
c++ -O2 -std=c++20 -pthread -DSPARSERREF_TEST_EXPECT_POOL -I. \
  tests/rational_matmul.cpp -lflint -lgmp -o rational_matmul
./rational_matmul 8    # the largest worker count to try
```

Each invocation also tests one
worker and a null pool. Exact dense oracles cover QQ/Fp, large coefficients,
skewed rows, cancellation, empty shapes and reuse. The 23-by-23 times 23-by-41
large-rational case crosses the default threshold at all listed widths. A
1601-row product exercises bounded task grouping. A wide sparse product has
2^20 columns and a compressed exact oracle that checks its complete support.

`SPARSERREF_TEST_EXPECT_POOL` changes only the harness: parked workers and a
queue observer deterministically assert that the large/tall calls enqueue
work and the wide call stays inline. There are no library instrumentation
hooks or polling timeouts. Compile without this test macro to compare arithmetic
against unmodified upstream, which is also expected to pass these exactness
guards; they are not failing-on-upstream numerical regressions.

Rebuild with `-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer`, or
separately with `-O1 -g -fsanitize=thread`, and repeat the allocated widths.
Prebuilt FLINT itself is not sanitizer-instrumented. The harness releases
worker FLINT caches and calls `flint_cleanup_master()` before exiting.
