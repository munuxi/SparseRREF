# Direct rational back-substitution and exactness tests

`triangular_solver` estimates rational work from the target-row count and the
pivot-row length. It dispatches at 256 entry operations, using the plain
`inline constexpr` threshold in `sparse_rref.h`. The comparison uses division
to avoid overflowing the work product. One-worker and fewer-than-two-target
updates stay inline. Word-size finite-field scheduling, pivot ordering,
per-row exact arithmetic and the wait between pivot columns are unchanged.
This targets direct rational calls, not the finite-field reconstruction stages.

Requested rational blocks are `min(rows, max(rows / 10, 64 * workers))`, with an
overflow-safe calculation. Small columns retain one task per target row; tall
columns approach ten rows per task instead of enqueuing all rows separately.
The threshold and balancing factor are empirical.

Build and run from the repository root; add include/library paths and `-lmpfr`
if your FLINT installation needs them:

```sh
c++ -O2 -std=c++20 -pthread -DSPARSERREF_TEST_EXPECT_POOL -I. \
  tests/rational_backsub.cpp -lflint -lgmp -o rational_backsub
./rational_backsub 8    # the largest worker count to try
```

Each invocation also uses
one worker. Dense exact Gauss-Jordan oracles cover upper/lower systems,
permuted pivots, QQ/Fp, large coefficients, empty/single-pivot cases, nonpivot
rows, both transpose overloads, pool reuse and pre-aborted calls. A separate
1601-row large-rational border system has a closed-form exact oracle for every
row and its complete support. Its first column update crosses the default
cutoff at every listed width and exercises bounded task grouping.

The test-only `SPARSERREF_TEST_EXPECT_POOL` flag enables deterministic queue
observation while workers are parked. This checks solver dispatch through the
precomputed-transpose overload, excluding transpose construction from the
observation. It adds no hooks to the library and uses no timeouts. Without
the flag, unmodified upstream is expected to pass the same exactness guards;
these are not failing-on-upstream numerical regressions.

Repeat with `-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer` or a
separate `-O1 -g -fsanitize=thread` build. Prebuilt FLINT itself remains
uninstrumented. The harness releases worker FLINT caches and calls
`flint_cleanup_master()` before exit.
