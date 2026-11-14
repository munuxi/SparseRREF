// Standalone exactness test; build instructions are in tests/rational_backsub.md.
#include "sparse_mat.h"
#include <atomic>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

using namespace SparseRREF;
template <class T> using Matrix = sparse_mat<T, int>;

static void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

// Park workers until this operation queues work (or returns inline). Observe
// only the solver with a precomputed transpose, not transpose construction.
template <class Function>
static void observe_pool_path(thread_pool& pool, bool expected, Function operation) {
    std::mutex mutex;
    std::condition_variable condition;
    bool released = false, observed = false;
    std::atomic<size_t> ready{0};
    std::atomic<bool> done{false};
    pool.wait(); // Also wait for a newly constructed pool's worker initialization.
    require(pool.get_tasks_total() == 0, "pool not idle before observation");
    for (size_t i = 0; i < pool.get_thread_count(); ++i)
        pool.detach_task([&] {
            ++ready;
            std::unique_lock lock(mutex);
            condition.wait(lock, [&] { return released; });
        });
    while (ready.load() != pool.get_thread_count()) std::this_thread::yield();
    std::thread observer([&] {
        while (!(observed = pool.get_tasks_queued() != 0) && !done.load())
            std::this_thread::yield();
        { std::lock_guard lock(mutex); released = true; }
        condition.notify_all();
    });
    try { operation(); }
    catch (...) { done = true; observer.join(); pool.wait(); throw; }
    done = true;
    observer.join();
    pool.wait();
    require(observed == expected, "unexpected back-substitution pool/inline path");
}

template <class T>
static T coefficient(int seed, bool large, const field_t& field) {
    if constexpr (std::is_same_v<T, rat_t>) {
        if (!large) return rat_t(seed % 13 - 6, seed % 5 + 1);
        int_t a(1), b(1);
        fmpz_mul_2exp(a.data(), a.data(), 129);
        fmpz_mul_2exp(b.data(), b.data(), 127);
        a += seed + 1; b += 2 * seed + 3;
        if (seed % 2) a = -a;
        return rat_t(a, b);
    } else return static_cast<ulong>(seed + 1) % field.get_prime();
}

static void large_border_case(int ordering, size_t threads) {
    const field_t field(RING::FIELD_QQ);
    constexpr int rank = 1601;
    const int anchor = ordering < 0 ? rank - 1 : 0;
    Matrix<rat_t> original(rank, rank + 1);
    std::vector<pivot_t<int>> pivots;
    std::vector<rat_t> expected_rhs(rank);
    for (int r = 0; r < rank; ++r) {
        pivots.push_back({r, r});
        const auto value = coefficient<rat_t>(r * 17 + 3, true, field);
        if (r != anchor && ordering > 0) original[r].push_back(anchor, value);
        original[r].push_back(r, rat_t(1));
        if (r != anchor && ordering < 0) original[r].push_back(anchor, value);
        original[r].push_back(rank, rat_t(r == anchor ? 3 : 1));
        expected_rhs[r] = r == anchor ? rat_t(3) : rat_t(1) - value * rat_t(3);
    }
    rref_option_t options;
    options->pool.reset(threads);
    options->pool.set_cleanup_func([] { flint_cleanup(); });
    for (int repeat = 0; repeat < 3; ++repeat) {
        auto actual = original;
        const auto transpose = build_pivot_transpose(actual, pivots, options->pool);
#ifdef SPARSERREF_TEST_EXPECT_POOL
        if (threads > 1)
            observe_pool_path(options->pool, true, [&] {
                triangular_solver(actual, transpose, pivots, field, options, ordering);
            });
        else
#endif
            triangular_solver(actual, transpose, pivots, field, options, ordering);
        // Eliminating the one border column leaves identity and b[r]-a[r]*b[anchor].
        // Checking its complete support gives an exact oracle without cubic dense work.
        for (int r = 0; r < rank; ++r) {
            const auto* diagonal = actual.find(r, r);
            const auto* rhs = actual.find(r, rank);
            require(diagonal && *diagonal == 1, "wrong border-system diagonal");
            require(rhs && *rhs == expected_rhs[r], "wrong border-system RHS");
            require(actual[r].nnz() == 2, "unexpected border-system support");
        }
    }
}

template <class T>
static void check_case(int rank, int ordering, bool large, size_t threads) {
    const field_t field(std::is_same_v<T, rat_t> ? RING::FIELD_QQ : RING::FIELD_Fp, 1000000007);
    const size_t rows = rank + 3, cols = rank + 5;
    std::vector<pivot_t<int>> pivots;
    std::vector<T> dense(rows * cols);
    for (int i = 0; i < rank; ++i) pivots.push_back({(i * 7) % rank, (i * 11) % rank});
    for (int i = 0; i < rank; ++i) {
        const auto [r, c] = pivots[i]; dense[r * cols + c] = T(1);
        for (int j = 0; j < rank; ++j) {
            if ((ordering < 0 ? j > i : j < i) && (i + j) % 4 != 0)
                dense[r * cols + pivots[j].c] = coefficient<T>(i * 31 + j * 7 + 3, large, field);
        }
        for (size_t j = rank; j < cols; ++j) dense[r * cols + j] = coefficient<T>(i * 17 + j, large, field);
    }
    // A nonpivot row is deliberately nonzero and must remain untouched.
    dense[(rows - 1) * cols + cols - 1] = T(3);
    Matrix<T> original(rows, cols);
    for (size_t r = 0; r < rows; ++r)
        for (size_t c = 0; c < cols; ++c)
            if (dense[r * cols + c] != 0) original[r].push_back(c, dense[r * cols + c]);
    auto expected = dense;
    // Dense Gauss-Jordan updates, independent of sparse row helpers/transpose.
    for (int step = 0; step < rank; ++step) {
        const auto [r, c] = pivots[ordering < 0 ? rank - 1 - step : step];
        for (const auto& pivot : pivots) {
            const auto target = pivot.r;
            if (target == r) continue;
            const auto value = expected[target * cols + c];
            for (size_t j = 0; j < cols; ++j) {
                if constexpr (std::is_same_v<T, rat_t>) expected[target * cols + j] -= value * expected[r * cols + j];
                else expected[target * cols + j] = nmod_sub(expected[target * cols + j], nmod_mul(value, expected[r * cols + j], field.mod), field.mod);
            }
        }
    }
    rref_option_t options;
    options->pool.reset(threads);
    options->pool.set_cleanup_func([] { flint_cleanup(); });
    for (int repeat = 0; repeat < 3; ++repeat) {
        auto actual = original;
        if (repeat % 2) {
            const auto transpose = build_pivot_transpose(actual, pivots, options->pool);
            triangular_solver(actual, transpose, pivots, field, options, ordering);
        } else triangular_solver(actual, pivots, field, options, ordering);
        for (size_t r = 0; r < rows; ++r) {
            size_t nonzeros = 0;
            for (size_t c = 0; c < cols; ++c) {
                const auto* entry = actual.find(r, c); const auto& value = expected[r * cols + c];
                require(entry ? *entry == value : value == 0, "wrong exact back-substitution entry");
                nonzeros += value != 0;
            }
            require(actual[r].nnz() == nonzeros, "noncanonical or duplicate output entries");
        }
    }
    options->abort = true;
    auto untouched = original;
    triangular_solver(untouched, pivots, field, options, ordering);
    for (size_t r = 0; r < rows; ++r)
        for (size_t c = 0; c < cols; ++c) {
            const auto* entry = untouched.find(r, c); const auto& value = dense[r * cols + c];
            require(entry ? *entry == value : value == 0, "pre-aborted input was modified");
        }
}

int main(int argc, char** argv) {
    try {
        const size_t width = argc > 1 ? std::stoul(argv[1]) : 2;
        require(width >= 1, "thread count must be positive");
        for (const size_t threads : {size_t(1), width})
            for (int ordering : {-1, 1}) {
                check_case<rat_t>(0, ordering, false, threads);
                check_case<rat_t>(1, ordering, true, threads);
                check_case<rat_t>(17, ordering, true, threads);
                check_case<rat_t>(40, ordering, false, threads);
                check_case<ulong>(40, ordering, false, threads);
                large_border_case(ordering, threads);
            }
        flint_cleanup_master();
        std::cout << "PASS rational_backsub: dense/closed-form exact oracles, QQ/Fp, upper/lower, large/tall, permuted pivots, empty/single, pool reuse and pre-abort; workers=" << width;
#ifdef SPARSERREF_TEST_EXPECT_POOL
        std::cout << "; solver pool path observed";
#endif
        std::cout << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
