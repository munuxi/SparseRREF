// Standalone exactness test; build instructions are in tests/rational_matmul.md.
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

// Park every worker before the call. Only work submitted by the operation can
// then appear in the queue. Release the workers when it does, or when a serial
// call returns. No sleeps, timeouts, or changes to library headers are needed.
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
    require(observed == expected, "unexpected multiplication pool/inline path");
}

static rat_t coefficient(int seed, bool large) {
    if (!large) return rat_t((seed % 19) - 9, (seed % 7) + 1);
    int_t numerator(1), denominator(1);
    fmpz_mul_2exp(numerator.data(), numerator.data(), 257);
    fmpz_mul_2exp(denominator.data(), denominator.data(), 251);
    numerator += seed + 1;
    denominator += 2 * seed + 3;
    if (seed % 2) numerator = -numerator;
    return rat_t(numerator, denominator);
}

template <class T>
static Matrix<T> make_matrix(int rows, int cols, bool large, bool skewed, const field_t& field) {
    Matrix<T> result(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (i % 11 == 3 || (skewed && i + 1 < rows && j > 1)) continue;
            const int seed = i * 43 + j * 17 + 7;
            if (seed % 5 == 0) continue;
            if constexpr (std::is_same_v<T, rat_t>) {
                auto value = coefficient(seed, large);
                if (value != 0) result[i].push_back(j, value);
            } else {
                result[i].push_back(j, static_cast<ulong>(seed + 1) % field.get_prime());
            }
        }
    }
    return result;
}

template <class T>
static std::vector<T> dense_product(const Matrix<T>& a, const Matrix<T>& b, const field_t& field) {
    // An independent dense triple loop, not sparse_mat_mul's scratch algorithm.
    std::vector<T> result(a.nrow * b.ncol);
    for (size_t i = 0; i < a.nrow; ++i)
        for (size_t j = 0; j < b.ncol; ++j)
            for (size_t k = 0; k < a.ncol; ++k) {
                const auto av = a.find(i, k), bv = b.find(k, j);
                if (!av || !bv) continue;
                if constexpr (std::is_same_v<T, rat_t>) result[i * b.ncol + j] += *av * *bv;
                else result[i * b.ncol + j] = nmod_add(result[i * b.ncol + j], nmod_mul(*av, *bv, field.mod), field.mod);
            }
    return result;
}

template <class T>
static void check_case(int rows, int inner, int cols, bool large, bool skewed, size_t threads,
                       bool observe_parallel = false) {
    const field_t field(std::is_same_v<T, rat_t> ? RING::FIELD_QQ : RING::FIELD_Fp, 1000000007);
    const auto a = make_matrix<T>(rows, inner, large, skewed, field);
    const auto b = make_matrix<T>(inner, cols, large, false, field);
    const auto expected = dense_product(a, b, field);
    thread_pool pool(threads);
    pool.set_cleanup_func([] { flint_cleanup(); });
    for (int repeat = 0; repeat < 3; ++repeat) {
        Matrix<T> actual;
        const parallel_option opt{repeat == 0 ? nullptr : &pool, {.matmul = matmul_schedule::WIDTH_SCALED_PRODUCT_CUTOFF}};
#ifdef SPARSERREF_TEST_EXPECT_POOL
        if (observe_parallel && repeat != 0 && threads > 1)
            observe_pool_path(pool, true, [&] { actual = sparse_mat_mul(a, b, field, opt); });
        else
#endif
            actual = sparse_mat_mul(a, b, field, opt);
        require(actual.nrow == a.nrow && actual.ncol == b.ncol, "wrong product shape");
        for (size_t i = 0; i < actual.nrow; ++i) {
            size_t nonzeros = 0;
            for (size_t j = 0; j < actual.ncol; ++j) {
                const auto* value = actual.find(i, j);
                const auto& target = expected[i * actual.ncol + j];
                require(value ? *value == target : target == 0, "wrong exact product entry");
                nonzeros += target != 0;
            }
            require(actual[i].nnz() == nonzeros, "noncanonical or duplicate output entries");
        }
    }
}

static void wide_sparse_case(size_t threads) {
    const field_t field(RING::FIELD_QQ);
    const auto a = make_matrix<rat_t>(65, 29, true, false, field);
    const auto compressed = make_matrix<rat_t>(29, 23, true, false, field);
    const auto expected = dense_product(a, compressed, field);
    constexpr size_t columns = 1u << 20;
    const auto mapped_column = [](size_t j) { return j * (columns / 23); };
    Matrix<rat_t> wide(compressed.nrow, columns);
    for (size_t i = 0; i < compressed.nrow; ++i)
        for (auto [column, value] : compressed[i]) wide[i].push_back(mapped_column(column), value);
    thread_pool pool(threads);
    pool.set_cleanup_func([] { flint_cleanup(); });
    Matrix<rat_t> actual;
    const parallel_option opt{&pool, {.matmul = matmul_schedule::WIDTH_SCALED_PRODUCT_CUTOFF}};
#ifdef SPARSERREF_TEST_EXPECT_POOL
    if (threads > 1)
        observe_pool_path(pool, false, [&] { actual = sparse_mat_mul(a, wide, field, opt); });
    else
#endif
        actual = sparse_mat_mul(a, wide, field, opt);
    require(actual.nrow == a.nrow && actual.ncol == columns, "wrong wide product shape");
    for (size_t i = 0; i < actual.nrow; ++i) {
        size_t nonzeros = 0;
        for (size_t j = 0; j < compressed.ncol; ++j) {
            const auto& target = expected[i * compressed.ncol + j];
            const auto* value = actual.find(i, mapped_column(j));
            require(value ? *value == target : target == 0, "wrong wide product entry");
            nonzeros += target != 0;
        }
        // These are all possible supported columns, so this also excludes stray entries.
        require(actual[i].nnz() == nonzeros, "unexpected wide product support");
    }
}

static void cancellation_case(size_t threads) {
    field_t field(RING::FIELD_QQ);
    Matrix<rat_t> a(2, 2), b(2, 2);
    a[0].push_back(0, rat_t(1)); a[0].push_back(1, rat_t(1));
    a[1].push_back(0, rat_t(-1)); a[1].push_back(1, rat_t(-1));
    for (int i = 0; i < 2; ++i) {
        b[i].push_back(0, rat_t(i == 0 ? 1 : -1));
        b[i].push_back(1, rat_t(i == 0 ? -3 : 3));
    }
    thread_pool pool(threads);
    pool.set_cleanup_func([] { flint_cleanup(); });
    const auto result = sparse_mat_mul(a, b, field, &pool);
    require(result.nnz() == 0, "cancelled entries were retained");
}

int main(int argc, char** argv) {
    try {
        const size_t width = argc > 1 ? std::stoul(argv[1]) : 2;
        require(width >= 1, "thread count must be positive");
        for (const size_t threads : {size_t(1), width}) {
            check_case<rat_t>(17, 19, 37, false, false, threads);
            check_case<rat_t>(33, 19, 37, true, true, threads);
            check_case<rat_t>(23, 23, 41, true, false, threads, true);
            check_case<rat_t>(1601, 5, 7, true, false, threads, true);
            wide_sparse_case(threads);
            check_case<rat_t>(1, 97, 43, true, false, threads);
            check_case<rat_t>(0, 7, 5, false, false, threads);
            check_case<rat_t>(4, 0, 5, false, false, threads);
            check_case<rat_t>(3, 4, 0, false, false, threads);
            check_case<ulong>(128, 24, 64, false, false, threads);
            cancellation_case(threads);
        }
        flint_cleanup_master();
        std::cout << "PASS rational_matmul: exact dense/compressed oracles, QQ/Fp, large/tall/wide, empty/one-row/skewed/cancellation, pool reuse; workers=" << width;
#ifdef SPARSERREF_TEST_EXPECT_POOL
        std::cout << "; pool and width-guard paths observed";
#endif
        std::cout << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
