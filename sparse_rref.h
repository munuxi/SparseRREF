/*
	Copyright (C) 2024 Zhenjie Li (Li, Zhenjie)

	This file is part of SparseRREF. The SparseRREF is free software:
	you can redistribute it and/or modify it under the terms of the MIT
	License.
*/

#ifndef SPARSE_RREF_H
#define SPARSE_RREF_H

#include <algorithm>
#include <bit>
#include <bitset>
#include <charconv> 
#include <chrono>
#include <climits>
#include <cmath>
#include <cstring>
#include <execution> 
#include <fstream>
#include <iomanip>
#include <iostream>
#include <list>
#include <map>
#include <numeric>
#include <queue>
#include <random>
#include <ranges>
#include <set>
#include <span>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

#include "thread_pool.hpp"

#ifdef NULL
#undef NULL
#endif
#define NULL nullptr

namespace SparseRREF {
	// version
	static const char version[] = "v0.3.2";

	enum SPARSE_FILE_TYPE {
		SPARSE_FILE_TYPE_PLAIN,
		SPARSE_FILE_TYPE_SMS,
		SPARSE_FILE_TYPE_MTX,
		SPARSE_FILE_TYPE_WXF,
		SPARSE_FILE_TYPE_BIN
	};

	// Memory management
	template <typename T>
	inline T* s_malloc(const size_t size) {
		return (T*)std::malloc(size * sizeof(T));
	}

	template <typename T>
	inline void s_free(T* s) {
		std::free(s);
	}

	template <typename T>
	inline T* s_realloc(T* s, const size_t size) {
		return (T*)std::realloc(s, size * sizeof(T));
	}

	template <typename T>
	void s_copy(T* des, T* ini, const size_t size) {
		if (des == ini)
			return;
		std::copy(ini, ini + size, des);
	}

	// thread
	using thread_pool = BS::thread_pool<>; // thread pool
	inline size_t thread_id() {
		return BS::this_thread::get_index().value();
	}

	// rref_option
	struct rref_option {
		bool verbose = false;
		bool is_back_sub = true;
		uint8_t method = 0;
		int print_step = 100;
		bool shrink_memory = false;
		std::function<int64_t(int64_t)> col_weight = [](int64_t i) { return i; };
		thread_pool pool = thread_pool(1); // default: thread pool with 1 thread
	};
	using rref_option_t = rref_option[1];

	inline size_t ctz(uint64_t x) {
		return std::countr_zero(x);
	}
	inline size_t clz(uint64_t x) {
		return std::countl_zero(x);
	}
	inline size_t popcount(uint64_t x) {
		return std::popcount(x);
	}

	template <typename T>
	inline uint8_t minimal_signed_bits(T x) noexcept {
		if (x >= INT8_MIN && x <= INT8_MAX) return 0;
		if (x >= INT16_MIN && x <= INT16_MAX) return 1;
		if (x >= INT32_MIN && x <= INT32_MAX) return 2;
		return 3; // for int64_t
	}

	template <typename T>
	inline uint8_t minimal_unsigned_bits(T x) noexcept {
		if (x <= UINT8_MAX) return 0;
		if (x <= UINT16_MAX) return 1;
		if (x <= UINT32_MAX) return 2;
		return 3; // for uint64_t
	}

	// string
	inline void DeleteSpaces(std::string& str) {
		str.erase(std::remove_if(str.begin(), str.end(),
			[](unsigned char x) { return std::isspace(x); }),
			str.end());
	}

	inline std::vector<std::string> SplitString(const std::string& s, const std::string delim) {
		size_t start = 0;
		size_t end = s.find(delim);
		std::vector<std::string> result;
		while (end != std::string::npos) {
			result.push_back(s.substr(start, end - start));
			start = end + delim.length();
			end = s.find(delim, start);
		}
		result.push_back(s.substr(start, end));
		return result;
	}

	uint64_t string_to_ull(std::string_view sv) {
		uint64_t result;
		auto [ptr, ec] = std::from_chars(sv.data(), sv.data() + sv.size(), result);
		if (ec != std::errc()) {
			throw std::runtime_error("Failed to parse number");
		}
		return result;
	}

	// time
	inline std::chrono::system_clock::time_point clocknow() {
		return std::chrono::system_clock::now();
	}

	inline double usedtime(std::chrono::system_clock::time_point start,
		std::chrono::system_clock::time_point end) {
		auto duration =
			std::chrono::duration_cast<std::chrono::microseconds>(end - start);
		return ((double)duration.count() * std::chrono::microseconds::period::num /
			std::chrono::microseconds::period::den);
	}

	// some algorithms
	template <typename T> std::vector<T> difference(std::vector<T> l) {
		std::vector<T> result;
		for (size_t i = 1; i < l.size(); i++) {
			result.push_back(l[i] - l[i - 1]);
		}
		return result;
	}

	template <typename T>
	int lexico_compare(const std::vector<T>& a, const std::vector<T>& b) {
		for (size_t i = 0; i < a.size(); i++) {
			if (a[i] < b[i])
				return -1;
			if (a[i] > b[i])
				return 1;
		}
		return 0;
	}

	template <typename T>
	int lexico_compare(const T* a, const T* b, const size_t len) {
		for (size_t i = 0; i < len; i++) {
			if (a[i] < b[i])
				return -1;
			if (a[i] > b[i])
				return 1;
		}
		return 0;
	}

	template <typename T>
	int lexico_compare(const T* a, const T* b, const std::vector<size_t>& perm) {
		for (auto i : perm) {
			if (a[i] < b[i])
				return -1;
			if (a[i] > b[i])
				return 1;
		}
		return 0;
	}

	// mulit for
	void multi_for(
		const std::vector<size_t>& start, 
		const std::vector<size_t>& end, 
		const std::function<void(std::vector<size_t>&)> func) {

		if (start.size() != end.size()) {
			std::cerr << "Error: start and end size not match." << std::endl;
			exit(1);
		}

		std::vector<size_t> index(start);
		size_t nt = start.size();

		while (true) {
			func(index);

			for (int i = nt - 1; i > -2; i--) {
				if (i == -1) 
					return;
				index[i]++;
				if (index[i] < end[i])
					break;
				index[i] = start[i];
			}
		}
	}

	// bit_array
	struct bit_array {
		constexpr static size_t bitset_size = 64; // 64 bits per bitset
		std::vector<std::bitset<bitset_size>> data;

		bit_array() {}

		void resize(size_t size) {
			auto len = size / bitset_size + 1;
			data.resize(len);
		}

		bit_array(size_t size) {
			resize(size);
		}

		~bit_array() {
			data.clear();
		}

		void insert(size_t val) {
			auto idx = val / bitset_size;
			auto pos = val % bitset_size;
			data[idx].set(pos);
		}

		bool test(size_t val) {
			auto idx = val / bitset_size;
			auto pos = val % bitset_size;
			return data[idx].test(pos);
		}

		bool count(size_t val) {
			return test(val);
		}

		void erase(size_t val) {
			auto idx = val / bitset_size;
			auto pos = val % bitset_size;
			data[idx].reset(pos);
		}

		void clear() {
			for (auto& d : data)
				d.reset();
		}

		bool operator[](size_t idx) {
			return count(idx);
		}

		size_t size() {
			return data.size();
		}

		size_t length() {
			return data.size() * bitset_size;
		}

		std::vector<size_t> nonzero() {
			std::vector<size_t> result;
			size_t tmp[bitset_size];
			size_t tmp_size = 0;
			for (size_t i = 0; i < data.size(); i++) {
				if (data[i].any()) {
					tmp_size = 0;
					uint64_t c = data[i].to_ullong();

					// only ctz version
					// while (c) {
					// 	auto ctzpos = ctz(c);
					// 	result.push_back(i * bitset_size + ctzpos);
					// 	c &= c - 1;
					// }

					while (c) {
						auto ctzpos = ctz(c);
						auto clzpos = bitset_size - 1 - clz(c);
						result.push_back(i * bitset_size + ctzpos);
						if (ctzpos == clzpos)
							break;
						tmp[tmp_size] = i * bitset_size + clzpos;
						tmp_size++;
						c = c ^ (1ULL << clzpos) ^ (1ULL << ctzpos);
					}
					for (size_t j = tmp_size; j > 0; j--) {
						result.push_back(tmp[j - 1]);
					}
				}
			}
			return result;
		}
	};

	template <typename T> inline T* binary_search(T* begin, T* end, T val) {
		auto ptr = std::lower_bound(begin, end, val);
		if (ptr == end || *ptr == val)
			return ptr;
		else
			return end;
	}

	template <typename T> inline T* lower_bound(T* begin, T* end, T* val, size_t rank) {
		if (rank == 1)
			return std::lower_bound(begin, end, *val);

		size_t left = 0;
		size_t right = (end - begin) / rank;

		while (left < right) {
			size_t mid = left + (right - left) / 2;
			if (lexico_compare(begin + rank * mid, val, rank) < 0)
				left = mid + 1;
			else
				right = mid;
		}

		return begin + rank * left;
	}

	template <typename T> inline T* binary_search(T* begin, T* end, uint16_t rank, T* val) {
		auto ptr = SparseRREF::lower_bound(begin, end, rank, val);
		if (ptr == end || std::equal(ptr, ptr + rank, val))
			return ptr;
		else
			return end;
	}

	template <typename T>
	std::vector<T> perm_init(T n) {
		std::vector<T> perm(n);
		std::iota(perm.begin(), perm.end(), 0);
		return perm;
	}

	std::vector<size_t> perm_inverse(const std::vector<size_t>& perm) {
		size_t n = perm.size();
		std::vector<size_t> result(n);
		for (size_t i = 0; i < n; i++)
			result[perm[i]] = i;
		return result;
	}

	std::vector<size_t> random_perm(size_t n) {
		auto permutation = perm_init(n);

		std::random_device rd;
		std::mt19937 g(rd());

		std::shuffle(permutation.begin(), permutation.end(), g);
		return permutation;
	}

	bool is_identity_perm(const std::vector<size_t>& perm) {
		size_t n = perm.size();
		for (size_t i = 0; i < n; i++) {
			if (perm[i] != i)
				return false;
		}
		return true;
	}

	template <typename T>
	void permute(const std::vector<size_t>& P, T* A, size_t block_size = 1) {
		std::vector<bool> visited(P.size(), false);

		auto permute_it = [&](auto& temp_block) {
			for (size_t i = 0; i < P.size(); ++i) {
				if (visited[i] || P[i] == i)
					continue;

				size_t j = i;
				for (size_t k = 0; k < block_size; ++k) 
					temp_block[k] = std::move(A[j * block_size + k]);

				while (!visited[j]) {
					visited[j] = true;
					size_t k = P[j];

					if (k == i) {
						for (size_t m = 0; m < block_size; ++m)
							A[j * block_size + m] = std::move(temp_block[m]);
						break;
					}

					for (size_t m = 0; m < block_size; ++m)
						A[j * block_size + m] = std::move(A[k * block_size + m]);
					j = k;
				}
			}
			};

		if (block_size < 32) {
			// Use a fixed-size array for small block sizes
			T temp_block[32];
			permute_it(temp_block);
		}
		else {
			std::vector<T> temp_block(block_size);
			permute_it(temp_block);
		}

	}

	template <typename T>
	void permute(const std::vector<size_t>& P, std::vector<T>& A, size_t block_size = 1) {
		permute(P, A.data(), block_size);
	}

	inline std::vector<size_t> swap_perm(size_t a, size_t b, size_t n) {
		std::vector<size_t> perm(n);
		for (size_t i = 0; i < n; i++)
			perm[i] = i;
		perm[a] = b;
		perm[b] = a;
		return perm;
	}

} // namespace SparseRREF

#endif