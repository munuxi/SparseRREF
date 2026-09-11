/*
	Copyright (C) 2024-2026 Zhenjie Li (Li, Zhenjie)

	This file is part of SparseRREF. The SparseRREF is free software:
	you can redistribute it and/or modify it under the terms of the MIT
	License.
*/


#ifndef SPARSE_TENSOR_H
#define SPARSE_TENSOR_H

#include "sparse_type.h"
#include <cstdint>
#include <limits>

namespace SparseRREF {
	// A and B need not be sorted: gen_perm() visits both in index order, so the index tuples are
	// concatenated in lexicographic order and C is sorted as well. The entries of C are written
	// directly: the row of A with permutation position r owns the slice [r * B.nnz(), (r + 1) *
	// B.nnz()) of C, so the slices do not overlap, their boundaries are known before any entry is
	// computed, and the parallel version needs neither a per block tensor nor a merge
	template <typename index_type, typename T>
	sparse_tensor<T, index_type, SPARSE_COO> tensor_product(
		const sparse_tensor<T, index_type, SPARSE_COO>& A,
		const sparse_tensor<T, index_type, SPARSE_COO>& B, const field_t& F,
		thread_pool* pool = nullptr) {

		std::vector<size_t> dimsB = B.dims();
		std::vector<size_t> dimsC = A.dims();
		dimsC.insert(dimsC.end(), dimsB.begin(), dimsB.end());

		sparse_tensor<T, index_type, SPARSE_COO> C(dimsC);

		if (A.nnz() == 0 || B.nnz() == 0) {
			return C;
		}

		const size_t rankA = A.rank();
		const size_t rankB = B.rank();
		const size_t rank = rankA + rankB;
		const size_t nnzA = A.nnz();
		const size_t nnzB = B.nnz();

		C.reserve(nnzA * nnzB);
		C.resize(nnzA * nnzB);

		auto permA = A.gen_perm();
		auto permB = B.gen_perm();

		auto fill_row = [&](const size_t r) {
			const auto posA = permA[r];
			const auto indexA = A.index(posA);
			const auto valA = A.val(posA);
			index_type* colptr = C.data.colptr + r * nnzB * rank;
			T* valptr = C.data.valptr + r * nnzB;
			for (size_t k = 0; k < nnzB; k++) {
				const auto posB = permB[k];
				s_copy(colptr + k * rank, indexA, rankA);
				s_copy(colptr + k * rank + rankA, B.index(posB), rankB);
				valptr[k] = scalar_mul(valA, B.val(posB), F);
			}
			};

		// every row costs the same, so handing out the rows hands out the work evenly
		constexpr size_t par_product_threshold = 1u << 17;
		if (pool != nullptr && nnzA * nnzB >= par_product_threshold) {
			const size_t nthread = pool->get_thread_count();
			if (nnzA >= 2 * nthread) {
				const size_t nblocks = nnzA < 64 * nthread ? nthread : 8 * nthread;
				pool->detach_loop(0, nnzA, fill_row, nblocks);
				pool->wait();
				return C;
			}
		}

		for (size_t r = 0; r < nnzA; r++)
			fill_row(r);

		return C;
	}

	// returned tensor is sorted
	template <typename index_type, typename T>
	sparse_tensor<T, index_type, SPARSE_COO> tensor_add(
		const sparse_tensor<T, index_type, SPARSE_COO>& A,
		const sparse_tensor<T, index_type, SPARSE_COO>& B,
		const field_t& F) {

		// if one of the tensors is empty, it is ok that dims of A or B are not defined
		if (A.alloc() == 0)
			return B;
		if (B.alloc() == 0)
			return A;

		if (A.rank() != B.rank()) {
			std::cerr << "Error: tensor_add: The dimensions of the two tensors do not match." << std::endl;
			return sparse_tensor<T, index_type, SPARSE_COO>();
		}

		for (size_t i = 0; i < A.rank(); i++) {
			if (A.dim(i) != B.dim(i)) {
				std::cerr << "Error: tensor_add: The dimensions of the two tensors do not match." << std::endl;
				return sparse_tensor<T, index_type, SPARSE_COO>();
			}
		}

		auto rank = A.rank();

		// if one of the tensors is zero
		if (A.nnz() == 0)
			return B;
		if (B.nnz() == 0)
			return A;

		sparse_tensor<T, index_type, SPARSE_COO> C(A.dims(), A.nnz() + B.nnz());

		auto Aperm = A.gen_perm();
		auto Bperm = B.gen_perm();

		// double pointer
		size_t i = 0, j = 0;
		// C.zero();
		while (i < A.nnz() && j < B.nnz()) {
			auto posA = Aperm[i];
			auto posB = Bperm[j];
			auto indexA = A.index(posA);
			auto indexB = B.index(posB);
			int cmp = lexico_compare(indexA, indexB, rank);

			if (cmp < 0) {
				C.push_back(indexA, A.val(posA));
				i++;
			}
			else if (cmp > 0) {
				C.push_back(indexB, B.val(posB));
				j++;
			}
			else {
				auto val = scalar_add(A.val(posA), B.val(posB), F);

				if (val != 0)
					C.push_back(indexA, val);
				i++; j++;
			}
		}
		while (i < A.nnz()) {
			auto posA = Aperm[i];
			C.push_back(A.index(posA), A.val(posA));
			i++;
		}
		while (j < B.nnz()) {
			auto posB = Bperm[j];
			C.push_back(B.index(posB), B.val(posB));
			j++;
		}

		return C;
	}

	// A += B, we assume that A and B are sorted
	template <typename index_type, typename T>
	void tensor_sum_replace(
		sparse_tensor<T, index_type, SPARSE_COO>& A,
		const sparse_tensor<T, index_type, SPARSE_COO>& B, const field_t& F) {

		// if one of the tensors is empty, it is ok that dims of A or B are not defined
		if (A.alloc() == 0) {
			A = B;
			return;
		}
		if (B.alloc() == 0)
			return;

		auto dimsC = A.dims();
		auto rank = A.rank();

		if (A.rank() != B.rank()) {
			std::cerr << "Error: tensor_sum_replace: The dimensions of the two tensors do not match." << std::endl;
			return;
		}

		for (size_t i = 0; i < A.rank(); i++) {
			if (A.dim(i) != B.dim(i)) {
				std::cerr << "Error: tensor_sum_replace: The dimensions of the two tensors do not match." << std::endl;
				return;
			}
		}

		if (!(A.check_sorted() && B.check_sorted())) {
			std::cerr << "Error: tensor_sum_replace: tensor_sum_replace: Both tensors must be sorted." << std::endl;
			return;
		}

		// if one of the tensors is zero
		if (A.nnz() == 0) {
			A = B;
			return;
		}
		if (B.nnz() == 0)
			return;

		if (&A == &B) {
			for (size_t i = 0; i < A.nnz(); i++) {
				A.val(i) = scalar_add(A.val(i), A.val(i), F);
			}
			return;
		}

		// double pointer, from the end to the beginning
		size_t ptr1 = A.nnz(), ptr2 = B.nnz();
		size_t ptr = A.nnz() + B.nnz();

		A.resize(ptr);

		while (ptr1 > 0 && ptr2 > 0) {
			int order = lexico_compare(A.index(ptr1 - 1), B.index(ptr2 - 1), rank);

			if (order == 0) {
				auto entry = scalar_add(A.val(ptr1 - 1), B.val(ptr2 - 1), F);
				if (entry != 0) {
					s_copy(A.index(ptr - 1), A.index(ptr1 - 1), rank);
					A.val(ptr - 1) = std::move(entry);
					ptr--;
				}
				ptr1--;
				ptr2--;
			}
			else if (order < 0) {
				s_copy(A.index(ptr - 1), B.index(ptr2 - 1), rank);
				A.val(ptr - 1) = B.val(ptr2 - 1);
				ptr2--;
				ptr--;
			}
			else {
				s_copy(A.index(ptr - 1), A.index(ptr1 - 1), rank);
				A.val(ptr - 1) = std::move(A.val(ptr1 - 1));
				ptr1--;
				ptr--;
			}
		}
		while (ptr2 > 0) {
			s_copy(A.index(ptr - 1), B.index(ptr2 - 1), rank);
			A.val(ptr - 1) = B.val(ptr2 - 1);
			ptr2--;
			ptr--;
		}

		// the merged entries were written at the end and A's untouched prefix is at the front,
		// so [ptr1, ptr) only holds stale copies: slide the merged part down to close the gap
		const size_t total = A.nnz();
		for (size_t i = 0; i < total - ptr; i++) {
			s_copy(A.index(ptr1 + i), A.index(ptr + i), rank);
			A.val(ptr1 + i) = std::move(A.val(ptr + i));
		}
		A.resize(ptr1 + (total - ptr));
	}

	// the result is sorted
	template <typename index_type, typename T>
	sparse_tensor<T, index_type, SPARSE_COO> tensor_contract(
		const sparse_tensor<T, index_type, SPARSE_COO>& A,
		const sparse_tensor<T, index_type, SPARSE_COO>& B,
		const std::vector<size_t>& i1, const std::vector<size_t>& i2,
		const field_t& F, thread_pool* pool = nullptr) {

		using index_v = std::vector<index_type>;
		using index_p = index_type*;

		if (i1.size() != i2.size()) {
			std::cerr << "Error: tensor_contract: The size of the two contract sets do not match." << std::endl;
			return sparse_tensor<T, index_type, SPARSE_COO>();
		}

		if (i1.size() == 0) {
			return tensor_product(A, B, F, pool);
		}

		// the indices of a contract set have to be in range and pairwise distinct: a repeated index
		// would make the tail of the index vector a non permutation and silently misplace the entries
		if (!in_range_and_distinct(i1, A.rank()) || !in_range_and_distinct(i2, B.rank())) {
			std::cerr << "Error: tensor_contract: The contract indices are out of range or repeated." << std::endl;
			return sparse_tensor<T, index_type, SPARSE_COO>();
		}

		auto dimsA = A.dims();
		auto dimsB = B.dims();

		for (size_t k = 0; k < i1.size(); k++) {
			if (dimsA[i1[k]] != dimsB[i2[k]]) {
				std::cerr << "Error: tensor_contract: The dimensions of the two tensors do not match." << std::endl;
				return sparse_tensor<T, index_type, SPARSE_COO>();
			}
		}

		// the dimensions of the result
		std::vector<size_t> dimsC, index_perm_A, index_perm_B;
		for (size_t k = 0; k < dimsA.size(); k++) {
			// if k is not in i1, we add it to dimsC and index_perm_A
			if (std::find(i1.begin(), i1.end(), k) == i1.end()) {
				dimsC.push_back(dimsA[k]);
				index_perm_A.push_back(k);
			}
		}
		index_perm_A.insert(index_perm_A.end(), i1.begin(), i1.end());
		for (size_t k = 0; k < dimsB.size(); k++) {
			// if k is not in i2, we add it to dimsC and index_perm_B
			if (std::find(i2.begin(), i2.end(), k) == i2.end()) {
				dimsC.push_back(dimsB[k]);
				index_perm_B.push_back(k);
			}
		}
		// B is ordered by the contract tuple and then by its free indices, so that the entries that
		// share a contract tuple are consecutive and a row of the result is built by looking up the
		// contract tuple of each of its entries in B, instead of by matching every pair of rows of A
		// and B
		index_perm_B.insert(index_perm_B.begin(), i2.begin(), i2.end());

		auto permA = A.gen_perm(index_perm_A);
		auto permB = B.gen_perm(index_perm_B);

		sparse_tensor<T, index_type, SPARSE_COO> C(dimsC);

		auto i1i2_size = i1.size();
		auto left_size_A = A.rank() - i1i2_size;
		auto left_size_B = B.rank() - i1i2_size;

		// an empty operand leaves the caches below empty, so there is nothing to read from them
		if (A.nnz() == 0 || B.nnz() == 0)
			return C;

		// the contract tuple and the free tuple of every entry, in the order of the permutations: the
		// loops below then compare neighbouring entries without following an index permutation
		// a tuple is packed into a number in mixed radix whenever its dimensions allow it, because a
		// packed key compares with one comparison instead of one per position, and because the keys of
		// the entries that start a run can be kept in a flat array, which makes the binary searches
		// below walk that array instead of chasing rowptrB. A tuple that does not fit keeps its cache
		// and is compared position by position
		constexpr size_t max_buckets = 1u << 20;
		constexpr size_t par_pack_threshold = 1u << 17;
		std::vector<size_t> cstride(i1i2_size, 1), stride_leftA(left_size_A, 1), stride_leftB(left_size_B, 1);
		bool keyed_contract = i1i2_size > 0, keyed_leftA = left_size_A > 0, keyed_leftB = left_size_B > 0;
		for (size_t l = i1i2_size; l-- > 1;) {
			const size_t dim = dimsA[i1[l]];
			if (dim == 0 || cstride[l] > std::numeric_limits<size_t>::max() / dim) {
				keyed_contract = false;
				break;
			}
			cstride[l - 1] = cstride[l] * dim;
		}
		for (size_t l = left_size_A; l-- > 1;) {
			const size_t dim = dimsA[index_perm_A[l]];
			if (dim == 0 || stride_leftA[l] > std::numeric_limits<size_t>::max() / dim) {
				keyed_leftA = false;
				break;
			}
			stride_leftA[l - 1] = stride_leftA[l] * dim;
		}
		for (size_t l = left_size_B; l-- > 1;) {
			const size_t dim = dimsB[index_perm_B[i1i2_size + l]];
			if (dim == 0 || stride_leftB[l] > std::numeric_limits<size_t>::max() / dim) {
				keyed_leftB = false;
				break;
			}
			stride_leftB[l - 1] = stride_leftB[l] * dim;
		}

		std::vector<index_type> index_A_cache, index_B_cache;
		std::vector<index_type> index_leftB_cache(left_size_B * B.nnz());
		std::vector<size_t> key_contract_A, key_contract_B, key_leftA, key_leftB;
		if (keyed_contract) {
			key_contract_A.resize(A.nnz());
			key_contract_B.resize(B.nnz());
		}
		if (keyed_leftA)
			key_leftA.resize(A.nnz());
		if (keyed_leftB)
			key_leftB.resize(B.nnz());

		// a label below zero would not order like the tuple once packed, so the keys are dropped when
		// an entry says otherwise, and the tuples are then compared position by position
		std::atomic<bool> negative{ false };
		for (size_t k = 0; k < A.nnz() && (keyed_contract || keyed_leftA); k++) {
			auto ptr = A.index(permA[k]);
			if (keyed_contract) {
				size_t key = 0;
				for (size_t l = 0; l < i1i2_size; l++) {
					const auto value = ptr[i1[l]];
					if (value < 0)
						negative.store(true, std::memory_order_relaxed);
					key += static_cast<size_t>(value) * cstride[l];
				}
				key_contract_A[k] = key;
			}
			if (keyed_leftA) {
				size_t key = 0;
				for (size_t l = 0; l < left_size_A; l++) {
					const auto value = ptr[index_perm_A[l]];
					if (value < 0)
						negative.store(true, std::memory_order_relaxed);
					key += static_cast<size_t>(value) * stride_leftA[l];
				}
				key_leftA[k] = key;
			}
		}

		// the same work on B, which holds most of the entries of the pair: the entries of B are the
		// inner loop of the contraction, so this pass is the longest serial stretch left in the
		// parallel path and it is handed to the pool
		auto pack_B = [&](const size_t k) {
			auto ptr = B.index(permB[k]);
			for (size_t l = 0; l < left_size_B; l++)
				index_leftB_cache[k * left_size_B + l] = ptr[index_perm_B[i1i2_size + l]];
			if (keyed_contract) {
				size_t key = 0;
				for (size_t l = 0; l < i1i2_size; l++) {
					const auto value = ptr[i2[l]];
					if (value < 0)
						negative.store(true, std::memory_order_relaxed);
					key += static_cast<size_t>(value) * cstride[l];
				}
				key_contract_B[k] = key;
			}
			if (keyed_leftB) {
				size_t key = 0;
				for (size_t l = 0; l < left_size_B; l++) {
					const auto value = ptr[index_perm_B[i1i2_size + l]];
					if (value < 0)
						negative.store(true, std::memory_order_relaxed);
					key += static_cast<size_t>(value) * stride_leftB[l];
				}
				key_leftB[k] = key;
			}
			};

		const size_t nthread = pool == nullptr ? 1 : pool->get_thread_count();
		if (pool != nullptr && B.nnz() >= par_pack_threshold) {
			pool->detach_loop(0, B.nnz(), pack_B, 4 * nthread);
			pool->wait();
		}
		else {
			for (size_t k = 0; k < B.nnz(); k++)
				pack_B(k);
		}

		if (negative.load(std::memory_order_relaxed)) {
			keyed_contract = false;
			keyed_leftA = false;
			keyed_leftB = false;
			key_contract_A.clear();
			key_contract_B.clear();
			key_leftA.clear();
			key_leftB.clear();
		}
		if (!keyed_contract) {
			index_A_cache.resize(i1i2_size * A.nnz());
			index_B_cache.resize(i1i2_size * B.nnz());
			for (size_t k = 0; k < A.nnz(); k++) {
				auto ptr = A.index(permA[k]);
				for (size_t l = 0; l < i1i2_size; l++)
					index_A_cache[k * i1i2_size + l] = ptr[i1[l]];
			}
			for (size_t k = 0; k < B.nnz(); k++) {
				auto ptr = B.index(permB[k]);
				for (size_t l = 0; l < i1i2_size; l++)
					index_B_cache[k * i1i2_size + l] = ptr[i2[l]];
			}
		}

		// the values of the entries in the order of the permutations: the loops below walk the runs in
		// that order, so reading the values through the permutation would gather them at random and
		// miss the cache on every step; the copy that builds these arrays is sequential
		std::vector<T> val_A(A.nnz()), val_B(B.nnz());
		for (size_t k = 0; k < A.nnz(); k++)
			val_A[k] = A.val(permA[k]);
		if (pool != nullptr && B.nnz() >= par_pack_threshold) {
			pool->detach_loop(0, B.nnz(), [&](const size_t k) { val_B[k] = B.val(permB[k]); }, 4 * nthread);
			pool->wait();
		}
		else {
			for (size_t k = 0; k < B.nnz(); k++)
				val_B[k] = B.val(permB[k]);
		}

		auto equal_except = [](const index_p a, const index_p b, const std::vector<size_t>& perm, const size_t len) {
			for (size_t i = 0; i < len; i++) {
				if (a[perm[i]] != b[perm[i]])
					return false;
			}
			return true;
			};

		// the rows of A: consecutive entries that share the free part, which the keys of the free part
		// of A tell apart without touching the index vectors again
		std::vector<size_t> rowptrA;
		rowptrA.push_back(0);
		if (keyed_leftA) {
			for (size_t k = 1; k < A.nnz(); k++) {
				if (key_leftA[k] != key_leftA[k - 1])
					rowptrA.push_back(k);
			}
		}
		else {
			for (size_t k = 1; k < A.nnz(); k++) {
				if (!equal_except(A.index(permA[rowptrA.back()]), A.index(permA[k]), index_perm_A, left_size_A))
					rowptrA.push_back(k);
			}
		}
		rowptrA.push_back(A.nnz());

		// the runs of B: consecutive entries that share the contract tuple
		std::vector<size_t> rowptrB;
		rowptrB.push_back(0);
		if (keyed_contract) {
			for (size_t k = 1; k < B.nnz(); k++) {
				if (key_contract_B[k] != key_contract_B[k - 1])
					rowptrB.push_back(k);
			}
		}
		else {
			for (size_t k = 1; k < B.nnz(); k++) {
				if (lexico_compare(index_B_cache.data() + (k - 1) * i1i2_size,
					index_B_cache.data() + k * i1i2_size, i1i2_size) != 0)
					rowptrB.push_back(k);
			}
		}
		rowptrB.push_back(B.nnz());

		const size_t runsB = rowptrB.size() - 1;

		// the contract tuple of the entries that start the runs, so that the search below reads one
		// flat array, and, when the contract tuple is a single label that is small enough, the run
		// that holds each label, so that the search of the run of an entry of A is one array read
		std::vector<size_t> run_key, run_of_value;
		if (keyed_contract) {
			run_key.resize(runsB);
			for (size_t r = 0; r < runsB; r++)
				run_key[r] = key_contract_B[rowptrB[r]];
		}
		if (keyed_contract && i1i2_size == 1 && dimsA[i1[0]] > 0 && dimsA[i1[0]] <= max_buckets) {
			run_of_value.assign(dimsA[i1[0]], std::numeric_limits<size_t>::max());
			for (size_t r = 0; r < runsB; r++)
				run_of_value[run_key[r]] = r;
		}

		// the runs of B that a row of A reaches: the entries of a row are ordered by their contract
		// tuple, so the run of B that holds a contract tuple is found by binary search, and the search
		// of the next entry starts at the run found for the previous one; when the contract tuple is a
		// single small label the lookup table answers in one read, and no search is needed at all.
		// The work of a row is the number of entries of B it reads, which the rows are handed out by
		// when the contraction runs on several threads
		auto select = [&](const size_t k, std::vector<size_t>& run_first, std::vector<size_t>& run_last,
			std::vector<T>* run_val) {
			const bool tabulated = !run_of_value.empty();
			size_t work = 0;
			size_t lo = 0;
			for (size_t ptrA = rowptrA[k]; ptrA < rowptrA[k + 1]; ptrA++) {
				size_t run;

				if (tabulated) {
					const size_t value = key_contract_A[ptrA];
					if (value >= run_of_value.size())
						continue;
					run = run_of_value[value];
					if (run == std::numeric_limits<size_t>::max())
						continue;
				}
				else if (keyed_contract) {
					const size_t key = key_contract_A[ptrA];
					size_t low = lo, high = runsB;
					while (low < high) {
						const size_t mid = low + (high - low) / 2;
						if (key > run_key[mid])
							low = mid + 1;
						else
							high = mid;
					}

					// every remaining contract tuple of the row is larger than every one of B
					if (low == runsB)
						break;

					lo = low;

					if (key != run_key[low])
						continue;

					run = low;
				}
				else {
					auto key = index_A_cache.data() + ptrA * i1i2_size;
					size_t low = lo, high = runsB;
					while (low < high) {
						const size_t mid = low + (high - low) / 2;
						if (lexico_compare(index_B_cache.data() + rowptrB[mid] * i1i2_size, key, i1i2_size) < 0)
							low = mid + 1;
						else
							high = mid;
					}

					if (low == runsB)
						break;

					lo = low;

					if (lexico_compare(index_B_cache.data() + rowptrB[low] * i1i2_size, key, i1i2_size) != 0)
						continue;

					run = low;
				}

				// the contract tuples of a row are distinct, so the runs it reaches are disjoint
				run_first.push_back(rowptrB[run]);
				run_last.push_back(rowptrB[run + 1]);
				work += rowptrB[run + 1] - rowptrB[run];
				if (run_val != nullptr)
					run_val->push_back(val_A[ptrA]);
			}
			return work;
			};

		auto method = [&](sparse_tensor<T, index_type>& C, size_t ss, size_t ee) {
			index_v indexC(dimsC.size());

			// the runs of B that the row that is being built reaches, the entry of A that reaches each
			// of them, the entry of B each run is read at, and the heap that merges the runs; the
			// buffers are reused across the rows
			std::vector<size_t> run_first, run_last, pos;
			std::vector<std::pair<size_t, size_t>> heap;
			std::vector<T> run_val;

			for (size_t k = ss; k < ee; k++) {
				// from rowptrA[k] to rowptrA[k + 1] are the same
				auto startA = rowptrA[k];

				for (size_t l = 0; l < left_size_A; l++)
					indexC[l] = A.index(permA[startA])[index_perm_A[l]];

				run_first.clear();
				run_last.clear();
				run_val.clear();
				select(k, run_first, run_last, &run_val);

				const size_t m = run_first.size();
				if (m == 0)
					continue;

				// the free tuples of a run are ordered and distinct, so a row that reaches a single run
				// that keeps at least one free index is written out as it is
				if (m == 1 && left_size_B > 0) {
					for (size_t ptrB = run_first[0]; ptrB < run_last[0]; ptrB++) {
						const T entry = scalar_mul(run_val[0], val_B[ptrB], F);
						if (entry != 0) {
							s_copy(indexC.data() + left_size_A, index_leftB_cache.data() + ptrB * left_size_B,
								left_size_B);
							C.push_back(indexC, entry);
						}
					}
				}
				// the runs of such a contraction hold one entry each, since a run is a maximal group of
				// entries that share the whole index vector when there is no free index, so the row adds
				// up to one entry of the result and there is nothing to merge
				else if (left_size_B == 0) {
					T entry = 0;
					for (size_t j = 0; j < m; j++) {
						for (size_t ptrB = run_first[j]; ptrB < run_last[j]; ptrB++)
							entry = scalar_add(entry, scalar_mul(run_val[j], val_B[ptrB], F), F);
					}
					if (entry != 0)
						C.push_back(indexC, entry);
				}
				// two runs merge with two pointers, which beats a heap of two entries
				else if (m == 2) {
					size_t p0 = run_first[0], p1 = run_first[1];
					const bool keyed = !key_leftB.empty();
					auto advance = [&](const size_t j, const size_t ptrB) {
						const T entry = scalar_mul(run_val[j], val_B[ptrB], F);
						if (entry != 0) {
							s_copy(indexC.data() + left_size_A, index_leftB_cache.data() + ptrB * left_size_B,
								left_size_B);
							C.push_back(indexC, entry);
						}
						};
					auto compare_free = [&](const size_t a, const size_t b) {
						if (keyed)
							return key_leftB[a] < key_leftB[b] ? -1 : key_leftB[a] > key_leftB[b] ? 1 : 0;
						return lexico_compare(index_leftB_cache.data() + a * left_size_B,
							index_leftB_cache.data() + b * left_size_B, left_size_B);
						};

					while (p0 < run_last[0] && p1 < run_last[1]) {
						const int cmp = compare_free(p0, p1);
						if (cmp < 0) {
							advance(0, p0);
							p0++;
						}
						else if (cmp > 0) {
							advance(1, p1);
							p1++;
						}
						else {
							const T entry = scalar_add(scalar_mul(run_val[0], val_B[p0], F),
								scalar_mul(run_val[1], val_B[p1], F), F);
							if (entry != 0) {
								s_copy(indexC.data() + left_size_A, index_leftB_cache.data() + p0 * left_size_B,
									left_size_B);
								C.push_back(indexC, entry);
							}
							p0++;
							p1++;
						}
					}
					while (p0 < run_last[0])
						advance(0, p0++);
					while (p1 < run_last[1])
						advance(1, p1++);
				}
				// a row that reaches several runs, or a run that keeps no free index, is built by merging
				// them: the free tuples of one run are ordered, so the smallest of the entries that wait
				// at the front of the runs comes first, and the entries that share a free tuple are added
				// up
				else {
					pos.resize(m);

					const bool keyed = !key_leftB.empty();

					// the heap holds the key of the entry that waits at the front of a run next to the
					// slot of that run: the order of two runs is then one comparison of two words that
					// sit in the same array, where reading the key through the position table is a load
					// that depends on the heap entry; a run whose free tuple has no key is ordered by
					// walking its tuple
					heap.clear();
					for (size_t j = 0; j < m; j++) {
						pos[j] = run_first[j];
						heap.emplace_back(keyed ? key_leftB[pos[j]] : 0, j);
					}

					auto free_front = [&](const size_t j) {
						return index_leftB_cache.data() + pos[j] * left_size_B;
						};
					auto later = [&](const std::pair<size_t, size_t>& a, const std::pair<size_t, size_t>& b) {
						if (keyed)
							return a.first > b.first;
						return lexico_compare(free_front(a.second), free_front(b.second), left_size_B) > 0;
						};
					std::make_heap(heap.begin(), heap.end(), later);

					while (!heap.empty()) {
						// free_min stays valid: the cache is not modified below, only the positions are
						const index_p free_min = free_front(heap.front().second);
						const size_t key_min = heap.front().first;
						T entry = 0;

						// every run whose front holds the smallest free tuple contributes to the same
						// entry of the result
						do {
							const size_t j = heap.front().second;
							std::pop_heap(heap.begin(), heap.end(), later);
							heap.pop_back();
							entry = scalar_add(entry, scalar_mul(run_val[j], val_B[pos[j]], F), F);
							if (++pos[j] < run_last[j]) {
								heap.emplace_back(keyed ? key_leftB[pos[j]] : 0, j);
								std::push_heap(heap.begin(), heap.end(), later);
							}
						} while (!heap.empty() && (keyed ? heap.front().first == key_min
							: lexico_compare(free_front(heap.front().second), free_min, left_size_B) == 0));

						if (entry != 0) {
							s_copy(indexC.data() + left_size_A, free_min, left_size_B);
							C.push_back(indexC, entry);
						}
					}
				}
			}
			};

		// parallel version
		if (pool != nullptr) {
			const size_t rows = rowptrA.size() - 1;

			if (rows < 2 * nthread) {
				method(C, 0, rows);
				return C;
			}

			size_t nblocks = rows < 64 * nthread ? nthread : 8 * nthread;

			// the work of a row is the number of entries of B that it reads, which varies widely from
			// row to row, so the rows are handed to the blocks by that work instead of by their number
			std::vector<size_t> work_till(rows + 1, 0);
			pool->detach_loop(0, rows, [&](const size_t k) {
				std::vector<size_t> run_first, run_last;
				work_till[k + 1] = select(k, run_first, run_last, nullptr);
				}, nblocks);
			pool->wait();
			for (size_t k = 0; k < rows; k++)
				work_till[k + 1] += work_till[k];

			const size_t allwork = work_till[rows];

			std::vector<std::pair<size_t, size_t>> ranges(nblocks);
			size_t start = 0;
			for (size_t i = 0; i < nblocks; i++) {
				const size_t target = static_cast<size_t>(static_cast<unsigned long long>(allwork) * (i + 1) / nblocks);
				size_t end = start;
				while (end < rows && work_till[end] < target)
					end++;
				if (end == start && end < rows)
					end++;
				ranges[i] = { start, end };
				start = end;
			}
			ranges[nblocks - 1].second = rows;

			std::vector<sparse_tensor<T, index_type, SPARSE_COO>> Cs(nblocks, C);

			pool->detach_sequence(0, nblocks, [&](size_t i) {
				method(Cs[i], ranges[i].first, ranges[i].second);
				});
			pool->wait();

			// merge the results
			size_t allnnz = 0;
			std::vector<size_t> start_pos(nblocks);
			for (size_t i = 0; i < nblocks; i++) {
				start_pos[i] = allnnz;
				allnnz += Cs[i].nnz();
			}

			C.reserve(allnnz);
			C.resize(allnnz);
			pool->detach_loop(0, nblocks, [&](size_t i) {
				const auto tmpnnz = Cs[i].nnz();
				T* valptr = C.data.valptr + start_pos[i];
				index_p colptr = C.data.colptr + start_pos[i] * C.rank();
				s_copy(colptr, Cs[i].data.colptr, tmpnnz * C.rank());
				s_copy(valptr, Cs[i].data.valptr, tmpnnz);
				Cs[i].clear();
				});
			pool->wait();

			return C;
		}
		else {
			method(C, 0, rowptrA.size() - 1);
			return C;
		}
	}

	template <typename index_type, typename T>
	sparse_tensor<T, index_type, SPARSE_COO> tensor_contract(
		const sparse_tensor<T, index_type, SPARSE_COO>& A,
		const sparse_tensor<T, index_type, SPARSE_COO>& B,
		const size_t i, const size_t j, const field_t& F, thread_pool* pool = nullptr) {

		return tensor_contract(A, B, std::vector<size_t>{ i }, std::vector<size_t>{ j }, F, pool);
	}

	// contract the a-th index of A with the 0-th index of B, then move the remaining indices of B into
	// the slot of the contracted index: the result has rank A.rank() + B.rank() - 2 and the order
	// [A[0..a-1], B[1], A[a+1..], B[2..]], so that contracting with a matrix reads like a matrix
	// product. The result is sorted unless sort_ind is false.
	template <typename index_type, typename T>
	sparse_tensor<T, index_type, SPARSE_COO> tensor_contract_2(
		const sparse_tensor<T, index_type, SPARSE_COO>& A,
		const sparse_tensor<T, index_type, SPARSE_COO>& B,
		const size_t a, const field_t& F, thread_pool* pool = nullptr, const bool sort_ind = true) {

		const size_t rankA = A.rank();
		const size_t rankB = B.rank();

		if (rankA == 0 || rankB == 0 || a >= rankA) {
			std::cerr << "Error: tensor_contract_2: cannot contract index " << a << " of a rank " << rankA
				<< " tensor with a rank " << rankB << " tensor." << std::endl;
			return sparse_tensor<T, index_type, SPARSE_COO>();
		}

		// tensor_contract returns the order [A except a] ++ [B except 0], so perm[i] is the old slot of
		// the new slot i, and the rank of the result fixes the size of perm
		auto C = tensor_contract(A, B, a, 0, F, pool);

		std::vector<size_t> perm;
		perm.reserve(rankA + rankB - 2);
		for (size_t k = 0; k < a; k++)
			perm.push_back(k);
		if (rankB > 1)
			perm.push_back(rankA - 1);
		for (size_t k = a; k + 1 < rankA; k++)
			perm.push_back(k);
		for (size_t k = rankA; k + 1 < rankA + rankB - 1; k++)
			perm.push_back(k);

		C.transpose_replace(perm, pool, sort_ind);

		return C;
	}

	// self contraction: C[rest] = sum_k A[..., i=k, ..., j=k, ...], requires i != j
	template <typename index_type, typename T>
	sparse_tensor<T, index_type, SPARSE_COO> tensor_contract(
		const sparse_tensor<T, index_type, SPARSE_COO>& A,
		const size_t i, const size_t j, const field_t& F, thread_pool* pool = nullptr) {

		using index_v = std::vector<index_type>;
		using index_p = index_type*;

		if (i >= A.rank() || j >= A.rank()) {
			std::cerr << "Error: tensor_contract: cannot contract index " << i << " and " << j << " of a rank "
				<< A.rank() << " tensor." << std::endl;
			return sparse_tensor<T, index_type, SPARSE_COO>();
		}

		if (i == j) {
			std::cerr << "Error: tensor_contract: The two contraction indices must be different." << std::endl;
			return sparse_tensor<T, index_type, SPARSE_COO>();
		}

		if (i > j)
			return tensor_contract(A, j, i, F, pool);

		// then i < j

		std::vector<size_t> dimsA = A.dims();
		auto rank = A.rank();

		std::vector<size_t> dimsC;
		for (size_t k = 0; k < dimsA.size(); k++) {
			if (k != i && k != j)
				dimsC.push_back(dimsA[k]);
		}

		std::vector<size_t> equal_ind_list;

		// search for the same indices
		constexpr size_t par_scan_threshold = 1u << 17;
		if (pool != nullptr && A.nnz() >= par_scan_threshold) {
			// the scan is a plain pass over the tensor and is what the serial version spends most of
			// its time on, so above the threshold it is worth handing its blocks to the pool
			const size_t nthread = pool->get_thread_count();
			const size_t nblocks = A.nnz() < 64 * nthread ? nthread : 8 * nthread;
			std::vector<std::vector<size_t>> parts(nblocks);

			pool->detach_loop(0, nblocks, [&](const size_t blk) {
				const size_t first = A.nnz() * blk / nblocks;
				const size_t last = A.nnz() * (blk + 1) / nblocks;
				auto& part = parts[blk];
				for (size_t k = first; k < last; k++) {
					if (A.index(k)[i] == A.index(k)[j])
						part.push_back(k);
				}
				}, nblocks);
			pool->wait();

			size_t total = 0;
			for (auto& part : parts)
				total += part.size();
			equal_ind_list.reserve(total);
			// the blocks cover increasing ranges of k, so the concatenation keeps the order
			for (auto& part : parts) {
				equal_ind_list.insert(equal_ind_list.end(), part.begin(), part.end());
				std::vector<size_t>().swap(part);
			}
		}
		else {
			for (size_t k = 0; k < A.nnz(); k++) {
				if (A.index(k)[i] == A.index(k)[j]) {
					equal_ind_list.push_back(k);
				}
			}
		}

		std::vector<size_t> index_perm;
		for (size_t k = 0; k < rank; k++) {
			if (k != i && k != j)
				index_perm.push_back(k);
		}
		index_perm.push_back(i);
		index_perm.push_back(j);

		auto perm = perm_init(equal_ind_list.size());
		auto by_index = [&](size_t a, size_t b) {
			return lexico_compare(A.index(equal_ind_list[a]), A.index(equal_ind_list[b]), index_perm) < 0;
			};
		// the matched entries are usually a small fraction of the tensor, so the sort is rarely the
		// bottleneck: above the threshold it is worth the threads, below it the serial version is
		// faster and a call without a pool (pool == nullptr) stays single threaded
		constexpr size_t par_sort_threshold = 1u << 16;
		if (pool != nullptr && equal_ind_list.size() >= par_sort_threshold)
			std::sort(std::execution::par, perm.begin(), perm.end(), by_index);
		else
			std::sort(perm.begin(), perm.end(), by_index);

		std::vector<size_t> rowptr;
		rowptr.push_back(0);
		auto equal_except_ij = [&](const index_p a, const index_p b) {
			// do not compare the i-th and j-th index
			for (size_t k = 0; k < rank; k++)
				if (k != i && k != j && a[k] != b[k])
					return false;
			return true;
			};

		for (size_t k = 1; k < equal_ind_list.size(); k++) {
			if (!equal_except_ij(A.index(equal_ind_list[perm[k]]), A.index(equal_ind_list[perm[rowptr.back()]])))
				rowptr.push_back(k);
		}
		rowptr.push_back(equal_ind_list.size());

		sparse_tensor<T, index_type, SPARSE_COO> C(dimsC);

		// the merge cost scales with the number of matched entries, so a small match set stays on one
		// thread even when a pool is available, since the dispatch overhead would dominate
		if (pool != nullptr && equal_ind_list.size() >= par_scan_threshold
			&& rowptr.size() - 1 >= 2 * pool->get_thread_count()) {
			const size_t nrows = rowptr.size() - 1;
			const size_t nthread = pool->get_thread_count();
			const size_t nblocks_max = nrows < 64 * nthread ? nthread : 8 * nthread;

			// a row is the set of entries that share the free tuple and its length varies widely, so
			// the blocks take whole rows and are cut by the accumulated length of the rows instead of
			// by their number
			std::vector<std::pair<size_t, size_t>> ranges;
			ranges.reserve(nblocks_max);
			const size_t target = rowptr[nrows] / nblocks_max + 1;
			size_t start = 0, work = 0;
			for (size_t k = 0; k + 1 < nrows; k++) {
				work += rowptr[k + 1] - rowptr[k];
				if (work >= target && ranges.size() + 1 < nblocks_max) {
					ranges.push_back({ start, k + 1 });
					start = k + 1;
					work = 0;
				}
			}
			ranges.push_back({ start, nrows });

			const size_t nblocks = ranges.size();

			// the blocks (not the threads) fix the order of the merged result, since the
			// scheduler decides which thread runs which block
			std::vector<sparse_tensor<T, index_type, SPARSE_COO>> Cs(nblocks, C);

			auto method = [&](const size_t blk) {
				index_v indexC;
				indexC.reserve(rank - 2);
				for (size_t k = ranges[blk].first; k < ranges[blk].second; k++) {
					// from rowptr[k] to rowptr[k + 1] are the same
					auto start = rowptr[k];
					auto end = rowptr[k + 1];
					T entry = 0;
					for (size_t m = start; m < end; m++) {
						entry = scalar_add(entry, A.val(equal_ind_list[perm[m]]), F);
					}
					if (entry != 0) {
						indexC.clear();
						for (size_t l = 0; l < A.rank(); l++)
							if (l != i && l != j)
								indexC.push_back(A.index(equal_ind_list[perm[start]])[l]);
						Cs[blk].push_back(indexC, entry);
					}
				}
				};

			pool->detach_sequence(0, nblocks, method);
			pool->wait();

			// merge the results: the blocks cover increasing rows, so Cs is ordered
			size_t allnnz = 0;
			std::vector<size_t> start_pos(nblocks);
			for (size_t blk = 0; blk < nblocks; blk++) {
				start_pos[blk] = allnnz;
				allnnz += Cs[blk].nnz();
			}

			C.reserve(allnnz);
			C.resize(allnnz);
			pool->detach_loop(0, nblocks, [&](size_t blk) {
				auto tmpnnz = Cs[blk].nnz();
				T* valptr = C.data.valptr + start_pos[blk];
				index_p colptr = C.data.colptr + start_pos[blk] * C.rank();
				s_copy(colptr, Cs[blk].data.colptr, tmpnnz * C.rank());
				s_copy(valptr, Cs[blk].data.valptr, tmpnnz);
				Cs[blk].clear();
				});
			pool->wait();
		}
		else {
			index_v indexC;
			indexC.reserve(rank - 2);
			for (size_t k = 0; k < rowptr.size() - 1; k++) {
				// from rowptr[k] to rowptr[k + 1] are the same
				auto start = rowptr[k];
				auto end = rowptr[k + 1];
				T entry = 0;
				for (size_t m = start; m < end; m++) {
					entry = scalar_add(entry, A.val(equal_ind_list[perm[m]]), F);
				}
				if (entry != 0) {
					indexC.clear();
					for (size_t l = 0; l < A.rank(); l++)
						if (l != i && l != j)
							indexC.push_back(A.index(equal_ind_list[perm[start]])[l]);
					C.push_back(indexC, entry);
				}
			}
		}

		return C;
	}

	template <typename index_type, typename T>
	sparse_tensor<T, index_type, SPARSE_COO> tensor_dot(
		const sparse_tensor<T, index_type, SPARSE_COO>& A,
		const sparse_tensor<T, index_type, SPARSE_COO>& B,
		const field_t& F, thread_pool* pool = nullptr) {

		// otherwise A.rank() - 1 wraps around
		if (A.rank() == 0 || B.rank() == 0) {
			std::cerr << "Error: tensor_dot: cannot contract a rank 0 tensor." << std::endl;
			return sparse_tensor<T, index_type, SPARSE_COO>();
		}

		return tensor_contract(A, B, A.rank() - 1, 0, F, pool);
	}

	// usually B is a matrix, and A is a tensor, we want to contract all the dimensions of A with B
	// e.g. change a basis of a tensor
	// we always require that B is sorted
	template <typename index_type, typename T>
	sparse_tensor<T, index_type, SPARSE_COO> tensor_transform(
		const sparse_tensor<T, index_type, SPARSE_COO>& A,
		const sparse_tensor<T, index_type, SPARSE_COO>& B,
		const size_t start_index, const field_t& F, thread_pool* pool = nullptr) {

		auto rank = A.rank();
		if (start_index >= rank) {
			std::cerr << "Error: tensor_transform: cannot start at index " << start_index << " of a rank "
				<< rank << " tensor." << std::endl;
			return sparse_tensor<T, index_type, SPARSE_COO>();
		}

		// A keeps its relative index order (B's remaining indices are appended at the end), so the next
		// index to contract with B is again at start_index: i only counts the contractions
		auto C = tensor_contract(A, B, start_index, 0, F, pool);
		for (size_t i = start_index + 1; i < rank; i++) {
			C = tensor_contract(C, B, start_index, 0, F, pool);
		}

		return C;
	}

	template <typename index_type, typename T>
	void tensor_transform_replace(
		sparse_tensor<T, index_type, SPARSE_COO>& A,
		const sparse_tensor<T, index_type, SPARSE_COO>& B,
		const size_t start_index, const field_t& F, thread_pool* pool = nullptr) {

		auto rank = A.rank();
		if (start_index >= rank) {
			std::cerr << "Error: tensor_transform_replace: cannot start at index " << start_index << " of a rank "
				<< rank << " tensor." << std::endl;
			return;
		}

		// start_index does not move either, see tensor_transform
		for (size_t i = start_index; i < rank; i++) {
			A = tensor_contract(A, B, start_index, 0, F, pool);
		}
	}

	// a hash map from a key of a fixed number of size_t to a value of type T, used by einstein_sum to
	// hold the partial products of the sum: the keys are stored in one flat array so that an entry is
	// visited without following a pointer and the table grows with a single reallocation. A bucket holds
	// the high bits of the hash of its entry next to the index of the entry, so that the keys are only
	// read for the buckets that agree with the hash of the key that is looked up. A key holds one
	// slot per label of the sum, the value of the label in the entry shifted by the smallest value of that
	// label plus one: a zero then marks a slot that is not part of the key, either because the label has
	// not been joined yet or because it has already been summed out, so the key is hashed and compared as
	// it is, without keeping a list of the labels that are still open.
	template <typename T>
	struct einstein_accumulator {
		size_t keylen = 0;
		size_t count = 0;
		std::vector<size_t> keys;
		std::vector<T> vals;
		std::vector<uint64_t> buckets;

		void init(const size_t keylen_, const size_t cap) {
			keylen = keylen_;
			count = 0;
			size_t size = 16;
			while (size < 2 * cap)
				size *= 2;
			keys.assign(size * keylen, 0);
			vals.assign(size, T(0));
			buckets.assign(size, 0);
		}

		static size_t mix(const size_t h, const size_t v) {
			return h ^ (v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2));
		}

		size_t hash_key(const size_t* key) const {
			size_t h = 0;
			for (size_t i = 0; i < keylen; i++)
				h = mix(h, key[i]);
			return h;
		}

		// the index of the entry with this key, which is appended with a zero value when it is new
		size_t find_or_insert(const size_t* key) {
			if (4 * (count + 1) > 3 * buckets.size())
				grow();
			const size_t mask = buckets.size() - 1;
			const size_t h = hash_key(key);
			const uint64_t tag = hash_tag(h);
			size_t b = h & mask;
			while (buckets[b] != 0) {
				if ((buckets[b] & tag_mask) == tag && std::equal(key, key + keylen, entry(bucket_index(buckets[b]))))
					return bucket_index(buckets[b]);
				b = (b + 1) & mask;
			}
			const size_t a = count++;
			buckets[b] = tag | index_field(a);
			std::copy(key, key + keylen, entry(a));
			vals[a] = T(0);
			return a;
		}

	private:
		// a free bucket is zero, which no entry can be: the index field of an entry is its index plus one
		static constexpr uint64_t tag_mask = 0xFFFFFFFFull << 32;

		// the number of entries of the table stays far below 2^32: an entry needs more than 32 bytes of
		// keys and the table keeps a bucket per entry
		static uint64_t index_field(const size_t a) { return static_cast<uint64_t>(static_cast<uint32_t>(a + 1)); }
		static size_t bucket_index(const uint64_t b) { return static_cast<size_t>(static_cast<uint32_t>(b - 1)); }

		// the high bits of the hash, used to skip the comparison of the keys of the buckets that hold the
		// entry of a different key
		static uint64_t hash_tag(const size_t h) { return static_cast<uint64_t>(h >> 32) << 32; }

		size_t* entry(const size_t i) { return keys.data() + i * keylen; }
		const size_t* entry(const size_t i) const { return keys.data() + i * keylen; }

		void grow() {
			const size_t size = 2 * buckets.size();
			keys.resize(size * keylen);
			vals.resize(size);
			buckets.assign(size, 0);
			const size_t mask = size - 1;
			for (size_t a = 0; a < count; a++) {
				const size_t h = hash_key(entry(a));
				size_t b = h & mask;
				while (buckets[b] != 0)
					b = (b + 1) & mask;
				buckets[b] = hash_tag(h) | index_field(a);
			}
		}
	};

	// tensors {A,B,...}
	// index_sets {{i1,j1,...}, {i2,j2,...}, ...}
	// |{i1,j1,...}| = A.rank(), |{i2,j2,...}| = B.rank(), ...
	// index with the same number will be contracted
	// and the other indices will be sorted
	// a label repeated inside a single index set is contracted as well, so A_{i,i} is the trace of A

	// e.g. D = einstein_sum({ A,B,C }, { {0,1,4}, {2,1}, {2,3} })
	// D_{i0,i3,i4} = sum_{i1,i2} A_{i0,i1,i4} B_{i2,i1} C_{i2,i3}

	// The tensors are joined one at a time with a hash join: the entries of the tensor that is being joined
	// are looked up by their values on the labels they have in common with the partial products, so the
	// work is proportional to the number of partial products that really match an entry, and not to the
	// product of the numbers of rows of the tensors as walking the whole cartesian product would be. A
	// label is summed out as soon as no tensor below can hold it anymore.
	template <typename index_type, typename T>
	sparse_tensor<T, index_type, SPARSE_COO> einstein_sum(
		const std::vector<sparse_tensor<T, index_type, SPARSE_COO>*> tensors,
		const std::vector<std::vector<size_t>> index_sets,
		const field_t& F, thread_pool* pool = nullptr) {

		using accumulator_t = einstein_accumulator<T>;

		auto nt = tensors.size();
		if (nt == 0 || nt != index_sets.size()) {
			std::cerr << "Error: einstein_sum: The number of tensors does not match the number of index sets." << std::endl;
			return sparse_tensor<T, index_type, SPARSE_COO>();
		}
		for (size_t i = 0; i < nt; i++) {
			if (tensors[i]->rank() != index_sets[i].size()) {
				std::cerr << "Error: einstein_sum: The rank of the tensor does not match the index set." << std::endl;
				return sparse_tensor<T, index_type, SPARSE_COO>();
			}
		}

		// now is the valid case

		// first case is zero
		for (size_t i = 0; i < nt; i++) {
			if (tensors[i]->nnz() == 0)
				return sparse_tensor<T, index_type, SPARSE_COO>();
		}

		// the slot of a label is its rank among the labels of the sum, so the slots are ordered by their
		// label: a free index, which is a label that occurs exactly once, is written at the slot of its
		// label, and the result comes out with its indices in ascending label order
		std::map<size_t, size_t> slot_of_label;
		for (size_t i = 0; i < nt; i++) {
			for (size_t j = 0; j < index_sets[i].size(); j++) {
				if (slot_of_label.find(index_sets[i][j]) == slot_of_label.end())
					slot_of_label.emplace(index_sets[i][j], slot_of_label.size());
			}
		}
		const size_t nlabels = slot_of_label.size();

		// the slot of every index of every tensor, the slots of a tensor in ascending order, and the
		// positions of the slots inside a tensor: a label that occurs more than once in the same tensor
		// is read at all of its positions, since only the diagonal of that tensor contributes
		std::vector<std::vector<size_t>> entry_slot(nt), each_slot(nt), slot_start(nt), slot_pos(nt);
		std::vector<size_t> occurrence(nlabels, 0), last_tensor(nlabels, 0), free_tensor(nlabels, 0), free_pos(nlabels, 0);
		for (size_t i = 0; i < nt; i++) {
			entry_slot[i].resize(index_sets[i].size());
			for (size_t j = 0; j < index_sets[i].size(); j++) {
				const size_t slot = slot_of_label[index_sets[i][j]];
				entry_slot[i][j] = slot;
				if (occurrence[slot] == 0) {
					free_tensor[slot] = i;
					free_pos[slot] = j;
				}
				occurrence[slot]++;
				last_tensor[slot] = i;
			}
			each_slot[i] = entry_slot[i];
			std::sort(each_slot[i].begin(), each_slot[i].end());
			each_slot[i].erase(std::unique(each_slot[i].begin(), each_slot[i].end()), each_slot[i].end());
			slot_start[i].push_back(0);
			for (auto slot : each_slot[i]) {
				for (size_t j = 0; j < entry_slot[i].size(); j++) {
					if (entry_slot[i][j] == slot)
						slot_pos[i].push_back(j);
				}
				slot_start[i].push_back(slot_pos[i].size());
			}
		}

		// an index is stored in a key shifted by the smallest value of its label plus one, so that a zero
		// can mark a slot that is not part of the key; distinct values of a label stay distinct, and the
		// range of the indices does not matter
		std::vector<index_type> lower(nlabels, 0);
		std::vector<bool> has_lower(nlabels, false);
		for (size_t i = 0; i < nt; i++) {
			for (size_t e = 0; e < tensors[i]->nnz(); e++) {
				const index_type* ptr = tensors[i]->index(e);
				for (size_t j = 0; j < entry_slot[i].size(); j++) {
					const size_t slot = entry_slot[i][j];
					if (!has_lower[slot]) {
						has_lower[slot] = true;
						lower[slot] = ptr[j];
					}
					else if (ptr[j] < lower[slot]) {
						lower[slot] = ptr[j];
					}
				}
			}
		}

		// the accumulator holds one entry per tuple of the labels that are open, and the value of an entry
		// is the sum of the products of the tensors joined so far that agree on that tuple; the empty tuple
		// with value one is the product of no tensor
		std::vector<bool> open(nlabels, false);
		accumulator_t acc;
		acc.init(nlabels, 1);
		std::vector<size_t> key(nlabels, 0);
		acc.find_or_insert(key.data());
		acc.vals[0] = 1;

		int nthread = 1;
		if (pool != nullptr)
			nthread = static_cast<int>(pool->get_thread_count());

		std::vector<std::vector<size_t>> entry_value(nthread, std::vector<size_t>(nlabels, 0));
		std::vector<std::vector<size_t>> new_key(nthread, std::vector<size_t>(nlabels, 0));
		std::vector<size_t> join_slot, new_slot, fuse_slot, drop_slot;
		constexpr size_t par_join_nnz = 1u << 15;

		for (size_t i = 0; i < nt; i++) {
			if (acc.count == 0)
				break;

			join_slot.clear();
			new_slot.clear();
			for (auto slot : each_slot[i]) {
				if (open[slot])
					join_slot.push_back(slot);
				else
					new_slot.push_back(slot);
			}

			// a label is summed out once no tensor below can hold it: its value is not constrained anymore,
			// so the entries that agree on the other slots are added up. A label that the tensor has in
			// common with the accumulator is left out of the new key right away, which adds it up while the
			// tensor is joined, and the others are summed out below.
			fuse_slot.clear();
			drop_slot.clear();
			for (auto slot : join_slot) {
				if (last_tensor[slot] == i && occurrence[slot] > 1)
					fuse_slot.push_back(slot);
			}
			for (auto slot : new_slot) {
				if (last_tensor[slot] == i && occurrence[slot] > 1)
					drop_slot.push_back(slot);
			}
			for (auto slot : new_slot) {
				if (last_tensor[slot] != i || occurrence[slot] == 1)
					open[slot] = true;
			}

			// an index of the accumulator over the tuple of the slots the tensor has in common with it:
			// one lookup reaches the whole chain of the partial products that an entry of the tensor
			// extends, and the entries that share a tuple are in the same chain
			size_t sub_size = 1;
			while (sub_size < 2 * acc.count)
				sub_size *= 2;
			std::vector<size_t> sub_head(sub_size, std::numeric_limits<size_t>::max()), sub_next(acc.count);
			for (size_t a = 0; a < acc.count; a++) {
				const size_t* akey = acc.keys.data() + a * nlabels;
				size_t h = 0;
				for (auto slot : join_slot)
					h = accumulator_t::mix(h, akey[slot]);
				const size_t b = h & (sub_size - 1);
				sub_next[a] = sub_head[b];
				sub_head[b] = a;
			}

			// one accumulator per thread, they are merged below
			std::vector<accumulator_t> next(nthread);
			const size_t nnz = tensors[i]->nnz();
			const size_t cap = std::min<size_t>(acc.count * nnz, 1u << 12);
			for (auto& a : next)
				a.init(nlabels, cap / nthread + 1);

			auto join_entry = [&](accumulator_t& out, std::vector<size_t>& value, std::vector<size_t>& newkey, const size_t e) {
				const T val = tensors[i]->val(e);
				if (val == 0)
					return;
				const index_type* ptr = tensors[i]->index(e);
				// only the diagonal of a tensor with a repeated label contributes
				for (size_t k = 0; k < each_slot[i].size(); k++) {
					const size_t* pos = slot_pos[i].data() + slot_start[i][k];
					const size_t np = slot_start[i][k + 1] - slot_start[i][k];
					const index_type num = ptr[pos[0]];
					for (size_t l = 1; l < np; l++) {
						if (ptr[pos[l]] != num)
							return;
					}
					value[each_slot[i][k]] = static_cast<size_t>(num - lower[each_slot[i][k]]) + 1;
				}

				size_t h = 0;
				for (auto slot : join_slot)
					h = accumulator_t::mix(h, value[slot]);
				for (size_t a = sub_head[h & (sub_size - 1)]; a != std::numeric_limits<size_t>::max(); a = sub_next[a]) {
					const size_t* akey = acc.keys.data() + a * nlabels;
					bool is_match = true;
					for (auto slot : join_slot) {
						if (akey[slot] != value[slot]) {
							is_match = false;
							break;
						}
					}
					if (!is_match)
						continue;
					const T prod = scalar_mul(acc.vals[a], val, F);
					if (prod == 0)
						continue;
					std::copy(akey, akey + nlabels, newkey.begin());
					for (auto slot : fuse_slot)
						newkey[slot] = 0;
					for (auto slot : new_slot)
						newkey[slot] = value[slot];
					const size_t pos = out.find_or_insert(newkey.data());
					out.vals[pos] = scalar_add(out.vals[pos], prod, F);
				}
				};

			if (pool != nullptr && nnz >= par_join_nnz) {
				auto fill = [&](const size_t ss, const size_t ee) {
					const size_t id = thread_id();
					for (size_t e = ss; e < ee; e++)
						join_entry(next[id], entry_value[id], new_key[id], e);
					};
				pool->detach_blocks(size_t(0), nnz, fill, static_cast<size_t>(nthread));
				pool->wait();
			}
			else {
				for (size_t e = 0; e < nnz; e++)
					join_entry(next[0], entry_value[0], new_key[0], e);
			}

			for (size_t t = 1; t < next.size(); t++) {
				for (size_t a = 0; a < next[t].count; a++) {
					const size_t pos = next[0].find_or_insert(next[t].keys.data() + a * nlabels);
					next[0].vals[pos] = scalar_add(next[0].vals[pos], next[t].vals[a], F);
				}
			}
			acc = std::move(next[0]);

			if (!drop_slot.empty()) {
				accumulator_t reduced;
				reduced.init(nlabels, acc.count);
				for (size_t a = 0; a < acc.count; a++) {
					const size_t* akey = acc.keys.data() + a * nlabels;
					std::copy(akey, akey + nlabels, key.begin());
					for (auto slot : drop_slot)
						key[slot] = 0;
					const size_t pos = reduced.find_or_insert(key.data());
					reduced.vals[pos] = scalar_add(reduced.vals[pos], acc.vals[a], F);
				}
				for (auto slot : drop_slot)
					open[slot] = false;
				acc = std::move(reduced);
			}
		}

		// every label that occurs more than once has been summed out, so the slots left in a key are the
		// free indices of the result
		std::vector<size_t> free_slot;
		std::vector<size_t> dimsC;
		for (size_t slot = 0; slot < nlabels; slot++) {
			if (occurrence[slot] == 1) {
				free_slot.push_back(slot);
				dimsC.push_back(tensors[free_tensor[slot]]->dim(free_pos[slot]));
			}
		}

		sparse_tensor<T, index_type, SPARSE_COO> C(dimsC);

		// the entries of the accumulator are in the order of its table, so they are sorted by their index,
		// which is what the caller expects
		std::vector<size_t> order(acc.count);
		for (size_t a = 0; a < order.size(); a++)
			order[a] = a;
		std::sort(order.begin(), order.end(), [&](const size_t a, const size_t b) {
			return std::lexicographical_compare(acc.keys.data() + a * nlabels, acc.keys.data() + (a + 1) * nlabels,
				acc.keys.data() + b * nlabels, acc.keys.data() + (b + 1) * nlabels);
			});

		std::vector<index_type> index(free_slot.size());
		C.reserve(acc.count);
		for (auto a : order) {
			if (acc.vals[a] == 0)
				continue;
			const size_t* akey = acc.keys.data() + a * nlabels;
			for (size_t j = 0; j < free_slot.size(); j++) {
				index[j] = static_cast<index_type>(static_cast<size_t>(lower[free_slot[j]]) + akey[free_slot[j]] - 1);
			}
			C.push_back(index, acc.vals[a]);
		}

		return C;
	}

	// IO

	template <typename ScalarType, typename IndexType, typename T>
	sparse_tensor<ScalarType, IndexType, SPARSE_COO> sparse_tensor_read(T& st, const field_t& F, thread_pool* pool = nullptr, const bool sort_ind = true) {
		if (!st.is_open())
			return sparse_tensor<ScalarType, IndexType, SPARSE_COO>();

		std::string line;
		std::vector<IndexType> index;
		std::vector<size_t> dims;
		sparse_tensor<ScalarType, IndexType> tensor;

		while (std::getline(st, line)) {
			if (line.empty() || line[0] == '%')
				continue;

			size_t start = 0;
			size_t end = line.find(' ');
			while (end != std::string::npos) {
				if (start != end) {
					dims.push_back(string_to_ull(line.substr(start, end - start)));
				}
				start = end + 1;
				end = line.find(' ', start);
			}
			if (start < line.size()) {
				size_t nnz = string_to_ull(line.substr(start));
				tensor = sparse_tensor<ScalarType, IndexType, SPARSE_COO>(dims, nnz);
				index.reserve(dims.size());
			}
			break;
		}

		while (std::getline(st, line)) {
			if (line.empty() || line[0] == '%')
				continue;

			index.clear();
			size_t start = 0;
			size_t end = line.find(' ');
			size_t count = 0;

			while (end != std::string::npos && count < dims.size()) {
				if (start != end) {
					index.push_back(static_cast<IndexType>(string_to_ull(line.substr(start, end - start)) - 1));
					count++;
				}
				start = end + 1;
				end = line.find(' ', start);
			}

			if (count != dims.size()) {
				std::cerr << "Error: sparse_tensor_read: wrong format in the tensor file" << std::endl;
				return sparse_tensor<ScalarType, IndexType, SPARSE_COO>();
			}

			ScalarType val;
			if constexpr (std::is_same_v<ScalarType, ulong>) {
				rat_t raw_val(line.substr(start));
				val = raw_val % F.mod;
			}
			else if constexpr (std::is_same_v<ScalarType, rat_t>) {
				val = rat_t(line.substr(start));
			}

			tensor.push_back(index, val);
		}
		
		if (sort_ind && !tensor.check_sorted())
			tensor.sort_indices(pool);

		return tensor;
	}

	template<typename T, typename IndexType, typename S>
	void sparse_tensor_write(S& st, const sparse_tensor<T, IndexType, SPARSE_COO>& tensor) {
		const auto& dims = tensor.dims();
		const size_t rank = dims.size();
		char num_buf[32];

		for (size_t i = 0; i < rank; ++i) {
			auto [ptr, ec] = std::to_chars(num_buf, num_buf + sizeof(num_buf), dims[i]);
			st.write(num_buf, ptr - num_buf);
			st.put(' ');
		}
		auto [ptr, ec] = std::to_chars(num_buf, num_buf + sizeof(num_buf), tensor.nnz());
		st.write(num_buf, ptr - num_buf);
		st.put('\n');

		std::vector<char> index_buf;
		index_buf.reserve(rank * 20 + 64);

		for (size_t i = 0; i < tensor.nnz(); ++i) {
			index_buf.clear();
			const auto& index = tensor.index(i);
			for (size_t j = 0; j < rank; ++j) {
				auto conv = std::to_chars(num_buf, num_buf + sizeof(num_buf), index[j] + 1);
				index_buf.insert(index_buf.end(), num_buf, conv.ptr);
				index_buf.push_back(' ');
			}
			st.write(index_buf.data(), index_buf.size());
			st << tensor.val(i) << '\n';
		}
	}

} // namespace SparseRREF

#endif