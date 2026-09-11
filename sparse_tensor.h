/*
	Copyright (C) 2024-2026 Zhenjie Li (Li, Zhenjie)

	This file is part of SparseRREF. The SparseRREF is free software:
	you can redistribute it and/or modify it under the terms of the MIT
	License.
*/


#ifndef SPARSE_TENSOR_H
#define SPARSE_TENSOR_H

#include "sparse_type.h"

namespace SparseRREF {
	// we assume that A, B are sorted, then C is also sorted
	template <typename index_type, typename T>
	sparse_tensor<T, index_type, SPARSE_COO> tensor_product(
		const sparse_tensor<T, index_type, SPARSE_COO>& A,
		const sparse_tensor<T, index_type, SPARSE_COO>& B, const field_t& F) {

		std::vector<size_t> dimsB = B.dims();
		std::vector<size_t> dimsC = A.dims();
		dimsC.insert(dimsC.end(), dimsB.begin(), dimsB.end());

		sparse_tensor<T, index_type, SPARSE_COO> C(dimsC);

		if (A.nnz() == 0 || B.nnz() == 0) {
			return C;
		}

		C.reserve(A.nnz() * B.nnz());
		std::vector<index_type> indexC;

		auto permA = A.gen_perm();
		auto permB = B.gen_perm();
		for (auto i : permA) {
			indexC = A.index_vector(i);
			for (auto j : permB) {
				indexC.insert(indexC.end(), B.index(j), B.index(j) + B.rank());
				C.push_back(indexC, scalar_mul(A.val(i), B.val(j), F));
				indexC.resize(A.rank());
			}
		}

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
					A.val(ptr - 1) = entry;
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
				A.val(ptr - 1) = A.val(ptr1 - 1);
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
			A.val(ptr1 + i) = A.val(ptr + i);
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
			return tensor_product(A, B, F);
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
		index_perm_B.insert(index_perm_B.end(), i2.begin(), i2.end());

		auto permA = A.gen_perm(index_perm_A);
		auto permB = B.gen_perm(index_perm_B);

		std::vector<size_t> rowptrA;
		std::vector<size_t> rowptrB;

		auto equal_except = [](const index_p a, const index_p b, const std::vector<size_t>& perm, const size_t len) {
			for (size_t i = 0; i < len; i++) {
				if (a[perm[i]] != b[perm[i]])
					return false;
			}
			return true;
			};

		auto i1i2_size = i1.size();
		auto left_size_A = A.rank() - i1i2_size;
		auto left_size_B = B.rank() - i1i2_size;

		rowptrA.push_back(0);
		for (size_t k = 1; k < A.nnz(); k++) {
			if (!equal_except(A.index(permA[rowptrA.back()]), A.index(permA[k]), index_perm_A, left_size_A))
				rowptrA.push_back(k);
		}
		rowptrA.push_back(A.nnz());

		rowptrB.push_back(0);
		for (size_t k = 1; k < B.nnz(); k++) {
			if (!equal_except(B.index(permB[rowptrB.back()]), B.index(permB[k]), index_perm_B, left_size_B))
				rowptrB.push_back(k);
		}
		rowptrB.push_back(B.nnz());

		sparse_tensor<T, index_type, SPARSE_COO> C(dimsC);
		// parallel version
		size_t nthread;
		if (pool == nullptr)
			nthread = 1;
		else
			nthread = pool->get_thread_count();

		std::vector<index_type> index_A_cache(i1i2_size * A.nnz());
		std::vector<index_type> index_B_cache(i1i2_size * B.nnz());

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

		auto method = [&](sparse_tensor<T, index_type>& C, size_t ss, size_t ee) {
			index_v indexC(dimsC.size());

			for (size_t k = ss; k < ee; k++) {
				// from rowptrA[k] to rowptrA[k + 1] are the same
				auto startA = rowptrA[k];
				auto endA = rowptrA[k + 1];

				for (size_t l = 0; l < left_size_A; l++)
					indexC[l] = A.index(permA[startA])[index_perm_A[l]];

				for (size_t l = 0; l < rowptrB.size() - 1; l++) {
					auto startB = rowptrB[l];
					auto endB = rowptrB[l + 1];

					// double pointer to calculate the inner product
					// since both are ordered, we can use binary search
					size_t ptrA = startA, ptrB = startB;
					T entry = 0;
					auto pA = index_A_cache.data() + ptrA * i1i2_size;
					auto pB = index_B_cache.data() + ptrB * i1i2_size;
					auto eA = index_A_cache.data() + endA * i1i2_size;
					auto eB = index_B_cache.data() + endB * i1i2_size;

					if (i1i2_size == 1) {
						while (ptrA < endA && ptrB < endB) {
							if (*pA < *pB) {
								pA = SparseRREF::lower_bound(pA, eA, pB, 1);
								ptrA = pA - index_A_cache.data();
							}
							else if (*pA > *pB) {
								pB = SparseRREF::lower_bound(pB, eB, pA, 1);
								ptrB = pB - index_B_cache.data();
							}
							else {
								entry = scalar_add(entry, scalar_mul(A.val(permA[ptrA]), B.val(permB[ptrB]), F), F);
								ptrA++; pA++;
								ptrB++; pB++;
							}
						}
					}
					else if (i1i2_size > 1) {
						while (ptrA < endA && ptrB < endB) {
							auto t1 = lexico_compare(pA, pB, i1i2_size);
							if (t1 < 0) {
								pA = SparseRREF::lower_bound(pA, eA, pB, i1i2_size);
								ptrA = ((pA - index_A_cache.data()) / i1i2_size);
							}
							else if (t1 > 0) {
								pB = SparseRREF::lower_bound(pB, eB, pA, i1i2_size);
								ptrB = ((pB - index_B_cache.data()) / i1i2_size);
							}
							else {
								entry = scalar_add(entry, scalar_mul(A.val(permA[ptrA]), B.val(permB[ptrB]), F), F);
								ptrA++; pA += i1i2_size;
								ptrB++; pB += i1i2_size;
							}
						}
					}

					if (entry != 0) {
						for (size_t l = 0; l < left_size_B; l++)
							indexC[left_size_A + l] = B.index(permB[startB])[index_perm_B[l]];

						C.push_back(indexC, entry);
					}
				}
			}
			};

		// parallel version

		if (pool != nullptr) {
			size_t nblocks = nthread;

			if ((rowptrA.size() - 1) < 2 * nthread) {
				method(C, 0, rowptrA.size() - 1);
				return C;
			}

			if (rowptrA.size() - 1 < 64 * nthread)
				nblocks = nthread;
			else
				nblocks = 8 * nthread;

			std::vector<sparse_tensor<T, index_type, SPARSE_COO>> Cs(nblocks, C);

			size_t base = (rowptrA.size() - 1) / nblocks;
			size_t rem = (rowptrA.size() - 1) % nblocks;

			std::vector<std::pair<size_t, size_t>> ranges(nblocks);
			size_t start = 0;
			for (int i = 0; i < nblocks; ++i) {
				size_t end = start + base + (i < rem ? 1 : 0);
				ranges[i] = { start, end };
				start = end;
			}
			pool->detach_sequence(0, nblocks, [&](size_t i) { method(Cs[i], ranges[i].first, ranges[i].second); });
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
				auto tmpnnz = Cs[i].nnz();
				T* valptr = C.data.valptr + start_pos[i];
				index_p colptr = C.data.colptr + start_pos[i] * C.rank();
				s_copy(colptr, Cs[i].data.colptr, tmpnnz * C.rank());
				for (size_t j = 0; j < tmpnnz; j++)
					valptr[j] = std::move(Cs[i].data.valptr[j]);
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
		for (size_t k = 0; k < A.nnz(); k++) {
			if (A.index(k)[i] == A.index(k)[j]) {
				equal_ind_list.push_back(k);
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
		std::sort(std::execution::par, perm.begin(), perm.end(), [&](size_t a, size_t b) {
			return lexico_compare(A.index(equal_ind_list[a]), A.index(equal_ind_list[b]), index_perm) < 0;
			});

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

		if (pool != nullptr) {
			const size_t nrows = rowptr.size() - 1;
			const size_t nblocks = std::min(pool->get_thread_count(), nrows);
			const size_t base = nrows / nblocks;
			const size_t rem = nrows % nblocks;

			// the blocks (not the threads) fix the order of the merged result, since the
			// scheduler decides which thread runs which block
			std::vector<std::pair<size_t, size_t>> ranges(nblocks);
			std::vector<sparse_tensor<T, index_type, SPARSE_COO>> Cs(nblocks, C);

			size_t start = 0;
			for (size_t blk = 0; blk < nblocks; blk++) {
				ranges[blk] = { start, start + base + (blk < rem ? 1 : 0) };
				start = ranges[blk].second;
			}

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
				for (size_t m = 0; m < tmpnnz; m++)
					valptr[m] = std::move(Cs[blk].data.valptr[m]);
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

	// tensors {A,B,...}
	// index_sets {{i1,j1,...}, {i2,j2,...}, ...}
	// |{i1,j1,...}| = A.rank(), |{i2,j2,...}| = B.rank(), ...
	// index with the same number will be contracted
	// and the other indices will be sorted

	// e.g. D = einstein_sum({ A,B,C }, { {0,1,4}, {2,1}, {2,3} })
	// D_{i0,i3,i4} = sum_{i1,i2} A_{i0,i1,i4} B_{i2,i1} C_{i2,i3}

	// TODO: it works, but the performance is not good
	// TODO: parallel version may be wrong
	template <typename index_type, typename T>
	sparse_tensor<T, index_type, SPARSE_COO> einstein_sum(
		const std::vector<sparse_tensor<T, index_type, SPARSE_COO>*> tensors,
		const std::vector<std::vector<size_t>> index_sets,
		const field_t& F, thread_pool* pool = nullptr) {

		auto nt = tensors.size();
		if (nt != index_sets.size()) {
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

		// compute the summed index
		std::map<size_t, std::vector<std::pair<size_t, size_t>>> index_map;
		for (size_t i = 0; i < nt; i++) {
			for (size_t j = 0; j < tensors[i]->rank(); j++) {
				index_map[index_sets[i][j]].push_back(std::make_pair(i, j));
			}
		}

		std::vector<std::vector<std::pair<size_t, size_t>>> contract_index;
		std::vector<std::pair<size_t, size_t>> free_index;
		for (auto& it : index_map) {
			if (it.second.size() > 1)
				contract_index.push_back(it.second);
			else
				free_index.push_back((it.second)[0]);
		}
		std::stable_sort(free_index.begin(), free_index.end(),
			[&index_sets](const std::pair<size_t, size_t>& a, const std::pair<size_t, size_t>& b) {
				return index_sets[a.first][a.second] < index_sets[b.first][b.second];
			});

		std::vector<std::vector<size_t>> each_free_index(nt);
		std::vector<std::vector<size_t>> each_perm(nt);
		for (auto [p, q] : free_index) {
			each_free_index[p].push_back(q);
			each_perm[p].push_back(q);
		}
		for (auto& a : contract_index) {
			for (auto [p, q] : a) {
				each_perm[p].push_back(q);
			}
		}

		// restore the perm of the tensor
		std::vector<std::pair<sparse_tensor<T, index_type, SPARSE_COO>*, std::vector<size_t>>> tensor_perm_map;
		std::vector<std::vector<size_t>> pindx;
		std::vector<size_t> tindx(nt);
		for (size_t i = 0; i < nt; i++) {
			auto checkexist = [&](const std::pair<sparse_tensor<T, index_type, SPARSE_COO>*, std::vector<size_t>>& a) {
				if (a.first != tensors[i])
					return false;
				if (a.second.size() != each_perm[i].size())
					return false;
				for (size_t j = 0; j < a.second.size(); j++) {
					if (a.second[j] != each_perm[i][j])
						return false;
				}
				return true;
				};
			bool is_exist = false;
			for (size_t j = 0; j < tensor_perm_map.size(); j++) {
				if (checkexist(tensor_perm_map[j])) {
					is_exist = true;
					tindx[i] = j;
				}
			}
			if (!is_exist) {
				tensor_perm_map.push_back(std::make_pair(tensors[i], each_perm[i]));
				pindx.push_back(tensors[i]->gen_perm(each_perm[i]));
				tindx[i] = pindx.size() - 1;
			}
		}


		std::vector<std::vector<size_t>> each_rowptr(nt);
		for (size_t i = 0; i < nt; i++) {
			each_rowptr[i].push_back(0);
			for (size_t j = 1; j < tensors[i]->nnz(); j++) {
				bool is_same = true;
				for (auto aa : each_free_index[i]) {
					if (tensors[i]->index(pindx[tindx[i]][j])[aa] != tensors[i]->index(pindx[tindx[i]][j - 1])[aa]) {
						is_same = false;
						break;
					}
				}
				if (!is_same)
					each_rowptr[i].push_back(j);
			}
			each_rowptr[i].push_back(tensors[i]->nnz());
		}

		// first compute the dims of the result
		std::vector<size_t> dimsC;
		for (auto& aa : free_index)
			dimsC.push_back(tensors[aa.first]->dim(aa.second));

		sparse_tensor<T, index_type, SPARSE_COO> C(dimsC);

		int nthread = 1;
		if (pool != nullptr) {
			nthread = pool->get_thread_count();
		}

		std::vector<sparse_tensor<T, index_type, SPARSE_COO>> Cs(nthread, C);

		auto method = [&](size_t ss, size_t ee) {
			int id = 0;
			if (pool != nullptr)
				id = thread_id();

			std::vector<size_t> ptrs(nt, 0);
			ptrs[0] = ss;
			std::vector<size_t> internel_ptrs(nt);

			std::vector<index_type> index(free_index.size());

			// the outer loop 
			// 0 <= ptrs[i] < each_rowptr[i].size() - 1
			while (true) {
				// the internel loop 
				// each_rowptr[i][ptrs[i]] <= internel_ptrs[i] <  to each_rowptr[i][ptrs[i] + 1]

				std::vector<size_t> start_ptrs(nt);
				std::vector<size_t> end_ptrs(nt);
				for (size_t i = 0; i < nt; i++) {
					start_ptrs[i] = each_rowptr[i][ptrs[i]];
					end_ptrs[i] = each_rowptr[i][ptrs[i] + 1];
				}

				T entry = 0;

				multi_for(start_ptrs, end_ptrs, [&](const std::vector<size_t>& internel_ptrs) {
					bool is_zero = false;
					for (auto& a : contract_index) {
						auto num = tensors[a[0].first]->index(pindx[tindx[a[0].first]][internel_ptrs[a[0].first]])[a[0].second];
						for (size_t j = 1; j < a.size(); j++) {
							if (num != tensors[a[j].first]->index(pindx[tindx[a[j].first]][internel_ptrs[a[j].first]])[a[j].second]) {
								is_zero = true;
								break;
							}
						}
					}

					if (!is_zero) {
						T tmp = 1;
						for (auto j = 0; j < nt; j++)
							tmp = scalar_mul(tmp, tensors[j]->val(pindx[tindx[j]][internel_ptrs[j]]), F);
						entry = scalar_add(entry, tmp, F);
					}
					});

				if (entry != 0) {
					// compute the index
					for (size_t j = 0; j < free_index.size(); j++) {
						index[j] = tensors[free_index[j].first]->index(ptrs[free_index[j].first])[free_index[j].second];
					}

					Cs[id].push_back(index, entry);
				}

				bool is_end = false;
				for (int i = nt - 1; i > -2; i--) {
					if (i == -1) {
						is_end = true;
						break;
					}
					ptrs[i]++;
					if (i == 0) {
						if (ptrs[i] < ee)
							break;
						ptrs[i] = ss;
					}
					else {
						if (ptrs[i] < each_rowptr[i].size() - 1)
							break;
						ptrs[i] = 0;
					}
				}

				if (is_end)
					break;
			}
			};

		if (pool == nullptr) {
			method(0, each_rowptr[0].size() - 1);
			return Cs[0];
		}

		pool->detach_blocks(0, each_rowptr[0].size() - 1, method, nthread);
		pool->wait();

		// merge the results
		size_t allnnz = 0;
		size_t nownnz = 0;
		for (size_t i = 0; i < nthread; i++) {
			allnnz += Cs[i].nnz();
		}

		C.reserve(allnnz);
		C.resize(allnnz);
		for (size_t i = 0; i < nthread; i++) {
			// it is ordered, so we can directly push them back
			auto tmpnnz = Cs[i].nnz();
			T* valptr = C.data.valptr + nownnz;
			index_type* colptr = C.data.colptr + nownnz * C.rank();
			s_copy(valptr, Cs[i].data.valptr, tmpnnz);
			s_copy(colptr, Cs[i].data.colptr, tmpnnz * C.rank());
			nownnz += tmpnnz;
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
					index.push_back(string_to_ull(line.substr(start, end - start)) - 1);
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
				auto [ptr, ec] = std::to_chars(num_buf, num_buf + sizeof(num_buf), index[j] + 1);
				index_buf.insert(index_buf.end(), num_buf, ptr);
				index_buf.push_back(' ');
			}
			st.write(index_buf.data(), index_buf.size());
			st << tensor.val(i) << '\n';
		}
	}

} // namespace SparseRREF

#endif