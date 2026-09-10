/*
	Copyright (C) 2024-2026 Zhenjie Li (Li, Zhenjie)

	This file is part of SparseRREF. The SparseRREF is free software:
	you can redistribute it and/or modify it under the terms of the MIT
	License.
*/


#ifndef SPARSE_MAT_H
#define SPARSE_MAT_H

#include "sparse_vec.h"

namespace SparseRREF {

	// first look for rows with only one nonzero value and eliminate them
	// we assume that mat is canonical, i.e. each index is sorted
	// and the result is also canonical
	template <typename T, typename index_t>
	size_t eliminate_row_with_one_nnz(sparse_mat_subview<T, index_t> mat, std::vector<index_t>& donelist,
		rref_option_t opt) {
		auto localcounter = 0;
		std::unordered_map<size_t, index_t> pivlist;
		bit_array collist(mat.ncol());
		for (size_t a = 0; a < mat.nrow(); a++) {
			size_t i = mat(a);
			if (donelist[i] != index_sval<index_t>())
				continue;
			if (mat[i].nnz() == 1) {
				if (!collist[mat[i](0)]) {
					localcounter++;
					pivlist[i] = mat[i](0);
					collist.insert(mat[i](0));
				}
			}
		}

		if (localcounter == 0)
			return localcounter;

		opt->pool.detach_loop(0, mat.nrow(), [&](size_t i) {
			bool is_changed = false;
			auto row = mat(i);
			for (auto [col, val] : mat[i]) {
				if (collist[col]) {
					if (pivlist.contains(row) && pivlist[row] == col)
						val = 1;
					else {
						val = 0;
						is_changed = true;
					}
				}
			}
			if (is_changed) {
				mat[i].canonicalize();
			}
			});

		for (auto [a, b] : pivlist)
			donelist[a] = b;

		opt->pool.wait();

		return localcounter;
	}

	template <typename index_t>
	size_t eliminate_row_with_one_nnz(sparse_mat_subview<bool, index_t> mat, std::vector<index_t>& donelist,
		rref_option_t opt) {
		auto localcounter = 0;
		std::unordered_map<size_t, index_t> pivlist;
		bit_array collist(mat.ncol());
		for (size_t a = 0; a < mat.nrow(); a++) {
			size_t i = mat(a);
			if (donelist[i] != index_sval<index_t>())
				continue;
			if (mat[i].nnz() == 1) {
				if (!collist[mat[i](0)]) {
					localcounter++;
					pivlist[i] = mat[i](0);
					collist.insert(mat[i](0));
				}
			}
		}

		if (localcounter == 0)
			return localcounter;

		constexpr index_t sv = index_sval<index_t>();

		opt->pool.detach_loop(0, mat.nrow(), [&](size_t i) {
			auto row = mat(i);
			bool is_changed = false;
			for (size_t j = 0; j < mat[i].nnz(); j++) {
				if (collist[mat[i](j)]) {
					if (!(pivlist.contains(row) && pivlist[row] == mat[i](j))) {
						mat[i](j) = sv; // mark as deleted
						is_changed = true;
					}
				}
			}
			if (is_changed) {
				size_t new_nnz = 0;
				for (size_t j = 0; j < mat[i].nnz(); j++) {
					if (mat[i](j) != sv) {
						if (new_nnz != j)
							mat[i](new_nnz) = mat[i](j);
						new_nnz++;
					}
				}
				mat[i].resize(new_nnz);
			}
			});

		for (auto [a, b] : pivlist)
			donelist[a] = b;

		opt->pool.wait();

		return localcounter;
	}

	template <typename T, typename index_t>
	size_t eliminate_row_with_one_nnz(sparse_mat<T, index_t>& mat, std::vector<index_t>& donelist,
		rref_option_t opt) {
		return eliminate_row_with_one_nnz(sparse_mat_subview<T, index_t>(mat), donelist, opt);
	}

	template <typename T, typename index_t>
	size_t eliminate_row_with_one_nnz_rec(sparse_mat_subview<T, index_t> mat, std::vector<index_t>& donelist,
		rref_option_t opt, int max_depth = 1024) {
		int depth = 0;
		size_t localcounter = 0;
		size_t count = 0;
		bool verbose = opt->verbose;

		size_t ndir = mat.ncol();

		size_t oldnnz = mat.nnz();
		int bitlen_nnz = num_digits(oldnnz) + 1;
		int bitlen_ndir = num_digits(ndir);

		// if the number of newly eliminated rows is less than 
		// 0.1% of the total number of eliminated rows, we stop
		do {
			localcounter = eliminate_row_with_one_nnz(mat, donelist, opt);
			count += localcounter;
			if (verbose) {
				oldnnz = mat.nnz();
				progress(opt, "Col")
					.add("%*zu/%zu", bitlen_ndir, count, ndir)
					.add("  rank: %*zu", bitlen_ndir, count)
					.add("  nnz: %*zu", bitlen_nnz, oldnnz)
					.add("  density: %8.6g%%", density_percent(oldnnz, mat.nrow(), mat.ncol()))
					.add("    ");
			}
			depth++;
			if (opt->abort)
				return count;

		} while (localcounter > 0 && depth < max_depth && localcounter * 1000 > count);
		return count;
	}

	template <typename T, typename index_t>
	size_t eliminate_row_with_one_nnz_rec(sparse_mat<T, index_t>& mat, std::vector<index_t>& donelist,
		rref_option_t opt, int max_depth = 1024) {
		return eliminate_row_with_one_nnz_rec(sparse_mat_subview<T, index_t>(mat), donelist, opt, max_depth);
	}

	// first choose the pivot with minimal col_weight
	// if the col_weight is negative, we do not choose it
	template <typename T, typename index_t>
	std::vector<pivot_t<index_t>> pivots_search(
		const sparse_mat<T, index_t>& mat, const std::vector<sparse_mat<bool, index_t>>& tranmat_vec,
		const std::vector<size_t>& leftrows, const std::vector<index_t>& leftcols,
		const std::function<int64_t(int64_t)>& col_weight = [](int64_t i) { return i; }) {

		std::deque<pivot_t<index_t>> pivots;
		// holds the used rows in the left look and the used columns in the right look
		bit_array dict(std::max((size_t)mat.nrow, (size_t)mat.ncol) + 1);

		// col_weight is a pure function of the column index and is queried for nearly
		// every scanned element below, so evaluate it once per column
		std::vector<int64_t> col_w(mat.ncol);
		for (index_t i = 0; i < mat.ncol; i++)
			col_w[i] = col_weight(i);

		std::vector<size_t> tranmat_nnz(mat.ncol, 0);
		for (auto& tranmat : tranmat_vec) {
			for (index_t col = 0; col < tranmat.nrow; col++) {
				tranmat_nnz[col] += tranmat[col].nnz();
			}
		}

		// leftlook first
		for (auto col : leftcols) {
			if (tranmat_nnz[col] == 0)
				continue;
			// negative weight means that we do not want to select this column
			if (col_w[col] < 0)
				continue;

			index_t row = 0;
			size_t mnnz = SIZE_MAX;
			bool flag = true;

			for (auto& tranmat : tranmat_vec) {
				for (auto r : tranmat[col].index_span()) {
					flag = !dict.test(r);
					if (!flag)
						break;
					size_t newnnz = mat[r].nnz();
					if (newnnz < mnnz) {
						row = r;
						mnnz = newnnz;
					}
					// make the result stable
					else if (newnnz == mnnz) {
						if (r < row)
							row = r;
					}
				}
				if (!flag)
					break;
			}
			if (!flag)
				continue;
			if (mnnz != SIZE_MAX) {
				pivots.emplace_front(row, col);
				dict.insert(row);
			}
		}

		// rightlook then
		dict.clear();

		for (auto [r, c] : pivots)
			dict.insert(c);

		for (auto row : leftrows) {
			index_t col = 0;
			size_t mnnz = SIZE_MAX;
			int64_t wcol = 0;
			bool flag = true;

			for (auto c : mat[row].index_span()) {
				int64_t wc = col_w[c];
				// negative weight means that we do not want to select this column
				if (wc < 0)
					continue;
				flag = !dict.test(c);
				if (!flag)
					break;
				if (tranmat_nnz[c] < mnnz) {
					mnnz = tranmat_nnz[c];
					col = c;
					wcol = wc;
				}
				// make the result stable
				else if (tranmat_nnz[c] == mnnz) {
					if (wc < wcol) {
						col = c;
						wcol = wc;
					}
					else if (wc == wcol && c < col)
						col = c;
				}
			}
			if (!flag)
				continue;
			if (mnnz != SIZE_MAX) {
				pivots.emplace_back(row, col);
				dict.insert(col);
			}
		}

		std::vector<pivot_t<index_t>> result(pivots.begin(), pivots.end());
		return result;
	}

	// first choose the pivot with minimal col_weight
	// if the col_weight is negative, we do not choose it
	// only right search version, do not need the full tranpose
	template <typename T, typename index_t>
	std::vector<pivot_t<index_t>> pivots_search_right(const sparse_mat<T, index_t>& mat,
		const std::vector<size_t>& leftrows, const std::vector<index_t>& leftcols,
		const std::function<int64_t(int64_t)>& col_weight = [](int64_t i) { return i; }) {

		std::vector<pivot_t<index_t>> pivots;
		bit_array c_dict((size_t)mat.ncol + 1);

		// col_weight is a pure function of the column index and is queried for nearly
		// every scanned element below, so evaluate it once per column
		std::vector<int64_t> col_w(mat.ncol);
		for (index_t i = 0; i < mat.ncol; i++)
			col_w[i] = col_weight(i);

		for (auto row : leftrows) {
			index_t col = 0;
			size_t mnnz = SIZE_MAX;
			int64_t wcol = 0;
			bool flag = true;

			for (auto c : mat[row].index_span()) {
				int64_t wc = col_w[c];
				// negative weight means that we do not want to select this column
				if (wc < 0)
					continue;
				flag = !c_dict.test(c);
				if (!flag)
					break;
				if (mnnz > 0) {
					mnnz = 0;
					col = c;
					wcol = wc;
				}
				// make the result stable
				else if (mnnz == 0) {
					if (wc < wcol) {
						col = c;
						wcol = wc;
					}
					else if (wc == wcol && c < col)
						col = c;
				}
			}
			if (!flag)
				continue;
			if (mnnz != SIZE_MAX) {
				pivots.emplace_back(row, col);
				c_dict.insert(col);
			}
		}

		return pivots;
	}

	template <typename T, typename index_t>
	std::vector<pivot_t<index_t>> pivots_search_left(const sparse_mat<T, index_t>& mat,
		const std::vector<sparse_mat<bool, index_t>>& tranmat_vec,
		const std::vector<size_t>& leftrows, const std::vector<index_t>& leftcols,
		const std::function<int64_t(int64_t)>& col_weight = [](int64_t i) { return i; }) {

		std::vector<pivot_t<index_t>> pivots;
		bit_array dict(mat.nrow);

		std::vector<size_t> tranmat_nnz(mat.ncol, 0);
		for (auto& tranmat : tranmat_vec) {
			for (index_t col = 0; col < tranmat.nrow; col++) {
				tranmat_nnz[col] += tranmat[col].nnz();
			}
		}

		// leftlook first
		for (auto col : leftcols) {
			if (tranmat_nnz[col] == 0)
				continue;
			// negative weight means that we do not want to select this column
			if (col_weight(col) < 0)
				continue;

			index_t row;
			size_t mnnz = SIZE_MAX;
			bool flag = true;

			for (auto& tranmat : tranmat_vec) {
				for (auto r : tranmat[col].index_span()) {
					flag = !dict.test(r);
					if (!flag)
						break;
					size_t newnnz = mat[r].nnz();
					if (newnnz < mnnz) {
						row = r;
						mnnz = newnnz;
					}
					// make the result stable
					else if (newnnz == mnnz) {
						if (r < row)
							row = r;
					}
				}
				if (!flag)
					break;
			}
			if (!flag)
				continue;

			if (mnnz != SIZE_MAX) {
				pivots.emplace_back(row, col);
				dict.insert(row);
			}
		}

		// reverse the ordering of pivots
		for (size_t i = 0; i < pivots.size() / 2; i++) {
			std::swap(pivots[i], pivots[pivots.size() - 1 - i]);
		}

		return pivots;
	}

	// upper solver : ordering = -1
	// lower solver : ordering = 1
	// tranmat in this version is precomputed
	template <typename T, typename index_t>
	void triangular_solver(sparse_mat<T, index_t>& mat,
		const sparse_mat<bool, index_t>& tranmat,
		const std::vector<pivot_t<index_t>>& pivots,
		const field_t& F, rref_option_t opt, int ordering) {
		bool verbose = opt->verbose;
		// a non-positive print_step would be an undefined modulo below; take it as
		// "print every row", which is what the other progress sites do
		size_t printstep = opt->print_step > 0 ? (size_t)opt->print_step : 1;
		auto& pool = opt->pool;

		bit_array rowlist(mat.nrow);
		for (auto [r, c] : pivots)
			rowlist.insert(r);

		size_t count = 0;
		bool printed = false;
		size_t nthreads = pool.get_thread_count();
		std::vector<index_t> thecol;
		// the reference time is kept across the whole loop, so that the reported
		// speed is the average over the rows handled since the last report
		auto start = SparseRREF::clocknow();
		for (size_t i = 0; i < pivots.size(); i++) {
			size_t index = i;
			if (ordering < 0)
				index = pivots.size() - 1 - i;
			auto [row, col] = pivots[index];
			thecol.clear();
			for (size_t j = 0; j < tranmat[col].nnz(); j++) {
				auto r = tranmat[col](j);
				if (rowlist[r] && r != row)
					thecol.push_back(r);
			}

			if constexpr (std::is_same_v<T, bool>) {
				pool.detach_loop<index_t>(0, thecol.size(), [&](index_t j) {
					auto r = thecol[j];
					sparse_vec_add(mat[r], mat[row], F);
					},
					((thecol.size() < 20 * nthreads) ? 0 : thecol.size() / 10));
			}
			else {
				pool.detach_loop<index_t>(0, thecol.size(), [&](index_t j) {
					auto r = thecol[j];
					sparse_vec_sub_mul(mat[r], mat[row], *mat.find(r, col), F);
					},
					((thecol.size() < 20 * nthreads) ? 0 : thecol.size() / 10));
			}
			pool.wait();

			count++;
			if (verbose && (i % printstep == 0 || i == pivots.size() - 1) && thecol.size() > 1) {
				auto end = SparseRREF::clocknow();
				auto now_nnz = mat.nnz();
				progress(opt, "Row")
					.add("%zu/%zu", i + 1, pivots.size())
					.add("  row to eliminate: %zu", thecol.size() - 1)
					.add("  nnz: %zu", now_nnz)
					.add("  density: %g%%", density_percent(now_nnz, mat.nrow, mat.ncol))
					.add("  speed: %g row/s", rate_per_second((double)count, SparseRREF::usedtime(start, end)));
				start = end;
				count = 0;
				printed = true;
			}
		}
		if (printed)
			progress_end(opt);
	}
	
	// in this version, we compute the transpose inside
	template <typename T, typename index_t>
	void triangular_solver(sparse_mat<T, index_t>& mat,
		const std::vector<pivot_t<index_t>>& pivots,
		const field_t& F, rref_option_t opt, int ordering) {

		auto& pool = opt->pool;
		auto nthreads = pool.get_thread_count();

		if (opt->abort)
			return;

		size_t mtx_size;
		if (nthreads > 16)
			mtx_size = 65536;
		else
			mtx_size = (size_t)1 << nthreads;
		std::vector<std::mutex> mtxes(mtx_size);

		// we only need to compute the transpose of the submatrix involving pivots
		sparse_mat<bool, index_t> tranmat(mat.ncol, mat.nrow);
		pool.detach_loop(0, pivots.size(), [&](size_t i) {
			auto [r, c] = pivots[i];
			for (auto [ind, val] : mat[r]) {
				std::lock_guard<std::mutex> lock(mtxes[ind % mtx_size]);
				tranmat[ind].push_back(r);
			}
			});
		pool.wait();
		
		triangular_solver(mat, tranmat, pivots, F, opt, ordering);
	}

	template <typename T, typename index_t>
	void triangular_solver(sparse_mat<T, index_t>& mat,
		std::vector<std::vector<pivot_t<index_t>>>& pivots,
		const field_t& F, rref_option_t opt, int ordering) {
		std::vector<pivot_t<index_t>> n_pivots;
		for (auto p : pivots)
			n_pivots.insert(n_pivots.end(), p.begin(), p.end());
		triangular_solver(mat, n_pivots, F, opt, ordering);
	}

	// dot_product
	template <typename T, typename index_t>
	sparse_vec<T, index_t> sparse_mat_dot_sparse_vec(
		const sparse_mat<T, index_t>& mat,
		const sparse_vec<T, index_t>& vec, const field_t& F) {

		sparse_vec<T, index_t> result;

		if (vec.nnz() == 0 || mat.nnz() == 0)
			return result;

		T tmp;
		for (size_t i = 0; i < mat.nrow; i++) {
			tmp = sparse_vec_dot(mat[i], vec, F);
			if (tmp != 0)
				result.push_back(i, tmp);
		}
		return result;
	}

	template <typename T, typename index_t>
	sparse_vec<T, index_t> sparse_mat_dot_dense_vec(
		const sparse_mat<T, index_t>& mat, const T* vec, const field_t& F) {

		sparse_vec<T, index_t> result;

		if (mat.nnz() == 0)
			return result;

		T tmp = 0;
		for (size_t i = 0; i < mat.nrow; i++) {
			tmp = sparse_vec_dot_dense_vec(mat[i], vec, F);
			if (tmp != 0)
				result.push_back(i, tmp);
		}
		return result;
	}

	// A = B * C
	template <typename T, typename index_t>
	sparse_mat<T, index_t> sparse_mat_mul(
		const sparse_mat<T, index_t>& B, const sparse_mat<T, index_t>& C,
		const field_t& F, thread_pool* pool = nullptr) {

		sparse_mat<T, index_t> A(B.nrow, C.ncol);
		auto nthreads = 1;
		if (pool)
			nthreads = pool->get_thread_count();

		std::vector<T> cache_densed_mat(A.ncol * nthreads);
		std::vector<SparseRREF::bit_array> nonzero_c(nthreads, A.ncol);

		auto pp = F.get_prime();

		auto method = [&](size_t id, size_t i) {
			auto& therow = B[i];
			if (therow.nnz() == 0)
				return;
			if (therow.nnz() == 1) {
				A[i] = C[therow(0)];
				sparse_vec_rescale(A[i], therow[0], F);
				return;
			}
			auto cache_dense_vec = cache_densed_mat.data() + id * A.ncol;
			auto& nonzero_c_vec = nonzero_c[id];
			nonzero_c_vec.clear();

			T scalar = therow[0];
			ulong e_pr;
			if constexpr (std::is_same_v<T, ulong>) {
				e_pr = n_mulmod_precomp_shoup(scalar, pp);
			}
			for (auto [ind, val] : C[therow(0)]) {
				nonzero_c_vec.insert(ind);
				if constexpr (std::is_same_v<T, ulong>) {
					cache_dense_vec[ind] = n_mulmod_shoup(scalar, val, e_pr, pp);
				}
				else if constexpr (std::is_same_v<T, rat_t>) {
					cache_dense_vec[ind] = scalar * val;
				}
			}

			for (size_t j = 1; j < therow.nnz(); j++) {
				scalar = therow[j];
				if constexpr (std::is_same_v<T, ulong>) {
					e_pr = n_mulmod_precomp_shoup(scalar, pp);
				}
				for (auto [ind, val] : C[therow(j)]) {
					if (!nonzero_c_vec.test(ind)) {
						nonzero_c_vec.insert(ind);
						cache_dense_vec[ind] = 0;
					}
					if constexpr (std::is_same_v<T, ulong>) {
						cache_dense_vec[ind] = _nmod_add(cache_dense_vec[ind],
							n_mulmod_shoup(scalar, val, e_pr, pp), F.mod);
					}
					else if constexpr (std::is_same_v<T, rat_t>) {
						cache_dense_vec[ind] += scalar * val;
					}
					if (cache_dense_vec[ind] == 0)
						nonzero_c_vec.erase(ind);
				}
			}

			auto pos = nonzero_c_vec.nonzero();
			A[i].reserve(pos.size());
			A[i].resize(pos.size());
			for (size_t j = 0; j < pos.size(); j++) {
				A[i](j) = pos[j];
				A[i][j] = cache_dense_vec[pos[j]];
			}
			};

		if (pool) {
			pool->detach_loop(0, B.nrow, [&](size_t i) {
				method(SparseRREF::thread_id(), i);
				}, ((B.nrow < 20 * nthreads) ? 0 : B.nrow / 10));
			pool->wait();
		}
		else {
			for (size_t i = 0; i < B.nrow; i++)
				method(0, i);
		}

		return A;
	}

	// A set of column indices supporting O(1) insert/erase/test/enumeration. On top
	// of the plain bit_array we keep an exact counter (so nnz() is O(1)) and a bitmap
	// with one bit per *word* of the bit_array that may be non-empty, so that the
	// final enumeration only visits the words that were touched instead of the whole
	// ncol/64 range. The word summary is only ever set, never cleared (clearing it
	// would need a branch on every erase); a stale bit is harmless because the
	// enumeration skips empty words and clears the summary on the way out.
	struct alignas(64) nonzero_cols {
		bit_array bits;
		bit_array word_bits;
		size_t nz = 0;

		explicit nonzero_cols(size_t ncol) : bits(ncol), word_bits((ncol >> 6) + 1) {}

		void insert(const size_t val) {
			auto idx = val >> 6;
			bits.data[idx] |= mask_table[val & 63];
			word_bits.data[idx >> 6] |= mask_table[idx & 63];
			nz++;
		}

		void erase(const size_t val) {
			bits.erase(val);
			nz--;
		}

		void xor_insert(const size_t val) {
			if (bits.test(val)) {
				bits.erase(val);
				nz--;
			}
			else {
				insert(val);
			}
		}

		bool test(const size_t val) const {
			return bits.test(val);
		}

		size_t nnz() const {
			return nz;
		}

		// write the set indices in increasing order into ptr, then clear the structure
		template <typename T>
		void nonzero_and_clear(T* ptr) {
			size_t pos = 0;
			auto& wdata = word_bits.data;
			for (size_t i = 0; i < wdata.size(); i++) {
				uint64_t wb = wdata[i];
				if (wb == 0)
					continue;
				wdata[i] = 0;
				while (wb) {
					auto idx = (i << 6) + ctz(wb);
					wb &= wb - 1;
					uint64_t word = bits.data[idx];
					bits.data[idx] = 0;
					auto base = idx << 6;
					while (word) {
						ptr[pos] = (T)(base + ctz(word));
						pos++;
						word &= word - 1;
					}
				}
			}
			nz = 0;
		}
	};

	// a helper class to store the buffer for schur_complete
	// it is used to avoid frequent memory allocation and deallocation
	template <typename T, typename index_t>
	struct schur_helper_buffer {
		size_t nthreads = 0;
		size_t ncol = 0;
		std::vector<T> cache_densed_mat;
		std::vector<nonzero_cols> nonzero_cs;
		std::vector<std::vector<index_t>> add_lists;
		std::vector<std::vector<index_t>> remove_lists;

		schur_helper_buffer(size_t nthreads_, size_t ncol_) : nthreads(nthreads_), ncol(ncol_) {
			cache_densed_mat = std::vector<T>(nthreads * ncol);
			nonzero_cs = std::vector<nonzero_cols>(nthreads, nonzero_cols(ncol));
			add_lists = std::vector<std::vector<index_t>>(nthreads);
			remove_lists = std::vector<std::vector<index_t>>(nthreads);
		}

		~schur_helper_buffer() = default;
		schur_helper_buffer(const schur_helper_buffer&) = delete;
		schur_helper_buffer& operator=(const schur_helper_buffer&) = delete;
		schur_helper_buffer(schur_helper_buffer&&) = default;
		schur_helper_buffer& operator=(schur_helper_buffer&&) = default;
	};

	// define a helper struct to store temporary variables for single-threaded schur_complete
	// pointers are used to avoid copying large data structures
	// but do not manage the memory of the pointers directly, the caller should manage the memory of the pointers
	template <typename T, typename index_t>
	struct schur_helper {
		T* cache_densed_vec = nullptr;
		nonzero_cols* nonzero_c = nullptr;
		std::vector<index_t>* add_list = nullptr;
		std::vector<index_t>* remove_list = nullptr;

		schur_helper(schur_helper_buffer<T, index_t>& buffer, size_t thread_id) {
			cache_densed_vec = buffer.cache_densed_mat.data() + thread_id * buffer.ncol;
			nonzero_c = &buffer.nonzero_cs[thread_id];
			add_list = &buffer.add_lists[thread_id];
			remove_list = &buffer.remove_lists[thread_id];
		}
	};

	// make sure nonzero_c and tmpvec are cleared before calling this function
	// after this function, nonzero_c and tmpvec are also cleared
	template <typename T, typename index_t>
	void schur_complete(sparse_mat<T, index_t>& mat, size_t k,
		const std::vector<pivot_t<index_t>>& pivots, 
		const field_t& F, schur_helper<T, index_t>& helper) {

		if (mat[k].nnz() == 0)
			return;

		T* tmpvec = helper.cache_densed_vec;
		auto& nonzero_c = *helper.nonzero_c;
		auto& add_list = *helper.add_list;
		auto& remove_list = *helper.remove_list;

		for (auto [ind, val] : mat[k]) {
			nonzero_c.insert(ind);
			tmpvec[ind] = val;
		}

		ulong e_pr;
		auto pp = F.get_prime();
		for (auto [r, c] : pivots) {
			if (!nonzero_c.test(c))
				continue;

			T entry = tmpvec[c];
			add_list.clear();
			remove_list.clear();

			if constexpr (std::is_same_v<T, ulong>) {
				e_pr = n_mulmod_precomp_shoup(entry, pp);
			}
			for (auto [ind, val] : mat[r]) {
				bool old_c = tmpvec[ind] == 0;
				if constexpr (std::is_same_v<T, ulong>) {
					tmpvec[ind] = _nmod_sub(tmpvec[ind],
						n_mulmod_shoup(entry, val, e_pr, pp), F.mod);
				}
				else if constexpr (std::is_same_v<T, rat_t>) {
					tmpvec[ind] -= entry * val;
				}
				else {
					tmpvec[ind] = scalar_sub(tmpvec[ind], scalar_mul(entry, val, F), F);
				}
				if (tmpvec[ind] == 0) {
					remove_list.push_back(ind);
				}
				else {
					if (old_c)
						add_list.push_back(ind);
				}
			}
			for (auto ind : add_list) nonzero_c.insert(ind);
			for (auto ind : remove_list) nonzero_c.erase(ind);
		}

		auto nnz = nonzero_c.nnz();
		if (mat[k].alloc() < nnz) {
			mat[k].reserve(nnz, false);
		}
		mat[k].resize(nnz);
		nonzero_c.nonzero_and_clear(mat[k].indices);
		for (size_t i = 0; i < nnz; i++) {
			mat[k][i] = tmpvec[mat[k](i)];
			tmpvec[mat[k](i)] = 0; // clear tmpvec for next use
		}
	}

	// tempvec is of useless, only nonzero_c is used
	template <typename index_t>
	void schur_complete(sparse_mat<bool, index_t>& mat, size_t k,
		const std::vector<pivot_t<index_t>>& pivots,
		const field_t& F, schur_helper<bool, index_t>& helper) {

		auto& nonzero_c = *helper.nonzero_c;

		auto nk = mat[k].nnz();
		if (nk == 0)
			return;

		for (size_t i = 0; i < nk; i++)
			nonzero_c.insert(mat[k](i));

		for (auto [r, c] : pivots) {
			if (!nonzero_c.test(c))
				continue;

			auto nr = mat[r].nnz();
			for (size_t i = 0; i < nr; i++)
				nonzero_c.xor_insert(mat[r](i));
		}

		auto nnz = nonzero_c.nnz();
		if (mat[k].alloc() < nnz) {
			mat[k].reserve(nnz, false);
		}
		mat[k].resize(nnz);
		nonzero_c.nonzero_and_clear(mat[k].indices);
	}

	template <typename T, typename index_t, size_t buffer_bit>
	void schur_complete_buffer(sparse_mat<T, index_t>& mat, size_t k,
		const std::vector<pivot_t<index_t>>& pivots,
		const field_t& F, schur_helper<T, index_t>& helper) {

		if constexpr (std::is_same_v<T, bool>) {
			schur_complete(mat, k, pivots, F, helper);
			return;
		}

		if (mat[k].nnz() == 0)
			return;

		T* tmpvec = helper.cache_densed_vec;
		auto& nonzero_c = *helper.nonzero_c;
		auto& add_list = *helper.add_list;
		auto& remove_list = *helper.remove_list;

		constexpr size_t buffer_size = (size_t)1 << buffer_bit;

		std::array<size_t, buffer_size> buffer = { 0 };
		for (auto [ind, val] : mat[k]) {
			nonzero_c.insert(ind);
			tmpvec[ind] = val;
			buffer[ind % buffer_size]++;
		}

		ulong e_pr;
		auto pp = F.get_prime();
		for (auto [r, c] : pivots) {
			if (buffer[c % buffer_size] == 0)
				continue;
			if (!nonzero_c.test(c))
				continue;

			T entry = tmpvec[c];
			add_list.clear();
			remove_list.clear();

			if constexpr (std::is_same_v<T, ulong>) {
				e_pr = n_mulmod_precomp_shoup(tmpvec[c], pp);
			}
			for (auto [ind, val] : mat[r]) {
				bool old_c = tmpvec[ind] == 0;
				if constexpr (std::is_same_v<T, ulong>) {
					tmpvec[ind] = _nmod_sub(tmpvec[ind],
						n_mulmod_shoup(entry, val, e_pr, pp), F.mod);
				}
				else if constexpr (std::is_same_v<T, rat_t>) {
					tmpvec[ind] -= entry * val;
				}
				else {
					tmpvec[ind] = scalar_sub(tmpvec[ind], scalar_mul(entry, val, F), F);
				}
				if (tmpvec[ind] == 0) {
					remove_list.push_back(ind);
				}
				else {
					if (old_c)
						add_list.push_back(ind);
				}
			}
			for (auto ind : add_list) {
				nonzero_c.insert(ind);
				buffer[ind % buffer_size]++;
			}
			for (auto ind : remove_list) {
				nonzero_c.erase(ind);
				buffer[ind % buffer_size]--;
			}
		}

		auto nnz = nonzero_c.nnz();
		if (mat[k].alloc() < nnz) {
			mat[k].reserve(nnz, false);
		}
		mat[k].resize(nnz);
		nonzero_c.nonzero_and_clear(mat[k].indices);
		for (size_t i = 0; i < nnz; i++) {
			mat[k][i] = tmpvec[mat[k](i)];
			tmpvec[mat[k](i)] = 0; // clear tmpvec for next use
		}
	}

	// Progress bookkeeping of the triangular back substitution, shared by all
	// recursion levels: every report covers the work completed since the previous
	// one, so a report issued at a level entry shows the real average speed of
	// the levels that came before instead of 0.
	struct triangular_progress_t {
		std::chrono::system_clock::time_point last_time; // time of the previous report
		double last_count;                               // completed pivots at the previous report
		bool reported;                                   // whether any line was already printed
	};

	template <typename T, typename index_t>
	void triangular_solver_2_rec(sparse_mat<T, index_t>& mat,
		const sparse_mat<bool, index_t>& tranmat,
		const std::vector<pivot_t<index_t>>& pivots,
		const field_t& F, rref_option_t opt, 
		schur_helper_buffer<T, index_t>& g_helper, size_t n_split, size_t rank, size_t& process,
		triangular_progress_t& tprogress) {

		if (opt->abort)
			return;

		const size_t printstep = opt->print_step > 0 ? (size_t)opt->print_step : 1;
		bool verbose = opt->verbose;
		auto& pool = opt->pool;
		opt->verbose = false;
		if (pivots.size() < n_split) {
			// this level performs the elimination itself, so it is the only one that
			// can report it; keep the caller's verbosity instead of staying silent
			opt->verbose = verbose;
			triangular_solver(mat, tranmat, pivots, F, opt, -1);
			process += pivots.size();
			return;
		}

		std::vector<pivot_t<index_t>> sub_pivots(pivots.end() - n_split, pivots.end());
		std::vector<pivot_t<index_t>> left_pivots(pivots.begin(), pivots.end() - n_split);

		std::unordered_set<index_t> pre_leftrows;
		for (auto [r, c] : sub_pivots) {
			for (auto i = 0; i < tranmat[c].nnz(); i++)
				pre_leftrows.insert(tranmat[c](i));
		}
		for (auto [r, c] : sub_pivots)
			pre_leftrows.erase(r);
		std::vector<index_t> leftrows(pre_leftrows.begin(), pre_leftrows.end());

		// for printing
		size_t now_nnz = mat.nnz();
		int bitlen_nnz = num_digits(now_nnz) + 1;
		int bitlen_nrow = num_digits(rank);

		double density = (double)now_nnz / (mat.nrow * mat.ncol);
		auto schur_complete_func = &schur_complete_buffer<T, index_t, 10>;
		if (100 * density > 0.01)
			schur_complete_func = &schur_complete<T, index_t>;

		std::atomic<size_t> cc = 0;
		pool.detach_blocks<size_t>(0, leftrows.size(), [&](const size_t s, const size_t e) {
			auto id = SparseRREF::thread_id();
			schur_helper<T, index_t> helper(g_helper, id);
			for (size_t i = s; i < e; i++) {
				schur_complete_func(mat, leftrows[i], sub_pivots, F, helper);
				cc++;
			}
			}, ((n_split < 20 * pool.get_thread_count()) ? 0 : leftrows.size() / 10));

		if (verbose) {
			while (cc.load(std::memory_order_relaxed) < leftrows.size()) {
				double status = 1.0 * sub_pivots.size() * cc.load(std::memory_order_relaxed) / leftrows.size();
				double done = (double)process + status;
				if (done - tprogress.last_count <= (double)printstep) {
					if (pool.wait_for(progress_poll_interval))
						break; // pool idle, no further progress can be reported
					continue;
				}
				auto now = SparseRREF::clocknow();
				now_nnz = mat.nnz();
				progress(opt, "Row")
					.add("%*zu/%zu", bitlen_nrow, (size_t)done, rank)
					.add("  nnz: %*zu", bitlen_nnz, now_nnz)
					.add("  density: %8.6g%%", density_percent(now_nnz, mat.nrow, mat.ncol))
					.add("  speed: %6.6g row/s", rate_per_second(done - tprogress.last_count, SparseRREF::usedtime(tprogress.last_time, now)))
					.add("    ");
				tprogress.last_time = now;
				tprogress.last_count = done;
				tprogress.reported = true;
			}
		}

		pool.wait();

		triangular_solver(mat, tranmat, sub_pivots, F, opt, -1);
		opt->verbose = verbose;
		process += sub_pivots.size();

		triangular_solver_2_rec(mat, tranmat, left_pivots, F, opt, g_helper, n_split, rank, process, tprogress);
	}

	template <typename T, typename index_t>
	void triangular_solver_2(sparse_mat<T, index_t>& mat, std::vector<pivot_t<index_t>>& pivots,
		const field_t& F, rref_option_t opt) {

		auto& pool = opt->pool;
		// prepare the tmp array
		auto nthreads = pool.get_thread_count();
		schur_helper_buffer<T, index_t> g_helper(nthreads, mat.ncol);

		if (opt->abort)
			return;

		size_t mtx_size;
		if (nthreads > 16)
			mtx_size = 65536;
		else
			mtx_size = (size_t)1 << nthreads;
		std::vector<std::mutex> mtxes(mtx_size);

		// we only need to compute the transpose of the submatrix involving pivots
		sparse_mat<bool, index_t> tranmat(mat.ncol, mat.nrow);
		pool.detach_loop(0, pivots.size(), [&](size_t i) {
			auto [r, c] = pivots[i];
			for (auto [ind, val] : mat[r]) {
				std::lock_guard<std::mutex> lock(mtxes[ind % mtx_size]);
				tranmat[ind].push_back(r);
			}
			});
		pool.wait();

		size_t process = 0;
		// TODO: better split strategy
		size_t n_split = std::max(pivots.size() / 128ULL, 1ULL << 10);
		size_t rank = pivots.size();
		triangular_progress_t tprogress{ SparseRREF::clocknow(), 0.0, false };
		triangular_solver_2_rec(mat, tranmat, pivots, F, opt, g_helper, n_split, rank, process, tprogress);

		// A level can only report while it is still polling, so the tail of the phase
		// (everything done after the last report) would stay invisible without this.
		if (opt->verbose && pivots.size() >= n_split && rank > tprogress.last_count) {
			auto now = SparseRREF::clocknow();
			size_t now_nnz = mat.nnz();
			int bitlen_nnz = num_digits(now_nnz) + 1;
			progress(opt, "Row")
				.add("%*zu/%zu", num_digits(rank), rank, rank)
				.add("  nnz: %*zu", bitlen_nnz, now_nnz)
				.add("  density: %8.6g%%", density_percent(now_nnz, mat.nrow, mat.ncol))
				.add("  speed: %6.6g row/s",
					rate_per_second(rank - tprogress.last_count, SparseRREF::usedtime(tprogress.last_time, now)))
				.add("    ");
			tprogress.reported = true;
		}

		if (opt->verbose && tprogress.reported)
			progress_end(opt);
	}

	template <typename T, typename index_t>
	void triangular_solver_2(sparse_mat<T, index_t>& mat,
		const std::vector<std::vector<pivot_t<index_t>>>& pivots,
		const field_t& F, rref_option_t opt) {

		std::vector<pivot_t<index_t>> n_pivots;
		// the first pivot is the row with only one nonzero value, so there is no need to do the elimination
		for (size_t i = 1; i < pivots.size(); i++)
			n_pivots.insert(n_pivots.end(), pivots[i].begin(), pivots[i].end());

		triangular_solver_2(mat, n_pivots, F, opt);
	}

	template <typename T, typename index_t>
	std::vector<pivot_t<index_t>> sparse_mat_direct_rref_part(sparse_mat<T, index_t>& mat,
		const std::vector<std::vector<pivot_t<index_t>>>& sub_pivots,
		const field_t& F, rref_option_t opt, schur_helper_buffer<T, index_t>& g_helper) {

		if (sub_pivots.size() <= 1)
			return std::vector<pivot_t<index_t>>();

		auto& pool = opt->pool;
		auto nthreads = pool.get_thread_count();
		
		std::vector<pivot_t<index_t>> combined_pivots;
		size_t nnz = 0;
		size_t total_rank = 0;
		for (auto& p : sub_pivots) {
			combined_pivots.insert(combined_pivots.end(), p.begin(), p.end());
			for (auto [r, c] : p) {
				nnz += mat[r].nnz();
				total_rank++;
			}
		}

		double density = (double)(nnz) / (total_rank * mat.ncol);
		auto schur_complete_func = &schur_complete_buffer<T, index_t, 10>;
		if (100 * density > 0.01)
			schur_complete_func = &schur_complete<T, index_t>;

		std::vector<size_t> leftrows;
		leftrows.reserve(total_rank);
		for (auto rp = sub_pivots.rbegin(); rp != sub_pivots.rend(); rp++) {
			for (auto p = rp->rbegin(); p != rp->rend(); p++) {
				leftrows.push_back(p->r);
			}
		}

		for (const auto& pivs : sub_pivots) {
			if (pivs.size() == 0)
				continue;

			// rescale the pivots
			pool.detach_loop(0, pivs.size(), [&](size_t j) {
				auto [r, c] = pivs[j];
				T scalar = scalar_inv(*mat.find(r, c), F);
				sparse_vec_rescale(mat[r], scalar, F);
				mat[r].reserve(mat[r].nnz());
				});

			// remove the used rows
			leftrows.resize(leftrows.size() - pivs.size());
			pool.wait();

			// upper solver
			pool.detach_blocks<size_t>(0, leftrows.size(), [&](const size_t s, const size_t e) {
				auto id = SparseRREF::thread_id();
				schur_helper<T, index_t> helper(g_helper, id);
				for (size_t j = s; j < e; j++) {
					schur_complete_func(mat, leftrows[j], pivs, F, helper);
					if (opt->abort)
						return;
				}
				}, ((leftrows.size() < 20 * nthreads) ? 0 : leftrows.size() / 10));

			pool.wait();
		}

		return combined_pivots;
	}

	// if already know the pivots, we can directly do the rref
	// if the pivots are of submatrix, it is dangerous to set clear_zero_row to true
	template <typename T, typename index_t>
	void sparse_mat_direct_rref(sparse_mat<T, index_t>& mat,
		const std::vector<std::vector<pivot_t<index_t>>>& pivots,
		const field_t& F, rref_option_t opt, const bool clear_zero_row = true) {
		auto& pool = opt->pool;
		constexpr index_t sv = index_sval<index_t>();

		// first set rows not in pivots to zero
		std::vector<index_t> rowset(mat.nrow, sv);
		size_t total_rank = 0;
		for (auto p : pivots) {
			total_rank += p.size();
			for (auto [r, c] : p)
				rowset[r] = c;
		}
		if (clear_zero_row) {
			// clear the rows not in pivots
			for (size_t i = 0; i < mat.nrow; i++)
				if (rowset[i] == sv)
					mat[i].clear();
		}
		else {
			std::fill(rowset.begin(), rowset.end(), 0);
		}

		for (auto [r, c] : pivots[0]) {
			mat[r].zero();
			mat[r].push_back(c, 1);
		}

		for (size_t i = 0; i < mat.nrow; i++) {
			mat[i].compress();
		}

		if (opt->eliminate_one_nnz) {
			std::vector<index_t> donelist(mat.nrow, sv);
			eliminate_row_with_one_nnz(mat, donelist, opt);
		}

		// then do the elimination parallelly
		auto nthreads = pool.get_thread_count();
		schur_helper_buffer<T, index_t> g_helper(nthreads, mat.ncol);

		size_t rank = pivots[0].size();

		double density = (double)mat.nnz() / (total_rank * mat.ncol);
		auto schur_complete_func = &schur_complete_buffer<T, index_t, 10>;
		if (100 * density > 0.01)
			schur_complete_func = &schur_complete<T, index_t>;

		std::vector<size_t> leftrows;
		leftrows.reserve(total_rank);
		for (auto rp = pivots.rbegin(); rp != pivots.rend(); rp++) {
			for (auto p = rp->rbegin(); p != rp->rend(); p++) {
				leftrows.push_back(p->r);
			}
		}
		leftrows.resize(leftrows.size() - pivots[0].size());
		
		int bitlen_nrow = num_digits(total_rank);
		int bitlen_nnz = num_digits(mat.nnz()) + 1;

		std::vector<pivot_t<index_t>> used_pivots;
		for (size_t i = 1; i < pivots.size(); i++) {
			if (pivots[i].size() == 0)
				continue;

			used_pivots.clear();
			if (pivots[i].size() < 512) {
				size_t trank = 0;
				std::vector<std::vector<pivot_t<index_t>>> sub_pivots;
				for (size_t ni = i; ni < pivots.size(); ni++) {
					trank += pivots[ni].size();
					sub_pivots.push_back(pivots[ni]);
					if (trank > 4096) {
						i = ni;
						break;
					}	
				}
				used_pivots = sparse_mat_direct_rref_part(mat, sub_pivots, F, opt, g_helper);
			}

			if (used_pivots.size() == 0) {
				used_pivots = pivots[i];
				// rescale the pivots
				pool.detach_loop(0, used_pivots.size(), [&](size_t j) {
					auto [r, c] = used_pivots[j];
					T scalar = scalar_inv(*mat.find(r, c), F);
					sparse_vec_rescale(mat[r], scalar, F);
					mat[r].reserve(mat[r].nnz());
					});
				pool.wait();
			}

			leftrows.resize(leftrows.size() - used_pivots.size());
			if (leftrows.size() == 0) // done
				break;

			// upper solver
			// TODO: check mode
			std::atomic<size_t> cc = 0;
			size_t old_cc = cc.load(std::memory_order_relaxed);
			pool.detach_blocks<size_t>(0, leftrows.size(), [&](const size_t s, const size_t e) {
				auto id = SparseRREF::thread_id();
				schur_helper<T, index_t> helper(g_helper, id);
				for (size_t j = s; j < e; j++) {
					schur_complete_func(mat, leftrows[j], used_pivots, F, helper);
					cc++;
					if (opt->abort)
						return;
				}
				}, ((leftrows.size() < 20 * nthreads) ? 0 : leftrows.size() / 10));

			if (opt->verbose) {
				auto cn = clocknow();
				const size_t printstep = opt->print_step > 0 ? (size_t)opt->print_step : 1;
				while (cc.load(std::memory_order_relaxed) < leftrows.size()) {
					if (opt->abort) {
						pool.purge();
						return;
					}
					const size_t cur_cc = cc.load(std::memory_order_relaxed);
					if (cur_cc - old_cc > printstep) {
						auto now_nnz = mat.nnz();
						progress(opt, "Row")
							.add("%*zu/%zu", bitlen_nrow, (size_t)std::floor(rank + (cur_cc * 1.0 / leftrows.size()) * used_pivots.size()), total_rank)
							.add("  nnz: %*zu", bitlen_nnz, now_nnz)
							.add("  alloc: %zu", mat.alloc())
							.add("  speed: %g row/s", rate_per_second((((cur_cc - old_cc) * 1.0 / leftrows.size()) * used_pivots.size()), usedtime(cn, clocknow())))
							.add("          ");
						old_cc = cur_cc;
						cn = clocknow();
					}
					// block until the pool makes progress instead of spinning
					pool.wait_for(progress_poll_interval);
				}
			}
			pool.wait();
			rank += used_pivots.size();
		}
		if (opt->verbose) {
			progress_end(opt);
		}

		if (opt->is_back_sub)
			triangular_solver_2(mat, pivots, F, opt);
	}

	template <typename T, typename index_t>
	std::vector<std::vector<pivot_t<index_t>>>
		sparse_mat_rref_forward(sparse_mat_subview<T, index_t> submat, const field_t& F, rref_option_t opt) {
		// first canonicalize, sort and compress the matrix

		sparse_mat<T, index_t>& mat = submat.get_mat();
		constexpr index_t sv = index_sval<index_t>();

		auto& pool = opt->pool;
		auto nthreads = pool.get_thread_count();

		pool.detach_loop(0, mat.nrow, [&](auto i) { mat[i].compress(); });

		// store the pivots that have been used
		// sv is not used
		std::vector<index_t> rowpivs(mat.nrow, sv);
		std::vector<std::vector<pivot_t<index_t>>> pivots;
		std::vector<pivot_t<index_t>> n_pivots;

		pool.wait();

		size_t now_nnz = mat.nnz();
		double density = (double)now_nnz / (mat.nrow * mat.ncol);
		auto schur_complete_func = &schur_complete_buffer<T, index_t, 10>;
		if (100 * density > 0.01)
			schur_complete_func = &schur_complete<T, index_t>;

		if (opt->abort)
			return pivots;

		// eliminate rows with only one non-zero entry
		if (opt->eliminate_one_nnz) {
			size_t count = eliminate_row_with_one_nnz_rec(submat, rowpivs, opt);
			now_nnz = mat.nnz();

			for (size_t i = 0; i < mat.nrow; i++) {
				if (opt->col_weight(rowpivs[i]) < 0)
					rowpivs[i] = sv; // mark as unused
				if (rowpivs[i] != sv)
					n_pivots.emplace_back(i, rowpivs[i]);
			}
		}
		pivots.push_back(n_pivots);

		if (opt->abort)
			return pivots;

		// use a vector to label the left columns
		std::vector<index_t> leftcols = perm_init((index_t)(mat.ncol));
		for (size_t i = 0; i < mat.nrow; i++) {
			if (rowpivs[i] != sv)
				leftcols[rowpivs[i]] = sv; // mark as used
		}
		std::erase_if(leftcols, [](index_t i) { return i == sv; });

		schur_helper_buffer<T, index_t> g_helper(nthreads, mat.ncol);

		std::vector<size_t> leftrows;
		leftrows.reserve(mat.nrow);
		submat.traverse([&rowpivs, &mat, &leftrows](size_t i) {
			if (rowpivs[i] == index_sval<index_t>() && mat[i].nnz() > 0)
				leftrows.push_back(i);
			}
		);

		int method = opt->method;

		std::vector<sparse_mat<bool, index_t>> tranmat_vec(nthreads);
		for (auto& tmat : tranmat_vec) {
			tmat = sparse_mat<bool, index_t>(mat.ncol, mat.nrow);
		}

		if (method != 1) {
			pool.detach_loop(0, leftrows.size(), [&](size_t i) {
				auto id = thread_id();
				auto r = leftrows[i];
				for (size_t j = 0; j < mat[r].nnz(); j++) {
					tranmat_vec[id][mat[r](j)].push_back(r);
				}
				});
			pool.wait();
		}

		if (method == 0 || method == 2) {
			// sort pivots by nnz, it will be faster
			std::ranges::stable_sort(leftcols, std::less{},
				[&tranmat_vec](size_t r) {
					size_t nnz = 0;
					for (auto& tmat : tranmat_vec) {
						nnz += tmat[r].nnz();
					}
					return nnz;
				});
		}

		// for printing
		double oldpr = 0;
		int bitlen_nnz = num_digits(now_nnz) + 1;
		int bitlen_ncol = num_digits(mat.ncol);

		bit_array tmp_set(mat.ncol);

		auto rank = pivots[0].size();
		while (rank < mat.ncol) {
			auto start = clocknow();

			std::vector<pivot_t<index_t>> ps;

			// case 1 is to use the right search method
			switch (method) {
			case 1: ps = pivots_search_right(mat, leftrows, leftcols, opt->col_weight); break;
			default: ps = pivots_search(mat, tranmat_vec, leftrows, leftcols, opt->col_weight); break;
			}

			if (ps.size() == 0)
				break;

			n_pivots.clear();
			for (auto& [r, c] : ps) {
				rowpivs[r] = c;
				n_pivots.emplace_back(r, c);
			}
			pivots.push_back(n_pivots);

			pool.detach_loop(0, n_pivots.size(), [&](size_t i) {
				auto [r, c] = n_pivots[i];
				T scalar = scalar_inv(*mat.find(r, c), F);
				sparse_vec_rescale(mat[r], scalar, F);
				mat[r].reserve(mat[r].nnz());
				});

			size_t n_leftrows = 0;
			for (size_t i = 0; i < leftrows.size(); i++) {
				auto row = leftrows[i];
				if (rowpivs[row] != sv)
					continue;
				if (mat[row].nnz() == 0) {
					mat[row].clear();
					continue;
				}
				leftrows[n_leftrows] = row;
				n_leftrows++;
			}
			leftrows.resize(n_leftrows);
			pool.wait();

			if (opt->abort)
				return pivots;

			auto print_info = [&](size_t kk, size_t x, bool& print_once, double& oldpr) {
				double pr = kk + (1.0 * ps.size() * x) / leftrows.size();
				// require real progress, otherwise a round-entry report would show speed 0
				if (opt->verbose && pr > oldpr && (print_once || pr - oldpr > opt->print_step)) {
					auto end = clocknow();
					now_nnz = mat.nnz();
					auto now_alloc = mat.alloc();
					progress(opt, "Col")
						.add("%*zu/%zu", bitlen_ncol, (size_t)pr, mat.ncol)
						.add("  rank: %*zu", bitlen_ncol, kk + ps.size())
						.add("  nnz: %*zu", bitlen_nnz, now_nnz)
						.add("  alloc: %*zu", bitlen_nnz, now_alloc)
						.add("  density: %8.6g%%", density_percent(now_nnz, mat.nrow, mat.ncol))
						.add("  speed: %8.6g col/s", rate_per_second(pr - oldpr, usedtime(start, end)))
						.add("    ");
					oldpr = pr;
					start = end;
					print_once = false;
				}
				};

			if (method == 1) {
				std::atomic<size_t> done_count = 0;
				pool.detach_blocks<size_t>(0, leftrows.size(), [&](const size_t s, const size_t e) {
					auto id = thread_id();
					schur_helper<T, index_t> helper(g_helper, id);
					for (size_t i = s; i < e; i++) {
						schur_complete_func(mat, leftrows[i], n_pivots, F, helper);
						done_count++;
						if (opt->abort)
							break;
					}
					}, (leftrows.size() < 20 * nthreads ? 0 : leftrows.size() / 10));

				// remove used cols
				size_t localcount = 0;
				tmp_set.clear();
				for (auto [r, c] : ps)
					tmp_set.insert(c);
				for (auto c : leftcols) {
					if (!tmp_set.test(c)) {
						leftcols[localcount] = c;
						localcount++;
					}
				}
				leftcols.resize(localcount);

				bool print_once = true; // print at least once

				localcount = 0;
				while (done_count.load(std::memory_order_relaxed) < leftrows.size()) {
					if (opt->abort) {
						pool.purge();
						return pivots;
					}

					print_info(rank, done_count.load(std::memory_order_relaxed), print_once, oldpr);
					// block until the pool makes progress instead of polling
					pool.wait_for(progress_poll_interval);
				}
				pool.wait();
			}
			else {
				std::vector<std::atomic<int>> flags(leftrows.size());
				pool.detach_blocks<size_t>(0, leftrows.size(), [&](const size_t s, const size_t e) {
					auto id = thread_id();
					schur_helper<T, index_t> helper(g_helper, id);
					for (size_t i = s; i < e; i++) {
						schur_complete_func(mat, leftrows[i], n_pivots, F, helper);
						flags[i].store(1, std::memory_order_release);
						if (opt->abort)
							break;
					}
					}, (leftrows.size() < 20 * nthreads ? 0 : leftrows.size() / 10));

				// remove used cols
				std::atomic<size_t> localcount = 0;
				tmp_set.clear();
				for (auto [r, c] : ps)
					tmp_set.insert(c);
				for (auto c : leftcols) {
					if (!tmp_set.test(c)) {
						leftcols[localcount] = c;
						localcount++;
						for (auto& tranmat : tranmat_vec)
							tranmat[c].zero();
					}
					else {
						for (auto& tranmat : tranmat_vec)
							tranmat[c].clear();
					}
				}
				leftcols.resize(localcount);

				bool print_once = true; // print at least once

				localcount = 0;
				while (localcount < leftrows.size()) {
					// if the pool is free and too many rows left, use pool
					if (localcount * 2 < leftrows.size() && pool.get_tasks_total() == 0) {
						std::vector<size_t> newleftrows;
						for (size_t i = 0; i < leftrows.size(); i++) {
							if (flags[i].load(std::memory_order_acquire))
								newleftrows.push_back(leftrows[i]);
						}

						pool.detach_loop(0, newleftrows.size(), [&](size_t i) {
							auto row = newleftrows[i];
							auto id = thread_id();
							for (size_t j = 0; j < mat[row].nnz(); j++) {
								auto col = mat[row](j);
								tranmat_vec[id][col].push_back(row, true);
							}
							localcount++;
							}, (newleftrows.size() < 20 * nthreads ? 0 : newleftrows.size() / 10));

						if (opt->verbose) {
							while (localcount.load(std::memory_order_relaxed) < leftrows.size() && !(opt->abort)) {
								print_info(rank, localcount.load(std::memory_order_relaxed), print_once, oldpr);
								// block until the pool makes progress instead of polling
								pool.wait_for(progress_poll_interval);
							}
						}

						pool.wait();
					}

					for (size_t i = 0; i < leftrows.size() && localcount < leftrows.size(); i++) {
						if (flags[i].load(std::memory_order_acquire)) {
							auto row = leftrows[i];
							for (size_t j = 0; j < mat[row].nnz(); j++) {
								tranmat_vec[0][mat[row](j)].push_back(row, true);
							}
							flags[i].store(0, std::memory_order_relaxed);
							localcount++;

							if (localcount * 2 < leftrows.size() && pool.get_tasks_total() == 0)
								break;
						}
					}

					if (opt->abort) {
						pool.purge();
						return pivots;
					}

					print_info(rank, localcount, print_once, oldpr);
				}
				pool.wait();
			}

			// if the number of new pivots is less than 1% of the total columns, 
			// it is very expansive to compute the transpose of the matrix
			// so we only search the right columns
			if (method == 2 && ps.size() < mat.ncol / 100) {
				method = 1;
				for (auto& tranmat : tranmat_vec)
					tranmat.clear();
			}

			rank += n_pivots.size();
		}

		if (opt->verbose)
			progress_message(opt, "\n** Rank: %zu nnz: %zu\n", (size_t)rank, mat.nnz());

		return pivots;
	}

	template <typename T, typename index_t>
	auto sparse_mat_rref_forward(sparse_mat<T, index_t>& mat, const field_t& F, rref_option_t opt) {
		return sparse_mat_rref_forward<T, index_t>(sparse_mat_subview<T, index_t>(mat), F, opt);
	}

	template <typename T, typename index_t>
	std::vector<std::vector<pivot_t<index_t>>>
		sparse_mat_rref_left(sparse_mat<T, index_t>& mat, const size_t fullrank, const field_t& F, rref_option_t opt) {
		// first canonicalize, sort and compress the matrix

		constexpr index_t sv = index_sval<index_t>();

		auto& pool = opt->pool;
		auto nthreads = pool.get_thread_count();
		size_t p_bound = fullrank;
		std::function<int64_t(int64_t)> col_weight = [&](int64_t i) {
			if (i < p_bound)
				return opt->col_weight(i);
			else
				return (int64_t)(-1);
			};

		pool.detach_loop(0, mat.nrow, [&](auto i) { mat[i].compress(); });

		auto printstep = opt->print_step;
		bool verbose = opt->verbose;

		size_t now_nnz = mat.nnz();
		double density = (double)now_nnz / (mat.nrow * mat.ncol);
		auto schur_complete_func = &schur_complete_buffer<T, index_t, 10>;
		if (100 * density > 0.01)
			schur_complete_func = &schur_complete<T, index_t>;

		// store the pivots that have been used
		// sv is not used
		std::vector<index_t> rowpivs(mat.nrow, sv);
		std::vector<std::vector<pivot_t<index_t>>> pivots;
		std::vector<pivot_t<index_t>> n_pivots;

		pool.wait();

		if (opt->abort)
			return pivots;

		if (opt->eliminate_one_nnz) {
			// eliminate rows with only one non-zero entry
			size_t count = eliminate_row_with_one_nnz_rec(mat, rowpivs, opt);
			now_nnz = mat.nnz();

			for (size_t i = 0; i < mat.nrow; i++) {
				if (opt->col_weight(rowpivs[i]) < 0)
					rowpivs[i] = sv; // mark as unused
				if (rowpivs[i] != sv)
					n_pivots.emplace_back(i, rowpivs[i]);
			}
		}
		pivots.push_back(n_pivots);

		if (opt->abort)
			return pivots;

		// use a vector to label the left columns
		std::vector<index_t> leftcols = perm_init((index_t)(mat.ncol));
		for (size_t i = 0; i < mat.nrow; i++) {
			if (rowpivs[i] != sv)
				leftcols[rowpivs[i]] = sv; // mark as used
		}
		std::erase_if(leftcols, [](index_t i) { return i == sv; });

		auto rank = pivots[0].size();
		size_t kk = rank;

		schur_helper_buffer<T, index_t> g_helper(nthreads, mat.ncol);

		std::vector<size_t> leftrows;
		leftrows.reserve(mat.nrow);
		for (size_t i = 0; i < mat.nrow; i++) {
			if (rowpivs[i] != sv || mat[i].nnz() == 0)
				continue;
			leftrows.push_back(i);
		}

		bool only_right_search = true;

		std::vector<sparse_mat<bool, index_t>> tranmat_vec(nthreads);
		for (auto& tmat : tranmat_vec) {
			tmat = sparse_mat<bool, index_t>(mat.ncol, mat.nrow);
		}

		if (opt->method != 1) {
			only_right_search = false;
			pool.detach_loop(0, leftrows.size(), [&](size_t i) {
				auto id = thread_id();
				auto r = leftrows[i];
				for (size_t j = 0; j < mat[r].nnz(); j++) {
					tranmat_vec[id][mat[r](j)].push_back(r);
				}
				});
			pool.wait();

			// sort pivots by nnz, it will be faster
			std::ranges::stable_sort(leftcols, std::less{},
				[&tranmat_vec](size_t r) {
					size_t nnz = 0;
					for (auto& tmat : tranmat_vec) {
						nnz += tmat[r].nnz();
					}
					return nnz;
				});
		}

		// for printing
		double oldpr = 0;
		int bitlen_nnz = num_digits(now_nnz) + 1;
		int bitlen_ncol = num_digits(mat.ncol);

		bit_array tmp_set(mat.ncol);

		while (kk < mat.ncol) {
			auto start = SparseRREF::clocknow();

			std::vector<pivot_t<index_t>> ps;

			if (only_right_search) {
				ps = pivots_search_right(mat, leftrows, leftcols, col_weight);
			}
			else {
				ps = pivots_search(mat, tranmat_vec, leftrows, leftcols, col_weight);
			}
			if (ps.size() == 0) {
				if (rank >= fullrank)
					break;
				else {
					p_bound += fullrank - rank;
					continue;
				}
			}

			n_pivots.clear();
			for (auto& [r, c] : ps) {
				rowpivs[r] = c;
				n_pivots.emplace_back(r, c);
			}
			pivots.push_back(n_pivots);
			rank += n_pivots.size();

			pool.detach_loop(0, n_pivots.size(), [&](size_t i) {
				auto [r, c] = n_pivots[i];
				T scalar = scalar_inv(*mat.find(r, c), F);
				sparse_vec_rescale(mat[r], scalar, F);
				mat[r].reserve(mat[r].nnz());
				});

			size_t n_leftrows = 0;
			for (size_t i = 0; i < leftrows.size(); i++) {
				auto row = leftrows[i];
				if (rowpivs[row] != sv)
					continue;
				if (mat[row].nnz() == 0) {
					mat[row].clear();
					continue;
				}
				leftrows[n_leftrows] = row;
				n_leftrows++;
			}
			leftrows.resize(n_leftrows);
			pool.wait();

			if (opt->abort)
				return pivots;

			auto print_info = [&](size_t kk, size_t x, bool& print_once, double& oldpr) {
				double pr = kk + (1.0 * ps.size() * x) / leftrows.size();
				// require real progress, otherwise a round-entry report would show speed 0
				if (verbose && pr > oldpr && (print_once || pr - oldpr > printstep)) {
					auto end = SparseRREF::clocknow();
					now_nnz = mat.nnz();
					auto now_alloc = mat.alloc();
					progress(opt, "Col")
						.add("%*zu/%zu", bitlen_ncol, (size_t)pr, mat.ncol)
						.add("  rank: %*zu", bitlen_ncol, rank)
						.add("  nnz: %*zu", bitlen_nnz, now_nnz)
						.add("  alloc: %*zu", bitlen_nnz, now_alloc)
						.add("  density: %8.6g%%", density_percent(now_nnz, mat.nrow, mat.ncol))
						.add("  speed: %8.6g col/s", rate_per_second(pr - oldpr, SparseRREF::usedtime(start, end)))
						.add("    ");
					oldpr = pr;
					start = end;
					print_once = false;
				}
				};

			if (only_right_search) {
				std::atomic<size_t> done_count = 0;
				pool.detach_blocks<size_t>(0, leftrows.size(), [&](const size_t s, const size_t e) {
					auto id = SparseRREF::thread_id();
					schur_helper<T, index_t> helper(g_helper, id);
					for (size_t i = s; i < e; i++) {
						schur_complete_func(mat, leftrows[i], n_pivots, F, helper);
						done_count++;
						if (opt->abort)
							break;
					}
					}, (leftrows.size() < 20 * nthreads ? 0 : leftrows.size() / 10));

				// remove used cols
				size_t localcount = 0;
				tmp_set.clear();
				for (auto [r, c] : ps)
					tmp_set.insert(c);
				for (auto c : leftcols) {
					if (!tmp_set.test(c)) {
						leftcols[localcount] = c;
						localcount++;
					}
				}
				leftcols.resize(localcount);

				bool print_once = true; // print at least once

				localcount = 0;
				while (done_count < leftrows.size()) {
					if (opt->abort) {
						pool.purge();
						return pivots;
					}

					print_info(kk, done_count, print_once, oldpr);
					pool.wait_for(progress_poll_interval);
				}
				pool.wait();
			}
			else {
				std::vector<std::atomic<int>> flags(leftrows.size());
				pool.detach_blocks<size_t>(0, leftrows.size(), [&](const size_t s, const size_t e) {
					auto id = SparseRREF::thread_id();
					schur_helper<T, index_t> helper(g_helper, id);
					for (size_t i = s; i < e; i++) {
						schur_complete_func(mat, leftrows[i], n_pivots, F, helper);
						flags[i].store(1, std::memory_order_release);
						if (opt->abort)
							break;
					}
					}, (leftrows.size() < 20 * nthreads ? 0 : leftrows.size() / 10));

				// remove used cols
				std::atomic<size_t> localcount = 0;
				tmp_set.clear();
				for (auto [r, c] : ps)
					tmp_set.insert(c);
				for (auto c : leftcols) {
					if (!tmp_set.test(c)) {
						leftcols[localcount] = c;
						localcount++;
						for (auto& tranmat : tranmat_vec)
							tranmat[c].zero();
					}
					else {
						for (auto& tranmat : tranmat_vec)
							tranmat[c].clear();
					}
				}
				leftcols.resize(localcount);

				bool print_once = true; // print at least once

				localcount = 0;
				while (localcount < leftrows.size()) {
					// if the pool is free and too many rows left, use pool
					if (localcount * 2 < leftrows.size() && pool.get_tasks_total() == 0) {
						std::vector<index_t> newleftrows;
						for (size_t i = 0; i < leftrows.size(); i++) {
							if (flags[i].load(std::memory_order_acquire))
								newleftrows.push_back(leftrows[i]);
						}

						pool.detach_loop(0, newleftrows.size(), [&](size_t i) {
							auto row = newleftrows[i];
							for (size_t j = 0; j < mat[row].nnz(); j++) {
								auto col = mat[row](j);
								auto id = thread_id();
								tranmat_vec[id][col].push_back(row, true);
							}
							localcount++;
							}, (newleftrows.size() < 20 * nthreads ? 0 : newleftrows.size() / 10));
						pool.wait();
					}

					for (size_t i = 0; i < leftrows.size() && localcount < leftrows.size(); i++) {
						if (flags[i].load(std::memory_order_acquire)) {
							auto row = leftrows[i];
							for (size_t j = 0; j < mat[row].nnz(); j++) {
								tranmat_vec[0][mat[row](j)].push_back(row, true);
							}
							flags[i].store(0, std::memory_order_relaxed);
							localcount++;

							if (localcount * 2 < leftrows.size() && pool.get_tasks_total() == 0)
								break;
						}
					}

					if (opt->abort) {
						pool.purge();
						return pivots;
					}

					print_info(kk, localcount, print_once, oldpr);
				}
				pool.wait();
			}

			// if the number of new pivots is less than 1% of the total columns, 
			// it is very expansive to compute the transpose of the matrix
			// so we only search the right columns
			if (opt->method == 2 && !only_right_search && ps.size() * 100 < mat.ncol) {
				only_right_search = true;
				for (auto& tranmat : tranmat_vec)
					tranmat.clear();
			}

			kk += ps.size();
		}

		if (verbose)
			progress_message(opt, "\n** Rank: %zu nnz: %zu\n", (size_t)rank, mat.nnz());

		return pivots;
	}

	template <typename T, typename index_t>
	std::vector<std::vector<pivot_t<index_t>>>
		sparse_mat_rref(sparse_mat<T, index_t>& mat, const field_t& F, rref_option_t opt) {

		auto pivots = sparse_mat_rref_forward(mat, F, opt);

		if (opt->shrink_memory) {
			opt->pool.detach_loop(0, mat.nrow, [&](auto i) {
				mat[i].reserve(mat[i].nnz());
				});
			opt->pool.wait();
		}

		if (opt->abort)
			return pivots;

		if (opt->is_back_sub) {
			if (opt->verbose)
				progress_message(opt, "\n>> Reverse solving: \n");
			// triangular_solver(mat, pivots, F, opt, -1);
			triangular_solver_2(mat, pivots, F, opt);
		}

		return pivots;
	}

	template <typename index_t>
	int_t sparse_mat_denominator_lcm(const sparse_mat<rat_t, index_t>& mat) {
		int_t d = 1;
		for (size_t i = 0; i < mat.nrow; i++) {
			d = Flint::LCM(d, sparse_vec_denominator_lcm(mat[i]));
		}
		return d;
	}

	template <typename index_t>
	int_t sparse_mat_height(const sparse_mat<rat_t, index_t>& mat) {
		if (mat.nrow == 0)
			return 1;

		int_t h = sparse_vec_height(mat[0]);
		for (size_t i = 1; i < mat.nrow; i++) {
			int_t hi = sparse_vec_height(mat[i]);
			if (hi > h)
				h = hi;
		}
		return h;
	}

	// The condition to stop reconstruct: H(d*E)*H(mat)*ncol < product of primes
	// where H is the height of a matrix (the maximal height of each entry), 
	// E is the reconstracted rref matrix
	// d is an integer such that d*E is a integer matrix
	// checkrank is only used for sparse_mat_inverse
	template <typename index_t>
	std::vector<std::vector<pivot_t<index_t>>> sparse_mat_rref_reconstruct(
		sparse_mat<rat_t, index_t>& mat, rref_option_t opt, const bool checkrank = false) {

		constexpr index_t sv = index_sval<index_t>();

		auto& pool = opt->pool;
		auto nthreads = pool.get_thread_count();

		pool.detach_loop(0, mat.nrow, [&](auto i) { mat[i].compress(); });
		pool.wait();

		ulong prime = n_nextprime(1ULL << 60, 0);
		field_t F(FIELD_Fp, prime);

		sparse_mat<ulong, index_t> matul(mat.nrow, mat.ncol);
		pool.detach_loop(0, mat.nrow, [&](auto i) {
			matul[i] = mat[i] % F.mod;
			});
		pool.wait();

		int_t m_height = sparse_mat_height(mat);

		auto cs = clocknow();
		auto pivots = sparse_mat_rref_forward(matul, F, opt);

		if (opt->abort)
			return pivots;

		if (checkrank) {
			size_t rank = 0;
			for (auto& p : pivots)
				rank += p.size();
			if (rank != mat.nrow)
				return pivots;
		}

		if (opt->is_back_sub)
			triangular_solver_2(matul, pivots, F, opt);

		int_t mod = prime;

		bool isok = true;
		sparse_mat<rat_t, index_t> matq(mat.nrow, mat.ncol);

		std::vector<size_t> leftrows;

		for (size_t i = 0; i < mat.nrow; i++) {
			size_t nnz = matul[i].nnz();
			if (nnz == 0)
				continue;
			leftrows.push_back(i);
			matq[i].reserve(nnz);
			matq[i].resize(nnz);
			for (size_t j = 0; j < nnz; j++) {
				matq[i](j) = matul[i](j);
				int_t mod1 = matul[i][j];
				if (isok)
					isok = rational_reconstruct(matq[i][j], mod1, mod);
			}
		}

		sparse_mat<int_t, index_t> matz(mat.nrow, mat.ncol);

		auto check_height_condition = [](const sparse_mat<rat_t, index_t>& matq,
			const int_t& mod, const int_t& m_height) -> bool {
			int_t d = sparse_mat_denominator_lcm(matq);
			int_t h = 1;
			for (size_t i = 0; i < matq.nrow; i++) {
				for (size_t j = 0; j < matq[i].nnz(); j++) {
					int_t hi = (matq[i][j] * d).num().abs(); // since denominator is 1, height() = num().abs()
					if (hi > h)
						h = hi;
				}
			}
			return m_height * matq.ncol * h < mod;
			};

		if (!isok || !check_height_condition(matq, mod, m_height)) {
			isok = false;
			for (size_t i = 0; i < mat.nrow; i++)
				matz[i] = matul[i];
		}

		auto verbose = opt->verbose;

		if (verbose) {
			progress_end(opt);
		}

		if (opt->abort)
			return pivots;

		// set rows not in pivots to zero
		if (!isok) {
			std::vector<index_t> rowset(mat.nrow, sv);
			for (auto p : pivots)
				for (auto [r, c] : p)
					rowset[r] = c;
			for (size_t i = 0; i < mat.nrow; i++)
				if (rowset[i] == sv) {
					mat[i].clear();
				}
		}

		while (!isok) {
			isok = true;
			prime = n_nextprime(prime, 0);
			auto ce = clocknow();
			
			if (verbose) {
				progress_message(opt, ">> Reconstruct failed, now mod ~ 2^%llu, used time: %g s\n",
					(unsigned long long)mod.bits(), usedtime(cs, ce));
			}
			cs = ce;
			int_t mod1 = mod * prime;
			F = field_t(FIELD_Fp, prime);
			pool.detach_loop(0, mat.nrow, [&](auto i) {
				matul[i] = mat[i] % F.mod;
				});
			pool.wait();
			sparse_mat_direct_rref(matul, pivots, F, opt);
			if (opt->is_back_sub) {
				opt->verbose = false;
				triangular_solver_2(matul, pivots, F, opt);
			}
			std::vector<int> flags(nthreads, 1);

			pool.detach_loop<size_t>(0, leftrows.size(), [&](size_t i) {
				size_t row = leftrows[i];
				auto id = SparseRREF::thread_id();
				for (size_t j = 0; j < matul[row].nnz(); j++) {
					matz[row][j] = CRT(matz[row][j], mod, matul[row][j], prime);
					if (flags[id])
						flags[id] = rational_reconstruct(matq[row][j], matz[row][j], mod1);
				}
				});

			pool.wait();
			for (auto f : flags)
				isok = isok && f;

			mod = mod1;

			// if ok, check height condition
			if (isok)
				isok = check_height_condition(matq, mod, m_height);
		}
		opt->verbose = verbose;

		if (opt->verbose) {
			progress_message(opt, "** Reconstruct success! Using mod ~ 2^%llu.                \n",
				(unsigned long long)mod.bits());
		}

		mat = matq;

		return pivots;
	}

	template <typename T, typename index_t>
	sparse_mat<T, index_t> sparse_mat_rref_kernel(const sparse_mat<T, index_t>& M,
		const std::vector<pivot_t<index_t>>& pivots, const field_t& F, rref_option_t opt) {

		auto rank = pivots.size();

		if (rank == M.ncol)
			return sparse_mat<T, index_t>();

		constexpr index_t sv = index_sval<index_t>();
		T m1 = scalar_neg((T)1, F);

		sparse_mat<T, index_t> K(M.ncol, M.ncol - rank);

		auto nonpivs = perm_init((index_t)M.ncol);

		for (auto [r, c] : pivots) {
			K[c] = M[r];
			*K[c].find(c) = (T)0;
			K[c].canonicalize();
			nonpivs[c] = sv;
		}
		std::erase_if(nonpivs, [](index_t i) { return i == sv; });
		std::vector<index_t> nonpivs_ord(M.ncol, sv);
		for (size_t i = 0; i < nonpivs.size(); i++)
			nonpivs_ord[nonpivs[i]] = (index_t)i;

		for (size_t i = 0; i < nonpivs.size(); i++) {
			K[nonpivs[i]].push_back((index_t)i, m1);
		}

		for (auto [r, c] : pivots) {
			for (size_t j = 0; j < K[c].nnz(); j++)
				K[c](j) = nonpivs_ord[K[c](j)];
		}

		return K;
	}

	template <typename T, typename index_t>
	sparse_mat<T, index_t> sparse_mat_rref_kernel(const sparse_mat<T, index_t>& M,
		const std::vector<std::vector<pivot_t<index_t>>>& pivots,
		const field_t& F, rref_option_t opt) {
		std::vector<pivot_t<index_t>> n_pivots;
		for (auto& p : pivots)
			n_pivots.insert(n_pivots.end(), p.begin(), p.end());
		return sparse_mat_rref_kernel(M, n_pivots, F, opt);
	}

	// error code:
	// 0 - sucess
	// 1 - matrix is not square
	// 2 - matrix is not invertible
	// 3 - Field is not supported
	template <typename T, typename index_t>
	int sparse_mat_inverse(sparse_mat<T, index_t>& M1, const sparse_mat<T, index_t>& M,
		const field_t& F, rref_option_t opt) noexcept {
		if (M.nrow != M.ncol) {
			std::cerr << "Error: sparse_mat_inverse: matrix is not square" << std::endl;
			return 1;
		}

		if (M.nnz() == 0) {
			std::cerr << "Error: sparse_mat_inverse: zero matrix" << std::endl;
			return 2;
		}

		auto& pool = opt->pool;

		// define the Augmented matrix
		M1 = M;
		M1.compress();
		for (size_t i = 0; i < M1.nrow; i++) {
			if (M1[i].nnz() == 0) {
				std::cerr << "Error: sparse_mat_inverse: matrix is not invertible" << std::endl;
				return 2;
			}
			M1[i].push_back(i + M.ncol, (T)1);
		}
		M1.ncol *= 2;

		// backup the option
		auto old_col_weight = opt->col_weight;
		std::function<int64_t(int64_t)> new_col_weight = [&](int64_t i) {
			return ((i < M.ncol) ? old_col_weight(i) : -1);
			};
		opt->col_weight = new_col_weight;
		bool is_back_sub = opt->is_back_sub;
		opt->is_back_sub = true;

		std::vector<std::vector<pivot_t<index_t>>> pivots;
		if (F.ring == RING::FIELD_Fp)
			pivots = sparse_mat_rref(M1, F, opt);
		else if (F.ring == RING::FIELD_QQ)
			if constexpr (std::is_same_v<T, rat_t>) {
				pivots = sparse_mat_rref_reconstruct(M1, opt, true);
			}
			else {
				// type is not rat_t when field is QQ
				std::cerr << "Warning: sparse_mat_inverse: type is not rat_t when field is QQ" << std::endl;
				pivots = sparse_mat_rref(M1, F, opt);
			}
		else {
			std::cerr << "Error: sparse_mat_inverse: field not supported" << std::endl;
			// restore the option
			opt->col_weight = old_col_weight;
			opt->is_back_sub = is_back_sub;
			return 3;
		}

		std::vector<pivot_t<index_t>> flatten_pivots;
		size_t rank = 0;
		for (auto& p : pivots) {
			rank += p.size();
			flatten_pivots.insert(flatten_pivots.end(), p.begin(), p.end());
		}

		if (rank != M.nrow) {
			std::cerr << "Error: sparse_mat_inverse: matrix is not invertible" << std::endl;
			// restore the option
			opt->col_weight = old_col_weight;
			opt->is_back_sub = is_back_sub;
			return 2;
		}

		auto perm = perm_init(M.nrow);
		std::sort(perm.begin(), perm.end(),
			[&](size_t a, size_t b) {
				return flatten_pivots[a].c < flatten_pivots[b].c;
			});
		for (size_t i = 0; i < M.nrow; i++) {
			perm[i] = flatten_pivots[perm[i]].r;
		}

		permute(perm, M1.rows);

		for (size_t i = 0; i < M1.nrow; i++) {
			// the first ncol columns is the identity matrix,
			// we need to remove it
			for (size_t j = 0; j < M1[i].nnz() - 1; j++) {
				M1[i][j] = M1[i][j + 1];
				M1[i](j) = M1[i](j + 1) - M.ncol;
			}
			M1[i].resize(M1[i].nnz() - 1);
		}
		M1.ncol = M.ncol;

		// restore the option
		opt->col_weight = old_col_weight;
		opt->is_back_sub = is_back_sub;

		return 0;
	}

	template <typename T, typename index_t>
	sparse_mat<T, index_t> sparse_mat_inverse(const sparse_mat<T, index_t>& M,
		const field_t& F, rref_option_t opt) {
		sparse_mat<T, index_t> res;
		int error = sparse_mat_inverse(res, M, F, opt);
		if (error != 0) 
			return sparse_mat<T, index_t>();
		return res;
	}

	// IO
	template <typename T, typename index_t, typename S>
	sparse_mat<T, index_t> sparse_mat_read(S& st, const field_t& F) {
		if (!st.is_open()) {
			std::cerr << "Error: sparse_mat_read: file not open." << std::endl;
			return sparse_mat<T, index_t>();
		}

		std::string line;
		std::vector<size_t> dims;
		sparse_mat<T, index_t> mat;

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
				// size_t nnz = string_to_ull(line.substr(start));
				if (dims.size() != 2) {
					std::cerr << "Error: sparse_mat_read: wrong format in the matrix file" << std::endl;
					return sparse_mat<T, index_t>();
				}
				mat = sparse_mat<T, index_t>(dims[0], dims[1]);
			}
			break;
		}

		while (std::getline(st, line)) {
			if (line.empty() || line[0] == '%')
				continue;

			bool is_end = false;
			size_t rowcol[2];
			size_t* rowcolptr = rowcol;
			size_t start = 0;
			size_t end = line.find(' ');
			size_t count = 0;

			while (end != std::string::npos && count < 2) {
				if (start != end) {
					auto val = string_to_ull(line.substr(start, end - start));
					if (val == 0) {
						is_end = true;
						break;
					}
					*rowcolptr = val - 1;
					rowcolptr++;
					count++;
				}
				start = end + 1;
				end = line.find(' ', start);
			}

			if (is_end)
				break;

			if (count != 2) {
				std::cerr << "Error: sparse_mat_read: wrong format in the matrix file" << std::endl;
				return sparse_mat<T, index_t>();
			}

			T val{};
			if constexpr (std::is_same_v<T, ulong>) {
				rat_t raw_val(line.substr(start));
				val = raw_val % F.mod;
			}
			else if constexpr (std::is_same_v<T, rat_t>) {
				val = rat_t(line.substr(start));
			}
			else if constexpr (std::is_same_v<T, int_t>) {
				val = int_t(line.substr(start));
			}

			mat[rowcol[0]].push_back(rowcol[1], val);
		}

		return mat;
	}

	template <typename T, typename S, typename index_t>
	void sparse_mat_write(sparse_mat<T, index_t>& mat, S& st, enum SPARSE_FILE_TYPE type) {
		switch (type) {
		case SPARSE_FILE_TYPE_PLAIN:
			st << mat.nrow << ' ' << mat.ncol << ' ' << mat.nnz() << '\n';
			break;
		case SPARSE_FILE_TYPE_MTX:
			if constexpr (std::is_same_v<T, ulong>) {
				st << "%%MatrixMarket matrix coordinate integer general\n";
			}
			st << mat.nrow << ' ' << mat.ncol << ' ' << mat.nnz() << '\n';
			break;
		case SPARSE_FILE_TYPE_SMS: {
			char type_char =
				std::is_same_v<T, rat_t> ? 'Q' :
				(std::is_same_v<T, ulong> || std::is_same_v<T, int_t>) ? 'M' : '\0';
			if (type_char == '\0') {
				return;
			}
			st << mat.nrow << ' ' << mat.ncol << ' ' << type_char << '\n';
			break;
		}
		default:
			return;
		}

		char num_buf[32];
		std::string line_buf;
		line_buf.reserve(mat.nnz() * 32);

		for (size_t i = 0; i < mat.nrow; ++i) {
			for (auto [ind, val] : mat[i]) {
				if (val == 0) continue;

				auto [ptr1, ec1] = std::to_chars(num_buf, num_buf + sizeof(num_buf), i + 1);
				line_buf.append(num_buf, ptr1);
				line_buf.push_back(' ');

				auto [ptr2, ec2] = std::to_chars(num_buf, num_buf + sizeof(num_buf), ind + 1);
				line_buf.append(num_buf, ptr2);
				line_buf.push_back(' ');

				if constexpr (std::is_integral_v<T>) {
					auto [ptr3, ec3] = std::to_chars(num_buf, num_buf + sizeof(num_buf), val);
					line_buf.append(num_buf, ptr3);
				}
				else if constexpr (std::is_same_v<T, rat_t> || std::is_same_v<T, int_t>) {
					line_buf += val.get_str();
				}

				line_buf.push_back('\n');
			}
		}

		st.write(line_buf.data(), line_buf.size());

		if (type == SPARSE_FILE_TYPE_SMS) {
			st << "0 0 0\n";
		}
	}

} // namespace SparseRREF

#endif