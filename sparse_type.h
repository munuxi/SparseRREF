/*
	Copyright (C) 2025-2026 Zhenjie Li (Li, Zhenjie), Xiang Li (Li, Xiang)

	This file is part of SparseRREF. The SparseRREF is free software:
	you can redistribute it and/or modify it under the terms of the MIT
	License.
*/


#ifndef SPARSE_TYPE_H
#define SPARSE_TYPE_H

#include "sparse_rref.h"
#include "scalar.h"

namespace SparseRREF {
	enum SPARSE_TYPE {
		SPARSE_CSR, // Compressed sparse row
		SPARSE_COO, // Coordinate list
		SPARSE_LR,   // List of rows
		SPARSE_OTHER  // Other types
	};

	template <Flint::builtin_integral index_t = int>
	struct pivot_t {
		index_t r;
		index_t c;
	};

	// sparse vector
	template <typename T, Flint::builtin_integral index_t = int> struct sparse_vec {
		index_t* indices = nullptr;
		T* entries = nullptr;
		size_t _nnz = 0;
		size_t _alloc = 0;

		struct de_iterator_ref {
			index_t& ind;
			T& val;
		};

		struct iterator {
			index_t* ind_ptr;
			T* val_ptr;

			iterator& operator++() { ind_ptr++; val_ptr++; return *this; }
			iterator operator++(int) { iterator tmp = *this; ind_ptr++; val_ptr++; return tmp; }
			iterator& operator--() { ind_ptr--; val_ptr--; return *this; }
			iterator operator--(int) { iterator tmp = *this; ind_ptr--; val_ptr--; return tmp; }
			iterator& operator+=(size_t n) { ind_ptr += n; val_ptr += n; return *this; }
			iterator& operator-=(size_t n) { ind_ptr -= n; val_ptr -= n; return *this; }
			iterator operator+(size_t n) const { iterator tmp = *this; tmp += n; return tmp; }
			iterator operator-(size_t n) const { iterator tmp = *this; tmp -= n; return tmp; }
			bool operator==(const iterator& other) const { return ind_ptr == other.ind_ptr; }
			bool operator!=(const iterator& other) const { return ind_ptr != other.ind_ptr; }

			de_iterator_ref operator*() const { return { *ind_ptr, *val_ptr }; }
		};

		// functions of iterator 
		iterator begin() { return { indices, entries }; }
		iterator end() { return { indices + _nnz, entries + _nnz }; }
		iterator begin() const { return { indices, entries }; }
		iterator end() const { return { indices + _nnz, entries + _nnz }; }
		iterator cbegin() const { return { indices, entries }; }
		iterator cend() const { return { indices + _nnz, entries + _nnz }; }

		auto index_span() const { return std::span<index_t>(indices, _nnz); }
		auto entry_span() const { return std::span<T>(entries, _nnz); }

		// C++23 is needed for zip_view
		// auto index_view() const { return std::ranges::subrange(indices, indices + _nnz); }
		// auto entry_view() const { return std::ranges::subrange(entries, entries + _nnz); }
		// auto combine_view() const { return std::ranges::zip_view(index_view(), entry_view()); }

		sparse_vec() {
			indices = nullptr;
			entries = nullptr;
			_nnz = 0;
			_alloc = 0;
		}

		void clear() {
			if (_alloc == 0)
				return;
			s_free(indices);
			indices = nullptr;
			for (size_t i = 0; i < _alloc; i++)
				entries[i].~T();
			s_free(entries);
			entries = nullptr;
			_alloc = 0;
			_nnz = 0;
		}

		~sparse_vec() {
			clear();
		}

		void reserve(const size_t n, const bool is_copy = true) {
			if (n == _alloc)
				return;

			if (n == 0) {
				clear();
				return;
			}

			if (_alloc == 0) {
				indices = s_malloc<index_t>(n);
				entries = s_malloc<T>(n);
				for (size_t i = 0; i < n; i++)
					new (entries + i) T();
				_alloc = n;
				return;
			}

			// when expanding or using realloc, sometimes we need to copy the old data
			// if is_copy is false, we do not make sure that the old data is copied to 
			// the new memory, it is useful when we just want to enlarge the memory
			if (!is_copy && n > _alloc) {
				auto ii = s_expand(indices, n);
				auto ee = s_expand(entries, n);
				if (ii == nullptr) {
					s_free(indices);
					indices = s_malloc<index_t>(n);
				}
				else {
					indices = ii;
				}
				if (ee == nullptr) {
					for (size_t i = 0; i < _alloc; i++) {
						entries[i].~T();
					}
					s_free(entries);
					entries = s_malloc<T>(n);
					for (size_t i = 0; i < n; i++) {
						new (entries + i) T();
					}
				}
				else {
					entries = ee;
					for (size_t i = _alloc; i < n; i++) {
						new (entries + i) T();
					}
				}

				_alloc = n;
				_nnz = 0;
				return;
			}

			indices = s_realloc(indices, n);

			if (n < _alloc) {
				for (size_t i = n; i < _alloc; i++)
					entries[i].~T();
				entries = s_realloc<T>(entries, n);
			}
			else {
				entries = s_realloc<T>(entries, n);
				for (size_t i = _alloc; i < n; i++)
					new (entries + i) T();
			}

			_alloc = n;
		}

		inline void zero() { _nnz = 0; }
		inline void resize(const size_t n) { _nnz = n; }

		inline void copy(const sparse_vec& l) {
			if (this == &l)
				return;
			if (_alloc < l._nnz)
				reserve(l._nnz);
			for (size_t i = 0; i < l._nnz; i++) {
				indices[i] = l.indices[i];
				entries[i] = l.entries[i];
			}
			_nnz = l._nnz;
		}

		inline sparse_vec(const sparse_vec& l) { copy(l); }

		inline size_t nnz() const { return _nnz; }
		inline size_t size() const { return _nnz; }
		inline size_t alloc() const { return _alloc; }

		sparse_vec(sparse_vec&& l) noexcept {
			indices = l.indices;
			entries = l.entries;
			_nnz = l._nnz;
			_alloc = l._alloc;
			l.indices = nullptr;
			l.entries = nullptr;
			l._nnz = 0;
			l._alloc = 0;
		}

		sparse_vec& operator=(const sparse_vec& l) {
			if (this == &l)
				return *this;

			copy(l);
			return *this;
		}

		sparse_vec& operator=(sparse_vec&& l) noexcept {
			if (this == &l)
				return *this;

			clear();
			indices = l.indices;
			entries = l.entries;
			_nnz = l._nnz;
			_alloc = l._alloc;
			l.indices = nullptr;
			l.entries = nullptr;
			l._nnz = 0;
			l._alloc = 0;
			return *this;
		}

		// this comparison does not clear zero elements / sort the order of indices
		bool operator==(const sparse_vec& l) const {
			if (this == &l)
				return true;
			if (_nnz != l._nnz)
				return false;
			return std::equal(indices, indices + _nnz, l.indices)
				&& std::equal(entries, entries + _nnz, l.entries);
		}

		// this comparison will clear zero elements / sort the order of indices
		bool is_equal_to(const sparse_vec& l) const {
			if (this == &l)
				return true;
			auto this_temp = *this;
			auto other_temp = l;
			this_temp.canonicalize();
			other_temp.canonicalize();
			if (this_temp._nnz != other_temp._nnz)
				return false;
			this_temp.sort_indices();
			other_temp.sort_indices();
			return std::equal(this_temp.indices, this_temp.indices + this_temp._nnz, other_temp.indices)
				&& std::equal(this_temp.entries, this_temp.entries + this_temp._nnz, other_temp.entries);
		}

		inline void push_back(const index_t index, const T& val) {
			if (_nnz + 1 > _alloc)
				reserve((1 + _alloc) * 2); // +1 to avoid _alloc = 0
			indices[_nnz] = index;
			entries[_nnz] = val;
			_nnz++;
		}

		inline void push_back(const index_t index, T&& val) {
			if (_nnz + 1 > _alloc)
				reserve((1 + _alloc) * 2); // +1 to avoid _alloc = 0
			indices[_nnz] = index;
			entries[_nnz] = std::move(val);
			_nnz++;
		}

		void pop_back() {
			if (_nnz != 0)
				_nnz--;
		}

		// take a span of elements
		// sparse_vec.take({start, end}) returns a sparse_vec with elements indexed in [start, end)
		// elements in the resulting sparse_vec are reindexed in [0, end - start)
		sparse_vec<T, index_t> take(const std::pair<index_t, index_t>& span, const bool reserve_nnz = true) const {
			if (span.first < 0 || span.second < 0)
				throw std::invalid_argument("sparse_vec.take: expect non-negative indices.");
			if (span.first > span.second)
				throw std::invalid_argument("sparse_vec.take: invalid span.");

			sparse_vec<T, index_t> result;
			if (reserve_nnz)
				result.reserve(_nnz);

			for (size_t i = 0; i < _nnz; i++) {
				if (indices[i] >= span.first && indices[i] < span.second) {
					result.push_back(indices[i] - span.first, entries[i]);
				}
			}
			result.compress();
			return result;
		}

		index_t& operator()(const size_t pos) { return indices[pos]; }
		const index_t& operator()(const size_t pos) const { return indices[pos]; }
		T& operator[](const size_t pos) { return entries[pos]; }
		const T& operator[](const size_t pos) const { return entries[pos]; }

		T* find(const index_t index, const bool isbinary = true) const {
			if (_nnz == 0)
				return nullptr;
			index_t* ptr;
			if (isbinary)
				ptr = SparseRREF::binary_search(indices, indices + _nnz, index);
			else
				ptr = std::find(indices, indices + _nnz, index);
			if (ptr == indices + _nnz)
				return nullptr;
			return entries + (ptr - indices);
		}

		// conversion functions
		template <typename U = T> requires std::is_integral_v<U> || std::is_same_v<U, int_t>
		operator sparse_vec<rat_t, index_t>() {
			sparse_vec<rat_t, index_t> result;
			result.reserve(_nnz);
			result.resize(_nnz);
			for (size_t i = 0; i < _nnz; i++) {
				result.indices[i] = indices[i];
				result.entries[i] = entries[i];
			}
			return result;
		}

		template <typename U = T> requires std::is_integral_v<U>
		operator sparse_vec<int_t, index_t>() {
			sparse_vec<int_t, index_t> result;
			result.reserve(_nnz);
			result.resize(_nnz);
			for (size_t i = 0; i < _nnz; i++) {
				result.indices[i] = indices[i];
				result.entries[i] = entries[i];
			}
			return result;
		}

		template <typename U = T> requires std::is_same_v<U, rat_t>
		sparse_vec<ulong, index_t> operator%(const nmod_t mod) const {
			sparse_vec<ulong, index_t> result;
			result.reserve(_nnz);
			result.resize(_nnz);
			for (size_t i = 0; i < _nnz; i++) {
				result.indices[i] = indices[i];
				result.entries[i] = entries[i] % mod;
			}
			return result;
		}

		template <typename U = T> requires std::is_same_v<U, rat_t>
		sparse_vec<ulong, index_t> operator%(const ulong p) const {
			sparse_vec<ulong, index_t> result;
			nmod_t mod;
			nmod_init(&mod, p);
			result.reserve(_nnz);
			result.resize(_nnz);
			for (size_t i = 0; i < _nnz; i++) {
				result.indices[i] = indices[i];
				result.entries[i] = entries[i] % mod;
			}
			return result;
		}

		void canonicalize() {
			size_t new_nnz = 0;
			for (size_t i = 0; i < _nnz; i++) {
				if (entries[i] != 0) {
					if (new_nnz != i) {
						indices[new_nnz] = indices[i];
						entries[new_nnz] = entries[i];
					}
					new_nnz++;
				}
			}
			_nnz = new_nnz;
		}

		void sort_indices() {
			if (_nnz <= 1 || std::is_sorted(indices, indices + _nnz))
				return;

			auto perm = perm_init(_nnz);
			std::ranges::sort(perm, [&](index_t a, index_t b) {
				return indices[a] < indices[b];
				});

			permute(perm, indices);
			permute(perm, entries);
		}

		void compress() {
			canonicalize();
			sort_indices();
			reserve(_nnz);
		}
	};

	template <Flint::builtin_integral index_t> struct sparse_vec<bool, index_t> {
		index_t* indices = nullptr;
		size_t _nnz = 0;
		size_t _alloc = 0;

		auto index_span() const { return std::span<index_t>(indices, _nnz); }

		sparse_vec() {
			indices = nullptr;
			_nnz = 0;
			_alloc = 0;
		}

		void clear() {
			if (_alloc != 0)
				s_free(indices);
			_alloc = 0;
			_nnz = 0;
		}

		~sparse_vec() { clear(); }

		void reserve(const size_t n) {
			if (n == _alloc)
				return;

			if (n == 0) {
				clear();
				return;
			}

			if (_alloc == 0) {
				indices = s_malloc<index_t>(n);
				_alloc = n;
				return;
			}

			indices = s_realloc(indices, n);
			_alloc = n;
		}

		void resize(const size_t n) { _nnz = n; }

		inline void copy(const sparse_vec& l) {
			if (this == &l)
				return;
			if (_alloc < l._nnz)
				reserve(l._nnz);
			for (size_t i = 0; i < l._nnz; i++) {
				indices[i] = l.indices[i];
			}
			_nnz = l._nnz;
		}

		sparse_vec(const sparse_vec& l) { copy(l); }
		size_t nnz() const { return _nnz; }
		size_t alloc() const { return _alloc; }

		sparse_vec(sparse_vec&& l) noexcept {
			indices = l.indices;
			_nnz = l._nnz;
			_alloc = l._alloc;
			l.indices = nullptr;
			l._nnz = 0;
			l._alloc = 0;
		}

		sparse_vec& operator=(const sparse_vec& l) {
			if (this == &l)
				return *this;

			copy(l);
			return *this;
		}

		sparse_vec& operator=(sparse_vec&& l) noexcept {
			if (this == &l)
				return *this;

			clear();
			indices = l.indices;
			_nnz = l._nnz;
			_alloc = l._alloc;
			l.indices = nullptr;
			l._nnz = 0;
			l._alloc = 0;
			return *this;
		}

		void push_back(const index_t index, const bool val = true) {
			if (_nnz + 1 > _alloc)
				reserve((1 + _alloc) * 2); // +1 to avoid _alloc = 0
			indices[_nnz] = index;
			_nnz++;
		}

		void pop_back() {
			if (_nnz != 0)
				_nnz--;
		}

		index_t& operator()(const size_t pos) { return indices[pos]; }
		const index_t& operator()(const size_t pos) const { return indices[pos]; }
		void zero() { _nnz = 0; }
		void sort_indices() { std::sort(indices, indices + _nnz); }
		void canonicalize() {}
		void compress() { sort_indices(); }
	};

	template <typename T, Flint::builtin_integral index_t = int> struct sparse_mat {
		size_t nrow = 0;
		size_t ncol = 0;
		std::vector<sparse_vec<T, index_t>> rows;

		void init(const size_t r, const size_t c) {
			nrow = r;
			ncol = c;
			rows = std::vector<sparse_vec<T, index_t>>(r);
		}

		sparse_mat() { nrow = 0; ncol = 0; }
		~sparse_mat() {}
		sparse_mat(const size_t r, const size_t c) { init(r, c); }

		sparse_vec<T, index_t>& operator[](const size_t i) { return rows[i]; }
		const sparse_vec<T, index_t>& operator[](const size_t i) const { return rows[i]; }

		T* find(const size_t row, const index_t col, const bool isbinary = true) const {
			return rows[row].find(col, isbinary);
		}

		sparse_mat(const sparse_mat& l) {
			init(l.nrow, l.ncol);
			rows = l.rows;
		}

		sparse_mat(sparse_mat&& l) noexcept {
			nrow = l.nrow;
			ncol = l.ncol;
			rows = std::move(l.rows);
			l.nrow = 0;
		}

		sparse_mat& operator=(const sparse_mat& l) {
			if (this == &l)
				return *this;
			nrow = l.nrow;
			ncol = l.ncol;
			rows = l.rows;
			return *this;
		}

		sparse_mat& operator=(sparse_mat&& l) noexcept {
			if (this == &l)
				return *this;
			nrow = l.nrow;
			ncol = l.ncol;
			rows = std::move(l.rows);
			l.nrow = 0;
			return *this;
		}

		void zero() {
			for (size_t i = 0; i < nrow; i++)
				rows[i].zero();
		}

		void clear() {
			for (size_t i = 0; i < nrow; i++) {
				rows[i].clear();
			}
			std::vector<sparse_vec<T, index_t>> tmp;
			rows.swap(tmp); // clear the vector and free memory
			nrow = 0;
			ncol = 0;
		}

		size_t nnz() const {
			size_t n = 0;
			for (size_t i = 0; i < nrow; i++)
				n += rows[i].nnz();
			return n;
		}

		size_t alloc() const {
			size_t n = 0;
			for (size_t i = 0; i < nrow; i++)
				n += rows[i].alloc();
			return n;
		}

		void compress(thread_pool* pool = nullptr) {
			if (pool == nullptr) {
				for (size_t i = 0; i < nrow; i++) {
					rows[i].compress();
				}
			}
			else {
				pool->detach_loop(0, nrow, [&](size_t i) {
					rows[i].compress();
					});
				pool->wait();
			}
		}

		void clear_zero_row() {
			size_t new_nrow = 0;
			for (size_t i = 0; i < nrow; i++) {
				if (rows[i].nnz() != 0) {
					std::swap(rows[new_nrow], rows[i]);
					new_nrow++;
				}
			}
			nrow = new_nrow;
			rows.resize(nrow);
			rows.shrink_to_fit();
		}

		sparse_mat<T, index_t> transpose() {
			sparse_mat<T, index_t> res(ncol, nrow);
			for (size_t i = 0; i < ncol; i++)
				res[i].zero();

			for (size_t i = 0; i < nrow; i++) {
				for (size_t j = 0; j < rows[i].nnz(); j++) {
					res[rows[i](j)].push_back(i, rows[i][j]);
				}
			}
			return res;
		}

		// take a span of rows
		// sparse_mat.take({start, end}) returns a sparse_mat with rows indexed in [start, end)
		sparse_mat<T, index_t> take(const std::pair<size_t, size_t>& span, thread_pool* pool = nullptr) const {
			if (span.second > nrow) {
				throw std::out_of_range("sparse_mat.take: [start, end) out of [0, nrow).");
			}
			if (span.first > span.second) {
				throw std::invalid_argument("sparse_mat.take: invalid span.");
			}

			sparse_mat<T, index_t> res(span.second - span.first, ncol);
			if (pool == nullptr) {
				res.rows.assign(rows.begin() + span.first, rows.begin() + span.second);
			}
			else {
				pool->detach_loop(span.first, span.second, [&](size_t i) {
					res[i - span.first] = rows[i];
					});
				pool->wait();
			}
			return res;
		}

		// take a span of elements
		// sparse_mat.take(levelspec, {start, end}) returns a sparse_mat whose elements have the levelspec-th index in range [start, end)
		// elements in the resulting sparse_mat have their levelspec-th indices reindexed in [0, end - start)
		sparse_mat<T, index_t> take(const size_t levelspec, const std::pair<index_t, index_t>& span, thread_pool* pool = nullptr) const {
			if (span.first < 0 || span.second < 0)
				throw std::invalid_argument("sparse_mat.take: expect non-negative indices.");
			if (span.first > span.second)
				throw std::invalid_argument("sparse_mat.take: invalid span.");
			if (levelspec > 1)
				throw std::invalid_argument("sparse_mat.take: levelspec must be 0 or 1.");

			if (levelspec == 0)
				return take(span, pool);

			// then levelspec == 1
			if (span.second > ncol)
				throw std::out_of_range("sparse_mat.take: [start, end) out of [0, ncol).");
			
			sparse_mat<T, index_t> res(nrow, span.second - span.first);
			if (pool == nullptr) {
				for (size_t i = 0; i < nrow; i++) {
					res.rows[i] = rows[i].take(span);
				}
			}
			else {
				pool->detach_loop(0, nrow, [&](size_t i) {
					res.rows[i] = rows[i].take(span);
					});
				pool->wait();
			}
			return res;
		}

		// if is_binary is true, we assume that the column indices are sorted in each row
		sparse_mat<T, index_t> submat(const std::pair<size_t, size_t>& rowspan,
			const std::pair<size_t, size_t>& colspan, const bool is_binary = true) const {

			if (colspan.second < colspan.first || rowspan.second < rowspan.first) {
				std::cerr << "sparse_mat.submat: invalid span." << std::endl;
				return sparse_mat<T, index_t>();
			}

			sparse_mat<T, index_t> res(rowspan.second - rowspan.first, colspan.second - colspan.first);

			if (is_binary) {
				for (size_t i = rowspan.first; i < rowspan.second; i++) {
					// binary search for the starting and end position
					auto indptr = rows[i].indices;
					auto it_start = std::lower_bound(indptr, indptr + rows[i].nnz(), colspan.first);
					auto it_end = std::lower_bound(it_start, indptr + rows[i].nnz(), colspan.second);
					res[i - rowspan.first].reserve(it_end - it_start);
					for (auto it = it_start; it != it_end; it++) {
						auto pos = it - indptr;
						res[i - rowspan.first].push_back(indptr[pos] - colspan.first, rows[i].entries[pos]);
					}
				}
			}
			else {
				for (size_t i = rowspan.first; i < rowspan.second; i++) {
					// use more memory but simpler
					res[i - rowspan.first].reserve(rows[i].nnz());
					for (size_t j = 0; j < rows[i].nnz(); j++) {
						auto col_index = rows[i](j);
						if (col_index >= colspan.first && col_index < colspan.second)
							res[i - rowspan.first].push_back(col_index - colspan.first, rows[i][j]);
					}
				}
			}
			return res;
		}

		// append other sparse_mat to this one
		void append(const sparse_mat<T, index_t>& other, thread_pool* pool = nullptr) {
			ncol = std::max(ncol, other.ncol);

			if (pool == nullptr) {
				rows.insert(rows.end(), other.rows.begin(), other.rows.end());
				nrow += other.nrow;
				return;
			}

			rows.reserve(nrow + other.nrow);
			pool->detach_loop(0, other.nrow, [&](size_t i) {
				rows.emplace_back(other.rows[i]);
				});
			pool->wait();
			nrow += other.nrow;
		}

		void append(sparse_mat<T, index_t>&& other) {
			ncol = std::max(ncol, other.ncol);

			rows.insert(rows.end(), std::make_move_iterator(other.rows.begin()), std::make_move_iterator(other.rows.end()));
			nrow += other.nrow;
			other.clear(); // clear the other matrix
		}

		// sort rows by nnz
		void sort_rows_by_nnz() {
			std::ranges::stable_sort(rows, std::less{}, &sparse_vec<T, index_t>::_nnz);
		}

		template <typename U = T> requires std::is_same_v<U, rat_t>
		sparse_mat<ulong> operator%(const nmod_t mod) const {
			sparse_mat<ulong> result(nrow, ncol);
			for (size_t i = 0; i < nrow; i++) {
				result[i] = rows[i] % mod;
			}
			return result;
		}

		template <typename U = T> requires std::is_same_v<U, rat_t>
		ulong height_bits() const {
			ulong max_height = 0;
			for (size_t i = 0; i < nrow; i++) {
				for (size_t j = 0; j < rows[i].nnz(); j++) {
					auto rr = rows[i][j].height_bits();
					if (rr > max_height)
						max_height = rr;
				}
			}

			return max_height;
		}
	};

	// CSR format for sparse tensor
	template <typename T, typename index_t> struct sparse_tensor_struct {
		size_t rank;
		size_t alloc;
		index_t* colptr;
		T* valptr;
		std::vector<size_t> dims;
		std::vector<size_t> rowptr;

		using index_v = std::vector<index_t>;
		using index_p = index_t*;
		using const_index_p = const index_t*;
		using interval_t = std::pair<index_t, index_t>;

		// empty constructor
		sparse_tensor_struct() {
			rank = 0;
			alloc = 0;
			colptr = nullptr;
			valptr = nullptr;
		}

		// Constructor with dimensions
		// we require that rank >= 2
		void init(const std::vector<size_t>& l, const size_t aoc = 8) {
			dims = l;
			rank = l.size();
			rowptr = std::vector<size_t>(l[0] + 1, 0);
			alloc = aoc;
			colptr = s_malloc<index_t>((rank - 1) * alloc);
			valptr = s_malloc<T>(alloc);
			for (size_t i = 0; i < alloc; i++)
				new (valptr + i) T();
		}

		sparse_tensor_struct(const std::vector<size_t>& l, const size_t aoc = 8) {
			init(l, aoc);
		}

		// Copy constructor
		sparse_tensor_struct(const sparse_tensor_struct& l) {
			init(l.dims, l.alloc);
			std::copy(l.rowptr.begin(), l.rowptr.end(), rowptr.begin());
			std::copy(l.colptr, l.colptr + alloc * (rank - 1), colptr);
			for (size_t i = 0; i < alloc; i++)
				valptr[i] = l.valptr[i];
		}

		// Move constructor
		sparse_tensor_struct(sparse_tensor_struct&& l) noexcept {
			dims = l.dims;
			rank = l.rank;
			rowptr = l.rowptr;
			alloc = l.alloc;
			colptr = l.colptr;
			l.colptr = nullptr;
			valptr = l.valptr;
			l.valptr = nullptr;
			l.alloc = 0; // important for no repeating clear
		}

		void clear() {
			if (alloc == 0)
				return;
			for (size_t i = 0; i < alloc; i++)
				valptr[i].~T();
			s_free(valptr);
			s_free(colptr);
			valptr = nullptr;
			colptr = nullptr;
			alloc = 0;
		}

		~sparse_tensor_struct() {
			clear();
		}

		void reserve(const size_t size) {
			if (size == alloc)
				return;
			if (size == 0) {
				clear();
				return;
			}
			if (alloc == 0) {
				alloc = size;
				colptr = s_malloc<index_t>(size * (rank - 1));
				valptr = s_malloc<T>(size);
				for (size_t i = 0; i < size; i++)
					new (valptr + i) T();
				return;
			}
			colptr = s_realloc<index_t>(colptr, size * (rank - 1));
			if (size > alloc) {
				valptr = s_realloc<T>(valptr, size);
				for (size_t i = alloc; i < size; i++)
					new (valptr + i) T();
			}
			else if (size < alloc) {
				for (size_t i = size; i < alloc; i++)
					valptr[i].~T();
				valptr = s_realloc<T>(valptr, size);
			}
			alloc = size;
		}

		// change the dimensions of the tensor
		// it is dangerous, only for internal use
		void change_dims(const std::vector<size_t>& new_dims) {
			dims = new_dims;
			rank = new_dims.size();
			colptr = s_realloc<index_t>(colptr, alloc * (rank - 1));
		}

		void zero() {
			if (rank != 0)
				std::fill(rowptr.begin(), rowptr.end(), 0);
		}

		inline size_t nnz() const {
			return rowptr[dims[0]];
		}

		// Copy assignment
		sparse_tensor_struct& operator=(const sparse_tensor_struct& l) {
			if (this == &l)
				return *this;
			auto nz = l.nnz();
			if (alloc == 0) {
				init(l.dims, nz);
				std::copy(l.rowptr.begin(), l.rowptr.end(), rowptr.begin());
				std::copy(l.colptr, l.colptr + nz * (rank - 1), colptr);
				std::copy(l.valptr, l.valptr + nz, valptr);
				return *this;
			}
			dims = l.dims;
			rank = l.rank;
			rowptr = l.rowptr;
			if (alloc < nz)
				reserve(nz);
			std::copy(l.colptr, l.colptr + nz * (rank - 1), colptr);
			std::copy(l.valptr, l.valptr + nz, valptr);
			return *this;
		}

		// Move assignment
		sparse_tensor_struct& operator=(sparse_tensor_struct&& l) noexcept {
			if (this == &l)
				return *this;
			clear();
			dims = l.dims;
			rank = l.rank;
			rowptr = l.rowptr;
			alloc = l.alloc;
			colptr = l.colptr;
			l.colptr = nullptr;
			valptr = l.valptr;
			l.valptr = nullptr;
			l.alloc = 0; // important for no repeating clear
			return *this;
		}

		std::vector<size_t> row_nums() const {
			return SparseRREF::difference(rowptr);
		}

		// nnz in the i-th row
		size_t row_nnz(const size_t i) const {
			return rowptr[i + 1] - rowptr[i];
		}

		// row index of the i-th entry
		size_t row_index(const size_t i) const {
			auto it = std::upper_bound(rowptr.begin(), rowptr.end(), i);
			return std::distance(rowptr.begin(), it) - 1;
		}

		// remove zero entries, double pointer
		void canonicalize() {
			size_t nnz_now = nnz();
			size_t index = 0;
			std::vector<size_t> newrowptr(dims[0] + 1);
			newrowptr[0] = 0;
			for (size_t i = 0; i < dims[0]; i++) {
				for (size_t j = rowptr[i]; j < rowptr[i + 1]; j++) {
					if (valptr[j] != 0) {
						s_copy(colptr + index * (rank - 1), colptr + j * (rank - 1), rank - 1);
						valptr[index] = valptr[j];
						index++;
					}
				}
				newrowptr[i + 1] = index;
			}
			rowptr = newrowptr;
		}

		std::pair<index_p, T*> row(const size_t i) {
			return { colptr + rowptr[i] * (rank - 1), valptr + rowptr[i] };
		}

		index_p entry_lower_bound(const_index_p l) {
			auto begin = row(l[0]).first;
			auto end = row(l[0] + 1).first;
			if (begin == end)
				return end;
			return SparseRREF::lower_bound(begin, end, l + 1, rank - 1);
		}

		index_p entry_lower_bound(const index_v& l) {
			return entry_lower_bound(l.data());
		}

		index_p entry_ptr(const index_p l) {
			auto ptr = entry_lower_bound(l);
			auto end = row(l[0] + 1).first;
			if (ptr == end || std::equal(ptr, ptr + rank - 1, l + 1))
				return ptr;
			else
				return end;
		}

		index_p entry_ptr(const index_v& l) {
			return entry_ptr(l.data());
		}

		// unordered, push back on the end of the row
		void push_back(const index_v& l, const T& val) {
			index_t row = l[0];
			size_t nnz = this->nnz();
			if (nnz + 1 > alloc)
				reserve((alloc + 1) * 2);
			size_t index = rowptr[row + 1];
			for (size_t i = nnz; i > index; i--) {
				auto tmpptr = colptr + (i - 1) * (rank - 1);
				std::copy_backward(tmpptr, tmpptr + (rank - 1), tmpptr + 2 * (rank - 1));
				valptr[i] = valptr[i - 1];
			}
			for (size_t i = 0; i < rank - 1; i++)
				colptr[index * (rank - 1) + i] = l[i + 1];
			valptr[index] = val;
			for (size_t i = row + 1; i <= dims[0]; i++)
				rowptr[i]++;
		}

		// ordered insert
		// mode = false: insert anyway
		// mode = true: insert and replace if exist
		void insert(const index_v& l, const T& val, bool mode = true) {
			size_t trow = l[0];
			size_t nnz = this->nnz();
			if (nnz + 1 > alloc)
				reserve((alloc + 1) * 2);
			auto ptr = entry_lower_bound(l);
			size_t index = (ptr - colptr) / (rank - 1);
			bool exist = (ptr != row(trow + 1).first && std::equal(ptr, ptr + rank - 1, l.data() + 1));
			if (!exist || !mode) {
				for (size_t i = nnz; i > index; i--) {
					auto tmpptr = colptr + (i - 1) * (rank - 1);
					std::copy_backward(tmpptr, tmpptr + (rank - 1), tmpptr + 2 * (rank - 1));
					valptr[i] = valptr[i - 1];
				}
				std::copy(l.begin() + 1, l.begin() + rank, colptr + index * (rank - 1));
				valptr[index] = val;
				for (size_t i = trow + 1; i <= dims[0]; i++)
					rowptr[i]++;
				return;
			}
			valptr[index] = val;
		}

		// ordered add one value
		void insert_add(const index_v& l, const T& val) {
			size_t trow = l[0];
			size_t nnz = this->nnz();
			if (nnz + 1 > alloc)
				reserve((alloc + 1) * 2);
			auto ptr = entry_lower_bound(l);
			size_t index = (ptr - colptr) / (rank - 1);
			bool exist = (ptr != row(trow + 1).first && std::equal(ptr, ptr + rank - 1, l.data() + 1));
			if (!exist) {
				for (size_t i = nnz; i > index; i--) {
					auto tmpptr = colptr + (i - 1) * (rank - 1);
					std::copy_backward(tmpptr, tmpptr + (rank - 1), tmpptr + 2 * (rank - 1));
					valptr[i] = valptr[i - 1];
				}
				std::copy(l.begin() + 1, l.begin() + rank, colptr + index * (rank - 1));
				valptr[index] = val;
				for (size_t i = trow + 1; i <= dims[0]; i++)
					rowptr[i]++;
				return;
			}
			valptr[index] += val;
		}

		/**
    	 * @brief Traverse tensor rows in the range `[row_start, row_end)` and apply function `func` to each entry. The function `row_init` is called before processing each row. Non-threaded version.
    	 *
    	 * @tparam F1 The type of the traversal function.
    	 * @tparam F2 The type of the row initialization function.
    	 * @param row_start The starting row index (inclusive).
		 * @param row_end The ending row index (exclusive).
		 * @param func The traversal function to apply to each entry. Takes at least two `size_t` arguments: row index and entry index. May also receive the return value of `row_init` (if not void) as an additional argument.
		 * @param row_init The row initialization function to be called before processing each row. Takes one `size_t` arguments: row index
    	 */
		template <typename F1, typename F2>
		inline void traverse(const size_t row_start, const size_t row_end, F1&& func, F2&& row_init) const {
			using F2_return_type = std::invoke_result_t<F2, size_t>;
			for (size_t i = row_start; i < row_end; i++) {
				if constexpr (std::is_void_v<F2_return_type>) {
					static_assert(std::is_invocable_v<F1, size_t, size_t>, "sparse_tensor_struct.traverse: func must be invocable with (size_t, size_t) when row_init returns void.");
					row_init(i);
					for (size_t j = rowptr[i]; j < rowptr[i + 1]; j++) {
						func(i, j);
					}
				}
				else {
					using F2_lvalue_type = std::conditional_t<std::is_reference_v<F2_return_type>, F2_return_type,std::add_lvalue_reference_t<F2_return_type>>;
					static_assert(std::is_invocable_v<F1, size_t, size_t, F2_lvalue_type>, "sparse_tensor_struct.traverse: func must be invocable with (size_t, size_t, row_init return type) when row_init does not return void.");
					auto row_data = row_init(i);
					for (size_t j = rowptr[i]; j < rowptr[i + 1]; j++) {
						func(i, j, row_data);
					}
				}
			}
		}

		/**
    	 * @brief Traverse tensor rows in the range `[row_start, row_end)` and apply function `func` to each entry using multi-threading. The function `row_init` is called before processing each row, and the function `block_init` is called before processing each block within a row. This will also setup a vector of `BS::blocks<size_t>` for each row to facilitate further parallel operations.
    	 *
    	 * @tparam F1 The type of the traversal function.
    	 * @tparam F2 The type of the row initialization function.
		 * @tparam F3 The type of the block initialization function.
    	 * @param row_start The starting row index (inclusive).
		 * @param row_end The ending row index (exclusive).
		 * @param func The traversal function to apply to each entry. Takes at least two `size_t` arguments: row index and entry index. May also receive the return values of `row_init` and `block_init` (if not void) as additional arguments.
		 * @param row_init The row initialization function to be called before processing each row. Takes two arguments: row index and `BS::blocks<size_t>` for that row.
		 * @param block_init The block initialization function to be called before processing each block within a row. Takes at least two arguments: row index and block index. May also receive the return value of `row_init` (if not void) as an additional argument.
		 * @param pool Pointer to a thread pool for parallel execution. Must not be `nullptr`.
		 * @return A vector of `BS::blocks<size_t>` for each row in the specified range.
    	 */
		template <typename F1, typename F2, typename F3>
		inline std::vector<BS::blocks<size_t>> traverse_setup_blocks(const size_t row_start, const size_t row_end, F1&& func, F2&& row_init, F3&& block_init, thread_pool* pool) const {
			std::vector<BS::blocks<size_t>> row_blocks;
			if (pool == nullptr)
				throw std::invalid_argument("sparse_tensor_struct.traverse_setup_blocks: thread pool is required for setting up blocks.");

			using F2_return_type = std::invoke_result_t<F2, size_t, BS::blocks<size_t>>;
			
			auto nthread = pool->get_thread_count();
			for (size_t i = row_start; i < row_end; i++) {
				// separate row elements into blocks
				const BS::blocks<size_t> blks(rowptr[i], rowptr[i + 1], nthread);
				row_blocks.push_back(blks);
				if constexpr (std::is_void_v<F2_return_type>) {
					using F3_return_type = std::invoke_result_t<F3, size_t, size_t>;
					row_init(i, blks);
					pool->detach_sequence(0, blks.get_num_blocks(), [&](size_t blk) {
						if constexpr (std::is_void_v<F3_return_type>) {
							static_assert(std::is_invocable_v<F1, size_t, size_t, size_t>, "sparse_tensor_struct.traverse_setup_blocks: func must be invocable with (size_t, size_t, size_t) when both row_init and block_init return void.");
							block_init(i, blk);
							for (size_t j = blks.start(blk); j < blks.end(blk); j++) {
								func(i, j, blk);
							}
						}
						else {
							using F3_lvalue_type = std::conditional_t<std::is_reference_v<F3_return_type>, F3_return_type,std::add_lvalue_reference_t<F3_return_type>>;
							static_assert(std::is_invocable_v<F1, size_t, size_t, size_t, F3_lvalue_type>, "sparse_tensor_struct.traverse_setup_blocks: func must be invocable with (size_t, size_t, size_t, block_init return type) when row_init returns void and block_init does not.");
							auto block_data = block_init(i, blk);
							for (size_t j = blks.start(blk); j < blks.end(blk); j++) {
								func(i, j, blk, block_data);
							}
						}
						});
					pool->wait();
				}
				else {
					using F2_lvalue_type = std::conditional_t<std::is_reference_v<F2_return_type>, F2_return_type,std::add_lvalue_reference_t<F2_return_type>>;
					using F3_return_type = std::invoke_result_t<F3, size_t, size_t, F2_lvalue_type>;
					auto row_data = row_init(i, blks);
					pool->detach_sequence(0, blks.get_num_blocks(), [&](size_t blk) {
						if constexpr (std::is_void_v<std::invoke_result_t<F3, size_t, size_t, F2_lvalue_type>>) {
							static_assert(std::is_invocable_v<F1, size_t, size_t, size_t, std::invoke_result_t<F2, size_t, BS::blocks<size_t>>>, "sparse_tensor_struct.traverse_setup_blocks: func must be invocable with (size_t, size_t, size_t, row_init return type) when block_init returns void and row_init does not.");
							block_init(i, blk, row_data);
							for (size_t j = blks.start(blk); j < blks.end(blk); j++) {
								func(i, j, blk, row_data);
							}
						}
						else {
							using F3_lvalue_type = std::conditional_t<std::is_reference_v<F3_return_type>, F3_return_type,std::add_lvalue_reference_t<F3_return_type>>;
							static_assert(std::is_invocable_v<F1, size_t, size_t, size_t, F2_lvalue_type, F3_lvalue_type>, "sparse_tensor_struct.traverse_setup_blocks: func must be invocable with (size_t, size_t, size_t, row_init return type, block_init return type) when both row_init and block_init do not return void.");
							auto block_data = block_init(i, blk, row_data);
							for (size_t j = blks.start(blk); j < blks.end(blk); j++) {
								func(i, j, blk, row_data, block_data);
							}
						}
						});
					pool->wait();
				}
			}
			
			return row_blocks;
		}

		/**
    	 * @brief Traverse tensor rows in the range `[row_start, row_end)` and apply function `func` to each entry using multi-threading. The function `row_init` is called before processing each row, and the function `block_init` is called before processing each block within a row. This will use the given `std::span<BS::blocks<size_t>>` for multi-threading.
    	 *
    	 * @tparam F1 The type of the traversal function.
    	 * @tparam F2 The type of the row initialization function.
		 * @tparam F3 The type of the block initialization function.
    	 * @param row_start The starting row index (inclusive).
		 * @param row_end The ending row index (exclusive).
		 * @param func The traversal function to apply to each entry. Takes at least two `size_t` arguments: row index and entry index. May also receive the return values of `row_init` and `block_init` (if not void) as additional arguments.
		 * @param row_init The row initialization function to be called before processing each row. Takes two arguments: row index (`size_t`) and `BS::blocks<size_t>` for that row.
		 * @param block_init The block initialization function to be called before processing each block within a row. Takes at least two `size_t` arguments: row index and block index. May also receive the return value of `row_init` (if not void) as an additional argument.
		 * @param row_blocks A `std::span<BS::blocks<size_t>>` containing the blocks for each row to be used.
		 * @param pool Pointer to a thread pool for parallel execution.
    	 */
		template <typename F1, typename F2, typename F3>
		inline void traverse_using_blocks(const size_t row_start, const size_t row_end, F1&& func, F2&& row_init, F3&& block_init, const std::span<BS::blocks<size_t>>& row_blocks, thread_pool* pool) const {
			if (pool == nullptr)
				throw std::invalid_argument("sparse_tensor_struct.traverse_using_blocks: thread pool is required for using blocks.");

			using F2_return_type = std::invoke_result_t<F2, size_t, BS::blocks<size_t>>;
			
			for (size_t i = row_start; i < row_end; i++) {
				const BS::blocks<size_t>& blks = row_blocks[i - row_start];
				if (blks.get_num_blocks() == 0)
					continue;
				if constexpr (std::is_void_v<F2_return_type>) {
					using F3_return_type = std::invoke_result_t<F3, size_t, size_t>;
					row_init(i, blks);
					pool->detach_sequence(0, blks.get_num_blocks(), [&](size_t blk) {
						if constexpr (std::is_void_v<F3_return_type>) {
							static_assert(std::is_invocable_v<F1, size_t, size_t, size_t>, "sparse_tensor_struct.traverse_using_blocks: func must be invocable with (size_t, size_t, size_t) when both row_init and block_init return void.");
							block_init(i, blk);
							for (size_t j = blks.start(blk); j < blks.end(blk); j++) {
								func(i, j, blk);
							}
						}
						else {
							using F3_lvalue_type = std::conditional_t<std::is_reference_v<F3_return_type>, F3_return_type,std::add_lvalue_reference_t<F3_return_type>>;
							static_assert(std::is_invocable_v<F1, size_t, size_t, size_t, F3_lvalue_type>, "sparse_tensor_struct.traverse_using_blocks: func must be invocable with (size_t, size_t, size_t, block_init return type) when row_init returns void and block_init does not.");
							auto block_data = block_init(i, blk);
							for (size_t j = blks.start(blk); j < blks.end(blk); j++) {
								func(i, j, blk, block_data);
							}
						}
						});
					pool->wait();
				}
				else {
					using F2_lvalue_type = std::conditional_t<std::is_reference_v<F2_return_type>, F2_return_type,std::add_lvalue_reference_t<F2_return_type>>;
					using F3_return_type = std::invoke_result_t<F3, size_t, size_t, F2_lvalue_type>;
					auto row_data = row_init(i, blks);
					pool->detach_sequence(0, blks.get_num_blocks(), [&](size_t blk) {
						if constexpr (std::is_void_v<F3_return_type>) {
							static_assert(std::is_invocable_v<F1, size_t, size_t, size_t, F2_lvalue_type>, "sparse_tensor_struct.traverse_using_blocks: func must be invocable with (size_t, size_t, size_t, row_init return type) when block_init returns void and row_init does not.");
							block_init(i, blk, row_data);
							for (size_t j = blks.start(blk); j < blks.end(blk); j++) {
								func(i, j, blk, row_data);
							}
						}
						else {
							using F3_lvalue_type = std::conditional_t<std::is_reference_v<F3_return_type>, F3_return_type,std::add_lvalue_reference_t<F3_return_type>>;
							static_assert(std::is_invocable_v<F1, size_t, size_t, size_t, F2_lvalue_type, F3_lvalue_type>, "sparse_tensor_struct.traverse_using_blocks: func must be invocable with (size_t, size_t, size_t, row_init return type, block_init return type) when both row_init and block_init do not return void.");
							auto block_data = block_init(i, blk, row_data);
							for (size_t j = blks.start(blk); j < blks.end(blk); j++) {
								func(i, j, blk, row_data, block_data);
							}
						}
						});
					pool->wait();
				}
			}			
		}

		// take a span of rows
		// sparse_tensor_struct.take({start, end}) returns a sparse_tensor with rows indexed in [start, end)
		sparse_tensor_struct<T, index_t> take(const interval_t& span, thread_pool* pool = nullptr) const {
			if (span.second > dims[0])
				throw std::invalid_argument("sparse_tensor_struct.take: [start, end) out of [0, dims[0]).");
			if (span.first > span.second)
				throw std::invalid_argument("sparse_tensor_struct.take: invalid span.");
			if (span.first < 0 || span.second < 0)
				throw std::invalid_argument("sparse_tensor_struct.take: expect non-negative indices.");

			std::vector<size_t> res_dims = dims;
			res_dims[0] = span.second - span.first;
			size_t res_nnz = rowptr[span.second] - rowptr[span.first];
			sparse_tensor_struct<T, index_t> B(res_dims, res_nnz);
			const size_t start = rowptr[span.first];
			const size_t end = rowptr[span.second];
			for (size_t i = span.first; i < span.second; i++) {
				B.rowptr[i - span.first + 1] = rowptr[i + 1] - start;
			}

			auto method = [&](size_t ss, size_t ee) {
				std::copy(colptr + ss * (rank - 1), colptr + ee * (rank - 1), B.colptr + (ss - start) * (rank - 1));
				std::copy(valptr + ss, valptr + ee, B.valptr + (ss - start));
				};

			if (pool == nullptr) {
				method(start, end);
			}
			else {
				pool->detach_blocks(start, end, method);
				pool->wait();
			}
			return B;
		}

		// take a span of elements
		// sparse_tensor_struct.take(levelspec, {start, end}) returns a sparse_tensor_struct whose elements have the levelspec-th index in range [start, end)
		// elements in the resulting sparse_tensor_struct have their levelspec-th indices reindexed in [0, end - start)
		sparse_tensor_struct<T, index_t> take(const size_t levelspec, const interval_t& span, thread_pool* pool = nullptr) const {
			if (levelspec == 0)
				return take(span, pool);

			else if (levelspec < rank) {
				if (span.second > dims[levelspec])
					throw std::out_of_range("sparse_tensor_struct.take: [start, end) out of [0, dims[levelspec]).");
				if (span.first > span.second)
					throw std::invalid_argument("sparse_tensor_struct.take: invalid span.");
				if (span.first < 0 || span.second < 0)
					throw std::invalid_argument("sparse_tensor_struct.take: expect non-negative indices.");

				std::vector<size_t> res_dims = dims;
				res_dims[levelspec] = span.second - span.first;
				std::vector<size_t> res_row_nnz(res_dims[0], 0);

				if (pool == nullptr) {
					// count nnz
					auto count_row_nnz = [&](size_t i, size_t j) {
						auto tmpptr = colptr + j * (rank - 1);
						if (tmpptr[levelspec - 1] >= span.first && tmpptr[levelspec - 1] < span.second) {
							res_row_nnz[i]++;
						}
					};
					traverse(0, dims[0], count_row_nnz, [](size_t) {});
					// construct result sparse tensor
					size_t res_nnz = std::accumulate(res_row_nnz.begin(), res_row_nnz.end(), (size_t)0);
					sparse_tensor_struct<T, index_t> B(res_dims, res_nnz);
					// set rowptr
					B.rowptr[0] = 0;
					for (size_t i = 0; i < res_dims[0]; i++) {
						B.rowptr[i + 1] = B.rowptr[i] + res_row_nnz[i];
					}
					// set colptr and valptr
					// row init: evaluate starting index in B
					auto row_init_indexing = [&](size_t i) -> size_t {
						return B.rowptr[i];
					};
					// copy entries
					auto copy_entry = [&](size_t i, size_t j, size_t& res_index) {
						auto tmpptr = colptr + j * (rank - 1);
						if (tmpptr[levelspec - 1] >= span.first && tmpptr[levelspec - 1] < span.second) {
							std::copy(tmpptr, tmpptr + rank - 1, B.colptr + res_index * (rank - 1));
							B.colptr[res_index * (rank - 1) + levelspec - 1] -= span.first;
							B.valptr[res_index] = valptr[j];
							res_index++;
						}
					};
					traverse(0, dims[0], copy_entry, row_init_indexing);
					return B;
				}
				else {
					auto nthread = pool->get_thread_count();
					// count nnz
					std::vector<std::vector<size_t>> res_row_block_nnz(res_dims[0]);
					// row init: initialize block nnz storage
					auto row_init_block_nnz = [&](size_t i, BS::blocks<size_t> blks) {
						res_row_block_nnz[i] = std::vector<size_t>(blks.get_num_blocks(), 0);
					};
					// count nnz in each block
					auto count_row_block_nnz = [&](size_t i, size_t j, size_t blk) {
						auto tmpptr = colptr + j * (rank - 1);
						if (tmpptr[levelspec - 1] >= span.first && tmpptr[levelspec - 1] < span.second)
							res_row_block_nnz[i][blk]++;
					};
					std::vector<BS::blocks<size_t>> row_blocks = traverse_setup_blocks(0, dims[0], count_row_block_nnz, row_init_block_nnz, [](size_t, size_t) {}, pool);
					// sum up block nnz
					for (size_t i = 0; i < dims[0]; i++) {
						res_row_nnz[i] = std::accumulate(res_row_block_nnz[i].begin(), res_row_block_nnz[i].end(), (size_t)0);
					}
					size_t res_nnz = std::accumulate(res_row_nnz.begin(), res_row_nnz.end(), (size_t)0);
					sparse_tensor_struct<T, index_t> B(res_dims, res_nnz);
					// set rowptr
					B.rowptr[0] = 0;
					for (size_t i = 0; i < res_dims[0]; i++) {
						B.rowptr[i + 1] = B.rowptr[i] + res_row_nnz[i];
					}
					// set colptr and valptr
					// row init: evaluate block offsets
					auto row_init_block_offset = [&](size_t i, BS::blocks<size_t> blks) -> std::vector<size_t> {
						size_t n_blocks = blks.get_num_blocks();
						std::vector<size_t> block_offset(n_blocks, 0);
						for (size_t j = 0; j < n_blocks - 1; j++) {
							block_offset[j + 1] = block_offset[j] + res_row_block_nnz[i][j];
						}
						return block_offset;
					};
					// block init: evaluate starting index in B
					auto block_init_indexing = [&](size_t i, size_t blk, const std::vector<size_t>& block_offset) -> size_t {
						return  B.rowptr[i] + block_offset[blk];
					};
					// copy entries
					auto copy_entry = [&](size_t i, size_t j, size_t blk, const std::vector<size_t>& block_offset, size_t& res_index) {
						auto tmpptr = colptr + j * (rank - 1);
						if (tmpptr[levelspec - 1] >= span.first && tmpptr[levelspec - 1] < span.second) {
							std::copy(tmpptr, tmpptr + rank - 1, B.colptr + res_index * (rank - 1));
							B.colptr[res_index * (rank - 1) + levelspec - 1] -= span.first;
							B.valptr[res_index] = valptr[j];
							res_index++;
						}
					};
					traverse_using_blocks(0, dims[0], copy_entry, row_init_block_offset, block_init_indexing, row_blocks, pool);
					return B;
				}
			}
			else {
				throw std::invalid_argument("sparse_tensor_struct.take: levelspec out of rank.");
			}
		}

		// extract a (rank-1) tensor by fixing the first index
		sparse_tensor_struct<T, index_t> extract(const index_t index, thread_pool* pool = nullptr) const {
			// Note: require sorted/perm to ensure correctness
			if (index < 0)
				throw std::invalid_argument("sparse_tensor_struct.extract: expect non-negative indices.");
			if (index >= static_cast<index_t>(dims[0]))
				throw std::out_of_range("sparse_tensor_struct.extract: index out of range.");

			std::vector<size_t> res_dims(rank - 1);
			for (size_t i = 0; i < rank - 1; i++)
				res_dims[i] = dims[i + 1];
			const size_t res_nnz = row_nnz(index);
			sparse_tensor_struct<T, index_t> B(res_dims, res_nnz);
			B.rowptr[0] = 0;
			if (pool == nullptr) {
				for (size_t j = rowptr[index]; j < rowptr[index + 1]; j++) {
					auto tmpptr = colptr + j * (rank - 1);
					B.rowptr[tmpptr[0] + 1]++;
					std::copy(tmpptr + 1, tmpptr + (rank - 1), B.colptr + (j - rowptr[index]) * (rank - 2));
				}
				for (size_t i = 0; i < res_dims[0]; i++) {
					B.rowptr[i + 1] += B.rowptr[i];
				}
				std::copy(valptr + rowptr[index], valptr + rowptr[index + 1], B.valptr);
				// if not sorted, we need to permute the entries
				if (!check_sorted()) {
					std::vector<size_t> perm = perm_init(res_nnz);
					std::sort(std::execution::par, perm.begin(), perm.end(), [&](size_t a, size_t b) {
						auto ptra = colptr + (rowptr[index] + a) * (rank - 1);
						auto ptrb = colptr + (rowptr[index] + b) * (rank - 1);
						return lexico_compare(ptra, ptrb, rank - 1) < 0;
						});
					permute(perm, B.colptr, rank - 2);
					permute(perm, B.valptr);
				}
			}
			else {
				auto nthread = pool->get_thread_count();
				std::vector<std::vector<size_t>> row_block_nnz(res_dims[0], std::vector<size_t>(nthread, 0)); 
				if (check_sorted()) {
					pool->detach_blocks(rowptr[index], rowptr[index + 1], [&](size_t ss, size_t ee) {
						for (size_t j = ss; j < ee; j++) {
							auto tmpptr = colptr + j * (rank - 1);
							auto id = SparseRREF::thread_id();
							row_block_nnz[tmpptr[0]][id]++;
							std::copy(tmpptr + 1, tmpptr + (rank - 1), B.colptr + (j - rowptr[index]) * (rank - 2));
						}
						std::copy(valptr + ss, valptr + ee, B.valptr + (ss - rowptr[index]));
						});
					pool->wait();
					// set rowptr
					for (size_t i = 0; i < res_dims[0]; i++) {
						B.rowptr[i + 1] = B.rowptr[i] + std::accumulate(row_block_nnz[i].begin(), row_block_nnz[i].end(), (size_t)0);
					}
				}
				else {
					std::vector<size_t> perm = perm_init(res_nnz);
					std::sort(std::execution::par, perm.begin(), perm.end(), [&](size_t a, size_t b) {
						auto ptra = colptr + (rowptr[index] + a) * (rank - 1);
						auto ptrb = colptr + (rowptr[index] + b) * (rank - 1);
						return lexico_compare(ptra, ptrb, rank - 1) < 0;
						});
					pool->detach_blocks(0, res_nnz, [&](size_t ss, size_t ee) {
						for (size_t j = ss; j < ee; j++) {
							auto oldptr = colptr + (rowptr[index] + perm[j]) * (rank - 1);
							auto newptr = B.colptr + j * (rank - 2);
							auto id = SparseRREF::thread_id();
							row_block_nnz[oldptr[0]][id]++;
							std::copy(oldptr + 1, oldptr + (rank - 1), newptr);
							B.valptr[j] = valptr[rowptr[index] + perm[j]];
						}
						});
					pool->wait();
					// set rowptr
					for (size_t i = 0; i < res_dims[0]; i++) {
						B.rowptr[i + 1] = B.rowptr[i] + std::accumulate(row_block_nnz[i].begin(), row_block_nnz[i].end(), (size_t)0);
					}
				}
			}
			return B;
		}

		// extract a (rank-1) tensor by fixing the levelspec-th index
		sparse_tensor_struct<T, index_t> extract(const size_t levelspec, const index_t index, thread_pool* pool = nullptr) const {
			// Note: require sorted/perm to ensure correctness if levelspec == 0
			if (levelspec == 0)
				return extract(index, pool);
			if (levelspec >= rank)
				throw std::invalid_argument("sparse_tensor_struct.extract: levelspec out of rank.");
			if (index < 0)
				throw std::invalid_argument("sparse_tensor_struct.extract: expect non-negative indices.");
			if (index >= static_cast<index_t>(dims[levelspec]))
				throw std::out_of_range("sparse_tensor_struct.extract: index out of range.");

			std::vector<size_t> res_dims(rank - 1);
			for (size_t i = 0; i < rank - 1; i++) {
				if (i < levelspec)
					res_dims[i] = dims[i];
				else
					res_dims[i] = dims[i + 1];
			}
			std::vector<size_t> res_row_nnz(res_dims[0], 0);

			if (pool == nullptr) {
				// count nnz
				auto count_row_nnz = [&](size_t i, size_t j) {
					auto tmpptr = colptr + j * (rank - 1);
					if (tmpptr[levelspec - 1] == index) {
						res_row_nnz[i]++;
					}
				};
				traverse(0, dims[0], count_row_nnz, [](size_t) {});
				size_t res_nnz = std::accumulate(res_row_nnz.begin(), res_row_nnz.end(), (size_t)0);
				sparse_tensor_struct<T, index_t> B(res_dims, res_nnz);
				// set rowptr
				B.rowptr[0] = 0;
				for (size_t i = 0; i < res_dims[0]; i++) {
					B.rowptr[i + 1] = B.rowptr[i] + res_row_nnz[i];
				}
				// set colptr and valptr
				// row init: evaluate starting index in B
				auto row_init_indexing = [&](size_t i) -> size_t {
					return B.rowptr[i];
				};
				// copy entries
				auto copy_entry = [&](size_t i, size_t j, size_t& res_index) {
					auto tmpptr = colptr + j * (rank - 1);
					if (tmpptr[levelspec - 1] == index) {
						std::copy(tmpptr, tmpptr + levelspec - 1, B.colptr + res_index * (rank - 2));
						std::copy(tmpptr + levelspec, tmpptr + (rank - 1), B.colptr + res_index * (rank - 2) + levelspec - 1);
						B.valptr[res_index] = valptr[j];
						res_index++;
					}
				};
				traverse(0, dims[0], copy_entry, row_init_indexing);
				return B;
			}
			else {
				auto nthread = pool->get_thread_count();
				// count nnz
				std::vector<std::vector<size_t>> res_row_block_nnz(res_dims[0]);
				auto row_init_block_nnz = [&](size_t i, BS::blocks<size_t> blks) {
					res_row_block_nnz[i] = std::vector<size_t>(blks.get_num_blocks(), 0);
				};
				auto count_row_block_nnz = [&](size_t i, size_t j, size_t blk) {
					auto tmpptr = colptr + j * (rank - 1);
					if (tmpptr[levelspec - 1] == index)
						res_row_block_nnz[i][blk]++;
				};
				std::vector<BS::blocks<size_t>> row_blocks = traverse_setup_blocks(0, dims[0], count_row_block_nnz, row_init_block_nnz, [](size_t, size_t) {}, pool);
				// sum up block nnz
				for (size_t i = 0; i < dims[0]; i++) {
					res_row_nnz[i] = std::accumulate(res_row_block_nnz[i].begin(), res_row_block_nnz[i].end(), (size_t)0);
				}
				size_t res_nnz = std::accumulate(res_row_nnz.begin(), res_row_nnz.end(), (size_t)0);
				sparse_tensor_struct<T, index_t> B(res_dims, res_nnz);
				// set rowptr
				B.rowptr[0] = 0;
				for (size_t i = 0; i < res_dims[0]; i++) {
					B.rowptr[i + 1] = B.rowptr[i] + res_row_nnz[i];
				}
				// set colptr and valptr
				auto row_init_block_offset = [&](size_t i, BS::blocks<size_t> blks) -> std::vector<size_t> {
					size_t n_blocks = blks.get_num_blocks();
					std::vector<size_t> block_offset(n_blocks, 0);
					for (size_t j = 0; j < n_blocks - 1; j++) {
						block_offset[j + 1] = block_offset[j] + res_row_block_nnz[i][j];
					}
					return block_offset;
				};
				auto block_init_indexing = [&](size_t i, size_t blk, const std::vector<size_t>& block_offset) -> size_t {
					return B.rowptr[i] + block_offset[blk];
				};
				auto copy_entry = [&](size_t i, size_t j, size_t blk, const std::vector<size_t>& block_offset, size_t& res_index) {
					auto tmpptr = colptr + j * (rank - 1);
					if (tmpptr[levelspec - 1] == index) {
						std::copy(tmpptr, tmpptr + levelspec - 1, B.colptr + res_index * (rank - 2));
						std::copy(tmpptr + levelspec, tmpptr + (rank - 1), B.colptr + res_index * (rank - 2) + levelspec - 1);
						B.valptr[res_index] = valptr[j];
						res_index++;
					}
				};
				// use the predefined blocks
				traverse_using_blocks(0, dims[0], copy_entry, row_init_block_offset, block_init_indexing, row_blocks, pool);
				return B;
			}
		}

		/* extract certain part from tensor, similar to Part in Mathematica 
		   part specification for each level:
		   - `index_t`: extract that index
		   - `std::pair<index_t, index_t>`: take the span `[first, second)`
		   Note: the result must be itself a sparse tensor_struct, i.e., has rank >= 2.
		*/
		sparse_tensor_struct<T, index_t> part(const std::vector<std::variant<index_t, interval_t>>& partspec, thread_pool* pool = nullptr) const {
			// Note: require sorted/perm to ensure correctness if partspec[0] is index_t
			if (partspec.size() != rank) {
				throw std::invalid_argument("sparse_tensor_struct.part: partspec size not equal to rank.");
			}

			// process partspec to get result dimensions
			std::vector<size_t> res_dims;
			std::vector<uint8_t> is_span(rank, 0); // record the partspec type at each level
			for (size_t i = 0; i < rank; i++) {
				if (std::holds_alternative<index_t>(partspec[i])) {
					index_t idx = std::get<index_t>(partspec[i]);
					if (idx < 0 || idx >= static_cast<index_t>(dims[i]))
						throw std::out_of_range("sparse_tensor_struct.part: index out of range.");
					is_span[i] = 0;
				}
				else {
					auto& span = std::get<interval_t>(partspec[i]);
					if (span.first < 0 || span.second < 0)
						throw std::invalid_argument("sparse_tensor_struct.part: expect non-negative indices.");
					if (span.first > span.second)
						throw std::invalid_argument("sparse_tensor_struct.part: invalid span.");
					if (span.second > static_cast<index_t>(dims[i]))
						throw std::out_of_range("sparse_tensor_struct.part: [start, end) out of [0, dims[i]).");
					if (span.first == 0 && span.second == dims[i])
						is_span[i] = 2; // full span
					else
						is_span[i] = 1; // partial span
					res_dims.push_back(span.second - span.first);
				}
			}
			// check result rank
			if (res_dims.size() < 2)
				throw std::invalid_argument("sparse_tensor_struct.part: result tensor rank < 2.");

			// Now introduce some helper variables and functions

			// levels which need to check (index or partial span), element: [level, type]
			std::vector<std::pair<size_t, uint8_t>> coord_check_levels;
			for (size_t i = 1; i < rank; i++) {
				if (is_span[i] != 2) {
					coord_check_levels.push_back(std::make_pair(i, is_span[i]));
				}
			}
			// check if an coordinate is in the partspec
			auto check_coordinate = [&](const_index_p coord) {
				for (const auto& [level, type] : coord_check_levels) {
					if (type == 0) {
						const index_t idx = std::get<index_t>(partspec[level]);
						if (coord[level - 1] != idx)
							return false;
					}
					else if (type == 1) {
						const auto& span = std::get<interval_t>(partspec[level]);
						if (coord[level - 1] < span.first || coord[level - 1] >= span.second)
							return false;
					}
				}
				return true;
			};

			// levels which need to copy (span), element: [[old level, new level - 1], reindex offset] (note that new level - 1 == its position in new_coord)
			std::vector<std::pair<std::pair<size_t, size_t>, index_t>> coord_copy_levels;
			for (size_t i = 1; i < rank; i++) {
				if (is_span[i] != 0) {
					coord_copy_levels.push_back(std::make_pair(std::make_pair(i, coord_copy_levels.size()), std::get<interval_t>(partspec[i]).first));
				}
			}
			// used when is_span[0] == 0, old colptr[new_row_level] becomes the row level (therefore chopped in new colptr)
			const size_t new_row_level = coord_copy_levels[0].first.first - 1;
			const index_t new_row_reindex = coord_copy_levels[0].second;
			// when is_span[0] == 0, old colptr[new_row_level] should not be copied to new colptr
			if (is_span[0] == 0) {
				coord_copy_levels.erase(coord_copy_levels.begin());
				// re-index new levels
				for (auto& [old_new_level, offset] : coord_copy_levels) {
					old_new_level.second -= 1;
				}
			}
			// copy colptr from old_coord to new_coord
			auto copy_colptr = [&](const_index_p old_coord, index_p new_coord) {
				for (const auto& [old_new_level, reindex] : coord_copy_levels) {
					auto& [old_level, new_pos] = old_new_level;
					new_coord[new_pos] = old_coord[old_level - 1] - reindex;
				}
			};

			// interval representation of rowspec
			interval_t row_span;
			if (is_span[0] == 0) {
				index_t idx = std::get<index_t>(partspec[0]);
				row_span = std::make_pair(idx, idx + 1);
			}
			else {
				row_span = std::get<interval_t>(partspec[0]);
			}
			// rowspec index, only used when is_span[0] == 0
			const size_t rowspec_idx = (is_span[0] == 0) ? std::get<index_t>(partspec[0]) : row_span.first;

			const bool sorted = check_sorted();

			// main processing
			sparse_tensor_struct<T, index_t> B(res_dims);
			std::vector<size_t> res_row_nnz(res_dims[0], 0);

			auto part_impl = [&]<bool SingleThread, bool RowspecIsSpan, bool Sorted>() {
				// get new row index
				auto new_row_index = [&](size_t i, const_index_p tmpptr) -> size_t {
					if constexpr (!RowspecIsSpan)
						return tmpptr[new_row_level] - new_row_reindex;
					// else
					return i - row_span.first;
				};

				// prepare perm if necessary
				std::vector<size_t> perm;
				if constexpr (!Sorted && !RowspecIsSpan) {
					perm = perm_init(rowptr[rowspec_idx + 1] - rowptr[rowspec_idx]);
					std::sort(std::execution::par, perm.begin(), perm.end(), [&](size_t a, size_t b) {
						auto ptra = colptr + (rowptr[rowspec_idx] + a) * (rank - 1);
						auto ptrb = colptr + (rowptr[rowspec_idx] + b) * (rank - 1);
						return lexico_compare(ptra, ptrb, rank - 1) < 0;
						});
				}
				// position of the sorted j-th element
				auto sorted_jth = [&](size_t j) -> size_t {
					if constexpr (!Sorted && !RowspecIsSpan)
						return rowptr[rowspec_idx] + perm[j - rowptr[rowspec_idx]];
					// else
					return j;
				};

				if constexpr (SingleThread) {
					// count nnz
					auto count_row_nnz = [&](size_t i, size_t j) {
						auto tmpptr = colptr + j * (rank - 1);
						if (check_coordinate(tmpptr))
							res_row_nnz[new_row_index(i, tmpptr)]++;
					};
					traverse(row_span.first, row_span.second, count_row_nnz, [](size_t) {});
					// reserve nnz for B
					size_t res_nnz = std::accumulate(res_row_nnz.begin(), res_row_nnz.end(), (size_t)0);
					B.reserve(res_nnz);
					// set rowptr
					B.rowptr[0] = 0;
					for (size_t i = 0; i < res_dims[0]; i++) {
						B.rowptr[i + 1] = B.rowptr[i] + res_row_nnz[i];
					}
					// set colptr and valptr
					// due to the complexity of !RowspecIsSpan case, we predefine the row starting index, instead of evaluating them during traversal
					std::vector<size_t> row_starting_index(B.rowptr.begin(), B.rowptr.end() - 1);
					// copy entries
					auto copy_entry = [&](size_t i, size_t j) {
						auto tmpptr = colptr + sorted_jth(j) * (rank - 1);
						if (check_coordinate(tmpptr)) {
							size_t& res_index = row_starting_index[new_row_index(i, tmpptr)];
							auto new_coord = B.colptr + res_index * (B.rank - 1);
							copy_colptr(tmpptr, new_coord);
							B.valptr[res_index] = valptr[sorted_jth(j)];
							res_index++;
						}
					};
					traverse(row_span.first, row_span.second, copy_entry, [](size_t) {});
				}
				else {
					auto nthread = pool->get_thread_count();
					// count nnz
					std::vector<std::vector<size_t>> row_block_nnz(res_dims[0]);
					if constexpr (!RowspecIsSpan)
						row_block_nnz.assign(res_dims[0], std::vector<size_t>(nthread, 0));
					// row init: initialize block nnz storage
					auto row_init_block_nnz = [&](size_t i, BS::blocks<size_t> blks) {
						if constexpr (RowspecIsSpan)
							row_block_nnz[i - row_span.first] = std::vector<size_t>(blks.get_num_blocks(), 0);
					};
					// count nnz in each block
					auto count_row_block_nnz = [&](size_t i, size_t j, size_t blk) {
						auto tmpptr = colptr + sorted_jth(j) * (rank - 1);
						if (check_coordinate(tmpptr)) {
							row_block_nnz[new_row_index(i, tmpptr)][blk]++;
						}
					};
					std::vector<BS::blocks<size_t>> row_blocks = traverse_setup_blocks(row_span.first, row_span.second, count_row_block_nnz, row_init_block_nnz, [](size_t, size_t) {}, pool);
					// sum up block nnz
					for (size_t i = 0; i < res_dims[0]; i++) {
						res_row_nnz[i] = std::accumulate(row_block_nnz[i].begin(), row_block_nnz[i].end(), (size_t)0);
					}
					size_t res_nnz = std::accumulate(res_row_nnz.begin(), res_row_nnz.end(), (size_t)0);
					B.reserve(res_nnz);
					// set rowptr
					B.rowptr[0] = 0;
					for (size_t i = 0; i < res_dims[0]; i++) {
						B.rowptr[i + 1] = B.rowptr[i] + res_row_nnz[i];
					}
					// set colptr and valptr
					// due to the complexity of !RowspecIsSpan case, we predefine the block offset/starting index, instead of evaluating them during traversal
					std::vector<std::vector<size_t>> row_block_starting_index(res_dims[0], std::vector<size_t>(nthread, 0));
					for (size_t i = 0; i < res_dims[0]; i++) {
						size_t offset = B.rowptr[i];
						if constexpr (RowspecIsSpan) {
							for (size_t blk = 0; blk < row_blocks[i].get_num_blocks(); blk++) {
								row_block_starting_index[i][blk] = offset;
								offset += row_block_nnz[i][blk];
							}
						}
						else {
							for (size_t blk = 0; blk < row_blocks[0].get_num_blocks(); blk++) {
								row_block_starting_index[i][blk] = offset;
								offset += row_block_nnz[i][blk];
							}
						}
					}
					// copy entries
					auto copy_entry = [&](size_t i, size_t j, size_t blk) {
						auto oldptr = colptr + sorted_jth(j) * (rank - 1);
						if (check_coordinate(oldptr)) {
							size_t& res_index = row_block_starting_index[new_row_index(i, oldptr)][blk];
							auto newptr = B.colptr + res_index * (B.rank - 1);
							copy_colptr(oldptr, newptr);
							B.valptr[res_index] = valptr[sorted_jth(j)];
							res_index++;
						}
					};
					// use the predefined blocks
					traverse_using_blocks(row_span.first, row_span.second, copy_entry, [](size_t, BS::blocks<size_t>) {}, [](size_t, size_t) {}, row_blocks, pool);
				}
			};

			if (pool == nullptr) {
				if (is_span[0] != 0) {
					if (sorted)
						part_impl.template operator()<true, true, true>();
					else
						part_impl.template operator()<true, true, false>();
				}
				else {
					if (sorted)
						part_impl.template operator()<true, false, true>();
					else
						part_impl.template operator()<true, false, false>();
				}
			}
			else {
				if (is_span[0] != 0) {
					if (sorted)
						part_impl.template operator()<false, true, true>();
					else
						part_impl.template operator()<false, true, false>();
				}
				else {
					if (sorted)
						part_impl.template operator()<false, false, true>();
					else
						part_impl.template operator()<false, false, false>();
				}
			}
			return B;
		}

		sparse_tensor_struct<T, index_t> transpose(const std::vector<size_t>& perm, thread_pool* pool = nullptr, const bool sort_ind = true) const {
			std::vector<size_t> l(rank);
			std::vector<size_t> lperm(rank);
			std::vector<index_t> old_coord(rank);
			std::vector<index_t> new_coord(rank);
			for (size_t i = 0; i < rank; i++)
				lperm[i] = dims[perm[i]];
			sparse_tensor_struct<T, index_t> B(lperm, nnz());
			for (size_t i = 0; i < dims[0]; i++) {
				for (size_t j = rowptr[i]; j < rowptr[i + 1]; j++) {
					old_coord[0] = i;
					auto tmpptr = colptr + j * (rank - 1);
					for (size_t k = 1; k < rank; k++)
						old_coord[k] = tmpptr[k - 1];
					for (size_t k = 0; k < rank; k++)
						new_coord[k] = old_coord[perm[k]];
					B.push_back(new_coord, valptr[j]);
				}
			}
			if (sort_ind)
				B.sort_indices(pool);
			return B;
		}

		bool check_sorted() const {
			for (size_t i = 0; i < dims[0]; i++) {
				size_t rownnz = rowptr[i + 1] - rowptr[i];
				if (rownnz < 2)
					continue;
				std::vector<size_t> perm = perm_init(rownnz);
				if (std::adjacent_find(std::execution::par, perm.begin(), perm.end(), [&](size_t a, size_t b) {
					auto ptra = colptr + (rowptr[i] + a) * (rank - 1);
					auto ptrb = colptr + (rowptr[i] + b) * (rank - 1);
					return lexico_compare(ptra, ptrb, rank - 1) > 0;
					}) != perm.end()) {
					return false;
				}
			}
			return true;
		}

		// multithread version use more memory and it will compress the tensor
		void sort_indices(thread_pool* pool = nullptr) {
			if (pool == nullptr) {
				for (size_t i = 0; i < dims[0]; i++) {
					size_t rownnz = rowptr[i + 1] - rowptr[i];
					std::vector<size_t> perm(rownnz);
					for (size_t j = 0; j < rownnz; j++)
						perm[j] = j;
					std::sort(std::execution::par, perm.begin(), perm.end(), [&](size_t a, size_t b) {
						auto ptra = colptr + (rowptr[i] + a) * (rank - 1);
						auto ptrb = colptr + (rowptr[i] + b) * (rank - 1);
						return lexico_compare(ptra, ptrb, rank - 1) < 0;
						});

					permute(perm, colptr + rowptr[i] * (rank - 1), rank - 1);
					permute(perm, valptr + rowptr[i]);
				}
				return;
			}

			auto nz = nnz();
			auto n_colptr = s_malloc<index_t>(nz * (rank - 1));
			auto n_valptr = s_malloc<T>(nz);
			for (size_t i = 0; i < nz; i++)
				new (n_valptr + i) T();

			for (size_t i = 0; i < dims[0]; i++) {
				size_t rownnz = rowptr[i + 1] - rowptr[i];
				std::vector<size_t> perm(rownnz);
				for (size_t j = 0; j < rownnz; j++)
					perm[j] = j;
				std::sort(std::execution::par, perm.begin(), perm.end(), [&](size_t a, size_t b) {
					auto ptra = colptr + (rowptr[i] + a) * (rank - 1);
					auto ptrb = colptr + (rowptr[i] + b) * (rank - 1);
					return lexico_compare(ptra, ptrb, rank - 1) < 0;
					});

				pool->detach_loop(0, rownnz, [&](size_t j) {
					auto oldptr = colptr + (rowptr[i] + perm[j]) * (rank - 1);
					auto newptr = n_colptr + (rowptr[i] + j) * (rank - 1);
					std::copy(oldptr, oldptr + (rank - 1), newptr);
					n_valptr[rowptr[i] + j] = valptr[rowptr[i] + perm[j]];
					});
				pool->wait();
			}
			for (size_t i = 0; i < alloc; i++)
				valptr[i].~T();
			s_free(colptr);
			s_free(valptr);
			colptr = n_colptr;
			valptr = n_valptr;
			alloc = nz;
		}
	};

	// define the default sparse tensor
	template <typename T, typename index_t = int, SPARSE_TYPE Type = SPARSE_COO> struct sparse_tensor;

	template <typename T, typename index_t> struct sparse_tensor<T, index_t, SPARSE_CSR> {
		sparse_tensor_struct<T, index_t> data;

		using index_v = std::vector<index_t>;
		using index_p = index_t*;
		using const_index_p = const index_t*;

		void clear() { data.clear(); }

		sparse_tensor() {}
		~sparse_tensor() {}
		sparse_tensor(const std::vector<size_t> l, size_t aoc = 8) : data(l, aoc) {}
		sparse_tensor(const sparse_tensor& l) : data(l.data) {}
		sparse_tensor(sparse_tensor&& l) noexcept : data(std::move(l.data)) {}
		sparse_tensor& operator=(const sparse_tensor& l) { data = l.data; return *this; }
		sparse_tensor& operator=(sparse_tensor&& l) noexcept { data = std::move(l.data); return *this; }

		size_t alloc() const { return data.alloc; }
		size_t rank() const { return data.rank; }
		size_t nnz() const { return data.rowptr[data.dims[0]]; }
		auto& rowptr() const { return data.rowptr; }
		auto& colptr() const { return data.colptr; }
		auto& valptr() const { return data.valptr; }
		auto& dims() const { return data.dims; }
		size_t dim(const size_t i) const { return data.dims[i]; }
		index_p index(const size_t i) { return data.colptr + i * (rank() - 1); }
		const_index_p index(const size_t i) const { return data.colptr + i * (rank() - 1); }
		T& val(const size_t i) { return data.valptr[i]; }
		const T& val(const size_t i) const { return data.valptr[i]; }
		bool check_sorted() const { return data.check_sorted(); }
		void zero() { data.zero(); }
		void insert(const index_v& l, const T& val, bool mode = true) { data.insert(l, val, mode); }
		void push_back(const index_v& l, const T& val) { data.push_back(l, val); }
		void canonicalize() { data.canonicalize(); }
		void sort_indices(thread_pool* pool = nullptr) { data.sort_indices(pool); }
		void reserve(const size_t size) { data.reserve(size); }
		sparse_tensor transpose(const std::vector<size_t>& perm, thread_pool* pool = nullptr, const bool sort_ind = true) const {
			sparse_tensor B;
			B.data = data.transpose(perm, pool, sort_ind);
			return B;
		}

		void change_dims(const std::vector<size_t>& new_dims) { data.change_dims(new_dims); }

		// index vector (row index included at first) of the i-th entry
		index_v index_vector(const size_t i) const {
			index_v result(rank());
			result[0] = data.row_index(i);
			for (size_t j = 1; j < rank(); j++)
				result[j] = index(i)[j - 1];
			return result;
		}

		// take a span of elements
		// sparse_tensor.take(levelspec, {start, end}) returns a sparse_tensor whose elements have the levelspec-th index in range [start, end)
		// elements in the resulting sparse_tensor have their levelspec-th indices reindexed in [0, end - start)
		sparse_tensor take(const size_t levelspec, const std::pair<size_t, size_t>& span, thread_pool* pool = nullptr) const {
			sparse_tensor B;
			B.data = data.take(levelspec, span, pool);
			return B;
		}

		// extract a (rank-1) tensor by fixing the levelspec-th index
		sparse_tensor extract(const size_t levelspec, const size_t index, thread_pool* pool = nullptr) const {
			// Note: require sorted/perm to ensure correctness if levelspec == 0
			sparse_tensor B;
			B.data = data.extract(levelspec, index, pool);
			return B;
		}

		/* extract certain part from tensor, similar to Part in Mathematica 
		   part specification for each level:
		   - `index_t`: extract that index
		   - `std::vector<std::pair<index_t, index_t>>`: take the union of spans `[first, second)`
		   Note: the result must be itself a sparse_tensor, i.e., has rank >= 2.
		*/
		sparse_tensor part(const std::vector<std::variant<index_t, std::pair<index_t, index_t>>>& partspec, thread_pool* pool = nullptr) const {
			// Note: require sorted/perm to ensure correctness if partspec[0] is index_t
			sparse_tensor B;
			B.data = data.part(partspec, pool);
			return B;
		}

		void convert_from_COO(const sparse_tensor<T, index_t, SPARSE_COO>& l, thread_pool* pool = nullptr) {
			// a rank 0 tensor has no CSR form: there is no row index to build the rowptr from, and the
			// data.dims[0] reads below would be out of bounds. The empty matrix is all that CSR can
			// express here, so report the lost entries and return it
			if (l.rank() == 0) {
				if (l.nnz() != 0)
					std::cerr << "Error: sparse_tensor: a rank 0 tensor cannot be converted to CSR" << std::endl;
				data.clear();
				data.init({ 0, 0 });
				return;
			}
			// Note: require sorted/perm to ensure correctness
			data.dims = l.data.dims;
			data.rank = l.data.rank;
			auto nnz = l.nnz();
			if (alloc() < nnz)
				reserve(nnz);
			std::copy(l.data.valptr, l.data.valptr + nnz, data.valptr);
			auto newrank = data.rank - 1;
			for (size_t i = 0; i < newrank; i++)
				data.dims[i] = data.dims[i + 1];
			data.dims.resize(newrank);
			data.rank = newrank;
			// then recompute the rowptr and colptr
			// first compute nnz for each row
			data.rowptr.resize(data.dims[0] + 1, 0);
			bool l_sorted = l.check_sorted();
			if (l_sorted || pool == nullptr) {
				for (size_t i = 0; i < nnz; i++) {
					auto oldptr = l.data.colptr + i * newrank;
					auto nowptr = data.colptr + i * (newrank - 1);
					data.rowptr[oldptr[0] + 1]++;
					std::copy(oldptr + 1, oldptr + newrank, nowptr);
				}
				for (size_t i = 0; i < data.dims[0]; i++)
					data.rowptr[i + 1] += data.rowptr[i];
			}
			if (!l_sorted) {
				std::vector<size_t> perm = perm_init(nnz);
				std::sort(std::execution::par, perm.begin(), perm.end(), [&](size_t a, size_t b) {
					auto ptra = l.data.colptr + a * newrank;
					auto ptrb = l.data.colptr + b * newrank;
					return lexico_compare(ptra, ptrb, newrank) < 0;
				});
				if (pool == nullptr) {
					permute(perm, data.colptr, newrank - 1);
					permute(perm, data.valptr);
				}
				else {
					auto nthread = pool->get_thread_count();
					std::vector<std::vector<size_t>> row_block_nnz(data.dims[0] + 1, std::vector<size_t>(nthread, 0));
					pool->detach_blocks(0, nnz, [&](size_t ss, size_t ee) {
						for (size_t j = ss; j < ee; j++) {
							auto oldptr = l.data.colptr + perm[j] * newrank;
							auto newptr = data.colptr + j * (newrank - 1);
							auto id = SparseRREF::thread_id();
							row_block_nnz[oldptr[0]][id]++;
							std::copy(oldptr + 1, oldptr + newrank, newptr);
							data.valptr[j] = l.data.valptr[perm[j]];
						}
					});
					pool->wait();
					for (size_t i = 0; i < data.dims[0]; i++) {
						data.rowptr[i + 1] = data.rowptr[i] + std::accumulate(row_block_nnz[i].begin(), row_block_nnz[i].end(), (size_t)0);
					}
				}
			}
		}

		void move_from_COO(sparse_tensor<T, index_t, SPARSE_COO>&& l, thread_pool* pool = nullptr) noexcept {
			// a rank 0 tensor has no CSR form: there is no row index to build the rowptr from, and the
			// data.dims[0] reads below would be out of bounds. The empty matrix is all that CSR can
			// express here, so report the lost entries and return it
			if (l.rank() == 0) {
				if (l.nnz() != 0)
					std::cerr << "Error: sparse_tensor: a rank 0 tensor cannot be converted to CSR" << std::endl;
				data.clear();
				data.init({ 0, 0 });
				return;
			}
			// Note: require sorted/perm to ensure correctness
			// use move to avoid memory problems, then no need to mention l
			data = std::move(l.data);
			// ensure sorted
			if (!data.check_sorted())
				data.sort_indices(pool);
			// remove the first dimension
			auto newrank = data.rank - 1;
			for (size_t i = 0; i < newrank; i++)
				data.dims[i] = data.dims[i + 1];
			data.dims.resize(newrank);
			// then recompute the rowptr and colptr
			// first compute nnz for each row
			std::vector<size_t> rowptr(data.dims[0] + 1, 0);
			auto nnz = data.rowptr[1];
			for (size_t i = 0; i < nnz; i++) {
				auto oldptr = data.colptr + i * newrank;
				auto nowptr = data.colptr + i * (newrank - 1);
				rowptr[oldptr[0] + 1]++;
				for (size_t j = 0; j < newrank - 1; j++)
					nowptr[j] = oldptr[j + 1];
			}
			for (size_t i = 0; i < data.dims[0]; i++)
				rowptr[i + 1] += rowptr[i];
			data.rowptr = rowptr;
			data.rank = newrank;
			// colptr only holds nnz * (rank - 1) indices now, so alloc must not claim more than that:
			// a later copy reads exactly alloc * (rank - 1) of them
			data.reserve(nnz);
		}

		// constructor from COO
		sparse_tensor(const sparse_tensor<T, index_t, SPARSE_COO>& l, thread_pool* pool = nullptr) { convert_from_COO(l, pool); }
		sparse_tensor& operator=(const sparse_tensor<T, index_t, SPARSE_COO>& l) {
			// Note: require sorted/perm to ensure correctness
			convert_from_COO(l);
			return *this;
		}

		// suppose that COO tensor is sorted
		sparse_tensor(sparse_tensor<T, index_t, SPARSE_COO>&& l, thread_pool* pool = nullptr) noexcept {
			// Note: require sorted/perm to ensure correctness
			move_from_COO(std::move(l), pool);
		}

		sparse_tensor& operator=(sparse_tensor<T, index_t, SPARSE_COO>&& l) noexcept {
			// Note: require sorted/perm to ensure correctness
			move_from_COO(std::move(l));
			return *this;
		}

		// convert a sparse matrix to a sparse tensor
		void convert_from_sparse_mat(const sparse_mat<T, index_t>& mat, thread_pool* pool = nullptr, const bool sort_ind = true) {
			data.dims = { mat.nrow, mat.ncol };
			data.rank = 2;

			auto nnz = mat.nnz();
			if (alloc() < nnz)
				reserve(nnz);

			// compute the rowptr
			data.rowptr.resize(mat.nrow + 1, 0);
			for (size_t i = 0; i < mat.nrow; i++) {
				data.rowptr[i + 1] = data.rowptr[i] + mat[i].nnz();
			}
			// copy the values and column indices
			if (pool == nullptr) {
				for (size_t i = 0; i < mat.nrow; i++) {
					std::copy(mat[i].indices, mat[i].indices + mat[i].nnz(), data.colptr + data.rowptr[i]);
					std::copy(mat[i].entries, mat[i].entries + mat[i].nnz(), data.valptr + data.rowptr[i]);
				}
			}
			else {
				pool->detach_loop(0, mat.nrow, [&](size_t i) {
					std::copy(mat[i].indices, mat[i].indices + mat[i].nnz(), data.colptr + data.rowptr[i]);
					std::copy(mat[i].entries, mat[i].entries + mat[i].nnz(), data.valptr + data.rowptr[i]);
					});
				pool->wait();
			}
			if (sort_ind && !check_sorted())
				sort_indices(pool);
		}

		sparse_mat<T, index_t> to_sparse_mat(thread_pool* pool = nullptr) const {
			if (rank() != 2) {
				std::cerr << "sparse_tensor.to_sparse_mat: rank must be 2" << std::endl;
				return sparse_mat<T, index_t>();
			}

			sparse_mat<T, index_t> mat(data.dims[0], data.dims[1]);
			if (pool == nullptr) {
				for (size_t i = 0; i < data.dims[0]; i++) {
					auto nz = data.rowptr[i + 1] - data.rowptr[i];
					mat[i].reserve(nz);
					mat[i].resize(nz);
					std::copy(data.colptr + data.rowptr[i], data.colptr + data.rowptr[i + 1], mat[i].indices);
					std::copy(data.valptr + data.rowptr[i], data.valptr + data.rowptr[i + 1], mat[i].entries);
				}
			}
			else {
				pool->detach_loop(0, data.dims[0], [&](size_t i) {
					auto nz = data.rowptr[i + 1] - data.rowptr[i];
					mat[i].reserve(nz);
					mat[i].resize(nz);
					std::copy(data.colptr + data.rowptr[i], data.colptr + data.rowptr[i + 1], mat[i].indices);
					std::copy(data.valptr + data.rowptr[i], data.valptr + data.rowptr[i + 1], mat[i].entries);
					});
				pool->wait();
			}
			return mat;
		}

		sparse_tensor(const sparse_mat<T, index_t>& mat, thread_pool* pool = nullptr) {
			convert_from_sparse_mat(mat, pool);
		}

		sparse_tensor& operator=(const sparse_mat<T, index_t>& mat) {
			convert_from_sparse_mat(mat);
			return *this;
		}

		// only for test
		void print_test() {
			for (size_t i = 0; i < data.dims[0]; i++) {
				for (size_t j = data.rowptr[i]; j < data.rowptr[i + 1]; j++) {
					std::cout << i << " ";
					for (size_t k = 0; k < data.rank - 1; k++)
						std::cout << (size_t)data.colptr[j * (data.rank - 1) + k] << " ";
					std::cout << " : " << data.valptr[j] << std::endl;
				}
			}
		}
	};

	template <typename index_t, typename T> struct sparse_tensor<T, index_t, SPARSE_COO> {
		sparse_tensor_struct<T, index_t> data;

		using index_v = std::vector<index_t>;
		using index_p = index_t*;
		using const_index_p = const index_t*;

		template <typename S, typename U = S> requires std::convertible_to<U, S>
		std::vector<S> prepend_num(const std::vector<S>& l, U num = 0) const {
			std::vector<S> lp;
			lp.reserve(l.size() + 1);
			lp.push_back(static_cast<S>(num));
			lp.insert(lp.end(), l.begin(), l.end());
			return lp;
		}

		void clear() { data.clear(); }
		void init(const std::vector<size_t>& l, size_t aoc = 8) {
			data.init(prepend_num(l, (size_t)1), aoc);
		}

		// the empty state is a valid rank 0 tensor, so that nnz(), rank(), dims() and
		// check_sorted() stay well defined on a tensor returned by a failed call
		sparse_tensor() {
			data.rank = 1;
			data.dims = { 1 };
			data.rowptr = { 0, 0 };
		}
		~sparse_tensor() {}
		sparse_tensor(const std::vector<size_t>& l, size_t aoc = 8) : data(prepend_num(l, (size_t)1), aoc) {}
		sparse_tensor(const sparse_tensor& l) : data(l.data) {}
		sparse_tensor(sparse_tensor&& l) noexcept : data(std::move(l.data)) {}
		sparse_tensor& operator=(const sparse_tensor& l) { data = l.data; return *this; }
		sparse_tensor& operator=(sparse_tensor&& l) noexcept { data = std::move(l.data); return *this; }

		// for the i-th column, return the indices
		index_p index(size_t i) const { return data.colptr + i * rank(); }
		T& val(size_t i) const { return data.valptr[i]; }

		index_v index_vector(size_t i) const {
			index_v result(rank());
			for (size_t j = 0; j < rank(); j++)
				result[j] = index(i)[j];
			return result;
		}

		inline size_t alloc() const { return data.alloc; }
		inline size_t nnz() const { return data.rowptr[1]; }
		inline size_t rank() const { return data.rank - 1; }
		inline std::vector<size_t> dims() const {
			std::vector<size_t> result(data.dims.begin() + 1, data.dims.end());
			return result;
		}
		inline size_t dim(size_t i) const { return data.dims[i + 1]; }
		inline bool check_sorted() const { return data.check_sorted(); }
		inline void zero() { data.zero(); }
		inline void reserve(size_t size) { data.reserve(size); }
		inline void resize(size_t new_nnz) {
			if (new_nnz > alloc())
				reserve(new_nnz);
			data.rowptr[1] = new_nnz;
		}

		// we assume that the tensor is sorted
		inline std::vector<size_t> rowptr() const {
			std::vector<size_t> result(dim(0) + 1);
			result[0] = 0;
			for (size_t i = 0; i < nnz(); i++) {
				result[index(i)[0] + 1]++;
			}
			for (size_t i = 0; i < dim(0); i++)
				result[i + 1] += result[i];
			return result;
		}

		// change the dimensions of the tensor
		// it is dangerous, only for internal use
		inline void change_dims(const std::vector<size_t>& new_dims) {
			data.change_dims(prepend_num(new_dims, (size_t)1));
		}

		inline void flatten(const std::vector<std::vector<size_t>>& pos) {
			auto r = rank();
			auto nr = pos.size();
			std::vector<index_t> newindex(nr);
			std::vector<size_t> new_dims(nr);
			auto old_dim = dims();
			auto init_ptr = data.colptr;

			// first compute new dimensions
			for (size_t i = 0; i < nr; i++) {
				new_dims[i] = 1;
				for (auto j : pos[i])
					new_dims[i] *= old_dim[j];
			}
			new_dims = prepend_num(new_dims, (size_t)1);

			for (size_t i = 0; i < nnz(); i++) {
				auto ptr = index(i);
				for (size_t j = 0; j < nr; j++) {
					newindex[j] = 0;
					for (auto k : pos[j])
						newindex[j] = newindex[j] * old_dim[k] + ptr[k];
				}
				for (size_t j = 0; j < nr; j++)
					init_ptr[i * nr + j] = newindex[j];
			}
			data.colptr = s_realloc(data.colptr, nr * nnz());

			// change the dimensions
			data.dims = new_dims;
			data.rank = nr + 1;
		}

		// reshape, for example {2,100} to {2,5,20}
		// TODO: check more examples
		inline void reshape(const std::vector<size_t>& new_dims) {
			auto old_dims = dims();
			index_t* newcolptr = s_malloc<index_t>(nnz() * new_dims.size());
			auto r = rank();

			int_t flatten_index = 0;
			int_t tmp;
			for (size_t i = 0; i < nnz(); i++) {
				auto ptr = index(i);
				flatten_index = 0;
				for (size_t j = 0; j < r; j++) {
					flatten_index *= old_dims[j];
					flatten_index += ptr[j];
				}
				for (auto j = new_dims.size(); j > 0; j--) {
					tmp = flatten_index % new_dims[j - 1];
					flatten_index /= new_dims[j - 1];
					newcolptr[i * new_dims.size() + j - 1] = tmp.to_si();
				}
			}
			s_free(data.colptr);
			data.colptr = newcolptr;
			data.dims = prepend_num(new_dims, (size_t)1);
			data.rank = new_dims.size() + 1;
		}

		inline void insert(const index_v& l, const T& val, bool mode = true) { data.insert(prepend_num(l), val, mode); }
		inline void insert_add(const index_v& l, const T& val) { data.insert_add(prepend_num(l), val); }
		void push_back(const_index_p l, const T& new_val) {
			auto n_nnz = nnz();
			if (n_nnz + 1 > data.alloc)
				reserve((data.alloc + 1) * 2);
			s_copy(index(n_nnz), l, rank());
			val(n_nnz) = new_val;
			data.rowptr[1]++; // increase the nnz
		}
		void push_back(const index_v& l, const T& new_val) { push_back(l.data(), new_val); }
		void pop_back() { if (data.rowptr[1] != 0) data.rowptr[1]--; }
		inline void canonicalize() { data.canonicalize(); }
		inline void sort_indices(thread_pool* pool = nullptr) { data.sort_indices(pool); }
		inline sparse_tensor transpose(const std::vector<size_t>& perm, thread_pool* pool = nullptr, const bool sort_ind = true) const {
			std::vector<size_t> perm_new(perm);
			for (auto& a : perm_new) { a++; }
			perm_new = prepend_num(perm_new, (size_t)0);
			sparse_tensor B;
			B.data = data.transpose(perm_new, pool, sort_ind);
			B.sort_indices(pool);
			return B;
		}

		// take a span of elements
		// sparse_tensor.take(levelspec, {start, end}) returns a sparse_tensor whose elements have the levelspec-th index in range [start, end)
		// elements in the resulting sparse_tensor have their levelspec-th indices reindexed in [0, end - start)
		sparse_tensor take(const size_t levelspec, const std::pair<size_t, size_t>& span, thread_pool* pool = nullptr) const {
			sparse_tensor B;
			B.data = data.take(levelspec + 1, span, pool);
			return B;
		}

		// extract a (rank-1) tensor by fixing the levelspec-th index
		sparse_tensor extract(const size_t levelspec, const size_t index, thread_pool* pool = nullptr) const {
			sparse_tensor B;
			B.data = data.extract(levelspec + 1, index, pool);
			return B;
		}

		/* extract certain part from tensor, similar to Part in Mathematica 
		   part specification for each level:
		   - `index_t`: extract that index
		   - `std::vector<std::pair<index_t, index_t>>`: take the union of spans `[first, second)`
		   Note: the result must be itself a sparse_tensor, i.e., has rank >= 2.
		*/
		sparse_tensor part(const std::vector<std::variant<index_t, std::pair<index_t, index_t>>>& partspec, thread_pool* pool = nullptr) const {
			std::vector<std::variant<index_t, std::pair<index_t, index_t>>> newpartspec;
			newpartspec.push_back(std::pair{0, 1}); // for the first dimension
			newpartspec.insert(newpartspec.end(), partspec.begin(), partspec.end());
			sparse_tensor B;
			B.data = data.part(newpartspec, pool);
			return B;
		}

		sparse_mat<T, index_t> to_sparse_mat(thread_pool* pool = nullptr, const bool sort_ind = true) {
			if (rank() != 2) {
				std::cerr << "sparse_tensor.to_sparse_mat: rank must be 2" << std::endl;
				return sparse_mat<T, index_t>();
			}
			if (sort_ind)
				sort_indices(pool);
			auto r = dim(0);
			auto c = dim(1);
			auto rptr = rowptr();

			sparse_mat<T, index_t> mat(r, c);
			if (pool == nullptr) {
				for (size_t i = 0; i < r; i++) {
					auto nz = rptr[i + 1] - rptr[i];
					mat[i].reserve(nz);
					mat[i].resize(nz);
					std::copy(data.valptr + rptr[i], data.valptr + rptr[i + 1], mat[i].entries);
					// skip the first index, which is the row index
					auto ptr = index(rptr[i]) + 1;
					for (size_t j = 0; j < nz; j++)
						mat[i].indices[j] = ptr[2 * j];
				}
			}
			else {
				pool->detach_loop(0, r, [&](size_t i) {
					auto nz = rptr[i + 1] - rptr[i];
					mat[i].reserve(nz);
					mat[i].resize(nz);
					std::copy(data.valptr + rptr[i], data.valptr + rptr[i + 1], mat[i].entries);
					// skip the first index, which is the row index
					auto ptr = index(rptr[i]) + 1;
					for (size_t j = 0; j < nz; j++)
						mat[i].indices[j] = ptr[2 * j];
					});
				pool->wait();
			}
			return mat;
		}

		// the permutation that orders the entries by the labels at the positions listed in order, or by
		// all the labels in the natural order when order is nullptr
		// the entries are usually already in that order, because the callers hand in tensors that were
		// built sorted or returned by an operation that sorts: verifying it costs one cheap pass, so
		// the identity permutation is returned as soon as the check succeeds
		std::vector<size_t> gen_perm_by(const std::vector<size_t>* order) const {
			// the comparators are written out twice instead of going through a common lambda: they run
			// once per entry of the sort, where an extra branch on order costs more than the pass above
			const auto r = rank();
			const auto nz = nnz();
			bool sorted = true;
			if (order == nullptr) {
				for (size_t i = 1; i < nz && sorted; i++)
					sorted = lexico_compare(index(i - 1), index(i), r) <= 0;
			}
			else {
				for (size_t i = 1; i < nz && sorted; i++)
					sorted = lexico_compare(index(i - 1), index(i), *order) <= 0;
			}

			std::vector<size_t> perm = perm_init(nz);
			if (sorted)
				return perm;

			if (order == nullptr) {
				std::sort(std::execution::par, perm.begin(), perm.end(), [&](size_t a, size_t b) {
					return lexico_compare(index(a), index(b), r) < 0;
					});
			}
			else {
				std::sort(std::execution::par, perm.begin(), perm.end(), [&](size_t a, size_t b) {
					return lexico_compare(index(a), index(b), *order) < 0;
					});
			}
			return perm;
		}

		std::vector<size_t> gen_perm() const { return gen_perm_by(nullptr); }

		std::vector<size_t> gen_perm(const std::vector<size_t>& index_perm) const {
			// index_perm is a permutation of the positions of the index vector, so it must have
			// exactly rank() entries; the caller is expected to check this, we only degrade to the
			// natural order here instead of handing back a perm of the wrong size
			if (index_perm.size() != rank()) {
				std::cerr << "Error: gen_perm: index_perm size is not equal to rank" << std::endl;
				return gen_perm();
			}

			return gen_perm_by(&index_perm);
		}

		void transpose_replace(const std::vector<size_t>& perm, thread_pool* pool = nullptr, const bool sort_ind = true) {
			// perm[i] is the old position of the new i-th index, so perm has to be a permutation of
			// [0, rank()); anything else would read out of bounds below
			std::vector<bool> used(rank(), false);
			bool valid = perm.size() == rank();
			for (size_t i = 0; valid && i < perm.size(); i++) {
				if (perm[i] >= rank() || used[perm[i]])
					valid = false;
				else
					used[perm[i]] = true;
			}
			if (!valid) {
				std::cerr << "Error: transpose_replace: perm must be a permutation of the indices" << std::endl;
				return;
			}

			std::vector<size_t> new_dims(rank() + 1);
			new_dims[0] = data.dims[0];

			for (size_t i = 0; i < rank(); i++)
				new_dims[i + 1] = data.dims[perm[i] + 1];
			data.dims = new_dims;

			auto method = [&](size_t ss, size_t ee) {
				std::vector<size_t> index_new(rank());
				for (size_t i = ss; i < ee; i++) {
					auto ptr = index(i);
					for (size_t j = 0; j < rank(); j++)
						index_new[j] = ptr[perm[j]];
					std::copy(index_new.begin(), index_new.end(), ptr);
				}
				};

			if (pool == nullptr)
				method(0, nnz());
			else {
				pool->detach_blocks(0, nnz(), method);
				pool->wait();
			}
			if (sort_ind)
				sort_indices(pool);
		}

		sparse_tensor<T, index_t, SPARSE_COO> chop(size_t pos, size_t aa) const {
			std::vector<size_t> dims_new = dims();
			dims_new.erase(dims_new.begin() + pos);
			sparse_tensor<T, index_t, SPARSE_COO> result(dims_new);
			index_v index_new;
			index_new.reserve(rank() - 1);
			for (size_t i = 0; i < nnz(); i++) {
				if (index(i)[pos] != aa)
					continue;
				for (size_t j = 0; j < rank(); j++) {
					if (j != pos)
						index_new.push_back(index(i)[j]);
				}
				result.push_back(index_new, val(i));
				index_new.clear();
			}
			return result;
		}

		// constructor from CSR
		sparse_tensor(const sparse_tensor<T, index_t, SPARSE_CSR>& l) {
			data.init(prepend_num(l.dims(), (size_t)1), l.nnz());
			resize(l.nnz());

			auto r = rank();
			auto n_row = dim(0);

			// first copy the data
			s_copy(data.valptr, l.data.valptr, l.nnz());

			// then deal with the indices
			for (size_t i = 0; i < n_row; i++) {
				for (size_t j = l.data.rowptr[i]; j < l.data.rowptr[i + 1]; j++) {
					auto tmp_index = index(j);
					tmp_index[0] = i;
					for (size_t k = 0; k < r - 1; k++)
						tmp_index[k + 1] = l.data.colptr[j * (r - 1) + k];
				}
			}
		}

		sparse_tensor& operator=(const sparse_tensor<T, index_t, SPARSE_CSR>& l) {
			if (alloc() == 0) {
				init(l.dims(), l.nnz());
			}
			else {
				change_dims(l.dims());
				reserve(l.nnz());
			}

			auto r = rank();
			auto n_row = dim(0);

			// first copy the data
			s_copy(data.valptr, l.data.valptr, l.nnz());

			// then deal with the indices
			for (size_t i = 0; i < n_row; i++) {
				for (size_t j = l.data.rowptr[i]; j < l.data.rowptr[i + 1]; j++) {
					auto tmp_index = index(j);
					tmp_index[0] = i;
					for (size_t k = 0; k < r - 1; k++)
						tmp_index[k + 1] = l.data.colptr[j * (r - 1) + k];
				}
			}

			return *this;
		}

		sparse_tensor& operator=(sparse_tensor<T, index_t, SPARSE_CSR>&& l) noexcept {
			data = std::move(l.data);
			if (data.alloc == 0)
				return *this;

			auto r = data.rank;
			auto n_row = data.dims[0];

			// recompute the index
			index_t* newcolptr = s_malloc<index_t>(data.alloc * r);
			auto newcolptr_j = newcolptr;
			auto nowcolptr_j = data.colptr;
			for (size_t i = 0; i < n_row; i++) {
				for (size_t j = data.rowptr[i]; j < data.rowptr[i + 1]; j++) {
					newcolptr_j[0] = i;
					s_copy(newcolptr_j + 1, nowcolptr_j, r - 1);
					newcolptr_j += r;
					nowcolptr_j += r - 1;
				}
			}
			s_free(data.colptr);
			data.colptr = newcolptr;

			data.rowptr = { 0, data.rowptr.back() };
			data.dims = prepend_num(data.dims, (size_t)1);
			data.rank++;

			return *this;
		}

		sparse_tensor(sparse_tensor<T, index_t, SPARSE_CSR>&& l) noexcept {
			data = std::move(l.data);
			if (data.alloc == 0)
				return;

			auto r = data.rank;
			auto n_row = data.dims[0];

			// recompute the index
			index_t* newcolptr = s_malloc<index_t>(data.alloc * r);
			auto newcolptr_j = newcolptr;
			auto nowcolptr_j = data.colptr;
			for (size_t i = 0; i < n_row; i++) {
				for (size_t j = data.rowptr[i]; j < data.rowptr[i + 1]; j++) {
					newcolptr_j[0] = i;
					s_copy(newcolptr_j + 1, nowcolptr_j, r - 1);
					newcolptr_j += r;
					nowcolptr_j += r - 1;
				}
			}
			s_free(data.colptr);
			data.colptr = newcolptr;

			data.rowptr = { 0, data.rowptr.back() };
			data.dims = prepend_num(data.dims, (size_t)1);
			data.rank++;
		}

		sparse_tensor(const sparse_mat<T, index_t>& mat, thread_pool* pool = nullptr) {
			*this = sparse_tensor<T, index_t, SPARSE_CSR>(mat, pool);
		}

		sparse_tensor& operator=(const sparse_mat<T, index_t>& mat) {
			*this = sparse_tensor<T, index_t, SPARSE_CSR>(mat);
			return *this;
		}

		void print_test() {
			for (size_t j = 0; j < data.rowptr[1]; j++) {
				for (size_t k = 0; k < data.rank - 1; k++)
					std::cout << (size_t)(data.colptr[j * (data.rank - 1) + k]) << " ";
				std::cout << " : " << data.valptr[j] << std::endl;
			}
		}
	};

	// some other functions

	// join two sparse matrices
	template <typename T, typename index_t>
	sparse_mat<T, index_t> sparse_mat_join(const sparse_mat<T, index_t>& A, const sparse_mat<T, index_t>& B, thread_pool* pool = nullptr) {
		sparse_mat<T, index_t> res(A.nrow + B.nrow, std::max(A.ncol, B.ncol));

		if (pool == nullptr) {
			std::copy(A.rows.begin(), A.rows.end(), res.rows.begin());
			std::copy(B.rows.begin(), B.rows.end(), res.rows.begin() + A.nrow);
		}
		else {
			pool->detach_loop(0, A.nrow, [&](size_t i) {
				res[i] = A[i];
				});
			pool->wait();
			pool->detach_loop(0, B.nrow, [&](size_t i) {
				res[i + A.nrow] = B[i];
				});
			pool->wait();
		}

		return res;
	}

	template <typename T, typename index_t>
	sparse_mat<T, index_t> sparse_mat_join(sparse_mat<T, index_t>&& A, sparse_mat<T, index_t>&& B) {
		sparse_mat<T, index_t> res = std::move(A);
		res.append(std::move(B));
		res.ncol = std::max(A.ncol, B.ncol);
		return res;
	}

	// split a sparse matrix into two parts
	template <typename T, typename index_t>
	std::pair<sparse_mat<T, index_t>, sparse_mat<T, index_t>> sparse_mat_split(const sparse_mat<T, index_t>& mat, const size_t split_row, thread_pool* pool = nullptr) {
		if (split_row > mat.nrow)
			throw std::out_of_range("sparse_mat_split: split_row out of range");

		sparse_mat<T, index_t> A(split_row, mat.ncol);
		sparse_mat<T, index_t> B(mat.nrow - split_row, mat.ncol);

		if (pool == nullptr) {
			std::copy(mat.rows.begin(), mat.rows.begin() + split_row, A.rows.begin());
			std::copy(mat.rows.begin() + split_row, mat.rows.end(), B.rows.begin());
		}
		else {
			pool->detach_loop(0, split_row, [&](size_t i) {
				A[i] = mat[i];
				});
			pool->wait();
			pool->detach_loop(split_row, mat.nrow, [&](size_t i) {
				B[i - split_row] = mat[i];
				});
			pool->wait();
		}

		return { A, B };
	}

	template <typename T, typename index_t>
	std::pair<sparse_mat<T, index_t>, sparse_mat<T, index_t>> sparse_mat_split(sparse_mat<T, index_t>&& mat, const size_t split_row) {
		if (split_row > mat.nrow)
			throw std::out_of_range("sparse_mat_split: split_row out of range");

		sparse_mat<T, index_t> A(split_row, mat.ncol);
		sparse_mat<T, index_t> B(mat.nrow - split_row, mat.ncol);

		A.rows = std::vector<sparse_vec<T, index_t>>(std::make_move_iterator(mat.rows.begin()),
			std::make_move_iterator(mat.rows.begin() + split_row));
		B.rows = std::vector<sparse_vec<T, index_t>>(std::make_move_iterator(mat.rows.begin() + split_row),
			std::make_move_iterator(mat.rows.end()));

		mat.clear();

		return { A, B };
	}

	// a submatrix view of sparse_mat
	// it only contains a pointer to the original matrix and a list of row indices
	// and it does not own any memory
	// to save memory, if rows[0] > mat.nrow, then it is a full view
	template <typename T, typename index_t>
	struct sparse_mat_subview {
		sparse_mat<T, index_t>* mat_ptr = nullptr;
		std::vector<size_t> rows;

		sparse_mat_subview() = default;
		~sparse_mat_subview() = default;

		sparse_mat_subview(sparse_mat<T, index_t>& mat_) {
			mat_ptr = &mat_;
			rows = { mat_.nrow + 1 }; // full view
		}

		sparse_mat_subview(const sparse_mat<T, index_t>& mat_) {
			mat_ptr = const_cast<sparse_mat<T, index_t>*>(&mat_);
			rows = { mat_.nrow + 1 }; // full view
		}

		sparse_mat_subview(sparse_mat<T, index_t>& mat_, const std::vector<size_t>& rows_) {
			mat_ptr = &mat_;
			rows = rows_;
		}

		sparse_mat_subview(const sparse_mat_subview& l) { mat_ptr = l.mat_ptr; rows = l.rows; }
		sparse_mat_subview(sparse_mat_subview&& l) noexcept { mat_ptr = l.mat_ptr; rows = std::move(l.rows); }
		sparse_mat_subview& operator=(const sparse_mat_subview& l) {
			if (this == &l)
				return *this;
			mat_ptr = l.mat_ptr;
			rows = l.rows;
			return *this;
		}
		sparse_mat_subview& operator=(sparse_mat_subview&& l) noexcept {
			if (this == &l)
				return *this;
			mat_ptr = l.mat_ptr;
			rows = std::move(l.rows);
			return *this;
		}

		size_t nrow() const {
			if (mat_ptr) {
				if (rows[0] > mat_ptr->nrow) // full view
					return mat_ptr->nrow;
				return rows.size();
			}
			return 0;
		}

		size_t ncol() const {
			if (mat_ptr)
				return mat_ptr->ncol;
			return 0;
		}

		size_t nnz() const {
			if (mat_ptr) {
				size_t result = 0;
				if (rows[0] > mat_ptr->nrow) // full view
					return mat_ptr->nnz();
				for (auto r : rows)
					result += (*mat_ptr)[r].nnz();
				return result;
			}
			return 0;
		}

		bool is_full() const {
			if (mat_ptr) {
				return rows[0] > mat_ptr->nrow; // full view
			}
			return false;
		}

		// access the i-th row of the subview, it is dangerous because we do not check the index
		sparse_vec<T, index_t>& operator[](size_t i) {
			if (rows[0] > mat_ptr->nrow) // full view
				return (*mat_ptr)[i];
			return (*mat_ptr)[rows[i]];
		}
		const sparse_vec<T, index_t>& operator[](size_t i) const {
			if (rows[0] > mat_ptr->nrow) // full view
				return (*mat_ptr)[i];
			return (*mat_ptr)[rows[i]];
		}

		const size_t operator()(size_t i) const {
			if (rows[0] > mat_ptr->nrow) // full view
				return i;
			return rows[i];
		}

		sparse_mat<T, index_t>& get_mat() {
			return *mat_ptr;
		}
		const sparse_mat<T, index_t>& get_mat() const {
			return *mat_ptr;
		}

		void traverse(std::function<void(size_t)> func) {
			if (mat_ptr == nullptr)
				return;
			if (rows[0] > mat_ptr->nrow) { // full view
				for (size_t i = 0; i < mat_ptr->nrow; i++)
					func(i);
			}
			else {
				for (auto r : rows)
					func(r);
			}
		}
	};

}

#endif