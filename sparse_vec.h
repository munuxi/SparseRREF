/*
	Copyright (C) 2024-2026 Zhenjie Li (Li, Zhenjie)

	This file is part of SparseRREF. The SparseRREF is free software:
	you can redistribute it and/or modify it under the terms of the MIT
	License.
*/

#ifndef SPARSE_VEC_H
#define SPARSE_VEC_H

#include "scalar.h"
#include "sparse_type.h"

namespace SparseRREF {

	template <typename index_t> using snmod_vec = sparse_vec<ulong, index_t>;
	template <typename index_t> using sfmpq_vec = sparse_vec<rat_t, index_t>;

	template <typename index_t, typename T>
	inline void sparse_vec_rescale(sparse_vec<T, index_t>& vec, const T& scalar, const field_t& F) {
		if (scalar == 1)
			return;
		if constexpr (std::is_same_v<T, ulong>) {
			auto pp = F.get_prime();
			ulong e_pr = n_mulmod_precomp_shoup(scalar, pp);
			for (size_t i = 0; i < vec.nnz(); i++)
				vec.entries[i] = n_mulmod_shoup(scalar, vec.entries[i], e_pr, pp);
		}
		else if constexpr (Flint::IsOneOf<T, int_t, rat_t>) {
			for (size_t i = 0; i < vec.nnz(); i++)
				vec.entries[i] *= scalar;
		}
	}

	template <typename index_t>
	void sparse_vec_rescale(sparse_vec<bool, index_t>& vec, const bool scalar, const field_t& F) {
		if (scalar == false)
			vec.zero();
	}

	// we assume that vec and src are sorted, and the result is also sorted
	template <typename index_t>
	int snmod_vec_add_mul(
		snmod_vec<index_t>& vec, const snmod_vec<index_t>& src,
		const ulong a, const field_t& F) {
		if (src.nnz() == 0)
			return 0;

		auto p = F.mod;

		if (vec.nnz() == 0) {
			vec = src;
			sparse_vec_rescale(vec, a, F);
			return 0;
		}

		ulong na = a;
		ulong na_pr = n_mulmod_precomp_shoup(na, p.n);

		size_t ptr1 = vec.nnz();
		size_t ptr2 = src.nnz();
		const size_t nnz_new = ptr1 + ptr2;
		size_t ptr = nnz_new;

		if (vec.alloc() < ptr)
			vec.reserve(ptr);
		vec.resize(ptr);

		while (ptr1 > 0 && ptr2 > 0) {
			if (vec(ptr1 - 1) == src(ptr2 - 1)) {
				auto entry =
					_nmod_add(vec[ptr1 - 1],
						n_mulmod_shoup(na, src[ptr2 - 1], na_pr, p.n), p);
				if (entry != 0) {
					vec(ptr - 1) = vec(ptr1 - 1);
					vec[ptr - 1] = entry;
					ptr--;
				}
				ptr1--;
				ptr2--;
			}
			else if (vec(ptr1 - 1) < src(ptr2 - 1)) {
				vec(ptr - 1) = src(ptr2 - 1);
				vec[ptr - 1] = n_mulmod_shoup(na, src[ptr2 - 1], na_pr, p.n);
				ptr2--;
				ptr--;
			}
			else {
				vec(ptr - 1) = vec(ptr1 - 1);
				vec[ptr - 1] = vec[ptr1 - 1];
				ptr1--;
				ptr--;
			}
		}
		while (ptr2 > 0) {
			vec(ptr - 1) = src(ptr2 - 1);
			vec[ptr - 1] = n_mulmod_shoup(na, src[ptr2 - 1], na_pr, p.n);
			ptr2--;
			ptr--;
		}

		const size_t res_nnz = nnz_new - ptr;
		const size_t gap = ptr - ptr1;
		if (gap != 0) {
			std::memmove(vec.indices + ptr1, vec.indices + ptr,
				res_nnz * sizeof(index_t));
			std::memmove(vec.entries + ptr1, vec.entries + ptr,
				res_nnz * sizeof(ulong));
		}
		vec.resize(ptr1 + res_nnz);

		if (vec.alloc() > 4 * vec.nnz())
			vec.reserve(2 * vec.nnz());

		return 0;
	}

	template <typename index_t, bool dir>
	int sfmpq_vec_addsub_mul(sfmpq_vec<index_t>& vec, const sfmpq_vec<index_t>& src, const rat_t& a) {
		if (src.nnz() == 0)
			return 0;

		if (vec.nnz() == 0) {
			vec = src;
			sparse_vec_rescale(vec, a, field_t());
			return 0;
		}

		rat_t na, entry;
		if constexpr (dir) {
			na = a;
		}
		else {
			na = -a;
		}

		size_t ptr1 = vec.nnz();
		size_t ptr2 = src.nnz();
		size_t ptr = vec.nnz() + src.nnz();

		if (vec._alloc < ptr)
			vec.reserve(ptr);
		vec.resize(ptr);

		while (ptr1 > 0 && ptr2 > 0) {
			if (vec(ptr1 - 1) == src(ptr2 - 1)) {
				entry = na * src[ptr2 - 1];
				entry += vec[ptr1 - 1];
				if (entry != 0) {
					vec(ptr - 1) = vec(ptr1 - 1);
					vec[ptr - 1] = entry;
					ptr--;
				}
				ptr1--;
				ptr2--;
			}
			else if (vec(ptr1 - 1) < src(ptr2 - 1)) {
				entry = na * src[ptr2 - 1];
				vec(ptr - 1) = src(ptr2 - 1);
				vec[ptr - 1] = entry;
				ptr2--;
				ptr--;
			}
			else {
				vec(ptr - 1) = vec(ptr1 - 1);
				vec[ptr - 1] = vec[ptr1 - 1];
				ptr1--;
				ptr--;
			}
		}
		while (ptr2 > 0) {
			entry = na * src[ptr2 - 1];
			vec(ptr - 1) = src(ptr2 - 1);
			vec[ptr - 1] = entry;
			ptr2--;
			ptr--;
		}

		// if ptr1 > 0, and ptr > 0
		for (size_t i = ptr1; i < ptr; i++) {
			vec[i] = 0;
		}

		vec.canonicalize();
		if (vec._alloc > 4 * vec.nnz())
			vec.reserve(2 * vec.nnz());

		return 0;
	}

	// it is just symmetric difference for bool vectors
	template <typename index_t>
	int sparse_vec_add(sparse_vec<bool, index_t>& vec, sparse_vec<bool, index_t>& src, const field_t& F) {
		if (src.nnz() == 0)
			return 0;

		if (vec.nnz() == 0) {
			vec = src;
			return 0;
		}

		auto a = vec.indices;
		auto a_end = vec.indices + vec.nnz();
		auto b = src.indices;
		auto b_end = src.indices + src.nnz();

		index_t* out = s_malloc<index_t>(vec.nnz() + src.nnz());
		auto out_start = out;
		auto out_alloc = vec.nnz() + src.nnz();

		while (a != a_end && b != b_end) {
			if (*a < *b)
				*out++ = *a++;
			else if (*b < *a)
				*out++ = *b++;
			else {
				++a; ++b;
			}
		}

		// remaining
		while (a != a_end) *out++ = *a++;
		while (b != b_end) *out++ = *b++;

		size_t new_nnz = out - out_start;
		s_free(vec.indices);
		vec.indices = out_start;
		vec._nnz = new_nnz;
		vec._alloc = out_alloc;

		return 0;
	}

	template <typename index_t>
	inline int sparse_vec_add(snmod_vec<index_t>& vec, const snmod_vec<index_t>& src, const field_t& F) {
		return snmod_vec_add_mul(vec, src, 1, F);
	}

	template <typename index_t>
	inline int sparse_vec_sub(snmod_vec<index_t>& vec, const snmod_vec<index_t>& src, const field_t& F) {
		return snmod_vec_add_mul(vec, src, F.get_prime() - 1, F);
	}

	template <typename index_t>
	inline int sparse_vec_add_mul(snmod_vec<index_t>& vec, const snmod_vec<index_t>& src, const ulong a, const field_t& F) {
		return snmod_vec_add_mul(vec, src, a, F);
	}

	template <typename index_t>
	inline int sparse_vec_add_mul(sfmpq_vec<index_t>& vec, const sfmpq_vec<index_t>& src, const rat_t& a, const field_t& F) {
		return sfmpq_vec_addsub_mul<index_t, true>(vec, src, a);
	}

	template <typename index_t>
	inline int sparse_vec_sub_mul(snmod_vec<index_t>& vec, const snmod_vec<index_t>& src, const ulong a, const field_t& F) {
		return snmod_vec_add_mul(vec, src, F.get_prime() - a, F);
	}

	template <typename index_t>
	inline int sparse_vec_sub_mul(sfmpq_vec<index_t>& vec, const sfmpq_vec<index_t>& src, const rat_t& a, const field_t& F) {
		return sfmpq_vec_addsub_mul<index_t, false>(vec, src, a);
	}

	// dot product
	template <typename index_t, typename T>
	T sparse_vec_dot(const sparse_vec<T, index_t>& v1, const sparse_vec<T, index_t>& v2, const field_t& F) {
		if (v1.nnz() == 0 || v2.nnz() == 0) {
			return T(0);
		}
		size_t ptr1 = 0, ptr2 = 0;
		T result = 0;
		auto e1 = v1.indices + v1.nnz();
		auto e2 = v2.indices + v2.nnz();
		while (ptr1 < v1.nnz() && ptr2 < v2.nnz()) {
			if (v1(ptr1) == v2(ptr2)) {
				result = scalar_add(result, scalar_mul(v1[ptr1], v2[ptr2], F), F);
				ptr1++;
				ptr2++;
			}
			else if (v1(ptr1) < v2(ptr2)) {
				ptr1 = std::lower_bound(v1.indices + ptr1, e1, v2(ptr2)) - v1.indices;
			}
			else {
				ptr2 = std::lower_bound(v2.indices + ptr2, e2, v1(ptr1)) - v2.indices;
			}
		}

		return result;
	}

	template <typename index_t>
	bool sparse_vec_dot(const sparse_vec<bool, index_t>& v1, const sparse_vec<bool, index_t>& v2, const field_t& F) {
		if (v1.nnz() == 0 || v2.nnz() == 0) {
			return false;
		}
		size_t ptr1 = 0, ptr2 = 0;
		bool result = false;
		auto e1 = v1.indices + v1.nnz();
		auto e2 = v2.indices + v2.nnz();
		while (ptr1 < v1.nnz() && ptr2 < v2.nnz()) {
			if (v1(ptr1) == v2(ptr2)) {
				// add one in GF(2), 0 to 1 and 1 to 0, so it is !result
				result = !result;
				ptr1++;
				ptr2++;
			}
			else if (v1(ptr1) < v2(ptr2)) {
				ptr1 = std::lower_bound(v1.indices + ptr1, e1, v2(ptr2)) - v1.indices;
			}
			else {
				ptr2 = std::lower_bound(v2.indices + ptr2, e2, v1(ptr1)) - v2.indices;
			}
		}

		return result;
	}

	// we do not check the boundry of v2, so it is not safe to use this function, 
	// be careful
	template <typename index_t, typename T>
	T sparse_vec_dot_dense_vec(const sparse_vec<T, index_t>& v1, const T* v2, const field_t& F) {
		if (v1.nnz() == 0) {
			return T(0);
		}

		T result = 0;
		for (size_t i = 0; i < v1.nnz(); i++) {
			result = scalar_add(result, scalar_mul(v1[i], v2[v1(i)], F), F);
		}

		return result;
	}

	template <typename index_t>
	bool sparse_vec_dot_dense_vec(const sparse_vec<bool, index_t>& v1, const bool* v2, const field_t& F) {
		if (v1.nnz() == 0)
			return bool(false);

		bool result = false;
		for (size_t i = 0; i < v1.nnz(); i++)
			result ^= v1[i] && v2[v1(i)];

		return result;
	}

	// the following two functions are used in sparse_mat_rref_reconstruct
	template <typename index_t>
	int_t sparse_vec_denominator_lcm(const sparse_vec<rat_t, index_t>& vec) {
		int_t d = 1;
		for (size_t i = 0; i < vec.nnz(); i++) {
			d = Flint::LCM(d, vec[i].den());
		}
		return d;
	}

	template <typename index_t>
	int_t sparse_vec_height(const sparse_vec<rat_t, index_t>& vec) {
		if (vec.nnz() == 0)
			return 1;
		int_t d = sparse_vec_denominator_lcm(vec);
		int_t h = (vec[0] * d).height();
		for (size_t i = 1; i < vec.nnz(); i++) {
			int_t hi = (vec[i] * d).height();
			if (hi > h)
				h = hi;
		}
		return h;
	}

	// debug only, not used to the large vector
	template <typename index_t, typename T> void print_vec_info(const sparse_vec<T, index_t>& vec) {
		std::cout << "-------------------" << std::endl;
		std::cout << "nnz: " << vec.nnz() << std::endl;
		std::cout << "indices: ";
		for (size_t i = 0; i < vec.nnz(); i++)
			std::cout << vec(i) << " ";
		std::cout << "\nentries: ";
		for (size_t i = 0; i < vec.nnz(); i++)
			std::cout << scalar_to_str(vec[i]) << " ";
		std::cout << std::endl;
	}

} // namespace SparseRREF

#endif
