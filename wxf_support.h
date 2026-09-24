/*
	Copyright (C) 2025-2026 Zhenjie Li (Li, Zhenjie)

	This file is part of SparseRREF. The SparseRREF is free software:
	you can redistribute it and/or modify it under the terms of the MIT
	License.
*/

#ifndef WXF_SUPPORT_H
#define WXF_SUPPORT_H

#include <algorithm>
#include <limits>
#include <type_traits>

#include "wxf_parser.h"
#include "sparse_type.h"

namespace SparseRREF {
	// sparse_tensor_read_wxf is the single implementation of the SparseArray layout; a matrix is a
	// rank 2 tensor, so sparse_mat_read_wxf reads a tensor and converts it. Forward declaration,
	// because the matrix reader comes first in this file.
	template <typename T, typename index_t>
	sparse_tensor<T, index_t, SPARSE_CSR> sparse_tensor_read_wxf(const std::vector<WXF_PARSER::Token>& tokens,
		const field_t& F, thread_pool* pool = nullptr, bool sort_ind = true);

	// Files larger than this are mapped instead of being copied into memory.
	inline constexpr uintmax_t wxf_mmap_threshold = 1ULL << 30;

	// The bytes a parser reads are only borrowed (WXF_PARSER::Parser keeps a pointer to them
	// and nothing else), so whoever owns them has to outlive the parse. A file above the
	// threshold is mapped, otherwise it is read into memory; either way the bytes stay alive
	// as long as this object does, and only one of the two is ever allocated.
	struct wxf_file_bytes {
		MMapFile mm;
		std::vector<uint8_t> buffer;
		std::string_view view;

		explicit wxf_file_bytes(const std::filesystem::path& file) {
			if (std::filesystem::file_size(file) > wxf_mmap_threshold) {
				std::string cc_str = std::filesystem::canonical(file).string();
				if (mmap_file(cc_str.c_str(), mm)) {
					view = mm.view;
					return;
				}
			}

			buffer = file_to_ustr(file);
			view = std::string_view((const char*)buffer.data(), buffer.size());
		}

		// MMapFile owns a mapping and must not be copied, so neither may this
		wxf_file_bytes(const wxf_file_bytes&) = delete;
		wxf_file_bytes& operator=(const wxf_file_bytes&) = delete;
	};

	namespace WXF_HELPER {
		// value fits in int64_t as an integer
		inline bool value_fits_si(const rat_t& value) { return value.is_integer() && call_flint(fmpz_fits_si, value.num_data()); }
		inline bool value_fits_si(const int_t& value) { return value.fits_si(); }
		inline bool value_fits_si(ulong value) { return value <= (ulong)(INT64_MAX); }

		// value to int64_t (unchecked!!)
		inline int64_t value_to_si(const rat_t& value) { return (int64_t)(call_flint(fmpz_get_si, value.num_data())); }
		inline int64_t value_to_si(const int_t& value) { return (int64_t)(value.to_si()); }
		inline int64_t value_to_si(ulong value) { return (int64_t)(value); }

		template <typename T>
			requires std::is_same_v<T, int_t> || std::is_same_v<T, ulong>
		inline void push_value(WXF_PARSER::Encoder& enc, const T& value) {
			if (value_fits_si(value))
				enc.push_integer(value_to_si(value));
			else
				enc.push_bigint(scalar_to_str(value));
		}

		// a numerator or a denominator: the small case is the common one and must not build an int_t
		inline void push_numerator(WXF_PARSER::Encoder& enc, const rat_t& value) {
			if (call_flint(fmpz_fits_si, value.num_data()))
				enc.push_integer((int64_t)(call_flint(fmpz_get_si, value.num_data())));
			else
				push_value(enc, value.num()); // does not fit int64: one copy, and only here
		}

		inline void push_denominator(WXF_PARSER::Encoder& enc, const rat_t& value) {
			if (call_flint(fmpz_fits_si, value.den_data()))
				enc.push_integer((int64_t)(call_flint(fmpz_get_si, value.den_data())));
			else
				push_value(enc, value.den());
		}

		inline void push_value(WXF_PARSER::Encoder& enc, const rat_t& value) {
			if (value.is_integer()) {
				push_numerator(enc, value);
			}
			else {
				// func,2,symbol,8,"Rational"
				enc.push_ustr("f\x02s\x08Rational");
				push_numerator(enc, value);
				push_denominator(enc, value);
			}
		}

		template <typename T>
			requires std::is_same_v<std::remove_cv_t<T>, ulong> || std::is_same_v<std::remove_cv_t<T>, int_t> || std::is_same_v<std::remove_cv_t<T>, rat_t>
		inline uint8_t vals_array_num_type(const std::span<T> vals) {
			if (vals.empty())
				return 0;
			if (!value_fits_si(vals[0]))
				return 102;
			int64_t min_val = value_to_si(vals[0]);
			int64_t max_val = min_val;
			for (size_t i = 1; i < vals.size(); i++) {
				if (!value_fits_si(vals[i]))
					return 102;
				auto value = value_to_si(vals[i]);
				if (value < min_val)
					min_val = value;
				if (value > max_val)
					max_val = value;
			}
			auto min_type = WXF_PARSER::minimal_signed_bits(min_val);
			auto max_type = WXF_PARSER::minimal_signed_bits(max_val);
			return std::max(min_type, max_type);
		}
	}

	template <typename T, typename index_t>
	sparse_mat<T, index_t> sparse_mat_read_wxf(const std::vector<WXF_PARSER::Token>& tokens, const field_t& F) {
		// a matrix is a rank 2 tensor; to_sparse_mat() also rejects any other rank
		return sparse_tensor_read_wxf<T, index_t>(tokens, F, nullptr, false).to_sparse_mat();
	}

	template <typename T, typename index_t>
	sparse_mat<T, index_t> sparse_mat_read_wxf(const std::filesystem::path file, const field_t& F) {
		if (!std::filesystem::exists(file)) {
			std::cerr << file << " does not exist." << std::endl;
			return sparse_mat<T, index_t>();
		}

		wxf_file_bytes bytes(file);
		WXF_PARSER::Parser wxf_parser(bytes.view);
		wxf_parser.parse();
		if (!WXF_PARSER::parse_ok(wxf_parser.err)) {
			std::cerr << "Error: sparse_mat_read: the WXF data is not valid ("
				<< WXF_PARSER::parse_error_code(wxf_parser.err) << ")" << std::endl;
			return sparse_mat<T, index_t>();
		}

		return sparse_mat_read_wxf<T, index_t>(wxf_parser.tokens, F);
	}

	// SparseArray[Automatic,dims,imp_val = 0,{1,{rowptr,colindex},vals}]
	template <typename T, typename index_t>
	std::vector<uint8_t> sparse_mat_write_wxf(const sparse_mat<T, index_t>& mat, bool include_head = true, bool mma_layout = false) {
		using namespace WXF_PARSER;

		// Wolfram cannot round trip a SparseArray with a zero dimension either: BinarySerialize of
		// SparseArray[{}, {0, 3}] is just List[], which reads back as {} and not as a SparseArray.
		// There is nothing to write, so say so instead of leaving an empty file behind.
		if (mat.nrow == 0 || mat.ncol == 0) {
			std::cerr << "Error: sparse_mat_write: a matrix with a zero dimension cannot be written as WXF" << std::endl;
			return std::vector<uint8_t>();
		}

		std::string_view ff_template = "SparseArray[Automatic,#dims,0,{1,{#rowptr,#colindex},#vals}]";

		size_t rank = 2;
		size_t nnz = mat.nnz();
		std::vector<size_t> dims_arr = { mat.nrow, mat.ncol };

		std::unordered_map<std::string, std::function<void(Encoder&)>> func_map;

		func_map["#dims"] = [&](Encoder& enc) {
			if (mma_layout) {
				auto num_type = WXF_PARSER::minimal_pos_signed_bits(*std::max_element(dims_arr.begin(), dims_arr.end()));
				enc.push_array({ 2 }, std::span(dims_arr), WXF_HEAD::array, num_type);
			}
			else {
				std::vector<int64_t> dims_arr_legacy = { (int64_t)mat.nrow, (int64_t)mat.ncol };
				enc.push_packed_array({ 2 }, dims_arr_legacy);
			}
			};

		func_map["#rowptr"] = [&](Encoder& enc) {
			if (mma_layout) {
				auto num_type = WXF_PARSER::minimal_pos_signed_bits(nnz);
				std::vector<size_t> rowptr(mat.nrow + 1);
				rowptr[0] = 0;
				for (size_t i = 0; i < mat.nrow; i++)
					rowptr[i + 1] = rowptr[i] + mat[i].nnz();
				enc.push_array({ mat.nrow + 1 }, std::span(rowptr), WXF_HEAD::array, num_type);
			}
			else {
				std::vector<int64_t> rowptr(mat.nrow + 1);
				rowptr[0] = 0;
				for (size_t i = 0; i < mat.nrow; i++) {
					rowptr[i + 1] = rowptr[i] + mat[i].nnz();
				}
				enc.push_packed_array({ mat.nrow + 1 }, rowptr);
			}
			};

		func_map["#colindex"] = [&](Encoder& enc) {
			uint8_t num_type = 0;
			if (mma_layout) {
				size_t max_colindex = 0;
				for (size_t i = 0; i < mat.nrow; i++) {
					auto index_span = mat[i].index_span();
					if (!index_span.empty()) {
						auto colindex = static_cast<size_t>(*std::max_element(index_span.begin(), index_span.end())) + 1;
						if (colindex > max_colindex)
							max_colindex = colindex;
					}
				}
				num_type = WXF_PARSER::minimal_pos_signed_bits(max_colindex);
			}
			else {
				num_type = minimal_pos_signed_bits(mat.ncol);
			}
			enc.push_array_info({ nnz, rank - 1 }, WXF_HEAD::array, num_type);
			if (num_type > 3) {
				std::cerr << "Error: sparse_mat_write_wxf: too large dimension" << std::endl;
				return;
			}
			enc.buffer.reserve(enc.buffer.size() + nnz * (rank - 1) * WXF_PARSER::size_of_arr_num_type(num_type));
			for (size_t i = 0; i < mat.nrow; i++) {
				if (mat[i].nnz() != 0)
					enc.push_array_data(mat[i].index_span(), num_type, [](const index_t& idx) {
						return idx + 1; // assumes idx + 1 does not exceed std::numeric_limits<index_t>::max()
						});
			}
			};

		func_map["#vals"] = [&](Encoder& enc) {
			if constexpr (std::is_same_v<T, rat_t> || std::is_same_v<T, int_t> || std::is_same_v<T, ulong>) {
				if (mma_layout) {
					uint8_t num_type = 0;
					for (size_t i = 0; i < mat.nrow; i++) {
						auto row_num_type = WXF_HELPER::vals_array_num_type(mat[i].entry_span());
						if (row_num_type == 102) {
							num_type = 102;
							break;
						}
						num_type = std::max(num_type, row_num_type);
					}

					if (num_type != 102) {
						enc.push_array_info({ nnz }, WXF_HEAD::array, num_type);
						enc.buffer.reserve(enc.buffer.size() + nnz * WXF_PARSER::size_of_arr_num_type(num_type));
						for (size_t i = 0; i < mat.nrow; i++) {
							if (mat[i].nnz() != 0)
								enc.push_array_data(mat[i].entry_span(), num_type, [](const T& value) {
									return WXF_HELPER::value_to_si(value);
									});
						}
					}
					else {
						enc.push_function("List", nnz);
						for (size_t i = 0; i < mat.nrow; i++) {
							for (size_t j = 0; j < mat[i].nnz(); j++) {
								WXF_HELPER::push_value(enc, mat[i][j]);
							}
						}
					}
				}
				else if constexpr (std::is_same_v<T, ulong>) {
					enc.push_array_info({ nnz }, WXF_HEAD::narray, 19);
					for (size_t i = 0; i < mat.nrow; i++) {
						enc.push_ustr(mat[i].entries, mat[i].nnz());
					}
				}
				else {
					enc.push_function("List", nnz);
					for (size_t i = 0; i < mat.nrow; i++) {
						for (size_t j = 0; j < mat[i].nnz(); j++)
							WXF_HELPER::push_value(enc, mat[i][j]);
					}
				}
			}
			};

		Encoder enc = fullform_to_wxf(ff_template, func_map, include_head);

		return enc.buffer;
	}

	// SparseArray[Automatic,dims,imp_val = 0,{1,{rowptr,colindex},vals}]
	template <typename T, typename index_t>
	sparse_tensor<T, index_t, SPARSE_CSR> sparse_tensor_read_wxf(const std::vector<WXF_PARSER::Token>& tokens, const field_t& F, thread_pool* pool, bool sort_ind) {
		using st = sparse_tensor<T, index_t, SPARSE_CSR>;

		if (tokens.size() < 13) {
			std::cerr << "Error: sparse_tensor_read: the WXF data is too short for a SparseArray" << std::endl;
			return st();
		}

		if (tokens[0].type != WXF_PARSER::WXF_HEAD::func ||
			tokens[0].length != 4 ||
			tokens[1].type != WXF_PARSER::WXF_HEAD::symbol ||
			tokens[1].get_string_view() != "SparseArray" ||
			tokens[2].type != WXF_PARSER::WXF_HEAD::symbol ||
			tokens[2].get_string_view() != "Automatic") {
			std::cerr << "Error: sparse_tensor_read: not a SparseArray" << std::endl;
			return st();
		}

		// see sparse_mat_read_wxf: dimensions and length share a union, so a part that is not an
		// array token (Wolfram writes List[] for the empty colindex/vals) must be rejected before
		// dimensions[...] is read, not misread
		auto is_array = [](const WXF_PARSER::Token& t) {
			return t.type == WXF_PARSER::WXF_HEAD::array || t.type == WXF_PARSER::WXF_HEAD::narray;
			};
		// the index parts are integers; a float/complex array would silently copy nothing
		auto is_int_array = [](const WXF_PARSER::Token& t) {
			if (t.type != WXF_PARSER::WXF_HEAD::array && t.type != WXF_PARSER::WXF_HEAD::narray)
				return false;
			switch (t.dimensions[0]) {
			case 0: case 1: case 2: case 3:
			case 16: case 17: case 18: case 19:
				return true;
			default:
				return false;
			}
			};

#define GENERATE_COPY_ARR(TYPE, CTYPE, FUNC) \
		case TYPE: { \
			auto sp = token.get_arr_span<CTYPE>(); \
			for (const auto& v : sp) {*out = FUNC(v); out++;} \
			break; }

#define TMP_IDENTITY_FUNC(x) (x)

#define GENERATE_ALL_ARR(FUNC2) \
		GENERATE_COPY_ARR(0, int8_t, FUNC2)  \
		GENERATE_COPY_ARR(1, int16_t, FUNC2) \
		GENERATE_COPY_ARR(2, int32_t, FUNC2) \
		GENERATE_COPY_ARR(3, int64_t, FUNC2) \
		GENERATE_COPY_ARR(16, uint8_t, FUNC2)  \
		GENERATE_COPY_ARR(17, uint16_t, FUNC2) \
		GENERATE_COPY_ARR(18, uint32_t, FUNC2) \
		GENERATE_COPY_ARR(19, uint64_t, FUNC2)

		auto copy_arr = [](const WXF_PARSER::Token& token, auto out) {
			int num_type = token.dimensions[0];
			switch (num_type) {
				GENERATE_ALL_ARR(TMP_IDENTITY_FUNC);
			default:
				std::cerr << "Error: sparse_tensor_read: read array fails" << std::endl;
				break;
			}
			};

		auto copy_modify_arr = [](const WXF_PARSER::Token& token, auto out, auto&& func) {
			int num_type = token.dimensions[0];
			switch (num_type) {
				GENERATE_ALL_ARR(func);
			default:
				std::cerr << "Error: sparse_tensor_read: read array fails" << std::endl;
				break;
			}
			};
#undef TMP_IDENTITY_FUNC
#undef GENERATE_COPY_ARR
#undef GENERATE_ALL_ARR

		// dims
		if (!is_int_array(tokens[3])) {
			std::cerr << "Error: sparse_tensor_read: the dimensions are not an integer array" << std::endl;
			return st();
		}
		std::vector<size_t> dims(tokens[3].dimensions[1]);
		copy_arr(tokens[3], dims.data());
		if (dims.size() < 2) {
			std::cerr << "Error: sparse_tensor_read: this is not a rank 2 or higher SparseArray" << std::endl;
			return st();
		}

		// imp_val
		if (tokens[4].get_integer() != 0) {
			std::cerr << "Error: sparse_tensor_read: the implicit value is not 0" << std::endl;
			return st();
		}

		// List of length 3
		// {1,{rowptr,colindex},vals}

		// check 
		if (tokens[5].type != WXF_PARSER::WXF_HEAD::func ||
			tokens[5].length != 3 ||
			tokens[6].type != WXF_PARSER::WXF_HEAD::symbol ||
			tokens[6].get_string_view() != "List" ||
			tokens[8].type != WXF_PARSER::WXF_HEAD::func ||
			tokens[8].length != 2 ||
			tokens[9].type != WXF_PARSER::WXF_HEAD::symbol ||
			tokens[9].get_string_view() != "List") {
			std::cerr << "Error: sparse_tensor_read: wrong format in SparseArray" << std::endl;
			return st();
		}

		// rowptr is tokens[10]
		if (!is_int_array(tokens[10])) {
			std::cerr << "Error: sparse_tensor_read: the row pointers are not an integer array" << std::endl;
			return st();
		}
		std::vector<size_t> rowptr(tokens[10].dimensions[1]);
		copy_arr(tokens[10], rowptr.data());
		if (rowptr.size() != dims[0] + 1) {
			std::cerr << "Error: sparse_tensor_read: the row pointers have the wrong length" << std::endl;
			return st();
		}
		size_t nz = rowptr.back();

		st tensor(dims, nz);
		tensor.data.rowptr = std::move(rowptr);
		if (nz == 0)
			return tensor;

		// colindex is tokens[11]
		if (!is_int_array(tokens[11])) {
			std::cerr << "Error: sparse_tensor_read: the column indices are not an integer array" << std::endl;
			return st();
		}
		if (WXF_PARSER::size_of_arr_num_type(tokens[11].dimensions[0]) > sizeof(index_t)) {
			std::cerr << "Error: sparse_tensor_read: the type of index is not enough for colindex" << std::endl;
			return st();
		}
		if (tokens[11].dimensions[1] != nz * (dims.size() - 1)) {
			std::cerr << "Error: sparse_tensor_read: the column indices have the wrong length" << std::endl;
			return st();
		}
		// mma is 1-based for colindex
		copy_modify_arr(tokens[11], tensor.data.colptr, [](auto v) { return v - 1; });

		// the row pointers have to be non-decreasing and every coordinate has to be inside its
		// dimension: a bogus index would be carried into the tensor -- and, read as a matrix, into
		// the rows of the matrix -- and corrupt whatever walks it later
		for (size_t i = 0; i < dims[0]; i++) {
			if (tensor.data.rowptr[i] > tensor.data.rowptr[i + 1]) {
				std::cerr << "Error: sparse_tensor_read: the row pointers are not increasing" << std::endl;
				return st();
			}
		}
		for (size_t j = 0; j < nz; j++) {
			auto ptr = tensor.data.colptr + j * (dims.size() - 1);
			for (size_t k = 0; k + 1 < dims.size(); k++) {
				if (ptr[k] < 0 || static_cast<size_t>(ptr[k]) >= dims[k + 1]) {
					std::cerr << "Error: sparse_tensor_read: an index is out of range" << std::endl;
					return st();
				}
			}
		}

		std::vector<char> buffer;
		auto get_int_from_sv = [&buffer](std::string_view sv) -> int_t {
			buffer.clear();
			buffer.reserve(sv.size() + 1);

			buffer.insert(buffer.end(), sv.begin(), sv.end());
			buffer.push_back('\0');
			return int_t(buffer.data());
			};

		auto get_int_from_tv = [&get_int_from_sv](const WXF_PARSER::Token& node) -> int_t {
			switch (node.type) {
			case WXF_PARSER::WXF_HEAD::i8:
			case WXF_PARSER::WXF_HEAD::i16:
			case WXF_PARSER::WXF_HEAD::i32:
			case WXF_PARSER::WXF_HEAD::i64:
				return int_t(node.get_integer());
			case WXF_PARSER::WXF_HEAD::bigint:
				return get_int_from_sv(node.get_string_view());
			default:
				std::cerr << "not a integer" << std::endl;
				return int_t(0);
			}
			};

		// the other is vals
		if (tokens[12].type == WXF_PARSER::WXF_HEAD::array ||
			tokens[12].type == WXF_PARSER::WXF_HEAD::narray) {
			if (tokens[12].dimensions[1] != nz) {
				std::cerr << "Error: sparse_tensor_read: the values have the wrong length" << std::endl;
				return st();
			}
			// this reader is exact: a float/complex value array has no representation here, and
			// copying nothing would silently turn it into zeros
			if (!is_int_array(tokens[12])) {
				std::cerr << "Error: sparse_tensor_read: the values are not integers or rationals" << std::endl;
				return st();
			}
			if constexpr (std::is_same_v<T, rat_t>)
				copy_arr(tokens[12], tensor.data.valptr);
			else if constexpr (std::is_same_v<T, ulong>) {
				copy_modify_arr(tokens[12], tensor.data.valptr, [&](auto v) {
					return nmod_set_si(v, F.mod);
					});
			}
		}
		else {
			// it is a list
			if (tokens[12].type != WXF_PARSER::WXF_HEAD::func ||
				tokens[12].length != nz ||
				tokens[13].type != WXF_PARSER::WXF_HEAD::symbol ||
				tokens[13].get_string_view() != "List") {
				std::cerr << "Error: sparse_tensor_read: wrong format in SparseArray" << std::endl;
				return st();
			}

			size_t pos = 14;
			T* vals = tensor.data.valptr;
			while (pos < tokens.size()) {
				auto& token = tokens[pos];
				T val;
				switch (token.type) {
				case WXF_PARSER::WXF_HEAD::i8:
				case WXF_PARSER::WXF_HEAD::i16:
				case WXF_PARSER::WXF_HEAD::i32:
				case WXF_PARSER::WXF_HEAD::i64:
					if constexpr (std::is_same_v<T, rat_t>) {
						val = token.get_integer();
					}
					else if constexpr (std::is_same_v<T, ulong>) {
						val = int_t(token.get_integer()) % F.mod;
					}
					break;
				case WXF_PARSER::WXF_HEAD::bigint:
					if constexpr (std::is_same_v<T, rat_t>) {
						val = get_int_from_tv(token);
					}
					else if constexpr (std::is_same_v<T, ulong>) {
						val = get_int_from_sv(token.get_string_view()) % F.mod;
					}
					break;
				case WXF_PARSER::WXF_HEAD::func: {
					auto ntoken = tokens[pos + 1];
					if (ntoken.get_string_view() == "Rational") {
						if (pos + 3 >= tokens.size()) {
							std::cerr << "Error: sparse_tensor_read: wrong format in SparseArray" << std::endl;
							return sparse_mat<T, index_t>();
						}

						int_t n_1 = get_int_from_tv(tokens[pos + 2]);
						int_t d_1 = get_int_from_tv(tokens[pos + 3]);
						if constexpr (std::is_same_v<T, rat_t>) {
							val = rat_t(std::move(n_1), std::move(d_1), true);
						}
						else if constexpr (std::is_same_v<T, ulong>) {
							val = rat_t(std::move(n_1), std::move(d_1), true) % F.mod;
						}
						pos += 3;
					}
					else {
						std::cerr << "Error: sparse_tensor_read: ";
						std::cerr << "not a SparseArray with rational / integer entries" << std::endl;
						return st();
					}
					break;
				}
				default:
					std::cerr << "Error: sparse_tensor_read: ";
					std::cerr << "not a SparseArray with rational / integer entries" << std::endl;
					return st();
					break;
				}
				*vals = val;
				vals++;
				pos++;
			}
		}
		
		if (sort_ind && !tensor.check_sorted())
			tensor.sort_indices(pool);

		return tensor;
	}

	template <typename T, typename index_t>
	auto sparse_tensor_read_wxf(const std::filesystem::path file, const field_t& F, thread_pool* pool = nullptr, bool sort_ind = true) {
		if (!std::filesystem::exists(file)) {
			std::cerr << file << " does not exist." << std::endl;
			return sparse_tensor<T, index_t, SPARSE_CSR>();
		}

		wxf_file_bytes bytes(file);
		WXF_PARSER::Parser wxf_parser(bytes.view);
		wxf_parser.parse();
		if (!WXF_PARSER::parse_ok(wxf_parser.err)) {
			std::cerr << "Error: sparse_tensor_read: the WXF data is not valid ("
				<< WXF_PARSER::parse_error_code(wxf_parser.err) << ")" << std::endl;
			return sparse_tensor<T, index_t, SPARSE_CSR>();
		}

		return sparse_tensor_read_wxf<T, index_t>(wxf_parser.tokens, F, pool, sort_ind);
	}

	// SparseArray[Automatic,dims,imp_val = 0,{1,{rowptr,colindex},vals}]
	template <typename T, typename index_t>
	std::vector<uint8_t> sparse_tensor_write_wxf(const sparse_tensor<T, index_t, SPARSE_CSR>& tensor, bool include_head = true, bool mma_layout = false) {
		using namespace WXF_PARSER;

		// e.g. Wolfram itself writes SparseArray[{}, {3, 0, 3}] to {{},{},{}},
		// which does not read back as a SparseArray, so there is nothing to write here
		if (std::find(tensor.dims().begin(), tensor.dims().end(), size_t(0)) != tensor.dims().end()) {
			std::cerr << "Error: sparse_tensor_write: a tensor with a zero dimension cannot be written as WXF" << std::endl;
			return std::vector<uint8_t>();
		}

		std::string_view ff_template = "SparseArray[Automatic,#dims,0,{1,{#rowptr,#colindex},#vals}]";

		auto rank = tensor.rank();
		auto nnz = tensor.nnz();
		std::vector<size_t> dims(tensor.dims().begin(), tensor.dims().end());

		std::unordered_map<std::string, std::function<void(Encoder&)>> func_map;

		func_map["#dims"] = [&](Encoder& enc) {
			if (mma_layout) {
				auto num_type = WXF_PARSER::minimal_pos_signed_bits(*std::max_element(dims.begin(), dims.end()));
				enc.push_array({ rank }, std::span(dims), WXF_HEAD::array, num_type);
			}
			else {
				std::vector<int64_t> dims_legacy(tensor.dims().begin(), tensor.dims().end());
				enc.push_packed_array({ rank }, dims_legacy);
			}
			};

		func_map["#rowptr"] = [&](Encoder& enc) {
			const auto& rowptr = tensor.data.rowptr;
			if (mma_layout) {
				auto num_type = WXF_PARSER::minimal_pos_signed_bits(nnz);
				enc.push_array({ rowptr.size() }, std::span(rowptr), WXF_HEAD::array, num_type);
			}
			else {
				enc.push_array({ rowptr.size() }, std::span(rowptr), WXF_HEAD::array, 3);
			}
			};


		func_map["#colindex"] = [&](Encoder& enc) {
			if (nnz == 0) {
				enc.push_function("List", 0);
				return;
			}
			auto col_span = std::span(tensor.data.colptr, (rank - 1) * nnz);
			uint8_t num_type = 0;
			if (mma_layout) {
				size_t max_colindex = 0;
				if (!col_span.empty())
					max_colindex = static_cast<size_t>(*std::max_element(col_span.begin(), col_span.end())) + 1;
				num_type = WXF_PARSER::minimal_pos_signed_bits(max_colindex);
			}
			else {
				num_type = minimal_pos_signed_bits(1 + *std::max_element(dims.begin() + 1, dims.end()));
			}
			enc.push_array({ nnz, rank - 1 }, col_span, WXF_HEAD::array, num_type, [](const index_t& idx) {
				return idx + 1; // assumes idx + 1 does not exceed std::numeric_limits<index_t>::max()
				});
			};

		func_map["#vals"] = [&](Encoder& enc) {
			if (nnz == 0) {
				enc.push_function("List", 0);
				return;
			}
			if constexpr (std::is_same_v<T, rat_t> || std::is_same_v<T, int_t> || std::is_same_v<T, ulong>) {
				auto val_span = std::span(tensor.data.valptr, nnz);
				if (mma_layout) {
					auto num_type = WXF_HELPER::vals_array_num_type(val_span);
					if (num_type != 102) {
						enc.push_array({ nnz }, val_span, WXF_HEAD::array, num_type, [](const T& value) {
							return WXF_HELPER::value_to_si(value);
							});
					}
					else {
						enc.push_function("List", nnz);
						for (size_t i = 0; i < nnz; i++) {
							WXF_HELPER::push_value(enc, tensor.val(i));
						}
					}
				}
				else if constexpr (std::is_same_v<T, ulong>) {
					enc.push_array_info({ nnz }, WXF_HEAD::narray, 19);
					enc.push_ustr(tensor.data.valptr, tensor.data.valptr + nnz);
				}
				else {
					enc.push_function("List", nnz);
					for (size_t i = 0; i < nnz; i++) {
						WXF_HELPER::push_value(enc, tensor.val(i));
					}
				}
			}
			};

		Encoder enc = fullform_to_wxf(ff_template, func_map, include_head);

		return enc.buffer;
	}

}

#endif // WXF_SUPPORT_H
