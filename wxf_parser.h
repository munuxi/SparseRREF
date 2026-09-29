/*
	Copyright (C) 2025 Zhenjie Li (Li, Zhenjie)

	You can redistribute it and/or modify it under the terms of the MIT
	License.
*/

/*
	WXF is a binary format for faithfully serializing Wolfram Language expressions
	in a form suitable for outside storage or interchange with other programs.
	WXF can readily be interpreted using low-level native types available in many
	programming languages, making it suitable as a format for reading and writing
	Wolfram Language expressions in other programming languages.

	The details of the WXF format are described in the Wolfram Language documentation:
	https://reference.wolfram.com/language/tutorial/WXFFormatDescription.html.en .

	The full list of supported types is given below:

	done	byte value  type of part
	*		102			function
	*		67			int8_t
	*		106			int16_t
	*		105			int32_t
	*		76			int64_t
	*		114			machine reals
	*		83			string
	*		66			binary string
	*		115			symbol
	*		73			big integer
	*		82			big real
	*		193			packed array
	*		194			numeric array
	*		65			association
	*		58			delayed rule in association
	*		45			rule in association
*/

#pragma once

#include <complex>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>

namespace WXF_PARSER {

	using complex_float_t = std::complex<float>;
	using complex_double_t = std::complex<double>;

	// Parser::err values.  `ok` means the buffer was consumed without complaint.
	enum class parse_error : int {
		ok = 0,
		invalid_head = 1,       // not a WXF stream (missing the 56 58 magic)
		unknown_type = 2,       // a head byte that is not a known WXF type
		truncated = 3,          // the buffer ends in the middle of a value
		bad_length = 4,         // a declared length/element count leaves the buffer
		bad_rank = 5,           // an array rank that is too large to be real
		bad_structure = 6,      // an expression tree that does not match the tokens
		no_data = 7             // nothing to decode
	};

	// helpers for callers that only test success or need the numeric code for a log
	inline constexpr bool parse_ok(const parse_error e) noexcept { return e == parse_error::ok; }
	inline constexpr int parse_error_code(const parse_error e) noexcept { return static_cast<int>(e); }

	// Bounds for the decoder.  Both are guards against malformed input, not format
	// limits: a rank this large cannot be backed by the bytes that declared it.
	inline constexpr size_t wxf_max_rank = 1000;
	// Depth limit for the FullForm reader/writer, so nested input cannot exhaust
	// the stack.
	inline constexpr size_t wxf_max_fullform_depth = 256;

	enum class WXF_HEAD {
		// function type
		func = 102,
		association = 65,
		delay_rule = 58,
		rule = 45,
		// string type
		symbol = 115,
		string = 83,
		binary_string = 66,
		bigint = 73,
		bigreal = 82,
		// number type
		i8 = 67,
		i16 = 106,
		i32 = 105,
		i64 = 76,
		f64 = 114,
		// array type
		array = 193,
		narray = 194
	};

	inline size_t size_of_head_num_type(const WXF_HEAD head) {
		switch (head) {
		case WXF_HEAD::i8:
			return sizeof(int8_t);
		case WXF_HEAD::i16:
			return sizeof(int16_t);
		case WXF_HEAD::i32:
			return sizeof(int32_t);
		case WXF_HEAD::i64:
			return sizeof(int64_t);
		case WXF_HEAD::f64:
			return sizeof(double);
		default:
			return 0;
		}
	}

	// array: head(numeric/packed array), num_type, rank, dimensions, data
	// for the num_type
	// 0 is int8_t      1 is int16_t
	// 2 is int32_t     3 is int64_t
	// 16 is uint8_t    17 is uint16_t ; only for numeric array
	// 18 is uint32_t   19 is uint64_t ; only for numeric array
	// 34 float         35 double
	// 51 complex float 52 complex double

	// only the last 3 bits are used to indicate the size
	constexpr size_t size_of_arr_num_type(const int num_type) {
		return size_t(1) << (num_type & 0b111);
	}

	template <typename T>
		requires std::is_integral_v<T>&& std::is_signed_v<T>
	constexpr uint8_t minimal_signed_bits(T x) noexcept {
		if (x >= INT8_MIN && x <= INT8_MAX) return 0;
		if (x >= INT16_MIN && x <= INT16_MAX) return 1;
		if (x >= INT32_MIN && x <= INT32_MAX) return 2;
		return 3; // for int64_t
	}

	// for positive signed/unsigned integer
	template <typename T>
	constexpr uint8_t minimal_pos_signed_bits(T x) noexcept {
		if (x <= INT8_MAX) return 0;
		if (x <= INT16_MAX) return 1;
		if (x <= INT32_MAX) return 2;
		if (x <= INT64_MAX) return 3;
		return 4; // for uint64_t
	}

	template <typename T>
		requires std::is_integral_v<T>&& std::is_unsigned_v<T>
	constexpr uint8_t minimal_unsigned_bits(T x) noexcept {
		if (x <= UINT8_MAX) return 0;
		if (x <= UINT16_MAX) return 1;
		if (x <= UINT32_MAX) return 2;
		return 3; // for uint64_t
	}

	inline void serialize_varint(std::vector<uint8_t>& buffer, uint64_t val) {
		uint8_t temp[10];
		size_t i = 0;

		do {
			temp[i] = val & 0x7F;
			val >>= 7;
			if (val != 0) temp[i] |= 0x80;
			++i;
		} while (val != 0);

		buffer.insert(buffer.end(), temp, temp + i);
	}

	template <typename T>
	void serialize_binary(std::vector<uint8_t>& buffer, const T& value) {
		const size_t old_size = buffer.size();
		buffer.resize(old_size + sizeof(T));
		std::memcpy(buffer.data() + old_size, &value, sizeof(T));
	}

	template <typename T>
	void serialize_binary(std::vector<uint8_t>& buffer, const T* valptr, const size_t len) {
		buffer.insert(buffer.end(), (uint8_t*)valptr, (uint8_t*)(valptr + len));
	}

	struct Encoder {
		std::vector<uint8_t> buffer;

		Encoder() = default;
		~Encoder() = default;
		Encoder(const Encoder&) = default;
		Encoder& operator=(const Encoder&) = default;
		Encoder(Encoder&&) = default;
		Encoder& operator=(Encoder&&) = default;

		void clear() { buffer.clear(); }

		// move from existing buffer
		Encoder(std::vector<uint8_t>&& buf) : buffer(std::move(buf)) {}

		// buffer management, so callers can pre-size and inspect the output
		const size_t size() const { return buffer.size(); }
		void reserve(const size_t new_size) { buffer.reserve(new_size); }
		void resize(const size_t new_size) { buffer.resize(new_size); }
		void resize(const size_t new_size, const uint8_t val) { buffer.resize(new_size, val); }

		// push ustr directly
		Encoder& push_ustr(const std::vector<uint8_t>& str) { buffer.insert(buffer.end(), str.begin(), str.end()); return *this; }
		Encoder& push_ustr(const std::string_view str) {
			buffer.insert(buffer.end(), (uint8_t*)str.data(), (uint8_t*)(str.data() + str.size())); return *this;
		}
		template<typename T>
		Encoder& push_ustr(const T* str_ptr, const size_t len) {
			buffer.insert(buffer.end(), (uint8_t*)str_ptr, (uint8_t*)(str_ptr + len)); return *this;
		}
		template<typename T>
		Encoder& push_ustr(T* start, T* end) {
			buffer.insert(buffer.end(), (uint8_t*)start, (uint8_t*)end); return *this;
		}

		// ---- local additions for SparseRREF, not present in upstream wxf_parser -------------
		// These two write the elements into a buffer after a per element transform, so an array
		// can be stored in a num_type narrower than the type it is given; the upstream API only
		// accepts data whose element type already matches the num_type.  wxf_support.h relies on
		// them, so a sync with upstream has to keep this block.

		// generate data as ustr in the back
		template<typename T, typename F>
			requires std::is_trivially_copyable_v<T> && std::is_invocable_r_v<T, F&, size_t>
		Encoder& generate_back_ustr(const size_t len, F&& func) {
			size_t old_size = buffer.size();
			buffer.resize(old_size + len * sizeof(T));
			uint8_t* data_ptr = buffer.data() + old_size;
			for (size_t i = 0; i < len; i++) {
				T value = func(i);
				std::memcpy(data_ptr + i * sizeof(T), &value, sizeof(T));
			}
			return *this;
		}

		// generate transformed data as ustr in the back
		template<typename T1, typename T2, typename F>
			requires std::is_trivially_copyable_v<T1> && std::is_invocable_r_v<T1, F&, const T2&>
		Encoder& transform_back_ustr(const T2* src_ptr, const size_t len, F&& func) {
			size_t old_size = buffer.size();
			buffer.resize(old_size + len * sizeof(T1));
			uint8_t* data_ptr = buffer.data() + old_size;
			for (size_t i = 0; i < len; i++) {
				T1 value = func(src_ptr[i]);
				std::memcpy(data_ptr + i * sizeof(T1), &value, sizeof(T1));
			}
			return *this;
		}

		Encoder& push_integer(const int64_t val) {
			auto num_type = minimal_signed_bits(val);
			switch (num_type) {
			case 0:
				buffer.push_back((uint8_t)WXF_HEAD::i8);
				serialize_binary(buffer, (int8_t)val);
				break;
			case 1:
				buffer.push_back((uint8_t)WXF_HEAD::i16);
				serialize_binary(buffer, (int16_t)val);
				break;
			case 2:
				buffer.push_back((uint8_t)WXF_HEAD::i32);
				serialize_binary(buffer, (int32_t)val);
				break;
			case 3:
				buffer.push_back((uint8_t)WXF_HEAD::i64);
				serialize_binary(buffer, val);
				break;
			default:
				break;
			}
			return *this;
		}

		Encoder& push_real(const double val) {
			buffer.push_back((uint8_t)WXF_HEAD::f64);
			serialize_binary(buffer, val);
			return *this;
		}

		// push string type struct: string/symbol/bigint/bigreal, default type is string
		Encoder& push_string(const std::string_view str, const WXF_HEAD type = WXF_HEAD::string) {
			buffer.push_back((uint8_t)type);
			serialize_varint(buffer, str.size());
			return push_ustr(str);
		}

		Encoder& push_symbol(const std::string_view sym) { return push_string(sym, WXF_HEAD::symbol); }
		Encoder& push_bigint(const std::string_view bigint_str) { return push_string(bigint_str, WXF_HEAD::bigint); }
		Encoder& push_bigreal(const std::string_view bigreal_str) { return push_string(bigreal_str, WXF_HEAD::bigreal); }
		Encoder& push_binary_string(const std::string_view bin_str) { return push_string(bin_str, WXF_HEAD::binary_string); }

		Encoder& push_function(const std::string_view head, const size_t num_vars) {
			buffer.push_back((uint8_t)WXF_HEAD::func);
			serialize_varint(buffer, num_vars);
			return push_string(head, WXF_HEAD::symbol);
		}

		Encoder& push_association(const size_t num_rules) {
			buffer.push_back((uint8_t)WXF_HEAD::association);
			serialize_varint(buffer, num_rules);
			return *this;
		}

		Encoder& push_rule() { buffer.push_back((uint8_t)WXF_HEAD::rule); return *this; }
		Encoder& push_delay_rule() { buffer.push_back((uint8_t)WXF_HEAD::delay_rule); return *this; }

		// return the total length of the array
		size_t push_array_info(const std::vector<size_t>& dimension_array, WXF_HEAD type, uint8_t num_type) {
			size_t all_len = 1;
			// [array_type, num_type, rank, dimensions...]
			buffer.push_back((uint8_t)type);
			buffer.push_back(num_type);
			serialize_varint(buffer, dimension_array.size());
			for (auto dim : dimension_array) {
				serialize_varint(buffer, dim);
				all_len *= dim;
			}
			return all_len;
		}

		// ---- local additions for SparseRREF, not present in upstream wxf_parser -------------
		// The same narrow num_type writes as above, through a transform or a generator, with
		// (push_array, push_generated_array) and without (push_array_data,
		// push_generated_array_data) the array header.  push_array here takes the transform and
		// keeps the identity default, so a plain four argument call still works; upstream has a
		// raw overload that refuses a num_type whose width differs from sizeof(T).

		template<typename T, typename F = std::identity>
			requires std::is_invocable_v<F&, const T&>
		Encoder& push_array_data(const std::span<T> data, uint8_t num_type, F&& func = std::identity{}) {
			// backup current size
			size_t old_size = buffer.size();
			using value_t = std::remove_cvref_t<std::invoke_result_t<F&, const T&>>;

			if constexpr (std::is_integral_v<value_t>) {
				const bool consistent_sign = (num_type >> 2) == (std::is_unsigned_v<value_t> ? (16 >> 2) : 0);
				if (size_of_arr_num_type(num_type) == sizeof(value_t) && consistent_sign) {
					if constexpr (std::is_same_v<std::remove_cvref_t<F>, std::identity>) {
						return push_ustr(data.data(), data.size());
					}
					else {
						return transform_back_ustr<value_t>(data.data(), data.size(), func);
					}
				}

#define APPEND_CASTED_ARRAY_DATA(TYPE) do {                                            \
				transform_back_ustr<TYPE>(data.data(), data.size(),                    \
					[&](const T& value) { return static_cast<TYPE>(func(value)); });   \
				} while (0)

				switch (num_type) {
				case 0: APPEND_CASTED_ARRAY_DATA(int8_t); break;
				case 1: APPEND_CASTED_ARRAY_DATA(int16_t); break;
				case 2: APPEND_CASTED_ARRAY_DATA(int32_t); break;
				case 3: APPEND_CASTED_ARRAY_DATA(int64_t); break;
				case 16: APPEND_CASTED_ARRAY_DATA(uint8_t); break;
				case 17: APPEND_CASTED_ARRAY_DATA(uint16_t); break;
				case 18: APPEND_CASTED_ARRAY_DATA(uint32_t); break;
				case 19: APPEND_CASTED_ARRAY_DATA(uint64_t); break;
				default: std::cerr << "Encoder::push_array_data: unsupported integer array num_type "
						<< static_cast<int>(num_type) << "." << std::endl;
					buffer.resize(old_size);
					break;
				}
				return *this;
#undef APPEND_CASTED_ARRAY_DATA
			}
			else {
				if constexpr (std::is_same_v<std::remove_cvref_t<F>, std::identity>) {
					return push_ustr(data.data(), data.size());
				}
				else {
					return transform_back_ustr<value_t>(data.data(), data.size(), func);
				}
			}
		}

		template<typename T, typename F = std::identity>
			requires std::is_invocable_v<F&, const T&>
		Encoder& push_array(const std::vector<size_t>& dimension_array, const std::span<T> data, WXF_HEAD type, uint8_t num_type, F&& func = std::identity{}) {
			// backup current size
			size_t old_size = buffer.size();
			using value_t = std::remove_cvref_t<std::invoke_result_t<F&, const T&>>;

			// [array_type, num_type, rank, dimensions..., data...]
			auto all_len = push_array_info(dimension_array, type, num_type);

			if (all_len != data.size()) {
				std::cerr << "Encoder::push_array: Data size does not match the dimension array." << std::endl;
				// restore buffer
				buffer.resize(old_size);
				return *this;
			}

			// the header reports size_of_arr_num_type(num_type) bytes per element, so
			// the data written for it has to match; push_array_data takes care of the
			// integral cases by casting, and rejects num_types it cannot serve
			if constexpr (std::is_integral_v<value_t>) {
				const bool supported_num_type = num_type <= 3 || (num_type >= 16 && num_type <= 19);
				if (!supported_num_type) {
					std::cerr << "Encoder::push_array: unsupported integer array num_type "
						<< static_cast<int>(num_type) << "." << std::endl;
					buffer.resize(old_size);
					return *this;
				}
			}
			// for the non-integral cases the width has to match exactly, because
			// push_array_data writes the elements as they are
			if constexpr (!std::is_integral_v<value_t>) {
				if (size_of_arr_num_type(num_type) != sizeof(value_t)) {
					std::cerr << "Encoder::push_array: num_type " << static_cast<int>(num_type)
						<< " means " << size_of_arr_num_type(num_type) << " bytes per element, but "
						<< sizeof(value_t) << " bytes were given." << std::endl;
					buffer.resize(old_size);
					return *this;
				}
			}
			return push_array_data(data, num_type, std::forward<F>(func));
		}

		template<typename F>
			requires std::is_invocable_v<F&, size_t>
		Encoder& push_generated_array_data(const size_t len, uint8_t num_type, F&& func) {
			// backup current size
			size_t old_size = buffer.size();
			using value_t = std::remove_cvref_t<std::invoke_result_t<F&, size_t>>;

			if constexpr (std::is_integral_v<value_t>) {
#define GENERATE_ARRAY_DATA(TYPE) do {                               \
				generate_back_ustr<TYPE>(len, [&](size_t i) {        \
					return static_cast<TYPE>(func(i));               \
				});} while (0)

				switch (num_type) {
				case 0: GENERATE_ARRAY_DATA(int8_t); break;
				case 1: GENERATE_ARRAY_DATA(int16_t); break;
				case 2: GENERATE_ARRAY_DATA(int32_t); break;
				case 3: GENERATE_ARRAY_DATA(int64_t); break;
				case 16: GENERATE_ARRAY_DATA(uint8_t); break;
				case 17: GENERATE_ARRAY_DATA(uint16_t); break;
				case 18: GENERATE_ARRAY_DATA(uint32_t); break;
				case 19: GENERATE_ARRAY_DATA(uint64_t); break;
				default: std::cerr << "Encoder::push_generated_array_data: unsupported integer array num_type "
						<< static_cast<int>(num_type) << "." << std::endl;
					buffer.resize(old_size);
					break;
				}
				return *this;
#undef GENERATE_ARRAY_DATA
			}
			else {
				return generate_back_ustr<value_t>(len, func);
			}
		}

		template<typename F>
			requires std::is_invocable_v<F&, size_t>
		Encoder& push_generated_array(const std::vector<size_t>& dimension_array, WXF_HEAD type, uint8_t num_type, F&& func) {
			// backup current size
			size_t old_size = buffer.size();
			using value_t = std::remove_cvref_t<std::invoke_result_t<F&, size_t>>;

			// [array_type, num_type, rank, dimensions..., data...]
			auto all_len = push_array_info(dimension_array, type, num_type);

			// push data
			if constexpr (std::is_integral_v<value_t>) {
				const bool supported_num_type = num_type <= 3 || (num_type >= 16 && num_type <= 19);
				if (!supported_num_type) {
					std::cerr << "Encoder::push_generated_array: unsupported integer array num_type "
						<< static_cast<int>(num_type) << "." << std::endl;
					buffer.resize(old_size);
					return *this;
				}
			}
			return push_generated_array_data(all_len, num_type, std::forward<F>(func));
		}

		// A packed array is limited by the format to signed integers, reals and
		// complex values: the unsigned element types (16..19) belong to the numeric
		// array only.  That is why the element type is constrained to signed here;
		// use push_numeric_array for unsigned data.
		template<typename T>
			requires std::is_integral_v<T>&& std::is_signed_v<T>
		Encoder& push_packed_array(const std::vector<size_t>& dimension_array, const std::span<const T> data) {
			const int num_type = minimal_signed_bits((std::numeric_limits<T>::max)());
			return push_array(dimension_array, data, WXF_HEAD::array, uint8_t(num_type));
		}

		Encoder& push_packed_array(const std::vector<size_t>& dimension_array, const std::span<const float> data) {
			return push_array(dimension_array, data, WXF_HEAD::array, 34);
		}

		Encoder& push_packed_array(const std::vector<size_t>& dimension_array, const std::span<const double> data) {
			return push_array(dimension_array, data, WXF_HEAD::array, 35);
		}

		Encoder& push_packed_array(const std::vector<size_t>& dimension_array, const std::span<const complex_float_t> data) {
			return push_array(dimension_array, data, WXF_HEAD::array, 51);
		}

		Encoder& push_packed_array(const std::vector<size_t>& dimension_array, const std::span<const complex_double_t> data) {
			return push_array(dimension_array, data, WXF_HEAD::array, 52);
		}

		template<typename T>
		Encoder& push_packed_array(const std::vector<size_t>& dimension_array, const std::vector<T>& data) {
			return push_packed_array(dimension_array, std::span<const T>(data));
		}

		// A numeric array is the only place where unsigned element types exist
		// (16..19), so this is the entry point for unsigned data.
		template<typename T>
			requires std::is_integral_v<T>
		Encoder& push_numeric_array(const std::vector<size_t>& dimension_array, const std::span<const T> data) {
			int num_type;
			if constexpr (std::is_signed_v<T>)
				num_type = minimal_signed_bits((std::numeric_limits<T>::max)());
			else
				num_type = 16 + minimal_unsigned_bits((std::numeric_limits<T>::max)());
			return push_array(dimension_array, data, WXF_HEAD::narray, uint8_t(num_type));
		}

		Encoder& push_numeric_array(const std::vector<size_t>& dimension_array, const std::span<const float> data) {
			return push_array(dimension_array, data, WXF_HEAD::narray, 34);
		}

		Encoder& push_numeric_array(const std::vector<size_t>& dimension_array, const std::span<const double> data) {
			return push_array(dimension_array, data, WXF_HEAD::narray, 35);
		}

		Encoder& push_numeric_array(const std::vector<size_t>& dimension_array, const std::span<const complex_float_t> data) {
			return push_array(dimension_array, data, WXF_HEAD::narray, 51);
		}

		Encoder& push_numeric_array(const std::vector<size_t>& dimension_array, const std::span<const complex_double_t> data) {
			return push_array(dimension_array, data, WXF_HEAD::narray, 52);
		}

		template<typename T>
		Encoder& push_numeric_array(const std::vector<size_t>& dimension_array, const std::vector<T>& data) {
			return push_numeric_array(dimension_array, std::span<const T>(data));
		}
	};

	struct Token {
		WXF_HEAD type;
		int rank = 0;
		union {
			// for number, string, symbol, bigint
			size_t length;
			// for array and narray, dimensions[0] is the type, dimensions[1] is the total flatten length
			// so the length is dimensions is rank + 2
			size_t* dimensions;
		};
		const uint8_t* data; // pointer to the data in the original buffer

		Token() : type(WXF_HEAD::i8), rank(0), length(0), data(nullptr) {}
		Token(const WXF_HEAD t, const size_t len, const uint8_t* d) : type(t), rank(0), length(len), data(d) {}
		Token(const WXF_HEAD t, const std::vector<size_t>& dims, const int num_type, const size_t len, const uint8_t* d) : type(t), data(d) {
			int r = int(dims.size());
			rank = r;
			dimensions = (size_t*)malloc((r + 2) * sizeof(size_t));
			dimensions[0] = num_type;
			dimensions[1] = len;
			for (auto i = 0; i < r; i++) {
				dimensions[i + 2] = dims[i];
			}
		}

		size_t dim(size_t i) const {
			if (rank > 0)
				return dimensions[i + 2];
			return length;
		}

		template<typename T> T* get_ptr() const { return (T*)data; }

		~Token() { free_dimensions(); }

		// `length` and `dimensions` are the same storage, so a copy has to decide
		// which one to *materialize*: either a deep copy of the block (when the
		// source still owns it), or a plain copy of the scalar.  A moved-from array
		// token has `dimensions == nullptr`, and that is exactly the case the old
		// code missed - it only looked at `type` and memcpy'd the nullptr.
		Token(const Token& other) : type(other.type), rank(other.rank), data(other.data) {
			if (other.is_array() && other.dimensions != nullptr) {
				dimensions = (size_t*)malloc((rank + 2) * sizeof(size_t));
				std::memcpy(dimensions, other.dimensions, (rank + 2) * sizeof(size_t));
			}
			else
				length = other.length;
		}

		Token(Token&& other) noexcept : type(other.type), rank(other.rank), data(other.data) {
			if (is_array())
				dimensions = other.dimensions; // steal the block ...
			else
				length = other.length;
			other.type = WXF_HEAD::i8;         // ... and leave the source in a well-defined
			other.rank = 0;                   // moved-from state: no longer an array, so
			other.length = 0;                 // destruction and copying are both harmless
		}

		Token& operator=(const Token& other) {
			if (this != &other)
				assign_from(other, /*steal=*/false);
			return *this;
		}

		Token& operator=(Token&& other) noexcept {
			if (this != &other)
				steal_from(other);
			return *this;
		}

		int64_t get_integer() const {
			// the payloads in a WXF stream are not guaranteed to be aligned, so
			// read through memcpy instead of dereferencing a reinterpreted pointer
			if (type == WXF_HEAD::i8) {
				int8_t v = 0;
				std::memcpy(&v, data, sizeof v);
				return v;
			}
			else if (type == WXF_HEAD::i16) {
				int16_t v = 0;
				std::memcpy(&v, data, sizeof v);
				return v;
			}
			else if (type == WXF_HEAD::i32) {
				int32_t v = 0;
				std::memcpy(&v, data, sizeof v);
				return v;
			}
			else if (type == WXF_HEAD::i64) {
				int64_t v = 0;
				std::memcpy(&v, data, sizeof v);
				return v;
			}
			else
				return 0;
		}

		double get_real() const {
			if (type == WXF_HEAD::f64) {
				double v = 0;
				std::memcpy(&v, data, sizeof v);
				return v;
			}
			else
				return 0;
		}

		std::string_view get_string_view() const {
			if (type == WXF_HEAD::symbol
				|| type == WXF_HEAD::bigint
				|| type == WXF_HEAD::bigreal
				|| type == WXF_HEAD::string
				|| type == WXF_HEAD::binary_string) {
				return std::string_view((const char*)data, length);
			}
			else
				return std::string_view();
		}

		bool is_array() const { return type == WXF_HEAD::array || type == WXF_HEAD::narray; }

		// an array token owns one block behind `dimensions`; every other type must
		// leave that union member alone
		void free_dimensions() {
			if (is_array()) {
				free(dimensions);
				dimensions = nullptr;
			}
		}

		// Leave a moved-from token in the default (non-array) state: its block has been
		// taken, so it must not be treated as an array again.  Running this *after* the
		// destination has taken the block is what keeps `dimensions` out of a const
		// path - which is why the union members need no `mutable`.
		static void mark_moved_from(Token& other) noexcept {
			other.type = WXF_HEAD::i8;
			other.rank = 0;
			other.length = 0;
		}

		// Shared by copy and move assignment so the two cannot drift apart:
		//  1. work out the block this token should end up with: a deep copy of the
		//     source's block (copy), the source's block itself (move), or none,
		//  2. release the block this token owns *before* overwriting the union -
		//     releasing after the overwrite is what used to leak it,
		//  3. materialize exactly one union member and copy the scalar fields.
		// The source is only ever read here; the move-specific "clear the source" step
		// lives in steal_from(), where the source is a non-const reference.
		void assign_from(const Token& other, const bool steal) noexcept {
			size_t* new_dimensions = nullptr;
			if (other.is_array() && other.dimensions != nullptr) {
				if (steal)
					new_dimensions = other.dimensions;
				else {
					new_dimensions = (size_t*)malloc((other.rank + 2) * sizeof(size_t));
					if (new_dimensions != nullptr)
						std::memcpy(new_dimensions, other.dimensions, (other.rank + 2) * sizeof(size_t));
				}
			}

			free_dimensions();

			type = other.type;
			rank = other.rank;
			if (new_dimensions != nullptr)
				dimensions = new_dimensions;
			else
				length = other.length;
			data = other.data;
		}

		// the move-assignment entry point
		void steal_from(Token& other) noexcept {
			assign_from(other, /*steal=*/true);
			if (other.is_array()) // an array source only keeps its type when the block
				mark_moved_from(other); // was nullptr, i.e. when there was nothing to move
		}

	public:

		template<typename T>
		std::span<const T> get_arr_span() const {
			if (type != WXF_HEAD::array && type != WXF_HEAD::narray)
				return std::span<const T>();
			return std::span<const T>((T*)data, dimensions[1]);
		}

		// debug only, print the token info
		template<typename T>
		void print(T& ss) const {
			auto& token = *this;
			switch (token.type) {
			case WXF_HEAD::i8:
				ss << "i8: " << token.get_integer() << std::endl;
				break;
			case WXF_HEAD::i16:
				ss << "i16: " << token.get_integer() << std::endl;
				break;
			case WXF_HEAD::i32:
				ss << "i32: " << token.get_integer() << std::endl;
				break;
			case WXF_HEAD::i64:
				ss << "i64: " << token.get_integer() << std::endl;
				break;
			case WXF_HEAD::f64:
				ss << "f64: " << token.get_real() << std::endl;
				break;
			case WXF_HEAD::symbol:
				ss << "symbol: " << token.get_string_view() << std::endl;
				break;
			case WXF_HEAD::bigint:
				ss << "bigint: " << token.get_string_view() << std::endl;
				break;
			case WXF_HEAD::bigreal:
				ss << "bigreal: " << token.get_string_view() << std::endl;
				break;
			case WXF_HEAD::string:
				ss << "string: " << token.get_string_view() << std::endl;
				break;
			case WXF_HEAD::binary_string:
				ss << "binary_string:" << token.get_string_view() << std::endl;
				break;
			case WXF_HEAD::func:
				ss << "func: " << token.length << " vars" << std::endl;
				break;
			case WXF_HEAD::association:
				ss << "association: " << token.length << " rules" << std::endl;
				break;
			case WXF_HEAD::delay_rule:
				ss << "delay_rule: " << token.length << std::endl;
				break;
			case WXF_HEAD::rule:
				ss << "rule: " << token.length << std::endl;
				break;
			case WXF_HEAD::array: {
				ss << "array: rank = " << token.rank << ", dimensions = ";
				size_t all_len = token.dimensions[1];
				for (int i = 0; i < token.rank; i++) {
					ss << token.dimensions[i + 2] << " ";
				}
				ss << std::endl;

				auto num_type = token.dimensions[0];
				ss << "data: ";
				if (token.data == nullptr)
					break;

#define PRINT_ARRAY_CASE(TYPE1, TYPE2) do {                      \
				for (size_t i = 0; i < all_len; i++) {           \
					ss << (TYPE2)(get_ptr<TYPE1>()[i]) << " ";   \
				}} while (0)

				switch (num_type) {
				case 0: PRINT_ARRAY_CASE(int8_t, int64_t); break;
				case 1: PRINT_ARRAY_CASE(int16_t, int64_t); break;
				case 2: PRINT_ARRAY_CASE(int32_t, int64_t); break;
				case 3: PRINT_ARRAY_CASE(int64_t, int64_t); break;
				case 34: PRINT_ARRAY_CASE(float, double); break;
				case 35: PRINT_ARRAY_CASE(double, double); break;
				case 51: PRINT_ARRAY_CASE(complex_float_t, complex_double_t); break;
				case 52: PRINT_ARRAY_CASE(complex_double_t, complex_double_t); break;
				default: std::cerr << "Unknown number type: " << num_type << std::endl; break;
				}
				ss << std::endl;
				break;
				}
			case WXF_HEAD::narray: {
				ss << "narray: rank = " << token.rank << ", dimensions = ";
				for (int i = 0; i < token.rank; i++) {
					ss << token.dimensions[i + 2] << " ";
				}
				ss << std::endl;

				size_t num_type = token.dimensions[0];
				size_t all_len = token.dimensions[1];

				ss << "data: ";
				if (token.data == nullptr)
					break;
				switch (num_type) {
				case 0: PRINT_ARRAY_CASE(int8_t, int64_t); break;
				case 1: PRINT_ARRAY_CASE(int16_t, int64_t); break;
				case 2: PRINT_ARRAY_CASE(int32_t, int64_t); break;
				case 3: PRINT_ARRAY_CASE(int64_t, int64_t); break;
				case 16: PRINT_ARRAY_CASE(uint8_t, uint64_t); break;
				case 17: PRINT_ARRAY_CASE(uint16_t, uint64_t); break;
				case 18: PRINT_ARRAY_CASE(uint32_t, uint64_t); break;
				case 19: PRINT_ARRAY_CASE(uint64_t, uint64_t); break;
				case 34: PRINT_ARRAY_CASE(float, double); break;
				case 35: PRINT_ARRAY_CASE(double, double); break;
				case 51: PRINT_ARRAY_CASE(complex_float_t, complex_double_t); break;
				case 52: PRINT_ARRAY_CASE(complex_double_t, complex_double_t); break;
				default: std::cerr << "Unknown number type: " << num_type << std::endl; break;
				}
#undef PRINT_ARRAY_CASE
				ss << std::endl;
				break;
			}
			default:
				std::cerr << "Unknown type: " << (int)token.type << std::endl;
				break;
			}
		}

		void print() const {
			print(std::cout);
		}
	};

	struct Parser {
		const uint8_t* buffer; // the buffer to read
		size_t pos = 0;
		size_t size = 0; // the size of the buffer
		parse_error err = parse_error::ok; // `ok` means no complaint was raised
		std::vector<Token> tokens;

		Parser(const uint8_t* buf, const size_t len) : buffer(buf), pos(0), size(len), err(parse_error::ok) {}
		Parser(const std::vector<uint8_t>& buf) : buffer(buf.data()), pos(0), size(buf.size()), err(parse_error::ok) {}
		Parser(const std::string_view buf) : buffer((const uint8_t*)buf.data()), pos(0), size(buf.size()), err(parse_error::ok) {}

		// default special member functions
		Parser() = default;
		~Parser() = default;
		Parser(const Parser&) = default;
		Parser& operator=(const Parser&) = default;
		Parser(Parser&&) noexcept = default;
		Parser& operator=(Parser&&) noexcept = default;

		// Read one base-128 varint.  A varint runs off the end of the buffer only
		// for truncated input, and then there is no value to return: `err` is set
		// and the caller must stop.  `pos` is always left at a readable position.
		inline uint64_t read_varint() {
			const uint8_t* const begin = buffer;
			const uint8_t* ptr = buffer + pos;
			const uint8_t* const end = buffer + size;
			uint64_t result = 0;
			int shift = 0;

			while (ptr < end && shift < 64) {
				const uint8_t b = *ptr++;
				result |= uint64_t(b & 0x7F) << shift;
				if (!(b & 0x80)) {
					pos = size_t(ptr - begin);
					return result;
				}
				shift += 7;
			}

			// either the buffer ended on a continuation byte, or ten continuation
			// bytes claimed a value that needs more than 64 bits
			pos = size_t(ptr - begin);
			err = parse_error::truncated;
			return 0;
		}

		void parse() {
			// check the file head
			if (pos == 0) {
				if (size < 2 || buffer == nullptr || buffer[0] != 56 || buffer[1] != 58) {
					std::cerr << "Invalid WXF file" << std::endl;
					err = parse_error::invalid_head;
					return;
				}
				pos = 2;
			}

			// a declared byte count is only usable while the whole payload is
			// inside the buffer
			auto payload_fits = [&](const size_t length) {
				if (length <= size - pos)
					return true;
				std::cerr << "Truncated WXF data: " << length
					<< " bytes declared, " << size - pos << " available at pos " << pos << std::endl;
				err = parse_error::bad_length;
				return false;
			};

			while (pos < size && parse_ok(err)) {
				WXF_HEAD type = (WXF_HEAD)(buffer[pos]); pos++;

				switch (type) {
				case WXF_HEAD::i8:
				case WXF_HEAD::i16:
				case WXF_HEAD::i32:
				case WXF_HEAD::i64:
				case WXF_HEAD::f64: {
					auto length = size_of_head_num_type(type);
					if (!payload_fits(length))
						return;
					tokens.emplace_back(type, length, buffer + pos);
					pos += length;
					break;
				}
				case WXF_HEAD::symbol:
				case WXF_HEAD::bigint:
				case WXF_HEAD::bigreal:
				case WXF_HEAD::string:
				case WXF_HEAD::binary_string: {
					auto length = read_varint();
					if (!parse_ok(err))
						return;
					if (!payload_fits(length))
						return;
					tokens.emplace_back(type, length, buffer + pos);
					pos += length;
					break;
				}
				case WXF_HEAD::func:
				case WXF_HEAD::association: {
					auto length = read_varint();
					if (!parse_ok(err))
						return;
					// the head of a function is a symbol that follows immediately
					const size_t head_len = (type == WXF_HEAD::func) ? 1 : 0;
					if (!payload_fits(head_len))
						return;
					tokens.emplace_back(type, length, buffer + pos);
					break;
				}
				case WXF_HEAD::delay_rule:
				case WXF_HEAD::rule:
					tokens.emplace_back(type, size_t(2), buffer + pos);
					break;
				case WXF_HEAD::array:
				case WXF_HEAD::narray: {
					const auto num_type = read_varint();
					if (!parse_ok(err))
						return;
					const auto r = read_varint();
					if (!parse_ok(err))
						return;
					if (r > wxf_max_rank) {
						std::cerr << "Invalid array rank: " << r << " at pos " << pos << std::endl;
						err = parse_error::bad_rank;
						return;
					}
					std::vector<size_t> dims(r);
					size_t all_len = 1;
					for (size_t i = 0; i < r; i++) {
						dims[i] = read_varint();
						if (!parse_ok(err))
							return;
						// the flattened length has to stay representable
						if (dims[i] != 0 && all_len > (std::numeric_limits<size_t>::max)() / dims[i]) {
							std::cerr << "Invalid array: the element count overflows at pos " << pos << std::endl;
							err = parse_error::bad_length;
							return;
						}
						all_len *= dims[i];
					}
					const size_t elem_size = size_of_arr_num_type(int(num_type));
					if (all_len > (size - pos) / elem_size) {
						std::cerr << "Truncated WXF data: " << all_len << " elements of "
							<< elem_size << " bytes declared, " << size - pos
							<< " bytes available at pos " << pos << std::endl;
						err = parse_error::bad_length;
						return;
					}
					const uint8_t* const data = buffer + pos;
					pos += all_len * elem_size;
					tokens.emplace_back(type, dims, int(num_type), all_len, data);
					break;
				}
				default:
					std::cerr << "Unknown head type: " << (int)type << " pos: " << pos << std::endl;
					err = parse_error::unknown_type;
					return;
				}
			}

			// a stream with a valid head but no token describes no expression at all
			if (parse_ok(err) && tokens.empty()) {
				std::cerr << "Empty WXF data: the head is valid but no token follows" << std::endl;
				err = parse_error::no_data;
			}
		}
	};

	struct expr_node {
		size_t index; // the index of the token in the tokens vector
		std::vector<expr_node> children;
		WXF_HEAD type;

		expr_node() : index(0), children(), type(WXF_HEAD::i8) {} // default constructor
		~expr_node() = default;

		expr_node(const expr_node&) = default; // copy constructor
		expr_node& operator=(const expr_node&) = default; // copy assignment operator
		expr_node(expr_node&& other) = default; // move constructor
		expr_node& operator=(expr_node&& other) = default; // move assignment operator

		expr_node(size_t idx, size_t sz, WXF_HEAD t) : index(idx), type(t) {
			if (sz > 0) 
				children.resize(sz);
		}

		size_t size() const { return children.size(); }
		bool has_children() const { return children.size() > 0; }
		const expr_node& operator[] (size_t i) const { return children[i]; }
		expr_node& operator[] (size_t i) { return children[i]; }
	};

	struct expr_tree {
		std::vector<Token> tokens;
		expr_node root;

		expr_tree() {} // default constructor
		expr_tree(Parser parser, size_t index, size_t size, WXF_HEAD type) : root(index, size, type) {
			tokens = std::move(parser.tokens);
		}

		expr_tree(const expr_tree&) = default; //  copy constructor
		expr_tree& operator=(const expr_tree&) = default; // copy assignment operator
		expr_tree(expr_tree&&) noexcept = default; // move constructor
		expr_tree& operator=(expr_tree&&) noexcept = default; // move assignment operator
		~expr_tree() = default;

		const Token& operator[](const expr_node& node) const {
			return tokens[node.index];
		}

		void print(std::ostream& ss, const expr_node& node, const int level = 0) const {
			for (int i = 0; i < level; i++)
				ss << "  ";
			ss << "Node type: " << (int)node.type << ", index: " << node.index << ", size: " << node.size() << std::endl;
			for (size_t i = 0; i < node.size(); i++) {
				print(ss, node.children[i], level + 1);
			}
		}

		void print(std::ostream& ss) const {
			print(ss, root, 0);
		}

		void print() const {
			print(std::cout);
		}
	};

	inline expr_tree make_expr_tree(Parser& parser) {
		expr_tree tree;
		if (!parse_ok(parser.err))
			return tree;

		tree.tokens = std::move(parser.tokens);

		auto total_len = tree.tokens.size();
		auto& tokens = tree.tokens;

		if (total_len == 0) {
			std::cerr << "Error: the input contains no token" << std::endl;
			parser.err = parse_error::no_data;
			return tree;
		}

		std::vector<expr_node*> expr_stack; // the stack to store the current father nodes
		std::vector<size_t> node_stack; // the vector to store the node index

		std::function<void(void)> move_to_next_node = [&]() {
			if (node_stack.empty())
				return;

			node_stack.back()++; // move to the next node
			if (node_stack.back() >= expr_stack.back()->size()) {
				expr_stack.pop_back(); // pop the current node
				node_stack.pop_back(); // pop the current node index
				move_to_next_node();
			}
			};

		// first we need to find the root node
		size_t pos = 0;
		auto& token = tokens[pos];
		if (token.type == WXF_HEAD::func) {
			// i + 1 is the head of the function (a symbol)
			tree.root = expr_node(pos + 1, token.length, token.type);
			pos += 2; // skip the head
		}
		else if (token.type == WXF_HEAD::association) {
			// association does not have a head
			tree.root = expr_node(pos + 1, token.length, token.type);
			pos += 1;
		}
		else {
			// if the token is not a function type, only one token is allowed
			tree.root = expr_node(pos, 0, token.type);
			return tree;
		}

		expr_stack.push_back(&(tree.root));
		node_stack.push_back(0);

		// a node that declares no children is complete the moment it is created:
		// move_to_next_node() only runs once a child has been stored, so without
		// this List[] (or any f[] / empty Association) would stay on the stack
		// and the whole tree would be reported as malformed
		if (tree.root.size() == 0)
			move_to_next_node();

		// now we need to parse the expression
		for (; pos < total_len; pos++) {
			if (node_stack.empty())
				break; // the declared tree is already complete; extra tokens are ignored
			auto& tok = tokens[pos];
			if (tok.type == WXF_HEAD::func || tok.type == WXF_HEAD::association) {
				// if the token is a function type, we need to create a new node
				auto node_pos = node_stack.back();
				auto parent = expr_stack.back();
				if (node_pos >= parent->size()) {
					std::cerr << "Error: malformed expression tree at token " << pos << std::endl;
					parser.err = parse_error::bad_structure;
					break;
				}
				auto& node = parent->children[node_pos];
				if (tok.type == WXF_HEAD::func) {
					node = expr_node(pos + 1, tok.length, tok.type);
					pos++; // skip the head
				}
				else
					node = expr_node(pos, tok.length, tok.type);
				expr_stack.push_back(&(node)); // push the new node to the stack
				node_stack.push_back(0); // push the new node index to the stack
				if (node.size() == 0) // f[] : nothing will ever complete this node
					move_to_next_node();
			}
			else if (tok.type == WXF_HEAD::delay_rule || tok.type == WXF_HEAD::rule) {
				// if the token is a rule type, we need to create a new node
				auto node_pos = node_stack.back();
				auto parent = expr_stack.back();
				if (node_pos >= parent->size()) {
					std::cerr << "Error: malformed expression tree at token " << pos << std::endl;
					parser.err = parse_error::bad_structure;
					break;
				}
				auto& node = parent->children[node_pos];
				node = expr_node(pos, 2, tok.type);
				expr_stack.push_back(&(node)); // push the new node to the stack
				node_stack.push_back(0); // push the new node index to the stack
			}
			else {
				// if the token is not a function type, we need to move to the next node
				auto node_pos = node_stack.back();
				auto parent = expr_stack.back();
				if (node_pos >= parent->size()) {
					std::cerr << "Error: malformed expression tree at token " << pos << std::endl;
					parser.err = parse_error::bad_structure;
					break;
				}
				auto& node = parent->children[node_pos];
				node = expr_node(pos, 0, tok.type);

				move_to_next_node();
			}
		}

		if (!node_stack.empty()) {
			std::cerr << "Error: not all nodes are parsed" << std::endl;
			parser.err = parse_error::bad_structure;
		}

		return tree;
	}

	inline expr_tree make_expr_tree(const uint8_t* str, const size_t len) {
		Parser parser(str, len);
		parser.parse();
		return make_expr_tree(parser);
	}

	inline expr_tree make_expr_tree(const std::vector<uint8_t>& str) {
		Parser parser(str);
		parser.parse();
		return make_expr_tree(parser);
	}

	inline expr_tree make_expr_tree(const std::string_view str) {
		Parser parser(str);
		parser.parse();
		return make_expr_tree(parser);
	}

} // namespace WXF_PARSER

/***********************************************************************************/

// a simple FullForm parser, we only support [0-9,a-z,A-Z,$] in symbol names
// it is used to write a template engine to generate WXF files
// and we add a special expression type starting with # for subexpression labels
// e.g. #x for a integer x, e.g. #1, #2, ...
// and then we can replace these labels with actual expressions when generating WXF files

// !! do not use for high percision numbers or very large integers, it only supports standard C++ number formats
// !! it is not efficient and robust enough now, make your template simple
// !! and handle complicated sub-expressions manually by using push_** in Encoder

namespace WXF_PARSER::FullForm {
	enum class atom_type {
		Integer,
		Real,
		String,
		Symbol,
		Expression,
		Null
	};

	struct atom_expression {
		atom_type type_;
		std::string value_;

		atom_expression(atom_type type, const std::string_view value)
			: type_(type), value_(value) {
		}

		const atom_type get_type() const { return type_; }
		const std::string& get_value() const { return value_; }

		std::string to_FullForm() const {
			switch (type_) {
			case atom_type::String:
				return "\"" + value_ + "\"";
			default:
				return value_;
			}
		}
	};

	struct lexer {
		std::string input_;
		size_t position_;
		size_t length_;

		char current_char() const {
			return position_ < length_ ? input_[position_] : '\0';
		}

		void advance() {
			if (position_ < length_) position_++;
		}

		void skip_ws() {
			while (position_ < length_ && std::isspace(static_cast<unsigned char>(current_char()))) {
				advance();
			}
		}

		enum token_type {
			IDENTIFIER,    // 
			EXPRESSION,    // start with #, use as a label of subexpression, it is not standard mathematica 
			INTEGER,       // 
			REAL,          // 
			STRING,        // 
			LBRACKET,      // [
			RBRACKET,      // ]
			COMMA,         // ,
			END            // 
		};

		struct token {
			token_type type;
			std::string value;
			size_t position;
		};

		lexer(const std::string_view input)
			: input_(input), position_(0), length_(input.length()) {
		}

		token nextToken() {
			skip_ws();

			if (position_ >= length_) {
				return { END, "", position_ };
			}

			size_t startPos = position_;
			char ch = current_char();

			// $ 
			if (std::isalpha(ch) || ch == '$') {
				std::string value;
				while (position_ < length_ &&
					(std::isalnum(static_cast<unsigned char>(current_char())) || current_char() == '$')) {
					value += current_char();
					advance();
				}
				return { IDENTIFIER, value, startPos };
			}

			// expression starting with #
			if (ch == '#') {
				std::string value;
				value += ch;
				advance();
				while (position_ < length_ &&
					(std::isalnum(static_cast<unsigned char>(current_char())) || current_char() == '$')) {
					value += current_char();
					advance();
				}
				return { EXPRESSION, value, startPos };
			}

			// number
			if (std::isdigit(static_cast<unsigned char>(ch)) || ch == '.' || ch == '-') {
				std::string value;
				bool hasDot = false;
				bool hasE = false;

				// minus sign
				if (ch == '-') {
					value += ch;
					advance();
					if (position_ >= length_ || !std::isdigit(static_cast<unsigned char>(current_char()))) {
						// single minus sign, return as IDENTIFIER
						return { IDENTIFIER, value, startPos };
					}
					ch = current_char();
				}

				// integer part
				while (position_ < length_ && std::isdigit(static_cast<unsigned char>(current_char()))) {
					value += current_char();
					advance();
				}

				// decimal part
				if (position_ < length_ && current_char() == '.') {
					hasDot = true;
					value += '.';
					advance();
					while (position_ < length_ && std::isdigit(static_cast<unsigned char>(current_char()))) {
						value += current_char();
						advance();
					}
				}

				// scientific notation
				if (position_ < length_ && (current_char() == 'e' || current_char() == 'E')) {
					hasE = true;
					value += current_char();
					advance();

					if (position_ < length_ && (current_char() == '+' || current_char() == '-')) {
						value += current_char();
						advance();
					}

					if (position_ >= length_ || !std::isdigit(static_cast<unsigned char>(current_char()))) {
						std::cerr << "Invalid scientific notation at position " << position_ << std::endl;
						return { END, "", startPos };
					}

					while (position_ < length_ && std::isdigit(static_cast<unsigned char>(current_char()))) {
						value += current_char();
						advance();
					}
				}

				if (hasDot || hasE) {
					return { REAL, value, startPos };
				}
				else {
					return { INTEGER, value, startPos };
				}
			}

			// string
			if (ch == '"') {
				std::string value;
				advance();

				while (position_ < length_ && current_char() != '"') {
					if (current_char() == '\\' && position_ + 1 < length_) {
						advance();
						char next = current_char();
						switch (next) {
						case 'n': value += '\n'; break;
						case 't': value += '\t'; break;
						case 'r': value += '\r'; break;
						case '"': value += '"'; break;
						case '\\': value += '\\'; break;
						default: value += '\\'; value += next; break;
						}
					}
					else {
						value += current_char();
					}
					advance();
				}

				if (position_ >= length_ || current_char() != '"') {
					std::cerr << "Unterminated string at position " << startPos << std::endl;
					return { END, "", startPos };
				}
				advance();  // skip closing "

				return { STRING, value, startPos };
			}

			// single-character tokens
			switch (ch) {
			case '[':
				advance();
				return { LBRACKET, "[", startPos };
			case ']':
				advance();
				return { RBRACKET, "]", startPos };
			case ',':
				advance();
				return { COMMA, ",", startPos };
			}

			std::cerr << "Unknown character '" << ch << "' at position " << position_ << std::endl;
			return { END, "", position_ };
		}
	};

	struct expression {
		atom_expression head_;
		std::vector<expression> args_;

		expression(const atom_expression& head) : head_(head), args_() {}
		expression(const atom_expression& head, const std::vector<expression>& args) : head_(head), args_(args) {}
		bool is_atom() const { return args_.size() == 0; }

		std::string to_FullForm() const {
			if (is_atom()) {
				return head_.to_FullForm();
			}
			else {
				std::string result = head_.to_FullForm() + "[";
				for (size_t i = 0; i < args_.size(); ++i) {
					result += args_[i].to_FullForm();
					if (i + 1 < args_.size()) {
						result += ", ";
					}
				}
				result += "]";
				return result;
			}
		}
	};

	struct parser {
		lexer lexer_;
		lexer::token currentToken_;

		void consume(lexer::token_type expected) {
			if (currentToken_.type == expected) {
				currentToken_ = lexer_.nextToken();
			}
			else {
				std::cerr << "Parse Error: Unexpected token '" << currentToken_.value
					<< "', expected " << static_cast<int>(expected)
					<< " at position " << currentToken_.position << std::endl;
			}
		}

		atom_expression parse_atom() {
			std::string value = currentToken_.value;

			switch (currentToken_.type) {
			case lexer::IDENTIFIER: {
				consume(lexer::IDENTIFIER);
				return atom_expression(atom_type::Symbol, value);
			}
			case lexer::EXPRESSION: {
				consume(lexer::EXPRESSION);
				return atom_expression(atom_type::Expression, value);
			}
			case lexer::INTEGER: {
				consume(lexer::INTEGER);
				return atom_expression(atom_type::Integer, value);
			}
			case lexer::REAL: {
				consume(lexer::REAL);
				return atom_expression(atom_type::Real, value);
			}
			case lexer::STRING: {
				consume(lexer::STRING);
				return atom_expression(atom_type::String, value);
			}
			default:
				std::cerr << "Parse Error: Unexpected token '" << currentToken_.value
					<< "' at position " << currentToken_.position << std::endl;
				return atom_expression(atom_type::Null, "");
			}
		}

		expression parse_expression(const size_t depth = 0) {
			if (depth > wxf_max_fullform_depth) {
				std::cerr << "Parse Error: FullForm is nested deeper than "
					<< wxf_max_fullform_depth << " levels at position " << currentToken_.position << std::endl;
				currentToken_ = { lexer::END, "", currentToken_.position };
				return atom_expression(atom_type::Null, "");
			}

			atom_expression head = parse_atom();

			if (currentToken_.type == lexer::LBRACKET) {
				consume(lexer::LBRACKET);
				std::vector<expression> args;

				if (currentToken_.type != lexer::RBRACKET) {
					args.push_back(parse_expression(depth + 1));

					while (currentToken_.type == lexer::COMMA) {
						consume(lexer::COMMA);
						args.push_back(parse_expression(depth + 1));
					}
				}

				// if we meet expr like f[], we use a null atom as argument
				if (args.size() == 0) {
					args.push_back(expression(atom_expression(atom_type::Null, "")));
				}

				consume(lexer::RBRACKET);
				return expression(head, args);
			}

			return head;
		}

		parser(const std::string_view input) : lexer_(input) {
			currentToken_ = lexer_.nextToken();
		}

		expression parse() {
			expression result = parse_expression();

			if (currentToken_.type != lexer::END) {
				std::cerr << "Parse Error: Unexpected token at end: " << currentToken_.value << std::endl;
			}

			return result;
		}
	};

	// we allow use { }, so we need to convert { } to List[ ]
	inline expression parse_FullForm(const std::string_view str) {
		std::string mod_str;
		mod_str.reserve(str.size() + 10);
		for (auto c : str) {
			if (c == '{')
				mod_str += "List[";
			else if (c == '}')
				mod_str += "]";
			else
				mod_str += c;
		}

		parser parser(mod_str);
		return parser.parse();
	}
} // namespace WXF_PARSER::FullForm

namespace WXF_PARSER {
	// we allow use a map to store function that generating sub-expressions
	inline void fullform_to_wxf(Encoder& encoder, const FullForm::expression& expr,
		const std::unordered_map<std::string, std::function<void(Encoder&)>>& map,
		const size_t depth = 0) {

		if (depth > wxf_max_fullform_depth) {
			std::cerr << "Error: FullForm is nested deeper than " << wxf_max_fullform_depth << " levels." << std::endl;
			return;
		}

		if (expr.is_atom()) {
			switch (expr.head_.get_type()) {
			case FullForm::atom_type::Integer:
				encoder.push_integer(std::stoll(expr.head_.get_value()));
				break;
			case FullForm::atom_type::Real:
				encoder.push_real(std::stod(expr.head_.get_value()));
				break;
			case FullForm::atom_type::String:
				encoder.push_string(expr.head_.get_value());
				break;
			case FullForm::atom_type::Symbol:
				encoder.push_symbol(expr.head_.get_value());
				break;
			case FullForm::atom_type::Null:
				break;
			case FullForm::atom_type::Expression: {
				auto& vv = expr.head_.get_value();
				auto it = map.find(vv);
				if (it != map.end())
					it->second(encoder);
				else
					std::cerr << "Error: expression id " << vv << " not found in map." << std::endl;
				break;
			}
			default:
				std::cerr << "Error: unknown atom type." << std::endl;
				break;
			}
		}
		else {
			size_t len = expr.args_.size();
			if (expr.args_[0].is_atom() &&
				expr.args_[0].head_.get_type() == FullForm::atom_type::Null) {
				len = 0;
			}
			std::string_view name = expr.head_.get_value();

			// Rule or DelayRule is not special in FullForm, we treat them as normal functions
			encoder.push_function(name, len);

			for (size_t i = 0; i < len; i++) {
				fullform_to_wxf(encoder, expr.args_[i], map, depth + 1);
			}
		}
	}

	inline void fullform_to_wxf(Encoder& encoder, const FullForm::expression& expr,
		const std::unordered_map<std::string, Encoder>& map) {
		std::unordered_map<std::string, std::function<void(Encoder&)>> func_map;
		for (const auto& [key, value] : map) {
			func_map[key] = [&value](Encoder& enc) {
				enc.push_ustr(value.buffer);
				};
		}
		fullform_to_wxf(encoder, expr, func_map);
	}

	template<typename MapType>
	Encoder fullform_to_wxf(const std::string_view ff_template, const MapType& map, bool include_head = true) {
		Encoder encoder;
		encoder.buffer.reserve(ff_template.size() * 32); // reserve some space
		if (include_head) {
			encoder.buffer.push_back(56); // WXF head
			encoder.buffer.push_back(58); // WXF head
		}
		fullform_to_wxf(encoder, FullForm::parse_FullForm(ff_template), map);
		return encoder;
	}
} // namespace WXF_PARSER