/*
	Copyright (C) 2025 Zhenjie Li (Li, Zhenjie)

	This file is part of SparseRREF. The SparseRREF is free software:
	you can redistribute it and/or modify it under the terms of the MIT
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

	We here intend to support import and export a SparseArray expression with rational
	entries, so some types are not supported, such as complex numbers.
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
	-		193			packed array
	-		194			numeric array
	*		65			association
	*		58			delayed rule in association
	*		45			rule in association

	* is supported, - is partially supported

	the struct of numeric/packed array is: num_type, rank, dimensions, data
	for the num_type
	0 is int8_t      1 is int16_t
	2 is int32_t     3 is int64_t
	16 is uint8_t    17 is uint16_t ; only for numeric array
	18 is uint32_t   19 is uint64_t ; only for numeric array
	34 float         35 double
	51 complex float 52 complex double
	we only support int8_t, int16_t, int32_t, int64_t, float, double
*/

#pragma once

#include <iostream>
#include <filesystem>
#include <fstream>
#include <functional>
#include <vector>
#include <string>
#include <cstdint>
#include <cstring>
#include <variant>

namespace WXF_PARSER {

	enum WXF_HEAD {
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

	template <typename T>
	inline int minimal_signed_bits(T x) noexcept {
		if (x >= INT8_MIN && x <= INT8_MAX) return 0;
		if (x >= INT16_MIN && x <= INT16_MAX) return 1;
		if (x >= INT32_MIN && x <= INT32_MAX) return 2;
		return 3; // for int64_t
	}

	template <typename T>
	inline int minimal_unsigned_bits(T x) noexcept {
		if (x <= UINT8_MAX) return 0;
		if (x <= UINT16_MAX) return 1;
		if (x <= UINT32_MAX) return 2;
		return 3; // for uint64_t
	}

	struct TOKEN {
		WXF_HEAD type;
		int rank = 0;
		union { 
			// for number, string, symbol, bigint
			uint64_t length;
			// for array and narray, dimensions[0] is the type, dimensions[1] is the total flatten length
			// so the length is dimensions is rank + 2
			uint64_t* dimensions; 
		};
		union { // data
			// number 
			int8_t i8;
			int16_t i16;
			int32_t i32;
			int64_t i64; // for i8, i16, i32, i64
			double d; 
			// array of numbers
			int8_t* i8_arr; 
			int16_t* i16_arr;
			int32_t* i32_arr; 
			int64_t* i64_arr;
			uint8_t* u8_arr;
			uint16_t* u16_arr;
			uint32_t* u32_arr;
			uint64_t* u64_arr; 
			float* f_arr;
			double* d_arr; 
			char* str; 
		};

		TOKEN() : type(WXF_HEAD::i8), rank(0), length(0), i64(0) {}

		uint64_t dim(int i) const {
			if (rank > 0) 
				return dimensions[i + 2];
			return length;
		}

		void clear() {
			if (type == WXF_HEAD::symbol 
				|| type == WXF_HEAD::bigint
				|| type == WXF_HEAD::bigreal
				|| type == WXF_HEAD::string
				|| type == WXF_HEAD::binary_string) {
				free(str);
			}
			else if (type == WXF_HEAD::array || type == WXF_HEAD::narray) {
				free(i8_arr);
				free(dimensions);
			}
			// no need to clear i, length, rank, type, as they are just basic types
		}

		~TOKEN() { clear(); }

		// disable copy constructor and copy assignment operator
		TOKEN(const TOKEN&) = delete; 
		TOKEN& operator=(const TOKEN&) = delete; 

		// move constructor
		TOKEN(TOKEN&& other) noexcept : type(other.type), rank(other.rank), length(other.length), i64(other.i64) {
			if (type == WXF_HEAD::symbol 
				|| type == WXF_HEAD::bigint 
				|| type == WXF_HEAD::bigreal
				|| type == WXF_HEAD::string 
				|| type == WXF_HEAD::binary_string) {
				str = other.str;
				other.str = nullptr;
			}
			else if (type == WXF_HEAD::array || type == WXF_HEAD::narray) {
				dimensions = other.dimensions;
				i8_arr = other.i8_arr; // for array, since it is union, we can use i8_arr for narray
				other.dimensions = nullptr;
				other.i8_arr = nullptr;
			}
		}

		int64_t get_int() const {
			switch (type) {
			case WXF_HEAD::i8: return i8;
			case WXF_HEAD::i16: return i16;
			case WXF_HEAD::i32: return i32;
			case WXF_HEAD::i64: return i64;
			default:
				std::cerr << "Cannot get int from non-integer token" << std::endl;
				return 0;
			}
		}

		uint8_t get_array_size() const {
			if (type != WXF_HEAD::array && type != WXF_HEAD::narray) {
				std::cerr << "Cannot copy array from non-array token" << std::endl;
				return 0;
			}

			switch (dimensions[0]) {
			case 0:  return sizeof(int8_t); 
			case 1:  return sizeof(int16_t); 
			case 2:  return sizeof(int32_t); 
			case 3:  return sizeof(int64_t); 
			case 16: return sizeof(uint8_t); 
			case 17: return sizeof(uint16_t); 
			case 18: return sizeof(uint32_t); 
			case 19: return sizeof(uint64_t); 
			case 34: return sizeof(float);
			case 35: return sizeof(double);
			default:
				std::cerr << "Unsupported type in array/narray" << std::endl;
				return 0;
			}
		}

		// symbol/bigint/string/binary_string, length, str
		TOKEN(WXF_HEAD t, const std::string& s) : type(t), rank(0) {
			length = s.size();
			str = (char*)malloc(length + 1);
			std::memcpy(str, s.data(), length);
			str[length] = '\0'; // null-terminate the string
		}

		// machine number, val (length is given by the sizeof(val))
		TOKEN(WXF_HEAD t, int8_t v) : type(t), rank(0), i8(v), length(1) {}
		TOKEN(WXF_HEAD t, int16_t v) : type(t), rank(0), i16(v), length(2) {}
		TOKEN(WXF_HEAD t, int32_t v) : type(t), rank(0), i32(v), length(3) {}
		TOKEN(WXF_HEAD t, int64_t v) : type(t), rank(0), i64(v), length(4) {}
		TOKEN(WXF_HEAD t, float v) : type(t), rank(0), d(v), length(2) {}
		TOKEN(WXF_HEAD t, double v) : type(t), rank(0), d(v), length(4) {}

		// array/narray
		TOKEN(WXF_HEAD t, const std::vector<size_t>& dims, int num_type, size_t len) : type(t) {
			int r = dims.size();
			rank = r;
			dimensions = (uint64_t*)malloc((r + 2) * sizeof(uint64_t));
			dimensions[0] = num_type;
			dimensions[1] = len;
			for (auto i = 0; i < r; i++) {
				dimensions[i + 2] = dims[i];
			}
			i8_arr = (int8_t*)malloc(len * get_array_size());
		}

		template<typename T>
		void print(T& ss) const {
			auto& token = *this;
			switch (token.type) {
				case WXF_HEAD::i8:
					ss << "i8: " << (int)token.i8 << std::endl;
					break;
				case WXF_HEAD::i16:
					ss << "i16: " << token.i16 << std::endl;
					break;
				case WXF_HEAD::i32:
					ss << "i32: " << token.i32 << std::endl;
					break;
				case WXF_HEAD::i64:
					ss << "i64: " << token.i64 << std::endl;
					break;
				case WXF_HEAD::f64:
					ss << "f64: " << token.d << std::endl;
					break;
				case WXF_HEAD::symbol:
					ss << "symbol: " << token.str << std::endl;
					break;
				case WXF_HEAD::bigint:
					ss << "bigint: " << token.str << std::endl;
					break;
				case WXF_HEAD::bigreal:
					ss << "bigreal: " << token.str << std::endl;
					break;
				case WXF_HEAD::string:
					ss << "string: " << token.str << std::endl;
					break;
				case WXF_HEAD::binary_string:
					ss << "binary_string: " << token.str << std::endl;
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
				case WXF_HEAD::array: 
				case WXF_HEAD::narray: {
					if (token.type == WXF_HEAD::array) 
						ss << "array: ";
					else 
						ss << "narray: ";

					ss << "rank = " << token.rank << ", dimensions = ";
					for (int i = 0; i < token.rank; i++) {
						ss << token.dimensions[i + 2] << " ";
					}
					ss << std::endl;

					size_t num_type = token.dimensions[0];
					size_t all_len = token.dimensions[1];

					ss << "data: ";
					switch (num_type) {
					case 0: for (size_t i = 0; i < all_len; i++) { ss << (int)token.i8_arr[i] << " "; } break;
					case 1: for (size_t i = 0; i < all_len; i++) { ss << token.i16_arr[i] << " "; } break;
					case 2: for (size_t i = 0; i < all_len; i++) { ss << token.i32_arr[i] << " "; } break;
					case 3: for (size_t i = 0; i < all_len; i++) { ss << token.i64_arr[i] << " "; } break;
					case 16: for (size_t i = 0; i < all_len; i++) { ss << (int)token.u8_arr[i] << " "; } break;
					case 17: for (size_t i = 0; i < all_len; i++) { ss << token.u16_arr[i] << " "; } break;
					case 18: for (size_t i = 0; i < all_len; i++) { ss << token.u32_arr[i] << " "; } break;
					case 19: for (size_t i = 0; i < all_len; i++) { ss << token.u64_arr[i] << " "; } break;
					case 34: for (size_t i = 0; i < all_len; i++) { ss << token.f_arr[i] << " "; } break;
					case 35: for (size_t i = 0; i < all_len; i++) { ss << token.d_arr[i] << " "; } break;
					default:
						std::cerr << "Unsupported type in array/narray" << std::endl;
						break;
					}
					ss << std::endl;
					break;
				}
				default:
					std::cerr << "Unknown type" << std::endl;
			}
		}

		void print() const {
			print(std::cout);
		}
	};

	std::vector<uint8_t> toVarint(const uint64_t val) {
		std::vector<uint8_t> bytes;
		auto tmp_val = val;
		bytes.reserve(16); // 10 bytes is the maximum length of varint, but we reserve more for efficiency
		while (tmp_val > 0) {
			uint8_t byte = tmp_val & 127;
			tmp_val >>= 7;
			if (tmp_val > 0) byte |= 128; // set the continuation bit
			bytes.push_back(byte);
		}
		return bytes;
	}

	template <typename T>
	void serialize_binary(std::vector<uint8_t>& buffer, const T& value) {
		const size_t old_size = buffer.size();
		buffer.resize(old_size + sizeof(T));
		std::memcpy(buffer.data() + old_size, &value, sizeof(T));
	}

	struct Parser {
		const uint8_t* buffer; // the buffer to read
		size_t pos = 0;
		size_t size = 0; // the size of the buffer
		std::vector<TOKEN> tokens; // the tokens read from the buffer

		Parser(const uint8_t* buf, const size_t len) : buffer(buf), pos(0), size(len) {}
		Parser(const std::vector<uint8_t>& buf) : buffer(buf.data()), pos(0), size(buf.size()) {}

		// we suppose that the length does not exceed 2^64 - 1 .. 
		uint64_t ReadVarint() {
			size_t count = 0;
			uint64_t result = 0;
			auto ptr = buffer + pos;

			while (pos < size && count < 8) {
				result |= (uint64_t)((*ptr) & 127) << (7 * count);
				count++; pos++;
				if (!((*ptr) & 128))
					break;
				ptr++;
			}

			return result;
		}

		void parseExpr() {
			// check the file head
			if (pos == 0) {
				if (size < 2 || buffer[0] != 56 || buffer[1] != 58) {
					std::cerr << "Invalid WXF file" << std::endl;
					return;
				}
				pos = 2; 
			}

			while (pos < size) {
				WXF_HEAD type = (WXF_HEAD)(buffer[pos]); pos++;

				if (pos == size)
					break;

				switch (type) {
					case WXF_HEAD::i8:
						tokens.push_back(makeNumber<int8_t>(type));
						break;
					case WXF_HEAD::i16:
						tokens.push_back(makeNumber<int16_t>(type));
						break;
					case WXF_HEAD::i32:
						tokens.push_back(makeNumber<int32_t>(type));
						break;
					case WXF_HEAD::i64:
						tokens.push_back(makeNumber<int64_t>(type));
						break;
					case WXF_HEAD::f64:
						tokens.push_back(makeNumber<double>(type));
						break;
					case WXF_HEAD::symbol: 
					case WXF_HEAD::bigint:
					case WXF_HEAD::bigreal:
					case WXF_HEAD::string:
					case WXF_HEAD::binary_string:
						tokens.push_back(makeString(type));
						break;
					case WXF_HEAD::func: 
					case WXF_HEAD::association:
						tokens.push_back(makeFunction(type));
						break;
					case WXF_HEAD::delay_rule:
					case WXF_HEAD::rule:
						tokens.push_back(makeRule(type));
						break;
					case WXF_HEAD::array: 
					case WXF_HEAD::narray:
						tokens.push_back(makeArray(type));
						break;
					default:
						std::cerr << "Unknown head type: " << (int)type << " pos: " << pos << std::endl;
						break;
				}
			}
		}

		// machine number, val (length is given by the sizeof(val))
		template <typename T>
		TOKEN makeNumber(WXF_HEAD type) {
			T val;
			std::memcpy(&val, buffer + pos, sizeof(val));
			pos += sizeof(T) / sizeof(uint8_t);
			
			return TOKEN(type, val);
		}

		// symbol/bigint/string/binary_string, length, str
		TOKEN makeString(WXF_HEAD type) {
			TOKEN node;
			node.type = type;
			node.length = ReadVarint();
			node.str = (char*)malloc((node.length + 1) * sizeof(char)); // +1 for null terminator
			std::memcpy(node.str, buffer + pos, node.length);
			node.str[node.length] = '\0'; // add null terminator
			pos += node.length;
			return node;
		}

		// func/association, length
		TOKEN makeFunction(WXF_HEAD type) {
			TOKEN node;
			node.type = type;
			node.length = ReadVarint();
			return node;
		}

		TOKEN makeRule(WXF_HEAD type) {
			TOKEN node;
			node.type = type;
			node.length = 2;
			return node;
		}

		TOKEN makeArray(WXF_HEAD type) {
			int num_type = (int)ReadVarint();
			int r = (int)ReadVarint();
			std::vector<size_t> dims(r);
			size_t all_len = 1;
			for (int i = 0; i < r; i++) {
				dims[i] = ReadVarint();
				all_len *= dims[i];
			}

			auto token = TOKEN(type, dims, num_type, all_len);
			auto ss = token.get_array_size();
			std::memcpy(token.i8_arr, buffer + pos, all_len * ss);
			pos += all_len * ss;
			return token;
		}
	};

	struct ExprNode {
		size_t index; // the index of the token in the tokens vector
		size_t size; // the size of the children
		std::unique_ptr<ExprNode[]> children; // the children of the node
		WXF_HEAD type; 

		ExprNode() : index(0), size(0), children(nullptr), type(WXF_HEAD::i8) {} // default constructor

		ExprNode(size_t idx, size_t sz, WXF_HEAD t) : index(idx), size(sz), type(t) {
			if (size > 0) {
				children = std::make_unique<ExprNode[]>(size);
			}
		}

		ExprNode(const ExprNode&) = delete; // disable copy constructor
		ExprNode& operator=(const ExprNode&) = delete; // disable copy assignment operator
		// move constructor
		ExprNode(ExprNode&& other) noexcept : index(other.index), size(other.size), 
			children(std::move(other.children)), type(other.type) {
			other.index = 0;
			other.size = 0;
			other.children = nullptr;
			other.type = WXF_HEAD::i8;
		}


		// move assignment operator
		ExprNode& operator=(ExprNode&& other) noexcept {
			if (this != &other) {
				index = other.index;
				size = other.size;
				children = std::move(other.children);
				type = other.type;
				other.size = 0;
				other.index = 0;
				other.children = nullptr;
				other.type = WXF_HEAD::i8;
			}
			return *this;
		}

		// destructor
		void clear() {
			index = 0;
			size = 0;
			type = WXF_HEAD::i8;
			if (children) {
				children.reset();
			}
		}

		~ExprNode(){
			clear();
		}

		bool has_children() const {
			return size > 0;
		}

		const ExprNode& operator[] (size_t i) const {
			return children[i];
		}

		ExprNode& operator[] (size_t i) {
			return children[i];
		}
	};

	//// debug only, print the small tree 
	//void printPrettyTree(const ExprNode& node, const std::string& prefix = "", bool isLast = true) {
	//	std::cout << prefix;
	//	std::cout << (isLast ? "└── " : "├── ");
	//	std::cout << node.index << std::endl;

	//	for (size_t i = 0; i < node.size; ++i) {
	//		bool lastChild = (i == node.size - 1);
	//		printPrettyTree(node.children[i], prefix + (isLast ? "    " : "│   "), lastChild);
	//	}
	//}

	// TODO
	void node_to_ustr(std::vector<uint8_t>& res, const std::vector<TOKEN>& tokens, const ExprNode& node) {
		auto& token = tokens[node.index];

		uint8_t short_buffer[16];
		auto toVarint = [&](uint64_t value) {
			uint8_t len = 0;
			auto tmp_val = value;
			while (tmp_val > 0) {
				uint8_t byte = tmp_val & 127;
				tmp_val >>= 7;
				if (tmp_val > 0) byte |= 128; // set the continuation bit
				short_buffer[len] = byte;
				len++;
			}
			return len; // return the length of the varint
			};

		auto push_symbol = [&](const std::string& str) {
			res.push_back(WXF_PARSER::symbol); res.push_back(str.size());
			res.insert(res.end(), str.begin(), str.end());
			};
		auto push_varint = [&](uint64_t size) {
			auto len = toVarint(size);
			res.insert(res.end(), short_buffer, short_buffer + len);
			};
		auto push_function = [&](const std::string& symbol, uint64_t size) {
			res.push_back(WXF_PARSER::func);
			push_varint(size);
			push_symbol(symbol);
			};

		switch (node.type) {
		case WXF_HEAD::i8:
			res.push_back(WXF_HEAD::i8);
			serialize_binary(res, token.i8);
			break;
		case WXF_HEAD::i16:
			res.push_back(WXF_HEAD::i16);
			serialize_binary(res, token.i16);
			break;
		case WXF_HEAD::i32:
			res.push_back(WXF_HEAD::i32);
			serialize_binary(res, token.i32);
			break;
		case WXF_HEAD::i64: 
			res.push_back(WXF_HEAD::i64);
			serialize_binary(res, token.i64);
			break;
		case WXF_HEAD::f64:
			res.push_back(WXF_HEAD::f64);
			serialize_binary(res, token.d);
			break;
		case WXF_HEAD::func:
		case WXF_HEAD::association: {
			res.push_back(node.type);
			push_varint(node.size);
			auto& head = tokens[node.index];
			res.push_back(head.type);
			push_varint(head.length);
			res.insert(res.end(), token.str, token.str + head.length);
			for (size_t i = 0; i < node.size; i++) {
				node_to_ustr(res, tokens, node.children[i]);
			}
			break;
		}
		case WXF_HEAD::rule:
		case WXF_HEAD::delay_rule:
			res.push_back(node.type);
			for (size_t i = 0; i < node.size; i++) {
				node_to_ustr(res, tokens, node.children[i]);
			}
			break;
		case WXF_HEAD::symbol:
		case WXF_HEAD::bigint:
		case WXF_HEAD::bigreal:
		case WXF_HEAD::string:
		case WXF_HEAD::binary_string:
			res.push_back(node.type);
			push_varint(token.length);
			res.insert(res.end(), token.str, token.str + token.length);
			break;
		case WXF_HEAD::array:
		case WXF_HEAD::narray: {
			res.push_back(node.type);
			res.push_back(token.dimensions[0]);
			push_varint(token.rank);
			for (auto i = 0; i < token.rank; i++) {
				push_varint(token.dimensions[i + 2]);
			}
			auto all_len = token.dimensions[1]; // total flatten length
			auto size_of_type = 0;
			switch (token.dimensions[0]) {
			case 0: size_of_type = sizeof(int8_t); break;
			case 1: size_of_type = sizeof(int16_t); break;
			case 2: size_of_type = sizeof(int32_t); break;
			case 3: size_of_type = sizeof(int64_t); break;
			case 16: size_of_type = sizeof(uint8_t); break;
			case 17: size_of_type = sizeof(uint16_t); break;
			case 18: size_of_type = sizeof(uint32_t); break;
			case 19: size_of_type = sizeof(uint64_t); break;
			case 34: size_of_type = sizeof(float); break;
			case 35: size_of_type = sizeof(double); break;
			default:
				std::cerr << "Unsupported type in array/narray: " << token.dimensions[0] << std::endl;
				return;
			}

			// we can use the same pointer for both array and narray, as they are union
			res.insert(res.end(), (uint8_t*)token.i8_arr, ((uint8_t*)token.i8_arr) + all_len * size_of_type);
			break;
		}
		default:
			break;
		}
	}

	struct ExprTree {
		std::vector<TOKEN> tokens;
		ExprNode root;

		ExprTree() {} // default constructor
		ExprTree(Parser parser, size_t index, size_t size, WXF_HEAD type) : root(index, size, type) {
			tokens = std::move(parser.tokens);
		}

		ExprTree(const ExprTree&) = delete; // disable copy constructor
		ExprTree& operator=(const ExprTree&) = delete; // disable copy assignment operator

		// move constructor
		ExprTree(ExprTree&& other) noexcept : tokens(std::move(other.tokens)), root(std::move(other.root)) {
			other.root.size = 0;
			other.root.index = 0;
			other.root.children = nullptr;
		}

		// move assignment operator
		ExprTree& operator=(ExprTree&& other) noexcept {
			if (this != &other) {
				tokens = std::move(other.tokens);
				root = std::move(other.root);
				other.root.size = 0;
				other.root.index = 0;
				other.root.children = nullptr;
			}
			return *this;
		}

		const TOKEN& operator[](const ExprNode& node) const {
			return tokens[node.index];
		}

		std::vector<uint8_t> to_ustr(bool include_head = true) const {
			std::vector<uint8_t> res;
			if (include_head) {
				res.push_back(56); // WXF head
				res.push_back(58); // WXF head
			}
			node_to_ustr(res, tokens, root);
			return res;
		}

		//void plot() const {
		//	printPrettyTree(root);
		//}
	};

	ExprTree MakeExprTree(Parser& parser) {
		ExprTree tree;
		tree.tokens = std::move(parser.tokens);
		auto total_len = tree.tokens.size();
		auto& tokens = tree.tokens;

		std::vector<ExprNode*> expr_stack; // the stack to store the current father nodes
		std::vector<size_t> node_stack; // the vector to store the node index

		std::function<void(void)> move_to_next_node = [&]() {
			if (node_stack.empty())
				return;

			node_stack.back()++; // move to the next node
			if (node_stack.back() >= expr_stack.back()->size) {
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
			tree.root = ExprNode(pos + 1, token.length, token.type);
			pos += 2; // skip the head
		}
		else if (token.type == WXF_HEAD::association) {
			// association does not have a head
			tree.root = ExprNode(pos + 1, token.length, token.type);
			pos += 1; 
		}
		else {
			// if the token is not a function type, only one token is allowed
			tree.root = ExprNode(pos, 0, token.type);
			return tree;
		}

		expr_stack.push_back(&(tree.root));
		node_stack.push_back(0);

		// now we need to parse the expression
		for (; pos < total_len; pos++) {
			auto& token = tokens[pos];
			if (token.type == WXF_HEAD::func || token.type == WXF_HEAD::association) {
				// if the token is a function type, we need to create a new node
				auto node_pos = node_stack.back();
				auto parent = expr_stack.back();
				auto& node = parent->children[node_pos];
				if (token.type == WXF_HEAD::func) {
					node = ExprNode(pos + 1, token.length, token.type);
					pos++; // skip the head
				}
				else
					node = ExprNode(pos, token.length, token.type);
				expr_stack.push_back(&(node)); // push the new node to the stack
				node_stack.push_back(0); // push the new node index to the stack
			}
			else if (token.type == WXF_HEAD::delay_rule || token.type == WXF_HEAD::rule) {
				// if the token is a rule type, we need to create a new node
				auto node_pos = node_stack.back();
				auto parent = expr_stack.back();
				auto& node = parent->children[node_pos];
				node = ExprNode(pos, 2, token.type);
				expr_stack.push_back(&(node)); // push the new node to the stack
				node_stack.push_back(0); // push the new node index to the stack
			}
			else {
				// if the token is not a function type, we need to move to the next node
				auto node_pos = node_stack.back();
				auto parent = expr_stack.back();
				auto& node = parent->children[node_pos];
				node = ExprNode(pos, 0, token.type);

				move_to_next_node();
			}
		}

		if (!node_stack.empty()) {
			std::cerr << "Error: not all nodes are parsed" << std::endl;
			for (auto& node : expr_stack) {
				node->clear();
			}
		}

		return tree;
	}

	ExprTree MakeExprTree(const uint8_t* str, const size_t len) {
		Parser parser(str, len);
		parser.parseExpr();
		return MakeExprTree(parser);
	}

	ExprTree MakeExprTree(const std::vector<uint8_t>& str) {
		Parser parser(str);
		parser.parseExpr();
		return MakeExprTree(parser);
	}

	ExprTree MakeExprTree(const std::filesystem::path filename) {
		if (!std::filesystem::exists(filename)) {
			std::cerr << "Error: File does not exist!" << std::endl;
			return ExprTree();
		}
		ExprTree expr_tree;
		std::ifstream file(filename, std::ios::binary | std::ios::ate);
		std::streamsize size = file.tellg();
		file.seekg(0, std::ios::beg);

		std::vector<uint8_t> buffer(size);
		if (!file.read((char*)buffer.data(), size)) {
			std::cerr << "Failed to read file!" << std::endl;
			return ExprTree();
		}

		return MakeExprTree(buffer);
	}

	// debug only: convert the expression tree to DOT format of Graphviz
	template <typename SS>
	void toDotFormat(const ExprTree& tree, SS& oss) {
		oss << "digraph ExprTree {\n";
		oss << "  node [shape=box];\n";

		auto& tokens = tree.tokens;

		int nodeId = 0;
		std::function<void(const ExprNode&)> traverse = [&](const ExprNode& node) {
			int currentId = nodeId++;

			oss << "  n" << currentId << " [label=\"";
			switch (tokens[node.index].type) {
			case WXF_HEAD::symbol:
				oss << tokens[node.index].str;
				break;
			case WXF_HEAD::i8:
				oss << (int)tokens[node.index].i8;
				break;
			case WXF_HEAD::i16:
				oss << tokens[node.index].i16;
				break;
			case WXF_HEAD::i32:
				oss << tokens[node.index].i32;
				break;
			case WXF_HEAD::i64:
				oss << tokens[node.index].i64;
				break;
			case WXF_HEAD::f64:
				oss << tokens[node.index].d;
				break;
			case WXF_HEAD::bigint:
				oss << "bigint " << node.index;
				break;
			case WXF_HEAD::bigreal:
				oss << "bigreal " << node.index;
				break;
			case WXF_HEAD::string:
				oss << "string " << node.index;
				break;
			case WXF_HEAD::binary_string:
				oss << "binary_string " << node.index;
				break;
			case WXF_HEAD::array:
				oss << "array " << node.index;
				break;
			case WXF_HEAD::narray:
				oss << "narray " << node.index;
				break;
			default:
				oss << node.index;
				break;
			}
			oss << "\"];\n";

			for (size_t i = 0; i < node.size; ++i) {
				int childId = nodeId;
				traverse(node.children[i]);
				oss << "  n" << currentId << " -> n" << childId << ";\n";
			}
			};

		traverse(tree.root);
		oss << "}\n";
	}

	/*
		an example to test:
			SparseArray[{{1, 1} -> 1/3.0, {1, 23133} -> 
			N[Pi, 100] + I N[E, 100], {44, 2} -> -(4/
			 33333333333333444333333335), {_, _} -> 0}]

		FullForm: 
			SparseArray[Automatic,List[44,23133],0,
			List[1,List[List[0,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
			2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3],
			List[List[1],List[23133],List[2]]],
			List[0.3333333333333333`,
			Complex[3.1415926535897932384626433832795028841971693993751058209
			749445923078164062862089986280348253421170679821480865191976`100.,
			2.7182818284590452353602874713526624977572470936999595749669676277
			240766303535475945713821785251664274274663919320031`100.],
			Rational[-4,33333333333333444333333335]]]]

		std::vector<uint8_t> test{ 56, 58, 102, 4, 115, 11, 83, 112, 97, 114, 115, 101, 65, 114, 114, \
								97, 121, 115, 9, 65, 117, 116, 111, 109, 97, 116, 105, 99, 193, 1, 1, \
								2, 44, 0, 93, 90, 67, 0, 102, 3, 115, 4, 76, 105, 115, 116, 67, 1, \
								102, 2, 115, 4, 76, 105, 115, 116, 193, 0, 1, 45, 0, 2, 2, 2, 2, 2, \
								2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, \
								2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 193, 1, 2, 3, 1, 1, \
								0, 93, 90, 2, 0, 102, 3, 115, 4, 76, 105, 115, 116, 114, 85, 85, 85, \
								85, 85, 85, 213, 63, 102, 2, 115, 7, 67, 111, 109, 112, 108, 101, \
								120, 82, 122, 51, 46, 49, 52, 49, 53, 57, 50, 54, 53, 51, 53, 56, 57, \
								55, 57, 51, 50, 51, 56, 52, 54, 50, 54, 52, 51, 51, 56, 51, 50, 55, \
								57, 53, 48, 50, 56, 56, 52, 49, 57, 55, 49, 54, 57, 51, 57, 57, 51, \
								55, 53, 49, 48, 53, 56, 50, 48, 57, 55, 52, 57, 52, 52, 53, 57, 50, \
								51, 48, 55, 56, 49, 54, 52, 48, 54, 50, 56, 54, 50, 48, 56, 57, 57, \
								56, 54, 50, 56, 48, 51, 52, 56, 50, 53, 51, 52, 50, 49, 49, 55, 48, \
								54, 55, 57, 56, 50, 49, 52, 56, 48, 56, 54, 53, 49, 57, 49, 57, 55, \
								54, 96, 49, 48, 48, 46, 82, 122, 50, 46, 55, 49, 56, 50, 56, 49, 56, \
								50, 56, 52, 53, 57, 48, 52, 53, 50, 51, 53, 51, 54, 48, 50, 56, 55, \
								52, 55, 49, 51, 53, 50, 54, 54, 50, 52, 57, 55, 55, 53, 55, 50, 52, \
								55, 48, 57, 51, 54, 57, 57, 57, 53, 57, 53, 55, 52, 57, 54, 54, 57, \
								54, 55, 54, 50, 55, 55, 50, 52, 48, 55, 54, 54, 51, 48, 51, 53, 51, \
								53, 52, 55, 53, 57, 52, 53, 55, 49, 51, 56, 50, 49, 55, 56, 53, 50, \
								53, 49, 54, 54, 52, 50, 55, 52, 50, 55, 52, 54, 54, 51, 57, 49, 57, \
								51, 50, 48, 48, 51, 49, 96, 49, 48, 48, 46, 102, 2, 115, 8, 82, 97, \
								116, 105, 111, 110, 97, 108, 67, 252, 73, 26, 51, 51, 51, 51, 51, 51, \
								51, 51, 51, 51, 51, 51, 51, 51, 52, 52, 52, 51, 51, 51, 51, 51, 51, \
								51, 51, 53 };

		print_tokens(example_test()):
			func: 4 vars
			symbol: SparseArray
			symbol: Automatic
			array: rank = 1, dimensions = 2
			data: 44 23133
			i8: 0
			func: 3 vars
			symbol: List
			i8: 1
			func: 2 vars
			symbol: List
			array: rank = 1, dimensions = 45
			data: 0 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 3
			array: rank = 2, dimensions = 3 1
			data: 1 23133 2
			func: 3 vars
			symbol: List
			f64: 0.333333
			func: 2 vars
			symbol: Complex
			bigreal: 3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679821480865191976`100.
			bigreal: 2.7182818284590452353602874713526624977572470936999595749669676277240766303535475945713821785251664274274663919320031`100.
			func: 2 vars
			symbol: Rational
			i8: -4
			bigint: 33333333333333444333333335
	*/

} // namespace WXF_PARSER