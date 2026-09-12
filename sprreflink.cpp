/*
	Copyright (C) 2024-2025 Zhenjie Li (Li, Zhenjie)

	This file is part of Sparse_rref. The Sparse_rref is free software:
	you can redistribute it and/or modify it under the terms of the MIT
	License.
*/

/*
	To compile the library, WolframLibrary.h, WolframSparseLibrary.h and 
	WolframNumericArrayLibrary.h are required,
	which are included in the Mathematica installation directory,
	which is $MATHEMATICA_HOME/SystemFiles/IncludeFiles/C in most cases.

	To use the library, load the package SparseRREF.wl, e.g.:
	```mathematica
	Needs["SparseRREF`"];
	mat = SparseArray @ { {10, 0, 20}, {30, 40, 50} };
	p = 7;
	{rref, kernel} = SparseRREF[mat, Modulus -> p, "OutputMode" -> "RREF,Kernel", "Method" -> "Hybrid", "Threads" -> 0];
	```
	See detailed instructions in SparseRREF.wl and in Readme.md.

	Progress output can be redirected to the kernel: the last argument of the two
	rref functions is a flag that, when True, makes every progress line be
	evaluated as SparseRREF`Private`sprrefLogPush["<line>"] while the computation
	is still running. This is what the "LogStream" option of the package uses.

	To load the functions in Mathematica manually, use the following code (as an example):

	```mathematica

	ratRREFLibFunction =
	  LibraryFunctionLoad[
		"sprreflink.dll",
		"sprref_rat_rref",
		{
		  {LibraryDataType[ByteArray], "Constant"},
		  Integer,
		  Integer,
		  True | False,
		  Integer,
		  True | False,
		  Integer,
		  True | False
		},
		{LibraryDataType[ByteArray], Automatic}
	  ];

	modRREFLibFunction =
	  LibraryFunctionLoad[
		"sprreflink.dll",
		"sprref_mod_rref",
		{
		  {LibraryDataType[SparseArray], "Constant"},
		  Integer,
		  Integer,
		  Integer,
		  True | False,
		  Integer,
		  True | False,
		  Integer,
		  True | False
		},
		{LibraryDataType[ByteArray], Automatic}
	  ];

	(* the first matrix is the result of rref, and the second is its kernel *)
	modrref[mat_SparseArray, p_?IntegerQ, outputMode_ : 0, method : 0, backSub : True, nthread_ : 1, verbose : False, printStep : 100] :=
		BinaryDeserialize[modRREFLibFunction[mat, p, outputMode, method, backSub, nthread, verbose, printStep]];

	ratrref[mat_SparseArray, outputMode_ : 0, method : 0, backSub : True, nthread_ : 1, verbose : False, printStep : 100] :=
		BinaryDeserialize[ratRREFLibFunction[BinarySerialize[mat], outputMode, method, backSub, nthread, verbose, printStep]];

	```
*/

#include <streambuf>
#include <string>
#include <utility>
#include "sparse_mat.h"
#include "sparse_tensor.h"
#include "wxf_support.h"
#include "WolframLibrary.h"
#include "WolframSparseLibrary.h"
#include "WolframNumericArrayLibrary.h"

using namespace SparseRREF;

EXTERN_C DLLEXPORT mint WolframLibrary_getVersion() {
    return WolframLibraryVersion;
}

EXTERN_C DLLEXPORT int WolframLibrary_initialize(WolframLibraryData ld) {
    return LIBRARY_NO_ERROR;
}

/*
	Kernel log stream.

	The progress output of a computation can be redirected to the Mathematica
	kernel: every completed line is evaluated as

		SparseRREF`Private`sprrefLogPush["<line>"]

	while the library function is still running, so the package can display the
	line right away (and keep it for a later SparseRREFLog[] call).
*/
namespace {

	// evaluateExpression is only usable with this mode value, the others crash the kernel
	constexpr int LIBRARY_EVALUATE_EXPRESSION = 6;

	// escape a line so that it can be embedded in a Wolfram Language string literal
	std::string escape_wl_string(const std::string& line) {
		std::string out;
		out.reserve(line.size() + 8);
		for (unsigned char c : line) {
			switch (c) {
			case '\\': out += "\\\\"; break;
			case '"': out += "\\\""; break;
			case '\t': out += "\\t"; break;
			default:
				// other control characters would not survive the round trip
				out += (c < 0x20 || c == 0x7f) ? ' ' : (char)c;
			}
		}
		return out;
	}

	void kernel_log_push(WolframLibraryData ld, const std::string& line) {
		if (ld->evaluateExpression == nullptr)
			return;
		std::string expression = "SparseRREF`Private`sprrefLogPush[\"" + escape_wl_string(line) + "\"]";
		ld->evaluateExpression(ld, expression.data(), LIBRARY_EVALUATE_EXPRESSION, 0, nullptr);
	}

	// Hands every completed line to the kernel. Progress lines always end with a
	// newline (a custom stream is never a terminal, so nothing is overwritten in
	// place); blank lines are dropped, they carry no information.
	class kernel_log_streambuf : public std::streambuf {
	public:
		explicit kernel_log_streambuf(WolframLibraryData ld) : ld_(ld) {}

	protected:
		int_type overflow(int_type c) override {
			if (traits_type::eq_int_type(c, traits_type::eof()))
				return traits_type::not_eof(c);
			push(traits_type::to_char_type(c));
			return c;
		}

		std::streamsize xsputn(const char* s, std::streamsize n) override {
			for (std::streamsize i = 0; i < n; i++)
				push(s[i]);
			return n;
		}

		int sync() override {
			emit_line();
			return 0;
		}

	private:
		void push(char c) {
			if (c == '\n')
				emit_line();
			else if (c != '\r')
				line_ += c;
		}

		void emit_line() {
			if (line_.empty())
				return;
			kernel_log_push(ld_, line_);
			line_.clear();
		}

		WolframLibraryData ld_;
		std::string line_;
	};
}

sparse_mat<ulong> MSparseArray_to_sparse_mat_ulong(WolframLibraryData ld, MArgument* arg, ulong p) {
	auto mat = MArgument_getMSparseArray(*arg);
	auto sf = ld->sparseLibraryFunctions;

	auto dims = sf->MSparseArray_getDimensions(mat);
	auto nrows = dims[0];
	auto ncols = dims[1];

	auto m_rowptr = sf->MSparseArray_getRowPointers(mat);
	auto m_colptr = sf->MSparseArray_getColumnIndices(mat);
	auto m_valptr = sf->MSparseArray_getExplicitValues(mat);

	// rowptr, valptr, colptr are managed by mathematica
	// do not free them
	mint* rowptr = ld->MTensor_getIntegerData(*m_rowptr);
	mint* valptr = ld->MTensor_getIntegerData(*m_valptr);
	mint* colptr = ld->MTensor_getIntegerData(*m_colptr);

	auto nnz = rowptr[nrows];

	// init a sparse matrix
	nmod_t pp;
	int_t tmp;
	nmod_init(&pp, (ulong)p);
	sparse_mat<ulong> A(nrows, ncols);

	for (auto i = 0; i < nrows; i++) {
		A[i].reserve(rowptr[i + 1] - rowptr[i]);
		for (auto k = rowptr[i]; k < rowptr[i + 1]; k++) {
			tmp = valptr[k];
			A[i].push_back(colptr[k] - 1, tmp % pp);
		}
	}

	return A;
}

int sparse_mat_ulong_to_MSparseArray(WolframLibraryData ld, MSparseArray& res, const sparse_mat<ulong>& A) {
	auto sf = ld->sparseLibraryFunctions;
	size_t nnz = A.nnz();
	MTensor pos, val, dim;
	if (!std::in_range<mint>(nnz))
		return LIBRARY_FUNCTION_ERROR;
	mint dims_r2[] = { static_cast<mint>(nnz), 2 };
	ld->MTensor_new(MType_Integer, 2, dims_r2, &pos);
	mint dims_r1[] = { static_cast<mint>(nnz) };
	ld->MTensor_new(MType_Integer, 1, dims_r1, &val);
	dims_r1[0] = 2;
	ld->MTensor_new(MType_Integer, 1, dims_r1, &dim);
	mint* dimdata = ld->MTensor_getIntegerData(dim);
	if (!std::in_range<mint>(A.nrow) || !std::in_range<mint>(A.ncol))
		return LIBRARY_FUNCTION_ERROR;
	dimdata[0] = static_cast<mint>(A.nrow);
	dimdata[1] = static_cast<mint>(A.ncol);
	mint* valdata = ld->MTensor_getIntegerData(val);
	mint* posdata = ld->MTensor_getIntegerData(pos);
	auto nownnz = 0;
	for (size_t i = 0; i < A.nrow; i++) {
		for (size_t j = 0; j < A[i].nnz(); j++) {
			posdata[2 * nownnz] = i + 1;
			posdata[2 * nownnz + 1] = A[i](j) + 1;
			valdata[nownnz] = A[i][j];
			nownnz++;
		}
	}
	auto err = sf->MSparseArray_fromExplicitPositions(pos, val, dim, 0, &res);
	ld->MTensor_free(pos);
	ld->MTensor_free(val);
	ld->MTensor_free(dim);
	return err;
}

sparse_tensor<ulong, int, SPARSE_CSR> MSparseArray_to_sparse_tensor_ulong(WolframLibraryData ld, MArgument* arg, ulong p) {
	auto tensor = MArgument_getMSparseArray(*arg);
	auto sf = ld->sparseLibraryFunctions;

	auto dims = sf->MSparseArray_getDimensions(tensor);
	auto rank = sf->MSparseArray_getRank(tensor);
	auto nrows = dims[0];

	auto m_rowptr = sf->MSparseArray_getRowPointers(tensor);
	auto m_colptr = sf->MSparseArray_getColumnIndices(tensor);
	auto m_valptr = sf->MSparseArray_getExplicitValues(tensor);

	// rowptr, valptr, colptr are managed by mathematica
	// do not free them
	mint* rowptr = ld->MTensor_getIntegerData(*m_rowptr);
	mint* valptr = ld->MTensor_getIntegerData(*m_valptr);
	mint* colptr = ld->MTensor_getIntegerData(*m_colptr);

	auto nnz = rowptr[nrows];

	// init a sparse tensor
	nmod_t pp;
	int_t tmp;
	nmod_init(&pp, (ulong)p);
	sparse_tensor<ulong, int, SPARSE_CSR> A(std::vector<size_t>(dims, dims + rank), nnz);

	auto& A_rowptr = A.data.rowptr;
	auto& A_colptr = A.data.colptr;
	auto& A_valptr = A.data.valptr;

	std::copy(rowptr, rowptr + nrows + 1, A_rowptr.begin());
	for (size_t i = 0; i < nnz; i++) {
		tmp = valptr[i];
		A_valptr[i] = tmp % pp;
	}
	for (size_t i = 0; i < nnz * (rank - 1); i++) {
		A_colptr[i] = colptr[i] - 1;
	}

	return A;
}

int sparse_tensor_ulong_to_MSparseArray(WolframLibraryData ld, MSparseArray& res, const sparse_tensor<ulong, int, SPARSE_CSR>& A) {
	auto sf = ld->sparseLibraryFunctions;
	size_t nnz = A.nnz();
	MTensor pos, val, dim;
	if (!std::in_range<mint>(nnz))
		return LIBRARY_FUNCTION_ERROR;
	if (!std::in_range<mint>(A.rank()))
		return LIBRARY_FUNCTION_ERROR;
	mint rank_mint = static_cast<mint>(A.rank());
	mint dims_r2[] = { static_cast<mint>(nnz), rank_mint };
	ld->MTensor_new(MType_Integer, 2, dims_r2, &pos);
	mint dims_r1[] = { static_cast<mint>(nnz) };
	ld->MTensor_new(MType_Integer, 1, dims_r1, &val);
	dims_r1[0] = rank_mint;
	ld->MTensor_new(MType_Integer, 1, dims_r1, &dim);
	mint* dimdata = ld->MTensor_getIntegerData(dim);
	for (size_t i = 0; i < A.rank(); i++) {
		if (!std::in_range<mint>(A.dim(i)))
			return LIBRARY_FUNCTION_ERROR;
		dimdata[i] = static_cast<mint>(A.dim(i));
	}
	mint* valdata = ld->MTensor_getIntegerData(val);
	mint* posdata = ld->MTensor_getIntegerData(pos);
	for (size_t i = 0; i < nnz; i++) {
		if (!std::in_range<mint>(A.val(i)))
			return LIBRARY_FUNCTION_ERROR;
		valdata[i] = static_cast<mint>(A.val(i));
		auto index_v = A.index_vector(i);
		for (size_t j = 0; j < A.rank(); j++) {
			if (!std::in_range<mint>(index_v[j] + 1))
				return LIBRARY_FUNCTION_ERROR;
			posdata[i * A.rank() + j] = static_cast<mint>(index_v[j] + 1);
		}
	}
	auto err = sf->MSparseArray_fromExplicitPositions(pos, val, dim, 0, &res);
	ld->MTensor_free(pos);
	ld->MTensor_free(val);
	ld->MTensor_free(dim);
	return err;
}

EXTERN_C DLLEXPORT int sprref_mod_tensor_contract(WolframLibraryData ld, mint Argc, MArgument* Args, MArgument Res) {
	if (Argc != 6)
		return LIBRARY_FUNCTION_ERROR;
	auto p = MArgument_getInteger(Args[2]);
	auto indexA = MArgument_getMTensor(Args[3]);
	auto indexB = MArgument_getMTensor(Args[4]);
	auto nthreads = MArgument_getInteger(Args[5]);

	if (ld->MTensor_getRank(indexA) != 1 || ld->MTensor_getRank(indexB) != 1)
		return LIBRARY_FUNCTION_ERROR;

	auto A = MSparseArray_to_sparse_tensor_ulong(ld, Args, (ulong)p);
	auto B = MSparseArray_to_sparse_tensor_ulong(ld, Args + 1, (ulong)p);

	sparse_tensor<ulong> tensorA(std::move(A));
	sparse_tensor<ulong> tensorB(std::move(B));

	auto startA = ld->MTensor_getIntegerData(indexA);
	auto startB = ld->MTensor_getIntegerData(indexB);
	size_t lenA = ld->MTensor_getFlattenedLength(indexA);
	size_t lenB = ld->MTensor_getFlattenedLength(indexB);

	if (lenA != lenB)
		return LIBRARY_FUNCTION_ERROR;

	std::vector<size_t> idxA(startA, startA + lenA);
	std::vector<size_t> idxB(startB, startB + lenB);
	for (auto& i : idxA)
		i--; // change to zero-based index
	for (auto& i : idxB)
		i--;

	// a contract index out of range or repeated makes tensor_contract place the entries wrongly
	if (!in_range_and_distinct(idxA, tensorA.rank()) || !in_range_and_distinct(idxB, tensorB.rank()))
		return LIBRARY_FUNCTION_ERROR;

	field_t F(FIELD_Fp, (ulong)p);
	int err = 0;
	MSparseArray result = 0;
	if (nthreads == 1) {
		auto C = tensor_contract(tensorA, tensorB, idxA, idxB, F, nullptr);
		err = sparse_tensor_ulong_to_MSparseArray(ld, result, C);
	}
	else {
		thread_pool pool(nthreads);
		auto tensorC = tensor_contract(tensorA, tensorB, idxA, idxB, F, &pool);
		sparse_tensor<ulong, int, SPARSE_CSR> C(std::move(tensorC), &pool);
		err = sparse_tensor_ulong_to_MSparseArray(ld, result, C);
	}
	if (err)
		return LIBRARY_FUNCTION_ERROR;
	MArgument_setMSparseArray(Res, result);
	return LIBRARY_NO_ERROR;
}

EXTERN_C DLLEXPORT int sprref_rat_tensor_contract(WolframLibraryData ld, mint Argc, MArgument* Args, MArgument Res) {
	if (Argc != 5)
		return LIBRARY_FUNCTION_ERROR;
	auto na_inA = MArgument_getMNumericArray(Args[0]);
	auto na_inB = MArgument_getMNumericArray(Args[1]);
	auto indexA = MArgument_getMTensor(Args[2]);
	auto indexB = MArgument_getMTensor(Args[3]);
	auto nthreads = MArgument_getInteger(Args[4]);
	numericarray_data_t type = MNumericArray_Type_Undef;
	auto naFuns = ld->numericarrayLibraryFunctions;

	type = naFuns->MNumericArray_getType(na_inA);
	if (type != MNumericArray_Type_UBit8)
		return LIBRARY_FUNCTION_ERROR;
	type = naFuns->MNumericArray_getType(na_inB);
	if (type != MNumericArray_Type_UBit8)
		return LIBRARY_FUNCTION_ERROR;

	auto in_strA = (uint8_t*)(naFuns->MNumericArray_getData(na_inA));
	auto lengthA = naFuns->MNumericArray_getFlattenedLength(na_inA);
	auto in_strB = (uint8_t*)(naFuns->MNumericArray_getData(na_inB));
	auto lengthB = naFuns->MNumericArray_getFlattenedLength(na_inB);

	if (ld->MTensor_getRank(indexA) != 1 || ld->MTensor_getRank(indexB) != 1)
		return LIBRARY_FUNCTION_ERROR;

	auto startA = ld->MTensor_getIntegerData(indexA);
	auto startB = ld->MTensor_getIntegerData(indexB);
	size_t lenA = ld->MTensor_getFlattenedLength(indexA);
	size_t lenB = ld->MTensor_getFlattenedLength(indexB);

	if (lenA != lenB)
		return LIBRARY_FUNCTION_ERROR;

	std::vector<size_t> idxA(startA, startA + lenA);
	std::vector<size_t> idxB(startB, startB + lenB);

	std::vector<uint8_t> res_str;
	uint8_t* out_str = nullptr;
	mint out_len = 0;
	auto err = LIBRARY_NO_ERROR;
	MNumericArray na_out = NULL;
	{
		field_t F(FIELD_QQ);
		sparse_tensor<rat_t> tensorA, tensorB;
		thread_pool pool(nthreads);
		thread_pool* pool_ptr = (nthreads == 1) ? nullptr : &pool;

		{
			WXF_PARSER::Parser parserA(in_strA, lengthA);
			parserA.parse();
			auto A = sparse_tensor_read_wxf<rat_t, int>(parserA.tokens, F, pool_ptr);
			tensorA = std::move(A);
		}

		{
			WXF_PARSER::Parser parserB(in_strB, lengthB);
			parserB.parse();
			auto B = sparse_tensor_read_wxf<rat_t, int>(parserB.tokens, F, pool_ptr);
			tensorB = std::move(B);
		}
		
		for (auto& i : idxA)
			i--; // change to zero-based index
		for (auto& i : idxB)
			i--;

		// a contract index out of range or repeated makes tensor_contract place the entries wrongly
		if (!in_range_and_distinct(idxA, tensorA.rank()) || !in_range_and_distinct(idxB, tensorB.rank()))
			return LIBRARY_FUNCTION_ERROR;

		auto tensorC = tensor_contract(tensorA, tensorB, idxA, idxB, F, pool_ptr);
		sparse_tensor<rat_t, int, SPARSE_CSR> C(std::move(tensorC), pool_ptr);
		res_str = sparse_tensor_write_wxf(C, true);

		// output the result(bit_array)
		out_len = res_str.size();
		auto err = naFuns->MNumericArray_new(type, 1, &out_len, &na_out);

		if (err)
			return LIBRARY_FUNCTION_ERROR;

		out_str = (uint8_t*)(naFuns->MNumericArray_getData(na_out));
	}

	std::memcpy(out_str, res_str.data(), res_str.size() * sizeof(uint8_t));

	MArgument_setMNumericArray(Res, na_out);

	return err;
}

EXTERN_C DLLEXPORT int sprref_mod_matmul(WolframLibraryData ld, mint Argc, MArgument* Args, MArgument Res) {
	if (Argc != 4)
		return LIBRARY_FUNCTION_ERROR;
	auto matA = MArgument_getMSparseArray(Args[0]);
	auto matB = MArgument_getMSparseArray(Args[1]);
	auto p = MArgument_getInteger(Args[2]);
	auto nthreads = MArgument_getInteger(Args[3]);
	auto sf = ld->sparseLibraryFunctions;
	auto ranksA = sf->MSparseArray_getRank(matA);
	if (ranksA != 2 && sf->MSparseArray_getImplicitValue(matA) != 0)
		return LIBRARY_FUNCTION_ERROR;
	auto ranksB = sf->MSparseArray_getRank(matB);
	if (ranksB != 2 && sf->MSparseArray_getImplicitValue(matB) != 0)
		return LIBRARY_FUNCTION_ERROR;
	auto A = MSparseArray_to_sparse_mat_ulong(ld, Args, (ulong)p);
	auto B = MSparseArray_to_sparse_mat_ulong(ld, Args + 1, (ulong)p);
	if (A.ncol != B.nrow)
		return LIBRARY_FUNCTION_ERROR;
	field_t F(FIELD_Fp, p);
	int err = 0;
	MSparseArray result = 0;
	if (nthreads == 1) {
		auto C = sparse_mat_mul(A, B, F, nullptr);
		err = sparse_mat_ulong_to_MSparseArray(ld, result, C);
	}
	else {
		thread_pool pool(nthreads);
		auto C = sparse_mat_mul(A, B, F, &pool);
		err = sparse_mat_ulong_to_MSparseArray(ld, result, C);
	}
	if (err)
		return LIBRARY_FUNCTION_ERROR;
	MArgument_setMSparseArray(Res, result);
	return LIBRARY_NO_ERROR;
}

EXTERN_C DLLEXPORT int sprref_rat_matmul(WolframLibraryData ld, mint Argc, MArgument* Args, MArgument Res) {
	if (Argc != 3)
		return LIBRARY_FUNCTION_ERROR;
	auto na_inA = MArgument_getMNumericArray(Args[0]);
	auto na_inB = MArgument_getMNumericArray(Args[1]);
	auto nthreads = MArgument_getInteger(Args[2]);

	numericarray_data_t type = MNumericArray_Type_Undef;
	auto naFuns = ld->numericarrayLibraryFunctions;

	type = naFuns->MNumericArray_getType(na_inA);
	if (type != MNumericArray_Type_UBit8)
		return LIBRARY_FUNCTION_ERROR;
	type = naFuns->MNumericArray_getType(na_inB);
	if (type != MNumericArray_Type_UBit8)
		return LIBRARY_FUNCTION_ERROR;

	auto in_strA = (uint8_t*)(naFuns->MNumericArray_getData(na_inA));
	auto lengthA = naFuns->MNumericArray_getFlattenedLength(na_inA);
	auto in_strB = (uint8_t*)(naFuns->MNumericArray_getData(na_inB));
	auto lengthB = naFuns->MNumericArray_getFlattenedLength(na_inB);

	std::vector<uint8_t> res_str;
	uint8_t* out_str = nullptr;
	mint out_len = 0;
	auto err = LIBRARY_NO_ERROR;
	MNumericArray na_out = NULL;
	{
		field_t F(FIELD_QQ);
		sparse_mat<rat_t, int> matA, matB;

		{
			WXF_PARSER::Parser parserA(in_strA, lengthA);
			parserA.parse();
			matA = sparse_mat_read_wxf<rat_t, int>(parserA.tokens, F);
		}

		{
			WXF_PARSER::Parser parserB(in_strB, lengthB);
			parserB.parse();
			matB = sparse_mat_read_wxf<rat_t, int>(parserB.tokens, F);
		}

		if (nthreads == 1) {
			auto matC = sparse_mat_mul(matA, matB, F, nullptr);
			res_str = sparse_mat_write_wxf(matC, true);
		}
		else {
			thread_pool pool(nthreads);
			auto matC = sparse_mat_mul(matA, matB, F, &pool);
			res_str = sparse_mat_write_wxf(matC, true);
		}

		// output the result(bit_array)
		out_len = res_str.size();
		auto err = naFuns->MNumericArray_new(type, 1, &out_len, &na_out);

		if (err)
			return LIBRARY_FUNCTION_ERROR;

		out_str = (uint8_t*)(naFuns->MNumericArray_getData(na_out));
	}

	std::memcpy(out_str, res_str.data(), res_str.size() * sizeof(uint8_t));

	MArgument_setMNumericArray(Res, na_out);

	return err;
}

// output_mode:
// 0: output the rref
// 1: output the rref and its kernel
// 2: output the rref and its pivots
// 3: output the rref, kernel and pivots
EXTERN_C DLLEXPORT int sprref_mod_rref(WolframLibraryData ld, mint Argc, MArgument* Args, MArgument Res) {
	if (Argc != 9)
		return LIBRARY_FUNCTION_ERROR;
	auto mat_in = MArgument_getMSparseArray(Args[0]);
	auto p = MArgument_getInteger(Args[1]);
	auto output_mode = MArgument_getInteger(Args[2]);
	auto method = MArgument_getInteger(Args[3]);
	auto is_back_sub = MArgument_getBoolean(Args[4]);
	auto nthreads = MArgument_getInteger(Args[5]);
	auto verbose = MArgument_getBoolean(Args[6]);
	auto print_step = MArgument_getInteger(Args[7]);
	auto log_to_kernel = MArgument_getBoolean(Args[8]);

	numericarray_data_t type = MNumericArray_Type_UBit8;
	auto sf = ld->sparseLibraryFunctions;
	auto naFuns = ld->numericarrayLibraryFunctions;

	auto ranks = sf->MSparseArray_getRank(mat_in);
	if (ranks != 2 && sf->MSparseArray_getImplicitValue(mat_in) != 0)
		return LIBRARY_FUNCTION_ERROR;

	auto mat = MSparseArray_to_sparse_mat_ulong(ld, Args, (ulong)p);

	WXF_PARSER::Encoder encoder;
	auto& res_str = encoder.buffer;
	uint8_t* out_str = nullptr;
	mint out_len = 0;
	auto err = LIBRARY_NO_ERROR;
	MNumericArray na_out = NULL;
	{
		field_t F(FIELD_Fp, p);

		// progress lines are forwarded to the kernel while the computation runs
		kernel_log_streambuf log_buffer(ld);
		std::ostream log_stream(&log_buffer);

		rref_option_t opt;
		if (method < 0 || method > 2)
			return LIBRARY_FUNCTION_ERROR;
		opt->method = method;
		opt->is_back_sub = is_back_sub;
		opt->pool.reset(nthreads);
		opt->verbose = verbose || log_to_kernel;
		opt->print_step = print_step;
		if (log_to_kernel)
			opt->progress_out = &log_stream;

		// The kernel is read off the identity pivot block of the RREF, which only
		// exists after the backward substitution: asking for it while disabling the
		// backward substitution is contradictory, so enable the backward
		// substitution instead of returning a kernel that does not annihilate the
		// matrix.
		if ((output_mode == 1 || output_mode == 3) && !opt->is_back_sub) {
			std::ostream& warn = log_to_kernel ? log_stream : std::cerr;
			warn << "Warning: the kernel requires the backward substitution, which"
				<< " BackwardSubstitution -> False disabled; enabling the backward"
				<< " substitution." << std::endl;
			opt->is_back_sub = true;
		}

		std::atomic<bool> cancel(false);
		std::thread check_cancel([&]() {
			while (!cancel) {
				cancel = ld->AbortQ();
				opt->abort = cancel.load();
				// wait for 100 ms before checking again
				std::this_thread::sleep_for(std::chrono::milliseconds(100));
			}
			});

		auto pivots = sparse_mat_rref(mat, F, opt);
		if (opt->abort) {
			// the matrix is only partially reduced, so nothing below is
			// meaningful (and evaluating it could read outside of its vectors)
			cancel = true;
			check_cancel.join();
			return LIBRARY_FUNCTION_ERROR;
		}
		std::vector<pivot_t<int>> pivots_vec;
		for (auto& p : pivots) {
			pivots_vec.insert(pivots_vec.end(), p.begin(), p.end());
		}

		sparse_mat<ulong> K;
		size_t len = 0;
		if (output_mode == 1 || output_mode == 3) {
			K = sparse_mat_rref_kernel(mat, pivots, F, opt);
			len = K.nrow;
		}

		{
			using namespace WXF_PARSER;

			std::vector<uint8_t> m_str;
			auto push_mat = [&](const auto& M) {
				m_str = sparse_mat_write_wxf(M, false);
				encoder.push_ustr(m_str);
				};
			auto push_pivots = [&]() {
				// rank 2, dimensions {pivots_vec.size(), 2}
				encoder.push_array_info({ pivots_vec.size(), 2 }, WXF_HEAD::array, 3);
				uint8_t int64_buf[16];
				for (auto& p : pivots_vec) {
					// output the pivot position
					int64_t row = p.r + 1; // 1-based index
					int64_t col = p.c + 1; // 1-based index
					memcpy(int64_buf, &row, sizeof(row));
					memcpy(int64_buf + sizeof(row), &col, sizeof(col));
					res_str.insert(res_str.end(), int64_buf, int64_buf + sizeof(row) + sizeof(col));
				}
				};
			
			switch (output_mode) {
			case 0: // output the rref
				res_str = sparse_mat_write_wxf(mat, true);
				break;
			case 1: {// output the rref and its kernel
				res_str.push_back(56); res_str.push_back(58);
				encoder.push_function("List", 2);
				push_mat(mat);
				if (len > 0)
					push_mat(K);
				else
					encoder.push_symbol("Null");
				break;
			}
			case 2: { // output the rref and its pivots
				res_str.push_back(56); res_str.push_back(58);
				encoder.push_function("List", 2);
				push_mat(mat);
				push_pivots();
				break;
			}
			case 3: { // output the rref, kernel and pivots
				res_str.push_back(56); res_str.push_back(58);
				encoder.push_function("List", 3);
				push_mat(mat);
				if (len > 0)
					push_mat(K);
				else
					encoder.push_symbol("Null");
				push_pivots();
				break;
			}
			default:
				cancel = true;
				check_cancel.join();
				return LIBRARY_FUNCTION_ERROR;
			}
		}

		cancel = true;
		check_cancel.join();

		if (opt->abort) {
			return LIBRARY_FUNCTION_ERROR;
		}

		// output the result(bit_array)
		out_len = res_str.size();
		auto err = naFuns->MNumericArray_new(type, 1, &out_len, &na_out);

		if (err)
			return LIBRARY_FUNCTION_ERROR;

		out_str = (uint8_t*)(naFuns->MNumericArray_getData(na_out));
	}

	std::memcpy(out_str, res_str.data(), res_str.size() * sizeof(uint8_t));

	MArgument_setMNumericArray(Res, na_out);

	return err;
}

// output_mode:
// 0: output the rref
// 1: output the rref and its kernel
// 2: output the rref and its pivots
// 3: output the rref, kernel and pivots
EXTERN_C DLLEXPORT int sprref_rat_rref(WolframLibraryData ld, mint Argc, MArgument* Args, MArgument Res) {
	if (Argc != 8)
		return LIBRARY_FUNCTION_ERROR;
	auto na_in = MArgument_getMNumericArray(Args[0]);
	auto output_mode = MArgument_getInteger(Args[1]);
	auto method = MArgument_getInteger(Args[2]);
	auto is_back_sub = MArgument_getBoolean(Args[3]);
	auto nthreads = MArgument_getInteger(Args[4]);
	auto verbose = MArgument_getBoolean(Args[5]);
	auto print_step = MArgument_getInteger(Args[6]);
	auto log_to_kernel = MArgument_getBoolean(Args[7]);

	numericarray_data_t type = MNumericArray_Type_Undef;
	auto naFuns = ld->numericarrayLibraryFunctions;

	type = naFuns->MNumericArray_getType(na_in);
	if (type != MNumericArray_Type_UBit8)
		return LIBRARY_FUNCTION_ERROR;

	auto in_str = (uint8_t*)(naFuns->MNumericArray_getData(na_in));
	auto length = naFuns->MNumericArray_getFlattenedLength(na_in);

	WXF_PARSER::Encoder encoder;
	auto& res_str = encoder.buffer;
	uint8_t* out_str = nullptr;
	mint out_len = 0;
	auto err = LIBRARY_NO_ERROR;
	MNumericArray na_out = NULL;
	{
		field_t F(FIELD_QQ);

		// progress lines are forwarded to the kernel while the computation runs
		kernel_log_streambuf log_buffer(ld);
		std::ostream log_stream(&log_buffer);

		WXF_PARSER::Parser parser(in_str, length);
		parser.parse();
		auto mat = sparse_mat_read_wxf<rat_t, int>(parser.tokens, F);

		rref_option_t opt;
		if (method < 0 || method > 2)
			return LIBRARY_FUNCTION_ERROR;
		opt->method = method;
		opt->is_back_sub = is_back_sub;
		opt->pool.reset(nthreads);
		opt->verbose = verbose || log_to_kernel;
		opt->print_step = print_step;
		if (log_to_kernel)
			opt->progress_out = &log_stream;

		// The kernel is read off the identity pivot block of the RREF, which only
		// exists after the backward substitution: asking for it while disabling the
		// backward substitution is contradictory, so enable the backward
		// substitution instead of returning a kernel that does not annihilate the
		// matrix.
		if ((output_mode == 1 || output_mode == 3) && !opt->is_back_sub) {
			std::ostream& warn = log_to_kernel ? log_stream : std::cerr;
			warn << "Warning: the kernel requires the backward substitution, which"
				<< " BackwardSubstitution -> False disabled; enabling the backward"
				<< " substitution." << std::endl;
			opt->is_back_sub = true;
		}

		std::atomic<bool> cancel(false);
		std::thread check_cancel([&]() {
			while (!cancel) {
				cancel = ld->AbortQ();
				opt->abort = cancel.load();
				// wait for 100 ms before checking again
				std::this_thread::sleep_for(std::chrono::milliseconds(100));
			}
			});

		auto pivots = sparse_mat_rref_reconstruct(mat, opt);
		if (opt->abort) {
			// the matrix is only partially reconstructed, so nothing below is
			// meaningful (and evaluating it could read outside of its vectors)
			cancel = true;
			check_cancel.join();
			return LIBRARY_FUNCTION_ERROR;
		}
		std::vector<pivot_t<int>> pivots_vec;
		for (auto& p : pivots) {
			pivots_vec.insert(pivots_vec.end(), p.begin(), p.end());
		}

		sparse_mat<rat_t> K;
		size_t len = 0;
		if (output_mode == 1 || output_mode == 3) {
			K = sparse_mat_rref_kernel(mat, pivots, F, opt);
			len = K.nrow;
		}

		{
			using namespace WXF_PARSER;

			std::vector<uint8_t> m_str;
			auto push_mat = [&](const auto& M) {
				m_str = sparse_mat_write_wxf(M, false);
				encoder.push_ustr(m_str);
				};
			auto push_pivots = [&]() {
				// rank 2, dimensions {pivots_vec.size(), 2}
				encoder.push_array_info({ pivots_vec.size(), 2 }, WXF_HEAD::array, 3);
				uint8_t int64_buf[16];
				for (auto& p : pivots_vec) {
					// output the pivot position
					int64_t row = p.r + 1; // 1-based index
					int64_t col = p.c + 1; // 1-based index
					memcpy(int64_buf, &row, sizeof(row));
					memcpy(int64_buf + sizeof(row), &col, sizeof(col));
					res_str.insert(res_str.end(), int64_buf, int64_buf + sizeof(row) + sizeof(col));
				}
				};
			
			switch (output_mode) {
			case 0: // output the rref
				res_str = sparse_mat_write_wxf(mat, true);
				break;
			case 1: {// output the rref and its kernel
				res_str.push_back(56); res_str.push_back(58);
				encoder.push_function("List", 2);
				push_mat(mat);
				if (len > 0)
					push_mat(K);
				else
					encoder.push_symbol("Null");
				break;
			}
			case 2: { // output the rref and its pivots
				res_str.push_back(56); res_str.push_back(58);
				encoder.push_function("List", 2);
				push_mat(mat);
				push_pivots();
				break;
			}
			case 3: { // output the rref, kernel and pivots
				res_str.push_back(56); res_str.push_back(58);
				encoder.push_function("List", 3);
				push_mat(mat);
				if (len > 0)
					push_mat(K);
				else
					encoder.push_symbol("Null");
				push_pivots();
				break;
			}
			default:
				cancel = true;
				check_cancel.join();
				return LIBRARY_FUNCTION_ERROR;
			}
		}

		cancel = true;
		check_cancel.join();

		if (opt->abort) {
			return LIBRARY_FUNCTION_ERROR;
		}

		// output the result(bit_array)
		out_len = res_str.size();
		auto err = naFuns->MNumericArray_new(type, 1, &out_len, &na_out);

		if (err)
			return LIBRARY_FUNCTION_ERROR;

		out_str = (uint8_t*)(naFuns->MNumericArray_getData(na_out));
	}

	std::memcpy(out_str, res_str.data(), res_str.size() * sizeof(uint8_t));

	MArgument_setMNumericArray(Res, na_out);

	return err;
}

EXTERN_C DLLEXPORT int sprref_mod_matinv(WolframLibraryData ld, mint Argc, MArgument* Args, MArgument Res) {
	if (Argc != 3)
		return LIBRARY_FUNCTION_ERROR;
	auto mat_in = MArgument_getMSparseArray(Args[0]);
	auto p = MArgument_getInteger(Args[1]);
	auto nthreads = MArgument_getInteger(Args[2]);
	auto sf = ld->sparseLibraryFunctions;
	auto ranks = sf->MSparseArray_getRank(mat_in);
	if (ranks != 2 && sf->MSparseArray_getImplicitValue(mat_in) != 0)
		return LIBRARY_FUNCTION_ERROR;
	auto mat = MSparseArray_to_sparse_mat_ulong(ld, Args, (ulong)p);
	if (mat.ncol != mat.nrow)
		return LIBRARY_FUNCTION_ERROR;
	field_t F(FIELD_Fp, p);
	int err = 0;
	MSparseArray result = 0;
	rref_option_t opt;
	opt->pool.reset(nthreads);
	decltype(mat) inv_mat;
	err = sparse_mat_inverse(inv_mat, mat, F, opt);
	if (err)
		return LIBRARY_FUNCTION_ERROR;
	err = sparse_mat_ulong_to_MSparseArray(ld, result, inv_mat);
	if (err)
		return LIBRARY_FUNCTION_ERROR;
	MArgument_setMSparseArray(Res, result);
	return LIBRARY_NO_ERROR;
}

EXTERN_C DLLEXPORT int sprref_rat_matinv(WolframLibraryData ld, mint Argc, MArgument* Args, MArgument Res) {
	if (Argc != 2)
		return LIBRARY_FUNCTION_ERROR;
	auto na_in = MArgument_getMNumericArray(Args[0]);
	auto nthreads = MArgument_getInteger(Args[1]);

	numericarray_data_t type = MNumericArray_Type_Undef;
	auto naFuns = ld->numericarrayLibraryFunctions;

	type = naFuns->MNumericArray_getType(na_in);
	if (type != MNumericArray_Type_UBit8)
		return LIBRARY_FUNCTION_ERROR;

	auto in_str = (uint8_t*)(naFuns->MNumericArray_getData(na_in));
	auto length = naFuns->MNumericArray_getFlattenedLength(na_in);

	std::vector<uint8_t> res_str;
	uint8_t* out_str = nullptr;
	mint out_len = 0;
	int err = 0;
	MNumericArray na_out = NULL;
	{
		field_t F(FIELD_QQ);

		WXF_PARSER::Parser parser(in_str, length);
		parser.parse();
		auto mat = sparse_mat_read_wxf<rat_t, int>(parser.tokens, F);

		rref_option_t opt;
		opt->pool.reset(nthreads);

		decltype(mat) inv_mat;
		err = sparse_mat_inverse(inv_mat, mat, F, opt);
		if (err)
			return LIBRARY_FUNCTION_ERROR;
		res_str = sparse_mat_write_wxf(inv_mat, true);

		// output the result(bit_array)
		out_len = res_str.size();
		auto err = naFuns->MNumericArray_new(type, 1, &out_len, &na_out);

		if (err)
			return LIBRARY_FUNCTION_ERROR;

		out_str = (uint8_t*)(naFuns->MNumericArray_getData(na_out));
	}

	std::memcpy(out_str, res_str.data(), res_str.size() * sizeof(uint8_t));

	MArgument_setMNumericArray(Res, na_out);

	return err;
}