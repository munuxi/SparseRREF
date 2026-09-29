/*
	Copyright (C) 2024-2025 Zhenjie Li (Li, Zhenjie)

	This file is part of Sparse_rref. The Sparse_rref is free software:
	you can redistribute it and/or modify it under the terms of the MIT
	License.
*/

/*
	To compile the library, WolframLibrary.h, WolframSparseLibrary.h, WolframNumericArrayLibrary.h
	and WolframIOLibraryFunctions.h are required,
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

	Both rref entry points read the matrix as WXF (the package sends BinarySerialize[mat]), so the
	mod p side is reduced by the same reader as the rational side.

	The rref runs as an asynchronous task: the task id comes back at once, the computation runs on a
	thread the kernel made for it, progress is raised as "log" events, and the result as "done". The
	package waits for those events in the language, so Alt+. stops a long computation at any moment:
	the cleanup removes the task, which the runner notices at its next progress line.

	This is what the interrupt needs. AbortQ() can hang when it is called from a thread Wolfram does
	not manage - the call then never returns, and whoever waits for that thread waits forever - so the
	kernel is asked from the task's own thread instead (asynchronousTaskAliveQ, raiseAsyncEvent).

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
		  {LibraryDataType[ByteArray], "Constant"},
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

	(* the asynchronous form, which the package uses. sprref_rat_rref_task and sprref_mod_rref_task
	   return the id of a task and raise the events "log", "done" and "error"; sprref_take_result
	   hands over the bytes of a finished task as the ByteArray to deserialize. They are loaded the
	   same way, see SparseRREF.wl. *)

	(* the first matrix is the result of rref, and the second is its kernel *)
	modrref[mat_SparseArray, p_?IntegerQ, outputMode_ : 0, method : 0, backSub : True, nthread_ : 1, verbose : False, printStep : 100] :=
		BinaryDeserialize[modRREFLibFunction[BinarySerialize[mat], p, outputMode, method, backSub, nthread, verbose, printStep]];

	ratrref[mat_SparseArray, outputMode_ : 0, method : 0, backSub : True, nthread_ : 1, verbose : False, printStep : 100] :=
		BinaryDeserialize[ratRREFLibFunction[BinarySerialize[mat], outputMode, method, backSub, nthread, verbose, printStep]];

	```
*/

#include <map>
#include <mutex>
#include <streambuf>
#include <string>
#include <utility>
#include "sparse_mat.h"
#include "sparse_tensor.h"
#include "wxf_support.h"
#include "WolframLibrary.h"
#include "WolframIOLibraryFunctions.h"
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

	// The option block of the call that runs right now, so the progress callback (which runs on the
	// thread of the call) can report an interrupt; AbortQ is only asked from there.
	rref_option*& current_option() {
		static rref_option* opt = nullptr;
		return opt;
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
			// this runs on the thread of the library call, so the kernel can be asked here
			if (rref_option* opt = current_option(); opt != nullptr && ld_->AbortQ != nullptr && ld_->AbortQ())
				opt->abort = true;
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
			if (!WXF_PARSER::parse_ok(parserA.err))
				return LIBRARY_FUNCTION_ERROR;
			auto A = sparse_tensor_read_wxf<rat_t, int>(parserA.tokens, F, pool_ptr);
			tensorA = std::move(A);
		}

		{
			WXF_PARSER::Parser parserB(in_strB, lengthB);
			parserB.parse();
			if (!WXF_PARSER::parse_ok(parserB.err))
				return LIBRARY_FUNCTION_ERROR;
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
			if (!WXF_PARSER::parse_ok(parserA.err))
				return LIBRARY_FUNCTION_ERROR;
			matA = sparse_mat_read_wxf<rat_t, int>(parserA.tokens, F);
		}

		{
			WXF_PARSER::Parser parserB(in_strB, lengthB);
			parserB.parse();
			if (!WXF_PARSER::parse_ok(parserB.err))
				return LIBRARY_FUNCTION_ERROR;
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

namespace {

	// The progress sink of an asynchronous task: it runs on the thread the kernel made for the task,
	// so the kernel may be asked (asynchronousTaskAliveQ) and reported to (raiseAsyncEvent) from here.
	class task_log_streambuf : public std::streambuf {
	public:
		task_log_streambuf(WolframIOLibrary_Functions io, mint id, rref_option_t opt)
			: io_(io), id_(id), opt_(opt) {}

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
			// a task that is gone (the Wolfram side removed it: that is the interrupt) stops the library
			const bool alive = io_->asynchronousTaskAliveQ == nullptr
				|| io_->asynchronousTaskAliveQ(id_) != 0;
			if (!alive) {
				opt_->abort = true;
				line_.clear();
				return;
			}
			if (line_.empty())
				return;
			if (io_->raiseAsyncEvent != nullptr) {
				DataStore ds = io_->createDataStore();
				io_->DataStore_addString(ds, (char*)line_.c_str());
				io_->raiseAsyncEvent(id_, (char*)"log", ds);
			}
			line_.clear();
		}

		WolframIOLibrary_Functions io_;
		mint id_;
		rref_option_t opt_;
		std::string line_;
	};

	// the byte array (an MNumericArray of UBit8) that carries a result
	MNumericArray na_from_bytes(WolframLibraryData ld, const std::vector<uint8_t>& bytes) {
		auto naFuns = ld->numericarrayLibraryFunctions;
		MNumericArray na = nullptr;
		mint len = (mint)bytes.size();
		if (naFuns->MNumericArray_new(MNumericArray_Type_UBit8, 1, &len, &na) != LIBRARY_NO_ERROR)
			return nullptr;
		std::memcpy(naFuns->MNumericArray_getData(na), bytes.data(), bytes.size());
		return na;
	}

	// The bytes of a finished task wait here and are picked up by sprref_take_result, which returns
	// them as a ByteArray. Through the event itself they would reach Wolfram as a list of integers.
	std::mutex result_mtx;
	std::map<mint, std::vector<uint8_t>> results;
	mint next_result_id = 1;

	mint put_result(std::vector<uint8_t>&& bytes) {
		std::lock_guard<std::mutex> lock(result_mtx);
		// bounded: a result nobody fetched (an abort right after "done") must not pile up
		while (results.size() >= 8)
			results.erase(results.begin());
		mint id = next_result_id++;
		results.emplace(id, std::move(bytes));
		return id;
	}

	void raise_result(WolframIOLibrary_Functions io, mint id, mint result_id) {
		DataStore ds = io->createDataStore();
		io->DataStore_addInteger(ds, result_id);
		if (io->raiseAsyncEvent != nullptr)
			io->raiseAsyncEvent(id, (char*)"done", ds);
	}

	void raise_error(WolframIOLibrary_Functions io, mint id, const std::string& message) {
		DataStore ds = io->createDataStore();
		io->DataStore_addString(ds, (char*)(message.empty() ? "the call failed" : message.c_str()));
		if (io->raiseAsyncEvent != nullptr)
			io->raiseAsyncEvent(id, (char*)"error", ds);
	}

	// One rref call: the serialized result, or an empty vector and a message. The option carries the
	// progress stream and the abort flag, so both entry points share this.
	template <typename T>
	std::vector<uint8_t> rref_result(sparse_mat<T, int>& mat, const field_t& F, int output_mode,
		rref_option_t opt, bool reconstruct, std::string& message) {

		using namespace WXF_PARSER;

		if (output_mode < 0 || output_mode > 3) {
			message = "rref: the output mode must be 0, 1, 2 or 3";
			return {};
		}
		if (opt->abort) {
			message = "rref: aborted";
			return {};
		}
		// the rational field can reconstruct the result, the prime field has its own rref
		std::vector<std::vector<pivot_t<int>>> pivots;
		if constexpr (std::is_same_v<T, rat_t>) {
			pivots = reconstruct ? sparse_mat_rref_reconstruct(mat, opt)
				: sparse_mat_rref(mat, F, opt);
		}
		else {
			(void)reconstruct;
			pivots = sparse_mat_rref(mat, F, opt);
		}
		if (opt->abort) {
			// the matrix is only partially reduced, so nothing below is meaningful (and evaluating it
			// could read outside of its vectors)
			message = "rref: aborted";
			return {};
		}
		std::vector<pivot_t<int>> pivots_vec;
		for (auto& round : pivots)
			pivots_vec.insert(pivots_vec.end(), round.begin(), round.end());

		sparse_mat<T> K;
		size_t len = 0;
		if (output_mode == 1 || output_mode == 3) {
			K = sparse_mat_rref_kernel(mat, pivots, F, opt);
			len = K.nrow;
		}

		Encoder encoder;
		auto& res_str = encoder.buffer;
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
		}
		return res_str;
	}
}

// output_mode:
// 0: output the rref
// 1: output the rref and its kernel
// 2: output the rref and its pivots
// 3: output the rref, kernel and pivots
//
// The input is WXF, as on the rational side (the reader does the reduction modulo p). This is the
// synchronous form, the fallback for a kernel without asynchronous tasks; the package uses
// sprref_mod_rref_task, which can be interrupted at any time.
EXTERN_C DLLEXPORT int sprref_mod_rref(WolframLibraryData ld, mint Argc, MArgument* Args, MArgument Res) {
	if (Argc != 9)
		return LIBRARY_FUNCTION_ERROR;
	auto na_in = MArgument_getMNumericArray(Args[0]);
	auto p = MArgument_getInteger(Args[1]);
	auto output_mode = MArgument_getInteger(Args[2]);
	auto method = MArgument_getInteger(Args[3]);
	auto is_back_sub = MArgument_getBoolean(Args[4]);
	auto nthreads = MArgument_getInteger(Args[5]);
	auto verbose = MArgument_getBoolean(Args[6]);
	auto print_step = MArgument_getInteger(Args[7]);
	auto log_to_kernel = MArgument_getBoolean(Args[8]);

	auto naFuns = ld->numericarrayLibraryFunctions;
	if (naFuns->MNumericArray_getType(na_in) != MNumericArray_Type_UBit8)
		return LIBRARY_FUNCTION_ERROR;
	if (method < 0 || method > 2 || output_mode < 0 || output_mode > 3)
		return LIBRARY_FUNCTION_ERROR;

	auto in_str = (uint8_t*)(naFuns->MNumericArray_getData(na_in));
	auto length = naFuns->MNumericArray_getFlattenedLength(na_in);

	field_t F(FIELD_Fp, p);

	// progress lines are forwarded to the kernel while the computation runs, and that callback is
	// also what notices an interrupt (the calling thread asking the kernel)
	kernel_log_streambuf log_buffer(ld);
	std::ostream log_stream(&log_buffer);

	WXF_PARSER::Parser parser(in_str, length);
	parser.parse();
	if (!WXF_PARSER::parse_ok(parser.err))
		return LIBRARY_FUNCTION_ERROR;
	auto mat = sparse_mat_read_wxf<ulong, int>(parser.tokens, F);
	if (mat.nrow == 0 && mat.ncol == 0) // not a matrix, or not readable
		return LIBRARY_FUNCTION_ERROR;

	rref_option_t opt;
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

	current_option() = opt;
	std::string message;
	auto res_str = rref_result<ulong>(mat, F, (int)output_mode, opt, false, message);
	current_option() = nullptr;
	if (res_str.empty())
		return LIBRARY_FUNCTION_ERROR;

	auto na_out = na_from_bytes(ld, res_str);
	if (na_out == nullptr)
		return LIBRARY_FUNCTION_ERROR;
	MArgument_setMNumericArray(Res, na_out);

	return LIBRARY_NO_ERROR;
}

// the asynchronous rref over Zp: the bytes are copied here, the rest runs on the task's thread
struct mod_rref_task_data {
	WolframIOLibrary_Functions io;
	std::vector<uint8_t> bytes;
	ulong p;
	int output_mode;
	int method;
	bool is_back_sub;
	int threads;
	int print_step;
	mint kernel_id = 0; // the id the kernel gave the task, see sweep_tasks
};

// A task is started by one thread and runs on another, so its data cannot live on the stack of the
// call that started it. The library holds it in these tables until the runner takes it over, which is
// its first line, so the struct itself is never allocated or freed by hand.
std::mutex task_mtx;
mint next_task_id = 1;
std::map<mint, mod_rref_task_data> mod_tasks;

// An entry that is still in the table was not taken over by its runner; if the kernel no longer has
// the task alive, no runner ever will, so that entry is garbage. The kernel's task id is kept on the
// entry for exactly this test and the table is swept before another task is added to it.
template <typename T>
void sweep_tasks(std::map<mint, T>& table) {
	for (auto it = table.begin(); it != table.end(); ) {
		WolframIOLibrary_Functions io = it->second.io;
		mint kid = it->second.kernel_id;
		if (kid != 0 && io->asynchronousTaskAliveQ != nullptr && !io->asynchronousTaskAliveQ(kid))
			it = table.erase(it);
		else
			++it;
	}
}

template <typename T>
void set_task_id(std::map<mint, T>& table, mint slot, mint kid) {
	std::lock_guard<std::mutex> lock(task_mtx);
	auto it = table.find(slot);
	if (it != table.end())
		it->second.kernel_id = kid;
}

mint put_mod_task(mod_rref_task_data&& data) {
	std::lock_guard<std::mutex> lock(task_mtx);
	sweep_tasks(mod_tasks);
	mint id = next_task_id++;
	mod_tasks.emplace(id, std::move(data));
	return id;
}

// the runner is handed the id of its entry, not a pointer to it
bool take_mod_task(mint id, mod_rref_task_data& data) {
	std::lock_guard<std::mutex> lock(task_mtx);
	auto it = mod_tasks.find(id);
	if (it == mod_tasks.end())
		return false;
	data = std::move(it->second);
	mod_tasks.erase(it);
	return true;
}

void mod_rref_task_runner(mint id, void* varg) {
	mod_rref_task_data d;
	if (!take_mod_task((mint)(size_t)varg, d))
		return;
	field_t F(FIELD_Fp, d.p);
	rref_option_t opt;
	opt->method = d.method;
	opt->is_back_sub = d.is_back_sub;
	opt->pool.reset(d.threads);
	opt->verbose = true; // the progress lines are the interrupt checkpoints
	opt->print_step = d.print_step;
	task_log_streambuf log_buffer(d.io, id, opt);
	std::ostream log_stream(&log_buffer);
	opt->progress_out = &log_stream;

	WXF_PARSER::Parser parser(d.bytes.data(), d.bytes.size());
	parser.parse();
	if (!WXF_PARSER::parse_ok(parser.err)) {
		raise_error(d.io, id, "rref: the input bytes are not a valid WXF array");
		return;
	}
	auto mat = sparse_mat_read_wxf<ulong, int>(parser.tokens, F);

	current_option() = opt;
	std::string message;
	auto bytes = mat.nrow == 0 && mat.ncol == 0
		? std::vector<uint8_t>()
		: rref_result<ulong>(mat, F, d.output_mode, opt, false, message);
	current_option() = nullptr;
	if (bytes.empty())
		raise_error(d.io, id, message.empty() ? "rref: the input is not a rank 2 array" : message);
	else
		raise_result(d.io, id, put_result(std::move(bytes)));
}

// the bytes of a finished task, as the ByteArray the Wolfram side deserializes; the entry is removed
// here, so it is fetched exactly once
EXTERN_C DLLEXPORT int sprref_take_result(WolframLibraryData ld, mint Argc, MArgument* Args, MArgument Res) {
	if (Argc != 1)
		return LIBRARY_FUNCTION_ERROR;
	std::lock_guard<std::mutex> lock(result_mtx);
	auto it = results.find(MArgument_getInteger(Args[0]));
	if (it == results.end())
		return LIBRARY_FUNCTION_ERROR;
	auto na_out = na_from_bytes(ld, it->second);
	results.erase(it);
	if (na_out == nullptr)
		return LIBRARY_FUNCTION_ERROR;
	MArgument_setMNumericArray(Res, na_out);
	return LIBRARY_NO_ERROR;
}

EXTERN_C DLLEXPORT int sprref_mod_rref_task(WolframLibraryData ld, mint Argc, MArgument* Args, MArgument Res) {
	if (Argc != 7)
		return LIBRARY_FUNCTION_ERROR;
	WolframIOLibrary_Functions io = ld->ioLibraryFunctions;
	if (io == nullptr || io->createAsynchronousTaskWithThread == nullptr)
		return LIBRARY_FUNCTION_ERROR;
	auto na_in = MArgument_getMNumericArray(Args[0]);
	auto p = MArgument_getInteger(Args[1]);
	auto output_mode = MArgument_getInteger(Args[2]);
	auto method = MArgument_getInteger(Args[3]);
	auto is_back_sub = MArgument_getBoolean(Args[4]);
	auto nthreads = MArgument_getInteger(Args[5]);
	auto print_step = MArgument_getInteger(Args[6]);
	auto naFuns = ld->numericarrayLibraryFunctions;

	if (naFuns->MNumericArray_getType(na_in) != MNumericArray_Type_UBit8)
		return LIBRARY_FUNCTION_ERROR;
	if (method < 0 || method > 2 || output_mode < 0 || output_mode > 3)
		return LIBRARY_FUNCTION_ERROR;

	// the argument belongs to the kernel and is only valid while this call runs, so the bytes are
	// taken over here (the runner gets our own copy)
	auto* data = (uint8_t*)(naFuns->MNumericArray_getData(na_in));
	auto length = naFuns->MNumericArray_getFlattenedLength(na_in);

	mod_rref_task_data d;
	d.io = io;
	d.bytes.assign(data, data + length);
	d.p = (ulong)p;
	d.output_mode = (int)output_mode;
	d.method = (int)method;
	d.is_back_sub = is_back_sub;
	d.threads = nthreads < 1 ? 1 : (int)nthreads;
	d.print_step = (int)print_step;
	mint slot = put_mod_task(std::move(d));
	mint id = io->createAsynchronousTaskWithThread(mod_rref_task_runner, (void*)(size_t)slot);
	if (io->asynchronousTaskAliveQ != nullptr && !io->asynchronousTaskAliveQ(id)) {
		// the kernel did not really start the task, so no runner will ever take this entry over
		mod_rref_task_data drop;
		take_mod_task(slot, drop);
		return LIBRARY_FUNCTION_ERROR;
	}
	set_task_id(mod_tasks, slot, id);
	MArgument_setInteger(Res, id);
	return LIBRARY_NO_ERROR;
}

// output_mode:
// 0: output the rref
// 1: output the rref and its kernel
// 2: output the rref and its pivots
// 3: output the rref, kernel and pivots
// The synchronous form: the fallback for a kernel without asynchronous tasks (the package uses
// sprref_rat_rref_task).
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

	auto naFuns = ld->numericarrayLibraryFunctions;

	if (naFuns->MNumericArray_getType(na_in) != MNumericArray_Type_UBit8)
		return LIBRARY_FUNCTION_ERROR;
	if (method < 0 || method > 2 || output_mode < 0 || output_mode > 3)
		return LIBRARY_FUNCTION_ERROR;

	auto in_str = (uint8_t*)(naFuns->MNumericArray_getData(na_in));
	auto length = naFuns->MNumericArray_getFlattenedLength(na_in);

	field_t F(FIELD_QQ);

	kernel_log_streambuf log_buffer(ld);
	std::ostream log_stream(&log_buffer);

	WXF_PARSER::Parser parser(in_str, length);
	parser.parse();
	if (!WXF_PARSER::parse_ok(parser.err))
		return LIBRARY_FUNCTION_ERROR;
	auto mat = sparse_mat_read_wxf<rat_t, int>(parser.tokens, F);

	rref_option_t opt;
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

	current_option() = opt;
	std::string message;
	auto res_str = rref_result<rat_t>(mat, F, (int)output_mode, opt, true, message);
	current_option() = nullptr;
	if (res_str.empty())
		return LIBRARY_FUNCTION_ERROR;

	auto na_out = na_from_bytes(ld, res_str);
	if (na_out == nullptr)
		return LIBRARY_FUNCTION_ERROR;
	MArgument_setMNumericArray(Res, na_out);

	return LIBRARY_NO_ERROR;
}

// the asynchronous rref over QQ: the bytes are copied here, the rest runs on the task's thread
struct rat_rref_task_data {
	WolframIOLibrary_Functions io;
	std::vector<uint8_t> bytes;
	int output_mode;
	int method;
	bool is_back_sub;
	int threads;
	int print_step;
	mint kernel_id = 0; // the id the kernel gave the task, see sweep_tasks
};

// the table of the QQ tasks, see mod_rref_task_data
std::map<mint, rat_rref_task_data> rat_tasks;

mint put_rat_task(rat_rref_task_data&& data) {
	std::lock_guard<std::mutex> lock(task_mtx);
	sweep_tasks(rat_tasks);
	mint id = next_task_id++;
	rat_tasks.emplace(id, std::move(data));
	return id;
}

bool take_rat_task(mint id, rat_rref_task_data& data) {
	std::lock_guard<std::mutex> lock(task_mtx);
	auto it = rat_tasks.find(id);
	if (it == rat_tasks.end())
		return false;
	data = std::move(it->second);
	rat_tasks.erase(it);
	return true;
}

void rat_rref_task_runner(mint id, void* varg) {
	rat_rref_task_data d;
	if (!take_rat_task((mint)(size_t)varg, d))
		return;
	field_t F(FIELD_QQ);
	rref_option_t opt;
	opt->method = d.method;
	opt->is_back_sub = d.is_back_sub;
	opt->pool.reset(d.threads);
	opt->verbose = true; // the progress lines are the interrupt checkpoints
	opt->print_step = d.print_step;
	task_log_streambuf log_buffer(d.io, id, opt);
	std::ostream log_stream(&log_buffer);
	opt->progress_out = &log_stream;

	WXF_PARSER::Parser parser(d.bytes.data(), d.bytes.size());
	parser.parse();
	if (!WXF_PARSER::parse_ok(parser.err)) {
		raise_error(d.io, id, "rref: the input bytes are not a valid WXF array");
		return;
	}
	auto mat = sparse_mat_read_wxf<rat_t, int>(parser.tokens, F);

	current_option() = opt;
	std::string message;
	auto bytes = rref_result<rat_t>(mat, F, d.output_mode, opt, true, message);
	current_option() = nullptr;
	if (bytes.empty())
		raise_error(d.io, id, message);
	else
		raise_result(d.io, id, put_result(std::move(bytes)));
}

EXTERN_C DLLEXPORT int sprref_rat_rref_task(WolframLibraryData ld, mint Argc, MArgument* Args, MArgument Res) {
	if (Argc != 6)
		return LIBRARY_FUNCTION_ERROR;
	WolframIOLibrary_Functions io = ld->ioLibraryFunctions;
	if (io == nullptr || io->createAsynchronousTaskWithThread == nullptr)
		return LIBRARY_FUNCTION_ERROR;
	auto na_in = MArgument_getMNumericArray(Args[0]);
	auto output_mode = MArgument_getInteger(Args[1]);
	auto method = MArgument_getInteger(Args[2]);
	auto is_back_sub = MArgument_getBoolean(Args[3]);
	auto nthreads = MArgument_getInteger(Args[4]);
	auto print_step = MArgument_getInteger(Args[5]);
	auto naFuns = ld->numericarrayLibraryFunctions;

	if (naFuns->MNumericArray_getType(na_in) != MNumericArray_Type_UBit8)
		return LIBRARY_FUNCTION_ERROR;
	if (method < 0 || method > 2 || output_mode < 0 || output_mode > 3)
		return LIBRARY_FUNCTION_ERROR;

	auto* data = (uint8_t*)(naFuns->MNumericArray_getData(na_in));
	auto length = naFuns->MNumericArray_getFlattenedLength(na_in);

	rat_rref_task_data d;
	d.io = io;
	d.bytes.assign(data, data + length);
	d.output_mode = (int)output_mode;
	d.method = (int)method;
	d.is_back_sub = is_back_sub;
	d.threads = nthreads < 1 ? 1 : (int)nthreads;
	d.print_step = (int)print_step;
	mint slot = put_rat_task(std::move(d));
	mint id = io->createAsynchronousTaskWithThread(rat_rref_task_runner, (void*)(size_t)slot);
	if (io->asynchronousTaskAliveQ != nullptr && !io->asynchronousTaskAliveQ(id)) {
		// the kernel did not really start the task, so no runner will ever take this entry over
		rat_rref_task_data drop;
		take_rat_task(slot, drop);
		return LIBRARY_FUNCTION_ERROR;
	}
	set_task_id(rat_tasks, slot, id);
	MArgument_setInteger(Res, id);
	return LIBRARY_NO_ERROR;
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
		if (!WXF_PARSER::parse_ok(parser.err))
			return LIBRARY_FUNCTION_ERROR;
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