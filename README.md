# Sparse RREF
(exact) Sparse Reduced Row Echelon Form (RREF) **with row and column permutations** in C++

---

*Sparse Gaussian elimination is more an art than a science.*  -- [Spasm](https://github.com/cbouilla/spasm) (Charles Bouillaguet)

----

This header-only library intends to compute the exact RREF with row and column permutations of a sparse matrix over finite field or rational field, which is a common problem in linear algebra, and is widely used for number theory, cryptography, theoretical physics, etc.. The code is based on the FLINT library, which is a C library for number theory. 

Some algorithms are inspired by [Spasm](https://github.com/cbouilla/spasm), but we do not depend on it. The algorithm here is definite (Spasm is random), so once the parameters are given, the result is stable, which is important for some purposes. 

### License

[The MIT License (MIT)](https://raw.githubusercontent.com/munuxi/sparse_mat/master/LICENSE)

### Dependence

The code mainly depends on [FLINT](https://flintlib.org/) to support arithmetic, and [BS::thread_pool](https://github.com/bshoshany/thread-pool),  [wxf_parser](https://github.com/munuxi/wxf_parser) and [argparse](https://github.com/p-ranav/argparse) (they are included) are also used to support thread pool, WXF format and parse args.

If one use functions on sparse_tensor, it also requires to link tbb (Threading Building Blocks) library (for GCC and CLANG), since the Parallel STL of C++20 is used there.

### What to compute?

For a sparse matrix $M$, the code computes its RREF $\Lambda$ with row and column permutations. Instead of permute the row and column directly, we keep the row and column ordering of the matrix, i.e. the i-th row/column of $\Lambda$ is the i-th row/column of $M$, and the row and column permutation of this RREF is implicitly given by its pivots, which is a list of pairs of (row,col). In the ordering of pivots, the submatrix $\Lambda[\text{rows in pivots},\text{cols in pivots}]$ of $\Lambda$ is an identity matrix (if `--no-backward-substitution` is enabled, it is upper triangular). 

### How to compile the code

We now only support the rational field $\mathbb Q$ and the $\mathbb Z/p\mathbb Z$, where $p$ is a prime less than $2^{\texttt{BIT}-1}$ (it's $2^{63}$ on a 64-bit machine), but it is possible to generalize to other fields/rings by some small modification.

It is recommended to use [mimalloc](https://github.com/microsoft/mimalloc) (or other similar library) to dynamically override the standard malloc, especially on Windows.

Example build commands (also add `-lpthread` if pthread is required by the compiler):

Standalone executable:
```bash
g++ main.cpp -o sparserref -O3 -std=c++20 -I$INCULDE -L$LIB -lflint -lgmp
```

Shared library for [Wolfram LibraryLink](https://reference.wolfram.com/language/guide/LibraryLink.html) API (used by Mathematica package [SparseRREF.wl](SparseRREF.wl), see below):
```bash
g++ sprreflink.cpp -fPIC -shared -O3 -std=c++20 -o sprreflink.dll -I$MATHEMATICA_HOME/SystemFiles/IncludeFiles/C -I$INCLUDE -L$LIB -lflint -lgmp
```
where we assume that the FLINT library and GMP library are installed in `$LIB`, the header files are in `$INCLUDE`,
and `$MATHEMATICA_HOME/SystemFiles/IncludeFiles/C` is the path of Mathematica C header files.

### How to use the code

#### Standalone executable

The file [main.cpp](main.cpp) is an example to use the header-only library, the help is

```
Usage: SparseRREF [--help] [--version] [--output VAR]
                  [--kernel] [--method VAR] [--output-pivots]
                  [--field VAR] [--prime VAR] [--threads VAR]
                  [--verbose] [--print_step VAR]
                  [--no-backward-substitution]
                  input_file

(exact) Sparse Reduced Row Echelon Form v0.4.0

Positional arguments:
  input_file                       input file in the Matrix Market exchange formats (MTX) or
                                   Sparse/Symbolic Matrix Storage (SMS)

Optional arguments:
  -h, --help                       shows help message and exits
  -v, --version                    prints version information and exits
  -o, --output                     output file in MTX format [default: "<input_file>.rref"]
  -k, --kernel                     output the kernel (null vectors)
  -m, --method                     method of RREF  [default: 0]
  -op, --output-pivots             output pivots
  -F, --field                      QQ: rational field
                                   Zp or Fp: Z/p for a prime p [default: "QQ"]
  -p, --prime                      a prime number, only vaild when field is Zp  [default: "34534567"]
  -t, --threads                    the number of threads  [default: 1]
  -V, --verbose                    prints information of calculation
  -ps, --print_step                print step when --verbose is enabled [default: 100]
  -nb, --no-backward-substitution  no backward substitution
```

With `--verbose`, progress lines are rewritten in place when the output is a terminal and
printed one line per sample (newline-terminated, no carriage returns) when it is redirected
or piped, so `sprref -V ... > log.txt` yields a clean, parseable log.

The format (matrix market-like format) of input file looks like this:
```
% some comments
number_of_rows number_of_columns number_of_non_zero_values
row_index_1 column_index_1 value_1
row_index_2 column_index_2 value_2
...
row_index_n column_index_n value_n
```
where the row and column indices are 1-based. 

We also support the SMS format. It looks like this:
```
number_of_rows number_of_columns value_type
row_index_1 column_index_1 value_1
row_index_2 column_index_2 value_2
...
row_index_n column_index_n value_n
0 0 0 
```
The last line is a dummy line, which is used to indicate the end of the matrix.

The main function is `sparse_mat_rref`, its output is its pivots, and it modifies the input matrix $M$ to its RREF $\Lambda$.

#### Mathematica API

The Mathematica package [SparseRREF.wl](SparseRREF.wl) allows to call functions exported in [sprreflink.cpp](sprreflink.cpp):

Example usage:
```mathematica
Needs["SparseRREF`"];
(* or: Needs["SparseRREF`", "/path/to/SparseRREF.wl"]; *)

(* rationals *)
mat = SparseArray @ { {1, 0, 2}, {1/2, 1/3, 1/4} };
rref = SparseRREF[mat];
{rref, kernel, pivots} = SparseRREF[mat, "OutputMode" -> "RREF,Kernel,Pivots", "Method" -> "Right", "BackwardSubstitution" -> True, "Threads" -> $ProcessorCount, "Verbose" -> True, "PrintStep" -> 10];

(* progress lines can be forwarded to the kernel while the computation runs *)
log = OpenWrite["rref.log"];
(* or "LogStream" -> Automatic to print them in this session *)
rref = SparseRREF[mat, "LogStream" -> log, "PrintStep" -> 10];
Close[log];
SparseRREFLog[] (* the log of that call, as a String *)

(* in a notebook, "LogStream" -> "Dynamic" refreshes the progress in a single
   temporary output cell instead of emitting one output cell per line *)
rref = SparseRREF[mat, "LogStream" -> "Dynamic", "PrintStep" -> 10];
SparseRREFLog[] (* every line, not only the last one shown in the cell *)

(* integers mod p *)
mat = SparseArray @ { {10, 0, 20}, {30, 40, 50} };
p = 7;
{rref, kernel} = SparseRREF[mat, Modulus -> p, "OutputMode" -> "RREF,Kernel", "Method" -> "Hybrid", "Threads" -> 0];
```

To use this package, you have to compile [sprreflink.cpp](sprreflink.cpp) to a shared library (`sprreflink.dll` on Windows, `sprreflink.so` on Linux, `sprreflink.dylib` on macOS) in the same directory with [SparseRREF.wl](SparseRREF.wl).

See comments in [SparseRREF.wl](SparseRREF.wl) for more details.

### BenchMark

We compare it with [Spasm](https://github.com/cbouilla/spasm). Platform and Configuration: 

	CPU: Intel(R) Core(TM) Ultra 9 185H (6P+8E+2LPE)
	MEM: 24.5G + SWAP on PCIE4.0 SSD 
	OS: Arch Linux x86-64
	Compiler: gcc (GCC) 15.2.1 20250813 with mimalloc
	FLINT: v3.1.2
	SparseRREF: v0.3.4
	Prime number: 1073741827 ~ 2^30
	Configuration: 
	  - Spasm: Default configuration for Spasm, first spasm_echelonize and then spasm_rref
	  - SparseRREF: -V -t 16 -F Zp -p 1073741827

First two test matrices come from https://hpac.imag.fr, bs comes from symbol bootstrap, ibp comes from IBP of Feynman integrals:

| Matrix   | (#row, #col, #non-zero-values, rank)   | Spasm (echelonize + rref)    | SparseRREF |
| -------- | -------------------------------------- | ---------------------------- | ---------- |
| GL7d24   | (21074, 105054, 593892, 18549)         | 20.8001s + 38.2s             | 2.93s      |
| M0,6-D10 | (1274688, 616320, 5342400, 493432)     | 49.9s + 19.3s                | 49.39s     |
| bs-1     | (202552, 64350, 11690309, 62130)       | 4.19241s + 1.1s              | 0.68s      |
| bs-2     | (709620, 732600, 48819232, 709620)     | too slow                     | 149.68s    |
| bs-3     | (10011551, 2958306, 33896262, 2867955) | 484s + 327.1s                | 34.00s     |
| ibp-1    | (69153, 73316, 1117324, 58252)         | (rank is wrong) 2543.92s + ? | 2.96s      |
| ibp-2    | (169323, 161970, 2801475, 135009)      | too slow                     | 15.27s     |

Some tests for Spasm are slow since the physical memory is not enough, and it uses swap. In the most of cases,
SparseRREF uses less memory than Spasm since its result has less non zero values.

### TODO

* Add document.
* Improve the algorithms.
* Add PLUQ decomposition.
* Add more fields/rings.
* Add more functions and parameters to Mathematica API

### Citing

```bibtex
@software{SparseRREF,
  author    = {Li, Zhenjie},
  title     = {{SparseRREF}: Exact Sparse Reduced Row Echelon Form in {C++}},
  version   = {0.4.0},
  doi       = {10.5281/zenodo.21278863},
  url       = {https://github.com/munuxi/SparseRREF}
}
```
