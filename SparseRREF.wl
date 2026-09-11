(* ::Package:: *)

(* 
  Mathematica interface for SparseRREF library.
  SparseRREF is a C++ library that computes exact RREF with row and column permutations of a sparse matrix over finite field or rational field.
  See details at https://github.com/munuxi/SparseRREF
  
  Prerequisites:
  - Compile sprreflink.cpp to shared library sprreflink.$EXT ($EXT = "dll" on Windows, "so" on Linux, "dylib" on macOS)
  - Store SparseRREF.wl in the same directory.
  
  Available functions:
  --------------------
  SparseRREF[mat, opts]
    Computes Row Reduced Echelon Form (and optionally kernel/pivots).
    
    Options:
    - "Modulus":
      0: compute over rational field (default).
      prime p: compute over finite field Z/p.
    - "OutputMode":
      0, "RREF": returns rref (default).
      1, "RREF,Kernel": returns {rref, kernel}.
      2, "RREF,Pivots": returns {rref, pivots}.
      3, "RREF,Kernel,Pivots": returns {rref, kernel, pivots}.
    - "Method":
      0, "RightAndLeft": right and left search (default).
      1, "Right": only right search (chooses the leftmost independent columns as pivots).
      2, "Hybrid": hybrid.
    - "BackwardSubstitution":
      True: submatrix rref[[ pivots[[All,1]], pivots[[All,2]] ]] is an identity matrix (default).
      False: submatrix is upper triangular.
    - "Threads": number of threads (Integer >= 0, with 0 meaning automatic).
    - "Verbose": True | False.
    - "PrintStep": Integer (print progress every n steps).
    - "LogStream":
      None: progress output is written to the kernel's stdout (default).
      Automatic: progress lines are printed in this session as they are produced.
      "Dynamic": progress is shown in a single temporary output cell, which is refreshed
        a few times per second while the computation runs. Use this in a notebook: the
        library may emit hundreds of lines, and appending all of them as separate cells
        is what makes the front end sluggish. Without a front end "Dynamic" falls back to
        Automatic.
      stream: progress lines are written to the OutputStream stream as they are produced.
      All but None also collect the lines, which SparseRREFLog[] then returns; they imply
      "Verbose" -> True, since the library only reports when it is verbose.
      The collected lines are kept even if the computation is aborted.
    
  SparseRREFLog[]
    Returns the log of the most recent SparseRREF call that used the "LogStream"
    option, as a String with one line per entry.
    
  SparseMatMul[matA, matB, opts]
    Computes the matrix multiplication of two sparse matrices.
    
    Options:
    - "Modulus":
      0: compute over rational field (default).
      prime p: compute over finite field Z/p.
    - "Threads": number of threads (Integer >= 0, with 0 meaning automatic).
    
  SparseTensorContract[tensorA, tensorB, IndexPairs, opts]
    Computes the tensor contraction of two sparse tensors, with the idxA-th index of tensorA contracted with the idxB-th index of tensorB for each {idxA, idxB} in IndexPairs.
    
    Options:
    - "Modulus":
      0: compute over rational field (default).
      prime p: compute over finite field Z/p.
    - "Threads": number of threads (Integer >= 0, with 0 meaning automatic).
    
  SparseTensorDot[tensorA, tensorB, opts]
    Computes the dot product of two sparse tensors.
    
    Options:
    - "Modulus":
      0: compute over rational field (default).
      prime p: compute over finite field Z/p.
    - "Threads": number of threads (Integer >= 0, with 0 meaning automatic).
    
  SparseMatInv[mat, opts]
    Computes the matrix inverse of a sparse matrix.
    
    Options:
    - "Modulus":
      0: compute over rational field (default).
      prime p: compute over finite field Z/p.
    - "Threads": number of threads (Integer >= 0, with 0 meaning automatic).

  Example usage:
  --------------
    Needs["SparseRREF`"];
    (* or: Needs["SparseRREF`", "/path/to/SparseRREF.wl"]; *)
    
    (* --- RREF Example --- *)    
    (* Rationals *)
    mat = SparseArray @ { {1, 0, 2}, {1/2, 1/3, 1/4} };
    rref = SparseRREF[mat];
    {rref, kernel, pivots} = SparseRREF[mat, "OutputMode" -> "RREF,Kernel,Pivots", "Method" -> "Right", "BackwardSubstitution" -> True, "Threads" -> $ProcessorCount, "Verbose" -> True, "PrintStep" -> 10];
    
    (* --- Log stream example --- *)
    log = OpenWrite["rref.log"];
    (* "LogStream" -> Automatic prints the progress in this session instead *)
    rref = SparseRREF[mat, "LogStream" -> log, "PrintStep" -> 10];
    Close[log];
    (* the log of that call is also available as a string *)
    logText = SparseRREFLog[];
    
    (* --- Log stream in a notebook --- *)
    (* "LogStream" -> "Dynamic" refreshes the progress in a single temporary cell
       instead of emitting one output cell per line *)
    rref = SparseRREF[mat, "LogStream" -> "Dynamic", "PrintStep" -> 10];
    (* the lines that were displayed, all of them, are still available afterwards *)
    logText = SparseRREFLog[];
    
    (* Finite Field *)
    mat = SparseArray @ { {10, 0, 20}, {30, 40, 50} };
    p = 7;
    {rref, kernel} = SparseRREF[mat, Modulus -> p, "OutputMode" -> "RREF,Kernel", "Method" -> "Hybrid", "Threads" -> 1];
    
    (* --- MatMul Example --- *)
    (* Rationals *)
    matA = SparseArray @ { {1, 0, 2}, {1/2, 1/3, 1/4} };
    matB = SparseArray @ { {0, 1}, {1, 0}, {1, 1} };
    matC = SparseMatMul[matA, matB, "Threads" -> 0];
    
    (* Finite Field *)
    matA = SparseArray @ { {10, 0, 20}, {30, 40, 50} };
    matB = SparseArray @ { {1, 2}, {3, 4}, {5, 6} };
    p = 11;
    matC = SparseMatMul[matA, matB, Modulus -> p, "Threads" -> 1];

    (* --- TensorContract Example --- *)
    (* Rationals *)
    tensorA = SparseArray @ LeviCivitaTensor[3];
    tensorB = SparseArray @ { {0, 1}, {1, 0}, {1, 1} };
    tensorC = SparseTensorContract[tensorA, tensorB, {{2, 1}}, "Threads" -> 0];
    (* tensorC == TensorContract[TensorProduct[tensorA, tensorB], {{2, 4}}] *)

    (* --- TensorDot Example --- *)
    (* Rationals *)
    tensorA = SparseArray @ LeviCivitaTensor[3];
    tensorB = SparseArray @ { {0, 1}, {1, 0}, {1, 1} };
    tensorC = SparseTensorDot[tensorA, tensorB, "Threads" -> 0];
    (* tensorC == tensorA . tensorB *)

    (* --- MatInv Example --- *)
    (* Rationals *)
    mat = SparseArray @ { {1, 0, 2}, {1/2, 1/3, 1/4}, {-1, 0, 1} };
    invMat = SparseMatInv[mat, "Threads" -> 0];

    (* Finite Field *)
    mat = SparseArray @ { {1, 2, 0}, {2, 1, 0}, {0, 0, 3} };
    p = 7;
    invMat = SparseMatInv[mat, Modulus -> p, "Threads" -> 1];
    (* invMat == SparseArray @ Inverse[mat, Modulus -> p] *)
*)


BeginPackage["SparseRREF`"];

Unprotect["SparseRREF`*"];


Options[SparseRREF] = {
  Modulus -> 0,
  "OutputMode" -> "RREF",
  "Method" -> "RightAndLeft",
  "BackwardSubstitution" -> True,
  "Threads" -> 1,
  "Verbose" -> False,
  "PrintStep" -> 100,
  "LogStream" -> None
};

SparseRREF::usage =
  "SparseRREF[mat, opts] computes the exact RREF of a sparse rational matrix " <>
  "or a sparse integer matrix modulo prime p (if Modulus -> p is specified). " <>
  "Default options: " <> ToString @ Options @ SparseRREF;

SyntaxInformation[SparseRREF] = {"ArgumentsPattern" -> {_, OptionsPattern[]}}

SparseRREF::findlib = "SparseRREF library \"`1`\" not found at `2`";
SparseRREF::optionvalue = "Invalid SparseRREF option value: `1` -> `2`. Allowed values: `3`";
SparseRREF::rettype = "SparseRREF should return SparseArray or List, but returned: `1`";


SparseRREFLog::usage =
  "SparseRREFLog[] returns the log of the most recent SparseRREF call that used the " <>
  "\"LogStream\" option, as a String with one line per entry. The log of an aborted " <>
  "call is returned as well.";


Options[SparseMatMul] = {
  Modulus -> 0,
  "Threads" -> 1
};

SparseMatMul::usage =
  "SparseMatMul[matA, matB, opts] computes the matrix multiplication of two sparse rational matrices " <>
  "or two sparse integer matrices modulo prime p (if Modulus -> p is specified). " <>
  "Default options: " <> ToString @ Options @ SparseMatMul;

SyntaxInformation[SparseMatMul] = {"ArgumentsPattern" -> {_, _, OptionsPattern[]}}

SparseMatMul::optionvalue = "Invalid SparseMatMul option value: `1` -> `2`. Allowed values: `3`";
SparseMatMul::rettype = "SparseMatMul should return SparseArray, but returned: `1`";
SparseMatMul::dims = "Matrices `1` and `2` have incompatible dimensions.";


Options[SparseTensorContract] = {
  Modulus -> 0,
  "Threads" -> 1
};

SparseTensorContract::usage =
  "SparseTensorContract[tensorA, tensorB, IndexPairs, opts] computes the tensor contraction of " <>
  "two sparse rational tensors or two sparse integer tensors modulo prime p (if Modulus -> p is specified), " <>
  "with the idxA-th index of tensorA contracted with the idxB-th index of tensorB for each {idxA, idxB} in IndexPairs. " <>
  "Default options: " <> ToString @ Options @ SparseTensorContract;

SyntaxInformation[SparseTensorContract] = {"ArgumentsPattern" -> {_, _, _, OptionsPattern[]}}

SparseTensorContract::optionvalue = "Invalid SparseTensorContract option value: `1` -> `2`. Allowed values: `3`";
SparseTensorContract::rettype = "SparseTensorContract should return SparseArray, but returned: `1`";
SparseTensorContract::dims = "Tensors `1` and `2` have incompatible dimensions for contraction.";
SparseTensorContract::indexpairs = "IndexPairs `1` is invalid.";


Options[SparseTensorDot] = Options[SparseTensorContract];

SparseTensorDot::usage =
  "SparseTensorDot[tensorA, tensorB, opts] computes the dot product of two sparse rational tensors " <>
  "or two sparse integer tensors modulo prime p (if Modulus -> p is specified). " <>
  "Default options: " <> ToString @ Options @ SparseTensorDot;

SyntaxInformation[SparseTensorDot] = {"ArgumentsPattern" -> {_, _, OptionsPattern[]}}

SparseTensorDot::optionvalue = "Invalid SparseTensorDot option value: `1` -> `2`. Allowed values: `3`";
SparseTensorDot::rettype = "SparseTensorDot should return SparseArray, but returned: `1`";
SparseTensorDot::dims = "Tensors `1` and `2` have incompatible dimensions for dot product.";


Options[SparseMatInv] = {
  Modulus -> 0,
  "Threads" -> 1
};

SparseMatInv::usage =
  "SparseMatInv[mat, opts] computes the matrix inverse of a sparse rational matrix " <>
  "or a sparse integer matrix modulo prime p (if Modulus -> p is specified). " <>
  "Default options: " <> ToString @ Options @ SparseMatInv;

SyntaxInformation[SparseMatInv] = {"ArgumentsPattern" -> {_, OptionsPattern[]}}

SparseMatInv::optionvalue = "Invalid SparseMatInv option value: `1` -> `2`. Allowed values: `3`";
SparseMatInv::rettype = "SparseMatInv should return SparseArray, but returned: `1`";
SparseMatInv::nonsquare = "Matrix `1` is not square.";
SparseMatInv::singular = "Matrix `1` is singular and not invertible.";


Begin["`Private`"];

(* Load SparseRREF library *)

$sparseRREFDirectory = DirectoryName[$InputFileName];
$sparseRREFLibName = "sprreflink";

(* TODO: shall we search in all directories from $LibraryPath? *)
$sparseRREFLib = FindLibrary @ FileNameJoin @ {$sparseRREFDirectory, $sparseRREFLibName};

If[FailureQ[$sparseRREFLib],
  Message[SparseRREF::findlib, $sparseRREFLibName, $sparseRREFDirectory];
];


ratRREFLibFunction =
  LibraryFunctionLoad[
    $sparseRREFLib,
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
    $sparseRREFLib,
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


ratMatMulLibFunction =
  LibraryFunctionLoad[
    $sparseRREFLib,
    "sprref_rat_matmul",
    {
      {LibraryDataType[ByteArray], "Constant"},
      {LibraryDataType[ByteArray], "Constant"},
      Integer
    },
    {LibraryDataType[ByteArray], Automatic}
  ];

modMatMulLibFunction =
  LibraryFunctionLoad[
    $sparseRREFLib,
    "sprref_mod_matmul",
    {
      {LibraryDataType[SparseArray], "Constant"},
      {LibraryDataType[SparseArray], "Constant"},
      Integer,
      Integer
    },
    {LibraryDataType[SparseArray], Automatic}
  ];


ratTensorContractLibFunction =
  LibraryFunctionLoad[
    $sparseRREFLib,
    "sprref_rat_tensor_contract",
    {
      {LibraryDataType[ByteArray], "Constant"},
      {LibraryDataType[ByteArray], "Constant"},
      {LibraryDataType[List, Integer, 1], "Constant"},
      {LibraryDataType[List, Integer, 1], "Constant"},
      Integer
    },
    {LibraryDataType[ByteArray], Automatic}
  ];

modTensorContractLibFunction =
  LibraryFunctionLoad[
    $sparseRREFLib,
    "sprref_mod_tensor_contract",
    {
      {LibraryDataType[SparseArray], "Constant"},
      {LibraryDataType[SparseArray], "Constant"},
      Integer,
      {LibraryDataType[List, Integer, 1], "Constant"},
      {LibraryDataType[List, Integer, 1], "Constant"},
      Integer
    },
    {LibraryDataType[SparseArray], Automatic}
  ];

ratMatInvLibFunction =
  LibraryFunctionLoad[
    $sparseRREFLib,
    "sprref_rat_matinv",
    {
      {LibraryDataType[ByteArray], "Constant"},
      Integer
    },
    {LibraryDataType[ByteArray], Automatic}
  ];

modMatInvLibFunction =
  LibraryFunctionLoad[
    $sparseRREFLib,
    "sprref_mod_matinv",
    {
      {LibraryDataType[SparseArray], "Constant"},
      Integer,
      Integer
    },
    {LibraryDataType[SparseArray], Automatic}
  ];


(* Helper functions: parse options, validate etc. *)

throwOptionError[optionName_?StringQ, optionValue_, allowedValues_] := (
  Message[
    SparseRREF::optionvalue,
    InputForm[optionName], 
    InputForm[optionValue],
    allowedValues
  ];
  Throw[$Failed];
);

methodToInteger = <|
  0 -> 0,
  1 -> 1,
  2 -> 2,
  "RightAndLeft" -> 0,
  "Right" -> 1,
  "Hybrid" -> 2
|>;

(* TODO: maybe allow arbitrary lists, e.g. {"Pivots", "RREF", "Kernel"}? *)
outputModeToInteger = <|
  0 -> 0,
  1 -> 1,
  2 -> 2,
  3 -> 3,
  "RREF" -> 0,
  "RREF,Kernel" -> 1,
  "RREF,Pivots" -> 2,
  "RREF,Kernel,Pivots" -> 3
|>;

parseModulus[0] := 0;
parseModulus[p_?PrimeQ] /; p > 0 := p;
parseModulus[p_] := throwOptionError["Modulus", p, "0 or prime number"];

parseMethod[method_] :=
  With[
    {$res = methodToInteger[method]},
    If[MissingQ[$res],
      throwOptionError["Method", method, InputForm @ Keys @ methodToInteger],
      $res
    ]
  ];

parseOutputMode[outputMode_] :=
  With[
    {$res = outputModeToInteger[outputMode]},
    If[MissingQ[$res],
      throwOptionError["OutputMode", outputMode, InputForm @ Keys @ outputModeToInteger],
      $res
    ]
  ];

parseBackwardSubstitution[b_?BooleanQ] := b;
parseBackwardSubstitution[b_] := throwOptionError["BackwardSubstitution", b, {True, False}];

parseThreads[threads_?IntegerQ] /; threads >= 0 := threads;
parseThreads[threads_] := throwOptionError["Threads", threads, "0,1,2..." ];

parseVerbose[b_?BooleanQ] := b;
parseVerbose[b_] := throwOptionError["Verbose", b, {True, False}];

parsePrintStep[ps_?IntegerQ] /; ps > 0 := ps;
parsePrintStep[ps_] := throwOptionError["PrintStep", ps, "1,2,3..."];

parseLogStream[None | Null] := None;
parseLogStream[Automatic] := Automatic;
parseLogStream["Dynamic"] := "Dynamic";
parseLogStream[stream_OutputStream] := stream;
parseLogStream[ls_] := throwOptionError["LogStream", ls, "None, Automatic, \"Dynamic\" or an OutputStream"];

logStreamQ[None] = False;
logStreamQ[_] = True;


(* Kernel log stream *)

(* Progress lines produced by the library are evaluated back into the kernel as
   sprrefLogPush["..."], as soon as they are produced. The destination is taken
   from $logDestination, which SparseRREF[] sets up for the duration of a call,
   and the lines are collected in $logLines so that SparseRREFLog[] can return
   them afterwards. *)
$logDestination = None;
$logLines = {};

(* Last line received so far, how many lines arrived, and the delay between two
   refreshes of the temporary cell that "LogStream" -> "Dynamic" uses. *)
$logLatest = "";
$logCount = 0;
$logRefresh = 0.2;

sprrefLogPush[line_String] := (
  (* nested instead of AppendTo[], to keep this O(1) per line *)
  $logLines = {$logLines, line};
  $logCount = $logCount + 1;
  Switch[$logDestination,
    Automatic, Print[line],
    "Dynamic", $logLatest = line,
    _OutputStream, WriteString[$logDestination, line, "\n"]; Flush[$logDestination],
    _, Null
  ];
  Null
);

checkResult[msg_, res_, pattern_] :=
  If[MatchQ[res, pattern],
    res,
    Message[msg, res];
    Throw[$Failed]
  ];
SetAttributes[checkResult, HoldFirst];


(* Temporary output cell used by "LogStream" -> "Dynamic": a single cell that is
   refreshed in place, instead of one output cell per progress line. The cell is
   removed when the computation finishes (or is aborted); the lines that went
   through it stay available through SparseRREFLog[]. *)
printLogCell[] := PrintTemporary @ Dynamic[
  Row[{"SparseRREF: ", $logCount, " line(s)  ", $logLatest}],
  UpdateInterval -> $logRefresh,
  TrackedSymbols :> {$logCount, $logLatest}
];


(* Define public function SparseRREF[] *)

SparseRREF[mat_SparseArray, opts : OptionsPattern[] ] :=
  Catch @ With[
    {
      $modulus = parseModulus @ OptionValue["Modulus"],
      $outputMode = parseOutputMode @ OptionValue["OutputMode"],
      $method = parseMethod @ OptionValue["Method"],
      $backwardSubstitution = parseBackwardSubstitution @ OptionValue["BackwardSubstitution"],
      $threads = parseThreads @ OptionValue["Threads"],
      $verbose = parseVerbose @ OptionValue["Verbose"],
      $printStep = parsePrintStep @ OptionValue["PrintStep"],
      $logStream = parseLogStream @ OptionValue["LogStream"]
    },
    If[logStreamQ[$logStream], $logLines = {}];
    Block[
      {
        (* without a front end there is no cell to refresh, print the lines instead *)
        $logDestination = If[$logStream === "Dynamic" && ! $Notebooks, Automatic, $logStream],
        $logLatest = "",
        $logCount = 0,
        $logCell = If[$logStream === "Dynamic" && $Notebooks, printLogCell[], Null],
        $result
      },
      Internal`WithLocalSettings[
        Null,
        $result = checkResult[
          SparseRREF::rettype,
          If[$modulus == 0,
            ratRREF[mat, $outputMode, $method, $backwardSubstitution, $threads,
              $verbose || logStreamQ[$logStream], $printStep, logStreamQ[$logStream]],
            modRREF[mat, $modulus, $outputMode, $method, $backwardSubstitution, $threads,
              $verbose || logStreamQ[$logStream], $printStep, logStreamQ[$logStream]]
          ],
          _SparseArray | _List
        ],
        (* the temporary cell is normally removed when the evaluation finishes; remove it
           here too, so that a computation that is aborted does not leave it behind *)
        If[MatchQ[$logCell, _CellObject], Quiet @ NotebookDelete[$logCell]]
      ];
      $result
    ]
  ];

SparseRREFLog[] := StringRiffle[Flatten[$logLines], "\n"];

ratRREF[
    mat_SparseArray,
    outputMode_?IntegerQ,
    method_?IntegerQ,
    backwardSubstitution_?BooleanQ,
    threads_?IntegerQ,
    verbose_?BooleanQ,
    printStep_?IntegerQ,
    log_?BooleanQ
  ] :=
  BinaryDeserialize @ ratRREFLibFunction[
    BinarySerialize[mat],
    outputMode,
    method,
    backwardSubstitution,
    threads,
    verbose,
    printStep,
    log
  ];

modRREF[
    mat_SparseArray,
    p_?PrimeQ,
    outputMode_?IntegerQ,
    method_?IntegerQ,
    backwardSubstitution_?BooleanQ,
    threads_?IntegerQ,
    verbose_?BooleanQ,
    printStep_?IntegerQ,
    log_?BooleanQ
  ] :=
  BinaryDeserialize @ modRREFLibFunction[
    mat,
    p,
    outputMode,
    method,
    backwardSubstitution,
    threads,
    verbose,
    printStep,
    log
  ];


(* Define public function SparseMatMul[] *)

SparseMatMul[matA_SparseArray, matB_SparseArray, opts : OptionsPattern[] ] :=
  Catch @ With[
    {
      $modulus = parseModulus @ OptionValue["Modulus"],
      $threads = parseThreads @ OptionValue["Threads"],
      $dimA = Dimensions[matA],
      $dimB = Dimensions[matB]
    },
    If[Last[$dimA] != First[$dimB],
      Message[SparseMatMul::dims, matA, matB];
      Throw[$Failed];
    ];
    checkResult[
      SparseMatMul::rettype,
      If[$modulus == 0,
        ratMatMul[matA, matB, $threads],
        modMatMul[matA, matB, $modulus, $threads]
      ],
      _SparseArray
    ]
  ];

ratMatMul[
    matA_SparseArray,
    matB_SparseArray,
    threads_?IntegerQ
  ] :=
  BinaryDeserialize @ ratMatMulLibFunction[
    BinarySerialize[matA],
    BinarySerialize[matB],
    threads
  ];

modMatMul[
    matA_SparseArray,
    matB_SparseArray,
    p_?PrimeQ,
    threads_?IntegerQ
  ] :=
  modMatMulLibFunction[
    matA,
    matB,
    p,
    threads
  ];


(* Define public function SparseTensorContract[] *)

SparseTensorContract[
    tensorA_SparseArray,
    tensorB_SparseArray,
    indexPairs : {{_Integer ..} ..},
    opts : OptionsPattern[]
  ] :=
  Catch @ With[
    {
      $modulus = parseModulus @ OptionValue["Modulus"],
      $threads = parseThreads @ OptionValue["Threads"],
      $dimA = Dimensions[tensorA],
      $dimB = Dimensions[tensorB],
      $idxA = indexPairs[[All, 1]],
      $idxB = indexPairs[[All, 2]]
    },
    If[
      Or[
        Max[$idxA] > Length[$dimA],
        Max[$idxB] > Length[$dimB],
        Min[$idxA] < 1,
        Min[$idxB] < 1,
        Length[$idxA] != Length[$idxB]
      ],
      Message[SparseTensorContract::indexpairs, indexPairs];
      Throw[$Failed];
    ];
    If[And @@ Thread[ $dimA[[ $idxA ]] == $dimB[[ $idxB ]] ] == False,
      Message[SparseTensorContract::dims, tensorA, tensorB];
      Throw[$Failed];
    ];
    checkResult[
      SparseTensorContract::rettype,
      If[$modulus == 0,
        ratTensorContract[tensorA, tensorB, $idxA, $idxB, $threads],
        modTensorContract[tensorA, tensorB, $modulus, $idxA, $idxB, $threads]
      ],
      _SparseArray
    ]
  ];

ratTensorContract[
    tensorA_SparseArray,
    tensorB_SparseArray,
    idxA : {_Integer ..},
    idxB : {_Integer ..},
    threads_?IntegerQ
  ] :=
  BinaryDeserialize @ ratTensorContractLibFunction[
    BinarySerialize[tensorA],
    BinarySerialize[tensorB],
    idxA,
    idxB,
    threads
  ];

modTensorContract[
    tensorA_SparseArray,
    tensorB_SparseArray,
    p_?PrimeQ,
    idxA : {_Integer ..},
    idxB : {_Integer ..},
    threads_?IntegerQ
  ] :=
  modTensorContractLibFunction[
    tensorA,
    tensorB,
    p,
    idxA,
    idxB,
    threads
  ];

SparseTensorDot[
    tensorA_SparseArray,
    tensorB_SparseArray,
    opts : OptionsPattern[]
  ] :=
  Catch @ With[
    {
      $modulus = parseModulus @ OptionValue["Modulus"],
      $threads = parseThreads @ OptionValue["Threads"],
      $dimA = Dimensions[tensorA],
      $dimB = Dimensions[tensorB]
    },
    If[Last[$dimA] != First[$dimB],
      Message[SparseTensorDot::dims, tensorA, tensorB];
      Throw[$Failed];
    ];
    checkResult[
      SparseTensorDot::rettype,
      If[$modulus == 0,
        ratTensorContract[tensorA, tensorB, {Length[$dimA]}, {1}, $threads],
        modTensorContract[tensorA, tensorB, $modulus, {Length[$dimA]}, {1}, $threads]
      ],
      _SparseArray
    ]
  ];


(* Define public function SparseMatInv[] *)

SparseMatInv[mat_SparseArray, opts : OptionsPattern[] ] :=
  Catch @ With[
    {
      $modulus = parseModulus @ OptionValue["Modulus"],
      $threads = parseThreads @ OptionValue["Threads"],
      $dim = Dimensions[mat]
    },
    If[Length[$dim] != 2 || $dim[[1]] != $dim[[2]],
      Message[SparseMatInv::nonsquare, mat];
      Throw[$Failed];
    ];
    checkResult[
      SparseMatInv::rettype,
      If[$modulus == 0,
        ratMatInv[mat, $threads],
        modMatInv[mat, $modulus, $threads]
      ],
      _SparseArray
    ]
  ];

ratMatInv[
    mat_SparseArray,
    threads_?IntegerQ
  ] :=
  BinaryDeserialize @ ratMatInvLibFunction[
    BinarySerialize[mat],
    threads
  ];

modMatInv[
    mat_SparseArray,
    p_?PrimeQ,
    threads_?IntegerQ
  ] :=
  modMatInvLibFunction[
    mat,
    p,
    threads
  ];


With[{syms = Names["SparseRREF`*"]},
  SetAttributes[syms, {Protected, ReadProtected}]
];  

End[];

EndPackage[];
