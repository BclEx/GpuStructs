// vdbe.h
#include <stdio.h>

namespace Core
{
	//#include "opcodes.h"

	// The Vdbe.aColName array contains 5n Mem structures, where n is the number of columns of data returned by the statement.
#define COLNAME_NAME     0
#define COLNAME_DECLTYPE 1
#define COLNAME_DATABASE 2
#define COLNAME_TABLE    3
#define COLNAME_COLUMN   4
#ifdef ENABLE_COLUMN_METADATA
#define COLNAME_N 5	// Number of COLNAME_xxx symbols
#else
#ifdef OMIT_DECLTYPE
#define COLNAME_N 1	// Store only the name
#else
#define COLNAME_N 2	// Store the name and decltype
#endif
#endif
#define ADDR(X)  (-1-(X))

	typedef unsigned bft;
	class Vdbe
	{
	public:

		enum P4T : int8
		{
			P4T_NOTUSED = 0,	// The P4 parameter is not used
			P4T_DYNAMIC = -1,	// Pointer to a string obtained from sqliteMalloc()
			P4T_STATIC = -2,	// Pointer to a static string
			P4T_COLLSEQ = -4,	// P4 is a pointer to a CollSeq structure
			P4T_FUNCDEF = -5,	// P4 is a pointer to a FuncDef structure
			P4T_KEYINFO = -6,	// P4 is a pointer to a KeyInfo structure
			P4T_VDBEFUNC =-7,	// P4 is a pointer to a VdbeFunc structure
			P4T_MEM  = -8,		// P4 is a pointer to a Mem*    structure
			P4T_TRANSIENT = 0,	// P4 is a pointer to a transient string
			P4T_VTAB = -10,		// P4 is a pointer to an sqlite3_vtab structure
			P4T_MPRINTF = -11,	// P4 is a string obtained from sqlite3_mprintf()
			P4T_REAL = -12,		// P4 is a 64-bit floating point value
			P4T_INT64 = -13,	// P4 is a 64-bit signed integer
			P4T_INT32 = -14,	// P4 is a 32-bit signed integer
			P4T_INTARRAY = -15,	// P4 is a vector of 32-bit integers
			P4T_SUBPROGRAM = -18, // P4 is a pointer to a SubProgram structure
			P4T_ADVANCE = -19,	// P4 is a pointer to BtreeNext() or BtreePrev()
			// When adding a P4 argument using P4_KEYINFO, a copy of the KeyInfo structure is made.  That copy is freed when the Vdbe is finalized.  But if the
			// argument is P4_KEYINFO_HANDOFF, the passed in pointer is used.  It still gets freed when the Vdbe is finalized so it still should be obtained
			// from a single sqliteMalloc().  But no copy is made and the calling function should *not* try to free the KeyInfo.
			P4T_KEYINFO_HANDOFF = -16,
			P4T_KEYINFO_STATIC = -17,
		};

		struct SubProgram;
		struct VdbeOp
		{
			uint8 Opcode;   // What operation to perform
			P4T P4Type;		// One of the P4_xxx constants for p4
			uint8 Opflags;  // Mask of the OPFLG_* flags in opcodes.h
			uint8 P5;       // Fifth parameter is an unsigned character
			int P1;         // First operand
			int P2;         // Second parameter (often the jump destination)
			int P3;         // The third parameter
			union
			{             
				int I;					// Integer value if p4type==P4_INT32
				void *P;				// Generic pointer
				char *Z;				// Pointer to data for string (char array) types
				int64 *I64;				// Used when p4type is P4_INT64
				double *Real;			// Used when p4type is P4_REAL
				FuncDef *Func;			// Used when p4type is P4_FUNCDEF
				VdbeFunc *VdbeFunc;		// Used when p4type is P4_VDBEFUNC
				CollSeq *Coll;			// Used when p4type is P4_COLLSEQ
				Mem *Mem;				// Used when p4type is P4_MEM
				VTable *VTable;			// Used when p4type is P4_VTAB
				KeyInfo *KeyInfo;		// Used when p4type is P4_KEYINFO
				int *Is;				// Used when p4type is P4_INTARRAY
				SubProgram *Program;	// Used when p4type is P4_SUBPROGRAM
				int (*Advance)(BtCursor *, int *);
			} P4;			// fourth parameter
#ifdef _DEBUG
			char *Comment;  // Comment to improve readability
#endif
#ifdef VDBE_PROFILE
			int Cnt;        // Number of times this instruction was executed
			uint64 Cycles;  // Total time spent executing this instruction
#endif
		};

		struct SubProgram
		{
			array_t<VdbeOp> *Ops;		// Array of opcodes for sub-program
			int Mems;                   // Number of memory cells required
			int Csrs;                   // Number of cursors required
			int Onces;                  // Number of OP_Once instructions
			void *Token;                // id that may be used to recursive triggers
			SubProgram *Next;           // Next sub-program already visited
		};

		struct VdbeOpList
		{
			uint8 Opcode;	// What operation to perform
			int8 P1;		// First operand
			int8 P2;		// Second parameter (often the jump destination)
			int8 P3;		// Third parameter
		};

		__device__ static Vdbe *Create(Context *);
		__device__ int AddOp0(int);
		__device__ int AddOp1(int, int);
		__device__ int AddOp2(int, int, int);
		__device__ int AddOp3(int, int, int, int);
		__device__ int AddOp4(int, int, int, int, const char *, int);
		__device__ int AddOp4Int(int, int, int, int, int);
		__device__ int AddOpList(int nOp, VdbeOpList const *aOp);
		__device__ void AddParseSchemaOp(int, char *);
		__device__ void ChangeP1(uint32 addr, int P1);
		__device__ void ChangeP2(uint32 addr, int P2);
		__device__ void ChangeP3(uint32 addr, int P3);
		__device__ void ChangeP5(uint8 P5);
		__device__ void JumpHere(int addr);
		__device__ void ChangeToNoop(int addr);
		__device__ void ChangeP4(int addr, const char *zP4, P4T N);
		__device__ void UsesBtree(int);
		__device__ VdbeOp *GetOp(int);
		__device__ int MakeLabel();
		__device__ void set_RunOnlyOnce();
		__device__ void Delete();
		__device__ static void ClearObject(Context *, Vdbe *);
		__device__ void MakeReady(Parse *);
		__device__ void ResolveLabel(int);
		__device__ int CurrentAddr();
#ifdef _DEBUG
		__device__ int AssertMayAbort(int);
		__device__ void set_Trace(FILE *);
#endif
		__device__ void ResetStepResult();
		__device__ void Rewind();
		__device__ void SetNumCols(int);
		__device__ int SetColName(int, int, const char *, void(*)(void *));
		__device__ void CountChanges();
		__device__ Context *get_Db();
		__device__ void SetSql(const char *z, int n, int);
		__device__ static void Swap(Vdbe *, Vdbe *);
		__device__ VdbeOp *TakeOpArray(int *, int *);
		__device__ Mem *GetValue(int, AFF);
		__device__ void SetVarmask(int);

		__device__ static void RecordUnpack(KeyInfo *, int, const void *, UnpackedRecord *);
		__device__ static int RecordCompare(int, const void *, UnpackedRecord *);
		__device__ static UnpackedRecord *AllocUnpackedRecord(KeyInfo *, char *, int, char **);

#ifndef OMIT_TRIGGER
		__device__ void LinkSubProgram(SubProgram *);
#endif

#ifndef NDEBUG
		//__device__ void Comment(const char*, ...);
		//__device__ void NoopComment(const char*, ...);
#define VdbeComment(X) Comment X
#define VdbeNoopComment(X) NoopComment X
#else
#define VdbeComment(X)
#define VdbeNoopComment(X)
#endif

	public: //was:private
		Context *Ctx;            // The database connection that owns this statement
		array_t<VdbeOp> Ops;        // Space to hold the virtual machine's program
		array_t<Mem> Mems;      // The memory locations
		Mem **Args;				// Arguments to currently executing user function
		Mem *ColNames;          // Column names to return
		Mem *ResultSet;			// Pointer to an array of results
		int OpsAlloc;           // Number of slots allocated for Ops[]
		array_t<int> *Labels;   // Space to hold the labels
		uint16 ResColumns;		// Number of columns in one row of the result set
		uint32 Magic;			// Magic number for sanity checking
		char *ErrMsg;			// Error message written here
		Vdbe *Prev, *Next;		// Linked list of VDBEs with the same Vdbe.db
		array_t<VdbeCursor *> Cursors;   // One element of this array for each open cursor
		array_t2<yVars, Mem> Vars;   // Values for the OP_Variable opcode.
		array_t2<yVars, char *> VarNames; // Name of variables
		uint32 CacheCtr;        // VdbeCursor row cache generation counter
		int PC;                 // The program counter
		RC RC_;					// Value to return
		uint8 ErrorAction;      // Recovery action to do in case of an error
		uint8 MinWriteFileFormat;  // Minimum file format for writable database files
		bft HasExplain:2;          // True if EXPLAIN present on SQL command
		bft InVtabMethod:2;     // See comments above
		bft ChangeCntOn:1;      // True to update the change-counter
		bft Expired:1;          // True if the VM needs to be recompiled
		bft RunOnlyOnce:1;      // Automatically expire on reset
		bft UsesStmtJournal:1;  // True if uses a statement journal
		bft ReadOnly:1;         // True for read-only statements
		bft IsPrepareV2:1;      // True if prepared with prepare_v2()
		bft DoingRerun:1;       // True if rerunning after an auto-reprepare
		int Changes;            // Number of db changes made since last reset
		yDbMask BtreeMask;      // Bitmask of db->aDb[] entries referenced
		yDbMask LockMask;       // Subset of btreeMask that requires a lock
		int StatementID;		// Statement number (or 0 if has not opened stmt)
		int Counters[3];        // Counters used by sqlite3_stmt_status()
#ifndef OMIT_TRACE
		int64 StartTime;        // Time when query started - used for profiling
#endif
		int64 FkConstraints;    // Number of imm. FK constraints this VM
		int64 StmtDefCons;		// Number of def. constraints when stmt started
		char *Sql;				// Text of the SQL statement that generated this
		void *FreeThis;         // Free this when deleting the vdbe
#ifdef _DEBUG
		FILE *Trace;            // Write an execution trace here, if not NULL
#endif
#ifdef ENABLE_TREE_EXPLAIN
		Explain *_explain;		// The explainer
		char *_explainString;    // Explanation of data structures
#endif
		array_t<VdbeFrame> Frames; // Parent frame
		VdbeFrame *DelFrames;   // List of frame objects to free on VM reset
		uint32 Expmask;         // Binding to these vars invalidates VM
		SubProgram *Programs;	// Linked list of all sub-programs used by VM
		array_t<uint8> OnceFlags; // Flags for OP_Once

		__device__ void FreeCursor(VdbeCursor *);
		__device__ void PopStack(int);
		__device__ static int CursorMoveto(VdbeCursor *);
#if defined(_DEBUG) || defined(VDBE_PROFILE)
		__device__ static void PrintOp(FILE *, int, VdbeOp *);
#endif
		__device__ static uint32 SerialTypeLen(uint32);
		__device__ static uint32 SerialType(Mem *, int);
		__device__ static uint32 SerialPut(unsigned char *, int, Mem *, int);
		__device__ static uint32 SerialGet(const unsigned char *, uint32, Mem *);
		__device__ static void DeleteAuxData(VdbeFunc *, int);

		__device__ static int sqlite2BtreeKeyCompare(BtCursor *, const void *, int, int, int *);
		__device__ static int IdxKeyCompare(VdbeCursor *, UnpackedRecord *, int *);
		__device__ static int IdxRowid(Context *, BtCursor *, int64 *);
		__device__ static int MemCompare(const Mem *mem1, const Mem *mem2, const CollSeq *coll); //@.mem
		__device__ int Exec();
		__device__ int List();
		__device__ int Halt();



		__device__ static const char *sqlite3OpcodeName(int);
		__device__ int CloseStatement(Vdbe *, int);
		__device__ static void FrameDelete(VdbeFrame *);
		__device__ static int FrameRestore(VdbeFrame *);
		__device__ static void MemStoreType(Mem *mem);
		__device__ int TransferError();

#pragma region Vdbe+Api

		// name1
		__device__ static RC Finalize(Vdbe *p);
		__device__ static RC Reset(Vdbe *p);
		__device__ RC ClearBindings(Vdbe *p);
		// value
		__device__ static const void *Value_Blob(Mem *p);
		__device__ static int Value_Bytes(Mem *p);
		__device__ static int Value_Bytes16(Mem *p);
		__device__ static double Value_Double(Mem *p);
		__device__ static int Value_Int(Mem *p);
		__device__ static int64 Value_Int64(Mem *p);
		__device__ static const unsigned char *Value_Text(Mem *p);
#ifndef OMIT_UTF16
		__device__ static const void *Value_Text16(Mem *p);
		__device__ static const void *Value_Text16be(Mem *P);
		__device__ static const void *Value_Text16le(Mem *p);
#endif
		__device__ static TYPE Value_Type(Mem *p);

		// results
		__device__ static void Result_Blob(FuncContext *fctx, const void *z, int n, void (*del)(void *));
		__device__ static void Result_Double(FuncContext *fctx, double value);
		__device__ static void Result_Error(FuncContext *fctx, const char *z, int n);
#ifndef OMIT_UTF16
		__device__ static void Result_Error16(FuncContext *fctx, const void *z, int n);
#endif
		__device__ static void Result_Int(FuncContext *fctx, int value);
		__device__ static void Result_Int64(FuncContext *fctx, int64 value);
		__device__ static void Result_Null(FuncContext *fctx);
		__device__ static void Result_Text(FuncContext *fctx, const char *z, int n, void (*del)(void *));
#ifndef OMIT_UTF16
		__device__ static void Result_Text16(FuncContext *fctx, const void *z, int n, void (*del)(void *));
		__device__ static void Result_Text16be(FuncContext *fctx, const void *z, int n, void (*del)(void *));
		__device__ static void Result_Text16le(FuncContext *fctx, const void *z, int n, void (*del)(void *));
#endif
		__device__ static void Result_Value(FuncContext *fctx, Mem *value);
		__device__ static void Result_ZeroBlob(FuncContext *fctx, int n);
		__device__ static void Result_ErrorCode(FuncContext *fctx, RC errCode);
		__device__ static void Result_ErrorOverflow(FuncContext *fctx);
		__device__ static void Result_ErrorNoMem(FuncContext *fctx);

		// step
		__device__ RC Step2();
		__device__ RC Step();

		// name3
		__device__ static void *User_Data(FuncContext *fctx);
		__device__ static Context *Context_Ctx(FuncContext *fctx);
		__device__ static void InvalidFunction(FuncContext *fctx, int notUsed1, Mem **notUsed2);
		__device__ static void *Agregate_Context(FuncContext *fctx, int bytes);
		__device__ static void *get_Auxdata(FuncContext *fctx, int arg);
		__device__ static void set_Auxdata(FuncContext *fctx, int args, void *aux, void (*delete_)(void*));
		__device__ int Column_Count(Vdbe *p);
		__device__ int Data_Count(Vdbe *p);

		// column
		__device__ static const void *Column_Blob(Vdbe *p, int i);
		__device__ static int Column_Bytes(Vdbe *p, int i);
		__device__ static int Column_Bytes16(Vdbe *p, int i);
		__device__ static double Column_Double(Vdbe *p, int i);
		__device__ static int Column_Int(Vdbe *p, int i);
		__device__ static int64 Column_Int64(Vdbe *p, int i);
		__device__ static const unsigned char *Column_Text(Vdbe *p, int i);
		__device__ static Mem *Column_Value(Vdbe *p, int i);
#ifndef OMIT_UTF16
		__device__ static const void *Column_Text16(Vdbe *p, int i);
#endif
		__device__ static TYPE Column_Type(Vdbe *p, int i);
		__device__ static const char *Column_Name(Vdbe *p, int n);
#ifndef OMIT_UTF16
		__device__ static const void *Column_Name16(Vdbe *p, int n);
#endif

#if defined(OMIT_DECLTYPE) && defined(ENABLE_COLUMN_METADATA)
#error "Must not define both OMIT_DECLTYPE and ENABLE_COLUMN_METADATA"
#endif
#ifndef OMIT_DECLTYPE
		__device__ static const char *Column_Decltype(Vdbe *p, int n);
#ifndef OMIT_UTF16
		__device__ static const void *Column_Decltype16(Vdbe *p, int n);
#endif
#endif
#ifdef ENABLE_COLUMN_METADATA
		__device__ static const char *Column_DatabaseName(Vdbe *p, int n);
#ifndef OMIT_UTF16
		__device__ static const void *Column_DatabaseName16(Vdbe *p, int n);
#endif
		__device__ static const char *Column_TableName(Vdbe *p, int n);
#ifndef OMIT_UTF16
		__device__ static const void *Column_TableName16(Vdbe *p, int n);
#endif
		__device__ static const char *Column_OriginName(Vdbe *p, int n);
#ifndef OMIT_UTF16
		__device__ static const void *Column_OriginName16(Vdbe *p, int n);
#endif
#endif
		// bind
		__device__ static RC Bind_Blob(Vdbe *p, int i, const void *z, int n, void (*del)(void *));
		__device__ static RC Bind_Double(Vdbe *p, int i, double value);
		__device__ static RC Bind_Int(Vdbe *p, int i, int value);
		__device__ static RC Bind_Int64(Vdbe *p, int i, int64 value);
		__device__ static RC Bind_Null(Vdbe *p, int i);
		__device__ static RC Bind_Text(Vdbe *p, int i, const char *z, int n, void (*del)(void *));
#ifndef OMIT_UTF16
		__device__ static RC Bind_Text16(Vdbe *p, int i, const void *z, int n, void (*del)(void *));
#endif
		__device__ static RC Bind_Value(Vdbe *p, int i, const Mem *value);
		__device__ static RC Bind_Zeroblob(Vdbe *p, int i, int n);
		__device__ static int Bind_ParameterCount(Vdbe *p);
		__device__ static const char *Bind_ParameterName(Vdbe *p, int i);
		__device__ static int ParameterIndex(Vdbe *p, const char *name, int nameLength);
		__device__ static int Bind_ParameterIndex(Vdbe *p, const char *name);
		__device__ static RC TransferBindings(Vdbe *from, Vdbe *to);

		// stmt		
		__device__ static Context *Stmt_Ctx(Vdbe *p);
		__device__ static bool Stmt_Readonly(Vdbe *p);
		__device__ static bool Stmt_Busy(Vdbe *p);
		__device__ static Vdbe *Stmt_Next(Context *ctx, Vdbe *p);
		__device__ static int Stmt_Status(Vdbe *p, OP op, bool resetFlag);

#pragma endregion

#pragma region Vdbe+Mem

		__device__ static RC ChangeEncoding(Mem *mem, TEXTENCODE newEncode);
		__device__ static RC MemGrow(Mem *mem, size_t newSize, bool preserve);
		__device__ static RC MemMakeWriteable(Mem *mem);
#ifndef OMIT_INCRBLOB
		__device__ static RC MemExpandBlob(Mem *mem);
#define ExpandBlob(P) (((P)->Flags & MEM_Zero)?MemExpandBlob(P):0)
#else
#define MemExpandBlob(x) RC_OK
#define ExpandBlob(P) RC_OK
#endif
		__device__ static RC MemNulTerminate(Mem *mem);
		__device__ static RC MemStringify(Mem *mem, TEXTENCODE encode);
		__device__ static RC MemFinalize(Mem *mem, FuncDef *func);
		__device__ static void MemReleaseExternal(Mem *mem);
		__device__ static void MemRelease(Mem *mem);
		__device__ static int64 IntValue(Mem *mem);
		__device__ static double RealValue(Mem *mem);
		__device__ static void IntegerAffinity(Mem *mem);
		__device__ static RC MemIntegerify(Mem *mem);
		__device__ static RC MemRealify(Mem *mem);
		__device__ static RC MemNumerify(Mem *mem);
		__device__ static void MemSetNull(Mem *mem);
		__device__ static void MemSetZeroBlob(Mem *mem, int n);
		__device__ static void MemSetInt64(Mem *mem, int64 value);
#ifdef OMIT_FLOATING_POINT
#define MemSetDouble MemSetInt64
#else
		__device__ static void MemSetDouble(Mem *mem, double value); //@
#endif
		__device__ static void MemSetRowSet(Mem *mem); //@
		__device__ static bool MemTooBig(Mem *mem); //@
#ifdef _DEBUG
		__device__ void MemAboutToChange(Vdbe *vdbe, Mem *mem); //@
#endif
		__device__ static void MemShallowCopy(Mem *to, const Mem *from, uint16 srcType); //@
		__device__ static RC MemCopy(Mem *to, const Mem *from); //@
		__device__ static void MemMove(Mem *to, Mem *from); //@
		__device__ static RC MemSetStr(Mem *mem, const char *z, int n, TEXTENCODE encode, void (*del)(void *)); //@
		__device__ static RC MemFromBtree(BtCursor *cursor, int offset, int amount, bool key, Mem *mem); //@
#define VdbeMemRelease(X) if((X)->Flags&(MEM_Agg|MEM_Dyn|MEM_RowSet|MEM_Frame)) Vdbe::MemReleaseExternal(X);

		// value
		__device__ static const void *ValueText(Mem *mem, TEXTENCODE encode);
		__device__ static Mem *ValueNew(Context *ctx);
		__device__ static RC ValueFromExpr(Context *ctx, Expr *expr, TEXTENCODE encode, AFF affinity, Mem **value);
		__device__ static void ValueSetStr(Mem *mem, int n, const void *z, TEXTENCODE encode, void (*del)(void *));
		__device__ static void ValueFree(Mem *mem);
		__device__ static int ValueBytes(Mem *mem, TEXTENCODE encode);

#pragma endregion

#pragma region Vdbe+Trace
#ifndef OMIT_TRACE
		__device__ char *ExpandSql(const char *rawSql);
#endif
#if defined(ENABLE_TREE_EXPLAIN)
		__device__ static void ExplainBegin(Vdbe *p);
		__device__ static void ExplainPrintf(Vdbe *p, const char *format, va_list args);
		__device__ static void ExplainNL(Vdbe *p);
		__device__ static void ExplainPush(Vdbe *p);
		__device__ static void ExplainPop(Vdbe *p);
		__device__ static void ExplainFinish(Vdbe *p);
		__device__ static const char *Explanation(Vdbe *p);
#endif
#pragma endregion

#pragma region Vdbe+Sort
		__device__ static RC SorterInit(Context *db, VdbeCursor *cursor); //@
		__device__ static void SorterClose(Context *db, VdbeCursor *cursor); //@
		__device__ static RC SorterRowkey(const VdbeCursor *cursor, Mem *mem); //@
		__device__ static RC SorterNext(Context *db, const VdbeCursor *cursor, bool *eof); //@
		__device__ static RC SorterRewind(Context *db, const VdbeCursor *cursor, bool *eof); //@
		__device__ static RC SorterWrite(Context *db, const VdbeCursor *cursor, Mem *mem); //@
		__device__ static RC SorterCompare(const VdbeCursor *cursor, Mem *mem, int *r); //@
#pragma endregion

#pragma region Vdbe+Utf
#ifndef OMIT_UTF16
		__device__ static RC MemTranslate(Mem *mem, TEXTENCODE desiredEncode);
		__device__ static int MemHandleBom(Mem *mem);
		__device__ static char *Utf16to8(Context *ctx, const void *z, int bytes, TEXTENCODE encode);
#ifdef ENABLE_STAT3
		__device__ static char *Utf8to16(Context *ctx, TEXTENCODE encode, char *z, int n, int *out_);
#endif
#endif
#pragma endregion

#if !defined(OMIT_SHARED_CACHE) && THREADSAFE>0
		__device__ void Enter();
		__device__ void Leave();
#else
#define sqlite3VdbeEnter(X)
#define sqlite3VdbeLeave(X)
#endif

#ifndef OMIT_FOREIGN_KEY
		__device__ int CheckFk(int);
#else
#define CheckFk(i) 0
#endif
#ifdef _DEBUG
		__device__ void PrintSql();
		__device__ static void MemPrettyPrint(Mem *mem, char *buf);
#endif
	};
}