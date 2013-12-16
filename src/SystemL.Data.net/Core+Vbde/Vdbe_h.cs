using FILE = System.IO.TextWriter;
using System;

#region Limits
#if !MAX_VARIABLE_NUMBER
using ynVar = System.Int16;
#else
using ynVar = System.Int32; 
#endif
#if MAX_ATTACHED
using yDbMask = System.Int64; 
#else
using yDbMask = System.Int32;
#endif
#endregion

namespace Core
{
    public partial class Vdbe
    {
        const int COLNAME_NAME = 0;
        const int COLNAME_DECLTYPE = 1;
        const int COLNAME_DATABASE = 2;
        const int COLNAME_TABLE = 3;
        const int COLNAME_COLUMN = 4;
#if ENABLE_COLUMN_METADATA
const int COLNAME_N = 5;     // Number of COLNAME_xxx symbols
#else
#if OMIT_DECLTYPE
const int COLNAME_N = 1;     // Number of COLNAME_xxx symbols
#else
        const int COLNAME_N = 2;
#endif
#endif
        static int ADDR(int x) { return -1 - x; }

        public enum P4T : sbyte
        {
            NOTUSED = 0,        // The P4 parameter is not used
            DYNAMIC = -1,       // Pointer to a string obtained from sqliteMalloc=();
            STATIC = -2,        // Pointer to a static string
            COLLSEQ = -4,       // P4 is a pointer to a CollSeq structure
            FUNCDEF = -5,       // P4 is a pointer to a FuncDef structure
            KEYINFO = -6,       // P4 is a pointer to a KeyInfo structure
            VDBEFUNC = -7,      // P4 is a pointer to a VdbeFunc structure
            MEM = -8,           // P4 is a pointer to a Mem*    structure
            TRANSIENT = 0,      // P4 is a pointer to a transient string
            VTAB = 10,          // P4 is a pointer to an sqlite3_vtab structure
            MPRINTF = -11,      // P4 is a string obtained from sqlite3_mprintf=();
            REAL = -12,         // P4 is a 64-bit floating point value
            INT64 = -13,        // P4 is a 64-bit signed integer
            INT32 = -14,        // P4 is a 32-bit signed integer
            INTARRAY = -15,     // P4 is a vector of 32-bit integers
            SUBPROGRAM = -18,   // P4 is a pointer to a SubProgram structure
            ADVANCE = -19,	    // P4 is a pointer to BtreeNext() or BtreePrev()
            // When adding a P4 argument using P4_KEYINFO, a copy of the KeyInfo structure is made.  That copy is freed when the Vdbe is finalized.  But if the
            // argument is P4_KEYINFO_HANDOFF, the passed in pointer is used.  It still gets freed when the Vdbe is finalized so it still should be obtained
            // from a single sqliteMalloc().  But no copy is made and the calling function should *not* try to free the KeyInfo.
            KEYINFO_HANDOFF = -16,
            KEYINFO_STATIC = -17,
        }

        public class VdbeOp
        {
            public byte Opcode;           // What operation to perform
            public P4T P4Type;            // One of the P4_xxx constants for p4
            public byte Opflags;            // Mask of the OPFLG_* flags in opcodes.h
            public byte P5;                 // Fifth parameter is an unsigned character
            public int P1;                // First operand
            public int P2;                // Second parameter (often the jump destination)
            public int P3;                // The third parameter
            public class P4_t
            {
                public int I;                // Integer value if p4type==P4_INT32
                public object P;             // Generic pointer
                public string Z;           // Pointer to data for string (char array) types
                public long I64;             // Used when p4type is P4_INT64
                public double Real;         // Used when p4type is P4_REAL
                public FuncDef Func;        // Used when p4type is P4_FUNCDEF
                public VdbeFunc VdbeFunc;   // Used when p4type is P4_VDBEFUNC
                public CollSeq Coll;        // Used when p4type is P4_COLLSEQ
                public Mem Mem;             // Used when p4type is P4_MEM
                public VTable Vtab;         // Used when p4type is P4_VTAB
                public KeyInfo KeyInfo;     // Used when p4type is P4_KEYINFO
                public int[] Is;             // Used when p4type is P4_INTARRAY
                public SubProgram Program;  // Used when p4type is P4_SUBPROGRAM
                public Func<Btree.BtCursor, int[], int> FuncDel;       // Used when p4type is P4_FUNCDEL 
            }
            public P4_t P4 = new P4_t(); // fourth parameter
#if DEBUG
            public string Comment;     // Comment to improve readability
#endif
#if VDBE_PROFILE
        public int cnt;             // Number of times this instruction was executed
        public ulong cycles;         // Total time spend executing this instruction
#endif
        }

        public class SubProgram
        {
            public array_t<VdbeOp> Ops;          // Array of opcodes for sub-program
            public int Mems;              // Number of memory cells required
            public int Csrs;              // Number of cursors required
            public int Token;             // id that may be used to recursive triggers
            public SubProgram Next;      // Next sub-program already visited
        }

        public struct VdbeOpList
        {
            public byte Opcode;  // What operation to perform
            public int P1;     // First operand
            public int P2;     // Second parameter (often the jump destination)
            public int P3;     // Third parameter
            public VdbeOpList(byte opcode, int p1, int p2, int p3)
            {
                Opcode = opcode;
                P1 = p1;
                P2 = p2;
                P3 = p3;
            }
        }

        //Vdbe *sqlite3VdbeCreate(sqlite3);
        //int sqlite3VdbeAddOp0(Vdbe*,int);
        //int sqlite3VdbeAddOp1(Vdbe*,int,int);
        //int sqlite3VdbeAddOp2(Vdbe*,int,int,int);
        //int sqlite3VdbeAddOp3(Vdbe*,int,int,int,int);
        //int sqlite3VdbeAddOp4(Vdbe*,int,int,int,int,string zP4,int);
        //int sqlite3VdbeAddOp4Int(Vdbe*,int,int,int,int,int);
        //int sqlite3VdbeAddOpList(Vdbe*, int nOp, VdbeOpList const *aOp);
        //void sqlite3VdbeAddParseSchemaOp(Vdbe*,int,char);
        //void sqlite3VdbeChangeP1(Vdbe*, int addr, int P1);
        //void sqlite3VdbeChangeP2(Vdbe*, int addr, int P2);
        //void sqlite3VdbeChangeP3(Vdbe*, int addr, int P3);
        //void sqlite3VdbeChangeP5(Vdbe*, u8 P5);
        //void sqlite3VdbeJumpHere(Vdbe*, int addr);
        //void sqlite3VdbeChangeToNoop(Vdbe*, int addr, int N);
        //void sqlite3VdbeChangeP4(Vdbe*, int addr, string zP4, int N);
        //void sqlite3VdbeUsesBtree(Vdbe*, int);
        //VdbeOp *sqlite3VdbeGetOp(Vdbe*, int);
        //int sqlite3VdbeMakeLabel(Vdbe);
        //void sqlite3VdbeRunOnlyOnce(Vdbe);
        //void sqlite3VdbeDelete(Vdbe);
        //void sqlite3VdbeDeleteObject(sqlite3*,Vdbe);
        //void sqlite3VdbeMakeReady(Vdbe*,Parse);
        //int sqlite3VdbeFinalize(Vdbe);
        //void sqlite3VdbeResolveLabel(Vdbe*, int);
        //int sqlite3VdbeCurrentAddr(Vdbe);
        //#if SQLITE_DEBUG
        //  int sqlite3VdbeAssertMayAbort(Vdbe *, int);
        //  void sqlite3VdbeTrace(Vdbe*,FILE);
        //#endif
        //void sqlite3VdbeResetStepResult(Vdbe);
        //void sqlite3VdbeRewind(Vdbe);
        //int sqlite3VdbeReset(Vdbe);
        //void sqlite3VdbeSetNumCols(Vdbe*,int);
        //int sqlite3VdbeSetColName(Vdbe*, int, int, string , void()(void));
        //void sqlite3VdbeCountChanges(Vdbe);
        //sqlite3 *sqlite3VdbeDb(Vdbe);
        //void sqlite3VdbeSetSql(Vdbe*, string z, int n, int);
        //void sqlite3VdbeSwap(Vdbe*,Vdbe);
        //VdbeOp *sqlite3VdbeTakeOpArray(Vdbe*, int*, int);
        //sqlite3_value *sqlite3VdbeGetValue(Vdbe*, int, u8);
        //void sqlite3VdbeSetVarmask(Vdbe*, int);
        //#if !SQLITE_OMIT_TRACE
        //  char *sqlite3VdbeExpandSql(Vdbe*, const char);
        //#endif

        //UnpackedRecord *sqlite3VdbeRecordUnpack(KeyInfo*,int,const void*,char*,int);
        //void sqlite3VdbeDeleteUnpackedRecord(UnpackedRecord);
        //int sqlite3VdbeRecordCompare(int,const void*,UnpackedRecord);

        //#if !OMIT_TRIGGER
        //void sqlite3VdbeLinkSubProgram(Vdbe *, SubProgram );
        //#endif
#if !NDEBUG
        static void VdbeComment(Vdbe v, string zFormat, params object[] ap) { sqlite3VdbeComment(v, zFormat, ap); }
        static void VdbeNoopComment(Vdbe v, string zFormat, params object[] ap) { sqlite3VdbeNoopComment(v, zFormat, ap); }
#else
    static void VdbeComment( Vdbe v, string zFormat, params object[] ap ) { }
    static void VdbeNoopComment( Vdbe v, string zFormat, params object[] ap ) { }
#endif

        //void sqlite3VdbeFreeCursor(Vdbe *, VdbeCursor);
        //void sqliteVdbePopStack(Vdbe*,int);
        //int sqlite3VdbeCursorMoveto(VdbeCursor);
        //#if (SQLITE_DEBUG) || defined(VDBE_PROFILE)
        //void sqlite3VdbePrintOp(FILE*, int, Op);
        //#endif
        //u32 sqlite3VdbeSerialTypeLen(u32);
        //u32 sqlite3VdbeSerialType(Mem*, int);
        //u32sqlite3VdbeSerialPut(unsigned char*, int, Mem*, int);
        //u32 sqlite3VdbeSerialGet(const unsigned char*, u32, Mem);
        //void sqlite3VdbeDeleteAuxData(VdbeFunc*, int);

        //int sqlite2BtreeKeyCompare(BtCursor *, const void *, int, int, int );
        //int sqlite3VdbeIdxKeyCompare(VdbeCursor*,UnpackedRecord*,int);
        //int sqlite3VdbeIdxRowid(sqlite3 *, i64 );
        //int sqlite3MemCompare(const Mem*, const Mem*, const CollSeq);
        //int sqlite3VdbeExec(Vdbe);
        //int sqlite3VdbeList(Vdbe);
        //int sqlite3VdbeHalt(Vdbe);
        //int sqlite3VdbeChangeEncoding(Mem *, int);
        //int sqlite3VdbeMemTooBig(Mem);
        //int sqlite3VdbeMemCopy(Mem*, const Mem);
        //void sqlite3VdbeMemShallowCopy(Mem*, const Mem*, int);
        //void sqlite3VdbeMemMove(Mem*, Mem);
        //int sqlite3VdbeMemNulTerminate(Mem);
        //int sqlite3VdbeMemSetStr(Mem*, const char*, int, u8, void()(void));
        //void sqlite3VdbeMemSetInt64(Mem*, i64);
#if OMIT_FLOATING_POINT
//# define sqlite3VdbeMemSetDouble sqlite3VdbeMemSetInt64
#else
        //void sqlite3VdbeMemSetDouble(Mem*, double);
#endif
        //void sqlite3VdbeMemSetNull(Mem);
        //void sqlite3VdbeMemSetZeroBlob(Mem*,int);
        //void sqlite3VdbeMemSetRowSet(Mem);
        //int sqlite3VdbeMemMakeWriteable(Mem);
        //int sqlite3VdbeMemStringify(Mem*, int);
        //i64 sqlite3VdbeIntValue(Mem);
        //int sqlite3VdbeMemIntegerify(Mem);
        //double sqlite3VdbeRealValue(Mem);
        //void sqlite3VdbeIntegerAffinity(Mem);
        //int sqlite3VdbeMemRealify(Mem);
        //int sqlite3VdbeMemNumerify(Mem);
        //int sqlite3VdbeMemFromBtree(BtCursor*,int,int,int,Mem);
        //void sqlite3VdbeMemRelease(Mem p);
        //void sqlite3VdbeMemReleaseExternal(Mem p);
        //int sqlite3VdbeMemFinalize(Mem*, FuncDef);
        //string sqlite3OpcodeName(int);
        //int sqlite3VdbeMemGrow(Mem pMem, int n, int preserve);
        //int sqlite3VdbeCloseStatement(Vdbe *, int);
        //void sqlite3VdbeFrameDelete(VdbeFrame);
        //int sqlite3VdbeFrameRestore(VdbeFrame );
        //void sqlite3VdbeMemStoreType(Mem *pMem);  

#if !OMIT_SHARED_CACHE && THREADSAFE
        //void Enter(Vdbe);
        //void Leave(Vdbe);
#else
    static void Enter( Vdbe p ) { }
    static void Leave( Vdbe p ) { }
#endif

#if DEBUG
        //void sqlite3VdbeMemPrepareToChange(Vdbe*,Mem);
#endif

#if !OMIT_FOREIGN_KEY
        //int sqlite3VdbeCheckFk(Vdbe *, int);
#else
        static int sqlite3VdbeCheckFk(Vdbe p, int i) { return 0; }
#endif

        //int sqlite3VdbeMemTranslate(Mem*, u8);
        //#if SQLITE_DEBUG
        //  void sqlite3VdbePrintSql(Vdbe);
        //  void sqlite3VdbeMemPrettyPrint(Mem pMem, string zBuf);
        //#endif
        //int sqlite3VdbeMemHandleBom(Mem pMem);

#if !OMIT_INCRBLOB
//      int sqlite3VdbeMemExpandBlob(Mem);
#else
        static RC sqlite3VdbeMemExpandBlob(Mem x) { return RC.OK; }
#endif

        public Context Db;              // The database connection that owns this statement
        public array_t<VdbeOp> Ops;     // Space to hold the virtual machine's program
        public array_t<Mem> Mems;       // The memory locations
        public Mem[] Args;              // Arguments to currently executing user function
        public Mem[] ColNames;          // Column names to return
        public Mem[] ResultSet;         // Pointer to an array of results
        public int OpsAlloc;            // Number of slots allocated for aOp[]
        public int LabelsAlloc;         // Number of slots allocated in aLabel[]
        public array_t<int> Labels;     // Space to hold the labels
        public ushort ResColumns;       // Number of columns in one row of the result set
        public uint Magic;              // Magic number for sanity checking
        public string ErrMsg;           // Error message written here
        public Vdbe Prev, Next;         // Linked list of VDBEs with the same Vdbe.db
        public array_t<VdbeCursor> Cursors; // One element of this array for each open cursor
        public array_t2<ynVar, Mem> Vars;       // Values for the OP_Variable opcode.
        public array_t2<ynVar, string> VarNames; // Name of variables
        public uint CacheCtr;           // VdbeCursor row cache generation counter
        public int PC;                  // The program counter
        public RC RC;                   // Value to return
        public byte ErrorAction;        // Recovery action to do in case of an error
        public int MinWriteFileFormat;  // Minimum file format for writable database files
        public byte HasExplain;         // True if EXPLAIN present on SQL command
        public byte InVtabMethod;       // See comments above
        public bool ChangeCntOn;        // True to update the change-counter
        public bool Expired;            // True if the VM needs to be recompiled
        public bool RunOnlyOnce;        // Automatically expire on reset
        public bool UsesStmtJournal;    // True if uses a statement journal
        public bool ReadOnly;           // True for read-only statements
        public bool IsPrepareV2;        // True if prepared with prepare_v2()
        public bool DoingRerun;         // True if rerunning after an auto-reprepare
        public int Changes;             // Number of db changes made since last reset
        public yDbMask BtreeMask;       // Bitmask of db.aDb[] entries referenced
        public yDbMask LockMask;        // Subset of btreeMask that requires a lock
        public int StatementID;         // Statement number (or 0 if has not opened stmt)
        public int[] Counters = new int[3]; // Counters used by sqlite3_stmt_status()
#if !OMIT_TRACE
      public long StartTime;            // Time when query started - used for profiling
#endif
        public long FkConstraints;      // Number of imm. FK constraints this VM
        public long StmtDefCons;        // Number of def. constraints when stmt started
        public string Sql = string.Empty;// Text of the SQL statement that generated this
        public object FreeThis;         // Free this when deleting the vdbe
#if DEBUG
        public FILE Trace;              // Write an execution trace here, if not NULL
#endif
#if ENABLE_TREE_EXPLAIN
		Explain Explain;		        // The explainer
		string ExplainString;           // Explanation of data structures
#endif
        public array_t<VdbeFrame> Frame; // Parent frame
        public VdbeFrame DelFrame;      // List of frame objects to free on VM reset
        public uint Expmask;            // Binding to these vars invalidates VM
        public SubProgram Program;      // Linked list of all sub-programs used by VM
        array_t<byte> OnceFlags;        // Flags for OP_Once

        public Vdbe Copy() { return (Vdbe)MemberwiseClone(); }
        public void CopyTo(Vdbe ct)
        {
            ct.Db = Db;
            ct.Ops = Ops; //ct.Ops.length = Ops.length;
            ct.Mems = Mems; //ct.Mems.length = Mems.length;
            ct.Args = Args; //ct.Args.length = Args.length;
            ct.ColNames = ColNames;
            ct.ResultSet = ResultSet;
            ct.OpsAlloc = OpsAlloc;
            ct.LabelsAlloc = LabelsAlloc;
            ct.Labels = Labels; //ct.Labels.length = Labels.length;
            ct.ResColumns = ResColumns;
            ct.Magic = Magic;
            ct.ErrMsg = ErrMsg;
            ct.Prev = Prev; ct.Next = Next;
            ct.Cursors = Cursors; //ct.Cursors.length = Cursors.length;
            ct.Vars = Vars; ct.Vars.length = Vars.length;
            ct.VarNames = VarNames; ct.VarNames.length = VarNames.length;
            ct.CacheCtr = CacheCtr;
            ct.PC = PC;
            ct.RC = RC;
            ct.ErrorAction = ErrorAction;
            ct.MinWriteFileFormat = MinWriteFileFormat;
            ct.HasExplain = HasExplain;
            ct.InVtabMethod = InVtabMethod;
            ct.ChangeCntOn = ChangeCntOn;
            ct.Expired = Expired;
            ct.RunOnlyOnce = RunOnlyOnce;
            ct.UsesStmtJournal = UsesStmtJournal;
            ct.ReadOnly = ReadOnly;
            ct.IsPrepareV2 = IsPrepareV2;
            ct.DoingRerun = DoingRerun;
            ct.Changes = Changes;
            ct.BtreeMask = BtreeMask;
            ct.LockMask = LockMask;
            ct.StatementID = StatementID;
            Counters.CopyTo(ct.Counters, 0);
#if !OMIT_TRACE
            ct.StartTime = startTime;
#endif
            ct.FkConstraints = FkConstraints;
            ct.StmtDefCons = StmtDefCons;
            ct.Sql = Sql;
            ct.FreeThis = FreeThis;
#if DEBUG
            ct.Trace = Trace;
#endif
#if ENABLE_TREE_EXPLAIN
		    ct.Explain = Explain;		        
		    ct.ExplainString = ExplainString;
#endif
            ct.Frame = Frame;
            ct.DelFrame = DelFrame;
            ct.Expmask = Expmask;
            ct.Program = Program;
            ct.OnceFlags = OnceFlags;
        }

        const uint VDBE_MAGIC_INIT = 0x26bceaa5;   // Building a VDBE program
        const uint VDBE_MAGIC_RUN = 0xbdf20da3;   // VDBE is ready to execute
        const uint VDBE_MAGIC_HALT = 0x519c2973;   // VDBE has completed execution
        const uint VDBE_MAGIC_DEAD = 0xb606c3c8;   // The VDBE has been deallocated
    }
}
