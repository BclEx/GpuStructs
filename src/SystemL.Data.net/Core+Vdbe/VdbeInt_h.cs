using Pid = System.UInt32;
using System;
using System.Diagnostics;
using System.Runtime.InteropServices;

using VdbeSorter = System.Object;

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
    public class VdbeCursor
    {
        public Btree.BtCursor Cursor;   // The cursor structure of the backend
        public Btree Bt;                // Separate file holding temporary table
        public KeyInfo KeyInfo;         // Info about index keys needed by index cursors
        public int Db;                  // Index of cursor database in db->aDb[] (or -1)
        public int PseudoTableReg;      // Register holding pseudotable content.
        public int Fields;              // Number of fields in the header
        public bool Zeroed;             // True if zeroed out and ready for reuse
        public bool RowidIsValid;       // True if lastRowid is valid
        public bool AtFirst;            // True if pointing to first entry
        public bool UseRandomRowid;     // Generate new record numbers semi-randomly
        public bool NullRow;            // True if pointing to a row with no data
        public bool deferredMoveto;     // A call to sqlite3BtreeMoveto() is needed
        public bool IsTable;            // True if a table requiring integer keys
        public bool IsIndex;            // True if an index containing keys only - no data
        public bool IsOrdered;          // True if the underlying table is BTREE_UNORDERED
        public bool IsSorter;		    // True if a new-style sorter
        public bool MultiPseudo;	    // Multi-register pseudo-cursor
#if !OMIT_VIRTUALTABLE
        public VTableCursor VtabCursor;   // The cursor for a virtual table
        public ITableModule IModule;      // Module for cursor pVtabCursor
#endif
        public long SeqCount;           // Sequence counter
        public long MovetoTarget;       // Argument to the deferred sqlite3BtreeMoveto()
        public long LastRowid;          // Last rowid from a Next or NextIdx operation
        public VdbeSorter Sorter;		// Sorter object for OP_SorterOpen cursors

        // Result of last sqlite3BtreeMoveto() done by an OP_NotExists or OP_IsUnique opcode on this cursor.
        public int SeekResult;

        // Cached information about the header for the data record that the cursor is currently pointing to.  Only valid if cacheStatus matches
        // Vdbe.cacheCtr.  Vdbe.cacheCtr will never take on the value of CACHE_STALE and so setting cacheStatus=CACHE_STALE guarantees that
        // the cache is out of date.
        //
        // aRow might point to (ephemeral) data for the current row, or it might be NULL.
        public uint CacheStatus;        // Cache is valid if this matches Vdbe.cacheCtr
        public int PayloadSize;         // Total number of bytes in the record
        public uint[] Types;            // Type values for all entries in the record
        public uint[] Offsets;          // Cached offsets to the start of each columns data
        public int Rows;                // Pointer to Data for the current row, if all on one page

        public VdbeCursor Copy() { return (VdbeCursor)MemberwiseClone(); }

        const int CACHE_STALE = 0; // A value for VdbeCursor.cacheValid that means the cache is always invalid.
    }

    public class VdbeFrame
    {
        public Vdbe V;                  // VM this frame belongs to
        public VdbeFrame Parent;        // Parent of this frame, or NULL if parent is main
        public Vdbe.VdbeOp[] Ops;       // Program instructions for parent frame
        public int OpsLength;
        public Mem[] Mems;              // Array of memory cells for parent frame
        public int MemsLength;
        public byte[] OnceFlags;        // Array of OP_Once flags for parent frame
        public int OnceFlagsLength;
        public VdbeCursor[] Cursors;    // Array of Vdbe cursors for parent frame
        public int CursorsLength;
        public int Token;               // Copy of SubProgram.token
        public long LastRowid;          // Last insert rowid (sqlite3.lastRowid)
        public int PC;                  // Program Counter in parent (calling) frame
        public int ChildMems;           // Number of memory cells for child frame
        public int ChildCursors;        // Number of cursors for child frame
        public int Changes;             // Statement changes (Vdbe.nChanges)
        // Needed for C# Implementation
        public Mem[] _ChildMems;        // Array of memory cells for child frame
        public VdbeCursor[] _ChildCursors; // Array of cursors for child frame
    }
    //#define VdbeFrameMem(p) ((Mem )&((u8 )p)[ROUND8(sizeof(VdbeFrame))])

    [Flags]
    public enum MEM : ushort
    {
        // If the MEM_Null flag is set, then the value is an SQL NULL value. No other flags may be set in this case.
        //
        // If the MEM_Str flag is set then Mem.z points at a string representation. Usually this is encoded in the same unicode encoding as the main
        // database (see below for exceptions). If the MEM_Term flag is also set, then the string is nul terminated. The MEM_Int and MEM_Real 
        // flags may coexist with the MEM_Str flag.
        Null = 0x0001,		// Value is NULL
        Str = 0x0002,		// Value is a string
        Int = 0x0004,		// Value is an integer
        Real = 0x0008,		// Value is a real number
        Blob = 0x0010,		// Value is a BLOB
        RowSet = 0x0020,	// Value is a RowSet object
        Frame = 0x0040,		// Value is a VdbeFrame object
        Invalid = 0x0080,	// Value is undefined
        Cleared = 0x0100,	// NULL set by OP_Null, not from data
        TypeMask = 0x01ff,  // Mask of type bits
        // Whenever Mem contains a valid string or blob representation, one of the following flags must be set to determine the memory management
        // policy for Mem.z.  The MEM_Term flag tells us whether or not the string is \000 or \u0000 terminated
        Term = 0x0200,		// String rep is nul terminated
        Dyn = 0x0400,		// Need to call sqliteFree() on Mem.z
        Static = 0x0800,	// Mem.z points to a static string
        Ephem = 0x1000,		// Mem.z points to an ephemeral string
        Agg = 0x2000,		// Mem.z points to an agg function context
#if !OMIT_INCRBLOB
        Zero = 0x4000,		// Mem.i contains count of 0s appended to blob
#else
        Zero = 0x0000,
#endif
    }

    public class Mem
    {
        public Context Ctx;              // The associated database connection
        public string Z;                // String value
        public double R;                // Real value
        public struct u_t
        {
            public long I;              // Integer value used when MEM_Int is set in flags
            public int Zero;            // Used when bit MEM_Zero is set in flags
            public FuncDef Def;         // Used only when flags==MEM_Agg
            public RowSet RowSet;       // Used only when flags==MEM_RowSet
            public VdbeFrame Frame;     // Used when flags==MEM_Frame
        };
        public u_t u;
        public int N;                   // Number of characters in string value, excluding '\0'
        public MEM Flags;               // Some combination of MEM_Null, MEM_Str, MEM_Dyn, etc.
        public TYPE Type;               // One of SQLITE_NULL, SQLITE_TEXT, SQLITE_INTEGER, etc
        public TEXTENCODE Encode;       // SQLITE_UTF8, SQLITE_UTF16BE, SQLITE_UTF16LE
#if DEBUG
        public Mem ScopyFrom;           // This Mem is a shallow copy of pScopyFrom
        public object Filler;           // So that sizeof(Mem) is a multiple of 8
#endif
        public Action<object> Del;      // If not null, call this function to delete Mem.z
        // Needed for C# Implementation
        public Mem _Mem;                // Used when C# overload Z as MEM space
        //public SumCtx _SumCtx;        // Used when C# overload Z as Sum context
        public Vdbe.SubProgram[] _SubProgram;   // Used when C# overload Z as SubProgram
        public Text.StringBuilder _StrAccum;    // Used when C# overload Z as STR context
        public object _MD5Context;    // Used when C# overload Z as MD5 context

        static void MemSetTypeFlag(Mem p, MEM f) { p.Flags = (p.Flags & ~(MEM.TypeMask | MEM.Zero) | f); }

#if DEBUG
        static bool memIsValid(Mem M) { return ((M).Flags & MEM.Invalid) == 0; }
#else
        static bool memIsValid( Mem M ) { return true; }
#endif

        public string GetBlob()
        {
            if (Flags & (MEM.Blob | MEM.Str))
            {
                sqlite3VdbeMemExpandBlob(p);
                Flags &= ~MEM.Str;
                Flags |= MEM.Blob;
                return (N != 0 ? Z : null);
            }
            return GetText();
        }
        public int GetBytes() { return Mem_Bytes(this, TEXTENCODE.UTF8); }
        public int GetBytes16() { return Mem_Bytes(this, TEXTENCODE.UTF16NATIVE); }
        public double GetDouble() { return sqlite3VdbeRealValue(this); }
        public int GetInt() { return (int)sqlite3VdbeIntValue(this); }
        public long GetInt64() { return sqlite3VdbeIntValue(this); }
        public string GetText() { return (const unsigned char *)Mem_Text(this, TEXTENCODE.UTF8); }
#if !OMIT_UTF16
        public string GetText16() { return Mem_Text(this, TEXTENCODE.UTF16NATIVE); }
        public string GetText16be() { return Mem_Text(this, TEXTENCODE.UTF16BE); }
        public string GetText16le() { return Mem_Text(this, TEXTENCODE.UTF16LE); }
#endif
        public TYPE GetType() { return Type; }

        public Mem() { }
        public Mem(Context db, string z, double r, int i, int n, MEM flags, TYPE type, TEXTENCODE encode
#if DEBUG
, Mem scopyFrom, object filler
#endif
)
        {
            Ctx = db;
            Z = z;
            R = r;
            u.I = i;
            N = n;
            Flags = flags;
#if DEBUG
            ScopyFrom = scopyFrom;
            Filler = filler;
#endif
            Type = type;
            Encode = encode;
        }

        public void memcopy(ref Mem ct)
        {
            if (ct == null) ct = new Mem();
            ct.u = u;
            ct.R = R;
            ct.Ctx = Ctx;
            ct.Z = Z;
            ct.N = N;
            ct.Flags = Flags;
            ct.Type = Type;
            ct.Encode = Encode;
            ct.Del = Del;
        }
    }

    public class VdbeFunc : FuncDef
    {
        public struct AuxData
        {
            public object Aux;                     // Aux data for the i-th argument
        }
        public int AuxsLength;                         // Number of entries allocated for apAux[]
        public AuxData[] Auxs = new AuxData[2]; // One slot for each function argument
    }

    public class FuncContext
    {
        public FuncDef Func;        // Pointer to function information.  MUST BE FIRST
        public VdbeFunc VdbeFunc;   // Auxilary data, if created.
        public Mem s = new Mem();         // The return value is stored here
        public Mem Mem;             // Memory cell used to store aggregate context
        public CollSeq Coll;        // Collating sequence
        public RC IsError;          // Error code returned by the function.
        public bool SkipFlag;			// Skip skip accumulator loading if true
    }
}
