using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
namespace Core
{
    public partial class SysEx
    {
        #region Log & Trace

#if DEBUG
        internal static bool OSTrace = false;
        internal static bool IOTrace = true;
        internal static void LOG(RC rc, string format, params object[] args) { Console.WriteLine("l:" + string.Format(format, args)); }
        internal static void OSTRACE(string format, params object[] args) { if (OSTrace) Console.WriteLine("a:" + string.Format(format, args)); }
        internal static void IOTRACE(string format, params object[] args) { if (IOTrace) Console.WriteLine("i:" + string.Format(format, args)); }
#else
        internal static void LOG(RC rc, string x, params object[] args) { }
        internal static void OSTRACE(string format, params object[] args) { }
        internal static void IOTRACE(string format, params object[] args) { }
#endif

        #endregion

        internal const int VERSION_NUMBER = 3007016;

        internal static void ASSERTCOVERAGE(bool p)
        {
        }

        #region Memory Allocation

        public static Action<object> DESTRUCTOR_DYNAMIC;

        [Flags]
        public enum MEMTYPE : byte
        {
            HEAP = 0x01,         // General heap allocations
            LOOKASIDE = 0x02,    // Might have been lookaside memory
            SCRATCH = 0x04,      // Scratch allocations
            PCACHE = 0x08,       // Page cache allocations
            DB = 0x10,           // Uses sqlite3DbMalloc, not sqlite_malloc
        }
        public static void BeginBenignAlloc() { }
        public static void EndBenignAlloc() { }
        public static byte[] Alloc(int size) { return new byte[size]; }
        public static byte[] Alloc(int size, bool clear) { return new byte[size]; }
        public static T[] Alloc<T>(byte s, int size) where T : struct { return new T[size / s]; }
        public static T[] Alloc<T>(byte s, int size, bool clear) where T : struct { return new T[size / s]; }
        public static byte[] TagAlloc(object tag, int size) { return new byte[size]; }
        public static byte[] TagAlloc(object tag, int size, bool clear) { return new byte[size]; }
        public static T[] TagAlloc<T>(object tag, int size) where T : struct { return new T[size]; }
        public static T[] TagAlloc<T>(object tag, byte s, int size) where T : struct { return new T[size / s]; }
        public static T[] TagAlloc<T>(object tag, byte s, int size, bool clear) where T : struct { return new T[size / s]; }
        public static int AllocSize(byte[] p)
        {
            Debug.Assert(MemdebugHasType(p, MEMTYPE.HEAP));
            Debug.Assert(MemdebugNoType(p, MEMTYPE.DB));
            return p.Length;
        }
        public static int TagAllocSize(object tag, byte[] p)
        {
            Debug.Assert(MemdebugHasType(p, MEMTYPE.HEAP));
            Debug.Assert(MemdebugNoType(p, MEMTYPE.DB));
            return p.Length;
        }
        public static void Free<T>(ref T p) where T : class { }
        public static void TagFree<T>(object tag, ref T p) where T : class { p = null; }
        public static byte[] ScratchAlloc(int size) { return new byte[size]; }
        public static void ScratchFree(ref byte[] p) { p = null; }
        public static bool HeapNearlyFull() { return false; }
        public static T[] Realloc<T>(int s, T[] p, int bytes)
        {
            var newT = new T[bytes / s];
            Array.Copy(p, newT, Math.Min(p.Length, newT.Length));
            return newT;
        }
        public static T[] TagRealloc<T>(object tag, int s, T[] p, int bytes)
        {
            var newT = new T[bytes / s];
            Array.Copy(p, newT, Math.Min(p.Length, newT.Length));
            return newT;
        }
        //
#if MEMDEBUG
        //public static void MemdebugSetType<T>(T X, MEMTYPE Y);
        //public static bool MemdebugHasType<T>(T X, MEMTYPE Y);
        //public static bool MemdebugNoType<T>(T X, MEMTYPE Y);
#else
        public static void MemdebugSetType<T>(T X, MEMTYPE Y) { }
        public static bool MemdebugHasType<T>(T X, MEMTYPE Y) { return true; }
        public static bool MemdebugNoType<T>(T X, MEMTYPE Y) { return true; }
#endif

        #endregion

        public static RC Initialize()
        {
            // mutex
            var rc = RC.OK; // = 3MutexEx::Initialize();
            if (rc != RC.OK) return rc;
            //rc = Alloc::Initialize();
            //rc = PCache.Initialize();
            //if (rc != RC.OK) return rc;
            rc = VSystem.Initialize();
            if (rc != RC.OK) return rc;
            //PCache.PageBufferSetup(_config.Page, _config.SizePage, _config.Pages);
            return RC.OK;
        }

        public static void Shutdown()
        {
            VSystem.Shutdown();
            //PCache.Shutdown();
            //Alloc.Shutdown();
            //MutexEx.Shutdown();
        }

        //internal static RC OSError(RC rc, string func, string path)
        //{
        //    var sf = new StackTrace(new StackFrame(true)).GetFrame(0);
        //    var errorID = (uint)Marshal.GetLastWin32Error();
        //    var message = Marshal.GetLastWin32Error().ToString();
        //    Debug.Assert(rc != RC.OK);
        //    if (path == null)
        //        path = string.Empty;
        //    int i;
        //    for (i = 0; i < message.Length && message[i] != '\r' && message[i] != '\n'; i++) ;
        //    message = message.Substring(0, i);
        //    //sqlite3_log("os_win.c:%d: (%d) %s(%s) - %s", sf.GetFileLineNumber(), errorID, func, sf.GetFileName(), message);
        //    return rc;
        //}

        public static void SKIP_UTF8(string z, ref int idx)
        {
            idx++;
            if (idx < z.Length && z[idx - 1] >= 0xC0)
                while (idx < z.Length && (z[idx] & 0xC0) == 0x80)
                    idx++;
        }
        public static void SKIP_UTF8(byte[] z, ref int idx)
        {
            idx++;
            if (idx < z.Length && z[idx - 1] >= 0xC0)
                while (idx < z.Length && (z[idx] & 0xC0) == 0x80)
                    idx++;
        }


        public static bool ALWAYS(bool x) { if (x != true) Debug.Assert(false); return x; }
        public static bool NEVER(bool x) { return x; }

        public static int ROUND8(int x) { return (x + 7) & ~7; }
        public static int ROUNDDOWN8(int x) { return x & ~7; }

#if BYTEALIGNED4
        public static bool HASALIGNMENT8(int x) { return true; }
#else
        public static bool HASALIGNMENT8(int x) { return true; }
#endif

#if DEBUG
        internal static RC CORRUPT_BKPT()
        {
            var sf = new StackTrace(new StackFrame(true)).GetFrame(0);
            LOG(RC.CANTOPEN, "database corruption at line {0} of [{1}]", sf.GetFileLineNumber(), sf.GetFileName());
            return RC.CORRUPT;
        }
        internal static RC MISUSE_BKPT()
        {
            var sf = new StackTrace(new StackFrame(true)).GetFrame(0);
            LOG(RC.CANTOPEN, "misuse at line {0} of [{1}]", sf.GetFileLineNumber(), sf.GetFileName());
            return RC.MISUSE;
        }
        internal static RC CANTOPEN_BKPT()
        {
            var sf = new StackTrace(new StackFrame(true)).GetFrame(0);
            LOG(RC.CANTOPEN, "cannot open file at line {0} of [{1}]", sf.GetFileLineNumber(), sf.GetFileName());
            return RC.CANTOPEN;
        }
#else
        internal static RC CORRUPT_BKPT() { return RC.CORRUPT; }
        internal static RC MISUSE_BKPT() { return RC.MISUSE; }
        internal static RC CANTOPEN_BKPT() { return RC.CANTOPEN; }
#endif

        #region For C#

        static byte[][] _scratch; // Scratch memory
        public static byte[][] ScratchAlloc(byte[][] cell, int n)
        {
            cell = _scratch;
            if (cell == null)
                cell = new byte[n < 200 ? 200 : n][];
            else if (cell.Length < n)
                Array.Resize(ref cell, n);
            _scratch = null;
            return cell;
        }

        public static void ScratchFree(byte[][] cell)
        {
            if (cell != null)
            {
                if (_scratch == null || _scratch.Length < cell.Length)
                {
                    Debug.Assert(MemdebugHasType(cell, MEMTYPE.SCRATCH));
                    Debug.Assert(MemdebugNoType(cell, ~MEMTYPE.SCRATCH));
                    MemdebugSetType(cell, MEMTYPE.HEAP);
                    _scratch = cell;
                }
                // larger Scratch 2 already in use, let the C# GC handle
                cell = null;
            }
        }

        #endregion

        internal static string Mprintf(object tag, string format, params object[] args)
        {
            throw new NotImplementedException();
        }
    }
}