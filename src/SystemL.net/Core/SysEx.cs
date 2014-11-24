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

        #region Memory Allocation

        public static Action<object> DESTRUCTOR_DYNAMIC;

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

    }
}