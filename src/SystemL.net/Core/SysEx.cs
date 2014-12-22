using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
namespace Core
{
    public class TagBase
    {
        public MutexEx Mutex;       // Connection mutex
        public bool MallocFailed;   // True if we have seen a malloc failure
        public RC ErrCode;          // Most recent error code (RC_*)
        public int ErrMask;         // & result codes with this before returning
    }

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

#if ENABLE_8_3_NAMES
        public static void FileSuffix3(string baseFilename, ref string z)
        {
#if ENABLE_8_3_NAMESx2
		if (!sqlite3_uri_boolean(baseFilename, "8_3_names", 0)) return;
#endif
            int sz = z.Length;
            int i;
            for (i = sz - 1; i > 0 && z[i] != '/' && z[i] != '.'; i--) { }
            if (z[i] == '.' && C._ALWAYS(sz > i + 4)) C._memmove(&z[i + 1], &z[sz - 3], 4);
        }
#else
        public static void FileSuffix3(string baseFilename, ref string z) { }
#endif

        public static RC ApiExit(TagBase tag, RC rc)
        {
            // If the ctx handle is not NULL, then we must hold the connection handle mutex here. Otherwise the read (and possible write) of db->mallocFailed 
            // is unsafe, as is the call to sqlite3Error().
            Debug.Assert(tag == null || MutexEx.Held(tag.Mutex));
            if (tag != null && (tag.MallocFailed || rc == RC.IOERR_NOMEM))
            {
                Error(tag, RC.NOMEM, null);
                tag.MallocFailed = false;
                rc = RC.NOMEM;
            }
            return (RC)((int)rc & (tag != null ? tag.ErrMask : 0xff));
        }

        void sqlite3Error(sqlite3 *db, int err_code, const char *zFormat, ...){
  if( db && (db->pErr || (db->pErr = sqlite3ValueNew(db))!=0) ){
    db->errCode = err_code;
    if( zFormat ){
      char *z;
      va_list ap;
      va_start(ap, zFormat);
      z = sqlite3VMPrintf(db, zFormat, ap);
      va_end(ap);
      sqlite3ValueSetStr(db->pErr, -1, z, SQLITE_UTF8, SQLITE_DYNAMIC);
    }else{
      sqlite3ValueSetStr(db->pErr, 0, 0, SQLITE_UTF8, SQLITE_STATIC);
    }
  }
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