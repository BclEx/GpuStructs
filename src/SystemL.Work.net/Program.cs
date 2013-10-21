using Core;
using Core.IO;
using System;

namespace GpuData
{
    class MainClass
    {
        public static void Main(string[] args)
        {
            SysEx.Initialize();
            //TestVFS();
            //Console.ReadKey();
        }

        static void TestVFS()
        {
            var vfs = VSystem.FindVfs("win32");
            if (vfs == null)
                throw new InvalidOperationException();
            var file = vfs.CreateOsFile();
            VSystem.OPEN flagOut;
            var rc = vfs.Open(@"C:\T_\Test.db", file, VSystem.OPEN.CREATE | VSystem.OPEN.READWRITE | VSystem.OPEN.MAIN_DB, out flagOut);
            if (rc != RC.OK)
                throw new InvalidOperationException();
            file.Write4(0, 12345);
            file.Close();
        }
    }
}
