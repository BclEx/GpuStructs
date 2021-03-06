#region Foreign-License
/*
Copyright (c) 2001 Bruce A. Johnson
Copyright (c) 2012 Sky Morey

See the file "license.terms" for information on usage and redistribution of this file, and for a DISCLAIMER OF ALL WARRANTIES.
*/
#endregion
using System;
using System.Text;
using System.Collections;
using System.IO;

namespace Tcl.Lang
{

    /// <summary> This class implements the built-in "encoding" command in Tcl.</summary>

    class EncodingCmd : ICommand
    {
        // FIXME: Make sure this is a global property and not a per-interp
        // property!
        internal static string systemTclEncoding = "utf-8";
        internal static Encoding systemJavaEncoding = UTF8Encoding.UTF8;

        internal static string[] tclNames = new string[] { "utf-8", "unicode", "ascii", "utf-7" };

        internal static readonly Encoding[] encodings = new Encoding[] { UTF8Encoding.UTF8, UnicodeEncoding.Unicode, ASCIIEncoding.Unicode, UTF7Encoding.UTF7 };

        internal static int[] bytesPerChar = new int[] { 1, 2, 1, 1 };

        private static readonly string[] validCmds = new string[] { "convertfrom", "convertto", "names", "system" };

        internal const int OPT_CONVERTFROM = 0;
        internal const int OPT_CONVERTTO = 1;
        internal const int OPT_NAMES = 2;
        internal const int OPT_SYSTEM = 3;

        /// <summary> This procedure is invoked to process the "encoding" Tcl command.
        /// See the user documentation for details on what it does.
        /// 
        /// </summary>
        /// <param name="interp">the current interpreter.
        /// </param>
        /// <param name="argv">command arguments.
        /// </param>

        public TCL.CompletionCode CmdProc(Interp interp, TclObject[] argv)
        {
            if (argv.Length < 2)
            {
                throw new TclNumArgsException(interp, 1, argv, "option ?arg ...?");
            }

            int index = TclIndex.Get(interp, argv[1], validCmds, "option", 0);

            switch (index)
            {

                case OPT_CONVERTTO:
                case OPT_CONVERTFROM:
                    {
                        string tclEncoding;
                        Encoding javaEncoding;
                        TclObject data;

                        if (argv.Length == 3)
                        {
                            tclEncoding = systemTclEncoding;
                            data = argv[2];
                        }
                        else if (argv.Length == 4)
                        {

                            tclEncoding = argv[2].ToString();
                            data = argv[3];
                        }
                        else
                        {
                            throw new TclNumArgsException(interp, 2, argv, "?encoding? data");
                        }

                        javaEncoding = getJavaName(tclEncoding);

                        if ((System.Object)javaEncoding == null)
                        {
                            throw new TclException(interp, "unknown encoding \"" + tclEncoding + "\"");
                        }

                        try
                        {
                            if (index == OPT_CONVERTFROM)
                            {
                                // Treat the string as binary data
                                byte[] bytes = TclByteArray.getBytes(interp, data);

                                // ATK
                                interp.SetResult(System.Text.Encoding.UTF8.GetString(bytes, 0, bytes.Length));
                            }
                            else
                            {
                                // Store the result as binary data


                                // ATK byte[] bytes = data.ToString().getBytes(javaEncoding);
                                byte[] bytes = System.Text.Encoding.UTF8.GetBytes(data.ToString());
                                interp.SetResult(TclByteArray.NewInstance(bytes));
                            }
                        }
                        catch (IOException ex)
                        {
                            throw new TclRuntimeError("Encoding.cmdProc() error: " + "unsupported java encoding \"" + javaEncoding + "\"");
                        }

                        break;
                    }

                case OPT_NAMES:
                    {
                        if (argv.Length > 2)
                        {
                            throw new TclNumArgsException(interp, 2, argv, null);
                        }

                        TclObject list = TclList.NewInstance();
                        for (int i = 0; i < tclNames.Length; i++)
                        {
                            TclList.Append(interp, list, TclString.NewInstance(tclNames[i]));
                        }
                        interp.SetResult(list);
                        break;
                    }

                case OPT_SYSTEM:
                    {
                        if (argv.Length > 3)
                            throw new TclNumArgsException(interp, 2, argv, "?encoding?");

                        if (argv.Length == 2)
                        {
                            interp.SetResult(systemTclEncoding);
                        }
                        else
                        {

                            string tclEncoding = argv[2].ToString();
                            Encoding javaEncoding = getJavaName(tclEncoding);

                            if (javaEncoding == null)
                            {
                                throw new TclException(interp, "unknown encoding \"" + tclEncoding + "\"");
                            }

                            systemTclEncoding = tclEncoding;
                            systemJavaEncoding = javaEncoding;
                        }

                        break;
                    }

                default:
                    {
                        throw new TclRuntimeError("Encoding.cmdProc() error: " + "incorrect index returned from TclIndex.get()");
                    }

            }
            return TCL.CompletionCode.RETURN;
        }

        internal static int getBytesPerChar(Encoding encoding)
        {
            return encoding.GetMaxByteCount(1);
        }

        internal static System.Text.Encoding getJavaName(string name)
        {
            for (int x = 0; x < EncodingCmd.tclNames.Length; x++)
            {
                if (EncodingCmd.tclNames[x] == name)
                    return EncodingCmd.encodings[x];
            }
            return null;
        }

        internal static string getTclName(Encoding encoding)
        {
            for (int x = 0; x < EncodingCmd.encodings.Length; x++)
            {
                if (EncodingCmd.encodings[x].EncodingName == encoding.EncodingName)
                    return EncodingCmd.tclNames[x];
            }
            return null;
        }
    }
}
