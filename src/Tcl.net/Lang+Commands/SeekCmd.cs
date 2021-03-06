#region Foreign-License
/*
Copyright (c) 1997 Sun Microsystems, Inc.
Copyright (c) 2012 Sky Morey

See the file "license.terms" for information on usage and redistribution of this file, and for a DISCLAIMER OF ALL WARRANTIES.
*/
#endregion
using System;
using System.IO;
namespace Tcl.Lang
{

    /// <summary> This class implements the built-in "seek" command in Tcl.</summary>

    class SeekCmd : ICommand
    {

        private static readonly string[] validOrigins = new string[] { "start", "current", "end" };

        internal const int OPT_START = 0;
        internal const int OPT_CURRENT = 1;
        internal const int OPT_END = 2;

        /// <summary> This procedure is invoked to process the "seek" Tcl command.
        /// See the user documentation for details on what it does.
        /// </summary>

        public TCL.CompletionCode CmdProc(Interp interp, TclObject[] argv)
        {

            Channel chan; /* The channel being operated on this method */
            int mode; /* Stores the search mode, either beg, cur or end
			* of file.  See the TclIO class for more info */

            if (argv.Length != 3 && argv.Length != 4)
            {
                throw new TclNumArgsException(interp, 1, argv, "channelId offset ?origin?");
            }

            // default is the beginning of the file

            mode = TclIO.SEEK_SET;
            if (argv.Length == 4)
            {
                int index = TclIndex.Get(interp, argv[3], validOrigins, "origin", 0);

                switch (index)
                {

                    case OPT_START:
                        {
                            mode = TclIO.SEEK_SET;
                            break;
                        }

                    case OPT_CURRENT:
                        {
                            mode = TclIO.SEEK_CUR;
                            break;
                        }

                    case OPT_END:
                        {
                            mode = TclIO.SEEK_END;
                            break;
                        }
                }
            }


            chan = TclIO.getChannel(interp, argv[1].ToString());
            if (chan == null)
            {

                throw new TclException(interp, "can not find channel named \"" + argv[1].ToString() + "\"");
            }
            long offset = TclInteger.Get(interp, argv[2]);

            try
            {
                chan.seek(interp, offset, mode);
            }
            catch (IOException e)
            {
                // FIXME: Need to figure out Tcl specific error conditions.
                // Should we also wrap an IOException in a ReflectException?
                throw new TclRuntimeError("SeekCmd.cmdProc() Error: IOException when seeking " + chan.ChanName + ":" + e.Message);
            }
            return TCL.CompletionCode.RETURN;
        }
    }
}
