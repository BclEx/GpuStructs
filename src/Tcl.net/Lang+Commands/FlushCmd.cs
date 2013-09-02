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

    /// <summary> This class implements the built-in "flush" command in Tcl.</summary>

    class FlushCmd : Command
    {

        /// <summary> This procedure is invoked to process the "flush" Tcl command.
        /// See the user documentation for details on what it does.
        /// 
        /// </summary>
        /// <param name="interp">the current interpreter.
        /// </param>
        /// <param name="argv">command arguments.
        /// </param>

        public TCL.CompletionCode cmdProc(Interp interp, TclObject[] argv)
        {

            Channel chan; /* The channel being operated on this method */

            if (argv.Length != 2)
            {
                throw new TclNumArgsException(interp, 1, argv, "channelId");
            }


            chan = TclIO.getChannel(interp, argv[1].ToString());
            if (chan == null)
            {

                throw new TclException(interp, "can not find channel named \"" + argv[1].ToString() + "\"");
            }

            try
            {
                chan.flush(interp);
            }
            catch (IOException e)
            {
                throw new TclRuntimeError("FlushCmd.cmdProc() Error: IOException when flushing " + chan.ChanName);
            }
            return TCL.CompletionCode.RETURN;
        }
    }
}
