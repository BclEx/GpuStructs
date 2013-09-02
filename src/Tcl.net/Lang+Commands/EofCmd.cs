#region Foreign-License
/*
Copyright (c) 1997 Sun Microsystems, Inc.
Copyright (c) 2012 Sky Morey

See the file "license.terms" for information on usage and redistribution of this file, and for a DISCLAIMER OF ALL WARRANTIES.
*/
#endregion
using System;
namespace Tcl.Lang
{

    /// <summary> This class implements the built-in "eof" command in Tcl.</summary>

    class EofCmd : ICommand
    {
        /// <summary> This procedure is invoked to process the "eof" Tcl command.
        /// See the user documentation for details on what it does.
        /// 
        /// </summary>
        /// <param name="interp">the current interpreter.
        /// </param>
        /// <param name="argv">command arguments.
        /// </param>

        public TCL.CompletionCode CmdProc(Interp interp, TclObject[] argv)
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

            if (chan.eof())
            {
                interp.setResult(TclInteger.NewInstance(1));
            }
            else
            {
                interp.setResult(TclInteger.NewInstance(0));
            }
            return TCL.CompletionCode.RETURN;
        }
    }
}
