#region Foreign-License
/*
Copyright (c) 1997 Sun Microsystems, Inc.
Copyright (c) 2012 Sky Morey

See the file "license.terms" for information on usage and redistribution of this file, and for a DISCLAIMER OF ALL WARRANTIES.
*/
#endregion
using System.Text;
using System.IO;
namespace Tcl.Lang
{

    /// <summary> This class implements the built-in "gets" command in Tcl.</summary>

    class GetsCmd : ICommand
    {

        /// <summary> This procedure is invoked to process the "gets" Tcl command.
        /// See the user documentation for details on what it does.
        /// 
        /// </summary>
        /// <param name="interp">the current interpreter.
        /// </param>
        /// <param name="argv">command arguments.
        /// </param>

        public TCL.CompletionCode CmdProc(Interp interp, TclObject[] argv)
        {

            bool writeToVar = false; // If true write to var passes as arg
            string varName = ""; // The variable to write value to
            Channel chan; // The channel being operated on
            int lineLen;
            TclObject line;

            if ((argv.Length < 2) || (argv.Length > 3))
            {
                throw new TclNumArgsException(interp, 1, argv, "channelId ?varName?");
            }

            if (argv.Length == 3)
            {
                writeToVar = true;

                varName = argv[2].ToString();
            }


            chan = TclIO.getChannel(interp, argv[1].ToString());
            if (chan == null)
            {

                throw new TclException(interp, "can not find channel named \"" + argv[1].ToString() + "\"");
            }

            try
            {
                line = TclString.NewInstance(new StringBuilder(64));
                lineLen = chan.read(interp, line, TclIO.READ_LINE, 0);
                if (lineLen < 0)
                {
                    // FIXME: Need more specific posix error codes!
                    if (!chan.eof() && !chan.isBlocked(interp))
                    {

                        throw new TclPosixException(interp, TclPosixException.EIO, true, "error reading \"" + argv[1].ToString() + "\"");
                    }
                    lineLen = -1;
                }
                if (writeToVar)
                {
                    interp.SetVar(varName, line, 0);
                    interp.SetResult(lineLen);
                }
                else
                {
                    interp.SetResult(line);
                }
            }
            catch (IOException e)
            {
                throw new TclRuntimeError("GetsCmd.cmdProc() Error: IOException when getting " + chan.ChanName + ": " + e.Message);
            }
            return TCL.CompletionCode.RETURN;
        }
    }
}
