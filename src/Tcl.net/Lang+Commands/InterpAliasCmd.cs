#region Foreign-License
/*
	Implements the built-in "interp" Tcl command.

Copyright (c) 2000 Christian Krone.
Copyright (c) 2012 Sky Morey

See the file "license.terms" for information on usage and redistribution of this file, and for a DISCLAIMER OF ALL WARRANTIES.
*/
#endregion
using System;
using System.Collections;

namespace Tcl.Lang
{

    /// <summary> This class implements the alias commands, which are created
    /// in response to the built-in "interp alias" command in Tcl.
    /// 
    /// </summary>

    class InterpAliasCmd : ICommandWithDispose
    {

        // Name of alias command in slave interp.

        internal TclObject name;

        // Interp in which target command will be invoked.

        private Interp targetInterp;

        // Tcl list making up the prefix of the target command to be invoked in
        // the target interpreter. Additional arguments specified when calling
        // the alias in the slave interp will be appended to the prefix before
        // the command is invoked.

        private TclObject prefix;

        // Source command in slave interpreter, bound to command that invokes
        // the target command in the target interpreter.

        private WrappedCommand slaveCmd;

        // Entry for the alias hash table in slave.
        // This is used by alias deletion to remove the alias from the slave
        // interpreter alias table.

        private string aliasEntry;

        // Interp in which the command is defined.
        // This is the interpreter with the aliasTable in Slave.

        private Interp slaveInterp;
        public TCL.CompletionCode CmdProc(Interp interp, TclObject[] argv)
        {
            targetInterp.preserve();
            targetInterp._nestLevel++;

            targetInterp.ResetResult();
            targetInterp.allowExceptions();

            // Append the arguments to the command prefix and invoke the command
            // in the target interp's global namespace.

            TclObject[] prefv = TclList.getElements(interp, prefix);
            TclObject cmd = TclList.NewInstance();
            cmd.Preserve();
            TclList.replace(interp, cmd, 0, 0, prefv, 0, prefv.Length - 1);
            TclList.replace(interp, cmd, prefv.Length, 0, argv, 1, argv.Length - 1);
            TclObject[] cmdv = TclList.getElements(interp, cmd);

            TCL.CompletionCode result = targetInterp.invoke(cmdv, Interp.INVOKE_NO_TRACEBACK);

            cmd.Release();
            targetInterp._nestLevel--;

            // Check if we are at the bottom of the stack for the target interpreter.
            // If so, check for special return codes.

            if (targetInterp._nestLevel == 0)
            {
                if (result == TCL.CompletionCode.RETURN)
                {
                    result = targetInterp.updateReturnInfo();
                }
                if (result != TCL.CompletionCode.OK && result != TCL.CompletionCode.ERROR)
                {
                    try
                    {
                        targetInterp.processUnexpectedResult(result);
                    }
                    catch (TclException e)
                    {
                        result = e.GetCompletionCode();
                    }
                }
            }

            targetInterp.release();
            interp.transferResult(targetInterp, result);
            return TCL.CompletionCode.RETURN;
        }
        public void Dispose()
        {
            if ((System.Object)aliasEntry != null)
            {
                SupportClass.HashtableRemove(slaveInterp._aliasTable, aliasEntry);
            }

            if (slaveCmd != null)
            {
                SupportClass.HashtableRemove(targetInterp._targetTable, slaveCmd);
            }

            name.Release();
            prefix.Release();
        }
        internal static void create(Interp interp, Interp slaveInterp, Interp masterInterp, TclObject name, TclObject targetName, int objIx, TclObject[] objv)
        {

            string inString = name.ToString();

            InterpAliasCmd alias = new InterpAliasCmd();

            alias.name = name;
            name.Preserve();

            alias.slaveInterp = slaveInterp;
            alias.targetInterp = masterInterp;

            alias.prefix = TclList.NewInstance();
            alias.prefix.Preserve();
            TclList.Append(interp, alias.prefix, targetName);
            TclList.insert(interp, alias.prefix, 1, objv, objIx, objv.Length - 1);

            slaveInterp.CreateCommand(inString, alias);
            alias.slaveCmd = NamespaceCmd.findCommand(slaveInterp, inString, null, 0);

            try
            {
                interp.preventAliasLoop(slaveInterp, alias.slaveCmd);
            }
            catch (TclException e)
            {
                // Found an alias loop!  The last call to Tcl_CreateObjCommand made
                // the alias point to itself.  Delete the command and its alias
                // record.  Be careful to wipe out its client data first, so the
                // command doesn't try to delete itself.

                slaveInterp.DeleteCommandFromToken(alias.slaveCmd);
                throw;
            }

            // Make an entry in the alias table. If it already exists delete
            // the alias command. Then retry.

            if (slaveInterp._aliasTable.ContainsKey(inString))
            {
                InterpAliasCmd oldAlias = (InterpAliasCmd)slaveInterp._aliasTable[inString];
                slaveInterp.DeleteCommandFromToken(oldAlias.slaveCmd);
            }

            alias.aliasEntry = inString;
            SupportClass.PutElement(slaveInterp._aliasTable, inString, alias);

            // Create the new command. We must do it after deleting any old command,
            // because the alias may be pointing at a renamed alias, as in:
            //
            // interp alias {} foo {} bar		# Create an alias "foo"
            // rename foo zop				# Now rename the alias
            // interp alias {} foo {} zop		# Now recreate "foo"...

            SupportClass.PutElement(masterInterp._targetTable, alias.slaveCmd, slaveInterp);

            interp.SetResult(name);
        }
        internal static void delete(Interp interp, Interp slaveInterp, TclObject name)
        {
            // If the alias has been renamed in the slave, the master can still use
            // the original name (with which it was created) to find the alias to
            // delete it.


            string inString = name.ToString();
            if (!slaveInterp._aliasTable.ContainsKey(inString))
            {
                throw new TclException(interp, "alias \"" + inString + "\" not found");
            }

            InterpAliasCmd alias = (InterpAliasCmd)slaveInterp._aliasTable[inString];
            slaveInterp.DeleteCommandFromToken(alias.slaveCmd);
        }
        internal static void describe(Interp interp, Interp slaveInterp, TclObject name)
        {
            // If the alias has been renamed in the slave, the master can still use
            // the original name (with which it was created) to find the alias to
            // describe it.


            string inString = name.ToString();
            if (slaveInterp._aliasTable.ContainsKey(inString))
            {
                InterpAliasCmd alias = (InterpAliasCmd)slaveInterp._aliasTable[inString];
                interp.SetResult(alias.prefix);
            }
        }
        internal static void list(Interp interp, Interp slaveInterp)
        {
            TclObject result = TclList.NewInstance();
            interp.SetResult(result);

            IEnumerator aliases = slaveInterp._aliasTable.Values.GetEnumerator();
            while (aliases.MoveNext())
            {
                InterpAliasCmd alias = (InterpAliasCmd)aliases.Current;
                TclList.Append(interp, result, alias.name);
            }
        }
        internal WrappedCommand getTargetCmd(Interp interp)
        {
            TclObject[] objv = TclList.getElements(interp, prefix);

            string targetName = objv[0].ToString();
            return NamespaceCmd.findCommand(targetInterp, targetName, null, 0);
        }
        internal static Interp getTargetInterp(Interp slaveInterp, string aliasName)
        {
            if (!slaveInterp._aliasTable.ContainsKey(aliasName))
            {
                return null;
            }

            InterpAliasCmd alias = (InterpAliasCmd)slaveInterp._aliasTable[aliasName];

            return alias.targetInterp;
        }
    } // end InterpAliasCmd
}
