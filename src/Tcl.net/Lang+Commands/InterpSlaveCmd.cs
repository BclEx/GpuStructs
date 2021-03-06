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

  /// <summary> This class implements the slave interpreter commands, which are created
  /// in response to the built-in "interp create" command in Tcl.
  /// 
  /// It is also used by the "interp" command to record and find information
  /// about slave interpreters. Maps from a command name in the master to
  /// information about a slave interpreter, e.g. what aliases are defined
  /// in it.
  /// </summary>

  class InterpSlaveCmd : ICommandWithDispose, IAssocData
  {

    private static readonly string[] options = new string[] { "alias", "aliases", "eval", "expose", "hide", "hidden", "issafe", "invokehidden", "marktrusted" };
    private const int OPT_ALIAS = 0;
    private const int OPT_ALIASES = 1;
    private const int OPT_EVAL = 2;
    private const int OPT_EXPOSE = 3;
    private const int OPT_HIDE = 4;
    private const int OPT_HIDDEN = 5;
    private const int OPT_ISSAFE = 6;
    private const int OPT_INVOKEHIDDEN = 7;
    private const int OPT_MARKTRUSTED = 8;

    private static readonly string[] hiddenOptions = new string[] { "-global", "--" };
    private const int OPT_HIDDEN_GLOBAL = 0;
    private const int OPT_HIDDEN_LAST = 1;

    // Master interpreter for this slave.

    internal Interp masterInterp;

    // Hash entry in masters slave table for this slave interpreter.
    // Used to find this record, and used when deleting the slave interpreter
    // to delete it from the master's table.

    internal string path;

    // The slave interpreter.

    internal Interp slaveInterp;

    // Interpreter object command.

    internal WrappedCommand interpCmd;
    public TCL.CompletionCode CmdProc( Interp interp, TclObject[] objv )
    {
      if ( objv.Length < 2 )
      {
        throw new TclNumArgsException( interp, 1, objv, "cmd ?arg ...?" );
      }
      int cmd = TclIndex.Get( interp, objv[1], options, "option", 0 );

      switch ( cmd )
      {

        case OPT_ALIAS:
          if ( objv.Length == 3 )
          {
            InterpAliasCmd.describe( interp, slaveInterp, objv[2] );
            return TCL.CompletionCode.RETURN;
          }

          if ( "".Equals( objv[3].ToString() ) )
          {
            if ( objv.Length == 4 )
            {
              InterpAliasCmd.delete( interp, slaveInterp, objv[2] );
              return TCL.CompletionCode.RETURN;
            }
          }
          else
          {
            InterpAliasCmd.create( interp, slaveInterp, interp, objv[2], objv[3], 4, objv );
            return TCL.CompletionCode.RETURN;
          }
          throw new TclNumArgsException( interp, 2, objv, "aliasName ?targetName? ?args..?" );

        case OPT_ALIASES:
          InterpAliasCmd.list( interp, slaveInterp );
          break;

        case OPT_EVAL:
          if ( objv.Length < 3 )
          {
            throw new TclNumArgsException( interp, 2, objv, "arg ?arg ...?" );
          }
          eval( interp, slaveInterp, 2, objv );
          break;

        case OPT_EXPOSE:
          if ( objv.Length < 3 || objv.Length > 4 )
          {
            throw new TclNumArgsException( interp, 2, objv, "hiddenCmdName ?cmdName?" );
          }
          expose( interp, slaveInterp, 2, objv );
          break;

        case OPT_HIDE:
          if ( objv.Length < 3 || objv.Length > 4 )
          {
            throw new TclNumArgsException( interp, 2, objv, "cmdName ?hiddenCmdName?" );
          }
          hide( interp, slaveInterp, 2, objv );
          break;

        case OPT_HIDDEN:
          if ( objv.Length != 2 )
          {
            throw new TclNumArgsException( interp, 2, objv, null );
          }
          InterpSlaveCmd.hidden( interp, slaveInterp );
          break;

        case OPT_ISSAFE:
          interp.SetResult( slaveInterp._isSafe );
          break;

        case OPT_INVOKEHIDDEN:
          bool global = false;
          int i;
          for ( i = 2; i < objv.Length; i++ )
          {

            if ( objv[i].ToString()[0] != '-' )
            {
              break;
            }
            int index = TclIndex.Get( interp, objv[i], hiddenOptions, "option", 0 );
            if ( index == OPT_HIDDEN_GLOBAL )
            {
              global = true;
            }
            else
            {
              i++;
              break;
            }
          }
          if ( objv.Length - i < 1 )
          {
            throw new TclNumArgsException( interp, 2, objv, "?-global? ?--? cmd ?arg ..?" );
          }
          InterpSlaveCmd.invokeHidden( interp, slaveInterp, global, i, objv );
          break;

        case OPT_MARKTRUSTED:
          if ( objv.Length != 2 )
          {
            throw new TclNumArgsException( interp, 2, objv, null );
          }
          markTrusted( interp, slaveInterp );
          break;
      }
      return TCL.CompletionCode.RETURN;
    }
    /// <summary>----------------------------------------------------------------------
    /// 
    /// disposeCmd --
    /// 
    /// Invoked when an object command for a slave interpreter is deleted;
    /// cleans up all state associated with the slave interpreter and destroys
    /// the slave interpreter.
    /// 
    /// Results:
    /// None.
    /// 
    /// Side effects:
    /// Cleans up all state associated with the slave interpreter and
    /// destroys the slave interpreter.
    /// 
    /// ----------------------------------------------------------------------
    /// </summary>

    public void Dispose()
    {
      // Unlink the slave from its master interpreter.

      SupportClass.HashtableRemove( masterInterp._slaveTable, path );

      // Set to null so that when the InterpInfo is cleaned up in the slave
      // it does not try to delete the command causing all sorts of grief.
      // See SlaveRecordDeleteProc().

      interpCmd = null;

      if ( slaveInterp != null )
      {
        slaveInterp.Dispose();
      }
    }
    public void Dispose( Interp interp )
    // Current interpreter.
    {
      // There shouldn't be any commands left.

      if ( !( interp._slaveTable.Count == 0 ) )
      {
        System.Console.Error.WriteLine( "InterpInfoDeleteProc: still exist commands" );
      }
      interp._slaveTable = null;

      // Tell any interps that have aliases to this interp that they should
      // delete those aliases.  If the other interp was already dead, it
      // would have removed the target record already. 

      // TODO ATK
      foreach ( WrappedCommand slaveCmd in new ArrayList( interp._targetTable.Keys ) )
      {
        Interp slaveInterp = (Interp)interp._targetTable[slaveCmd];
        slaveInterp.DeleteCommandFromToken( slaveCmd );
      }
      interp._targetTable = null;

      if ( interp._interpChanTable != null )
      {
        foreach ( Channel channel in new ArrayList( interp._interpChanTable.Values ) )
        {
          TclIO.unregisterChannel( interp, channel );
        }
      }

      if ( interp._slave.interpCmd != null )
      {
        // Tcl_DeleteInterp() was called on this interpreter, rather
        // "interp delete" or the equivalent deletion of the command in the
        // master.  First ensure that the cleanup callback doesn't try to
        // delete the interp again.

        interp._slave.slaveInterp = null;
        interp._slave.masterInterp.DeleteCommandFromToken( interp._slave.interpCmd );
      }

      // There shouldn't be any aliases left.

      if ( !( interp._aliasTable.Count == 0 ) )
      {
        System.Console.Error.WriteLine( "InterpInfoDeleteProc: still exist aliases" );
      }
      interp._aliasTable = null;
    }
    internal static Interp create( Interp interp, TclObject path, bool safe )
    {
      Interp masterInterp;
      string pathString;

      TclObject[] objv = TclList.getElements( interp, path );

      if ( objv.Length < 2 )
      {
        masterInterp = interp;

        pathString = path.ToString();
      }
      else
      {
        TclObject obj = TclList.NewInstance();

        TclList.insert( interp, obj, 0, objv, 0, objv.Length - 2 );
        masterInterp = InterpCmd.getInterp( interp, obj );

        pathString = objv[objv.Length - 1].ToString();
      }
      if ( !safe )
      {
        safe = masterInterp._isSafe;
      }

      if ( masterInterp._slaveTable.ContainsKey( pathString ) )
      {
        throw new TclException( interp, "interpreter named \"" + pathString + "\" already exists, cannot create" );
      }

      Interp slaveInterp = new Interp();
      InterpSlaveCmd slave = new InterpSlaveCmd();

      slaveInterp._slave = slave;
      slaveInterp.SetAssocData( "InterpSlaveCmd", slave );

      slave.masterInterp = masterInterp;
      slave.path = pathString;
      slave.slaveInterp = slaveInterp;

      masterInterp.CreateCommand( pathString, slaveInterp._slave );
      slaveInterp._slave.interpCmd = NamespaceCmd.findCommand( masterInterp, pathString, null, 0 );

      SupportClass.PutElement( masterInterp._slaveTable, pathString, slaveInterp._slave );

      slaveInterp.SetVar( "tcl_interactive", "0", TCL.VarFlag.GLOBAL_ONLY );

      // Inherit the recursion limit.

      slaveInterp._maxNestingDepth = masterInterp._maxNestingDepth;

      if ( safe )
      {
        try
        {
          makeSafe( slaveInterp );
        }
        catch ( TclException e )
        {
          SupportClass.WriteStackTrace( e, Console.Error );
        }
      }
      else
      {
        //Tcl_Init(slaveInterp);
      }

      return slaveInterp;
    }
    internal static void eval( Interp interp, Interp slaveInterp, int objIx, TclObject[] objv )
    {
      TCL.CompletionCode result;

      slaveInterp.preserve();
      slaveInterp.allowExceptions();

      try
      {
        if ( objIx + 1 == objv.Length )
        {
          slaveInterp.Eval( objv[objIx], 0 );
        }
        else
        {
          TclObject obj = TclList.NewInstance();
          for ( int ix = objIx; ix < objv.Length; ix++ )
          {
            TclList.Append( interp, obj, objv[ix] );
          }
          obj.Preserve();
          slaveInterp.Eval( obj, 0 );
          obj.Release();
        }
        result = slaveInterp._returnCode;
      }
      catch ( TclException e )
      {
        result = e.GetCompletionCode();
      }

      slaveInterp.release();
      interp.transferResult( slaveInterp, result );
    }
    internal static void expose( Interp interp, Interp slaveInterp, int objIx, TclObject[] objv )
    {
      if ( interp._isSafe )
      {
        throw new TclException( interp, "permission denied: " + "safe interpreter cannot expose commands" );
      }

      int nameIdx = objv.Length - objIx == 1 ? objIx : objIx + 1;

      try
      {

        slaveInterp.exposeCommand( objv[objIx].ToString(), objv[nameIdx].ToString() );
      }
      catch ( TclException e )
      {
        interp.transferResult( slaveInterp, e.GetCompletionCode() );
        throw;
      }
    }
    internal static void hide( Interp interp, Interp slaveInterp, int objIx, TclObject[] objv )
    {
      if ( interp._isSafe )
      {
        throw new TclException( interp, "permission denied: " + "safe interpreter cannot hide commands" );
      }

      int nameIdx = objv.Length - objIx == 1 ? objIx : objIx + 1;

      try
      {

        slaveInterp.hideCommand( objv[objIx].ToString(), objv[nameIdx].ToString() );
      }
      catch ( TclException e )
      {
        interp.transferResult( slaveInterp, e.GetCompletionCode() );
        throw;
      }
    }
    internal static void hidden( Interp interp, Interp slaveInterp )
    {
      if ( slaveInterp._hiddenCmdTable == null )
      {
        return;
      }

      TclObject result = TclList.NewInstance();
      interp.SetResult( result );

      IEnumerator hiddenCmds = slaveInterp._hiddenCmdTable.Keys.GetEnumerator();
      while ( hiddenCmds.MoveNext() )
      {
        string cmdName = (string)hiddenCmds.Current;
        TclList.Append( interp, result, TclString.NewInstance( cmdName ) );
      }
    }
    internal static void invokeHidden( Interp interp, Interp slaveInterp, bool global, int objIx, TclObject[] objv )
    {
      TCL.CompletionCode result;

      if ( interp._isSafe )
      {
        throw new TclException( interp, "not allowed to " + "invoke hidden commands from safe interpreter" );
      }

      slaveInterp.preserve();
      slaveInterp.allowExceptions();

      TclObject[] localObjv = new TclObject[objv.Length - objIx];
      for ( int i = 0; i < objv.Length - objIx; i++ )
      {
        localObjv[i] = objv[i + objIx];
      }

      try
      {
        if ( global )
        {
          slaveInterp.invokeGlobal( localObjv, Interp.INVOKE_HIDDEN );
        }
        else
        {
          slaveInterp.invoke( localObjv, Interp.INVOKE_HIDDEN );
        }
        result = slaveInterp._returnCode;
      }
      catch ( TclException e )
      {
        result = e.GetCompletionCode();
      }

      slaveInterp.release();
      interp.transferResult( slaveInterp, result );
    }
    internal static void markTrusted( Interp interp, Interp slaveInterp )
    {
      if ( interp._isSafe )
      {
        throw new TclException( interp, "permission denied: " + "safe interpreter cannot mark trusted" );
      }
      slaveInterp._isSafe = false;
    }
    private static void makeSafe( Interp interp )
    {
      Channel chan; // Channel to remove from safe interpreter.

      interp.hideUnsafeCommands();

      interp._isSafe = true;

      //  Unsetting variables : (which should not have been set 
      //  in the first place, but...)

      // No env array in a safe slave.

      try
      {
        interp.UnsetVar( "env", TCL.VarFlag.GLOBAL_ONLY );
      }
      catch ( TclException e )
      {
      }

      // Remove unsafe parts of tcl_platform

      try
      {
        interp.UnsetVar( "tcl_platform", "os", TCL.VarFlag.GLOBAL_ONLY );
      }
      catch ( TclException e )
      {
      }
      try
      {
        interp.UnsetVar( "tcl_platform", "osVersion", TCL.VarFlag.GLOBAL_ONLY );
      }
      catch ( TclException e )
      {
      }
      try
      {
        interp.UnsetVar( "tcl_platform", "machine", TCL.VarFlag.GLOBAL_ONLY );
      }
      catch ( TclException e )
      {
      }
      try
      {
        interp.UnsetVar( "tcl_platform", "user", TCL.VarFlag.GLOBAL_ONLY );
      }
      catch ( TclException e )
      {
      }

      // Unset path informations variables
      // (the only one remaining is [info nameofexecutable])

      try
      {
        interp.UnsetVar( "tclDefaultLibrary", TCL.VarFlag.GLOBAL_ONLY );
      }
      catch ( TclException e )
      {
      }
      try
      {
        interp.UnsetVar( "tcl_library", TCL.VarFlag.GLOBAL_ONLY );
      }
      catch ( TclException e )
      {
      }
      try
      {
        interp.UnsetVar( "tcl_pkgPath", TCL.VarFlag.GLOBAL_ONLY );
      }
      catch ( TclException e )
      {
      }

      // Remove the standard channels from the interpreter; safe interpreters
      // do not ordinarily have access to stdin, stdout and stderr.
      //
      // NOTE: These channels are not added to the interpreter by the
      // Tcl_CreateInterp call, but may be added later, by another I/O
      // operation. We want to ensure that the interpreter does not have
      // these channels even if it is being made safe after being used for
      // some time..

      chan = TclIO.GetStdChannel( StdChannel.STDIN );
      if ( chan != null )
      {
        TclIO.unregisterChannel( interp, chan );
      }
      chan = TclIO.GetStdChannel( StdChannel.STDOUT );
      if ( chan != null )
      {
        TclIO.unregisterChannel( interp, chan );
      }
      chan = TclIO.GetStdChannel( StdChannel.STDERR );
      if ( chan != null )
      {
        TclIO.unregisterChannel( interp, chan );
      }
    }
  } // end InterpSlaveCmd
}
