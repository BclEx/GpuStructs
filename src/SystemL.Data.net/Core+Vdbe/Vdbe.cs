using FILE = System.IO.TextWriter;
using System;
using System.Diagnostics;
using System.IO;
using System.Text;

#region Limits
#if MAX_ATTACHED
using yDbMask = System.Int64; 
#else
using yDbMask = System.Int32;
#endif
#endregion

namespace Core
{
    public partial class Vdbe
    {
        #region Preamble

        //#if DEBUG
        //        static void MemAboutToChange_(Vdbe P, Mem M) { P.MemAboutToChange(P, M); }
        //#else
        //        static void MemAboutToChange_(Vdbe P, Mem M) {}
        //#endif

#if TEST
        static int g_search_count = 0;
        static int g_interrupt_count = 0;
        static int g_sort_count = 0;
        static int g_max_blobsize = 0;
        static void UpdateMaxBlobsize(Mem p) { if ((p.Flags & (MEM.Str | MEM.Blob)) != 0 && p.N > g_max_blobsize) g_max_blobsize = p.N; }
        static int g_found_count = 0;
        static void UPDATE_MAX_BLOBSIZE(Mem P) { UpdateMaxBlobsize(P); }
#else
        static void UPDATE_MAX_BLOBSIZE(Mem P) { }
#endif
        //: static void Deephemeralize(Mem P) { }

        #endregion

        #region Name2

        public static void MemStoreType(Mem mem)
        {
            MEM flags = mem.Flags;
            if ((flags & MEM.Null) != 0) { mem.Type = TYPE.NULL; mem.Z = null; }
            else if ((flags & MEM.Int) != 0) mem.Type = TYPE.INTEGER;
            else if ((flags & MEM.Real) != 0) mem.Type = TYPE.FLOAT;
            else if ((flags & MEM.Str) != 0) mem.Type = TYPE.TEXT;
            else mem.Type = TYPE.BLOB;
        }

        static VdbeCursor AllocateCursor(Vdbe p, int curID, int fields, int db, bool isBtreeCursor)
        {
            // Find the memory cell that will be used to store the blob of memory required for this VdbeCursor structure. It is convenient to use a 
            // vdbe memory cell to manage the memory allocation required for a VdbeCursor structure for the following reasons:
            //
            //   * Sometimes cursor numbers are used for a couple of different purposes in a vdbe program. The different uses might require
            //     different sized allocations. Memory cells provide growable allocations.
            //
            //   * When using ENABLE_MEMORY_MANAGEMENT, memory cell buffers can be freed lazily via the sqlite3_release_memory() API. This
            //     minimizes the number of malloc calls made by the system.
            //
            // Memory cells for cursors are allocated at the top of the address space. Memory cell (p->nMem) corresponds to cursor 0. Space for
            // cursor 1 is managed by memory cell (p->nMem-1), etc.
            VdbeCursor cx = null;

            Debug.Assert(curID < p.Cursors.length);
            if (p.Cursors[curID] != null)
            {
                p.FreeCursor(p.Cursors[curID]);
                p.Cursors[curID] = null;
            }
            //: if (Vdbe.MemGrow(mem, bytes, false) == RC.OK)
            {
                p.Cursors[curID] = cx = new VdbeCursor();
                cx.Db = db;
                cx.Fields = fields;
                if (fields != 0)
                    cx.Types = new uint[fields];
                if (isBtreeCursor)
                {
                    cx.Cursor = sqlite3MemMallocBtCursor(cx.Cursor);
                    Btree.CursorZero(cx.Cursor);
                }
            }
            return cx;
        }

        static void ApplyNumericAffinity(Mem rec)
        {
            if ((rec.Flags & (MEM.Real | MEM.Int)) == 0)
            {
                double r = 0.0;
                long i = 0;
                TEXTENCODE encode = rec.Encode;
                if ((rec.Flags & MEM.Str) == 0) return;
                if (!ConvertEx.Atof(rec.Z, ref r, rec.N, encode)) return;
                if (ConvertEx.Atoi64(rec.Z, out i, rec.N, encode) == 0)
                {
                    rec.u.I = i;
                    rec.Flags |= MEM.Int;
                }
                else
                {
                    rec.R = r;
                    rec.Flags |= MEM.Real;
                }
            }
        }

        static void ApplyAffinity(Mem rec, AFF affinity, TEXTENCODE encode)
        {
            if (affinity == AFF.TEXT)
            {
                // Only attempt the conversion to TEXT if there is an integer or real representation (blob and NULL do not get converted) but no string representation.
                if ((rec.Flags & MEM.Str) == 0 && (rec.Flags & (MEM.Real | MEM.Int)) != 0)
                    MemStringify(rec, encode);
                //if ((rec.flags & (MEM.Blob | MEM.Str)) == (MEM.Blob | MEM.Str))
                //{
                //    var sb = new StringBuilder(rec.zBLOB.Length);
                //    for (int i = 0; i < rec.zBLOB.Length; i++)
                //        sb.Append((char)rec.zBLOB[i]);
                //    rec.Z = sb.ToString();
                //    C._free(ref rec.zBLOB);
                //    rec.flags &= ~MEM_Blob;
                //}
                rec.Flags &= ~(MEM.Real | MEM.Int);
            }
            else if (affinity != AFF.NONE)
            {
                Debug.Assert(affinity == AFF.INTEGER || affinity == AFF.REAL || affinity == AFF.NUMERIC);
                ApplyNumericAffinity(rec);
                if ((rec.Flags & MEM.Real) != 0)
                    IntegerAffinity(rec);
            }
        }

        public static TYPE ValueNumericType(Mem mem)
        {
            if (mem.Type == TYPE.TEXT)
            {
                ApplyNumericAffinity(mem);
                MemStoreType(mem);
            }
            return mem.Type;
        }

        static void ValueApplyAffinity(Mem mem, char affinity, TEXTENCODE encode)
        {
            ApplyAffinity(mem, (AFF)affinity, encode);
        }

#if DEBUG
        static StringBuilder _csr = new StringBuilder(100);
        static readonly string[] _encnames = new string[] { "(X)", "(8)", "(16LE)", "(16BE)" };
        public static void MemPrettyPrint(Mem mem, StringBuilder buf)
        {
            buf.Length = 0;
            _csr.Length = 0;

            MEM f = mem.Flags;
            if ((f & MEM.Blob) != 0)
            {
                char c;
                if ((f & MEM.Dyn) != 0)
                {
                    c = 'z';
                    Debug.Assert((f & (MEM.Static | MEM.Ephem)) == 0);
                }
                else if ((f & MEM.Static) != 0)
                {
                    c = 't';
                    Debug.Assert((f & (MEM.Dyn | MEM.Ephem)) == 0);
                }
                else if ((f & MEM.Ephem) != 0)
                {
                    c = 'e';
                    Debug.Assert((f & (MEM.Static | MEM.Dyn)) == 0);
                }
                else
                    c = 's';

                buf.Append(c);
                buf.AppendFormat("{0}[", mem.N);
                int i; for (i = 0; i < 16 && i < mem.N; i++)
                    buf.AppendFormat("{0:%02X}", ((int)mem.Z[i] & 0xFF));
                for (i = 0; i < 16 && i < mem.N; i++)
                {
                    char z = (char)mem.Z[i];
                    if (z < 32 || z > 126) buf.Append('.');
                    else buf.Append(z);
                }
                buf.AppendFormat("]{0}", _encnames[(byte)mem.Encode]);
                if ((f & MEM.Zero) != 0)
                    buf.AppendFormat("+{0}z", mem.u.Zeros);
            }
            else if ((f & MEM.Str) != 0)
            {
                buf.Append(' ');
                if ((f & MEM.Dyn) != 0)
                {
                    buf.Append('z');
                    Debug.Assert((f & (MEM.Static | MEM.Ephem)) == 0);
                }
                else if ((f & MEM.Static) != 0)
                {
                    buf.Append('t');
                    Debug.Assert((f & (MEM.Dyn | MEM.Ephem)) == 0);
                }
                else if ((f & MEM.Ephem) != 0)
                {
                    buf.Append('s');
                    Debug.Assert((f & (MEM.Static | MEM.Dyn)) == 0);
                }
                else
                    buf.Append('s');

                buf.AppendFormat("{0}", mem.N);
                buf.Append('[');
                for (int j = 0; j < 15 && j < mem.N; j++)
                {
                    byte c = (byte)mem.Z[j];
                    buf.Append(c >= 0x20 && c < 0x7f ? (char)c : '.');
                }
                buf.Append(']');
                buf.Append(_encnames[(byte)mem.Encode]);
            }
        }

        static void MemTracePrint(FILE _out, Mem p)
        {
            if ((p.Flags & MEM.Null) != 0) _out.Write(" NULL");
            else if ((p.Flags & (MEM.Int | MEM.Str)) == (MEM.Int | MEM.Str)) _out.Write(" si:%lld", p.u.I);
#if !OMIT_FLOATING_POINT
            else if ((p.Flags & MEM.Int) != 0) _out.Write(" i:%lld", p.u.I);
#endif
            else if ((p.Flags & MEM.Real) != 0) _out.Write(" r:%g", p.R);
            else if ((p.Flags & MEM.RowSet) != 0) _out.Write(" (rowset)");
            else
            {
                StringBuilder buf = new StringBuilder(200);
                MemPrettyPrint(p, buf);
                _out.Write(" ");
                _out.Write("%s", buf);
            }
        }
        static void RegisterTrace(FILE _out, int iReg, Mem p)
        {
            _out.Write("reg[%d] = ", iReg);
            MemTracePrint(_out, p);
            _out.Write("\n");
        }

        static void REGISTER_TRACE(Vdbe p, int R, Mem M) { if (p.Trace != null) RegisterTrace(p.Trace, R, M); }
#else
        static void REGISTER_TRACE(Vdbe p, int R, Mem M) { }
#endif

#if !NDEBUG
        static int CheckSavepointCount(Context db)
        {
            int n = 0;
            for (Savepoint p = db.Savepoints; p != null; p = p.Next) n++;
            Debug.Assert((db.SavepointsLength + db.IsTransactionSavepoint) == n);
            return 1;
        }
#else
        static int CheckSavepointCount(Context db) { return 1; }
#endif

        static void ImportVtabErrMsg(Vdbe p, IVTable vtab)
        {
            Context db = p.Ctx;
            C._tagfree(db, ref p.ErrMsg);
            p.ErrMsg = vtab.ErrMsg;
            C._free(ref vtab.ErrMsg);
            vtab.ErrMsg = null;
        }

        #endregion

        #region Main

        public RC Exec()
        {
            VdbeOp[] ops = Ops.data; // Copy of p.aOp
            VdbeOp op; // Current operation
            RC rc = RC.OK; // Value to return
            Context ctx = Ctx; // The database
            byte resetSchemaOnFault = 0; // Reset schema after an error if positive
            TEXTENCODE encoding = E.CTXENCODE(ctx); // The database encoding
            Mem[] mems = Mems.data; // Copy of p.mems
            Mem in1 = null; // 1st input operand
            Mem in2 = null; // 2nd input operand
            Mem in3 = null; // 3rd input operand
            Mem out_ = null; // Output operand
            int compare = 0;  // Result of last OP_Compare operation
            int[] permutes = null; // Permutation of columns for OP_Compare
            long lastRowid = ctx.LastRowID; // Saved value of the last insert ROWID

            //// INSERT STACK UNION HERE ////
            Debug.Assert(Magic == VDBE_MAGIC_RUN); // sqlite3_step() verifies this
            Enter();
            if (RC_ == RC.NOMEM)
                goto no_mem; // This happens if a malloc() inside a call to sqlite3_column_text() or sqlite3_column_text16() failed.
            Debug.Assert(RC_ == RC.OK || RC_ == RC.BUSY);
            RC_ = RC.OK;
            Debug.Assert(HasExplain == 0);
            ResultSet = null;
            ctx.BusyHandler.Busys = 0;
            if (ctx.u1.IsInterrupted) goto abort_due_to_interrupt; //CHECK_FOR_INTERRUPT;
#if !OMIT_TRACE && ENABLE_IOTRACE
            IOTraceSql();
#endif
#if !OMIT_PROGRESS_CALLBACK
            bool checkProgress = (ctx.Progress != null); // True if progress callbacks are enabled
            int progressOps = 0; // Opcodes executed since progress callback.
#endif
#if DEBUG
            C._benignalloc_begin();
            if (PC == 0 && (ctx.Flags & Context.FLAG.VdbeListing) != 0)
            {
                Console.Write("VDBE Program Listing:\n");
                PrintSql();
                for (int i = 0; i < Ops.length; i++)
                    PrintOp(Console.Out, i, Ops[i]);
            }
            C._benignalloc_end();
#endif
            int pc = 0; // The program counter
            for (pc = PC; rc == RC.OK; pc++)
            {
                Debug.Assert(pc >= 0 && pc < Ops.length);
                if (ctx.MallocFailed) goto no_mem;
#if VDBE_PROFILE
                int origPc = pc; // Program counter at start of opcode
                ulong start = _hwtime(); // CPU clock count at start of opcode
#endif
                op = ops[pc];

#if DEBUG
                // Only allow tracing if SQLITE_DEBUG is defined.
                if (Trace != null)
                {
                    if (pc == 0)
                    {
                        Console.Write("VDBE Execution Trace:\n");
                        PrintSql();
                    }
                    PrintOp(Trace, pc, op);
                }
#endif

#if TEST
                // Check to see if we need to simulate an interrupt.  This only happens if we have a special test build.
                if (g_interrupt_count > 0)
                {
                    g_interrupt_count--;
                    if (g_interrupt_count == 0)
                        Main.Interrupt(ctx);
                }
#endif

#if !OMIT_PROGRESS_CALLBACK
                // Call the progress callback if it is configured and the required number of VDBE ops have been executed (either since this invocation of
                // sqlite3VdbeExec() or since last time the progress callback was called). If the progress callback returns non-zero, exit the virtual machine with
                // a return code SQLITE_ABORT.
                if (checkProgress)
                {
                    if (ctx.ProgressOps == progressOps)
                    {
                        int prc = ctx.Progress(ctx.ProgressArg);
                        if (prc != 0)
                        {
                            rc = RC.INTERRUPT;
                            goto vdbe_error_halt;
                        }
                        progressOps = 0;
                    }
                    progressOps++;
                }
#endif

                // On any opcode with the "out2-prerelease" tag, free any external allocations out of mem[p2] and set mem[p2] to be
                // an undefined integer.  Opcodes will either fill in the integer value or convert mem[p2] to a different type.
                Debug.Assert(op.Opflags == g_opcodeProperty[op.Opcode]);
                if ((op.Opflags & OPFLG.OUT2_PRERELEASE) != 0)
                {
                    Debug.Assert(op.P2 > 0);
                    Debug.Assert(op.P2 <= Mems.length);
                    out_ = mems[op.P2];
                    MemAboutToChange(this, out_);
                    MemRelease(out_);
                    out_.Flags = MEM.Int;
                }

#if DEBUG
                // Sanity checking on other operands
                if ((op.Opflags & OPFLG.IN1) != 0)
                {
                    Debug.Assert(op.P1 > 0);
                    Debug.Assert(op.P1 <= Mems.length);
                    Debug.Assert(E.MemIsValid(mems[op.P1]));
                    REGISTER_TRACE(this, op.P1, mems[op.P1]);
                }
                if ((op.Opflags & OPFLG.IN2) != 0)
                {
                    Debug.Assert(op.P2 > 0);
                    Debug.Assert(op.P2 <= Mems.length);
                    Debug.Assert(E.MemIsValid(mems[op.P2]));
                    REGISTER_TRACE(this, op.P2, mems[op.P2]);
                }
                if ((op.Opflags & OPFLG.IN3) != 0)
                {
                    Debug.Assert(op.P3 > 0);
                    Debug.Assert(op.P3 <= Mems.length);
                    Debug.Assert(E.MemIsValid(mems[op.P3]));
                    REGISTER_TRACE(this, op.P3, mems[op.P3]);
                }
                if ((op.Opflags & OPFLG.OUT2) != 0)
                {
                    Debug.Assert(op.P2 > 0);
                    Debug.Assert(op.P2 <= Mems.length);
                    MemAboutToChange(this, mems[op.P2]);
                }
                if ((op.Opflags & OPFLG.OUT3) != 0)
                {
                    Debug.Assert(op.P3 > 0);
                    Debug.Assert(op.P3 <= Mems.length);
                    MemAboutToChange(this, mems[op.P3]);
                }
#endif

                // What follows is a massive switch statement where each case implements a separate instruction in the virtual machine.  If we follow the usual
                // indentation conventions, each case should be indented by 6 spaces.  But that is a lot of wasted space on the left margin.  So the code within
                // the switch statement will break with convention and be flush-left. Another big comment (similar to this one) will mark the point in the code where
                // we transition back to normal indentation.
                //
                // The formatting of each case is important.  The makefile for SQLite generates two C files "opcodes.h" and "opcodes.c" by scanning this
                // file looking for lines that begin with "case OP_".  The opcodes.h files will be filled with #defines that give unique integer values to each
                // opcode and the opcodes.c file is filled with an array of strings where each string is the symbolic name for the corresponding opcode.  If the
                // case statement is followed by a comment of the form "/# same as ... #/" that comment is used to determine the particular value of the opcode.
                //
                // Other keywords in the comment that follows each case are used to construct the OPFLG_INITIALIZER value that initializes opcodeProperty[].
                // Keywords include: in1, in2, in3, out2_prerelease, out2, out3.  See the mkopcodeh.awk script for additional information.
                //
                // Documentation about VDBE opcodes is generated by scanning this file for lines of that contain "Opcode:".  That line and all subsequent
                // comment lines are used in the generation of the opcode.html documentation file.
                //
                // SUMMARY:
                //
                //     Formatting is important to scripts that scan this file.
                //     Do not deviate from the formatting style currently in use.
                switch (op.Opcode)
                {
                    case OP.Goto: // jump
                        {
                            // Opcode:  Goto * P2 * * *
                            //
                            // An unconditional jump to address P2. The next instruction executed will be 
                            // the one at index P2 from the beginning of the program.
                            if (ctx.u1.IsInterrupted) goto abort_due_to_interrupt; //CHECK_FOR_INTERRUPT;
                            pc = op.P2 - 1;
                            break;
                        }
                    case OP.Gosub: // jump
                        {
                            // Opcode:  Gosub P1 P2 * * *
                            //
                            // Write the current address onto register P1 and then jump to address P2.
                            in1 = mems[op.P1];
                            Debug.Assert((in1.Flags & MEM.Dyn) == 0);
                            MemAboutToChange(this, in1);
                            in1.Flags = MEM.Int;
                            in1.u.I = pc;
                            REGISTER_TRACE(this, op.P1, in1);
                            pc = op.P2 - 1;
                            break;
                        }
                    case OP.Return:  // in1
                        {
                            // Opcode:  Return P1 * * * *
                            //
                            // Jump to the next instruction after the address in register P1.
                            in1 = mems[op.P1];
                            Debug.Assert((in1.Flags & MEM.Int) != 0);
                            pc = (int)in1.u.I;
                            break;
                        }
                    case OP.Yield: // in1
                        {
                            // Opcode:  Yield P1 * * * *
                            //
                            // Swap the program counter with the value in register P1.
                            in1 = mems[op.P1];
                            Debug.Assert((in1.Flags & MEM.Dyn) == 0);
                            in1.Flags = MEM.Int;
                            int pcDest = (int)in1.u.I;
                            in1.u.I = pc;
                            REGISTER_TRACE(this, op.P1, in1);
                            pc = pcDest;
                            break;
                        }
                    case OP.HaltIfNull: // in3
                        {
                            // Opcode:  HaltIfNull  P1 P2 P3 P4 *
                            //
                            // Check the value in register P3.  If it is NULL then Halt using parameter P1, P2, and P4 as if this were a Halt instruction.  If the
                            // value in register P3 is not NULL, then this routine is a no-op.
                            in3 = mems[op.P3];
                            if ((in3.Flags & MEM.Null) == 0) break;
                            goto case OP.Halt;
                        }
                    // Fall through into OP_Halt
                    case OP.Halt:
                        {
                            // Opcode:  Halt P1 P2 * P4 *
                            //
                            // Exit immediately.  All open cursors, etc are closed automatically.
                            //
                            // P1 is the result code returned by sqlite3_exec(), sqlite3_reset(), or sqlite3_finalize().  For a normal halt, this should be SQLITE_OK (0).
                            // For errors, it can be some other value.  If P1!=0 then P2 will determine whether or not to rollback the current transaction.  Do not rollback
                            // if P2==OE_Fail. Do the rollback if P2==OE_Rollback.  If P2==OE_Abort, then back out all changes that have occurred during this execution of the
                            // VDBE, but do not rollback the transaction. 
                            //
                            // If P4 is not null then it is an error message string.
                            //
                            // There is an implied "Halt 0 0 0" instruction inserted at the very end of every program.  So a jump past the last instruction of the program
                            // is the same as executing Halt.
                            if (op.P1 == (int)RC.OK && Frames != null)
                            {
                                in3 = mems[op.P3];
                                // Halt the sub-program. Return control to the parent frame.
                                VdbeFrame frame = Frames;
                                Frames = frame.Parent;
                                FramesLength--;
                                SetChanges(ctx, Changes);
                                pc = FrameRestore(frame);
                                lastRowid = ctx.LastRowID;
                                if (op.P2 == (int)OE.Ignore)
                                {
                                    // Instruction pc is the OP_Program that invoked the sub-program currently being halted. If the p2 instruction of this OP_Halt
                                    // instruction is set to OE_Ignore, then the sub-program is throwing an IGNORE exception. In this case jump to the address specified
                                    // as the p2 of the calling OP_Program.
                                    pc = Ops[pc].P2 - 1;
                                }
                                ops = Ops.data;
                                mems = Mems.data;
                                break;
                            }

                            RC_ = (RC)op.P1;
                            ErrorAction = (OE)op.P2;
                            PC = pc;
                            if (op.P4.Z != null)
                            {
                                Debug.Assert(RC_ != RC.OK);
                                C._setstring(ref ErrMsg, ctx, "%s", op.P4.Z);
                                C.ASSERTCOVERAGE(SysEx._GlobalStatics.Log != null);
                                SysEx.LOG((RC)op.P1, "abort at %d in [%s]: %s", pc, Sql_, op.P4.Z);
                            }
                            else if (RC_ != 0)
                            {
                                C.ASSERTCOVERAGE(SysEx._GlobalStatics.Log != null);
                                SysEx.LOG((RC)op.P1, "constraint failed at %d in [%s]", pc, Sql_);
                            }
                            rc = Halt();
                            Debug.Assert(rc == RC.BUSY || rc == RC.OK || rc == RC.ERROR);
                            if (rc == RC.BUSY)
                                RC_ = rc = RC.BUSY;
                            else
                            {
                                Debug.Assert(rc == RC.OK || RC_ == RC.CONSTRAINT);
                                Debug.Assert(rc == RC.OK || ctx.DeferredCons > 0);
                                rc = (RC_ != 0 ? RC.ERROR : RC.DONE);
                            }
                            goto vdbe_return;
                        }
                    case OP.Integer: // out2-prerelease
                        {
                            // Opcode: Integer P1 P2 * * *
                            //
                            // The 32-bit integer value P1 is written into register P2.
                            out_.u.I = op.P1;
                            break;
                        }
                    case OP.Int64: // out2-prerelease
                        {
                            // Opcode: Int64 * P2 * P4 *
                            //
                            // P4 is a pointer to a 64-bit integer value. Write that value into register P2.
                            Debug.Assert(op.P4.I64 != 0);
                            out_.u.I = op.P4.I64;
                            break;
                        }
#if !OMIT_FLOATING_POINT
                    case OP.Real: // same as TK_FLOAT, out2-prerelease
                        {
                            // Opcode: Real * P2 * P4 *
                            //
                            // P4 is a pointer to a 64-bit floating point value. Write that value into register P2.
                            out_.Flags = MEM.Real;
                            Debug.Assert(!double.IsNaN(op.P4.Real));
                            out_.R = op.P4.Real;
                            break;
                        }
#endif
                    case OP.String8:// same as TK_STRING, out2-prerelease
                        {
                            // Opcode: String8 * P2 * P4 *
                            //
                            // P4 points to a nul terminated UTF-8 string. This opcode is transformed into an OP_String before it is executed for the first time.
                            Debug.Assert(op.P4.Z != null);
                            op.Opcode = OP.String;
                            op.P1 = op.P4.Z.Length;
#if !OMIT_UTF16
                            if (encoding != TEXTENCODE.UTF8)
                            {
                                rc = MemSetStr(out_, op.P4.Z, -1, TEXTENCODE.UTF8, C.DESTRUCTOR_STATIC);
                                if (rc == RC.TOOBIG) goto too_big;
                                if (ChangeEncoding(out_, encoding) != RC.OK) goto no_mem;
                                Debug.Assert(out_.Malloc == out_.Z);
                                Debug.Assert((out_.Flags & MEM.Dyn) != 0);
                                out_.Malloc = null;
                                out_.Flags |= MEM.Static;
                                out_.Flags &= ~MEM.Dyn;
                                if (op.P4Type == Vdbe.P4T.DYNAMIC)
                                    C._tagfree(ctx, ref op.P4.Z);
                                op.P4Type = P4T.DYNAMIC;
                                op.P4.Z = out_.Z;
                                op.P1 = out_.N;
                            }
#endif
                            if (op.P1 > ctx.Limits[(int)LIMIT.LENGTH])
                                goto too_big;
                            goto case OP.String;
                        }
                    // Fall through to the next case, OP_String
                    case OP.String: // out2-prerelease
                        {
                            // Opcode: String P1 P2 * P4 *
                            //
                            // The string value P4 of length P1 (bytes) is stored in register P2.
                            C._free(ref out_.Z_);
                            Debug.Assert(op.P4.Z != null);
                            out_.Flags = MEM.Str | MEM.Static | MEM.Term;
                            out_.Z = op.P4.Z;
                            out_.N = op.P1;
                            out_.Encode = encoding;
                            UPDATE_MAX_BLOBSIZE(out_);
                            break;
                        }
                    case OP.Null: // out2-prerelease
                        {
                            // Opcode: Null P1 P2 P3 * *
                            //
                            // Write a NULL into registers P2.  If P3 greater than P2, then also write NULL into register P3 and every register in between P2 and P3.  If P3
                            // is less than P2 (typically P3 is zero) then only register P2 is set to NULL.
                            //
                            // If the P1 value is non-zero, then also set the MEM_Cleared flag so that NULL values will not compare equal even if SQLITE_NULLEQ is set on OP_Ne or OP_Eq.
                            int cnt = op.P3 - op.P2;
                            Debug.Assert(op.P3 <= Mems.length);
                            MEM nullFlag;
                            out_.Flags = nullFlag = (op.P1 != 0 ? (MEM.Null | MEM.Cleared) : MEM.Null);
                            while (cnt > 0)
                            {
                                out_++;
                                MemAboutToChange(this, out_);
                                MemRelease(out_);
                                out_.Flags = nullFlag;
                                cnt--;
                            }
                            break;
                        }
                    case OP.Blob:  // out2-prerelease
                        {
                            // Opcode: Blob P1 P2 * P4
                            //
                            // P4 points to a blob of data P1 bytes long.  Store this blob in register P2.
                            Debug.Assert(op.P1 <= CORE_MAX_LENGTH);
                            MemSetStr(out_, op.P4.Z, op.P1, 0, null);
                            out_.Encode = encoding;
                            UPDATE_MAX_BLOBSIZE(out_);
                            break;
                        }
                    case OP.Variable: // out2-prerelease
                        {
                            // Opcode: Variable P1 P2 * P4 *
                            //
                            // Transfer the values of bound parameter P1 into register P2
                            //
                            // If the parameter is named, then its name appears in P4 and P3==1. The P4 value is used by sqlite3_bind_parameter_name().
                            Debug.Assert(op.P1 >= 0 && op.P1 <= Vars.length);
                            Debug.Assert(op.P4.Z == null || op.P4.Z == p.VarNames[op.P1 - 1]);
                            Mem var = Vars[op.P1 - 1]; // Value being transferred
                            if (MemTooBig(var))
                                goto too_big;
                            MemShallowCopy(out_, var, MEM.Static);
                            UPDATE_MAX_BLOBSIZE(out_);
                            break;
                        }
                    // Opcode: Move P1 P2 P3 * *
                    //
                    // Move the values in register P1..P1+P3 over into registers P2..P2+P3.  Registers P1..P1+P3 are
                    // left holding a NULL.  It is an error for register ranges P1..P1+P3 and P2..P2+P3 to overlap.
                    case OP.Move:
                        {
                            int n = op.P3; // Number of registers left to copy
                            int p1 = op.P1; // Register to copy from
                            int p2 = op.P2; // Register to copy to
                            Debug.Assert(n > 0 && p1 > 0 && p2 > 0);
                            Debug.Assert(p1 + n <= p2 || p2 + n <= p1);

                            in1 = mems[op.P1];
                            out_ = mems[op.P2];
                            while (n-- != 0)
                            {
                                in1 = mems[p1 + op.P3 - n - 1];
                                out_ = mems[p2];
                                //Debug.Assert(out_ <= mems[Mems.length]);
                                //Debug.Assert(in1 <= mems[Mems.length]);
                                Debug.Assert(E.MemIsValid(in1));
                                MemAboutToChange(this, out_);
                                //byte[] malloc = out_.Malloc; // Holding variable for allocated memory
                                //out_.Malloc = null;
                                MemMove(out_, in1);
#if DEBUG
                                //if (out_.ScopyFrom >= mems[p1] && out_.ScopyFrom < mems[p1 + op.P3])
                                //    out_.ScopyFrom += p1 - op.P2;
#endif
                                //in1.Malloc = malloc;
                                REGISTER_TRACE(p2++, out_);
                            }
                            break;
                        }
                    case OP.Copy:
                        {
                            // Opcode: Copy P1 P2 P3 * *
                            //
                            // Make a copy of registers P1..P1+P3 into registers P2..P2+P3.
                            //
                            // This instruction makes a deep copy of the value.  A duplicate is made of any string or blob constant.  See also OP_SCopy.
                            int n = op.P3;
                            in1 = mems[op.P1];
                            out_ = mems[op.P2];
                            Debug.Assert(out_ != in1);
                            int x = 0; // C#
                            while (true)
                            {
                                in1 = mems[op.P1 + x];
                                out_ = mems[op.P2 + x];
                                MemShallowCopy(out_, in1, MEM.Ephem);
                                Deephemeralize(out_);
#if DEBUG
                                out_.ScopyFrom = null;
#endif
                                REGISTER_TRACE(op.P2 + op.P3 - n, out_);
                                if ((n--) == 0) break;
                                x++; // C#
                            }
                            break;
                        }
                    case OP.SCopy:  // in1, out2
                        {
                            // Opcode: SCopy P1 P2 * * *
                            //
                            // Make a shallow copy of register P1 into register P2.
                            //
                            // This instruction makes a shallow copy of the value.  If the value is a string or blob, then the copy is only a pointer to the
                            // original and hence if the original changes so will the copy. Worse, if the original is deallocated, the copy becomes invalid.
                            // Thus the program must guarantee that the original will not change during the lifetime of the copy.  Use OP_Copy to make a complete copy.
                            in1 = mems[op.P1];
                            out_ = mems[op.P2];
                            Debug.Assert(out_ != in1);
                            MemShallowCopy(out_, in1, MEM.Ephem);
#if DEBUG
                            if (out_.ScopyFrom == null) out_.ScopyFrom = in1;
#endif
                            REGISTER_TRACE(op.P2, out_);
                            break;
                        }
                    case OP.ResultRow:
                        {
                            // Opcode: ResultRow P1 P2 * * *
                            //
                            // The registers P1 through P1+P2-1 contain a single row of results. This opcode causes the sqlite3_step() call to terminate
                            // with an SQLITE_ROW return code and it sets up the sqlite3_stmt structure to provide access to the top P1 values as the result row.
                            Debug.Assert(ResColumns == op.P2);
                            Debug.Assert(op.P1 > 0);
                            Debug.Assert(op.P1 + op.P2 <= Mems.length + 1);

                            // If this statement has violated immediate foreign key constraints, do not return the number of rows modified. And do not RELEASE the statement
                            // transaction. It needs to be rolled back.
                            if ((rc = CheckFk(false)) != RC.OK)
                            {
                                Debug.Assert((ctx.Flags & Context.FLAG.CountRows) != 0);
                                Debug.Assert(UsesStmtJournal);
                                break;
                            }

                            // If the SQLITE_CountRows flag is set in sqlite3.flags mask, then DML statements invoke this opcode to return the number of rows 
                            // modified to the user. This is the only way that a VM that opens a statement transaction may invoke this opcode.
                            //
                            // In case this is such a statement, close any statement transaction opened by this VM before returning control to the user. This is to
                            // ensure that statement-transactions are always nested, not overlapping. If the open statement-transaction is not closed here, then the user
                            // may step another VM that opens its own statement transaction. This may lead to overlapping statement transactions.
                            //
                            // The statement transaction is never a top-level transaction.  Hence the RELEASE call below can never fail.
                            Debug.Assert(StatementID == 0 || (ctx.Flags & Context.FLAG.CountRows) != 0);
                            rc = CloseStatement(IPager.SAVEPOINT.RELEASE);
                            if (C._NEVER(rc != RC.OK))
                                break;

                            // Invalidate all ephemeral cursor row caches
                            CacheCtr = (CacheCtr + 2) | 1;

                            // Make sure the results of the current row are \000 terminated and have an assigned type.  The results are de-ephemeralized as a side effect.
                            //Mem[] mems2 = ResultSet = mems[op.P1];
                            ResultSet = new Mem[op.P2];
                            for (int i = 0; i < op.P2; i++)
                            {
                                ResultSet[i] = mems[op.P1 + i];
                                Debug.Assert(E.MemIsValid(ResultSet[i]));
                                Deephemeralize(ResultSet[i]);
                                Debug.Assert((ResultSet[i].Flags & MEM.Ephem) == 0 || (ResultSet[i].Flags & (MEM.Str | MEM.Blob)) == 0);
                                MemNulTerminate(ResultSet[i]);
                                MemStoreType(ResultSet[i]);
                                REGISTER_TRACE(op.P1 + i, ResultSet[i]);
                            }
                            if (ctx.MallocFailed) goto no_mem;

                            // Return SQLITE_ROW
                            PC = pc + 1;
                            rc = RC.ROW;
                            goto vdbe_return;
                        }
                    case OP.Concat: // same as TK_CONCAT, in1, in2, out3
                        {
                            // Opcode: Concat P1 P2 P3 * *
                            //
                            // Add the text in register P1 onto the end of the text in register P2 and store the result in register P3.
                            // If either the P1 or P2 text are NULL then store NULL in P3.
                            //
                            //   P3 = P2 || P1
                            //
                            // It is illegal for P1 and P3 to be the same register. Sometimes, if P3 is the same register as P2, the implementation is able to avoid a memcpy().
                            in1 = mems[op.P1];
                            in2 = mems[op.P2];
                            out_ = mems[op.P3];
                            Debug.Assert(in1 != out_);
                            if (((in1.Flags | in2.Flags) & MEM.Null) != 0)
                            {
                                MemSetNull(out_);
                                break;
                            }
                            if (E.ExpandBlob(in1) != 0 || E.ExpandBlob(in2) != 0) goto no_mem;
                            if (((in1.Flags & (MEM.Str | MEM.Blob)) == 0) && MemStringify(in1, encoding) != 0) goto no_mem; // Stringify(in1, encoding);
                            if (((in2.Flags & (MEM.Str | MEM.Blob)) == 0) && MemStringify(in2, encoding) != 0) goto no_mem; // Stringify(in2, encoding);
                            long bytes = in1.N + in2.N;
                            if (bytes > ctx.Limits[(int)LIMIT.LENGTH])
                                goto too_big;
                            E.MemSetTypeFlag(out_, MEM.Str);
                            //:if (MemGrow(out_, (int)bytes + 2, out_ == in2))
                            //:    goto no_mem;
                            //:if (out_ != in2)
                            //:    _memcpy(out_.Z, in2.Z, in2.N);
                            //:_memcpy(out_.Z[in2.N], in1.Z, in1.N);
                            if (in2.Z != null && in2.Z.Length >= in2.N)
                                if (in1.Z != null)
                                    out_.Z = in2.Z.Substring(0, in2.N) + (in1.N < in1.Z.Length ? in1.Z.Substring(0, in1.N) : in1.Z);
                                else
                                {
                                    if ((in1.Flags & MEM.Blob) == 0) // String as Blob
                                    {
                                        StringBuilder sb = new StringBuilder(in1.N);
                                        for (int i = 0; i < in1.N; i++)
                                            sb.Append((byte)in1.Z_[i]);
                                        out_.Z = in2.Z.Substring(0, in2.N) + sb.ToString();
                                    }
                                    else // UTF-8 Blob
                                        out_.Z = in2.Z.Substring(0, in2.N) + Encoding.UTF8.GetString(in1.Z_, 0, in1.Z_.Length);
                                }
                            else
                            {
                                out_.Z_ = C._alloc(in1.N + in2.N);
                                Buffer.BlockCopy(in2.Z_, 0, out_.Z_, 0, in2.N);
                                if (in1.Z_ != null)
                                    Buffer.BlockCopy(in1.Z_, 0, out_.Z_, in2.N, in1.N);
                                else
                                    for (int i = 0; i < in1.N; i++)
                                        out_.Z_[in2.N + i] = (byte)in1.Z[i];
                            }
                            //out_.Z[byte] = 0;
                            //out_.Z[byte + 1] = 0;
                            out_.Flags |= MEM.Term;
                            out_.N = (int)bytes;
                            out_.Encode = encoding;
                            UPDATE_MAX_BLOBSIZE(out_);
                            break;
                        }
                    case OP.Add: // same as TK_PLUS, in1, in2, ref3
                    case OP.Subtract: // same as TK_MINUS, in1, in2, ref3
                    case OP.Multiply: // same as TK_STAR, in1, in2, ref3
                    case OP.Divide: // same as TK_SLASH, in1, in2, ref3
                    case OP.Remainder: // same as TK_REM, in1, in2, ref3
                        {
                            // Opcode: Add P1 P2 P3 * *
                            //
                            // Add the value in register P1 to the value in register P2 and store the result in register P3.
                            // If either input is NULL, the result is NULL.
                            //
                            // Opcode: Multiply P1 P2 P3 * *
                            //
                            //
                            // Multiply the value in register P1 by the value in register P2 and store the result in register P3.
                            // If either input is NULL, the result is NULL.
                            //
                            // Opcode: Subtract P1 P2 P3 * *
                            //
                            // Subtract the value in register P1 from the value in register P2 and store the result in register P3.
                            // If either input is NULL, the result is NULL.
                            //
                            // Opcode: Divide P1 P2 P3 * *
                            //
                            // Divide the value in register P1 by the value in register P2 and store the result in register P3 (P3=P2/P1). If the value in 
                            // register P1 is zero, then the result is NULL. If either input is NULL, the result is NULL.
                            //
                            // Opcode: Remainder P1 P2 P3 * *
                            //
                            // Compute the remainder after integer division of the value in register P1 by the value in register P2 and store the result in P3.
                            // If the value in register P2 is zero the result is NULL. If either operand is NULL, the result is NULL.
                            bool intint; // Started out as two integer operands
                            long iA; // Integer value of left operand
                            long iB = 0; // Integer value of right operand
                            double rA; // Real value of left operand
                            double rB; // Real value of right operand
                            in1 = mems[op.P1];
                            ApplyNumericAffinity(in1);
                            in2 = mems[op.P2];
                            ApplyNumericAffinity(in2);
                            out_ = mems[op.P3];
                            MEM flags = (in1.Flags | in2.Flags); // Combined MEM_* flags from both inputs
                            if ((flags & MEM.Null) != 0) goto arithmetic_result_is_null;
                            bool fp_math = false;
                            if ((in1.Flags & in2.Flags & MEM.Int) == MEM.Int)
                            {
                                iA = in1.u.I;
                                iB = in2.u.I;
                                intint = true;
                                switch (op.Opcode)
                                {
                                    case OP.Add: if (MathEx.Add(ref iB, iA)) fp_math = true; break; // goto fp_math
                                    case OP.Subtract: if (MathEx.Sub(ref iB, iA)) fp_math = true; break; // goto fp_math
                                    case OP.Multiply: if (MathEx.Mul(ref iB, iA)) fp_math = true; break; // goto fp_math
                                    case OP.Divide:
                                        {
                                            if (iA == 0) goto arithmetic_result_is_null;
                                            if (iA == -1 && iB == long.MinValue) { fp_math = true; break; } // goto fp_math
                                            iB /= iA;
                                            break;
                                        }
                                    default:
                                        {
                                            if (iA == 0) goto arithmetic_result_is_null;
                                            if (iA == -1) iA = 1;
                                            iB %= iA;
                                            break;
                                        }
                                }
                            }
                            if (!fp_math)
                            {
                                out_.u.I = iB;
                                E.MemSetTypeFlag(out_, MEM.Int);
                            }
                            else
                            {
                                //fp_math:
                                rA = Vdbe.RealValue(in1);
                                rB = Vdbe.RealValue(in2);
                                switch (op.Opcode)
                                {
                                    case OP.Add: rB += rA; break;
                                    case OP.Subtract: rB -= rA; break;
                                    case OP.Multiply: rB *= rA; break;
                                    case OP.Divide:
                                        {
                                            // (double)0 In case of SQLITE_OMIT_FLOATING_POINT...
                                            if (rA == (double)0) goto arithmetic_result_is_null;
                                            rB /= rA;
                                            break;
                                        }
                                    default:
                                        {
                                            iA = (long)rA;
                                            iB = (long)rB;
                                            if (iA == 0) goto arithmetic_result_is_null;
                                            if (iA == -1) iA = 1;
                                            rB = (double)(iB % iA);
                                            break;
                                        }
                                }
#if OMIT_FLOATING_POINT
                                out_->u.I = rB;
                                MemSetTypeFlag(out_, MEM.Int);
#else
                                if (double.IsNaN(rB))
                                    goto arithmetic_result_is_null;
                                out_.R = rB;
                                E.MemSetTypeFlag(out_, MEM.Real);
                                if ((flags & MEM.Real) == 0)
                                    IntegerAffinity(out_);
#endif
                            }
                            break;
                        arithmetic_result_is_null:
                            MemSetNull(out_);
                            break;
                        }
                    case OP.CollSeq:
                        {
                            // Opcode: CollSeq P1 * * P4
                            //
                            // P4 is a pointer to a CollSeq struct. If the next call to a user function or aggregate calls sqlite3GetFuncCollSeq(), this collation sequence will
                            // be returned. This is used by the built-in min(), max() and nullif() functions.
                            //
                            // If P1 is not zero, then it is a register that a subsequent min() or max() aggregate will set to 1 if the current row is not the minimum or
                            // maximum.  The P1 register is initialized to 0 by this instruction.
                            //
                            // The interface used by the implementation of the aforementioned functions to retrieve the collation sequence set by this opcode is not available
                            // publicly, only to user functions defined in func.c.
                            Debug.Assert(op.P4Type == Vdbe.P4T.COLLSEQ);
                            if (op.P1 != 0)
                                MemSetInt64(mems[op.P1], 0);
                            break;
                        }
                    case OP.Function:
                        {
                            // Opcode: Function P1 P2 P3 P4 P5
                            //
                            // Invoke a user function (P4 is a pointer to a Function structure that defines the function) with P5 arguments taken from register P2 and
                            // successors.  The result of the function is stored in register P3. Register P3 must not be one of the function inputs.
                            //
                            // P1 is a 32-bit bitmask indicating whether or not each argument to the function was determined to be constant at compile time. If the first
                            // argument was constant then bit 0 of P1 is set. This is used to determine whether meta data associated with a user function argument using the
                            // sqlite3_set_auxdata() API may be safely retained until the next invocation of this opcode.
                            //
                            // See also: AggStep and AggFinal
                            int n = op.P5;

                            Mem[] vals = Args;
                            Debug.Assert(vals != null || n == 0);
                            Debug.Assert(op.P3 > 0 && op.P3 <= Mems.length);
                            out_ = mems[op.P3];
                            MemAboutToChange(this, out_);

                            Debug.Assert(n == 0 || (op.P2 > 0 && op.P2 + n <= Mems.length + 1));
                            Debug.Assert(op.P3 < op.P2 || op.P3 >= op.P2 + n);
                            Mem arg;
                            for (int i = 0; i < n; i++)
                            {
                                arg = mems[op.P2 + i];
                                Debug.Assert(E.MemIsValid(arg));
                                vals[i] = arg;
                                Deephemeralize(arg);
                                MemStoreType(arg);
                                REGISTER_TRACE(op.P2 + i, arg);
                            }

                            Debug.Assert(op.P4Type == Vdbe.P4T.FUNCDEF || op.P4Type == Vdbe.P4T.VDBEFUNC);
                            FuncContext fctx = new FuncContext();
                            if (op.P4Type == Vdbe.P4T.FUNCDEF)
                            {
                                fctx.Func = op.P4.Func;
                                fctx.VdbeFunc = null;
                            }
                            else
                            {
                                fctx.VdbeFunc = (VdbeFunc)op.P4.VdbeFunc;
                                fctx.Func = fctx.VdbeFunc.Func;
                            }

                            fctx.S.Flags = MEM.Null;
                            fctx.S.Ctx = ctx;
                            fctx.S.Del = null;
                            //: fctx.S.Malloc = null;

                            // The output cell may already have a buffer allocated. Move the pointer to fctx.s so in case the user-function can use
                            // the already allocated buffer instead of allocating a new one.
                            MemMove(fctx.S, out_);
                            E.MemSetTypeFlag(fctx.S, MEM.Null);

                            fctx.IsError = (RC)0;
                            if ((fctx.Func.Flags & FUNC.NEEDCOLL) != 0)
                            {
                                //Debug.Assert(op > ops);
                                Debug.Assert(ops[pc - 1].P4Type == Vdbe.P4T.COLLSEQ);
                                Debug.Assert(ops[pc - 1].Opcode == OP.CollSeq);
                                fctx.Coll = Ops[pc - 1].P4.Coll;
                            }
                            ctx.LastRowID = lastRowid;
                            fctx.Func.Func(fctx, n, vals); // IMP: R-24505-23230
                            lastRowid = ctx.LastRowID;

                            // If any auxiliary data functions have been called by this user function, immediately call the destructor for any non-static values.
                            if (fctx.VdbeFunc != null)
                            {
                                DeleteAuxData(fctx.VdbeFunc, op.P1);
                                op.P4.VdbeFunc = fctx.VdbeFunc;
                                op.P4Type = Vdbe.P4T.VDBEFUNC;
                            }

                            if (ctx.MallocFailed)
                            {
                                // Even though a malloc() has failed, the implementation of the user function may have called an sqlite3_result_XXX() function
                                // to return a value. The following call releases any resources associated with such a value.
                                MemRelease(fctx.S);
                                goto no_mem;
                            }

                            // If the function returned an error, throw an exception
                            if (fctx.IsError != 0)
                            {
                                C._setstring(ref ErrMsg, ctx, Vdbe.Value_Text(fctx.S));
                                rc = fctx.IsError;
                            }

                            // Copy the result of the function into register P3
                            ChangeEncoding(fctx.S, encoding);
                            MemMove(out_, fctx.S);
                            if (MemTooBig(out_))
                                goto too_big;
#if false
                            // The app-defined function has done something that as caused this statement to expire.  (Perhaps the function called sqlite3_exec()
                            // with a CREATE TABLE statement.)
                            if (Expired) rc = RC.ABORT;
#endif
                            REGISTER_TRACE(op.P3, out_);
                            UPDATE_MAX_BLOBSIZE(out_);
                            break;
                        }


                    case OP.BitAnd: // same as TK_BITAND, in1, in2, ref3
                    case OP.BitOr: // same as TK_BITOR, in1, in2, ref3
                    case OP.ShiftLeft: // same as TK_LSHIFT, in1, in2, ref3
                    case OP.ShiftRight: // same as TK_RSHIFT, in1, in2, ref3
                        {
                            // Opcode: BitAnd P1 P2 P3 * *
                            //
                            // Take the bit-wise AND of the values in register P1 and P2 and store the result in register P3.
                            // If either input is NULL, the result is NULL.
                            //
                            // Opcode: BitOr P1 P2 P3 * *
                            //
                            // Take the bit-wise OR of the values in register P1 and P2 and store the result in register P3.
                            // If either input is NULL, the result is NULL.
                            //
                            // Opcode: ShiftLeft P1 P2 P3 * *
                            //
                            // Shift the integer value in register P2 to the left by the number of bits specified by the integer in register P1.
                            // Store the result in register P3. If either input is NULL, the result is NULL.
                            //
                            // Opcode: ShiftRight P1 P2 P3 * *
                            //
                            // Shift the integer value in register P2 to the right by the number of bits specified by the integer in register P1.
                            // Store the result in register P3. If either input is NULL, the result is NULL.
                            in1 = mems[op.P1];
                            in2 = mems[op.P2];
                            out_ = mems[op.P3];
                            if (((in1.Flags | in2.Flags) & MEM.Null) != 0)
                            {
                                MemSetNull(out_);
                                break;
                            }
                            long iA = IntValue(in2);
                            long iB = IntValue(in1);
                            OP op2 = op.Opcode;
                            if (op2 == OP.BitAnd)
                                iA &= iB;
                            else if (op2 == OP.BitOr)
                                iA |= iB;
                            else if (iB != 0)
                            {
                                Debug.Assert(op2 == OP.ShiftRight || op2 == OP.ShiftLeft);

                                // If shifting by a negative amount, shift in the other direction
                                if (iB < 0)
                                {
                                    Debug.Assert(OP.ShiftRight == OP.ShiftLeft + 1);
                                    op2 = (OP)(2 * (int)OP.ShiftLeft + 1 - op2);
                                    iB = (iB > -64 ? -iB : 64);
                                }

                                if (iB >= 64)
                                    iA = (iA >= 0 || op2 == OP.ShiftLeft ? 0 : -1);
                                else
                                {
                                    ulong uA;
                                    if (op2 == OP.ShiftLeft)
                                        iA = iA << (int)iB; //: uA = (ulong)(iA << 0); //: memcpy(uA, iA, sizeof(uA));
                                    else
                                    {
                                        iA = iA >> (int)iB; //: uA = (ulong)(iA << 0); //: memcpy(uA, iA, sizeof(uA));
                                        // Sign-extend on a right shift of a negative number
                                        //:if (iA < 0) uA |= (((0xffffffff) << (byte)32) | 0xffffffff) << (byte)(64 - iB);
                                    }
                                    //: iA = (long)(uA << 0); //: memcpy(iA, uA, sizeof(iA));
                                }
                            }
                            out_.u.I = iA;
                            E.MemSetTypeFlag(out_, MEM.Int);
                            break;
                        }
                    case OP.AddImm: // in1
                        {
                            // Opcode: AddImm  P1 P2 * * *
                            //
                            // Add the constant P2 to the value in register P1. The result is always an integer.
                            //
                            // To force any register to be an integer, just add 0.
                            in1 = mems[op.P1];
                            MemAboutToChange(this, in1);
                            MemIntegerify(in1);
                            in1.u.I += op.P2;
                            break;
                        }
                    case OP.MustBeInt: // jump, in1
                        {
                            // Opcode: MustBeInt P1 P2 * * *
                            // 
                            // Force the value in register P1 to be an integer.  If the value in P1 is not an integer and cannot be converted into an integer
                            // without data loss, then jump immediately to P2, or if P2==0 raise an SQLITE_MISMATCH exception.
                            in1 = mems[op.P1];
                            ApplyAffinity(in1, AFF.NUMERIC, encoding);
                            if ((in1.Flags & MEM.Int) == 0)
                            {
                                if (op.P2 == 0)
                                {
                                    rc = RC.MISMATCH;
                                    goto abort_due_to_error;
                                }
                                else
                                    pc = op.P2 - 1;
                            }
                            else
                                E.MemSetTypeFlag(in1, MEM.Int);
                            break;
                        }
#if !OMIT_FLOATING_POINT
                    case OP.RealAffinity: // in1
                        {
                            // Opcode: RealAffinity P1 * * * *
                            //
                            // If register P1 holds an integer convert it to a real value.
                            //
                            // This opcode is used when extracting information from a column that has REAL affinity.  Such column values may still be stored as
                            // integers, for space efficiency, but after extraction we want them to have only a real value.
                            in1 = mems[op.P1];
                            if ((in1.Flags & MEM.Int) != 0)
                                MemRealify(in1);
                            break;
                        }
#endif
#if !OMIT_CAST
                    case OP.ToText: // same as TK_TO_TEXT, in1
                        {
                            // Opcode: ToText P1 * * * *
                            //
                            // Force the value in register P1 to be text. If the value is numeric, convert it to a string using the
                            // equivalent of printf().  Blob values are unchanged and are afterwards simply interpreted as text.
                            //
                            // A NULL value is not changed by this routine.  It remains NULL.
                            in1 = mems[op.P1];
                            MemAboutToChange(this, in1);
                            if ((in1.Flags & MEM.Null) != 0) break;
                            Debug.Assert(MEM.Str == (MEM)((int)MEM.Blob >> 3));
                            in1.Flags |= (MEM)((int)(in1.Flags & MEM.Blob) >> 3);
                            ApplyAffinity(in1, AFF.TEXT, encoding);
                            rc = E.ExpandBlob(in1);
                            Debug.Assert((in1.Flags & MEM.Str) != 0 || ctx.MallocFailed);
                            in1.Flags &= ~(MEM.Int | MEM.Real | MEM.Blob | MEM.Zero);
                            UPDATE_MAX_BLOBSIZE(in1);
                            break;
                        }
                    case OP.ToBlob: // same as TK_TO_BLOB, in1
                        {
                            // Opcode: ToBlob P1 * * * *
                            //
                            // Force the value in register P1 to be a BLOB. If the value is numeric, convert it to a string first.
                            // Strings are simply reinterpreted as blobs with no change to the underlying data.
                            //
                            // A NULL value is not changed by this routine.  It remains NULL.
                            in1 = mems[op.P1];
                            if ((in1.Flags & MEM.Null) != 0) break;
                            if ((in1.Flags & MEM.Blob) == 0)
                            {
                                ApplyAffinity(in1, AFF.TEXT, encoding);
                                Debug.Assert((in1.Flags & MEM.Str) != 0 || ctx.MallocFailed);
                                E.MemSetTypeFlag(in1, MEM.Blob);
                            }
                            else
                                in1.Flags &= ~(MEM.TypeMask & ~MEM.Blob);
                            UPDATE_MAX_BLOBSIZE(in1);
                            break;
                        }
                    case OP.ToNumeric: // same as TK_TO_NUMERIC, in1
                        {
                            // Opcode: ToNumeric P1 * * * *
                            //
                            // Force the value in register P1 to be numeric (either an integer or a floating-point number.)
                            // If the value is text or blob, try to convert it to an using the equivalent of atoi() or atof() and store 0 if no such conversion is possible.
                            //
                            // A NULL value is not changed by this routine.  It remains NULL.
                            in1 = mems[op.P1];
                            MemNumerify(in1);
                            break;
                        }
#endif
                    case OP.ToInt:// same as TK_TO_INT, in1
                        {
                            // Opcode: ToInt P1 * * * *
                            //
                            // Force the value in register P1 to be an integer.  If The value is currently a real number, drop its fractional part.
                            // If the value is text or blob, try to convert it to an integer using the equivalent of atoi() and store 0 if no such conversion is possible.
                            //
                            // A NULL value is not changed by this routine.  It remains NULL.
                            in1 = mems[op.P1];
                            if ((in1.Flags & MEM.Null) == 0)
                                MemIntegerify(in1);
                            break;
                        }
#if !OMIT_CAST && !OMIT_FLOATING_POINT
                    case OP.ToReal: // same as TK_TO_REAL, in1
                        {
                            // Opcode: ToReal P1 * * * *
                            //
                            // Force the value in register P1 to be a floating point number. If The value is currently an integer, convert it.
                            // If the value is text or blob, try to convert it to an integer using the equivalent of atoi() and store 0.0 if no such conversion is possible.
                            //
                            // A NULL value is not changed by this routine.  It remains NULL.
                            in1 = mems[op.P1];
                            MemAboutToChange(this, in1);
                            if ((in1.Flags & MEM.Null) == 0)
                                MemRealify(in1);
                            break;
                        }
#endif
                    case OP.Eq: // same as TK_EQ, jump, in1, in3
                    case OP.Ne: // same as TK_NE, jump, in1, in3
                    case OP.Lt: // same as TK_LT, jump, in1, in3
                    case OP.Le: // same as TK_LE, jump, in1, in3
                    case OP.Gt: // same as TK_GT, jump, in1, in3
                    case OP.Ge: // same as TK_GE, jump, in1, in3
                        {
                            // Opcode: Lt P1 P2 P3 P4 P5
                            //
                            // Compare the values in register P1 and P3.  If reg(P3)<reg(P1) then jump to address P2.  
                            //
                            // If the SQLITE_JUMPIFNULL bit of P5 is set and either reg(P1) or reg(P3) is NULL then take the jump.  If the SQLITE_JUMPIFNULL 
                            // bit is clear then fall through if either operand is NULL.
                            //
                            // The SQLITE_AFF_MASK portion of P5 must be an affinity character - SQLITE_AFF_TEXT, SQLITE_AFF_INTEGER, and so forth. An attempt is made 
                            // to coerce both inputs according to this affinity before the comparison is made. If the SQLITE_AFF_MASK is 0x00, then numeric
                            // affinity is used. Note that the affinity conversions are stored back into the input registers P1 and P3.  So this opcode can cause
                            // persistent changes to registers P1 and P3.
                            //
                            // Once any conversions have taken place, and neither value is NULL, the values are compared. If both values are blobs then memcmp() is
                            // used to determine the results of the comparison.  If both values are text, then the appropriate collating function specified in
                            // P4 is  used to do the comparison.  If P4 is not specified then memcmp() is used to compare text string.  If both values are
                            // numeric, then a numeric comparison is used. If the two values are of different types, then numbers are considered less than
                            // strings and strings are considered less than blobs.
                            //
                            // If the SQLITE_STOREP2 bit of P5 is set, then do not jump.  Instead, store a boolean result (either 0, or 1, or NULL) in register P2.
                            //
                            // If the SQLITE_NULLEQ bit is set in P5, then NULL values are considered equal to one another, provided that they do not have their MEM_Cleared
                            // bit set.
                            //
                            // Opcode: Ne P1 P2 P3 P4 P5
                            //
                            // This works just like the Lt opcode except that the jump is taken if the operands in registers P1 and P3 are not equal.  See the Lt opcode for
                            // additional information.
                            //
                            // If SQLITE_NULLEQ is set in P5 then the result of comparison is always either true or false and is never NULL.  If both operands are NULL then the result
                            // of comparison is false.  If either operand is NULL then the result is true. If neither operand is NULL the result is the same as it would be if
                            // the SQLITE_NULLEQ flag were omitted from P5.
                            //
                            // Opcode: Eq P1 P2 P3 P4 P5
                            //
                            // This works just like the Lt opcode except that the jump is taken if the operands in registers P1 and P3 are equal.
                            // See the Lt opcode for additional information.
                            //
                            // If SQLITE_NULLEQ is set in P5 then the result of comparison is always either true or false and is never NULL.  If both operands are NULL then the result
                            // of comparison is true.  If either operand is NULL then the result is false. If neither operand is NULL the result is the same as it would be if
                            // the SQLITE_NULLEQ flag were omitted from P5.
                            //
                            // Opcode: Le P1 P2 P3 P4 P5
                            //
                            // This works just like the Lt opcode except that the jump is taken if the content of register P3 is less than or equal to the content of
                            // register P1.  See the Lt opcode for additional information.
                            //
                            // Opcode: Gt P1 P2 P3 P4 P5
                            //
                            // This works just like the Lt opcode except that the jump is taken if the content of register P3 is greater than the content of
                            // register P1.  See the Lt opcode for additional information.
                            //
                            // Opcode: Ge P1 P2 P3 P4 P5
                            //
                            // This works just like the Lt opcode except that the jump is taken if the content of register P3 is greater than or equal to the content of
                            // register P1.  See the Lt opcode for additional information.
                            in1 = mems[op.P1];
                            in3 = mems[op.P3];
                            MEM flags1 = in1.Flags; // Copy of initial value of in1->flags
                            MEM flags3 = in3.Flags; // Copy of initial value of in3->flags
                            int res = 0; // Result of the comparison of in1 against in3
                            if (((flags1 | flags3) & MEM.Null) != 0)
                            {
                                // One or both operands are NULL
                                if (((AFF)op.P5 & AFF.BIT_NULLEQ) != 0)
                                {
                                    // If SQLITE_NULLEQ is set (which will only happen if the operator is OP_Eq or OP_Ne) then take the jump or not depending on whether or not both operands are null.
                                    Debug.Assert(op.Opcode == OP.Eq || op.Opcode == OP.Ne);
                                    Debug.Assert((flags1 & MEM.Cleared) == 0);
                                    res = ((flags1 & flags3 & MEM.Null) == 0 ? 1 : 0); // Results are equal/not equal
                                }
                                else
                                {
                                    // SQLITE_NULLEQ is clear and at least one operand is NULL, then the result is always NULL. The jump is taken if the SQLITE_JUMPIFNULL bit is set.
                                    if (((AFF)op.P5 & AFF.BIT_STOREP2) != 0)
                                    {
                                        out_ = mems[op.P2];
                                        E.MemSetTypeFlag(out_, MEM.Null);
                                        REGISTER_TRACE(this, op.P2, out_);
                                    }
                                    else if (((AFF)op.P5 & AFF.BIT_JUMPIFNULL) != 0)
                                        pc = op.P2 - 1;
                                    break;
                                }
                            }
                            else
                            {
                                // Neither operand is NULL.  Do a comparison.
                                AFF affinity = ((AFF)op.P5 & AFF.MASK); // Affinity to use for comparison
                                if (affinity != 0)
                                {
                                    ApplyAffinity(in1, affinity, encoding);
                                    ApplyAffinity(in3, affinity, encoding);
                                    if (ctx.MallocFailed) goto no_mem;
                                }

                                Debug.Assert(op.P4Type == Vdbe.P4T.COLLSEQ || op.P4.Coll == null);
                                E.ExpandBlob(in1);
                                E.ExpandBlob(in3);
                                res = MemCompare(in3, in1, op.P4.Coll);
                            }
                            switch (op.Opcode)
                            {
                                case OP.Eq: res = (res == 0 ? 1 : 0); break;
                                case OP.Ne: res = (res != 0 ? 1 : 0); break;
                                case OP.Lt: res = (res < 0 ? 1 : 0); break;
                                case OP.Le: res = (res <= 0 ? 1 : 0); break;
                                case OP.Gt: res = (res > 0 ? 1 : 0); break;
                                default: res = (res >= 0 ? 1 : 0); break;
                            }

                            if (((AFF)op.P5 & AFF.BIT_STOREP2) != 0)
                            {
                                out_ = mems[op.P2];
                                MemAboutToChange(this, out_);
                                E.MemSetTypeFlag(out_, MEM.Int);
                                out_.u.I = res;
                                REGISTER_TRACE(p, op.P2, out_);
                            }
                            else if (res != 0)
                                pc = op.P2 - 1;

                            // Undo any changes made by applyAffinity() to the input registers.
                            in1.Flags = (in1.Flags & ~MEM.TypeMask) | (flags1 & MEM.TypeMask);
                            in3.Flags = (in3.Flags & ~MEM.TypeMask) | (flags3 & MEM.TypeMask);
                            break;
                        }
                    case OP.Permutation:
                        {
                            // Opcode: Permutation * * * P4 *
                            //
                            // Set the permutation used by the OP_Compare operator to be the array of integers in P4.
                            //
                            // The permutation is only valid until the next OP_Compare that has the OPFLAG_PERMUTE bit set in P5. Typically the OP_Permutation should 
                            // occur immediately prior to the OP_Compare.
                            Debug.Assert(op.P4Type == Vdbe.P4T.INTARRAY);
                            Debug.Assert(op.P4.Is != null);
                            permutes = op.P4.Is;
                            break;
                        }
                    case OP.Compare:
                        {
                            // Opcode: Compare P1 P2 P3 P4 P5
                            //
                            // Compare two vectors of registers in reg(P1)..reg(P1+P3-1) (call this vector "A") and in reg(P2)..reg(P2+P3-1) ("B").  Save the result of
                            // the comparison for use by the next OP_Jump instruct.
                            //
                            // If P5 has the OPFLAG_PERMUTE bit set, then the order of comparison is determined by the most recent OP_Permutation operator.  If the
                            // OPFLAG_PERMUTE bit is clear, then register are compared in sequential order.
                            //
                            // P4 is a KeyInfo structure that defines collating sequences and sort orders for the comparison.  The permutation applies to registers
                            // only.  The KeyInfo elements are used sequentially.
                            //
                            // The comparison is a sort comparison, so NULLs compare equal, NULLs are less than numbers, numbers are less than strings,
                            // and strings are less than blobs.
                            if (((OPFLAG)op.P5 & OPFLAG.PERMUTE) == 0) permutes = null;
                            int n = op.P3;
                            KeyInfo keyInfo = op.P4.KeyInfo;
                            Debug.Assert(n > 0);
                            Debug.Assert(keyInfo != null);
                            int p1 = op.P1;
                            int p2 = op.P2;
#if DEBUG
                            if (permutes != null)
                            {
                                int k, max = 0;
                                for (k = 0; k < n; k++) if (permutes[k] > max) max = permutes[k];
                                Debug.Assert(p1 > 0 && p1 + max <= Mems.length + 1);
                                Debug.Assert(p2 > 0 && p2 + max <= Mems.length + 1);
                            }
                            else
                            {
                                Debug.Assert(p1 > 0 && p1 + n <= Mems.length + 1);
                                Debug.Assert(p2 > 0 && p2 + n <= Mems.length + 1);
                            }
#endif
                            for (int i = 0; i < n; i++)
                            {
                                int idx = (permutes != null ? permutes[i] : i);
                                Debug.Assert(E.MemIsValid(mems[p1 + idx]));
                                Debug.Assert(E.MemIsValid(mems[p2 + idx]));
                                REGISTER_TRACE(p1 + idx, mems[p1 + idx]);
                                REGISTER_TRACE(p2 + idx, mems[p2 + idx]);
                                Debug.Assert(i < keyInfo.Fields);
                                CollSeq coll = keyInfo.Colls[i]; // Collating sequence to use on this term
                                SO rev = keyInfo.SortOrders[i]; // True for DESCENDING sort order
                                compare = sqlite3MemCompare(mems[p1 + idx], mems[p2 + idx], coll);
                                if (compare != 0)
                                {
                                    if (rev != 0)
                                        compare = -compare;
                                    break;
                                }
                            }
                            permutes = null;
                            break;
                        }
                    case OP.Jump: // jump
                        {
                            // Opcode: Jump P1 P2 P3 * *
                            //
                            // Jump to the instruction at address P1, P2, or P3 depending on whether in the most recent OP_Compare instruction the P1 vector was less than
                            // equal to, or greater than the P2 vector, respectively.
                            if (compare < 0) pc = op.P1 - 1;
                            else if (compare == 0) pc = op.P2 - 1;
                            else pc = op.P3 - 1;
                            break;
                        }
                    case OP.And: // same as TK_AND, in1, in2, ref3
                    case OP.Or: // same as TK_OR, in1, in2, ref3
                        {
                            // Opcode: And P1 P2 P3 * *
                            //
                            // Take the logical AND of the values in registers P1 and P2 and write the result into register P3.
                            //
                            // If either P1 or P2 is 0 (false) then the result is 0 even if the other input is NULL.  A NULL and true or two NULLs give
                            // a NULL output.
                            //
                            // Opcode: Or P1 P2 P3 * *
                            //
                            // Take the logical OR of the values in register P1 and P2 and store the answer in register P3.
                            //
                            // If either P1 or P2 is nonzero (true) then the result is 1 (true) even if the other input is NULL.  A NULL and false or two NULLs
                            // give a NULL output.
                            in1 = mems[op.P1];
                            int v1 = ((in1.Flags & MEM.Null) != 0 ? 2 : (IntValue(in1) != 0 ? 1 : 0)); // Left operand:  0==FALSE, 1==TRUE, 2==UNKNOWN or NULL
                            in2 = mems[op.P2];
                            int v2 = ((in2.Flags & MEM.Null) != 0 ? 2 : (IntValue(in2) != 0 ? 1 : 0)); // Right operand: 0==FALSE, 1==TRUE, 2==UNKNOWN or NULL
                            if (op.Opcode == OP.And)
                            {
                                byte[] and_logic = new byte[] { 0, 0, 0, 0, 1, 2, 0, 2, 2 };
                                v1 = and_logic[v1 * 3 + v2];
                            }
                            else
                            {
                                byte[] or_logic = new byte[] { 0, 1, 2, 1, 1, 1, 2, 1, 2 };
                                v1 = or_logic[v1 * 3 + v2];
                            }
                            out_ = mems[op.P3];
                            if (v1 == 2)
                                E.MemSetTypeFlag(out_, MEM.Null);
                            else
                            {
                                out_.u.I = v1;
                                E.MemSetTypeFlag(out_, MEM.Int);
                            }
                            break;
                        }
                    case OP.Not: // same as TK_NOT, in1
                        {
                            // Opcode: Not P1 P2 * * *
                            //
                            // Interpret the value in register P1 as a boolean value.  Store the boolean complement in register P2.  If the value in register P1 is
                            // NULL, then a NULL is stored in P2.
                            in1 = mems[op.P1];
                            out_ = mems[op.P2];
                            if ((in1.Flags & MEM.Null) != 0)
                                MemSetNull(out_);
                            else
                                MemSetInt64(out_, IntValue(in1) == 0 ? 1 : 0);
                            break;
                        }
                    case OP.BitNot: // same as TK_BITNOT, in1
                        {
                            // Opcode: BitNot P1 P2 * * *
                            //
                            // Interpret the content of register P1 as an integer.  Store the ones-complement of the P1 value into register P2.  If P1 holds
                            // a NULL then store a NULL in P2.
                            in1 = mems[op.P1];
                            out_ = mems[op.P2];
                            if ((in1.Flags & MEM.Null) != 0)
                                MemSetNull(out_);
                            else
                                MemSetInt64(out_, ~IntValue(in1));
                            break;
                        }
                    case OP.If:
                    case OP.IfNot:
                        {
                            // Opcode: If P1 P2 P3 * *
                            //
                            // Jump to P2 if the value in register P1 is true.  The value is considered true if it is numeric and non-zero.  If the value
                            // in P1 is NULL then take the jump if P3 is non-zero.
                            //
                            // Opcode: IfNot P1 P2 P3 * *
                            //
                            // Jump to P2 if the value in register P1 is False.  The value is considered false if it has a numeric value of zero.  If the value
                            // in P1 is NULL then take the jump if P3 is zero.
                            int c;
                            in1 = mems[op.P1];
                            if ((in1.Flags & MEM.Null) != 0)
                                c = op.P3;
                            else
                            {
#if OMIT_FLOATING_POINT
                                c = (IntValue(in1) != 0 ? 1 : 0);
#else
                                c = (RealValue(in1) != 0.0 ? 1 : 0);
#endif
                                if (op.Opcode == OP.IfNot) c = !c;
                            }
                            if (c != 0)
                                pc = op.P2 - 1;
                            break;
                        }
                    case OP.IsNull: // same as TK_ISNULL, jump, in1
                        {
                            // Opcode: IsNull P1 P2 * * *
                            //
                            // Jump to P2 if the value in register P1 is NULL.
                            in1 = mems[op.P1];
                            if ((in1.Flags & MEM.Null) != 0)
                                pc = op.P2 - 1;
                            break;
                        }
                    case OP.NotNull: // same as TK_NOTNULL, jump, in1
                        {
                            // Opcode: NotNull P1 P2 * * *
                            //
                            // Jump to P2 if the value in register P1 is not NULL.  
                            in1 = mems[op.P1];
                            if ((in1.Flags & MEM.Null) == 0)
                                pc = op.P2 - 1;
                            break;
                        }
                    case OP.Column:
                        {
                            // Opcode: Column P1 P2 P3 P4 P5
                            //
                            // Interpret the data that cursor P1 points to as a structure built using the MakeRecord instruction.  (See the MakeRecord opcode for additional
                            // information about the format of the data.)  Extract the P2-th column from this record.  If there are less that (P2+1) 
                            // values in the record, extract a NULL.
                            //
                            // The value extracted is stored in register P3.
                            //
                            // If the column contains fewer than P2 fields, then extract a NULL.  Or, if the P4 argument is a P4_MEM use the value of the P4 argument as
                            // the result.
                            //
                            // If the OPFLAG_CLEARCACHE bit is set on P5 and P1 is a pseudo-table cursor, then the cache of the cursor is reset prior to extracting the column.
                            // The first OP_Column against a pseudo-table after the value of the content register has changed should have this bit set.
                            //
                            // If the OPFLAG_LENGTHARG and OPFLAG_TYPEOFARG bits are set on P5 when the result is guaranteed to only be used as the argument of a length()
                            // or typeof() function, respectively.  The loading of large blobs can be skipped for length() and all content loading can be skipped for typeof().
                            int p1 = op.P1; // P1 value of the opcode
                            int p2 = op.P2; // column number to retrieve
                            Mem sMem = C._alloc(sMem); // For storing the record being decoded
                            Debug.Assert(p1 < Cursors.length);
                            Debug.Assert(op.P3 > 0 && op.P3 <= Mems.length);
                            Mem dest = mems[op.P3]; // Where to write the extracted value
                            MemAboutToChange(this, dest);

                            // This block sets the variable payloadSize to be the total number of bytes in the record.
                            //
                            // zRec is set to be the complete text of the record if it is available. The complete record text is always available for pseudo-tables
                            // If the record is stored in a cursor, the complete record text might be available in the  pC->aRow cache.  Or it might not be.
                            // If the data is unavailable,  zRec is set to NULL.
                            //
                            // We also compute the number of columns in the record.  For cursors, the number of columns is stored in the VdbeCursor.Fields element.
                            VdbeCursor c = Cursors[p1]; // The VDBE cursor
                            Debug.Assert(c != null);
#if !OMIT_VIRTUALTABLE
                            Debug.Assert(c.VtabCursor == null);
#endif

                            Btree.BtCursor crsr = c.Cursor; // The BTree cursor
                            uint payloadSize = 0; // Number of bytes in the record
                            long payloadSize64 = 0; // Number of bytes in the record
                            byte[] rec = null; // Pointer to complete record-data
                            if (crsr != null)
                            {
                                // The record is stored in a B-Tree
                                rc = CursorMoveto(c);
                                if (rc != 0) goto abort_due_to_error;
                                if (c.NullRow)
                                    payloadSize = 0;
                                else if (c.CacheStatus == CacheCtr)
                                {
                                    payloadSize = (uint)c.PayloadSize;
                                    rec = C._alloc((int)payloadSize);
                                    Buffer.BlockCopy(crsr.Info.Cell, c.Rows, rec, 0, (int)payloadSize);
                                }
                                else if (c.IsIndex)
                                {
                                    Debug.Assert(Btree.CursorIsValid(crsr));
                                    rc = Btree.KeySize(crsr, ref payloadSize64);
                                    Debug.Assert(rc == RC.OK); // True because of CursorMoveto() call above
                                    // sqlite3BtreeParseCellPtr() uses getVarint32() to extract the payload size, so it is impossible for payloadSize64 to be larger than 32 bits.
                                    Debug.Assert(((ulong)payloadSize64 & uint.MaxValue) == (ulong)payloadSize64);
                                    payloadSize = (uint)payloadSize64;
                                }
                                else
                                {
                                    Debug.Assert(Btree.CursorIsValid(crsr));
                                    rc = Btree.DataSize(crsr, ref payloadSize);
                                    Debug.Assert(rc == RC.OK); // DataSize() cannot fail
                                }
                            }
                            else if (c.PseudoTableReg > 0)
                            {
                                // The record is the sole entry of a pseudo-table
                                Mem reg = mems[c.PseudoTableReg]; // PseudoTable input register
                                if (c.MultiPseudo)
                                {
                                    MemShallowCopy(dest, reg + p2, MEM.Ephem);
                                    Deephemeralize(dest);
                                    goto op_column_out;
                                }
                                Debug.Assert((reg.Flags & MEM.Blob) != 0);
                                Debug.Assert(E.MemIsValid(reg));
                                payloadSize = (uint)reg.N;
                                rec = reg.Z_;
                                c.CacheStatus = (((OPFLAG)op.P5 & OPFLAG.CLEARCACHE) != 0 ? E.CACHE_STALE : CacheCtr);
                                Debug.Assert(payloadSize == 0 || rec != null);
                            }
                            else
                                payloadSize = 0; // Consider the row to be NULL

                            // If payloadSize is 0, then just store a NULL.  This can happen because of nullRow or because of a corrupt database.
                            if (payloadSize == 0)
                            {
                                E.MemSetTypeFlag(dest, MEM.Null);
                                goto op_column_out;
                            }
                            Debug.Assert(ctx.Limits[(int)LIMIT.LENGTH] >= 0);
                            if (payloadSize > (uint)ctx.Limits[(int)LIMIT.LENGTH])
                                goto too_big;

                            int fields = c.Fields; // number of fields in the record
                            Debug.Assert(p2 < fields);

                            // Read and parse the table header.  Store the results of the parse into the record header cache fields of the cursor.
                            uint[] types = c.Types; // aType[i] holds the numeric type of the i-th column
                            uint[] offsets; // aOffset[i] is offset to start of data for i-th column
                            byte[] data = null; // Part of the record being decoded
                            int len; // The length of the serialized data for the column
                            uint t; // A type code from the record header
                            if (c.CacheStatus == CacheCtr)
                                offsets = c.Offsets;
                            else
                            {
                                Debug.Assert(types != null);
                                int avail = 0; // Number of bytes of available data
                                //: c.Offsets = offsets = types[fields];
                                offsets = new uint[fields];
                                c.Offsets = offsets;
                                c.PayloadSize = (int)payloadSize;
                                c.CacheStatus = CacheCtr;

                                // Figure out how many bytes are in the header
                                if (rec != null)
                                    data = rec;
                                else
                                {
                                    data = (c.IsIndex ? Btree.KeyFetch(crsr, ref avail, ref c.Rows) : Btree.DataFetch(crsr, ref avail, ref c.Rows));
                                    // If KeyFetch()/DataFetch() managed to get the entire payload, save the payload in the pC->aRow cache.  That will save us from
                                    // having to make additional calls to fetch the content portion of the record.
                                    Debug.Assert(avail >= 0);
                                    if (payloadSize <= (uint)avail)
                                    {
                                        rec = data;
                                        //c.Rows = data;
                                    }
                                    else
                                        c.Rows = -1; //: c.Rows = null;
                                }
                                // The following assert is true in all cases except when the database file has been corrupted externally.
                                //Debug.Assert(rec != 0 || avail >= payloadSize || avail >= 9);
                                uint offset; // Offset into the data
                                int sizeHdr = ConvertEx.GetVarint32(data, out offset); // Size of the header size field at start of record

                                // Make sure a corrupt database has not given us an oversize header. Do this now to avoid an oversize memory allocation.
                                //
                                // Type entries can be between 1 and 5 bytes each.  But 4 and 5 byte types use so much data space that there can only be 4096 and 32 of
                                // them, respectively.  So the maximum header length results from a 3-byte type for each of the maximum of 32768 columns plus three
                                // extra bytes for the header length itself.  32768*3 + 3 = 98307.
                                if (offset > 98307)
                                {
                                    rc = SysEx.CORRUPT_BKPT();
                                    goto op_column_out;
                                }

                                // Compute in len the number of bytes of data we need to read in order to get nField type values.  offset is an upper bound on this.  But
                                // nField might be significantly less than the true number of columns in the table, and in that case, 5*nField+3 might be smaller than offset.
                                // We want to minimize len in order to limit the size of the memory allocation, especially if a corrupt database file has caused offset
                                // to be oversized. Offset is limited to 98307 above.  But 98307 might still exceed Robson memory allocation limits on some configurations.
                                // On systems that cannot tolerate large memory allocations, nField*5+3 will likely be much smaller since nField will likely be less than
                                // 20 or so.  This insures that Robson memory allocation limits are not exceeded even for corrupt database files.
                                len = fields * 5 + 3;
                                if (len > (int)offset) len = (int)offset;

                                // The KeyFetch() or DataFetch() above are fast and will get the entire record header in most cases.  But they will fail to get the complete
                                // record header if the record header does not fit on a single page in the B-Tree.  When that happens, use sqlite3VdbeMemFromBtree() to
                                // acquire the complete header text.
                                if (rec == null && avail < len)
                                {
                                    sMem.Flags = 0;
                                    sMem.Ctx = null;
                                    rc = MemFromBtree(crsr, 0, len, c.IsIndex, sMem);
                                    if (rc != RC.OK)
                                        goto op_column_out;
                                    data = sMem.Z_;
                                }
                                int endHdr = len; //: data[len]; // Pointer to first byte after the header
                                int idx = sizeHdr; //: data[sizeHdr]; // Index into header

                                // Scan the header and use it to fill in the aType[] and aOffset[] arrays.  aType[i] will contain the type integer for the i-th
                                // column and aOffset[i] will contain the offset from the beginning of the record to the start of the data for the i-th column
                                for (int i = 0; i < fields; i++)
                                {
                                    if (idx < endHdr)
                                    {
                                        offsets[i] = offset;
                                        if (data[idx] < 0x80)
                                        {
                                            t = data[idx];
                                            idx++;
                                        }
                                        else
                                            idx += ConvertEx.GetVarint32(data, idx, out t);
                                        types[i] = t;
                                        uint sizeField = SerialTypeLen(t); // Number of bytes in the content of a field
                                        offset += sizeField;
                                        if (offset < sizeField) // True if offset overflows
                                        {
                                            idx = int.MaxValue; // Forces SQLITE_CORRUPT return below
                                            break;
                                        }
                                    }
                                    else
                                        // If i is less that nField, then there are fewer fields in this record than SetNumColumns indicated there are columns in the
                                        // table. Set the offset for any extra columns not present in the record to 0. This tells code below to store the default value
                                        // for the column instead of deserializing a value from the record.
                                        offsets[i] = 0;
                                }
                                MemRelease(sMem);
                                sMem.Flags = MEM.Null;

                                // If we have read more header data than was contained in the header, or if the end of the last field appears to be past the end of the
                                // record, or if the end of the last field appears to be before the end of the record (when all fields present), then we must be dealing 
                                // with a corrupt database.
                                if ((idx > endHdr) || (offset > payloadSize) || (idx == endHdr && offset != payloadSize))
                                {
                                    rc = SysEx.CORRUPT_BKPT();
                                    goto op_column_out;
                                }
                            }

                            // Get the column information. If aOffset[p2] is non-zero, then deserialize the value from the record. If aOffset[p2] is zero,
                            // then there are not enough fields in the record to satisfy the request.  In this case, set the value NULL or to P4 if P4 is
                            // a pointer to a Mem object.
                            if (offsets[p2] != 0)
                            {
                                Debug.Assert(rc == RC.OK);
                                if (rec != null)
                                {
                                    E.VdbeMemRelease(dest);
                                    SerialGet(rec, (int)offsets[p2], types[p2], dest);
                                }
                                else
                                {
                                    // This branch happens only when the row overflows onto multiple pages
                                    t = types[p2];
                                    if (((OPFLAG)op.P5 & (OPFLAG.LENGTHARG | OPFLAG.TYPEOFARG)) != 0 && ((t >= 12 && (t & 1) == 0) || ((OPFLAG)op.P5 & OPFLAG.TYPEOFARG) != 0))
                                    {
                                        // Content is irrelevant for the typeof() function and for the length(X) function if X is a blob.  So we might as well use
                                        // bogus content rather than reading content from disk.  NULL works for text and blob and whatever is in the payloadSize64 variable
                                        // will work for everything else.
                                        data = (t < 12 ? BitConverter.GetBytes(payloadSize64) : null);
                                    }
                                    else
                                    {
                                        len = (int)SerialTypeLen(types[p2]);
                                        MemMove(sMem, dest);
                                        rc = MemFromBtree(crsr, (int)offsets[p2], len, c.IsIndex, sMem);
                                        if (rc != RC.OK)
                                            goto op_column_out;
                                        data = sMem.Z_;
                                        sMem.Z_ = null;
                                    }
                                    SerialGet(data, types[p2], dest);
                                }
                                dest.Encode = encoding;
                            }
                            else
                            {
                                if (op.P4Type == P4T.MEM)
                                    MemShallowCopy(dest, op.P4.Mem, MEM.Static);
                                else
                                    E.MemSetTypeFlag(dest, MEM.Null);
                            }

                            // If we dynamically allocated space to hold the data (in the sqlite3VdbeMemFromBtree() call above) then transfer control of that
                            // dynamically allocated space over to the pDest structure. This prevents a memory copy.
                            //: if (sMem.Malloc != null)
                            //: {
                            //:     Debug.Assert(sMem.Z == sMem.Malloc);
                            //:     Debug.Assert(sMem.Del == null);
                            //:     Debug.Assert((dest.Flags & MEM.Dyn) == 0);
                            //:     Debug.Assert((dest.Flags & (MEM.Blob | MEM.Str)) == 0 || dest.Z == sMem.Z);
                            //:     dest.Flags &= ~(MEM.Ephem | MEM.Static);
                            //:     dest.Flags |= MEM.Term;
                            //:     dest.Z = sMem.Z;
                            //:     dest.Malloc = sMem.zMalloc;
                            //: }

                            rc = MemMakeWriteable(dest);

                        op_column_out:
                            UPDATE_MAX_BLOBSIZE(dest);
                            REGISTER_TRACE(op.P3, dest);
                            break;
                        }
                    case OP.Affinity:
                        {
                            // Opcode: Affinity P1 P2 * P4 *
                            //
                            // Apply affinities to a range of P2 registers starting with P1.
                            //
                            // P4 is a string that is P2 characters long. The nth character of the string indicates the column affinity that should be used for the nth
                            // memory cell in the range.
                            string affinity = op.P4.Z; // The affinity to be applied
                            Debug.Assert(affinity != null);
                            Debug.Assert(affinity.Length <= op.P2); //: affinity[op.P2] == 0
                            //: in1 = mems[op.P1];
                            AFF aff; // A single character of affinity
                            for (int aIdx = 0; aIdx < affinity.Length; aIdx++) //: while ((aff = *(affinity++)) != 0)
                            {
                                aff = (AFF)affinity[aIdx];
                                in1 = mems[op.P1 + aIdx];
                                //: Debug.Assert( in1 <= mems[Mems.length]);
                                Debug.Assert(E.MemIsValid(in1));
                                E.ExpandBlob(in1);
                                ApplyAffinity(in1, aff, encoding);
                                //: in1++;
                            }
                            break;
                        }
                    case OP.MakeRecord:
                        {
                            // Opcode: MakeRecord P1 P2 P3 P4 *
                            //
                            // Convert P2 registers beginning with P1 into the [record format] use as a data record in a database table or as a key
                            // in an index.  The OP_Column opcode can decode the record later.
                            //
                            // P4 may be a string that is P2 characters long.  The nth character of the string indicates the column affinity that should be used for the nth
                            // field of the index key.
                            //
                            // The mapping from character to affinity is given by the SQLITE_AFF_ macros defined in sqliteInt.h.
                            //
                            // If P4 is NULL then all index fields have the affinity NONE.

                            // Assuming the record contains N fields, the record format looks like this:
                            //
                            // ------------------------------------------------------------------------
                            // | hdr-size | type 0 | type 1 | ... | type N-1 | data0 | ... | data N-1 | 
                            // ------------------------------------------------------------------------
                            //
                            // Data(0) is taken from register P1.  Data(1) comes from register P1+1 and so froth.
                            //
                            // Each type field is a varint representing the serial type of the corresponding data element (see sqlite3VdbeSerialType()). The
                            // hdr-size field is also a varint which is the offset from the beginning of the record to data0.
                            ulong dataLength = 0; // Number of bytes of data space
                            int hdrLength = 0; // Number of bytes of header space
                            int zeros = 0; // Number of zero bytes at the end of the record
                            int fields = op.P1; // Number of fields in the record
                            string affinity = (op.P4.Z ?? string.Empty); // The affinity string for the record
                            Debug.Assert(fields > 0 && op.P2 > 0 && op.P2 + fields <= Mems.length + 1);
                            //: Mem data0 = mems[fields]; // First field to be combined into the record
                            fields = op.P2;
                            //: Mem last =  data0[fields - 1]; // Last field of the record
                            int fileFormat = MinWriteFileFormat; // File format to use for encoding

                            // Identify the output register
                            Debug.Assert(op.P3 < op.P1 || op.P3 >= op.P1 + op.P2);
                            out_ = mems[op.P3];
                            MemAboutToChange(this, out_);

                            // Loop through the elements that will make up the record to figure out how much space is required for the new record.

                            Mem rec; // The new record
                            uint serialType; // Type field
                            for (int d0 = 0; d0 < fields; d0++)
                            {
                                rec = mems[op.P1 + d0];
                                Debug.Assert(E.MemIsValid(rec));
                                if (d0 < affinity.Length && affinity[d0] != '\0')
                                    ApplyAffinity(rec, (AFF)affinity[d0], encoding);
                                if ((rec.Flags & MEM.Zero) != 0 && rec.N > 0)
                                    MemExpandBlob(rec);
                                serialType = SerialType(rec, fileFormat);
                                int len = (int)SerialTypeLen(serialType); // Length of a field
                                dataLength += (ulong)len;
                                hdrLength += ConvertEx.GetVarintLength(serialType);
                                if ((rec.Flags & MEM.Zero) != 0)
                                    zeros += rec.u.Zeros; // Only pure zero-filled BLOBs can be input to this Opcode. We do not allow blobs with a prefix and a zero-filled tail.
                                else if (len != 0)
                                    zeros = 0;
                            }

                            // Add the initial header varint and total the size
                            int varintLength; // Number of bytes in a varint
                            hdrLength += varintLength = ConvertEx.GetVarintLength((ulong)hdrLength);
                            if (varintLength < ConvertEx.GetVarintLength((ulong)hdrLength))
                                hdrLength++;
                            long bytes = (long)((ulong)hdrLength + dataLength - (ulong)zeros); // Data space required for this record
                            if (bytes > ctx.Limits[(int)LIMIT.LENGTH])
                                goto too_big;

                            // Make sure the output register has a buffer large enough to store the new record. The output register (op->P3) is not allowed to
                            // be one of the input registers (because the following call to sqlite3VdbeMemGrow() could clobber the value before it is used).
                            //: if (MemGrow(out_, (int)bytes, 0) != 0)
                            //:     goto no_mem;
                            byte[] newRecord = C._alloc((int)bytes); //: out_.Z; // A buffer to hold the data for the new record

                            // Write the record
                            int i = ConvertEx.PutVarint32(newRecord, hdrLength); // Space used in zNewRecord[]
                            for (int d0 = 0; d0 < fields; d0++)
                            {
                                rec = mems[op.P1 + d0];
                                serialType = SerialType(rec, fileFormat);
                                i += ConvertEx.PutVarint32(newRecord, i, (int)serialType); // serial type
                            }
                            for (int d0 = 0; d0 < fields; d0++) // serial data
                            {
                                rec = mems[op.P1 + d0];
                                i += (int)SerialPut(newRecord, i, (int)bytes - i, rec, fileFormat);
                            }
                            Debug.Assert(i == bytes);

                            Debug.Assert(op.P3 > 0 && op.P3 <= Mems.length);
                            out_.Z_ = newRecord;
                            out_.Z = null;
                            out_.N = (int)bytes;
                            out_.Flags = MEM.Blob | MEM.Dyn;
                            out_.Del = null;
                            if (zeros != 0)
                            {
                                out_.u.Zeros = zeros;
                                out_.Flags |= MEM.Zero;
                            }
                            out_.Encode = TEXTENCODE.UTF8; // In case the blob is ever converted to text
                            REGISTER_TRACE(p, op.P3, out_);
                            UPDATE_MAX_BLOBSIZE(out_);
                            break;
                        }
#if !OMIT_BTREECOUNT
                    case OP.Count: // out2-prerelease
                        {
                            // Opcode: Count P1 P2 * * *
                            //
                            // Store the number of entries (an integer value) in the table or index opened by cursor P1 in register P2
                            long entrys = 0;
                            Btree.BtCursor crsr = p.apCsr[op.P1].pCursor;
                            if (crsr != null)
                                rc = Btree.Count(crsr, ref entrys);
                            else
                                entrys = 0;
                            out_.u.I = entrys;
                            break;
                        }
#endif
                    case OP.Savepoint:
                        {
                            // Opcode: Savepoint P1 * * P4 *
                            //
                            // Open, release or rollback the savepoint named by parameter P4, depending on the value of P1. To open a new savepoint, P1==0. To release (commit) an
                            // existing savepoint, P1==1, or to rollback an existing savepoint P1==2.
                            IPager.SAVEPOINT p1 = (IPager.SAVEPOINT)op.P1; // Value of P1 operand 
                            string name = op.P4.Z; // Name of savepoint

                            // Assert that the p1 parameter is valid. Also that if there is no open transaction, then there cannot be any savepoints.
                            Debug.Assert(ctx.Savepoints == null || ctx.AutoCommit == 0);
                            Debug.Assert(p1 == IPager.SAVEPOINT.BEGIN || p1 == IPager.SAVEPOINT.RELEASE || p1 == IPager.SAVEPOINT.ROLLBACK);
                            Debug.Assert(ctx.Savepoints != null || ctx.IsTransactionSavepoint == 0);
                            Debug.Assert(CheckSavepointCount(ctx) != 0);

                            if (p1 == IPager.SAVEPOINT.BEGIN)
                            {
                                if (ctx.WriteVdbeCnt > 0)
                                {
                                    // A new savepoint cannot be created if there are active write statements (i.e. open read/write incremental blob handles).
                                    C._setstring(ref ErrMsg, ctx, "cannot open savepoint - SQL statements in progress");
                                    rc = RC.BUSY;
                                }
                                else
                                {
                                    int nameLength = name.Length;

#if !OMIT_VIRTUALTABLE
                                    // This call is Ok even if this savepoint is actually a transaction savepoint (and therefore should not prompt xSavepoint()) callbacks.
                                    // If this is a transaction savepoint being opened, it is guaranteed that the ctx->aVTrans[] array is empty.
                                    Debug.Assert(ctx.AutoCommit == 0 || ctx.VTrans.length == 0);
                                    rc = VTable.Savepoint(ctx, IPager.SAVEPOINT.BEGIN, ctx.Statements + ctx.SavepointsLength);
                                    if (rc != RC.OK) goto abort_due_to_error;
#endif

                                    // Create a new savepoint structure.
                                    Savepoint newSavepoint = new Savepoint();
                                    if (newSavepoint != null)
                                    {
                                        newSavepoint.Name = name;

                                        // If there is no open transaction, then mark this as a special "transaction savepoint".
                                        if (ctx.AutoCommit != 0)
                                        {
                                            ctx.AutoCommit = 0;
                                            ctx.IsTransactionSavepoint = 1;
                                        }
                                        else
                                            ctx.SavepointsLength++;

                                        // Link the new savepoint into the database handle's list.
                                        newSavepoint.Next = ctx.Savepoints;
                                        ctx.Savepoints = newSavepoint;
                                        newSavepoint.DeferredCons = ctx.DeferredCons;
                                    }
                                }
                            }
                            else
                            {
                                // Find the named savepoint. If there is no such savepoint, then an an error is returned to the user.
                                int savepointId = 0;
                                Savepoint savepoint; for (savepoint = ctx.Savepoints; savepoint != null && !string.Equals(savepoint.Name, name, StringComparison.OrdinalIgnoreCase); savepoint = savepoint.Next)
                                    savepointId++;
                                if (savepoint == null)
                                {
                                    C._setstring(ref ErrMsg, ctx, "no such savepoint: %s", name);
                                    rc = RC.ERROR;
                                }
                                else if (ctx.WriteVdbeCnt > 0 || (p1 == IPager.SAVEPOINT.ROLLBACK && ctx.ActiveVdbeCnt > 1))
                                {
                                    // It is not possible to release (commit) a savepoint if there are active write statements.
                                    C._setstring(ref ErrMsg, ctx, "cannot %s savepoint - SQL statements in progress");
                                    rc = RC.BUSY;
                                }
                                else
                                {
                                    // Determine whether or not this is a transaction savepoint. If so, and this is a RELEASE command, then the current transaction is committed. 
                                    int isTransaction = (savepoint.Next == null && ctx.IsTransactionSavepoint != 0 ? 1 : 0);
                                    if (isTransaction != 0 && p1 == IPager.SAVEPOINT.RELEASE)
                                    {
                                        if ((rc = CheckFk(true)) != RC.OK)
                                            goto vdbe_return;
                                        ctx.AutoCommit = 1;
                                        if (Halt() == RC.BUSY)
                                        {
                                            PC = pc;
                                            ctx.AutoCommit = 0;
                                            RC_ = rc = RC.BUSY;
                                            goto vdbe_return;
                                        }
                                        ctx.IsTransactionSavepoint = 0;
                                        rc = RC_;
                                    }
                                    else
                                    {
                                        savepointId = ctx.SavepointsLength - savepointId - 1;
                                        int ii;
                                        if (p1 == IPager.SAVEPOINT.ROLLBACK)
                                            for (ii = 0; ii < ctx.DBs.length; ii++)
                                                ctx.DBs[ii].Bt.TripAllCursors(RC.ABORT);
                                        for (ii = 0; ii < ctx.DBs.length; ii++)
                                        {
                                            rc = ctx.DBs[ii].Bt.Savepoint(p1, savepointId);
                                            if (rc != RC.OK)
                                                goto abort_due_to_error;
                                        }
                                        if (p1 == IPager.SAVEPOINT.ROLLBACK && (ctx.Flags & Context.FLAG.InternChanges) != 0)
                                        {
                                            ExpirePreparedStatements(ctx);
                                            Parse.ResetAllSchemasOfConnection(ctx);
                                            ctx.Flags = (ctx.Flags | Context.FLAG.InternChanges);
                                        }
                                    }

                                    // Regardless of whether this is a RELEASE or ROLLBACK, destroy all savepoints nested inside of the savepoint being operated on.
                                    while (ctx.Savepoints != savepoint)
                                    {
                                        Savepoint tmp = ctx.Savepoints;
                                        ctx.Savepoints = tmp.Next;
                                        C._tagfree(ctx, ref tmp);
                                        ctx.SavepointsLength--;
                                    }

                                    // If it is a RELEASE, then destroy the savepoint being operated on too. If it is a ROLLBACK TO, then set the number of deferred 
                                    // constraint violations present in the database to the value stored when the savepoint was created.
                                    if (p1 == IPager.SAVEPOINT.RELEASE)
                                    {
                                        Debug.Assert(savepoint == ctx.Savepoints);
                                        ctx.Savepoints = savepoint.Next;
                                        C._tagfree(ctx, ref savepoint);
                                        if (isTransaction == 0)
                                            ctx.SavepointsLength--;
                                    }
                                    else
                                        ctx.DeferredCons = savepoint.DeferredCons;

                                    if (isTransaction == 0)
                                    {
                                        rc = VTable.Savepoint(ctx, p1, savepointId);
                                        if (rc != RC.OK) goto abort_due_to_error;
                                    }
                                }
                            }
                            break;
                        }
                    case OP.AutoCommit:
                        {
                            int desiredAutoCommit = (byte)op.P1;
                            int rollbackId = op.P2;
                            bool turnOnAC = (desiredAutoCommit != 0 && ctx.AutoCommit == 0);
                            Debug.Assert(desiredAutoCommit != 0 || desiredAutoCommit == 0);
                            Debug.Assert(desiredAutoCommit != 0 || rollbackId == 0);
                            Debug.Assert(ctx.ActiveVdbeCnt > 0); // At least this one VM is active
#if false
                            if (turnOnAC && rollbackId != 0 && ctx.ActiveVdbeCnt > 1)
                            {
                                // If this instruction implements a ROLLBACK and other VMs are still running, and a transaction is active, return an error indicating
                                // that the other VMs must complete first. 
                                C._setstring(ref ErrMsg, ctx, "cannot rollback transaction - SQL statements in progress");
                                rc = RC.BUSY;
                            }
                            else
#endif
                            if (turnOnAC && 0 == rollbackId && ctx.WriteVdbeCnt > 0)
                            {
                                // If this instruction implements a COMMIT and other VMs are writing return an error indicating that the other VMs must complete first. 
                                C._setstring(ref ErrMsg, ctx, "cannot commit transaction - SQL statements in progress");
                                rc = RC.BUSY;
                            }
                            else if (desiredAutoCommit != ctx.AutoCommit)
                            {
                                if (rollbackId != 0)
                                {
                                    Debug.Assert(desiredAutoCommit != 0);
                                    Main.RollbackAll(ctx, RC.ABORT_ROLLBACK);
                                    ctx.AutoCommit = 1;
                                }
                                else if ((rc = CheckFk(true)) != RC.OK)
                                    goto vdbe_return;
                                else
                                {
                                    ctx.AutoCommit = (byte)desiredAutoCommit;
                                    if (Halt() == RC.BUSY)
                                    {
                                        PC = pc;
                                        ctx.AutoCommit = (byte)(desiredAutoCommit == 0 ? 1 : 0);
                                        RC_ = rc = RC.BUSY;
                                        goto vdbe_return;
                                    }
                                }
                                Debug.Assert(ctx.Statements == 0);
                                Main.CloseSavepoints(ctx);
                                rc = (RC_ == RC.OK ? RC.DONE : RC.ERROR);
                                goto vdbe_return;
                            }
                            else
                            {
                                C._setstring(ref ErrMsg, ctx, (desiredAutoCommit == 0 ? "cannot start a transaction within a transaction" : (rollbackId != 0 ? "cannot rollback - no transaction is active" : "cannot commit - no transaction is active")));
                                rc = RC.ERROR;
                            }
                            break;
                        }
                    case OP.Transaction:
                        {
                            // Opcode: Transaction P1 P2 * * *
                            //
                            // Begin a transaction.  The transaction ends when a Commit or Rollback opcode is encountered.  Depending on the ON CONFLICT setting, the
                            // transaction might also be rolled back if an error is encountered.
                            //
                            // P1 is the index of the database file on which the transaction is started.  Index 0 is the main database file and index 1 is the
                            // file used for temporary tables.  Indices of 2 or more are used for attached databases.
                            //
                            // If P2 is non-zero, then a write-transaction is started.  A RESERVED lock is obtained on the database file when a write-transaction is started.  No
                            // other process can start another write transaction while this transaction is underway.  Starting a write transaction also creates a rollback journal. A
                            // write transaction must be started before any changes can be made to the database.  If P2 is 2 or greater then an EXCLUSIVE lock is also obtained
                            // on the file.
                            //
                            // If a write-transaction is started and the Vdbe.usesStmtJournal flag is true (this flag is set if the Vdbe may modify more than one row and may
                            // throw an ABORT exception), a statement transaction may also be opened. More specifically, a statement transaction is opened iff the database
                            // connection is currently not in autocommit mode, or if there are other active statements. A statement transaction allows the changes made by this
                            // VDBE to be rolled back after an error without having to roll back the entire transaction. If no error is encountered, the statement transaction
                            // will automatically commit when the VDBE halts.
                            //
                            // If P2 is zero, then a read-lock is obtained on the database file.
                            Debug.Assert(op.P1 >= 0 && op.P1 < ctx.DBs.length);
                            Debug.Assert((BtreeMask & (((yDbMask)1) << op.P1)) != 0);
                            Btree bt = ctx.DBs[op.P1].Bt;
                            if (bt != null)
                            {
                                rc = bt.BeginTrans(op.P2);
                                if (rc == RC.BUSY)
                                {
                                    PC = pc;
                                    RC_ = rc = RC.BUSY;
                                    goto vdbe_return;
                                }
                                if (rc != RC.OK)
                                    goto abort_due_to_error;
                                if (op.P2 != 0 && UsesStmtJournal && (ctx.AutoCommit == 0 || ctx.ActiveVdbeCnt > 1))
                                {
                                    Debug.Assert(bt.IsInTrans());
                                    if (StatementID == 0)
                                    {
                                        Debug.Assert(ctx.Statements >= 0 && ctx.SavepointsLength >= 0);
                                        ctx.Statements++;
                                        StatementID = ctx.SavepointsLength + ctx.Statements;
                                    }
                                    rc = VTable.Savepoint(ctx, IPager.SAVEPOINT.BEGIN, StatementID - 1);
                                    if (rc == RC.OK)
                                        rc = bt.BeginStmt(StatementID);
                                    // Store the current value of the database handles deferred constraint counter. If the statement transaction needs to be rolled back,
                                    // the value of this counter needs to be restored too.
                                    StmtDefCons = ctx.DeferredCons;
                                }
                            }
                            break;
                        }
                    case OP.ReadCookie: // out2-prerelease
                        {
                            // Opcode: ReadCookie P1 P2 P3 * *
                            //
                            // Read cookie number P3 from database P1 and write it into register P2. P3==1 is the schema version.  P3==2 is the database format.
                            // P3==3 is the recommended pager cache size, and so forth.  P1==0 is the main database file and P1==1 is the database file used to store
                            // temporary tables.
                            //
                            // There must be a read-lock on the database (either a transaction must be started or there must be an open cursor) before
                            // executing this instruction.
                            int db = op.P1;
                            Btree.META cookie = (Btree.META)op.P3;
                            Debug.Assert(op.P3 < Btree.N_BTREE_META);
                            Debug.Assert(db >= 0 && db < ctx.DBs.length);
                            Debug.Assert(ctx.DBs[db].Bt != null);
                            Debug.Assert((BtreeMask & (((yDbMask)1) << db)) != 0);
                            uint meta = 0;
                            ctx.DBs[db].Bt.GetMeta(cookie, ref meta);
                            out_.u.I = (int)meta;
                            break;
                        }
                    case OP.SetCookie: // in3
                        {
                            Debug.Assert(op.P2 < Btree.N_BTREE_META);
                            Debug.Assert(op.P1 >= 0 && op.P1 < ctx.DBs.length);
                            Debug.Assert((BtreeMask & (((yDbMask)1) << op.P1)) != 0);
                            Context.DB db = ctx.DBs[op.P1];
                            Debug.Assert(db.Bt != null);
                            Debug.Assert(Btree.SchemaMutexHeld(ctx, op.P1, null));
                            in3 = mems[op.P3];
                            MemIntegerify(in3);
                            // See note about index shifting on OP_ReadCookie
                            rc = db.Bt.UpdateMeta((Btree.META)op.P2, (uint)in3.u.I);
                            if ((Btree.META)op.P2 == Btree.META.SCHEMA_VERSION)
                            {
                                // When the schema cookie changes, record the new cookie internally
                                db.Schema.SchemaCookie = (int)in3.u.I;
                                ctx.Flags |= Context.FLAG.InternChanges;
                            }
                            else if ((Btree.META)op.P2 == Btree.META.FILE_FORMAT)
                                // Record changes in the file format
                                db.Schema.FileFormat = (byte)in3.u.I;
                            if (op.P1 == 1)
                            {
                                // Invalidate all prepared statements whenever the TEMP database schema is changed.  Ticket #1644
                                ExpirePreparedStatements(ctx);
                                Expired = false;
                            }
                            break;
                        }
                    case OP.VerifyCookie:
                        {
                            // Opcode: VerifyCookie P1 P2 P3 * *
                            //
                            // Check the value of global database parameter number 0 (the schema version) and make sure it is equal to P2 and that the
                            // generation counter on the local schema parse equals P3.
                            //
                            // P1 is the database number which is 0 for the main database file and 1 for the file holding temporary tables and some higher number
                            // for auxiliary databases.
                            //
                            // The cookie changes its value whenever the database schema changes. This operation is used to detect when that the cookie has changed
                            // and that the current process needs to reread the schema.
                            //
                            // Either a transaction needs to have been started or an OP_Open needs to be executed (to establish a read lock) before this opcode is
                            // invoked.
                            Debug.Assert(op.P1 >= 0 && op.P1 < ctx.DBs.length);
                            Debug.Assert((BtreeMask & ((yDbMask)1 << op.P1)) != 0);
                            Debug.Assert(Btree.SchemaMutexHeld(ctx, op.P1, null));
                            Btree bt = ctx.DBs[op.P1].Bt;
                            uint meta = 0;
                            uint gen;
                            if (bt != null)
                            {
                                bt.GetMeta(Btree.META.SCHEMA_VERSION, ref meta);
                                gen = (uint)ctx.DBs[op.P1].Schema.Generation;
                            }
                            else
                                gen = meta = 0;
                            if (meta != op.P2 || gen != op.P3)
                            {
                                C._tagfree(ctx, ref ErrMsg);
                                ErrMsg = "database schema has changed";
                                // If the schema-cookie from the database file matches the cookie stored with the in-memory representation of the schema, do
                                // not reload the schema from the database file.
                                //
                                // If virtual-tables are in use, this is not just an optimization. Often, v-tables store their data in other SQLite tables, which
                                // are queried from within xNext() and other v-table methods using prepared queries. If such a query is out-of-date, we do not want to
                                // discard the database schema, as the user code implementing the v-table would have to be ready for the sqlite3_vtab structure itself
                                // to be invalidated whenever sqlite3_step() is called from within a v-table method.
                                if (ctx.DBs[op.P1].Schema.SchemaCookie != meta)
                                    Parse.ResetOneSchema(ctx, op.P1);

                                Expired = true;
                                rc = RC.SCHEMA;
                            }
                            break;
                        }
                    case OP.OpenRead:
                    case OP.OpenWrite:
                        {
                            // Opcode: OpenRead P1 P2 P3 P4 P5
                            //
                            // Open a read-only cursor for the database table whose root page is P2 in a database file.  The database file is determined by P3. 
                            // P3==0 means the main database, P3==1 means the database used for temporary tables, and P3>1 means used the corresponding attached
                            // database.  Give the new cursor an identifier of P1.  The P1 values need not be contiguous but all P1 values should be small integers.
                            // It is an error for P1 to be negative.
                            //
                            // If P5!=0 then use the content of register P2 as the root page, not the value of P2 itself.
                            //
                            // There will be a read lock on the database whenever there is an open cursor.  If the database was unlocked prior to this instruction
                            // then a read lock is acquired as part of this instruction.  A read lock allows other processes to read the database but prohibits
                            // any other process from modifying the database.  The read lock is released when all cursors are closed.  If this instruction attempts
                            // to get a read lock but fails, the script terminates with an SQLITE_BUSY error code.
                            //
                            // The P4 value may be either an integer (P4_INT32) or a pointer to a KeyInfo structure (P4_KEYINFO). If it is a pointer to a KeyInfo 
                            // structure, then said structure defines the content and collating sequence of the index being opened. Otherwise, if P4 is an integer 
                            // value, it is set to the number of columns in the table.
                            //
                            // See also OpenWrite.
                            //
                            // Opcode: OpenWrite P1 P2 P3 P4 P5
                            //
                            // Open a read/write cursor named P1 on the table or index whose root page is P2.  Or if P5!=0 use the content of register P2 to find the
                            // root page.
                            //
                            // The P4 value may be either an integer (P4_INT32) or a pointer to a KeyInfo structure (P4_KEYINFO). If it is a pointer to a KeyInfo 
                            // structure, then said structure defines the content and collating sequence of the index being opened. Otherwise, if P4 is an integer 
                            // value, it is set to the number of columns in the table, or to the largest index of any column of the table that is actually used.
                            //
                            // This instruction works just like OpenRead except that it opens the cursor in read/write mode.  For a given table, there can be one or more read-only
                            // cursors or a single read/write cursor but not both.
                            //
                            // See also OpenRead.
                            Debug.Assert(((OPFLAG)op.P5 & (OPFLAG.P2ISREG | OPFLAG.BULKCSR)) == (OPFLAG)op.P5);
                            Debug.Assert(op.Opcode == OP.OpenWrite || op.P5 == 0);
                            if (Expired)
                            {
                                rc = RC.ABORT;
                                break;
                            }

                            int fields = 0;
                            KeyInfo keyInfo = null;
                            int p2 = op.P2;
                            int db = op.P3;
                            Debug.Assert(db >= 0 && db < ctx.DBs.length);
                            Debug.Assert((BtreeMask & (((yDbMask)1) << db)) != 0);
                            Context.DB dbAsObj = ctx.DBs[db];
                            Btree x = dbAsObj.Bt;
                            Debug.Assert(x != null);
                            int wrFlag;
                            if (op.Opcode == OP.OpenWrite)
                            {
                                wrFlag = 1;
                                Debug.Assert(Btree.SchemaMutexHeld(ctx, db, null));
                                if (dbAsObj.Schema.FileFormat < MinWriteFileFormat)
                                    MinWriteFileFormat = dbAsObj.Schema.FileFormat;
                            }
                            else
                                wrFlag = 0;
                            if (((OPFLAG)op.P5 & OPFLAG.P2ISREG) != 0)
                            {
                                Debug.Assert(p2 > 0);
                                Debug.Assert(p2 <= Mems.length);
                                in2 = mems[p2];
                                Debug.Assert(E.MemIsValid(in2));
                                Debug.Assert((in2.Flags & MEM.Int) != 0);
                                MemIntegerify(in2);
                                p2 = (int)in2.u.I;
                                // The p2 value always comes from a prior OP_CreateTable opcode and that opcode will always set the p2 value to 2 or more or else fail.
                                // If there were a failure, the prepared statement would have halted before reaching this instruction.
                                if (C._NEVER(p2 < 2))
                                {
                                    rc = SysEx.CORRUPT_BKPT();
                                    goto abort_due_to_error;
                                }
                            }
                            if (op.P4Type == P4T.KEYINFO)
                            {
                                keyInfo = op.P4.KeyInfo;
                                keyInfo.Encode = E.CTXENCODE(ctx);
                                fields = keyInfo.Fields + 1;
                            }
                            else if (op.P4Type == P4T.INT32)
                                fields = op.P4.I;
                            Debug.Assert(op.P1 >= 0);
                            VdbeCursor cur = AllocateCursor(this, op.P1, fields, db, true);
                            if (cur == null) goto no_mem;
                            cur.NullRow = true;
                            cur.IsOrdered = true;
                            rc = x.Cursor(p2, wrFlag, keyInfo, cur.Cursor);
                            cur.KeyInfo = keyInfo;
                            Debug.Assert(OPFLAG.BULKCSR == BTREE_BULKLOAD);

                            // Since it performs no memory allocation or IO, the only value that sqlite3BtreeCursor() may return is SQLITE_OK.
                            Debug.Assert(rc == RC.OK);

                            // Set the VdbeCursor.isTable and isIndex variables. Previous versions of SQLite used to check if the root-page flags were sane at this point
                            // and report database corruption if they were not, but this check has since moved into the btree layer.
                            cur.IsTable = (op.P4Type != P4T.KEYINFO);
                            cur.IsIndex = !cur.IsTable;
                            break;
                        }
                    case OP.OpenAutoindex:
                    case OP.OpenEphemeral:
                        {
                            // Opcode: OpenEphemeral P1 P2 * P4 P5
                            //
                            // Open a new cursor P1 to a transient table. The cursor is always opened read/write even if 
                            // the main database is read-only.  The ephemeral table is deleted automatically when the cursor is closed.
                            //
                            // P2 is the number of columns in the ephemeral table. The cursor points to a BTree table if P4==0 and to a BTree index
                            // if P4 is not 0.  If P4 is not NULL, it points to a KeyInfo structure that defines the format of keys in the index.
                            //
                            // This opcode was once called OpenTemp.  But that created confusion because the term "temp table", might refer either
                            // to a TEMP table at the SQL level, or to a table opened by this opcode.  Then this opcode was call OpenVirtual.  But
                            // that created confusion with the whole virtual-table idea.
                            //
                            // The P5 parameter can be a mask of the BTREE_* flags defined in btree.h.  These flags control aspects of the operation of
                            // the btree.  The BTREE_OMIT_JOURNAL and BTREE_SINGLE flags are added automatically.
                            //
                            // Opcode: OpenAutoindex P1 P2 * P4 *
                            //
                            // This opcode works the same as OP_OpenEphemeral.  It has a different name to distinguish its use.  Tables created using
                            // by this opcode will be used for automatically created transient indices in joins.
                            const VSystem.OPEN vfsFlags = VSystem.OPEN.READWRITE | VSystem.OPEN.CREATE | VSystem.OPEN.EXCLUSIVE | VSystem.OPEN.DELETEONCLOSE | VSystem.OPEN.TRANSIENT_DB;
                            Debug.Assert(op.P1 >= 0);
                            VdbeCursor cx = AllocateCursor(this, op.P1, op.P2, -1, true);
                            if (cx == null)
                                goto no_mem;
                            cx.NullRow = true;
                            rc = Btree.Open(ctx.Vfs, null, ctx, ref cx.Bt, Btree.OPEN.OMIT_JOURNAL | Btree.OPEN.SINGLE | (Btree.OPEN)op.P5, vfsFlags);
                            if (rc == RC.OK)
                                rc = cx.Bt.BeginTrans(1);
                            if (rc == RC.OK)
                            {
                                // If a transient index is required, create it by calling sqlite3BtreeCreateTable() with the BTREE_BLOBKEY flag before
                                // opening it. If a transient table is required, just use the automatically created table with root-page 1 (an BLOB_INTKEY table).
                                if (op.P4.KeyInfo != null)
                                {
                                    Debug.Assert(op.P4Type == P4T.KEYINFO);
                                    int pgno = 0;
                                    rc = Btree.CreateTable(cx.Bt, ref pgno, BTREE_BLOBKEY);
                                    if (rc == RC.OK)
                                    {
                                        Debug.Assert(pgno == MASTER_ROOT + 1);
                                        rc = cx.Bt.Cursor(pgno, 1, op.P4.KeyInfo, cx.Cursor);
                                        cx.KeyInfo = op.P4.KeyInfo;
                                        cx.KeyInfo.Encode = E.CTXENCODE(ctx);
                                    }
                                    cx.IsTable = false;
                                }
                                else
                                {
                                    rc = cx.Bt.Cursor(MASTER_ROOT, 1, null, cx.Cursor);
                                    cx.IsTable = true;
                                }
                            }
                            cx.IsOrdered = (op.P5 != BTREE_UNORDERED);
                            cx.IsIndex = !cx.IsTable;
                            break;
                        }
                    case OP.SorterOpen:
                        {
                            // Opcode: SorterOpen P1 P2 * P4 *
                            //
                            // This opcode works like OP_OpenEphemeral except that it opens a transient index that is specifically designed to sort large
                            // tables using an external merge-sort algorithm.
                            VdbeCursor cur = AllocateCursor(this, op.P1, op.P2, -1, true);
                            if (cur == null) goto no_mem;
                            cur.KeyInfo = op.P4.KeyInfo;
                            cur.KeyInfo.Encode = E.CTXENCODE(ctx);
                            cur.IsSorter = true;
                            rc = SorterInit(ctx, cur);
                            break;
                        }
                    case OP.OpenPseudo:
                        {
                            // Opcode: OpenPseudo P1 P2 P3 * P5
                            //
                            // Open a new cursor that points to a fake table that contains a single row of data.  The content of that one row in the content of memory
                            // register P2 when P5==0.  In other words, cursor P1 becomes an alias for the MEM_Blob content contained in register P2.  When P5==1, then the
                            // row is represented by P3 consecutive registers beginning with P2.
                            //
                            // A pseudo-table created by this opcode is used to hold a single row output from the sorter so that the row can be decomposed into
                            // individual columns using the OP_Column opcode.  The OP_Column opcode is the only cursor opcode that works with a pseudo-table.
                            //
                            // P3 is the number of fields in the records that will be stored by the pseudo-table.
                            Debug.Assert(op.P1 >= 0);
                            VdbeCursor cur = AllocateCursor(this, op.P1, op.P3, -1, false);
                            if (cur == null) goto no_mem;
                            cur.NullRow = true;
                            cur.PseudoTableReg = op.P2;
                            cur.IsTable = true;
                            cur.IsIndex = false;
                            cur.MultiPseudo = op.P5;
                            break;
                        }
                    case OP.Close:
                        {
                            // Opcode: Close P1 * * * *
                            //
                            // Close a cursor previously opened as P1.  If P1 is not currently open, this instruction is a no-op.
                            Debug.Assert(op.P1 >= 0 && op.P1 < Cursors.length);
                            FreeCursor(Cursors[op.P1]);
                            Cursors[op.P1] = null;
                            break;
                        }



                    case OP.SeekLt: // jump, in3
                    case OP.SeekLe: // jump, in3
                    case OP.SeekGe: // jump, in3
                    case OP.SeekGt: // jump, in3
                        {
                            // Opcode: SeekGe P1 P2 P3 P4 *
                            //
                            // If cursor P1 refers to an SQL table (B-Tree that uses integer keys), use the value in register P3 as the key.  If cursor P1 refers 
                            // to an SQL index, then P3 is the first in an array of P4 registers that are used as an unpacked index key. 
                            //
                            // Reposition cursor P1 so that  it points to the smallest entry that is greater than or equal to the key value. If there are no records 
                            // greater than or equal to the key and P2 is not zero, then jump to P2.
                            //
                            // See also: Found, NotFound, Distinct, SeekLt, SeekGt, SeekLe
                            //
                            // Opcode: SeekGt P1 P2 P3 P4 *
                            //
                            // If cursor P1 refers to an SQL table (B-Tree that uses integer keys), use the value in register P3 as a key. If cursor P1 refers 
                            // to an SQL index, then P3 is the first in an array of P4 registers that are used as an unpacked index key. 
                            //
                            // Reposition cursor P1 so that  it points to the smallest entry that is greater than the key value. If there are no records greater than 
                            // the key and P2 is not zero, then jump to P2.
                            //
                            // See also: Found, NotFound, Distinct, SeekLt, SeekGe, SeekLe
                            //
                            // Opcode: SeekLt P1 P2 P3 P4 * 
                            //
                            // If cursor P1 refers to an SQL table (B-Tree that uses integer keys), use the value in register P3 as a key. If cursor P1 refers 
                            // to an SQL index, then P3 is the first in an array of P4 registers that are used as an unpacked index key. 
                            //
                            // Reposition cursor P1 so that  it points to the largest entry that is less than the key value. If there are no records less than 
                            // the key and P2 is not zero, then jump to P2.
                            //
                            // See also: Found, NotFound, Distinct, SeekGt, SeekGe, SeekLe
                            //
                            // Opcode: SeekLe P1 P2 P3 P4 *
                            //
                            // If cursor P1 refers to an SQL table (B-Tree that uses integer keys), use the value in register P3 as a key. If cursor P1 refers 
                            // to an SQL index, then P3 is the first in an array of P4 registers that are used as an unpacked index key. 
                            //
                            // Reposition cursor P1 so that it points to the largest entry that is less than or equal to the key value. If there are no records 
                            // less than or equal to the key and P2 is not zero, then jump to P2.
                            //
                            // See also: Found, NotFound, Distinct, SeekGt, SeekGe, SeekLt
                            int res = 0;
                            Debug.Assert(op.P1 >= 0 && op.P1 < Cursors.length);
                            Debug.Assert(op.P2 != 0);
                            VdbeCursor cur = Cursors[op.P1];
                            Debug.Assert(cur != null);
                            Debug.Assert(cur.PseudoTableReg == 0);
                            Debug.Assert(OP.SeekLe == OP.SeekLt + 1);
                            Debug.Assert(OP.SeekGe == OP.SeekLt + 2);
                            Debug.Assert(OP.SeekGt == OP.SeekLt + 3);
                            Debug.Assert(cur.IsOrdered);
                            UnpackedRecord r = new UnpackedRecord();
                            if (cur.Cursor != null)
                            {
                                OP oc = op.Opcode;
                                cur.NullRow = false;
                                if (cur.IsTable)
                                {
                                    // The input value in P3 might be of any type: integer, real, string, blob, or NULL.  But it needs to be an integer before we can do the seek, so covert it.
                                    in3 = mems[op.P3];
                                    ApplyNumericAffinity(in3);
                                    long keyId = IntValue(in3); // The rowid we are to seek to
                                    cur.RowidIsValid = false;

                                    // If the P3 value could not be converted into an integer without loss of information, then special processing is required...
                                    if ((in3.Flags & MEM.Int) == 0)
                                    {
                                        if ((in3.Flags & MEM.Real) == 0)
                                        {
                                            // If the P3 value cannot be converted into any kind of a number, then the seek is not possible, so jump to P2
                                            pc = op.P2 - 1;
                                            break;
                                        }
                                        // If we reach this point, then the P3 value must be a floating point number.
                                        Debug.Assert((in3.Flags & MEM.Real) != 0);

                                        if (keyId == long.MinValue && (in3.R < (double)keyId || in3.R > 0))
                                        {
                                            // The P3 value is too large in magnitude to be expressed as an integer.
                                            res = 1;
                                            if (in3.R < 0)
                                            {
                                                if (oc >= OP.SeekGe)
                                                {
                                                    Debug.Assert(oc == OP.SeekGe || oc == OP.SeekGt);
                                                    rc = Btree.First(cur.Cursor, ref res);
                                                    if (rc != RC.OK)
                                                        goto abort_due_to_error;
                                                }
                                            }
                                            else
                                            {
                                                if (oc <= OP.SeekLe)
                                                {
                                                    Debug.Assert(oc == OP.SeekLt || oc == OP.SeekLe);
                                                    rc = Btree.Last(cur.Cursor, ref res);
                                                    if (rc != RC.OK)
                                                        goto abort_due_to_error;
                                                }
                                            }
                                            if (res != 0)
                                                pc = op.P2 - 1;
                                            break;
                                        }
                                        else if (oc == OP.SeekLt || oc == OP.SeekGe)
                                        {
                                            // Use the ceiling() function to convert real.int
                                            if (in3.R > (double)keyId) keyId++;
                                        }
                                        else
                                        {
                                            // Use the floor() function to convert real.int
                                            Debug.Assert(oc == OP.SeekLe || oc == OP.SeekGt);
                                            if (in3.R < (double)keyId) keyId--;
                                        }
                                    }
                                    rc = Btree.MovetoUnpacked(cur.Cursor, null, keyId, 0, ref res);
                                    if (rc != RC.OK)
                                        goto abort_due_to_error;
                                    if (res == 0)
                                    {
                                        cur.RowidIsValid = true;
                                        cur.LastRowid = keyId;
                                    }
                                }
                                else
                                {
                                    int fields = op.P4.I;
                                    Debug.Assert(op.P4Type == P4T.INT32);
                                    Debug.Assert(fields > 0);
                                    r.KeyInfo = cur.KeyInfo;
                                    r.Fields = (ushort)fields;

                                    // The next line of code computes as follows, only faster:
                                    //   r.Flags = (oc == OP_SeekGt || oc == OP_SeekLe ? UNPACKED_INCRKEY : 0);
                                    r.Flags = (UNPACKED)((int)UNPACKED.INCRKEY * (1 & ((int)oc - (int)OP.SeekLt)));
                                    Debug.Assert(oc != OP.SeekGt || r.Flags == UNPACKED.INCRKEY);
                                    Debug.Assert(oc != OP.SeekLe || r.Flags == UNPACKED.INCRKEY);
                                    Debug.Assert(oc != OP.SeekGe || r.Flags == 0);
                                    Debug.Assert(oc != OP.SeekLt || r.Flags == 0);

                                    r.Mems = new Mem[r.Fields]; for (int i = 0; i < r.Fields; i++)
                                    {
                                        r.Mems[i] = mems[op.P3 + i]; //: r.mems = mems[op.P3];
#if DEBUG
                                        Debug.Assert(E.MemIsValid(r.Mems[i]));
#endif
                                    }
                                    E.ExpandBlob(r.Mems[0]);
                                    rc = Btree.MovetoUnpacked(cur.Cursor, r, 0, 0, ref res);
                                    if (rc != RC.OK)
                                        goto abort_due_to_error;
                                    cur.RowidIsValid = false;
                                }
                                cur.DeferredMoveto = false;
                                cur.CacheStatus = CACHE_STALE;
#if TEST
                                f_search_count++;
#endif
                                if (oc >= OP.SeekGe)
                                {
                                    Debug.Assert(oc == OP.SeekGe || oc == OP.SeekGt);
                                    if (res < 0 || (res == 0 && oc == OP.SeekGt))
                                    {
                                        rc = Btree.Next(cur.Cursor, ref res);
                                        if (rc != Rc.OK) goto abort_due_to_error;
                                        cur.RowidIsValid = false;
                                    }
                                    else
                                        res = 0;
                                }
                                else
                                {
                                    Debug.Assert(oc == OP.SeekLt || oc == OP.SeekLe);
                                    if (res > 0 || (res == 0 && oc == OP.SeekLt))
                                    {
                                        rc = Btree.Previous(cur.Cursor, ref res);
                                        if (rc != RC.OK) goto abort_due_to_error;
                                        cur.RowidIsValid = false;
                                    }
                                    else
                                        res = (Btree.Eof(cur.Cursor) ? 1 : 0); // res might be negative because the table is empty.  Check to see if this is the case.
                                }
                                Debug.Assert(op.P2 > 0);
                                if (res != 0)
                                    pc = op.P2 - 1;
                            }
                            else
                                // This happens when attempting to open the sqlite3_master table for read access returns SQLITE_EMPTY. In this case always
                                // take the jump (since there are no records in the table).
                                pc = op.P2 - 1;
                            break;
                        }
                    case OP.Seek: // in2
                        {
                            // Opcode: Seek P1 P2 * * *
                            //
                            // P1 is an open table cursor and P2 is a rowid integer.  Arrange for P1 to move so that it points to the rowid given by P2.
                            //
                            // This is actually a deferred seek.  Nothing actually happens until the cursor is used to read a record.  That way, if no reads
                            // occur, no unnecessary I/O happens.
                            Debug.Assert(op.P1 >= 0 && op.P1 < Cursors.length);
                            VdbeCursor cur = Cursors[op.P1];
                            Debug.Assert(C._ALWAYS(cur != null));
                            if (cur.Cursor != null)
                            {
                                Debug.Assert(cur.IsTable);
                                cur.NullRow = false;
                                in2 = mems[op.P2];
                                cur.MovetoTarget = IntValue(in2);
                                cur.RowidIsValid = false;
                                cur.DeferredMoveto = true;
                            }
                            break;
                        }
                    case OP.NotFound: // jump, in3
                    case OP.Found: // jump, in3
                        {
                            // Opcode: Found P1 P2 P3 P4 *
                            //
                            // If P4==0 then register P3 holds a blob constructed by MakeRecord.  If P4>0 then register P3 is the first of P4 registers that form an unpacked record.
                            //
                            // Cursor P1 is on an index btree.  If the record identified by P3 and P4 is a prefix of any entry in P1 then a jump is made to P2 and
                            // P1 is left pointing at the matching entry.
                            //
                            // Opcode: NotFound P1 P2 P3 P4 *
                            //
                            // If P4==0 then register P3 holds a blob constructed by MakeRecord.  If P4>0 then register P3 is the first of P4 registers that form an unpacked record.
                            // 
                            // Cursor P1 is on an index btree.  If the record identified by P3 and P4 is not the prefix of any entry in P1 then a jump is made to P2.  If P1 
                            // does contain an entry whose prefix matches the P3/P4 record then control falls through to the next instruction and P1 is left pointing at the matching entry.
                            //
                            // See also: Found, NotExists, IsUnique
                            int res = 0;
#if TEST
                            g_found_count++;
#endif
                            bool alreadyExists = false;
                            Debug.Assert(op.P1 >= 0 && op.P1 < Cursors.length);
                            Debug.Assert(op.P4Type == Vdbe.P4T.INT32);
                            VdbeCursor cur = Cursors[op.P1];
                            Debug.Assert(cur != null);
                            in3 = mems[op.P3];
                            if (C._ALWAYS(cur.Cursor != null))
                            {
                                Debug.Assert(!cur.IsTable);
                                UnpackedRecord idxKey;
                                UnpackedRecord r = new UnpackedRecord();
                                if (op.P4.I > 0)
                                {
                                    r.KeyInfo = cur.KeyInfo;
                                    r.Fields = (ushort)op.P4.I;
                                    r.Mems = new Mem[r.Fields]; for (int i = 0; i < r.Mems.Length; i++)
                                    {
                                        r.Mems[i] = mems[op.P3 + i]; //: r.mems = mems[op.P3];
#if DEBUG
                                        Debug.Assert(E.MemIsValid(r.Mems[i]));
#endif
                                    }
                                    r.Flags = UNPACKED.PREFIX_MATCH;
                                    idxKey = r;
                                }
                                else
                                {
                                    //UnpackedRecord tempRecs = new UnpackedRecord();
                                    idxKey = AllocUnpackedRecord(cur.KeyInfo);
                                    if (idxKey == null) goto no_mem;
                                    Debug.Assert((in3.Flags & MEM.Blob) != 0);
                                    Debug.Assert((in3.Flags & MEM.Zero) == 0); // zeroblobs already expanded
                                    idxKey = RecordUnpack(cur.KeyInfo, in3.N, in3.Z_, idxKey);
                                    idxKey.Flags |= UNPACKED.PREFIX_MATCH;
                                }
                                rc = Btree.MovetoUnpacked(cur.Cursor, idxKey, 0, 0, ref res);
                                //if (op.P4.I == 0)
                                //    DeleteUnpackedRecord(idxKey);
                                if (rc != RC.OK)
                                    break;
                                alreadyExists = (res == 0);
                                cur.DeferredMoveto = false;
                                cur.CacheStatus = CACHE_STALE;
                            }
                            if (op.Opcode == OP.Found)
                            {
                                if (alreadyExists) pc = op.P2 - 1;
                            }
                            else
                            {
                                if (!alreadyExists) pc = op.P2 - 1;
                            }
                            break;
                        }
                    case OP.IsUnique: // jump, in3
                        {
                            // Opcode: IsUnique P1 P2 P3 P4 *
                            //
                            // Cursor P1 is open on an index b-tree - that is to say, a btree which no data and where the key are records generated by OP_MakeRecord with
                            // the list field being the integer ROWID of the entry that the index entry refers to.
                            //
                            // The P3 register contains an integer record number. Call this record number R. Register P4 is the first in a set of N contiguous registers
                            // that make up an unpacked index key that can be used with cursor P1. The value of N can be inferred from the cursor. N includes the rowid
                            // value appended to the end of the index record. This rowid value may or may not be the same as R.
                            //
                            // If any of the N registers beginning with register P4 contains a NULL value, jump immediately to P2.
                            //
                            // Otherwise, this instruction checks if cursor P1 contains an entry where the first (N-1) fields match but the rowid value at the end
                            // of the index entry is not R. If there is no such entry, control jumps to instruction P2. Otherwise, the rowid of the conflicting index
                            // entry is copied to register P3 and control falls through to the next instruction.
                            //
                            // See also: NotFound, NotExists, Found
                            in3 = mems[op.P3];
                            // Assert that the values of parameters P1 and P4 are in range.
                            Debug.Assert(op.P4Type == P4T.INT32);
                            Debug.Assert(op.P4.I > 0 && op.P4.I <= Mems.length);
                            Debug.Assert(op.P1 >= 0 && op.P1 < Cursors.length);

                            // Find the index cursor.
                            VdbeCursor cur = Cursors[op.P1];
                            Debug.Assert(!cur.DeferredMoveto);
                            cur.SeekResult = 0;
                            cur.CacheStatus = CACHE_STALE;
                            Btree.BtCursor crsr = cur.Cursor;

                            // If any of the values are NULL, take the jump.
                            ushort fields = cur.KeyInfo.Fields;
                            Mem[] maxs = new Mem[fields + 1];
                            for (ushort ii = 0; ii < fields; ii++)
                            {
                                maxs[ii] = mems[op.P4.I + ii];
                                if ((maxs[ii].Flags & MEM.Null) != 0)
                                {
                                    pc = op.P2 - 1;
                                    crsr = null;
                                    break;
                                }
                            }
                            maxs[fields] = new Mem();
                            Debug.Assert((maxs[fields].Flags & MEM.Null) == 0);

                            if (crsr != null)
                            {
                                // Populate the index search key.
                                UnpackedRecord r = new UnpackedRecord(); // B-Tree index search key
                                r.KeyInfo = cur.KeyInfo;
                                r.Fields = (ushort)(fields + 1);
                                r.Flags = UNPACKED.PREFIX_SEARCH;
                                r.Mems = maxs;
#if DEBUG
                                for (int i = 0; i < r.Fields; i++) Debug.Assert(E.MemIsValid(r.Mems[i]));
#endif
                                // Extract the value of R from register P3.
                                MemIntegerify(in3);
                                long R = in3.u.I; // Rowid stored in register P3

                                // Search the B-Tree index. If no conflicting record is found, jump to P2. Otherwise, copy the rowid of the conflicting record to
                                // register P3 and fall through to the next instruction.
                                rc = Btree.MovetoUnpacked(crsr, r, 0, 0, ref cur.SeekResult);
                                if ((r.Flags & UNPACKED.PREFIX_SEARCH) != 0 || r.Rowid == R)
                                    pc = op.P2 - 1;
                                else
                                    in3.u.I = r.Rowid;
                            }
                            break;
                        }
                    case OP.NotExists: // jump, in3
                        {
                            // Opcode: NotExists P1 P2 P3 * *
                            //
                            // Use the content of register P3 as an integer key.  If a record with that key does not exist in table of P1, then jump to P2. 
                            // If the record does exist, then fall through.  The cursor is left pointing to the record if it exists.
                            //
                            // The difference between this operation and NotFound is that this operation assumes the key is an integer and that P1 is a table whereas
                            // NotFound assumes key is a blob constructed from MakeRecord and P1 is an index.
                            //
                            // See also: Found, NotFound, IsUnique
                            int res;
                            in3 = mems[op.P3];
                            Debug.Assert((in3.Flags & MEM.Int) != 0);
                            Debug.Assert(op.P1 >= 0 && op.P1 < Cursors.length);
                            VdbeCursor cur = Cursors[op.P1];
                            Debug.Assert(cur != null);
                            Debug.Assert(cur.IsTable);
                            Debug.Assert(cur.PseudoTableReg == 0);
                            Btree.BtCursor crsr = cur.Cursor;
                            if (crsr != null)
                            {
                                res = 0;
                                long keyId = in3.u.I;
                                rc = Btree.MovetoUnpacked(crsr, null, (long)keyId, 0, ref res);
                                cur.LastRowid = in3.u.I;
                                cur.RowidIsValid = (res == 0);
                                cur.NullRow = false;
                                cur.CacheStatus = CACHE_STALE;
                                cur.DeferredMoveto = false;
                                if (res != 0)
                                {
                                    pc = op.P2 - 1;
                                    Debug.Assert(!cur.RowidIsValid);
                                }
                                cur.SeekResult = res;
                            }
                            else
                            {
                                // This happens when an attempt to open a read cursor on the sqlite_master table returns SQLITE_EMPTY.
                                pc = op.P2 - 1;
                                Debug.Assert(!cur.RowidIsValid);
                                cur.SeekResult = 0;
                            }
                            break;
                        }
                    case OP.Sequence:// out2-prerelease
                        {
                            // Opcode: Sequence P1 P2 * * *
                            //
                            // Find the next available sequence number for cursor P1. Write the sequence number into register P2.
                            // The sequence number on the cursor is incremented after this instruction.  
                            Debug.Assert(op.P1 >= 0 && op.P1 < Cursors.length);
                            Debug.Assert(Cursors[op.P1] != null);
                            out_.u.I = (long)Cursors[op.P1].SeqCount++;
                            break;
                        }
                    case OP.NewRowid: // out2-prerelease
                        {
                            // Opcode: NewRowid P1 P2 P3 * *
                            //
                            // Get a new integer record number (a.k.a "rowid") used as the key to a table. The record number is not previously used as a key in the database
                            // table that cursor P1 points to.  The new record number is written written to register P2.
                            //
                            // If P3>0 then P3 is a register in the root frame of this VDBE that holds the largest previously generated record number. No new record numbers are
                            // allowed to be less than this value. When this value reaches its maximum, an SQLITE_FULL error is generated. The P3 register is updated with the '
                            // generated record number. This P3 mechanism is used to help implement the AUTOINCREMENT feature.
                            long v = 0; // The new rowid
                            int res = 0; // Result of an sqlite3BtreeLast()
                            Debug.Assert(op.P1 >= 0 && op.P1 < Cursors.length);
                            VdbeCursor cur = Cursors[op.P1]; // Cursor of table to get the new rowid
                            Debug.Assert(cur != null);
                            if (C._NEVER(cur.Cursor == null))
                            { } // The zero initialization above is all that is needed
                            else
                            {
#if _32BIT_ROWID                                
                                const int MAX_ROWID = int.MaxValue;
#else
                                const long MAX_ROWID = long.MaxValue; // Some compilers complain about constants of the form 0x7fffffffffffffff. Others complain about 0x7ffffffffffffffffLL.  The following macro seems to provide the constant while making all compilers happy.
#endif
                                // The next rowid or record number (different terms for the same thing) is obtained in a two-step algorithm.
                                //
                                // First we attempt to find the largest existing rowid and add one to that.  But if the largest existing rowid is already the maximum
                                // positive integer, we have to fall through to the second probabilistic algorithm
                                //
                                // The second algorithm is to select a rowid at random and see if it already exists in the table.  If it does not exist, we have
                                // succeeded.  If the random rowid does exist, we select a new one and try again, up to 100 times.
                                Debug.Assert(cur.IsTable);
                                if (!cur.UseRandomRowid)
                                {
                                    v = Btree.GetCachedRowid(cur.Cursor);
                                    if (v == 0)
                                    {
                                        rc = Btree.Last(cur.Cursor, ref res);
                                        if (rc != RC.OK)
                                            goto abort_due_to_error;
                                        if (res != 0)
                                            v = 1; // IMP: R-61914-48074
                                        else
                                        {
                                            Debug.Assert(Btree.CursorIsValid(cur.Cursor));
                                            rc = Btree.KeySize(cur.Cursor, ref v);
                                            Debug.Assert(rc == RC.OK); // Cannot fail following BtreeLast()
                                            if (v == MAX_ROWID)
                                                cur.UseRandomRowid = true;
                                            else
                                                v++; // IMP: R-29538-34987
                                        }
                                    }
#if !OMIT_AUTOINCREMENT
                                    if (op.P3 != 0)
                                    {
                                        // Assert that P3 is a valid memory cell.
                                        Debug.Assert(op.P3 > 0);
                                        Mem mem; // Register holding largest rowid for AUTOINCREMENT
                                        VdbeFrame frame; // Root frame of VDBE
                                        if (Frames != null)
                                        {
                                            for (frame = Frames; frame.Parent != null; frame = frame.Parent) ;
                                            // Assert that P3 is a valid memory cell.
                                            Debug.Assert(op.P3 <= frame.Mems.length);
                                            mem = frame.Mems[op.P3];
                                        }
                                        else
                                        {
                                            // Assert that P3 is a valid memory cell.
                                            Debug.Assert(op.P3 <= Mems.length);
                                            mem = mems[op.P3];
                                            MemAboutToChange(this, mem);
                                        }
                                        Debug.Assert(E.MemIsValid(mem));

                                        REGISTER_TRACE(op.P3, mem);
                                        MemIntegerify(mem);
                                        Debug.Assert((mem.Flags & MEM.Int) != 0); // mem(P3) holds an integer
                                        if (mem.u.I == MAX_ROWID || cur.UseRandomRowid)
                                        {
                                            rc = RC.FULL; // IMP: R-12275-61338
                                            goto abort_due_to_error;
                                        }
                                        if (v < (mem.u.I + 1))
                                            v = (int)(mem.u.I + 1);
                                        mem.u.I = (long)v;
                                    }
#endif

                                    Btree.SetCachedRowid(cur.Cursor, (v < MAX_ROWID ? v + 1 : 0));
                                }
                                if (cur.UseRandomRowid)
                                {
                                    // IMPLEMENTATION-OF: R-07677-41881 If the largest ROWID is equal to the largest possible integer (9223372036854775807) then the database
                                    // engine starts picking positive candidate ROWIDs at random until it finds one that is not previously used. */
                                    Debug.Assert(op.P3 == 0); // We cannot be in random rowid mode if this is an AUTOINCREMENT table. on the first attempt, simply do one more than previous
                                    v = lastRowid;
                                    v &= (MAX_ROWID >> 1); // ensure doesn't go negative
                                    v++; // ensure non-zero
                                    int cnt = 0; // Counter to limit the number of searches
                                    while ((rc = Btree.MovetoUnpacked(cur.Cursor, null, v, 0, ref res)) == RC.OK && res == 0 && ++cnt < 100)
                                    {
                                        // collision - try another random rowid
                                        SysEx.PutRandom(sizeof(long), ref v);
                                        if (cnt < 5)
                                            v &= 0xffffff; // try "small" random rowids for the initial attempts
                                        else
                                            v &= (MAX_ROWID >> 1); // ensure doesn't go negative
                                        v++; // ensure non-zero
                                    }
                                    if (rc == RC.OK && res == 0)
                                    {
                                        rc = RC.FULL; // IMP: R-38219-53002
                                        goto abort_due_to_error;
                                    }
                                    Debug.Assert(v > 0); // EV: R-40812-03570
                                }
                                cur.RowidIsValid = false;
                                cur.DeferredMoveto = false;
                                cur.CacheStatus = CACHE_STALE;
                            }
                            out_.u.I = (long)v;
                            break;
                        }


#if false
          case OP_Insert:
          case OP_InsertInt:
            {
              Mem pData;        /* MEM cell holding data for the record to be inserted */
              Mem pKey;         /* MEM cell holding key  for the record */
              i64 iKey;         /* The integer ROWID or key for the record to be inserted */
              VdbeCursor pC;    /* Cursor to table into which insert is written */
              int nZero;        /* Number of zero-bytes to append */
              int seekResult;   /* Result of prior seek or 0 if no USESEEKRESULT flag */
              string zDb;       /* database name - used by the update hook */
              string zTbl;      /* Table name - used by the opdate hook */
              int op;           /* Opcode for update hook: SQLITE_UPDATE or SQLITE_INSERT */

              pData = mems[op.P2];
              Debug.Assert( op.P1 >= 0 && op.P1 < p.nCursor );
              Debug.Assert( memIsValid( pData ) );
              pC = p.apCsr[op.P1];
              Debug.Assert( pC != null );
              Debug.Assert( pC.pCursor != null );
              Debug.Assert( pC.pseudoTableReg == 0 );
              Debug.Assert( pC.isTable );
              REGISTER_TRACE( p, op.P2, pData );

              if ( op.opcode == OP_Insert )
              {
                pKey = mems[op.P3];
                Debug.Assert( ( pKey.flags & MEM_Int ) != 0 );
                Debug.Assert( memIsValid( pKey ) );
                REGISTER_TRACE( p, op.P3, pKey );
                iKey = pKey.u.i;
              }
              else
              {
                Debug.Assert( op.opcode == OP_InsertInt );
                iKey = op.P3;
              }

              if ( ( op.P5 & OPFLAG_NCHANGE ) != 0 )
                p.nChange++;
              if ( ( op.P5 & OPFLAG_LASTROWID ) != 0 )
                ctx.lastRowid = lastRowid = iKey;
              if ( ( pData.flags & MEM_Null ) != 0 )
              {
                sqlite3_free( ref pData.zBLOB );
                pData.z = null;
                pData.n = 0;
              }
              else
              {
                Debug.Assert( ( pData.flags & ( MEM_Blob | MEM_Str ) ) != 0 );
              }
              seekResult = ( ( op.P5 & OPFLAG_USESEEKRESULT ) != 0 ? pC.seekResult : 0 );
              if ( ( pData.flags & MEM_Zero ) != 0 )
              {
                nZero = pData.u.nZero;
              }
              else
              {
                nZero = 0;
              }
              rc = sqlite3BtreeInsert( pC.pCursor, null, iKey,
              pData.zBLOB
              , pData.n, nZero,
              ( op.P5 & OPFLAG_APPEND ) != 0 ? 1 : 0, seekResult
              );

              pC.rowidIsValid = false;
              pC.deferredMoveto = false;
              pC.cacheStatus = CACHE_STALE;

              /* Invoke the update-hook if required. */
              if ( rc == SQLITE_OK && ctx.xUpdateCallback != null && op.P4.z != null )
              {
                zDb = ctx.aDb[pC.iDb].zName;
                zTbl = op.P4.z;
                op = ( ( op.P5 & OPFLAG_ISUPDATE ) != 0 ? SQLITE_UPDATE : SQLITE_INSERT );
                Debug.Assert( pC.isTable );
                ctx.xUpdateCallback( ctx.pUpdateArg, op, zDb, zTbl, iKey );
                Debug.Assert( pC.iDb >= 0 );
              }
              break;
            }

          /* Opcode: Delete P1 P2 * P4 *
          **
          ** Delete the record at which the P1 cursor is currently pointing.
          **
          ** The cursor will be left pointing at either the next or the previous
          ** record in the table. If it is left pointing at the next record, then
          ** the next Next instruction will be a no-op.  Hence it is OK to delete
          ** a record from within an Next loop.
          **
          ** If the OPFLAG_NCHANGE flag of P2 is set, then the row change count is
          ** incremented (otherwise not).
          **
          ** P1 must not be pseudo-table.  It has to be a real table with
          ** multiple rows.
          **
          ** If P4 is not NULL, then it is the name of the table that P1 is
          ** pointing to.  The update hook will be invoked, if it exists.
          ** If P4 is not NULL then the P1 cursor must have been positioned
          ** using OP_NotFound prior to invoking this opcode.
          */
          case OP_Delete:
            {
              i64 iKey;
              VdbeCursor pC;

              iKey = 0;
              Debug.Assert( op.P1 >= 0 && op.P1 < p.nCursor );
              pC = p.apCsr[op.P1];
              Debug.Assert( pC != null );
              Debug.Assert( pC.pCursor != null );  /* Only valid for real tables, no pseudotables */

              /* If the update-hook will be invoked, set iKey to the rowid of the
              ** row being deleted.
              */
              if ( ctx.xUpdateCallback != null && op.P4.z != null )
              {
                Debug.Assert( pC.isTable );
                Debug.Assert( pC.rowidIsValid );  /* lastRowid set by previous OP_NotFound */
                iKey = pC.lastRowid;
              }

              /* The OP_Delete opcode always follows an OP_NotExists or OP_Last or
              ** OP_Column on the same table without any intervening operations that
              ** might move or invalidate the cursor.  Hence cursor pC is always pointing
              ** to the row to be deleted and the sqlite3VdbeCursorMoveto() operation
              ** below is always a no-op and cannot fail.  We will run it anyhow, though,
              ** to guard against future changes to the code generator.
              **/
              Debug.Assert( pC.deferredMoveto == false );
              rc = sqlite3VdbeCursorMoveto( pC );
              if ( NEVER( rc != SQLITE_OK ) )
                goto abort_due_to_error;
              sqlite3BtreeSetCachedRowid( pC.pCursor, 0 );
              rc = sqlite3BtreeDelete( pC.pCursor );
              pC.cacheStatus = CACHE_STALE;

              /* Invoke the update-hook if required. */
              if ( rc == SQLITE_OK && ctx.xUpdateCallback != null && op.P4.z != null )
              {
                string zDb = ctx.aDb[pC.iDb].zName;
                string zTbl = op.P4.z;
                ctx.xUpdateCallback( ctx.pUpdateArg, SQLITE_DELETE, zDb, zTbl, iKey );
                Debug.Assert( pC.iDb >= 0 );
              }
              if ( ( op.P2 & OPFLAG_NCHANGE ) != 0 )
                p.nChange++;
              break;
            }

          /* Opcode: ResetCount P1 * *
          **
          ** The value of the change counter is copied to the database handle
          ** change counter (returned by subsequent calls to sqlite3_changes()).
          ** Then the VMs internal change counter resets to 0.
          ** This is used by trigger programs.
          */
          case OP_ResetCount:
            {
              sqlite3VdbeSetChanges( ctx, p.nChange );
              p.nChange = 0;
              break;
            }

          /* Opcode: RowData P1 P2 * * *
          **
          ** Write into register P2 the complete row data for cursor P1.
          ** There is no interpretation of the data.
          ** It is just copied onto the P2 register exactly as
          ** it is found in the database file.
          **
          ** If the P1 cursor must be pointing to a valid row (not a NULL row)
          ** of a real table, not a pseudo-table.
          */
          /* Opcode: RowKey P1 P2 * * *
          **
          ** Write into register P2 the complete row key for cursor P1.
          ** There is no interpretation of the data.
          ** The key is copied onto the P3 register exactly as
          ** it is found in the database file.
          **
          ** If the P1 cursor must be pointing to a valid row (not a NULL row)
          ** of a real table, not a pseudo-table.
          */
          case OP_RowKey:
          case OP_RowData:
            {
              VdbeCursor pC;
              BtCursor crsr;
              uint32 n;
              i64 n64;

              n = 0;
              n64 = 0;

              out_ = mems[op.P2];
              memAboutToChange( p, out_ );

              /* Note that RowKey and RowData are really exactly the same instruction */
              Debug.Assert( op.P1 >= 0 && op.P1 < p.nCursor );
              pC = p.apCsr[op.P1];
              Debug.Assert( pC.isTable || op.opcode == OP_RowKey );
              Debug.Assert( pC.isIndex || op.opcode == OP_RowData );
              Debug.Assert( pC != null );
              Debug.Assert( pC.nullRow == false );
              Debug.Assert( pC.pseudoTableReg == 0 );
              Debug.Assert( pC.pCursor != null );
              crsr = pC.pCursor;
              Debug.Assert( sqlite3BtreeCursorIsValid( crsr ) );

              /* The OP_RowKey and OP_RowData opcodes always follow OP_NotExists or
              ** OP_Rewind/Op_Next with no intervening instructions that might invalidate
              ** the cursor.  Hence the following sqlite3VdbeCursorMoveto() call is always
              ** a no-op and can never fail.  But we leave it in place as a safety.
              */
              Debug.Assert( pC.deferredMoveto == false );
              rc = sqlite3VdbeCursorMoveto( pC );
              if ( NEVER( rc != SQLITE_OK ) )
                goto abort_due_to_error;
              if ( pC.isIndex )
              {
                Debug.Assert( !pC.isTable );
                rc = sqlite3BtreeKeySize( crsr, ref n64 );
                Debug.Assert( rc == SQLITE_OK );    /* True because of CursorMoveto() call above */
                if ( n64 > ctx.aLimit[SQLITE_LIMIT_LENGTH] )
                {
                  goto too_big;
                }
                n = (uint32)n64;
              }
              else
              {
                rc = sqlite3BtreeDataSize( crsr, ref n );
                Debug.Assert( rc == SQLITE_OK );    /* DataSize() cannot fail */
                if ( n > (uint32)ctx.aLimit[SQLITE_LIMIT_LENGTH] )
                {
                  goto too_big;
                }
                if ( sqlite3VdbeMemGrow( out_, (int)n, 0 ) != 0 )
                {
                  goto no_mem;
                }
              }
              out_.n = (int)n;
              if ( pC.isIndex )
              {
                out_.zBLOB = sqlite3Malloc( (int)n );
                rc = sqlite3BtreeKey( crsr, 0, n, out_.zBLOB );
              }
              else
              {
                out_.zBLOB = sqlite3Malloc( (int)crsr.info.nData );
                rc = sqlite3BtreeData( crsr, 0, (uint32)n, out_.zBLOB );
              }
              MemSetTypeFlag( out_, MEM_Blob );
              out_.enc = SQLITE_UTF8;  /* In case the blob is ever cast to text */
#if SQLITE_TEST
              UPDATE_MAX_BLOBSIZE( out_ );
#endif
              break;
            }

          /* Opcode: Rowid P1 P2 * * *
          **
          ** Store in register P2 an integer which is the key of the table entry that
          ** P1 is currently point to.
          **
          ** P1 can be either an ordinary table or a virtual table.  There used to
          ** be a separate OP_VRowid opcode for use with virtual tables, but this
          ** one opcode now works for both table types.
          */
          case OP_Rowid:
            {                 /* out2-prerelease */
              VdbeCursor pC;
              i64 v;
              sqlite3_vtab pVtab;
              sqlite3_module pModule;

              v = 0;

              Debug.Assert( op.P1 >= 0 && op.P1 < p.nCursor );
              pC = p.apCsr[op.P1];
              Debug.Assert( pC != null );
              Debug.Assert( pC.pseudoTableReg == 0 );
              if ( pC.nullRow )
              {
                out_.flags = MEM_Null;
                break;
              }
              else if ( pC.deferredMoveto )
              {
                v = pC.movetoTarget;
#if !SQLITE_OMIT_VIRTUALTABLE
              }
              else if ( pC.pVtabCursor!=null )
              {
                pVtab = pC.pVtabCursor.pVtab;
                pModule = pVtab.pModule;
                Debug.Assert( pModule.xRowid != null );
                rc = pModule.xRowid( pC.pVtabCursor, out v );
                importVtabErrMsg( p, pVtab );
#endif //* SQLITE_OMIT_VIRTUALTABLE */
              }
              else
              {
                Debug.Assert( pC.pCursor != null );
                rc = sqlite3VdbeCursorMoveto( pC );
                if ( rc != 0 )
                  goto abort_due_to_error;
                if ( pC.rowidIsValid )
                {
                  v = pC.lastRowid;
                }
                else
                {
                  rc = sqlite3BtreeKeySize( pC.pCursor, ref v );
                  Debug.Assert( rc == SQLITE_OK );  /* Always so because of CursorMoveto() above */
                }
              }
              out_.u.i = (long)v;
              break;
            }

          /* Opcode: NullRow P1 * * * *
          **
          ** Move the cursor P1 to a null row.  Any OP_Column operations
          ** that occur while the cursor is on the null row will always
          ** write a NULL.
          */
          case OP_NullRow:
            {
              VdbeCursor pC;

              Debug.Assert( op.P1 >= 0 && op.P1 < p.nCursor );
              pC = p.apCsr[op.P1];
              Debug.Assert( pC != null );
              pC.nullRow = true;
              pC.rowidIsValid = false;
              if ( pC.pCursor != null )
              {
                sqlite3BtreeClearCursor( pC.pCursor );
              }
              break;
            }

          /* Opcode: Last P1 P2 * * *
          **
          ** The next use of the Rowid or Column or Next instruction for P1
          ** will refer to the last entry in the database table or index.
          ** If the table or index is empty and P2>0, then jump immediately to P2.
          ** If P2 is 0 or if the table or index is not empty, fall through
          ** to the following instruction.
          */
          case OP_Last:
            {        /* jump */
              VdbeCursor pC;
              BtCursor crsr;
              int res = 0;

              Debug.Assert( op.P1 >= 0 && op.P1 < p.nCursor );
              pC = p.apCsr[op.P1];
              Debug.Assert( pC != null );
              crsr = pC.pCursor;
              if ( crsr == null )
              {
                res = 1;
              }
              else
              {
                rc = sqlite3BtreeLast( crsr, ref res );
              }
              pC.nullRow = res == 1 ? true : false;
              pC.deferredMoveto = false;
              pC.rowidIsValid = false;
              pC.cacheStatus = CACHE_STALE;
              if ( op.P2 > 0 && res != 0 )
              {
                pc = op.P2 - 1;
              }
              break;
            }


          /* Opcode: Sort P1 P2 * * *
          **
          ** This opcode does exactly the same thing as OP_Rewind except that
          ** it increments an undocumented global variable used for testing.
          **
          ** Sorting is accomplished by writing records into a sorting index,
          ** then rewinding that index and playing it back from beginning to
          ** end.  We use the OP_Sort opcode instead of OP_Rewind to do the
          ** rewinding so that the global variable will be incremented and
          ** regression tests can determine whether or not the optimizer is
          ** correctly optimizing out sorts.
          */
          case OP_Sort:
            {        /* jump */
#if SQLITE_TEST
#if !TCLSH
              sqlite3_sort_count++;
              sqlite3_search_count--;
#else
              sqlite3_sort_count.iValue++;
              sqlite3_search_count.iValue--;
#endif
#endif
              p.aCounter[SQLITE_STMTSTATUS_SORT - 1]++;
              /* Fall through into OP_Rewind */
              goto case OP_Rewind;
            }
          /* Opcode: Rewind P1 P2 * * *
          **
          ** The next use of the Rowid or Column or Next instruction for P1
          ** will refer to the first entry in the database table or index.
          ** If the table or index is empty and P2>0, then jump immediately to P2.
          ** If P2 is 0 or if the table or index is not empty, fall through
          ** to the following instruction.
          */
          case OP_Rewind:
            {        /* jump */
              VdbeCursor pC;
              BtCursor crsr;
              int res = 0;

              Debug.Assert( op.P1 >= 0 && op.P1 < p.nCursor );
              pC = p.apCsr[op.P1];
              Debug.Assert( pC != null );
              res = 1;
              if ( ( crsr = pC.pCursor ) != null )
              {
                rc = sqlite3BtreeFirst( crsr, ref res );
                pC.atFirst = res == 0 ? true : false;
                pC.deferredMoveto = false;
                pC.cacheStatus = CACHE_STALE;
                pC.rowidIsValid = false;
              }
              pC.nullRow = res == 1 ? true : false;
              Debug.Assert( op.P2 > 0 && op.P2 < p.nOp );
              if ( res != 0 )
              {
                pc = op.P2 - 1;
              }
              break;
            }

          /* Opcode: Next P1 P2 * * P5
          **
          ** Advance cursor P1 so that it points to the next key/data pair in its
          ** table or index.  If there are no more key/value pairs then fall through
          ** to the following instruction.  But if the cursor advance was successful,
          ** jump immediately to P2.
          **
          ** The P1 cursor must be for a real table, not a pseudo-table.
          **
          ** See also: Prev
          */
          /* Opcode: Prev P1 P2 * * *
          **
          ** Back up cursor P1 so that it points to the previous key/data pair in its
          ** table or index.  If there is no previous key/value pairs then fall through
          ** to the following instruction.  But if the cursor backup was successful,
          ** jump immediately to P2.
          **
          ** The P1 cursor must be for a real table, not a pseudo-table.
          **
          ** If P5 is positive and the jump is taken, then event counter
          ** number P5-1 in the prepared statement is incremented.
          **
          */
          case OP_Prev:          /* jump */
          case OP_Next:
            {        /* jump */
              VdbeCursor pC;
              BtCursor crsr;
              int res;

              if ( ctx.u1.isInterrupted )
                goto abort_due_to_interrupt; //CHECK_FOR_INTERRUPT;
              Debug.Assert( op.P1 >= 0 && op.P1 < p.nCursor );
              Debug.Assert( op.P5 <= ArraySize( p.aCounter ) );
              pC = p.apCsr[op.P1];
              if ( pC == null )
              {
                break;  /* See ticket #2273 */
              }
              crsr = pC.pCursor;
              if ( crsr == null )
              {
                pC.nullRow = true;
                break;
              }
              res = 1;
              Debug.Assert( !pC.deferredMoveto );
              rc = op.opcode == OP_Next ? sqlite3BtreeNext( crsr, ref res ) :
              sqlite3BtreePrevious( crsr, ref res );
              pC.nullRow = res == 1 ? true : false;
              pC.cacheStatus = CACHE_STALE;
              if ( res == 0 )
              {
                pc = op.P2 - 1;
                if ( op.P5 != 0 )
                  p.aCounter[op.P5 - 1]++;
#if SQLITE_TEST
#if !TCLSH
                sqlite3_search_count++;
#else
                sqlite3_search_count.iValue++;
#endif
#endif
              }
              pC.rowidIsValid = false;
              break;
            }

          /* Opcode: IdxInsert P1 P2 P3 * P5
          **
          ** Register P2 holds an SQL index key made using the
          ** MakeRecord instructions.  This opcode writes that key
          ** into the index P1.  Data for the entry is nil.
          **
          ** P3 is a flag that provides a hint to the b-tree layer that this
          ** insert is likely to be an append.
          **
          ** This instruction only works for indices.  The equivalent instruction
          ** for tables is OP_Insert.
          */
          case OP_IdxInsert:
            {        /* in2 */
              VdbeCursor pC;
              BtCursor crsr;
              int nKey;
              byte[] zKey;
              Debug.Assert( op.P1 >= 0 && op.P1 < p.nCursor );
              pC = p.apCsr[op.P1];
              Debug.Assert( pC != null );
              in2 = mems[op.P2];
              Debug.Assert( ( in2.flags & MEM_Blob ) != 0 );
              crsr = pC.pCursor;
              if ( ALWAYS( crsr != null ) )
              {
                Debug.Assert( !pC.isTable );
                ExpandBlob( in2 );
                if ( rc == SQLITE_OK )
                {
                  nKey = in2.n;
                  zKey = ( in2.flags & MEM_Blob ) != 0 ? in2.zBLOB : Encoding.UTF8.GetBytes( in2.z );
                  rc = sqlite3BtreeInsert( crsr, zKey, nKey, null, 0, 0, ( op.P3 != 0 ) ? 1 : 0,
                  ( ( op.P5 & OPFLAG_USESEEKRESULT ) != 0 ? pC.seekResult : 0 )
                  );
                  Debug.Assert( !pC.deferredMoveto );
                  pC.cacheStatus = CACHE_STALE;
                }
              }
              break;
            }


          /* Opcode: IdxDelete P1 P2 P3 * *
          **
          ** The content of P3 registers starting at register P2 form
          ** an unpacked index key. This opcode removes that entry from the
          ** index opened by cursor P1.
          */
          case OP_IdxDelete:
            {
              VdbeCursor pC;
              BtCursor crsr;
              int res;
              UnpackedRecord r;

              res = 0;
              r = new UnpackedRecord();

              Debug.Assert( op.P3 > 0 );
              Debug.Assert( op.P2 > 0 && op.P2 + op.P3 <= Mems.length + 1 );
              Debug.Assert( op.P1 >= 0 && op.P1 < p.nCursor );
              pC = p.apCsr[op.P1];
              Debug.Assert( pC != null );
              crsr = pC.pCursor;
              if ( ALWAYS( crsr != null ) )
              {
                r.pKeyInfo = pC.pKeyInfo;
                r.nField = (u16)op.P3;
                r.flags = 0;
                r.mems = new Mem[r.nField];
                for ( int ra = 0; ra < r.nField; ra++ )
                {
                  r.mems[ra] = mems[op.P2 + ra];
#if SQLITE_DEBUG
                  Debug.Assert( memIsValid( r.mems[ra] ) );
#endif
                }
                rc = sqlite3BtreeMovetoUnpacked( crsr, r, 0, 0, ref res );
                if ( rc == SQLITE_OK && res == 0 )
                {
                  rc = sqlite3BtreeDelete( crsr );
                }
                Debug.Assert( !pC.deferredMoveto );
                pC.cacheStatus = CACHE_STALE;
              }
              break;
            }

          /* Opcode: IdxRowid P1 P2 * * *
          **
          ** Write into register P2 an integer which is the last entry in the record at
          ** the end of the index key pointed to by cursor P1.  This integer should be
          ** the rowid of the table entry to which this index entry points.
          **
          ** See also: Rowid, MakeRecord.
          */
          case OP_IdxRowid:
            {              /* out2-prerelease */
              BtCursor crsr;
              VdbeCursor pC;
              i64 rowid;

              rowid = 0;

              Debug.Assert( op.P1 >= 0 && op.P1 < p.nCursor );
              pC = p.apCsr[op.P1];
              Debug.Assert( pC != null );
              crsr = pC.pCursor;
              out_.flags = MEM_Null;
              if ( ALWAYS( crsr != null ) )
              {
                rc = sqlite3VdbeCursorMoveto( pC );
                if ( NEVER( rc != 0 ) )
                  goto abort_due_to_error;
                Debug.Assert( !pC.deferredMoveto );
                Debug.Assert( !pC.isTable );
                if ( !pC.nullRow )
                {
                  rc = sqlite3VdbeIdxRowid( ctx, crsr, ref rowid );
                  if ( rc != SQLITE_OK )
                  {
                    goto abort_due_to_error;
                  }
                  out_.u.i = rowid;
                  out_.flags = MEM_Int;
                }
              }
              break;
            }

          /* Opcode: IdxGE P1 P2 P3 P4 P5
          **
          ** The P4 register values beginning with P3 form an unpacked index
          ** key that omits the ROWID.  Compare this key value against the index
          ** that P1 is currently pointing to, ignoring the ROWID on the P1 index.
          **
          ** If the P1 index entry is greater than or equal to the key value
          ** then jump to P2.  Otherwise fall through to the next instruction.
          **
          ** If P5 is non-zero then the key value is increased by an epsilon
          ** prior to the comparison.  This make the opcode work like IdxGT except
          ** that if the key from register P3 is a prefix of the key in the cursor,
          ** the result is false whereas it would be true with IdxGT.
          */
          /* Opcode: IdxLT P1 P2 P3 P4 P5
          **
          ** The P4 register values beginning with P3 form an unpacked index
          ** key that omits the ROWID.  Compare this key value against the index
          ** that P1 is currently pointing to, ignoring the ROWID on the P1 index.
          **
          ** If the P1 index entry is less than the key value then jump to P2.
          ** Otherwise fall through to the next instruction.
          **
          ** If P5 is non-zero then the key value is increased by an epsilon prior
          ** to the comparison.  This makes the opcode work like IdxLE.
          */
          case OP_IdxLT:          /* jump */
          case OP_IdxGE:
            {        /* jump */
              VdbeCursor pC;
              int res;
              UnpackedRecord r;

              res = 0;
              r = new UnpackedRecord();

              Debug.Assert( op.P1 >= 0 && op.P1 < p.nCursor );
              pC = p.apCsr[op.P1];
              Debug.Assert( pC != null );
              Debug.Assert( pC.isOrdered );
              if ( ALWAYS( pC.pCursor != null ) )
              {
                Debug.Assert( pC.deferredMoveto == false );
                Debug.Assert( op.P5 == 0 || op.P5 == 1 );
                Debug.Assert( op.p4type == P4_INT32 );
                r.pKeyInfo = pC.pKeyInfo;
                r.nField = (u16)op.P4.i;
                if ( op.P5 != 0 )
                {
                  r.flags = UNPACKED_INCRKEY | UNPACKED_IGNORE_ROWID;
                }
                else
                {
                  r.flags = UNPACKED_IGNORE_ROWID;
                }
                r.mems = new Mem[r.nField];
                for ( int rI = 0; rI < r.nField; rI++ )
                {
                  r.mems[rI] = mems[op.P3 + rI];// r.mems = mems[op.P3];
#if SQLITE_DEBUG
                  Debug.Assert( memIsValid( r.mems[rI] ) );
#endif
                }
                rc = sqlite3VdbeIdxKeyCompare( pC, r, ref res );
                if ( op.opcode == OP_IdxLT )
                {
                  res = -res;
                }
                else
                {
                  Debug.Assert( op.opcode == OP_IdxGE );
                  res++;
                }
                if ( res > 0 )
                {
                  pc = op.P2 - 1;
                }
              }
              break;
            }

          /* Opcode: Destroy P1 P2 P3 * *
          **
          ** Delete an entire database table or index whose root page in the database
          ** file is given by P1.
          **
          ** The table being destroyed is in the main database file if P3==0.  If
          ** P3==1 then the table to be clear is in the auxiliary database file
          ** that is used to store tables create using CREATE TEMPORARY TABLE.
          **
          ** If AUTOVACUUM is enabled then it is possible that another root page
          ** might be moved into the newly deleted root page in order to keep all
          ** root pages contiguous at the beginning of the database.  The former
          ** value of the root page that moved - its value before the move occurred -
          ** is stored in register P2.  If no page
          ** movement was required (because the table being dropped was already
          ** the last one in the database) then a zero is stored in register P2.
          ** If AUTOVACUUM is disabled then a zero is stored in register P2.
          **
          ** See also: Clear
          */
          case OP_Destroy:
            {     /* out2-prerelease */
              int iMoved = 0;
              int iCnt;
              Vdbe pVdbe;
              int iDb;

#if !SQLITE_OMIT_VIRTUALTABLE
              iCnt = 0;
              for ( pVdbe = ctx.pVdbe; pVdbe != null; pVdbe = pVdbe.pNext )
              {
                if ( pVdbe.magic == VDBE_MAGIC_RUN && pVdbe.inVtabMethod < 2 && pVdbe.pc >= 0 )
                {
                  iCnt++;
                }
              }
#else
              iCnt = db.activeVdbeCnt;
#endif
              out_.flags = MEM_Null;
              if ( iCnt > 1 )
              {
                rc = SQLITE_LOCKED;
                p.errorAction = OE_Abort;
              }
              else
              {
                iDb = op.P3;
                Debug.Assert( iCnt == 1 );
                Debug.Assert( ( p.btreeMask & ( ( (yDbMask)1 ) << iDb ) ) != 0 );
                rc = sqlite3BtreeDropTable( ctx.aDb[iDb].pBt, op.P1, ref iMoved );
                out_.flags = MEM_Int;
                out_.u.i = iMoved;
#if !SQLITE_OMIT_AUTOVACUUM
                if ( rc == SQLITE_OK && iMoved != 0 )
                {
                  sqlite3RootPageMoved( ctx, iDb, iMoved, op.P1 );
                  /* All OP_Destroy operations occur on the same btree */
                  Debug.Assert( resetSchemaOnFault == 0 || resetSchemaOnFault == iDb + 1 );
                  resetSchemaOnFault = (u8)( iDb + 1 );
                }
#endif
              }
              break;
            }

          /* Opcode: Clear P1 P2 P3
          **
          ** Delete all contents of the database table or index whose root page
          ** in the database file is given by P1.  But, unlike Destroy, do not
          ** remove the table or index from the database file.
          **
          ** The table being clear is in the main database file if P2==0.  If
          ** P2==1 then the table to be clear is in the auxiliary database file
          ** that is used to store tables create using CREATE TEMPORARY TABLE.
          **
          ** If the P3 value is non-zero, then the table referred to must be an
          ** intkey table (an SQL table, not an index). In this case the row change
          ** count is incremented by the number of rows in the table being cleared.
          ** If P3 is greater than zero, then the value stored in register P3 is
          ** also incremented by the number of rows in the table being cleared.
          **
          ** See also: Destroy
          */
          case OP_Clear:
            {
              int nChange;

              nChange = 0;
              Debug.Assert( ( p.btreeMask & ( ( (yDbMask)1 ) << op.P2 ) ) != 0 );
              int iDummy0 = 0;
              if ( op.P3 != 0 )
                rc = sqlite3BtreeClearTable( ctx.aDb[op.P2].pBt, op.P1, ref nChange );
              else
                rc = sqlite3BtreeClearTable( ctx.aDb[op.P2].pBt, op.P1, ref iDummy0 );
              if ( op.P3 != 0 )
              {
                p.nChange += nChange;
                if ( op.P3 > 0 )
                {
                  Debug.Assert( memIsValid( mems[op.P3] ) );
                  memAboutToChange( p, mems[op.P3] );
                  mems[op.P3].u.i += nChange;
                }
              }
              break;
            }

          /* Opcode: CreateTable P1 P2 * * *
          **
          ** Allocate a new table in the main database file if P1==0 or in the
          ** auxiliary database file if P1==1 or in an attached database if
          ** P1>1.  Write the root page number of the new table into
          ** register P2
          **
          ** The difference between a table and an index is this:  A table must
          ** have a 4-byte integer key and can have arbitrary data.  An index
          ** has an arbitrary key but no data.
          **
          ** See also: CreateIndex
          */
          /* Opcode: CreateIndex P1 P2 * * *
          **
          ** Allocate a new index in the main database file if P1==0 or in the
          ** auxiliary database file if P1==1 or in an attached database if
          ** P1>1.  Write the root page number of the new table into
          ** register P2.
          **
          ** See documentation on OP_CreateTable for additional information.
          */
          case OP_CreateIndex:            /* out2-prerelease */
          case OP_CreateTable:
            {          /* out2-prerelease */
              int pgno;
              int flags;
              Db pDb;

              pgno = 0;
              Debug.Assert( op.P1 >= 0 && op.P1 < ctx.nDb );
              Debug.Assert( ( p.btreeMask & ( ( (yDbMask)1 ) << op.P1 ) ) != 0 );
              pDb = ctx.aDb[op.P1];
              Debug.Assert( pDb.pBt != null );
              if ( op.opcode == OP_CreateTable )
              {
                /* flags = BTREE_INTKEY; */
                flags = BTREE_INTKEY;
              }
              else
              {
                flags = BTREE_BLOBKEY;
              }
              rc = sqlite3BtreeCreateTable( pDb.pBt, ref pgno, flags );
              out_.u.i = pgno;
              break;
            }

          /* Opcode: ParseSchema P1 * * P4 *
          **
          ** Read and parse all entries from the SQLITE_MASTER table of database P1
          ** that match the WHERE clause P4. 
          **
          ** This opcode invokes the parser to create a new virtual machine,
          ** then runs the new virtual machine.  It is thus a re-entrant opcode.
          */
          case OP_ParseSchema:
            {
              int iDb;
              string zMaster;
              string zSql;
              InitData initData;

              /* Any prepared statement that invokes this opcode will hold mutexes
              ** on every btree.  This is a prerequisite for invoking
              ** sqlite3InitCallback().
              */
#if SQLITE_DEBUG
              for ( iDb = 0; iDb < db.nDb; iDb++ )
              {
                Debug.Assert( iDb == 1 || sqlite3BtreeHoldsMutex( db.aDb[iDb].pBt ) );
              }
#endif

              iDb = op.P1;
              Debug.Assert( iDb >= 0 && iDb < ctx.nDb );
              Debug.Assert( DbHasProperty( ctx, iDb, DB_SchemaLoaded ) );
              /* Used to be a conditional */
              {
                zMaster = E.SCHEMA_TABLE( iDb );
                initData = new InitData();
                initData.db = ctx;
                initData.iDb = op.P1;
                initData.pzErrMsg = p.zErrMsg;
                zSql = sqlite3MPrintf( ctx,
                "SELECT name, rootpage, sql FROM '%q'.%s WHERE %s ORDER BY rowid",
                ctx.aDb[iDb].zName, zMaster, op.P4.z );
                if ( String.IsNullOrEmpty( zSql ) )
                {
                  rc = SQLITE_NOMEM;
                }
                else
                {
                  Debug.Assert( 0 == ctx.init.busy );
                  ctx.init.busy = 1;
                  initData.rc = SQLITE_OK;
                  //Debug.Assert( 0 == db.mallocFailed );
                  rc = sqlite3_exec( ctx, zSql, (dxCallback)sqlite3InitCallback, (object)initData, 0 );
                  if ( rc == SQLITE_OK )
                    rc = initData.rc;
                  sqlite3DbFree( ctx, ref zSql );
                  ctx.init.busy = 0;
                }
              }
              if ( rc == SQLITE_NOMEM )
              {
                goto no_mem;
              }
              break;
            }

#if  !SQLITE_OMIT_ANALYZE
          /* Opcode: LoadAnalysis P1 * * * *
**
** Read the sqlite_stat1 table for database P1 and load the content
** of that table into the internal index hash table.  This will cause
** the analysis to be used when preparing all subsequent queries.
*/
          case OP_LoadAnalysis:
            {
              Debug.Assert( op.P1 >= 0 && op.P1 < ctx.nDb );
              rc = sqlite3AnalysisLoad( ctx, op.P1 );
              break;
            }
#endif // * !SQLITE_OMIT_ANALYZE) */

          /* Opcode: DropTable P1 * * P4 *
**
** Remove the internal (in-memory) data structures that describe
** the table named P4 in database P1.  This is called after a table
** is dropped in order to keep the internal representation of the
** schema consistent with what is on disk.
*/
          case OP_DropTable:
            {
              sqlite3UnlinkAndDeleteTable( ctx, op.P1, op.P4.z );
              break;
            }

          /* Opcode: DropIndex P1 * * P4 *
          **
          ** Remove the internal (in-memory) data structures that describe
          ** the index named P4 in database P1.  This is called after an index
          ** is dropped in order to keep the internal representation of the
          ** schema consistent with what is on disk.
          */
          case OP_DropIndex:
            {
              sqlite3UnlinkAndDeleteIndex( ctx, op.P1, op.P4.z );
              break;
            }

          /* Opcode: DropTrigger P1 * * P4 *
          **
          ** Remove the internal (in-memory) data structures that describe
          ** the trigger named P4 in database P1.  This is called after a trigger
          ** is dropped in order to keep the internal representation of the
          ** schema consistent with what is on disk.
          */
          case OP_DropTrigger:
            {
              sqlite3UnlinkAndDeleteTrigger( ctx, op.P1, op.P4.z );
              break;
            }


#if !SQLITE_OMIT_INTEGRITY_CHECK
          /* Opcode: IntegrityCk P1 P2 P3 * P5
**
** Do an analysis of the currently open database.  Store in
** register P1 the text of an error message describing any problems.
** If no problems are found, store a NULL in register P1.
**
** The register P3 contains the maximum number of allowed errors.
** At most reg(P3) errors will be reported.
** In other words, the analysis stops as soon as reg(P1) errors are
** seen.  Reg(P1) is updated with the number of errors remaining.
**
** The root page numbers of all tables in the database are integer
** stored in reg(P1), reg(P1+1), reg(P1+2), ....  There are P2 tables
** total.
**
** If P5 is not zero, the check is done on the auxiliary database
** file, not the main database file.
**
** This opcode is used to implement the integrity_check pragma.
*/
          case OP_IntegrityCk:
            {
              int nRoot;       /* Number of tables to check.  (Number of root pages.) */
              int[] aRoot = null;     /* Array of rootpage numbers for tables to be checked */
              int j;           /* Loop counter */
              int nErr = 0;    /* Number of errors reported */
              string z;        /* Text of the error report */
              Mem pnErr;       /* Register keeping track of errors remaining */

              nRoot = op.P2;
              Debug.Assert( nRoot > 0 );
              aRoot = sqlite3Malloc( aRoot, ( nRoot + 1 ) );// sqlite3DbMallocRaw(db, sizeof(int) * (nRoot + 1));
              if ( aRoot == null )
                goto no_mem;
              Debug.Assert( op.P3 > 0 && op.P3 <= Mems.length );
              pnErr = mems[op.P3];
              Debug.Assert( ( pnErr.flags & MEM_Int ) != 0 );
              Debug.Assert( ( pnErr.flags & ( MEM_Str | MEM_Blob ) ) == 0 );
              in1 = mems[op.P1];
              for ( j = 0; j < nRoot; j++ )
              {
                aRoot[j] = (int)sqlite3VdbeIntValue( p.mems[op.P1 + j] ); // in1[j]);
              }
              aRoot[j] = 0;
              Debug.Assert( op.P5 < ctx.nDb );
              Debug.Assert( ( p.btreeMask & ( ( (yDbMask)1 ) << op.P5 ) ) != 0 );
              z = sqlite3BtreeIntegrityCheck( ctx.aDb[op.P5].pBt, aRoot, nRoot,
              (int)pnErr.u.i, ref nErr );
              sqlite3DbFree( ctx, ref aRoot );
              pnErr.u.i -= nErr;
              sqlite3VdbeMemSetNull( in1 );
              if ( nErr == 0 )
              {
                Debug.Assert( z == "" );
              }
              else if ( String.IsNullOrEmpty( z ) )
              {
                goto no_mem;
              }
              else
              {
                sqlite3VdbeMemSetStr( in1, z, -1, SQLITE_UTF8, null ); //sqlite3_free );
              }
#if SQLITE_TEST
              UPDATE_MAX_BLOBSIZE( in1 );
#endif
              sqlite3VdbeChangeEncoding( in1, encoding );
              break;
            }
#endif // * SQLITE_OMIT_INTEGRITY_CHECK */

          /* Opcode: RowSetAdd P1 P2 * * *
**
** Insert the integer value held by register P2 into a boolean index
** held in register P1.
**
** An assertion fails if P2 is not an integer.
*/
          case OP_RowSetAdd:
            {       /* in1, in2 */
              in1 = mems[op.P1];
              in2 = mems[op.P2];
              Debug.Assert( ( in2.flags & MEM_Int ) != 0 );
              if ( ( in1.flags & MEM_RowSet ) == 0 )
              {
                sqlite3VdbeMemSetRowSet( in1 );
                if ( ( in1.flags & MEM_RowSet ) == 0 )
                  goto no_mem;
              }
              sqlite3RowSetInsert( in1.u.pRowSet, in2.u.i );
              break;
            }
          /* Opcode: RowSetRead P1 P2 P3 * *
          **
          ** Extract the smallest value from boolean index P1 and put that value into
          ** register P3.  Or, if boolean index P1 is initially empty, leave P3
          ** unchanged and jump to instruction P2.
          */
          case OP_RowSetRead:
            {       /* jump, in1, ref3 */
              i64 val = 0;
              if ( ctx.u1.isInterrupted )
                goto abort_due_to_interrupt; //CHECK_FOR_INTERRUPT;
              in1 = mems[op.P1];
              if ( ( in1.flags & MEM_RowSet ) == 0
              || sqlite3RowSetNext( in1.u.pRowSet, ref val ) == 0
              )
              {
                /* The boolean index is empty */
                sqlite3VdbeMemSetNull( in1 );
                pc = op.P2 - 1;
              }
              else
              {
                /* A value was pulled from the index */
                sqlite3VdbeMemSetInt64( mems[op.P3], val );
              }
              break;
            }

          /* Opcode: RowSetTest P1 P2 P3 P4
          **
          ** Register P3 is assumed to hold a 64-bit integer value. If register P1
          ** contains a RowSet object and that RowSet object contains
          ** the value held in P3, jump to register P2. Otherwise, insert the
          ** integer in P3 into the RowSet and continue on to the
          ** next opcode.
          **
          ** The RowSet object is optimized for the case where successive sets
          ** of integers, where each set contains no duplicates. Each set
          ** of values is identified by a unique P4 value. The first set
          ** must have P4==0, the final set P4=-1.  P4 must be either -1 or
          ** non-negative.  For non-negative values of P4 only the lower 4
          ** bits are significant.
          **
          ** This allows optimizations: (a) when P4==0 there is no need to test
          ** the rowset object for P3, as it is guaranteed not to contain it,
          ** (b) when P4==-1 there is no need to insert the value, as it will
          ** never be tested for, and (c) when a value that is part of set X is
          ** inserted, there is no need to search to see if the same value was
          ** previously inserted as part of set X (only if it was previously
          ** inserted as part of some other set).
          */
          case OP_RowSetTest:
            {                     /* jump, in1, in3 */
              int iSet;
              int exists;

              in1 = mems[op.P1];
              in3 = mems[op.P3];
              iSet = op.P4.i;
              Debug.Assert( ( in3.flags & MEM_Int ) != 0 );

              /* If there is anything other than a rowset object in memory cell P1,
              ** delete it now and initialize P1 with an empty rowset
              */
              if ( ( in1.flags & MEM_RowSet ) == 0 )
              {
                sqlite3VdbeMemSetRowSet( in1 );
                if ( ( in1.flags & MEM_RowSet ) == 0 )
                  goto no_mem;
              }

              Debug.Assert( op.p4type == P4_INT32 );
              Debug.Assert( iSet == -1 || iSet >= 0 );
              if ( iSet != 0 )
              {
                exists = sqlite3RowSetTest( in1.u.pRowSet,
                (u8)( iSet >= 0 ? iSet & 0xf : 0xff ),
                in3.u.i );
                if ( exists != 0 )
                {
                  pc = op.P2 - 1;
                  break;
                }
              }
              if ( iSet >= 0 )
              {
                sqlite3RowSetInsert( in1.u.pRowSet, in3.u.i );
              }
              break;
            }

#if !SQLITE_OMIT_TRIGGER

          /* Opcode: Program P1 P2 P3 P4 *
**
** Execute the trigger program passed as P4 (type P4_SUBPROGRAM). 
**
** P1 contains the address of the memory cell that contains the first memory 
** cell in an array of values used as arguments to the sub-program. P2 
** contains the address to jump to if the sub-program throws an IGNORE 
** exception using the RAISE() function. Register P3 contains the address 
** of a memory cell in this (the parent) VM that is used to allocate the 
** memory required by the sub-vdbe at runtime.
**
** P4 is a pointer to the VM containing the trigger program.
*/
          case OP_Program:
            {        /* jump */
              int nMem;              /* Number of memory registers for sub-program */
              int nByte;             /* Bytes of runtime space required for sub-program */
              Mem pRt;               /* Register to allocate runtime space */
              Mem pMem = null;       /* Used to iterate through memory cells */
              //Mem pEnd;            /* Last memory cell in new array */
              VdbeFrame pFrame;      /* New vdbe frame to execute in */
              SubProgram pProgram;   /* Sub-program to execute */
              int t;                 /* Token identifying trigger */

              pProgram = op.P4.pProgram;
              pRt = mems[op.P3];
              Debug.Assert( memIsValid( pRt ) );
              Debug.Assert( pProgram.nOp > 0 );

              /* If the p5 flag is clear, then recursive invocation of triggers is 
              ** disabled for backwards compatibility (p5 is set if this sub-program
              ** is really a trigger, not a foreign key action, and the flag set
              ** and cleared by the "PRAGMA recursive_triggers" command is clear).
              ** 
              ** It is recursive invocation of triggers, at the SQL level, that is 
              ** disabled. In some cases a single trigger may generate more than one 
              ** SubProgram (if the trigger may be executed with more than one different 
              ** ON CONFLICT algorithm). SubProgram structures associated with a
              ** single trigger all have the same value for the SubProgram.token 
              ** variable.  */
              if ( op.P5 != 0 )
              {
                t = pProgram.token;
                for ( pFrame = p.pFrame; pFrame != null && pFrame.token != t; pFrame = pFrame.pParent )
                  ;
                if ( pFrame != null )
                  break;
              }

              if ( p.nFrame >= ctx.aLimit[SQLITE_LIMIT_TRIGGER_DEPTH] )
              {
                rc = SQLITE_ERROR;
                C._setstring( ref p.zErrMsg, ctx, "too many levels of trigger recursion" );
                break;
              }

              /* Register pRt is used to store the memory required to save the state
              ** of the current program, and the memory required at runtime to execute
              ** the trigger program. If this trigger has been fired before, then pRt 
              ** is already allocated. Otherwise, it must be initialized.  */
              if ( ( pRt.flags & MEM_Frame ) == 0 )
              {
                /* SubProgram.nMem is set to the number of memory cells used by the 
                ** program stored in SubProgram.aOp. As well as these, one memory
                ** cell is required for each cursor used by the program. Set local
                ** variable nMem (and later, VdbeFrame.nChildMem) to this value.
                */
                nMem = pProgram.nMem + pProgram.nCsr;
                //nByte = ROUND8( sizeof( VdbeFrame ) )
                //+ nMem * sizeof( Mem )
                //+ pProgram.nCsr * sizeof( VdbeCursor* );
                pFrame = new VdbeFrame();// sqlite3DbMallocZero( db, nByte );
                //if ( !pFrame )
                //{
                //  goto no_mem;
                //}
                sqlite3VdbeMemRelease( pRt );
                pRt.flags = MEM_Frame;
                pRt.u.pFrame = pFrame;

                pFrame.v = p;
                pFrame.nChildMem = nMem;
                pFrame.nChildCsr = pProgram.nCsr;
                pFrame.pc = pc;
                pFrame.mems = p.mems;
                pFrame.nMem = Mems.length;
                pFrame.apCsr = p.apCsr;
                pFrame.nCursor = p.nCursor;
                pFrame.aOp = p.aOp;
                pFrame.nOp = p.nOp;
                pFrame.token = pProgram.token;

                // &VdbeFrameMem( pFrame )[pFrame.nChildMem];
                // mems is 1 based, so allocate 1 extra cell under C#
                pFrame.aChildMem = new Mem[pFrame.nChildMem + 1];
                for ( int i = 0; i < pFrame.aChildMem.Length; i++ )//pMem = VdbeFrameMem( pFrame ) ; pMem != pEnd ; pMem++ )
                {
                  //pFrame.mems[i] = pFrame.mems[pFrame.nMem+i];
                  pMem = sqlite3Malloc( pMem );
                  pMem.flags = MEM_Null;
                  pMem.db = ctx;
                  pFrame.aChildMem[i] = pMem;
                }
                pFrame.aChildCsr = new VdbeCursor[pFrame.nChildCsr];
                for ( int i = 0; i < pFrame.nChildCsr; i++ )
                  pFrame.aChildCsr[i] = new VdbeCursor();
              }
              else
              {
                pFrame = pRt.u.pFrame;
                Debug.Assert( pProgram.nMem + pProgram.nCsr == pFrame.nChildMem );
                Debug.Assert( pProgram.nCsr == pFrame.nChildCsr );
                Debug.Assert( pc == pFrame.pc );
              }

              p.nFrame++;
              pFrame.pParent = p.pFrame;
              pFrame.lastRowid = lastRowid;
              pFrame.nChange = p.nChange;
              p.nChange = 0;
              p.pFrame = pFrame;
              p.mems = mems = pFrame.aChildMem; // &VdbeFrameMem( pFrame )[-1];
              Mems.length = pFrame.nChildMem;
              p.nCursor = (u16)pFrame.nChildCsr;
              p.apCsr = pFrame.aChildCsr;// (VdbeCursor *)&mems[p->nMem+1];
              p.aOp = ops = pProgram.aOp;
              p.nOp = pProgram.nOp;
              pc = -1;

              break;
            }

          /* Opcode: Param P1 P2 * * *
          **
          ** This opcode is only ever present in sub-programs called via the 
          ** OP_Program instruction. Copy a value currently stored in a memory 
          ** cell of the calling (parent) frame to cell P2 in the current frames 
          ** address space. This is used by trigger programs to access the new.* 
          ** and old.* values.
          **
          ** The address of the cell in the parent frame is determined by adding
          ** the value of the P1 argument to the value of the P1 argument to the
          ** calling OP_Program instruction.
          */
          case OP_Param:
            {           /* out2-prerelease */
              VdbeFrame pFrame;
              Mem pIn;
              pFrame = p.pFrame;
              pIn = pFrame.mems[op.P1 + pFrame.aOp[pFrame.pc].p1];
              sqlite3VdbeMemShallowCopy( out_, pIn, MEM_Ephem );
              break;
            }
#endif // * #if !SQLITE_OMIT_TRIGGER */

#if !SQLITE_OMIT_FOREIGN_KEY
          /* Opcode: FkCounter P1 P2 * * *
**
** Increment a "constraint counter" by P2 (P2 may be negative or positive).
** If P1 is non-zero, the database constraint counter is incremented 
** (deferred foreign key constraints). Otherwise, if P1 is zero, the 
** statement counter is incremented (immediate foreign key constraints).
*/
          case OP_FkCounter:
            {
              if ( op.P1 != 0 )
              {
                ctx.nDeferredCons += op.P2;
              }
              else
              {
                p.nFkConstraint += op.P2;
              }
              break;
            }

          /* Opcode: FkIfZero P1 P2 * * *
          **
          ** This opcode tests if a foreign key constraint-counter is currently zero.
          ** If so, jump to instruction P2. Otherwise, fall through to the next 
          ** instruction.
          **
          ** If P1 is non-zero, then the jump is taken if the database constraint-counter
          ** is zero (the one that counts deferred constraint violations). If P1 is
          ** zero, the jump is taken if the statement constraint-counter is zero
          ** (immediate foreign key constraint violations).
          */
          case OP_FkIfZero:
            {         /* jump */
              if ( op.P1 != 0 )
              {
                if ( ctx.nDeferredCons == 0 )
                  pc = op.P2 - 1;
              }
              else
              {
                if ( p.nFkConstraint == 0 )
                  pc = op.P2 - 1;
              }
              break;
            }
#endif //* #if !SQLITE_OMIT_FOREIGN_KEY */

#if !SQLITE_OMIT_AUTOINCREMENT
          /* Opcode: MemMax P1 P2 * * *
**
** P1 is a register in the root frame of this VM (the root frame is
** different from the current frame if this instruction is being executed
** within a sub-program). Set the value of register P1 to the maximum of 
** its current value and the value in register P2.
**
** This instruction throws an error if the memory cell is not initially
** an integer.
*/
          case OP_MemMax:
            {        /* in2 */
              Mem _pIn1;
              VdbeFrame pFrame;
              if ( p.pFrame != null )
              {
                for ( pFrame = p.pFrame; pFrame.pParent != null; pFrame = pFrame.pParent )
                  ;
                _pIn1 = pFrame.mems[op.P1];
              }
              else
              {
                _pIn1 = mems[op.P1];
              }
              Debug.Assert( memIsValid( _pIn1 ) );
              sqlite3VdbeMemIntegerify( _pIn1 );
              in2 = mems[op.P2];
              sqlite3VdbeMemIntegerify( in2 );
              if ( _pIn1.u.i < in2.u.i )
              {
                _pIn1.u.i = in2.u.i;
              }
              break;
            }
#endif // * SQLITE_OMIT_AUTOINCREMENT */

          /* Opcode: IfPos P1 P2 * * *
**
** If the value of register P1 is 1 or greater, jump to P2.
**
** It is illegal to use this instruction on a register that does
** not contain an integer.  An Debug.Assertion fault will result if you try.
*/
          case OP_IfPos:
            {        /* jump, in1 */
              in1 = mems[op.P1];
              Debug.Assert( ( in1.flags & MEM_Int ) != 0 );
              if ( in1.u.i > 0 )
              {
                pc = op.P2 - 1;
              }
              break;
            }

          /* Opcode: IfNeg P1 P2 * * *
          **
          ** If the value of register P1 is less than zero, jump to P2.
          **
          ** It is illegal to use this instruction on a register that does
          ** not contain an integer.  An Debug.Assertion fault will result if you try.
          */
          case OP_IfNeg:
            {        /* jump, in1 */
              in1 = mems[op.P1];
              Debug.Assert( ( in1.flags & MEM_Int ) != 0 );
              if ( in1.u.i < 0 )
              {
                pc = op.P2 - 1;
              }
              break;
            }

          /* Opcode: IfZero P1 P2 P3 * *
          **
          ** The register P1 must contain an integer.  Add literal P3 to the
          ** value in register P1.  If the result is exactly 0, jump to P2. 
          **
          ** It is illegal to use this instruction on a register that does
          ** not contain an integer.  An assertion fault will result if you try.
          */
          case OP_IfZero:
            {        /* jump, in1 */
              in1 = mems[op.P1];
              Debug.Assert( ( in1.flags & MEM_Int ) != 0 );
              in1.u.i += op.P3;
              if ( in1.u.i == 0 )
              {
                pc = op.P2 - 1;
              }
              break;
            }

          /* Opcode: AggStep * P2 P3 P4 P5
          **
          ** Execute the step function for an aggregate.  The
          ** function has P5 arguments.   P4 is a pointer to the FuncDef
          ** structure that specifies the function.  Use register
          ** P3 as the accumulator.
          **
          ** The P5 arguments are taken from register P2 and its
          ** successors.
          */
          case OP_AggStep:
            {
              int n;
              int i;
              Mem pMem;
              Mem pRec;
              sqlite3_context ctx = new sqlite3_context();
              sqlite3_value[] apVal;

              n = op.P5;
              Debug.Assert( n >= 0 );
              //pRec = mems[op.P2];
              apVal = p.apArg;
              Debug.Assert( apVal != null || n == 0 );
              for ( i = 0; i < n; i++ )//, pRec++)
              {
                pRec = mems[op.P2 + i];
                Debug.Assert( memIsValid( pRec ) );
                apVal[i] = pRec;
                memAboutToChange( p, pRec );
                sqlite3VdbeMemStoreType( pRec );
              }
              ctx.pFunc = op.P4.pFunc;
              Debug.Assert( op.P3 > 0 && op.P3 <= Mems.length );
              ctx.pMem = pMem = mems[op.P3];
              pMem.n++;
              ctx.s.flags = MEM_Null;
              ctx.s.z = null;
              //ctx.s.zMalloc = null;
              ctx.s.xDel = null;
              ctx.s.db = ctx;
              ctx.isError = 0;
              ctx.pColl = null;
              if ( ( ctx.pFunc.flags & SQLITE_FUNC_NEEDCOLL ) != 0 )
              {
                Debug.Assert( pc > 0 );//op > p.aOp );
                Debug.Assert( p.aOp[pc - 1].p4type == P4_COLLSEQ ); //op[-1].p4type == P4_COLLSEQ );
                Debug.Assert( p.aOp[pc - 1].opcode == OP_CollSeq ); // op[-1].opcode == OP_CollSeq );
                ctx.pColl = p.aOp[pc - 1].p4.pColl;
                ;// op[-1].p4.pColl;
              }
              ctx.pFunc.xStep( ctx, n, apVal ); /* IMP: R-24505-23230 */
              if ( ctx.isError != 0 )
              {
                C._setstring( ref p.zErrMsg, ctx, sqlite3_value_text( ctx.s ) );
                rc = ctx.isError;
              }
              sqlite3VdbeMemRelease( ctx.s );
              break;
            }

          /* Opcode: AggFinal P1 P2 * P4 *
          **
          ** Execute the finalizer function for an aggregate.  P1 is
          ** the memory location that is the accumulator for the aggregate.
          **
          ** P2 is the number of arguments that the step function takes and
          ** P4 is a pointer to the FuncDef for this function.  The P2
          ** argument is not used by this opcode.  It is only there to disambiguate
          ** functions that can take varying numbers of arguments.  The
          ** P4 argument is only needed for the degenerate case where
          ** the step function was not previously called.
          */
          case OP_AggFinal:
            {
              Mem pMem;
              Debug.Assert( op.P1 > 0 && op.P1 <= Mems.length );
              pMem = mems[op.P1];
              Debug.Assert( ( pMem.flags & ~( MEM_Null | MEM_Agg ) ) == 0 );
              rc = sqlite3VdbeMemFinalize( pMem, op.P4.pFunc );
              p.mems[op.P1] = pMem;
              if ( rc != 0 )
              {
                C._setstring( ref p.zErrMsg, ctx, sqlite3_value_text( pMem ) );
              }
              sqlite3VdbeChangeEncoding( pMem, encoding );
#if SQLITE_TEST
              UPDATE_MAX_BLOBSIZE( pMem );
#endif
              if ( sqlite3VdbeMemTooBig( pMem ) )
              {
                goto too_big;
              }
              break;
            }


#if !SQLITE_OMIT_WAL
/* Opcode: Checkpoint P1 P2 P3 * *
**
** Checkpoint database P1. This is a no-op if P1 is not currently in
** WAL mode. Parameter P2 is one of SQLITE_CHECKPOINT_PASSIVE, FULL
** or RESTART.  Write 1 or 0 into mem[P3] if the checkpoint returns
** SQLITE_BUSY or not, respectively.  Write the number of pages in the
** WAL after the checkpoint into mem[P3+1] and the number of pages
** in the WAL that have been checkpointed after the checkpoint
** completes into mem[P3+2].  However on an error, mem[P3+1] and
** mem[P3+2] are initialized to -1.
*/
cDebug.Ase OP_Checkpoint: {
  aRes[0] = 0;
  aRes[1] = aRes[2] = -1;
  Debug.Assert( op.P2==SQLITE_CHECKPOINT_PDebug.AsSIVE
       || op.P2==SQLITE_CHECKPOINT_FULL
       || op.P2==SQLITE_CHECKPOINT_RESTART
  );
  rc = sqlite3Checkpoint(ctx, op.P1, op.P2, ref aRes[1], ref aRes[2]);
  if( rc==SQLITE_BUSY ){
    rc = SQLITE_OK;
    aRes[0] = 1;
  }
  for(i=0, pMem = mems[op.P3]; i<3; i++, pMem++){
    sqlite3VdbeMemSetInt64(pMem, (i64)aRes[i]);
  }
  break;
};  
#endif

#if !SQLITE_OMIT_PRAGMA
          /* Opcode: JournalMode P1 P2 P3 * P5
**
** Change the journal mode of database P1 to P3. P3 must be one of the
** PAGER_JOURNALMODE_XXX values. If changing between the various rollback
** modes (delete, truncate, persist, off and memory), this is a simple
** operation. No IO is required.
**
** If changing into or out of WAL mode the procedure is more complicated.
**
** Write a string containing the final journal-mode to register P2.
*/
          case OP_JournalMode:
            {    /* out2-prerelease */
              Btree pBt;                      /* Btree to change journal mode of */
              Pager pPager;                   /* Pager associated with pBt */
              int eNew;                       /* New journal mode */
              int eOld;                       /* The old journal mode */
              string zFilename;               /* Name of database file for pPager */

              eNew = op.P3;
              Debug.Assert( eNew == PAGER_JOURNALMODE_DELETE
              || eNew == PAGER_JOURNALMODE_TRUNCATE
              || eNew == PAGER_JOURNALMODE_PERSIST
              || eNew == PAGER_JOURNALMODE_OFF
              || eNew == PAGER_JOURNALMODE_MEMORY
              || eNew == PAGER_JOURNALMODE_WAL
              || eNew == PAGER_JOURNALMODE_QUERY
              );
              Debug.Assert( op.P1 >= 0 && op.P1 < ctx.nDb );

              pBt = ctx.aDb[op.P1].pBt;
              pPager = sqlite3BtreePager( pBt );
              eOld = sqlite3PagerGetJournalMode( pPager );
              if ( eNew == PAGER_JOURNALMODE_QUERY )
                eNew = eOld;
              if ( 0 == sqlite3PagerOkToChangeJournalMode( pPager ) )
                eNew = eOld;

#if !SQLITE_OMIT_WAL
zFilename = sqlite3PagerFilename(pPager);

/* Do not allow a transition to journal_mode=WAL for a database
** in temporary storage or if the VFS does not support shared memory 
*/
if( eNew==PAGER_JOURNALMODE_WAL
&& (zFilename[0]==0                         /* Temp file */
|| !sqlite3PagerWalSupported(pPager))   /* No shared-memory support */
){
eNew = eOld;
}

if( (eNew!=eOld)
&& (eOld==PAGER_JOURNALMODE_WAL || eNew==PAGER_JOURNALMODE_WAL)
){
if( null==ctx.autoCommit || ctx.activeVdbeCnt>1 ){
rc = SQLITE_ERROR;
C._setstring(&p.zErrMsg, ctx, 
"cannot change %s wal mode from within a transaction",
(eNew==PAGER_JOURNALMODE_WAL ? "into" : "out of")
);
break;
}else{

if( eOld==PAGER_JOURNALMODE_WAL ){
/* If leaving WAL mode, close the log file. If successful, the call
** to PagerCloseWal() checkpoints and deletes the write-ahead-log 
** file. An EXCLUSIVE lock may still be held on the database file 
** after a successful return. 
*/
rc = sqlite3PagerCloseWal(pPager);
if( rc==SQLITE_OK ){
sqlite3PagerSetJournalMode(pPager, eNew);
}
}else if( eOld==PAGER_JOURNALMODE_MEMORY ){
/* Cannot transition directly from MEMORY to WAL.  Use mode OFF
** as an intermediate */
sqlite3PagerSetJournalMode(pPager, PAGER_JOURNALMODE_OFF);
}

/* Open a transaction on the database file. Regardless of the journal
** mode, this transaction always uses a rollback journal.
*/
Debug.Assert( sqlite3BtreeIsInTrans(pBt)==0 );
if( rc==SQLITE_OK ){
rc = sqlite3BtreeSetVersion(pBt, (eNew==PAGER_JOURNALMODE_WAL ? 2 : 1));
}
}
}
#endif //* ifndef SQLITE_OMIT_WAL */

              if ( rc != 0 )
              {
                eNew = eOld;
              }
              eNew = sqlite3PagerSetJournalMode( pPager, eNew );

              out_ = mems[op.P2];
              out_.flags = MEM_Str | MEM_Static | MEM_Term;
              out_.z = sqlite3JournalModename( eNew );
              out_.n = sqlite3Strlen30( out_.z );
              out_.enc = SQLITE_UTF8;
              sqlite3VdbeChangeEncoding( out_, encoding );
              break;
            };
#endif //* SQLITE_OMIT_PRAGMA */

#if  !SQLITE_OMIT_VACUUM && !SQLITE_OMIT_ATTACH
          /* Opcode: Vacuum * * * * *
**
** Vacuum the entire database.  This opcode will cause other virtual
** machines to be created and run.  It may not be called from within
** a transaction.
*/
          case OP_Vacuum:
            {
              rc = sqlite3RunVacuum( ref p.zErrMsg, ctx );
              break;
            }
#endif

#if  !SQLITE_OMIT_AUTOVACUUM
          /* Opcode: IncrVacuum P1 P2 * * *
**
** Perform a single step of the incremental vacuum procedure on
** the P1 database. If the vacuum has finished, jump to instruction
** P2. Otherwise, fall through to the next instruction.
*/
          case OP_IncrVacuum:
            {        /* jump */
              Btree pBt;

              Debug.Assert( op.P1 >= 0 && op.P1 < ctx.nDb );
              Debug.Assert( ( p.btreeMask & ( ( (yDbMask)1 ) << op.P1 ) ) != 0 );
              pBt = ctx.aDb[op.P1].pBt;
              rc = sqlite3BtreeIncrVacuum( pBt );
              if ( rc == SQLITE_DONE )
              {
                pc = op.P2 - 1;
                rc = SQLITE_OK;
              }
              break;
            }
#endif

          /* Opcode: Expire P1 * * * *
**
** Cause precompiled statements to become expired. An expired statement
** fails with an error code of SQLITE_SCHEMA if it is ever executed
** (via sqlite3_step()).
**
** If P1 is 0, then all SQL statements become expired. If P1 is non-zero,
** then only the currently executing statement is affected.
*/
          case OP_Expire:
            {
              if ( op.P1 == 0 )
              {
                sqlite3ExpirePreparedStatements( ctx );
              }
              else
              {
                p.expired = true;
              }
              break;
            }

#if !SQLITE_OMIT_SHARED_CACHE
/* Opcode: TableLock P1 P2 P3 P4 *
**
** Obtain a lock on a particular table. This instruction is only used when
** the shared-cache feature is enabled.
**
** P1 is the index of the database in sqlite3.aDb[] of the database
** on which the lock is acquired.  A readlock is obtained if P3==0 or
** a write lock if P3==1.
**
** P2 contains the root-page of the table to lock.
**
** P4 contains a pointer to the name of the table being locked. This is only
** used to generate an error message if the lock cannot be obtained.
*/
case OP_TableLock:
{
u8 isWriteLock = (u8)op.P3;
if( isWriteLock || 0==(ctx.flags&SQLITE_ReadUncommitted) ){
int p1 = op.P1; 
Debug.Assert( p1 >= 0 && p1 < ctx.nDb );
Debug.Assert( ( p.btreeMask & ( ((yDbMask)1) << p1 ) ) != 0 );
Debug.Assert( isWriteLock == 0 || isWriteLock == 1 );
rc = sqlite3BtreeLockTable( ctx.aDb[p1].pBt, op.P2, isWriteLock );
if ( ( rc & 0xFF ) == SQLITE_LOCKED )
{
string z = op.P4.z;
C._setstring( ref p.zErrMsg, ctx, "database table is locked: ", z );
}
}
break;
}
#endif // * SQLITE_OMIT_SHARED_CACHE */
#endif

                    #region Virtual Table
#if !OMIT_VIRTUALTABLE
                    case OP.VBegin:
                        {
                            // Opcode: VBegin * * * P4 *
                            //
                            // P4 may be a pointer to an sqlite3_vtab structure. If so, call the xBegin method for that table.
                            //
                            // Also, whether or not P4 is set, check that this is not being called from within a callback to a virtual table xSync() method. If it is, the error
                            // code will be set to SQLITE_LOCKED.
                            VTable vtable = op.P4.VTable;
                            rc = VTable.Begin(ctx, vtable);
                            if (vtable != null) ImportVtabErrMsg(this, vtable.IVTable);
                            break;
                        }
                    case OP.VCreate:
                        {
                            // Opcode: VCreate P1 * * P4 *
                            //
                            // P4 is the name of a virtual table in database P1. Call the xCreate method for that table.
                            rc = VTable.CallCreate(ctx, op.P1, op.P4.Z, ref ErrMsg);
                            break;
                        }
                    case OP.VDestroy:
                        {
                            // Opcode: VDestroy P1 * * P4 *
                            //
                            // P4 is the name of a virtual table in database P1.  Call the xDestroy method of that table.
                            InVtabMethod = 2;
                            rc = VTable.CallDestroy(ctx, op.P1, op.P4.Z);
                            InVtabMethod = 0;
                            break;
                        }
                    case OP.VOpen:
                        {
                            // Opcode: VOpen P1 * * P4 *
                            //
                            // P4 is a pointer to a virtual table object, an sqlite3_vtab structure. P1 is a cursor number.  This opcode opens a cursor to the virtual
                            // table and stores that cursor in P1.
                            VTable vtable = op.P4.VTable.IVTable;
                            ITableModule module = (ITableModule)vtable.Module;
                            Debug.Assert(vtable != null && module != null);
                            IVTableCursor vtabCursor;
                            rc = module.Open(vtable, out vtabCursor);
                            ImportVtabErrMsg(this, vtable);
                            if (rc == RC.OK)
                            {
                                // Initialize sqlite3_vtab_cursor base class
                                vtabCursor.VTable = vtable;

                                // Initialise vdbe cursor object
                                VdbeCursor cur = AllocateCursor(this, op.P1, 0, -1, false);
                                if (cur != null)
                                {
                                    cur.VtabCursor = vtabCursor;
                                    cur.IModule = vtabCursor.IVTable.Module;
                                }
                                else
                                {
                                    ctx.MallocFailed = true;
                                    module.Close(ref vtabCursor);
                                }
                            }
                            break;
                        }
                    case OP.VFilter: // jump
                        {
                            // Opcode: VFilter P1 P2 P3 P4 *
                            //
                            // P1 is a cursor opened using VOpen.  P2 is an address to jump to if the filtered result set is empty.
                            //
                            // P4 is either NULL or a string that was generated by the xBestIndex method of the module.  The interpretation of the P4 string is left
                            // to the module implementation.
                            //
                            // This opcode invokes the xFilter method on the virtual table specified by P1.  The integer query plan parameter to xFilter is stored in register
                            // P3. Register P3+1 stores the argc parameter to be passed to the xFilter method. Registers P3+2..P3+1+argc are the argc
                            // additional parameters which are passed to xFilter as argv. Register P3+2 becomes argv[0] when passed to xFilter.
                            //
                            // A jump is made to P2 if the result set after filtering would be empty.
                            int res;
                            Mem query = mems[op.P3];
                            Mem argc = mems[op.P3 + 1]; //: query[1];
                            VdbeCursor cur = Cursors[op.P1];
                            Debug.Assert(E.MemIsValid(query));
                            REGISTER_TRACE(op.P3, query);
                            Debug.Assert(cur.VtabCursor != null);
                            IVTableCursor vtabCursor = cur.VtabCursor;
                            VTable vtable = vtabCursor.IVTable;
                            ITableModule module = vtable.Module;

                            // Grab the index number and argc parameters
                            Debug.Assert((query.Flags & MEM.Int) != 0 && argc.Flags == MEM.Int);
                            int argsLength = (int)argc.u.I;
                            int queryLength = (int)query.u.I;

                            // Invoke the xFilter method
                            {
                                res = 0;
                                Mem[] args = Args;
                                for (int i = 0; i < argsLength; i++)
                                {
                                    args[i] = mems[(op.P3 + 1) + i + 1]; //: args[i] = argc[i + 1];
                                    MemStoreType(args[i]);
                                }

                                InVtabMethod = 1;
                                rc = module.Filter(vtabCursor, queryLength, op.P4.Z, argsLength, args);
                                InVtabMethod = 0;
                                ImportVtabErrMsg(this, vtable);
                                if (rc == RC.OK)
                                    res = module.Eof(vtabCursor);

                                if (res != 0)
                                    pc = op.P2 - 1;
                            }
                            cur.NullRow = false;
                            break;
                        }
                    case OP.VColumn:
                        {
                            // Opcode: VColumn P1 P2 P3 * *
                            //
                            // Store the value of the P2-th column of the row of the virtual-table that the P1 cursor is pointing to into register P3.
                            VdbeCursor cur = Cursors[op.P1];
                            Debug.Assert(cur.VtabCursor != null);
                            Debug.Assert(op.P3 > 0 && op.P3 <= Mems.length);
                            Mem dest = mems[op.P3];
                            MemAboutToChange(this, dest);
                            if (cur.NullRow)
                            {
                                MemSetNull(dest);
                                break;
                            }
                            IVTable vtable = cur.VtabCursor.IVTable;
                            ITableModule module = vtable.IModule;
                            Debug.Assert(module.Column != null);
                            FuncContext sContext = new FuncContext();

                            /* The output cell may already have a buffer allocated. Move
                            ** the current contents to sContext.s so in case the user-function
                            ** can use the already allocated buffer instead of allocating a
                            ** new one.
                            */
                            sqlite3VdbeMemMove(sContext.s, dest);
                            MemSetTypeFlag(sContext.s, MEM_Null);

                            rc = module.xColumn(cur.pVtabCursor, sContext, op.P2);
                            importVtabErrMsg(p, vtable);

                            if (sContext.isError != 0)
                            {
                                rc = sContext.isError;
                            }

                            /* Copy the result of the function to the P3 register. We
                            ** do this regardless of whether or not an error occurred to ensure any
                            ** dynamic allocation in sContext.s (a Mem struct) is  released.
                            */
                            sqlite3VdbeChangeEncoding(sContext.s, encoding);
                            sqlite3VdbeMemMove(dest, sContext.s);
                            REGISTER_TRACE(p, op.P3, dest);
                            UPDATE_MAX_BLOBSIZE(dest);
                            if (sqlite3VdbeMemTooBig(dest))
                            {
                                goto too_big;
                            }
                            break;
                        }
                    case OP.VNext: // jump
                        {
                            sqlite3_vtab pVtab;
                            sqlite3_module pModule;
                            int res;
                            VdbeCursor pCur;

                            res = 0;
                            pCur = p.apCsr[op.P1];
                            Debug.Assert(pCur.pVtabCursor != null);
                            if (pCur.nullRow)
                            {
                                break;
                            }
                            pVtab = pCur.pVtabCursor.pVtab;
                            pModule = pVtab.pModule;
                            Debug.Assert(pModule.xNext != null);

                            /* Invoke the xNext() method of the module. There is no way for the
                            ** underlying implementation to return an error if one occurs during
                            ** xNext(). Instead, if an error occurs, true is returned (indicating that
                            ** data is available) and the error code returned when xColumn or
                            ** some other method is next invoked on the save virtual table cursor.
                            */
                            p.inVtabMethod = 1;
                            rc = pModule.xNext(pCur.pVtabCursor);
                            p.inVtabMethod = 0;
                            importVtabErrMsg(p, pVtab);
                            if (rc == SQLITE_OK)
                            {
                                res = pModule.xEof(pCur.pVtabCursor);
                            }

                            if (0 == res)
                            {
                                /* If there is data, jump to P2 */
                                pc = op.P2 - 1;
                            }
                            break;
                        }
                    case OP_VRename:
                        {
                            sqlite3_vtab pVtab;
                            Mem pName;

                            pVtab = op.P4.pVtab.pVtab;
                            pName = mems[op.P1];
                            Debug.Assert(pVtab.pModule.xRename != null);
                            Debug.Assert(memIsValid(pName));
                            REGISTER_TRACE(p, op.P1, pName);
                            Debug.Assert((pName.flags & MEM_Str) != 0);
                            rc = pVtab.pModule.xRename(pVtab, pName.z);
                            importVtabErrMsg(p, pVtab);
                            p.expired = false;
                            break;
                        }
                    case OP_VUpdate:
                        {
                            sqlite3_vtab pVtab;
                            sqlite3_module pModule;
                            int nArg;
                            int i;
                            sqlite_int64 rowid = 0;
                            Mem[] apArg;
                            Mem pX;

                            Debug.Assert(op.P2 == 1 || op.P5 == OE_Fail || op.P5 == OE_Rollback
                                   || op.P5 == OE_Abort || op.P5 == OE_Ignore || op.P5 == OE_Replace
                              );
                            pVtab = op.P4.pVtab.pVtab;
                            pModule = (sqlite3_module)pVtab.pModule;
                            nArg = op.P2;
                            Debug.Assert(op.p4type == P4_VTAB);
                            if (ALWAYS(pModule.xUpdate))
                            {
                                u8 vtabOnConflict = ctx.vtabOnConflict;
                                apArg = p.apArg;
                                //pX = mems[op.P3];
                                for (i = 0; i < nArg; i++)
                                {
                                    pX = mems[op.P3 + i];
                                    Debug.Assert(memIsValid(pX));
                                    memAboutToChange(p, pX);
                                    sqlite3VdbeMemStoreType(pX);
                                    apArg[i] = pX;
                                    //pX++;
                                }
                                ctx.vtabOnConflict = op.P5;
                                rc = pModule.xUpdate(pVtab, nArg, apArg, out rowid);
                                ctx.vtabOnConflict = vtabOnConflict;
                                importVtabErrMsg(p, pVtab);
                                if (rc == SQLITE_OK && op.P1 != 0)
                                {
                                    Debug.Assert(nArg > 1 && apArg[0] != null && (apArg[0].flags & MEM_Null) != 0);
                                    ctx.lastRowid = lastRowid = rowid;
                                }
                                if (rc == SQLITE_CONSTRAINT && op.P4.pVtab.bConstraint != 0)
                                {
                                    if (op.P5 == OE_Ignore)
                                    {
                                        rc = SQLITE_OK;
                                    }
                                    else
                                    {
                                        p.errorAction = op.P5 == (u8)OE_Replace ? (u8)OE_Abort : op.P5;
                                    }
                                }
                                else
                                {
                                    p.nChange++;
                                }
                            }
                            break;
                        }
#endif
                    #endregion

#if !OMIT_PAGER_PRAGMAS
                    case OP.Pagecount: // out2-prerelease
                        {
                            // Opcode: Pagecount P1 P2 * * *
                            //
                            // Write the current number of pages in database P1 to memory cell P2.
                            out_.u.I = Btree.LastPage(ctx.DBs[op.P1].Bt);
                            break;
                        }
                    case OP.MaxPgcnt: // out2-prerelease
                        {
                            // Opcode: MaxPgcnt P1 P2 P3 * *
                            //
                            // Try to set the maximum page count for database P1 to the value in P3. Do not let the maximum page count fall below the current page count and
                            // do not change the maximum page count value if P3==0.
                            //
                            // Store the maximum page count after the change in register P2.
                            Btree bt = ctx.DBs[op.P1].Bt;
                            long newMax = 0;
                            if (op.P3 != 0)
                            {
                                newMax = Btree.LastPage(bt);
                                if (newMax < op.P3) newMax = op.P3;
                            }
                            out_.u.I = (long)Btree.MaxPageCount(bt, (int)newMax);
                            break;
                        }
#endif
#if !OMIT_TRACE
                    case OP.Trace:
                        {
                            // Opcode: Trace * * * P4 *
                            //
                            // If tracing is enabled (by the sqlite3_trace()) interface, then the UTF-8 string contained in P4 is emitted on the trace callback.
                            string trace;
                            string z;
                            if (ctx.Trace != null && !DoingRerun && (trace = (op.P4.Z != null ? op.P4.Z : Sql_)) != null)
                            {
                                z = ExpandSql(trace);
                                ctx.Trace(ctx.TraceArg, z);
                                C._tagfree(ctx, ref z);
                            }
#if DEBUG
                            if ((ctx.Flags & Context.FLAG.SqlTrace) != 0 && (trace = (op.P4.Z != null ? op.P4.Z : Sql_)) != null)
                                sqlite3DebugPrintf("SQL-trace: %s\n", trace);
#endif
                            break;
                        }
#endif
                    default: // This is really OP_Noop and OP_Explain
                        {
                            // Opcode: Noop * * * * *
                            //
                            // Do nothing.  This instruction is often useful as a jump destination.
                            //
                            // The magic Explain opcode are only inserted when explain==2 (which is to say when the EXPLAIN QUERY PLAN syntax is used.)
                            // This opcode records information from the optimizer.  It is the the same as a no-op.  This opcodesnever appears in a real VM program.
                            Debug.Assert(op.Opcode == OP.Noop || op.Opcode == OP.Explain);
                            break;
                        }
                }
                // The cases of the switch statement above this line should all be indented by 6 spaces.  But the left-most 6 spaces have been removed to improve the
                // readability.  From this point on down, the normal indentation rules are restored.

#if VDBE_PROFILE
                {
                    ulong elapsed = C._hwtime() - start;
                    op.Cycles += elapsed;
                    op.Cnt++;
#if false
Console.Write("%10llu ", elapsed);
PrintOp(Console.Out, origPc, ops[origPc]);
#endif
                }
#endif

#if !NDEBUG
                // The following code adds nothing to the actual functionality of the program.  It is only here for testing and debugging.
                // On the other hand, it does burn CPU cycles every time through the evaluator loop.  So we can leave it out when NDEBUG is defined.
                Debug.Assert(pc >= -1 && pc < Ops.length);
#if DEBUG
                if (Trace != null)
                {
                    if (rc != 0)
                        fprintf(Trace, "rc=%d\n", rc);
                    if ((op.Opflags & (OPFLG.OUT2_PRERELEASE | OPFLG.OUT2)) != 0)
                        RegisterTrace(Trace, op.P2, mems[op.P2]);
                    if ((op.Opflags & OPFLG.OUT3) != 0)
                        RegisterTrace(Trace, op.P3, mems[op.P3]);
                }
#endif
#endif
            }  // The end of the for(;;) loop the loops through opcodes

        // If we reach this point, it means that execution is finished with an error of some kind.
        vdbe_error_halt:
            Debug.Assert(rc != 0);
            RC_ = rc;
            C.ASSERTCOVERAGE(SysEx._GlobalStatics.Log != null);
            SysEx.LOG(rc, "statement aborts at %d: [%s] %s", pc, Sql_, ErrMsg);
            Halt();
            if (rc == RC.IOERR_NOMEM) ctx.MallocFailed = true;
            rc = RC.ERROR;
            if (resetSchemaOnFault > 0)
                Parse.ResetOneSchema(ctx, resetSchemaOnFault - 1);

        // This is the only way out of this procedure.  We have to release the mutexes on btrees that were acquired at the top.
        vdbe_return:
            ctx.LastRowID = lastRowid;
            Leave();
            return rc;

        // Jump to here if a string or blob larger than CORE_MAX_LENGTH is encountered.
        too_big:
            C._setstring(ref ErrMsg, ctx, "string or blob too big");
            rc = RC.TOOBIG;
            goto vdbe_error_halt;

        // Jump to here if a malloc() fails.
        no_mem:
            ctx.MallocFailed = true;
            C._setstring(ref ErrMsg, ctx, "out of memory");
            rc = RC.NOMEM;
            goto vdbe_error_halt;

        // Jump to here for any other kind of fatal error.  The "rc" variable should hold the error number.
        abort_due_to_error:
            Debug.Assert(ErrMsg != null);
            if (ctx.MallocFailed) rc = RC.NOMEM;
            if (rc != RC.IOERR_NOMEM)
                C._setstring(ref ErrMsg, ctx, "%s", Main.ErrStr(rc));
            goto vdbe_error_halt;

        // Jump to here if the sqlite3_interrupt() API sets the interrupt flag.
        abort_due_to_interrupt:
            Debug.Assert(ctx.u1.IsInterrupted);
            rc = RC.INTERRUPT;
            RC_ = rc;
            C._setstring(ref ErrMsg, ctx, Main.ErrStr(rc));
            goto vdbe_error_halt;
        }

        #endregion
    }
}
