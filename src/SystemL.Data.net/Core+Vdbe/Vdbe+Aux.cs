using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using FILE = System.IO.TextWriter;
using Pid = System.UInt32;
#region Limits
#if !MAX_VARIABLE_NUMBER
using yVar = System.Int16;
#else
    using yVar = System.Int32; 
#endif
#if MAX_ATTACHED
    using yDbMask = System.Int64;
#else
using yDbMask = System.Int32;
#endif
#endregion
using P4_t = Core.Vdbe.VdbeOp.P4_t;

namespace Core
{
    public partial class Vdbe
    {
        public static Vdbe Create(Context ctx)
        {
            Vdbe p = new Vdbe();
            if (p == null) return null;
            p.Ctx = ctx;
            if (ctx.Vdbes != null)
                ctx.Vdbes.Prev = p;
            p.Next = ctx.Vdbes;
            p.Prev = null;
            ctx.Vdbes = p;
            p.Magic = VDBE_MAGIC_INIT;
            return p;
        }

        public static void SetSql(Vdbe p, string z, int n, bool isPrepareV2)
        {
            if (p == null) return;
#if OMIT_TRACE && !ENABLE_SQLLOG
            if (isPrepareV2 == 0) return;
#endif
            Debug.Assert(p.Sql_ == null);
            p.Sql_ = z.Substring(0, n); //: _tagstrndup(p->Ctx, z, n);
            p.IsPrepareV2 = isPrepareV2;
        }

        public static string Sql(Vdbe stmt)
        {
            Vdbe p = (Vdbe)stmt;
            return (p != null && p.IsPrepareV2 ? p.Sql_ : null);
        }

        public static void Swap(Vdbe a, Vdbe b)
        {
            Vdbe tmp_ = new Vdbe();
            a._memcpy(tmp_);
            b._memcpy(a);
            tmp_._memcpy(b);
            Vdbe tmp = tmp = a.Next;
            a.Next = b.Next;
            b.Next = tmp;
            tmp = a.Prev;
            a.Prev = b.Prev;
            b.Prev = tmp;
            string tmpSql = a.Sql_;
            a.Sql_ = b.Sql_;
            b.Sql_ = tmpSql;
            b.IsPrepareV2 = a.IsPrepareV2;
        }

#if DEBUG
        public void set_Trace(FILE trace)
        {
            Trace = trace;
        }
#endif

        static RC GrowOps(Vdbe p)
        {
            int newLength = (p.OpsAlloc != 0 ? p.OpsAlloc * 2 : 1024 / 1);
            p.OpsAlloc = newLength;
            C._tagrealloc_or_create(p.Ctx, ref p.Ops.data, newLength);
            return (p.Ops.data != null ? RC.OK : RC.NOMEM);
        }

        public int AddOp3(OP op, int p1, int p2, int p3)
        {
            int i = Ops.length;
            Debug.Assert(Magic == VDBE_MAGIC_INIT);
            Debug.Assert((int)op > 0 && (int)op < 0xff);
            if (OpsAlloc <= i)
                if (GrowOps(this) != RC.OK)
                    return 1;
            Ops.length++;
            if (Ops[i] == null) Ops[i] = new VdbeOp();
            VdbeOp opAsObj = Ops[i];
            opAsObj.Opcode = op;
            opAsObj.P5 = 0;
            opAsObj.P1 = p1;
            opAsObj.P2 = p2;
            opAsObj.P3 = p3;
            opAsObj.P4.P = null;
            opAsObj.P4Type = Vdbe.P4T.NOTUSED;
#if DEBUG
            opAsObj.Comment = null;
            if ((Ctx.Flags & BContext.FLAG.VdbeAddopTrace) != 0)
                PrintOp(null, i, Ops[i]);
#endif
#if VDBE_PROFILE
            opAsObj.Cycles = 0;
            opAsObj.Cnt = 0;
#endif
            return i;
        }
        public int AddOp0(OP op) { return AddOp3(op, 0, 0, 0); }
        public int AddOp1(OP op, int p1) { return AddOp3(op, p1, 0, 0); }
        public int AddOp2(OP op, int p1, bool b2) { return AddOp3(op, p1, (int)(b2 ? 1 : 0), 0); }
        public int AddOp2(OP op, int p1, int p2) { return AddOp3(op, p1, p2, 0); }
        public int AddOp4(OP op, int p1, int p2, int p3, int p4, Vdbe.P4T p4t) // int
        {
            int addr = AddOp3(op, p1, p2, p3);
            ChangeP4(addr, new P4_t { I = p4 }, p4t);
            return addr;
        }
        public int AddOp4(OP op, int p1, int p2, int p3, char p4, Vdbe.P4T p4t) // char
        {
            int addr = AddOp3(op, p1, p2, p3);
            ChangeP4(addr, new P4_t { Z = p4.ToString() }, p4t);
            return addr;
        }
        public int AddOp4(OP op, int p1, int p2, int p3, StringBuilder p4, Vdbe.P4T p4t) // StringBuilder
        {
            int addr = AddOp3(op, p1, p2, p3);
            ChangeP4(addr, new P4_t { Z = p4.ToString() }, p4t);
            return addr;
        }
        public int AddOp4(OP op, int p1, int p2, int p3, string p4, Vdbe.P4T p4t) // string
        {
            int addr = AddOp3(op, p1, p2, p3);
            ChangeP4(addr, new P4_t { Z = p4 }, p4t);
            return addr;
        }
        public int AddOp4(OP op, int p1, int p2, int p3, byte[] p4, Vdbe.P4T p4t) // byte[]
        {
            Debug.Assert(op == OP.Null || p4 != null);
            int addr = AddOp3(op, p1, p2, p3);
            ChangeP4(p, addr, new P4_t { Z = Encoding.UTF8.GetString(p4, 0, p4.Length) }, p4t);
            return addr;
        }
        public int AddOp4(OP op, int p1, int p2, int p3, int[] p4, Vdbe.P4T p4t) // P4T_INTARRAY
        {
            Debug.Assert(p4 != null);
            int addr = AddOp3(op, p1, p2, p3);
            ChangeP4(addr, new P4_t { Is = p4 }, p4t);
            return addr;
        }
        public int AddOp4(OP op, int p1, int p2, int p3, long p4, Vdbe.P4T p4t) // P4T_INT64
        {
            int addr = AddOp3(op, p1, p2, p3);
            ChangeP4(addr, new P4_t { I64 = p4 }, p4t);
            return addr;
        }
        public int AddOp4(OP op, int p1, int p2, int p3, double p4, Vdbe.P4T p4t) // DOUBLE (REAL)
        {
            int addr = AddOp3(op, p1, p2, p3);
            ChangeP4(addr, new P4_t { Real = p4 }, p4t);
            return addr;
        }
        public int AddOp4(OP op, int p1, int p2, int p3, FuncDef p4, Vdbe.P4T p4t) // FUNCDEF
        {
            int addr = AddOp3(op, p1, p2, p3);
            ChangeP4(addr, new P4_t { Func = p4 }, p4t);
            return addr;
        }
        public int AddOp4(OP op, int p1, int p2, int p3, CollSeq p4, Vdbe.P4T p4t) // CollSeq
        {
            int addr = AddOp3(op, p1, p2, p3);
            ChangeP4(addr, new P4_t { Coll = p4 }, p4t);
            return addr;
        }
        public int AddOp4(OP op, int p1, int p2, int p3, KeyInfo p4, Vdbe.P4T p4t) // KeyInfo
        {
            int addr = AddOp3(op, p1, p2, p3);
            ChangeP4(addr, new P4_t { KeyInfo = p4 }, p4t);
            return addr;
        }

#if !OMIT_VIRTUALTABLE
        public int AddOp4(OP op, int p1, int p2, int p3, VTable p4, Vdbe.P4T p4t) // VTable
        {
            Debug.Assert(p4 != null);
            int addr = AddOp3(op, p1, p2, p3);
            ChangeP4(p, addr, new P4_t { VTable = p4 }, p4t);
            return addr;
        }
#endif

        public void AddParseSchemaOp(int db, string where_)
        {
            int addr = AddOp3(OP.ParseSchema, db, 0, 0);
            ChangeP4(addr, where_, Vdbe.P4T.DYNAMIC);
            for (int j = 0; j < Ctx.DBs.length; j++) UsesBtree(j);
        }

        public int AddOp4Int(OP op, int p1, int p2, int p3, int p4)
        {
            int addr = AddOp3(op, p1, p2, p3);
            ChangeP4(addr, new P4_t { I = p4 }, Vdbe.P4T.INT32);
            return addr;
        }

        public int MakeLabel()
        {
            int i = Labels.length++;
            Debug.Assert(Magic == VDBE_MAGIC_INIT);
            C._tagrealloc_or_create(Ctx, ref Labels.data, (i * 2 + 1));
            if (Labels.data != null)
                Labels[i] = -1;
            return -1 - i;
        }

        public void ResolveLabel(int x)
        {
            int j = -1 - x;
            Debug.Assert(Magic == VDBE_MAGIC_INIT);
            Debug.Assert(j >= 0 && j < Labels.length);
            if (Labels.data != null)
                Labels[j] = Ops.length;
        }

        public void set_RunOnlyOnce()
        {
            RunOnlyOnce = true;
        }

#if DEBUG

        public class VdbeOpIter
        {
            public Vdbe V;                    // Vdbe to iterate through the opcodes of
            public array_t<SubProgram> Subs;     // Array of subprograms
            public int Addr;                  // Address of next instruction to return
            public int SubId;                 // 0 = main program, 1 = first sub-program etc.
        }

        static VdbeOp OpIterNext(VdbeOpIter p)
        {
            VdbeOp r = null;
            if (p.SubId <= p.Subs.length)
            {
                Vdbe v = p.V;
                array_t<VdbeOp> ops = (p.SubId == 0 ? v.Ops : p.Subs[p.SubId - 1].Ops);
                Debug.Assert(p.Addr < ops.length);

                r = ops[p.Addr];
                p.Addr++;
                if (p.Addr == ops.length)
                {
                    p.SubId++;
                    p.Addr = 0;
                }
                if (r.P4Type == Vdbe.P4T.SUBPROGRAM)
                {
                    int bytes = p.Subs.length + 1;
                    int j;
                    for (j = 0; j < p.Subs.length; j++)
                        if (p.Subs[j] == r.P4.Program) break;
                    if (j == p.Subs.length)
                    {
                        C._tagrealloc_or_free2(v.Ctx, ref p.Subs.data, bytes);
                        p.Subs[p.Subs.length++] = r.P4.Program;
                    }
                }
            }
            return r;
        }

        public bool AssertMayAbort(bool mayAbort)
        {
            VdbeOpIter sIter;
            sIter = new VdbeOpIter();
            sIter.V = this;

            bool hasAbort = false;
            VdbeOp op;
            while ((op = OpIterNext(sIter)) != null)
            {
                OP opcode = (OP)op.Opcode;
                if (opcode == OP.Destroy || opcode == OP.VUpdate || opcode == OP.VRename
#if !OMIT_FOREIGN_KEY
 || (opcode == OP.FkCounter && op.P1 == 0 && op.P2 == 1)
#endif
 || ((opcode == OP.Halt || opcode == OP.HaltIfNull)
 && (op.P1 == (int)RC.CONSTRAINT && op.P2 == (int)OE.Abort)))
                {
                    hasAbort = true;
                    break;
                }
            }
            C._tagfree(Ctx, ref sIter.Subs.data);

            // Return true if hasAbort==mayAbort. Or if a malloc failure occurred. If malloc failed, then the while() loop above may not have iterated
            // through all opcodes and hasAbort may be set incorrectly. Return true for this case to prevent the assert() in the callers frame from failing.            return (hasAbort == mayAbort) ? 1 : 0;//v.db.mallocFailed !=0|| hasAbort==mayAbort );
            return (Ctx.MallocFailed || hasAbort == mayAbort);
        }
#endif

        static void ResolveP2Values(Vdbe p, ref int maxFuncArgs)
        {
            int maxArgs = maxFuncArgs;
            int[] labels = p.Labels.data;
            p.ReadOnly = true;

            Vdbe.VdbeOp op;
            int i;
            for (i = 0; i < p.Ops.length; i++)
            {
                op = p.Ops[i];
                OP opcode = op.Opcode;
                op.Opflags = E._opcodeProperty[(int)opcode];
                if (opcode == OP.Function || opcode == OP.AggStep)
                {
                    if (op.P5 > maxArgs) maxArgs = op.P5;
                }
                else if ((opcode == OP.Transaction && op.P2 != 0) || opcode == OP.Vacuum)
                {
                    p.ReadOnly = false;

                }
#if !OMIT_VIRTUALTABLE
                else if (opcode == OP.VUpdate)
                {
                    if (op.P2 > maxArgs) maxArgs = op.P2;
                }
                else if (opcode == OP.VFilter)
                {
                    Debug.Assert(p.Ops.length - i >= 3);
                    Debug.Assert(p.Ops[i - 1].Opcode == OP.Integer);
                    int n = p.Ops[i - 1].P1;
                    if (n > maxArgs) maxArgs = n;
                }
#endif
                else if (opcode == OP.Next || opcode == OP.SorterNext)
                {
                    op.P4.Advance = Btree.Next_;
                    op.P4Type = Vdbe.P4T.ADVANCE;
                }
                else if (opcode == OP.Prev)
                {
                    op.P4.Advance = Btree.Previous;
                    op.P4Type = Vdbe.P4T.ADVANCE;
                }
                if ((op.Opflags & OPFLG.JUMP) != 0 && op.P2 < 0)
                {
                    Debug.Assert(-1 - op.P2 < p.Labels.length);
                    op.P2 = labels[-1 - op.P2];
                }
            }
            C._tagfree(p.Ctx, ref p.Labels.data);
            p.Labels.data = null;
            maxFuncArgs = maxArgs;
        }

        public int CurrentAddr()
        {
            Debug.Assert(Magic == VDBE_MAGIC_INIT);
            return Ops.length;
        }

        public VdbeOp[] TakeOpArray(ref int opsLength, ref int maxArgs)
        {
            VdbeOp[] ops = Ops.data;
            Debug.Assert(ops != null && !Ctx.MallocFailed);
            Debug.Assert(BtreeMask == 0); // Check that sqlite3VdbeUsesBtree() was not called on this VM
            ResolveP2Values(this, ref maxArgs);
            opsLength = Ops.length;
            Ops.data = null;
            return ops;
        }

        public int AddOpList(int opsLength, VdbeOpList[] ops)
        {
            Debug.Assert(Magic == VDBE_MAGIC_INIT);
            if (Ops.length + opsLength > OpsAlloc && GrowOps(this) != 0)
                return 0;
            int addr = Ops.length;
            if (C._ALWAYS(opsLength > 0))
            {
                VdbeOpList in_;
                for (int i = 0; i < opsLength; i++)
                {
                    in_ = ops[i];
                    int p2 = in_.P2;
                    if (Ops[i + addr] == null)
                        Ops[i + addr] = new VdbeOp();
                    VdbeOp out_ = Ops[i + addr];
                    out_.Opcode = in_.Opcode;
                    out_.P1 = in_.P1;
                    out_.P2 = (p2 < 0 && ((OPFLG)E._opcodeProperty[(int)out_.Opcode] & OPFLG.JUMP) != 0 ? addr + (-1 - p2) : p2);
                    out_.P3 = in_.P3;
                    out_.P4Type = Vdbe.P4T.NOTUSED;
                    out_.P4.P = null;
                    out_.P5 = 0;
#if DEBUG
                    out_.Comment = null;
                    if ((Ctx.Flags & BContext.FLAG.VdbeAddopTrace) != 0)
                        PrintOp(null, i + addr, Ops[i + addr]);
#endif
                }
                Ops.length += opsLength;
            }
            return addr;
        }

        public void ChangeP1(int addr, int val) { if (Ops.length > addr) Ops[addr].P1 = val; }
        public void ChangeP2(int addr, int val) { if (Ops.length > addr) Ops[addr].P2 = val; }
        public void ChangeP3(int addr, int val) { if (Ops.length > addr) Ops[addr].P3 = val; }
        public void ChangeP5(byte val) { if (Ops.data != null) { Debug.Assert(Ops.length > 0); Ops[Ops.length - 1].P5 = val; } }

        public void JumpHere(int addr)
        {
            Debug.Assert(addr >= 0);
            ChangeP2(addr, Ops.length);
        }

        static void FreeEphemeralFunction(Context ctx, FuncDef def)
        {
            if (C._ALWAYS(def != null) && (def.Flags & FUNC.EPHEM) != 0)
                C._tagfree(ctx, ref def);
        }
        static void FreeP4(Context ctx, P4T p4t, object p4)
        {
            Debug.Assert(ctx != null);
            if (p4 != null)
            {
                switch (p4t)
                {
                    case P4T.REAL:
                    case P4T.INT64:
                    case P4T.DYNAMIC:
                    case P4T.KEYINFO:
                    case P4T.INTARRAY:
                    case P4T.KEYINFO_HANDOFF:
                        {
                            C._tagfree(ctx, ref p4);
                            break;
                        }
                    case P4T.MPRINTF:
                        {
                            if (ctx.BytesFreed == 0) C._free(ref p4);
                            break;
                        }
                    case P4T.VDBEFUNC:
                        {
                            VdbeFunc vdbeFunc = (VdbeFunc)p4;
                            FreeEphemeralFunction(ctx, vdbeFunc);
                            if (ctx.BytesFreed == 0) Vdbe.DeleteAuxData(vdbeFunc, 0);
                            C._tagfree(ctx, ref vdbeFunc);
                            break;
                        }
                    case P4T.FUNCDEF:
                        {
                            FreeEphemeralFunction(ctx, (FuncDef)p4);
                            break;
                        }
                    case P4T.MEM:
                        {
                            if (ctx.BytesFreed == 0)
                                Vdbe.ValueFree(ref (Mem)p4);
                            else
                            {
                                Mem p = (Mem)p4;
                                //C._tagfree(ctx, ref p.Malloc);
                                C._tagfree(ctx, ref p);
                            }
                            break;
                        }
                    case P4T.VTAB:
                        {
                            if (ctx.BytesFreed == 0) ((VTable)p4).Unlock();
                            break;
                        }
                }
            }
        }

        static void VdbeFreeOpArray(Context ctx, ref VdbeOp[] ops, int opsLength)
        {
            if (ops != null)
            {
                for (int opIdx = ops.Length; opIdx < opsLength; opIdx++)
                {
                    VdbeOp op = ops[opIdx];
                    FreeP4(ctx, op.P4Type, op.P4.P);
#if DEBUG
                    C._tagfree(ctx, ref op.Comment);
#endif
                }
            }
            C._tagfree(ctx, ref ops);
        }

        public void LinkSubProgram(SubProgram p)
        {
            p.Next = Programs;
            Programs = p;
        }

        public void ChangeToNoop(int addr)
        {
            if (Ops.data != null)
            {
                VdbeOp op = Ops[addr];
                FreeP4(Ctx, op.P4Type, op.P4.P);
                op._memset();
                op.Opcode = OP.Noop;
            }
        }

        public void ChangeP4(int addr, CollSeq coll, int n) { ChangeP4(addr, new P4_t { Coll = coll }, n); } // P4_COLLSEQ 
        public void ChangeP4(int addr, FuncDef func, int n) { ChangeP4(addr, new P4_t { Func = func }, n); } // P4_FUNCDEF
        public void ChangeP4(int addr, int i, int n) { ChangeP4(addr, new P4_t { I = i }, n); } // P4_INT32
        public void ChangeP4(int addr, KeyInfo keyInfo, int n) { ChangeP4(addr, new P4_t { KeyInfo = keyInfo }, n); } // P4T_KEYINFO
        public void ChangeP4(int addr, char c, int n) { ChangeP4(addr, new P4_t { Z = c.ToString() }, n); } // CHAR
        public void ChangeP4(int addr, Mem m, int n) { ChangeP4(addr, new P4_t { Mem = m }, n); } // MEM
        public void ChangeP4(int addr, string z, Action<object> notUsed1) { ChangeP4(addr, new P4_t { Z = z }, (int)P4T.DYNAMIC); } // STRING + Type
        public void ChangeP4(int addr, SubProgram program, int n) { ChangeP4(addr, new P4_t { Program = program }, n); } // SUBPROGRAM
        public void ChangeP4(int addr, string z, int n) { ChangeP4(addr, new P4_t { Z = (n > 0 && n <= z.Length ? z.Substring(0, n) : z) }, n); }
        public void ChangeP4(int addr, P4_t p4, int n)
        {
            Debug.Assert(Magic == VDBE_MAGIC_INIT);
            Context ctx = Ctx;
            if (Ops.data == null || ctx.MallocFailed)
            {
                if (n != (int)P4T.KEYINFO && n != (int)P4T.VTAB)
                    FreeP4(ctx, (P4T)n, p4);
                return;
            }
            Debug.Assert(Ops.length > 0);
            Debug.Assert(addr < Ops.length);
            if (addr < 0)
                addr = Ops.length - 1;
            VdbeOp op = Ops[addr];
            FreeP4(ctx, op.P4Type, op.P4.P);
            op.P4.P = null;
            if (n == (int)P4T.INT32)
            {
                // Note: this cast is safe, because the origin data point was an int that was cast to a (string ).
                op.P4.I = p4.I; //: PTR_TO_INT(p4);
                op.P4Type = P4T.INT32;
            }
            else if (n == (int)P4T.INT64)
            {
                op.P4.I64 = p4.I64;
                op.P4Type = (P4T)n;
            }
            else if (n == (int)P4T.REAL)
            {
                op.P4.Real = p4.Real;
                op.P4Type = (P4T)n;
            }
            else if (p4 == null)
            {
                op.P4.P = null;
                op.P4Type = P4T.NOTUSED;
            }
            else if (n == (int)P4T.KEYINFO)
            {
                int fields = p4.KeyInfo.Fields;
                KeyInfo keyInfo = new KeyInfo();
                op.P4.KeyInfo = keyInfo;
                if (keyInfo != null)
                {
                    keyInfo = p4.KeyInfo._memcpy();
                    op.P4Type = P4T.KEYINFO;
                }
                else
                {
                    ctx.MallocFailed = true;
                    op.P4Type = P4T.NOTUSED;
                }
            }
            else if (n == (int)P4T.KEYINFO_HANDOFF)
            {
                op.P4.KeyInfo = p4.KeyInfo;
                op.P4Type = P4T.KEYINFO;
            }
            else if (n == (int)P4T.FUNCDEF)
            {
                op.P4.Func = p4.Func;
                op.P4Type = P4T.FUNCDEF;
            }
            else if (n == (int)P4T.COLLSEQ)
            {
                op.P4.Coll = p4.Coll;
                op.P4Type = P4T.COLLSEQ;
            }
            else if (n == (int)P4T.DYNAMIC || n == (int)P4T.STATIC || n == (int)P4T.MPRINTF)
            {
                op.P4.Z = p4.Z;
                op.P4Type = P4T.DYNAMIC;
            }
            else if (n == (int)P4T.MEM)
            {
                op.P4.Mem = p4.Mem;
                op.P4Type = P4T.MEM;
            }
            else if (n == (int)P4T.INTARRAY)
            {
                op.P4.Is = p4.Is;
                op.P4Type = P4T.INTARRAY;
            }
            else if (n == (int)P4T.SUBPROGRAM)
            {
                op.P4.Program = p4.Program;
                op.P4Type = P4T.SUBPROGRAM;
            }
            else if (n == (int)P4T.VTAB)
            {
                op.P4.VTable = p4.VTable;
                op.P4Type = P4T.VTAB;
                p4.VTable.Lock();
                Debug.Assert(p4.VTable.Ctx == ctx);
            }
            else if (n < 0)
            {
                op.P4.P = p4.P;
                op.P4Type = (P4T)n;
            }
            else
            {
                //: if (n == 0) n = _strlen30(p4);
                op.P4.Z = p4.Z;
                op.P4Type = P4T.DYNAMIC;
            }
        }

#if !NDEBUG
        public static void Comment(Vdbe p, string format, params object[] args)
        {
            if (p == null) return;
            Debug.Assert(p.Ops.length > 0 || p.Ops.data == null);
            Debug.Assert(p.Ops.data == null || p.Ops[p.Ops.length - 1].Comment == null || p.Ctx.MallocFailed);
            if (p.Ops.length != 0)
            {
                string z = C._vmtagprintf(p.Ctx, format, args);
                p.Ops[p.Ops.length - 1].Comment = z;
            }
        }
        public static void NoopComment(Vdbe p, string format, params object[] args)
        {
            if (p == null) return;
            p.AddOp0(OP.Noop);
            Debug.Assert(p.Ops.length > 0 || p.Ops.data == null);
            Debug.Assert(p.Ops.data == null || p.Ops[p.Ops.length - 1].Comment == null || p.Ctx.MallocFailed);
            if (p.Ops.length != 0)
            {
                string z = C._vmtagprintf(p.Ctx, format, args);
                p.Ops[p.Ops.length - 1].Comment = z;
            }
        }
#else
        public static void VdbeComment(Vdbe p, string fmt, params object[] args) { }
        public static void VdbeNoopComment(Vdbe p, string fmt, params object[] args) { }
#endif

        const VdbeOp _dummy = new VdbeOp();  // Ignore the MSVC warning about no initializer
        public VdbeOp GetOp(int addr)
        {
            // C89 specifies that the constant "dummy" will be initialized to all zeros, which is correct.  MSVC generates a warning, nevertheless.
            Debug.Assert(Magic == VDBE_MAGIC_INIT);
            if (addr < 0)
            {
#if OMIT_TRACE
                if (Ops.length == 0) return _dummy;
#endif
                addr = Ops.length - 1;
            }
            Debug.Assert((addr >= 0 && addr < Ops.length) || Ctx.MallocFailed);
            return (Ctx.MallocFailed ? _dummy : Ops[addr]);
        }

#if !OMIT_EXPLAIN || !NDEBUG || VDBE_PROFILE || DEBUG
        //static StringBuilder temp = new StringBuilder(100);
        static string DisplayP4(VdbeOp op, StringBuilder temp, int tempLength)
        {
            temp.Length = 0;
            Debug.Assert(tempLength >= 20);
            switch (op.P4Type)
            {
                case P4T.KEYINFO_STATIC:
                case P4T.KEYINFO:
                    {
                        KeyInfo keyInfo = op.P4.KeyInfo;
                        Debug.Assert(keyInfo.SortOrders != null);
                        C.__snprintf(temp, tempLength, "keyinfo(%d", keyInfo.Fields);
                        int i = temp.Length;
                        for (int j = 0; j < keyInfo.Fields; j++)
                        {
                            CollSeq coll = keyInfo.Colls[j];
                            if (coll != null)
                            {
                                string collName = coll.Name;
                                int collNameLength = collName.Length;
                                if (i + collNameLength > tempLength)
                                {
                                    temp.Append(",...");
                                    break;
                                }
                                temp.Append(",");
                                if (keyInfo.SortOrders != null && keyInfo.SortOrders[j] != 0)
                                    temp.Append("-");
                                temp.Append(coll.Name);
                                i += collNameLength;
                            }
                            else if (i + 4 < tempLength)
                            {
                                temp.Append(",nil");
                                i += 4;
                            }
                        }
                        temp.Append(")");
                        Debug.Assert(i < tempLength);
                        break;
                    }
                case P4T.COLLSEQ:
                    {
                        CollSeq coll = op.P4.Coll;
                        C.__snprintf(temp, tempLength, "collseq(%.20s)", (coll != null ? coll.Name : "null"));
                        break;
                    }
                case P4T.FUNCDEF:
                    {
                        FuncDef def = op.P4.Func;
                        C.__snprintf(temp, tempLength, "%s(%d)", def.Name, def.Args);
                        break;
                    }
                case P4T.INT64:
                    {
                        C.__snprintf(temp, tempLength, "%lld", op.P4.I64);
                        break;
                    }
                case P4T.INT32:
                    {
                        C.__snprintf(temp, tempLength, "%d", op.P4.I);
                        break;
                    }
                case P4T.REAL:
                    {
                        C.__snprintf(temp, tempLength, "%.16g", op.P4.Real);
                        break;
                    }
                case P4T.MEM:
                    {
                        Mem mem = op.P4.Mem;
                        Debug.Assert((mem.Flags & MEM.Null) == 0);
                        if ((mem.Flags & MEM.Str) != 0) temp.Append(mem.Z);
                        else if ((mem.Flags & MEM.Int) != 0) C.__snprintf(temp, tempLength, "%lld", mem.u.I);
                        else if ((mem.Flags & MEM.Real) != 0) C.__snprintf(temp, tempLength, "%.16g", mem.R);
                        else { Debug.Assert((mem.Flags & MEM.Blob) != 0); temp = new StringBuilder("(blob)"); }
                        break;
                    }
#if !OMIT_VIRTUALTABLE
                case P4T.VTAB:
                    {
                        IVTable vtable = op.P4.VTable.IVTable;
                        C.__snprintf(temp, tempLength, "vtab:%p:%p", vtable, vtable.IModule);
                        break;
                    }
#endif
                case P4T.INTARRAY:
                    {
                        C.__snprintf(temp, tempLength, "intarray");
                        break;
                    }
                case P4T.SUBPROGRAM:
                    {
                        C.__snprintf(temp, tempLength, "program");
                        break;
                    }
                default:
                    {
                        if (op.P4.Z != null)
                            temp.Append(op.P4.Z);
                        break;
                    }
            }
            Debug.Assert(temp != null);
            return temp.ToString();
        }
#endif

        public void UsesBtree(int i)
        {
            Debug.Assert(i >= 0 && i < Ctx.DBs.length && i < (int)sizeof(yDbMask) * 8);
            Debug.Assert(i < (int)sizeof(yDbMask) * 8);
            BtreeMask |= ((yDbMask)1) << i;
            if (i != 1 && Ctx.DBs[i].Bt.Sharable())
                LockMask |= ((yDbMask)1) << i;
        }

#if !(OMIT_SHARED_CACHE) && THREADSAFE
        public void Enter()
        {
            if (LockMask == 0) return;  // The common case
            Context ctx = Ctx;
            array_t<BContext.DB> dbs = ctx.DBs;
            int i;
            yDbMask mask;
            for (i = 0, mask = 1; i < dbs.length; i++, mask += mask)
                if (i != 1 && (mask & LockMask) != 0 && C._ALWAYS(dbs[i].Bt != null))
                    dbs[i].Bt.Enter();
        }

        public void Leave()
        {
            if (LockMask == 0) return;  // The common case
            Context ctx = Ctx;
            array_t<BContext.DB> dbs = ctx.DBs;
            int i;
            yDbMask mask;
            for (i = 0, mask = 1; i < dbs.length; i++, mask += mask)
                if (i != 1 && (mask & LockMask) != 0 && C._ALWAYS(dbs[i].Bt != null))
                    dbs[i].Bt.Leave();
        }
#endif


#if VDBE_PROFILE || DEBUG
        public static void PrintOp(FILE out_, int pc, VdbeOp op)
        {
            if (out_ == null) out_ = Console.Out;
            StringBuilder ptr = new StringBuilder(50);
            string p4 = DisplayP4(op, ptr, ptr.Length);
            StringBuilder out2_ = new StringBuilder(10);
            C.__snprintf(out2_, 999, "%4d %-13s %4d %4d %4d %-4s %.2X %s\n", pc,
            OpcodeName(op.Opcode), op.P1, op.P2, op.P3, p4, op.P5,
#if  DEBUG
 (op.Comment != null ? op.Comment : string.Empty)
#else
 string.Empty
#endif
);
            out_.Write(out2_);
        }
#endif

        static void ReleaseMemArray(Mem[] p, int n) { ReleaseMemArray(p, 0, n); }
        static void ReleaseMemArray(Mem[] p, int offset, int n)
        {
            if (p != null && p.Length > offset && p[offset] != null && n != 0)
            {
                Context ctx = p[offset].Ctx;
                bool mallocFailed = ctx.MallocFailed;
                if (ctx != null && ctx.BytesFreed != 0)
                {
                    for (int pIdx = offset; pIdx < n; pIdx++)
                        C._tagfree(ctx, ref p[pIdx].TextBuilder_);
                    return;
                }
                Mem end = p[n];
                for (int pIdx = offset; pIdx < n; pIdx++)
                {
                    var p2 = p[pIdx];
                    Debug.Assert((p[pIdx + 1]) == end || n == 1 || pIdx == p.Length - 1 || p[pIdx].Ctx == p[pIdx + 1].Ctx);

                    // This block is really an inlined version of sqlite3VdbeMemRelease() that takes advantage of the fact that the memory cell value is 
                    // being set to NULL after releasing any dynamic resources.
                    //
                    // The justification for duplicating code is that according to callgrind, this causes a certain test case to hit the CPU 4.7 
                    // percent less (x86 linux, gcc version 4.1.2, -O6) than if sqlite3MemRelease() were called from here. With -O2, this jumps
                    // to 6.6 percent. The test case is inserting 1000 rows into a table with no indexes using a single prepared INSERT statement, bind() 
                    // and reset(). Inserts are grouped into a transaction.
                    if (p2 != null)
                    {
                        if ((p2.Flags & (MEM.Agg | MEM.Dyn | MEM.Frame | MEM.RowSet)) != 0)
                            MemRelease(p2);
                        else if (p2.Mem_ != null)
                        {
                            C._tagfree(ctx, ref p2.Mem_);
                            p2.Mem_ = null;
                        }
                        else if (p2.TextBuilder_ != null)
                        {
                            C._tagfree(ctx, ref p2.TextBuilder_);
                            p2.TextBuilder_ = null;
                        }
                        p2.Flags = MEM.Invalid;
                        p2.Z = null;
                    }
                }
                ctx.MallocFailed = mallocFailed;
            }
        }

        public static void FrameDelete(VdbeFrame p)
        {
            int i;
            //Mem[] mems = VdbeFrameMem(p);
            VdbeCursor[] cursors = p.aChildCsr;// (VdbeCursor)aMem[p.nChildMem];
            for (i = 0; i < p.nChildCsr; i++)
                FreeCursor(p.v, cursors[i]);
            ReleaseMemArray(p.aChildMem, p.nChildMem);
            p = null;// sqlite3DbFree( p.v.db, p );
        }

#if !OMIT_EXPLAIN
        public RC List()
        {
            Context ctx = Ctx; // The database connection
            int i;

            Debug.Assert(HasExplain != 0);
            Debug.Assert(Magic == VDBE_MAGIC_RUN);
            Debug.Assert(RC_ == RC.OK || RC_ == RC.BUSY || RC_ == RC.NOMEM);

            // Even though this opcode does not use dynamic strings for the result, result columns may become dynamic if the user calls
            // sqlite3_column_text16(), causing a translation to UTF-16 encoding.
            if (ResultSet == null) ResultSet = new Mem[0]; //Mem* pMem = p.pResultSet = p.aMem[1];
            ReleaseMemArray(ResultSet, 8);
            ResultSet = null;

            if (RC_ == RC.NOMEM)
            {
                // This happens if a malloc() inside a call to sqlite3_column_text() or sqlite3_column_text16() failed.
                ctx.MallocFailed = true;
                return RC.ERROR;
            }

            // When the number of output rows reaches rows, that means the listing has finished and sqlite3_step() should return SQLITE_DONE.
            // rows is the sum of the number of rows in the main program, plus the sum of the number of rows in all trigger subprograms encountered
            // so far.  The rows value will increase as new trigger subprograms are encountered, but p->pc will eventually catch up to rows.
            int rows = Ops.length; // Stop when row count reaches this
            SubProgram[] subs = null; // Array of sub-vdbes
            Mem sub = null; // Memory cell hold array of subprogs
            int subsLength = 0; // Number of sub-vdbes seen so far

            if (HasExplain == 1)
            {
                // The first 8 memory cells are used for the result set.  So we will commandeer the 9th cell to use as storage for an array of pointers
                // to trigger subprograms.  The VDBE is guaranteed to have at least 9 cells.
                Debug.Assert(Mems.length > 9);
                sub = Mems[9];
                if ((sub.Flags & MEM.Blob) != 0)
                {
                    // On the first call to sqlite3_step(), sub will hold a NULL.  It is initialized to a BLOB by the P4_SUBPROGRAM processing logic below
                    subsLength = subs.Length; //: sub->N/sizeof(Vdbe *);
                    subs = Mems[9].SubProgram_; //: (SubProgram **)sub->Z
                }
                for (i = 0; i < subsLength; i++)
                    rows += subs[i].Ops.length;
            }

            int memIdx = 0;
            if (memIdx >= ResultSet.Length) Array.Resize(ref ResultSet, 8 + ResultSet.Length);
            ResultSet[memIdx] = C._alloc(ResultSet[memIdx]);
            Mem mem = ResultSet[memIdx++]; // First Mem of result set

            do
            {
                i = PC++;
            } while (i < rows && HasExplain == 2 && Ops[i].Opcode != OP.Explain);

            RC rc = RC.OK;
            if (i >= rows)
            {
                RC_ = RC.OK;
                rc = RC.DONE;
            }
            else if (ctx.u1.IsInterrupted)
            {
                RC_ = RC.INTERRUPT;
                rc = RC.ERROR;
                sqlite3SetString(ref ErrMsg, ctx, sqlite3ErrStr(RC_));
            }
            else
            {
                string z;
                VdbeOp op;
                if (i < Ops.length)
                    op = Ops[i]; // The output line number is small enough that we are still in the main program.
                else
                {
                    // We are currently listing subprograms.  Figure out which one and pick up the appropriate opcode.
                    i -= Ops.length;
                    int j;
                    for (j = 0; i >= subs[j].Ops.length; j++)
                        i -= subs[j].Ops.length;
                    op = subs[j].Ops[i];
                }
                if (HasExplain == 1)
                {
                    mem.Flags = MEM.Int;
                    mem.Type = TYPE.INTEGER;
                    mem.u.I = i; // Program counter
                    if (ResultSet[memIdx] == null) ResultSet[memIdx] = C._alloc(ResultSet[memIdx]);
                    mem = ResultSet[memIdx++]; //: mem++;

                    mem.Flags = MEM.Static | MEM.Str | MEM.Term;
                    mem.Z = OpcodeName(op.Opcode); // Opcode
                    Debug.Assert(mem.Z != null);
                    mem.N = mem.Z.Length;
                    mem.Type = TYPE.TEXT;
                    mem.Encode = TEXTENCODE.UTF8;
                    if (ResultSet[memIdx] == null) ResultSet[memIdx] = C._alloc(ResultSet[memIdx]);
                    mem = ResultSet[memIdx++]; //: mem++;

                    // When an OP_Program opcode is encounter (the only opcode that has a P4_SUBPROGRAM argument), expand the size of the array of subprograms
                    // kept in p->aMem[9].z to hold the new program - assuming this subprogram has not already been seen.
                    if (op.P4Type == P4T.SUBPROGRAM)
                    {
                        //: int bytes = (subsLength+1)*sizeof(SubProgram *);
                        int j;
                        for (j = 0; j < subsLength; j++)
                            if (subs[j] == op.P4.Program)
                                break;
                        if (j == subsLength)
                        {
                            //: && MemGrow(sub, bytes, subsLength != 0) == RC.OK)
                            Array.Resize(ref subs, subsLength + 1);
                            sub.SubProgram_ = subs; //: (SubProgram)pSub.z;
                            subs[subsLength++] = op.P4.Program;
                            sub.Flags |= MEM.Blob;
                            sub.N = 0; //: subsLength*sizeof(SubProgram *);
                        }
                    }
                }

                mem.Flags = MEM.Int;
                mem.u.I = op.P1; // P1
                mem.Type = TYPE.INTEGER;
                if (ResultSet[memIdx] == null) ResultSet[memIdx] = C._alloc(ResultSet[memIdx]);
                mem = ResultSet[memIdx++]; //: mem++;

                mem.Flags = MEM.Int;
                mem.u.I = op.P2; // P2
                mem.Type = TYPE.INTEGER;
                if (ResultSet[memIdx] == null) ResultSet[memIdx] = C._alloc(ResultSet[memIdx]);
                mem = ResultSet[memIdx++]; //: mem++;

                mem.Flags = MEM.Int;
                mem.u.I = op.P3; // P3
                mem.Type = TYPE.INTEGER;
                if (ResultSet[memIdx] == null) ResultSet[memIdx] = C._alloc(ResultSet[memIdx]);
                mem = ResultSet[memIdx++]; //: mem++;

                //if (MemGrow(mem, 32, 0) != 0) // P4
                //{
                //    Debug.Assert(ctx.MallocFailed);
                //    return RC.ERROR;
                //}

                mem.Flags = MEM.Dyn | MEM.Str | MEM.Term;
                z = DisplayP4(op, mem.Z, 32);
                if (z != mem.Z)
                    MemSetStr(mem, z, -1, TEXTENCODE.UTF8, null);
                else
                {
                    Debug.Assert(mem.Z != null);
                    mem.N = mem.Z.Length;
                    mem.Encode = TEXTENCODE.UTF8;
                }
                mem.Type = TYPE.TEXT;
                if (ResultSet[memIdx] == null) ResultSet[memIdx] = C._alloc(ResultSet[memIdx]);
                mem = ResultSet[memIdx++]; //: mem++;

                if (HasExplain == 1)
                {
                    //if (MemGrow(mem, 4, 0) != 0)
                    //{
                    //    Debug.Assert(ctx.MallocFailed);
                    //    return RC.ERROR;
                    //}
                    mem.Flags = MEM.Dyn | MEM.Str | MEM.Term;
                    mem.N = 2;
                    mem.Z = op.P5.ToString("x2"); //: __snprintf(mem->Z, 3, "%.2x", op->P5); // P5
                    mem.Type = TYPE.TEXT;
                    mem.Encode = TEXTENCODE.UTF8;
                    if (ResultSet[memIdx] == null) ResultSet[memIdx] = C._alloc(ResultSet[memIdx]);
                    mem = ResultSet[memIdx++]; // mem++;

#if DEBUG
                    if (op.Comment != null)
                    {
                        mem.Flags = MEM.Str | MEM.Term;
                        mem.Z = op.Comment; // Comment
                        mem.N = (mem.Z != null ? mem.Z.Length : 0);
                        mem.Encode = TEXTENCODE.UTF8;
                        mem.Type = TYPE.TEXT;
                    }
                    else
#endif
                    {
                        mem.Flags = MEM.Null;
                        mem.Type = TYPE.NULL;
                    }
                }

                ResColumns = (ushort)(8 - 4 * (HasExplain - 1));
                RC_ = RC.OK;
                rc = RC.ROW;
            }
            return rc;
        }
#endif

#if  DEBUG
        public void PrintSql()
        {
            int opsLength = Ops.length;
            if (opsLength < 1) return;
            VdbeOp op = Ops[0];
            if (op.Opcode == OP.Trace && op.P4.Z != null)
            {
                string z = op.P4.Z;
                z = z.Trim(); //: while (_isspace(*z)) z++;
                Console.Write("SQL: [{0}]\n", z);
            }
        }
#endif

#if !OMIT_TRACE && ENABLE_IOTRACE
        public void IOTraceSql()
        {
            if (!SysEx.IOTrace) return;
            int opsLength = Ops.length;
            if (opsLength < 1) return;
            VdbeOp op = Ops[0];
            if (op.Opcode == OP.Trace && op.P4.Z != null)
            {
                string z = string.Empty;
                C.__snprintf(z, 1000, "%s", op.P4.Z);
                SysEx.IOTRACE("SQL %s\n", z.Trim());
            }
        }
#endif

        //: __device__ static void *AllocSpace(void *buf, int bytes, uint8 **from, uint8 *end, int *bytesOut)
        //: {
        //:     _assert(SysEx_HASALIGNMENT8(*from));
        //:     if (buf) return buf;
        //:     bytes = SysEx_ROUND8(bytes);
        //:     if (&(*from)[bytes] <= end)
        //:     {
        //:         buf = (void *)*from;
        //:         *from += bytes;
        //:     }
        //:     else
        //:         *bytesOut += bytes;
        //:     return buf;
        //: }

        public void Rewind()
        {
            Debug.Assert(Magic == VDBE_MAGIC_INIT);
            Debug.Assert(Ops.length > 0); // There should be at least one opcode.

            // Set the magic to VDBE_MAGIC_RUN sooner rather than later.
            Magic = VDBE_MAGIC_RUN;
#if DEBUG
            for (int i = 1; i < Mems.length; i++)
                Debug.Assert(Mems[i].Ctx == Ctx);
#endif
            PC = -1;
            RC_ = RC.OK;
            ErrorAction = OE.Abort;
            Magic = VDBE_MAGIC_RUN;
            Changes = 0;
            CacheCtr = 1;
            MinWriteFileFormat = 255;
            StatementID = 0;
            FkConstraints = 0;
#if VDBE_PROFILE
            for (int i = 0; i < Ops.length; i++)
            {
                Ops[i].Cnt = 0;
                Ops[i].Cycles = 0;
            }
#endif
        }

        public void MakeReady(Parse parse)
        {
            Debug.Assert(parse != null);
            Debug.Assert(Magic == VDBE_MAGIC_INIT);
            Context ctx = Ctx; // The database connection
            Debug.Assert(!ctx.MallocFailed);
            int vars = parse.Vars.length; // Number of parameters
            int mems = parse.Mems; // Number of VM memory registers
            int cursors = parse.Tabs; // Number of cursors required
            int args = parse.MaxArgs; // Number of arguments in subprograms
            int onces = parse.Onces; // Number of OP_Once instructions
            if (onces == 0) onces = 1; // Ensure at least one byte in p->aOnceFlag[]

            // For each cursor required, also allocate a memory cell. Memory cells (nMem+1-nCursor)..nMem, inclusive, will never be used by
            // the vdbe program. Instead they are used to allocate space for VdbeCursor/BtCursor structures. The blob of memory associated with 
            // cursor 0 is stored in memory cell nMem. Memory cell (nMem-1) stores the blob of memory associated with cursor 1, etc.
            //
            // See also: allocateCursor().
            mems += cursors;

            // Allocate space for memory registers, SQL variables, VDBE cursors and an array to marshal SQL function arguments in.
            //: uint8* csr = (uint8*)&Ops[Ops.length]; // Memory avaliable for allocation
            //: uint8* end = (uint8*)&Ops[OpsAlloc]; // First byte past end of zCsr[]
            ResolveP2Values(this, ref args);
            UsesStmtJournal = (parse.IsMultiWrite != 0 && parse.MayAbort);
            if (parse.Explain != 0 && mems < 10)
                mems = 10;
            //: _memset(csr, 0, end - csr);
            //: csr += (csr - (uint8*)0) & 7;
            //: _assert(SysEx_HASALIGNMENT8(csr));
            Expired = false;

            // Memory for registers, parameters, cursor, etc, is allocated in two passes.  On the first pass, we try to reuse unused space at the 
            // end of the opcode array.  If we are unable to satisfy all memory requirements by reusing the opcode array tail, then the second
            // pass will fill in the rest using a fresh allocation.  
            //
            // This two-pass approach that reuses as much memory as possible from the leftover space at the end of the opcode array can significantly
            // reduce the amount of memory held by a prepared statement.
            int n;
            //int bytes; // How much extra memory is needed
            //: do
            //: {
            //:     bytes = 0;
            //:     Mems.data = (Mem*)AllocSpace(Mems.data, mems * sizeof(Mem), &csr, end, &bytes);
            //:     Vars.data = (Mem*)AllocSpace(Vars.data, vars * sizeof(Mem), &csr, end, &bytes);
            //:     Args = (Mem**)AllocSpace(Args, args * sizeof(Mem*), &csr, end, &bytes);
            //:     VarNames.data = (char**)AllocSpace(VarNames.data, vars * sizeof(char*), &csr, end, &bytes);
            //:     Cursors.data = (VdbeCursor**)AllocSpace(Cursors.data, cursors * sizeof(VdbeCursor*), &csr, end, &bytes);
            //:     OnceFlags.data = (uint8*)AllocSpace(OnceFlags.data, onces, &csr, end, &bytes);
            //:     if (bytes)
            //:         FreeThis = _tagalloc2(ctx, bytes, true);
            //:     csr = FreeThis;
            //:     end = &csr[bytes];
            //: } while (bytes && !ctx->MallocFailed);
            Mems.data = new Mem[mems + 1]; //C#: aMem is 1 based, so allocate 1 extra cell
            Vars.data = new Mem[vars == 0 ? 1 : vars];
            Args = new Mem[args == 0 ? 1 : args];
            VarNames.data = new string[vars == 0 ? 1 : vars];            
            Cursors.data = new VdbeCursor[cursors == 0 ? 1 : cursors]; Cursors[0] = new VdbeCursor();
            OnceFlags.data = new byte[onces == 0 ? 1 : onces];

            Cursors.length = cursors;
            OnceFlags.length = onces;
            if (Vars.data != null)
            {
                Vars.length = (yVar)vars;
                for (n = 0; n < vars; n++)
                {
                    Vars[n] = C._alloc(Vars[n])
                    Vars[n].Flags = MEM.Null;
                    Vars[n].Ctx = ctx;
                }
            }
            if (VarNames.data != null)
            {
                VarNames.length = parse.Vars.length;
                for (n = 0; n < VarNames.length; n++)
                {
                    VarNames[n] = parse.Vars[n]; //: _memcpy(VarNames.data, parse->Vars.data, VarNames.length*sizeof(VarNames[0]));
                    parse.Vars[n] = string.Empty; //: _memset(parse->Vars.data, 0, parse->Vars.length*sizeof(parse->Vars[0]));
                }
            }
            if (Mems.data != null)
            {
                //: Mems.data--; // aMem[] goes from 1..nMem
                Mems.length = mems; // not from 0..nMem-1
                for (n = 0; n <= mems; n++)
                {
                    Mems[n] = C._alloc(Mems[n]);
                    Mems[n].Flags = MEM.Null;
                    Mems[n].Ctx = ctx;
                }
            }
            HasExplain = parse.Explain;
            Rewind();
        }

        public void FreeCursor(VdbeCursor cur)
        {
            if (cur == null)
                return;

            if (cur.Bt != null)
                cur.Bt.Close(); // The pCx.pCursor will be close automatically, if it exists, by the call above.
            else if (cur.Cursor != null)
                Btree.CloseCursor(cur.Cursor);
#if !OMIT_VIRTUALTABLE
            if (cur.VtabCursor != null)
            {
                IVTableCursor vtabCursor = cur.VtabCursor;
                ITableModule module = cur.IModule;
                InVtabMethod = 1;
                module.Close(vtabCursor);
                InVtabMethod = 0;
            }
#endif
        }

        public static int FrameRestore(VdbeFrame frame)
        {
            Vdbe v = frame.V;
            v.OnceFlags.data = frame.OnceFlags.data;
            v.OnceFlags.length = frame.OnceFlags.length;
            v.Ops.data = frame.Ops.data;
            v.Ops.length = frame.Ops.length;
            v.Mems.data = frame.Mems.data;
            v.Mems.length = frame.Mems.length;
            v.Cursors.data = frame.Cursors.data;
            v.Cursors.length = frame.Cursors.length;
            v.Ctx.LastRowID = frame.LastRowID;
            v.Changes = frame.Changes;
            return frame.PC;
        }

        static void CloseAllCursors(Vdbe p)
        {
            if (p.Frames.data != null)
            {
                VdbeFrame frame;
                for (frame = p.Frames.data; frame.Parent != null; frame = frame.Parent) { }
                FrameRestore(frame);
            }
            p.Frames.data = null;
            p.Frames.length = 0;
            if (p.Cursors.data != null)
            {
                int i;
                for (i = 0; i < p.Cursors.length; i++)
                {
                    VdbeCursor cur = p.Cursors[i];
                    if (cur != null)
                    {
                        p.FreeCursor(cur);
                        p.Cursors[i] = null;
                    }
                }
            }
            if (p.Mems.data != null)
                ReleaseMemArray(p.Mems.data, 1, p.Mems.length);
            while (p.DelFrames != null)
            {
                VdbeFrame del = p.DelFrame;
                p.DelFrame = del.Parent;
                FrameDelete(del);
            }
        }

        static void Cleanup(Vdbe p)
        {
            Context ctx = p.Ctx;
#if DEBUG
            // Execute assert() statements to ensure that the Vdbe.apCsr[] and Vdbe.aMem[] arrays have already been cleaned up.
            int i;
            if (p.Cursors.data != null) for (i = 0; i <= p.Cursors.length; i++) Debug.Assert(p.Cursors[i] == null);
            if (p.Mems.data != null)
                for (i = 1; i <= p.Mems.length; i++) Debug.Assert(p.Mems[i].Flags == MEM.Invalid);
#endif
            C._tagfree(ctx, ref p.ErrMsg);
            p.ErrMsg = null;
            p.ResultSet = null;
        }

        public void SetNumCols(int resColumns)
        {
            Context ctx = Ctx;
            ReleaseMemArray(ColNames, ResColumns * COLNAME_N);
            C._tagfree(ctx, ref ColNames);
            int n = resColumns * COLNAME_N;
            ResColumns = (ushort)resColumns;
            Mem colName;
            ColNames = new Mem[n]; //: (Mem *)_tagalloc2(ctx, sizeof(Mem)*n, true);
            if (ColNames == null) return;
            while (n-- > 0)
            {
                ColNames[n] = C._tagalloc(ColNames[n]);
                colName = ColNames[n];
                colName.Flags = MEM.Null;
                colName.Ctx = ctx;
            }
        }

        public RC SetColName(int idx, int var, string name, Action<object> del)
        {
            Debug.Assert(idx < ResColumns);
            Debug.Assert(var < COLNAME_N);
            if (Ctx.MallocFailed)
            {
                Debug.Assert(name == null || del != C.DESTRUCTOR_DYNAMIC);
                return RC.NOMEM;
            }
            Debug.Assert(ColNames != null);
            Mem colName = ColNames[idx + var * ResColumns];
            RC rc = MemSetStr(colName, name, -1, TEXTENCODE.UTF8, del);
            Debug.Assert(rc != 0 || name == null || (colName.Flags & MEM.Term) != 0);
            return rc;
        }

        static RC VdbeCommit(Context ctx, Vdbe p)
        {
#if !OMIT_VIRTUALTABLE
            // With this option, sqlite3VtabSync() is defined to be simply SQLITE_OK so p is not used. 
#endif

            // Before doing anything else, call the xSync() callback for any virtual module tables written in this transaction. This has to
            // be done before determining whether a master journal file is required, as an xSync() callback may add an attached database
            // to the transaction.
            RC rc = VTable.Sync(ctx, ref p.ErrMsg);

            int trans = 0; // Number of databases with an active write-transaction
            bool needXcommit = false;

            // This loop determines (a) if the commit hook should be invoked and (b) how many database files have open write transactions, not 
            // including the temp database. (b) is important because if more than one database file has an open write transaction, a master journal
            // file is required for an atomic commit.
            int i;
            for (i = 0; rc == RC.OK && i < ctx.DBs.length; i++)
            {
                Btree bt = ctx.DBs[i].Bt;
                if (bt.IsInTrans())
                {
                    needXcommit = true;
                    if (i != 1) trans++;
                    bt.Enter();
                    rc = bt.get_Pager().ExclusiveLock();
                    bt.Leave();
                }
            }
            if (rc != RC.OK)
                return rc;

            // If there are any write-transactions at all, invoke the commit hook
            if (needXcommit && ctx.CommitCallback != null)
            {
                rc = ctx.CommitCallback(ctx.CommitArg);
                if (rc != 0)
                    return RC.CONSTRAINT_COMMITHOOK;
            }

            // The simple case - no more than one database file (not counting the TEMP database) has a transaction active.   There is no need for the
            // master-journal.
            //
            // If the return value of sqlite3BtreeGetFilename() is a zero length string, it means the main database is :memory: or a temp file.  In 
            // that case we do not support atomic multi-file commits, so use the simple case then too.
            if (ctx.DBs[0].Bt.get_Filename().Length == 0 || trans <= 1)
            {
                for (i = 0; rc == RC.OK && i < ctx.DBs.length; i++)
                {
                    Btree bt = ctx.DBs[i].Bt;
                    if (bt != null)
                        rc = bt.CommitPhaseOne(null);
                }

                // Do the commit only if all databases successfully complete phase 1. If one of the BtreeCommitPhaseOne() calls fails, this indicates an
                // IO error while deleting or truncating a journal file. It is unlikely, but could happen. In this case abandon processing and return the error.
                for (i = 0; rc == RC.OK && i < ctx.DBs.length; i++)
                {
                    Btree bt = ctx.DBs[i].Bt;
                    if (bt != null)
                        rc = bt.CommitPhaseTwo(false);
                }
                if (rc == RC.OK)
                    VTable.Commit(ctx);
            }

            // The complex case - There is a multi-file write-transaction active. This requires a master journal file to ensure the transaction is
            // committed atomicly.
#if !OMIT_DISKIO
            else
            {
                sqlite3_vfs pVfs = ctx.pVfs;
                bool needSync = false;
                string zMaster = "";   /* File-name for the master journal */
                string zMainFile = sqlite3BtreeGetFilename(ctx.aDb[0].pBt);
                sqlite3_file pMaster = null;
                i64 offset = 0;
                int res = 0;

                /* Select a master journal file name */
                do
                {
                    i64 iRandom = 0;
                    sqlite3DbFree(ctx, ref zMaster);
                    sqlite3_randomness(sizeof(u32), ref iRandom);//random.Length
                    zMaster = sqlite3MPrintf(ctx, "%s-mj%08X", zMainFile, iRandom & 0x7fffffff);
                    //if (!zMaster)
                    //{
                    //  return SQLITE_NOMEM;
                    //}
                    sqlite3FileSuffix3(zMainFile, zMaster);
                    rc = sqlite3OsAccess(pVfs, zMaster, SQLITE_ACCESS_EXISTS, ref res);
                } while (rc == SQLITE_OK && res == 1);
                if (rc == SQLITE_OK)
                {
                    /* Open the master journal. */
                    rc = sqlite3OsOpenMalloc(ref pVfs, zMaster, ref pMaster,
                    SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE |
                    SQLITE_OPEN_EXCLUSIVE | SQLITE_OPEN_MASTER_JOURNAL, ref rc
                    );
                }
                if (rc != SQLITE_OK)
                {
                    sqlite3DbFree(ctx, ref zMaster);
                    return rc;
                }

                /* Write the name of each database file in the transaction into the new
                ** master journal file. If an error occurs at this point close
                ** and delete the master journal file. All the individual journal files
                ** still have 'null' as the master journal pointer, so they will roll
                ** back independently if a failure occurs.
                */
                for (i = 0; i < ctx.nDb; i++)
                {
                    Btree pBt = ctx.aDb[i].pBt;
                    if (sqlite3BtreeIsInTrans(pBt))
                    {
                        string zFile = sqlite3BtreeGetJournalname(pBt);
                        if (zFile == null)
                        {
                            continue;  /* Ignore TEMP and :memory: databases */
                        }
                        Debug.Assert(zFile != "");
                        if (!needSync && 0 == sqlite3BtreeSyncDisabled(pBt))
                        {
                            needSync = true;
                        }
                        rc = sqlite3OsWrite(pMaster, Encoding.UTF8.GetBytes(zFile), sqlite3Strlen30(zFile), offset);
                        offset += sqlite3Strlen30(zFile);
                        if (rc != SQLITE_OK)
                        {
                            sqlite3OsCloseFree(pMaster);
                            sqlite3OsDelete(pVfs, zMaster, 0);
                            sqlite3DbFree(ctx, ref zMaster);
                            return rc;
                        }
                    }
                }

                /* Sync the master journal file. If the IOCAP_SEQUENTIAL device
                ** flag is set this is not required.
                */
                if (needSync
                && 0 == (sqlite3OsDeviceCharacteristics(pMaster) & SQLITE_IOCAP_SEQUENTIAL)
                && SQLITE_OK != (rc = sqlite3OsSync(pMaster, SQLITE_SYNC_NORMAL))
                )
                {
                    sqlite3OsCloseFree(pMaster);
                    sqlite3OsDelete(pVfs, zMaster, 0);
                    sqlite3DbFree(ctx, ref zMaster);
                    return rc;
                }

                /* Sync all the db files involved in the transaction. The same call
                ** sets the master journal pointer in each individual journal. If
                ** an error occurs here, do not delete the master journal file.
                **
                ** If the error occurs during the first call to
                ** sqlite3BtreeCommitPhaseOne(), then there is a chance that the
                ** master journal file will be orphaned. But we cannot delete it,
                ** in case the master journal file name was written into the journal
                ** file before the failure occurred.
                */
                for (i = 0; rc == SQLITE_OK && i < ctx.nDb; i++)
                {
                    Btree pBt = ctx.aDb[i].pBt;
                    if (pBt != null)
                    {
                        rc = sqlite3BtreeCommitPhaseOne(pBt, zMaster);
                    }
                }
                sqlite3OsCloseFree(pMaster);
                Debug.Assert(rc != SQLITE_BUSY);
                if (rc != SQLITE_OK)
                {
                    sqlite3DbFree(ctx, ref zMaster);
                    return rc;
                }

                /* Delete the master journal file. This commits the transaction. After
                ** doing this the directory is synced again before any individual
                ** transaction files are deleted.
                */
                rc = sqlite3OsDelete(pVfs, zMaster, 1);
                sqlite3DbFree(ctx, ref zMaster);
                if (rc != 0)
                {
                    return rc;
                }

                /* All files and directories have already been synced, so the following
                ** calls to sqlite3BtreeCommitPhaseTwo() are only closing files and
                ** deleting or truncating journals. If something goes wrong while
                ** this is happening we don't really care. The integrity of the
                ** transaction is already guaranteed, but some stray 'cold' journals
                ** may be lying around. Returning an error code won't help matters.
                */
#if SQLITE_TEST
        disable_simulated_io_errors();
#endif
                sqlite3BeginBenignMalloc();
                for (i = 0; i < ctx.nDb; i++)
                {
                    Btree pBt = ctx.aDb[i].pBt;
                    if (pBt != null)
                    {
                        sqlite3BtreeCommitPhaseTwo(pBt, 0);
                    }
                }
                sqlite3EndBenignMalloc();
#if SQLITE_TEST
        enable_simulated_io_errors();
#endif
                sqlite3VtabCommit(ctx);
            }
#endif

            return rc;
        }

        /*
        ** This routine checks that the sqlite3.activeVdbeCnt count variable
        ** matches the number of vdbe's in the list sqlite3.pVdbe that are
        ** currently active. An Debug.Assertion fails if the two counts do not match.
        ** This is an internal self-check only - it is not an essential processing
        ** step.
        **
        ** This is a no-op if NDEBUG is defined.
        */
#if !NDEBUG
        static void checkActiveVdbeCnt(sqlite3 db)
        {
            Vdbe p;
            int cnt = 0;
            int nWrite = 0;
            p = db.pVdbe;
            while (p != null)
            {
                if (p.magic == VDBE_MAGIC_RUN && p.pc >= 0)
                {
                    cnt++;
                    if (p.readOnly == false)
                        nWrite++;
                }
                p = p.pNext;
            }
            Debug.Assert(cnt == db.activeVdbeCnt);
            Debug.Assert(nWrite == db.writeVdbeCnt);
        }
#else
//#define checkActiveVdbeCnt(x)
static void checkActiveVdbeCnt( sqlite3 db ){}
#endif

        static void invalidateCursorsOnModifiedBtrees(sqlite3 db)
        {
            int i;
            for (i = 0; i < db.nDb; i++)
            {
                Btree p = db.aDb[i].pBt;
                if (p != null && sqlite3BtreeIsInTrans(p))
                {
                    sqlite3BtreeTripAllCursors(p, SQLITE_ABORT);
                }
            }
        }

        /*
        ** If the Vdbe passed as the first argument opened a statement-transaction,
        ** close it now. Argument eOp must be either SAVEPOINT_ROLLBACK or
        ** SAVEPOINT_RELEASE. If it is SAVEPOINT_ROLLBACK, then the statement
        ** transaction is rolled back. If eOp is SAVEPOINT_RELEASE, then the
        ** statement transaction is commtted.
        **
        ** If an IO error occurs, an SQLITE_IOERR_XXX error code is returned.
        ** Otherwise SQLITE_OK.
        */
        static int sqlite3VdbeCloseStatement(Vdbe p, int eOp)
        {
            sqlite3 db = p.db;
            int rc = SQLITE_OK;
            /* If p->iStatement is greater than zero, then this Vdbe opened a 
            ** statement transaction that should be closed here. The only exception
            ** is that an IO error may have occured, causing an emergency rollback.
            ** In this case (db->nStatement==0), and there is nothing to do.
            */
            if (db.nStatement != 0 && p.iStatement != 0)
            {
                int i;
                int iSavepoint = p.iStatement - 1;

                Debug.Assert(eOp == SAVEPOINT_ROLLBACK || eOp == SAVEPOINT_RELEASE);
                Debug.Assert(db.nStatement > 0);
                Debug.Assert(p.iStatement == (db.nStatement + db.nSavepoint));

                for (i = 0; i < db.nDb; i++)
                {
                    int rc2 = SQLITE_OK;
                    Btree pBt = db.aDb[i].pBt;
                    if (pBt != null)
                    {
                        if (eOp == SAVEPOINT_ROLLBACK)
                        {
                            rc2 = sqlite3BtreeSavepoint(pBt, SAVEPOINT_ROLLBACK, iSavepoint);
                        }
                        if (rc2 == SQLITE_OK)
                        {
                            rc2 = sqlite3BtreeSavepoint(pBt, SAVEPOINT_RELEASE, iSavepoint);
                        }
                        if (rc == SQLITE_OK)
                        {
                            rc = rc2;
                        }
                    }
                }
                db.nStatement--;
                p.iStatement = 0;

                if (rc == SQLITE_OK)
                {
                    if (eOp == SAVEPOINT_ROLLBACK)
                    {
                        rc = sqlite3VtabSavepoint(db, SAVEPOINT_ROLLBACK, iSavepoint);
                    }
                    if (rc == SQLITE_OK)
                    {
                        rc = sqlite3VtabSavepoint(db, SAVEPOINT_RELEASE, iSavepoint);
                    }
                }

                /* If the statement transaction is being rolled back, also restore the 
                ** database handles deferred constraint counter to the value it had when 
                ** the statement transaction was opened.  */
                if (eOp == SAVEPOINT_ROLLBACK)
                {
                    db.nDeferredCons = p.nStmtDefCons;
                }
            }
            return rc;
        }

        /*
        ** This function is called when a transaction opened by the database 
        ** handle associated with the VM passed as an argument is about to be 
        ** committed. If there are outstanding deferred foreign key constraint
        ** violations, return SQLITE_ERROR. Otherwise, SQLITE_OK.
        **
        ** If there are outstanding FK violations and this function returns 
        ** SQLITE_ERROR, set the result of the VM to SQLITE_CONSTRAINT and write
        ** an error message to it. Then return SQLITE_ERROR.
        */
#if !SQLITE_OMIT_FOREIGN_KEY
        static int sqlite3VdbeCheckFk(Vdbe p, int deferred)
        {
            sqlite3 db = p.db;
            if ((deferred != 0 && db.nDeferredCons > 0) || (0 == deferred && p.nFkConstraint > 0))
            {
                p.rc = SQLITE_CONSTRAINT;
                p.errorAction = OE_Abort;
                sqlite3SetString(ref p.zErrMsg, db, "foreign key constraint failed");
                return SQLITE_ERROR;
            }
            return SQLITE_OK;
        }
#endif

        /*
** This routine is called the when a VDBE tries to halt.  If the VDBE
** has made changes and is in autocommit mode, then commit those
** changes.  If a rollback is needed, then do the rollback.
**
** This routine is the only way to move the state of a VM from
** SQLITE_MAGIC_RUN to SQLITE_MAGIC_HALT.  It is harmless to
** call this on a VM that is in the SQLITE_MAGIC_HALT state.
**
** Return an error code.  If the commit could not complete because of
** lock contention, return SQLITE_BUSY.  If SQLITE_BUSY is returned, it
** means the close did not happen and needs to be repeated.
*/
        static int sqlite3VdbeHalt(Vdbe p)
        {
            int rc;                         /* Used to store transient return codes */
            sqlite3 db = p.db;

            /* This function contains the logic that determines if a statement or
            ** transaction will be committed or rolled back as a result of the
            ** execution of this virtual machine.
            **
            ** If any of the following errors occur:
            **
            **     SQLITE_NOMEM
            **     SQLITE_IOERR
            **     SQLITE_FULL
            **     SQLITE_INTERRUPT
            **
            ** Then the internal cache might have been left in an inconsistent
            ** state.  We need to rollback the statement transaction, if there is
            ** one, or the complete transaction if there is no statement transaction.
            */

            //if ( p.db.mallocFailed != 0 )
            //{
            //  p.rc = SQLITE_NOMEM;
            //}
            CloseAllCursors(p);
            if (p.magic != VDBE_MAGIC_RUN)
            {
                return SQLITE_OK;
            }
            checkActiveVdbeCnt(db);

            /* No commit or rollback needed if the program never started */
            if (p.pc >= 0)
            {
                int mrc;   /* Primary error code from p.rc */
                int eStatementOp = 0;
                bool isSpecialError = false;            /* Set to true if a 'special' error */

                /* Lock all btrees used by the statement */
                Enter(p);
                /* Check for one of the special errors */
                mrc = p.rc & 0xff;
                Debug.Assert(p.rc != SQLITE_IOERR_BLOCKED);  /* This error no longer exists */
                isSpecialError = mrc == SQLITE_NOMEM || mrc == SQLITE_IOERR
                || mrc == SQLITE_INTERRUPT || mrc == SQLITE_FULL;
                if (isSpecialError)
                {
                    /* If the query was read-only and the error code is SQLITE_INTERRUPT, 
                    ** no rollback is necessary. Otherwise, at least a savepoint 
                    ** transaction must be rolled back to restore the database to a 
                    ** consistent state.
                    **
                    ** Even if the statement is read-only, it is important to perform
                    ** a statement or transaction rollback operation. If the error 
                    ** occured while writing to the journal, sub-journal or database
                    ** file as part of an effort to free up cache space (see function
                    ** pagerStress() in pager.c), the rollback is required to restore 
                    ** the pager to a consistent state.
                    */
                    if (!p.readOnly || mrc != SQLITE_INTERRUPT)
                    {
                        if ((mrc == SQLITE_NOMEM || mrc == SQLITE_FULL) && p.usesStmtJournal)
                        {
                            eStatementOp = SAVEPOINT_ROLLBACK;
                        }
                        else
                        {
                            /* We are forced to roll back the active transaction. Before doing
                            ** so, abort any other statements this handle currently has active.
                            */
                            invalidateCursorsOnModifiedBtrees(db);
                            sqlite3RollbackAll(db);
                            sqlite3CloseSavepoints(db);
                            db.autoCommit = 1;
                        }
                    }
                }

                /* Check for immediate foreign key violations. */
                if (p.rc == SQLITE_OK)
                {
                    sqlite3VdbeCheckFk(p, 0);
                }

                /* If the auto-commit flag is set and this is the only active writer
                ** VM, then we do either a commit or rollback of the current transaction.
                **
                ** Note: This block also runs if one of the special errors handled
                ** above has occurred.
                */
                if (!sqlite3VtabInSync(db)
                && db.autoCommit != 0
                && db.writeVdbeCnt == ((p.readOnly == false) ? 1 : 0)
                )
                {
                    if (p.rc == SQLITE_OK || (p.errorAction == OE_Fail && !isSpecialError))
                    {
                        rc = sqlite3VdbeCheckFk(p, 1);
                        if (rc != SQLITE_OK)
                        {
                            if (NEVER(p.readOnly))
                            {
                                sqlite3VdbeLeave(p);
                                return SQLITE_ERROR;
                            }
                            rc = SQLITE_CONSTRAINT;
                        }
                        else
                        {
                            /* The auto-commit flag is true, the vdbe program was successful 
                            ** or hit an 'OR FAIL' constraint and there are no deferred foreign
                            ** key constraints to hold up the transaction. This means a commit 
                            ** is required. */
                            rc = VdbeCommit(db, p);
                        }
                        if (rc == SQLITE_BUSY && p.readOnly)
                        {
                            sqlite3VdbeLeave(p);
                            return SQLITE_BUSY;
                        }
                        else if (rc != SQLITE_OK)
                        {
                            p.rc = rc;
                            sqlite3RollbackAll(db);
                        }
                        else
                        {
                            db.nDeferredCons = 0;
                            sqlite3CommitInternalChanges(db);
                        }
                    }
                    else
                    {
                        sqlite3RollbackAll(db);
                    }
                    db.nStatement = 0;
                }
                else if (eStatementOp == 0)
                {
                    if (p.rc == SQLITE_OK || p.errorAction == OE_Fail)
                    {
                        eStatementOp = SAVEPOINT_RELEASE;
                    }
                    else if (p.errorAction == OE_Abort)
                    {
                        eStatementOp = SAVEPOINT_ROLLBACK;
                    }
                    else
                    {
                        invalidateCursorsOnModifiedBtrees(db);
                        sqlite3RollbackAll(db);
                        sqlite3CloseSavepoints(db);
                        db.autoCommit = 1;
                    }
                }

                /* If eStatementOp is non-zero, then a statement transaction needs to
                ** be committed or rolled back. Call sqlite3VdbeCloseStatement() to
                ** do so. If this operation returns an error, and the current statement
                ** error code is SQLITE_OK or SQLITE_CONSTRAINT, then promote the
                ** current statement error code.
                */
                if (eStatementOp != 0)
                {
                    rc = sqlite3VdbeCloseStatement(p, eStatementOp);
                    if (rc != 0)
                    {
                        if (p.rc == SQLITE_OK || p.rc == SQLITE_CONSTRAINT)
                        {
                            p.rc = rc;
                            sqlite3DbFree(db, ref p.zErrMsg);
                            p.zErrMsg = null;
                        }
                        invalidateCursorsOnModifiedBtrees(db);
                        sqlite3RollbackAll(db);
                        sqlite3CloseSavepoints(db);
                        db.autoCommit = 1;
                    }
                }

                /* If this was an INSERT, UPDATE or DELETE and no statement transaction
                ** has been rolled back, update the database connection change-counter.
                */
                if (p.changeCntOn)
                {
                    if (eStatementOp != SAVEPOINT_ROLLBACK)
                    {
                        sqlite3VdbeSetChanges(db, p.nChange);
                    }
                    else
                    {
                        sqlite3VdbeSetChanges(db, 0);
                    }
                    p.nChange = 0;
                }

                /* Rollback or commit any schema changes that occurred. */
                if (p.rc != SQLITE_OK && (db.flags & SQLITE_InternChanges) != 0)
                {
                    sqlite3ResetInternalSchema(db, -1);
                    db.flags = (db.flags | SQLITE_InternChanges);
                }

                /* Release the locks */
                sqlite3VdbeLeave(p);
            }

            /* We have successfully halted and closed the VM.  Record this fact. */
            if (p.pc >= 0)
            {
                db.activeVdbeCnt--;
                if (!p.readOnly)
                {
                    db.writeVdbeCnt--;
                }
                Debug.Assert(db.activeVdbeCnt >= db.writeVdbeCnt);
            }
            p.magic = VDBE_MAGIC_HALT;
            checkActiveVdbeCnt(db);
            //if ( p.db.mallocFailed != 0 )
            //{
            //  p.rc = SQLITE_NOMEM;
            //}
            /* If the auto-commit flag is set to true, then any locks that were held
            ** by connection db have now been released. Call sqlite3ConnectionUnlocked()
            ** to invoke any required unlock-notify callbacks.
            */
            if (db.autoCommit != 0)
            {
                sqlite3ConnectionUnlocked(db);
            }

            Debug.Assert(db.activeVdbeCnt > 0 || db.autoCommit == 0 || db.nStatement == 0);
            return (p.rc == SQLITE_BUSY ? SQLITE_BUSY : SQLITE_OK);
        }


        /*
        ** Each VDBE holds the result of the most recent sqlite3_step() call
        ** in p.rc.  This routine sets that result back to SQLITE_OK.
        */
        static void sqlite3VdbeResetStepResult(Vdbe p)
        {
            p.rc = SQLITE_OK;
        }

        /*
        ** Clean up a VDBE after execution but do not delete the VDBE just yet.
        ** Write any error messages into pzErrMsg.  Return the result code.
        **
        ** After this routine is run, the VDBE should be ready to be executed
        ** again.
        **
        ** To look at it another way, this routine resets the state of the
        ** virtual machine from VDBE_MAGIC_RUN or VDBE_MAGIC_HALT back to
        ** VDBE_MAGIC_INIT.
        */
        static int sqlite3VdbeReset(Vdbe p)
        {
            sqlite3 db;
            db = p.db;

            /* If the VM did not run to completion or if it encountered an
            ** error, then it might not have been halted properly.  So halt
            ** it now.
            */
            sqlite3VdbeHalt(p);

            /* If the VDBE has be run even partially, then transfer the error code
            ** and error message from the VDBE into the main database structure.  But
            ** if the VDBE has just been set to run but has not actually executed any
            ** instructions yet, leave the main database error information unchanged.
            */
            if (p.pc >= 0)
            {
                //if ( p.zErrMsg != 0 ) // Always exists under C#
                {
                    sqlite3BeginBenignMalloc();
                    sqlite3ValueSetStr(db.pErr, -1, p.zErrMsg == null ? "" : p.zErrMsg, SQLITE_UTF8, SQLITE_TRANSIENT);
                    sqlite3EndBenignMalloc();
                    db.errCode = p.rc;
                    sqlite3DbFree(db, ref p.zErrMsg);
                    p.zErrMsg = "";
                }
                //else if ( p.rc != 0 )
                //{
                //  sqlite3Error( db, p.rc, 0 );
                //}
                //else
                //{
                //  sqlite3Error( db, SQLITE_OK, 0 );
                //}
                if (p.runOnlyOnce != 0)
                    p.expired = true;
            }
            else if (p.rc != 0 && p.expired)
            {
                /* The expired flag was set on the VDBE before the first call
                ** to sqlite3_step(). For consistency (since sqlite3_step() was
                ** called), set the database error in this case as well.
                */
                sqlite3Error(db, p.rc, 0);
                sqlite3ValueSetStr(db.pErr, -1, p.zErrMsg, SQLITE_UTF8, SQLITE_TRANSIENT);
                sqlite3DbFree(db, ref p.zErrMsg);
                p.zErrMsg = "";
            }

            /* Reclaim all memory used by the VDBE
            */
            Cleanup(p);

            /* Save profiling information from this VDBE run.
            */
#if  VDBE_PROFILE && TODO
{
FILE *out = fopen("vdbe_profile.out", "a");
if( out ){
int i;
fprintf(out, "---- ");
for(i=0; i<p.nOp; i++){
fprintf(out, "%02x", p.aOp[i].opcode);
}
fprintf(out, "\n");
for(i=0; i<p.nOp; i++){
fprintf(out, "%6d %10lld %8lld ",
p.aOp[i].cnt,
p.aOp[i].cycles,
p.aOp[i].cnt>0 ? p.aOp[i].cycles/p.aOp[i].cnt : 0
);
sqlite3VdbePrintOp(out, i, p.aOp[i]);
}
fclose(out);
}
}
#endif
            p.magic = VDBE_MAGIC_INIT;
            return p.rc & db.errMask;
        }

        /*
        ** Clean up and delete a VDBE after execution.  Return an integer which is
        ** the result code.  Write any error message text into pzErrMsg.
        */
        static int sqlite3VdbeFinalize(ref Vdbe p)
        {
            int rc = SQLITE_OK;
            if (p.magic == VDBE_MAGIC_RUN || p.magic == VDBE_MAGIC_HALT)
            {
                rc = sqlite3VdbeReset(p);
                Debug.Assert((rc & p.db.errMask) == rc);
            }
            sqlite3VdbeDelete(ref p);
            return rc;
        }

        /*
        ** Call the destructor for each auxdata entry in pVdbeFunc for which
        ** the corresponding bit in mask is clear.  Auxdata entries beyond 31
        ** are always destroyed.  To destroy all auxdata entries, call this
        ** routine with mask==0.
        */
        static void sqlite3VdbeDeleteAuxData(VdbeFunc pVdbeFunc, int mask)
        {
            int i;
            for (i = 0; i < pVdbeFunc.nAux; i++)
            {
                AuxData pAux = pVdbeFunc.apAux[i];
                if ((i > 31 || (mask & (((u32)1) << i)) == 0 && pAux.pAux != null))
                {
                    if (pAux.pAux != null && pAux.pAux is IDisposable)
                    {
                        (pAux.pAux as IDisposable).Dispose();
                    }
                    pAux.pAux = null;
                }
            }
        }

        /*
        ** Free all memory associated with the Vdbe passed as the second argument.
        ** The difference between this function and sqlite3VdbeDelete() is that
        ** VdbeDelete() also unlinks the Vdbe from the list of VMs associated with
        ** the database connection.
        */
        static void sqlite3VdbeDeleteObject(sqlite3 db, ref Vdbe p)
        {
            SubProgram pSub, pNext;
            int i;
            Debug.Assert(p.db == null || p.db == db);
            ReleaseMemArray(p.aVar, p.nVar);
            ReleaseMemArray(p.aColName, p.nResColumn, COLNAME_N);
            for (pSub = p.pProgram; pSub != null; pSub = pNext)
            {
                pNext = pSub.pNext;
                VdbeFreeOpArray(db, ref pSub.aOp, pSub.nOp);
                sqlite3DbFree(db, ref pSub);
            }
            //for ( i = p->nzVar - 1; i >= 0; i-- )
            //  sqlite3DbFree( db, p.azVar[i] );
            VdbeFreeOpArray(db, ref p.aOp, p.nOp);
            sqlite3DbFree(db, ref p.aLabel);
            sqlite3DbFree(db, ref p.aColName);
            sqlite3DbFree(db, ref p.zSql);
            sqlite3DbFree(db, ref p.pFree);
            // Free memory allocated from db within p
            //sqlite3DbFree( db, p );
        }

        /*
        ** Delete an entire VDBE.
        */
        static void sqlite3VdbeDelete(ref Vdbe p)
        {
            sqlite3 db;
            if (NEVER(p == null))
                return;
            Cleanup(p);
            db = p.db;
            if (p.pPrev != null)
            {
                p.pPrev.pNext = p.pNext;
            }
            else
            {
                Debug.Assert(db.pVdbe == p);
                db.pVdbe = p.pNext;
            }
            if (p.pNext != null)
            {
                p.pNext.pPrev = p.pPrev;
            }
            p.magic = VDBE_MAGIC_DEAD;
            p.db = null;
            sqlite3VdbeDeleteObject(db, ref p);
        }

        /*
        ** Make sure the cursor p is ready to read or write the row to which it
        ** was last positioned.  Return an error code if an OOM fault or I/O error
        ** prevents us from positioning the cursor to its correct position.
        **
        ** If a MoveTo operation is pending on the given cursor, then do that
        ** MoveTo now.  If no move is pending, check to see if the row has been
        ** deleted out from under the cursor and if it has, mark the row as
        ** a NULL row.
        **
        ** If the cursor is already pointing to the correct row and that row has
        ** not been deleted out from under the cursor, then this routine is a no-op.
        */
        static int sqlite3VdbeCursorMoveto(VdbeCursor p)
        {
            if (p.deferredMoveto)
            {
                int res = 0;
                int rc;
#if  SQLITE_TEST
        //extern int sqlite3_search_count;
#endif
                Debug.Assert(p.isTable);
                rc = sqlite3BtreeMovetoUnpacked(p.pCursor, null, p.movetoTarget, 0, ref res);
                if (rc != 0)
                    return rc;
                p.lastRowid = p.movetoTarget;
                if (res != 0)
                    return SQLITE_CORRUPT_BKPT();
                p.rowidIsValid = true;
#if  SQLITE_TEST
#if !TCLSH
        sqlite3_search_count++;
#else
        sqlite3_search_count.iValue++;
#endif
#endif
                p.deferredMoveto = false;
                p.cacheStatus = CACHE_STALE;
            }
            else if (ALWAYS(p.pCursor != null))
            {
                int hasMoved = 0;
                int rc = sqlite3BtreeCursorHasMoved(p.pCursor, ref hasMoved);
                if (rc != 0)
                    return rc;
                if (hasMoved != 0)
                {
                    p.cacheStatus = CACHE_STALE;
                    p.nullRow = true;
                }
            }
            return SQLITE_OK;
        }

        /*
        ** The following functions:
        **
        ** sqlite3VdbeSerialType()
        ** sqlite3VdbeSerialTypeLen()
        ** sqlite3VdbeSerialLen()
        ** sqlite3VdbeSerialPut()
        ** sqlite3VdbeSerialGet()
        **
        ** encapsulate the code that serializes values for storage in SQLite
        ** data and index records. Each serialized value consists of a
        ** 'serial-type' and a blob of data. The serial type is an 8-byte unsigned
        ** integer, stored as a varint.
        **
        ** In an SQLite index record, the serial type is stored directly before
        ** the blob of data that it corresponds to. In a table record, all serial
        ** types are stored at the start of the record, and the blobs of data at
        ** the end. Hence these functions allow the caller to handle the
        ** serial-type and data blob seperately.
        **
        ** The following table describes the various storage classes for data:
        **
        **   serial type        bytes of data      type
        **   --------------     ---------------    ---------------
        **      0                     0            NULL
        **      1                     1            signed integer
        **      2                     2            signed integer
        **      3                     3            signed integer
        **      4                     4            signed integer
        **      5                     6            signed integer
        **      6                     8            signed integer
        **      7                     8            IEEE float
        **      8                     0            Integer constant 0
        **      9                     0            Integer constant 1
        **     10,11                               reserved for expansion
        **    N>=12 and even       (N-12)/2        BLOB
        **    N>=13 and odd        (N-13)/2        text
        **
        ** The 8 and 9 types were added in 3.3.0, file format 4.  Prior versions
        ** of SQLite will not understand those serial types.
        */

        /*
        ** Return the serial-type for the value stored in pMem.
        */
        static u32 sqlite3VdbeSerialType(Mem pMem, int file_format)
        {
            int flags = pMem.flags;
            int n;

            if ((flags & MEM_Null) != 0)
            {
                return 0;
            }
            if ((flags & MEM_Int) != 0)
            {
                /* Figure out whether to use 1, 2, 4, 6 or 8 bytes. */
                const i64 MAX_6BYTE = ((((i64)0x00008000) << 32) - 1);
                i64 i = pMem.u.i;
                u64 u;
                if (file_format >= 4 && (i & 1) == i)
                {
                    return 8 + (u32)i;
                }
                if (i < 0)
                {
                    if (i < (-MAX_6BYTE))
                        return 6;
                    /* Previous test prevents:  u = -(-9223372036854775808) */
                    u = (u64)(-i);
                }
                else
                {
                    u = (u64)i;
                }
                if (u <= 127)
                    return 1;
                if (u <= 32767)
                    return 2;
                if (u <= 8388607)
                    return 3;
                if (u <= 2147483647)
                    return 4;
                if (u <= MAX_6BYTE)
                    return 5;
                return 6;
            }
            if ((flags & MEM_Real) != 0)
            {
                return 7;
            }
            Debug.Assert( /* pMem.db.mallocFailed != 0 || */ (flags & (MEM_Str | MEM_Blob)) != 0);
            n = pMem.n;
            if ((flags & MEM_Zero) != 0)
            {
                n += pMem.u.nZero;
            }
            else if ((flags & MEM_Blob) != 0)
            {
                n = pMem.zBLOB != null ? pMem.zBLOB.Length : pMem.z != null ? pMem.z.Length : 0;
            }
            else
            {
                if (pMem.z != null)
                    n = Encoding.UTF8.GetByteCount(pMem.n < pMem.z.Length ? pMem.z.Substring(0, pMem.n) : pMem.z);
                else
                    n = pMem.zBLOB.Length;
                pMem.n = n;
            }

            Debug.Assert(n >= 0);
            return (u32)((n * 2) + 12 + (((flags & MEM_Str) != 0) ? 1 : 0));
        }

        /*
        ** Return the length of the data corresponding to the supplied serial-type.
        */
        static u32[] aSize = new u32[] { 0, 1, 2, 3, 4, 6, 8, 8, 0, 0, 0, 0 };
        static u32 sqlite3VdbeSerialTypeLen(u32 serial_type)
        {
            if (serial_type >= 12)
            {
                return (u32)((serial_type - 12) / 2);
            }
            else
            {
                return aSize[serial_type];
            }
        }

        /*
        ** If we are on an architecture with mixed-endian floating
        ** points (ex: ARM7) then swap the lower 4 bytes with the
        ** upper 4 bytes.  Return the result.
        **
        ** For most architectures, this is a no-op.
        **
        ** (later):  It is reported to me that the mixed-endian problem
        ** on ARM7 is an issue with GCC, not with the ARM7 chip.  It seems
        ** that early versions of GCC stored the two words of a 64-bit
        ** float in the wrong order.  And that error has been propagated
        ** ever since.  The blame is not necessarily with GCC, though.
        ** GCC might have just copying the problem from a prior compiler.
        ** I am also told that newer versions of GCC that follow a different
        ** ABI get the byte order right.
        **
        ** Developers using SQLite on an ARM7 should compile and run their
        ** application using -DSQLITE_DEBUG=1 at least once.  With DEBUG
        ** enabled, some Debug.Asserts below will ensure that the byte order of
        ** floating point values is correct.
        **
        ** (2007-08-30)  Frank van Vugt has studied this problem closely
        ** and has send his findings to the SQLite developers.  Frank
        ** writes that some Linux kernels offer floating point hardware
        ** emulation that uses only 32-bit mantissas instead of a full
        ** 48-bits as required by the IEEE standard.  (This is the
        ** CONFIG_FPE_FASTFPE option.)  On such systems, floating point
        ** byte swapping becomes very complicated.  To avoid problems,
        ** the necessary byte swapping is carried out using a 64-bit integer
        ** rather than a 64-bit float.  Frank assures us that the code here
        ** works for him.  We, the developers, have no way to independently
        ** verify this, but Frank seems to know what he is talking about
        ** so we trust him.
        */
#if  SQLITE_MIXED_ENDIAN_64BIT_FLOAT
//static u64 floatSwap(u64 in){
//  union {
//    u64 r;
//    u32 i[2];
//  } u;
//  u32 t;

//  u.r = in;
//  t = u.i[0];
//  u.i[0] = u.i[1];
//  u.i[1] = t;
//  return u.r;
//}
//# define swapMixedEndianFloat(X)  X = floatSwap(X)
#else
        //# define swapMixedEndianFloat(X)
#endif

        /*
** Write the serialized data blob for the value stored in pMem into
** buf. It is assumed that the caller has allocated sufficient space.
** Return the number of bytes written.
**
** nBuf is the amount of space left in buf[].  nBuf must always be
** large enough to hold the entire field.  Except, if the field is
** a blob with a zero-filled tail, then buf[] might be just the right
** size to hold everything except for the zero-filled tail.  If buf[]
** is only big enough to hold the non-zero prefix, then only write that
** prefix into buf[].  But if buf[] is large enough to hold both the
** prefix and the tail then write the prefix and set the tail to all
** zeros.
**
** Return the number of bytes actually written into buf[].  The number
** of bytes in the zero-filled tail is included in the return value only
** if those bytes were zeroed in buf[].
*/
        static u32 sqlite3VdbeSerialPut(byte[] buf, int offset, int nBuf, Mem pMem, int file_format)
        {
            u32 serial_type = sqlite3VdbeSerialType(pMem, file_format);
            u32 len;

            /* Integer and Real */
            if (serial_type <= 7 && serial_type > 0)
            {
                u64 v;
                u32 i;
                if (serial_type == 7)
                {
                    //Debug.Assert( sizeof( v) == sizeof(pMem.r));
#if WINDOWS_PHONE || WINDOWS_MOBILE
v = (ulong)BitConverter.ToInt64(BitConverter.GetBytes(pMem.r),0);
#else
                    v = (ulong)BitConverter.DoubleToInt64Bits(pMem.r);// memcpy( &v, pMem.r, v ).Length;
#endif
#if  SQLITE_MIXED_ENDIAN_64BIT_FLOAT
swapMixedEndianFloat( v );
#endif
                }
                else
                {
                    v = (ulong)pMem.u.i;
                }
                len = i = sqlite3VdbeSerialTypeLen(serial_type);
                Debug.Assert(len <= (u32)nBuf);
                while (i-- != 0)
                {
                    buf[offset + i] = (u8)(v & 0xFF);
                    v >>= 8;
                }
                return len;
            }

            /* String or blob */
            if (serial_type >= 12)
            {
                // TO DO -- PASS TESTS WITH THIS ON Debug.Assert( pMem.n + ( ( pMem.flags & MEM_Zero ) != 0 ? pMem.u.nZero : 0 ) == (int)sqlite3VdbeSerialTypeLen( serial_type ) );
                Debug.Assert(pMem.n <= nBuf);
                if ((len = (u32)pMem.n) != 0)
                    if (pMem.zBLOB == null && String.IsNullOrEmpty(pMem.z))
                    {
                    }
                    else if (pMem.zBLOB != null && ((pMem.flags & MEM_Blob) != 0 || pMem.z == null))
                        Buffer.BlockCopy(pMem.zBLOB, 0, buf, offset, (int)len);//memcpy( buf, pMem.z, len );
                    else
                        Buffer.BlockCopy(Encoding.UTF8.GetBytes(pMem.z), 0, buf, offset, (int)len);//memcpy( buf, pMem.z, len );
                if ((pMem.flags & MEM_Zero) != 0)
                {
                    len += (u32)pMem.u.nZero;
                    Debug.Assert(nBuf >= 0);
                    if (len > (u32)nBuf)
                    {
                        len = (u32)nBuf;
                    }
                    Array.Clear(buf, offset + pMem.n, (int)(len - pMem.n));// memset( &buf[pMem.n], 0, len - pMem.n );
                }
                return len;
            }

            /* NULL or constants 0 or 1 */
            return 0;
        }

        /*
        ** Deserialize the data blob pointed to by buf as serial type serial_type
        ** and store the result in pMem.  Return the number of bytes read.
        */
        static u32 sqlite3VdbeSerialGet(
        byte[] buf,         /* Buffer to deserialize from */
        int offset,         /* Offset into Buffer */
        u32 serial_type,    /* Serial type to deserialize */
        Mem pMem            /* Memory cell to write value into */
        )
        {
            switch (serial_type)
            {
                case 10:   /* Reserved for future use */
                case 11:   /* Reserved for future use */
                case 0:
                    {  /* NULL */
                        pMem.flags = MEM_Null;
                        pMem.n = 0;
                        pMem.z = null;
                        pMem.zBLOB = null;
                        break;
                    }
                case 1:
                    { /* 1-byte signed integer */
                        pMem.u.i = (sbyte)buf[offset + 0];
                        pMem.flags = MEM_Int;
                        return 1;
                    }
                case 2:
                    { /* 2-byte signed integer */
                        pMem.u.i = (int)((((sbyte)buf[offset + 0]) << 8) | buf[offset + 1]);
                        pMem.flags = MEM_Int;
                        return 2;
                    }
                case 3:
                    { /* 3-byte signed integer */
                        pMem.u.i = (int)((((sbyte)buf[offset + 0]) << 16) | (buf[offset + 1] << 8) | buf[offset + 2]);
                        pMem.flags = MEM_Int;
                        return 3;
                    }
                case 4:
                    { /* 4-byte signed integer */
                        pMem.u.i = (int)(((sbyte)buf[offset + 0] << 24) | (buf[offset + 1] << 16) | (buf[offset + 2] << 8) | buf[offset + 3]);
                        pMem.flags = MEM_Int;
                        return 4;
                    }
                case 5:
                    { /* 6-byte signed integer */
                        u64 x = (ulong)((((sbyte)buf[offset + 0]) << 8) | buf[offset + 1]);
                        u32 y = (u32)((buf[offset + 2] << 24) | (buf[offset + 3] << 16) | (buf[offset + 4] << 8) | buf[offset + 5]);
                        x = (x << 32) | y;
                        pMem.u.i = (i64)x;
                        pMem.flags = MEM_Int;
                        return 6;
                    }
                case 6:   /* 8-byte signed integer */
                case 7:
                    { /* IEEE floating point */
                        u64 x;
                        u32 y;
#if !NDEBUG && !SQLITE_OMIT_FLOATING_POINT
                        /* Verify that integers and floating point values use the same
** byte order.  Or, that if SQLITE_MIXED_ENDIAN_64BIT_FLOAT is
** defined that 64-bit floating point values really are mixed
** endian.
*/
                        const u64 t1 = ((u64)0x3ff00000) << 32;
                        const double r1 = 1.0;
                        u64 t2 = t1;
#if  SQLITE_MIXED_ENDIAN_64BIT_FLOAT
swapMixedEndianFloat(t2);
#endif
                        Debug.Assert(sizeof(double) == sizeof(u64) && memcmp(BitConverter.GetBytes(r1), BitConverter.GetBytes(t2), sizeof(double)) == 0);//Debug.Assert( sizeof(r1)==sizeof(t2) && memcmp(&r1, t2, sizeof(r1))==0 );
#endif

                        x = (u64)((buf[offset + 0] << 24) | (buf[offset + 1] << 16) | (buf[offset + 2] << 8) | buf[offset + 3]);
                        y = (u32)((buf[offset + 4] << 24) | (buf[offset + 5] << 16) | (buf[offset + 6] << 8) | buf[offset + 7]);
                        x = (x << 32) | y;
                        if (serial_type == 6)
                        {
                            pMem.u.i = (i64)x;
                            pMem.flags = MEM_Int;
                        }
                        else
                        {
                            Debug.Assert(sizeof(i64) == 8 && sizeof(double) == 8);
#if  SQLITE_MIXED_ENDIAN_64BIT_FLOAT
swapMixedEndianFloat(x);
#endif
#if WINDOWS_PHONE || WINDOWS_MOBILE
              pMem.r = BitConverter.ToDouble(BitConverter.GetBytes((long)x), 0);
#else
                            pMem.r = BitConverter.Int64BitsToDouble((long)x);// memcpy(pMem.r, x, sizeof(x))
#endif
                            pMem.flags = (u16)(sqlite3IsNaN(pMem.r) ? MEM_Null : MEM_Real);
                        }
                        return 8;
                    }
                case 8:    /* Integer 0 */
                case 9:
                    {  /* Integer 1 */
                        pMem.u.i = serial_type - 8;
                        pMem.flags = MEM_Int;
                        return 0;
                    }
                default:
                    {
                        u32 len = (serial_type - 12) / 2;
                        pMem.n = (int)len;
                        pMem.xDel = null;
                        if ((serial_type & 0x01) != 0)
                        {
                            pMem.flags = MEM_Str | MEM_Ephem;
                            if (len <= buf.Length - offset)
                            {
                                pMem.z = Encoding.UTF8.GetString(buf, offset, (int)len);//memcpy( buf, pMem.z, len );
                                pMem.n = pMem.z.Length;
                            }
                            else
                            {
                                pMem.z = ""; // Corrupted Data
                                pMem.n = 0;
                            }
                            pMem.zBLOB = null;
                        }
                        else
                        {
                            pMem.z = null;
                            pMem.zBLOB = sqlite3Malloc((int)len);
                            pMem.flags = MEM_Blob | MEM_Ephem;
                            if (len <= buf.Length - offset)
                            {
                                Buffer.BlockCopy(buf, offset, pMem.zBLOB, 0, (int)len);//memcpy( buf, pMem.z, len );
                            }
                            else
                            {
                                Buffer.BlockCopy(buf, offset, pMem.zBLOB, 0, buf.Length - offset - 1);
                            }
                        }
                        return len;
                    }
            }
            return 0;
        }

        static int sqlite3VdbeSerialGet(
        byte[] buf,     /* Buffer to deserialize from */
        u32 serial_type,              /* Serial type to deserialize */
        Mem pMem                     /* Memory cell to write value into */
        )
        {
            switch (serial_type)
            {
                case 10:   /* Reserved for future use */
                case 11:   /* Reserved for future use */
                case 0:
                    {  /* NULL */
                        pMem.flags = MEM_Null;
                        break;
                    }
                case 1:
                    { /* 1-byte signed integer */
                        pMem.u.i = (sbyte)buf[0];
                        pMem.flags = MEM_Int;
                        return 1;
                    }
                case 2:
                    { /* 2-byte signed integer */
                        pMem.u.i = (int)(((buf[0]) << 8) | buf[1]);
                        pMem.flags = MEM_Int;
                        return 2;
                    }
                case 3:
                    { /* 3-byte signed integer */
                        pMem.u.i = (int)(((buf[0]) << 16) | (buf[1] << 8) | buf[2]);
                        pMem.flags = MEM_Int;
                        return 3;
                    }
                case 4:
                    { /* 4-byte signed integer */
                        pMem.u.i = (int)((buf[0] << 24) | (buf[1] << 16) | (buf[2] << 8) | buf[3]);
                        pMem.flags = MEM_Int;
                        return 4;
                    }
                case 5:
                    { /* 6-byte signed integer */
                        u64 x = (ulong)(((buf[0]) << 8) | buf[1]);
                        u32 y = (u32)((buf[2] << 24) | (buf[3] << 16) | (buf[4] << 8) | buf[5]);
                        x = (x << 32) | y;
                        pMem.u.i = (i64)x;
                        pMem.flags = MEM_Int;
                        return 6;
                    }
                case 6:   /* 8-byte signed integer */
                case 7:
                    { /* IEEE floating point */
                        u64 x;
                        u32 y;
#if !NDEBUG && !SQLITE_OMIT_FLOATING_POINT
                        /* Verify that integers and floating point values use the same
** byte order.  Or, that if SQLITE_MIXED_ENDIAN_64BIT_FLOAT is
** defined that 64-bit floating point values really are mixed
** endian.
*/
                        const u64 t1 = ((u64)0x3ff00000) << 32;
                        const double r1 = 1.0;
                        u64 t2 = t1;
#if  SQLITE_MIXED_ENDIAN_64BIT_FLOAT
swapMixedEndianFloat(t2);
#endif
                        Debug.Assert(sizeof(double) == sizeof(u64) && memcmp(BitConverter.GetBytes(r1), BitConverter.GetBytes(t2), sizeof(double)) == 0);//Debug.Assert( sizeof(r1)==sizeof(t2) && memcmp(&r1, t2, sizeof(r1))==0 );
#endif

                        x = (u64)((buf[0] << 24) | (buf[1] << 16) | (buf[2] << 8) | buf[3]);
                        y = (u32)((buf[4] << 24) | (buf[5] << 16) | (buf[6] << 8) | buf[7]);
                        x = (x << 32) | y;
                        if (serial_type == 6)
                        {
                            pMem.u.i = (i64)x;
                            pMem.flags = MEM_Int;
                        }
                        else
                        {
                            Debug.Assert(sizeof(i64) == 8 && sizeof(double) == 8);
#if  SQLITE_MIXED_ENDIAN_64BIT_FLOAT
swapMixedEndianFloat(x);
#endif
#if WINDOWS_PHONE || WINDOWS_MOBILE
              pMem.r = BitConverter.ToDouble(BitConverter.GetBytes((long)x), 0);
#else
                            pMem.r = BitConverter.Int64BitsToDouble((long)x);// memcpy(pMem.r, x, sizeof(x))
#endif
                            pMem.flags = MEM_Real;
                        }
                        return 8;
                    }
                case 8:    /* Integer 0 */
                case 9:
                    {  /* Integer 1 */
                        pMem.u.i = serial_type - 8;
                        pMem.flags = MEM_Int;
                        return 0;
                    }
                default:
                    {
                        int len = (int)((serial_type - 12) / 2);
                        pMem.xDel = null;
                        if ((serial_type & 0x01) != 0)
                        {
                            pMem.flags = MEM_Str | MEM_Ephem;
                            pMem.z = Encoding.UTF8.GetString(buf, 0, len);//memcpy( buf, pMem.z, len );
                            pMem.n = pMem.z.Length;// len;
                            pMem.zBLOB = null;
                        }
                        else
                        {
                            pMem.flags = MEM_Blob | MEM_Ephem;
                            pMem.zBLOB = sqlite3Malloc(len);
                            buf.CopyTo(pMem.zBLOB, 0);
                            pMem.n = len;// len;
                            pMem.z = null;
                        }
                        return len;
                    }
            }
            return 0;
        }

        /*
        ** Given the nKey-byte encoding of a record in pKey[], parse the
        ** record into a UnpackedRecord structure.  Return a pointer to
        ** that structure.
        **
        ** The calling function might provide szSpace bytes of memory
        ** space at pSpace.  This space can be used to hold the returned
        ** VDbeParsedRecord structure if it is large enough.  If it is
        ** not big enough, space is obtained from sqlite3Malloc().
        **
        ** The returned structure should be closed by a call to
        ** sqlite3VdbeDeleteUnpackedRecord().
        */
        static UnpackedRecord sqlite3VdbeRecordUnpack(
        KeyInfo pKeyInfo,   /* Information about the record format */
        int nKey,           /* Size of the binary record */
        byte[] pKey,        /* The binary record */
        UnpackedRecord pSpace, //  char *pSpace,          /* Unaligned space available to hold the object */
        int szSpace         /* Size of pSpace[] in bytes */
        )
        {
            byte[] aKey = pKey;
            UnpackedRecord p;     /* The unpacked record that we will return */
            int nByte;            /* Memory space needed to hold p, in bytes */
            int d;
            u32 idx;
            int u;                /* Unsigned loop counter */
            int szHdr = 0;
            Mem pMem;
            int nOff;           /* Increase pSpace by this much to 8-byte align it */

            /*
            ** We want to shift the pointer pSpace up such that it is 8-byte aligned.
            ** Thus, we need to calculate a value, nOff, between 0 and 7, to shift
            ** it by.  If pSpace is already 8-byte aligned, nOff should be zero.
            */
            //nOff = ( 8 - ( SQLITE_PTR_TO_INT( pSpace ) & 7 ) ) & 7;
            //pSpace += nOff;
            //szSpace -= nOff;
            //nByte = ROUND8( sizeof( UnpackedRecord ) ) + sizeof( Mem ) * ( pKeyInfo->nField + 1 );
            //if ( nByte > szSpace)
            //{
            //var  p = new UnpackedRecord();//sqlite3DbMallocRaw(pKeyInfo.db, nByte);
            //  if ( p == null ) return null;
            //  p.flags = UNPACKED_NEED_FREE | UNPACKED_NEED_DESTROY;
            //}
            //else
            {
                p = pSpace;//(UnpackedRecord)pSpace;
                p.flags = UNPACKED_NEED_DESTROY;
            }
            p.pKeyInfo = pKeyInfo;
            p.nField = (u16)(pKeyInfo.nField + 1);
            //p->aMem = pMem = (Mem)&( (char)p )[ROUND8( sizeof( UnpackedRecord ) )];
            //Debug.Assert( EIGHT_BYTE_ALIGNMENT( pMem ) );
            p.aMem = new Mem[p.nField + 1];
            idx = (u32)getVarint32(aKey, 0, out szHdr);// GetVarint( aKey, szHdr );
            d = (int)szHdr;
            u = 0;
            while (idx < (int)szHdr && u < p.nField && d <= nKey)
            {
                p.aMem[u] = sqlite3Malloc(p.aMem[u]);
                pMem = p.aMem[u];
                u32 serial_type = 0;

                idx += (u32)getVarint32(aKey, idx, out serial_type);// GetVarint( aKey + idx, serial_type );
                pMem.enc = pKeyInfo.enc;
                pMem.db = pKeyInfo.db;
                /* pMem->flags = 0; // sqlite3VdbeSerialGet() will set this for us */
                //pMem.zMalloc = null;
                d += (int)sqlite3VdbeSerialGet(aKey, d, serial_type, pMem);
                //pMem++;
                u++;
            }
            Debug.Assert(u <= pKeyInfo.nField + 1);
            p.nField = (u16)u;
            return p;// (void)p;
        }

        /*
        ** This routine destroys a UnpackedRecord object.
        */
        static void sqlite3VdbeDeleteUnpackedRecord(UnpackedRecord p)
        {
#if SQLITE_DEBUG
      int i;
      Mem pMem;
      Debug.Assert( p != null );
      Debug.Assert( ( p.flags & UNPACKED_NEED_DESTROY ) != 0 );
      //for ( i = 0, pMem = p->aMem ; i < p->nField ; i++, pMem++ )
      //{
      //  /* The unpacked record is always constructed by the
      //  ** sqlite3VdbeUnpackRecord() function above, which makes all
      //  ** strings and blobs static.  And none of the elements are
      //  ** ever transformed, so there is never anything to delete.
      //  */
      //  if ( NEVER( pMem->zMalloc ) ) sqlite3VdbeMemRelease( pMem );
      //}
#endif
            if ((p.flags & UNPACKED_NEED_FREE) != 0)
            {
                sqlite3DbFree(p.pKeyInfo.db, ref p.aMem);
                p = null;
            }
        }

        /*
        ** This function compares the two table rows or index records
        ** specified by {nKey1, pKey1} and pPKey2.  It returns a negative, zero
        ** or positive integer if key1 is less than, equal to or
        ** greater than key2.  The {nKey1, pKey1} key must be a blob
        ** created by th OP_MakeRecord opcode of the VDBE.  The pPKey2
        ** key must be a parsed key such as obtained from
        ** sqlite3VdbeParseRecord.
        **
        ** Key1 and Key2 do not have to contain the same number of fields.
        ** The key with fewer fields is usually compares less than the
        ** longer key.  However if the UNPACKED_INCRKEY flags in pPKey2 is set
        ** and the common prefixes are equal, then key1 is less than key2.
        ** Or if the UNPACKED_MATCH_PREFIX flag is set and the prefixes are
        ** equal, then the keys are considered to be equal and
        ** the parts beyond the common prefix are ignored.
        **
        ** If the UNPACKED_IGNORE_ROWID flag is set, then the last byte of
        ** the header of pKey1 is ignored.  It is assumed that pKey1 is
        ** an index key, and thus ends with a rowid value.  The last byte
        ** of the header will therefore be the serial type of the rowid:
        ** one of 1, 2, 3, 4, 5, 6, 8, or 9 - the integer serial types.
        ** The serial type of the final rowid will always be a single byte.
        ** By ignoring this last byte of the header, we force the comparison
        ** to ignore the rowid at the end of key1.
        */

        static Mem mem1 = new Mem();
        // ALTERNATE FORM for C#
        static int sqlite3VdbeRecordCompare(
        int nKey1, byte[] pKey1,    /* Left key */
        UnpackedRecord pPKey2       /* Right key */
        )
        {
            return sqlite3VdbeRecordCompare(nKey1, pKey1, 0, pPKey2);
        }

        static int sqlite3VdbeRecordCompare(
        int nKey1, byte[] pKey1,    /* Left key */
        int offset,
        UnpackedRecord pPKey2       /* Right key */
        )
        {
            int d1;            /* Offset into aKey[] of next data element */
            u32 idx1;          /* Offset into aKey[] of next header element */
            u32 szHdr1;        /* Number of bytes in header */
            int i = 0;
            int nField;
            int rc = 0;

            byte[] aKey1 = new byte[pKey1.Length - offset];
            //Buffer.BlockCopy( pKey1, offset, aKey1, 0, aKey1.Length );
            KeyInfo pKeyInfo;

            pKeyInfo = pPKey2.pKeyInfo;
            mem1.enc = pKeyInfo.enc;
            mem1.db = pKeyInfo.db;
            /* mem1.flags = 0;  // Will be initialized by sqlite3VdbeSerialGet() */
            //  VVA_ONLY( mem1.zMalloc = 0; ) /* Only needed by Debug.Assert() statements */

            /* Compilers may complain that mem1.u.i is potentially uninitialized.
            ** We could initialize it, as shown here, to silence those complaints.
            ** But in fact, mem1.u.i will never actually be used uninitialized, and doing 
            ** the unnecessary initialization has a measurable negative performance
            ** impact, since this routine is a very high runner.  And so, we choose
            ** to ignore the compiler warnings and leave this variable uninitialized.
            */
            /*  mem1.u.i = 0;  // not needed, here to silence compiler warning */

            idx1 = (u32)((szHdr1 = pKey1[offset]) <= 0x7f ? 1 : getVarint32(pKey1, offset, out szHdr1));// GetVarint( aKey1, szHdr1 );
            d1 = (int)szHdr1;
            if ((pPKey2.flags & UNPACKED_IGNORE_ROWID) != 0)
            {
                szHdr1--;
            }
            nField = pKeyInfo.nField;
            while (idx1 < szHdr1 && i < pPKey2.nField)
            {
                u32 serial_type1;

                /* Read the serial types for the next element in each key. */
                idx1 += (u32)((serial_type1 = pKey1[offset + idx1]) <= 0x7f ? 1 : getVarint32(pKey1, (uint)(offset + idx1), out serial_type1)); //GetVarint( aKey1 + idx1, serial_type1 );
                if (d1 <= 0 || d1 >= nKey1 && sqlite3VdbeSerialTypeLen(serial_type1) > 0)
                    break;

                /* Extract the values to be compared.
                */
                d1 += (int)sqlite3VdbeSerialGet(pKey1, offset + d1, serial_type1, mem1);//sqlite3VdbeSerialGet( aKey1, d1, serial_type1, mem1 );

                /* Do the comparison
                */
                rc = sqlite3MemCompare(mem1, pPKey2.aMem[i], i < nField ? pKeyInfo.aColl[i] : null);
                if (rc != 0)
                {
                    //Debug.Assert( mem1.zMalloc==null );  /* See comment below */

                    /* Invert the result if we are using DESC sort order. */
                    if (pKeyInfo.aSortOrder != null && i < nField && pKeyInfo.aSortOrder[i] != 0)
                    {
                        rc = -rc;
                    }

                    /* If the PREFIX_SEARCH flag is set and all fields except the final
                    ** rowid field were equal, then clear the PREFIX_SEARCH flag and set
                    ** pPKey2->rowid to the value of the rowid field in (pKey1, nKey1).
                    ** This is used by the OP_IsUnique opcode.
                    */
                    if ((pPKey2.flags & UNPACKED_PREFIX_SEARCH) != 0 && i == (pPKey2.nField - 1))
                    {
                        Debug.Assert(idx1 == szHdr1 && rc != 0);
                        Debug.Assert((mem1.flags & MEM_Int) != 0);
                        pPKey2.flags = (ushort)(pPKey2.flags & ~UNPACKED_PREFIX_SEARCH);
                        pPKey2.rowid = mem1.u.i;
                    }

                    return rc;
                }
                i++;
            }

            /* No memory allocation is ever used on mem1.  Prove this using
            ** the following Debug.Assert().  If the Debug.Assert() fails, it indicates a
            ** memory leak and a need to call sqlite3VdbeMemRelease(&mem1).
            */
            //Debug.Assert( mem1.zMalloc==null );

            /* rc==0 here means that one of the keys ran out of fields and
            ** all the fields up to that point were equal. If the UNPACKED_INCRKEY
            ** flag is set, then break the tie by treating key2 as larger.
            ** If the UPACKED_PREFIX_MATCH flag is set, then keys with common prefixes
            ** are considered to be equal.  Otherwise, the longer key is the
            ** larger.  As it happens, the pPKey2 will always be the longer
            ** if there is a difference.
            */
            Debug.Assert(rc == 0);
            if ((pPKey2.flags & UNPACKED_INCRKEY) != 0)
            {
                rc = -1;
            }
            else if ((pPKey2.flags & UNPACKED_PREFIX_MATCH) != 0)
            {
                /* Leave rc==0 */
            }
            else if (idx1 < szHdr1)
            {
                rc = 1;
            }
            return rc;
        }

        /*
        ** pCur points at an index entry created using the OP_MakeRecord opcode.
        ** Read the rowid (the last field in the record) and store it in *rowid.
        ** Return SQLITE_OK if everything works, or an error code otherwise.
        **
        ** pCur might be pointing to text obtained from a corrupt database file.
        ** So the content cannot be trusted.  Do appropriate checks on the content.
        */
        static int sqlite3VdbeIdxRowid(sqlite3 db, BtCursor pCur, ref i64 rowid)
        {
            i64 nCellKey = 0;
            int rc;
            u32 szHdr = 0;        /* Size of the header */
            u32 typeRowid = 0;    /* Serial type of the rowid */
            u32 lenRowid;       /* Size of the rowid */
            Mem m = null;
            Mem v = null;
            v = sqlite3Malloc(v);
            UNUSED_PARAMETER(db);

            /* Get the size of the index entry.  Only indices entries of less
            ** than 2GiB are support - anything large must be database corruption.
            ** Any corruption is detected in sqlite3BtreeParseCellPtr(), though, so
            ** this code can safely assume that nCellKey is 32-bits  
            */
            Debug.Assert(sqlite3BtreeCursorIsValid(pCur));
            rc = sqlite3BtreeKeySize(pCur, ref nCellKey);
            Debug.Assert(rc == SQLITE_OK);     /* pCur is always valid so KeySize cannot fail */
            Debug.Assert(((u32)nCellKey & SQLITE_MAX_U32) == (u64)nCellKey);

            /* Read in the complete content of the index entry */
            m = sqlite3Malloc(m);
            // memset(&m, 0, sizeof(m));
            rc = sqlite3VdbeMemFromBtree(pCur, 0, (int)nCellKey, true, m);
            if (rc != 0)
            {
                return rc;
            }

            /* The index entry must begin with a header size */
            getVarint32(m.zBLOB, 0, out szHdr);
            testcase(szHdr == 3);
            testcase(szHdr == m.n);
            if (unlikely(szHdr < 3 || (int)szHdr > m.n))
            {
                goto idx_rowid_corruption;
            }

            /* The last field of the index should be an integer - the ROWID.
            ** Verify that the last entry really is an integer. */
            getVarint32(m.zBLOB, szHdr - 1, out typeRowid);
            testcase(typeRowid == 1);
            testcase(typeRowid == 2);
            testcase(typeRowid == 3);
            testcase(typeRowid == 4);
            testcase(typeRowid == 5);
            testcase(typeRowid == 6);
            testcase(typeRowid == 8);
            testcase(typeRowid == 9);
            if (unlikely(typeRowid < 1 || typeRowid > 9 || typeRowid == 7))
            {
                goto idx_rowid_corruption;
            }
            lenRowid = (u32)sqlite3VdbeSerialTypeLen(typeRowid);
            testcase((u32)m.n == szHdr + lenRowid);
            if (unlikely((u32)m.n < szHdr + lenRowid))
            {
                goto idx_rowid_corruption;
            }

            /* Fetch the integer off the end of the index record */
            sqlite3VdbeSerialGet(m.zBLOB, (int)(m.n - lenRowid), typeRowid, v);
            rowid = v.u.i;
            sqlite3VdbeMemRelease(m);
            return SQLITE_OK;

        /* Jump here if database corruption is detected after m has been
        ** allocated.  Free the m object and return SQLITE_CORRUPT. */
        idx_rowid_corruption:
            //testcase( m.zMalloc != 0 );
            sqlite3VdbeMemRelease(m);
            return SQLITE_CORRUPT_BKPT();
        }

        /*
        ** Compare the key of the index entry that cursor pC is pointing to against
        ** the key string in pUnpacked.  Write into *pRes a number
        ** that is negative, zero, or positive if pC is less than, equal to,
        ** or greater than pUnpacked.  Return SQLITE_OK on success.
        **
        ** pUnpacked is either created without a rowid or is truncated so that it
        ** omits the rowid at the end.  The rowid at the end of the index entry
        ** is ignored as well.  Hence, this routine only compares the prefixes 
        ** of the keys prior to the final rowid, not the entire key.
        */
        static int sqlite3VdbeIdxKeyCompare(
        VdbeCursor pC,              /* The cursor to compare against */
        UnpackedRecord pUnpacked,   /* Unpacked version of key to compare against */
        ref int res                 /* Write the comparison result here */
        )
        {
            i64 nCellKey = 0;
            int rc;
            BtCursor pCur = pC.pCursor;
            Mem m = null;

            Debug.Assert(sqlite3BtreeCursorIsValid(pCur));
            rc = sqlite3BtreeKeySize(pCur, ref nCellKey);
            Debug.Assert(rc == SQLITE_OK);    /* pCur is always valid so KeySize cannot fail */
            /* nCellKey will always be between 0 and 0xffffffff because of the say
            ** that btreeParseCellPtr() and sqlite3GetVarint32() are implemented */
            if (nCellKey <= 0 || nCellKey > 0x7fffffff)
            {
                res = 0;
                return SQLITE_CORRUPT_BKPT();
            }

            m = sqlite3Malloc(m);
            // memset(&m, 0, sizeof(m));
            rc = sqlite3VdbeMemFromBtree(pC.pCursor, 0, (int)nCellKey, true, m);
            if (rc != 0)
            {
                return rc;
            }
            Debug.Assert((pUnpacked.flags & UNPACKED_IGNORE_ROWID) != 0);
            res = sqlite3VdbeRecordCompare(m.n, m.zBLOB, pUnpacked);
            sqlite3VdbeMemRelease(m);
            return SQLITE_OK;
        }

        /*
        ** This routine sets the value to be returned by subsequent calls to
        ** sqlite3_changes() on the database handle 'db'.
        */
        static void sqlite3VdbeSetChanges(sqlite3 db, int nChange)
        {
            Debug.Assert(sqlite3_mutex_held(db.mutex));
            db.nChange = nChange;
            db.nTotalChange += nChange;
        }

        /*
        ** Set a flag in the vdbe to update the change counter when it is finalised
        ** or reset.
        */
        static void sqlite3VdbeCountChanges(Vdbe v)
        {
            v.changeCntOn = true;
        }

        /*
        ** Mark every prepared statement associated with a database connection
        ** as expired.
        **
        ** An expired statement means that recompilation of the statement is
        ** recommend.  Statements expire when things happen that make their
        ** programs obsolete.  Removing user-defined functions or collating
        ** sequences, or changing an authorization function are the types of
        ** things that make prepared statements obsolete.
        */
        static void sqlite3ExpirePreparedStatements(sqlite3 db)
        {
            Vdbe p;
            for (p = db.pVdbe; p != null; p = p.pNext)
            {
                p.expired = true;
            }
        }

        /*
        ** Return the database associated with the Vdbe.
        */
        static sqlite3 sqlite3VdbeDb(Vdbe v)
        {
            return v.db;
        }
        /*
        ** Return a pointer to an sqlite3_value structure containing the value bound
        ** parameter iVar of VM v. Except, if the value is an SQL NULL, return 
        ** 0 instead. Unless it is NULL, apply affinity aff (one of the SQLITE_AFF_*
        ** constants) to the value before returning it.
        **
        ** The returned value must be freed by the caller using sqlite3ValueFree().
        */
        static sqlite3_value sqlite3VdbeGetValue(Vdbe v, int iVar, u8 aff)
        {
            Debug.Assert(iVar > 0);
            if (v != null)
            {
                Mem pMem = v.aVar[iVar - 1];
                if (0 == (pMem.flags & MEM_Null))
                {
                    sqlite3_value pRet = sqlite3ValueNew(v.db);
                    if (pRet != null)
                    {
                        sqlite3VdbeMemCopy((Mem)pRet, pMem);
                        sqlite3ValueApplyAffinity(pRet, (char)aff, SQLITE_UTF8);
                        sqlite3VdbeMemStoreType((Mem)pRet);
                    }
                    return pRet;
                }
            }
            return null;
        }

        /*
        ** Configure SQL variable iVar so that binding a new value to it signals
        ** to sqlite3_reoptimize() that re-preparing the statement may result
        ** in a better query plan.
        */
        static void sqlite3VdbeSetVarmask(Vdbe v, int iVar)
        {
            Debug.Assert(iVar > 0);
            if (iVar > 32)
            {
                v.expmask = 0xffffffff;
            }
            else
            {
                v.expmask |= ((u32)1 << (iVar - 1));
            }
        }
    }
}
