#region OMIT_TRIGGER
#if !OMIT_TRIGGER
using System;
using System.Diagnostics;
using System.Text;

namespace Core
{
    public partial class Trigger
    {
        public static void DeleteTriggerStep(Context ctx, ref TriggerStep triggerStep)
        {
            while (triggerStep != null)
            {
                TriggerStep tmp = triggerStep;
                triggerStep = triggerStep.Next;

                Expr.Delete(ctx, ref tmp.Where);
                Expr.ListDelete(ctx, ref tmp.ExprList);
                Select.Delete(ctx, ref tmp.Select);
                IdList.Delete(ctx, ref tmp.IdList);

                triggerStep = null;
                SysEx.TagFree(ctx, ref tmp);
            }
        }

        public static Trigger TriggerList(Parse parse, Table table)
        {
            Schema tmpSchema = parse.Ctx.DBs[1].Schema;
            Trigger list = null; // List of triggers to return
            if (parse.DisableTriggers)
                return null;
            if (tmpSchema != table.Schema)
            {
                Debug.Assert(Btree.SchemaMutexHeld(parse.Ctx, 0, tmpSchema));
                for (HashElem p = tmpSchema.TriggerHash.First; p != null; p = p.Next)
                {
                    Trigger trig = (Trigger)p.Data;
                    if (trig.TabSchema == table.Schema && trig.Table.Equals(table.Name, StringComparison.InvariantCultureIgnoreCase))
                    {
                        trig.Next = (list != null ? list : table.Trigger);
                        list = trig;
                    }
                }
            }

            return (list != null ? list : table.Trigger);
        }

        public static void BeginTrigger(Parse parse, Token name1, Token name2, TK trTm, TK op, IdList columns, SrcList tableName, Expr when, bool isTemp, int noErr)
        {
            Trigger trigger = null; // The new trigger
            Table table; // Table that the trigger fires off of
            Context ctx = parse.Ctx; // The database connection
            Token name = null; // The unqualified db name
            DbFixer fix = new DbFixer(); // State vector for the DB fixer
            int tabDB; // Index of the database holding pTab

            Debug.Assert(name1 != null);   // pName1.z might be NULL, but not pName1 itself
            Debug.Assert(name2 != null);
            Debug.Assert(op == TK.INSERT || op == TK.UPDATE || op == TK.DELETE);
            Debug.Assert(op > 0 && op < (TK)0xff);

            int db; // The database to store the trigger in
            if (isTemp)
            {
                // If TEMP was specified, then the trigger name may not be qualified.
                if (name2.length > 0)
                {
                    parse.ErrorMsg("temporary trigger may not have qualified name");
                    goto trigger_cleanup;
                }
                db = 1;
                name = name1;
            }
            else
            {
                // Figure out the db that the the trigger will be created in
                db = parse.TwoPartName(name1, name2, ref name);
                if (db < 0)
                    goto trigger_cleanup;
            }
            if (tableName == null || ctx.MallocFailed)
                goto trigger_cleanup;

            // A long-standing parser bug is that this syntax was allowed:
            //    CREATE TRIGGER attached.demo AFTER INSERT ON attached.tab ....
            //                                                 ^^^^^^^^
            // To maintain backwards compatibility, ignore the database name on pTableName if we are reparsing our of SQLITE_MASTER.
            if (ctx.Init.Busy && db != 1)
            {
                SysEx.TagFree(ctx, ref tableName.Ids[0].Database);
                tableName.Ids[0].Database = null;
            }

            // If the trigger name was unqualified, and the table is a temp table, then set iDb to 1 to create the trigger in the temporary database.
            // If sqlite3SrcListLookup() returns 0, indicating the table does not exist, the error is caught by the block below.
            //? if (tableName == null) goto trigger_cleanup;
            table = sqlite3SrcListLookup(parse, tableName);
            if (ctx.Init.Busy == null && name2.length == 0 && table != null && table.Schema == ctx.DBs[1].Schema)
                db = 1;

            // Ensure the table name matches database name and that the table exists
            if (ctx.MallocFailed) goto trigger_cleanup;
            Debug.Assert(tableName.Srcs == 1);
            if (sqlite3FixInit(fix, parse, db, "trigger", name) != 0 && sqlite3FixSrcList(fix, tableName) != 0)
                goto trigger_cleanup;
            table = sqlite3SrcListLookup(parse, tableName);
            if (table == null)
            {
                // The table does not exist.
                if (ctx.Init.DB == 1)
                {
                    // Ticket #3810.
                    // Normally, whenever a table is dropped, all associated triggers are dropped too.  But if a TEMP trigger is created on a non-TEMP table
                    // and the table is dropped by a different database connection, the trigger is not visible to the database connection that does the
                    // drop so the trigger cannot be dropped.  This results in an "orphaned trigger" - a trigger whose associated table is missing.
                    ctx.Init.OrphanTrigger = 1;
                }
                goto trigger_cleanup;
            }
            if (IsVirtual(table))
            {
                parse.ErrorMsg("cannot create triggers on virtual tables");
                goto trigger_cleanup;
            }

            // Check that the trigger name is not reserved and that no trigger of the specified name exists
            string nameAsString = sqlite3NameFromToken(ctx, name);
            if (nameAsString == null || RC.OK != sqlite3CheckObjectName(parse, nameAsString))
                goto trigger_cleanup;
            Debug.Assert(Btree.SchemaMutexHeld(ctx, db, null));
            if (ctx.DBs[db].Schema.TriggerHash.Find(nameAsString, nameAsString.Length, (Trigger)null) != null)
            {
                if (noErr == 0)
                    parse.ErrorMsg(ctx, "trigger %T already exists", name);
                else
                {
                    Debug.Assert(!ctx.Init.Busy);
                    sqlite3CodeVerifySchema(parse, db);
                }
                goto trigger_cleanup;
            }

            // Do not create a trigger on a system table
            if (table.Name.StartsWith("sqlite_", StringComparison.InvariantCultureIgnoreCase))
            {
                parse.ErrorMsg("cannot create trigger on system table");
                parse.Errs++;
                goto trigger_cleanup;
            }

            // INSTEAD of triggers are only for views and views only support INSTEAD of triggers.
            if (table.Select != null && trTm != TK.INSTEAD)
            {
                parse.ErrorMsg("cannot create %s trigger on view: %S", (trTm == TK.BEFORE ? "BEFORE" : "AFTER"), tableName, 0);
                goto trigger_cleanup;
            }
            if (table.Select == null && trTm == TK.INSTEAD)
            {
                parse.ErrorMsg("cannot create INSTEAD OF trigger on table: %S", tableName, 0);
                goto trigger_cleanup;
            }
            tabDB = sqlite3SchemaToIndex(db, table.Schema);

#if !OMIT_AUTHORIZATION
            {
                int code = AUTH.CREATE_TRIGGER;
                string dbName = ctx.DBs[tabDb].Name;
                string dbTrigName = (isTemp ? ctx.DBs[1].Name : dbName);
                if (tabDB == 1 || isTemp) code = SQLITE_CREATE_TEMP_TRIGGER;
                if (Auth.Check(parse, code, nameAsString, table.Name, dbTrigName) || Auth.Check(parse, AUTH.INSERT, SCHEMA_TABLE(tabDb), 0, dbName))
                    goto trigger_cleanup;
            }
#endif

            // INSTEAD OF triggers can only appear on views and BEFORE triggers cannot appear on views.  So we might as well translate every
            // INSTEAD OF trigger into a BEFORE trigger.  It simplifies code elsewhere.
            if (trTm == TK.INSTEAD)
                trTm = TK.BEFORE;

            // Build the Trigger object
            trigger = new Trigger(); //: (Trigger *)SysEx::TagAlloc(db, sizeof(Trigger), true);
            if (trigger == null) goto trigger_cleanup;
            trigger.Name = name;
            trigger.Table = tableName.Ids[0].Name; //: SysEx::TagStrDup(ctx, tableName->Ids[0].Name);
            trigger.Schema = ctx.DBs[db].Schema;
            trigger.TabSchema = table.Schema;
            trigger.OP = (byte)op;
            trigger.TR_TM = (trTm == TK.BEFORE ? TRIGGER.BEFORE : TRIGGER.AFTER);
            trigger.When = Expr.Dup(db, when, EXPRDUP.REDUCE);
            trigger.Columns = IdListDup(ctx, columns);
            Debug.Assert(parse.NewTrigger == null);
            parse.NewTrigger = trigger;

        trigger_cleanup:
            SysEx.TagFree(ctx, ref name);
            SrcListDelete(ctx, ref tableName);
            IdListDelete(ctx, ref columns);
            ExprDelete(ctx, ref when);
            if (parse.NewTrigger == null)
                DeleteTrigger(ctx, ref trigger);
            else
                Debug.Assert(parse.NewTrigger == trigger);
        }

        static void sqlite3FinishTrigger(Parse parse, TriggerStep stepList, Token all)
        {
            Trigger trig = parse.NewTrigger; // Trigger being finished
            Context ctx = parse.Ctx; // The database
            DbFixer sFix = new DbFixer(); // Fixer object
            Token nameToken = new Token(); // Trigger name for error reporting

            parse.NewTrigger = null;
            if (SysEx.NEVER(parse.Errs != 0) || trig == null)
                goto triggerfinish_cleanup;
            string name = trig.Name; // Name of trigger
            int db = sqlite3SchemaToIndex(parse.Ctx, trig.Schema); // Database containing the trigger
            trig.StepList = stepList;
            while (stepList != null)
            {
                stepList.Trig = trig;
                stepList = stepList.pNext;
            }
            nameToken.data = trig.Name;
            nameToken.length = nameToken.data.Length;
            if (sqlite3FixInit(sFix, parse, db, "trigger", nameToken) != 0 && sqlite3FixTriggerStep(sFix, trig.step_list) != 0)
                goto triggerfinish_cleanup;

            // if we are not initializing, build the sqlite_master entry
            if (ctx.Init.Busy)
            {
                // Make an entry in the sqlite_master table
                Vdbe v = parse.GetVdbe();
                if (v == null) goto triggerfinish_cleanup;
                parse.BeginWriteOperation(0, db);
                string z = all.data.Substring(0, all.length); //: SysEx::TagStrNDup(ctx, (char *)all->data, all->length);
                parse.NestedParse("INSERT INTO %Q.%s VALUES('trigger',%Q,%Q,0,'CREATE TRIGGER %q')", ctx.DBs[db].Name, SCHEMA_TABLE(db), name, trig.Table, z);
                SysEx.TagFree(ctx, ref z);
                parse.ChangeCookie(db);
                v.AddParseSchemaOp(db, SysEx.Mprintf(ctx, "type='trigger' AND name='%q'", name));
            }

            if (!ctx.Init.Busy)
            {
                Trigger link = trig;
                Debug.Assert(Btree.SchemaMutexHeld(ctx, db, null));
                trig = ctx.DBs[db].Schema.TriggerHash.Insert(name, name.Length, trig);
                if (trig != null)
                    ctx.MallocFailed = true;
                else if (link.Schema == link.TabSchema)
                {
                    int tableLength = link.Table.Length;
                    Table table = (Table)link.TabSchema.TableHash.Find(link.Table, tableLength, (Table)null);
                    Debug.Assert(table != null);
                    link.Next = table.Trigger;
                    table.Trigger = link;
                }
            }

        triggerfinish_cleanup:
            DeleteTrigger(ctx, ref trig);
            Debug.Assert(parse.NewTrigger == null);
            DeleteTriggerStep(ctx, ref stepList);
        }

        public static TriggerStep TriggerSelectStep(Context ctx, Select select)
        {
            TriggerStep triggerStep = new TriggerStep(); //: SysEx::TagAlloc(ctx, sizeof(TriggerStep), true);
            if (triggerStep == null)
            {
                SelectDelete(ctx, ref select);
                return null;
            }
            triggerStep.OP = TK.SELECT;
            triggerStep.Select spSelect;
            triggerStep.orconf = OE.Default;
            return triggerStep;
        }

        static TriggerStep TriggerStepAllocate(Context ctx, byte op, Token name)
        {
            TriggerStep triggerStep = new TriggerStep(); //: SysEx::TagAlloc(ctx, sizeof(TriggerStep) + name->length, true);
            if (triggerStep != null)
            {
                string z = name.data;
                triggerStep.target.data = z;
                triggerStep.target.length = name.length;
                triggerStep.OP = op;
            }
            return triggerStep;
        }

        // OVERLOADS, so I don't need to rewrite parse.c
        public static TriggerStep TriggerInsertStep(Context ctx, Token tableName, IdList column, int null_4, int null_5, OE orconf) { return TriggerInsertStep(ctx, tableName, column, null, null, orconf); }
        public static TriggerStep TriggerInsertStep(Context ctx, Token tableName, IdList column, ExprList list, int null_5, OE orconf) { return TriggerInsertStep(ctx, tableName, column, list, null, orconf); }
        public static TriggerStep TriggerInsertStep(Context ctx, Token tableName, IdList column, int null_4, Select select, OE orconf) { return TriggerInsertStep(ctx, tableName, column, null, select, orconf); }
        public static TriggerStep TriggerInsertStep(Context ctx, Token tableName, IdList column, ExprList list, Select select, OE orconf)
        {
            Debug.Assert(list == null || select == null);
            Debug.Assert(list != null || select != null || ctx.MallocFailed);
            TriggerStep triggerStep = TriggerStepAllocate(ctx, TK.INSERT, tableName);
            if (triggerStep != null)
            {
                triggerStep.Select = SelectDup(ctx, select, EXPRDUP_REDUCE);
                triggerStep.IdList = column;
                triggerStep.ExprList = ExprListDup(ctx, list, EXPRDUP_REDUCE);
                triggerStep.Orconf = orconf;
            }
            else
                IdListDelete(ctx, ref column);
            ExprListDelete(ctx, ref list);
            SelectDelete(ctx, ref select);
            return triggerStep;
        }

        public static TriggerStep TriggerUpdateStep(Context ctx, Token tableName, ExprList list, Expr where, OE orconf)
        {
            TriggerStep triggerStep = triggerStepAllocate(ctx, TK_UPDATE, tableName);
            if (triggerStep != null)
            {
                triggerStep.ExprList = ExprListDup(ctx, list, EXPRDUP_REDUCE);
                triggerStep.Where = ExprDup(ctx, where, EXPRDUP_REDUCE);
                triggerStep.Orconf = orconf;
            }
            ExprListDelete(ctx, ref list);
            ExprDelete(ctx, ref where);
            return triggerStep;
        }

        public static TriggerStep TriggerDeleteStep(Context ctx, Token tableName, Expr where_)
        {
            TriggerStep triggerStep = TriggerStepAllocate(ctx, TK_DELETE, tableName);
            if (triggerStep != null)
            {
                triggerStep.Where = sqlite3ExprDup(ctx, where_, EXPRDUP_REDUCE);
                triggerStep.Orconf = OE_Default;
            }
            ExprDelete(ctx, ref where_);
            return triggerStep;
        }

        public static void DeleteTrigger(Context ctx, ref Trigger trigger)
        {
            if (trigger == null) return;
            DeleteTriggerStep(ctx, ref trigger.StepList);
            SysEx.TagFree(ctx, ref trigger.Name);
            SysEx.TagFree(ctx, ref trigger.Table);
            ExprDelete(ctx, ref trigger.When);
            IdListDelete(ctx, ref trigger.Columns);
            SysEx.TagFree(ctx, ref trigger);
            trigger = null;
        }

        public static void DropTrigger(Parse parse, SrcList name, int noErr)
        {
            Context ctx = parse.db;
            if (ctx.MallocFailed || parse.ReadSchema(parse) != RC.OK)
                goto drop_trigger_cleanup;

            Debug.Assert(name.nSrc == 1);
            string dbName = name.Ids[0].Database;
            string nameAsString = name.Ids[0].Name;
            int nameLength = nameAsString.Length;
            Debug.Assert(dbName != null || Btree.BtreeHoldsAllMutexes(ctx));
            Trigger trigger = null;
            for (int i = OMIT_TEMPDB; i < ctx.nDb; i++)
            {
                int j = (i < 2 ? i ^ 1 : i); // Search TEMP before MAIN
                if (dbName != null && !string.Equals(ctx.DBs[j].Name, dbName, StringComparison.InvariantCultureIgnoreCase)) continue;
                Debug.Assert(Btree.SchemaMutexHeld(ctx, j, null));
                trigger = ctx.DBs[j].Schema.TriggerHash.Find(nameAsString, nameLength, (Trigger)null);
                if (trigger != null) break;
            }
            if (trigger == null)
            {
                if (noErr == 0)
                    parse.ErrorMsg("no such trigger: %S", name, 0);
                else
                    parse.CodeVerifyNamedSchema(dbName);
                parse.CheckSchema = true;
                goto drop_trigger_cleanup;
            }
            DropTriggerPtr(parse, trigger);

        drop_trigger_cleanup:
            SrcListDelete(ctx, ref name);
        }

        static Table TableOfTrigger(Trigger trigger)
        {
            int tableLength = trigger.Table.Length;
            return trigger.TabSchema.TableHash.Find(trigger.table, tableLength, (Table)null);
        }

        VdbeOpList[] _dropTrigger = new VdbeOpList[]  {
            new VdbeOpList( OP_Rewind,     0, ADDR(9),  0),
            new VdbeOpList( OP_String8,    0, 1,        0), /* 1 */
            new VdbeOpList( OP_Column,     0, 1,        2),
            new VdbeOpList( OP_Ne,         2, ADDR(8),  1),
            new VdbeOpList( OP_String8,    0, 1,        0), /* 4: "trigger" */
            new VdbeOpList( OP_Column,     0, 0,        2),
            new VdbeOpList( OP_Ne,         2, ADDR(8),  1),
            new VdbeOpList( OP_Delete,     0, 0,        0),
            new VdbeOpList( OP_Next,       0, ADDR(1),  0), /* 8 */
        };
        public static void DropTriggerPtr(Parse parse, Trigger trigger)
        {
            Context ctx = parse.Ctx;
            int db = ctx.SchemaToIndex(trigger.Schema);
            Debug.Assert(db >= 0 && db < ctx.DBs.length);
            Table table = tableOfTrigger(trigger);
            Debug.Assert(table != null);
            Debug.Assert(table.Schema == trigger.Schema || db == 1);
#if !OMIT_AUTHORIZATION
            {
                int code = AUTH.DROP_TRIGGER;
                string dbName = ctx.Dbs[db].Name;
                string tableName = SCHEMA_TABLE(db);
                if (db == 1) code = AUTH.DROP_TEMP_TRIGGER;
                if (Auth.Check(parse, code, trigger.Name, table.Name, dbName) || Auth.Check(parse, AUTH.DELETE, tableName, 0, dbName))
                    return;
            }
#endif

            // Generate code to destroy the database record of the trigger.
            Debug.Assert(table != null);
            Vdbe v = parse.GetVdbe();
            if (v != null)
            {
                parse.BeginWriteOperation(0, db);
                parse.OpenMasterTable(db);
                int base_ = v.AddOpList(v, _dropTrigger.Length, _dropTrigger);
                v->ChangeP4(base_ + 1, trigger.Name, Vdbe.P4T.TRANSIENT);
                v->ChangeP4(base_ + 4, "trigger", Vdbe.P4T.STATIC);
                parse->ChangeCookie(db);
                v.AddOp2(O._Close, 0, 0);
                v.AddOp4(OP.DropTrigger, db, 0, 0, trigger.Name, 0);
                if (parse.Mems < 3)
                    parse.Mems = 3;
            }
        }

        public static void UnlinkAndDeleteTrigger(Context ctx, int db, string name)
        {
            Debug.Assert(sqlite3SchemaMutexHeld(ctx, db, null));
            Trigger trigger = ctx.DBs[db].Schema.TriggerHash.Insert(name, name.Length, (Trigger)null);
            if (SysEx_ALWAYS(trigger != null))
            {
                if (trigger.Schema == trigger.TabSchema)
                {
                    Table table = TableOfTrigger(trigger);
                    //: Trigger** pp;
                    //: for (pp = &table->Trigger; *pp != trigger; pp = &((*pp)->Next)) ;
                    //: *pp = (*pp)->Next;
                    if (table.Trigger == trigger)
                        table.Trigger = trigger.Next;
                    else
                    {
                        Trigger cc = table.Trigger;
                        while (cc != null)
                        {
                            if (cc.Next == trigger)
                            {
                                cc.Next = cc.Next.Next;
                                break;
                            }
                            cc = cc.Next;
                        }
                        Debug.Assert(cc != null);
                    }
                }
                DeleteTrigger(ctx, ref trigger);
                ctx.Flags |= Context.FLAG.InternChanges;
            }
        }

        static bool CheckColumnOverlap(IdList idList, ExprList eList)
        {
            int e;
            if (idList == null || SysEx.NEVER(eList == null))
                return true;
            for (e = 0; e < eList.Exprs; e++)
            {
                if (IdListIndex(idList, eList.Ids[e].Name) >= 0)
                    return true;
            }
            return false;
        }

        public static Trigger TriggersExist(Parse parse, Table table, TK op, ExprList changes, out int maskOut)
        {
            int mask = 0;
            Trigger list = null;
            if ((parse.Ctx.Flags & Context.FLAG.EnableTrigger) != 0)
                list = TriggerList(parse, table);
            Debug.Assert(list == null || !IsVirtual(table));
            for (Trigger p = list; p != null; p = p.Next)
                if (p.OP == op && CheckColumnOverlap(p.Columns, changes))
                    mask |= p.tr_tm;
            maskOut = mask;
            return (mask != 0 ? list : null);
        }

        static SrcList targetSrcList(Parse parse, TriggerStep step)
        {
            Context ctx = parse.Ctx;
            SrcList src = sqlite3SrcListAppend(parse.db, 0, step.target, 0); // SrcList to be returned
            if (src != null)
            {
                Debug.Assert(src.Srcs > 0);
                Debug.Assert(src.Ids != null);
                int db = Btree.SchemaToIndex(ctx, step.Trig.Schema); // Index of the database to use
                if (db == 0 || db >= 2)
                {
                    Debug.Assert(db < ctx.DBs.length);
                    src.Ids[src.Srcs - 1].Database = ctx.DBs[db].Name; //: SysEx::TagStrDup(ctx, ctx->DBs[db].Name);
                }
            }
            return src;
        }

        static int CodeTriggerProgram(Parse parse, TriggerStep stepList, int orconf)
        {
            Vdbe v = parse.Vdbe;
            Context ctx = parse.Ctx;

            Debug.Assert(parse.TriggerTab != null && parse.Toplevel != null);
            Debug.Assert(stepList != null);
            Debug.Assert(v != null);
            for (TriggerStep step = stepList; step != null; step = step.pNext)
            {
                // Figure out the ON CONFLICT policy that will be used for this step of the trigger program. If the statement that caused this trigger
                // to fire had an explicit ON CONFLICT, then use it. Otherwise, use the ON CONFLICT policy that was specified as part of the trigger
                // step statement. Example:
                //
                //   CREATE TRIGGER AFTER INSERT ON t1 BEGIN;
                //     INSERT OR REPLACE INTO t2 VALUES(new.a, new.b);
                //   END;
                //
                //   INSERT INTO t1 ... ;            -- insert into t2 uses REPLACE policy
                //   INSERT OR IGNORE INTO t1 ... ;  -- insert into t2 uses IGNORE policy
                parse.Orconf = (orconf == OE.Default ? step.orconf : (OE)orconf);
                switch (step.OP)
                {
                    case TK.UPDATE:
                        Update(parse,
                          targetSrcList(parse, step),
                          ExprListDup(ctx, step.ExprList, 0),
                          ExprDup(ctx, step.Where, 0),
                          parse.eOrconf);
                        break;
                    case TK.INSERT:
                        Insert(parse,
                          targetSrcList(parse, step),
                          ExprListDup(ctx, step.ExprList, 0),
                          SelectDup(ctx, step.Select, 0),
                          IdListDup(ctx, step.IdList),
                          parse.eOrconf);
                        break;
                    case TK_DELETE:
                        DeleteFrom(parse,
                          targetSrcList(parse, step),
                          ExprDup(ctx, step.Where, 0));
                        break;
                    default:
                        Debug.Assert(step.OP == TK.SELECT);
                        SelectDest sDest = new SelectDest();
                        Select select = SelectDup(ctx, step.Select, 0);
                        SelectDestInit(sDest, SRT_Discard, 0);
                        Select(parse, select, ref sDest);
                        SelectDelete(ctx, ref select);
                        break;
                }
                if (step.OP != TK.SELECT)
                    v.AddOp0(OP.ResetCount);
            }

            return 0;
        }

#if DEBUG
        static string OnErrorText(int onError)
        {
            switch (onError)
            {
                case OE_Abort: return "abort";
                case OE_Rollback: return "rollback";
                case OE_Fail: return "fail";
                case OE_Replace: return "replace";
                case OE_Ignore: return "ignore";
                case OE_Default: return "default";
            }
            return "n/a";
        }
#endif

        static void TransferParseError(Parse to, Parse from)
        {
            Debug.Assert(string.IsNullOrEmpty(from.ErrMsg) || from.Errs != 0);
            Debug.Assert(string.IsNullOrEmpty(to.ErrMsg) || to.Errs != 0);
            if (to.Errs == 0)
            {
                to.ErrMsg = from.ErrMsg;
                to.Errs = from.Errs;
            }
            else
                SysEx.TagFree(from.Ctx, ref from.ErrMsg);
        }

        static TriggerPrg CodeRowTrigger(Parse parse, Trigger trigger, Table table, OE orconf)
        {
            Parse top = ParseToplevel(parse);
            Context ctx = parse.Ctx; // Database handle

            Debug.Assert(trigger.Name == null || table == TableOfTrigger(trigger));
            Debug.Assert(top.Vdbe != null);

            // Allocate the TriggerPrg and SubProgram objects. To ensure that they are freed if an error occurs, link them into the Parse.pTriggerPrg 
            // list of the top-level Parse object sooner rather than later.
            TriggerPrg prg = new TriggerPrg(); // Value to return //: SysEx::TagAlloc(ctx, sizeof(TriggerPrg), true);
            if (prg == null) return null;
            prg.Next = top.TriggerPrg;
            top.TriggerPrg = prg;
            SubProgram program; // Sub-vdbe for trigger program
            prg.Program = program = new SubProgram();// sqlite3DbMallocZero( db, sizeof( SubProgram ) );
            if (program == null) return null;
            top.Vdbe.LinkSubProgram(program);
            prg.Trigger = trigger;
            prg.Orconf = orconf;
            prg.Colmasks[0] = 0xffffffff;
            prg.Colmasks[1] = 0xffffffff;


            // Allocate and populate a new Parse context to use for coding the trigger sub-program.
            Parse subParse = new Parse(); // Parse context for sub-vdbe //: SysEx::ScratchAlloc(ctx, sizeof(Parse), true);
            if (subParse == null) return null;
            NameContext sNC = new NameContext(); // Name context for sub-vdbe
            sNC.Parse = subParse;
            subParse.Ctx = ctx;
            subParse.TriggerTab = table;
            subParse.Toplevel = top;
            subParse.AuthContext = trigger.Name;
            subParse.TriggerOp = trigger.OP;
            subParse.QueryLoop = parse.QueryLoop;

            int endTrigger = 0; // Label to jump to if WHEN is false
            Vdbe v = subParse.GetVdbe(); // Temporary VM
            if (v != null)
            {
#if DEBUG
                v.VdbeComment("Start: %s.%s (%s %s%s%s ON %s)",
                    (trigger.Name != null ? trigger.zName : ""), OnErrorText(orconf),
                    (trigger.tr_tm == TRIGGER_BEFORE ? "BEFORE" : "AFTER"),
                    (trigger.OP == TK.UPDATE ? "UPDATE" : ""),
                    (trigger.OP == TK.INSERT ? "INSERT" : ""),
                    (trigger.OP == TK.DELETE ? "DELETE" : ""),
                    table.Name);
#endif
#if !OMIT_TRACE
                v.ChangeP4(-1, SysEx.Mprintf(ctx, "-- TRIGGER %s", trigger.Name), Vdbe.P4T.DYNAMIC);
#endif

                // If one was specified, code the WHEN clause. If it evaluates to false (or NULL) the sub-vdbe is immediately halted by jumping to the 
                // OP_Halt inserted at the end of the program.
                if (trigger.When != null)
                {
                    Expr when = sqlite3ExprDup(ctx, trigger.pWhen, 0); // Duplicate of trigger WHEN expression
                    if (ResolveExprNames(sNC, ref when) == RC.OK && !ctx.MallocFailed)
                    {
                        endTrigger = v.MakeLabel();
                        ExprIfFalse(subParse, when, endTrigger, SQLITE_JUMPIFNULL);
                    }
                    ExprDelete(ctx, ref when);
                }

                // Code the trigger program into the sub-vdbe.
                CodeTriggerProgram(subParse, trigger.StepList, orconf);

                // Insert an OP_Halt at the end of the sub-program.
                if (endTrigger != 0)
                    v.ResolveLabel(endTrigger);
                v.AddOp0(OP.Halt);
#if DEBUG
                v.VdbeComment("End: %s.%s", trigger.Name, OnErrorText(orconf));
#endif
                TransferParseError(parse, subParse);
                if (!db.MallocFailed)
                    program.OPs = v.TakeOpArray(ref program.OPs.length, ref top.MaxArgs);
                program.Mem = subParse.Mems;
                program.Csrs = subParse.Tabs;
                program.Token = trigger.GetHashCode();
                prg.Colmasks[0] = subParse.Oldmask;
                prg.Colmasks[1] = subParse.Newmask;
                Vdbe.Delete(ref v);
            }

            Debug.Assert(subParse.Ainc == null && subParse.ZombieTab == null);
            Debug.Assert(subParse.TriggerPrg == null && subParse.MaxArgs == 0);
            SysEx.ScratchFree(ctx, subParse);

            return prg;
        }

        static TriggerPrg GetRowTrigger(Parse parse, Trigger trigger, Table table, int orconf)
        {
            Parse root = ParseToplevel(parse);
            Debug.Assert(trigger.Name == null || table == TableOfTrigger(trigger));

            // It may be that this trigger has already been coded (or is in the process of being coded). If this is the case, then an entry with
            // a matching TriggerPrg.pTrigger field will be present somewhere in the Parse.pTriggerPrg list. Search for such an entry.
            TriggerPrg prg;
            for (prg = root.pTriggerPrg; prg != null && (prg.Trigger != trigger || prg.orconf != orconf); prg = prg.Next) ;
            // If an existing TriggerPrg could not be located, create a new one.
            if (prg == null)
                prg = CodeRowTrigger(parse, trigger, table, orconf);
            return prg;
        }

        static void sqlite3CodeRowTriggerDirect(Parse parse, Trigger p, Table table, int reg, int orconf, int ignoreJump)
        {
            Vdbe v = parse.GetVdbe(); // Main VM
            TriggerPrg prg = GetRowTrigger(parse, p, table, orconf);
            Debug.Assert(prg != null || parse.Errs != 0 || parse.Ctx.MallocFailed);

            // Code the OP_Program opcode in the parent VDBE. P4 of the OP_Program is a pointer to the sub-vdbe containing the trigger program.
            if (prg != null)
            {
                bool recursive = (!string.IsNullOrEmpty(p.Name) && (parse.Ctx.Flags & Context.FLAG.RecTriggers) == 0);
                v.AddOp3(OP.Program, reg, ignoreJump, ++parse.Mems);
                v.ChangeP4(-1, prg.pProgram, Vdbe.P4T.SUBPROGRAM);
#if DEBUG
                v.VdbeComment("Call: %s.%s", (!string.IsNullOrEmpty(p.Name) ? p.Name : "fkey"), OnErrorText(orconf));
#endif
                // Set the P5 operand of the OP_Program instruction to non-zero if recursive invocation of this trigger program is disallowed. Recursive
                // invocation is disallowed if (a) the sub-program is really a trigger, not a foreign key action, and (b) the flag to enable recursive triggers is clear.
                v.ChangeP5((int)(recursive ? 1 : 0));
            }
        }

        static void sqlite3CodeRowTrigger(Parse parse, Trigger trigger, int op, ExprList changes, int tr_tm, Table table, int reg, int orconf, int ignoreJump)
        {
            Debug.Assert(op == TK.UPDATE || op == TK.INSERT || op == TK.DELETE);
            Debug.Assert(tr_tm == TRIGGER_BEFORE || tr_tm == TRIGGER_AFTER);
            Debug.Assert((op == TK.UPDATE) == (changes != null));

            for (Trigger p = trigger; p != null; p = p.Next)
            {
                // Sanity checking:  The schema for the trigger and for the table are always defined.  The trigger must be in the same schema as the table or else it must be a TEMP trigger.
                Debug.Assert(p.Schema != null);
                Debug.Assert(p.TabSchema != null);
                Debug.Assert(p.Schema == p.TabSchema || p.Schema == parse.Ctx.DBs[1].Schema);

                // Determine whether we should code this trigger
                if (p.OP == op && p.tr_tm == tr_tm && CheckColumnOverlap(p.Columns, changes) != 0)
                    sqlite3CodeRowTriggerDirect(parse, p, table, reg, orconf, ignoreJump);
            }
        }

        static uint32 sqlite3TriggerColmask(Parse parse, Trigger trigger, ExprList changes, bool isNew, int tr_tm, Table table, int orconf)
        {
            TK op = (changes != null ? TK.UPDATE : TK.DELETE);
            int isNewId = (isNew ? 1 : 0);
            uint32 mask = 0;
            for (Trigger p = trigger; p != null; p = p.pNext)
                if (p.OP == op && (tr_tm & p.tr_tm) != 0 && CheckColumnOverlap(p.Columns, changes))
                {
                    TriggerPrg prg = GetRowTrigger(parse, p, table, orconf);
                    if (prg != null)
                        mask |= prg.Colmasks[isNewId];
                }
            return mask;
        }
    }
}
#endif
#endregion