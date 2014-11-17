// trigger.c
#ifndef OMIT_TRIGGER
#include "Core+Vdbe.cu.h"

namespace Core
{
	__device__ void Trigger::DeleteTriggerStep(Context *ctx, TriggerStep *triggerStep)
	{
		while (triggerStep)
		{
			TriggerStep *tmp = triggerStep;
			triggerStep = triggerStep->Next;

			Expr::Delete(ctx, tmp->Where);
			Expr::ListDelete(ctx, tmp->ExprList);
			Select::Delete(ctx, tmp->Select);
			IdList::Delete(ctx, tmp->IdList);

			_tagfree(ctx, tmp);
		}
	}

	__device__ Trigger *Trigger::List(Parse *parse, Table *table)
	{
		Schema *const tmpSchema = parse->Ctx->DBs[1].Schema;
		Trigger *list = nullptr; // List of triggers to return
		if (parse->DisableTriggers)
			return nullptr;
		if (tmpSchema != table->Schema)
		{
			_assert(Btree::SchemaMutexHeld(parse->Ctx, 0, tmpSchema));
			for (HashElem *p = tmpSchema->TriggerHash.First; p; p = p->Next)
			{
				Trigger *trig = (Trigger *)p->Data;
				if (trig->TabSchema == table->Schema && !_strcmp(trig->Table, table->Name))
				{
					trig->Next = (list ? list : table->Trigger);
					list = trig;
				}
			}
		}

		return (list ? list : table->Trigger);
	}

	void sqlite3BeginTrigger(Parse *parse,Token *name1, Token *name2, int trTm, int op, IdList *columns, SrcList *tableName, Expr *when, bool isTemp, int noErr)
	{
		Context *ctx = parse->Ctx; // The database connection
		_assert(name1); // pName1->z might be NULL, but not pName1 itself
		_assert(name2);
		_assert(op == TK_INSERT || op == TK_UPDATE || op == TK_DELETE);
		_assert(op > 0 && op<0xff);
		Trigger *trigger = nullptr;  // The new trigger
		DbFixer fix;  // State vector for the DB fixer
		int tabDb; // Index of the database holding pTab

		int db; // The database to store the trigger in
		Token *name; // The unqualified db name
		if (isTemp)
		{
			// If TEMP was specified, then the trigger name may not be qualified.
			if (name2->length > 0)
			{
				parse->ErrorMsg("temporary trigger may not have qualified name");
				goto trigger_cleanup;
			}
			db = 1;
			name = name1;
		}
		else
		{
			// Figure out the db that the trigger will be created in
			db = parse->TwoPartName(name1, name2, &name);
			if (db < 0)
				goto trigger_cleanup;
		}
		if (!tableName || ctx->MallocFailed)
			goto trigger_cleanup;


		// A long-standing parser bug is that this syntax was allowed:
		//
		//    CREATE TRIGGER attached.demo AFTER INSERT ON attached.tab ....
		//                                                 ^^^^^^^^
		// To maintain backwards compatibility, ignore the database name on pTableName if we are reparsing our of SQLITE_MASTER.
		if (ctx->Init.Busy && db != 1)
		{
			_tagfree(ctx, tableName->Ids[0].Database);
			tableName->Ids[0].Database = nullptr;
		}

		// If the trigger name was unqualified, and the table is a temp table, then set iDb to 1 to create the trigger in the temporary database.
		// If sqlite3SrcListLookup() returns 0, indicating the table does not exist, the error is caught by the block below.
		Table *table = sqlite3SrcListLookup(parse, tableName); // Table that the trigger fires off of
		if (!ctx->Init.Busy && name2->length == 0 && table && table->Schema == ctx->DBs[1].Schema)
			db = 1;

		// Ensure the table name matches database name and that the table exists
		if (ctx->MallocFailed) goto trigger_cleanup;
		_assert(tableName->Srcs == 1);
		if (sqlite3FixInit(&fix, parse, db, "trigger", name) && sqlite3FixSrcList(&sFix, tableName))
			goto trigger_cleanup;
		table = sqlite3SrcListLookup(parse, tableName);
		if (!table)
		{
			// The table does not exist.
			if (ctx->Init.DB == 1)
			{
				// Ticket #3810.
				// Normally, whenever a table is dropped, all associated triggers are dropped too.  But if a TEMP trigger is created on a non-TEMP table
				// and the table is dropped by a different database connection, the trigger is not visible to the database connection that does the
				// drop so the trigger cannot be dropped.  This results in an "orphaned trigger" - a trigger whose associated table is missing.
				ctx->Init.OrphanTrigger = 1;
			}
			goto trigger_cleanup;
		}
		if (IsVirtual(table))
		{
			parse->ErrorMsg("cannot create triggers on virtual tables");
			goto trigger_cleanup;
		}

		// Check that the trigger name is not reserved and that no trigger of the specified name exists
		char *nameAsString = sqlite3NameFromToken(ctx, name); // Name of the trigger
		if (!nameAsString || sqlite3CheckObjectName(parse, nameAsString) != RC_OK)
			goto trigger_cleanup;
		_assert(Btree::SchemaMutexHeld(ctx, db, 0));
		if (ctx->DBs[db].Schema->TriggerHash.Find(nameAsString, _strlen30(nameAsString)))
		{
			if (!noErr)
				parse->ErrorMsg("trigger %T already exists", name);
			else
			{
				_assert(!ctx->Init.Busy);
				sqlite3CodeVerifySchema(parse, db);
			}
			goto trigger_cleanup;
		}

		// Do not create a trigger on a system table
		if (!_strncmp(table->Name, "sqlite_", 7))
		{
			parse->ErrorMsg("cannot create trigger on system table");
			parse->Errs++;
			goto trigger_cleanup;
		}

		// INSTEAD of triggers are only for views and views only support INSTEAD of triggers.
		if (table->Select && trRm != TK_INSTEAD)
		{
			parse->ErrorMsg("cannot create %s trigger on view: %S",  (trTm == TK_BEFORE ? "BEFORE" : "AFTER"), tableName, 0);
			goto trigger_cleanup;
		}
		if (!table->Select && trRm == TK_INSTEAD)
		{
			parse->ErrorMsg("cannot create INSTEAD OF trigger on table: %S", tableName, 0);
			goto trigger_cleanup;
		}
		tabDb = sqlite3SchemaToIndex(ctx, table->Schema);

#ifndef OMIT_AUTHORIZATION
		{
			int code = AUTH_CREATE_TRIGGER;
			const char *dbName = ctx->DBs[tabDB].zName;
			const char *dbTrigName = (isTemp ? ctx->DBs[1].Name : dbName);
			if (tabDB == 1 || isTemp) code = SQLITE_CREATE_TEMP_TRIGGER;
			if (Auth::Check(parse, code, nameAsString, table->Name, dbTrigName) || Auth::Check(parse, AUTH_INSERT, SCHEMA_TABLE(tabDb), 0, dbName))
				goto trigger_cleanup;
		}
#endif

		// INSTEAD OF triggers can only appear on views and BEFORE triggers cannot appear on views.  So we might as well translate every
		// INSTEAD OF trigger into a BEFORE trigger.  It simplifies code elsewhere.
		if (trTm == TK_INSTEAD)
			trTm = TK_BEFORE;

		// Build the Trigger object
		trigger = (Trigger *)_tagalloc(ctx, sizeof(Trigger), true);
		if (!trigger) goto trigger_cleanup;
		trigger->Name = name; name = nullptr;
		trigger->Table = _tagstrdup(ctx, tableName->Ids[0].Name);
		trigger->Schema = ctx->DBs[db].Schema;
		trigger->TabSchema = tab->Schema;
		trigger->OP = (uint8)op;
		trigger->TR_tm = (trTm == TK_BEFORE ? TRIGGER_BEFORE : TRIGGER_AFTER);
		trigger->When = Expr::Dup(ctx, when, EXPRDUP_REDUCE);
		trigger->Columns = IdListDup(ctx, columns);
		_assert(parse->NewTrigger == 0);
		parse->NewTrigger = trigger;

trigger_cleanup:
		_tagfree(ctx, name);
		SrcListDelete(ctx, tableName);
		IdListDelete(ctx, columns);
		Expr::Delete(ctx, when);
		if (!pParse->NewTrigger)
			DeleteTrigger(ctx, trigger);
		else
			_assert(parse->NewTrigger == trigger);
	}

	void sqlite3FinishTrigger(Parse *parse, TriggerStep *stepList, Token *all)
	{
		Trigger *trig = parse->NewTrigger; // Trigger being finished
		Context *ctx = parse->Ctx; // The database
		DbFixer sFix; // Fixer object
		Token nameToken; // Trigger name for error reporting

		parse->NewTrigger = nullptr;
		if (_NEVER(parse->Errs) || !trig) goto triggerfinish_cleanup;
		char *name = trig->Name; // Name of trigger
		int db = sqlite3SchemaToIndex(ctx, trig->Schema); // Database containing the trigger
		trig->StepList = stepList;
		while (stepList)
		{
			stepList->Trig = trig;
			stepList = stepList->Next;
		}
		nameToken.data = trig->Name;
		nameToken.length = _strlen30(nameToken.data);
		if (sqlite3FixInit(&sFix, parse, db, "trigger", &nameToken) && sqlite3FixTriggerStep(&sFix, trig->StepList))
			goto triggerfinish_cleanup;

		// if we are not initializing, build the sqlite_master entry
		if (!ctx->Init.Busy)
		{
			// Make an entry in the sqlite_master table
			Vdbe *v = parse->GetVdbe();
			if (!v) goto triggerfinish_cleanup;
			parse->BeginWriteOperation(0, db);
			char *z = _tagstrndup(ctx, (char *)all->data, all->length);
			parse->NestedParse("INSERT INTO %Q.%s VALUES('trigger',%Q,%Q,0,'CREATE TRIGGER %q')", ctx->DBs[db].Name, SCHEMA_TABLE(db), name, trig->table, z);
			_tagfree(ctx, z);
			parse->ChangeCookie(db);
			v->AddParseSchemaOp(db, _mprintf(ctx, "type='trigger' AND name='%q'", name));
		}

		if (ctx->Init.Busy)
		{
			Trigger *link = trig;
			_assert(Btree::SchemaMutexHeld(ctx, db, 0));
			trig = (Trigger *)ctx->DBs[db].Schema->TriggerHash.Insert(name, _strlen30(zName), trig);
			if (trig)
				ctx->MallocFailed = true;
			else if (link->Schema == link->TabSchema)
			{
				int tableLength = _strlen30(link->Table);
				Table *table = link->TabSchema->TableHash.Find(link->table, tableLength);
				_assert(table);
				link->Next = tab->Trigger;
				tab->Trigger = link;
			}
		}

triggerfinish_cleanup:
		DeleteTrigger(ctx, trig);
		_assert(!parse->NewTrigger);
		DeleteTriggerStep(ctx, stepList);
	}

	TriggerStep *sqlite3TriggerSelectStep(Context *ctx, Select *select)
	{
		TriggerStep *triggerStep = (TriggerStep *)_tagalloc(ctx, sizeof(TriggerStep), true);
		if (!triggerStep)
		{
			SelectDelete(ctx, select);
			return nullptr;
		}
		triggerStep->OP = TK_SELECT;
		triggerStep->Select = select;
		triggerStep->Orconf = OE_Default;
		return triggerStep;
	}

	static TriggerStep *TriggerStepAllocate(Context *ctx, uint8 op, Token *name)
	{
		TriggerStep *triggerStep = (TriggerStep *)_tagalloc(ctx, sizeof(TriggerStep) + name->length, true);
		if (triggerStep)
		{
			char *z = (char *)&triggerStep[1];
			_memcpy(z, name->data, name->length);
			triggerStep->Target.data = data;
			triggerStep->Target.length = name->length;
			triggerStep->OP = op;
		}
		return triggerStep;
	}

	TriggerStep *sqlite3TriggerInsertStep(Context *ctx, Token *tableName, IdList *column, ExprList *list, Select *select, OE orconf)
	{
		_assert(list == 0 || select == 0);
		_assert(list != 0 || select != 0 || ctx->MallocFailed);
		TriggerStep *triggerStep = TriggerStepAllocate(ctx, TK_INSERT, tableName);
		if (triggerStep)
		{
			triggerStep->Select = SelectDup(ctx, select, EXPRDUP_REDUCE);
			triggerStep->IdList = column;
			triggerStep->ExprList = ExprListDup(ctx, list, EXPRDUP_REDUCE);
			triggerStep->Orconf = orconf;
		}
		else
			IdListDelete(ctx, column);
		ExprListDelete(ctx, list);
		SelectDelete(ctx, select);
		return triggerStep;
	}

	TriggerStep *sqlite3TriggerUpdateStep(Context *ctx, Token *tableName, ExprList *list, Expr *where, OE orconf)
	{
		TriggerStep *triggerStep = TriggerStepAllocate(ctx, TK_UPDATE, tableName);
		if (triggerStep)
		{
			triggerStep->ExprList = ExprList::Dup(ctx, lList, EXPRDUP_REDUCE);
			triggerStep->Where = Expr::Dup(ctx, where, EXPRDUP_REDUCE);
			triggerStep->Orconf = orconf;
		}
		ExprListDelete(ctx, list);
		ExprDelete(ctx, where);
		return triggerStep;
	}

	TriggerStep *sqlite3TriggerDeleteStep(Context *ctx, Token *pTableName, Expr *where_)
	{
		TriggerStep *pTriggerStep = TriggerStepAllocate(ctx, TK_DELETE, pTableName);
		if (triggerStep)
		{
			triggerStep->Where = Expr::Dup(ctx, where, EXPRDUP_REDUCE);
			triggerStep->Orconf = OE_Default;
		}
		ExprDelete(ctx, where);
		return pTriggerStep;
	}

	void sqlite3DeleteTrigger(Context *ctx, Trigger *trigger)
	{
		if (!trigger) return;
		DeleteTriggerStep(ctx, trigger->StepList);
		_tagfree(ctx, trigger->Name);
		_tagfree(ctx, trigger->Table);
		ExprDelete(ctx, trigger->When);
		IdListDelete(ctx, trigger->Columns);
		_tagfree(ctx, trigger);
	}

	void sqlite3DropTrigger(Parse *parse, SrcList *name, int noErr)
	{
		Context *ctx = parse->Ctx;
		if (ctx->MallocFailed || parse->ReadSchema() != RC_OK)
			goto drop_trigger_cleanup;

		_assert(name->Srcs == 1);
		const char *dbName = name->Ids[0].Database;
		const char *nameAsString = name->Ids[0].Name;
		int nameLength = _strlen30(nameAsString);
		_assert(dbName != 0 || Btree::BtreeHoldsAllMutexes(ctx));
		Trigger *trigger = nullptr;
		for (int i = OMIT_TEMPDB; i < ctx->DBs.length; i++)
		{
			int j = (i < 2 ? i^1 : i); // Search TEMP before MAIN
			if (dbName && _strcmp(ctx->DBs[j].Name, dbName)) continue;
			_assert(Btree::SchemaMutexHeld(ctx, j, 0));
			trigger = (Trigger *)ctx->DBs[j].Schema->TriggerHash.Find(nameAsString, nameLength);
			if (trigger) break;
		}
		if (!trigger)
		{
			if (!noErr)
				parse->ErrorMsg("no such trigger: %S", name, 0);
			else
				parse->CodeVerifyNamedSchema(dbName);
			parse->CheckSchema = true;
			goto drop_trigger_cleanup;
		}
		DropTriggerPtr(parse, trigger);

drop_trigger_cleanup:
		SrcListDelete(ctx, name);
	}

	static Table *TableOfTrigger(Trigger *trigger)
	{
		int tableLength = _strlen30(trigger->Table);
		return trigger->TabSchema->TableHash.Find(trigger->Table, tableLength);
	}

	static const VdbeOpList _dropTrigger[] = {
		{ OP_Rewind,     0, ADDR(9),  0},
		{ OP_String8,    0, 1,        0}, /* 1 */
		{ OP_Column,     0, 1,        2},
		{ OP_Ne,         2, ADDR(8),  1},
		{ OP_String8,    0, 1,        0}, /* 4: "trigger" */
		{ OP_Column,     0, 0,        2},
		{ OP_Ne,         2, ADDR(8),  1},
		{ OP_Delete,     0, 0,        0},
		{ OP_Next,       0, ADDR(1),  0}, /* 8 */
	};
	void sqlite3DropTriggerPtr(Parse *parse, Trigger *trigger)
	{
		Context *ctx = parse->Ctx;
		int db = ctx->SchemaToIndex(trigger->Schema);
		_assert(db >= 0 && db < ctx->DB.length);
		Table *table = TableOfTrigger(trigger);
		_assert(table);
		_assert(table->Schema == trigger->Schema || db == 1);
#ifndef OMIT_AUTHORIZATION
		{
			int code = AUTH::DROP_TRIGGER;
			const char *dbName = ctx->DBs[db].Name;
			const char *tableName = SCHEMA_TABLE(db);
			if (db == 1) code = AUTH::DROP_TEMP_TRIGGER;
			if (Auth::Check(parse, code, trigger->Name, table->Name, dbName) || Auth::Check(parse, AUTH::DELETE, tableName, 0, dbName))
				return;
		}
#endif

		// Generate code to destroy the database record of the trigger.
		_assert(table);
		Vdbe *v = parse->GetVdbe();
		if (v)
		{
			parse->BeginWriteOperation(0, db);
			parse->OpenMasterTable(db);
			int base_ = v->AddOpList(_lengthof(_dropTrigger), _dropTrigger);
			v->ChangeP4(base_+1, trigger->Name, Vdbe::P4T_TRANSIENT);
			v->ChangeP4(base_+4, "trigger", Vdbe::P4T_STATIC);
			parse->ChangeCookie(db);
			v->AddOp2(OP_Close, 0, 0);
			v->AddOp4(OP_DropTrigger, db, 0, 0, trigger->Name, 0);
			if (parse->Mems < 3)
				parse->Mems = 3;
		}
	}

	void sqlite3UnlinkAndDeleteTrigger(Context *ctx, int db, const char *name)
	{
		_assert(Btree::SchemaMutexHeld(ctx, db, 0));
		Trigger *trigger = ctx->DBs[db].Schema->TriggerHash.Insert(name, _strlen30(name), 0);
		if (_ALWAYS(trigger))
		{
			if (trigger->Schema == trigger->TabSchema)
			{
				Table *table = TableOfTrigger(trigger);
				Trigger **pp;
				for (pp = &table->Trigger; *pp != trigger; pp = &((*pp)->Next));
				*pp = (*pp)->Next;
			}
			sqlite3DeleteTrigger(ctx, trigger);
			ctx->Flags |= Context::FLAG_InternChanges;
		}
	}

	static int checkColumnOverlap(IdList *idList, ExprList *eList)
	{
		if (!idList || SysEx::NEVER(!eList)) return 1;
		for (int e = 0; e < eList->Exprs; e++)
			if (IdListIndex(idList, eList->Ids[e].Name) >= 0) return 1;
		return 0; 
	}

	Trigger *sqlite3TriggersExist(Parse *parse, Table *table, int op, ExprList *changes, int *maskOut)
	{
		int mask = 0;
		Trigger *list = nullptr;
		if ((parse->Ctx->Flags & Context::FLAG_EnableTrigger) != 0)
			list = TriggerList(parse, table);
		_assert(!list || !IsVirtual(pTab));
		for (Trigger *p = list; p; p = p->Next)
			if (p->OP == op && CheckColumnOverlap(p->Columns, changes))
				mask |= p->tr_tm;
		if (maskOut)
			*maskOut = mask;
		return (mask ? list : nullptr);
	}

	static SrcList *TargetSrcList(Parse *parse, TriggerStep *step)
	{
		Context *ctx = parse->Ctx;
		SrcList *src = sqlite3SrcListAppend(ctx, 0, &step->Target, 0); // SrcList to be returned
		if (src)
		{
			_assert(src->Srcs > 0);
			_assert(src->Ids != 0 );
			int db = Btree::SchemaToIndex(ctx, step->Trig->Schema); // Index of the database to use
			if (db == 0 || db >= 2)
			{
				_assert( db < ctx->DBs.length);
				src->Ids[src->Srcs-1].Database = _tagstrdup(ctx, ctx->DBs[db].Name);
			}
		}
		return src;
	}

	static int CodeTriggerProgram(Parse *parse, TriggerStep *stepList, int orconf)
	{
		Vdbe *v = parse->Vdbe;
		Context *ctx = parse->Ctx;
		_assert(parse->TriggerTab && parse->Toplevel);
		_assert(stepList);
		_assert(v);
		for (TriggerStep *step = stepList; step; step = step->Next)
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
			parse->Orconf = (orconf == OE_Default ? step->orconf : (uint8)orconf);

			// Clear the cookieGoto flag. When coding triggers, the cookieGoto variable is used as a flag to indicate to sqlite3ExprCodeConstants()
			// that it is not safe to refactor constants (this happens after the start of the first loop in the SQL statement is coded - at that 
			// point code may be conditionally executed, so it is no longer safe to initialize constant register values).
			_assert(parse->CookieGoto == 0 || parse->CookieGoto == -1);
			parse->CookieGoto = 0;

			switch (step->OP)
			{
			case TK_UPDATE:
				Update(parse, 
					targetSrcList(parse, step),
					ExprListDup(ctx, step->ExprList, 0), 
					ExprDup(ctx, step->Where, 0), 
					parse->eOrconf);
				break;

			case TK_INSERT:
				Insert(parse, 
					targetSrcList(parse, step),
					ExprListDup(ctx, step->ExprList, 0), 
					SelectDup(ctx, step->Select, 0), 
					IdListDup(ctx, step->IdList), 
					parse->eOrconf);
				break;

			case TK_DELETE: {
				DeleteFrom(parse, 
					SrcList(parse, step),
					ExprDup(ctx, step->Where, 0));
				break;
							}
			default: _assert(step->OP == TK_SELECT);
				SelectDest sDest;
				Select *select = SelectDup(ctx, step->Select, 0);
				SelectDestInit(&sDest, SRT_Discard, 0);
				Select(parse, select, &sDest);
				SelectDelete(ctx, select);
				break;

			} 
			if (step->OP != TK_SELECT)
				v->AddOp0(OP_ResetCount);
		}

		return 0;
	}

#ifdef DEBUG
	static const char *OnErrorText(int onError)
	{
		switch (onError)
		{
		case OE_Abort:    return "abort";
		case OE_Rollback: return "rollback";
		case OE_Fail:     return "fail";
		case OE_Replace:  return "replace";
		case OE_Ignore:   return "ignore";
		case OE_Default:  return "default";
		}
		return "n/a";
	}
#endif

	static void TransferParseError(Parse *pTo, Parse *pFrom)
	{
		_assert(from->ErrMsg == nullptr || from->Errs);
		_assert(to->ErrMsg == nullptr || to->Errs);
		if (to->Errs == 0)
		{
			to->ErrMsg = pFrom->ErrMsg;
			to->Errs = from->Errs;
		}
		else
			_tagfree(from->Ctx, from->ErrMsg);
	}

	static TriggerPrg *CodeRowTrigger(Parse *parse, Trigger *trigger, Table *table, int orconf)
	{
		Parse *top = ParseToplevel(parse);
		Context *ctx = parse->Ctx; // Database handle

		_assert(trigger->Name == nullptr || table == TableOfTrigger(trigger));
		_assert(top->Vdbe);

		// Allocate the TriggerPrg and SubProgram objects. To ensure that they are freed if an error occurs, link them into the Parse.pTriggerPrg 
		// list of the top-level Parse object sooner rather than later.
		TriggerPrg *prg = (TriggerPrg *)_tagalloc(ctx, sizeof(TriggerPrg), true); // Value to return
		if (!prg) return nullptr;
		prg->Next = top->TriggerPrg;
		top->TriggerPrg = prg;
		SubProgram *program; // Sub-vdbe for trigger program
		prg->Program = program = (SubProgram *)_tagalloc(ctx, sizeof(SubProgram), true);
		if (!program) return nullptr;
		top->Vdbe->LinkSubProgram(program);
		prg->Trigger = trigger;
		prg->Orconf = orconf;
		prg->Colmasks[0] = 0xffffffff;
		prg->Colmasks[1] = 0xffffffff;

		// Allocate and populate a new Parse context to use for coding the trigger sub-program.
		Parse *subParse = (Parse *)_stackalloc(ctx, sizeof(Parse), true); // Parse context for sub-vdbe
		if (!subParse) return nullptr;
		NameContext sNC; _memset(&sNC, 0, sizeof(sNC)); // Name context for sub-vdbe
		sNC.Parse = subParse;
		subParse->Ctx = ctx;
		subParse->TriggerTab = tab;
		subParse->Toplevel = top;
		subParse->AuthContext = trigger->Name;
		subParse->TriggerOp = trigger->OP;
		subParse->QueryLoop = parse->QueryLoop;

		int endTrigger = 0; // Label to jump to if WHEN is false
		Vdbe *v = subParse->GetVdbe(); // Temporary VM
		if (v)
		{
			v->Comment("Start: %s.%s (%s %s%s%s ON %s)", 
				trigger->Name, OnErrorText(orconf),
				(trigger->Tr_tm == TRIGGER_BEFORE ? "BEFORE" : "AFTER"),
				(trigger->OP == TK_UPDATE ? "UPDATE" : ""),
				(trigger->OP == TK_INSERT ? "INSERT" : ""),
				(trigger->OP == TK_DELETE ? "DELETE" : ""),
				tab->Name);
#ifndef OMIT_TRACE
			v->ChangeP4(-1, _mprintf(ctx, "-- TRIGGER %s", trigger->Name), Vdbe::P4T_DYNAMIC);
#endif

			// If one was specified, code the WHEN clause. If it evaluates to false (or NULL) the sub-vdbe is immediately halted by jumping to the 
			// OP_Halt inserted at the end of the program.
			if (trigger->When)
			{
				Expr *when = ExprDup(ctx, trigger->When, 0); // Duplicate of trigger WHEN expression
				if (ResolveExprNames(&sNC, when) == RC_OK && !ctx->MallocFailed)
				{
					endTrigger = v->MakeLabel(v);
					ExprIfFalse(subParse, when, endTrigger, SQLITE_JUMPIFNULL);
				}
				ExprDelete(ctx, when);
			}

			// Code the trigger program into the sub-vdbe.
			CodeTriggerProgram(subParse, trigger->StepList, orconf);

			// Insert an OP_Halt at the end of the sub-program.
			if (endTrigger)
				v->ResolveLabel(endTrigger);
			v->AddOp0(OP_Halt);
			v->Comment(v, "End: %s.%s", trigger->Name, OnErrorText(orconf));

			TransferParseError(parse, subParse);
			if (!ctx->MallocFailed)
				program->OPs = v->TakeOpArray(&program->OP.length, &top->MaxArgs);
			program->Mems = subParse->Mems;
			program->Csrs = subParse->Tabs;
			program->Onces = subParse->Onces;
			program->Token = (void *)trigger;
			prg->Colmasks[0] = subParse->Oldmask;
			prg->Colmasks[1] = subParse->Newmask;
			Vdbe::Delete(v);
		}

		_assert(!subParse->Ainc && !subParse->ZombieTab);
		_assert(!subParse->TriggerPrg && !subParse->MaxArgs);
		_stackfree(ctx, subParse);

		return prg;
	}

	static TriggerPrg *GetRowTrigger(Parse *parse, Trigger *trigger, Table *table,  int orconf)
	{
		Parse *root = ParseToplevel(parse);
		_assert(!trigger->Name || table == TableOfTrigger(trigger));

		// It may be that this trigger has already been coded (or is in the process of being coded). If this is the case, then an entry with
		// a matching TriggerPrg.pTrigger field will be present somewhere in the Parse.pTriggerPrg list. Search for such an entry.
		TriggerPrg *prg;
		for (prg = root->TriggerPrg; prg && (prg->Trigger != trigger || prg->Orconf != orconf); prg = prg->Next) ;
		// If an existing TriggerPrg could not be located, create a new one.
		if (!prg)
			prg = CodeRowTrigger(parse, trigger, table, orconf);
		return prg;
	}

	void sqlite3CodeRowTriggerDirect(Parse *parse, Trigger *p, Table *table, int reg, int orconf, int ignoreJump)
	{
		Vdbe *v = parse->GetVdbe(); // Main VM
		TriggerPrg *prg = GetRowTrigger(parse, p, table, orconf);
		_assert(prg || parse->Errs || parse->Ctx->MallocFailed);

		// Code the OP_Program opcode in the parent VDBE. P4 of the OP_Program is a pointer to the sub-vdbe containing the trigger program.
		if (prg)
		{
			bool recursive = (p->Name && (parse->Ctx->Flags & Context::FLAG_RecTriggers) == 0);

			v->AddOp3(v, OP_Program, reg, ignoreJump, ++parse->Mems);
			v->ChangeP4(v, -1, (const char *)prg->Program, Vdbe::P4T_SUBPROGRAM);
#if DEBUG
			v->VdbeComment("Call: %s.%s", (p->Name ? p->Name : "fkey"), OnErrorText(orconf));
#endif

			// Set the P5 operand of the OP_Program instruction to non-zero if recursive invocation of this trigger program is disallowed. Recursive
			// invocation is disallowed if (a) the sub-program is really a trigger, not a foreign key action, and (b) the flag to enable recursive triggers is clear.
			v->ChangeP5((uint8)recursive);
		}
	}

	void sqlite3CodeRowTrigger(Parse *parse, Trigger *trigger, int op, ExprList *changes, int tr_tm, Table *table, int reg, int orconf, int ignoreJump)
	{
		_assert(op == TK_UPDATE || op == TK_INSERT || op == TK_DELETE );
		_assert(tr_tm == TRIGGER_BEFORE || tr_tm == TRIGGER_AFTER );
		_assert((op == TK_UPDATE) == (changes != nullptr));

		for (Trigger *p = trigger; p; p = p->Next)
		{
			// Sanity checking:  The schema for the trigger and for the table are always defined.  The trigger must be in the same schema as the table or else it must be a TEMP trigger.
			_assert(p->Schema);
			_assert(p->TabSchema);
			_assert(p->Schema == p->TabSchema || p->Schema == parse->Ctx->DBs[1].Schema);

			// Determine whether we should code this trigger
			if (p->OP == op && p->tr_tm == tr_tm && CheckColumnOverlap(p->Columns, changes))
				sqlite3CodeRowTriggerDirect(parse, p, table, reg, orconf, ignoreJump);
		}
	}

	uint32 sqlite3TriggerColmask(Parse *parse, Trigger *trigger, ExprList *changes, bool isNew, int tr_tm, Table *table, int orconf)
	{
		const int op = (changes ? TK_UPDATE : TK_DELETE);
		int isNewId = (isNew ? 1 : 0);
		uint32 mask = 0;
		for (Trigger *p = trigger; p; p = p->Next)
			if (p->OP == op && (tr_tm & p->tr_tm) && CheckColumnOverlap(p->Columns, changes))
			{
				TriggerPrg *prg = GetRowTrigger(parse, p, table, orconf);
				if (prg)
					mask |= prg->Colmasks[isNewId];
			}
			return mask;
	}

#endif
}