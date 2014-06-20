#include "..\Core+Syntax.cu.h"

namespace Core { namespace Command
{

#ifndef OMIT_VIRTUALTABLE
	static void UpdateVirtualTable(Parse *parse, SrcList *src, Table *table, ExprList *changes, Expr *rowidExpr, int *xref, Expr *where, int onError);
#endif

	__device__ void Update::ColumnDefault(Vdbe *v, Table *table, int i, int regId)
	{
		_assert(table);
		if (!table->Select)
		{
			TEXTENCODE encode = CTXENCODE(v->Ctx);
			Column *col = &table->Cols[i];
			v->VdbeComment("%s.%s", table->Name, col->Name);
			_assert(i < table->Cols.length);
			Mem *value;
			sqlite3ValueFromExpr(v->ctx, col->Dflt, textencode, col->Affinity, &value);
			if (value)
				v->ChangeP4(-1, (const char *)value, Context::P4T_MEM);
#ifndef OMIT_FLOATING_POINT
			if (regId >= 0 && table->Cols[i].Affinity == AFF_REAL)
				v->AddOp1(OP_RealAffinity, regId);
#endif
		}
	}

	__device__ void Update::Update(Parse *parse, SrcList *tabList, ExprList *changes, Expr *where_, OE onError)
	{
		int i, j;              // Loop counters

		AuthContext sContext;  // The authorization context
		_memset(&sContext, 0, sizeof(sContext));
		Context *ctx = parse->Ctx; // The database structure
		if (parse->Errs || ctx->MallocFailed)
			goto update_cleanup;
		_assert(tabList->Srcs == 1);

		// Locate the table which we want to update. 
		Table *table = sqlite3SrcListLookup(parse, tabList); // The table to be updated
		if (!table) goto update_cleanup;
		int db = Context::SchemaToIndex(ctx, table->Schema); // Database containing the table being updated

		// Figure out if we have any triggers and if the table being updated is a view.
#ifndef OMIT_TRIGGER
		int tmask; // Mask of TRIGGER_BEFORE|TRIGGER_AFTER
		Trigger *trigger = sqlite3TriggersExist(parse, table, TK_UPDATE, changes, &tmask); // List of triggers on table, if required
		bool isView = (table->Select != nullptr); // True when updating a view (INSTEAD OF trigger)
		assert(trigger || tmask == 0);
#else
#define tmask 0
#define trigger nullptr
#define isView false
#endif
#ifdef OMIT_VIEW
#undef isView
#define isView false
#endif
		if (sqlite3ViewGetColumnNames(parse, table) || sqlite3IsReadOnly(parse, table, tmask))
			goto update_cleanup;
		int *xrefs = (int *)SysEx::TagAlloc(ctx, sizeof(int) * table->Cols.length); // xrefs[i] is the index in pChanges->a[] of the an expression for the i-th column of the table. xrefs[i]==-1 if the i-th column is not changed.
		if (!xrefs) goto update_cleanup;
		for (i = 0; i < table->Cols.length; i++) xrefs[i] = -1;

		// Allocate a cursors for the main database table and for all indices. The index cursors might not be used, but if they are used they
		// need to occur right after the database cursor.  So go ahead and allocate enough space, just in case.
		int curId; // VDBE Cursor number of table
		tabList->Ids[0].Cursor = curId = parse->Tabs++;
		Index *idx; // For looping over indices
		for (idx = table->Index; idx; idx = idx->Next)
			parse->Tabs++;

		// Initialize the name-context
		NameContext sNC; // The name-context to resolve expressions in
		_memset(&sNC, 0, sizeof(sNC));
		sNC.Parse = parse;
		sNC.SrcList = tabList;

		// Resolve the column names in all the expressions of the of the UPDATE statement.  Also find the column index
		// for each column to be updated in the pChanges array.  For each column to be updated, make sure we have authorization to change that column.
		bool chngRowid = false; // True if the record number is being changed
		Expr *rowidExpr = nullptr; // Expression defining the new record number
		for (i = 0; i < changes->Exprs; i++)
		{
			if (sqlite3ResolveExprNames(&sNC, changes->Ids[i].Expr))
				goto update_cleanup;
			for (j = 0; j < table->Cols.length; j++)
			{
				if (!_strcmp(table->Cols[j].Name, changes->Ids[i].Name))
				{
					if (j == table->PKey)
					{
						chngRowid = true;
						rowidExpr = changes->Ids[i].Expr;
					}
					xrefs[j] = i;
					break;
				}
			}
			if (j >= table->Cols.length)
			{
				if (sqlite3IsRowid(changes->Ids[i].Name))
				{
					chngRowid = true;
					rowidExpr = changes->Ids[i].Expr;
				}
				else
				{
					parse->ErrorMsg("no such column: %s", changes->Ids[i].Name);
					parse->CheckSchema = 1;
					goto update_cleanup;
				}
			}
#ifndef OMIT_AUTHORIZATION
			{
				ARC rc = Auth::Check(parse, AUTH_UPDATE, table->Name, table->Cols[j].Name, ctx->DBs[db].Name);
				if (rc == ARC_DENY) goto update_cleanup;
				else if (rc == ARC_IGNORE) xrefs[j] = -1;
			}
#endif
		}

		bool hasFK = sqlite3FkRequired(parse, table, xrefs, chngRowid); // True if foreign key processing is required

		// Allocate memory for the array aRegIdx[].  There is one entry in the array for each index associated with table being updated.  Fill in
		// the value with a register number for indices that are to be used and with zero for unused indices.
		int idxLength; // Number of indices that need updating
		for (idxLength = 0, idx = table->Index; idx; idx = idx->Next, idxLength++) ;
		int *regIdxs = 0; // One register assigned to each index to be updated
		if (idxLength > 0)
		{
			regIdxs = (int *)SysEx::TagAlloc(ctx, sizeof(Index *) * idxLength);
			if (!regIdxs) goto update_cleanup;
		}
		for (j = 0, idx = table->Index; idx; idx = idx->Next, j++)
		{
			int regId;
			if (hasFK || chngRowid)
				regId = ++parse->Mems;
			else
			{
				regId = 0;
				for (i = 0; i < idx->Columns.length; i++)
					if (xrefs[idx->Columns[i]] >= 0)
					{
						regId = ++parse->Mems;
						break;
					}
			}
			regIdxs[j] = regId;
		}

		// Begin generating code.
		Vdbe *v = parse->GetVdbe(); // The virtual database engine
		if (!v) goto update_cleanup;
		if (parse->Nested == 0) v->CountChanges();
		parse->BeginWriteOperation(1, db);

#ifndef OMIT_VIRTUALTABLE
		// Virtual tables must be handled separately
		if (IsVirtual(table))
		{
			UpdateVirtualTable(parse, tabList, table, changes, rowidExpr, xrefs, where_, onError);
			where_ = nullptr;
			tabList = nullptr;
			goto update_cleanup;
		}
#endif

		// Register Allocations
		int regRowCount = 0;   // A count of rows changed
		int regOldRowid;       // The old rowid
		int regNewRowid;       // The new rowid
		int regNew;            // Content of the NEW.* table in triggers
		int regOld = 0;        // Content of OLD.* table in triggers
		int regRowSet = 0;     // Rowset of rows to be updated

		// Allocate required registers.
		regRowSet = ++parse->Mems;
		regOldRowid = regNewRowid = ++parse->Mems;
		if (trigger || hasFK)
		{
			regOld = parse->Mems + 1;
			parse->Mems += table->Cols.length;
		}
		if (chngRowid || trigger || hasFK)
			regNewRowid = ++parse->Mems;
		regNew = parse->Mems + 1;
		parse->Mems += table->Cols.length;

		// Start the view context.
		if (isView)
			Auth::ContextPush(parse, &sContext, table->Name);

		// If we are trying to update a view, realize that view into a ephemeral table.
#if !defined(OMIT_VIEW) && !defined(OMIT_TRIGGER)
		if (isView)
			sqlite3MaterializeView(parse, table, where_, curId);
#endif

		// Resolve the column names in all the expressions in the WHERE clause.
		if (sqlite3ResolveExprNames(&sNC, where_))
			goto update_cleanup;

		// Begin the database scan
		v->AddOp3(OP_Null, 0, regRowSet, regOldRowid);
		WhereInfo *winfo = Where::Begin(parse, tabList, where_, 0, 0, WHERE_ONEPASS_DESIRED, 0); // Information about the WHERE clause
		if (!winfo) goto update_cleanup;
		bool okOnePass = winfo->OkOnePass; // True for one-pass algorithm without the FIFO

		// Remember the rowid of every item to be updated.
		v->AddOp2(OP_Rowid, curId, regOldRowid);
		if (!okOnePass)
			v->AddOp2(OP_RowSetAdd, regRowSet, regOldRowid);

		// End the database scan loop.
		Where::End(winfo);

		// Initialize the count of updated rows
		if ((ctx->Flags & Context::FLAG_CountRows) && !parse->TriggerTab)
		{
			regRowCount = ++parse->Mems;
			v->AddOp2(OP_Integer, 0, regRowCount);
		}

		bool openAll = false;       // True if all indices need to be opened
		if (!isView)
		{
			// Open every index that needs updating.  Note that if any index could potentially invoke a REPLACE conflict resolution 
			// action, then we need to open all indices because we might need to be deleting some records.
			if (!okOnePass) sqlite3OpenTable(parse, curId, db, table, OP_OpenWrite); 
			if (onError == OE_Replace)
				openAll = true;
			else
			{
				openAll = false;
				for (idx = table->Index; idx; idx = idx->Next)
				{
					if (idx->OnError == OE_Replace)
					{
						openAll = true;
						break;
					}
				}
			}
			for (i = 0, idx = table->Index; idx; idx = idx->Next, i++)
			{
				_assert(regIdxs);
				if (openAll || regIdxs[i] > 0)
				{
					KeyInfo *key = sqlite3IndexKeyinfo(parse, idx);
					v->AddOp4(OP_OpenWrite, curId+i+1, idx->Id, db, (char *)key, Vdbe::P4T_KEYINFO_HANDOFF);
					_assert(parse->Tabs > curId+i+1);
				}
			}
		}

		// Top of the update loop
		int addr = 0; // VDBE instruction address of the start of the loop
		if (okOnePass)
		{
			int a1 = v->AddOp1(OP_NotNull, regOldRowid);
			addr = v->AddOp0(OP_Goto);
			v->JumpHere(a1);
		}
		else
			addr = v->AddOp3(OP_RowSetRead, regRowSet, 0, regOldRowid);

		// Make cursor iCur point to the record that is being updated. If this record does not exist for some reason (deleted by a trigger,
		// for example, then jump to the next iteration of the RowSet loop.
		v->AddOp3(OP_NotExists, curId, addr, regOldRowid);

		// If the record number will change, set register regNewRowid to contain the new value. If the record number is not being modified,
		// then regNewRowid is the same register as regOldRowid, which is already populated.
		_assert(chngRowid || trigger || hasFK || regOldRowid == regNewRowid);
		if (chngRowid)
		{
			sqlite3ExprCode(parse, rowidExpr, regNewRowid);
			v->AddOp1(OP_MustBeInt, regNewRowid);
		}

		// If there are triggers on this table, populate an array of registers with the required old.* column data.
		if (hasFK || trigger)
		{
			uint32 oldmask = (hasFK ? sqlite3FkOldmask(parse, table) : 0);
			oldmask |= sqlite3TriggerColmask(parse, trigger, changes, 0, TRIGGER_BEFORE|TRIGGER_AFTER, table, onError);
			for (i = 0; i < table->Cols.length; i++)
			{
				if (xrefs[i] < 0 || oldmask == 0xffffffff || (i < 32 && (oldmask & (1<<i))))
					sqlite3ExprCodeGetColumnOfTable(v, table, curId, i, regOld+i);
				else
					v->AddOp2(v, OP_Null, 0, regOld+i);
			}
			if (!chngRowid)
				v->AddOp2(v, OP_Copy, regOldRowid, regNewRowid);
		}

		// Populate the array of registers beginning at regNew with the new row data. This array is used to check constaints, create the new
		// table and index records, and as the values for any new.* references made by triggers.
		//
		// If there are one or more BEFORE triggers, then do not populate the registers associated with columns that are (a) not modified by
		// this UPDATE statement and (b) not accessed by new.* references. The values for registers not modified by the UPDATE must be reloaded from 
		// the database after the BEFORE triggers are fired anyway (as the trigger may have modified them). So not loading those that are not going to
		// be used eliminates some redundant opcodes.
		int newmask = sqlite3TriggerColmask(parse, trigger, changes, 1, TRIGGER_BEFORE, table, onError); // Mask of NEW.* columns accessed by BEFORE triggers
		v->AddOp3(OP_Null, 0, regNew, regNew+table->nCol-1);
		for (i = 0; i < table->Cols.length; i++)
		{
			if (i == table->PKey)
			{
				//!v->AddOp2(OP_Null, 0, regNew+i);
			}
			else
			{
				j = xrefs[i];
				if (j >= 0)
					sqlite3ExprCode(parse, changes->Ids[j].Expr, regNew+i);
				else if ((tmask & TRIGGER_BEFORE) == 0 || i > 31 || (newmask & (1<<i)))
				{
					// This branch loads the value of a column that will not be changed into a register. This is done if there are no BEFORE triggers, or
					// if there are one or more BEFORE triggers that use this value via a new.* reference in a trigger program.
					ASSERTCOVERAGE(i == 31);
					ASSERTCOVERAGE(i == 32);
					v->AddOp3(OP_Column, curId, i, regNew+i);
					v->ColumnDefault(table, i, regNew+i);
				}
			}
		}

		// Fire any BEFORE UPDATE triggers. This happens before constraints are verified. One could argue that this is wrong.
		if (tmask & TRIGGER_BEFORE)
		{
			v->AddOp2(OP_Affinity, regNew, table->Cols.length);
			sqlite3TableAffinityStr(v, table);
			sqlite3CodeRowTrigger(parse, trigger, TK_UPDATE, changes, TRIGGER_BEFORE, table, regOldRowid, onError, addr);

			// The row-trigger may have deleted the row being updated. In this case, jump to the next row. No updates or AFTER triggers are 
			// required. This behavior - what happens when the row being updated is deleted or renamed by a BEFORE trigger - is left undefined in the documentation.
			v->AddOp3(OP_NotExists, curId, addr, regOldRowid);

			// If it did not delete it, the row-trigger may still have modified some of the columns of the row being updated. Load the values for 
			// all columns not modified by the update statement into their registers in case this has happened.
			for (i = 0; i < table->Cols.length; i++)
				if (xrefs[i] < 0 && i != table->PKey)
				{
					v->AddOp3(OP_Column, curId, i, regNew+i);
					v->ColumnDefault(table, i, regNew+i);
				}
		}

		if (!isView)
		{
			int j1; // Address of jump instruction

			// Do constraint checks.
			sqlite3GenerateConstraintChecks(parse, table, curId, regNewRowid, regIdxs, (chngRowid ? regOldRowid : 0), 1, onError, addr, 0);

			// Do FK constraint checks.
			if (hasFK)
				sqlite3FkCheck(parse, table, regOldRowid, 0);

			// Delete the index entries associated with the current record.
			j1 = v->AddOp3(OP_NotExists, curId, 0, regOldRowid);
			sqlite3GenerateRowIndexDelete(parse, table, curId, regIdxs);

			// If changing the record number, delete the old record.
			if (hasFK || chngRowid)
				v->AddOp2(OP_Delete, curId, 0);
			v->JumpHere(j1);

			if (hasFK)
				sqlite3FkCheck(parse, table, 0, regNewRowid);

			// Insert the new index entries and the new record.
			sqlite3CompleteInsertion(parse, table, curId, regNewRowid, regIdxs, 1, 0, 0);

			// Do any ON CASCADE, SET NULL or SET DEFAULT operations required to handle rows (possibly in other tables) that refer via a foreign key
			// to the row just updated.
			if (hasFK)
				sqlite3FkActions(parse, table, changes, regOldRowid);
		}

		// Increment the row counter 
		if ((ctx->Flags & Context::FLAG_CountRows) && !parse->TriggerTab)
			v->AddOp2(OP_AddImm, regRowCount, 1);

		sqlite3CodeRowTrigger(parse, trigger, TK_UPDATE, changes, TRIGGER_AFTER, table, regOldRowid, onError, addr);

		// Repeat the above with the next record to be updated, until all record selected by the WHERE clause have been updated.
		v->AddOp2(OP_Goto, 0, addr);
		v->JumpHere(addr);

		// Close all tables
		for (i = 0, idx = table->Index; idx; idx = idx->Next, i++)
		{
			_assert(regIdxs);
			if (openAll || regIdxs[i] > 0)
				v-?AddOp2(OP_Close, curId+i+1, 0);
		}
		v->AddOp2(OP_Close, curId, 0);

		// Update the sqlite_sequence table by storing the content of the maximum rowid counter values recorded while inserting into
		// autoincrement tables.
		if (parse->Nested == 0 && parse->TriggerTab == 0)
			sqlite3AutoincrementEnd(parse);

		/*
		** Return the number of rows that were changed. If this routine is 
		** generating code because of a call to sqlite3NestedParse(), do not
		** invoke the callback function.
		*/
		if( (ctx->flags&SQLITE_CountRows) && !pParse->pTriggerTab && !pParse->nested ){
			sqlite3VdbeAddOp2(v, OP_ResultRow, regRowCount, 1);
			sqlite3VdbeSetNumCols(v, 1);
			sqlite3VdbeSetColName(v, 0, COLNAME_NAME, "rows updated", SQLITE_STATIC);
		}

update_cleanup:
		sqlite3AuthContextPop(&sContext);
		sqlite3DbFree(ctx, aRegIdx);
		sqlite3DbFree(ctx, xrefs);
		sqlite3SrcListDelete(ctx, pTabList);
		sqlite3ExprListDelete(ctx, pChanges);
		sqlite3ExprDelete(ctx, pWhere);
		return;
	}
	/* Make sure "isView" and other macros defined above are undefined. Otherwise
	** thely may interfere with compilation of other functions in this file
	** (or in another file, if this file becomes part of the amalgamation).  */
#ifdef isView
#undef isView
#endif
#ifdef pTrigger
#undef pTrigger
#endif

#ifndef OMIT_VIRTUALTABLE
	static void UpdateVirtualTable(Parse *parse, SrcList *src, Table *table, ExprList *changes, Expr *rowid, int *xref, Expr *where, OE onError)
	{
		Vdbe *v = pParse->pVdbe;  // Virtual machine under construction
		ExprList *pEList = 0;     // The result set of the SELECT statement
		Select *pSelect = 0;      // The SELECT statement
		Expr *pExpr;              // Temporary expression
		int ephemTab;             // Table holding the result of the SELECT
		int i;                    // Loop counter
		int addr;                 // Address of top of loop
		int iReg;                 // First register in set passed to OP_VUpdate
		Context *ctx = pParse->Ctx; // Database connection
		const char *pVTab = (const char*)sqlite3GetVTable(ctx, table);
		SelectDest dest;

		/* Construct the SELECT statement that will find the new values for
		** all updated rows. 
		*/
		pEList = sqlite3ExprListAppend(pParse, 0, sqlite3Expr(ctx, TK_ID, "_rowid_"));
		if( pRowid ){
			pEList = sqlite3ExprListAppend(pParse, pEList,
				sqlite3ExprDup(ctx, pRowid, 0));
		}
		assert( table->iPKey<0 );
		for(i=0; i<table->nCol; i++){
			if( xrefs[i]>=0 ){
				pExpr = sqlite3ExprDup(ctx, pChanges->a[xrefs[i]].pExpr, 0);
			}else{
				pExpr = sqlite3Expr(ctx, TK_ID, table->aCol[i].zName);
			}
			pEList = sqlite3ExprListAppend(pParse, pEList, pExpr);
		}
		pSelect = sqlite3SelectNew(pParse, pEList, pSrc, pWhere, 0, 0, 0, 0, 0, 0);

		/* Create the ephemeral table into which the update results will
		** be stored.
		*/
		assert( v );
		ephemTab = pParse->nTab++;
		sqlite3VdbeAddOp2(v, OP_OpenEphemeral, ephemTab, table->nCol+1+(pRowid!=0));
		sqlite3VdbeChangeP5(v, BTREE_UNORDERED);

		/* fill the ephemeral table 
		*/
		sqlite3SelectDestInit(&dest, SRT_Table, ephemTab);
		sqlite3Select(pParse, pSelect, &dest);

		/* Generate code to scan the ephemeral table and call VUpdate. */
		iReg = ++pParse->nMem;
		pParse->nMem += table->nCol+1;
		addr = sqlite3VdbeAddOp2(v, OP_Rewind, ephemTab, 0);
		sqlite3VdbeAddOp3(v, OP_Column,  ephemTab, 0, iReg);
		sqlite3VdbeAddOp3(v, OP_Column, ephemTab, (pRowid?1:0), iReg+1);
		for(i=0; i<table->nCol; i++){
			sqlite3VdbeAddOp3(v, OP_Column, ephemTab, i+1+(pRowid!=0), iReg+2+i);
		}
		sqlite3VtabMakeWritable(pParse, table);
		sqlite3VdbeAddOp4(v, OP_VUpdate, 0, table->nCol+2, iReg, pVTab, P4_VTAB);
		sqlite3VdbeChangeP5(v, onError==OE_Default ? OE_Abort : onError);
		sqlite3MayAbort(pParse);
		sqlite3VdbeAddOp2(v, OP_Next, ephemTab, addr+1);
		sqlite3VdbeJumpHere(v, addr);
		sqlite3VdbeAddOp2(v, OP_Close, ephemTab, 0);

		/* Cleanup */
		sqlite3SelectDelete(ctx, pSelect);  
	}
#endif
} }