#pragma region OMIT_FOREIGN_KEY
#ifndef OMIT_FOREIGN_KEY
#include "Core+Syntax.cu.h"
#include "..\Core+Vbde\Vdbe.cu.h"

namespace Core
{
#ifndef OMIT_TRIGGER

	__device__ int Parse::FKLocateIndex(Table *parent, FKey *fkey, Index **indexOut, int **colsOut)
	{
		int colsLength = fkey->Cols.length; // Number of columns in parent key
		char *key = fkey->Cols[0].Col; // Name of left-most parent key column

		// The caller is responsible for zeroing output parameters.
		_assert(indexOut && *indexOut == nullptr);
		_assert(!colsOut || *colsOut == nullptr);

		// If this is a non-composite (single column) foreign key, check if it maps to the INTEGER PRIMARY KEY of table pParent. If so, leave *ppIdx 
		// and *paiCol set to zero and return early. 
		//
		// Otherwise, for a composite foreign key (more than one column), allocate space for the aiCol array (returned via output parameter *paiCol).
		// Non-composite foreign keys do not require the aiCol array.
		int *cols = nullptr; // Value to return via *paiCol
		if (colsLength == 1)
		{
			// The FK maps to the IPK if any of the following are true:
			//   1) There is an INTEGER PRIMARY KEY column and the FK is implicitly mapped to the primary key of table pParent, or
			//   2) The FK is explicitly mapped to a column declared as INTEGER PRIMARY KEY.
			if (parent->PKey >= 0)
				if (!key || !_strcmp(parent->Cols[parent->PKey].Name, key)) return 0;
		}
		else if (colsOut)
		{
			_assert(colsLength > 1);
			cols = (int *)SysEx::TagAlloc(Ctx, colsLength*sizeof(int));
			if (!cols) return 1;
			*colsOut = cols;
		}

		Index *index = nullptr; // Value to return via *ppIdx
		for (index = parent->Index; index; index = index->Next)
		{
			if (index->Columns.length == colsLength && index->OnError != OE_None)
			{ 
				// pIdx is a UNIQUE index (or a PRIMARY KEY) and has the right number of columns. If each indexed column corresponds to a foreign key
				// column of pFKey, then this index is a winner.  */
				if (!key)
				{
					// If zKey is NULL, then this foreign key is implicitly mapped to the PRIMARY KEY of table pParent. The PRIMARY KEY index may be 
					// identified by the test (Index.autoIndex==2).
					if (index->AutoIndex == 2)
					{
						if (cols)
							for (int i = 0; i < colsLength; i++) cols[i] = fkey->Cols[i].From;
						break;
					}
				}
				else
				{
					// If zKey is non-NULL, then this foreign key was declared to map to an explicit list of columns in table pParent. Check if this
					// index matches those columns. Also, check that the index uses the default collation sequences for each column.
					int i, j;
					for (i = 0; i < colsLength; i++)
					{
						int col = index->Columns[i]; // Index of column in parent tbl
						// If the index uses a collation sequence that is different from the default collation sequence for the column, this index is
						// unusable. Bail out early in this case.
						char *dfltColl = parent->Cols[col].Coll;// Def. collation for column
						if (!dfltColl)
							dfltColl = "BINARY";
						if (_strcmp(index->CollNames[i], dfltColl)) break;

						char *indexCol = parent->Cols[col].Name; // Name of indexed column
						for (j = 0; j < colsLength; j++)
						{
							if (!_strcmp(fkey->Cols[j].Col, indexCol))
							{
								if (cols) cols[i] = fkey->Cols[j].From;
								break;
							}
						}
						if (j == colsLength) break;
					}
					if (i == colsLength) break; // pIdx is usable
				}
			}
		}

		if (!index)
		{
			if (!DisableTriggers)
				ErrorMsg("foreign key mismatch - \"%w\" referencing \"%w\"", fkey->From->Name, fkey->To);
			SysEx::TagFree(Ctx, cols);
			return 1;
		}

		*indexOut = index;
		return 0;
	}

	__device__ static void FKLookupParent(Parse *parse, int iDb, Table *table, Index *index, FKey *fkey, int *cols, int regDataId, int incr, bool isIgnore)
	{
		Vdbe *v = parse->GetVdbe(); // Vdbe to add code to
		int curId = parse->Tabs - 1; // Cursor number to use
		int okId = v->MakeLabel(); // jump here if parent key found

		// If nIncr is less than zero, then check at runtime if there are any outstanding constraints to resolve. If there are not, there is no need
		// to check if deleting this row resolves any outstanding violations.
		//
		// Check if any of the key columns in the child table row are NULL. If any are, then the constraint is considered satisfied. No need to 
		// search for a matching row in the parent table.
		int i;
		if (incr < 0)
			v->AddOp2(OP_FkIfZero, fkey->IsDeferred, okId);
		for (i = 0; i < fkey->Cols.length; i++)
		{
			int regId = cols[i] + regDataId + 1;
			v->AddOp2(OP_IsNull, regId, okId);
		}

		if (!isIgnore)
		{
			if (!index)
			{
				// If pIdx is NULL, then the parent key is the INTEGER PRIMARY KEY column of the parent table (table pTab).
				int mustBeIntId; 
				int regTempId = parse->GetTempReg();

				// Invoke MustBeInt to coerce the child key value to an integer (i.e. apply the affinity of the parent key). If this fails, then there
				// is no matching parent key. Before using MustBeInt, make a copy of the value. Otherwise, the value inserted into the child key column
				// will have INTEGER affinity applied to it, which may not be correct.
				v->AddOp2(OP_SCopy, cols[0]+1+regDataId, regTempId);
				int mustBeIntId = v->AddOp2(OP_MustBeInt, regTempId, 0);// Address of MustBeInt instruction

				// If the parent table is the same as the child table, and we are about to increment the constraint-counter (i.e. this is an INSERT operation),
				// then check if the row being inserted matches itself. If so, do not increment the constraint-counter.
				if (table == fkey->From && incr == 1)
					v->AddOp3(OP_Eq, regDataId, okId, regTempId);

				parse->OpenTable(curId, db, table, OP_OpenRead);
				v->AddOp3(OP_NotExists, curId, 0, regTempId);
				v->AddOp2(OP_Goto, 0, okId);
				v->JumpHere(v->CurrentAddr()-2);
				v->JumpHere(mustBeIntId);
				parse->ReleaseTempReg(regTempId);
			}
			else
			{
				int colsLength = fkey->Cols.length;
				int regTempId = parse->GetTempRange(colsLength);
				int regRecId = parse->GetTempReg();
				KeyInfo *key = parse->IndexKeyinfo(index);

				v->AddOp3(OP_OpenRead, curId, index->Id, db);
				v->ChangeP4(-1, (char *)key, Vdbe::P4T_KEYINFO_HANDOFF);
				for (i = 0; i < colsLength; i++)
					v->AddOp2(OP_Copy, cols[i]+1+regDataId, regTempId+i);

				// If the parent table is the same as the child table, and we are about to increment the constraint-counter (i.e. this is an INSERT operation),
				// then check if the row being inserted matches itself. If so, do not increment the constraint-counter. 
				//
				// If any of the parent-key values are NULL, then the row cannot match itself. So set JUMPIFNULL to make sure we do the OP_Found if any
				// of the parent-key values are NULL (at this point it is known that none of the child key values are).
				if (table == fkey->From && incr == 1)
				{
					int jumpId = v->CurrentAddr() + colsLength + 1;
					for (i = 0; i < colsLength; i++)
					{
						int childId = cols[i]+1+regDataId;
						int parentId = index->Columns[i]+1+regDataId;
						_assert(cols[i] != table->PKey);
						if (index->Columns[i] == table->PKey)
							parentId = regDataId; // The parent key is a composite key that includes the IPK column
						v->AddOp3(OP_Ne, childId, jumpId, parentId);
						v->ChangeP5(SQLITE_JUMPIFNULL);
					}
					v->AddOp2(OP_Goto, 0, okId);
				}

				v->AddOp3(OP_MakeRecord, regTempId, colsLength, regRecId);
				v->ChangeP4(-1, v->IndexAffinityStr(index), Vdbe::P4T_TRANSIENT);
				v->AddOp4Int(OP_Found, curId, okId, regRecId, 0);

				parse->ReleaseTempReg(regRec);
				parse->ReleaseTempRange(regTempId, colsLength);
			}
		}

		if (!fkey->IsDeferred && !parse->Toplevel && !parse->IsMultiWrite)
		{
			// Special case: If this is an INSERT statement that will insert exactly one row into the table, raise a constraint immediately instead of
			// incrementing a counter. This is necessary as the VM code is being generated for will not open a statement transaction.
			_assert(incr == 1);
			HaltConstraint(parse, SQLITE_CONSTRAINT_FOREIGNKEY, OE_Abort, "foreign key constraint failed", Vdbe::P4T_STATIC);
		}
		else
		{
			if (incr > 0 && !fkey->IsDeferred)
				Parse_Toplevel(parse)->MayAbort = 1;
			v->AddOp2(OP_FkCounter, fkey->IsDeferred, incr);
		}

		v->ResolveLabel(okId);
		v->AddOp1(OP_Close, curId);
	}

	__device__ static void FKScanChildren(Parse *parse, SrcList *src, Table *table, Index *index, FKey *fkey, int *cols, int regDataId, int incr)
	{
		Context *ctx = parse->Ctx; // Database handle
		Vdbe *v = parse->GetVdbe();
		_assert(!index || index->Table == table);
		int iFkIfZero = 0; // Address of OP_FkIfZero
		if (incr < 0)
			iFkIfZero = v->AddOp2(OP_FkIfZero, fkey->IsDeferred, 0);

		// Create an Expr object representing an SQL expression like:
		//
		//   <parent-key1> = <child-key1> AND <parent-key2> = <child-key2> ...
		//
		// The collation sequence used for the comparison should be that of the parent key columns. The affinity of the parent key column should
		// be applied to each child key value before the comparison takes place.
		Expr *where_ = nullptr; // WHERE clause to scan with
		for (int i = 0; i < fkey->Cols.length; i++)
		{
			int col; // Index of column in child table
			Expr *left = Expr::Expr(ctx, TK_REGISTER, 0); // Value from parent table row
			if (left)
			{
				// Set the collation sequence and affinity of the LHS of each TK_EQ expression to the parent key column defaults.
				if (index)
				{
					col = index->Columns[i];
					Column *colObj = &table->Cols[col];
					if (table->PKey == col) col = -1;
					left->TableIdx = regDataId + col + 1;
					left->Aff = colObj->Affinity;
					const char *collName = colObj->Coll;
					if (collName == 0) collName = ctx->DefaultColl->Name;
					left = Expr::AddCollateString(parse, left, collName);
				}
				else
				{
					left->TableIdx = regDataId;
					left->Affinity = AFF_INTEGER;
				}
			}
			col = (cols ? cols[i] : fkey->Cols[0].From);
			_assert(col >= 0);
			const char *colName = fkey->From->Cols[col].Name; // Name of column in child table
			Expr *right = Expr::Expr(ctx, TK_ID, colName); // Column ref to child table
			Expr *eq = Expr::PExpr(parse, TK_EQ, left, right, 0); // Expression (left = right)
			where_ = Expr::And(ctx, where_, eq);
		}

		// If the child table is the same as the parent table, and this scan is taking place as part of a DELETE operation (operation D.2), omit the
		// row being deleted from the scan by adding ($rowid != rowid) to the WHERE clause, where $rowid is the rowid of the row being deleted.
		if (table == fkey->From && incr > 0)
		{
			Expr *left = Expr::Expr(ctx, TK_REGISTER, 0); // Value from parent table row
			Expr *right = Expr::Expr(ctx, TK_COLUMN, 0); // Column ref to child table
			if (left && right)
			{
				left->TableIdx = regDataId;
				left->Affinity = AFF_INTEGER;
				right->TableIdx = src->Ids[0].Cursor;
				right->ColumnIdx= -1;
			}
			Expr *eq = Expr::PExpr(parse, TK_NE, left, right, 0); // Expression (left = right)
			where_ = Expr::And(ctx, where_, eq);
		}

		// Resolve the references in the WHERE clause.
		NameContext nameContext; // Context used to resolve WHERE clause
		_memset(&nameContext, 0, sizeof(NameContext));
		nameContext.SrcList = src;
		nameContext.Parse = parse;
		ResolveExprNames(&nameContext, where_);

		// Create VDBE to loop through the entries in src that match the WHERE clause. If the constraint is not deferred, throw an exception for
		// each row found. Otherwise, for deferred constraints, increment the deferred constraint counter by incr for each row selected.
		WhereInfo *whereInfo = Where::Begin(parse, src, where_, 0, 0, 0, 0);  // Context used by sqlite3WhereXXX()
		if (incr > 0 && !fkey->IsDeferred)
			Parse_Toplevel(parse)->MayAbort = true;
		v->AddOp2(OP_FkCounter, fkey->IsDeferred, incr);
		if (whereInfo)
			Where::End(whereInfo);

		// Clean up the WHERE clause constructed above.
		Expr::Delete(ctx, where_);
		if (fkIfZero)
			v->JumpHere(fkIfZero);
	}

	__device__ FKey *Parse::FkReferences(Table *table)
	{
		int nameLength = _strlen30(table->Name);
		return (FKey *)table->Schema->FKeyHash.Find(table->Name, nameLength);
	}

	__device__ static void FKTriggerDelete(Context *ctx, Trigger *p)
	{
		if (p)
		{
			TriggerStep *step = p->StepList;
			Expr::Delete(ctx, step->Where);
			Expr::ListDelete(ctx, step->ExprList);
			Select::Delete(ctx, step->Select);
			Expr::Delete(ctx, p->When);
			SysEx::TagFree(ctx, p);
		}
	}

	__device__ void Parse::FKDropTable(Parse *pParse, SrcList *pName, Table *pTab)
	{
		sqlite3 *db = pParse->db;
		if( (db->flags&SQLITE_ForeignKeys) && !IsVirtual(pTab) && !pTab->pSelect ){
			int iSkip = 0;
			Vdbe *v = sqlite3GetVdbe(pParse);

			assert( v );                  /* VDBE has already been allocated */
			if( sqlite3FkReferences(pTab)==0 ){
				/* Search for a deferred foreign key constraint for which this table
				** is the child table. If one cannot be found, return without 
				** generating any VDBE code. If one can be found, then jump over
				** the entire DELETE if there are no outstanding deferred constraints
				** when this statement is run.  */
				FKey *p;
				for(p=pTab->pFKey; p; p=p->pNextFrom){
					if( p->isDeferred ) break;
				}
				if( !p ) return;
				iSkip = sqlite3VdbeMakeLabel(v);
				sqlite3VdbeAddOp2(v, OP_FkIfZero, 1, iSkip);
			}

			pParse->disableTriggers = 1;
			sqlite3DeleteFrom(pParse, sqlite3SrcListDup(db, pName, 0), 0);
			pParse->disableTriggers = 0;

			/* If the DELETE has generated immediate foreign key constraint 
			** violations, halt the VDBE and return an error at this point, before
			** any modifications to the schema are made. This is because statement
			** transactions are not able to rollback schema changes.  */
			sqlite3VdbeAddOp2(v, OP_FkIfZero, 0, sqlite3VdbeCurrentAddr(v)+2);
			sqlite3HaltConstraint(pParse, SQLITE_CONSTRAINT_FOREIGNKEY,
				OE_Abort, "foreign key constraint failed", P4_STATIC
				);

			if( iSkip ){
				sqlite3VdbeResolveLabel(v, iSkip);
			}
		}
	}

	__device__ void Parse::FKCheck(Parse *pParse, Table *pTab, int regOld, int regNew)
	{
		sqlite3 *db = pParse->db;       /* Database handle */
		FKey *pFKey;                    /* Used to iterate through FKs */
		int iDb;                        /* Index of database containing pTab */
		const char *zDb;                /* Name of database containing pTab */
		int isIgnoreErrors = pParse->disableTriggers;

		/* Exactly one of regOld and regNew should be non-zero. */
		assert( (regOld==0)!=(regNew==0) );

		/* If foreign-keys are disabled, this function is a no-op. */
		if( (db->flags&SQLITE_ForeignKeys)==0 ) return;

		iDb = sqlite3SchemaToIndex(db, pTab->pSchema);
		zDb = db->aDb[iDb].zName;

		/* Loop through all the foreign key constraints for which pTab is the
		** child table (the table that the foreign key definition is part of).  */
		for(pFKey=pTab->pFKey; pFKey; pFKey=pFKey->pNextFrom){
			Table *pTo;                   /* Parent table of foreign key pFKey */
			Index *pIdx = 0;              /* Index on key columns in pTo */
			int *aiFree = 0;
			int *aiCol;
			int iCol;
			int i;
			int isIgnore = 0;

			/* Find the parent table of this foreign key. Also find a unique index 
			** on the parent key columns in the parent table. If either of these 
			** schema items cannot be located, set an error in pParse and return 
			** early.  */
			if( pParse->disableTriggers ){
				pTo = sqlite3FindTable(db, pFKey->zTo, zDb);
			}else{
				pTo = sqlite3LocateTable(pParse, 0, pFKey->zTo, zDb);
			}
			if( !pTo || sqlite3FkLocateIndex(pParse, pTo, pFKey, &pIdx, &aiFree) ){
				assert( isIgnoreErrors==0 || (regOld!=0 && regNew==0) );
				if( !isIgnoreErrors || db->mallocFailed ) return;
				if( pTo==0 ){
					/* If isIgnoreErrors is true, then a table is being dropped. In this
					** case SQLite runs a "DELETE FROM xxx" on the table being dropped
					** before actually dropping it in order to check FK constraints.
					** If the parent table of an FK constraint on the current table is
					** missing, behave as if it is empty. i.e. decrement the relevant
					** FK counter for each row of the current table with non-NULL keys.
					*/
					Vdbe *v = sqlite3GetVdbe(pParse);
					int iJump = sqlite3VdbeCurrentAddr(v) + pFKey->nCol + 1;
					for(i=0; i<pFKey->nCol; i++){
						int iReg = pFKey->aCol[i].iFrom + regOld + 1;
						sqlite3VdbeAddOp2(v, OP_IsNull, iReg, iJump);
					}
					sqlite3VdbeAddOp2(v, OP_FkCounter, pFKey->isDeferred, -1);
				}
				continue;
			}
			assert( pFKey->nCol==1 || (aiFree && pIdx) );

			if( aiFree ){
				aiCol = aiFree;
			}else{
				iCol = pFKey->aCol[0].iFrom;
				aiCol = &iCol;
			}
			for(i=0; i<pFKey->nCol; i++){
				if( aiCol[i]==pTab->iPKey ){
					aiCol[i] = -1;
				}
#ifndef OMIT_AUTHORIZATION
				// Request permission to read the parent key columns. If the authorization callback returns SQLITE_IGNORE, behave as if any
				// values read from the parent table are NULL.
				if (db->xAuth)
				{
					int rcauth;
					char *zCol = pTo->aCol[pIdx ? pIdx->aiColumn[i] : pTo->iPKey].zName;
					rcauth = sqlite3AuthReadCol(pParse, pTo->zName, zCol, iDb);
					isIgnore = (rcauth==SQLITE_IGNORE);
				}
#endif
			}

			/* Take a shared-cache advisory read-lock on the parent table. Allocate 
			** a cursor to use to search the unique index on the parent key columns 
			** in the parent table.  */
			sqlite3TableLock(pParse, iDb, pTo->tnum, 0, pTo->zName);
			pParse->nTab++;

			if( regOld!=0 ){
				/* A row is being removed from the child table. Search for the parent.
				** If the parent does not exist, removing the child row resolves an 
				** outstanding foreign key constraint violation. */
				fkLookupParent(pParse, iDb, pTo, pIdx, pFKey, aiCol, regOld, -1,isIgnore);
			}
			if( regNew!=0 ){
				/* A row is being added to the child table. If a parent row cannot
				** be found, adding the child row has violated the FK constraint. */ 
				fkLookupParent(pParse, iDb, pTo, pIdx, pFKey, aiCol, regNew, +1,isIgnore);
			}

			sqlite3DbFree(db, aiFree);
		}

		/* Loop through all the foreign key constraints that refer to this table */
		for(pFKey = sqlite3FkReferences(pTab); pFKey; pFKey=pFKey->pNextTo){
			Index *pIdx = 0;              /* Foreign key index for pFKey */
			SrcList *pSrc;
			int *aiCol = 0;

			if( !pFKey->isDeferred && !pParse->pToplevel && !pParse->isMultiWrite ){
				assert( regOld==0 && regNew!=0 );
				/* Inserting a single row into a parent table cannot cause an immediate
				** foreign key violation. So do nothing in this case.  */
				continue;
			}

			if( sqlite3FkLocateIndex(pParse, pTab, pFKey, &pIdx, &aiCol) ){
				if( !isIgnoreErrors || db->mallocFailed ) return;
				continue;
			}
			assert( aiCol || pFKey->nCol==1 );

			/* Create a SrcList structure containing a single table (the table 
			** the foreign key that refers to this table is attached to). This
			** is required for the sqlite3WhereXXX() interface.  */
			pSrc = sqlite3SrcListAppend(db, 0, 0, 0);
			if( pSrc ){
				struct SrcList_item *pItem = pSrc->a;
				pItem->pTab = pFKey->pFrom;
				pItem->zName = pFKey->pFrom->zName;
				pItem->pTab->nRef++;
				pItem->iCursor = pParse->nTab++;

				if( regNew!=0 ){
					fkScanChildren(pParse, pSrc, pTab, pIdx, pFKey, aiCol, regNew, -1);
				}
				if( regOld!=0 ){
					/* If there is a RESTRICT action configured for the current operation
					** on the parent table of this FK, then throw an exception 
					** immediately if the FK constraint is violated, even if this is a
					** deferred trigger. That's what RESTRICT means. To defer checking
					** the constraint, the FK should specify NO ACTION (represented
					** using OE_None). NO ACTION is the default.  */
					fkScanChildren(pParse, pSrc, pTab, pIdx, pFKey, aiCol, regOld, 1);
				}
				pItem->zName = 0;
				sqlite3SrcListDelete(db, pSrc);
			}
			sqlite3DbFree(db, aiCol);
		}
	}

#define COLUMN_MASK(x) (((x)>31) ? 0xffffffff : ((u32)1<<(x)))

	__device__ uint32 Parse::FKOldmask(Parse *pParse, Table *pTab)
	{
		u32 mask = 0;
		if( pParse->db->flags&SQLITE_ForeignKeys ){
			FKey *p;
			int i;
			for(p=pTab->pFKey; p; p=p->pNextFrom){
				for(i=0; i<p->nCol; i++) mask |= COLUMN_MASK(p->aCol[i].iFrom);
			}
			for(p=sqlite3FkReferences(pTab); p; p=p->pNextTo){
				Index *pIdx = 0;
				sqlite3FkLocateIndex(pParse, pTab, p, &pIdx, 0);
				if( pIdx ){
					for(i=0; i<pIdx->nColumn; i++) mask |= COLUMN_MASK(pIdx->aiColumn[i]);
				}
			}
		}
		return mask;
	}

	__device__ int Parse::FKRequired(Parse *pParse, Table *pTab, int *aChange, int chngRowid)
	{
		if( pParse->db->flags&SQLITE_ForeignKeys ){
			if( !aChange ){
				/* A DELETE operation. Foreign key processing is required if the 
				** table in question is either the child or parent table for any 
				** foreign key constraint.  */
				return (sqlite3FkReferences(pTab) || pTab->pFKey);
			}else{
				/* This is an UPDATE. Foreign key processing is only required if the
				** operation modifies one or more child or parent key columns. */
				int i;
				FKey *p;

				/* Check if any child key columns are being modified. */
				for(p=pTab->pFKey; p; p=p->pNextFrom){
					for(i=0; i<p->nCol; i++){
						int iChildKey = p->aCol[i].iFrom;
						if( aChange[iChildKey]>=0 ) return 1;
						if( iChildKey==pTab->iPKey && chngRowid ) return 1;
					}
				}

				/* Check if any parent key columns are being modified. */
				for(p=sqlite3FkReferences(pTab); p; p=p->pNextTo){
					for(i=0; i<p->nCol; i++){
						char *zKey = p->aCol[i].zCol;
						int iKey;
						for(iKey=0; iKey<pTab->nCol; iKey++){
							Column *pCol = &pTab->aCol[iKey];
							if( (zKey ? !sqlite3StrICmp(pCol->zName, zKey)
								: (pCol->colFlags & COLFLAG_PRIMKEY)!=0) ){
									if( aChange[iKey]>=0 ) return 1;
									if( iKey==pTab->iPKey && chngRowid ) return 1;
							}
						}
					}
				}
			}
		}
		return 0;
	}

	__device__ static Trigger *FKActionTrigger(Parse *pParse, Table *pTab, FKey *pFKey, ExprList *pChanges)
	{
		sqlite3 *db = pParse->db;       /* Database handle */
		int action;                     /* One of OE_None, OE_Cascade etc. */
		Trigger *pTrigger;              /* Trigger definition to return */
		int iAction = (pChanges!=0);    /* 1 for UPDATE, 0 for DELETE */

		action = pFKey->aAction[iAction];
		pTrigger = pFKey->apTrigger[iAction];

		if( action!=OE_None && !pTrigger ){
			u8 enableLookaside;           /* Copy of db->lookaside.bEnabled */
			char const *zFrom;            /* Name of child table */
			int nFrom;                    /* Length in bytes of zFrom */
			Index *pIdx = 0;              /* Parent key index for this FK */
			int *aiCol = 0;               /* child table cols -> parent key cols */
			TriggerStep *pStep = 0;        /* First (only) step of trigger program */
			Expr *pWhere = 0;             /* WHERE clause of trigger step */
			ExprList *pList = 0;          /* Changes list if ON UPDATE CASCADE */
			Select *pSelect = 0;          /* If RESTRICT, "SELECT RAISE(...)" */
			int i;                        /* Iterator variable */
			Expr *pWhen = 0;              /* WHEN clause for the trigger */

			if( sqlite3FkLocateIndex(pParse, pTab, pFKey, &pIdx, &aiCol) ) return 0;
			assert( aiCol || pFKey->nCol==1 );

			for(i=0; i<pFKey->nCol; i++){
				Token tOld = { "old", 3 };  /* Literal "old" token */
				Token tNew = { "new", 3 };  /* Literal "new" token */
				Token tFromCol;             /* Name of column in child table */
				Token tToCol;               /* Name of column in parent table */
				int iFromCol;               /* Idx of column in child table */
				Expr *pEq;                  /* tFromCol = OLD.tToCol */

				iFromCol = aiCol ? aiCol[i] : pFKey->aCol[0].iFrom;
				assert( iFromCol>=0 );
				tToCol.z = pIdx ? pTab->aCol[pIdx->aiColumn[i]].zName : "oid";
				tFromCol.z = pFKey->pFrom->aCol[iFromCol].zName;

				tToCol.n = sqlite3Strlen30(tToCol.z);
				tFromCol.n = sqlite3Strlen30(tFromCol.z);

				/* Create the expression "OLD.zToCol = zFromCol". It is important
				** that the "OLD.zToCol" term is on the LHS of the = operator, so
				** that the affinity and collation sequence associated with the
				** parent table are used for the comparison. */
				pEq = sqlite3PExpr(pParse, TK_EQ,
					sqlite3PExpr(pParse, TK_DOT, 
					sqlite3PExpr(pParse, TK_ID, 0, 0, &tOld),
					sqlite3PExpr(pParse, TK_ID, 0, 0, &tToCol)
					, 0),
					sqlite3PExpr(pParse, TK_ID, 0, 0, &tFromCol)
					, 0);
				pWhere = sqlite3ExprAnd(db, pWhere, pEq);

				/* For ON UPDATE, construct the next term of the WHEN clause.
				** The final WHEN clause will be like this:
				**
				**    WHEN NOT(old.col1 IS new.col1 AND ... AND old.colN IS new.colN)
				*/
				if( pChanges ){
					pEq = sqlite3PExpr(pParse, TK_IS,
						sqlite3PExpr(pParse, TK_DOT, 
						sqlite3PExpr(pParse, TK_ID, 0, 0, &tOld),
						sqlite3PExpr(pParse, TK_ID, 0, 0, &tToCol),
						0),
						sqlite3PExpr(pParse, TK_DOT, 
						sqlite3PExpr(pParse, TK_ID, 0, 0, &tNew),
						sqlite3PExpr(pParse, TK_ID, 0, 0, &tToCol),
						0),
						0);
					pWhen = sqlite3ExprAnd(db, pWhen, pEq);
				}

				if( action!=OE_Restrict && (action!=OE_Cascade || pChanges) ){
					Expr *pNew;
					if( action==OE_Cascade ){
						pNew = sqlite3PExpr(pParse, TK_DOT, 
							sqlite3PExpr(pParse, TK_ID, 0, 0, &tNew),
							sqlite3PExpr(pParse, TK_ID, 0, 0, &tToCol)
							, 0);
					}else if( action==OE_SetDflt ){
						Expr *pDflt = pFKey->pFrom->aCol[iFromCol].pDflt;
						if( pDflt ){
							pNew = sqlite3ExprDup(db, pDflt, 0);
						}else{
							pNew = sqlite3PExpr(pParse, TK_NULL, 0, 0, 0);
						}
					}else{
						pNew = sqlite3PExpr(pParse, TK_NULL, 0, 0, 0);
					}
					pList = sqlite3ExprListAppend(pParse, pList, pNew);
					sqlite3ExprListSetName(pParse, pList, &tFromCol, 0);
				}
			}
			sqlite3DbFree(db, aiCol);

			zFrom = pFKey->pFrom->zName;
			nFrom = sqlite3Strlen30(zFrom);

			if( action==OE_Restrict ){
				Token tFrom;
				Expr *pRaise; 

				tFrom.z = zFrom;
				tFrom.n = nFrom;
				pRaise = sqlite3Expr(db, TK_RAISE, "foreign key constraint failed");
				if( pRaise ){
					pRaise->affinity = OE_Abort;
				}
				pSelect = sqlite3SelectNew(pParse, 
					sqlite3ExprListAppend(pParse, 0, pRaise),
					sqlite3SrcListAppend(db, 0, &tFrom, 0),
					pWhere,
					0, 0, 0, 0, 0, 0
					);
				pWhere = 0;
			}

			/* Disable lookaside memory allocation */
			enableLookaside = db->lookaside.bEnabled;
			db->lookaside.bEnabled = 0;

			pTrigger = (Trigger *)sqlite3DbMallocZero(db, 
				sizeof(Trigger) +         /* struct Trigger */
				sizeof(TriggerStep) +     /* Single step in trigger program */
				nFrom + 1                 /* Space for pStep->target.z */
				);
			if( pTrigger ){
				pStep = pTrigger->step_list = (TriggerStep *)&pTrigger[1];
				pStep->target.z = (char *)&pStep[1];
				pStep->target.n = nFrom;
				memcpy((char *)pStep->target.z, zFrom, nFrom);

				pStep->pWhere = sqlite3ExprDup(db, pWhere, EXPRDUP_REDUCE);
				pStep->pExprList = sqlite3ExprListDup(db, pList, EXPRDUP_REDUCE);
				pStep->pSelect = sqlite3SelectDup(db, pSelect, EXPRDUP_REDUCE);
				if( pWhen ){
					pWhen = sqlite3PExpr(pParse, TK_NOT, pWhen, 0, 0);
					pTrigger->pWhen = sqlite3ExprDup(db, pWhen, EXPRDUP_REDUCE);
				}
			}

			/* Re-enable the lookaside buffer, if it was disabled earlier. */
			db->lookaside.bEnabled = enableLookaside;

			sqlite3ExprDelete(db, pWhere);
			sqlite3ExprDelete(db, pWhen);
			sqlite3ExprListDelete(db, pList);
			sqlite3SelectDelete(db, pSelect);
			if( db->mallocFailed==1 ){
				fkTriggerDelete(db, pTrigger);
				return 0;
			}
			assert( pStep!=0 );

			switch( action ){
			case OE_Restrict:
				pStep->op = TK_SELECT; 
				break;
			case OE_Cascade: 
				if( !pChanges ){ 
					pStep->op = TK_DELETE; 
					break; 
				}
			default:
				pStep->op = TK_UPDATE;
			}
			pStep->pTrig = pTrigger;
			pTrigger->pSchema = pTab->pSchema;
			pTrigger->pTabSchema = pTab->pSchema;
			pFKey->apTrigger[iAction] = pTrigger;
			pTrigger->op = (pChanges ? TK_UPDATE : TK_DELETE);
		}

		return pTrigger;
	}

	__device__ void Parse::FKActions(Parse *pParse, Table *pTab, ExprList *pChanges, int regOld)
	{
		/* If foreign-key support is enabled, iterate through all FKs that 
		** refer to table pTab. If there is an action associated with the FK 
		** for this operation (either update or delete), invoke the associated 
		** trigger sub-program.  */
		if( pParse->db->flags&SQLITE_ForeignKeys ){
			FKey *pFKey;                  /* Iterator variable */
			for(pFKey = sqlite3FkReferences(pTab); pFKey; pFKey=pFKey->pNextTo){
				Trigger *pAction = fkActionTrigger(pParse, pTab, pFKey, pChanges);
				if( pAction ){
					sqlite3CodeRowTriggerDirect(pParse, pAction, pTab, regOld, OE_Abort, 0);
				}
			}
		}
	}

#endif

	__device__ void Parse::FKDelete(sqlite3 *db, Table *pTab)
	{
		FKey *pFKey;                    /* Iterator variable */
		FKey *pNext;                    /* Copy of pFKey->pNextFrom */

		assert( db==0 || sqlite3SchemaMutexHeld(db, 0, pTab->pSchema) );
		for(pFKey=pTab->pFKey; pFKey; pFKey=pNext){

			/* Remove the FK from the fkeyHash hash table. */
			if( !db || db->pnBytesFreed==0 ){
				if( pFKey->pPrevTo ){
					pFKey->pPrevTo->pNextTo = pFKey->pNextTo;
				}else{
					void *p = (void *)pFKey->pNextTo;
					const char *z = (p ? pFKey->pNextTo->zTo : pFKey->zTo);
					sqlite3HashInsert(&pTab->pSchema->fkeyHash, z, sqlite3Strlen30(z), p);
				}
				if( pFKey->pNextTo ){
					pFKey->pNextTo->pPrevTo = pFKey->pPrevTo;
				}
			}

			/* EV: R-30323-21917 Each foreign key constraint in SQLite is
			** classified as either immediate or deferred.
			*/
			assert( pFKey->isDeferred==0 || pFKey->isDeferred==1 );

			/* Delete any triggers created to implement actions for this FK. */
#ifndef SQLITE_OMIT_TRIGGER
			fkTriggerDelete(db, pFKey->apTrigger[0]);
			fkTriggerDelete(db, pFKey->apTrigger[1]);
#endif

			pNext = pFKey->pNextFrom;
			sqlite3DbFree(db, pFKey);
		}
	}

}
#endif
#pragma endregion
