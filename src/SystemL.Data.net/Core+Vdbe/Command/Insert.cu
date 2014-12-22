// insert.c
#include "..\Core+Vdbe.cu.h"
#include "..\VdbeInt.h"

namespace Core { namespace Command
{
	__device__ void Insert::OpenTable(Parse *p, int cur, int db, Table *table, OP opcode)
	{
		_assert(!IsVirtual(table));
		Vdbe *v = p->GetVdbe();
		_assert(opcode == OP_OpenWrite || opcode == OP_OpenRead);
		sqlite3TableLock(p, db, table->Id, (opcode == OP_OpenWrite ? 1 : 0), table->Name);
		v->AddOp3(opcode, cur, table->Id, db);
		v->ChangeP4(-1, INT_TO_PTR(table->Cols.length), Vdbe::P4T_INT32);
		v->Comment("%s", table->Name);
	}

	__device__ const char *Insert::IndexAffinityStr(Vdbe *v, Index *index)
	{
		if (!index->ColAff)
		{
			// The first time a column affinity string for a particular index is required, it is allocated and populated here. It is then stored as
			// a member of the Index structure for subsequent use.
			//
			// The column affinity string will eventually be deleted by sqliteDeleteIndex() when the Index structure itself is cleaned up.
			Table *table = index->Table;
			Context *ctx = v->Ctx;
			index->ColAff = (char *)_tagalloc(nullptr, index->Columns.length+2);
			if (!index->ColAff)
			{
				ctx->MallocFailed = true;
				return nullptr;
			}
			int n;
			for (n = 0; n < index->Columns.length; n++)
				index->ColAff[n] = table->Cols[index->Columns[n]].Affinity;
			index->ColAff[n++] = AFF_INTEGER;
			index->ColAff[n] = 0;
		}
		return index->ColAff;
	}

	__device__ void Insert::TableAffinityStr(Vdbe *v, Table *table)
	{
		// The first time a column affinity string for a particular table is required, it is allocated and populated here. It is then 
		// stored as a member of the Table structure for subsequent use.
		//
		// The column affinity string will eventually be deleted by sqlite3DeleteTable() when the Table structure itself is cleaned up.
		if (!table->ColAff)
		{
			Context *ctx = v->Ctx;
			char *colAff = (char *)_tagalloc(0, table->Cols.length+1);
			if (!colAff)
			{
				ctx->MallocFailed = true;
				return;
			}
			for (int i = 0; i < table->Cols.length; i++)
				colAff[i] = table->Cols[i].Affinity;
			colAff[table->Cols.length] = '\0';
			table->ColAff = colAff;
		}
		v->ChangeP4(-1, table->ColAff, Vdbe::P4T_TRANSIENT);
	}

	__device__ static bool ReadsTable(Parse *p, int startAddr, int db, Table *table)
	{
		Vdbe *v = p->GetVdbe();
		int end = v->CurrentAddr();
#ifndef OMIT_VIRTUALTABLE
		VTable *vtable = (IsVirtual(table) ? VTable::GetVTable(p->Ctx, table) : nullptr);
#endif
		for (int i = startAddr; i < end; i++)
		{
			Vdbe::VdbeOp *op = v->GetOp(i);
			_assert(op != nullptr);
			if (op->Opcode == OP_OpenRead && op->P3 == db)
			{
				int id = op->P2;
				if (id == table->Id)
					return true;
				for (Index *index = table->Index; index; index = index->Next)
					if (id == index->Id)
						return true;
			}
#ifndef OMIT_VIRTUALTABLE
			if (op->Opcode == OP_VOpen && op->P4.VTable == vtable)
			{
				_assert(op->P4.VTable != nullptr);
				_assert(op->P4Type == Vdbe::P4T_VTAB);
				return true;
			}
#endif
		}
		return false;
	}

#ifndef OMIT_AUTOINCREMENT
	__device__ static int AutoIncBegin(Parse *parse, int db, Table *table)
	{
		int memId = 0; // Register holding maximum rowid
		if (table->TabFlags & TF_Autoincrement)
		{
			Parse *toplevel = Parse::Toplevel(parse->Toplevel);
			AutoincInfo *info = toplevel->Ainc;
			while (info && info->Table  != table) info = info->Next;
			if (!info)
			{
				info = (AutoincInfo *)_tagalloc(parse->Ctx, sizeof(*info));
				if (!info) return 0;
				info->Next = toplevel->Ainc;
				toplevel->Ainc = info;
				info->Table = table;
				info->DB = db;
				toplevel->Mems++;					// Register to hold name of table
				info->RegCtr = ++toplevel->Mems;	// Max rowid register
				toplevel->Mems++;					// Rowid in sqlite_sequence
			}
			memId = info->RegCtr;
		}
		return memId;
	}

	__device__ void Insert::AutoincrementBegin(Parse *parse)
	{
		Context *ctx = parse->Ctx; // The database connection
		Vdbe *v = parse->V; // VDBE under construction

		// This routine is never called during trigger-generation.  It is only called from the top-level
		_assert(parse->TriggerTab == nullptr);
		_assert(parse == Parse::Toplevel(parse));

		_assert(v); // We failed long ago if this is not so
		for (AutoincInfo *p = parse->Ainc; p; p = p->Next) // Information about an AUTOINCREMENT
		{
			Context::DB *dbAsObj = &ctx->DBs[p->DB];  // Database only autoinc table
			int memId = p->RegCtr; // Register holding max rowid
			_assert(Btree::SchemaMutexHeld(ctx, 0, dbAsObj->Schema));
			OpenTable(parse, 0, p->DB, dbAsObj->Schema->SeqTable, OP_OpenRead);
			v->AddOp3(OP_Null, 0, memId, memId+1);
			int addr = v->CurrentAddr(); // A VDBE address
			v->AddOp4(OP_String8, 0, memId-1, 0, p->Table->Name, 0);
			v->AddOp2(OP_Rewind, 0, addr+9);
			v->AddOp3(OP_Column, 0, 0, memId);
			v->AddOp3(OP_Ne, memId-1, addr+7, memId);
			v->ChangeP5(SQLITE_JUMPIFNULL);
			v->AddOp2(OP_Rowid, 0, memId+1);
			v->AddOp3(OP_Column, 0, 1, memId);
			v->AddOp2(OP_Goto, 0, addr+9);
			v->AddOp2(OP_Next, 0, addr+2);
			v->AddOp2(OP_Integer, 0, memId);
			v->AddOp0(OP_Close);
		}
	}

	__device__ static void AutoIncStep(Parse *parse, int memId, int regRowid)
	{
		if (memId > 0)
			parse->V->AddOp2(OP_MemMax, memId, regRowid);
	}

	__device__ void Insert::AutoincrementEnd(Parse *parse)
	{
		Vdbe *v = parse->V;
		Context *ctx = parse->Ctx;

		_assert(v);
		for (AutoincInfo *p = parse->Ainc; p; p = p->Next)
		{
			Context::DB *dbAsObj = &ctx->DBs[p->DB];
			int memId = p->RegCtr;

			int rec = Expr::GetTempReg(parse);
			_assert(Btree::SchemaMutexHeld(ctx, 0, dbAsObj->Schema));
			OpenTable(parse, 0, p->DB, dbAsObj->Schema->SeqTable, OP_OpenWrite);
			int j1 = v->AddOp1(OP_NotNull, memId+1);
			int j2 = v->AddOp0(OP_Rewind);
			int j3 = v->AddOp3(OP_Column, 0, 0, rec);
			int j4 = v->AddOp3(OP_Eq, memId-1, 0, rec);
			v->AddOp2(OP_Next, 0, j3);
			v->JumpHere(j2);
			v->AddOp2(OP_NewRowid, 0, memId+1);
			int j5 = v->AddOp0(OP_Goto);
			v->JumpHere(j4);
			v->AddOp2(OP_Rowid, 0, memId+1);
			v->JumpHere(j1);
			v->JumpHere(j5);
			v->AddOp3(OP_MakeRecord, memId-1, 2, rec);
			v->AddOp3(OP_Insert, 0, rec, memId+1);
			v->ChangeP5(Vdbe::OPFLAG_APPEND);
			v->AddOp0(OP_Close);
			Expr::ReleaseTempReg(parse, rec);
		}
	}
#else
#define AutoIncBegin(A,B,C) (0)
#define AutoIncStep(A,B,C)
#endif

	__device__ RC Insert::CodeCoroutine(Parse *parse, Select *select, SelectDest *dest)
	{
		int regYield = ++parse->Mems; // Register holding co-routine entry-point
		int regEof = ++parse->Mems; // Register holding co-routine completion flag
		Vdbe *v = parse->GetVdbe(); // VDBE under construction
		int addrTop = v->CurrentAddr(); // Top of the co-routine
		v->AddOp2(OP_Integer, addrTop+2, regYield); // X <- A
		v->Comment("Co-routine entry point");
		v->AddOp2(OP_Integer, 0, regEof); // EOF <- 0
		v->Comment("Co-routine completion flag");
		Select::DestInit(dest, SRT_Coroutine, regYield);
		int j1 = v->AddOp2(OP_Goto, 0, 0); // Jump instruction
		RC rc = Select::Select_(parse, select, dest);
		_assert(parse->Errs == 0 || rc);
		if (parse->Ctx->MallocFailed && rc == RC_OK) rc = RC_NOMEM;
		if (rc) return rc;
		v->AddOp2(OP_Integer, 1, regEof); // EOF <- 1
		v->AddOp1(OP_Yield, regYield); // yield X
		v->AddOp2(OP_Halt, SQLITE_INTERNAL, OE_Abort);
		v->Comment("End of coroutine");
		v->JumpHere(j1); // label B:
		return rc;
	}

	// Forward declaration
	__device__ static int XferOptimization(Parse *parse, Table *dest, Select *select, int onError, int dbDestId);
	__device__ void Insert::Insert_(Parse *parse, SrcList *tabList, ExprList *list, Select *select, IdList *column, int onError)
	{
		;          
		pTab;          
		const char *zDb;      // Name of the database holding this table
		int i, j, idx;        // Loop counters
		Vdbe *v;              // Generate code into this virtual machine
		Index *pIdx;          // For looping over indices of the table
		int nColumn;          // Number of columns in the data
		int nHidden = 0;      // Number of hidden columns if TABLE is virtual
		int baseCur = 0;      // VDBE Cursor number for pTab
		int keyColumn = -1;   // Column that is the INTEGER PRIMARY KEY
		int endOfLoop;        // Label for the end of the insertion loop
		int useTempTable = 0; // Store SELECT results in intermediate table
		int srcTab = 0;       // Data comes from this temporary cursor if >=0
		int addrInsTop = 0;   // Jump to label "D"
		int addrCont = 0;     // Top of insert loop. Label "C" in templates 3 and 4
		int addrSelect = 0;   // Address of coroutine that implements the SELECT

		int iDb;              // Index of database holding TABLE
		Context::DB *pDb;              // The database containing table being inserted into
		int appendFlag = 0;   // True if the insert is likely to be an append

		// Register allocations
		int regFromSelect = 0;// Base register for data coming from SELECT
		int regAutoinc = 0;   // Register holding the AUTOINCREMENT counter
		int regRowCount = 0;  // Memory cell used for the row counter
		int regIns;           // Block of regs holding rowid+data being inserted
		int regRowid;         // registers holding insert rowid
		int regData;          // register holding first column to insert
		int regEof = 0;       // Register recording end of SELECT data
		int *aRegIdx = 0;     // One register allocated to each index

#ifndef OMIT_TRIGGER
		int isView;                 /* True if attempting to insert into a view */
		Trigger *pTrigger;          /* List of triggers on table, if required */
		int tmask;                  /* Mask of trigger times */
#endif

		Context *ctx = parse->Ctx; // The main database structure
		SelectDest dest; // Destination for SELECT on rhs of INSERT
		_memset(&dest, 0, sizeof(dest));
		if (parse->Errs || ctx->MallocFailed)
			goto insert_cleanup;

		// Locate the table into which we will be inserting new information.
		_assert(tabList->Srcs == 1);
		char *tableName = tabList->Ids[0].Name; // Name of the table into which we are inserting
		if( NEVER(tableName==0) ) goto insert_cleanup;
		Table *table = sqlite3SrcListLookup(pParse, pTabList); // The table to insert into.  aka TABLE
		if( table==0 ){
			goto insert_cleanup;
		}
		iDb = sqlite3SchemaToIndex(db, table->pSchema);
		assert( iDb<db->nDb );
		pDb = &db->aDb[iDb];
		zDb = pDb->zName;
		if( sqlite3AuthCheck(pParse, SQLITE_INSERT, table->zName, 0, zDb) ){
			goto insert_cleanup;
		}

		/* Figure out if we have any triggers and if the table being
		** inserted into is a view
		*/
#ifndef OMIT_TRIGGER
		pTrigger = sqlite3TriggersExist(pParse, table, TK_INSERT, 0, &tmask);
		isView = table->pSelect!=0;
#else
# define pTrigger 0
# define tmask 0
# define isView 0
#endif
#ifdef OMIT_VIEW
# undef isView
# define isView 0
#endif
		assert( (pTrigger && tmask) || (pTrigger==0 && tmask==0) );

		/* If table is really a view, make sure it has been initialized.
		** ViewGetColumnNames() is a no-op if table is not a view (or virtual 
		** module table).
		*/
		if( sqlite3ViewGetColumnNames(pParse, table) ){
			goto insert_cleanup;
		}

		/* Ensure that:
		*  (a) the table is not read-only, 
		*  (b) that if it is a view then ON INSERT triggers exist
		*/
		if( sqlite3IsReadOnly(pParse, table, tmask) ){
			goto insert_cleanup;
		}

		/* Allocate a VDBE
		*/
		v = sqlite3GetVdbe(pParse);
		if( v==0 ) goto insert_cleanup;
		if( pParse->nested==0 ) sqlite3VdbeCountChanges(v);
		sqlite3BeginWriteOperation(pParse, pSelect || pTrigger, iDb);

#ifndef OMIT_XFER_OPT
		/* If the statement is of the form
		**
		**       INSERT INTO <table1> SELECT * FROM <table2>;
		**
		** Then special optimizations can be applied that make the transfer
		** very fast and which reduce fragmentation of indices.
		**
		** This is the 2nd template.
		*/
		if( pColumn==0 && xferOptimization(pParse, table, pSelect, onError, iDb) ){
			assert( !pTrigger );
			assert( pList==0 );
			goto insert_end;
		}
#endif

		/* If this is an AUTOINCREMENT table, look up the sequence number in the
		** sqlite_sequence table and store it in memory cell regAutoinc.
		*/
		regAutoinc = autoIncBegin(pParse, iDb, table);

		/* Figure out how many columns of data are supplied.  If the data
		** is coming from a SELECT statement, then generate a co-routine that
		** produces a single row of the SELECT on each invocation.  The
		** co-routine is the common header to the 3rd and 4th templates.
		*/
		if( pSelect ){
			/* Data is coming from a SELECT.  Generate a co-routine to run that
			** SELECT. */
			int rc = sqlite3CodeCoroutine(pParse, pSelect, &dest);
			if( rc ) goto insert_cleanup;

			regEof = dest.iSDParm + 1;
			regFromSelect = dest.iSdst;
			assert( pSelect->pEList );
			nColumn = pSelect->pEList->nExpr;
			assert( dest.nSdst==nColumn );

			/* Set useTempTable to TRUE if the result of the SELECT statement
			** should be written into a temporary table (template 4).  Set to
			** FALSE if each* row of the SELECT can be written directly into
			** the destination table (template 3).
			**
			** A temp table must be used if the table being updated is also one
			** of the tables being read by the SELECT statement.  Also use a 
			** temp table in the case of row triggers.
			*/
			if( pTrigger || readsTable(pParse, addrSelect, iDb, table) ){
				useTempTable = 1;
			}

			if( useTempTable ){
				/* Invoke the coroutine to extract information from the SELECT
				** and add it to a transient table srcTab.  The code generated
				** here is from the 4th template:
				**
				**      B: open temp table
				**      L: yield X
				**         if EOF goto M
				**         insert row from R..R+n into temp table
				**         goto L
				**      M: ...
				*/
				int regRec;          /* Register to hold packed record */
				int regTempRowid;    /* Register to hold temp table ROWID */
				int addrTop;         /* Label "L" */
				int addrIf;          /* Address of jump to M */

				srcTab = pParse->nTab++;
				regRec = sqlite3GetTempReg(pParse);
				regTempRowid = sqlite3GetTempReg(pParse);
				sqlite3VdbeAddOp2(v, OP_OpenEphemeral, srcTab, nColumn);
				addrTop = sqlite3VdbeAddOp1(v, OP_Yield, dest.iSDParm);
				addrIf = sqlite3VdbeAddOp1(v, OP_If, regEof);
				sqlite3VdbeAddOp3(v, OP_MakeRecord, regFromSelect, nColumn, regRec);
				sqlite3VdbeAddOp2(v, OP_NewRowid, srcTab, regTempRowid);
				sqlite3VdbeAddOp3(v, OP_Insert, srcTab, regRec, regTempRowid);
				sqlite3VdbeAddOp2(v, OP_Goto, 0, addrTop);
				sqlite3VdbeJumpHere(v, addrIf);
				sqlite3ReleaseTempReg(pParse, regRec);
				sqlite3ReleaseTempReg(pParse, regTempRowid);
			}
		}else{
			/* This is the case if the data for the INSERT is coming from a VALUES
			** clause
			*/
			NameContext sNC;
			memset(&sNC, 0, sizeof(sNC));
			sNC.pParse = pParse;
			srcTab = -1;
			assert( useTempTable==0 );
			nColumn = pList ? pList->nExpr : 0;
			for(i=0; i<nColumn; i++){
				if( sqlite3ResolveExprNames(&sNC, pList->a[i].pExpr) ){
					goto insert_cleanup;
				}
			}
		}

		/* Make sure the number of columns in the source data matches the number
		** of columns to be inserted into the table.
		*/
		if( IsVirtual(table) ){
			for(i=0; i<table->nCol; i++){
				nHidden += (IsHiddenColumn(&table->aCol[i]) ? 1 : 0);
			}
		}
		if( pColumn==0 && nColumn && nColumn!=(table->nCol-nHidden) ){
			sqlite3ErrorMsg(pParse, 
				"table %S has %d columns but %d values were supplied",
				pTabList, 0, table->nCol-nHidden, nColumn);
			goto insert_cleanup;
		}
		if( pColumn!=0 && nColumn!=pColumn->nId ){
			sqlite3ErrorMsg(pParse, "%d values for %d columns", nColumn, pColumn->nId);
			goto insert_cleanup;
		}

		/* If the INSERT statement included an IDLIST term, then make sure
		** all elements of the IDLIST really are columns of the table and 
		** remember the column indices.
		**
		** If the table has an INTEGER PRIMARY KEY column and that column
		** is named in the IDLIST, then record in the keyColumn variable
		** the index into IDLIST of the primary key column.  keyColumn is
		** the index of the primary key as it appears in IDLIST, not as
		** is appears in the original table.  (The index of the primary
		** key in the original table is table->iPKey.)
		*/
		if( pColumn ){
			for(i=0; i<pColumn->nId; i++){
				pColumn->a[i].idx = -1;
			}
			for(i=0; i<pColumn->nId; i++){
				for(j=0; j<table->nCol; j++){
					if( sqlite3StrICmp(pColumn->a[i].zName, table->aCol[j].zName)==0 ){
						pColumn->a[i].idx = j;
						if( j==table->iPKey ){
							keyColumn = i;
						}
						break;
					}
				}
				if( j>=table->nCol ){
					if( sqlite3IsRowid(pColumn->a[i].zName) ){
						keyColumn = i;
					}else{
						sqlite3ErrorMsg(pParse, "table %S has no column named %s",
							pTabList, 0, pColumn->a[i].zName);
						pParse->checkSchema = 1;
						goto insert_cleanup;
					}
				}
			}
		}

		/* If there is no IDLIST term but the table has an integer primary
		** key, the set the keyColumn variable to the primary key column index
		** in the original table definition.
		*/
		if( pColumn==0 && nColumn>0 ){
			keyColumn = table->iPKey;
		}

		/* Initialize the count of rows to be inserted
		*/
		if( db->flags & SQLITE_CountRows ){
			regRowCount = ++pParse->nMem;
			sqlite3VdbeAddOp2(v, OP_Integer, 0, regRowCount);
		}

		/* If this is not a view, open the table and and all indices */
		if( !isView ){
			int nIdx;

			baseCur = pParse->nTab;
			nIdx = sqlite3OpenTableAndIndices(pParse, table, baseCur, OP_OpenWrite);
			aRegIdx = sqlite3DbMallocRaw(db, sizeof(int)*(nIdx+1));
			if( aRegIdx==0 ){
				goto insert_cleanup;
			}
			for(i=0; i<nIdx; i++){
				aRegIdx[i] = ++pParse->nMem;
			}
		}

		/* This is the top of the main insertion loop */
		if( useTempTable ){
			/* This block codes the top of loop only.  The complete loop is the
			** following pseudocode (template 4):
			**
			**         rewind temp table
			**      C: loop over rows of intermediate table
			**           transfer values form intermediate table into <table>
			**         end loop
			**      D: ...
			*/
			addrInsTop = sqlite3VdbeAddOp1(v, OP_Rewind, srcTab);
			addrCont = sqlite3VdbeCurrentAddr(v);
		}else if( pSelect ){
			/* This block codes the top of loop only.  The complete loop is the
			** following pseudocode (template 3):
			**
			**      C: yield X
			**         if EOF goto D
			**         insert the select result into <table> from R..R+n
			**         goto C
			**      D: ...
			*/
			addrCont = sqlite3VdbeAddOp1(v, OP_Yield, dest.iSDParm);
			addrInsTop = sqlite3VdbeAddOp1(v, OP_If, regEof);
		}

		/* Allocate registers for holding the rowid of the new row,
		** the content of the new row, and the assemblied row record.
		*/
		regRowid = regIns = pParse->nMem+1;
		pParse->nMem += table->nCol + 1;
		if( IsVirtual(table) ){
			regRowid++;
			pParse->nMem++;
		}
		regData = regRowid+1;

		/* Run the BEFORE and INSTEAD OF triggers, if there are any
		*/
		endOfLoop = sqlite3VdbeMakeLabel(v);
		if( tmask & TRIGGER_BEFORE ){
			int regCols = sqlite3GetTempRange(pParse, table->nCol+1);

			/* build the NEW.* reference row.  Note that if there is an INTEGER
			** PRIMARY KEY into which a NULL is being inserted, that NULL will be
			** translated into a unique ID for the row.  But on a BEFORE trigger,
			** we do not know what the unique ID will be (because the insert has
			** not happened yet) so we substitute a rowid of -1
			*/
			if( keyColumn<0 ){
				sqlite3VdbeAddOp2(v, OP_Integer, -1, regCols);
			}else{
				int j1;
				if( useTempTable ){
					sqlite3VdbeAddOp3(v, OP_Column, srcTab, keyColumn, regCols);
				}else{
					assert( pSelect==0 );  /* Otherwise useTempTable is true */
					sqlite3ExprCode(pParse, pList->a[keyColumn].pExpr, regCols);
				}
				j1 = sqlite3VdbeAddOp1(v, OP_NotNull, regCols);
				sqlite3VdbeAddOp2(v, OP_Integer, -1, regCols);
				sqlite3VdbeJumpHere(v, j1);
				sqlite3VdbeAddOp1(v, OP_MustBeInt, regCols);
			}

			/* Cannot have triggers on a virtual table. If it were possible,
			** this block would have to account for hidden column.
			*/
			assert( !IsVirtual(table) );

			/* Create the new column data
			*/
			for(i=0; i<table->nCol; i++){
				if( pColumn==0 ){
					j = i;
				}else{
					for(j=0; j<pColumn->nId; j++){
						if( pColumn->a[j].idx==i ) break;
					}
				}
				if( (!useTempTable && !pList) || (pColumn && j>=pColumn->nId) ){
					sqlite3ExprCode(pParse, table->aCol[i].pDflt, regCols+i+1);
				}else if( useTempTable ){
					sqlite3VdbeAddOp3(v, OP_Column, srcTab, j, regCols+i+1); 
				}else{
					assert( pSelect==0 ); /* Otherwise useTempTable is true */
					sqlite3ExprCodeAndCache(pParse, pList->a[j].pExpr, regCols+i+1);
				}
			}

			/* If this is an INSERT on a view with an INSTEAD OF INSERT trigger,
			** do not attempt any conversions before assembling the record.
			** If this is a real table, attempt conversions as required by the
			** table column affinities.
			*/
			if( !isView ){
				sqlite3VdbeAddOp2(v, OP_Affinity, regCols+1, table->nCol);
				sqlite3TableAffinityStr(v, table);
			}

			/* Fire BEFORE or INSTEAD OF triggers */
			sqlite3CodeRowTrigger(pParse, pTrigger, TK_INSERT, 0, TRIGGER_BEFORE, 
				table, regCols-table->nCol-1, onError, endOfLoop);

			sqlite3ReleaseTempRange(pParse, regCols, table->nCol+1);
		}

		/* Push the record number for the new entry onto the stack.  The
		** record number is a randomly generate integer created by NewRowid
		** except when the table has an INTEGER PRIMARY KEY column, in which
		** case the record number is the same as that column. 
		*/
		if( !isView ){
			if( IsVirtual(table) ){
				/* The row that the VUpdate opcode will delete: none */
				sqlite3VdbeAddOp2(v, OP_Null, 0, regIns);
			}
			if( keyColumn>=0 ){
				if( useTempTable ){
					sqlite3VdbeAddOp3(v, OP_Column, srcTab, keyColumn, regRowid);
				}else if( pSelect ){
					sqlite3VdbeAddOp2(v, OP_SCopy, regFromSelect+keyColumn, regRowid);
				}else{
					VdbeOp *pOp;
					sqlite3ExprCode(pParse, pList->a[keyColumn].pExpr, regRowid);
					pOp = sqlite3VdbeGetOp(v, -1);
					if( ALWAYS(pOp) && pOp->opcode==OP_Null && !IsVirtual(table) ){
						appendFlag = 1;
						pOp->opcode = OP_NewRowid;
						pOp->p1 = baseCur;
						pOp->p2 = regRowid;
						pOp->p3 = regAutoinc;
					}
				}
				/* If the PRIMARY KEY expression is NULL, then use OP_NewRowid
				** to generate a unique primary key value.
				*/
				if( !appendFlag ){
					int j1;
					if( !IsVirtual(table) ){
						j1 = sqlite3VdbeAddOp1(v, OP_NotNull, regRowid);
						sqlite3VdbeAddOp3(v, OP_NewRowid, baseCur, regRowid, regAutoinc);
						sqlite3VdbeJumpHere(v, j1);
					}else{
						j1 = sqlite3VdbeCurrentAddr(v);
						sqlite3VdbeAddOp2(v, OP_IsNull, regRowid, j1+2);
					}
					sqlite3VdbeAddOp1(v, OP_MustBeInt, regRowid);
				}
			}else if( IsVirtual(table) ){
				sqlite3VdbeAddOp2(v, OP_Null, 0, regRowid);
			}else{
				sqlite3VdbeAddOp3(v, OP_NewRowid, baseCur, regRowid, regAutoinc);
				appendFlag = 1;
			}
			autoIncStep(pParse, regAutoinc, regRowid);

			/* Push onto the stack, data for all columns of the new entry, beginning
			** with the first column.
			*/
			nHidden = 0;
			for(i=0; i<table->nCol; i++){
				int iRegStore = regRowid+1+i;
				if( i==table->iPKey ){
					/* The value of the INTEGER PRIMARY KEY column is always a NULL.
					** Whenever this column is read, the record number will be substituted
					** in its place.  So will fill this column with a NULL to avoid
					** taking up data space with information that will never be used. */
					sqlite3VdbeAddOp2(v, OP_Null, 0, iRegStore);
					continue;
				}
				if( pColumn==0 ){
					if( IsHiddenColumn(&table->aCol[i]) ){
						assert( IsVirtual(table) );
						j = -1;
						nHidden++;
					}else{
						j = i - nHidden;
					}
				}else{
					for(j=0; j<pColumn->nId; j++){
						if( pColumn->a[j].idx==i ) break;
					}
				}
				if( j<0 || nColumn==0 || (pColumn && j>=pColumn->nId) ){
					sqlite3ExprCode(pParse, table->aCol[i].pDflt, iRegStore);
				}else if( useTempTable ){
					sqlite3VdbeAddOp3(v, OP_Column, srcTab, j, iRegStore); 
				}else if( pSelect ){
					sqlite3VdbeAddOp2(v, OP_SCopy, regFromSelect+j, iRegStore);
				}else{
					sqlite3ExprCode(pParse, pList->a[j].pExpr, iRegStore);
				}
			}

			/* Generate code to check constraints and generate index keys and
			** do the insertion.
			*/
#ifndef OMIT_VIRTUALTABLE
			if( IsVirtual(table) ){
				const char *pVTab = (const char *)sqlite3GetVTable(db, table);
				sqlite3VtabMakeWritable(pParse, table);
				sqlite3VdbeAddOp4(v, OP_VUpdate, 1, table->nCol+2, regIns, pVTab, P4_VTAB);
				sqlite3VdbeChangeP5(v, onError==OE_Default ? OE_Abort : onError);
				sqlite3MayAbort(pParse);
			}else
#endif
			{
				int isReplace;    /* Set to true if constraints may cause a replace */
				sqlite3GenerateConstraintChecks(pParse, table, baseCur, regIns, aRegIdx,
					keyColumn>=0, 0, onError, endOfLoop, &isReplace
					);
				sqlite3FkCheck(pParse, table, 0, regIns);
				sqlite3CompleteInsertion(
					pParse, table, baseCur, regIns, aRegIdx, 0, appendFlag, isReplace==0
					);
			}
		}

		/* Update the count of rows that are inserted
		*/
		if( (db->flags & SQLITE_CountRows)!=0 ){
			sqlite3VdbeAddOp2(v, OP_AddImm, regRowCount, 1);
		}

		if( pTrigger ){
			/* Code AFTER triggers */
			sqlite3CodeRowTrigger(pParse, pTrigger, TK_INSERT, 0, TRIGGER_AFTER, 
				table, regData-2-table->nCol, onError, endOfLoop);
		}

		/* The bottom of the main insertion loop, if the data source
		** is a SELECT statement.
		*/
		sqlite3VdbeResolveLabel(v, endOfLoop);
		if( useTempTable ){
			sqlite3VdbeAddOp2(v, OP_Next, srcTab, addrCont);
			sqlite3VdbeJumpHere(v, addrInsTop);
			sqlite3VdbeAddOp1(v, OP_Close, srcTab);
		}else if( pSelect ){
			sqlite3VdbeAddOp2(v, OP_Goto, 0, addrCont);
			sqlite3VdbeJumpHere(v, addrInsTop);
		}

		if( !IsVirtual(table) && !isView ){
			/* Close all tables opened */
			sqlite3VdbeAddOp1(v, OP_Close, baseCur);
			for(idx=1, pIdx=table->pIndex; pIdx; pIdx=pIdx->pNext, idx++){
				sqlite3VdbeAddOp1(v, OP_Close, idx+baseCur);
			}
		}

insert_end:
		/* Update the sqlite_sequence table by storing the content of the
		** maximum rowid counter values recorded while inserting into
		** autoincrement tables.
		*/
		if( pParse->nested==0 && pParse->pTriggerTab==0 ){
			sqlite3AutoincrementEnd(pParse);
		}

		/*
		** Return the number of rows inserted. If this routine is 
		** generating code because of a call to sqlite3NestedParse(), do not
		** invoke the callback function.
		*/
		if( (db->flags&SQLITE_CountRows) && !pParse->nested && !pParse->pTriggerTab ){
			sqlite3VdbeAddOp2(v, OP_ResultRow, regRowCount, 1);
			sqlite3VdbeSetNumCols(v, 1);
			sqlite3VdbeSetColName(v, 0, COLNAME_NAME, "rows inserted", SQLITE_STATIC);
		}

insert_cleanup:
		sqlite3SrcListDelete(db, pTabList);
		sqlite3ExprListDelete(db, pList);
		sqlite3SelectDelete(db, pSelect);
		sqlite3IdListDelete(db, pColumn);
		sqlite3DbFree(db, aRegIdx);
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
#ifdef tmask
#undef tmask
#endif

	void sqlite3GenerateConstraintChecks(Parse *pParse, Table *table, int baseCur, int regRowid, int *aRegIdx, int rowidChng, int isUpdate, int overrideError, int ignoreDest, int *pbMayReplace)
	{
		int i;              /* loop counter */
		Vdbe *v;            /* VDBE under constrution */
		int nCol;           /* Number of columns */
		int onError;        /* Conflict resolution strategy */
		int j1;             /* Addresss of jump instruction */
		int j2 = 0, j3;     /* Addresses of jump instructions */
		int regData;        /* Register containing first data column */
		int iCur;           /* Table cursor number */
		Index *pIdx;         /* Pointer to one of the indices */
		sqlite3 *db;         /* Database connection */
		int seenReplace = 0; /* True if REPLACE is used to resolve INT PK conflict */
		int regOldRowid = (rowidChng && isUpdate) ? rowidChng : regRowid;

		db = pParse->db;
		v = sqlite3GetVdbe(pParse);
		assert( v!=0 );
		assert( table->pSelect==0 );  /* This table is not a VIEW */
		nCol = table->nCol;
		regData = regRowid + 1;

		/* Test all NOT NULL constraints.
		*/
		for(i=0; i<nCol; i++){
			if( i==table->iPKey ){
				continue;
			}
			onError = table->aCol[i].notNull;
			if( onError==OE_None ) continue;
			if( overrideError!=OE_Default ){
				onError = overrideError;
			}else if( onError==OE_Default ){
				onError = OE_Abort;
			}
			if( onError==OE_Replace && table->aCol[i].pDflt==0 ){
				onError = OE_Abort;
			}
			assert( onError==OE_Rollback || onError==OE_Abort || onError==OE_Fail
				|| onError==OE_Ignore || onError==OE_Replace );
			switch( onError ){
			case OE_Abort:
				sqlite3MayAbort(pParse);
			case OE_Rollback:
			case OE_Fail: {
				char *zMsg;
				sqlite3VdbeAddOp3(v, OP_HaltIfNull,
					SQLITE_CONSTRAINT_NOTNULL, onError, regData+i);
				zMsg = sqlite3MPrintf(db, "%s.%s may not be NULL",
					table->zName, table->aCol[i].zName);
				sqlite3VdbeChangeP4(v, -1, zMsg, P4_DYNAMIC);
				break;
						  }
			case OE_Ignore: {
				sqlite3VdbeAddOp2(v, OP_IsNull, regData+i, ignoreDest);
				break;
							}
			default: {
				assert( onError==OE_Replace );
				j1 = sqlite3VdbeAddOp1(v, OP_NotNull, regData+i);
				sqlite3ExprCode(pParse, table->aCol[i].pDflt, regData+i);
				sqlite3VdbeJumpHere(v, j1);
				break;
					 }
			}
		}

		// Test all CHECK constraints
#ifndef OMIT_CHECK
		if( table->pCheck && (db->flags & SQLITE_IgnoreChecks)==0 ){
			ExprList *pCheck = table->pCheck;
			pParse->ckBase = regData;
			onError = overrideError!=OE_Default ? overrideError : OE_Abort;
			for(i=0; i<pCheck->nExpr; i++){
				int allOk = sqlite3VdbeMakeLabel(v);
				sqlite3ExprIfTrue(pParse, pCheck->a[i].pExpr, allOk, SQLITE_JUMPIFNULL);
				if( onError==OE_Ignore ){
					sqlite3VdbeAddOp2(v, OP_Goto, 0, ignoreDest);
				}else{
					char *zConsName = pCheck->a[i].zName;
					if( onError==OE_Replace ) onError = OE_Abort; /* IMP: R-15569-63625 */
					if( zConsName ){
						zConsName = sqlite3MPrintf(db, "constraint %s failed", zConsName);
					}else{
						zConsName = 0;
					}
					sqlite3HaltConstraint(pParse, SQLITE_CONSTRAINT_CHECK,
						onError, zConsName, P4_DYNAMIC);
				}
				sqlite3VdbeResolveLabel(v, allOk);
			}
		}
#endif

		/* If we have an INTEGER PRIMARY KEY, make sure the primary key
		** of the new record does not previously exist.  Except, if this
		** is an UPDATE and the primary key is not changing, that is OK.
		*/
		if( rowidChng ){
			onError = table->keyConf;
			if( overrideError!=OE_Default ){
				onError = overrideError;
			}else if( onError==OE_Default ){
				onError = OE_Abort;
			}

			if( isUpdate ){
				j2 = sqlite3VdbeAddOp3(v, OP_Eq, regRowid, 0, rowidChng);
			}
			j3 = sqlite3VdbeAddOp3(v, OP_NotExists, baseCur, 0, regRowid);
			switch( onError ){
			default: {
				onError = OE_Abort;
				/* Fall thru into the next case */
					 }
			case OE_Rollback:
			case OE_Abort:
			case OE_Fail: {
				sqlite3HaltConstraint(pParse, SQLITE_CONSTRAINT_PRIMARYKEY,
					onError, "PRIMARY KEY must be unique", P4_STATIC);
				break;
						  }
			case OE_Replace: {
				/* If there are DELETE triggers on this table and the
				** recursive-triggers flag is set, call GenerateRowDelete() to
				** remove the conflicting row from the table. This will fire
				** the triggers and remove both the table and index b-tree entries.
				**
				** Otherwise, if there are no triggers or the recursive-triggers
				** flag is not set, but the table has one or more indexes, call 
				** GenerateRowIndexDelete(). This removes the index b-tree entries 
				** only. The table b-tree entry will be replaced by the new entry 
				** when it is inserted.  
				**
				** If either GenerateRowDelete() or GenerateRowIndexDelete() is called,
				** also invoke MultiWrite() to indicate that this VDBE may require
				** statement rollback (if the statement is aborted after the delete
				** takes place). Earlier versions called sqlite3MultiWrite() regardless,
				** but being more selective here allows statements like:
				**
				**   REPLACE INTO t(rowid) VALUES($newrowid)
				**
				** to run without a statement journal if there are no indexes on the
				** table.
				*/
				Trigger *pTrigger = 0;
				if( db->flags&SQLITE_RecTriggers ){
					pTrigger = sqlite3TriggersExist(pParse, table, TK_DELETE, 0, 0);
				}
				if( pTrigger || sqlite3FkRequired(pParse, table, 0, 0) ){
					sqlite3MultiWrite(pParse);
					sqlite3GenerateRowDelete(
						pParse, table, baseCur, regRowid, 0, pTrigger, OE_Replace
						);
				}else if( table->pIndex ){
					sqlite3MultiWrite(pParse);
					sqlite3GenerateRowIndexDelete(pParse, table, baseCur, 0);
				}
				seenReplace = 1;
				break;
							 }
			case OE_Ignore: {
				assert( seenReplace==0 );
				sqlite3VdbeAddOp2(v, OP_Goto, 0, ignoreDest);
				break;
							}
			}
			sqlite3VdbeJumpHere(v, j3);
			if( isUpdate ){
				sqlite3VdbeJumpHere(v, j2);
			}
		}

		/* Test all UNIQUE constraints by creating entries for each UNIQUE
		** index and making sure that duplicate entries do not already exist.
		** Add the new records to the indices as we go.
		*/
		for(iCur=0, pIdx=table->pIndex; pIdx; pIdx=pIdx->pNext, iCur++){
			int regIdx;
			int regR;

			if( aRegIdx[iCur]==0 ) continue;  /* Skip unused indices */

			/* Create a key for accessing the index entry */
			regIdx = sqlite3GetTempRange(pParse, pIdx->nColumn+1);
			for(i=0; i<pIdx->nColumn; i++){
				int idx = pIdx->aiColumn[i];
				if( idx==table->iPKey ){
					sqlite3VdbeAddOp2(v, OP_SCopy, regRowid, regIdx+i);
				}else{
					sqlite3VdbeAddOp2(v, OP_SCopy, regData+idx, regIdx+i);
				}
			}
			sqlite3VdbeAddOp2(v, OP_SCopy, regRowid, regIdx+i);
			sqlite3VdbeAddOp3(v, OP_MakeRecord, regIdx, pIdx->nColumn+1, aRegIdx[iCur]);
			sqlite3VdbeChangeP4(v, -1, sqlite3IndexAffinityStr(v, pIdx), P4_TRANSIENT);
			sqlite3ExprCacheAffinityChange(pParse, regIdx, pIdx->nColumn+1);

			/* Find out what action to take in case there is an indexing conflict */
			onError = pIdx->onError;
			if( onError==OE_None ){ 
				sqlite3ReleaseTempRange(pParse, regIdx, pIdx->nColumn+1);
				continue;  /* pIdx is not a UNIQUE index */
			}
			if( overrideError!=OE_Default ){
				onError = overrideError;
			}else if( onError==OE_Default ){
				onError = OE_Abort;
			}
			if( seenReplace ){
				if( onError==OE_Ignore ) onError = OE_Replace;
				else if( onError==OE_Fail ) onError = OE_Abort;
			}

			/* Check to see if the new index entry will be unique */
			regR = sqlite3GetTempReg(pParse);
			sqlite3VdbeAddOp2(v, OP_SCopy, regOldRowid, regR);
			j3 = sqlite3VdbeAddOp4(v, OP_IsUnique, baseCur+iCur+1, 0,
				regR, SQLITE_INT_TO_PTR(regIdx),
				P4_INT32);
			sqlite3ReleaseTempRange(pParse, regIdx, pIdx->nColumn+1);

			/* Generate code that executes if the new index entry is not unique */
			assert( onError==OE_Rollback || onError==OE_Abort || onError==OE_Fail
				|| onError==OE_Ignore || onError==OE_Replace );
			switch( onError ){
			case OE_Rollback:
			case OE_Abort:
			case OE_Fail: {
				int j;
				StrAccum errMsg;
				const char *zSep;
				char *zErr;

				sqlite3StrAccumInit(&errMsg, 0, 0, 200);
				errMsg.db = db;
				zSep = pIdx->nColumn>1 ? "columns " : "column ";
				for(j=0; j<pIdx->nColumn; j++){
					char *zCol = table->aCol[pIdx->aiColumn[j]].zName;
					sqlite3StrAccumAppend(&errMsg, zSep, -1);
					zSep = ", ";
					sqlite3StrAccumAppend(&errMsg, zCol, -1);
				}
				sqlite3StrAccumAppend(&errMsg,
					pIdx->nColumn>1 ? " are not unique" : " is not unique", -1);
				zErr = sqlite3StrAccumFinish(&errMsg);
				sqlite3HaltConstraint(pParse, SQLITE_CONSTRAINT_UNIQUE,
					onError, zErr, 0);
				sqlite3DbFree(errMsg.db, zErr);
				break;
						  }
			case OE_Ignore: {
				assert( seenReplace==0 );
				sqlite3VdbeAddOp2(v, OP_Goto, 0, ignoreDest);
				break;
							}
			default: {
				Trigger *pTrigger = 0;
				assert( onError==OE_Replace );
				sqlite3MultiWrite(pParse);
				if( db->flags&SQLITE_RecTriggers ){
					pTrigger = sqlite3TriggersExist(pParse, table, TK_DELETE, 0, 0);
				}
				sqlite3GenerateRowDelete(
					pParse, table, baseCur, regR, 0, pTrigger, OE_Replace
					);
				seenReplace = 1;
				break;
					 }
			}
			sqlite3VdbeJumpHere(v, j3);
			sqlite3ReleaseTempReg(pParse, regR);
		}

		if( pbMayReplace ){
			*pbMayReplace = seenReplace;
		}
	}

	__device__ void Insert::CompleteInsertion(Parse *parse, Table *table, int baseCur, int regRowid, int *regIdxs, bool isUpdate, bool appendBias, bool useSeekResult)
	{
		Vdbe *v = parse->GetVdbe();
		_assert(v);
		_assert(!table->Select);  // This table is not a VIEW
		int indexsLength;
		Index *index;
		for (indexsLength = 0, index = table->Index; index; index = index->Next, indexsLength++) { }
		for (int i = indexsLength-1; i >= 0; i--)
		{
			if (regIdxs[i] == 0) continue;
			v->AddOp2(OP_IdxInsert, baseCur+i+1, regIdxs[i]);
			if (useSeekResult)
				v->ChangeP5(Vdbe::OPFLAG_USESEEKRESULT);
		}
		int regData = regRowid + 1;
		int regRec = Expr::GetTempReg(parse);
		v->AddOp3(v, OP_MakeRecord, regData, table->Cols.length, regRec);
		TableAffinityStr(v, table);
		Expr::CacheAffinityChange(parse, regData, table->Cols.length);
		Vdbe::OPFLAG pik_flags;
		if (parse->Nested)
			pik_flags = 0;
		else
		{
			pik_flags = Vdbe::OPFLAG_NCHANGE;
			pik_flags |= (isUpdate ? Vdbe::OPFLAG_ISUPDATE : Vdbe::OPFLAG_LASTROWID);
		}
		if (appendBias) pik_flags |= Vdbe::OPFLAG_APPEND;
		if (useSeekResult) pik_flags |= Vdbe::OPFLAG_USESEEKRESULT;
		v->AddOp3(OP_Insert, baseCur, regRec, regRowid);
		if (!parse->Nested)
			v->ChangeP4(-1, table->Name, Vdbe::P4T_TRANSIENT);
		v->ChangeP5(pik_flags);
	}

	__device__ int Insert::OpenTableAndIndices(Parse *parse, Table *table, int baseCur, OP op)
	{
		if (IsVirtual(table)) return 0;
		int db = Prepare::SchemaToIndex(parse->Ctx, table->Schema);
		Vdbe *v = parse->GetVdbe();
		_assert(v);
		OpenTable(parse, baseCur, db, table, op);
		int i;
		Index *index;
		for (i = 1, index = table->Index; index; index = index->Next, i++)
		{
			KeyInfo *key = parse->IndexKeyinfo(index);
			_assert(index->Schema == table->Schema);
			v->AddOp4(op, i+baseCur, index->Id, db, (char *)key, Vdbe::P4T_KEYINFO_HANDOFF);
			v->Comment("%s", index->Name);
		}
		if (parse->Tabs < baseCur+i)
			parse->Tabs = baseCur+i;
		return i-1;
	}

#ifdef TEST
	__device__ int _xferopt_count;
#endif

#ifndef OMIT_XFER_OPT
	__device__ static bool XferCompatibleCollation(const char *z1, const char *z2)
	{
		if (!z1) return (z2 == nullptr);
		if (!z2) return false;
		return !_strcmp(z1, z2);
	}

	__device__ static bool XferCompatibleIndex(Index *dest, Index *src)
	{
		_assert(dest && src);
		_assert(dest->Table != src->Table);
		if (dest->Columns.length != src->Columns.length) return false;			// Different number of columns
		if (dest->OnError != src->OnError) return false;						// Different conflict resolution strategies
		for (int i = 0; i < src->Columns.length; i++)
		{
			if (src->Columns[i] != dest->Columns[i]) return false;				// Different columns indexed
			if (src->SortOrders[i] != dest->SortOrders[i]) return false;		// Different sort orders
			if (!XferCompatibleCollation(src->CollNames[i], dest->CollNames[i])) return false; // Different collating sequences
		}
		// If no test above fails then the indices must be compatible
		return true;
	}

	__device__ static bool XferOptimization(Parse *parse, Table *dest, Select *select, OE onError, int dbDestId)
	{
		if (!select) return false;									// Must be of the form  INSERT INTO ... SELECT ...
		if (sqlite3TriggerList(parse, dest)) return false;			// tab1 must not have triggers
#ifndef OMIT_VIRTUALTABLE
		if (dest->TabFlags & TF_Virtual) return false;				// tab1 must not be a virtual table
#endif
		if (onError == OE_Default)
		{
			if (dest->PKey >= 0) onError = dest->KeyConf;
			if (onError == OE_Default) onError = OE_Abort;
		}
		_assert(select->Src);										// allocated even if there is no FROM clause
		if (select->Src->Srcs != 1) return false;					// FROM clause must have exactly one term
		if (select->Src->Ids[0].Select) return false;				// FROM clause cannot contain a subquery
		if (select->Where) return false;							// SELECT may not have a WHERE clause
		if (select->OrderBy) return false;							// SELECT may not have an ORDER BY clause
		// Do not need to test for a HAVING clause.  If HAVING is present but there is no ORDER BY, we will get an error.
		if (select->GroupBy) return false;							// SELECT may not have a GROUP BY clause
		if (select->Limit) return false;							// SELECT may not have a LIMIT clause
		_assert(select->Offset == nullptr);								// Must be so if pLimit==0
		if (select->Prior) return false;							// SELECT may not be a compound query
		if (select->SelFlags & SF_Distinct) return false;			// SELECT may not be DISTINCT
		ExprList *list = select->EList;							// The result set of the SELECT
		_assert(list != nullptr);
		if (list->Exprs != 1) return false;						// The result set must have exactly one column
		_assert(list->Ids[0].Expr);
		if (list->Ids[0].Expr->OP != TK_ALL) return false;			// The result set must be the special operator "*"

		// At this point we have established that the statement is of the correct syntactic form to participate in this optimization.  Now
		// we have to check the semantics.
		SrcList::SrcListItem *item = select->Src->Ids;				// An element of pSelect->pSrc
		Table *src = parse->LocateTableItem(false, item);			// The table in the FROM clause of SELECT
		if (!src) return false;										// FROM clause does not contain a real table
		if (src == dest) return false;								// tab1 and tab2 may not be the same table
#ifndef OMIT_VIRTUALTABLE
		if (src->TabFlags & TF_Virtual) return false;				// tab2 must not be a virtual table
#endif
		if (src->Select) return false;								// tab2 may not be a view
		if (dest->Cols.length != src->Cols.length) return false;	// Number of columns must be the same in tab1 and tab2
		if (dest->PKey != src->PKey) return false;					// Both tables must have the same INTEGER PRIMARY KEY
		for (int i = 0; i < dest->Cols.length; i++)
		{
			if (dest->Cols[i].Affinity != src->Cols[i].Affinity) return false; // Affinity must be the same on all columns
			if (!XferCompatibleCollation(dest->Cols[i].Coll, src->Cols[i].Coll)) return false; // Collating sequence must be the same on all columns
			if (dest->Cols[i].NotNull && !src->Cols[i].NotNull) return false; // tab2 must be NOT NULL if tab1 is
		}
		bool destHasUniqueIdx = false; // True if pDest has a UNIQUE index
		Index *srcIdx, *destIdx; // Source and destination indices
		for (destIdx = dest->Index; destIdx; destIdx = destIdx->Next)
		{
			if (destIdx->OnError != OE_None) destHasUniqueIdx = true;
			for (srcIdx = src->Index; srcIdx; srcIdx = srcIdx->Next)
				if (XferCompatibleIndex(destIdx, srcIdx)) break;
			if (!srcIdx) return false; // pDestIdx has no corresponding index in pSrc
		}
#ifndef OMIT_CHECK
		if (dest->Check && Expr::ListCompare(src->Check, dest->Check)) return false; // Tables have different CHECK constraints.  Ticket #2252
#endif
#ifndef OMIT_FOREIGN_KEY
		// Disallow the transfer optimization if the destination table constains any foreign key constraints.  This is more restrictive than necessary.
		// But the main beneficiary of the transfer optimization is the VACUUM command, and the VACUUM command disables foreign key constraints.  So
		// the extra complication to make this rule less restrictive is probably not worth the effort.  Ticket [6284df89debdfa61db8073e062908af0c9b6118e]
		if ((parse->Ctx->Flags & Context::FLAG_ForeignKeys) != 0 && dest->FKeys != nullptr) return false;
#endif
		if ((parse->Ctx->Flags & Context::FLAG_CountRows) != 0) return false;  // xfer opt does not play well with PRAGMA count_changes

		// If we get this far, it means that the xfer optimization is at least a possibility, though it might only work if the destination
		// table (tab1) is initially empty.
#ifdef TEST
		_xferopt_count++;
#endif
		int dbSrcId = Prepare::SchemaToIndex(parse->Ctx, src->Schema); // The database of pSrc
		Vdbe *v = parse->GetVdbe(); // The VDBE we are building
		parse->CodeVerifySchema(dbSrcId);
		int srcId = parse->Tabs++; // Cursors from source and destination
		int destId = parse->Tabs++; // Cursors from source and destination
		int regAutoinc = AutoIncBegin(parse, dbDestId, dest); // Memory register used by AUTOINC
		OpenTable(parse, destId, dbDestId, dest, OP_OpenWrite);
		int addr1; // Loop addresses
		int emptyDestTest; // Address of test for empty pDest
		if ((dest->PKey < 0 && dest->Index != nullptr) ||		// (1)
			destHasUniqueIdx ||									// (2)
			(onError != OE_Abort && onError != OE_Rollback))	// (3)
		{
			// In some circumstances, we are able to run the xfer optimization only if the destination table is initially empty.  This code makes
			// that determination.  Conditions under which the destination must be empty:
			//
			// (1) There is no INTEGER PRIMARY KEY but there are indices. (If the destination is not initially empty, the rowid fields
			//     of index entries might need to change.)
			//
			// (2) The destination has a unique index.  (The xfer optimization is unable to test uniqueness.)
			//
			// (3) onError is something other than OE_Abort and OE_Rollback.
			addr1 = v->AddOp2(OP_Rewind, destId, 0);
			emptyDestTest = v->AddOp2(OP_Goto, 0, 0);
			v->JumpHere(addr1);
		}
		else
			emptyDestTest = 0;
		OpenTable(parse, srcId, dbSrcId, src, OP_OpenRead);
		int emptySrcTest = v->AddOp2(OP_Rewind, srcId, 0); // Address of test for empty pSrc
		int regData = Expr::GetTempReg(parse); // Registers holding data and rowid
		int regRowid = Expr::GetTempReg(parse); // Registers holding data and rowid
		if (dest->PKey >= 0)
		{
			addr1 = v->AddOp2(OP_Rowid, srcId, regRowid);
			int addr2 = v->AddOp3(OP_NotExists, destId, 0, regRowid); // Loop addresses
			sqlite3HaltConstraint(parse, RC_CONSTRAINT_PRIMARYKEY, onError, "PRIMARY KEY must be unique", Vdbe::P4T_STATIC);
			v->JumpHere(addr2);
			AutoIncStep(parse, regAutoinc, regRowid);
		}
		else if (!dest->Index)
			addr1 = v->AddOp2(OP_NewRowid, destId, regRowid);
		else
		{
			addr1 = v->AddOp2(OP_Rowid, srcId, regRowid);
			_assert((dest->TabFlags & TF_Autoincrement) == 0);
		}
		v->AddOp2(OP_RowData, srcId, regData);
		v->AddOp3(OP_Insert, destId, regData, regRowid);
		v->ChangeP5(Vdbe::OPFLAG_NCHANGE|Vdbe::OPFLAG_LASTROWID|Vdbe::OPFLAG_APPEND);
		v->ChangeP4(-1, dest->Name, 0);
		v->AddOp2(OP_Next, srcId, addr1);
		for (destIdx = dest->Index; destIdx; destIdx = destIdx->Next)
		{
			for (srcIdx = src->Index; _ALWAYS(srcIdx); srcIdx = srcIdx->Next)
				if (XferCompatibleIndex(destIdx, srcIdx)) break;
			_assert(srcIdx);
			v->AddOp2(OP_Close, srcId, 0);
			v->AddOp2(OP_Close, destId, 0);
			KeyInfo *key = parse->IndexKeyinfo(srcIdx); // Key information for an index
			v->AddOp4(OP_OpenRead, srcId, srcIdx->Id, dbSrcId, (char *)key, Vdbe::P4T_KEYINFO_HANDOFF);
			v->Comment("%s", srcIdx->Name);
			key = parse->IndexKeyinfo(destIdx);
			v->AddOp4(OP_OpenWrite, destId, destIdx->Id, dbDestId, (char *)key, Vdbe::P4T_KEYINFO_HANDOFF);
			v->Comment("%s", destIdx->Name);
			addr1 = v->AddOp2(OP_Rewind, srcId, 0);
			v->AddOp2(OP_RowKey, srcId, regData);
			v->AddOp3(OP_IdxInsert, destId, regData, 1);
			v->AddOp2(OP_Next, srcId, addr1+1);
			v->JumpHere(addr1);
		}
		v->JumpHere(emptySrcTest);
		Expr::ReleaseTempReg(parse, regRowid);
		Expr::ReleaseTempReg(parse, regData);
		v->AddOp2(OP_Close, srcId, 0);
		v->AddOp2(OP_Close, destId, 0);
		if (emptyDestTest)
		{
			v->AddOp2(OP_Halt, RC_OK, 0);
			v->JumpHere(emptyDestTest);
			v->AddOp2(OP_Close, destId, 0);
			return false;
		}
		return true;
	}
#endif

}}