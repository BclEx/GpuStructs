#include "Core+Syntax.cu.h"

namespace Core
{

#define STRICMP(x, y) (__tolower(*(unsigned char *)(x))==__tolower(*(unsigned char *)(y))&&!_strcmp((x)+1,(y)+1))

	__device__ void Parse::BeginParse(bool explainFlag)
	{
		Explain = explainFlag;
		VarsSeen = 0;
	}

#ifndef OMIT_SHARED_CACHE
	struct TableLock
	{
		int DB;					// The database containing the table to be locked
		int Table;				// The root page of the table to be locked
		bool IsWriteLock;		// True for write lock.  False for a read lock
		const char *Name;		// Name of the table
	};

	__device__ void Parse::TableLock(int db, int table, bool isWriteLock, const char *name)
	{
		_assert(db >= 0);
		Parse *toplevel = Toplevel();
		Core::TableLock *tableLock;
		for (int i = 0; i < toplevel->TableLocks.length; i++)
		{
			tableLock = &toplevel->TableLocks[i];
			if (tableLock->DB == db && tableLock->Table == table)
			{
				tableLock->IsWriteLock |= isWriteLock;
				return;
			}
		}
		int bytes = sizeof(TableLock) * (toplevel->TableLocks.length + 1);
		toplevel->TableLocks = (Core::TableLock *)SysEx::TagRellocOrFree(toplevel->Ctx, toplevel->TableLocks, bytes);
		if (toplevel->TableLocks)
		{
			tableLock = &toplevel->TableLocks[toplevel->TableLocks.length++];
			tableLock->DB = db;
			tableLock->Table = table;
			tableLock->IsWriteLock = isWriteLock;
			tableLock->Name = name;
		}
		else
		{
			toplevel->TableLocks.length = 0;
			toplevel->Ctx->MallocFailed = true;
		}
	}

	__device__ static void CodeTableLocks(Parse *parse)
	{
		Vdbe *vdbe = parse->GetVdbe();
		_assert(vdbe); // sqlite3GetVdbe cannot fail: VDBE already allocated
		for (int i = 0; i < parse->TableLocks.length; i++)
		{
			TableLock *tableLock = &parse->TableLocks[i];
			int p1 = p->DB;
			Vdbe::AddOp4(vdbe, OP_TableLock, p1, tableLock->Table, tableLock->IsWriteLock, tableLock->Name, P4_STATIC);
		}
	}
#else
#define CodeTableLocks(x)
#endif

	__device__ void Parse::FinishCoding()
	{
		_assert(!Toplevel);
		Context *ctx = Ctx;
		if (ctx->MallocFailed || Nested || Errs) return;

		// Begin by generating some termination code at the end of the vdbe program
		Vdbe *v = GetVdbe();
		_assert(!IsMultiWrite || Vdbe::AssertMayAbort(v, MayAbort));
		if (v)
		{
			Vdbe::AddOp0(v, OP_Halt);

			// The cookie mask contains one bit for each database file open. (Bit 0 is for main, bit 1 is for temp, and so forth.)  Bits are
			// set for each database that is used.  Generate code to start a transaction on each used database and to verify the schema cookie
			// on each used database.
			if (CookieGoto > 0)
			{
				Vdbe::JumpHere(v, CookieGoto - 1);
				int db; yDbMask mask; 
				for (int db = 0, mask = 1; db < ctx->DBs.length; mask <<= 1, db++)
				{
					if ((mask & CookieMask) == 0)
						continue;
					Vdbe::UsesBtree(v, db);
					Vdbe::AddOp2(v, OP_Transaction, db, (mask & WriteMask) != 0);
					if (!ctx->Init.Busy)
					{
						_assert(Btree::SchemaMutexHeld(db, db, nullptr));
						Vdbe::AddOp3(v, OP_VerifyCookie, db, CookieValue[db], ctx->DBs[db].Schema->Generation);
					}
				}
#ifndef OMIT_VIRTUALTABLE
				{
					for (int i = 0; i < VTableLocks.length; i++)
					{
						char *vtable = (char *)VTable::GetVTable(ctx, VTableLocks[i]);
						sqlite3VdbeAddOp4(v, OP_VBegin, 0, 0, 0, vtable, P4_VTAB);
					}
					VTableLocks.length = 0;
				}
#endif

				// Once all the cookies have been verified and transactions opened, obtain the required table-locks. This is a no-op unless the 
				// shared-cache feature is enabled.
				CodeTableLocks(this);
				// Initialize any AUTOINCREMENT data structures required.
				sqlite3AutoincrementBegin(this);
				// Finally, jump back to the beginning of the executable code.
				Vdbe::AddOp2(v, OP_Goto, 0, CookieGoto);
			}
		}

		// Get the VDBE program ready for execution
		if (v && SysEx_ALWAYS(Errs == 0) && !ctx->MallocFailed)
		{
#ifdef DEBUG
			FILE *trace = ((ctx->Flags & BContext::FLAG_VdbeTrace) != 0 ? stdout : 0);
			Vdbe::Trace(v, trace);
#endif
			_assert(CacheLevel == 0);  // Disables and re-enables match
			// A minimum of one cursor is required if autoincrement is used See ticket [a696379c1f08866]
			if ( pParse->pAinc!=0 && pParse->nTab==0 ) pParse->nTab = 1;
			Vdbe::MakeReady(v, this);
			RC = RC_DONE;
			ColNamesSet = 0;
		}
		else
			RC = RC_ERROR;
		Tabs = 0;
		Mems = 0;
		Sets = 0;
		VarsSeen = 0;
		CookieMask = 0;
		CookieGoto = 0;
	}

	__device__ void Parse::NestedParse(const char *format, void *args)
	{
		if (Errs) return;
		_assert(Nested < 10); // Nesting should only be of limited depth
		//va_list ap;
		//va_start(ap, zFormat);
		//char *sql = sqlite3VMPrintf(Ctx, zFormat, ap);
		//va_end(ap);
		char *sql = nullptr;
		if (!sql)
			return; // A malloc must have failed
		Nested++;
#define SAVE_SZ (sizeof(Parse) - offsetof(Parse, VarsSeen))
		char saveBuf[SAVE_SZ];
		_memcpy(saveBuf, &VarsSeen, SAVE_SZ);
		_memset(&VarsSeen, 0, SAVE_SZ);
		char *errMsg = nullptr;
		RunParser(sql, &errMsg);
		SysEx::TagFree(db, errMsg);
		SysEx::TagFree(db, sql);
		_memcpy(&VarsSeen, saveBuf, SAVE_SZ);
		Nested--;
	}

	__device__ Table *Parse::FindTable(Context *ctx, const char *name, const char *database)
	{
		_assert(name);
		int nameLength = _strlen30(name);
		// All mutexes are required for schema access.  Make sure we hold them. */
		_assert(database || Btree::HoldsAllMutexes(ctx));
		Table *table = nullptr;
		for (int i = OMIT_TEMPDB; i < ctx->DBs.length; i++)
		{
			int j = (i < 2 ? i ^ 1 : i); // Search TEMP before MAIN
			if (database && _strcmp(database, ctx->DBs[j].Name))
				continue;
			_assert(Btree::SchemaMutexHeld(db, j, 0));
			table = (Table *)ctx->DBs[j].Schema->TableHash.Find(name, nameLength);
			if (table)
				break;
		}
		return table;
	}

	__device__ Table *Parse::LocateTable(bool isView, const char *name, const char *database)
	{
		// Read the database schema. If an error occurs, leave an error message and code in pParse and return NULL.
		if (sqlite3ReadSchema(pParse) != RC_OK)
			return nullptr;

		Table *table = FindTable(Ctx, name, database);
		if (!table)
		{
			const char *msg = (isView ? "no such view" : "no such table");
			if (database)
				ErrorMsg("%s: %s.%s", msg, database, name);
			else
				ErrorMsg("%s: %s", msg, name);
			CheckSchema = 1;
		}
		return table;
	}

	__device__ Table *Parse::LocateTableItem(bool isView,  SrcList_item *p)
	{
		_assert(!p->Schema || !p->Database);
		const char *database;
		if (p->Schema)
		{
			int db = sqlite3SchemaToIndex(Ctx, p->Schema);
			database = Ctx->DBs[db].Name;
		}
		else
			database = p->Database;
		return LocateTable(isView, p->Name, database);
	}

	__device__ Index *Parse::FindIndex(Context *ctx, const char *name, const char *database)
	{
		// All mutexes are required for schema access.  Make sure we hold them.
		_assert(database Btree::HoldsAllMutexes(ctx));
		Index *p = nullptr;
		int nameLength = _strlen30(name);
		for (int i = OMIT_TEMPDB; i < ctx->DBs.length; i++)
		{
			int j = (i < 2 ? i ^ 1 : i); // Search TEMP before MAIN
			Schema *schema = ctx->DBs[j].Schema;
			_assert(schema);
			if (database && _strcmp(database, ctx->DBs[j].Name))
				continue;
			_assert(Btree::SchemaMutexHeld(db, j, 0));
			p = (Index *)schema->IndexHash.Find(name, nameLength);
			if (p)
				break;
		}
		return p;
	}

	__device__ static void FreeIndex(Context *ctx, Index *index)
	{
#ifndef OMIT_ANALYZE
		sqlite3DeleteIndexSamples(ctx, index);
#endif
		SysEx::TagFree(ctx, index->ColAff);
		SysEx::TagFree(ctx, index);
	}

	__device__ void Parse::UnlinkAndDeleteIndex(Context *ctx, int db, const char *indexName)
	{
		_assert(Btree::SchemaMutexHeld(ctx, db, 0));
		Hash *hash = &ctx->DBs[db].Schema->IndexHash;
		int indexNameLength = _strlen30(indexName);
		Index *index = (Index *)hash->Insert(indexName, indexNameLength, 0);
		if (SysEx_ALWAYS(index))
		{
			if (index->Table->Index == index)
				index->Table->Index = index->Next;
			else
			{
				// Justification of ALWAYS();  The index must be on the list of indices.
				Index *p = index->Table->Index;
				while (SysEx_ALWAYS(p) && p->Next != index)
					p = p->Next;
				if (SysEx_ALWAYS(p && p->Next == index))
					p->Next = index->Next;
			}
			FreeIndex(ctx, index);
		}
		ctx->Flags |= Context::FLAG_InternChanges;
	}

	__device__ void Parse::CollapseDatabaseArray(Context *ctx)
	{
		int i, j;
		for (i = j = 2; i < ctx->DBs.length; i++)
		{
			Context::DB *db = &ctx->DBs[i];
			if (!db->Bt)
			{
				SysEx::TagFree(ctx, db->Name);
				db->Name = nullptr;
				continue;
			}
			if (j < i)
				ctx->DBs[j] = ctx->DBs[i];
			j++;
		}
		_memset(&ctx->DBs[j], 0, (ctx->DBs.length - j) * sizeof(ctx->DBs[0]));
		ctx->DBs.length = j;
		if (ctx->DBs.length <= 2 && ctx->DBs.data != ctx->StaticDBs)
		{
			_memcpy(ctx->StaticDBs, ctx->DBs.data, 2 * sizeof(ctx->DBs[0]));
			SysEx::TagFree(ctx, ctx->DBs);
			ctx->DBs = ctx->StaticDBs;
		}
	}

	__device__ void Parse::ResetOneSchema(Context *ctx, int db)
	{
		_assert(db < ctx->DBs.length);
		// Case 1:  Reset the single schema identified by iDb
		Context::DB *db2 = &ctx->DBs[db];
		_assert(Btree::SchemaMutexHeld(ctx, db, 0));
		_assert(db2->Schema);
		sqlite3SchemaClear(db2->Schema);
		// If any database other than TEMP is reset, then also reset TEMP since TEMP might be holding triggers that reference tables in the other database.
		if (db != 1)
		{
			db2 = &ctx->DBs[1];
			_assert(db2->Schema);
			sqlite3SchemaClear(db2->Schema);
		}
		return;
	}

	__device__ void Parse::ResetAllSchemasOfConnection(Context *ctx)
	{
		Btree::EnterAll(ctx);
		for (int i = 0; i < ctx->DBs.length; i++)
		{
			Context::DB *db = &ctx->DBs[i];
			if (db->Schema)
				sqlite3SchemaClear(db->Schema);
		}
		ctx->Flags &= ~Context::FLAG_InternChanges;
		VTable::UnlockList(ctx);
		Btree::LeaveAll(ctx);
		CollapseDatabaseArray(ctx);
	}

	__device__ void Parse::CommitInternalChanges(Context *ctx)
	{
		ctx->Flags &= ~Context::FLAG_InternChanges;
	}

	__device__ static void SqliteDeleteColumnNames(Context *ctx, Table *table)
	{
		_assert(table);
		Column *col;
		if ((col = table->Cols) != nullptr)
		{
			for (int i = 0; i < table->Cols.length; i++, col++)
			{
				SysEx::TagFree(ctx, col->Name);
				sqlite3ExprDelete(ctx, col->Dflt);
				SysEx::TagFree(ctx, col->DfltName);
				SysEx::TagFree(ctx, col->Type);
				SysEx::TagFree(ctx, col->Coll);
			}
			SysEx::TagFree(ctx, table->Cols);
		}
	}

	__device__ void Parse::DeleteTable(Context *ctx, Table *table)
	{
		_assert(!table || table->Refs > 0);

		// Do not delete the table until the reference count reaches zero.
		if (!table) return;
		if (((!ctx || ctx->BusyHandler == 0) && (--table->Refs) > 0)) return;

		// Record the number of outstanding lookaside allocations in schema Tables prior to doing any free() operations.  Since schema Tables do not use
		// lookaside, this number should not change.
#if !DEBUG
		int lookaside = (ctx && (table->TabFlags & TF_Ephemeral) == 0 ? ctx->Lookaside.nOut : 0); // Used to verify lookaside not used for schema 
#endif

		/* Delete all indices associated with this table. */
		Index *index, *next;
		for (index = table->Index; index; index = next)
		{
			next = index->Next;
			_assert(index->Schema == table->Schema);
			if (!ctx || ctx->BytesFreed == 0)
			{
				char *name = index->Name; 
				ASSERTONLY(Index *oldIndex = (Index *)) index->Schema->IndexHash.Insert(name, _strlen30(name), nullptr);
				_assert(!ctx || Btree::SchemaMutexHeld(ctx, 0, index->Schema));
				_assert(!oldIndex || oldIndex == index);
			}
			FreeIndex(ctx, index);
		}

		// Delete any foreign keys attached to this table.
		sqlite3FkDelete(ctx, table);

		// Delete the Table structure itself.
		sqliteDeleteColumnNames(ctx, table);
		SysEx::TagFree(ctx, table->Name);
		SysEx::TagFree(ctx, table->ColAff);
		sqlite3SelectDelete(db, pTable->pSelect);
#ifndef OMIT_CHECK
		sqlite3ExprListDelete(db, pTable->pCheck);
#endif
#ifndef OMIT_VIRTUALTABLE
		VTable::Clear(ctx, table);
#endif
		SysEx::TagFree(ctx, table);

		// Verify that no lookaside memory was used by schema tables
		_assert(lookaside == 0 || lookaside == ctx->Lookaside.nOut);
	}

	__device__ void Parse::UnlinkAndDeleteTable(Context *ctx, int db, const char *tableName)
	{
		_assert(ctx);
		_assert(db >= 0 && db < ctx->DBs.length);
		_assert(tableName);
		_assert(Btree::SchemaMutexHeld(ctx, db, nullptr));
		ASSERTCOVERAGE(tablenName[0] == nullptr);  // Zero-length table names are allowed
		Context::DB *db2 = &ctx->DBs[db];
		Table *table = (Table *)db2->Schema->TableHash.Insert(tableName, _strlen30(tableName), nullptr);
		sqlite3DeleteTable(ctx, table);
		ctx->Flags |= Context::FLAG_InternChanges;
	}

	__device__ char *Parse::NameFromToken(Context *ctx, Token *name)
	{
		if (name)
		{
			char *nameAsString = SysEx::TagStrNDup(ctx, name->data, name->length);
			sqlite3Dequote(nameAsString);
			return nameAsString;
		}
		return nullptr;
	}

	__device__ void Parse::OpenMasterTable(int db)
	{
		Vdbe *v = GetVdbe();
		TableLock(this, db, MASTER_ROOT, 1, SCHEMA_TABLE(db));
		Vdbe::AddOp3(v, OP_OpenWrite, 0, MASTER_ROOT, db);
		Vdbe::ChangeP4(v, -1, (char *)5, P4_INT32);  // 5 column table
		if (Tabs == 0)
			Tabs = 1;
	}

	__device__ int Parse::FindDbName(Context *ctx, const char *name)
	{
		int db = -1; // Database number
		if (name)
		{
			int nameLength = _strlen30(name);
			Context::DB *db2;
			for (db = (ctx->DBs.length - 1), db2 = &ctx->DBs[db]; db >= 0; db--, db2--)
				if ((!OMIT_TEMPDB || db != 1) && _strlen30(db2->Name) == nameLength && !_strcmp(db2->Name, name))
					break;
		}
		return db;
	}

	__device__ int Parse::FindDb(Context *ctx, Token *name)
	{
		char *nameAsString = NameFromToken(db, name); // Name we are searching for
		int db = FindDbName(ctx, nameAsString); // Database number                    
		SysEx::TagFree(ctx, nameAsString);
		return db;
	}

	__device__ int Parse::TwoPartName(Token *name1, Token *name2, Token **unqual)
	{
		Context *ctx = Ctx;
		int db; // Database holding the object
		if (SysEx_ALWAYS(name2 != nullptr) && name2->length > 0)
		{
			if (ctx->Init.Busy)
			{
				ErrorMsg("corrupt database");
				Errs++;
				return -1;
			}
			*unqual = name2;
			db = FindDb(ctx, name1);
			if (db < 0)
			{
				ErrorMsg("unknown database %T", name1);
				Errs++;
				return -1;
			}
		}
		else
		{
			_assert(ctx->Init.DB == 0 || ctx->Init.Busy);
			db = ctx->Init.DB;
			*unqual = name1;
		}
		return db;
	}

	__device__ RC Parse::CheckObjectName(const char *name)
	{
		if (!Ctx->Init.Busy && Nested == 0 && (Ctx->Flags & Context::FLAG_WriteSchema) == 0 && !_strnCmp(name, "sqlite_", 7))
		{
			ErrorMsg("object name reserved for internal use: %s", name);
			return RC_ERROR;
		}
		return RC_OK;
	}

	__device__ void Parse::StartTable(Token *name1, Token *name2, bool isTemp, bool isView, bool isVirtual, bool noErr)
	{
		// The table or view name to create is passed to this routine via tokens pName1 and pName2. If the table name was fully qualified, for example:
		//
		// CREATE TABLE xxx.yyy (...);
		// 
		// Then pName1 is set to "xxx" and pName2 "yyy". On the other hand if the table name is not fully qualified, i.e.:
		//
		// CREATE TABLE yyy(...);
		//
		// Then pName1 is set to "yyy" and pName2 is "".
		//
		// The call below sets the unqual pointer to point at the token (pName1 or pName2) that stores the unqualified table name. The variable iDb is
		// set to the index of the database that the table or view is to be created in.
		Token *unqual; // Unqualified name of the table to create
		int db = TwoPartName(name1, name2, &unqual); // Database number to create the table in
		if (db < 0)
			return;
		if (!OMIT_TEMPDB && isTemp && name2->length > 0 && db != 1)
		{
			// If creating a temp table, the name may not be qualified. Unless the database name is "temp" anyway. 
			ErrorMsg("temporary table name must be unqualified");
			return;
		}
		if (!OMIT_TEMPDB && isTemp)
			db = 1;
		NameToken = *unqual;
		Context *ctx = Ctx;
		char *name = NameFromToken(ctx, unqual); // The name of the new table
		if (!name)
			return;
		Vdbe *v;
		Table *table;
		if (CheckObjectName(name) != RC_OK)
			goto begin_table_error;
		if (ctx->Init.DB == 1)
			isTemp = 1;
#ifndef OMIT_AUTHORIZATION
		_assert((isTemp & 1) == isTemp);
		{
			int code;
			char *databaseName = ctx->DBs[db].Name;
			if (sqlite3AuthCheck(this, SQLITE_INSERT, SCHEMA_TABLE(isTemp), 0, databaseName))
				goto begin_table_error;
			if (isView)
				code = (!OMIT_TEMPDB && isTemp ? CREATE_TEMP_VIEW : CREATE_VIEW);
			else
				code = (!OMIT_TEMPDB && isTemp ? CREATE_TEMP_TABLE : CREATE_TABLE);
			if (!isVirtual && sqlite3AuthCheck(this, code, name, 0, databaseName))
				goto begin_table_error;
		}
#endif

		// Make sure the new table name does not collide with an existing index or table name in the same database.  Issue an error message if
		// it does. The exception is if the statement being parsed was passed to an sqlite3_declare_vtab() call. In that case only the column names
		// and types will be used, so there is no need to test for namespace collisions.
		if (!INDECLARE_VTABLE(this))
		{
			char *databaseName = ctx->DBs[db].Name;
			if (ReadSchema() != RC_OK)
				goto begin_table_error;
			table = FindTable(ctx, name, databaseName);
			if (table)
			{
				if (!noErr)
					ErrorMsg("table %T already exists", unqual);
				else
				{
					_assert(!ctx->Init.Busy);
					CodeVerifySchema(this, db);
				}
				goto begin_table_error;
			}
			if (FindIndex(ctx, name, databaseName))
			{
				ErrorMsg("there is already an index named %s", name);
				goto begin_table_error;
			}
		}

		table = SysEx::TagAlloc(ctx, sizeof(Table), true);
		if (!table)
		{
			ctx->MallocFailed = true;
			RC = Rc_NOMEM;
			Errs++;
			goto begin_table_error;
		}
		table->Name = name;
		table->PKey = -1;
		table->Schema = ctx->DBs[db].Schema;
		table->Refs = 1;
		table->RowEst = 1000000;
		_assert(!NewTable);
		NewTable = table;

		// If this is the magic sqlite_sequence table used by autoincrement, then record a pointer to this table in the main database structure
		// so that INSERT can find the table easily.
#ifndef OMIT_AUTOINCREMENT
		if (!Nested && !_strcmp(name, "sqlite_sequence"))
		{
			_assert(Btree::SchemaMutexHeld(ctx, db, nullptr));
			table->Schema->SeqTable = table;
		}
#endif

		// Begin generating the code that will insert the table record into the SQLITE_MASTER table.  Note in particular that we must go ahead
		// and allocate the record number for the table entry now.  Before any PRIMARY KEY or UNIQUE keywords are parsed.  Those keywords will cause
		// indices to be created and the table record must come before the indices.  Hence, the record number for the table must be allocated now.
		if (!ctx->Init.Busy && (v = GetVdbe()) != nullptr)
		{
			sqlite3BeginWriteOperation(this, 0, db);

#ifndef OMIT_VIRTUALTABLE
			if (isVirtual)
				Vdbe::AddOp0(v, OP_VBegin);
#endif

			// If the file format and encoding in the database have not been set, set them now.
			int reg1 = RegRowid = ++Mems;
			int reg2 = RegRoot = ++Mems;
			int reg3 = ++Mems;
			Vdbe::AddOp3(v, OP_ReadCookie, db, reg3, BTREE_FILE_FORMAT);
			Vdbe::UsesBtree(v, db);
			int j1 = Vdbe::AddOp1(v, OP_If, reg3);
			int fileFormat = ((ctx->Flags & Context::FLAG_LegacyFileFmt) != 0 ? 1 : MAX_FILE_FORMAT);
			Vdbe::AddOp2(v, OP_Integer, fileFormat, reg3);
			Vdbe::AddOp3(v, OP_SetCookie, db, BTREE_FILE_FORMAT, reg3);
			Vdbe::AddOp2(v, OP_Integer, ENC(ctx), reg3);
			Vdbe::AddOp3(v, OP_SetCookie, db, BTREE_TEXT_ENCODING, reg3);
			Vdbe::JumpHere(v, j1);

			// This just creates a place-holder record in the sqlite_master table. The record created does not contain anything yet.  It will be replaced
			// by the real entry in code generated at sqlite3EndTable().
			//
			// The rowid for the new entry is left in register pParse->regRowid. The root page number of the new table is left in reg pParse->regRoot.
			// The rowid and root page number values are needed by the code that sqlite3EndTable will generate.
#if !defined(OMIT_VIEW) || !defined(OMIT_VIRTUALTABLE)
			if (isView || isVirtual)
				Vdbe::AddOp2(v, OP_Integer, 0, reg2);
			else
#endif
				Vdbe::AddOp2(v, OP_CreateTable, db, reg2);
			OpenMasterTable(db);
			Vdbe::AddOp2(v, OP_NewRowid, 0, reg1);
			Vdbe::AddOp2(v, OP_Null, 0, reg3);
			Vdbe::AddOp3(v, OP_Insert, 0, reg3, reg1);
			Vdbe::ChangeP5(v, OPFLAG_APPEND);
			Vdbe::AddOp0(v, OP_Close);
		}
		return;

begin_table_error:
		SysEx::TagFree(ctx, name);
		return;
	}

	__device__ void Parser::AddColumn(Token *name)
	{
		Table *table;
		if ((table = NewTable) == nullptr)
			return;
		Context *ctx = Ctx;
#if MAX_COLUMN
		if (table->Cols.length + 1 > ctx->Limits[LIMIT_COLUMN])
		{
			ErrorMsg("too many columns on %s", table->Name);
			return;
		}
#endif
		char *nameAsString = NameFromToken(ctx, name);
		if (!nameAsString)
			return;
		for (int i = 0; i < table->Cols.length; i++)
		{
			if (STRICMP(nameAsString, table->Cols[i].Name))
			{
				ErrorMsg("duplicate column name: %s", nameAsString);
				SysEx::TagFree(ctx, nameAsString);
				return;
			}
		}
		if ((table->Cols.length & 0x7) == 0)
		{
			Column *newCols = (Column *)SysEx::TagRealloc(ctx, table->Cols, (table->Cols.length + 8) * sizeof(table->Cols[0]));
			if (!newCols)
			{
				SysEx::TagFree(ctx, nameAsString);
				return;
			}
			table->Cols = newCols;
		}
		Column *col = &table->Cols[table->Cols.length];
		_memset(col, 0, sizeof(table->Csol[0]));
		col->Name = nameAsStringz;

		// If there is no type specified, columns have the default affinity 'NONE'. If there is a type specified, then sqlite3AddColumnType() will
		// be called next to set pCol->affinity correctly.
		col->Affinity = AFF_NONE;
		table->Cols.length++;
	}

	__device__ void Parse::AddNotNull(uint8 onError)
	{
		Table *table = NewTable;
		if (!table || SysEx_NEVER(table->Cols.length < 1))
			return;
		table->Cols[table->Cols.length - 1].NotNull = onError;
	}

	__device__ AFF Parse::AffinityType(const char *data)
	{
		uint32 h = 0;
		AFF aff = AFF_NUMERIC;
		if (data)
			while (data[0])
			{
				h = (h << 8) + _tolower((*data) & 0xff);
				data++;
				if (h == (('c'<<24)+('h'<<16)+('a'<<8)+'r')) aff = AFF_TEXT; // CHAR
				else if (h == (('c'<<24)+('l'<<16)+('o'<<8)+'b')) aff = AFF_TEXT; // CLOB
				else if (h == (('t'<<24)+('e'<<16)+('x'<<8)+'t')) aff = AFF_TEXT; // TEXT
				else if (h == (('b'<<24)+('l'<<16)+('o'<<8)+'b') && (aff == AFF_NUMERIC || aff == AFF_REAL)) aff = AFF_NONE; // BLOB
#ifndef OMIT_FLOATING_POINT
				else if (h == (('r'<<24)+('e'<<16)+('a'<<8)+'l') && aff == AFF_NUMERIC) aff = AFF_REAL; // REAL
				else if (h == (('f'<<24)+('l'<<16)+('o'<<8)+'a') && aff == AFF_NUMERIC) aff = AFF_REAL; // FLOA
				else if (h == (('d'<<24)+('o'<<16)+('u'<<8)+'b') && aff == AFF_NUMERIC) aff = AFF_REAL; // DOUB
#endif
				else if ((h & 0x00FFFFFF) == (('i'<<16)+('n'<<8)+'t')) { aff = AFF_INTEGER; break; }  // INT
			}
			return aff;
	}

	__device__ void Parse::AddColumnType(Token *type)
	{
		Table *table = NewTable;
		if (!table || SysEx_NEVER(table->Cols.length < 1))
			return;
		Column *col = &table->Cols[table->Cols.length - 1];
		_assert(col->Type == nullptr);
		col->Type = NameFromToken(Ctx, type);
		col->Affinity = AffinityType(col->Type);
	}

	__device__ void Parse::AddDefaultValue(ExprSpan *span)
	{
		Context *ctx = Ctx;
		Table *table = NewTable;
		if (table)
		{
			Column *col = &(table->Cols[table->Cols.length - 1]);
			if (!Expr::IsConstantOrFunction(span->Expr))
				ErrorMsg("default value of column [%s] is not constant", col->Name);
			else
			{
				// A copy of pExpr is used instead of the original, as pExpr contains tokens that point to volatile memory. The 'span' of the expression
				// is required by pragma table_info.
				Expr::Delete(ctx, col->Dflt);
				col->Dflt = Expr::ExprDup(ctx, span->Expr, EXPRDUP_REDUCE);
				SysEx::TagFree(ctx, col->DfltName);
				col->DfltName = SysEx::TagStrNDup(ctx, (char *)span->Start, (int)(span->End - span->Start));
			}
		}
		Expr::Delete(ctx, span->Expr);
	}

	__device__ void Parse::AddPrimaryKey(ExprList *list, uint8 onError, bool autoInc, int sortOrder)
	{
		Table *table = NewTable;
		if (!table || INDECLARE_VTABLE(this))
			goto primary_key_exit;
		if ((table->TabFlags & TF_HasPrimaryKey) != 0)
		{
			ErrorMsg("table \"%s\" has more than one primary key", table->Name);
			goto primary_key_exit;
		}
		table->TabFlags |= TF_HasPrimaryKey;
		int col = -1;
		if (!list)
		{
			col = table->Cols.length - 1;
			table->Cols[col].ColFlags |= COLFLAG_PRIMKEY;
		}
		else
		{
			for (int i = 0; i < list->nExpr; i++)
			{
				for (col = 0; col < table->Cols.length; col++)
					if (!_strcmp(list->a[i].Name, table->Cols[col].Name))
						break;
				if (col < table->Cols.length)
					table->Cols[col].ColFlags |= COLFLAG_PRIMKEY;
			}
			if (list->nExpr > 1)
				col = -1;
		}
		char *type = nullptr;
		if (col >= 0 && col < table->Cols.length)
			type = table->Cols[col].Type;
		if (type && !_strcmp(type, "INTEGER") && sortOrder == SO_ASC)
		{
			table->PKey = col;
			table->KeyConf = onError;
			table->TabFlags |= (autoInc ? TF_Autoincrement : 0);
		}
		else if (autoInc)
		{
#ifndef OMIT_AUTOINCREMENT
			ErrorMsg("AUTOINCREMENT is only allowed on an INTEGER PRIMARY KEY");
#endif
		}
		else
		{
			Index *index = CreateIndex(0, 0, 0, list, onError, 0, 0, sortOrder, 0);
			if (index)
				index->AutoIndex = 2;
			list = nullptr;
		}
primary_key_exit:
		Expr::ListDelete(Ctx, list);
		return;
	}

	__device__ void Parse::AddCheckConstraint(Expr *checkExpr)
	{
#ifndef OMIT_CHECK
		Table *table = NewTable;
		if (table && !INDECLARE_VTABLE(this))
		{
			table->Check = Expr::ListAppend(this, table->Check, checkExpr);
			if (ConstraintName.length)
				Expr::ListSetName(this, table->Check, &ConstraintName, 1);
		}
		else
#endif
			Expr::Delete(Ctx, checkExpr);
	}

	__device__ void Parse::AddCollateType(Token *token)
	{
		Table *table;
		if (!(table = NewTable))
			return;
		int col = table->Cols.length - 1;
		Context *ctx = Ctx;
		char *collName = NameFromToken(ctx, token); // Dequoted name of collation sequence
		if (!collName)
			return;
		if (sqlite3LocateCollSeq(this, collName))
		{
			table->Cols[col].Coll = collName;
			// If the column is declared as "<name> PRIMARY KEY COLLATE <type>", then an index may have been created on this column before the
			// collation type was added. Correct this if it is the case.
			for (Index *index = table->Index; index; index = index->Next)
			{
				_assert(index->Column == 1);
				if (index->Columns[0] == i)
					index->CollNames[0] = table->Cols[i].Coll;
			}
		}
		else
			SysEx::TagFree(ctx, collName);
	}

	__device__ CollSeq *Parse::LocateCollSeq(const char *name)
	{
		Context *ctx = Ctx;
		uint8 enc = ENC(ctx);
		bool initBusy = ctx->Init.Busy;
		CollSeq *coll = FindCollSeq(ctx, enc, name, initBusy);
		if (!initBusy && (!coll || !coll->Cmp))
			coll = GetCollSeq(this, enc, coll, name);
		return coll;
	}

	__device__ void Parse::ChangeCookie(int db)
	{
		int r1 = GetTempReg();
		Context *ctx = Ctx;
		Vdbe *v = Vdbe;
		_assert(Btree::SchemaMutexHeld(ctx, db, nullptr));
		Vdbe::AddOp2(v, OP_Integer, ctx->DBs[db].Schema->SchemaCookie + 1, r1);
		Vdbe::AddOp3(v, OP_SetCookie, db, BTREE_SCHEMA_VERSION, r1);
		ReleaseTempReg(r1);
	}

	__device__ static int IdentLength(const char *z)
	{
		int size;
		for (size = 0; *z; size++, z++)
			if (*z == '"')
				size++;
		return size + 2;
	}

	/*
	** The first parameter is a pointer to an output buffer. The second 
	** parameter is a pointer to an integer that contains the offset at
	** which to write into the output buffer. This function copies the
	** nul-terminated string pointed to by the third parameter, zSignedIdent,
	** to the specified offset in the buffer and updates *pIdx to refer
	** to the first byte after the last byte written before returning.
	** 
	** If the string zSignedIdent consists entirely of alpha-numeric
	** characters, does not begin with a digit and is not an SQL keyword,
	** then it is copied to the output buffer exactly as it is. Otherwise,
	** it is quoted using double-quotes.
	*/
	static void IdentPut(char *z, int *pIdx, char *zSignedIdent)
	{
		unsigned char *zIdent = (unsigned char*)zSignedIdent;
		int i, j, needQuote;
		i = *pIdx;

		for(j=0; zIdent[j]; j++){
			if( !sqlite3Isalnum(zIdent[j]) && zIdent[j]!='_' ) break;
		}
		needQuote = sqlite3Isdigit(zIdent[0]) || sqlite3KeywordCode(zIdent, j)!=TK_ID;
		if( !needQuote ){
			needQuote = zIdent[j];
		}

		if( needQuote ) z[i++] = '"';
		for(j=0; zIdent[j]; j++){
			z[i++] = zIdent[j];
			if( zIdent[j]=='"' ) z[i++] = '"';
		}
		if( needQuote ) z[i++] = '"';
		z[i] = 0;
		*pIdx = i;
	}

	/*
	** Generate a CREATE TABLE statement appropriate for the given
	** table.  Memory to hold the text of the statement is obtained
	** from sqliteMalloc() and must be freed by the calling function.
	*/
	static char *createTableStmt(sqlite3 *db, Table *p){
		int i, k, n;
		char *zStmt;
		char *zSep, *zSep2, *zEnd;
		Column *pCol;
		n = 0;
		for(pCol = p->aCol, i=0; i<p->nCol; i++, pCol++){
			n += identLength(pCol->zName) + 5;
		}
		n += identLength(p->zName);
		if( n<50 ){ 
			zSep = "";
			zSep2 = ",";
			zEnd = ")";
		}else{
			zSep = "\n  ";
			zSep2 = ",\n  ";
			zEnd = "\n)";
		}
		n += 35 + 6*p->nCol;
		zStmt = sqlite3DbMallocRaw(0, n);
		if( zStmt==0 ){
			db->mallocFailed = 1;
			return 0;
		}
		sqlite3_snprintf(n, zStmt, "CREATE TABLE ");
		k = sqlite3Strlen30(zStmt);
		identPut(zStmt, &k, p->zName);
		zStmt[k++] = '(';
		for(pCol=p->aCol, i=0; i<p->nCol; i++, pCol++){
			static const char * const azType[] = {
				/* SQLITE_AFF_TEXT    */ " TEXT",
				/* SQLITE_AFF_NONE    */ "",
				/* SQLITE_AFF_NUMERIC */ " NUM",
				/* SQLITE_AFF_INTEGER */ " INT",
				/* SQLITE_AFF_REAL    */ " REAL"
			};
			int len;
			const char *zType;

			sqlite3_snprintf(n-k, &zStmt[k], zSep);
			k += sqlite3Strlen30(&zStmt[k]);
			zSep = zSep2;
			identPut(zStmt, &k, pCol->zName);
			assert( pCol->affinity-SQLITE_AFF_TEXT >= 0 );
			assert( pCol->affinity-SQLITE_AFF_TEXT < ArraySize(azType) );
			testcase( pCol->affinity==SQLITE_AFF_TEXT );
			testcase( pCol->affinity==SQLITE_AFF_NONE );
			testcase( pCol->affinity==SQLITE_AFF_NUMERIC );
			testcase( pCol->affinity==SQLITE_AFF_INTEGER );
			testcase( pCol->affinity==SQLITE_AFF_REAL );

			zType = azType[pCol->affinity - SQLITE_AFF_TEXT];
			len = sqlite3Strlen30(zType);
			assert( pCol->affinity==SQLITE_AFF_NONE 
				|| pCol->affinity==sqlite3AffinityType(zType) );
			memcpy(&zStmt[k], zType, len);
			k += len;
			assert( k<=n );
		}
		sqlite3_snprintf(n-k, &zStmt[k], "%s", zEnd);
		return zStmt;
	}

	/*
	** This routine is called to report the final ")" that terminates
	** a CREATE TABLE statement.
	**
	** The table structure that other action routines have been building
	** is added to the internal hash tables, assuming no errors have
	** occurred.
	**
	** An entry for the table is made in the master table on disk, unless
	** this is a temporary table or db->init.busy==1.  When db->init.busy==1
	** it means we are reading the sqlite_master table because we just
	** connected to the database or because the sqlite_master table has
	** recently changed, so the entry for this table already exists in
	** the sqlite_master table.  We do not want to create it again.
	**
	** If the pSelect argument is not NULL, it means that this routine
	** was called to create a table generated from a 
	** "CREATE TABLE ... AS SELECT ..." statement.  The column names of
	** the new table will match the result set of the SELECT.
	*/
	void sqlite3EndTable(
		Parse *pParse,          /* Parse context */
		Token *pCons,           /* The ',' token after the last column defn. */
		Token *pEnd,            /* The final ')' token in the CREATE TABLE */
		Select *pSelect         /* Select from a "CREATE ... AS SELECT" */
		){
			Table *p;
			sqlite3 *db = pParse->db;
			int iDb;

			if( (pEnd==0 && pSelect==0) || db->mallocFailed ){
				return;
			}
			p = pParse->pNewTable;
			if( p==0 ) return;

			assert( !db->init.busy || !pSelect );

			iDb = sqlite3SchemaToIndex(db, p->pSchema);

#ifndef SQLITE_OMIT_CHECK
			/* Resolve names in all CHECK constraint expressions.
			*/
			if( p->pCheck ){
				SrcList sSrc;                   /* Fake SrcList for pParse->pNewTable */
				NameContext sNC;                /* Name context for pParse->pNewTable */
				ExprList *pList;                /* List of all CHECK constraints */
				int i;                          /* Loop counter */

				memset(&sNC, 0, sizeof(sNC));
				memset(&sSrc, 0, sizeof(sSrc));
				sSrc.nSrc = 1;
				sSrc.a[0].zName = p->zName;
				sSrc.a[0].pTab = p;
				sSrc.a[0].iCursor = -1;
				sNC.pParse = pParse;
				sNC.pSrcList = &sSrc;
				sNC.ncFlags = NC_IsCheck;
				pList = p->pCheck;
				for(i=0; i<pList->nExpr; i++){
					if( sqlite3ResolveExprNames(&sNC, pList->a[i].pExpr) ){
						return;
					}
				}
			}
#endif /* !defined(SQLITE_OMIT_CHECK) */

			/* If the db->init.busy is 1 it means we are reading the SQL off the
			** "sqlite_master" or "sqlite_temp_master" table on the disk.
			** So do not write to the disk again.  Extract the root page number
			** for the table from the db->init.newTnum field.  (The page number
			** should have been put there by the sqliteOpenCb routine.)
			*/
			if( db->init.busy ){
				p->tnum = db->init.newTnum;
			}

			/* If not initializing, then create a record for the new table
			** in the SQLITE_MASTER table of the database.
			**
			** If this is a TEMPORARY table, write the entry into the auxiliary
			** file instead of into the main database file.
			*/
			if( !db->init.busy ){
				int n;
				Vdbe *v;
				char *zType;    /* "view" or "table" */
				char *zType2;   /* "VIEW" or "TABLE" */
				char *zStmt;    /* Text of the CREATE TABLE or CREATE VIEW statement */

				v = sqlite3GetVdbe(pParse);
				if( NEVER(v==0) ) return;

				sqlite3VdbeAddOp1(v, OP_Close, 0);

				/* 
				** Initialize zType for the new view or table.
				*/
				if( p->pSelect==0 ){
					/* A regular table */
					zType = "table";
					zType2 = "TABLE";
#ifndef SQLITE_OMIT_VIEW
				}else{
					/* A view */
					zType = "view";
					zType2 = "VIEW";
#endif
				}

				/* If this is a CREATE TABLE xx AS SELECT ..., execute the SELECT
				** statement to populate the new table. The root-page number for the
				** new table is in register pParse->regRoot.
				**
				** Once the SELECT has been coded by sqlite3Select(), it is in a
				** suitable state to query for the column names and types to be used
				** by the new table.
				**
				** A shared-cache write-lock is not required to write to the new table,
				** as a schema-lock must have already been obtained to create it. Since
				** a schema-lock excludes all other database users, the write-lock would
				** be redundant.
				*/
				if( pSelect ){
					SelectDest dest;
					Table *pSelTab;

					assert(pParse->nTab==1);
					sqlite3VdbeAddOp3(v, OP_OpenWrite, 1, pParse->regRoot, iDb);
					sqlite3VdbeChangeP5(v, OPFLAG_P2ISREG);
					pParse->nTab = 2;
					sqlite3SelectDestInit(&dest, SRT_Table, 1);
					sqlite3Select(pParse, pSelect, &dest);
					sqlite3VdbeAddOp1(v, OP_Close, 1);
					if( pParse->nErr==0 ){
						pSelTab = sqlite3ResultSetOfSelect(pParse, pSelect);
						if( pSelTab==0 ) return;
						assert( p->aCol==0 );
						p->nCol = pSelTab->nCol;
						p->aCol = pSelTab->aCol;
						pSelTab->nCol = 0;
						pSelTab->aCol = 0;
						sqlite3DeleteTable(db, pSelTab);
					}
				}

				/* Compute the complete text of the CREATE statement */
				if( pSelect ){
					zStmt = createTableStmt(db, p);
				}else{
					n = (int)(pEnd->z - pParse->sNameToken.z) + 1;
					zStmt = sqlite3MPrintf(db, 
						"CREATE %s %.*s", zType2, n, pParse->sNameToken.z
						);
				}

				/* A slot for the record has already been allocated in the 
				** SQLITE_MASTER table.  We just need to update that slot with all
				** the information we've collected.
				*/
				sqlite3NestedParse(pParse,
					"UPDATE %Q.%s "
					"SET type='%s', name=%Q, tbl_name=%Q, rootpage=#%d, sql=%Q "
					"WHERE rowid=#%d",
					db->aDb[iDb].zName, SCHEMA_TABLE(iDb),
					zType,
					p->zName,
					p->zName,
					pParse->regRoot,
					zStmt,
					pParse->regRowid
					);
				sqlite3DbFree(db, zStmt);
				sqlite3ChangeCookie(pParse, iDb);

#ifndef SQLITE_OMIT_AUTOINCREMENT
				/* Check to see if we need to create an sqlite_sequence table for
				** keeping track of autoincrement keys.
				*/
				if( p->tabFlags & TF_Autoincrement ){
					Db *pDb = &db->aDb[iDb];
					assert( sqlite3SchemaMutexHeld(db, iDb, 0) );
					if( pDb->pSchema->pSeqTab==0 ){
						sqlite3NestedParse(pParse,
							"CREATE TABLE %Q.sqlite_sequence(name,seq)",
							pDb->zName
							);
					}
				}
#endif

				/* Reparse everything to update our internal data structures */
				sqlite3VdbeAddParseSchemaOp(v, iDb,
					sqlite3MPrintf(db, "tbl_name='%q'", p->zName));
			}


			/* Add the table to the in-memory representation of the database.
			*/
			if( db->init.busy ){
				Table *pOld;
				Schema *pSchema = p->pSchema;
				assert( sqlite3SchemaMutexHeld(db, iDb, 0) );
				pOld = sqlite3HashInsert(&pSchema->tblHash, p->zName,
					sqlite3Strlen30(p->zName),p);
				if( pOld ){
					assert( p==pOld );  /* Malloc must have failed inside HashInsert() */
					db->mallocFailed = 1;
					return;
				}
				pParse->pNewTable = 0;
				db->flags |= SQLITE_InternChanges;

#ifndef SQLITE_OMIT_ALTERTABLE
				if( !p->pSelect ){
					const char *zName = (const char *)pParse->sNameToken.z;
					int nName;
					assert( !pSelect && pCons && pEnd );
					if( pCons->z==0 ){
						pCons = pEnd;
					}
					nName = (int)((const char *)pCons->z - zName);
					p->addColOffset = 13 + sqlite3Utf8CharLen(zName, nName);
				}
#endif
			}
	}

#ifndef SQLITE_OMIT_VIEW
	/*
	** The parser calls this routine in order to create a new VIEW
	*/
	void sqlite3CreateView(
		Parse *pParse,     /* The parsing context */
		Token *pBegin,     /* The CREATE token that begins the statement */
		Token *pName1,     /* The token that holds the name of the view */
		Token *pName2,     /* The token that holds the name of the view */
		Select *pSelect,   /* A SELECT statement that will become the new view */
		int isTemp,        /* TRUE for a TEMPORARY view */
		int noErr          /* Suppress error messages if VIEW already exists */
		){
			Table *p;
			int n;
			const char *z;
			Token sEnd;
			DbFixer sFix;
			Token *pName = 0;
			int iDb;
			sqlite3 *db = pParse->db;

			if( pParse->nVar>0 ){
				sqlite3ErrorMsg(pParse, "parameters are not allowed in views");
				sqlite3SelectDelete(db, pSelect);
				return;
			}
			sqlite3StartTable(pParse, pName1, pName2, isTemp, 1, 0, noErr);
			p = pParse->pNewTable;
			if( p==0 || pParse->nErr ){
				sqlite3SelectDelete(db, pSelect);
				return;
			}
			sqlite3TwoPartName(pParse, pName1, pName2, &pName);
			iDb = sqlite3SchemaToIndex(db, p->pSchema);
			if( sqlite3FixInit(&sFix, pParse, iDb, "view", pName)
				&& sqlite3FixSelect(&sFix, pSelect)
				){
					sqlite3SelectDelete(db, pSelect);
					return;
			}

			/* Make a copy of the entire SELECT statement that defines the view.
			** This will force all the Expr.token.z values to be dynamically
			** allocated rather than point to the input string - which means that
			** they will persist after the current sqlite3_exec() call returns.
			*/
			p->pSelect = sqlite3SelectDup(db, pSelect, EXPRDUP_REDUCE);
			sqlite3SelectDelete(db, pSelect);
			if( db->mallocFailed ){
				return;
			}
			if( !db->init.busy ){
				sqlite3ViewGetColumnNames(pParse, p);
			}

			/* Locate the end of the CREATE VIEW statement.  Make sEnd point to
			** the end.
			*/
			sEnd = pParse->sLastToken;
			if( ALWAYS(sEnd.z[0]!=0) && sEnd.z[0]!=';' ){
				sEnd.z += sEnd.n;
			}
			sEnd.n = 0;
			n = (int)(sEnd.z - pBegin->z);
			z = pBegin->z;
			while( ALWAYS(n>0) && sqlite3Isspace(z[n-1]) ){ n--; }
			sEnd.z = &z[n-1];
			sEnd.n = 1;

			/* Use sqlite3EndTable() to add the view to the SQLITE_MASTER table */
			sqlite3EndTable(pParse, 0, &sEnd, 0);
			return;
	}
#endif /* SQLITE_OMIT_VIEW */

#if !defined(SQLITE_OMIT_VIEW) || !defined(SQLITE_OMIT_VIRTUALTABLE)
	/*
	** The Table structure pTable is really a VIEW.  Fill in the names of
	** the columns of the view in the pTable structure.  Return the number
	** of errors.  If an error is seen leave an error message in pParse->zErrMsg.
	*/
	int sqlite3ViewGetColumnNames(Parse *pParse, Table *pTable){
		Table *pSelTab;   /* A fake table from which we get the result set */
		Select *pSel;     /* Copy of the SELECT that implements the view */
		int nErr = 0;     /* Number of errors encountered */
		int n;            /* Temporarily holds the number of cursors assigned */
		sqlite3 *db = pParse->db;  /* Database connection for malloc errors */
		int (*xAuth)(void*,int,const char*,const char*,const char*,const char*);

		assert( pTable );

#ifndef SQLITE_OMIT_VIRTUALTABLE
		if( sqlite3VtabCallConnect(pParse, pTable) ){
			return SQLITE_ERROR;
		}
		if( IsVirtual(pTable) ) return 0;
#endif

#ifndef SQLITE_OMIT_VIEW
		/* A positive nCol means the columns names for this view are
		** already known.
		*/
		if( pTable->nCol>0 ) return 0;

		/* A negative nCol is a special marker meaning that we are currently
		** trying to compute the column names.  If we enter this routine with
		** a negative nCol, it means two or more views form a loop, like this:
		**
		**     CREATE VIEW one AS SELECT * FROM two;
		**     CREATE VIEW two AS SELECT * FROM one;
		**
		** Actually, the error above is now caught prior to reaching this point.
		** But the following test is still important as it does come up
		** in the following:
		** 
		**     CREATE TABLE main.ex1(a);
		**     CREATE TEMP VIEW ex1 AS SELECT a FROM ex1;
		**     SELECT * FROM temp.ex1;
		*/
		if( pTable->nCol<0 ){
			sqlite3ErrorMsg(pParse, "view %s is circularly defined", pTable->zName);
			return 1;
		}
		assert( pTable->nCol>=0 );

		/* If we get this far, it means we need to compute the table names.
		** Note that the call to sqlite3ResultSetOfSelect() will expand any
		** "*" elements in the results set of the view and will assign cursors
		** to the elements of the FROM clause.  But we do not want these changes
		** to be permanent.  So the computation is done on a copy of the SELECT
		** statement that defines the view.
		*/
		assert( pTable->pSelect );
		pSel = sqlite3SelectDup(db, pTable->pSelect, 0);
		if( pSel ){
			u8 enableLookaside = db->lookaside.bEnabled;
			n = pParse->nTab;
			sqlite3SrcListAssignCursors(pParse, pSel->pSrc);
			pTable->nCol = -1;
			db->lookaside.bEnabled = 0;
#ifndef SQLITE_OMIT_AUTHORIZATION
			xAuth = db->xAuth;
			db->xAuth = 0;
			pSelTab = sqlite3ResultSetOfSelect(pParse, pSel);
			db->xAuth = xAuth;
#else
			pSelTab = sqlite3ResultSetOfSelect(pParse, pSel);
#endif
			db->lookaside.bEnabled = enableLookaside;
			pParse->nTab = n;
			if( pSelTab ){
				assert( pTable->aCol==0 );
				pTable->nCol = pSelTab->nCol;
				pTable->aCol = pSelTab->aCol;
				pSelTab->nCol = 0;
				pSelTab->aCol = 0;
				sqlite3DeleteTable(db, pSelTab);
				assert( sqlite3SchemaMutexHeld(db, 0, pTable->pSchema) );
				pTable->pSchema->flags |= DB_UnresetViews;
			}else{
				pTable->nCol = 0;
				nErr++;
			}
			sqlite3SelectDelete(db, pSel);
		} else {
			nErr++;
		}
#endif /* SQLITE_OMIT_VIEW */
		return nErr;  
	}
#endif /* !defined(SQLITE_OMIT_VIEW) || !defined(SQLITE_OMIT_VIRTUALTABLE) */

#ifndef SQLITE_OMIT_VIEW
	/*
	** Clear the column names from every VIEW in database idx.
	*/
	static void sqliteViewResetAll(sqlite3 *db, int idx){
		HashElem *i;
		assert( sqlite3SchemaMutexHeld(db, idx, 0) );
		if( !DbHasProperty(db, idx, DB_UnresetViews) ) return;
		for(i=sqliteHashFirst(&db->aDb[idx].pSchema->tblHash); i;i=sqliteHashNext(i)){
			Table *pTab = sqliteHashData(i);
			if( pTab->pSelect ){
				sqliteDeleteColumnNames(db, pTab);
				pTab->aCol = 0;
				pTab->nCol = 0;
			}
		}
		DbClearProperty(db, idx, DB_UnresetViews);
	}
#else
# define sqliteViewResetAll(A,B)
#endif /* SQLITE_OMIT_VIEW */

	/*
	** This function is called by the VDBE to adjust the internal schema
	** used by SQLite when the btree layer moves a table root page. The
	** root-page of a table or index in database iDb has changed from iFrom
	** to iTo.
	**
	** Ticket #1728:  The symbol table might still contain information
	** on tables and/or indices that are the process of being deleted.
	** If you are unlucky, one of those deleted indices or tables might
	** have the same rootpage number as the real table or index that is
	** being moved.  So we cannot stop searching after the first match 
	** because the first match might be for one of the deleted indices
	** or tables and not the table/index that is actually being moved.
	** We must continue looping until all tables and indices with
	** rootpage==iFrom have been converted to have a rootpage of iTo
	** in order to be certain that we got the right one.
	*/
#ifndef SQLITE_OMIT_AUTOVACUUM
	void sqlite3RootPageMoved(sqlite3 *db, int iDb, int iFrom, int iTo){
		HashElem *pElem;
		Hash *pHash;
		Db *pDb;

		assert( sqlite3SchemaMutexHeld(db, iDb, 0) );
		pDb = &db->aDb[iDb];
		pHash = &pDb->pSchema->tblHash;
		for(pElem=sqliteHashFirst(pHash); pElem; pElem=sqliteHashNext(pElem)){
			Table *pTab = sqliteHashData(pElem);
			if( pTab->tnum==iFrom ){
				pTab->tnum = iTo;
			}
		}
		pHash = &pDb->pSchema->idxHash;
		for(pElem=sqliteHashFirst(pHash); pElem; pElem=sqliteHashNext(pElem)){
			Index *pIdx = sqliteHashData(pElem);
			if( pIdx->tnum==iFrom ){
				pIdx->tnum = iTo;
			}
		}
	}
#endif

	/*
	** Write code to erase the table with root-page iTable from database iDb.
	** Also write code to modify the sqlite_master table and internal schema
	** if a root-page of another table is moved by the btree-layer whilst
	** erasing iTable (this can happen with an auto-vacuum database).
	*/ 
	static void destroyRootPage(Parse *pParse, int iTable, int iDb){
		Vdbe *v = sqlite3GetVdbe(pParse);
		int r1 = sqlite3GetTempReg(pParse);
		sqlite3VdbeAddOp3(v, OP_Destroy, iTable, r1, iDb);
		sqlite3MayAbort(pParse);
#ifndef SQLITE_OMIT_AUTOVACUUM
		/* OP_Destroy stores an in integer r1. If this integer
		** is non-zero, then it is the root page number of a table moved to
		** location iTable. The following code modifies the sqlite_master table to
		** reflect this.
		**
		** The "#NNN" in the SQL is a special constant that means whatever value
		** is in register NNN.  See grammar rules associated with the TK_REGISTER
		** token for additional information.
		*/
		sqlite3NestedParse(pParse, 
			"UPDATE %Q.%s SET rootpage=%d WHERE #%d AND rootpage=#%d",
			pParse->db->aDb[iDb].zName, SCHEMA_TABLE(iDb), iTable, r1, r1);
#endif
		sqlite3ReleaseTempReg(pParse, r1);
	}

	/*
	** Write VDBE code to erase table pTab and all associated indices on disk.
	** Code to update the sqlite_master tables and internal schema definitions
	** in case a root-page belonging to another table is moved by the btree layer
	** is also added (this can happen with an auto-vacuum database).
	*/
	static void destroyTable(Parse *pParse, Table *pTab){
#ifdef SQLITE_OMIT_AUTOVACUUM
		Index *pIdx;
		int iDb = sqlite3SchemaToIndex(pParse->db, pTab->pSchema);
		destroyRootPage(pParse, pTab->tnum, iDb);
		for(pIdx=pTab->pIndex; pIdx; pIdx=pIdx->pNext){
			destroyRootPage(pParse, pIdx->tnum, iDb);
		}
#else
		/* If the database may be auto-vacuum capable (if SQLITE_OMIT_AUTOVACUUM
		** is not defined), then it is important to call OP_Destroy on the
		** table and index root-pages in order, starting with the numerically 
		** largest root-page number. This guarantees that none of the root-pages
		** to be destroyed is relocated by an earlier OP_Destroy. i.e. if the
		** following were coded:
		**
		** OP_Destroy 4 0
		** ...
		** OP_Destroy 5 0
		**
		** and root page 5 happened to be the largest root-page number in the
		** database, then root page 5 would be moved to page 4 by the 
		** "OP_Destroy 4 0" opcode. The subsequent "OP_Destroy 5 0" would hit
		** a free-list page.
		*/
		int iTab = pTab->tnum;
		int iDestroyed = 0;

		while( 1 ){
			Index *pIdx;
			int iLargest = 0;

			if( iDestroyed==0 || iTab<iDestroyed ){
				iLargest = iTab;
			}
			for(pIdx=pTab->pIndex; pIdx; pIdx=pIdx->pNext){
				int iIdx = pIdx->tnum;
				assert( pIdx->pSchema==pTab->pSchema );
				if( (iDestroyed==0 || (iIdx<iDestroyed)) && iIdx>iLargest ){
					iLargest = iIdx;
				}
			}
			if( iLargest==0 ){
				return;
			}else{
				int iDb = sqlite3SchemaToIndex(pParse->db, pTab->pSchema);
				assert( iDb>=0 && iDb<pParse->db->nDb );
				destroyRootPage(pParse, iLargest, iDb);
				iDestroyed = iLargest;
			}
		}
#endif
	}

	/*
	** Remove entries from the sqlite_statN tables (for N in (1,2,3))
	** after a DROP INDEX or DROP TABLE command.
	*/
	static void sqlite3ClearStatTables(
		Parse *pParse,         /* The parsing context */
		int iDb,               /* The database number */
		const char *zType,     /* "idx" or "tbl" */
		const char *zName      /* Name of index or table */
		){
			int i;
			const char *zDbName = pParse->db->aDb[iDb].zName;
			for(i=1; i<=3; i++){
				char zTab[24];
				sqlite3_snprintf(sizeof(zTab),zTab,"sqlite_stat%d",i);
				if( sqlite3FindTable(pParse->db, zTab, zDbName) ){
					sqlite3NestedParse(pParse,
						"DELETE FROM %Q.%s WHERE %s=%Q",
						zDbName, zTab, zType, zName
						);
				}
			}
	}

	/*
	** Generate code to drop a table.
	*/
	void sqlite3CodeDropTable(Parse *pParse, Table *pTab, int iDb, int isView){
		Vdbe *v;
		sqlite3 *db = pParse->db;
		Trigger *pTrigger;
		Db *pDb = &db->aDb[iDb];

		v = sqlite3GetVdbe(pParse);
		assert( v!=0 );
		sqlite3BeginWriteOperation(pParse, 1, iDb);

#ifndef SQLITE_OMIT_VIRTUALTABLE
		if( IsVirtual(pTab) ){
			sqlite3VdbeAddOp0(v, OP_VBegin);
		}
#endif

		/* Drop all triggers associated with the table being dropped. Code
		** is generated to remove entries from sqlite_master and/or
		** sqlite_temp_master if required.
		*/
		pTrigger = sqlite3TriggerList(pParse, pTab);
		while( pTrigger ){
			assert( pTrigger->pSchema==pTab->pSchema || 
				pTrigger->pSchema==db->aDb[1].pSchema );
			sqlite3DropTriggerPtr(pParse, pTrigger);
			pTrigger = pTrigger->pNext;
		}

#ifndef SQLITE_OMIT_AUTOINCREMENT
		/* Remove any entries of the sqlite_sequence table associated with
		** the table being dropped. This is done before the table is dropped
		** at the btree level, in case the sqlite_sequence table needs to
		** move as a result of the drop (can happen in auto-vacuum mode).
		*/
		if( pTab->tabFlags & TF_Autoincrement ){
			sqlite3NestedParse(pParse,
				"DELETE FROM %Q.sqlite_sequence WHERE name=%Q",
				pDb->zName, pTab->zName
				);
		}
#endif

		/* Drop all SQLITE_MASTER table and index entries that refer to the
		** table. The program name loops through the master table and deletes
		** every row that refers to a table of the same name as the one being
		** dropped. Triggers are handled separately because a trigger can be
		** created in the temp database that refers to a table in another
		** database.
		*/
		sqlite3NestedParse(pParse, 
			"DELETE FROM %Q.%s WHERE tbl_name=%Q and type!='trigger'",
			pDb->zName, SCHEMA_TABLE(iDb), pTab->zName);
		if( !isView && !IsVirtual(pTab) ){
			destroyTable(pParse, pTab);
		}

		/* Remove the table entry from SQLite's internal schema and modify
		** the schema cookie.
		*/
		if( IsVirtual(pTab) ){
			sqlite3VdbeAddOp4(v, OP_VDestroy, iDb, 0, 0, pTab->zName, 0);
		}
		sqlite3VdbeAddOp4(v, OP_DropTable, iDb, 0, 0, pTab->zName, 0);
		sqlite3ChangeCookie(pParse, iDb);
		sqliteViewResetAll(db, iDb);
	}

	/*
	** This routine is called to do the work of a DROP TABLE statement.
	** pName is the name of the table to be dropped.
	*/
	void sqlite3DropTable(Parse *pParse, SrcList *pName, int isView, int noErr){
		Table *pTab;
		Vdbe *v;
		sqlite3 *db = pParse->db;
		int iDb;

		if( db->mallocFailed ){
			goto exit_drop_table;
		}
		assert( pParse->nErr==0 );
		assert( pName->nSrc==1 );
		if( noErr ) db->suppressErr++;
		pTab = sqlite3LocateTableItem(pParse, isView, &pName->a[0]);
		if( noErr ) db->suppressErr--;

		if( pTab==0 ){
			if( noErr ) sqlite3CodeVerifyNamedSchema(pParse, pName->a[0].zDatabase);
			goto exit_drop_table;
		}
		iDb = sqlite3SchemaToIndex(db, pTab->pSchema);
		assert( iDb>=0 && iDb<db->nDb );

		/* If pTab is a virtual table, call ViewGetColumnNames() to ensure
		** it is initialized.
		*/
		if( IsVirtual(pTab) && sqlite3ViewGetColumnNames(pParse, pTab) ){
			goto exit_drop_table;
		}
#ifndef SQLITE_OMIT_AUTHORIZATION
		{
			int code;
			const char *zTab = SCHEMA_TABLE(iDb);
			const char *zDb = db->aDb[iDb].zName;
			const char *zArg2 = 0;
			if( sqlite3AuthCheck(pParse, SQLITE_DELETE, zTab, 0, zDb)){
				goto exit_drop_table;
			}
			if( isView ){
				if( !OMIT_TEMPDB && iDb==1 ){
					code = SQLITE_DROP_TEMP_VIEW;
				}else{
					code = SQLITE_DROP_VIEW;
				}
#ifndef SQLITE_OMIT_VIRTUALTABLE
			}else if( IsVirtual(pTab) ){
				code = SQLITE_DROP_VTABLE;
				zArg2 = sqlite3GetVTable(db, pTab)->pMod->zName;
#endif
			}else{
				if( !OMIT_TEMPDB && iDb==1 ){
					code = SQLITE_DROP_TEMP_TABLE;
				}else{
					code = SQLITE_DROP_TABLE;
				}
			}
			if( sqlite3AuthCheck(pParse, code, pTab->zName, zArg2, zDb) ){
				goto exit_drop_table;
			}
			if( sqlite3AuthCheck(pParse, SQLITE_DELETE, pTab->zName, 0, zDb) ){
				goto exit_drop_table;
			}
		}
#endif
		if( sqlite3StrNICmp(pTab->zName, "sqlite_", 7)==0 
			&& sqlite3StrNICmp(pTab->zName, "sqlite_stat", 11)!=0 ){
				sqlite3ErrorMsg(pParse, "table %s may not be dropped", pTab->zName);
				goto exit_drop_table;
		}

#ifndef SQLITE_OMIT_VIEW
		/* Ensure DROP TABLE is not used on a view, and DROP VIEW is not used
		** on a table.
		*/
		if( isView && pTab->pSelect==0 ){
			sqlite3ErrorMsg(pParse, "use DROP TABLE to delete table %s", pTab->zName);
			goto exit_drop_table;
		}
		if( !isView && pTab->pSelect ){
			sqlite3ErrorMsg(pParse, "use DROP VIEW to delete view %s", pTab->zName);
			goto exit_drop_table;
		}
#endif

		/* Generate code to remove the table from the master table
		** on disk.
		*/
		v = sqlite3GetVdbe(pParse);
		if( v ){
			sqlite3BeginWriteOperation(pParse, 1, iDb);
			sqlite3ClearStatTables(pParse, iDb, "tbl", pTab->zName);
			sqlite3FkDropTable(pParse, pName, pTab);
			sqlite3CodeDropTable(pParse, pTab, iDb, isView);
		}

exit_drop_table:
		sqlite3SrcListDelete(db, pName);
	}

	/*
	** This routine is called to create a new foreign key on the table
	** currently under construction.  pFromCol determines which columns
	** in the current table point to the foreign key.  If pFromCol==0 then
	** connect the key to the last column inserted.  pTo is the name of
	** the table referred to.  pToCol is a list of tables in the other
	** pTo table that the foreign key points to.  flags contains all
	** information about the conflict resolution algorithms specified
	** in the ON DELETE, ON UPDATE and ON INSERT clauses.
	**
	** An FKey structure is created and added to the table currently
	** under construction in the pParse->pNewTable field.
	**
	** The foreign key is set for IMMEDIATE processing.  A subsequent call
	** to sqlite3DeferForeignKey() might change this to DEFERRED.
	*/
	void sqlite3CreateForeignKey(
		Parse *pParse,       /* Parsing context */
		ExprList *pFromCol,  /* Columns in this table that point to other table */
		Token *pTo,          /* Name of the other table */
		ExprList *pToCol,    /* Columns in the other table */
		int flags            /* Conflict resolution algorithms. */
		){
			sqlite3 *db = pParse->db;
#ifndef SQLITE_OMIT_FOREIGN_KEY
			FKey *pFKey = 0;
			FKey *pNextTo;
			Table *p = pParse->pNewTable;
			int nByte;
			int i;
			int nCol;
			char *z;

			assert( pTo!=0 );
			if( p==0 || IN_DECLARE_VTAB ) goto fk_end;
			if( pFromCol==0 ){
				int iCol = p->nCol-1;
				if( NEVER(iCol<0) ) goto fk_end;
				if( pToCol && pToCol->nExpr!=1 ){
					sqlite3ErrorMsg(pParse, "foreign key on %s"
						" should reference only one column of table %T",
						p->aCol[iCol].zName, pTo);
					goto fk_end;
				}
				nCol = 1;
			}else if( pToCol && pToCol->nExpr!=pFromCol->nExpr ){
				sqlite3ErrorMsg(pParse,
					"number of columns in foreign key does not match the number of "
					"columns in the referenced table");
				goto fk_end;
			}else{
				nCol = pFromCol->nExpr;
			}
			nByte = sizeof(*pFKey) + (nCol-1)*sizeof(pFKey->aCol[0]) + pTo->n + 1;
			if( pToCol ){
				for(i=0; i<pToCol->nExpr; i++){
					nByte += sqlite3Strlen30(pToCol->a[i].zName) + 1;
				}
			}
			pFKey = sqlite3DbMallocZero(db, nByte );
			if( pFKey==0 ){
				goto fk_end;
			}
			pFKey->pFrom = p;
			pFKey->pNextFrom = p->pFKey;
			z = (char*)&pFKey->aCol[nCol];
			pFKey->zTo = z;
			memcpy(z, pTo->z, pTo->n);
			z[pTo->n] = 0;
			sqlite3Dequote(z);
			z += pTo->n+1;
			pFKey->nCol = nCol;
			if( pFromCol==0 ){
				pFKey->aCol[0].iFrom = p->nCol-1;
			}else{
				for(i=0; i<nCol; i++){
					int j;
					for(j=0; j<p->nCol; j++){
						if( sqlite3StrICmp(p->aCol[j].zName, pFromCol->a[i].zName)==0 ){
							pFKey->aCol[i].iFrom = j;
							break;
						}
					}
					if( j>=p->nCol ){
						sqlite3ErrorMsg(pParse, 
							"unknown column \"%s\" in foreign key definition", 
							pFromCol->a[i].zName);
						goto fk_end;
					}
				}
			}
			if( pToCol ){
				for(i=0; i<nCol; i++){
					int n = sqlite3Strlen30(pToCol->a[i].zName);
					pFKey->aCol[i].zCol = z;
					memcpy(z, pToCol->a[i].zName, n);
					z[n] = 0;
					z += n+1;
				}
			}
			pFKey->isDeferred = 0;
			pFKey->aAction[0] = (u8)(flags & 0xff);            /* ON DELETE action */
			pFKey->aAction[1] = (u8)((flags >> 8 ) & 0xff);    /* ON UPDATE action */

			assert( sqlite3SchemaMutexHeld(db, 0, p->pSchema) );
			pNextTo = (FKey *)sqlite3HashInsert(&p->pSchema->fkeyHash, 
				pFKey->zTo, sqlite3Strlen30(pFKey->zTo), (void *)pFKey
				);
			if( pNextTo==pFKey ){
				db->mallocFailed = 1;
				goto fk_end;
			}
			if( pNextTo ){
				assert( pNextTo->pPrevTo==0 );
				pFKey->pNextTo = pNextTo;
				pNextTo->pPrevTo = pFKey;
			}

			/* Link the foreign key to the table as the last step.
			*/
			p->pFKey = pFKey;
			pFKey = 0;

fk_end:
			sqlite3DbFree(db, pFKey);
#endif /* !defined(SQLITE_OMIT_FOREIGN_KEY) */
			sqlite3ExprListDelete(db, pFromCol);
			sqlite3ExprListDelete(db, pToCol);
	}

	/*
	** This routine is called when an INITIALLY IMMEDIATE or INITIALLY DEFERRED
	** clause is seen as part of a foreign key definition.  The isDeferred
	** parameter is 1 for INITIALLY DEFERRED and 0 for INITIALLY IMMEDIATE.
	** The behavior of the most recently created foreign key is adjusted
	** accordingly.
	*/
	void sqlite3DeferForeignKey(Parse *pParse, int isDeferred){
#ifndef SQLITE_OMIT_FOREIGN_KEY
		Table *pTab;
		FKey *pFKey;
		if( (pTab = pParse->pNewTable)==0 || (pFKey = pTab->pFKey)==0 ) return;
		assert( isDeferred==0 || isDeferred==1 ); /* EV: R-30323-21917 */
		pFKey->isDeferred = (u8)isDeferred;
#endif
	}

	/*
	** Generate code that will erase and refill index *pIdx.  This is
	** used to initialize a newly created index or to recompute the
	** content of an index in response to a REINDEX command.
	**
	** if memRootPage is not negative, it means that the index is newly
	** created.  The register specified by memRootPage contains the
	** root page number of the index.  If memRootPage is negative, then
	** the index already exists and must be cleared before being refilled and
	** the root page number of the index is taken from pIndex->tnum.
	*/
	static void sqlite3RefillIndex(Parse *pParse, Index *pIndex, int memRootPage){
		Table *pTab = pIndex->pTable;  /* The table that is indexed */
		int iTab = pParse->nTab++;     /* Btree cursor used for pTab */
		int iIdx = pParse->nTab++;     /* Btree cursor used for pIndex */
		int iSorter;                   /* Cursor opened by OpenSorter (if in use) */
		int addr1;                     /* Address of top of loop */
		int addr2;                     /* Address to jump to for next iteration */
		int tnum;                      /* Root page of index */
		Vdbe *v;                       /* Generate code into this virtual machine */
		KeyInfo *pKey;                 /* KeyInfo for index */
		int regRecord;                 /* Register holding assemblied index record */
		sqlite3 *db = pParse->db;      /* The database connection */
		int iDb = sqlite3SchemaToIndex(db, pIndex->pSchema);

#ifndef SQLITE_OMIT_AUTHORIZATION
		if( sqlite3AuthCheck(pParse, SQLITE_REINDEX, pIndex->zName, 0,
			db->aDb[iDb].zName ) ){
				return;
		}
#endif

		/* Require a write-lock on the table to perform this operation */
		sqlite3TableLock(pParse, iDb, pTab->tnum, 1, pTab->zName);

		v = sqlite3GetVdbe(pParse);
		if( v==0 ) return;
		if( memRootPage>=0 ){
			tnum = memRootPage;
		}else{
			tnum = pIndex->tnum;
			sqlite3VdbeAddOp2(v, OP_Clear, tnum, iDb);
		}
		pKey = sqlite3IndexKeyinfo(pParse, pIndex);
		sqlite3VdbeAddOp4(v, OP_OpenWrite, iIdx, tnum, iDb, 
			(char *)pKey, P4_KEYINFO_HANDOFF);
		sqlite3VdbeChangeP5(v, OPFLAG_BULKCSR|((memRootPage>=0)?OPFLAG_P2ISREG:0));

		/* Open the sorter cursor if we are to use one. */
		iSorter = pParse->nTab++;
		sqlite3VdbeAddOp4(v, OP_SorterOpen, iSorter, 0, 0, (char*)pKey, P4_KEYINFO);

		/* Open the table. Loop through all rows of the table, inserting index
		** records into the sorter. */
		sqlite3OpenTable(pParse, iTab, iDb, pTab, OP_OpenRead);
		addr1 = sqlite3VdbeAddOp2(v, OP_Rewind, iTab, 0);
		regRecord = sqlite3GetTempReg(pParse);

		sqlite3GenerateIndexKey(pParse, pIndex, iTab, regRecord, 1);
		sqlite3VdbeAddOp2(v, OP_SorterInsert, iSorter, regRecord);
		sqlite3VdbeAddOp2(v, OP_Next, iTab, addr1+1);
		sqlite3VdbeJumpHere(v, addr1);
		addr1 = sqlite3VdbeAddOp2(v, OP_SorterSort, iSorter, 0);
		if( pIndex->onError!=OE_None ){
			int j2 = sqlite3VdbeCurrentAddr(v) + 3;
			sqlite3VdbeAddOp2(v, OP_Goto, 0, j2);
			addr2 = sqlite3VdbeCurrentAddr(v);
			sqlite3VdbeAddOp3(v, OP_SorterCompare, iSorter, j2, regRecord);
			sqlite3HaltConstraint(pParse, SQLITE_CONSTRAINT_UNIQUE,
				OE_Abort, "indexed columns are not unique", P4_STATIC
				);
		}else{
			addr2 = sqlite3VdbeCurrentAddr(v);
		}
		sqlite3VdbeAddOp2(v, OP_SorterData, iSorter, regRecord);
		sqlite3VdbeAddOp3(v, OP_IdxInsert, iIdx, regRecord, 1);
		sqlite3VdbeChangeP5(v, OPFLAG_USESEEKRESULT);
		sqlite3ReleaseTempReg(pParse, regRecord);
		sqlite3VdbeAddOp2(v, OP_SorterNext, iSorter, addr2);
		sqlite3VdbeJumpHere(v, addr1);

		sqlite3VdbeAddOp1(v, OP_Close, iTab);
		sqlite3VdbeAddOp1(v, OP_Close, iIdx);
		sqlite3VdbeAddOp1(v, OP_Close, iSorter);
	}

	/*
	** Create a new index for an SQL table.  pName1.pName2 is the name of the index 
	** and pTblList is the name of the table that is to be indexed.  Both will 
	** be NULL for a primary key or an index that is created to satisfy a
	** UNIQUE constraint.  If pTable and pIndex are NULL, use pParse->pNewTable
	** as the table to be indexed.  pParse->pNewTable is a table that is
	** currently being constructed by a CREATE TABLE statement.
	**
	** pList is a list of columns to be indexed.  pList will be NULL if this
	** is a primary key or unique-constraint on the most recent column added
	** to the table currently under construction.  
	**
	** If the index is created successfully, return a pointer to the new Index
	** structure. This is used by sqlite3AddPrimaryKey() to mark the index
	** as the tables primary key (Index.autoIndex==2).
	*/
	Index *sqlite3CreateIndex(
		Parse *pParse,     /* All information about this parse */
		Token *pName1,     /* First part of index name. May be NULL */
		Token *pName2,     /* Second part of index name. May be NULL */
		SrcList *pTblName, /* Table to index. Use pParse->pNewTable if 0 */
		ExprList *pList,   /* A list of columns to be indexed */
		int onError,       /* OE_Abort, OE_Ignore, OE_Replace, or OE_None */
		Token *pStart,     /* The CREATE token that begins this statement */
		Token *pEnd,       /* The ")" that closes the CREATE INDEX statement */
		int sortOrder,     /* Sort order of primary key when pList==NULL */
		int ifNotExist     /* Omit error if index already exists */
		){
			Index *pRet = 0;     /* Pointer to return */
			Table *pTab = 0;     /* Table to be indexed */
			Index *pIndex = 0;   /* The index to be created */
			char *zName = 0;     /* Name of the index */
			int nName;           /* Number of characters in zName */
			int i, j;
			Token nullId;        /* Fake token for an empty ID list */
			DbFixer sFix;        /* For assigning database names to pTable */
			int sortOrderMask;   /* 1 to honor DESC in index.  0 to ignore. */
			sqlite3 *db = pParse->db;
			Db *pDb;             /* The specific table containing the indexed database */
			int iDb;             /* Index of the database that is being written */
			Token *pName = 0;    /* Unqualified name of the index to create */
			struct ExprList_item *pListItem; /* For looping over pList */
			int nCol;
			int nExtra = 0;
			char *zExtra;

			assert( pStart==0 || pEnd!=0 ); /* pEnd must be non-NULL if pStart is */
			assert( pParse->nErr==0 );      /* Never called with prior errors */
			if( db->mallocFailed || IN_DECLARE_VTAB ){
				goto exit_create_index;
			}
			if( SQLITE_OK!=sqlite3ReadSchema(pParse) ){
				goto exit_create_index;
			}

			/*
			** Find the table that is to be indexed.  Return early if not found.
			*/
			if( pTblName!=0 ){

				/* Use the two-part index name to determine the database 
				** to search for the table. 'Fix' the table name to this db
				** before looking up the table.
				*/
				assert( pName1 && pName2 );
				iDb = sqlite3TwoPartName(pParse, pName1, pName2, &pName);
				if( iDb<0 ) goto exit_create_index;
				assert( pName && pName->z );

#ifndef SQLITE_OMIT_TEMPDB
				/* If the index name was unqualified, check if the table
				** is a temp table. If so, set the database to 1. Do not do this
				** if initialising a database schema.
				*/
				if( !db->init.busy ){
					pTab = sqlite3SrcListLookup(pParse, pTblName);
					if( pName2->n==0 && pTab && pTab->pSchema==db->aDb[1].pSchema ){
						iDb = 1;
					}
				}
#endif

				if( sqlite3FixInit(&sFix, pParse, iDb, "index", pName) &&
					sqlite3FixSrcList(&sFix, pTblName)
					){
						/* Because the parser constructs pTblName from a single identifier,
						** sqlite3FixSrcList can never fail. */
						assert(0);
				}
				pTab = sqlite3LocateTableItem(pParse, 0, &pTblName->a[0]);
				assert( db->mallocFailed==0 || pTab==0 );
				if( pTab==0 ) goto exit_create_index;
				assert( db->aDb[iDb].pSchema==pTab->pSchema );
			}else{
				assert( pName==0 );
				assert( pStart==0 );
				pTab = pParse->pNewTable;
				if( !pTab ) goto exit_create_index;
				iDb = sqlite3SchemaToIndex(db, pTab->pSchema);
			}
			pDb = &db->aDb[iDb];

			assert( pTab!=0 );
			assert( pParse->nErr==0 );
			if( sqlite3StrNICmp(pTab->zName, "sqlite_", 7)==0 
				&& sqlite3StrNICmp(&pTab->zName[7],"altertab_",9)!=0 ){
					sqlite3ErrorMsg(pParse, "table %s may not be indexed", pTab->zName);
					goto exit_create_index;
			}
#ifndef SQLITE_OMIT_VIEW
			if( pTab->pSelect ){
				sqlite3ErrorMsg(pParse, "views may not be indexed");
				goto exit_create_index;
			}
#endif
#ifndef SQLITE_OMIT_VIRTUALTABLE
			if( IsVirtual(pTab) ){
				sqlite3ErrorMsg(pParse, "virtual tables may not be indexed");
				goto exit_create_index;
			}
#endif

			/*
			** Find the name of the index.  Make sure there is not already another
			** index or table with the same name.  
			**
			** Exception:  If we are reading the names of permanent indices from the
			** sqlite_master table (because some other process changed the schema) and
			** one of the index names collides with the name of a temporary table or
			** index, then we will continue to process this index.
			**
			** If pName==0 it means that we are
			** dealing with a primary key or UNIQUE constraint.  We have to invent our
			** own name.
			*/
			if( pName ){
				zName = sqlite3NameFromToken(db, pName);
				if( zName==0 ) goto exit_create_index;
				assert( pName->z!=0 );
				if( SQLITE_OK!=sqlite3CheckObjectName(pParse, zName) ){
					goto exit_create_index;
				}
				if( !db->init.busy ){
					if( sqlite3FindTable(db, zName, 0)!=0 ){
						sqlite3ErrorMsg(pParse, "there is already a table named %s", zName);
						goto exit_create_index;
					}
				}
				if( sqlite3FindIndex(db, zName, pDb->zName)!=0 ){
					if( !ifNotExist ){
						sqlite3ErrorMsg(pParse, "index %s already exists", zName);
					}else{
						assert( !db->init.busy );
						sqlite3CodeVerifySchema(pParse, iDb);
					}
					goto exit_create_index;
				}
			}else{
				int n;
				Index *pLoop;
				for(pLoop=pTab->pIndex, n=1; pLoop; pLoop=pLoop->pNext, n++){}
				zName = sqlite3MPrintf(db, "sqlite_autoindex_%s_%d", pTab->zName, n);
				if( zName==0 ){
					goto exit_create_index;
				}
			}

			/* Check for authorization to create an index.
			*/
#ifndef SQLITE_OMIT_AUTHORIZATION
			{
				const char *zDb = pDb->zName;
				if( sqlite3AuthCheck(pParse, SQLITE_INSERT, SCHEMA_TABLE(iDb), 0, zDb) ){
					goto exit_create_index;
				}
				i = SQLITE_CREATE_INDEX;
				if( !OMIT_TEMPDB && iDb==1 ) i = SQLITE_CREATE_TEMP_INDEX;
				if( sqlite3AuthCheck(pParse, i, zName, pTab->zName, zDb) ){
					goto exit_create_index;
				}
			}
#endif

			/* If pList==0, it means this routine was called to make a primary
			** key out of the last column added to the table under construction.
			** So create a fake list to simulate this.
			*/
			if( pList==0 ){
				nullId.z = pTab->aCol[pTab->nCol-1].zName;
				nullId.n = sqlite3Strlen30((char*)nullId.z);
				pList = sqlite3ExprListAppend(pParse, 0, 0);
				if( pList==0 ) goto exit_create_index;
				sqlite3ExprListSetName(pParse, pList, &nullId, 0);
				pList->a[0].sortOrder = (u8)sortOrder;
			}

			/* Figure out how many bytes of space are required to store explicitly
			** specified collation sequence names.
			*/
			for(i=0; i<pList->nExpr; i++){
				Expr *pExpr = pList->a[i].pExpr;
				if( pExpr ){
					CollSeq *pColl = sqlite3ExprCollSeq(pParse, pExpr);
					if( pColl ){
						nExtra += (1 + sqlite3Strlen30(pColl->zName));
					}
				}
			}

			/* 
			** Allocate the index structure. 
			*/
			nName = sqlite3Strlen30(zName);
			nCol = pList->nExpr;
			pIndex = sqlite3DbMallocZero(db, 
				ROUND8(sizeof(Index)) +              /* Index structure  */
				ROUND8(sizeof(tRowcnt)*(nCol+1)) +   /* Index.aiRowEst   */
				sizeof(char *)*nCol +                /* Index.azColl     */
				sizeof(int)*nCol +                   /* Index.aiColumn   */
				sizeof(u8)*nCol +                    /* Index.aSortOrder */
				nName + 1 +                          /* Index.zName      */
				nExtra                               /* Collation sequence names */
				);
			if( db->mallocFailed ){
				goto exit_create_index;
			}
			zExtra = (char*)pIndex;
			pIndex->aiRowEst = (tRowcnt*)&zExtra[ROUND8(sizeof(Index))];
			pIndex->azColl = (char**)
				((char*)pIndex->aiRowEst + ROUND8(sizeof(tRowcnt)*nCol+1));
			assert( EIGHT_BYTE_ALIGNMENT(pIndex->aiRowEst) );
			assert( EIGHT_BYTE_ALIGNMENT(pIndex->azColl) );
			pIndex->aiColumn = (int *)(&pIndex->azColl[nCol]);
			pIndex->aSortOrder = (u8 *)(&pIndex->aiColumn[nCol]);
			pIndex->zName = (char *)(&pIndex->aSortOrder[nCol]);
			zExtra = (char *)(&pIndex->zName[nName+1]);
			memcpy(pIndex->zName, zName, nName+1);
			pIndex->pTable = pTab;
			pIndex->nColumn = pList->nExpr;
			pIndex->onError = (u8)onError;
			pIndex->autoIndex = (u8)(pName==0);
			pIndex->pSchema = db->aDb[iDb].pSchema;
			assert( sqlite3SchemaMutexHeld(db, iDb, 0) );

			/* Check to see if we should honor DESC requests on index columns
			*/
			if( pDb->pSchema->file_format>=4 ){
				sortOrderMask = -1;   /* Honor DESC */
			}else{
				sortOrderMask = 0;    /* Ignore DESC */
			}

			/* Scan the names of the columns of the table to be indexed and
			** load the column indices into the Index structure.  Report an error
			** if any column is not found.
			**
			** TODO:  Add a test to make sure that the same column is not named
			** more than once within the same index.  Only the first instance of
			** the column will ever be used by the optimizer.  Note that using the
			** same column more than once cannot be an error because that would 
			** break backwards compatibility - it needs to be a warning.
			*/
			for(i=0, pListItem=pList->a; i<pList->nExpr; i++, pListItem++){
				const char *zColName = pListItem->zName;
				Column *pTabCol;
				int requestedSortOrder;
				CollSeq *pColl;                /* Collating sequence */
				char *zColl;                   /* Collation sequence name */

				for(j=0, pTabCol=pTab->aCol; j<pTab->nCol; j++, pTabCol++){
					if( sqlite3StrICmp(zColName, pTabCol->zName)==0 ) break;
				}
				if( j>=pTab->nCol ){
					sqlite3ErrorMsg(pParse, "table %s has no column named %s",
						pTab->zName, zColName);
					pParse->checkSchema = 1;
					goto exit_create_index;
				}
				pIndex->aiColumn[i] = j;
				if( pListItem->pExpr
					&& (pColl = sqlite3ExprCollSeq(pParse, pListItem->pExpr))!=0
					){
						int nColl;
						zColl = pColl->zName;
						nColl = sqlite3Strlen30(zColl) + 1;
						assert( nExtra>=nColl );
						memcpy(zExtra, zColl, nColl);
						zColl = zExtra;
						zExtra += nColl;
						nExtra -= nColl;
				}else{
					zColl = pTab->aCol[j].zColl;
					if( !zColl ){
						zColl = "BINARY";
					}
				}
				if( !db->init.busy && !sqlite3LocateCollSeq(pParse, zColl) ){
					goto exit_create_index;
				}
				pIndex->azColl[i] = zColl;
				requestedSortOrder = pListItem->sortOrder & sortOrderMask;
				pIndex->aSortOrder[i] = (u8)requestedSortOrder;
			}
			sqlite3DefaultRowEst(pIndex);

			if( pTab==pParse->pNewTable ){
				/* This routine has been called to create an automatic index as a
				** result of a PRIMARY KEY or UNIQUE clause on a column definition, or
				** a PRIMARY KEY or UNIQUE clause following the column definitions.
				** i.e. one of:
				**
				** CREATE TABLE t(x PRIMARY KEY, y);
				** CREATE TABLE t(x, y, UNIQUE(x, y));
				**
				** Either way, check to see if the table already has such an index. If
				** so, don't bother creating this one. This only applies to
				** automatically created indices. Users can do as they wish with
				** explicit indices.
				**
				** Two UNIQUE or PRIMARY KEY constraints are considered equivalent
				** (and thus suppressing the second one) even if they have different
				** sort orders.
				**
				** If there are different collating sequences or if the columns of
				** the constraint occur in different orders, then the constraints are
				** considered distinct and both result in separate indices.
				*/
				Index *pIdx;
				for(pIdx=pTab->pIndex; pIdx; pIdx=pIdx->pNext){
					int k;
					assert( pIdx->onError!=OE_None );
					assert( pIdx->autoIndex );
					assert( pIndex->onError!=OE_None );

					if( pIdx->nColumn!=pIndex->nColumn ) continue;
					for(k=0; k<pIdx->nColumn; k++){
						const char *z1;
						const char *z2;
						if( pIdx->aiColumn[k]!=pIndex->aiColumn[k] ) break;
						z1 = pIdx->azColl[k];
						z2 = pIndex->azColl[k];
						if( z1!=z2 && sqlite3StrICmp(z1, z2) ) break;
					}
					if( k==pIdx->nColumn ){
						if( pIdx->onError!=pIndex->onError ){
							/* This constraint creates the same index as a previous
							** constraint specified somewhere in the CREATE TABLE statement.
							** However the ON CONFLICT clauses are different. If both this 
							** constraint and the previous equivalent constraint have explicit
							** ON CONFLICT clauses this is an error. Otherwise, use the
							** explicitly specified behavior for the index.
							*/
							if( !(pIdx->onError==OE_Default || pIndex->onError==OE_Default) ){
								sqlite3ErrorMsg(pParse, 
									"conflicting ON CONFLICT clauses specified", 0);
							}
							if( pIdx->onError==OE_Default ){
								pIdx->onError = pIndex->onError;
							}
						}
						goto exit_create_index;
					}
				}
			}

			/* Link the new Index structure to its table and to the other
			** in-memory database structures. 
			*/
			if( db->init.busy ){
				Index *p;
				assert( sqlite3SchemaMutexHeld(db, 0, pIndex->pSchema) );
				p = sqlite3HashInsert(&pIndex->pSchema->idxHash, 
					pIndex->zName, sqlite3Strlen30(pIndex->zName),
					pIndex);
				if( p ){
					assert( p==pIndex );  /* Malloc must have failed */
					db->mallocFailed = 1;
					goto exit_create_index;
				}
				db->flags |= SQLITE_InternChanges;
				if( pTblName!=0 ){
					pIndex->tnum = db->init.newTnum;
				}
			}

			/* If the db->init.busy is 0 then create the index on disk.  This
			** involves writing the index into the master table and filling in the
			** index with the current table contents.
			**
			** The db->init.busy is 0 when the user first enters a CREATE INDEX 
			** command.  db->init.busy is 1 when a database is opened and 
			** CREATE INDEX statements are read out of the master table.  In
			** the latter case the index already exists on disk, which is why
			** we don't want to recreate it.
			**
			** If pTblName==0 it means this index is generated as a primary key
			** or UNIQUE constraint of a CREATE TABLE statement.  Since the table
			** has just been created, it contains no data and the index initialization
			** step can be skipped.
			*/
			else{ /* if( db->init.busy==0 ) */
				Vdbe *v;
				char *zStmt;
				int iMem = ++pParse->nMem;

				v = sqlite3GetVdbe(pParse);
				if( v==0 ) goto exit_create_index;


				/* Create the rootpage for the index
				*/
				sqlite3BeginWriteOperation(pParse, 1, iDb);
				sqlite3VdbeAddOp2(v, OP_CreateIndex, iDb, iMem);

				/* Gather the complete text of the CREATE INDEX statement into
				** the zStmt variable
				*/
				if( pStart ){
					assert( pEnd!=0 );
					/* A named index with an explicit CREATE INDEX statement */
					zStmt = sqlite3MPrintf(db, "CREATE%s INDEX %.*s",
						onError==OE_None ? "" : " UNIQUE",
						(int)(pEnd->z - pName->z) + 1,
						pName->z);
				}else{
					/* An automatic index created by a PRIMARY KEY or UNIQUE constraint */
					/* zStmt = sqlite3MPrintf(""); */
					zStmt = 0;
				}

				/* Add an entry in sqlite_master for this index
				*/
				sqlite3NestedParse(pParse, 
					"INSERT INTO %Q.%s VALUES('index',%Q,%Q,#%d,%Q);",
					db->aDb[iDb].zName, SCHEMA_TABLE(iDb),
					pIndex->zName,
					pTab->zName,
					iMem,
					zStmt
					);
				sqlite3DbFree(db, zStmt);

				/* Fill the index with data and reparse the schema. Code an OP_Expire
				** to invalidate all pre-compiled statements.
				*/
				if( pTblName ){
					sqlite3RefillIndex(pParse, pIndex, iMem);
					sqlite3ChangeCookie(pParse, iDb);
					sqlite3VdbeAddParseSchemaOp(v, iDb,
						sqlite3MPrintf(db, "name='%q' AND type='index'", pIndex->zName));
					sqlite3VdbeAddOp1(v, OP_Expire, 0);
				}
			}

			/* When adding an index to the list of indices for a table, make
			** sure all indices labeled OE_Replace come after all those labeled
			** OE_Ignore.  This is necessary for the correct constraint check
			** processing (in sqlite3GenerateConstraintChecks()) as part of
			** UPDATE and INSERT statements.  
			*/
			if( db->init.busy || pTblName==0 ){
				if( onError!=OE_Replace || pTab->pIndex==0
					|| pTab->pIndex->onError==OE_Replace){
						pIndex->pNext = pTab->pIndex;
						pTab->pIndex = pIndex;
				}else{
					Index *pOther = pTab->pIndex;
					while( pOther->pNext && pOther->pNext->onError!=OE_Replace ){
						pOther = pOther->pNext;
					}
					pIndex->pNext = pOther->pNext;
					pOther->pNext = pIndex;
				}
				pRet = pIndex;
				pIndex = 0;
			}

			/* Clean up before exiting */
exit_create_index:
			if( pIndex ){
				sqlite3DbFree(db, pIndex->zColAff);
				sqlite3DbFree(db, pIndex);
			}
			sqlite3ExprListDelete(db, pList);
			sqlite3SrcListDelete(db, pTblName);
			sqlite3DbFree(db, zName);
			return pRet;
	}

	/*
	** Fill the Index.aiRowEst[] array with default information - information
	** to be used when we have not run the ANALYZE command.
	**
	** aiRowEst[0] is suppose to contain the number of elements in the index.
	** Since we do not know, guess 1 million.  aiRowEst[1] is an estimate of the
	** number of rows in the table that match any particular value of the
	** first column of the index.  aiRowEst[2] is an estimate of the number
	** of rows that match any particular combiniation of the first 2 columns
	** of the index.  And so forth.  It must always be the case that
	*
	**           aiRowEst[N]<=aiRowEst[N-1]
	**           aiRowEst[N]>=1
	**
	** Apart from that, we have little to go on besides intuition as to
	** how aiRowEst[] should be initialized.  The numbers generated here
	** are based on typical values found in actual indices.
	*/
	void sqlite3DefaultRowEst(Index *pIdx){
		tRowcnt *a = pIdx->aiRowEst;
		int i;
		tRowcnt n;
		assert( a!=0 );
		a[0] = pIdx->pTable->nRowEst;
		if( a[0]<10 ) a[0] = 10;
		n = 10;
		for(i=1; i<=pIdx->nColumn; i++){
			a[i] = n;
			if( n>5 ) n--;
		}
		if( pIdx->onError!=OE_None ){
			a[pIdx->nColumn] = 1;
		}
	}

	/*
	** This routine will drop an existing named index.  This routine
	** implements the DROP INDEX statement.
	*/
	void sqlite3DropIndex(Parse *pParse, SrcList *pName, int ifExists){
		Index *pIndex;
		Vdbe *v;
		sqlite3 *db = pParse->db;
		int iDb;

		assert( pParse->nErr==0 );   /* Never called with prior errors */
		if( db->mallocFailed ){
			goto exit_drop_index;
		}
		assert( pName->nSrc==1 );
		if( SQLITE_OK!=sqlite3ReadSchema(pParse) ){
			goto exit_drop_index;
		}
		pIndex = sqlite3FindIndex(db, pName->a[0].zName, pName->a[0].zDatabase);
		if( pIndex==0 ){
			if( !ifExists ){
				sqlite3ErrorMsg(pParse, "no such index: %S", pName, 0);
			}else{
				sqlite3CodeVerifyNamedSchema(pParse, pName->a[0].zDatabase);
			}
			pParse->checkSchema = 1;
			goto exit_drop_index;
		}
		if( pIndex->autoIndex ){
			sqlite3ErrorMsg(pParse, "index associated with UNIQUE "
				"or PRIMARY KEY constraint cannot be dropped", 0);
			goto exit_drop_index;
		}
		iDb = sqlite3SchemaToIndex(db, pIndex->pSchema);
#ifndef SQLITE_OMIT_AUTHORIZATION
		{
			int code = SQLITE_DROP_INDEX;
			Table *pTab = pIndex->pTable;
			const char *zDb = db->aDb[iDb].zName;
			const char *zTab = SCHEMA_TABLE(iDb);
			if( sqlite3AuthCheck(pParse, SQLITE_DELETE, zTab, 0, zDb) ){
				goto exit_drop_index;
			}
			if( !OMIT_TEMPDB && iDb ) code = SQLITE_DROP_TEMP_INDEX;
			if( sqlite3AuthCheck(pParse, code, pIndex->zName, pTab->zName, zDb) ){
				goto exit_drop_index;
			}
		}
#endif

		/* Generate code to remove the index and from the master table */
		v = sqlite3GetVdbe(pParse);
		if( v ){
			sqlite3BeginWriteOperation(pParse, 1, iDb);
			sqlite3NestedParse(pParse,
				"DELETE FROM %Q.%s WHERE name=%Q AND type='index'",
				db->aDb[iDb].zName, SCHEMA_TABLE(iDb), pIndex->zName
				);
			sqlite3ClearStatTables(pParse, iDb, "idx", pIndex->zName);
			sqlite3ChangeCookie(pParse, iDb);
			destroyRootPage(pParse, pIndex->tnum, iDb);
			sqlite3VdbeAddOp4(v, OP_DropIndex, iDb, 0, 0, pIndex->zName, 0);
		}

exit_drop_index:
		sqlite3SrcListDelete(db, pName);
	}

	/*
	** pArray is a pointer to an array of objects. Each object in the
	** array is szEntry bytes in size. This routine uses sqlite3DbRealloc()
	** to extend the array so that there is space for a new object at the end.
	**
	** When this function is called, *pnEntry contains the current size of
	** the array (in entries - so the allocation is ((*pnEntry) * szEntry) bytes
	** in total).
	**
	** If the realloc() is successful (i.e. if no OOM condition occurs), the
	** space allocated for the new object is zeroed, *pnEntry updated to
	** reflect the new size of the array and a pointer to the new allocation
	** returned. *pIdx is set to the index of the new array entry in this case.
	**
	** Otherwise, if the realloc() fails, *pIdx is set to -1, *pnEntry remains
	** unchanged and a copy of pArray returned.
	*/
	void *sqlite3ArrayAllocate(
		sqlite3 *db,      /* Connection to notify of malloc failures */
		void *pArray,     /* Array of objects.  Might be reallocated */
		int szEntry,      /* Size of each object in the array */
		int *pnEntry,     /* Number of objects currently in use */
		int *pIdx         /* Write the index of a new slot here */
		){
			char *z;
			int n = *pnEntry;
			if( (n & (n-1))==0 ){
				int sz = (n==0) ? 1 : 2*n;
				void *pNew = sqlite3DbRealloc(db, pArray, sz*szEntry);
				if( pNew==0 ){
					*pIdx = -1;
					return pArray;
				}
				pArray = pNew;
			}
			z = (char*)pArray;
			memset(&z[n * szEntry], 0, szEntry);
			*pIdx = n;
			++*pnEntry;
			return pArray;
	}

	/*
	** Append a new element to the given IdList.  Create a new IdList if
	** need be.
	**
	** A new IdList is returned, or NULL if malloc() fails.
	*/
	IdList *sqlite3IdListAppend(sqlite3 *db, IdList *pList, Token *pToken){
		int i;
		if( pList==0 ){
			pList = sqlite3DbMallocZero(db, sizeof(IdList) );
			if( pList==0 ) return 0;
		}
		pList->a = sqlite3ArrayAllocate(
			db,
			pList->a,
			sizeof(pList->a[0]),
			&pList->nId,
			&i
			);
		if( i<0 ){
			sqlite3IdListDelete(db, pList);
			return 0;
		}
		pList->a[i].zName = sqlite3NameFromToken(db, pToken);
		return pList;
	}

	/*
	** Delete an IdList.
	*/
	void sqlite3IdListDelete(sqlite3 *db, IdList *pList){
		int i;
		if( pList==0 ) return;
		for(i=0; i<pList->nId; i++){
			sqlite3DbFree(db, pList->a[i].zName);
		}
		sqlite3DbFree(db, pList->a);
		sqlite3DbFree(db, pList);
	}

	/*
	** Return the index in pList of the identifier named zId.  Return -1
	** if not found.
	*/
	int sqlite3IdListIndex(IdList *pList, const char *zName){
		int i;
		if( pList==0 ) return -1;
		for(i=0; i<pList->nId; i++){
			if( sqlite3StrICmp(pList->a[i].zName, zName)==0 ) return i;
		}
		return -1;
	}

	/*
	** Expand the space allocated for the given SrcList object by
	** creating nExtra new slots beginning at iStart.  iStart is zero based.
	** New slots are zeroed.
	**
	** For example, suppose a SrcList initially contains two entries: A,B.
	** To append 3 new entries onto the end, do this:
	**
	**    sqlite3SrcListEnlarge(db, pSrclist, 3, 2);
	**
	** After the call above it would contain:  A, B, nil, nil, nil.
	** If the iStart argument had been 1 instead of 2, then the result
	** would have been:  A, nil, nil, nil, B.  To prepend the new slots,
	** the iStart value would be 0.  The result then would
	** be: nil, nil, nil, A, B.
	**
	** If a memory allocation fails the SrcList is unchanged.  The
	** db->mallocFailed flag will be set to true.
	*/
	SrcList *sqlite3SrcListEnlarge(
		sqlite3 *db,       /* Database connection to notify of OOM errors */
		SrcList *pSrc,     /* The SrcList to be enlarged */
		int nExtra,        /* Number of new slots to add to pSrc->a[] */
		int iStart         /* Index in pSrc->a[] of first new slot */
		){
			int i;

			/* Sanity checking on calling parameters */
			assert( iStart>=0 );
			assert( nExtra>=1 );
			assert( pSrc!=0 );
			assert( iStart<=pSrc->nSrc );

			/* Allocate additional space if needed */
			if( pSrc->nSrc+nExtra>pSrc->nAlloc ){
				SrcList *pNew;
				int nAlloc = pSrc->nSrc+nExtra;
				int nGot;
				pNew = sqlite3DbRealloc(db, pSrc,
					sizeof(*pSrc) + (nAlloc-1)*sizeof(pSrc->a[0]) );
				if( pNew==0 ){
					assert( db->mallocFailed );
					return pSrc;
				}
				pSrc = pNew;
				nGot = (sqlite3DbMallocSize(db, pNew) - sizeof(*pSrc))/sizeof(pSrc->a[0])+1;
				pSrc->nAlloc = (u16)nGot;
			}

			/* Move existing slots that come after the newly inserted slots
			** out of the way */
			for(i=pSrc->nSrc-1; i>=iStart; i--){
				pSrc->a[i+nExtra] = pSrc->a[i];
			}
			pSrc->nSrc += (i16)nExtra;

			/* Zero the newly allocated slots */
			memset(&pSrc->a[iStart], 0, sizeof(pSrc->a[0])*nExtra);
			for(i=iStart; i<iStart+nExtra; i++){
				pSrc->a[i].iCursor = -1;
			}

			/* Return a pointer to the enlarged SrcList */
			return pSrc;
	}


	/*
	** Append a new table name to the given SrcList.  Create a new SrcList if
	** need be.  A new entry is created in the SrcList even if pTable is NULL.
	**
	** A SrcList is returned, or NULL if there is an OOM error.  The returned
	** SrcList might be the same as the SrcList that was input or it might be
	** a new one.  If an OOM error does occurs, then the prior value of pList
	** that is input to this routine is automatically freed.
	**
	** If pDatabase is not null, it means that the table has an optional
	** database name prefix.  Like this:  "database.table".  The pDatabase
	** points to the table name and the pTable points to the database name.
	** The SrcList.a[].zName field is filled with the table name which might
	** come from pTable (if pDatabase is NULL) or from pDatabase.  
	** SrcList.a[].zDatabase is filled with the database name from pTable,
	** or with NULL if no database is specified.
	**
	** In other words, if call like this:
	**
	**         sqlite3SrcListAppend(D,A,B,0);
	**
	** Then B is a table name and the database name is unspecified.  If called
	** like this:
	**
	**         sqlite3SrcListAppend(D,A,B,C);
	**
	** Then C is the table name and B is the database name.  If C is defined
	** then so is B.  In other words, we never have a case where:
	**
	**         sqlite3SrcListAppend(D,A,0,C);
	**
	** Both pTable and pDatabase are assumed to be quoted.  They are dequoted
	** before being added to the SrcList.
	*/
	SrcList *sqlite3SrcListAppend(
		sqlite3 *db,        /* Connection to notify of malloc failures */
		SrcList *pList,     /* Append to this SrcList. NULL creates a new SrcList */
		Token *pTable,      /* Table to append */
		Token *pDatabase    /* Database of the table */
		){
			struct SrcList_item *pItem;
			assert( pDatabase==0 || pTable!=0 );  /* Cannot have C without B */
			if( pList==0 ){
				pList = sqlite3DbMallocZero(db, sizeof(SrcList) );
				if( pList==0 ) return 0;
				pList->nAlloc = 1;
			}
			pList = sqlite3SrcListEnlarge(db, pList, 1, pList->nSrc);
			if( db->mallocFailed ){
				sqlite3SrcListDelete(db, pList);
				return 0;
			}
			pItem = &pList->a[pList->nSrc-1];
			if( pDatabase && pDatabase->z==0 ){
				pDatabase = 0;
			}
			if( pDatabase ){
				Token *pTemp = pDatabase;
				pDatabase = pTable;
				pTable = pTemp;
			}
			pItem->zName = sqlite3NameFromToken(db, pTable);
			pItem->zDatabase = sqlite3NameFromToken(db, pDatabase);
			return pList;
	}

	/*
	** Assign VdbeCursor index numbers to all tables in a SrcList
	*/
	void sqlite3SrcListAssignCursors(Parse *pParse, SrcList *pList){
		int i;
		struct SrcList_item *pItem;
		assert(pList || pParse->db->mallocFailed );
		if( pList ){
			for(i=0, pItem=pList->a; i<pList->nSrc; i++, pItem++){
				if( pItem->iCursor>=0 ) break;
				pItem->iCursor = pParse->nTab++;
				if( pItem->pSelect ){
					sqlite3SrcListAssignCursors(pParse, pItem->pSelect->pSrc);
				}
			}
		}
	}

	/*
	** Delete an entire SrcList including all its substructure.
	*/
	void sqlite3SrcListDelete(sqlite3 *db, SrcList *pList){
		int i;
		struct SrcList_item *pItem;
		if( pList==0 ) return;
		for(pItem=pList->a, i=0; i<pList->nSrc; i++, pItem++){
			sqlite3DbFree(db, pItem->zDatabase);
			sqlite3DbFree(db, pItem->zName);
			sqlite3DbFree(db, pItem->zAlias);
			sqlite3DbFree(db, pItem->zIndex);
			sqlite3DeleteTable(db, pItem->pTab);
			sqlite3SelectDelete(db, pItem->pSelect);
			sqlite3ExprDelete(db, pItem->pOn);
			sqlite3IdListDelete(db, pItem->pUsing);
		}
		sqlite3DbFree(db, pList);
	}

	/*
	** This routine is called by the parser to add a new term to the
	** end of a growing FROM clause.  The "p" parameter is the part of
	** the FROM clause that has already been constructed.  "p" is NULL
	** if this is the first term of the FROM clause.  pTable and pDatabase
	** are the name of the table and database named in the FROM clause term.
	** pDatabase is NULL if the database name qualifier is missing - the
	** usual case.  If the term has a alias, then pAlias points to the
	** alias token.  If the term is a subquery, then pSubquery is the
	** SELECT statement that the subquery encodes.  The pTable and
	** pDatabase parameters are NULL for subqueries.  The pOn and pUsing
	** parameters are the content of the ON and USING clauses.
	**
	** Return a new SrcList which encodes is the FROM with the new
	** term added.
	*/
	SrcList *sqlite3SrcListAppendFromTerm(
		Parse *pParse,          /* Parsing context */
		SrcList *p,             /* The left part of the FROM clause already seen */
		Token *pTable,          /* Name of the table to add to the FROM clause */
		Token *pDatabase,       /* Name of the database containing pTable */
		Token *pAlias,          /* The right-hand side of the AS subexpression */
		Select *pSubquery,      /* A subquery used in place of a table name */
		Expr *pOn,              /* The ON clause of a join */
		IdList *pUsing          /* The USING clause of a join */
		){
			struct SrcList_item *pItem;
			sqlite3 *db = pParse->db;
			if( !p && (pOn || pUsing) ){
				sqlite3ErrorMsg(pParse, "a JOIN clause is required before %s", 
					(pOn ? "ON" : "USING")
					);
				goto append_from_error;
			}
			p = sqlite3SrcListAppend(db, p, pTable, pDatabase);
			if( p==0 || NEVER(p->nSrc==0) ){
				goto append_from_error;
			}
			pItem = &p->a[p->nSrc-1];
			assert( pAlias!=0 );
			if( pAlias->n ){
				pItem->zAlias = sqlite3NameFromToken(db, pAlias);
			}
			pItem->pSelect = pSubquery;
			pItem->pOn = pOn;
			pItem->pUsing = pUsing;
			return p;

append_from_error:
			assert( p==0 );
			sqlite3ExprDelete(db, pOn);
			sqlite3IdListDelete(db, pUsing);
			sqlite3SelectDelete(db, pSubquery);
			return 0;
	}

	/*
	** Add an INDEXED BY or NOT INDEXED clause to the most recently added 
	** element of the source-list passed as the second argument.
	*/
	void sqlite3SrcListIndexedBy(Parse *pParse, SrcList *p, Token *pIndexedBy){
		assert( pIndexedBy!=0 );
		if( p && ALWAYS(p->nSrc>0) ){
			struct SrcList_item *pItem = &p->a[p->nSrc-1];
			assert( pItem->notIndexed==0 && pItem->zIndex==0 );
			if( pIndexedBy->n==1 && !pIndexedBy->z ){
				/* A "NOT INDEXED" clause was supplied. See parse.y 
				** construct "indexed_opt" for details. */
				pItem->notIndexed = 1;
			}else{
				pItem->zIndex = sqlite3NameFromToken(pParse->db, pIndexedBy);
			}
		}
	}

	/*
	** When building up a FROM clause in the parser, the join operator
	** is initially attached to the left operand.  But the code generator
	** expects the join operator to be on the right operand.  This routine
	** Shifts all join operators from left to right for an entire FROM
	** clause.
	**
	** Example: Suppose the join is like this:
	**
	**           A natural cross join B
	**
	** The operator is "natural cross join".  The A and B operands are stored
	** in p->a[0] and p->a[1], respectively.  The parser initially stores the
	** operator with A.  This routine shifts that operator over to B.
	*/
	void sqlite3SrcListShiftJoinType(SrcList *p){
		if( p ){
			int i;
			assert( p->a || p->nSrc==0 );
			for(i=p->nSrc-1; i>0; i--){
				p->a[i].jointype = p->a[i-1].jointype;
			}
			p->a[0].jointype = 0;
		}
	}

	/*
	** Begin a transaction
	*/
	void sqlite3BeginTransaction(Parse *pParse, int type){
		sqlite3 *db;
		Vdbe *v;
		int i;

		assert( pParse!=0 );
		db = pParse->db;
		assert( db!=0 );
		/*  if( db->aDb[0].pBt==0 ) return; */
		if( sqlite3AuthCheck(pParse, SQLITE_TRANSACTION, "BEGIN", 0, 0) ){
			return;
		}
		v = sqlite3GetVdbe(pParse);
		if( !v ) return;
		if( type!=TK_DEFERRED ){
			for(i=0; i<db->nDb; i++){
				sqlite3VdbeAddOp2(v, OP_Transaction, i, (type==TK_EXCLUSIVE)+1);
				sqlite3VdbeUsesBtree(v, i);
			}
		}
		sqlite3VdbeAddOp2(v, OP_AutoCommit, 0, 0);
	}

	/*
	** Commit a transaction
	*/
	void sqlite3CommitTransaction(Parse *pParse){
		Vdbe *v;

		assert( pParse!=0 );
		assert( pParse->db!=0 );
		if( sqlite3AuthCheck(pParse, SQLITE_TRANSACTION, "COMMIT", 0, 0) ){
			return;
		}
		v = sqlite3GetVdbe(pParse);
		if( v ){
			sqlite3VdbeAddOp2(v, OP_AutoCommit, 1, 0);
		}
	}

	/*
	** Rollback a transaction
	*/
	void sqlite3RollbackTransaction(Parse *pParse){
		Vdbe *v;

		assert( pParse!=0 );
		assert( pParse->db!=0 );
		if( sqlite3AuthCheck(pParse, SQLITE_TRANSACTION, "ROLLBACK", 0, 0) ){
			return;
		}
		v = sqlite3GetVdbe(pParse);
		if( v ){
			sqlite3VdbeAddOp2(v, OP_AutoCommit, 1, 1);
		}
	}

	/*
	** This function is called by the parser when it parses a command to create,
	** release or rollback an SQL savepoint. 
	*/
	void sqlite3Savepoint(Parse *pParse, int op, Token *pName){
		char *zName = sqlite3NameFromToken(pParse->db, pName);
		if( zName ){
			Vdbe *v = sqlite3GetVdbe(pParse);
#ifndef SQLITE_OMIT_AUTHORIZATION
			static const char * const az[] = { "BEGIN", "RELEASE", "ROLLBACK" };
			assert( !SAVEPOINT_BEGIN && SAVEPOINT_RELEASE==1 && SAVEPOINT_ROLLBACK==2 );
#endif
			if( !v || sqlite3AuthCheck(pParse, SQLITE_SAVEPOINT, az[op], zName, 0) ){
				sqlite3DbFree(pParse->db, zName);
				return;
			}
			sqlite3VdbeAddOp4(v, OP_Savepoint, op, 0, 0, zName, P4_DYNAMIC);
		}
	}

	/*
	** Make sure the TEMP database is open and available for use.  Return
	** the number of errors.  Leave any error messages in the pParse structure.
	*/
	int sqlite3OpenTempDatabase(Parse *pParse){
		sqlite3 *db = pParse->db;
		if( db->aDb[1].pBt==0 && !pParse->explain ){
			int rc;
			Btree *pBt;
			static const int flags = 
				SQLITE_OPEN_READWRITE |
				SQLITE_OPEN_CREATE |
				SQLITE_OPEN_EXCLUSIVE |
				SQLITE_OPEN_DELETEONCLOSE |
				SQLITE_OPEN_TEMP_DB;

			rc = sqlite3BtreeOpen(db->pVfs, 0, db, &pBt, 0, flags);
			if( rc!=SQLITE_OK ){
				sqlite3ErrorMsg(pParse, "unable to open a temporary database "
					"file for storing temporary tables");
				pParse->rc = rc;
				return 1;
			}
			db->aDb[1].pBt = pBt;
			assert( db->aDb[1].pSchema );
			if( SQLITE_NOMEM==sqlite3BtreeSetPageSize(pBt, db->nextPagesize, -1, 0) ){
				db->mallocFailed = 1;
				return 1;
			}
		}
		return 0;
	}

	/*
	** Generate VDBE code that will verify the schema cookie and start
	** a read-transaction for all named database files.
	**
	** It is important that all schema cookies be verified and all
	** read transactions be started before anything else happens in
	** the VDBE program.  But this routine can be called after much other
	** code has been generated.  So here is what we do:
	**
	** The first time this routine is called, we code an OP_Goto that
	** will jump to a subroutine at the end of the program.  Then we
	** record every database that needs its schema verified in the
	** pParse->cookieMask field.  Later, after all other code has been
	** generated, the subroutine that does the cookie verifications and
	** starts the transactions will be coded and the OP_Goto P2 value
	** will be made to point to that subroutine.  The generation of the
	** cookie verification subroutine code happens in sqlite3FinishCoding().
	**
	** If iDb<0 then code the OP_Goto only - don't set flag to verify the
	** schema on any databases.  This can be used to position the OP_Goto
	** early in the code, before we know if any database tables will be used.
	*/
	void sqlite3CodeVerifySchema(Parse *pParse, int iDb){
		Parse *pToplevel = sqlite3ParseToplevel(pParse);

#ifndef SQLITE_OMIT_TRIGGER
		if( pToplevel!=pParse ){
			/* This branch is taken if a trigger is currently being coded. In this
			** case, set cookieGoto to a non-zero value to show that this function
			** has been called. This is used by the sqlite3ExprCodeConstants()
			** function. */
			pParse->cookieGoto = -1;
		}
#endif
		if( pToplevel->cookieGoto==0 ){
			Vdbe *v = sqlite3GetVdbe(pToplevel);
			if( v==0 ) return;  /* This only happens if there was a prior error */
			pToplevel->cookieGoto = sqlite3VdbeAddOp2(v, OP_Goto, 0, 0)+1;
		}
		if( iDb>=0 ){
			sqlite3 *db = pToplevel->db;
			yDbMask mask;

			assert( iDb<db->nDb );
			assert( db->aDb[iDb].pBt!=0 || iDb==1 );
			assert( iDb<SQLITE_MAX_ATTACHED+2 );
			assert( sqlite3SchemaMutexHeld(db, iDb, 0) );
			mask = ((yDbMask)1)<<iDb;
			if( (pToplevel->cookieMask & mask)==0 ){
				pToplevel->cookieMask |= mask;
				pToplevel->cookieValue[iDb] = db->aDb[iDb].pSchema->schema_cookie;
				if( !OMIT_TEMPDB && iDb==1 ){
					sqlite3OpenTempDatabase(pToplevel);
				}
			}
		}
	}

	/*
	** If argument zDb is NULL, then call sqlite3CodeVerifySchema() for each 
	** attached database. Otherwise, invoke it for the database named zDb only.
	*/
	void sqlite3CodeVerifyNamedSchema(Parse *pParse, const char *zDb){
		sqlite3 *db = pParse->db;
		int i;
		for(i=0; i<db->nDb; i++){
			Db *pDb = &db->aDb[i];
			if( pDb->pBt && (!zDb || 0==sqlite3StrICmp(zDb, pDb->zName)) ){
				sqlite3CodeVerifySchema(pParse, i);
			}
		}
	}

	/*
	** Generate VDBE code that prepares for doing an operation that
	** might change the database.
	**
	** This routine starts a new transaction if we are not already within
	** a transaction.  If we are already within a transaction, then a checkpoint
	** is set if the setStatement parameter is true.  A checkpoint should
	** be set for operations that might fail (due to a constraint) part of
	** the way through and which will need to undo some writes without having to
	** rollback the whole transaction.  For operations where all constraints
	** can be checked before any changes are made to the database, it is never
	** necessary to undo a write and the checkpoint should not be set.
	*/
	void sqlite3BeginWriteOperation(Parse *pParse, int setStatement, int iDb){
		Parse *pToplevel = sqlite3ParseToplevel(pParse);
		sqlite3CodeVerifySchema(pParse, iDb);
		pToplevel->writeMask |= ((yDbMask)1)<<iDb;
		pToplevel->isMultiWrite |= setStatement;
	}

	/*
	** Indicate that the statement currently under construction might write
	** more than one entry (example: deleting one row then inserting another,
	** inserting multiple rows in a table, or inserting a row and index entries.)
	** If an abort occurs after some of these writes have completed, then it will
	** be necessary to undo the completed writes.
	*/
	void sqlite3MultiWrite(Parse *pParse){
		Parse *pToplevel = sqlite3ParseToplevel(pParse);
		pToplevel->isMultiWrite = 1;
	}

	/* 
	** The code generator calls this routine if is discovers that it is
	** possible to abort a statement prior to completion.  In order to 
	** perform this abort without corrupting the database, we need to make
	** sure that the statement is protected by a statement transaction.
	**
	** Technically, we only need to set the mayAbort flag if the
	** isMultiWrite flag was previously set.  There is a time dependency
	** such that the abort must occur after the multiwrite.  This makes
	** some statements involving the REPLACE conflict resolution algorithm
	** go a little faster.  But taking advantage of this time dependency
	** makes it more difficult to prove that the code is correct (in 
	** particular, it prevents us from writing an effective
	** implementation of sqlite3AssertMayAbort()) and so we have chosen
	** to take the safe route and skip the optimization.
	*/
	void sqlite3MayAbort(Parse *pParse){
		Parse *pToplevel = sqlite3ParseToplevel(pParse);
		pToplevel->mayAbort = 1;
	}

	/*
	** Code an OP_Halt that causes the vdbe to return an SQLITE_CONSTRAINT
	** error. The onError parameter determines which (if any) of the statement
	** and/or current transaction is rolled back.
	*/
	void sqlite3HaltConstraint(
		Parse *pParse,    /* Parsing context */
		int errCode,      /* extended error code */
		int onError,      /* Constraint type */
		char *p4,         /* Error message */
		int p4type        /* P4_STATIC or P4_TRANSIENT */
		){
			Vdbe *v = sqlite3GetVdbe(pParse);
			assert( (errCode&0xff)==SQLITE_CONSTRAINT );
			if( onError==OE_Abort ){
				sqlite3MayAbort(pParse);
			}
			sqlite3VdbeAddOp4(v, OP_Halt, errCode, onError, 0, p4, p4type);
	}

	/*
	** Check to see if pIndex uses the collating sequence pColl.  Return
	** true if it does and false if it does not.
	*/
#ifndef SQLITE_OMIT_REINDEX
	static int collationMatch(const char *zColl, Index *pIndex){
		int i;
		assert( zColl!=0 );
		for(i=0; i<pIndex->nColumn; i++){
			const char *z = pIndex->azColl[i];
			assert( z!=0 );
			if( 0==sqlite3StrICmp(z, zColl) ){
				return 1;
			}
		}
		return 0;
	}
#endif

	/*
	** Recompute all indices of pTab that use the collating sequence pColl.
	** If pColl==0 then recompute all indices of pTab.
	*/
#ifndef SQLITE_OMIT_REINDEX
	static void reindexTable(Parse *pParse, Table *pTab, char const *zColl){
		Index *pIndex;              /* An index associated with pTab */

		for(pIndex=pTab->pIndex; pIndex; pIndex=pIndex->pNext){
			if( zColl==0 || collationMatch(zColl, pIndex) ){
				int iDb = sqlite3SchemaToIndex(pParse->db, pTab->pSchema);
				sqlite3BeginWriteOperation(pParse, 0, iDb);
				sqlite3RefillIndex(pParse, pIndex, -1);
			}
		}
	}
#endif

	/*
	** Recompute all indices of all tables in all databases where the
	** indices use the collating sequence pColl.  If pColl==0 then recompute
	** all indices everywhere.
	*/
#ifndef SQLITE_OMIT_REINDEX
	static void reindexDatabases(Parse *pParse, char const *zColl){
		Db *pDb;                    /* A single database */
		int iDb;                    /* The database index number */
		sqlite3 *db = pParse->db;   /* The database connection */
		HashElem *k;                /* For looping over tables in pDb */
		Table *pTab;                /* A table in the database */

		assert( sqlite3BtreeHoldsAllMutexes(db) );  /* Needed for schema access */
		for(iDb=0, pDb=db->aDb; iDb<db->nDb; iDb++, pDb++){
			assert( pDb!=0 );
			for(k=sqliteHashFirst(&pDb->pSchema->tblHash);  k; k=sqliteHashNext(k)){
				pTab = (Table*)sqliteHashData(k);
				reindexTable(pParse, pTab, zColl);
			}
		}
	}
#endif

	/*
	** Generate code for the REINDEX command.
	**
	**        REINDEX                            -- 1
	**        REINDEX  <collation>               -- 2
	**        REINDEX  ?<database>.?<tablename>  -- 3
	**        REINDEX  ?<database>.?<indexname>  -- 4
	**
	** Form 1 causes all indices in all attached databases to be rebuilt.
	** Form 2 rebuilds all indices in all databases that use the named
	** collating function.  Forms 3 and 4 rebuild the named index or all
	** indices associated with the named table.
	*/
#ifndef SQLITE_OMIT_REINDEX
	void sqlite3Reindex(Parse *pParse, Token *pName1, Token *pName2){
		CollSeq *pColl;             /* Collating sequence to be reindexed, or NULL */
		char *z;                    /* Name of a table or index */
		const char *zDb;            /* Name of the database */
		Table *pTab;                /* A table in the database */
		Index *pIndex;              /* An index associated with pTab */
		int iDb;                    /* The database index number */
		sqlite3 *db = pParse->db;   /* The database connection */
		Token *pObjName;            /* Name of the table or index to be reindexed */

		/* Read the database schema. If an error occurs, leave an error message
		** and code in pParse and return NULL. */
		if( SQLITE_OK!=sqlite3ReadSchema(pParse) ){
			return;
		}

		if( pName1==0 ){
			reindexDatabases(pParse, 0);
			return;
		}else if( NEVER(pName2==0) || pName2->z==0 ){
			char *zColl;
			assert( pName1->z );
			zColl = sqlite3NameFromToken(pParse->db, pName1);
			if( !zColl ) return;
			pColl = sqlite3FindCollSeq(db, ENC(db), zColl, 0);
			if( pColl ){
				reindexDatabases(pParse, zColl);
				sqlite3DbFree(db, zColl);
				return;
			}
			sqlite3DbFree(db, zColl);
		}
		iDb = sqlite3TwoPartName(pParse, pName1, pName2, &pObjName);
		if( iDb<0 ) return;
		z = sqlite3NameFromToken(db, pObjName);
		if( z==0 ) return;
		zDb = db->aDb[iDb].zName;
		pTab = sqlite3FindTable(db, z, zDb);
		if( pTab ){
			reindexTable(pParse, pTab, 0);
			sqlite3DbFree(db, z);
			return;
		}
		pIndex = sqlite3FindIndex(db, z, zDb);
		sqlite3DbFree(db, z);
		if( pIndex ){
			sqlite3BeginWriteOperation(pParse, 0, iDb);
			sqlite3RefillIndex(pParse, pIndex, -1);
			return;
		}
		sqlite3ErrorMsg(pParse, "unable to identify the object to be reindexed");
	}
#endif

	/*
	** Return a dynamicly allocated KeyInfo structure that can be used
	** with OP_OpenRead or OP_OpenWrite to access database index pIdx.
	**
	** If successful, a pointer to the new structure is returned. In this case
	** the caller is responsible for calling sqlite3DbFree(db, ) on the returned 
	** pointer. If an error occurs (out of memory or missing collation 
	** sequence), NULL is returned and the state of pParse updated to reflect
	** the error.
	*/
	KeyInfo *sqlite3IndexKeyinfo(Parse *pParse, Index *pIdx){
		int i;
		int nCol = pIdx->nColumn;
		int nBytes = sizeof(KeyInfo) + (nCol-1)*sizeof(CollSeq*) + nCol;
		sqlite3 *db = pParse->db;
		KeyInfo *pKey = (KeyInfo *)sqlite3DbMallocZero(db, nBytes);

		if( pKey ){
			pKey->db = pParse->db;
			pKey->aSortOrder = (u8 *)&(pKey->aColl[nCol]);
			assert( &pKey->aSortOrder[nCol]==&(((u8 *)pKey)[nBytes]) );
			for(i=0; i<nCol; i++){
				char *zColl = pIdx->azColl[i];
				assert( zColl );
				pKey->aColl[i] = sqlite3LocateCollSeq(pParse, zColl);
				pKey->aSortOrder[i] = pIdx->aSortOrder[i];
			}
			pKey->nField = (u16)nCol;
		}

		if( pParse->nErr ){
			sqlite3DbFree(db, pKey);
			pKey = 0;
		}
		return pKey;
	}
}