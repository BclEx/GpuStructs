#include "Core+Syntax.cu.h"
#include "..\Core+Vbde\Vdbe.cu.h"
//#include <stddef.h>

namespace Core
{
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
		Parse *toplevel = Parse_Toplevel(this);
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
		int bytes = sizeof(Core::TableLock) * (toplevel->TableLocks.length + 1);
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
		Vdbe *v = parse->GetVdbe();
		_assert(v); // sqlite3GetVdbe cannot fail: VDBE already allocated
		for (int i = 0; i < parse->TableLocks.length; i++)
		{
			TableLock *tableLock = &parse->TableLocks[i];
			int p1 = tableLock->DB;
			v->AddOp4(OP_TableLock, p1, tableLock->Table, tableLock->IsWriteLock, tableLock->Name, Vdbe::P4T_STATIC);
		}
	}
#else
#define CodeTableLocks(x)
#endif

	__device__ void Parse::FinishCoding()
	{
		_assert(!Toplevel);
		Context *ctx = Ctx;
		if (ctx->MallocFailed || Nested || Errs)
			return;

		// Begin by generating some termination code at the end of the vdbe program
		Vdbe *v = GetVdbe();
		_assert(!IsMultiWrite || v->AssertMayAbort(MayAbort));
		if (v)
		{
			v->AddOp0(OP_Halt);

			// The cookie mask contains one bit for each database file open. (Bit 0 is for main, bit 1 is for temp, and so forth.)  Bits are
			// set for each database that is used.  Generate code to start a transaction on each used database and to verify the schema cookie
			// on each used database.
			if (CookieGoto > 0)
			{
				v->JumpHere(CookieGoto - 1);
				int db; yDbMask mask; 
				for (db = 0, mask = 1; db < ctx->DBs.length; mask <<= 1, db++)
				{
					if ((mask & CookieMask) == 0)
						continue;
					v->UsesBtree(db);
					v->AddOp2(OP_Transaction, db, (mask & WriteMask) != 0);
					if (!ctx->Init.Busy)
					{
						_assert(Btree::SchemaMutexHeld(ctx, db, nullptr));
						v->AddOp3(OP_VerifyCookie, db, CookieValue[db], ctx->DBs[db].Schema->Generation);
					}
				}
#ifndef OMIT_VIRTUALTABLE
				{
					for (int i = 0; i < VTableLocks.length; i++)
					{
						char *vtable = (char *)VTable::GetVTable(ctx, VTableLocks[i]);
						v->AddOp4(OP_VBegin, 0, 0, 0, vtable, Vdbe::P4T_VTAB);
					}
					VTableLocks.length = 0;
				}
#endif

				// Once all the cookies have been verified and transactions opened, obtain the required table-locks. This is a no-op unless the 
				// shared-cache feature is enabled.
				CodeTableLocks(this);
				// Initialize any AUTOINCREMENT data structures required.
				AutoincrementBegin();
				// Finally, jump back to the beginning of the executable code.
				v->AddOp2(OP_Goto, 0, CookieGoto);
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
			if (Ainc != 0 && Tabs == 0)
				Tabs = 1;
			v->MakeReady(this);
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

	__device__ void Parse::NestedParse(const char *format, void **args)
	{
		if (Errs)
			return;
		_assert(Nested < 10); // Nesting should only be of limited depth
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
		Context *ctx = Ctx;
		SysEx::TagFree(ctx, errMsg);
		SysEx::TagFree(ctx, sql);
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
			_assert(Btree::SchemaMutexHeld(ctx, j, 0));
			table = (Table *)ctx->DBs[j].Schema->TableHash.Find(name, nameLength);
			if (table)
				break;
		}
		return table;
	}

	__device__ Table *Parse::LocateTable(bool isView, const char *name, const char *database)
	{
		// Read the database schema. If an error occurs, leave an error message and code in pParse and return NULL.
		if (ReadSchema() != RC_OK)
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

	__device__ Table *Parse::LocateTableItem(bool isView,  SrcList::SrcListItem *item)
	{
		_assert(!item->Schema || !item->Database);
		const char *database;
		if (item->Schema)
		{
			int db = SchemaToIndex(Ctx, item->Schema);
			database = Ctx->DBs[db].Name;
		}
		else
			database = item->Database;
		return LocateTable(isView, item->Name, database);
	}

	__device__ Index *Parse::FindIndex(Context *ctx, const char *name, const char *database)
	{
		// All mutexes are required for schema access.  Make sure we hold them.
		_assert(Btree::HoldsAllMutexes(ctx));
		Index *p = nullptr;
		int nameLength = _strlen30(name);
		for (int i = OMIT_TEMPDB; i < ctx->DBs.length; i++)
		{
			int j = (i < 2 ? i ^ 1 : i); // Search TEMP before MAIN
			Schema *schema = ctx->DBs[j].Schema;
			_assert(schema);
			if (database && _strcmp(database, ctx->DBs[j].Name))
				continue;
			_assert(Btree::SchemaMutexHeld(ctx, j, 0));
			p = (Index *)schema->IndexHash.Find(name, nameLength);
			if (p)
				break;
		}
		return p;
	}

	__device__ static void FreeIndex(Context *ctx, Index *index)
	{
#ifndef OMIT_ANALYZE
		Parse::DeleteIndexSamples(ctx, index);
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
		Context::DB *dbobj = &ctx->DBs[db];
		_assert(Btree::SchemaMutexHeld(ctx, db, 0));
		_assert(dbobj->Schema);
		Callback::SchemaClear(dbobj->Schema);
		// If any database other than TEMP is reset, then also reset TEMP since TEMP might be holding triggers that reference tables in the other database.
		if (db != 1)
		{
			dbobj = &ctx->DBs[1];
			_assert(dbobj->Schema);
			Callback::SchemaClear(dbobj->Schema);
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
				Callback::SchemaClear(db->Schema);
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

	__device__ static void DeleteColumnNames(Context *ctx, Table *table)
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
		int lookaside = (ctx && (table->TabFlags & TF_Ephemeral) == 0 ? ctx->Lookaside.Outs : 0); // Used to verify lookaside not used for schema 
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
		FkDelete(ctx, table);

		// Delete the Table structure itself.
		DeleteColumnNames(ctx, table);
		SysEx::TagFree(ctx, table->Name);
		SysEx::TagFree(ctx, table->ColAff);
		Select::SelectDelete(db, table->Select);
#ifndef OMIT_CHECK
		Expr::ExprListDelete(db, table->Check);
#endif
#ifndef OMIT_VIRTUALTABLE
		VTable::Clear(ctx, table);
#endif
		SysEx::TagFree(ctx, table);

		// Verify that no lookaside memory was used by schema tables
		_assert(!lookaside || lookaside == ctx->Lookaside.Outs);
	}

	__device__ void Parse::UnlinkAndDeleteTable(Context *ctx, int db, const char *tableName)
	{
		_assert(ctx);
		_assert(db >= 0 && db < ctx->DBs.length);
		_assert(tableName);
		_assert(Btree::SchemaMutexHeld(ctx, db, nullptr));
		ASSERTCOVERAGE(tableName[0] == nullptr);  // Zero-length table names are allowed
		Context::DB *db2 = &ctx->DBs[db];
		Table *table = (Table *)db2->Schema->TableHash.Insert(tableName, _strlen30(tableName), nullptr);
		DeleteTable(ctx, table);
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
		TableLock(db, MASTER_ROOT, true, SCHEMA_TABLE(db));
		v->AddOp3(OP_OpenWrite, 0, MASTER_ROOT, db);
		v->ChangeP4(-1, (char *)5, Vdbe::P4T_INT32);  // 5 column table
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
		if (!Ctx->Init.Busy && Nested == 0 && (Ctx->Flags & Context::FLAG_WriteSchema) == 0 && !_strncmp(name, "sqlite_", 7))
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
			if (AuthCheck(this, SQLITE_INSERT, SCHEMA_TABLE(isTemp), 0, databaseName))
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

		table = (Table *)SysEx::TagAlloc(ctx, sizeof(Table), true);
		if (!table)
		{
			ctx->MallocFailed = true;
			RC = RC_NOMEM;
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
				v->AddOp0(OP_VBegin);
#endif

			// If the file format and encoding in the database have not been set, set them now.
			int reg1 = RegRowid = ++Mems;
			int reg2 = RegRoot = ++Mems;
			int reg3 = ++Mems;
			v->AddOp3(OP_ReadCookie, db, reg3, BTREE_FILE_FORMAT);
			v->UsesBtree(db);
			int j1 = v->AddOp1(OP_If, reg3);
			int fileFormat = ((ctx->Flags & Context::FLAG_LegacyFileFmt) != 0 ? 1 : MAX_FILE_FORMAT);
			v->AddOp2(OP_Integer, fileFormat, reg3);
			v->AddOp3(OP_SetCookie, db, BTREE_FILE_FORMAT, reg3);
			v->AddOp2(OP_Integer, CTXENCODE(ctx), reg3);
			v->AddOp3(OP_SetCookie, db, BTREE_TEXT_ENCODING, reg3);
			v->JumpHere(j1);

			// This just creates a place-holder record in the sqlite_master table. The record created does not contain anything yet.  It will be replaced
			// by the real entry in code generated at sqlite3EndTable().
			//
			// The rowid for the new entry is left in register pParse->regRowid. The root page number of the new table is left in reg pParse->regRoot.
			// The rowid and root page number values are needed by the code that sqlite3EndTable will generate.
#if !defined(OMIT_VIEW) || !defined(OMIT_VIRTUALTABLE)
			if (isView || isVirtual)
				v->AddOp2(OP_Integer, 0, reg2);
			else
#endif
				v->AddOp2(OP_CreateTable, db, reg2);
			OpenMasterTable(db);
			v->AddOp2(OP_NewRowid, 0, reg1);
			v->AddOp2(OP_Null, 0, reg3);
			v->AddOp3(OP_Insert, 0, reg3, reg1);
			v->ChangeP5(OPFLAG_APPEND);
			v->AddOp0(OP_Close);
		}
		return;

begin_table_error:
		SysEx::TagFree(ctx, name);
		return;
	}

	__device__ void Parse::AddColumn(Token *name)
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
			if (_strcmp(nameAsString, table->Cols[i].Name))
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
		_memset(col, 0, sizeof(table->Cols[0]));
		col->Name = nameAsString;

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
				h = (h << 8) + __tolower((*data) & 0xff);
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
			if (!span->Expr->IsConstantOrFunction())
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

	__device__ void Parse::AddPrimaryKey(ExprList *list, OE onError, bool autoInc, int sortOrder)
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
			for (int i = 0; i < list->Exprs; i++)
			{
				for (col = 0; col < table->Cols.length; col++)
					if (!_strcmp(list->Ids[i].Name, table->Cols[col].Name))
						break;
				if (col < table->Cols.length)
					table->Cols[col].ColFlags |= COLFLAG_PRIMKEY;
			}
			if (list->Exprs > 1)
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
		Expr::ExprListDelete(Ctx, list);
		return;
	}

	__device__ void Parse::AddCheckConstraint(Expr *checkExpr)
	{
#ifndef OMIT_CHECK
		Table *table = NewTable;
		if (table && !INDECLARE_VTABLE(this))
		{
			table->Check = ExprListAppend(table->Check, checkExpr);
			if (ConstraintName.length)
				ExprListSetName(table->Check, &ConstraintName, 1);
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
		if (LocateCollSeq(collName))
		{
			table->Cols[col].Coll = collName;
			// If the column is declared as "<name> PRIMARY KEY COLLATE <type>", then an index may have been created on this column before the
			// collation type was added. Correct this if it is the case.
			for (Index *index = table->Index; index; index = index->Next)
			{
				_assert(index->Column == 1);
				if (index->Columns[0] == col)
					index->CollNames[0] = table->Cols[col].Coll;
			}
		}
		else
			SysEx::TagFree(ctx, collName);
	}

	__device__ CollSeq *Parse::LocateCollSeq(const char *name)
	{
		Context *ctx = Ctx;
		TEXTENCODE encode = CTXENCODE(ctx);
		bool initBusy = ctx->Init.Busy;
		CollSeq *coll = Callback::FindCollSeq(ctx, encode, name, initBusy);
		if (!initBusy && (!coll || !coll->Cmp))
			coll = Callback::GetCollSeq(this, encode, coll, name);
		return coll;
	}

	__device__ void Parse::ChangeCookie(int db)
	{
		int r1 = GetTempReg();
		Context *ctx = Ctx;
		Vdbe *v = V;
		_assert(Btree::SchemaMutexHeld(ctx, db, nullptr));
		v->AddOp2(OP_Integer, ctx->DBs[db].Schema->SchemaCookie + 1, r1);
		v->AddOp3(OP_SetCookie, db, BTREE_SCHEMA_VERSION, r1);
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

	__device__ static void IdentPut(char *z, int *idx, char *signedIdent)
	{
		unsigned char *ident = (unsigned char *)signedIdent;
		int i = *idx;
		int j;
		for (j = 0; ident[j]; j++)
			if (!_isalnum(ident[j]) && ident[j] != '_')
				break;
		bool needQuote = (_isdigit(ident[0]) || KeywordCode(ident, j) != TK_ID);
		if (!needQuote) needQuote = ident[j];
		else z[i++] = '"';
		for (j = 0; ident[j]; j++)
		{
			z[i++] = ident[j];
			if (ident[j] == '"') z[i++] = '"';
		}
		if (needQuote) z[i++] = '"';
		z[i] = 0;
		*idx = i;
	}

	__constant__ static const char * const _types[] = {
		" TEXT",	// AFF_TEXT
		"",			// AFF_NONE
		" NUM",		// AFF_NUMERIC
		" INT",		// AFF_INTEGER
		" REAL"};	// AFF_REAL   
	__device__ static char *CreateTableStmt(Context *ctx, Table *table)
	{		
		Column *col;
		int i, n = 0;
		for (col = table->Cols, i = 0; i < table->Cols.length; i++, col++)
			n += IdentLength(col->Name) + 5;
		n += IdentLength(table->Name);
		char *sep, *sep2, *end;
		if (n < 50)
		{ 
			sep = "";
			sep2 = ",";
			end = ")";
		}
		else
		{
			sep = "\n  ";
			sep2 = ",\n  ";
			end = "\n)";
		}
		n += 35 + 6 * table->Cols.length;
		char *stmt = (char *)SysEx::TagAlloc(0, n);
		if (!stmt)
		{
			ctx->MallocFailed = true;
			return nullptr;
		}
		__snprintf(stmt, n, "CREATE TABLE ");
		int k = _strlen30(stmt);
		IdentPut(stmt, &k, table->Name);
		stmt[k++] = '(';
		for (col = table->Cols, i = 0; i < table->Cols.length; i++, col++)
		{
			int len;
			__snprintf(&stmt[k], n - k, sep);
			k += _strlen30(&stmt[k]);
			sep = sep2;
			IdentPut(stmt, &k, col->Name);
			_assert(col->Affinity - AFF_TEXT >= 0);
			_assert(col->Affinity - AFF_TEXT < _arrayLength(types));
			ASSERTCOVERAGE(col->Affinity == AFF_TEXT);
			ASSERTCOVERAGE(col->Affinity == AFF_NONE);
			ASSERTCOVERAGE(col->Affinity == AFF_NUMERIC);
			ASSERTCOVERAGE(col->Affinity == AFF_INTEGER);
			ASSERTCOVERAGE(col->Affinity == AFF_REAL);

			const char *type = _types[col->Affinity - AFF_TEXT];
			int typeLength = _strlen30(type);
			_assert(col->Affinity == AFF_NONE || col->Affinity == sqlite3AffinityType(type));
			_memcpy(&stmt[k], type, typeLength);
			k += typeLength;
			_assert(k <= n);
		}
		__snprintf(&stmt[k], n - k, "%s", end);
		return stmt;
	}

	__device__ void Parse::EndTable(Token *cons, Token *end, Select *select)
	{
		Context *ctx = Ctx;
		if ((!end && !select) || ctx->MallocFailed)
			return;
		Table *table = NewTable;
		if (!table)
			return;
		_assert(!ctx->Init.Busy || !select);

		int db = SchemaToIndex(ctx, table->Schema);

#ifndef OMIT_CHECK
		// Resolve names in all CHECK constraint expressions.
		if (table->Check)
		{
			SrcList src; // Fake SrcList for pParse->pNewTable
			NameContext nc; // Name context for pParse->pNewTable
			_memset(&nc, 0, sizeof(nc));
			_memset(&src, 0, sizeof(src));
			src.Srcs = 1;
			src.Ids[0].Name = table->Name;
			src.Ids[0].Table = table;
			src.Ids[0].Cursor = -1;
			nc.Parse = this;
			nc.SrcList = &src;
			nc.NCFlags = NC_IsCheck;
			ExprList *list = table->Check; // List of all CHECK constraints
			for (int i = 0; i < list->Exprs; i++)
				if (Resolve::ExprNames(&nc, list->Ids[i].Expr))
					return;
		}
#endif

		// If the db->init.busy is 1 it means we are reading the SQL off the "sqlite_master" or "sqlite_temp_master" table on the disk.
		// So do not write to the disk again.  Extract the root page number for the table from the db->init.newTnum field.  (The page number
		// should have been put there by the sqliteOpenCb routine.)
		if (ctx->Init.Busy)
			table->Id = ctx->Init.NewTid;

		// If not initializing, then create a record for the new table in the SQLITE_MASTER table of the database.
		// If this is a TEMPORARY table, write the entry into the auxiliary file instead of into the main database file.
		if (!ctx->Init.Busy)
		{
			Vdbe *v = GetVdbe();
			if (SysEx_NEVER(v == 0))
				return;
			v->AddOp1(OP_Close, 0);
			// Initialize zType for the new view or table.
			char *type; // "view" or "table" */
			char *type2; // "VIEW" or "TABLE" */
			if (!table->Select)
			{
				// A regular table
				type = "table";
				type2 = "TABLE";
			}
#ifndef OMIT_VIEW
			else
			{
				// A view
				type = "view";
				type2 = "VIEW";

			}
#endif

			// If this is a CREATE TABLE xx AS SELECT ..., execute the SELECT statement to populate the new table. The root-page number for the
			// new table is in register pParse->regRoot.
			//
			// Once the SELECT has been coded by sqlite3Select(), it is in a suitable state to query for the column names and types to be used
			// by the new table.
			//
			// A shared-cache write-lock is not required to write to the new table, as a schema-lock must have already been obtained to create it. Since
			// a schema-lock excludes all other database users, the write-lock would be redundant.
			if (select)
			{
				_assert(Tabs == 1);
				v->AddOp3(OP_OpenWrite, 1, RegRoot, ctx);
				v->ChangeP5(OPFLAG_P2ISREG);
				Tabs = 2;
				SelectDest dest;
				SelectDestInit(&dest, SRT_Table, 1);
				Select(select, &dest);
				v->AddOp1(OP_Close, 1);
				if (Errs == 0)
				{
					Table *selectTable = ResultSetOfSelect(select);
					if (!selectTable)
						return;
					_assert(!table->Cols);
					table->Cols.length = selectTable->Cols.length;
					table->Cols = selectTable->Cols;
					selectTable->Cols.length = 0;
					selectTable->Cols = nullptr;
					DeleteTable(ctx, selectTable);
				}
			}

			// Compute the complete text of the CREATE statement
			int n;
			char *stmt; // Text of the CREATE TABLE or CREATE VIEW statement
			if (select)
				stmt = CreateTableStmt(ctx, table);
			else
			{
				n = (int)(end->data - NameToken.data) + 1;
				stmt = SysEx::Mprintf(ctx, "CREATE %s %.*s", type2, n, NameToken.data);
			}

			// A slot for the record has already been allocated in the SQLITE_MASTER table.  We just need to update that slot with all
			// the information we've collected.
			void *args[8] = { ctx->DBs[db].Name, SCHEMA_TABLE(db),
				type, table->Name, table->Name, RegRoot, stmt,
				RegRowid };
			NestedParse(
				"UPDATE %Q.%s "
				"SET type='%s', name=%Q, tbl_name=%Q, rootpage=#%d, sql=%Q "
				"WHERE rowid=#%d", args);
			SysEx::TagFree(ctx, stmt);
			ChangeCookie(db);

#ifndef OMIT_AUTOINCREMENT
			// Check to see if we need to create an sqlite_sequence table for keeping track of autoincrement keys.
			if (table->TabFlags & TF_Autoincrement)
			{
				Context::DB *dbobj = &ctx->DBs[db];
				_assert(Btree::SchemaMutexHeld(ctx, db, nullptr));
				if (!dbobj->Schema->SeqTable)
				{
					void *args2[1] = { dbobj->Name };
					NestedParse("CREATE TABLE %Q.sqlite_sequence(name,seq)", args2);
				}
			}
#endif

			// Reparse everything to update our internal data structures
			v->AddParseSchemaOp(db, SysEx::Mprintf(ctx, "tbl_name='%q'", table->Name));
		}

		// Add the table to the in-memory representation of the database.
		if (ctx->Init.Busy)
		{
			_assert(Btree::SchemaMutexHeld(ctx, db, nullptr));
			Schema *schema = table->Schema;
			Table *oldTable = (Table *)schema->TableHash.Insert(table->Name, _strlen30(table->Name), table);
			if (oldTable)
			{
				_assert(table == oldTable);  // Malloc must have failed inside HashInsert()
				ctx->MallocFailed = true;
				return;
			}
			NewTable = nullptr;
			ctx->Flags |= Context::FLAG_InternChanges;
#ifndef OMIT_ALTERTABLE
			if (!table->Select)
			{
				const char *name = (const char *)NameToken.data;
				_assert(!select && cons && end); // SKY::THIS MIGHT FAIL BECAUSE TESTING DATA NOT THE OBJ
				if (!cons->data)
					cons = end;
				int nameLength = (int)((const char *)cons->data - name);
				table->AddColOffset = 13 + Utf8CharLen(name, nameLength);
			}
#endif
		}
	}

#ifndef OMIT_VIEW
	__device__ void Parse::CreateView(Token *begin, Token *name1, Token *name2, Select *select, bool isTemp, bool noErr)
	{
		Context *ctx = Ctx;
		if (Vars.length > 0)
		{
			ErrorMsg("parameters are not allowed in views");
			SelectDelete(ctx, select);
			return;
		}
		StartTable(name1, name2, isTemp, 1, 0, noErr);
		Table *table = NewTable;
		if (!table || Errs)
		{
			SelectDelete(ctx, select);
			return;
		}
		Token *name = nullptr;
		TwoPartName(name1, name2, &name);
		int db = SchemaToIndex(ctx, table->Schema);
		DbFixer fix;
		if (FixInit(&fix, this, db, "view", name) && FixSelect(&fix, select))
		{
			SelectDelete(ctx, select);
			return;
		}

		// Make a copy of the entire SELECT statement that defines the view. This will force all the Expr.token.z values to be dynamically
		// allocated rather than point to the input string - which means that they will persist after the current sqlite3_exec() call returns.
		table->Select = SelectDup(ctx, select, EXPRDUP_REDUCE);
		SelectDelete(ctx, select);
		if (ctx->MallocFailed)
			return;
		if (!ctx->Init.Busy)
			ViewGetColumnNames(table);

		// Locate the end of the CREATE VIEW statement.  Make sEnd point to the end.
		Token end = LastToken;
		if (SysEx_ALWAYS(end[0] != 0) && end[0] != ';')
			end.data += end.length;
		end.length = 0;

		int n = (int)(end.data - begin->data);
		const char *z = begin->data;
		while (SysEx_ALWAYS(n > 0) && _isspace(z[n - 1])) { n--; }
		end.data = &z[n - 1];
		end.length = 1;

		// Use sqlite3EndTable() to add the view to the SQLITE_MASTER table
		EndTable(0, &end, 0);
		return;
	}
#endif

#if !defined(OMIT_VIEW) || !defined(OMIT_VIRTUALTABLE)
	__device__ int Parse::ViewGetColumnNames(Table *table)
	{
		_assert(table);
		Context *ctx = Ctx;  // Database connection for malloc errors
#ifndef OMIT_VIRTUALTABLE
		if (VTable::CallConnect(this, table))
			return RC_ERROR;
		if (IsVirtual(table))
			return 0;
#endif
#ifndef OMIT_VIEW
		// A positive nCol means the columns names for this view are already known.
		if (table->Cols.length > 0)
			return 0;

		// A negative nCol is a special marker meaning that we are currently trying to compute the column names.  If we enter this routine with
		// a negative nCol, it means two or more views form a loop, like this:
		//     CREATE VIEW one AS SELECT * FROM two;
		//     CREATE VIEW two AS SELECT * FROM one;
		// Actually, the error above is now caught prior to reaching this point. But the following test is still important as it does come up
		// in the following:
		//     CREATE TABLE main.ex1(a);
		//     CREATE TEMP VIEW ex1 AS SELECT a FROM ex1;
		//     SELECT * FROM temp.ex1;
		if (table->Cols.length < 0)
		{
			ErrorMsg("view %s is circularly defined", table->Name);
			return 1;
		}
		_assert(table->Cols.length >= 0);

		// If we get this far, it means we need to compute the table names. Note that the call to sqlite3ResultSetOfSelect() will expand any
		// "*" elements in the results set of the view and will assign cursors to the elements of the FROM clause.  But we do not want these changes
		// to be permanent.  So the computation is done on a copy of the SELECT statement that defines the view.
		_assert(table->Select);
		int errs = 0; // Number of errors encountered
		Select *select = SelectDup(ctx, table->Select, 0); // Copy of the SELECT that implements the view
		if (select)
		{
			bool enableLookaside = ctx->Lookaside.Enabled;
			int n = Tabs; // Temporarily holds the number of cursors assigned
			SrcListAssignCursors(select->Src);
			table->Cols.length = -1;
			ctx->Lookaside.Enabled = false;
#ifndef OMIT_AUTHORIZATION
			int (*auth)(void*,int,const char*,const char*,const char*,const char*) = ctx->Auth;
			ctx->Auth = nullptr;
			Table *selectTable = ResultSetOfSelect(select); // A fake table from which we get the result set
			ctx->Auth = auth;
#else
			Table *selectTable = ResultSetOfSelect(select);
#endif
			ctx->Lookaside.Enabled = enableLookaside;
			Tabs = n;
			if (selectTable)
			{
				_assert(table->Cols.data == nullptr);
				table->Cols.length = selectTable->Cols.length;
				table->Cols.data = selectTable->Cols.data;
				selectTable->Cols.length = 0;
				selectTable->Cols.data = nullptr;
				DeleteTable(ctx, selectTable);
				_assert(Btree::SchemaMutexHeld(ctx, 0, table->Schema));
				table->Schema->Flags |= SCHEMA_UnresetViews;
			}
			else
			{
				table->Cols.length = 0;
				errs++;
			}
			SelectDelete(ctx, select);
		}
		else
			errs++;
#endif
		return errs;  
	}
#endif

#ifndef OMIT_VIEW
	__device__ void Parse::ViewResetAll(Context *ctx, int db)
	{
		i;
		_assert(Btree::SchemaMutexHeld(ctx, db, 0));
		if (!DbHasProperty(ctx, db, FLAG_UnresetViews))
			return;
		for (HashElem *i = (HashElem *)ctx->DBs[db].Schema->TableHash.First(); i; i = i->Next)
		{
			Table *table = (Table *)i->Data;
			if (table->Select)
			{
				DeleteColumnNames(ctx, table);
				table->Cols.data = nullptr;
				table->Cols.length = 0;
			}
		}
		DbClearProperty(ctx, db, FLAG_UnresetViews);
	}
#else
	__device__ void Parse::ViewResetAll(Context *ctx, int db) { }
#endif

#ifndef OMIT_AUTOVACUUM
	__device__ void Parse::RootPageMoved(Context *ctx, int db, int from, int to)
	{
		_assert(Btree::SchemaMutexHeld(ctx, db, 0));
		Context::DB *dbobj = &ctx->DBs[db];
		Hash *hash = &dbobj->Schema->TableHash;
		for (HashElem *elem = (HashElem *)hash->First; elem; elem = elem->Next)
		{
			Table *table = (Table *)elem->Data;
			if (table->Id == from)
				table->Id = to;
		}
		hash = &dbobj->Schema->IndexHash;
		for (HashElem *elem = (HashElem *)hash->First; elem; elem = elem->Next)
		{
			Index *index = (Index *)elem->Data;
			if (index->Id == from)
				index->Id = to;
		}
	}
#endif

	__device__ static void DestroyRootPage(Parse *parse, int table, int db)
	{
		Vdbe *v = parse->GetVdbe();
		int r1 = parse->GetTempReg();
		v->AddOp3(OP_Destroy, table, r1, db);
		parse->MayAbort();
#ifndef OMIT_AUTOVACUUM
		// OP_Destroy stores an in integer r1. If this integer is non-zero, then it is the root page number of a table moved to
		// location iTable. The following code modifies the sqlite_master table to reflect this.
		//
		// The "#NNN" in the SQL is a special constant that means whatever value is in register NNN.  See grammar rules associated with the TK_REGISTER
		// token for additional information.
		void *args[5] = { parse->Ctx->DBs[db].Name, SCHEMA_TABLE(db), table, r1, r1 };
		parse->NestedParse("UPDATE %Q.%s SET rootpage=%d WHERE #%d AND rootpage=#%d", args);
#endif
		parse->ReleaseTempReg(r1);
	}

	__device__ static void DestroyTable(Parse *parse, Table *table)
	{
#ifdef OMIT_AUTOVACUUM
		int db = SchemaToIndex(parse->Ctx, table->Schema);
		DestroyRootPage(parse, table->Id, db);
		for (Index *index = table->Index; index; index = index->Next)
			DestroyRootPage(parse, index->Id, db);
#else
		// If the database may be auto-vacuum capable (if SQLITE_OMIT_AUTOVACUUM is not defined), then it is important to call OP_Destroy on the
		// table and index root-pages in order, starting with the numerically largest root-page number. This guarantees that none of the root-pages
		// to be destroyed is relocated by an earlier OP_Destroy. i.e. if the following were coded:
		//
		// OP_Destroy 4 0
		// ...
		// OP_Destroy 5 0
		//
		// and root page 5 happened to be the largest root-page number in the database, then root page 5 would be moved to page 4 by the 
		// "OP_Destroy 4 0" opcode. The subsequent "OP_Destroy 5 0" would hit a free-list page.
		int tableId = table->Id;
		int destroyedId = 0;
		while (true)
		{
			int largestId = 0;
			if (destroyedId == 0 || tableId < destroyedId)
				largestId = tableId;
			for (Index *index = table->Index; index; index = index->Next)
			{
				int indexId = index->Id;
				_assert(index->Schema == table->Schema);
				if ((destroyedId == 0 || indexId < destroyedId) && indexId > largestId)
					largestId = indexId;
			}
			if (largestId == 0)
				return;
			else
			{
				int db = SchemaToIndex(parse->Ctx, table->Schema);
				_assert(db >= 0 && db < parse->Ctx->DBs.length);
				DestroyRootPage(parse, largestId, db);
				destroyedId = largestId;
			}
		}
#endif
	}

	__device__ void Parse::ClearStatTables(int db, const char *type, const char *name)
	{
		const char *databaseName = Ctx->DBs[db].Name;
		for (int i = 1; i <= 3; i++)
		{
			char tableName[24];
			__snprintf(tableName, sizeof(tableName), "sqlite_stat%d", i);
			if (FindTable(Ctx, tableName, databaseName))
			{
				void *args[4] = { databaseName, tableName, type, name };
				NestedParse("DELETE FROM %Q.%s WHERE %s=%Q", args);
			}
		}
	}

	__device__ void Parse::CodeDropTable(Table *table, int db, bool isView)
	{
		Context *ctx = Ctx;
		Context::DB *dbobj = &ctx->DBs[db];

		Vdbe *v = GetVdbe();
		_assert(v);
		BeginWriteOperation(1, db);

#ifndef OMIT_VIRTUALTABLE
		if (IsVirtual(pTab))
			v->AddOp0(OP_VBegin);
#endif

		// Drop all triggers associated with the table being dropped. Code is generated to remove entries from sqlite_master and/or
		// sqlite_temp_master if required.
		Trigger *trigger = sqlite3TriggerList(this, table);
		while (trigger)
		{
			_assert(trigger->Schema == table->Schema || trigger->Schema == ctx->DBs[1].Schema);
			sqlite3DropTriggerPtr(this, trigger);
			trigger = trigger->Next;
		}

#ifndef OMIT_AUTOINCREMENT
		// Remove any entries of the sqlite_sequence table associated with the table being dropped. This is done before the table is dropped
		// at the btree level, in case the sqlite_sequence table needs to move as a result of the drop (can happen in auto-vacuum mode).
		if (table->TabFlags & TF_Autoincrement)
		{
			void *args[2] = { dbobj->Name, table->Name };
			NestedParse("DELETE FROM %Q.sqlite_sequence WHERE name=%Q", args);
		}
#endif

		// Drop all SQLITE_MASTER table and index entries that refer to the table. The program name loops through the master table and deletes
		// every row that refers to a table of the same name as the one being dropped. Triggers are handled separately because a trigger can be
		// created in the temp database that refers to a table in another database.
		void *args2[3] = { dbobj->Name, SCHEMA_TABLE(db), table->Name };
		NestedParse("DELETE FROM %Q.%s WHERE tbl_name=%Q and type!='trigger'", args2);
		if (!isView && !IsVirtual(table))
			DestroyTable(this, table);

		// Remove the table entry from SQLite's internal schema and modify the schema cookie.
		if (IsVirtual(table))
			v->AddOp4(OP_VDestroy, db, 0, 0, table->Name, 0);
		v->AddOp4(OP_DropTable, db, 0, 0, table->Name, 0);
		ChangeCookie(db);
		ViewResetAll(ctx, db);
	}

	__device__ void Parse::DropTable(SrcList *name, bool isView, bool noErr)
	{
		Context *ctx = Ctx;
		if (ctx->MallocFailed)
			goto exit_drop_table;

		_assert(Errs == 0);
		_assert(name->Srcs.length == 1);
		if (noErr)
			ctx->SuppressErr++;
		Table *table = LocateTableItem(this, isView, &name->a[0]);
		if (noErr)
			ctx->SuppressErr--;

		if (!table)
		{
			if (noErr)
				CodeVerifyNamedSchema(this, name->a[0].Database);
			goto exit_drop_table;
		}
		int db = SchemaToIndex(ctx, table->Schema);
		_assert(db >= 0 && db < ctx->DBs.length);

		// If pTab is a virtual table, call ViewGetColumnNames() to ensure it is initialized.
		if (IsVirtual(table) && ViewGetColumnNames(this, table))
			goto exit_drop_table;
#ifndef OMIT_AUTHORIZATION
		{
			const char *tableName = SCHEMA_TABLE(db);
			const char *databaseName = ctx->DBs[db].Name;
			if (AuthCheck(this, SQLITE_DELETE, tableName, 0, databaseName))
				goto exit_drop_table;
			int code;
			const char *arg2;
			if (isView)
			{
				code = (!OMIT_TEMPDB && db == 1 ? DROP_TEMP_VIEW : DROP_VIEW);
				arg2 = 0;
#ifndef OMIT_VIRTUALTABLE
			}
			else if (IsVirtual(table))
			{
				code = DROP_VTABLE;
				arg2 = VTable::GetVTable(ctx, table)->Module->Name;
#endif
			}
			else
			{
				code = (!OMIT_TEMPDB && db == 1 ? DROP_TEMP_TABLE : DROP_TABLE);
				arg2 = 0;
			}
			if (AuthCheck(this, code, table->Name, arg2, databaseName) || AuthCheck(this, SQLITE_DELETE, table->Name, 0, databaseName))
				goto exit_drop_table;
		}
#endif
		if (!_strncmp(table->Name, "sqlite_", 7) && _strncmp(table->Name, "sqlite_stat", 11))
		{
			ErrorMsg("table %s may not be dropped", table->Name);
			goto exit_drop_table;
		}

#ifndef OMIT_VIEW
		// Ensure DROP TABLE is not used on a view, and DROP VIEW is not used on a table.
		if (isView && !table->Select)
		{
			ErrorMsg("use DROP TABLE to delete table %s", table->Name);
			goto exit_drop_table;
		}
		if (!isView && table->Select)
		{
			ErrorMsg("use DROP VIEW to delete view %s", table->Name);
			goto exit_drop_table;
		}
#endif

		// Generate code to remove the table from the master table on disk.
		Vdbe *v = GetVdbe();
		if (v)
		{
			BeginWriteOperation(1, db);
			ClearStatTables(db, "tbl", table->Name);
			FkDropTable(name, table);
			CodeDropTable(table, db, isView);
		}

exit_drop_table:
		SrcListDelete(ctx, name);
	}

	__device__ void Parse::CreateForeignKey(ExprList *fromCol, Token *to, ExprList *toCol, int flags)
	{
		Context *ctx = Ctx;
#ifndef OMIT_FOREIGN_KEY
		FKey *nextTo;
		Table *table = NewTable;
		int i;

		_assert(!to);
		FKey *fkey = nullptr;
		if (!table || INDECLARE_VTABLE(this))
			goto fk_end;
		int cols;
		if (!fromCol)
		{
			int col = table->Cols.length-1;
			if (SysEx_NEVER(col < 0))
				goto fk_end;
			if (toCol && toCol->Exprs != 1)
			{
				ErrorMsg("foreign key on %s should reference only one column of table %T", table->Cols[col].Name, to);
				goto fk_end;
			}
			cols = 1;
		}
		else if (toCol && toCol->Exprs != fromCol->Exprs)
		{
			ErrorMsg("number of columns in foreign key does not match the number of columns in the referenced table");
			goto fk_end;
		}
		else
			cols = fromCol->Exprs;
		int bytes = sizeof(*fkey) + (cols-1)*sizeof(fkey->Cols[0]) + to->length + 1;
		if (toCol)
			for (i = 0; i < toCol->Exprs; i++)
				bytes += _strlen30(toCol->Ids[i].Name) + 1;
		fkey = (FKey *)SysEx::TagAlloc(ctx, bytes);
		if (!fkey)
			goto fk_end;
		fkey->From = table;
		fkey->NextFrom = table->FKeys;
		char *z = (char *)&fkey->Cols[cols];
		fkey->To = z;
		_memcpy(z, to->data, to->length);
		z[to->length] = 0;
		Dequote(z);
		z += to->length+1;
		fkey->Cols.length = cols;
		if (!fromCol)
			fkey->Cols[0].From = table->Cols.length-1;
		else
		{
			for (i = 0; i < cols; i++)
			{
				int j;
				for (j = 0; j < table->Cols.length; j++)
				{
					if (!_strcmp(table->Cols[j].Name, fromCol->Ids[i].Name))
					{
						fkey->Cols[i].From = j;
						break;
					}
				}
				if (j >= table->Cols.length)
				{
					ErrorMsg("unknown column \"%s\" in foreign key definition", fromCol->a[i].Name);
					goto fk_end;
				}
			}
		}
		if (toCol)
		{
			for (i = 0; i < cols; i++)
			{
				int n = _strlen30(toCol->Ids[i].Name);
				fkey->Cols[i].Col = z;
				_memcpy(z, toCol->Ids[i].Name, n);
				z[n] = 0;
				z += n+1;
			}
		}
		fkey->IsDeferred = false;
		fkey->Actions[0] = (uint8)(flags & 0xff);            // ON DELETE action
		fkey->Actions[1] = (uint8)((flags >> 8 ) & 0xff);    // ON UPDATE action

		_assert(Btree::SchemaMutexHeld(ctx, 0, table->Schema));
		nextTo = (FKey *)table->Schema->FKeyHash.Insert(fkey->To, _strlen30(fkey->To), (void *)fkey);
		if (nextTo == fkKey)
		{
			ctx->MallocFailed = true;
			goto fk_end;
		}
		if (nextTo)
		{
			_assert(nextTo->PrevTo == nullptr);
			fkey->NextTo = nextTo;
			nextTo->PrevTo = fkey;
		}

		// Link the foreign key to the table as the last step.
		table->FKey = fkey;
		fkey = nullptr;

fk_end:
		SysEx::TagFree(ctx, fkey);
#endif
		Expr::ExprListDelete(ctx, fromCol);
		Expr::ExprListDelete(ctx, toCol);
	}

	__device__ void Parse::DeferForeignKey(bool isDeferred)
	{
#ifndef OMIT_FOREIGN_KEY
		Table *table;
		FKey *fkey;
		if (!(table = NewTable) || !(fkey = table->FKeys)) return;
		fkey->IsDeferred = isDeferred;
#endif
	}

	__device__ void Parse::RefillIndex(Index *index, int memRootPage)
	{
		Context *ctx = Ctx; // The database connection
		int db = SchemaToIndex(ctx, index->Schema);

#ifndef OMIT_AUTHORIZATION
		if (AuthCheck(this, SQLITE_REINDEX, index->Name, 0, ctx->DBs[db].Name))
			return;
#endif

		// Require a write-lock on the table to perform this operation
		Table *table = index->Table; // The table that is indexed
		TableLock(db, table->Id, 1, table->Name);

		Vdbe *v = GetVdbe(); // Generate code into this virtual machine
		if (!v) return;
		int tid; // Root page of index
		if (memRootPage >= 0)
			tid = memRootPage;
		else
		{
			tid = index->Id;
			v->AddOp2(OP_Clear, tid, db);
		}
		int indexIdx = Tabs++; // Btree cursor used for pIndex
		KeyInfo *key = IndexKeyinfo(index); // KeyInfo for index
		v->AddOp4(OP_OpenWrite, indexIdx, tid, db, (char *)key, Vdbe::P4T_KEYINFO_HANDOFF);
		v->ChangeP5(OPFLAG_BULKCSR | (memRootPage >= 0 ? OPFLAG_P2ISREG : 0));

		// Open the sorter cursor if we are to use one.
		int sorterIdx = Tabs++; // Cursor opened by OpenSorter (if in use)
		v->AddOp4(OP_SorterOpen, sorterIdx, 0, 0, (char *)key, Vdbe::P4T_KEYINFO);

		// Open the table. Loop through all rows of the table, inserting index records into the sorter.
		int tableIdx = Tabs++; // Btree cursor used for pTab
		OpenTable(tableIdx, db, table, OP_OpenRead);
		int addr1 = v->AddOp2(OP_Rewind, tableIdx, 0); // Address of top of loop
		int regRecord = GetTempReg(); // Register holding assemblied index record

		GenerateIndexKey(index, tableIdx, regRecord, 1);
		v->AddOp2(OP_SorterInsert, sorterIdx, regRecord);
		v->AddOp2(OP_Next, tableIdx, addr1+1);
		v->JumpHere(addr1);
		addr1 = v->AddOp2(OP_SorterSort, sorterIdx, 0);
		int addr2; // Address to jump to for next iteration
		if (indexIdx->OnError != OE_None)
		{
			int j2 = v->CurrentAddr() + 3;
			v->AddOp2(OP_Goto, 0, j2);
			addr2 = v->CurrentAddr();
			v->AddOp3(OP_SorterCompare, sorterIdx, j2, regRecord);
			HaltConstraint(SQLITE_CONSTRAINT_UNIQUE, OE_Abort, "indexed columns are not unique", Vdbe::P4T_STATIC);
		}
		else
			addr2 = sqlite3VdbeCurrentAddr(v);
		v->AddOp2(OP_SorterData, sorterIdx, regRecord);
		v->AddOp3(OP_IdxInsert, indexIdx, regRecord, 1);
		v->ChangeP5(OPFLAG_USESEEKRESULT);
		ReleaseTempReg(regRecord);
		v->AddOp2(OP_SorterNext, sorterIdx, addr2);
		v->JumpHere(addr1);
		v->AddOp1(OP_Close, tableIdx);
		v->AddOp1(OP_Close, indexIdx);
		v->AddOp1(OP_Close, sorterIdx);
	}

	__device__ Index *Parse::CreateIndex(Token *name1, Token *name2, SrcList *tableName, ExprList *list, OE onError, Token *start, Token *end, int sortOrder, bool ifNotExist)
	{
		//Index *pRet = 0;     // Pointer to return
		int i, j;

		int sortOrderMask;   // 1 to honor DESC in index.  0 to ignore.
		Context *ctx = Ctx;


		ExprList::ExprListItem *listItem; // For looping over pList
		int nCol;
		char *zExtra;

		_assert(start || !end); // pEnd must be non-NULL if pStart is
		_assert(Errs == 0); // Never called with prior errors
		if (ctx->MallocFailed || INDECLARE_VTABLE(this))
			goto exit_create_index;
		if (ReadSchema() != RC_OK)
			goto exit_create_index;

		// Find the table that is to be indexed.  Return early if not found.
		int db; // Index of the database that is being written
		Table *table = nullptr; // Table to be indexed
		DbFixer fix; // For assigning database names to pTable
		Token *nameAsToken = nullptr; // Unqualified name of the index to create
		if (tableName)
		{
			// Use the two-part index name to determine the database to search for the table. 'Fix' the table name to this db
			// before looking up the table.
			_assert(name1 && name2);
			db = TwoPartName(name1, name2, &nameAsToken);
			if (db < 0)
				goto exit_create_index;
			_assert(nameAsToken && nameAsToken->data);
#ifndef OMIT_TEMPDB
			// If the index name was unqualified, check if the table is a temp table. If so, set the database to 1. Do not do this
			// if initialising a database schema.
			if (!ctx->Init.Busy)
			{
				table = Delete::SrcListLookup(tableName);
				if (name2->length == 0 && table && table->Schema == ctx->DBs[1].Schema)
					db = 1;
			}
#endif
			// Because the parser constructs pTblName from a single identifier, sqlite3FixSrcList can never fail.
			if (Attach::FixInit(&fix, this, db, "index", nameAsToken) && Attach::FixSrcList(&fix, tableName))
				_assert(0);
			table = LocateTableItem(false, &tableName->Ids[0]);
			_assert(!ctx->MallocFailed || !table);
			if (!table)
				goto exit_create_index;
			_assert(ctx->DBs[db].Schema == table->Schema);
		}
		else
		{
			_assert(!nameAsToken);
			_assert(!start);
			table = NewTable;
			if (!table)
				goto exit_create_index;
			db = SchemaToIndex(ctx, table->Schema);
		}
		Context::DB *dbobj = &ctx->DBs[db]; // The specific table containing the indexed database
		_assert(table);
		_assert(Errs == 0);
		if (!_strncmp(table->Name, "sqlite_", 7) && _strncmp(&table->Name[7], "altertab_", 9) != 0)
		{
			ErrorMsg("table %s may not be indexed", table->Name);
			goto exit_create_index;
		}
#ifndef OMIT_VIEW
		if (table->Select)
		{
			ErrorMsg("views may not be indexed");
			goto exit_create_index;
		}
#endif
#ifndef OMIT_VIRTUALTABLE
		if (IsVirtual(table))
		{
			ErrorMsg("virtual tables may not be indexed");
			goto exit_create_index;
		}
#endif

		// Find the name of the index.  Make sure there is not already another index or table with the same name.  
		//
		// Exception:  If we are reading the names of permanent indices from the sqlite_master table (because some other process changed the schema) and
		// one of the index names collides with the name of a temporary table or index, then we will continue to process this index.
		//
		// If pName==0 it means that we are dealing with a primary key or UNIQUE constraint.  We have to invent our own name.
		char *name = nullptr; // Name of the index
		if (nameAsToken)
		{
			name = NameFromToken(ctx, nameAsToken);
			if (!name)
				goto exit_create_index;
			_assert(nameAsToken->data != nullptr);
			if (CheckObjectName(name) != RC_OK)
				goto exit_create_index;
			if (!ctx->Init.Busy)
			{
				if (FindTable(ctx, name, 0))
				{
					ErrorMsg("there is already a table named %s", name);
					goto exit_create_index;
				}
			}
			if (FindIndex(ctx, name, dbobj->Name))
			{
				if (!ifNotExist)
					ErrorMsg("index %s already exists", name);
				else
				{
					_assert(!ctx->Init.Busy);
					CodeVerifySchema(db);
				}
				goto exit_create_index;
			}
		}
		else
		{
			int n;
			Index *loop;
			for (loop = table->Index, n = 1; loop; loop = loop->Next, n++) { }
			name = SysEx::Mprintf(ctx, "sqlite_autoindex_%s_%d", table->Name, n);
			if (!name)
				goto exit_create_index;
		}

		// Check for authorization to create an index.
#ifndef OMIT_AUTHORIZATION
		{
			const char *databaseName = dbobj->Name;
			if (Auth::AuthCheck(this, AUTH_INSERT, SCHEMA_TABLE(db), 0, databaseName))
				goto exit_create_index;
			i = (!OMIT_TEMPDB2 && db == 1 ? AUTH_CREATE_TEMP_INDEX : AUTH_CREATE_INDEX);
			if (AuthCheck(this, i, name, table->Name, databaseName))
				goto exit_create_index;
		}
#endif

		// If pList==0, it means this routine was called to make a primary key out of the last column added to the table under construction.
		// So create a fake list to simulate this.
		if (!list)
		{
			Token nullId; // Fake token for an empty ID list
			nullId.data = table->Cols[table->Cols.length-1].Name;
			nullId.length = _strlen30((char *)nullId.data);
			list = ExprList::Append(this, 0, 0);
			if (!list)
				goto exit_create_index;
			ExprList::SetName(this, list, &nullId, 0);
			list->a[0].SortOrder = (uint8)sortOrder;
		}

		// Figure out how many bytes of space are required to store explicitly specified collation sequence names.
		int extras = 0;
		for (i = 0; i < list->Exprs; i++)
		{
			Expr *expr = list->Ids[i].Expr;
			if (expr)
			{
				CollSeq *coll = Expr::CollSeq(this, expr);
				if (coll)
					extras += (1 + _strlen30(coll->Name));
			}
		}

		// Allocate the index structure. 
		int nameLength = _strlen30(name); // Number of characters in zName
		int cols = list->Exprs;
		Index *index = (Index *)SysEx::TagAlloc(ctx, 
			SysEx_ROUND8(sizeof(Index)) +			// Index structure
			SysEx_ROUND8(sizeof(tRowcnt)*(cols+1)) +// Index.aiRowEst
			sizeof(char *)*cols +					// Index.azColl
			sizeof(int)*cols +						// Index.aiColumn
			sizeof(uint8)*cols +                    // Index.aSortOrder
			nameLength + 1 +                        // Index.zName
			extraLength);								// Collation sequence names
		if (ctx->MallocFailed)
			goto exit_create_index;
		extra = (char *)index;
		index->RowEsts = (tRowcnt *)&extra[SysEx_ROUND8(sizeof(Index))];
		index->CollNames = (char **)((char *)index->RowEsts + SysEx_ROUND8(sizeof(tRowcnt)*cols+1));
		_assert(SysEx_HASALIGNMENT8(index->RowEsts) );
		_assert(SysEx_HASALIGNMENT8(index->CollNames) );
		index->Columns = (int *)(&index->CollNames[cols]);
		index->SortOrders = (uint8 *)(&index->Columns[cols]);
		index->Name = (char *)(&index->SortOrders[cols]);
		extra = (char *)(&pIndex->zName[nName+1]);
		_memcpy(index->Name, name, nameLength+1);
		index->Table = table;
		index->nColumn = pList->nExpr;
		index->onError = (u8)onError;
		index->autoIndex = (u8)(pName==0);
		index->pSchema = db->aDb[iDb].pSchema;
		_assert(Btree::SchemaMutexHeld(db, iDb, 0) );

		// Check to see if we should honor DESC requests on index columns
		sortOrderMask = (pDb->Schema->FileFormat >= 4 ? -1 : 0); // Honor/Ignore DESC

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

	__device__ void sqlite3DefaultRowEst(Index *idx)
	{
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