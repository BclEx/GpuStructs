// vtab.c
#ifndef OMIT_VIRTUALTABLE
#include "Core+Vdbe.cu.h"

namespace Core
{
	struct VTableContext
	{
		VTable *VTable;   // The virtual table being constructed
		Table *Table;		// The Table object to which the virtual table belongs
	};

	__device__ RC VTable::CreateModule(Context *ctx, const char *name, const ITableModule *imodule, void *aux, void (*destroy)(void *))
	{
		RC rc = RC_OK;
		MutexEx::Enter(ctx->Mutex);
		int nameLength = _strlen30(name);
		if (ctx->Modules.Find(name, nameLength))
			rc = SysEx_MISUSE_BKPT;
		else
		{
			TableModule *module = (TableModule *)SysEx::TagAlloc(ctx, sizeof(TableModule) + nameLength + 1);
			if (module)
			{
				char *nameCopy = (char *)(&module[1]);
				_memcpy(nameCopy, name, nameLength+1);
				module->Name = nameCopy;
				module->IModule = imodule;
				module->Aux = aux;
				module->Destroy = destroy;
				TableModule *delModule = (TableModule *)ctx->Modules.Insert(nameCopy, nameLength, (void *)module);
				_assert(!delModule || delModule == module);
				if (delModule)
				{
					ctx->MallocFailed = true;
					SysEx::TagFree(ctx, delModule);
				}
			}
		}
		rc = Context::ApiExit(ctx, rc);
		if (rc != RC_OK && destroy) destroy(aux);
		MutexEx::Leave(ctx->Mutex);
		return rc;
	}

	__device__ void VTable::Lock()
	{
		Refs++;
	}

	__device__ VTable *VTable::GetVTable(Context *ctx, Table *table)
	{
		_assert(IsVirtual(table));
		VTable *vtable;
		for (vtable = table->VTables; vtable && vtable->Ctx != ctx; vtable = vtable->Next);
		return vtable;
	}

	__device__ void VTable::Unlock()
	{
		Context *ctx = Ctx;
		_assert(ctx);
		_assert(Refs > 0);
		_assert(ctx->Magic == MAGIC_OPEN || ctx->Magic == MAGIC_ZOMBIE);
		Refs--;
		if (Refs == 0)
		{
			if (IVTable)
				((ITableModule *)IVTable->IModule)->Disconnect(this->IVTable);
			SysEx::TagFree(ctx, this);
		}
	}

	__device__ static VTable *VTableDisconnectAll(Context *ctx, Table *table)
	{
		// Assert that the mutex (if any) associated with the BtShared database that contains table p is held by the caller. See header comments 
		// above function sqlite3VtabUnlockList() for an explanation of why this makes it safe to access the sqlite3.pDisconnect list of any
		// database connection that may have an entry in the p->pVTable list.
		_assert(ctx == nullptr || Btree::SchemaMutexHeld(ctx, 0, table->Schema));
		VTable *r = nullptr;
		VTable *vtables = table->VTables;
		table->VTables = nullptr;
		while (vtables)
		{
			VTable *next = vtables->Next;
			Context *ctx2 = vtables->Ctx;
			_assert(ctx2);
			if (ctx2 == ctx)
			{
				r = vtables;
				table->VTables = r;
				r->Next = nullptr;
			}
			else
			{
				vtables->Next = ctx2->Disconnect;
				ctx2->Disconnect = vtables;
			}
			vtables = next;
		}
		_assert(!ctx || r);
		return r;
	}

	__device__ void VTable::Disconnect(Context *ctx, Table *table)
	{
		_assert(IsVirtual(table));
		_assert(Btree::HoldsAllMutexes(ctx));
		_assert(MutexEx::Held(ctx->Mutex));
		for (VTable **pvtable = &table->VTables; *pvtable; pvtable = &(*pvtable)->Next)
			if ((*pvtable)->Ctx == ctx)
			{
				VTable *vtable = *pvtable;
				*pvtable = vtable->Next;
				vtable->Unlock();
				break;
			}
	}

	__device__ void VTable::UnlockList(Context *ctx)
	{
		_assert(Btree::HoldsAllMutexes(ctx));
		_assert(MutexEx::Held(ctx->Mutex) );
		VTable *vtable = ctx->Disconnect;
		ctx->Disconnect = nullptr;
		if (vtable)
		{
			Vdbe::ExpirePreparedStatements(ctx);
			do
			{
				VTable *next = vtable->Next;
				vtable->Unlock();
				vtable = next;
			} while (vtable);
		}
	}

	__device__ void VTable::Clear(Context *ctx, Table *table)
	{
		if (!ctx || ctx->BytesFreed == 0)
			VTableDisconnectAll(nullptr, table);
		if (table->ModuleArgs)
		{
			for (int i = 0; i < table->ModuleArgs.length; i++)
				if (i != 1)
					SysEx::TagFree(ctx, table->ModuleArgs[i]);
			SysEx::TagFree(ctx, table->ModuleArgs);
		}
	}

	__device__ static void AddModuleArgument(Context *ctx, Table *table, char *arg)
	{
		int i = table->ModuleArgs.length++;
		int bytes = sizeof(char *) * (1 + table->ModuleArgs.length);
		char **moduleArgs = (char **)SysEx::TagRealloc(ctx, table->ModuleArgs, bytes);
		if (!moduleArgs)
		{
			for (int j = 0; j < i; j++)
				SysEx::TagFree(ctx, table->ModuleArgs[j]);
			SysEx::TagFree(ctx, arg);
			SysEx::TagFree(ctx, table->ModuleArgs);
			table->ModuleArgs.length = 0;
		}
		else
		{
			moduleArgs[i] = arg;
			moduleArgs[i + 1] = 0;
		}
		table->ModuleArgs = moduleArgs;
	}

	__device__ void VTable::BeginParse(Parse *parse, Token *name1, Token *name2, Token *moduleName, int ifNotExists)
	{
		sqlite3StartTable(parse, name1, name2, 0, 0, 1, ifNotExists);
		Table *table = parse->NewTable; // The new virtual table
		if (!table) return;
		_assert(table->Index == 0);

		Context *ctx = parse->Ctx; // Database connection
		int dbidx = sqlite3SchemaToIndex(ctx, table->Schema); // The database the table is being created in
		_assert(dbidx >= 0);

		table->TabFlags |= TF_Virtual;
		table->ModuleArgs.length = 0;
		AddModuleArgument(ctx, table, sqlite3NameFromToken(ctx, moduleName));
		AddModuleArgument(ctx, table, 0);
		AddModuleArgument(ctx, table, SysEx::TagStrDup(ctx, table->Name));
		parse->NameToken.length = (int)(&moduleName[moduleName->length] - name1);

#ifndef OMIT_AUTHORIZATION
		// Creating a virtual table invokes the authorization callback twice. The first invocation, to obtain permission to INSERT a row into the
		// sqlite_master table, has already been made by sqlite3StartTable(). The second call, to obtain permission to create the table, is made now.
		if (table->ModuleArgs)
			sqlite3AuthCheck(parse, OP_CREATE_VTABLE, table->Name, table->ModuleArgs[0], parse->Ctx->DBs[dbidx].Name);
#endif
	}

	__device__ static void AddArgumentToVTable(Parse *parse)
	{
		if (parse->Arg && parse->NewTable)
		{
			const char *z = (const char*)parse->Arg;
			int length = parse->Arg.length;
			Context *ctx = parse->Ctx;
			AddModuleArgument(ctx, parse->NewTable, SysEx::TagStrNDup(ctx, z, length));
		}
	}

	__device__ void VTable::FinishParse(Parse *parse, Token *end)
	{
		Table *table = parse->NewTable;  // The table being constructed
		Context *ctx = parse->Ctx;         // The database connection
		if (!table)
			return;
		AddArgumentToVTable(parse);
		parse->Arg = nullptr;
		if (table->ModuleArgs.length < 1)
			return;

		// If the CREATE VIRTUAL TABLE statement is being entered for the first time (in other words if the virtual table is actually being
		// created now instead of just being read out of sqlite_master) then do additional initialization work and store the statement text
		// in the sqlite_master table.
		if (!ctx->Init.Busy)
		{
			// Compute the complete text of the CREATE VIRTUAL TABLE statement
			if (end)
				parse->NameToken.length = (int)(end->data - parse->NameToken) + end->length;
			char *stmt = SysEx::Mprintf(ctx, "CREATE VIRTUAL TABLE %T", &parse->NameToken);

			// A slot for the record has already been allocated in the SQLITE_MASTER table.  We just need to update that slot with all
			// the information we've collected.  
			//
			// The VM register number pParse->regRowid holds the rowid of an entry in the sqlite_master table tht was created for this vtab
			// by sqlite3StartTable().
			int dbidx = sqlite3SchemaToIndex(ctx, table->Schema);
			sqlite3NestedParse(parse,
				"UPDATE %Q.%s "
				"SET type='table', name=%Q, tbl_name=%Q, rootpage=0, sql=%Q "
				"WHERE rowid=#%d",
				ctx->DBs[dbidx].Name, SCHEMA_TABLE(dbidx),
				table->Name,
				table->Name,
				stmt,
				parse->RegRowid);
			SysEx::TagFree(ctx, stmt);
			Vdbe *v = Vdbe::GetVdbe(parse);
			sqlite3ChangeCookie(parse, dbidx);

			Vdbe::AddOp2(v, OP_Expire, 0, 0);
			char *where = SysEx::Mprintf(ctx, "name='%q' AND type='table'", table->Name);
			Vdbe::AddParseSchemaOp(v, dbidx, where);
			Vdbe::AddOp4(v, OP_VCreate, dbidx, 0, 0, table->Name, _strlen30(table->Name) + 1);
		}

		// If we are rereading the sqlite_master table create the in-memory record of the table. The xConnect() method is not called until
		// the first time the virtual table is used in an SQL statement. This allows a schema that contains virtual tables to be loaded before
		// the required virtual table implementations are registered.
		else
		{
			Schema *schema = table->Schema;
			const char *name = table->Name;
			int nameLength = _strlen30(name);
			_assert(Btree::SchemaMutexHeld(ctx, 0, schema));
			Table *oldTable = (Table *)schema->TableHash.Insert(name, nameLength, table);
			if (oldTable)
			{
				ctx->MallocFailed = true;
				_assert(table == oldTable);  // Malloc must have failed inside HashInsert()
				return;
			}
			parse->NewTable = nullptr;
		}
	}

	__device__ void VTable::ArgInit(Parse *parse)
	{
		AddArgumentToVTable(parse);
		parse->Arg = nullptr;
		parse->Arg.length = 0;
	}


	__device__ void VTable::ArgExtend(Parse *parse, Token *token)
	{
		Token *arg = &parse->Arg;
		if (arg == nullptr)
		{
			arg->data = token->data;
			arg->length = token->length;
		}
		else
		{
			_assert(arg < token);
			arg->length = (int)(&token[token->length] - arg);
		}
	}

	__device__ static RC VTableCallConstructor(Context *ctx,  Table *table, TableModule *module, RC (*construct)(Context *, void *, int, const char *const*, IVTable **, char**), char **perror)
	{
		char *moduleName = SysEx::Mprintf(ctx, "%s", table->Name);
		if (!moduleName)
			return RC_NOMEM;

		VTable *vtable = (VTable *)SysEx::TagAlloc(ctx, sizeof(VTable), true);
		if (!vtable)
		{
			SysEx::TagFree(ctx, moduleName);
			return RC_NOMEM;
		}
		vtable->Ctx = ctx;
		vtable->Module = module;

		int dbidx = sqlite3SchemaToIndex(ctx, table->Schema);
		table->ModuleArgs[1] = ctx->DBs[dbidx].Name;

		// Invoke the virtual table constructor
		_assert(&ctx->VTableCtx);
		_assert(construct);
		VTableContext vtableCtx;
		vtableCtx.Table = table;
		vtableCtx.VTable = vtable;
		VTableContext *priorCtx = ctx->VTableCtx;
		ctx->VTableCtx = &vtableCtx;
		const char *const*args = (const char *const*)table->ModuleArgs;
		int argsLength = table->ModuleArgs.length;
		char *error = nullptr;
		RC rc = construct(ctx, module->Aux, argsLength, args, &vtable->IVTable, &error);
		ctx->VTableCtx = priorCtx;
		if (rc == RC_NOMEM)
			ctx->MallocFailed = true;

		if (rc != RC_OK)
		{
			if (!error)
				*perror = SysEx::Mprintf(ctx, "vtable constructor failed: %s", moduleName);
			else
			{
				*perror = SysEx::Mprintf(ctx, "%s", error);
				SysEx::Free(error);
			}
			SysEx::TagFree(ctx, vtable);
		}
		else if (SysEx_ALWAYS(vtable->IVTable))
		{
			// Justification of ALWAYS():  A correct vtab constructor must allocate the sqlite3_vtab object if successful.
			vtable->IVTable->IModule = module->IModule;
			vtable->Refs = 1;
			if (vtableCtx.Table)
			{
				*perror = SysEx::Mprintf(ctx, "vtable constructor did not declare schema: %s", table->Name);
				vtable->Unlock();
				rc = RC_ERROR;
			}
			else
			{
				// If everything went according to plan, link the new VTable structure into the linked list headed by pTab->pVTable. Then loop through the 
				// columns of the table to see if any of them contain the token "hidden". If so, set the Column COLFLAG_HIDDEN flag and remove the token from
				// the type string.
				vtable->Next = table->VTables;
				table->VTables = vtable;
				for (int col = 0; col < table->Cols.length; col++)
				{
					char *type = table->Cols[col].Type;
					if (!type) continue;
					int typeLength = _strlen30(type);
					int i = 0;
					if (_strncmp("hidden", type, 6) || (type[6] && type[6] != ' '))
					{
						for (i = 0; i < typeLength; i++)
							if (!_strncmp(" hidden", &type[i], 7) && (type[i + 7] == '\0' || type[i + 7] == ' '))
							{
								i++;
								break;
							}
					}
					if (i < typeLength)
					{
						int del = 6 + (type[i + 6] ? 1 : 0);
						for (int j = i; (j + del) <= typeLength; j++)
							type[j] = type[j + del];
						if (type[i] == '\0' && i > 0)
						{
							_assert(type[i - 1] == ' ');
							type[i - 1] = '\0';
						}
						table->Cols[col].ColFlags |= COLFLAG_HIDDEN;
					}
				}
			}
		}

		SysEx::TagFree(ctx, moduleName);
		return rc;
	}

	__device__ RC VTable::CallConnect(Parse *parse, Table *table)
	{
		_assert(table);
		Context *ctx = parse->Ctx;
		if ((table->TabFlags & TF_Virtual) == 0 || VTable::GetVTable(ctx, table))
			return RC_OK;

		// Locate the required virtual table module
		const char *moduleName = table->ModuleArgs[0];
		TableModule *module = (TableModule *)ctx->Modules.Find(moduleName, _strlen30(moduleName));
		if (!module)
		{
			Context::ErrorMsg(parse, "no such module: %s", moduleName);
			return RC_ERROR;
		}

		char *error = nullptr;
		RC rc = VTableCallConstructor(ctx, table, module, module->IModule->Connect, &error);
		if (rc != RC_OK)
			Context::ErrorMsg(parse, "%s", error);
		SysEx::TagFree(ctx, error);
		return rc;
	}

	__device__ static RC GrowVTrans(Context *ctx)
	{
		const int ARRAY_INCR = 5;
		// Grow the sqlite3.aVTrans array if required
		if ((ctx->VTrans.length % ARRAY_INCR) == 0)
		{
			int bytes = sizeof(IVTable *) * (ctx->VTrans.length + ARRAY_INCR);
			VTable **vtrans = (VTable **)SysEx::TagRealloc(ctx, (void *)ctx->VTrans, bytes);
			if (!vtrans)
				return RC_NOMEM;
			_memset(&vtrans[ctx->VTrans.length], 0, sizeof(IVTable *) * ARRAY_INCR);
			ctx->VTrans = vtrans;
		}
		return RC_OK;
	}

	__device__ static void AddToVTrans(Context *ctx, VTable *vtable)
	{
		// Add pVtab to the end of sqlite3.aVTrans
		ctx->VTrans[ctx->VTrans.length++] = vtable;
		vtable->Lock();
	}

	__device__ RC VTable::CallCreate(Context *ctx, int dbidx, const char *tableName, char **error)
	{

		Table *table = sqlite3FindTable(ctx, tableName, ctx->DBs[dbidx].Name);
		_assert(table && (table->TabFlags & TF_Virtual) != 0 && !table->VTable);

		// Locate the required virtual table module
		const char *moduleName = table->ModuleArgs[0];
		TableModule *module = (TableModule *)ctx->Modules.Find(moduleName, _strlen30(moduleName));

		// If the module has been registered and includes a Create method, invoke it now. If the module has not been registered, return an 
		// error. Otherwise, do nothing.
		RC rc = RC_OK;
		if (!module)
		{
			*error = SysEx::Mprintf(ctx, "no such module: %s", moduleName);
			rc = RC_ERROR;
		}
		else
			rc = VTableCallConstructor(ctx, table, module, module->IModule->Create, error);

		// Justification of ALWAYS():  The xConstructor method is required to create a valid sqlite3_vtab if it returns SQLITE_OK.
		if (rc == RC_OK && SysEx_ALWAYS(VTable::GetVTable(ctx, table)))
		{
			rc = GrowVTrans(ctx);
			if (rc == RC_OK)
				AddToVTrans(ctx, VTable::GetVTable(ctx, table));
		}
		return rc;
	}

	__device__ RC VTable::DeclareVTable(Context *ctx, const char *createTable)
	{
		MutexEx::Enter(ctx->Mutex);
		Table *table;
		if (!ctx->VTableCtx || !(table = ctx->VTableCtx->Table))
		{
			Context::Error(ctx, RC_MISUSE, 0);
			MutexEx::Leave(ctx->Mutex);
			return SysEx_MISUSE_BKPT;
		}
		_assert((table->TabFlags & TF_Virtual) != 0);

		RC rc = RC_OK;
		Parse *parse = (Parse *)SysEx::ScratchAlloc(ctx, sizeof(Parse));
		if (!parse)
			rc = RC_NOMEM;
		else
		{
			parse->DeclareVTable = true;
			parse->Ctx = ctx;
			parse->QueryLoops = 1;
			char *error = nullptr;
			if (sqlite3RunParser(parse, createTable, &error) == RC_OK  && parse->NewTable && !ctx->MallocFailed && !parse->NewTable->Select && (parse->NewTable->TabFlags & TF_Virtual) == 0)
			{
				if (!table->Cols)
				{
					table->Cols = parse->NewTable->Cols;
					table->Cols.length = parse->NewTable->Cols.length;
					parse->NewTable->Cols.length = 0;
					parse->NewTable->Cols = nullptr;
				}
				ctx->VTableCtx->Table = nullptr;
			}
			else
			{
				Context::Error(ctx, RC_ERROR, (error ? "%s" : 0), error);
				SysEx::TagFree(ctx, error);
				rc = RC_ERROR;
			}
			parse->DeclareVTable = false;

			if (parse->Vdbe)
				Vdbe::Finalize(parse->Vdbe);
			sqlite3DeleteTable(ctx, parse->NewTable);
			SysEx::ScratchFree(ctx, parse);
		}

		_assert((rc & 0xff) == rc);
		rc = Context::ApiExit(ctx, rc);
		MutexEx::Leave(ctx->Mutex);
		return rc;
	}

	__device__ RC VTable::CallDestroy(Context *ctx, int dbidx, const char *tableName)
	{
		RC rc = RC_OK;
		Table *table = sqlite3FindTable(ctx, tableName, ctx->DBs[dbidx].Name);
		if (SysEx_ALWAYS(table != 0 && table->VTables != 0))
		{
			VTable *vtable = VTableDisconnectAll(ctx, table);
			_assert(rc == RC_OK);
			rc = ((ITableModule *)vtable->Module->IModule)->Destroy(vtable->IVTable);

			// Remove the sqlite3_vtab* from the aVTrans[] array, if applicable
			if (rc == RC_OK)
			{
				_assert(table->VTables == vtable && vtable->Next == nullptr);
				vtable->IVTable = nullptr;
				table->VTables = nullptr;
				vtable->Unlock();
			}
		}
		return rc;
	}

	__device__ static void CallFinaliser(Context *ctx, int offset)
	{
		if (ctx->VTrans)
		{
			for (int i = 0; i < ctx->VTrans.length; i++)
			{
				VTable *vtable = ctx->VTrans[i];
				IVTable *ivtable = vtable->IVTable;
				if (ivtable)
				{
					RC (*x)(IVTable *); // = *(int (**)(IVTable *))((char *)ivtable->IModule + offset);
					if (offset == 0)
						x = ivtable->IModule->Rollback;
					else if (offset == 1)
						x = ivtable->IModule->Commit;
					if (x)
						x(ivtable);
				}
				vtable->Savepoint = 0;
				vtable->Unlock();
			}
			SysEx::TagFree(ctx, ctx->VTrans);
			ctx->VTrans.length = 0;
			ctx->VTrans = nullptr;
		}
	}

	__device__ RC VTable::Sync(Context *ctx, char **error)
	{
		RC rc = RC_OK;
		VTable **vtrans = ctx->VTrans;
		ctx->VTrans = nullptr;
		for (int i = 0; rc == RC_OK && i < ctx->VTrans.length; i++)
		{
			RC (*x)(Core::IVTable *);
			Core::IVTable *ivtable = vtrans[i]->IVTable;
			if (ivtable && (x = ivtable->IModule->Sync))
			{
				rc = x(ivtable);
				SysEx::TagFree(ctx, *error);
				*error = SysEx::TagStrDup(ctx, ivtable->ErrMsg);
				SysEx::Free(ivtable->ErrMsg);
			}
		}
		ctx->VTrans = vtrans;
		return rc;
	}

	__device__ RC VTable::Rollback(Context *ctx)
	{
		CallFinaliser(ctx, 0);
		return RC_OK;
	}

	__device__ RC VTable::Commit(Context *ctx)
	{
		CallFinaliser(ctx, 1);
		return RC_OK;
	}

	__device__ RC VTable::Begin(Context *ctx, VTable *vtable)
	{
		// Special case: If ctx->aVTrans is NULL and ctx->nVTrans is greater than zero, then this function is being called from within a
		// virtual module xSync() callback. It is illegal to write to virtual module tables in this case, so return SQLITE_LOCKED.
		if (VTable_InSync(ctx))
			return RC_LOCKED;
		if (!vtable)
			return RC_OK;
		RC rc = RC_OK;
		const ITableModule *imodule = vtable->IVTable->IModule;
		if (imodule->Begin)
		{
			// If pVtab is already in the aVTrans array, return early
			for (int i = 0; i < ctx->VTrans.length; i++)
				if (ctx->VTrans[i] == vtable)
					return RC_OK;
			// Invoke the xBegin method. If successful, add the vtab to the sqlite3.aVTrans[] array.
			rc = GrowVTrans(ctx);
			if (rc == RC_OK)
			{
				rc = ((ITableModule *)imodule)->Begin(vtable->IVTable);
				if (rc == RC_OK)
					AddToVTrans(ctx, vtable);
			}
		}
		return rc;
	}

	__device__ RC VTable::Savepoint(Context *ctx, int op, IPager::SAVEPOINT savepoint)
	{
		_assert(op == IPager::SAVEPOINT_RELEASE || op == IPager::SAVEPOINT_ROLLBACK || op == IPager::SAVEPOINT_BEGIN);
		_assert(savepoint >= 0);
		RC rc = RC_OK;
		if (ctx->VTrans)
			for (int i = 0; rc == RC_OK && i < ctx->VTrans.length; i++)
			{
				VTable *vtable = ctx->VTrans[i];
				const ITableModule *imodule = vtable->Module->IModule;
				if (vtable->IVTable && imodule->Version >= 2)
				{
					RC (*method)(Core::IVTable *, int);
					switch (op)
					{
					case IPager::SAVEPOINT_BEGIN:
						method = imodule->Savepoint;
						vtable->Savepoints = savepoint + 1;
						break;
					case IPager::SAVEPOINT_ROLLBACK:
						method = imodule->RollbackTo;
						break;
					default:
						method = imodule->Release;
						break;
					}
					if (method && vtable->Savepoints > (int)savepoint)
						rc = method(vtable->IVTable, savepoint);
				}
			}
			return rc;
	}

	__device__ FuncDef *VTable::OverloadFunction(Context *ctx, FuncDef *def, int argsLength, Expr *expr)
	{
		// Check to see the left operand is a column in a virtual table
		if (SysEx_NEVER(expr == nullptr)) return def;
		if (expr->OP != TK_COLUMN) return def;
		Table *table = expr->Table;
		if (SysEx_NEVER(table == nullptr)) return def;
		if ((table->TabFlags & TF_Virtual) == 0) return def;
		Core::IVTable *ivtable = VTable::GetVTable(ctx, table)->IVTable;
		_assert(ivtable != nullptr);
		_assert(ivtable->IModule != nullptr);
		ITableModule *imodule = (ITableModule *)ivtable->IModule;
		if (imodule->FindFunction == nullptr) return def;

		// Call the xFindFunction method on the virtual table implementation to see if the implementation wants to overload this function 
		char *lowerName = SysEx::TagStrDup(ctx, def->Name);
		RC rc = RC_OK;
		void (*func)(FuncContext *, int, Mem **) = nullptr;
		void *args = nullptr;
		if (lowerName)
		{
			for (unsigned char *z = (unsigned char*)lowerName; *z; z++)
				*z = __tolower(*z);
			rc = imodule->FindFunction(ivtable, argsLength, lowerName, &func, &args);
			SysEx::TagFree(ctx, lowerName);
		}
		if (rc == RC_OK)
			return def;

		// Create a new ephemeral function definition for the overloaded function
		FuncDef *newFunc = (FuncDef *)SysEx::TagAlloc(ctx, sizeof(FuncDef) + _strlen30(def->Name) + 1, true);
		if (!newFunc) return def;
		*newFunc = *def;
		newFunc->Name = (char *)&newFunc[1];
		_memcpy(newFunc->Name, def->Name, _strlen30(def->Name) + 1);
		newFunc->Func = func;
		newFunc->UserData = args;
		newFunc->Flags |= FUNC_EPHEM;
		return newFunc;
	}

	__device__ void VTable::MakeWritable(Parse *parse, Table *table)
	{
		_assert(IsVirtual(table));
		Parse *toplevel = parse->TopLevel();
		for (int i = 0; i < toplevel->VTableLocks.length; i++)
			if (table == toplevel->VTableLocks[i]) return;
		int newSize = (toplevel->VTableLocks.length + 1) * sizeof(toplevel->VTableLocks[0]);
		Table **vtablelocks = (Table **)SysEx::Realloc(toplevel->VTableLocks, newSize);
		if (vtablelocks)
		{
			toplevel->VTableLocks = vtablelocks;
			toplevel->VTableLocks[toplevel->VTableLocks.length++] = table;
		}
		else
			toplevel->Ctx->MallocFailed = true;
	}

	__device__ CONFLICT VTable::OnConflict(Context *ctx)
	{
		static const CONFLICT map[] =
		{
			CONFLICT_ROLLBACK,
			CONFLICT_ABORT,
			CONFLICT_FAIL,
			CONFLICT_IGNORE,
			CONFLICT_REPLACE,
		};
		_assert(OE_Rollback == 1 && OE_Abort == 2 && OE_Fail == 3);
		_assert(OE_Ignore == 4 && OE_Replace == 5);
		_assert(ctx->VTableOnConflict >= 1 && ctx->VTableOnConflict <= 5);
		return map[ctx->VTableOnConflict - 1];
	}

	__device__ RC VTable::Config(Context *ctx, VTABLECONFIG op, void *arg1)
	{
		RC rc = RC_OK;
		MutexEx::Enter(ctx->Mutex);
		switch (op)
		{
		case VTABLECONFIG_CONSTRAINT:
			VTableContext *p = ctx->VTableCtx;
			if (!p)
				rc = SysEx_MISUSE_BKPT;
			else
			{
				_assert(!p->Table || (p->Table->TabFlags & TF_Virtual) != 0);
				p->VTable->Constraint = (bool)arg1;
			}
			break;
		default:
			rc = SysEx_MISUSE_BKPT;
			break;
		}
		if (rc != RC_OK)
			Context::Error(ctx, rc, nullptr);
		MutexEx::Leave(ctx->Mutex);
		return rc;
	}
}
#endif