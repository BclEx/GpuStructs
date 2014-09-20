// analyze.c
#pragma region OMIT_ANALYZE
#ifndef OMIT_ANALYZE
#include "..\Core+Vdbe.cu.h"

namespace Core { namespace Command
{
	__device__ static const struct {
		const char *Name;
		const char *Cols;
	} _tables[] = {
		{ "sqlite_stat1", "tbl,idx,stat" },
#ifdef ENABLE_STAT3
		{ "sqlite_stat3", "tbl,idx,neq,nlt,ndlt,sample" },
#endif
	};

	__device__ static void OpenStatTable(Parse *parse, int db, int statCur, const char *where_, const char *whereType)
	{
		int roots[] = {0, 0};
		uint8 createTbls[] = {0, 0};

		Context *ctx = parse->Ctx;
		Vdbe *v = parse->GetVdbe();
		if (!v) return;
		_assert(Btree::HoldsAllMutexes(ctx));
		_assert(v->VdbeCtx() == ctx);
		Context::DB *dbObj = &ctx->DBs[db];

		// Create new statistic tables if they do not exist, or clear them if they do already exist.
		for (int i = 0; i < __arrayStaticLength(_tables); i++)
		{
			const char *table = _tables[i].Name;
			Table *stat;
			if ((stat = sqlite3FindTable(db, table, dbObj->Name)) == nullptr)
			{
				// The sqlite_stat[12] table does not exist. Create it. Note that a side-effect of the CREATE TABLE statement is to leave the rootpage 
				// of the new table in register pParse->regRoot. This is important because the OpenWrite opcode below will be needing it.
				void *args[] = { dbObj->Name, (void *)table, (void *)_tables[i].Cols }; parse->NestedParse("CREATE TABLE %Q.%s(%s)", args);
				roots[i] = parse->RegRoot;
				createTbls[i] = OPFLAG_P2ISREG;
			}
			else
			{
				// The table already exists. If zWhere is not NULL, delete all entries associated with the table zWhere. If zWhere is NULL, delete the
				// entire contents of the table.
				roots[i] = stat->Id;
				sqlite3TableLock(parse, db, roots[i], 1, table);
				if (where_)
				{
					void *args[] = { dbObj->Name, (void *)table, (void *)whereType, (void *)where_ }; parse->NestedParse("DELETE FROM %Q.%s WHERE %s=%Q", args);
				}
				else
					v->AddOp2(OP_Clear, roots[i], db); // The sqlite_stat[12] table already exists.  Delete all rows.
			}
		}

		// Open the sqlite_stat[13] tables for writing.
		for (int i = 0; i < __arrayStaticLength(_tables); i++)
		{
			v->AddOp3(OP_OpenWrite, statCur+i, roots[i], db);
			v->ChangeP4(-1, (char *)3, Vdbe::P4T_INT32);
			v->ChangeP5(createTbls[i]);
		}
	}

#ifndef STAT3_SAMPLES
#define STAT3_SAMPLES 24
#endif

	struct Stat3Accum
	{
		tRowcnt Rows;				// Number of rows in the entire table
		tRowcnt PSamples;			// How often to do a periodic sample
		int Min;					// Index of entry with minimum nEq and hash
		int MaxSamples;             // Maximum number of samples to accumulate
		//int Samples;              // Current number of samples
		uint32 Prn;                 // Pseudo-random number used for sampling
		struct Stat3Sample
		{
			int64 Rowid;            // Rowid in main table of the key
			tRowcnt Eq;             // sqlite_stat3.nEq
			tRowcnt Lt;             // sqlite_stat3.nLt
			tRowcnt DLt;            // sqlite_stat3.nDLt
			bool IsPSample;         // True if a periodic sample
			uint32 Hash;            // Tiebreaker hash
		};
		array_t<Stat3Sample> a; // An array of samples
	};

#ifdef ENABLE_STAT3
	__device__ static void Stat3Init_(FuncContext *funcCtx, int argc, Mem **argv)
	{
		tRowcnt rows = (tRowcnt)sqlite3_value_int64(argv[0]);
		int maxSamples = sqlite3_value_int(argv[1]);
		int n = sizeof(*p) + sizeof(p->a[0])*maxSamples;
		Stat3Accum *p = (Stat3Accum *)SysEx::Alloc(n);
		if (!p)
		{
			sqlite3_result_error_nomem(funcCtx);
			return;
		}
		p->a = (Stat3Accum::Stat3Sample *)&p[1];
		p->Rows = rows;
		p->MaxSamples = maxSamples;
		p->PSamples = rows/(maxSamples/3+1) + 1;
		sqlite3_randomness(sizeof(p->Prn), &p->Prn);
		sqlite3_result_blob(funcCtx, p, sizeof(p), SysEx::Free);
	}
	__device__ static const FuncDef Stat3InitFuncdef =
	{
		2,					// nArg
		TEXTENCODE_UTF8,	// iPrefEnc
		(FUNC)0,			// flags
		nullptr,            // pUserData
		nullptr,            // pNext
		Stat3Init_,			// xFunc
		nullptr,            // xStep
		nullptr,            // xFinalize
		"stat3_init",		// zName
		nullptr,            // pHash
		nullptr             // pDestructor
	};

	__device__ static void Stat3Push_(FuncContext *funcCtx, int argc, Mem **argv)
	{
		tRowcnt eq = sqlite3_value_int64(argv[0]);
		if (eq == 0) return;
		tRowcnt lt = sqlite3_value_int64(argv[1]);
		tRowcnt dLt = sqlite3_value_int64(argv[2]);
		int64 rowid = sqlite3_value_int64(argv[3]);
		Stat3Accum *p = (Stat3Accum *)sqlite3_value_blob(argv[4]);
		bool isPSample = false;
		bool doInsert = false;
		int min = p->Min;
		uint32 h = p->Prn = p->Prn*1103515245 + 12345;
		if ((lt/p->PSamples) != ((eq+lt)/p->PSamples)) doInsert = isPSample = true;
		else if (p->a.length < p->MaxSamples) doInsert = true;
		else if (eq > p->a[min].Eq || (eq == p->a[min].Eq && h > p->a[min].Hash)) doInsert = true;
		if (!doInsert) return;
		Stat3Accum::Stat3Sample *sample;
		if (p->a.length == p->MaxSamples)
		{
			_assert(p->a.length - min - 1 >= 0);
			_memmove(&p->a[min], &p->a[min+1], sizeof(p->a[0])*(p->a.length-min-1));
			sample = &p->a[p->a.length-1];
		}
		else
			sample = &p->a[p->a.length++];
		sample->Rowid = rowid;
		sample->Eq = eq;
		sample->Lt = lt;
		sample->DLt = dLt;
		sample->Hash = h;
		sample->IsPSample = isPSample;

		// Find the new minimum
		if (p->a.length == p->MaxSamples)
		{
			sample = p->a;
			int i = 0;
			while (sample->IsPSample)
			{
				i++;
				sample++;
				_assert(i < p->a.length);
			}
			eq = sample->Eq;
			h = sample->Hash;
			min = i;
			for (i++, sample++; i < p->a.length; i++, sample++)
			{
				if (sample->IsPSample) continue;
				if (sample->Eq < eq || (sample->Eq == eq && sample->Hash < h))
				{
					min = i;
					eq = sample->Eq;
					h = sample->Hash;
				}
			}
			p->Min = min;
		}
	}
	__device__ static const FuncDef Stat3PushFuncdef =
	{
		5,					// nArg
		TEXTENCODE_UTF8,    // iPrefEnc
		(FUNC)0,			// flags
		nullptr,            // pUserData
		nullptr,            // pNext
		Stat3Push_,			// xFunc
		nullptr,            // xStep
		nullptr,            // xFinalize
		"stat3_push",		// zName
		nullptr,            // pHash
		nullptr             // pDestructor
	};

	__device__ static void Stat3Get_(FuncContext *funcCtx, int argc, Mem **argv)
	{
		int n = sqlite3_value_int(argv[1]);
		Stat3Accum *p = (Stat3Accum *)sqlite3_value_blob(argv[0]);
		_assert(p);
		if (p->a.length <= n) return;
		switch (argc)
		{
		case 2:  sqlite3_result_int64(funcCtx, p->a[n].Rowid); break;
		case 3:  sqlite3_result_int64(funcCtx, p->a[n].Eq);    break;
		case 4:  sqlite3_result_int64(funcCtx, p->a[n].Lt);    break;
		default: sqlite3_result_int64(funcCtx, p->a[n].DLt);   break;
		}
	}
	__device__ static const FuncDef Stat3GetFuncdef =
	{
		-1,					// nArg
		TEXTENCODE_UTF8,	// iPrefEnc
		(FUNC)0,			// flags
		nullptr,            // pUserData
		nullptr,            // pNext
		Stat3Get_,			// xFunc
		nullptr,            // xStep
		nullptr,            // xFinalize
		"stat3_get",		// zName
		nullptr,            // pHash
		nullptr             // pDestructor
	};

#endif

	__device__ static void AnalyzeOneTable(Parse *parse, Table *table, Index *onlyIdx, int statCurId, int memId)
	{
		Context *ctx = parse->Ctx;    // Database handle
		int i;                       // Loop counter
		int regTabname = memId++;     // Register containing table name
		int regIdxname = memId++;     // Register containing index name
		int regStat1 = memId++;       // The stat column of sqlite_stat1
#ifdef ENABLE_STAT3
		int regNumEq = regStat1;     // Number of instances.  Same as regStat1
		int regNumLt = memId++;       // Number of keys less than regSample
		int regNumDLt = memId++;      // Number of distinct keys less than regSample
		int regSample = memId++;      // The next sample value
		int regRowid = regSample;    // Rowid of a sample
		int regAccum = memId++;       // Register to hold Stat3Accum object
		int regLoop = memId++;        // Loop counter
		int regCount = memId++;       // Number of rows in the table or index
		int regTemp1 = memId++;       // Intermediate register
		int regTemp2 = memId++;       // Intermediate register
		int tabCurId = parse->Tabs++; // Table cursor
#endif
		int regCol = memId++;         // Content of a column in analyzed table
		int regRec = memId++;         // Register holding completed record
		int regTemp = memId++;        // Temporary use register
		int regNewRowid = memId++;    // Rowid for the inserted record

		Vdbe *v = parse->GetVdbe(); // The virtual machine being built up
		if (!v || SysEx_NEVER(!table))
			return;
		if (table->Id == 0 || !_strncmp(table->Name, "sqlite_", 7)) // Do not gather statistics on views or virtual tables + Do not gather statistics on system tables
			return;
		_assert(Btree::HoldsAllMutexes(ctx));
		int db = sqlite3SchemaToIndex(ctx, table->Schema); // Index of database containing table
		_assert(db >= 0);
		_assert(Btree::SchemaMutexHeld(ctx, db, 0));
#ifndef OMIT_AUTHORIZATION
		if (Auth::Check(parse, AUTH_ANALYZE, table->Name, 0, ctx->DBs[db].Name))
			return;
#endif

		// Establish a read-lock on the table at the shared-cache level.
		sqlite3TableLock(parse, db, table->Id, 0, table->Name);

		int zeroRows = -1;          // Jump from here if number of rows is zero
		int idxCurId = parse->Tabs++; // Cursor open on index being analyzed
		v->AddOp4(OP_String8, 0, regTabname, 0, table->Name, 0);
		for (Index *idx = table->Index; idx; idx = idx->Next) // An index to being analyzed
		{
			if (onlyIdx && onlyIdx != idx) continue;
			v->VdbeNoopComment("Begin analysis of %s", idx->Name);
			int cols = idx->Columns.length;
			int *chngAddrs = (int *)SysEx::TagAlloc(ctx, sizeof(int)*cols); // Array of jump instruction addresses
			if (!chngAddrs) continue;
			KeyInfo *key = sqlite3IndexKeyinfo(parse, idx);
			if (memId+1+(cols*2) > parse->Mems)
				parse->Mems = memId+1+(cols*2);

			// Open a cursor to the index to be analyzed.
			_assert(db == sqlite3SchemaToIndex(ctx, idx->Schema));
			v->AddOp4(OP_OpenRead, idxCurId, idx->Id, db, (char *)key, Vdbe::P4T_KEYINFO_HANDOFF);
			v->VdbeComment("%s", idx->Name);

			// Populate the register containing the index name.
			v->AddOp4(OP_String8, 0, regIdxname, 0, idx->Name, 0);

#ifdef ENABLE_STAT3
			bool once = false; // One-time initialization
			if (once)
			{
				once = false;
				sqlite3OpenTable(parse, tabCurId, db, table, OP_OpenRead);
			}
			v->AddOp2(OP_Count, idxCurId, regCount);
			v->AddOp2(OP_Integer, SQLITE_STAT3_SAMPLES, regTemp1);
			v->AddOp2(OP_Integer, 0, regNumEq);
			v->AddOp2(OP_Integer, 0, regNumLt);
			v->AddOp2(OP_Integer, -1, regNumDLt);
			v->AddOp3(OP_Null, 0, regSample, regAccum);
			v->AddOp4(OP_Function, 1, regCount, regAccum, (char*)&Stat3InitFuncdef, Vdbe::P4T_FUNCDEF);
			v->ChangeP5(2);
#endif

			// The block of memory cells initialized here is used as follows.
			//
			//    memId:                
			//        The total number of rows in the table.
			//
			//    memId+1 .. memId+nCol: 
			//        Number of distinct entries in index considering the left-most N columns only, where N is between 1 and nCol, 
			//        inclusive.
			//
			//    memId+nCol+1 .. Mem+2*nCol:  
			//        Previous value of indexed columns, from left to right.
			//
			// Cells memId through memId+nCol are initialized to 0. The others are initialized to contain an SQL NULL.
			for (i = 0; i <= cols; i++)
				v->AddOp2(OP_Integer, 0, memId+i);
			for (i = 0; i < cols; i++)
				v->AddOp2(OP_Null, 0, memId+cols+i+1);

			// Start the analysis loop. This loop runs through all the entries in the index b-tree.
			int endOfLoop = v->MakeLabel(); // The end of the loop
			v->AddOp2(OP_Rewind, idxCurId, endOfLoop);
			int topOfLoop = v->CurrentAddr(); // The top of the loop
			v->AddOp2(OP_AddImm, memId, 1);  // Increment row counter

			int addrIfNot = 0; // address of OP_IfNot
			for (i = 0; i < cols; i++)
			{
				v->AddOp3(OP_Column, idxCurId, i, regCol);
				if (i == 0) // Always record the very first row
					addrIfNot = v->AddOp1(OP_IfNot, memId+1);
				_assert(idx->CollNames && idx->CollNames[i]);
				CollSeq *coll = sqlite3LocateCollSeq(parse, idx->CollNames[i]);
				chngAddrs[i] = v->AddOp4(OP_Ne, regCol, 0, memId+cols+i+1, (char *)coll, Vdbe::P4T_COLLSEQ);
				v->ChangeP5(SQLITE_NULLEQ);
				v->VdbeComment("jump if column %d changed", i);
#ifdef ENABLE_STAT3
				if (i == 0)
				{
					v->AddOp2(OP_AddImm, regNumEq, 1);
					v->VdbeComment("incr repeat count");
				}
#endif
			}
			v->AddOp2(OP_Goto, 0, endOfLoop);
			for (i = 0; i < cols; i++)
			{
				v->JumpHere(chngAddrs[i]); // Set jump dest for the OP_Ne
				if (i == 0)
				{
					v->JumpHere(addrIfNot); // Jump dest for OP_IfNot
#ifdef ENABLE_STAT3
					v->AddOp4(OP_Function, 1, regNumEq, regTemp2, (char *)&Stat3PushFuncdef, Vdbe::P4T_FUNCDEF);
					v->ChangeP5(5);
					v->AddOp3(OP_Column, idxCurId, idx->Columns.length, regRowid);
					v->AddOp3(OP_Add, regNumEq, regNumLt, regNumLt);
					v->AddOp2(OP_AddImm, regNumDLt, 1);
					v->AddOp2(OP_Integer, 1, regNumEq);
#endif        
				}
				v->AddOp2(OP_AddImm, memId+i+1, 1);
				v->AddOp3(OP_Column, idxCurId, i, memId+cols+i+1);
			}
			SysEx::TagFree(ctx, chngAddrs);

			// Always jump here after updating the memId+1...memId+1+nCol counters
			v->ResolveLabel(endOfLoop);

			v->AddOp2(OP_Next, idxCurId, topOfLoop);
			v->AddOp1(OP_Close, idxCurId);
#ifdef ENABLE_STAT3
			v->AddOp4(OP_Function, 1, regNumEq, regTemp2, (char *)&Stat3PushFuncdef, Vdbe::P4T_FUNCDEF);
			v->ChangeP5(5);
			v->AddOp2(OP_Integer, -1, regLoop);
			int shortJump =  v->AddOp2(OP_AddImm, regLoop, 1); // Instruction address
			v->AddOp4(OP_Function, 1, regAccum, regTemp1, (char *)&Stat3GetFuncdef, Vdbe::P4T_FUNCDEF);
			v->ChangeP5(2);
			v->AddOp1(OP_IsNull, regTemp1);
			v->AddOp3(OP_NotExists, tabCurId, shortJump, regTemp1);
			v->AddOp3(OP_Column, tabCurId, idx->Columns[0], regSample);
			sqlite3ColumnDefault(v, table, idx->Columns[0], regSample);
			v->AddOp4(OP_Function, 1, regAccum, regNumEq, (char *)&Stat3GetFuncdef, Vdbe::P4T_FUNCDEF);
			v->ChangeP5(3);
			v->AddOp4(OP_Function, 1, regAccum, regNumLt, (char *)&Stat3GetFuncdef, Vdbe::P4T_FUNCDEF);
			v->ChangeP5(4);
			v->AddOp4(OP_Function, 1, regAccum, regNumDLt, (char *)&Stat3GetFuncdef, Vdbe::P4T_FUNCDEF);
			v->ChangeP5(5);
			v->AddOp4(OP_MakeRecord, regTabname, 6, regRec, "bbbbbb", 0);
			v->AddOp2(OP_NewRowid, statCurId+1, regNewRowid);
			v->AddOp3(OP_Insert, statCurId+1, regRec, regNewRowid);
			v->AddOp2(OP_Goto, 0, shortJump);
			v->JumpHere(shortJump+2);
#endif        

			// Store the results in sqlite_stat1.
			//
			// The result is a single row of the sqlite_stat1 table.  The first two columns are the names of the table and index.  The third column
			// is a string composed of a list of integer statistics about the index.  The first integer in the list is the total number of entries
			// in the index.  There is one additional integer in the list for each column of the table.  This additional integer is a guess of how many
			// rows of the table the index will select.  If D is the count of distinct values and K is the total number of rows, then the integer is computed
			// as:
			//        I = (K+D-1)/D
			// If K==0 then no entry is made into the sqlite_stat1 table.  
			// If K>0 then it is always the case the D>0 so division by zero is never possible.
			v->AddOp2(OP_SCopy, memId, regStat1);
			if (zeroRows < 0)
				zeroRows = v->AddOp1(OP_IfNot, memId);
			for (i = 0; i < cols; i++)
			{
				v->AddOp4(OP_String8, 0, regTemp, 0, " ", 0);
				v->AddOp3(OP_Concat, regTemp, regStat1, regStat1);
				v->AddOp3(OP_Add, memId, memId+i+1, regTemp);
				v->AddOp2(OP_AddImm, regTemp, -1);
				v->AddOp3(OP_Divide, memId+i+1, regTemp, regTemp);
				v->AddOp1(OP_ToInt, regTemp);
				v->AddOp3(OP_Concat, regTemp, regStat1, regStat1);
			}
			v->AddOp4(OP_MakeRecord, regTabname, 3, regRec, "aaa", 0);
			v->AddOp2(OP_NewRowid, statCurId, regNewRowid);
			v->AddOp3(OP_Insert, statCurId, regRec, regNewRowid);
			v->ChangeP5(OPFLAG_APPEND);
		}

		// If the table has no indices, create a single sqlite_stat1 entry containing NULL as the index name and the row count as the content.
		if (!table->Index)
		{
			v->AddOp3(OP_OpenRead, idxCurId, table->Id, db);
			v->VdbeComment("%s", table->Name);
			v->AddOp2(OP_Count, idxCurId, regStat1);
			v->AddOp1(OP_Close, idxCurId);
			zeroRows = v->AddOp1(OP_IfNot, regStat1);
		}
		else
		{
			v->JumpHere(zeroRows);
			zeroRows = v->AddOp0(OP_Goto);
		}
		v->AddOp2(OP_Null, 0, regIdxname);
		v->AddOp4(OP_MakeRecord, regTabname, 3, regRec, "aaa", 0);
		v->AddOp2(OP_NewRowid, statCurId, regNewRowid);
		v->AddOp3(OP_Insert, statCurId, regRec, regNewRowid);
		v->ChangeP5(OPFLAG_APPEND);
		if (parse->Mems < regRec) parse->Mems = regRec;
		v->JumpHere(zeroRows);
	}

	__device__ static void LoadAnalysis(Parse *parse, int db)
	{
		Vdbe *v = parse->GetVdbe();
		if (v)
			v->AddOp1(OP_LoadAnalysis, db);
	}

	__device__ static void AnalyzeDatabase(Parse *parse, int db)
	{
		Context *ctx = parse->Ctx;
		Schema *schema = ctx->DBs[db].Schema; // Schema of database db
		parse->BeginWriteOperation(0, db);
		int statCurId = parse->Tabs;
		parse->Tabs += 3;
		OpenStatTable(parse, db, statCurId, 0, 0);
		int memId = parse->Mems+1;
		_assert(sqlite3SchemaMutexHeld(ctx, db, 0));
		for (HashElem *k = schema->TableHash.First; k; k = k->Next)
		{
			Table *table = (Table *)k->Data;
			AnalyzeOneTable(parse, table, 0, statCurId, memId);
		}
		LoadAnalysis(parse, db);
	}

	__device__ static void AnalyzeTable(Parse *parse, Table *table, Index *onlyIdx)
	{
		_assert(table && Btree::HoldsAllMutexes(parse->Ctx));
		int db = Btree::SchemaToIndex(parse->Ctx, table->Schema);
		parse->BeginWriteOperation(0, db);
		int statCurId = parse->Tabs;
		parse->Tabs += 3;
		if (onlyIdx)

			OpenStatTable(parse, db, statCurId, onlyIdx->Name, "idx");
		else
			OpenStatTable(parse, db, statCurId, table->Name, "tbl");
		AnalyzeOneTable(parse, table, onlyIdx, statCurId, parse->Mems+1);
		LoadAnalysis(parse, db);
	}

	__device__ void Analyze::Analyze(Parse *parse, Token *name1, Token *name2)
	{
		Context *ctx = parse->Ctx;
		// Read the database schema. If an error occurs, leave an error message and code in pParse and return NULL.
		_assert(Btree::HoldsAllMutexes(ctx));
		if (sqlite3ReadSchema(parse) != RC_OK)
			return;

		_assert(name2 || name1);
		int db;
		if (!name1)
		{
			// Form 1:  Analyze everything
			for (int i = 0; i < ctx->DBs.length; i++)
			{
				if (i == 1) continue; // Do not analyze the TEMP database
				AnalyzeDatabase(parse, i);
			}
		}
		else if (name2->length == 0)
		{
			// Form 2:  Analyze the database or table named
			int db = sqlite3FindDb(ctx, name1);
			if (db >= 0)
				AnalyzeDatabase(parse, db);
			else
			{
				char *z = sqlite3NameFromToken(ctx, name1);
				if (z)
				{
					Table *table;
					Index *idx;
					if ((idx = sqlite3FindIndex(ctx, z, nullptr)) != nullptr)
						AnalyzeTable(parse, idx->Table, idx);
					else if ((table = sqlite3LocateTable(parse, 0, z, nullptr)) != nullptr)
						AnalyzeTable(parse, table, nullptr);
					SysEx::TagFree(ctx, z);
				}
			}
		}
		else
		{
			// Form 3: Analyze the fully qualified table name
			Token *tableName;
			int db = sqlite3TwoPartName(parse, name1, name2, &tableName);
			if (db >= 0)
			{
				char *dbName = ctx->DBs[db].Name;
				char *z = sqlite3NameFromToken(ctx, tableName);
				if (z)
				{
					Table *table;
					Index *idx;
					if ((idx = sqlite3FindIndex(ctx, z, dbName)) != nullptr)
						AnalyzeTable(parse, idx->Table, idx);
					else if ((table = sqlite3LocateTable(parse, 0, z, dbName)) != nullptr)
						AnalyzeTable(parse, table, nullptr);
					SysEx::TagFree(ctx, z);
				}
			}   
		}
	}

	struct AnalysisInfo
	{
		Context *Ctx;
		const char *Database;
	};

	__device__ static int AnalysisLoader(void *data, int argc, char **argv, char **notUsed)
	{
		AnalysisInfo *info = (AnalysisInfo *)data;
		_assert(argc == 3);
		if (argv == 0 || argv[0] == 0 || argv[2] == 0)
			return 0;
		Table *table = sqlite3FindTable(info->Ctx, argv[0], info->Database);
		if (!table)
			return 0;
		Index *index = (argv[1] ? sqlite3FindIndex(info->Ctx, argv[1], info->Database) : nullptr);
		int n = (index ? index->Columns.length : 0);
		const char *z = argv[2];
		for (int i = 0; *z && i <= n; i++)
		{
			tRowcnt v = 0;
			int c;
			while ((c = z[0]) >= '0' && c <= '9')
			{
				v = v*10 + c - '0';
				z++;
			}
			if (i == 0) table->RowEst = v;
			if (!index) break;
			index->RowEsts[i] = v;
			if (*z == ' ') z++;
			if (!_strcmp(z, "unordered"))
			{
				index->Unordered = true;
				break;
			}
		}
		return 0;
	}

	__device__ void Analyze::DeleteIndexSamples(Context *ctx, Index *idx)
	{
#ifdef ENABLE_STAT3
		if (idx->Samples)
		{
			for (int j = 0; j < idx->Samples.length; j++)
			{
				IndexSample *p = &idx->Samples[j];
				if (p->Type == TYPE_TEXT || p->Type == TYPE_BLOB)
					SysEx::TagFree(ctx, p->u.Z);
			}
			SysEx::TagFree(ctx, idx->Samples);
		}
		if (ctx && ctx->BytesFreed == 0)
		{
			idx->Samples.length = 0;
			idx->Samples.data = nullptr;
		}
#endif
	}

#ifdef ENABLE_STAT3
	__device__ static RC LoadStat3(Context *ctx, const char *dbName)
	{
		_assert(!ctx->Lookaside.Enabled);
		if (!sqlite3FindTable(ctx, "sqlite_stat3", dbName))
			return RC_OK;

		char *sql = SysEx::Mprintf(ctx, 
			"SELECT idx,count(*) FROM %Q.sqlite_stat3 GROUP BY idx", dbName); // Text of the SQL statement
		if (!sql)
			return RC_NOMEM;
		sqlite3_stmt *stmt = nullptr; // An SQL statement being run
		RC rc = sqlite3_prepare(ctx, sql, -1, &stmt, 0); // Result codes from subroutines
		SysEx::TagFree(ctx, sql);
		if (rc) return rc;

		while (sqlite3_step(stmt) == SQLITE_ROW)
		{
			char *indexName = (char *)sqlite3_column_text(stmt, 0); // Index name
			if (!indexName) continue;
			int samplesLength = sqlite3_column_int(stmt, 1); // Number of samples
			Index *idx = sqlite3FindIndex(ctx, indexName, dbName); // Pointer to the index object
			if (!idx) continue;
			_assert(idx->Samples.length == 0);
			idx->Samples.length = samplesLength;
			idx->Samples.data = SysEx::TagAlloc(ctx, samplesLength*sizeof(IndexSample));
			idx->AvgEq = idx->aiRowEst[1];
			if (!idx->Samples.data)
			{
				ctx->MallocFailed = true;
				sqlite3_finalize(stmt);
				return RC_NOMEM;
			}
		}
		rc = sqlite3_finalize(stmt);
		if (rc) return rc;

		sql = SysEx::Mprintf(ctx, 
			"SELECT idx,neq,nlt,ndlt,sample FROM %Q.sqlite_stat3", dbName);
		if (!sql)
			return RC_NOMEM;
		rc = sqlite3_prepare(ctx, sql, -1, &stmt, 0);
		SysEx::TagFree(ctx, sql);
		if (rc) return rc;

		Index *prevIdx = nullptr; // Previous index in the loop
		int idxId = 0; // slot in pIdx->aSample[] for next sample
		while (sqlite3_step(stmt) == SQLITE_ROW)
		{
			char *indexName = (char *)sqlite3_column_text(stmt, 0); // Index name
			if (!indexName) continue;
			Index *idx = sqlite3FindIndex(ctx, indexName, dbName); // Pointer to the index object
			if (!idx) continue;
			if (idx == prevIdx)
				idxId++;
			else
			{
				prevIdx = idx;
				idxId = 0;
			}
			_assert(idxId < idx->Samples.length);
			IndexSample *sample = &idx->Samples[idxId]; // A slot in pIdx->aSample[]
			sample->Eq = (tRowcnt)sqlite3_column_int64(stmt, 1);
			sample->Lt = (tRowcnt)sqlite3_column_int64(stmt, 2);
			sample->DLt = (tRowcnt)sqlite3_column_int64(stmt, 3);
			if (idxId == idx->Samples.length-1)
			{
				if (sample->DLt > 0)
				{
					tRowcnt sumEq;  // Sum of the nEq values
					for (int i = 0, sumEq = 0; i <= idxId-1; i++) sumEq += idx->Samples[i].Eq;
					idx->AvgEq = (sample->Lt - sumEq) / sample->DLt;
				}
				if (idx->AvgEq <= 0) idx->AvgEq = 1;
			}
			TYPE type = sqlite3_column_type(stmt, 4); // Datatype of a sample
			sample->Type = type;
			switch (type)
			{
			case TYPE_INTEGER: {
				sample->u.I = sqlite3_column_int64(stmt, 4);
				break; }
			case TYPE_FLOAT: {
				sample->u.R = sqlite3_column_double(stmt, 4);
				break; }
			case TYPE_NULL: {
				break; }
			default: _assert(type == TYPE_TEXT || type == TYPE_BLOB ); {
				const char *z = (const char *)(type == TYPE_BLOB ? sqlite3_column_blob(stmt, 4) : sqlite3_column_text(stmt, 4));
				int n = (z ? sqlite3_column_bytes(stmt, 4) : 0);
				sample->Bytes = n;
				if (n < 1 )
					sample->u.Z = nullptr;
				else
				{
					sample->u.Z = SysEx::TagAlloc(ctx, n);
					if (!sample->u.Z)
					{
						ctx->MallocFailed = true;
						sqlite3_finalize(stmt);
						return RC_NOMEM;
					}
					_memcpy(sample->u.Z, z, n);
				} }
			}
		}
		return sqlite3_finalize(stmt);
	}
#endif

	__device__ RC Analyze::AnalysisLoad(Context *ctx, int db)
	{
		_assert(db >= 0 && db < ctx->DBs.length);
		_assert(ctx->DBs[db].Bt != nullptr);

		// Clear any prior statistics
		_assert(Btree::SchemaMutexHeld(ctx, db, 0));
		for (HashElem *i = ctx->DBs[db].Schema->IndexHash.First; i; i = i->Next)
		{
			Index *idx = (Index *)i->Data;
			sqlite3DefaultRowEst(idx);
#ifdef ENABLE_STAT3
			sqlite3DeleteIndexSamples(ctx, idx);
			idx->Samples.data = nullptr;
#endif
		}

		// Check to make sure the sqlite_stat1 table exists
		AnalysisInfo sInfo;
		sInfo.Ctx = ctx;
		sInfo.Database = ctx->DBs[db].Name;
		if (sqlite3FindTable(ctx, "sqlite_stat1", sInfo.Database) == 0)
			return RC_ERROR;

		// Load new statistics out of the sqlite_stat1 table
		char *sql = SysEx::Mprintf(ctx, 
			"SELECT tbl,idx,stat FROM %Q.sqlite_stat1", sInfo.Database);
		RC rc;
		if (!sql)
			rc = RC_NOMEM;
		else
		{
			rc = sqlite3_exec(ctx, sql, AnalysisLoader, &sInfo, 0);
			SysEx::TagFree(ctx, sql);
		}

		// Load the statistics from the sqlite_stat3 table.
#ifdef ENABLE_STAT3
		if (rc == RC_OK)
		{
			bool lookasideEnabled = ctx->Lookaside.Enabled;
			ctx->Lookaside.Enabled = false;
			rc = LoadStat3(ctx, sInfo.Database);
			ctx->Lookaside.Enabled = lookasideEnabled;
		}
#endif

		if (rc == RC_NOMEM)
			ctx->MallocFailed = true;
		return rc;
	}

} }
#endif
#pragma endregion