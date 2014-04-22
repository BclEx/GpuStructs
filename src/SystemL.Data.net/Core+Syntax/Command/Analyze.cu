// analyze.c
#pragma region OMIT_ANALYZE
#ifndef OMIT_ANALYZE
#include "..\Core+Syntax.cu.h"

namespace Core { namespace Command
{
	static const struct {
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
		Context::DB *dbAsObj = &ctx->DBs[db];

		// Create new statistic tables if they do not exist, or clear them if they do already exist.
		for (int i = 0; i < __arrayStaticLength(_tables); i++)
		{
			const char *table = _tables[i].Name;
			Table *stat;
			if ((stat = sqlite3FindTable(db, table, dbAsObj->Name)) == nullptr)
			{
				// The sqlite_stat[12] table does not exist. Create it. Note that a side-effect of the CREATE TABLE statement is to leave the rootpage 
				// of the new table in register pParse->regRoot. This is important because the OpenWrite opcode below will be needing it.
				parse->NestedParse("CREATE TABLE %Q.%s(%s)", dbAsObj->Name, table, _tables[i].Cols);
				roots[i] = parse->RegRoot;
				createTbls[i] = OPFLAG_P2ISREG;
			}
			else
			{
				// The table already exists. If zWhere is not NULL, delete all entries associated with the table zWhere. If zWhere is NULL, delete the
				// entire contents of the table.
				roots[i] = stat->tnum;
				sqlite3TableLock(parse, db, roots[i], 1, table);
				if (where_)
					parse->NestedParse("DELETE FROM %Q.%s WHERE %s=%Q", dbAsObj->Name, table, whereType, where_);
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

	/*
	** Three SQL functions - stat3_init(), stat3_push(), and stat3_pop() -
	** share an instance of the following structure to hold their state
	** information.
	*/
	typedef struct Stat3Accum Stat3Accum;
	struct Stat3Accum {
		tRowcnt nRow;             /* Number of rows in the entire table */
		tRowcnt nPSample;         /* How often to do a periodic sample */
		int iMin;                 /* Index of entry with minimum nEq and hash */
		int mxSample;             /* Maximum number of samples to accumulate */
		int nSample;              /* Current number of samples */
		u32 iPrn;                 /* Pseudo-random number used for sampling */
		struct Stat3Sample {
			i64 iRowid;                /* Rowid in main table of the key */
			tRowcnt nEq;               /* sqlite_stat3.nEq */
			tRowcnt nLt;               /* sqlite_stat3.nLt */
			tRowcnt nDLt;              /* sqlite_stat3.nDLt */
			u8 isPSample;              /* True if a periodic sample */
			u32 iHash;                 /* Tiebreaker hash */
		} *a;                     /* An array of samples */
	};

#ifdef ENABLE_STAT3
	/*
	** Implementation of the stat3_init(C,S) SQL function.  The two parameters
	** are the number of rows in the table or index (C) and the number of samples
	** to accumulate (S).
	**
	** This routine allocates the Stat3Accum object.
	**
	** The return value is the Stat3Accum object (P).
	*/
	static void stat3Init(
		sqlite3_context *context,
		int argc,
		sqlite3_value **argv
		){
			Stat3Accum *p;
			tRowcnt nRow;
			int mxSample;
			int n;

			UNUSED_PARAMETER(argc);
			nRow = (tRowcnt)sqlite3_value_int64(argv[0]);
			mxSample = sqlite3_value_int(argv[1]);
			n = sizeof(*p) + sizeof(p->a[0])*mxSample;
			p = sqlite3MallocZero( n );
			if( p==0 ){
				sqlite3_result_error_nomem(context);
				return;
			}
			p->a = (struct Stat3Sample*)&p[1];
			p->nRow = nRow;
			p->mxSample = mxSample;
			p->nPSample = p->nRow/(mxSample/3+1) + 1;
			sqlite3_randomness(sizeof(p->iPrn), &p->iPrn);
			sqlite3_result_blob(context, p, sizeof(p), sqlite3_free);
	}
	static const FuncDef stat3InitFuncdef = {
		2,                /* nArg */
		SQLITE_UTF8,      /* iPrefEnc */
		0,                /* flags */
		0,                /* pUserData */
		0,                /* pNext */
		stat3Init,        /* xFunc */
		0,                /* xStep */
		0,                /* xFinalize */
		"stat3_init",     /* zName */
		0,                /* pHash */
		0                 /* pDestructor */
	};


	/*
	** Implementation of the stat3_push(nEq,nLt,nDLt,rowid,P) SQL function.  The
	** arguments describe a single key instance.  This routine makes the 
	** decision about whether or not to retain this key for the sqlite_stat3
	** table.
	**
	** The return value is NULL.
	*/
	static void stat3Push(
		sqlite3_context *context,
		int argc,
		sqlite3_value **argv
		){
			Stat3Accum *p = (Stat3Accum*)sqlite3_value_blob(argv[4]);
			tRowcnt nEq = sqlite3_value_int64(argv[0]);
			tRowcnt nLt = sqlite3_value_int64(argv[1]);
			tRowcnt nDLt = sqlite3_value_int64(argv[2]);
			i64 rowid = sqlite3_value_int64(argv[3]);
			u8 isPSample = 0;
			u8 doInsert = 0;
			int iMin = p->iMin;
			struct Stat3Sample *pSample;
			int i;
			u32 h;

			UNUSED_PARAMETER(context);
			UNUSED_PARAMETER(argc);
			if( nEq==0 ) return;
			h = p->iPrn = p->iPrn*1103515245 + 12345;
			if( (nLt/p->nPSample)!=((nEq+nLt)/p->nPSample) ){
				doInsert = isPSample = 1;
			}else if( p->nSample<p->mxSample ){
				doInsert = 1;
			}else{
				if( nEq>p->a[iMin].nEq || (nEq==p->a[iMin].nEq && h>p->a[iMin].iHash) ){
					doInsert = 1;
				}
			}
			if( !doInsert ) return;
			if( p->nSample==p->mxSample ){
				assert( p->nSample - iMin - 1 >= 0 );
				memmove(&p->a[iMin], &p->a[iMin+1], sizeof(p->a[0])*(p->nSample-iMin-1));
				pSample = &p->a[p->nSample-1];
			}else{
				pSample = &p->a[p->nSample++];
			}
			pSample->iRowid = rowid;
			pSample->nEq = nEq;
			pSample->nLt = nLt;
			pSample->nDLt = nDLt;
			pSample->iHash = h;
			pSample->isPSample = isPSample;

			/* Find the new minimum */
			if( p->nSample==p->mxSample ){
				pSample = p->a;
				i = 0;
				while( pSample->isPSample ){
					i++;
					pSample++;
					assert( i<p->nSample );
				}
				nEq = pSample->nEq;
				h = pSample->iHash;
				iMin = i;
				for(i++, pSample++; i<p->nSample; i++, pSample++){
					if( pSample->isPSample ) continue;
					if( pSample->nEq<nEq
						|| (pSample->nEq==nEq && pSample->iHash<h)
						){
							iMin = i;
							nEq = pSample->nEq;
							h = pSample->iHash;
					}
				}
				p->iMin = iMin;
			}
	}
	static const FuncDef stat3PushFuncdef = {
		5,                /* nArg */
		SQLITE_UTF8,      /* iPrefEnc */
		0,                /* flags */
		0,                /* pUserData */
		0,                /* pNext */
		stat3Push,        /* xFunc */
		0,                /* xStep */
		0,                /* xFinalize */
		"stat3_push",     /* zName */
		0,                /* pHash */
		0                 /* pDestructor */
	};

	/*
	** Implementation of the stat3_get(P,N,...) SQL function.  This routine is
	** used to query the results.  Content is returned for the Nth sqlite_stat3
	** row where N is between 0 and S-1 and S is the number of samples.  The
	** value returned depends on the number of arguments.
	**
	**   argc==2    result:  rowid
	**   argc==3    result:  nEq
	**   argc==4    result:  nLt
	**   argc==5    result:  nDLt
	*/
	static void stat3Get(
		sqlite3_context *context,
		int argc,
		sqlite3_value **argv
		){
			int n = sqlite3_value_int(argv[1]);
			Stat3Accum *p = (Stat3Accum*)sqlite3_value_blob(argv[0]);

			assert( p!=0 );
			if( p->nSample<=n ) return;
			switch( argc ){
			case 2:  sqlite3_result_int64(context, p->a[n].iRowid); break;
			case 3:  sqlite3_result_int64(context, p->a[n].nEq);    break;
			case 4:  sqlite3_result_int64(context, p->a[n].nLt);    break;
			default: sqlite3_result_int64(context, p->a[n].nDLt);   break;
			}
	}
	static const FuncDef stat3GetFuncdef = {
		-1,               /* nArg */
		SQLITE_UTF8,      /* iPrefEnc */
		0,                /* flags */
		0,                /* pUserData */
		0,                /* pNext */
		stat3Get,         /* xFunc */
		0,                /* xStep */
		0,                /* xFinalize */
		"stat3_get",     /* zName */
		0,                /* pHash */
		0                 /* pDestructor */
	};
#endif

	static void analyzeOneTable(Parse *pParse, Table *pTab, Index *pOnlyIdx, int iStatCur, int iMem)
	{
		sqlite3 *db = pParse->db;    /* Database handle */
		Index *pIdx;                 /* An index to being analyzed */
		int iIdxCur;                 /* Cursor open on index being analyzed */
		Vdbe *v;                     /* The virtual machine being built up */
		int i;                       /* Loop counter */
		int topOfLoop;               /* The top of the loop */
		int endOfLoop;               /* The end of the loop */
		int jZeroRows = -1;          /* Jump from here if number of rows is zero */
		int iDb;                     /* Index of database containing pTab */
		int regTabname = iMem++;     /* Register containing table name */
		int regIdxname = iMem++;     /* Register containing index name */
		int regStat1 = iMem++;       /* The stat column of sqlite_stat1 */
#ifdef SQLITE_ENABLE_STAT3
		int regNumEq = regStat1;     /* Number of instances.  Same as regStat1 */
		int regNumLt = iMem++;       /* Number of keys less than regSample */
		int regNumDLt = iMem++;      /* Number of distinct keys less than regSample */
		int regSample = iMem++;      /* The next sample value */
		int regRowid = regSample;    /* Rowid of a sample */
		int regAccum = iMem++;       /* Register to hold Stat3Accum object */
		int regLoop = iMem++;        /* Loop counter */
		int regCount = iMem++;       /* Number of rows in the table or index */
		int regTemp1 = iMem++;       /* Intermediate register */
		int regTemp2 = iMem++;       /* Intermediate register */
		int once = 1;                /* One-time initialization */
		int shortJump = 0;           /* Instruction address */
		int iTabCur = pParse->nTab++; /* Table cursor */
#endif
		int regCol = iMem++;         /* Content of a column in analyzed table */
		int regRec = iMem++;         /* Register holding completed record */
		int regTemp = iMem++;        /* Temporary use register */
		int regNewRowid = iMem++;    /* Rowid for the inserted record */


		v = sqlite3GetVdbe(pParse);
		if( v==0 || NEVER(pTab==0) ){
			return;
		}
		if( pTab->tnum==0 ){
			/* Do not gather statistics on views or virtual tables */
			return;
		}
		if( sqlite3_strnicmp(pTab->zName, "sqlite_", 7)==0 ){
			/* Do not gather statistics on system tables */
			return;
		}
		assert( sqlite3BtreeHoldsAllMutexes(db) );
		iDb = sqlite3SchemaToIndex(db, pTab->pSchema);
		assert( iDb>=0 );
		assert( sqlite3SchemaMutexHeld(db, iDb, 0) );
#ifndef SQLITE_OMIT_AUTHORIZATION
		if( sqlite3AuthCheck(pParse, SQLITE_ANALYZE, pTab->zName, 0,
			db->aDb[iDb].zName ) ){
				return;
		}
#endif

		/* Establish a read-lock on the table at the shared-cache level. */
		sqlite3TableLock(pParse, iDb, pTab->tnum, 0, pTab->zName);

		iIdxCur = pParse->nTab++;
		sqlite3VdbeAddOp4(v, OP_String8, 0, regTabname, 0, pTab->zName, 0);
		for(pIdx=pTab->pIndex; pIdx; pIdx=pIdx->pNext){
			int nCol;
			KeyInfo *pKey;
			int addrIfNot = 0;           /* address of OP_IfNot */
			int *aChngAddr;              /* Array of jump instruction addresses */

			if( pOnlyIdx && pOnlyIdx!=pIdx ) continue;
			VdbeNoopComment((v, "Begin analysis of %s", pIdx->zName));
			nCol = pIdx->nColumn;
			aChngAddr = sqlite3DbMallocRaw(db, sizeof(int)*nCol);
			if( aChngAddr==0 ) continue;
			pKey = sqlite3IndexKeyinfo(pParse, pIdx);
			if( iMem+1+(nCol*2)>pParse->nMem ){
				pParse->nMem = iMem+1+(nCol*2);
			}

			/* Open a cursor to the index to be analyzed. */
			assert( iDb==sqlite3SchemaToIndex(db, pIdx->pSchema) );
			sqlite3VdbeAddOp4(v, OP_OpenRead, iIdxCur, pIdx->tnum, iDb,
				(char *)pKey, P4_KEYINFO_HANDOFF);
			VdbeComment((v, "%s", pIdx->zName));

			/* Populate the register containing the index name. */
			sqlite3VdbeAddOp4(v, OP_String8, 0, regIdxname, 0, pIdx->zName, 0);

#ifdef SQLITE_ENABLE_STAT3
			if( once ){
				once = 0;
				sqlite3OpenTable(pParse, iTabCur, iDb, pTab, OP_OpenRead);
			}
			sqlite3VdbeAddOp2(v, OP_Count, iIdxCur, regCount);
			sqlite3VdbeAddOp2(v, OP_Integer, SQLITE_STAT3_SAMPLES, regTemp1);
			sqlite3VdbeAddOp2(v, OP_Integer, 0, regNumEq);
			sqlite3VdbeAddOp2(v, OP_Integer, 0, regNumLt);
			sqlite3VdbeAddOp2(v, OP_Integer, -1, regNumDLt);
			sqlite3VdbeAddOp3(v, OP_Null, 0, regSample, regAccum);
			sqlite3VdbeAddOp4(v, OP_Function, 1, regCount, regAccum,
				(char*)&stat3InitFuncdef, P4_FUNCDEF);
			sqlite3VdbeChangeP5(v, 2);
#endif /* SQLITE_ENABLE_STAT3 */

			/* The block of memory cells initialized here is used as follows.
			**
			**    iMem:                
			**        The total number of rows in the table.
			**
			**    iMem+1 .. iMem+nCol: 
			**        Number of distinct entries in index considering the 
			**        left-most N columns only, where N is between 1 and nCol, 
			**        inclusive.
			**
			**    iMem+nCol+1 .. Mem+2*nCol:  
			**        Previous value of indexed columns, from left to right.
			**
			** Cells iMem through iMem+nCol are initialized to 0. The others are 
			** initialized to contain an SQL NULL.
			*/
			for(i=0; i<=nCol; i++){
				sqlite3VdbeAddOp2(v, OP_Integer, 0, iMem+i);
			}
			for(i=0; i<nCol; i++){
				sqlite3VdbeAddOp2(v, OP_Null, 0, iMem+nCol+i+1);
			}

			/* Start the analysis loop. This loop runs through all the entries in
			** the index b-tree.  */
			endOfLoop = sqlite3VdbeMakeLabel(v);
			sqlite3VdbeAddOp2(v, OP_Rewind, iIdxCur, endOfLoop);
			topOfLoop = sqlite3VdbeCurrentAddr(v);
			sqlite3VdbeAddOp2(v, OP_AddImm, iMem, 1);  /* Increment row counter */

			for(i=0; i<nCol; i++){
				CollSeq *pColl;
				sqlite3VdbeAddOp3(v, OP_Column, iIdxCur, i, regCol);
				if( i==0 ){
					/* Always record the very first row */
					addrIfNot = sqlite3VdbeAddOp1(v, OP_IfNot, iMem+1);
				}
				assert( pIdx->azColl!=0 );
				assert( pIdx->azColl[i]!=0 );
				pColl = sqlite3LocateCollSeq(pParse, pIdx->azColl[i]);
				aChngAddr[i] = sqlite3VdbeAddOp4(v, OP_Ne, regCol, 0, iMem+nCol+i+1,
					(char*)pColl, P4_COLLSEQ);
				sqlite3VdbeChangeP5(v, SQLITE_NULLEQ);
				VdbeComment((v, "jump if column %d changed", i));
#ifdef SQLITE_ENABLE_STAT3
				if( i==0 ){
					sqlite3VdbeAddOp2(v, OP_AddImm, regNumEq, 1);
					VdbeComment((v, "incr repeat count"));
				}
#endif
			}
			sqlite3VdbeAddOp2(v, OP_Goto, 0, endOfLoop);
			for(i=0; i<nCol; i++){
				sqlite3VdbeJumpHere(v, aChngAddr[i]);  /* Set jump dest for the OP_Ne */
				if( i==0 ){
					sqlite3VdbeJumpHere(v, addrIfNot);   /* Jump dest for OP_IfNot */
#ifdef SQLITE_ENABLE_STAT3
					sqlite3VdbeAddOp4(v, OP_Function, 1, regNumEq, regTemp2,
						(char*)&stat3PushFuncdef, P4_FUNCDEF);
					sqlite3VdbeChangeP5(v, 5);
					sqlite3VdbeAddOp3(v, OP_Column, iIdxCur, pIdx->nColumn, regRowid);
					sqlite3VdbeAddOp3(v, OP_Add, regNumEq, regNumLt, regNumLt);
					sqlite3VdbeAddOp2(v, OP_AddImm, regNumDLt, 1);
					sqlite3VdbeAddOp2(v, OP_Integer, 1, regNumEq);
#endif        
				}
				sqlite3VdbeAddOp2(v, OP_AddImm, iMem+i+1, 1);
				sqlite3VdbeAddOp3(v, OP_Column, iIdxCur, i, iMem+nCol+i+1);
			}
			sqlite3DbFree(db, aChngAddr);

			/* Always jump here after updating the iMem+1...iMem+1+nCol counters */
			sqlite3VdbeResolveLabel(v, endOfLoop);

			sqlite3VdbeAddOp2(v, OP_Next, iIdxCur, topOfLoop);
			sqlite3VdbeAddOp1(v, OP_Close, iIdxCur);
#ifdef SQLITE_ENABLE_STAT3
			sqlite3VdbeAddOp4(v, OP_Function, 1, regNumEq, regTemp2,
				(char*)&stat3PushFuncdef, P4_FUNCDEF);
			sqlite3VdbeChangeP5(v, 5);
			sqlite3VdbeAddOp2(v, OP_Integer, -1, regLoop);
			shortJump = 
				sqlite3VdbeAddOp2(v, OP_AddImm, regLoop, 1);
			sqlite3VdbeAddOp4(v, OP_Function, 1, regAccum, regTemp1,
				(char*)&stat3GetFuncdef, P4_FUNCDEF);
			sqlite3VdbeChangeP5(v, 2);
			sqlite3VdbeAddOp1(v, OP_IsNull, regTemp1);
			sqlite3VdbeAddOp3(v, OP_NotExists, iTabCur, shortJump, regTemp1);
			sqlite3VdbeAddOp3(v, OP_Column, iTabCur, pIdx->aiColumn[0], regSample);
			sqlite3ColumnDefault(v, pTab, pIdx->aiColumn[0], regSample);
			sqlite3VdbeAddOp4(v, OP_Function, 1, regAccum, regNumEq,
				(char*)&stat3GetFuncdef, P4_FUNCDEF);
			sqlite3VdbeChangeP5(v, 3);
			sqlite3VdbeAddOp4(v, OP_Function, 1, regAccum, regNumLt,
				(char*)&stat3GetFuncdef, P4_FUNCDEF);
			sqlite3VdbeChangeP5(v, 4);
			sqlite3VdbeAddOp4(v, OP_Function, 1, regAccum, regNumDLt,
				(char*)&stat3GetFuncdef, P4_FUNCDEF);
			sqlite3VdbeChangeP5(v, 5);
			sqlite3VdbeAddOp4(v, OP_MakeRecord, regTabname, 6, regRec, "bbbbbb", 0);
			sqlite3VdbeAddOp2(v, OP_NewRowid, iStatCur+1, regNewRowid);
			sqlite3VdbeAddOp3(v, OP_Insert, iStatCur+1, regRec, regNewRowid);
			sqlite3VdbeAddOp2(v, OP_Goto, 0, shortJump);
			sqlite3VdbeJumpHere(v, shortJump+2);
#endif        

			/* Store the results in sqlite_stat1.
			**
			** The result is a single row of the sqlite_stat1 table.  The first
			** two columns are the names of the table and index.  The third column
			** is a string composed of a list of integer statistics about the
			** index.  The first integer in the list is the total number of entries
			** in the index.  There is one additional integer in the list for each
			** column of the table.  This additional integer is a guess of how many
			** rows of the table the index will select.  If D is the count of distinct
			** values and K is the total number of rows, then the integer is computed
			** as:
			**
			**        I = (K+D-1)/D
			**
			** If K==0 then no entry is made into the sqlite_stat1 table.  
			** If K>0 then it is always the case the D>0 so division by zero
			** is never possible.
			*/
			sqlite3VdbeAddOp2(v, OP_SCopy, iMem, regStat1);
			if( jZeroRows<0 ){
				jZeroRows = sqlite3VdbeAddOp1(v, OP_IfNot, iMem);
			}
			for(i=0; i<nCol; i++){
				sqlite3VdbeAddOp4(v, OP_String8, 0, regTemp, 0, " ", 0);
				sqlite3VdbeAddOp3(v, OP_Concat, regTemp, regStat1, regStat1);
				sqlite3VdbeAddOp3(v, OP_Add, iMem, iMem+i+1, regTemp);
				sqlite3VdbeAddOp2(v, OP_AddImm, regTemp, -1);
				sqlite3VdbeAddOp3(v, OP_Divide, iMem+i+1, regTemp, regTemp);
				sqlite3VdbeAddOp1(v, OP_ToInt, regTemp);
				sqlite3VdbeAddOp3(v, OP_Concat, regTemp, regStat1, regStat1);
			}
			sqlite3VdbeAddOp4(v, OP_MakeRecord, regTabname, 3, regRec, "aaa", 0);
			sqlite3VdbeAddOp2(v, OP_NewRowid, iStatCur, regNewRowid);
			sqlite3VdbeAddOp3(v, OP_Insert, iStatCur, regRec, regNewRowid);
			sqlite3VdbeChangeP5(v, OPFLAG_APPEND);
		}

		/* If the table has no indices, create a single sqlite_stat1 entry
		** containing NULL as the index name and the row count as the content.
		*/
		if( pTab->pIndex==0 ){
			sqlite3VdbeAddOp3(v, OP_OpenRead, iIdxCur, pTab->tnum, iDb);
			VdbeComment((v, "%s", pTab->zName));
			sqlite3VdbeAddOp2(v, OP_Count, iIdxCur, regStat1);
			sqlite3VdbeAddOp1(v, OP_Close, iIdxCur);
			jZeroRows = sqlite3VdbeAddOp1(v, OP_IfNot, regStat1);
		}else{
			sqlite3VdbeJumpHere(v, jZeroRows);
			jZeroRows = sqlite3VdbeAddOp0(v, OP_Goto);
		}
		sqlite3VdbeAddOp2(v, OP_Null, 0, regIdxname);
		sqlite3VdbeAddOp4(v, OP_MakeRecord, regTabname, 3, regRec, "aaa", 0);
		sqlite3VdbeAddOp2(v, OP_NewRowid, iStatCur, regNewRowid);
		sqlite3VdbeAddOp3(v, OP_Insert, iStatCur, regRec, regNewRowid);
		sqlite3VdbeChangeP5(v, OPFLAG_APPEND);
		if( pParse->nMem<regRec ) pParse->nMem = regRec;
		sqlite3VdbeJumpHere(v, jZeroRows);
	}

	static void LoadAnalysis(Parse *pParse, int iDb)
	{
		Vdbe *v = sqlite3GetVdbe(pParse);
		if( v ){
			sqlite3VdbeAddOp1(v, OP_LoadAnalysis, iDb);
		}
	}

	static void AnalyzeDatabase(Parse *pParse, int iDb)
	{
		sqlite3 *db = pParse->db;
		Schema *pSchema = db->aDb[iDb].pSchema;    /* Schema of database iDb */
		HashElem *k;
		int iStatCur;
		int iMem;

		sqlite3BeginWriteOperation(pParse, 0, iDb);
		iStatCur = pParse->nTab;
		pParse->nTab += 3;
		openStatTable(pParse, iDb, iStatCur, 0, 0);
		iMem = pParse->nMem+1;
		assert( sqlite3SchemaMutexHeld(db, iDb, 0) );
		for(k=sqliteHashFirst(&pSchema->tblHash); k; k=sqliteHashNext(k)){
			Table *pTab = (Table*)sqliteHashData(k);
			analyzeOneTable(pParse, pTab, 0, iStatCur, iMem);
		}
		loadAnalysis(pParse, iDb);
	}


	static void AnalyzeTable(Parse *pParse, Table *pTab, Index *pOnlyIdx)
	{
		int iDb;
		int iStatCur;

		assert( pTab!=0 );
		assert( sqlite3BtreeHoldsAllMutexes(pParse->db) );
		iDb = sqlite3SchemaToIndex(pParse->db, pTab->pSchema);
		sqlite3BeginWriteOperation(pParse, 0, iDb);
		iStatCur = pParse->nTab;
		pParse->nTab += 3;
		if( pOnlyIdx ){
			openStatTable(pParse, iDb, iStatCur, pOnlyIdx->zName, "idx");
		}else{
			openStatTable(pParse, iDb, iStatCur, pTab->zName, "tbl");
		}
		analyzeOneTable(pParse, pTab, pOnlyIdx, iStatCur, pParse->nMem+1);
		loadAnalysis(pParse, iDb);
	}

	/*
	** Generate code for the ANALYZE command.  The parser calls this routine
	** when it recognizes an ANALYZE command.
	**
	**        ANALYZE                            -- 1
	**        ANALYZE  <database>                -- 2
	**        ANALYZE  ?<database>.?<tablename>  -- 3
	**
	** Form 1 causes all indices in all attached databases to be analyzed.
	** Form 2 analyzes all indices the single database named.
	** Form 3 analyzes all indices associated with the named table.
	*/
	void sqlite3Analyze(Parse *pParse, Token *pName1, Token *pName2)
	{
		sqlite3 *db = pParse->db;
		int iDb;
		int i;
		char *z, *zDb;
		Table *pTab;
		Index *pIdx;
		Token *pTableName;

		/* Read the database schema. If an error occurs, leave an error message
		** and code in pParse and return NULL. */
		assert( sqlite3BtreeHoldsAllMutexes(pParse->db) );
		if( SQLITE_OK!=sqlite3ReadSchema(pParse) ){
			return;
		}

		assert( pName2!=0 || pName1==0 );
		if( pName1==0 ){
			/* Form 1:  Analyze everything */
			for(i=0; i<db->nDb; i++){
				if( i==1 ) continue;  /* Do not analyze the TEMP database */
				analyzeDatabase(pParse, i);
			}
		}else if( pName2->n==0 ){
			/* Form 2:  Analyze the database or table named */
			iDb = sqlite3FindDb(db, pName1);
			if( iDb>=0 ){
				analyzeDatabase(pParse, iDb);
			}else{
				z = sqlite3NameFromToken(db, pName1);
				if( z ){
					if( (pIdx = sqlite3FindIndex(db, z, 0))!=0 ){
						analyzeTable(pParse, pIdx->pTable, pIdx);
					}else if( (pTab = sqlite3LocateTable(pParse, 0, z, 0))!=0 ){
						analyzeTable(pParse, pTab, 0);
					}
					sqlite3DbFree(db, z);
				}
			}
		}else{
			/* Form 3: Analyze the fully qualified table name */
			iDb = sqlite3TwoPartName(pParse, pName1, pName2, &pTableName);
			if( iDb>=0 ){
				zDb = db->aDb[iDb].zName;
				z = sqlite3NameFromToken(db, pTableName);
				if( z ){
					if( (pIdx = sqlite3FindIndex(db, z, zDb))!=0 ){
						analyzeTable(pParse, pIdx->pTable, pIdx);
					}else if( (pTab = sqlite3LocateTable(pParse, 0, z, zDb))!=0 ){
						analyzeTable(pParse, pTab, 0);
					}
					sqlite3DbFree(db, z);
				}
			}   
		}
	}

	/*
	** Used to pass information from the analyzer reader through to the
	** callback routine.
	*/
	typedef struct analysisInfo analysisInfo;
	struct analysisInfo {
		sqlite3 *db;
		const char *zDatabase;
	};

	/*
	** This callback is invoked once for each index when reading the
	** sqlite_stat1 table.  
	**
	**     argv[0] = name of the table
	**     argv[1] = name of the index (might be NULL)
	**     argv[2] = results of analysis - on integer for each column
	**
	** Entries for which argv[1]==NULL simply record the number of rows in
	** the table.
	*/
	static int analysisLoader(void *pData, int argc, char **argv, char **NotUsed){
		analysisInfo *pInfo = (analysisInfo*)pData;
		Index *pIndex;
		Table *pTable;
		int i, c, n;
		tRowcnt v;
		const char *z;

		assert( argc==3 );
		UNUSED_PARAMETER2(NotUsed, argc);

		if( argv==0 || argv[0]==0 || argv[2]==0 ){
			return 0;
		}
		pTable = sqlite3FindTable(pInfo->db, argv[0], pInfo->zDatabase);
		if( pTable==0 ){
			return 0;
		}
		if( argv[1] ){
			pIndex = sqlite3FindIndex(pInfo->db, argv[1], pInfo->zDatabase);
		}else{
			pIndex = 0;
		}
		n = pIndex ? pIndex->nColumn : 0;
		z = argv[2];
		for(i=0; *z && i<=n; i++){
			v = 0;
			while( (c=z[0])>='0' && c<='9' ){
				v = v*10 + c - '0';
				z++;
			}
			if( i==0 ) pTable->nRowEst = v;
			if( pIndex==0 ) break;
			pIndex->aiRowEst[i] = v;
			if( *z==' ' ) z++;
			if( strcmp(z, "unordered")==0 ){
				pIndex->bUnordered = 1;
				break;
			}
		}
		return 0;
	}

	/*
	** If the Index.aSample variable is not NULL, delete the aSample[] array
	** and its contents.
	*/
	void sqlite3DeleteIndexSamples(sqlite3 *db, Index *pIdx){
#ifdef SQLITE_ENABLE_STAT3
		if( pIdx->aSample ){
			int j;
			for(j=0; j<pIdx->nSample; j++){
				IndexSample *p = &pIdx->aSample[j];
				if( p->eType==SQLITE_TEXT || p->eType==SQLITE_BLOB ){
					sqlite3DbFree(db, p->u.z);
				}
			}
			sqlite3DbFree(db, pIdx->aSample);
		}
		if( db && db->pnBytesFreed==0 ){
			pIdx->nSample = 0;
			pIdx->aSample = 0;
		}
#else
		UNUSED_PARAMETER(db);
		UNUSED_PARAMETER(pIdx);
#endif
	}

#ifdef ENABLE_STAT3
	/*
	** Load content from the sqlite_stat3 table into the Index.aSample[]
	** arrays of all indices.
	*/
	static int loadStat3(sqlite3 *db, const char *zDb){
		int rc;                       /* Result codes from subroutines */
		sqlite3_stmt *pStmt = 0;      /* An SQL statement being run */
		char *zSql;                   /* Text of the SQL statement */
		Index *pPrevIdx = 0;          /* Previous index in the loop */
		int idx = 0;                  /* slot in pIdx->aSample[] for next sample */
		int eType;                    /* Datatype of a sample */
		IndexSample *pSample;         /* A slot in pIdx->aSample[] */

		assert( db->lookaside.bEnabled==0 );
		if( !sqlite3FindTable(db, "sqlite_stat3", zDb) ){
			return SQLITE_OK;
		}

		zSql = sqlite3MPrintf(db, 
			"SELECT idx,count(*) FROM %Q.sqlite_stat3"
			" GROUP BY idx", zDb);
		if( !zSql ){
			return SQLITE_NOMEM;
		}
		rc = sqlite3_prepare(db, zSql, -1, &pStmt, 0);
		sqlite3DbFree(db, zSql);
		if( rc ) return rc;

		while( sqlite3_step(pStmt)==SQLITE_ROW ){
			char *zIndex;   /* Index name */
			Index *pIdx;    /* Pointer to the index object */
			int nSample;    /* Number of samples */

			zIndex = (char *)sqlite3_column_text(pStmt, 0);
			if( zIndex==0 ) continue;
			nSample = sqlite3_column_int(pStmt, 1);
			pIdx = sqlite3FindIndex(db, zIndex, zDb);
			if( pIdx==0 ) continue;
			assert( pIdx->nSample==0 );
			pIdx->nSample = nSample;
			pIdx->aSample = sqlite3DbMallocZero(db, nSample*sizeof(IndexSample));
			pIdx->avgEq = pIdx->aiRowEst[1];
			if( pIdx->aSample==0 ){
				db->mallocFailed = 1;
				sqlite3_finalize(pStmt);
				return SQLITE_NOMEM;
			}
		}
		rc = sqlite3_finalize(pStmt);
		if( rc ) return rc;

		zSql = sqlite3MPrintf(db, 
			"SELECT idx,neq,nlt,ndlt,sample FROM %Q.sqlite_stat3", zDb);
		if( !zSql ){
			return SQLITE_NOMEM;
		}
		rc = sqlite3_prepare(db, zSql, -1, &pStmt, 0);
		sqlite3DbFree(db, zSql);
		if( rc ) return rc;

		while( sqlite3_step(pStmt)==SQLITE_ROW ){
			char *zIndex;   /* Index name */
			Index *pIdx;    /* Pointer to the index object */
			int i;          /* Loop counter */
			tRowcnt sumEq;  /* Sum of the nEq values */

			zIndex = (char *)sqlite3_column_text(pStmt, 0);
			if( zIndex==0 ) continue;
			pIdx = sqlite3FindIndex(db, zIndex, zDb);
			if( pIdx==0 ) continue;
			if( pIdx==pPrevIdx ){
				idx++;
			}else{
				pPrevIdx = pIdx;
				idx = 0;
			}
			assert( idx<pIdx->nSample );
			pSample = &pIdx->aSample[idx];
			pSample->nEq = (tRowcnt)sqlite3_column_int64(pStmt, 1);
			pSample->nLt = (tRowcnt)sqlite3_column_int64(pStmt, 2);
			pSample->nDLt = (tRowcnt)sqlite3_column_int64(pStmt, 3);
			if( idx==pIdx->nSample-1 ){
				if( pSample->nDLt>0 ){
					for(i=0, sumEq=0; i<=idx-1; i++) sumEq += pIdx->aSample[i].nEq;
					pIdx->avgEq = (pSample->nLt - sumEq)/pSample->nDLt;
				}
				if( pIdx->avgEq<=0 ) pIdx->avgEq = 1;
			}
			eType = sqlite3_column_type(pStmt, 4);
			pSample->eType = (u8)eType;
			switch( eType ){
			case SQLITE_INTEGER: {
				pSample->u.i = sqlite3_column_int64(pStmt, 4);
				break;
								 }
			case SQLITE_FLOAT: {
				pSample->u.r = sqlite3_column_double(pStmt, 4);
				break;
							   }
			case SQLITE_NULL: {
				break;
							  }
			default: assert( eType==SQLITE_TEXT || eType==SQLITE_BLOB ); {
				const char *z = (const char *)(
					(eType==SQLITE_BLOB) ?
					sqlite3_column_blob(pStmt, 4):
					sqlite3_column_text(pStmt, 4)
					);
				int n = z ? sqlite3_column_bytes(pStmt, 4) : 0;
				pSample->nByte = n;
				if( n < 1){
					pSample->u.z = 0;
				}else{
					pSample->u.z = sqlite3DbMallocRaw(db, n);
					if( pSample->u.z==0 ){
						db->mallocFailed = 1;
						sqlite3_finalize(pStmt);
						return SQLITE_NOMEM;
					}
					memcpy(pSample->u.z, z, n);
				}
					 }
			}
		}
		return sqlite3_finalize(pStmt);
	}
#endif

	/*
	** Load the content of the sqlite_stat1 and sqlite_stat3 tables. The
	** contents of sqlite_stat1 are used to populate the Index.aiRowEst[]
	** arrays. The contents of sqlite_stat3 are used to populate the
	** Index.aSample[] arrays.
	**
	** If the sqlite_stat1 table is not present in the database, SQLITE_ERROR
	** is returned. In this case, even if SQLITE_ENABLE_STAT3 was defined 
	** during compilation and the sqlite_stat3 table is present, no data is 
	** read from it.
	**
	** If SQLITE_ENABLE_STAT3 was defined during compilation and the 
	** sqlite_stat3 table is not present in the database, SQLITE_ERROR is
	** returned. However, in this case, data is read from the sqlite_stat1
	** table (if it is present) before returning.
	**
	** If an OOM error occurs, this function always sets db->mallocFailed.
	** This means if the caller does not care about other errors, the return
	** code may be ignored.
	*/
	int sqlite3AnalysisLoad(sqlite3 *db, int iDb){
		analysisInfo sInfo;
		HashElem *i;
		char *zSql;
		int rc;

		assert( iDb>=0 && iDb<db->nDb );
		assert( db->aDb[iDb].pBt!=0 );

		/* Clear any prior statistics */
		assert( sqlite3SchemaMutexHeld(db, iDb, 0) );
		for(i=sqliteHashFirst(&db->aDb[iDb].pSchema->idxHash);i;i=sqliteHashNext(i)){
			Index *pIdx = sqliteHashData(i);
			sqlite3DefaultRowEst(pIdx);
#ifdef SQLITE_ENABLE_STAT3
			sqlite3DeleteIndexSamples(db, pIdx);
			pIdx->aSample = 0;
#endif
		}

		/* Check to make sure the sqlite_stat1 table exists */
		sInfo.db = db;
		sInfo.zDatabase = db->aDb[iDb].zName;
		if( sqlite3FindTable(db, "sqlite_stat1", sInfo.zDatabase)==0 ){
			return SQLITE_ERROR;
		}

		/* Load new statistics out of the sqlite_stat1 table */
		zSql = sqlite3MPrintf(db, 
			"SELECT tbl,idx,stat FROM %Q.sqlite_stat1", sInfo.zDatabase);
		if( zSql==0 ){
			rc = SQLITE_NOMEM;
		}else{
			rc = sqlite3_exec(db, zSql, analysisLoader, &sInfo, 0);
			sqlite3DbFree(db, zSql);
		}


		/* Load the statistics from the sqlite_stat3 table. */
#ifdef ENABLE_STAT3
		if( rc==SQLITE_OK ){
			int lookasideEnabled = db->lookaside.bEnabled;
			db->lookaside.bEnabled = 0;
			rc = loadStat3(db, sInfo.zDatabase);
			db->lookaside.bEnabled = lookasideEnabled;
		}
#endif

		if( rc==SQLITE_NOMEM ){
			db->mallocFailed = 1;
		}
		return rc;
	}

} }
#endif
#pragma endregion
