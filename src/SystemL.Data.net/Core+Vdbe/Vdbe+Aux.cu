#include "vdbeInt.cu.h"

#pragma region Preamble
// Properties of opcodes.  The OPFLG_INITIALIZER macro is created by mkopcodeh.awk during compilation.  Data is obtained
// from the comments following the "case OP_xxxx:" statements in the vdbe.c file.  
const unsigned char _opcodeProperty[] = OPFLG_INITIALIZER;
#pragma endregion

#pragma region Aux

__device__ Vdbe *Vdbe::Create(Context *ctx)
{
	Vdbe *p = (Vdbe *)_tagalloc2(ctx, sizeof(Vdbe), true);
	if (!p) return nullptr;
	p->Ctx = ctx;
	if (ctx->Vdbes)
		ctx->Vdbes->Prev = p;
	p->Next = ctx->Vdbes;
	p->Prev = nullptr;
	ctx->Vdbes = p;
	p->Magic = VDBE_MAGIC_INIT;
	return p;
}

__device__ void Vdbe::SetSql(Vdbe *p, const char *z, int n, bool isPrepareV2)
{
	if (!p) return;
#if defined(OMIT_TRACE) && !defined(ENABLE_SQLLOG)
	if (!isPrepareV2) return;
#endif
	_assert(!p->Sql_);
	p->Sql_ = _tagstrndup(p->Ctx, z, n);
	p->IsPrepareV2 = (bool)isPrepareV2;
}

__device__ const char *Vdbe::Sql(Vdbe *stmt)
{
	Vdbe *p = (Vdbe *)stmt;
	return (p && p->IsPrepareV2 ? p->Sql_ : nullptr);
}

__device__ void Vdbe::Swap(Vdbe *a, Vdbe *b)
{
	Vdbe tmp_ = *a;
	*a = *b;
	*b = tmp_;
	Vdbe *tmp = a->Next;
	a->Next = b->Next;
	b->Next = tmp;
	tmp = a->Prev;
	a->Prev = b->Prev;
	b->Prev = tmp;
	char *tmpSql = a->Sql_;
	a->Sql_ = b->Sql_;
	b->Sql_ = tmpSql;
	b->IsPrepareV2 = a->IsPrepareV2;
}

#ifdef _DEBUG
__device__ void Vdbe::set_Trace(FILE *trace)
{
	Trace = trace;
}
#endif

__device__ static RC GrowOps(Vdbe *p)
{
	int newLength = (p->OpsAlloc ? p->OpsAlloc*2 : (int)(1024/sizeof(Vdbe::VdbeOp)));
	Vdbe::VdbeOp *newData = (Vdbe::VdbeOp *)_tagrealloc(p->Ctx, p->Ops.data, newLength * sizeof(Vdbe::VdbeOp));
	if (newData)
	{
		p->OpsAlloc = _tagallocsize(p->Ctx, newData)/sizeof(Vdbe::VdbeOp);
		p->Ops.data = newData;
	}
	return (newData ? RC_OK : RC_NOMEM);
}

__device__ int Vdbe::AddOp3(OP op, int p1, int p2, int p3)
{
	int i = Ops.length;
	_assert(Magic == VDBE_MAGIC_INIT);
	_assert(op > 0 && op < 0xff);
	if (OpsAlloc <= i)
		if (GrowOps(this))
			return 1;
	Ops.length++;
	Vdbe::VdbeOp *opAsObj = &Ops[i];
	opAsObj->Opcode = op;
	opAsObj->P5 = 0;
	opAsObj->P1 = p1;
	opAsObj->P2 = p2;
	opAsObj->P3 = p3;
	opAsObj->P4.P = nullptr;
	opAsObj->P4Type = Vdbe::P4T_NOTUSED;
#ifdef _DEBUG
	opAsObj->Comment = nullptr;
	if (Ctx->Flags & BContext::FLAG_VdbeAddopTrace)
		PrintOp(nullptr, i, &Ops[i]);
#endif
#ifdef VDBE_PROFILE
	opAsObj->Cycles = 0;
	opAsObj->Cnt = 0;
#endif
	return i;
}
__device__ int Vdbe::AddOp0(OP op) { return AddOp3(op, 0, 0, 0); }
__device__ int Vdbe::AddOp1(OP op, int p1) { return AddOp3(op, p1, 0, 0); }
__device__ int Vdbe::AddOp2(OP op, int p1, int p2) { return AddOp3(op, p1, p2, 0); }
__device__ int Vdbe::AddOp4(OP op, int p1, int p2, int p3, const char *p4, Vdbe::P4T p4t)
{
	int addr = AddOp3(op, p1, p2, p3);
	ChangeP4(addr, p4, p4t);
	return addr;
}

__device__ void Vdbe::AddParseSchemaOp(int db, char *where_)
{
	int addr = AddOp3(OP_ParseSchema, db, 0, 0);
	ChangeP4(addr, where_, Vdbe::P4T_DYNAMIC);
	for (int j = 0; j < Ctx->DBs.length; j++) UsesBtree(j);
}

__device__ int Vdbe::AddOp4Int(OP op, int p1, int p2, int p3, int p4)
{
	int addr = AddOp3(op, p1, p2, p3);
	ChangeP4(addr, INT_TO_PTR(p4), Vdbe::P4T_INT32);
	return addr;
}

__device__ int Vdbe::MakeLabel()
{
	int i = Labels.length++;
	_assert(Magic == VDBE_MAGIC_INIT);
	if ((i & (i-1)) == 0) //sky? always
		p->Labels.data = _tagrealloc_or_free(Ctx, Labels.data, (i*2+1)*sizeof(Labels[0]));
	if (p->Label.data)
		p->Label[i] = -1;
	return -1 - i;
}

__device__ void Vdbe::ResolveLabel(int x)
{
	int j = -1-x;
	_assert(Magic == VDBE_MAGIC_INIT);
	_assert(j >= 0 && j < Labels.length);
	if (Labels.data)
		Labels[j] = Ops.length;
}

__device__ void Vdbe::set_RunOnlyOnce()
{
	RunOnlyOnce = true;
}

#ifdef _DEBUG

struct VdbeOpIter
{
	Vdbe *V;					// Vdbe to iterate through the opcodes of
	array_t<SubProgram> Subs;	// Array of subprograms
	int Addr;					// Address of next instruction to return
	int SubId;					// 0 = main program, 1 = first sub-program etc.
};

static Vdbe::VdbeOp *OpIterNext(VdbeOpIter *p)
{
	Vdbe::VdbeOp *r = nullptr;
	if (p->SubId <= p->Subs.length)
	{
		Vdbe *v = p->V;
		array_t<Vdbe::VdbeOp> ops = (p->SubId == 0 ? v->Ops : p->Subs[p->SubId-1]->Ops);
		_assert(p->Addr < ops.Length);

		r = &ops[p->Addr];
		p->Addr++;
		if (p->Addr == ops.length)
		{
			p->SubId++;
			p->Addr = 0;
		}
		if (r->P4Type == Vdbe::P4T_SUBPROGRAM)
		{
			int j;
			for (j = 0; j < p->Subs.length; j++)
				if (p->Subs[j] == r->P4.Program) break;
			if (j == p->Subs.length)
			{
				int bytes = (p->Subs.length+1)*sizeof(SubProgram *);
				p->Subs.data = _tagrealloc_or_free(v->Ctx, p->Subs.data, bytes);
				if (!p->Subs.data)
					r = nullptr;
				else
					p->Subs[p->Subs.length++] = r->P4.Program;
			}
		}
	}
	return r;
}

__device__ bool Vdbe::AssertMayAbort(bool mayAbort)
{
	VdbeOpIter sIter;
	_memset(&sIter, 0, sizeof(sIter));
	sIter.V = this;

	bool hasAbort = false;
	Vdbe::VdbeOp *op;
	while ((op = OpIterNext(&sIter)) != nullptr)
	{
		OP opcode = op->Opcode;
		if (opcode == OP_Destroy || opcode == OP_VUpdate || opcode == OP_VRename 
#ifndef OMIT_FOREIGN_KEY
			|| (opcode == OP_FkCounter && op->P1 == 0 && op->P2 == 1) 
#endif
			|| ((opcode == OP_Halt || opcode == OP_HaltIfNull) 
			&& ((op->P1&0xff) == RC_CONSTRAINT && op->P2 == OE_Abort)))
		{
			hasAbort = true;
			break;
		}
	}
	_tagfree(Ctx, sIter.Subs.data);

	// Return true if hasAbort==mayAbort. Or if a malloc failure occurred. If malloc failed, then the while() loop above may not have iterated
	// through all opcodes and hasAbort may be set incorrectly. Return true for this case to prevent the assert() in the callers frame from failing.
	return (Ctx->MallocFailed || hasAbort == mayAbort);
}
#endif

__device__ static void ResolveP2Values(Vdbe *p, int *maxFuncArgs)
{
	int maxArgs = *maxFuncArgs;
	int *labels = p->Labels.data;
	p->ReadOnly = true;

	Vdbe::VdbeOp *op;
	int i;
	for (op = p->Ops, i = p->Ops.length-1; i >= 0; i--, op++)
	{
		OP opcode = op->Opcode;
		op->Opflags = _opcodeProperty[opcode];
		if (opcode == OP_Function || opcode == OP_AggStep)
		{
			if (op->P5 > maxArgs) maxArgs = op->P5;
		}
		else if ((opcode == OP_Transaction && op->P2 != 0) || opcode == OP_Vacuum)
		{
			p->ReadOnly = false;
		}
#ifndef OMIT_VIRTUALTABLE
		else if (opcode == OP_VUpdate)
		{
			if (op->P2 > maxArgs) maxArgs = op->P2;
		}
		else if (opcode == OP_VFilter)
		{
			_assert(p->Ops.length - i >= 3);
			_assert(op[-1].Opcode == OP_Integer);
			int n = op[-1].P1;
			if (n > maxArgs) maxArgs = n;
		}
#endif
		else if (opcode == OP_Next || opcode == OP_SorterNext)
		{
			op->P4.Advance = Btree::Next_;
			op->P4Type = Vdbe::P4T_ADVANCE;
		}
		else if (opcode == OP_Prev)
		{
			op->P4.Advance = Btree::Previous;
			op->P4Type = Vdbe::P4T_ADVANCE;
		}
		if ((op->Opflags & OPFLG_JUMP) != 0 && op->P2 < 0)
		{
			_assert(-1 - op->P2 < p->Labels.length);
			op->P2 = labels[-1 - op->P2];
		}
	}
	_tagfree(p->Ctx, p->Labels.data);
	p->Labels.data = nullptr;
	*maxFuncArgs = maxArgs;
}

__device__ int Vdbe::CurrentAddr()
{
	_assert(Magic == VDBE_MAGIC_INIT);
	return Ops.length;
}

__device__ Vdbe::VdbeOp *Vdbe::TakeOpArray(int *opsLength, int *maxArgs)
{
	Vdbe::VdbeOp *ops = Ops.data;
	_assert(ops && !Ctx->MallocFailed);
	_assert(BtreeMask == 0); // Check that sqlite3VdbeUsesBtree() was not called on this VM
	ResolveP2Values(this, maxArgs);
	*opsLength = Ops.length;
	Ops.data = nullptr;
	return ops;
}

__device__ int Vdbe::AddOpList(int opsLength, VdbeOpList const *ops)
{
	_assert(Magic == VDBE_MAGIC_INIT);
	if (Ops.length + opsLength > OpsAlloc && GrowOps(this))
		return 0;
	int addr = Ops.length;
	if (_ALWAYS(opsLength > 0))
	{
		VdbeOpList const *in_ = ops;
		for (int i = 0; i < opsLength; i++, in_++)
		{
			int p2 = in_->P2;
			VdbeOp *out_ = &p->aOp[i+addr];
			out_->Opcode = in_->Opcode;
			out_->P1 = in_->P1;
			out_->P2 = (p2 < 0 && (_opcodeProperty[out_->Opcode] & OPFLG_JUMP) != 0 ? addr + ADDR(p2) : p2);
			out_->P3 = in_->P3;
			out_->P4Type = Vdbe::P4T_NOTUSED;
			out_->P4.P = nullptr;
			out_->P5 = 0;
#ifdef _DEBUG
			out_->Comment = nullptr;
			if (Ctx->Flags & BContext::FLAG_VdbeAddopTrace)
				PrintOp(nullptr, i+addr, &Ops[i+addr]);
#endif
		}
		Ops.length += opsLength;
	}
	return addr;
}

__device__ void Vdbe::ChangeP1(uint32 addr, int val) { if ((uint32)Ops.length > addr) Ops[addr].P1 = val; }
__device__ void Vdbe::ChangeP2(uint32 addr, int val) { if ((uint32)Ops.length > addr) Ops[addr].P2 = val; }
__device__ void Vdbe::ChangeP3(uint32 addr, int val) { if ((uint32)Ops.length > addr) Ops[addr].P3 = val; }
__device__ void Vdbe::ChangeP5(uint8 val) { if (Ops.data) { _assert(Ops.length > 0); Ops[p->Ops.length-1].P5 = val; } }

__device__ void Vdbe::JumpHere(int addr)
{
	_assert(addr >= 0 || Ctx->MallocFailed);
	if (addr >= 0) ChangeP2(addr, Ops.length);
}

__device__ static void FreeEphemeralFunction(Context *ctx, FuncDef *def)
{
	if (_ALWAYS(def) && (def->Flags & FUNC_EPHEM) != 0)
		_tagfree(ctx, def);
}
__device__ static void VdbeFreeOpArray(Context *ctx, Vdbe::VdbeOp *ops, int opsLength);
__device__ static void FreeP4(Context *ctx, Vdbe::P4T p4t, void *p4)
{
	_assert(ctx);
	if (p4)
	{
		switch (p4t)
		{
		case Vdbe::P4T_REAL:
		case Vdbe::P4T_INT64:
		case Vdbe::P4T_DYNAMIC:
		case Vdbe::P4T_KEYINFO:
		case Vdbe::P4T_INTARRAY:
		case Vdbe::P4T_KEYINFO_HANDOFF: {
			_tagfree(ctx, p4);
			break; }
		case Vdbe::P4T_MPRINTF: {
			if (ctx->BytesFreed == 0) _free(p4);
			break; }
		case Vdbe::P4T_VDBEFUNC: {
			VdbeFunc *vdbeFunc = (VdbeFunc *)p4;
			FreeEphemeralFunction(ctx, vdbeFunc);
			if (ctx->BytesFreed == 0) Vdbe::DeleteAuxData(vdbeFunc, 0);
			_tagfree(ctx, vdbeFunc);
			break; }
		case Vdbe::P4T_FUNCDEF: {
			FreeEphemeralFunction(ctx, (FuncDef *)p4);
			break; }
		case Vdbe::P4T_MEM: {
			if (ctx->BytesFreed == 0)
				Vdbe::ValueFree((Mem *)p4);
			else
			{
				Mem *p = (Mem *)p4;
				_tagfree(ctx, p->Malloc);
				_tagfree(ctx, p);
			}
			break; }
		case Vdbe::P4T_VTAB : {
			if (ctx->BytesFreed == 0) ((VTable *)p4)->Unlock();
			break; }
		}
	}
}

__device__ static void VdbeFreeOpArray(Context *ctx, Vdbe::VdbeOp *ops, int opsLength)
{
	if (ops)
	{
		for (Vdbe::VdbeOp *op = ops; op < &ops[opsLength]; op++)
		{
			FreeP4(ctx, op->P4Type, op->P4.P);
#ifdef _DEBUG
			_tagfree(ctx, op->Comment);
#endif     
		}
	}
	_tagfree(ctx, ops);
}

__device__ void Vdbe::LinkSubProgram(SubProgram *p)
{
	p->Next = Programs;
	Programs = p;
}

__device__ void Vdbe::ChangeToNoop(int addr)
{
	if (Ops.data)
	{
		Vdbe::VdbeOp *op = &Ops[addr];
		FreeP4(Ctx, op->P4Type, op->P4.P);
		_memset(op, 0, sizeof(op[0]));
		op->Opcode = OP_Noop;
	}
}

__device__ void Vdbe::ChangeP4(int addr, const char *p4, int n)
{
	_assert(Magic == VDBE_MAGIC_INIT);
	Context *ctx = Ctx;
	if (!Ops.data || ctx->MallocFailed)
	{
		if (n != P4T_KEYINFO && n != P4T_VTAB)
			FreeP4(ctx, (P4T)n, (void *)*(char **)&p4);
		return;
	}
	_assert(Ops.length > 0);
	_assert(addr < Ops.length);
	if (addr < 0)
		addr = Ops.length - 1;
	Vdbe::VdbeOp *op = &Ops[addr];
	_assert(op->P4Type == P4T_NOTUSED || op->P4Rype == P4T_INT32);
	FreeP4(ctx, op->P4Type, op->P4.P);
	op->P4.P = nullptr;
	if ((P4T)n == P4T_INT32)
	{
		// Note: this cast is safe, because the origin data point was an int that was cast to a (const char *).
		op->P4.I = PTR_TO_INT(p4);
		op->P4Type = P4T_INT32;
	}
	else if (!p4)
	{
		op->P4.P = nullptr;
		op->P4Type = P4T_NOTUSED;
	}
	else if ((P4T)n == P4T_KEYINFO)
	{
		int fields = ((KeyInfo *)p4)->Fields;
		KeyInfo *keyInfo;
		int bytes = sizeof(*keyInfo) + (fields-1)*sizeof(keyInfo->Colls[0]) + fields;
		keyInfo = (KeyInfo *)_tagalloc(nullptr, bytes);
		op->P4.KeyInfo = keyInfo;
		if (keyInfo)
		{
			_memcpy((char *)keyInfo, p4, bytes - fields);
			uint8 *sortOrders = keyInfo->SortOrders;
			_assert(sortOrders != nullptr);
			keyInfo->SortOrders = (unsigned char *)&keyInfo->Colls[fields];
			_memcpy(keyInfo->SortOrders, sortOrders, fields);
			op->P4Type = P4T_KEYINFO;
		}
		else
		{
			ctx->MallocFailed = true;
			op->P4Type = P4T_NOTUSED;
		}
	}
	else if ((P4T)n == P4T_KEYINFO_HANDOFF)
	{
		op->P4.P = (void *)p4;
		op->P4Type = P4T_KEYINFO;
	}
	else if ((P4T)n == P4T_VTAB)
	{
		op->P4.P = (void *)p4;
		op->P4Type = P4T_VTAB;
		(((VTable *)p4))->Lock();
		_assert(((VTable *)p4)->Ctx == ctx);
	}
	else if (n < 0)
	{
		op->P4.P = (void *)p4;
		op->P4Type = (P4T)n;
	}
	else
	{
		if (n == 0) n = _strlen30(p4);
		op->P4.Z = _tagstrndup(ctx, p4, n);
		op->P4Type = P4T_DYNAMIC;
	}
}

#ifndef NDEBUG
__device__ static void VdbeVComment(Vdbe *p, const char *zFormat, va_list args)
{
	_assert(p->nOp>0 || p->aOp==0 );
	_assert(p->aOp==0 || p->aOp[p->nOp-1].zComment==0 || p->db->mallocFailed );
	if( p->nOp ){
		assert( p->aOp );
		sqlite3DbFree(p->db, p->aOp[p->nOp-1].zComment);
		p->aOp[p->nOp-1].zComment = sqlite3VMPrintf(p->db, zFormat, args);
	}
}
void Vdbe::Comment(Vdbe *p, const char *fmt, va_list args)
{
	if (p)
	{
		va_start(ap, zFormat);
		VdbeVComment(p, fmt, args);
		va_end(ap);
	}
}
void Vdbe::NoopComment(Vdbe *p, const char *fmt, ...)
{
	va_list ap;
	if( p ){
		sqlite3VdbeAddOp0(p, OP_Noop);
		va_start(ap, zFormat);
		vdbeVComment(p, zFormat, ap);
		va_end(ap);
	}
}
#endif

VdbeOp *sqlite3VdbeGetOp(Vdbe *p, int addr){
	/* C89 specifies that the constant "dummy" will be initialized to all
	** zeros, which is correct.  MSVC generates a warning, nevertheless. */
	static VdbeOp dummy;  /* Ignore the MSVC warning about no initializer */
	assert( p->magic==VDBE_MAGIC_INIT );
	if( addr<0 ){
#ifdef SQLITE_OMIT_TRACE
		if( p->nOp==0 ) return (VdbeOp*)&dummy;
#endif
		addr = p->nOp - 1;
	}
	assert( (addr>=0 && addr<p->nOp) || p->db->mallocFailed );
	if( p->db->mallocFailed ){
		return (VdbeOp*)&dummy;
	}else{
		return &p->aOp[addr];
	}
}

#if !defined(SQLITE_OMIT_EXPLAIN) || !defined(NDEBUG) \
	|| defined(VDBE_PROFILE) || defined(SQLITE_DEBUG)
/*
** Compute a string that describes the P4 parameter for an opcode.
** Use zTemp for any required temporary buffer space.
*/
static char *displayP4(Op *pOp, char *zTemp, int nTemp){
	char *zP4 = zTemp;
	assert( nTemp>=20 );
	switch( pOp->p4type ){
	case P4_KEYINFO_STATIC:
	case P4_KEYINFO: {
		int i, j;
		KeyInfo *pKeyInfo = pOp->p4.pKeyInfo;
		assert( pKeyInfo->aSortOrder!=0 );
		sqlite3_snprintf(nTemp, zTemp, "keyinfo(%d", pKeyInfo->nField);
		i = sqlite3Strlen30(zTemp);
		for(j=0; j<pKeyInfo->nField; j++){
			CollSeq *pColl = pKeyInfo->aColl[j];
			const char *zColl = pColl ? pColl->zName : "nil";
			int n = sqlite3Strlen30(zColl);
			if( i+n>nTemp-6 ){
				memcpy(&zTemp[i],",...",4);
				break;
			}
			zTemp[i++] = ',';
			if( pKeyInfo->aSortOrder[j] ){
				zTemp[i++] = '-';
			}
			memcpy(&zTemp[i], zColl, n+1);
			i += n;
		}
		zTemp[i++] = ')';
		zTemp[i] = 0;
		assert( i<nTemp );
		break;
					 }
	case P4_COLLSEQ: {
		CollSeq *pColl = pOp->p4.pColl;
		sqlite3_snprintf(nTemp, zTemp, "collseq(%.20s)", pColl->zName);
		break;
					 }
	case P4_FUNCDEF: {
		FuncDef *pDef = pOp->p4.pFunc;
		sqlite3_snprintf(nTemp, zTemp, "%s(%d)", pDef->zName, pDef->nArg);
		break;
					 }
	case P4_INT64: {
		sqlite3_snprintf(nTemp, zTemp, "%lld", *pOp->p4.pI64);
		break;
				   }
	case P4_INT32: {
		sqlite3_snprintf(nTemp, zTemp, "%d", pOp->p4.i);
		break;
				   }
	case P4_REAL: {
		sqlite3_snprintf(nTemp, zTemp, "%.16g", *pOp->p4.pReal);
		break;
				  }
	case P4_MEM: {
		Mem *pMem = pOp->p4.pMem;
		if( pMem->flags & MEM_Str ){
			zP4 = pMem->z;
		}else if( pMem->flags & MEM_Int ){
			sqlite3_snprintf(nTemp, zTemp, "%lld", pMem->u.i);
		}else if( pMem->flags & MEM_Real ){
			sqlite3_snprintf(nTemp, zTemp, "%.16g", pMem->r);
		}else if( pMem->flags & MEM_Null ){
			sqlite3_snprintf(nTemp, zTemp, "NULL");
		}else{
			assert( pMem->flags & MEM_Blob );
			zP4 = "(blob)";
		}
		break;
				 }
#ifndef SQLITE_OMIT_VIRTUALTABLE
	case P4_VTAB: {
		sqlite3_vtab *pVtab = pOp->p4.pVtab->pVtab;
		sqlite3_snprintf(nTemp, zTemp, "vtab:%p:%p", pVtab, pVtab->pModule);
		break;
				  }
#endif
	case P4_INTARRAY: {
		sqlite3_snprintf(nTemp, zTemp, "intarray");
		break;
					  }
	case P4_SUBPROGRAM: {
		sqlite3_snprintf(nTemp, zTemp, "program");
		break;
						}
	case P4_ADVANCE: {
		zTemp[0] = 0;
		break;
					 }
	default: {
		zP4 = pOp->p4.z;
		if( zP4==0 ){
			zP4 = zTemp;
			zTemp[0] = 0;
		}
			 }
	}
	assert( zP4!=0 );
	return zP4;
}
#endif

/*
** Declare to the Vdbe that the BTree object at db->aDb[i] is used.
**
** The prepared statements need to know in advance the complete set of
** attached databases that will be use.  A mask of these databases
** is maintained in p->btreeMask.  The p->lockMask value is the subset of
** p->btreeMask of databases that will require a lock.
*/
void sqlite3VdbeUsesBtree(Vdbe *p, int i){
	assert( i>=0 && i<p->db->nDb && i<(int)sizeof(yDbMask)*8 );
	assert( i<(int)sizeof(p->btreeMask)*8 );
	p->btreeMask |= ((yDbMask)1)<<i;
	if( i!=1 && sqlite3BtreeSharable(p->db->aDb[i].pBt) ){
		p->lockMask |= ((yDbMask)1)<<i;
	}
}

#if !defined(SQLITE_OMIT_SHARED_CACHE) && SQLITE_THREADSAFE>0
/*
** If SQLite is compiled to support shared-cache mode and to be threadsafe,
** this routine obtains the mutex associated with each BtShared structure
** that may be accessed by the VM passed as an argument. In doing so it also
** sets the BtShared.db member of each of the BtShared structures, ensuring
** that the correct busy-handler callback is invoked if required.
**
** If SQLite is not threadsafe but does support shared-cache mode, then
** sqlite3BtreeEnter() is invoked to set the BtShared.db variables
** of all of BtShared structures accessible via the database handle 
** associated with the VM.
**
** If SQLite is not threadsafe and does not support shared-cache mode, this
** function is a no-op.
**
** The p->btreeMask field is a bitmask of all btrees that the prepared 
** statement p will ever use.  Let N be the number of bits in p->btreeMask
** corresponding to btrees that use shared cache.  Then the runtime of
** this routine is N*N.  But as N is rarely more than 1, this should not
** be a problem.
*/
void sqlite3VdbeEnter(Vdbe *p){
	int i;
	yDbMask mask;
	sqlite3 *db;
	Db *aDb;
	int nDb;
	if( p->lockMask==0 ) return;  /* The common case */
	db = p->db;
	aDb = db->aDb;
	nDb = db->nDb;
	for(i=0, mask=1; i<nDb; i++, mask += mask){
		if( i!=1 && (mask & p->lockMask)!=0 && ALWAYS(aDb[i].pBt!=0) ){
			sqlite3BtreeEnter(aDb[i].pBt);
		}
	}
}
#endif

#if !defined(SQLITE_OMIT_SHARED_CACHE) && SQLITE_THREADSAFE>0
/*
** Unlock all of the btrees previously locked by a call to sqlite3VdbeEnter().
*/
void sqlite3VdbeLeave(Vdbe *p){
	int i;
	yDbMask mask;
	sqlite3 *db;
	Db *aDb;
	int nDb;
	if( p->lockMask==0 ) return;  /* The common case */
	db = p->db;
	aDb = db->aDb;
	nDb = db->nDb;
	for(i=0, mask=1; i<nDb; i++, mask += mask){
		if( i!=1 && (mask & p->lockMask)!=0 && ALWAYS(aDb[i].pBt!=0) ){
			sqlite3BtreeLeave(aDb[i].pBt);
		}
	}
}
#endif

#if defined(VDBE_PROFILE) || defined(SQLITE_DEBUG)
/*
** Print a single opcode.  This routine is used for debugging only.
*/
void sqlite3VdbePrintOp(FILE *pOut, int pc, Op *pOp){
	char *zP4;
	char zPtr[50];
	static const char *zFormat1 = "%4d %-13s %4d %4d %4d %-4s %.2X %s\n";
	if( pOut==0 ) pOut = stdout;
	zP4 = displayP4(pOp, zPtr, sizeof(zPtr));
	fprintf(pOut, zFormat1, pc, 
		sqlite3OpcodeName(pOp->opcode), pOp->p1, pOp->p2, pOp->p3, zP4, pOp->p5,
#ifdef SQLITE_DEBUG
		pOp->zComment ? pOp->zComment : ""
#else
		""
#endif
		);
	fflush(pOut);
}
#endif

/*
** Release an array of N Mem elements
*/
static void releaseMemArray(Mem *p, int N){
	if( p && N ){
		Mem *pEnd;
		sqlite3 *db = p->db;
		u8 malloc_failed = db->mallocFailed;
		if( db->pnBytesFreed ){
			for(pEnd=&p[N]; p<pEnd; p++){
				sqlite3DbFree(db, p->zMalloc);
			}
			return;
		}
		for(pEnd=&p[N]; p<pEnd; p++){
			assert( (&p[1])==pEnd || p[0].db==p[1].db );

			/* This block is really an inlined version of sqlite3VdbeMemRelease()
			** that takes advantage of the fact that the memory cell value is 
			** being set to NULL after releasing any dynamic resources.
			**
			** The justification for duplicating code is that according to 
			** callgrind, this causes a certain test case to hit the CPU 4.7 
			** percent less (x86 linux, gcc version 4.1.2, -O6) than if 
			** sqlite3MemRelease() were called from here. With -O2, this jumps
			** to 6.6 percent. The test case is inserting 1000 rows into a table 
			** with no indexes using a single prepared INSERT statement, bind() 
			** and reset(). Inserts are grouped into a transaction.
			*/
			if( p->flags&(MEM_Agg|MEM_Dyn|MEM_Frame|MEM_RowSet) ){
				sqlite3VdbeMemRelease(p);
			}else if( p->zMalloc ){
				sqlite3DbFree(db, p->zMalloc);
				p->zMalloc = 0;
			}

			p->flags = MEM_Invalid;
		}
		db->mallocFailed = malloc_failed;
	}
}

/*
** Delete a VdbeFrame object and its contents. VdbeFrame objects are
** allocated by the OP_Program opcode in sqlite3VdbeExec().
*/
void sqlite3VdbeFrameDelete(VdbeFrame *p){
	int i;
	Mem *aMem = VdbeFrameMem(p);
	VdbeCursor **apCsr = (VdbeCursor **)&aMem[p->nChildMem];
	for(i=0; i<p->nChildCsr; i++){
		sqlite3VdbeFreeCursor(p->v, apCsr[i]);
	}
	releaseMemArray(aMem, p->nChildMem);
	sqlite3DbFree(p->v->db, p);
}

#ifndef SQLITE_OMIT_EXPLAIN
/*
** Give a listing of the program in the virtual machine.
**
** The interface is the same as sqlite3VdbeExec().  But instead of
** running the code, it invokes the callback once for each instruction.
** This feature is used to implement "EXPLAIN".
**
** When p->explain==1, each instruction is listed.  When
** p->explain==2, only OP_Explain instructions are listed and these
** are shown in a different format.  p->explain==2 is used to implement
** EXPLAIN QUERY PLAN.
**
** When p->explain==1, first the main program is listed, then each of
** the trigger subprograms are listed one by one.
*/
int sqlite3VdbeList(
	Vdbe *p                   /* The VDBE */
	){
		int nRow;                            /* Stop when row count reaches this */
		int nSub = 0;                        /* Number of sub-vdbes seen so far */
		SubProgram **apSub = 0;              /* Array of sub-vdbes */
		Mem *pSub = 0;                       /* Memory cell hold array of subprogs */
		sqlite3 *db = p->db;                 /* The database connection */
		int i;                               /* Loop counter */
		int rc = SQLITE_OK;                  /* Return code */
		Mem *pMem = &p->aMem[1];             /* First Mem of result set */

		assert( p->explain );
		assert( p->magic==VDBE_MAGIC_RUN );
		assert( p->rc==SQLITE_OK || p->rc==SQLITE_BUSY || p->rc==SQLITE_NOMEM );

		/* Even though this opcode does not use dynamic strings for
		** the result, result columns may become dynamic if the user calls
		** sqlite3_column_text16(), causing a translation to UTF-16 encoding.
		*/
		releaseMemArray(pMem, 8);
		p->pResultSet = 0;

		if( p->rc==SQLITE_NOMEM ){
			/* This happens if a malloc() inside a call to sqlite3_column_text() or
			** sqlite3_column_text16() failed.  */
			db->mallocFailed = 1;
			return SQLITE_ERROR;
		}

		/* When the number of output rows reaches nRow, that means the
		** listing has finished and sqlite3_step() should return SQLITE_DONE.
		** nRow is the sum of the number of rows in the main program, plus
		** the sum of the number of rows in all trigger subprograms encountered
		** so far.  The nRow value will increase as new trigger subprograms are
		** encountered, but p->pc will eventually catch up to nRow.
		*/
		nRow = p->nOp;
		if( p->explain==1 ){
			/* The first 8 memory cells are used for the result set.  So we will
			** commandeer the 9th cell to use as storage for an array of pointers
			** to trigger subprograms.  The VDBE is guaranteed to have at least 9
			** cells.  */
			assert( p->nMem>9 );
			pSub = &p->aMem[9];
			if( pSub->flags&MEM_Blob ){
				/* On the first call to sqlite3_step(), pSub will hold a NULL.  It is
				** initialized to a BLOB by the P4_SUBPROGRAM processing logic below */
				nSub = pSub->n/sizeof(Vdbe*);
				apSub = (SubProgram **)pSub->z;
			}
			for(i=0; i<nSub; i++){
				nRow += apSub[i]->nOp;
			}
		}

		do{
			i = p->pc++;
		}while( i<nRow && p->explain==2 && p->aOp[i].opcode!=OP_Explain );
		if( i>=nRow ){
			p->rc = SQLITE_OK;
			rc = SQLITE_DONE;
		}else if( db->u1.isInterrupted ){
			p->rc = SQLITE_INTERRUPT;
			rc = SQLITE_ERROR;
			sqlite3SetString(&p->zErrMsg, db, "%s", sqlite3ErrStr(p->rc));
		}else{
			char *z;
			Op *pOp;
			if( i<p->nOp ){
				/* The output line number is small enough that we are still in the
				** main program. */
				pOp = &p->aOp[i];
			}else{
				/* We are currently listing subprograms.  Figure out which one and
				** pick up the appropriate opcode. */
				int j;
				i -= p->nOp;
				for(j=0; i>=apSub[j]->nOp; j++){
					i -= apSub[j]->nOp;
				}
				pOp = &apSub[j]->aOp[i];
			}
			if( p->explain==1 ){
				pMem->flags = MEM_Int;
				pMem->type = SQLITE_INTEGER;
				pMem->u.i = i;                                /* Program counter */
				pMem++;

				pMem->flags = MEM_Static|MEM_Str|MEM_Term;
				pMem->z = (char*)sqlite3OpcodeName(pOp->opcode);  /* Opcode */
				assert( pMem->z!=0 );
				pMem->n = sqlite3Strlen30(pMem->z);
				pMem->type = SQLITE_TEXT;
				pMem->enc = SQLITE_UTF8;
				pMem++;

				/* When an OP_Program opcode is encounter (the only opcode that has
				** a P4_SUBPROGRAM argument), expand the size of the array of subprograms
				** kept in p->aMem[9].z to hold the new program - assuming this subprogram
				** has not already been seen.
				*/
				if( pOp->p4type==P4_SUBPROGRAM ){
					int nByte = (nSub+1)*sizeof(SubProgram*);
					int j;
					for(j=0; j<nSub; j++){
						if( apSub[j]==pOp->p4.pProgram ) break;
					}
					if( j==nSub && SQLITE_OK==sqlite3VdbeMemGrow(pSub, nByte, nSub!=0) ){
						apSub = (SubProgram **)pSub->z;
						apSub[nSub++] = pOp->p4.pProgram;
						pSub->flags |= MEM_Blob;
						pSub->n = nSub*sizeof(SubProgram*);
					}
				}
			}

			pMem->flags = MEM_Int;
			pMem->u.i = pOp->p1;                          /* P1 */
			pMem->type = SQLITE_INTEGER;
			pMem++;

			pMem->flags = MEM_Int;
			pMem->u.i = pOp->p2;                          /* P2 */
			pMem->type = SQLITE_INTEGER;
			pMem++;

			pMem->flags = MEM_Int;
			pMem->u.i = pOp->p3;                          /* P3 */
			pMem->type = SQLITE_INTEGER;
			pMem++;

			if( sqlite3VdbeMemGrow(pMem, 32, 0) ){            /* P4 */
				assert( p->db->mallocFailed );
				return SQLITE_ERROR;
			}
			pMem->flags = MEM_Dyn|MEM_Str|MEM_Term;
			z = displayP4(pOp, pMem->z, 32);
			if( z!=pMem->z ){
				sqlite3VdbeMemSetStr(pMem, z, -1, SQLITE_UTF8, 0);
			}else{
				assert( pMem->z!=0 );
				pMem->n = sqlite3Strlen30(pMem->z);
				pMem->enc = SQLITE_UTF8;
			}
			pMem->type = SQLITE_TEXT;
			pMem++;

			if( p->explain==1 ){
				if( sqlite3VdbeMemGrow(pMem, 4, 0) ){
					assert( p->db->mallocFailed );
					return SQLITE_ERROR;
				}
				pMem->flags = MEM_Dyn|MEM_Str|MEM_Term;
				pMem->n = 2;
				sqlite3_snprintf(3, pMem->z, "%.2x", pOp->p5);   /* P5 */
				pMem->type = SQLITE_TEXT;
				pMem->enc = SQLITE_UTF8;
				pMem++;

#ifdef SQLITE_DEBUG
				if( pOp->zComment ){
					pMem->flags = MEM_Str|MEM_Term;
					pMem->z = pOp->zComment;
					pMem->n = sqlite3Strlen30(pMem->z);
					pMem->enc = SQLITE_UTF8;
					pMem->type = SQLITE_TEXT;
				}else
#endif
				{
					pMem->flags = MEM_Null;                       /* Comment */
					pMem->type = SQLITE_NULL;
				}
			}

			p->nResColumn = 8 - 4*(p->explain-1);
			p->pResultSet = &p->aMem[1];
			p->rc = SQLITE_OK;
			rc = SQLITE_ROW;
		}
		return rc;
}
#endif /* SQLITE_OMIT_EXPLAIN */

#ifdef SQLITE_DEBUG
/*
** Print the SQL that was used to generate a VDBE program.
*/
void sqlite3VdbePrintSql(Vdbe *p){
	int nOp = p->nOp;
	VdbeOp *pOp;
	if( nOp<1 ) return;
	pOp = &p->aOp[0];
	if( pOp->opcode==OP_Trace && pOp->p4.z!=0 ){
		const char *z = pOp->p4.z;
		while( sqlite3Isspace(*z) ) z++;
		printf("SQL: [%s]\n", z);
	}
}
#endif

#if !defined(SQLITE_OMIT_TRACE) && defined(SQLITE_ENABLE_IOTRACE)
/*
** Print an IOTRACE message showing SQL content.
*/
void sqlite3VdbeIOTraceSql(Vdbe *p){
	int nOp = p->nOp;
	VdbeOp *pOp;
	if( sqlite3IoTrace==0 ) return;
	if( nOp<1 ) return;
	pOp = &p->aOp[0];
	if( pOp->opcode==OP_Trace && pOp->p4.z!=0 ){
		int i, j;
		char z[1000];
		sqlite3_snprintf(sizeof(z), z, "%s", pOp->p4.z);
		for(i=0; sqlite3Isspace(z[i]); i++){}
		for(j=0; z[i]; i++){
			if( sqlite3Isspace(z[i]) ){
				if( z[i-1]!=' ' ){
					z[j++] = ' ';
				}
			}else{
				z[j++] = z[i];
			}
		}
		z[j] = 0;
		sqlite3IoTrace("SQL %s\n", z);
	}
}
#endif /* !SQLITE_OMIT_TRACE && SQLITE_ENABLE_IOTRACE */

/*
** Allocate space from a fixed size buffer and return a pointer to
** that space.  If insufficient space is available, return NULL.
**
** The pBuf parameter is the initial value of a pointer which will
** receive the new memory.  pBuf is normally NULL.  If pBuf is not
** NULL, it means that memory space has already been allocated and that
** this routine should not allocate any new memory.  When pBuf is not
** NULL simply return pBuf.  Only allocate new memory space when pBuf
** is NULL.
**
** nByte is the number of bytes of space needed.
**
** *ppFrom points to available space and pEnd points to the end of the
** available space.  When space is allocated, *ppFrom is advanced past
** the end of the allocated space.
**
** *pnByte is a counter of the number of bytes of space that have failed
** to allocate.  If there is insufficient space in *ppFrom to satisfy the
** request, then increment *pnByte by the amount of the request.
*/
static void *allocSpace(
	void *pBuf,          /* Where return pointer will be stored */
	int nByte,           /* Number of bytes to allocate */
	u8 **ppFrom,         /* IN/OUT: Allocate from *ppFrom */
	u8 *pEnd,            /* Pointer to 1 byte past the end of *ppFrom buffer */
	int *pnByte          /* If allocation cannot be made, increment *pnByte */
	){
		assert( EIGHT_BYTE_ALIGNMENT(*ppFrom) );
		if( pBuf ) return pBuf;
		nByte = ROUND8(nByte);
		if( &(*ppFrom)[nByte] <= pEnd ){
			pBuf = (void*)*ppFrom;
			*ppFrom += nByte;
		}else{
			*pnByte += nByte;
		}
		return pBuf;
}

/*
** Rewind the VDBE back to the beginning in preparation for
** running it.
*/
void sqlite3VdbeRewind(Vdbe *p){
#if defined(SQLITE_DEBUG) || defined(VDBE_PROFILE)
	int i;
#endif
	assert( p!=0 );
	assert( p->magic==VDBE_MAGIC_INIT );

	/* There should be at least one opcode.
	*/
	assert( p->nOp>0 );

	/* Set the magic to VDBE_MAGIC_RUN sooner rather than later. */
	p->magic = VDBE_MAGIC_RUN;

#ifdef SQLITE_DEBUG
	for(i=1; i<p->nMem; i++){
		assert( p->aMem[i].db==p->db );
	}
#endif
	p->pc = -1;
	p->rc = SQLITE_OK;
	p->errorAction = OE_Abort;
	p->magic = VDBE_MAGIC_RUN;
	p->nChange = 0;
	p->cacheCtr = 1;
	p->minWriteFileFormat = 255;
	p->iStatement = 0;
	p->nFkConstraint = 0;
#ifdef VDBE_PROFILE
	for(i=0; i<p->nOp; i++){
		p->aOp[i].cnt = 0;
		p->aOp[i].cycles = 0;
	}
#endif
}

/*
** Prepare a virtual machine for execution for the first time after
** creating the virtual machine.  This involves things such
** as allocating stack space and initializing the program counter.
** After the VDBE has be prepped, it can be executed by one or more
** calls to sqlite3VdbeExec().  
**
** This function may be called exact once on a each virtual machine.
** After this routine is called the VM has been "packaged" and is ready
** to run.  After this routine is called, futher calls to 
** sqlite3VdbeAddOp() functions are prohibited.  This routine disconnects
** the Vdbe from the Parse object that helped generate it so that the
** the Vdbe becomes an independent entity and the Parse object can be
** destroyed.
**
** Use the sqlite3VdbeRewind() procedure to restore a virtual machine back
** to its initial state after it has been run.
*/
void sqlite3VdbeMakeReady(
	Vdbe *p,                       /* The VDBE */
	Parse *pParse                  /* Parsing context */
	){
		sqlite3 *db;                   /* The database connection */
		int nVar;                      /* Number of parameters */
		int nMem;                      /* Number of VM memory registers */
		int nCursor;                   /* Number of cursors required */
		int nArg;                      /* Number of arguments in subprograms */
		int nOnce;                     /* Number of OP_Once instructions */
		int n;                         /* Loop counter */
		u8 *zCsr;                      /* Memory available for allocation */
		u8 *zEnd;                      /* First byte past allocated memory */
		int nByte;                     /* How much extra memory is needed */

		assert( p!=0 );
		assert( p->nOp>0 );
		assert( pParse!=0 );
		assert( p->magic==VDBE_MAGIC_INIT );
		db = p->db;
		assert( db->mallocFailed==0 );
		nVar = pParse->nVar;
		nMem = pParse->nMem;
		nCursor = pParse->nTab;
		nArg = pParse->nMaxArg;
		nOnce = pParse->nOnce;
		if( nOnce==0 ) nOnce = 1; /* Ensure at least one byte in p->aOnceFlag[] */

		/* For each cursor required, also allocate a memory cell. Memory
		** cells (nMem+1-nCursor)..nMem, inclusive, will never be used by
		** the vdbe program. Instead they are used to allocate space for
		** VdbeCursor/BtCursor structures. The blob of memory associated with 
		** cursor 0 is stored in memory cell nMem. Memory cell (nMem-1)
		** stores the blob of memory associated with cursor 1, etc.
		**
		** See also: allocateCursor().
		*/
		nMem += nCursor;

		/* Allocate space for memory registers, SQL variables, VDBE cursors and 
		** an array to marshal SQL function arguments in.
		*/
		zCsr = (u8*)&p->aOp[p->nOp];       /* Memory avaliable for allocation */
		zEnd = (u8*)&p->aOp[p->nOpAlloc];  /* First byte past end of zCsr[] */

		resolveP2Values(p, &nArg);
		p->usesStmtJournal = (u8)(pParse->isMultiWrite && pParse->mayAbort);
		if( pParse->explain && nMem<10 ){
			nMem = 10;
		}
		memset(zCsr, 0, zEnd-zCsr);
		zCsr += (zCsr - (u8*)0)&7;
		assert( EIGHT_BYTE_ALIGNMENT(zCsr) );
		p->expired = 0;

		/* Memory for registers, parameters, cursor, etc, is allocated in two
		** passes.  On the first pass, we try to reuse unused space at the 
		** end of the opcode array.  If we are unable to satisfy all memory
		** requirements by reusing the opcode array tail, then the second
		** pass will fill in the rest using a fresh allocation.  
		**
		** This two-pass approach that reuses as much memory as possible from
		** the leftover space at the end of the opcode array can significantly
		** reduce the amount of memory held by a prepared statement.
		*/
		do {
			nByte = 0;
			p->aMem = allocSpace(p->aMem, nMem*sizeof(Mem), &zCsr, zEnd, &nByte);
			p->aVar = allocSpace(p->aVar, nVar*sizeof(Mem), &zCsr, zEnd, &nByte);
			p->apArg = allocSpace(p->apArg, nArg*sizeof(Mem*), &zCsr, zEnd, &nByte);
			p->azVar = allocSpace(p->azVar, nVar*sizeof(char*), &zCsr, zEnd, &nByte);
			p->apCsr = allocSpace(p->apCsr, nCursor*sizeof(VdbeCursor*),
				&zCsr, zEnd, &nByte);
			p->aOnceFlag = allocSpace(p->aOnceFlag, nOnce, &zCsr, zEnd, &nByte);
			if( nByte ){
				p->pFree = sqlite3DbMallocZero(db, nByte);
			}
			zCsr = p->pFree;
			zEnd = &zCsr[nByte];
		}while( nByte && !db->mallocFailed );

		p->nCursor = nCursor;
		p->nOnceFlag = nOnce;
		if( p->aVar ){
			p->nVar = (ynVar)nVar;
			for(n=0; n<nVar; n++){
				p->aVar[n].flags = MEM_Null;
				p->aVar[n].db = db;
			}
		}
		if( p->azVar ){
			p->nzVar = pParse->nzVar;
			memcpy(p->azVar, pParse->azVar, p->nzVar*sizeof(p->azVar[0]));
			memset(pParse->azVar, 0, pParse->nzVar*sizeof(pParse->azVar[0]));
		}
		if( p->aMem ){
			p->aMem--;                      /* aMem[] goes from 1..nMem */
			p->nMem = nMem;                 /*       not from 0..nMem-1 */
			for(n=1; n<=nMem; n++){
				p->aMem[n].flags = MEM_Invalid;
				p->aMem[n].db = db;
			}
		}
		p->explain = pParse->explain;
		sqlite3VdbeRewind(p);
}

/*
** Close a VDBE cursor and release all the resources that cursor 
** happens to hold.
*/
void sqlite3VdbeFreeCursor(Vdbe *p, VdbeCursor *pCx){
	if( pCx==0 ){
		return;
	}
	sqlite3VdbeSorterClose(p->db, pCx);
	if( pCx->pBt ){
		sqlite3BtreeClose(pCx->pBt);
		/* The pCx->pCursor will be close automatically, if it exists, by
		** the call above. */
	}else if( pCx->pCursor ){
		sqlite3BtreeCloseCursor(pCx->pCursor);
	}
#ifndef SQLITE_OMIT_VIRTUALTABLE
	if( pCx->pVtabCursor ){
		sqlite3_vtab_cursor *pVtabCursor = pCx->pVtabCursor;
		const sqlite3_module *pModule = pCx->pModule;
		p->inVtabMethod = 1;
		pModule->xClose(pVtabCursor);
		p->inVtabMethod = 0;
	}
#endif
}

/*
** Copy the values stored in the VdbeFrame structure to its Vdbe. This
** is used, for example, when a trigger sub-program is halted to restore
** control to the main program.
*/
int sqlite3VdbeFrameRestore(VdbeFrame *pFrame){
	Vdbe *v = pFrame->v;
	v->aOnceFlag = pFrame->aOnceFlag;
	v->nOnceFlag = pFrame->nOnceFlag;
	v->aOp = pFrame->aOp;
	v->nOp = pFrame->nOp;
	v->aMem = pFrame->aMem;
	v->nMem = pFrame->nMem;
	v->apCsr = pFrame->apCsr;
	v->nCursor = pFrame->nCursor;
	v->db->lastRowid = pFrame->lastRowid;
	v->nChange = pFrame->nChange;
	return pFrame->pc;
}

/*
** Close all cursors.
**
** Also release any dynamic memory held by the VM in the Vdbe.aMem memory 
** cell array. This is necessary as the memory cell array may contain
** pointers to VdbeFrame objects, which may in turn contain pointers to
** open cursors.
*/
static void closeAllCursors(Vdbe *p){
	if( p->pFrame ){
		VdbeFrame *pFrame;
		for(pFrame=p->pFrame; pFrame->pParent; pFrame=pFrame->pParent);
		sqlite3VdbeFrameRestore(pFrame);
	}
	p->pFrame = 0;
	p->nFrame = 0;

	if( p->apCsr ){
		int i;
		for(i=0; i<p->nCursor; i++){
			VdbeCursor *pC = p->apCsr[i];
			if( pC ){
				sqlite3VdbeFreeCursor(p, pC);
				p->apCsr[i] = 0;
			}
		}
	}
	if( p->aMem ){
		releaseMemArray(&p->aMem[1], p->nMem);
	}
	while( p->pDelFrame ){
		VdbeFrame *pDel = p->pDelFrame;
		p->pDelFrame = pDel->pParent;
		sqlite3VdbeFrameDelete(pDel);
	}
}

/*
** Clean up the VM after execution.
**
** This routine will automatically close any cursors, lists, and/or
** sorters that were left open.  It also deletes the values of
** variables in the aVar[] array.
*/
static void Cleanup(Vdbe *p){
	sqlite3 *db = p->db;

#ifdef SQLITE_DEBUG
	/* Execute assert() statements to ensure that the Vdbe.apCsr[] and 
	** Vdbe.aMem[] arrays have already been cleaned up.  */
	int i;
	if( p->apCsr ) for(i=0; i<p->nCursor; i++) assert( p->apCsr[i]==0 );
	if( p->aMem ){
		for(i=1; i<=p->nMem; i++) assert( p->aMem[i].flags==MEM_Invalid );
	}
#endif

	sqlite3DbFree(db, p->zErrMsg);
	p->zErrMsg = 0;
	p->pResultSet = 0;
}

/*
** Set the number of result columns that will be returned by this SQL
** statement. This is now set at compile time, rather than during
** execution of the vdbe program so that sqlite3_column_count() can
** be called on an SQL statement before sqlite3_step().
*/
void sqlite3VdbeSetNumCols(Vdbe *p, int nResColumn){
	Mem *pColName;
	int n;
	sqlite3 *db = p->db;

	releaseMemArray(p->aColName, p->nResColumn*COLNAME_N);
	sqlite3DbFree(db, p->aColName);
	n = nResColumn*COLNAME_N;
	p->nResColumn = (u16)nResColumn;
	p->aColName = pColName = (Mem*)sqlite3DbMallocZero(db, sizeof(Mem)*n );
	if( p->aColName==0 ) return;
	while( n-- > 0 ){
		pColName->flags = MEM_Null;
		pColName->db = p->db;
		pColName++;
	}
}

/*
** Set the name of the idx'th column to be returned by the SQL statement.
** zName must be a pointer to a nul terminated string.
**
** This call must be made after a call to sqlite3VdbeSetNumCols().
**
** The final parameter, xDel, must be one of SQLITE_DYNAMIC, SQLITE_STATIC
** or SQLITE_TRANSIENT. If it is SQLITE_DYNAMIC, then the buffer pointed
** to by zName will be freed by sqlite3DbFree() when the vdbe is destroyed.
*/
int sqlite3VdbeSetColName(
	Vdbe *p,                         /* Vdbe being configured */
	int idx,                         /* Index of column zName applies to */
	int var,                         /* One of the COLNAME_* constants */
	const char *zName,               /* Pointer to buffer containing name */
	void (*xDel)(void*)              /* Memory management strategy for zName */
	){
		int rc;
		Mem *pColName;
		assert( idx<p->nResColumn );
		assert( var<COLNAME_N );
		if( p->db->mallocFailed ){
			assert( !zName || xDel!=SQLITE_DYNAMIC );
			return SQLITE_NOMEM;
		}
		assert( p->aColName!=0 );
		pColName = &(p->aColName[idx+var*p->nResColumn]);
		rc = sqlite3VdbeMemSetStr(pColName, zName, -1, SQLITE_UTF8, xDel);
		assert( rc!=0 || !zName || (pColName->flags&MEM_Term)!=0 );
		return rc;
}

/*
** A read or write transaction may or may not be active on database handle
** db. If a transaction is active, commit it. If there is a
** write-transaction spanning more than one database file, this routine
** takes care of the master journal trickery.
*/
static int vdbeCommit(sqlite3 *db, Vdbe *p){
	int i;
	int nTrans = 0;  /* Number of databases with an active write-transaction */
	int rc = SQLITE_OK;
	int needXcommit = 0;

#ifdef SQLITE_OMIT_VIRTUALTABLE
	/* With this option, sqlite3VtabSync() is defined to be simply 
	** SQLITE_OK so p is not used. 
	*/
	UNUSED_PARAMETER(p);
#endif

	/* Before doing anything else, call the xSync() callback for any
	** virtual module tables written in this transaction. This has to
	** be done before determining whether a master journal file is 
	** required, as an xSync() callback may add an attached database
	** to the transaction.
	*/
	rc = sqlite3VtabSync(db, &p->zErrMsg);

	/* This loop determines (a) if the commit hook should be invoked and
	** (b) how many database files have open write transactions, not 
	** including the temp database. (b) is important because if more than 
	** one database file has an open write transaction, a master journal
	** file is required for an atomic commit.
	*/ 
	for(i=0; rc==SQLITE_OK && i<db->nDb; i++){ 
		Btree *pBt = db->aDb[i].pBt;
		if( sqlite3BtreeIsInTrans(pBt) ){
			needXcommit = 1;
			if( i!=1 ) nTrans++;
			sqlite3BtreeEnter(pBt);
			rc = sqlite3PagerExclusiveLock(sqlite3BtreePager(pBt));
			sqlite3BtreeLeave(pBt);
		}
	}
	if( rc!=SQLITE_OK ){
		return rc;
	}

	/* If there are any write-transactions at all, invoke the commit hook */
	if( needXcommit && db->xCommitCallback ){
		rc = db->xCommitCallback(db->pCommitArg);
		if( rc ){
			return SQLITE_CONSTRAINT_COMMITHOOK;
		}
	}

	/* The simple case - no more than one database file (not counting the
	** TEMP database) has a transaction active.   There is no need for the
	** master-journal.
	**
	** If the return value of sqlite3BtreeGetFilename() is a zero length
	** string, it means the main database is :memory: or a temp file.  In 
	** that case we do not support atomic multi-file commits, so use the 
	** simple case then too.
	*/
	if( 0==sqlite3Strlen30(sqlite3BtreeGetFilename(db->aDb[0].pBt))
		|| nTrans<=1
		){
			for(i=0; rc==SQLITE_OK && i<db->nDb; i++){
				Btree *pBt = db->aDb[i].pBt;
				if( pBt ){
					rc = sqlite3BtreeCommitPhaseOne(pBt, 0);
				}
			}

			/* Do the commit only if all databases successfully complete phase 1. 
			** If one of the BtreeCommitPhaseOne() calls fails, this indicates an
			** IO error while deleting or truncating a journal file. It is unlikely,
			** but could happen. In this case abandon processing and return the error.
			*/
			for(i=0; rc==SQLITE_OK && i<db->nDb; i++){
				Btree *pBt = db->aDb[i].pBt;
				if( pBt ){
					rc = sqlite3BtreeCommitPhaseTwo(pBt, 0);
				}
			}
			if( rc==SQLITE_OK ){
				sqlite3VtabCommit(db);
			}
	}

	/* The complex case - There is a multi-file write-transaction active.
	** This requires a master journal file to ensure the transaction is
	** committed atomicly.
	*/
#ifndef SQLITE_OMIT_DISKIO
	else{
		sqlite3_vfs *pVfs = db->pVfs;
		int needSync = 0;
		char *zMaster = 0;   /* File-name for the master journal */
		char const *zMainFile = sqlite3BtreeGetFilename(db->aDb[0].pBt);
		sqlite3_file *pMaster = 0;
		i64 offset = 0;
		int res;
		int retryCount = 0;
		int nMainFile;

		/* Select a master journal file name */
		nMainFile = sqlite3Strlen30(zMainFile);
		zMaster = sqlite3MPrintf(db, "%s-mjXXXXXX9XXz", zMainFile);
		if( zMaster==0 ) return SQLITE_NOMEM;
		do {
			u32 iRandom;
			if( retryCount ){
				if( retryCount>100 ){
					sqlite3_log(SQLITE_FULL, "MJ delete: %s", zMaster);
					sqlite3OsDelete(pVfs, zMaster, 0);
					break;
				}else if( retryCount==1 ){
					sqlite3_log(SQLITE_FULL, "MJ collide: %s", zMaster);
				}
			}
			retryCount++;
			sqlite3_randomness(sizeof(iRandom), &iRandom);
			sqlite3_snprintf(13, &zMaster[nMainFile], "-mj%06X9%02X",
				(iRandom>>8)&0xffffff, iRandom&0xff);
			/* The antipenultimate character of the master journal name must
			** be "9" to avoid name collisions when using 8+3 filenames. */
			assert( zMaster[sqlite3Strlen30(zMaster)-3]=='9' );
			sqlite3FileSuffix3(zMainFile, zMaster);
			rc = sqlite3OsAccess(pVfs, zMaster, SQLITE_ACCESS_EXISTS, &res);
		}while( rc==SQLITE_OK && res );
		if( rc==SQLITE_OK ){
			/* Open the master journal. */
			rc = sqlite3OsOpenMalloc(pVfs, zMaster, &pMaster, 
				SQLITE_OPEN_READWRITE|SQLITE_OPEN_CREATE|
				SQLITE_OPEN_EXCLUSIVE|SQLITE_OPEN_MASTER_JOURNAL, 0
				);
		}
		if( rc!=SQLITE_OK ){
			sqlite3DbFree(db, zMaster);
			return rc;
		}

		/* Write the name of each database file in the transaction into the new
		** master journal file. If an error occurs at this point close
		** and delete the master journal file. All the individual journal files
		** still have 'null' as the master journal pointer, so they will roll
		** back independently if a failure occurs.
		*/
		for(i=0; i<db->nDb; i++){
			Btree *pBt = db->aDb[i].pBt;
			if( sqlite3BtreeIsInTrans(pBt) ){
				char const *zFile = sqlite3BtreeGetJournalname(pBt);
				if( zFile==0 ){
					continue;  /* Ignore TEMP and :memory: databases */
				}
				assert( zFile[0]!=0 );
				if( !needSync && !sqlite3BtreeSyncDisabled(pBt) ){
					needSync = 1;
				}
				rc = sqlite3OsWrite(pMaster, zFile, sqlite3Strlen30(zFile)+1, offset);
				offset += sqlite3Strlen30(zFile)+1;
				if( rc!=SQLITE_OK ){
					sqlite3OsCloseFree(pMaster);
					sqlite3OsDelete(pVfs, zMaster, 0);
					sqlite3DbFree(db, zMaster);
					return rc;
				}
			}
		}

		/* Sync the master journal file. If the IOCAP_SEQUENTIAL device
		** flag is set this is not required.
		*/
		if( needSync 
			&& 0==(sqlite3OsDeviceCharacteristics(pMaster)&SQLITE_IOCAP_SEQUENTIAL)
			&& SQLITE_OK!=(rc = sqlite3OsSync(pMaster, SQLITE_SYNC_NORMAL))
			){
				sqlite3OsCloseFree(pMaster);
				sqlite3OsDelete(pVfs, zMaster, 0);
				sqlite3DbFree(db, zMaster);
				return rc;
		}

		/* Sync all the db files involved in the transaction. The same call
		** sets the master journal pointer in each individual journal. If
		** an error occurs here, do not delete the master journal file.
		**
		** If the error occurs during the first call to
		** sqlite3BtreeCommitPhaseOne(), then there is a chance that the
		** master journal file will be orphaned. But we cannot delete it,
		** in case the master journal file name was written into the journal
		** file before the failure occurred.
		*/
		for(i=0; rc==SQLITE_OK && i<db->nDb; i++){ 
			Btree *pBt = db->aDb[i].pBt;
			if( pBt ){
				rc = sqlite3BtreeCommitPhaseOne(pBt, zMaster);
			}
		}
		sqlite3OsCloseFree(pMaster);
		assert( rc!=SQLITE_BUSY );
		if( rc!=SQLITE_OK ){
			sqlite3DbFree(db, zMaster);
			return rc;
		}

		/* Delete the master journal file. This commits the transaction. After
		** doing this the directory is synced again before any individual
		** transaction files are deleted.
		*/
		rc = sqlite3OsDelete(pVfs, zMaster, 1);
		sqlite3DbFree(db, zMaster);
		zMaster = 0;
		if( rc ){
			return rc;
		}

		/* All files and directories have already been synced, so the following
		** calls to sqlite3BtreeCommitPhaseTwo() are only closing files and
		** deleting or truncating journals. If something goes wrong while
		** this is happening we don't really care. The integrity of the
		** transaction is already guaranteed, but some stray 'cold' journals
		** may be lying around. Returning an error code won't help matters.
		*/
		disable_simulated_io_errors();
		sqlite3BeginBenignMalloc();
		for(i=0; i<db->nDb; i++){ 
			Btree *pBt = db->aDb[i].pBt;
			if( pBt ){
				sqlite3BtreeCommitPhaseTwo(pBt, 1);
			}
		}
		sqlite3EndBenignMalloc();
		enable_simulated_io_errors();

		sqlite3VtabCommit(db);
	}
#endif

	return rc;
}

/* 
** This routine checks that the sqlite3.activeVdbeCnt count variable
** matches the number of vdbe's in the list sqlite3.pVdbe that are
** currently active. An assertion fails if the two counts do not match.
** This is an internal self-check only - it is not an essential processing
** step.
**
** This is a no-op if NDEBUG is defined.
*/
#ifndef NDEBUG
static void checkActiveVdbeCnt(sqlite3 *db){
	Vdbe *p;
	int cnt = 0;
	int nWrite = 0;
	p = db->pVdbe;
	while( p ){
		if( p->magic==VDBE_MAGIC_RUN && p->pc>=0 ){
			cnt++;
			if( p->readOnly==0 ) nWrite++;
		}
		p = p->pNext;
	}
	assert( cnt==db->activeVdbeCnt );
	assert( nWrite==db->writeVdbeCnt );
}
#else
#define checkActiveVdbeCnt(x)
#endif

/*
** If the Vdbe passed as the first argument opened a statement-transaction,
** close it now. Argument eOp must be either SAVEPOINT_ROLLBACK or
** SAVEPOINT_RELEASE. If it is SAVEPOINT_ROLLBACK, then the statement
** transaction is rolled back. If eOp is SAVEPOINT_RELEASE, then the 
** statement transaction is commtted.
**
** If an IO error occurs, an SQLITE_IOERR_XXX error code is returned. 
** Otherwise SQLITE_OK.
*/
int sqlite3VdbeCloseStatement(Vdbe *p, int eOp){
	sqlite3 *const db = p->db;
	int rc = SQLITE_OK;

	/* If p->iStatement is greater than zero, then this Vdbe opened a 
	** statement transaction that should be closed here. The only exception
	** is that an IO error may have occurred, causing an emergency rollback.
	** In this case (db->nStatement==0), and there is nothing to do.
	*/
	if( db->nStatement && p->iStatement ){
		int i;
		const int iSavepoint = p->iStatement-1;

		assert( eOp==SAVEPOINT_ROLLBACK || eOp==SAVEPOINT_RELEASE);
		assert( db->nStatement>0 );
		assert( p->iStatement==(db->nStatement+db->nSavepoint) );

		for(i=0; i<db->nDb; i++){ 
			int rc2 = SQLITE_OK;
			Btree *pBt = db->aDb[i].pBt;
			if( pBt ){
				if( eOp==SAVEPOINT_ROLLBACK ){
					rc2 = sqlite3BtreeSavepoint(pBt, SAVEPOINT_ROLLBACK, iSavepoint);
				}
				if( rc2==SQLITE_OK ){
					rc2 = sqlite3BtreeSavepoint(pBt, SAVEPOINT_RELEASE, iSavepoint);
				}
				if( rc==SQLITE_OK ){
					rc = rc2;
				}
			}
		}
		db->nStatement--;
		p->iStatement = 0;

		if( rc==SQLITE_OK ){
			if( eOp==SAVEPOINT_ROLLBACK ){
				rc = sqlite3VtabSavepoint(db, SAVEPOINT_ROLLBACK, iSavepoint);
			}
			if( rc==SQLITE_OK ){
				rc = sqlite3VtabSavepoint(db, SAVEPOINT_RELEASE, iSavepoint);
			}
		}

		/* If the statement transaction is being rolled back, also restore the 
		** database handles deferred constraint counter to the value it had when 
		** the statement transaction was opened.  */
		if( eOp==SAVEPOINT_ROLLBACK ){
			db->nDeferredCons = p->nStmtDefCons;
		}
	}
	return rc;
}

/*
** This function is called when a transaction opened by the database 
** handle associated with the VM passed as an argument is about to be 
** committed. If there are outstanding deferred foreign key constraint
** violations, return SQLITE_ERROR. Otherwise, SQLITE_OK.
**
** If there are outstanding FK violations and this function returns 
** SQLITE_ERROR, set the result of the VM to SQLITE_CONSTRAINT_FOREIGNKEY
** and write an error message to it. Then return SQLITE_ERROR.
*/
#ifndef SQLITE_OMIT_FOREIGN_KEY
int sqlite3VdbeCheckFk(Vdbe *p, int deferred){
	sqlite3 *db = p->db;
	if( (deferred && db->nDeferredCons>0) || (!deferred && p->nFkConstraint>0) ){
		p->rc = SQLITE_CONSTRAINT_FOREIGNKEY;
		p->errorAction = OE_Abort;
		sqlite3SetString(&p->zErrMsg, db, "foreign key constraint failed");
		return SQLITE_ERROR;
	}
	return SQLITE_OK;
}
#endif

/*
** This routine is called the when a VDBE tries to halt.  If the VDBE
** has made changes and is in autocommit mode, then commit those
** changes.  If a rollback is needed, then do the rollback.
**
** This routine is the only way to move the state of a VM from
** SQLITE_MAGIC_RUN to SQLITE_MAGIC_HALT.  It is harmless to
** call this on a VM that is in the SQLITE_MAGIC_HALT state.
**
** Return an error code.  If the commit could not complete because of
** lock contention, return SQLITE_BUSY.  If SQLITE_BUSY is returned, it
** means the close did not happen and needs to be repeated.
*/
int sqlite3VdbeHalt(Vdbe *p){
	int rc;                         /* Used to store transient return codes */
	sqlite3 *db = p->db;

	/* This function contains the logic that determines if a statement or
	** transaction will be committed or rolled back as a result of the
	** execution of this virtual machine. 
	**
	** If any of the following errors occur:
	**
	**     SQLITE_NOMEM
	**     SQLITE_IOERR
	**     SQLITE_FULL
	**     SQLITE_INTERRUPT
	**
	** Then the internal cache might have been left in an inconsistent
	** state.  We need to rollback the statement transaction, if there is
	** one, or the complete transaction if there is no statement transaction.
	*/

	if( p->db->mallocFailed ){
		p->rc = SQLITE_NOMEM;
	}
	if( p->aOnceFlag ) memset(p->aOnceFlag, 0, p->nOnceFlag);
	closeAllCursors(p);
	if( p->magic!=VDBE_MAGIC_RUN ){
		return SQLITE_OK;
	}
	checkActiveVdbeCnt(db);

	/* No commit or rollback needed if the program never started */
	if( p->pc>=0 ){
		int mrc;   /* Primary error code from p->rc */
		int eStatementOp = 0;
		int isSpecialError;            /* Set to true if a 'special' error */

		/* Lock all btrees used by the statement */
		sqlite3VdbeEnter(p);

		/* Check for one of the special errors */
		mrc = p->rc & 0xff;
		assert( p->rc!=SQLITE_IOERR_BLOCKED );  /* This error no longer exists */
		isSpecialError = mrc==SQLITE_NOMEM || mrc==SQLITE_IOERR
			|| mrc==SQLITE_INTERRUPT || mrc==SQLITE_FULL;
		if( isSpecialError ){
			/* If the query was read-only and the error code is SQLITE_INTERRUPT, 
			** no rollback is necessary. Otherwise, at least a savepoint 
			** transaction must be rolled back to restore the database to a 
			** consistent state.
			**
			** Even if the statement is read-only, it is important to perform
			** a statement or transaction rollback operation. If the error 
			** occurred while writing to the journal, sub-journal or database
			** file as part of an effort to free up cache space (see function
			** pagerStress() in pager.c), the rollback is required to restore 
			** the pager to a consistent state.
			*/
			if( !p->readOnly || mrc!=SQLITE_INTERRUPT ){
				if( (mrc==SQLITE_NOMEM || mrc==SQLITE_FULL) && p->usesStmtJournal ){
					eStatementOp = SAVEPOINT_ROLLBACK;
				}else{
					/* We are forced to roll back the active transaction. Before doing
					** so, abort any other statements this handle currently has active.
					*/
					sqlite3RollbackAll(db, SQLITE_ABORT_ROLLBACK);
					sqlite3CloseSavepoints(db);
					db->autoCommit = 1;
				}
			}
		}

		/* Check for immediate foreign key violations. */
		if( p->rc==SQLITE_OK ){
			sqlite3VdbeCheckFk(p, 0);
		}

		/* If the auto-commit flag is set and this is the only active writer 
		** VM, then we do either a commit or rollback of the current transaction. 
		**
		** Note: This block also runs if one of the special errors handled 
		** above has occurred. 
		*/
		if( !sqlite3VtabInSync(db) 
			&& db->autoCommit 
			&& db->writeVdbeCnt==(p->readOnly==0) 
			){
				if( p->rc==SQLITE_OK || (p->errorAction==OE_Fail && !isSpecialError) ){
					rc = sqlite3VdbeCheckFk(p, 1);
					if( rc!=SQLITE_OK ){
						if( NEVER(p->readOnly) ){
							sqlite3VdbeLeave(p);
							return SQLITE_ERROR;
						}
						rc = SQLITE_CONSTRAINT_FOREIGNKEY;
					}else{ 
						/* The auto-commit flag is true, the vdbe program was successful 
						** or hit an 'OR FAIL' constraint and there are no deferred foreign
						** key constraints to hold up the transaction. This means a commit 
						** is required. */
						rc = vdbeCommit(db, p);
					}
					if( rc==SQLITE_BUSY && p->readOnly ){
						sqlite3VdbeLeave(p);
						return SQLITE_BUSY;
					}else if( rc!=SQLITE_OK ){
						p->rc = rc;
						sqlite3RollbackAll(db, SQLITE_OK);
					}else{
						db->nDeferredCons = 0;
						sqlite3CommitInternalChanges(db);
					}
				}else{
					sqlite3RollbackAll(db, SQLITE_OK);
				}
				db->nStatement = 0;
		}else if( eStatementOp==0 ){
			if( p->rc==SQLITE_OK || p->errorAction==OE_Fail ){
				eStatementOp = SAVEPOINT_RELEASE;
			}else if( p->errorAction==OE_Abort ){
				eStatementOp = SAVEPOINT_ROLLBACK;
			}else{
				sqlite3RollbackAll(db, SQLITE_ABORT_ROLLBACK);
				sqlite3CloseSavepoints(db);
				db->autoCommit = 1;
			}
		}

		/* If eStatementOp is non-zero, then a statement transaction needs to
		** be committed or rolled back. Call sqlite3VdbeCloseStatement() to
		** do so. If this operation returns an error, and the current statement
		** error code is SQLITE_OK or SQLITE_CONSTRAINT, then promote the
		** current statement error code.
		*/
		if( eStatementOp ){
			rc = sqlite3VdbeCloseStatement(p, eStatementOp);
			if( rc ){
				if( p->rc==SQLITE_OK || (p->rc&0xff)==SQLITE_CONSTRAINT ){
					p->rc = rc;
					sqlite3DbFree(db, p->zErrMsg);
					p->zErrMsg = 0;
				}
				sqlite3RollbackAll(db, SQLITE_ABORT_ROLLBACK);
				sqlite3CloseSavepoints(db);
				db->autoCommit = 1;
			}
		}

		/* If this was an INSERT, UPDATE or DELETE and no statement transaction
		** has been rolled back, update the database connection change-counter. 
		*/
		if( p->changeCntOn ){
			if( eStatementOp!=SAVEPOINT_ROLLBACK ){
				sqlite3VdbeSetChanges(db, p->nChange);
			}else{
				sqlite3VdbeSetChanges(db, 0);
			}
			p->nChange = 0;
		}

		/* Release the locks */
		sqlite3VdbeLeave(p);
	}

	/* We have successfully halted and closed the VM.  Record this fact. */
	if( p->pc>=0 ){
		db->activeVdbeCnt--;
		if( !p->readOnly ){
			db->writeVdbeCnt--;
		}
		assert( db->activeVdbeCnt>=db->writeVdbeCnt );
	}
	p->magic = VDBE_MAGIC_HALT;
	checkActiveVdbeCnt(db);
	if( p->db->mallocFailed ){
		p->rc = SQLITE_NOMEM;
	}

	/* If the auto-commit flag is set to true, then any locks that were held
	** by connection db have now been released. Call sqlite3ConnectionUnlocked() 
	** to invoke any required unlock-notify callbacks.
	*/
	if( db->autoCommit ){
		sqlite3ConnectionUnlocked(db);
	}

	assert( db->activeVdbeCnt>0 || db->autoCommit==0 || db->nStatement==0 );
	return (p->rc==SQLITE_BUSY ? SQLITE_BUSY : SQLITE_OK);
}


/*
** Each VDBE holds the result of the most recent sqlite3_step() call
** in p->rc.  This routine sets that result back to SQLITE_OK.
*/
void sqlite3VdbeResetStepResult(Vdbe *p){
	p->rc = SQLITE_OK;
}

/*
** Copy the error code and error message belonging to the VDBE passed
** as the first argument to its database handle (so that they will be 
** returned by calls to sqlite3_errcode() and sqlite3_errmsg()).
**
** This function does not clear the VDBE error code or message, just
** copies them to the database handle.
*/
int sqlite3VdbeTransferError(Vdbe *p){
	sqlite3 *db = p->db;
	int rc = p->rc;
	if( p->zErrMsg ){
		u8 mallocFailed = db->mallocFailed;
		sqlite3BeginBenignMalloc();
		sqlite3ValueSetStr(db->pErr, -1, p->zErrMsg, SQLITE_UTF8, SQLITE_TRANSIENT);
		sqlite3EndBenignMalloc();
		db->mallocFailed = mallocFailed;
		db->errCode = rc;
	}else{
		sqlite3Error(db, rc, 0);
	}
	return rc;
}

#ifdef SQLITE_ENABLE_SQLLOG
/*
** If an SQLITE_CONFIG_SQLLOG hook is registered and the VM has been run, 
** invoke it.
*/
static void vdbeInvokeSqllog(Vdbe *v){
	if( sqlite3GlobalConfig.xSqllog && v->rc==SQLITE_OK && v->zSql && v->pc>=0 ){
		char *zExpanded = sqlite3VdbeExpandSql(v, v->zSql);
		assert( v->db->init.busy==0 );
		if( zExpanded ){
			sqlite3GlobalConfig.xSqllog(
				sqlite3GlobalConfig.pSqllogArg, v->db, zExpanded, 1
				);
			sqlite3DbFree(v->db, zExpanded);
		}
	}
}
#else
# define vdbeInvokeSqllog(x)
#endif

/*
** Clean up a VDBE after execution but do not delete the VDBE just yet.
** Write any error messages into *pzErrMsg.  Return the result code.
**
** After this routine is run, the VDBE should be ready to be executed
** again.
**
** To look at it another way, this routine resets the state of the
** virtual machine from VDBE_MAGIC_RUN or VDBE_MAGIC_HALT back to
** VDBE_MAGIC_INIT.
*/
int sqlite3VdbeReset(Vdbe *p){
	sqlite3 *db;
	db = p->db;

	/* If the VM did not run to completion or if it encountered an
	** error, then it might not have been halted properly.  So halt
	** it now.
	*/
	sqlite3VdbeHalt(p);

	/* If the VDBE has be run even partially, then transfer the error code
	** and error message from the VDBE into the main database structure.  But
	** if the VDBE has just been set to run but has not actually executed any
	** instructions yet, leave the main database error information unchanged.
	*/
	if( p->pc>=0 ){
		vdbeInvokeSqllog(p);
		sqlite3VdbeTransferError(p);
		sqlite3DbFree(db, p->zErrMsg);
		p->zErrMsg = 0;
		if( p->runOnlyOnce ) p->expired = 1;
	}else if( p->rc && p->expired ){
		/* The expired flag was set on the VDBE before the first call
		** to sqlite3_step(). For consistency (since sqlite3_step() was
		** called), set the database error in this case as well.
		*/
		sqlite3Error(db, p->rc, 0);
		sqlite3ValueSetStr(db->pErr, -1, p->zErrMsg, SQLITE_UTF8, SQLITE_TRANSIENT);
		sqlite3DbFree(db, p->zErrMsg);
		p->zErrMsg = 0;
	}

	/* Reclaim all memory used by the VDBE
	*/
	Cleanup(p);

	/* Save profiling information from this VDBE run.
	*/
#ifdef VDBE_PROFILE
	{
		FILE *out = fopen("vdbe_profile.out", "a");
		if( out ){
			int i;
			fprintf(out, "---- ");
			for(i=0; i<p->nOp; i++){
				fprintf(out, "%02x", p->aOp[i].opcode);
			}
			fprintf(out, "\n");
			for(i=0; i<p->nOp; i++){
				fprintf(out, "%6d %10lld %8lld ",
					p->aOp[i].cnt,
					p->aOp[i].cycles,
					p->aOp[i].cnt>0 ? p->aOp[i].cycles/p->aOp[i].cnt : 0
					);
				sqlite3VdbePrintOp(out, i, &p->aOp[i]);
			}
			fclose(out);
		}
	}
#endif
	p->magic = VDBE_MAGIC_INIT;
	return p->rc & db->errMask;
}

/*
** Clean up and delete a VDBE after execution.  Return an integer which is
** the result code.  Write any error message text into *pzErrMsg.
*/
int sqlite3VdbeFinalize(Vdbe *p){
	int rc = SQLITE_OK;
	if( p->magic==VDBE_MAGIC_RUN || p->magic==VDBE_MAGIC_HALT ){
		rc = sqlite3VdbeReset(p);
		assert( (rc & p->db->errMask)==rc );
	}
	sqlite3VdbeDelete(p);
	return rc;
}

/*
** Call the destructor for each auxdata entry in pVdbeFunc for which
** the corresponding bit in mask is clear.  Auxdata entries beyond 31
** are always destroyed.  To destroy all auxdata entries, call this
** routine with mask==0.
*/
void sqlite3VdbeDeleteAuxData(VdbeFunc *pVdbeFunc, int mask){
	int i;
	for(i=0; i<pVdbeFunc->nAux; i++){
		struct AuxData *pAux = &pVdbeFunc->apAux[i];
		if( (i>31 || !(mask&(((u32)1)<<i))) && pAux->pAux ){
			if( pAux->xDelete ){
				pAux->xDelete(pAux->pAux);
			}
			pAux->pAux = 0;
		}
	}
}

/*
** Free all memory associated with the Vdbe passed as the second argument,
** except for object itself, which is preserved.
**
** The difference between this function and sqlite3VdbeDelete() is that
** VdbeDelete() also unlinks the Vdbe from the list of VMs associated with
** the database connection and frees the object itself.
*/
void sqlite3VdbeClearObject(sqlite3 *db, Vdbe *p){
	SubProgram *pSub, *pNext;
	int i;
	assert( p->db==0 || p->db==db );
	releaseMemArray(p->aVar, p->nVar);
	releaseMemArray(p->aColName, p->nResColumn*COLNAME_N);
	for(pSub=p->pProgram; pSub; pSub=pNext){
		pNext = pSub->pNext;
		vdbeFreeOpArray(db, pSub->aOp, pSub->nOp);
		sqlite3DbFree(db, pSub);
	}
	for(i=p->nzVar-1; i>=0; i--) sqlite3DbFree(db, p->azVar[i]);
	vdbeFreeOpArray(db, p->aOp, p->nOp);
	sqlite3DbFree(db, p->aLabel);
	sqlite3DbFree(db, p->aColName);
	sqlite3DbFree(db, p->zSql);
	sqlite3DbFree(db, p->pFree);
#if defined(SQLITE_ENABLE_TREE_EXPLAIN)
	sqlite3DbFree(db, p->zExplain);
	sqlite3DbFree(db, p->pExplain);
#endif
}

/*
** Delete an entire VDBE.
*/
void sqlite3VdbeDelete(Vdbe *p){
	sqlite3 *db;

	if( NEVER(p==0) ) return;
	db = p->db;
	assert( sqlite3_mutex_held(db->mutex) );
	sqlite3VdbeClearObject(db, p);
	if( p->pPrev ){
		p->pPrev->pNext = p->pNext;
	}else{
		assert( db->pVdbe==p );
		db->pVdbe = p->pNext;
	}
	if( p->pNext ){
		p->pNext->pPrev = p->pPrev;
	}
	p->magic = VDBE_MAGIC_DEAD;
	p->db = 0;
	sqlite3DbFree(db, p);
}

/*
** Make sure the cursor p is ready to read or write the row to which it
** was last positioned.  Return an error code if an OOM fault or I/O error
** prevents us from positioning the cursor to its correct position.
**
** If a MoveTo operation is pending on the given cursor, then do that
** MoveTo now.  If no move is pending, check to see if the row has been
** deleted out from under the cursor and if it has, mark the row as
** a NULL row.
**
** If the cursor is already pointing to the correct row and that row has
** not been deleted out from under the cursor, then this routine is a no-op.
*/
int sqlite3VdbeCursorMoveto(VdbeCursor *p){
	if( p->deferredMoveto ){
		int res, rc;
#ifdef SQLITE_TEST
		extern int sqlite3_search_count;
#endif
		assert( p->isTable );
		rc = sqlite3BtreeMovetoUnpacked(p->pCursor, 0, p->movetoTarget, 0, &res);
		if( rc ) return rc;
		p->lastRowid = p->movetoTarget;
		if( res!=0 ) return SQLITE_CORRUPT_BKPT;
		p->rowidIsValid = 1;
#ifdef SQLITE_TEST
		sqlite3_search_count++;
#endif
		p->deferredMoveto = 0;
		p->cacheStatus = CACHE_STALE;
	}else if( ALWAYS(p->pCursor) ){
		int hasMoved;
		int rc = sqlite3BtreeCursorHasMoved(p->pCursor, &hasMoved);
		if( rc ) return rc;
		if( hasMoved ){
			p->cacheStatus = CACHE_STALE;
			p->nullRow = 1;
		}
	}
	return SQLITE_OK;
}

/*
** The following functions:
**
** sqlite3VdbeSerialType()
** sqlite3VdbeSerialTypeLen()
** sqlite3VdbeSerialLen()
** sqlite3VdbeSerialPut()
** sqlite3VdbeSerialGet()
**
** encapsulate the code that serializes values for storage in SQLite
** data and index records. Each serialized value consists of a
** 'serial-type' and a blob of data. The serial type is an 8-byte unsigned
** integer, stored as a varint.
**
** In an SQLite index record, the serial type is stored directly before
** the blob of data that it corresponds to. In a table record, all serial
** types are stored at the start of the record, and the blobs of data at
** the end. Hence these functions allow the caller to handle the
** serial-type and data blob separately.
**
** The following table describes the various storage classes for data:
**
**   serial type        bytes of data      type
**   --------------     ---------------    ---------------
**      0                     0            NULL
**      1                     1            signed integer
**      2                     2            signed integer
**      3                     3            signed integer
**      4                     4            signed integer
**      5                     6            signed integer
**      6                     8            signed integer
**      7                     8            IEEE float
**      8                     0            Integer constant 0
**      9                     0            Integer constant 1
**     10,11                               reserved for expansion
**    N>=12 and even       (N-12)/2        BLOB
**    N>=13 and odd        (N-13)/2        text
**
** The 8 and 9 types were added in 3.3.0, file format 4.  Prior versions
** of SQLite will not understand those serial types.
*/

/*
** Return the serial-type for the value stored in pMem.
*/
u32 sqlite3VdbeSerialType(Mem *pMem, int file_format){
	int flags = pMem->flags;
	int n;

	if( flags&MEM_Null ){
		return 0;
	}
	if( flags&MEM_Int ){
		/* Figure out whether to use 1, 2, 4, 6 or 8 bytes. */
#   define MAX_6BYTE ((((i64)0x00008000)<<32)-1)
		i64 i = pMem->u.i;
		u64 u;
		if( i<0 ){
			if( i<(-MAX_6BYTE) ) return 6;
			/* Previous test prevents:  u = -(-9223372036854775808) */
			u = -i;
		}else{
			u = i;
		}
		if( u<=127 ){
			return ((i&1)==i && file_format>=4) ? 8+(u32)u : 1;
		}
		if( u<=32767 ) return 2;
		if( u<=8388607 ) return 3;
		if( u<=2147483647 ) return 4;
		if( u<=MAX_6BYTE ) return 5;
		return 6;
	}
	if( flags&MEM_Real ){
		return 7;
	}
	assert( pMem->db->mallocFailed || flags&(MEM_Str|MEM_Blob) );
	n = pMem->n;
	if( flags & MEM_Zero ){
		n += pMem->u.nZero;
	}
	assert( n>=0 );
	return ((n*2) + 12 + ((flags&MEM_Str)!=0));
}

/*
** Return the length of the data corresponding to the supplied serial-type.
*/
u32 sqlite3VdbeSerialTypeLen(u32 serial_type){
	if( serial_type>=12 ){
		return (serial_type-12)/2;
	}else{
		static const u8 aSize[] = { 0, 1, 2, 3, 4, 6, 8, 8, 0, 0, 0, 0 };
		return aSize[serial_type];
	}
}

/*
** If we are on an architecture with mixed-endian floating 
** points (ex: ARM7) then swap the lower 4 bytes with the 
** upper 4 bytes.  Return the result.
**
** For most architectures, this is a no-op.
**
** (later):  It is reported to me that the mixed-endian problem
** on ARM7 is an issue with GCC, not with the ARM7 chip.  It seems
** that early versions of GCC stored the two words of a 64-bit
** float in the wrong order.  And that error has been propagated
** ever since.  The blame is not necessarily with GCC, though.
** GCC might have just copying the problem from a prior compiler.
** I am also told that newer versions of GCC that follow a different
** ABI get the byte order right.
**
** Developers using SQLite on an ARM7 should compile and run their
** application using -DSQLITE_DEBUG=1 at least once.  With DEBUG
** enabled, some asserts below will ensure that the byte order of
** floating point values is correct.
**
** (2007-08-30)  Frank van Vugt has studied this problem closely
** and has send his findings to the SQLite developers.  Frank
** writes that some Linux kernels offer floating point hardware
** emulation that uses only 32-bit mantissas instead of a full 
** 48-bits as required by the IEEE standard.  (This is the
** CONFIG_FPE_FASTFPE option.)  On such systems, floating point
** byte swapping becomes very complicated.  To avoid problems,
** the necessary byte swapping is carried out using a 64-bit integer
** rather than a 64-bit float.  Frank assures us that the code here
** works for him.  We, the developers, have no way to independently
** verify this, but Frank seems to know what he is talking about
** so we trust him.
*/
#ifdef SQLITE_MIXED_ENDIAN_64BIT_FLOAT
static u64 floatSwap(u64 in){
	union {
		u64 r;
		u32 i[2];
	} u;
	u32 t;

	u.r = in;
	t = u.i[0];
	u.i[0] = u.i[1];
	u.i[1] = t;
	return u.r;
}
# define swapMixedEndianFloat(X)  X = floatSwap(X)
#else
# define swapMixedEndianFloat(X)
#endif

/*
** Write the serialized data blob for the value stored in pMem into 
** buf. It is assumed that the caller has allocated sufficient space.
** Return the number of bytes written.
**
** nBuf is the amount of space left in buf[].  nBuf must always be
** large enough to hold the entire field.  Except, if the field is
** a blob with a zero-filled tail, then buf[] might be just the right
** size to hold everything except for the zero-filled tail.  If buf[]
** is only big enough to hold the non-zero prefix, then only write that
** prefix into buf[].  But if buf[] is large enough to hold both the
** prefix and the tail then write the prefix and set the tail to all
** zeros.
**
** Return the number of bytes actually written into buf[].  The number
** of bytes in the zero-filled tail is included in the return value only
** if those bytes were zeroed in buf[].
*/ 
u32 sqlite3VdbeSerialPut(u8 *buf, int nBuf, Mem *pMem, int file_format){
	u32 serial_type = sqlite3VdbeSerialType(pMem, file_format);
	u32 len;

	/* Integer and Real */
	if( serial_type<=7 && serial_type>0 ){
		u64 v;
		u32 i;
		if( serial_type==7 ){
			assert( sizeof(v)==sizeof(pMem->r) );
			memcpy(&v, &pMem->r, sizeof(v));
			swapMixedEndianFloat(v);
		}else{
			v = pMem->u.i;
		}
		len = i = sqlite3VdbeSerialTypeLen(serial_type);
		assert( len<=(u32)nBuf );
		while( i-- ){
			buf[i] = (u8)(v&0xFF);
			v >>= 8;
		}
		return len;
	}

	/* String or blob */
	if( serial_type>=12 ){
		assert( pMem->n + ((pMem->flags & MEM_Zero)?pMem->u.nZero:0)
			== (int)sqlite3VdbeSerialTypeLen(serial_type) );
		assert( pMem->n<=nBuf );
		len = pMem->n;
		memcpy(buf, pMem->z, len);
		if( pMem->flags & MEM_Zero ){
			len += pMem->u.nZero;
			assert( nBuf>=0 );
			if( len > (u32)nBuf ){
				len = (u32)nBuf;
			}
			memset(&buf[pMem->n], 0, len-pMem->n);
		}
		return len;
	}

	/* NULL or constants 0 or 1 */
	return 0;
}

/*
** Deserialize the data blob pointed to by buf as serial type serial_type
** and store the result in pMem.  Return the number of bytes read.
*/ 
u32 sqlite3VdbeSerialGet(
	const unsigned char *buf,     /* Buffer to deserialize from */
	u32 serial_type,              /* Serial type to deserialize */
	Mem *pMem                     /* Memory cell to write value into */
	){
		switch( serial_type ){
		case 10:   /* Reserved for future use */
		case 11:   /* Reserved for future use */
		case 0: {  /* NULL */
			pMem->flags = MEM_Null;
			break;
				}
		case 1: { /* 1-byte signed integer */
			pMem->u.i = (signed char)buf[0];
			pMem->flags = MEM_Int;
			return 1;
				}
		case 2: { /* 2-byte signed integer */
			pMem->u.i = (((signed char)buf[0])<<8) | buf[1];
			pMem->flags = MEM_Int;
			return 2;
				}
		case 3: { /* 3-byte signed integer */
			pMem->u.i = (((signed char)buf[0])<<16) | (buf[1]<<8) | buf[2];
			pMem->flags = MEM_Int;
			return 3;
				}
		case 4: { /* 4-byte signed integer */
			pMem->u.i = (buf[0]<<24) | (buf[1]<<16) | (buf[2]<<8) | buf[3];
			pMem->flags = MEM_Int;
			return 4;
				}
		case 5: { /* 6-byte signed integer */
			u64 x = (((signed char)buf[0])<<8) | buf[1];
			u32 y = (buf[2]<<24) | (buf[3]<<16) | (buf[4]<<8) | buf[5];
			x = (x<<32) | y;
			pMem->u.i = *(i64*)&x;
			pMem->flags = MEM_Int;
			return 6;
				}
		case 6:   /* 8-byte signed integer */
		case 7: { /* IEEE floating point */
			u64 x;
			u32 y;
#if !defined(NDEBUG) && !defined(SQLITE_OMIT_FLOATING_POINT)
			/* Verify that integers and floating point values use the same
			** byte order.  Or, that if SQLITE_MIXED_ENDIAN_64BIT_FLOAT is
			** defined that 64-bit floating point values really are mixed
			** endian.
			*/
			static const u64 t1 = ((u64)0x3ff00000)<<32;
			static const double r1 = 1.0;
			u64 t2 = t1;
			swapMixedEndianFloat(t2);
			assert( sizeof(r1)==sizeof(t2) && memcmp(&r1, &t2, sizeof(r1))==0 );
#endif

			x = (buf[0]<<24) | (buf[1]<<16) | (buf[2]<<8) | buf[3];
			y = (buf[4]<<24) | (buf[5]<<16) | (buf[6]<<8) | buf[7];
			x = (x<<32) | y;
			if( serial_type==6 ){
				pMem->u.i = *(i64*)&x;
				pMem->flags = MEM_Int;
			}else{
				assert( sizeof(x)==8 && sizeof(pMem->r)==8 );
				swapMixedEndianFloat(x);
				memcpy(&pMem->r, &x, sizeof(x));
				pMem->flags = sqlite3IsNaN(pMem->r) ? MEM_Null : MEM_Real;
			}
			return 8;
				}
		case 8:    /* Integer 0 */
		case 9: {  /* Integer 1 */
			pMem->u.i = serial_type-8;
			pMem->flags = MEM_Int;
			return 0;
				}
		default: {
			u32 len = (serial_type-12)/2;
			pMem->z = (char *)buf;
			pMem->n = len;
			pMem->xDel = 0;
			if( serial_type&0x01 ){
				pMem->flags = MEM_Str | MEM_Ephem;
			}else{
				pMem->flags = MEM_Blob | MEM_Ephem;
			}
			return len;
				 }
		}
		return 0;
}

/*
** This routine is used to allocate sufficient space for an UnpackedRecord
** structure large enough to be used with sqlite3VdbeRecordUnpack() if
** the first argument is a pointer to KeyInfo structure pKeyInfo.
**
** The space is either allocated using sqlite3DbMallocRaw() or from within
** the unaligned buffer passed via the second and third arguments (presumably
** stack space). If the former, then *ppFree is set to a pointer that should
** be eventually freed by the caller using sqlite3DbFree(). Or, if the 
** allocation comes from the pSpace/szSpace buffer, *ppFree is set to NULL
** before returning.
**
** If an OOM error occurs, NULL is returned.
*/
UnpackedRecord *sqlite3VdbeAllocUnpackedRecord(
	KeyInfo *pKeyInfo,              /* Description of the record */
	char *pSpace,                   /* Unaligned space available */
	int szSpace,                    /* Size of pSpace[] in bytes */
	char **ppFree                   /* OUT: Caller should free this pointer */
	){
		UnpackedRecord *p;              /* Unpacked record to return */
		int nOff;                       /* Increment pSpace by nOff to align it */
		int nByte;                      /* Number of bytes required for *p */

		/* We want to shift the pointer pSpace up such that it is 8-byte aligned.
		** Thus, we need to calculate a value, nOff, between 0 and 7, to shift 
		** it by.  If pSpace is already 8-byte aligned, nOff should be zero.
		*/
		nOff = (8 - (SQLITE_PTR_TO_INT(pSpace) & 7)) & 7;
		nByte = ROUND8(sizeof(UnpackedRecord)) + sizeof(Mem)*(pKeyInfo->nField+1);
		if( nByte>szSpace+nOff ){
			p = (UnpackedRecord *)sqlite3DbMallocRaw(pKeyInfo->db, nByte);
			*ppFree = (char *)p;
			if( !p ) return 0;
		}else{
			p = (UnpackedRecord*)&pSpace[nOff];
			*ppFree = 0;
		}

		p->aMem = (Mem*)&((char*)p)[ROUND8(sizeof(UnpackedRecord))];
		assert( pKeyInfo->aSortOrder!=0 );
		p->pKeyInfo = pKeyInfo;
		p->nField = pKeyInfo->nField + 1;
		return p;
}

/*
** Given the nKey-byte encoding of a record in pKey[], populate the 
** UnpackedRecord structure indicated by the fourth argument with the
** contents of the decoded record.
*/ 
void sqlite3VdbeRecordUnpack(
	KeyInfo *pKeyInfo,     /* Information about the record format */
	int nKey,              /* Size of the binary record */
	const void *pKey,      /* The binary record */
	UnpackedRecord *p      /* Populate this structure before returning. */
	){
		const unsigned char *aKey = (const unsigned char *)pKey;
		int d; 
		u32 idx;                        /* Offset in aKey[] to read from */
		u16 u;                          /* Unsigned loop counter */
		u32 szHdr;
		Mem *pMem = p->aMem;

		p->flags = 0;
		assert( EIGHT_BYTE_ALIGNMENT(pMem) );
		idx = getVarint32(aKey, szHdr);
		d = szHdr;
		u = 0;
		while( idx<szHdr && u<p->nField && d<=nKey ){
			u32 serial_type;

			idx += getVarint32(&aKey[idx], serial_type);
			pMem->enc = pKeyInfo->enc;
			pMem->db = pKeyInfo->db;
			/* pMem->flags = 0; // sqlite3VdbeSerialGet() will set this for us */
			pMem->zMalloc = 0;
			d += sqlite3VdbeSerialGet(&aKey[d], serial_type, pMem);
			pMem++;
			u++;
		}
		assert( u<=pKeyInfo->nField + 1 );
		p->nField = u;
}

/*
** This function compares the two table rows or index records
** specified by {nKey1, pKey1} and pPKey2.  It returns a negative, zero
** or positive integer if key1 is less than, equal to or 
** greater than key2.  The {nKey1, pKey1} key must be a blob
** created by th OP_MakeRecord opcode of the VDBE.  The pPKey2
** key must be a parsed key such as obtained from
** sqlite3VdbeParseRecord.
**
** Key1 and Key2 do not have to contain the same number of fields.
** The key with fewer fields is usually compares less than the 
** longer key.  However if the UNPACKED_INCRKEY flags in pPKey2 is set
** and the common prefixes are equal, then key1 is less than key2.
** Or if the UNPACKED_MATCH_PREFIX flag is set and the prefixes are
** equal, then the keys are considered to be equal and
** the parts beyond the common prefix are ignored.
*/
int sqlite3VdbeRecordCompare(
	int nKey1, const void *pKey1, /* Left key */
	UnpackedRecord *pPKey2        /* Right key */
	){
		int d1;            /* Offset into aKey[] of next data element */
		u32 idx1;          /* Offset into aKey[] of next header element */
		u32 szHdr1;        /* Number of bytes in header */
		int i = 0;
		int nField;
		int rc = 0;
		const unsigned char *aKey1 = (const unsigned char *)pKey1;
		KeyInfo *pKeyInfo;
		Mem mem1;

		pKeyInfo = pPKey2->pKeyInfo;
		mem1.enc = pKeyInfo->enc;
		mem1.db = pKeyInfo->db;
		/* mem1.flags = 0;  // Will be initialized by sqlite3VdbeSerialGet() */
		VVA_ONLY( mem1.zMalloc = 0; ) /* Only needed by assert() statements */

			/* Compilers may complain that mem1.u.i is potentially uninitialized.
			** We could initialize it, as shown here, to silence those complaints.
			** But in fact, mem1.u.i will never actually be used uninitialized, and doing 
			** the unnecessary initialization has a measurable negative performance
			** impact, since this routine is a very high runner.  And so, we choose
			** to ignore the compiler warnings and leave this variable uninitialized.
			*/
			/*  mem1.u.i = 0;  // not needed, here to silence compiler warning */

			idx1 = getVarint32(aKey1, szHdr1);
		d1 = szHdr1;
		nField = pKeyInfo->nField;
		assert( pKeyInfo->aSortOrder!=0 );
		while( idx1<szHdr1 && i<pPKey2->nField ){
			u32 serial_type1;

			/* Read the serial types for the next element in each key. */
			idx1 += getVarint32( aKey1+idx1, serial_type1 );
			if( d1>=nKey1 && sqlite3VdbeSerialTypeLen(serial_type1)>0 ) break;

			/* Extract the values to be compared.
			*/
			d1 += sqlite3VdbeSerialGet(&aKey1[d1], serial_type1, &mem1);

			/* Do the comparison
			*/
			rc = sqlite3MemCompare(&mem1, &pPKey2->aMem[i],
				i<nField ? pKeyInfo->aColl[i] : 0);
			if( rc!=0 ){
				assert( mem1.zMalloc==0 );  /* See comment below */

				/* Invert the result if we are using DESC sort order. */
				if( i<nField && pKeyInfo->aSortOrder[i] ){
					rc = -rc;
				}

				/* If the PREFIX_SEARCH flag is set and all fields except the final
				** rowid field were equal, then clear the PREFIX_SEARCH flag and set 
				** pPKey2->rowid to the value of the rowid field in (pKey1, nKey1).
				** This is used by the OP_IsUnique opcode.
				*/
				if( (pPKey2->flags & UNPACKED_PREFIX_SEARCH) && i==(pPKey2->nField-1) ){
					assert( idx1==szHdr1 && rc );
					assert( mem1.flags & MEM_Int );
					pPKey2->flags &= ~UNPACKED_PREFIX_SEARCH;
					pPKey2->rowid = mem1.u.i;
				}

				return rc;
			}
			i++;
		}

		/* No memory allocation is ever used on mem1.  Prove this using
		** the following assert().  If the assert() fails, it indicates a
		** memory leak and a need to call sqlite3VdbeMemRelease(&mem1).
		*/
		assert( mem1.zMalloc==0 );

		/* rc==0 here means that one of the keys ran out of fields and
		** all the fields up to that point were equal. If the UNPACKED_INCRKEY
		** flag is set, then break the tie by treating key2 as larger.
		** If the UPACKED_PREFIX_MATCH flag is set, then keys with common prefixes
		** are considered to be equal.  Otherwise, the longer key is the 
		** larger.  As it happens, the pPKey2 will always be the longer
		** if there is a difference.
		*/
		assert( rc==0 );
		if( pPKey2->flags & UNPACKED_INCRKEY ){
			rc = -1;
		}else if( pPKey2->flags & UNPACKED_PREFIX_MATCH ){
			/* Leave rc==0 */
		}else if( idx1<szHdr1 ){
			rc = 1;
		}
		return rc;
}


/*
** pCur points at an index entry created using the OP_MakeRecord opcode.
** Read the rowid (the last field in the record) and store it in *rowid.
** Return SQLITE_OK if everything works, or an error code otherwise.
**
** pCur might be pointing to text obtained from a corrupt database file.
** So the content cannot be trusted.  Do appropriate checks on the content.
*/
int sqlite3VdbeIdxRowid(sqlite3 *db, BtCursor *pCur, i64 *rowid){
	i64 nCellKey = 0;
	int rc;
	u32 szHdr;        /* Size of the header */
	u32 typeRowid;    /* Serial type of the rowid */
	u32 lenRowid;     /* Size of the rowid */
	Mem m, v;

	UNUSED_PARAMETER(db);

	/* Get the size of the index entry.  Only indices entries of less
	** than 2GiB are support - anything large must be database corruption.
	** Any corruption is detected in sqlite3BtreeParseCellPtr(), though, so
	** this code can safely assume that nCellKey is 32-bits  
	*/
	assert( sqlite3BtreeCursorIsValid(pCur) );
	VVA_ONLY(rc =) sqlite3BtreeKeySize(pCur, &nCellKey);
	assert( rc==SQLITE_OK );     /* pCur is always valid so KeySize cannot fail */
	assert( (nCellKey & SQLITE_MAX_U32)==(u64)nCellKey );

	/* Read in the complete content of the index entry */
	memset(&m, 0, sizeof(m));
	rc = sqlite3VdbeMemFromBtree(pCur, 0, (int)nCellKey, 1, &m);
	if( rc ){
		return rc;
	}

	/* The index entry must begin with a header size */
	(void)getVarint32((u8*)m.z, szHdr);
	testcase( szHdr==3 );
	testcase( szHdr==m.n );
	if( unlikely(szHdr<3 || (int)szHdr>m.n) ){
		goto idx_rowid_corruption;
	}

	/* The last field of the index should be an integer - the ROWID.
	** Verify that the last entry really is an integer. */
	(void)getVarint32((u8*)&m.z[szHdr-1], typeRowid);
	testcase( typeRowid==1 );
	testcase( typeRowid==2 );
	testcase( typeRowid==3 );
	testcase( typeRowid==4 );
	testcase( typeRowid==5 );
	testcase( typeRowid==6 );
	testcase( typeRowid==8 );
	testcase( typeRowid==9 );
	if( unlikely(typeRowid<1 || typeRowid>9 || typeRowid==7) ){
		goto idx_rowid_corruption;
	}
	lenRowid = sqlite3VdbeSerialTypeLen(typeRowid);
	testcase( (u32)m.n==szHdr+lenRowid );
	if( unlikely((u32)m.n<szHdr+lenRowid) ){
		goto idx_rowid_corruption;
	}

	/* Fetch the integer off the end of the index record */
	sqlite3VdbeSerialGet((u8*)&m.z[m.n-lenRowid], typeRowid, &v);
	*rowid = v.u.i;
	sqlite3VdbeMemRelease(&m);
	return SQLITE_OK;

	/* Jump here if database corruption is detected after m has been
	** allocated.  Free the m object and return SQLITE_CORRUPT. */
idx_rowid_corruption:
	testcase( m.zMalloc!=0 );
	sqlite3VdbeMemRelease(&m);
	return SQLITE_CORRUPT_BKPT;
}

/*
** Compare the key of the index entry that cursor pC is pointing to against
** the key string in pUnpacked.  Write into *pRes a number
** that is negative, zero, or positive if pC is less than, equal to,
** or greater than pUnpacked.  Return SQLITE_OK on success.
**
** pUnpacked is either created without a rowid or is truncated so that it
** omits the rowid at the end.  The rowid at the end of the index entry
** is ignored as well.  Hence, this routine only compares the prefixes 
** of the keys prior to the final rowid, not the entire key.
*/
int sqlite3VdbeIdxKeyCompare(
	VdbeCursor *pC,             /* The cursor to compare against */
	UnpackedRecord *pUnpacked,  /* Unpacked version of key to compare against */
	int *res                    /* Write the comparison result here */
	){
		i64 nCellKey = 0;
		int rc;
		BtCursor *pCur = pC->pCursor;
		Mem m;

		assert( sqlite3BtreeCursorIsValid(pCur) );
		VVA_ONLY(rc =) sqlite3BtreeKeySize(pCur, &nCellKey);
		assert( rc==SQLITE_OK );    /* pCur is always valid so KeySize cannot fail */
		/* nCellKey will always be between 0 and 0xffffffff because of the say
		** that btreeParseCellPtr() and sqlite3GetVarint32() are implemented */
		if( nCellKey<=0 || nCellKey>0x7fffffff ){
			*res = 0;
			return SQLITE_CORRUPT_BKPT;
		}
		memset(&m, 0, sizeof(m));
		rc = sqlite3VdbeMemFromBtree(pC->pCursor, 0, (int)nCellKey, 1, &m);
		if( rc ){
			return rc;
		}
		assert( pUnpacked->flags & UNPACKED_PREFIX_MATCH );
		*res = sqlite3VdbeRecordCompare(m.n, m.z, pUnpacked);
		sqlite3VdbeMemRelease(&m);
		return SQLITE_OK;
}

/*
** This routine sets the value to be returned by subsequent calls to
** sqlite3_changes() on the database handle 'db'. 
*/
void sqlite3VdbeSetChanges(sqlite3 *db, int nChange){
	assert( sqlite3_mutex_held(db->mutex) );
	db->nChange = nChange;
	db->nTotalChange += nChange;
}

/*
** Set a flag in the vdbe to update the change counter when it is finalised
** or reset.
*/
void sqlite3VdbeCountChanges(Vdbe *v){
	v->changeCntOn = 1;
}

/*
** Mark every prepared statement associated with a database connection
** as expired.
**
** An expired statement means that recompilation of the statement is
** recommend.  Statements expire when things happen that make their
** programs obsolete.  Removing user-defined functions or collating
** sequences, or changing an authorization function are the types of
** things that make prepared statements obsolete.
*/
void sqlite3ExpirePreparedStatements(sqlite3 *db){
	Vdbe *p;
	for(p = db->pVdbe; p; p=p->pNext){
		p->expired = 1;
	}
}

/*
** Return the database associated with the Vdbe.
*/
sqlite3 *sqlite3VdbeDb(Vdbe *v){
	return v->db;
}

/*
** Return a pointer to an sqlite3_value structure containing the value bound
** parameter iVar of VM v. Except, if the value is an SQL NULL, return 
** 0 instead. Unless it is NULL, apply affinity aff (one of the SQLITE_AFF_*
** constants) to the value before returning it.
**
** The returned value must be freed by the caller using sqlite3ValueFree().
*/
sqlite3_value *sqlite3VdbeGetValue(Vdbe *v, int iVar, u8 aff){
	assert( iVar>0 );
	if( v ){
		Mem *pMem = &v->aVar[iVar-1];
		if( 0==(pMem->flags & MEM_Null) ){
			sqlite3_value *pRet = sqlite3ValueNew(v->db);
			if( pRet ){
				sqlite3VdbeMemCopy((Mem *)pRet, pMem);
				sqlite3ValueApplyAffinity(pRet, aff, SQLITE_UTF8);
				sqlite3VdbeMemStoreType((Mem *)pRet);
			}
			return pRet;
		}
	}
	return 0;
}

/*
** Configure SQL variable iVar so that binding a new value to it signals
** to sqlite3_reoptimize() that re-preparing the statement may result
** in a better query plan.
*/
void sqlite3VdbeSetVarmask(Vdbe *v, int iVar){
	assert( iVar>0 );
	if( iVar>32 ){
		v->expmask = 0xffffffff;
	}else{
		v->expmask |= ((u32)1 << (iVar-1));
	}
}

#pragma endregion