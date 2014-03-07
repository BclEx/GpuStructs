#include "Core+Syntax.cu.h"
#include "..\Core+Vbde\Vdbe.cu.h"

namespace Core
{
	__device__ AFF Expr::Affinity()
	{
		Expr *expr = SkipCollate();
		int op = expr->OP;
		if (op == TK_SELECT)
		{
			_assert(expr->Flags&EP_xIsSelect);
			return expr->x.Select->EList->a[0].Expr->Affinity();
		}
#ifndef OMIT_CAST
		if (op == TK_CAST)
		{
			_assert(!ExprHasProperty(expr, EP_IntValue));
			return AffinityType(expr->u.Token);
		}
#endif
		if ((op == TK_AGG_COLUMN || op == TK_COLUMN || op == TK_REGISTER) && expr->Table)
		{
			// op == TK_REGISTER && expr->Table happens when pExpr was originally a TK_COLUMN but was previously evaluated and cached in a register
			int j = expr->ColumnIdx;
			if (j < 0)
				return AFF_INTEGER;
			_assert(expr->Table && j < expr->Table->Cols.length);
			return expr->Table->Cols[j].Affinity;
		}
		return expr->Affinity;
	}

	__device__ Expr *Parse::ExprAddCollateToken(Expr *expr, Token *collName)
	{
		if (collName->length > 0)
		{
			Expr *newExpr = Expr::Alloc(Ctx, TK_COLLATE, collName, 1);
			if (newExpr)
			{
				newExpr->Left = expr;
				newExpr->Flags |= EP_Collate;
				expr = newExpr;
			}
		}
		return expr;
	}

	__device__ Expr *Parse::ExprAddCollateString(Expr *expr, const char *z)
	{
		_assert(z);
		Token s;
		s.data = (char *)z;
		s.length = _strlen30(s);
		return ExprAddCollateToken(expr, &s);
	}

	__device__ Expr *Expr::SkipCollate()
	{
		Expr *expr = this;
		while (expr && (expr->OP == TK_COLLATE || expr->OP == TK_AS))
			expr = expr->Left;
		return expr;
	}

	__device__ CollSeq *Parse::ExprCollSeq(Expr *expr)
	{
		Context *ctx = Ctx;
		CollSeq *coll = nullptr;
		Expr *p = expr;
		while (p)
		{
			int op = p->OP;
			if (op == TK_CAST || op == TK_UPLUS)
			{
				p = p->Left;
				continue;
			}
			_assert(op != TK_REGISTER || p->OP2 != TK_COLLATE);
			if (op == TK_COLLATE)
			{
				if (ctx->Init.Busy) // Do not report errors when parsing while the schema 
					coll = FindCollSeq(ctx, ENC(ctx), p->u.Token, 0);
				else
					coll = GetCollSeq(ENC(ctx), 0, p->u.Token);
				break;
			}
			if (p->Table && (op == TK_AGG_COLUMN || op == TK_COLUMN || op == TK_REGISTER || op == TK_TRIGGER))
			{
				// op==TK_REGISTER && p->pTab!=0 happens when pExpr was originally a TK_COLUMN but was previously evaluated and cached in a register
				int j = p->ColumnIdx;
				if (j >= 0)
				{
					const char *nameColl = p->Table->Cols[j].Coll;
					coll = FindCollSeq(ctx, ENC(ctx), nameColl, 0);
				}
				break;
			}
			if (p->Flags & EP_Collate)
				p = (SysEx_ALWAYS(p->Left) && (p->Left->Flags & EP_Collate) != 0 ? p->Left : p->Right);
			else
				break;
		}
		if (CheckCollSeq(coll))
			coll = nullptr;
		return coll;
	}

	__device__ AFF Expr::CompareAffinity(AFF aff2)
	{
		AFF aff1 = Affinity();
		if (aff1 && aff2) // Both sides of the comparison are columns. If one has numeric affinity, use that. Otherwise use no affinity.
			return (IsNumericAffinity(aff1) || IsNumericAffinity(aff2) ? AFF_NUMERIC : AFF_NONE);
		else if (!aff1 && !aff2) // Neither side of the comparison is a column.  Compare the results directly.
			return AFF_NONE;
		else // One side is a column, the other is not. Use the columns affinity.
		{
			_assert(aff1 == 0 || aff2 == 0);
			return (aff1 + aff2);
		}
	}

	__device__ static AFF ComparisonAffinity(Expr *expr)
	{
		_assert(expr->OP == TK_EQ || expr->OP == TK_IN || expr->OP == TK_LT ||
			expr->OP == TK_GT || expr->OP == TK_GE || expr->OP == TK_LE ||
			expr->OP == TK_NE || expr->OP == TK_IS || expr->OP == TK_ISNOT);
		_assert(expr->Left);
		AFF aff = expr->Left->Affinity();
		if (expr->Right)
			aff = expr->Right->CompareAffinity(aff);
		else if (ExprHasProperty(expr, EP_xIsSelect))
			aff = expr->x.Select->EList->a[0].Expr->CompareAffinity(aff);
		else if (!aff)
			aff = AFF_NONE;
		return aff;
	}

	__device__ bool Expr::ValidIndexAffinity(AFF indexAff)
	{
		AFF aff = ComparisonAffinity(this);
		switch (aff)
		{
		case AFF_NONE:
			return true;
		case AFF_TEXT:
			return (indexAff == AFF_TEXT);
		default:
			return IsNumericAffinity(indexAff);
		}
	}

	__device__ static AFF BinaryCompareP5(Expr *expr1, Expr *expr2, uint8 jumpIfNull)
	{
		AFF aff = expr2->Affinity();
		aff = expr1->CompareAffinity(aff) | (uint8)jumpIfNull;
		return aff;
	}

	__device__ CollSeq *Parse::BinaryCompareCollSeq(Expr *left, Expr *right)
	{
		_assert(left);
		CollSeq *coll;
		if (left->Flags & EP_Collate)
			coll = ExprCollSeq(left);
		else if (right && (right->Flags & EP_Collate))
			coll = ExprCollSeq(right);
		else{
			coll = ExprCollSeq(left);
			if (!coll)
				coll = ExprCollSeq(right);
		}
		return coll;
	}

	__device__ static int CodeCompare(Parse *parse, Expr *left, Expr *right, int opcode, int in1, int in2, int dest, int jumpIfNull)
	{
		CollSeq *p4 = parse->BinaryCompareCollSeq(left, right);
		int p5 = BinaryCompareP5(left, right, jumpIfNull);
		Vdbe *v = parse->V;
		int addr = v->AddOp4(opcode, in2, dest, in1, (void *)p4, Vdbe::P4T_COLLSEQ);
		v->ChangeP5((uint8)p5);
		return addr;
	}

#if MAX_EXPR_DEPTH>0
	__device__ RC Parse::ExprCheckHeight(int height)
	{
		RC rc = RC_OK;
		int maxHeight = Ctx->Limits[LIMIT_EXPR_DEPTH];
		if (height > maxHeight)
		{
			ErrorMsg("Expression tree is too large (maximum depth %d)", maxHeight);
			rc = RC_ERROR;
		}
		return rc;
	}

	__device__ static void HeightOfExpr(Expr *p, int *height)
	{
		if (p)
			if (p->Height > *height)
				*height = p->Height;
	}
	__device__ static void HeightOfExprList(ExprList *p, int *height)
	{
		if (p)
			for (int i = 0; i < p->Exprs.length; i++)
				HeightOfExpr(p->a[i].Expr, height);
	}
	__device__ static void HeightOfSelect(Select *p, int *height)
	{
		if (p)
		{
			HeightOfExpr(p->Where, hight);
			HeightOfExpr(p->Having, height);
			HeightOfExpr(p->Limit, height);
			HeightOfExpr(p->Offset, height);
			HeightOfExprList(p->EList, height);
			HeightOfExprList(p->GroupBy, height);
			HeightOfExprList(p->OrderBy, height);
			HeightOfSelect(p->Prior, height);
		}
	}

	__device__ static void ExprSetHeight(Expr *p)
	{
		int height = 0;
		HeightOfExpr(p->Left, &height);
		HeightOfExpr(p->Right, &height);
		if (ExprHasProperty(p, EP_xIsSelect))
			HeightOfSelect(p->x.Select, &height);
		else
			HeightOfExprList(p->x.List, &height);
	}
	p->Height = height + 1;
}
__device__ void Parse::ExprSetHeight(Expr *expr)
{
	ExprSetHeight(expr);
	ExprCheckHeight(expr->Height);
}

__device__ int Expr::SelectExprHeight(Select *select)
{
	int height = 0;
	HeightOfSelect(select, &height);
	return height;
}
#else
#define ExprSetHeight(y)
#endif

	__device__ Expr *Expr::Alloc(Context *ctx, int op, const Token *token, bool dequote)
	{
		int extraSize = 0;
		int value = 0;
		if (token)
		{
			if (op != TK_INTEGER || !token->data || ConvertEx::GetInt32(token, &value) == 0)
			{
				extraSize = token->length + 1;
				_assert(value >= 0);
			}
		}
		Expr *newExpr = (Expr *)SysEx::TagAlloc(ctx, sizeof(Expr)+extraSize, true);
		if (newExpr)
		{
			newExpr->OP = (uint8)op;
			newExpr->Agg = -1;
			if (token)
			{
				if (extraSize == 0)
				{
					newExpr->Flags |= EP_IntValue;
					newExpr->u.I = value;
				}
				else
				{
					int c;
					newExpr->u.Token = (char *)&newExpr[1];
					_assert(token->data || token->length == 0);
					if (token->length)
						_memcpy(newExpr->u.Token, token->data, token->length);
					newExpr->u.Token[token->length] = 0;
					if (dequote && extraSize >= 3  && ((c = token->data[0]) == '\'' || c == '"' || c == '[' || c == '`'))
					{
						sqlite3Dequote(newExpr->u.Token);
						if (c == '"')
							newExpr->Flags |= EP_DblQuoted;
					}
				}
			}
#if MAX_EXPR_DEPTH>0
			newExpr->Height = 1;
#endif  
		}
		return newExpr;
	}

	__device__ Expr *Expr::Alloc(Context *ctx, int op, const char *token)
	{
		Token x;
		x.data = (char *)token;
		x.length = (token ? _strlen30(token) : 0);
		return Alloc(ctx, op, &x, false);
	}

	__device__ void Expr::AttachSubtrees(Context *ctx, Expr *root, Expr *left, Expr *right)
	{
		if (!root)
		{
			_assert(ctx->MallocFailed);
			Expr::Delete(ctx, left);
			Expr::Delete(ctx, right);
		}
		else
		{
			if (right)
			{
				root->Right = right;
				root->Flags |= EP_Collate & right->Flags;
			}
			if (left)
			{
				root->Left = left;
				root->Flags |= EP_Collate & left->Flags;
			}
			ExprSetHeight(root);
		}
	}

	__device__ Expr *Parse::PExpr(int op, Expr *left, Expr *right, const Token *token)
	{
		Expr *p;
		if (op == TK_AND && left && right) // Take advantage of short-circuit false optimization for AND
			p = Expr::And(Ctx, left, right);
		else
		{
			p = Expr::Alloc(Ctx, op, token, true);
			Expr::AttachSubtrees(Ctx, p, left, right);
		}
		if (p)
			ExprCheckHeight(p->Height);
		return p;
	}

	__device__ static bool ExprAlwaysFalse(Expr *p)
	{
		if (ExprHasProperty(p, EP_FromJoin)) return false;
		int v = 0;
		if (!ExprIsInteger(p, &v)) return false;
		return !v;
	}

	__device__ Expr *Expr::And(Context *ctx, Expr *left, Expr *right)
	{
		if (!left)
			return right;
		else if (!right)
			return left;
		else if (ExprAlwaysFalse(left) || ExprAlwaysFalse(right))
		{
			Delete(ctx, left);
			Delete(ctx, right);
			return Alloc(ctx, TK_INTEGER, &sqlite3IntTokens[0], false);
		}
		else
		{
			Expr *newExpr = Alloc(ctx, TK_AND, nullptr, false);
			AttachSubtrees(ctx, newExpr, left, right);
			return newExpr;
		}
	}

	__device__ Expr *Parse::ExprFunction(ExprList *list, Token *token)
	{
		_assert(token);
		Context *ctx = Ctx;
		Expr *newExpr = Expr::Alloc(ctx, TK_FUNCTION, token, true);
		if (!newExpr)
		{
			Expr::ListDelete(ctx, list); // Avoid memory leak when malloc fails
			return nullptr;
		}
		newExpr->x.List = list;
		_assert(!ExprHasProperty(newExpr, EP_xIsSelect));
		ExprSetHeight(newExpr);
		return newExpr;
	}


	__device__ void Parse::ExprAssignVarNumber(Expr *expr)
	{
		if (!expr)
			return;
		_assert(!ExprHasAnyProperty(expr, EP_IntValue|EP_Reduced|EP_TokenOnly));
		const char *z = expr->u.Token;
		_assert(z && z[0] != 0);
		Context *ctx = Ctx;
		if (z[1] == 0) 
		{
			_assert(z[0] == '?');
			// Wildcard of the form "?".  Assign the next variable number
			expr->ColumnIdx = (yVars)(++VarsSeen);
		}
		else
		{
			yVars x = 0;
			uint32 length = _strlen30(z);
			if (z[0] == '?') 
			{
				// Wildcard of the form "?nnn".  Convert "nnn" to an integer and use it as the variable number
				int64 i;
				bool ok = !ConvertEx::Atoi64(&z[1], &i, length-1, TEXTENCODE_UTF8);
				expr->ColumnIdx = x = (yVars)i;
				ASSERTCOVERAGE(i == 0);
				ASSERTCOVERAGE(i == 1);
				ASSERTCOVERAGE(i == ctx->Limits[LIMIT_VARIABLE_NUMBER]-1);
				ASSERTCOVERAGE(i == ctx->Limits[LIMIT_VARIABLE_NUMBER]);
				if (!ok || i < 1 || i > ctx->Limits[LIMIT_VARIABLE_NUMBER])
				{
					ErrorMsg("variable number must be between ?1 and ?%d", ctx->Limits[LIMIT_VARIABLE_NUMBER]);
					x = 0;
				}
				if (i > VarsSeen)
					VarsSeen = (int)i;
			}
			else
			{
				// Wildcards like ":aaa", "$aaa" or "@aaa".  Reuse the same variable number as the prior appearance of the same name, or if the name
				// has never appeared before, reuse the same variable number
				yVars i;
				for (i = 0; i < Vars.length; i++)
				{
					if (Vars[i] && !_strcmp(Vars[i], z))
					{
						expr->ColumnIdx = x = (yVars)i + 1;
						break;
					}
				}
				if (x == 0)
					x = expr->ColumnIdx = (yVars)(++VarsSeen);
			}
			if (x > 0)
			{
				if (x > Vars.length)
				{
					char **a = (char **)SysEx::TagRealloc(ctx, Vars.data, x * sizeof(a[0]));
					if (!a) return;  // Error reported through db->mallocFailed
					Vars.data = a;
					_memset(&a[Vars.length], 0, (x - Vars.length)*sizeof(a[0]));
					Vars.length = x;
				}
				if (z[0] != '?' || Vars[x-1] == nullptr)
				{
					SysEx::TagFree(ctx, Vars[x-1]);
					Vars[x-1] = SysEx::TagStrNDup(ctx, z, n);
				}
			}
		} 
		if (!Errs && VarsSeen > ctx->Limits[LIMIT_VARIABLE_NUMBER])
			ErrorMsg("too many SQL variables");
	}

	__device__ void Expr::Delete(Context *ctx, Expr *expr)
	{
		if (!expr) return;
		// Sanity check: Assert that the IntValue is non-negative if it exists
		_assert(!ExprHasProperty(expr, EP_IntValue) || expr->u.I >= 0);
		if (!ExprHasAnyProperty(expr, EP_TokenOnly))
		{
			Delete(ctx, expr->Left);
			Delete(ctx, expr->Right);
			if (!ExprHasProperty(expr, EP_Reduced) && (expr->Flags2 & EP2_MallocedToken))
				SysEx::TagFree(ctx, expr->u.Token);
			if (ExprHasProperty(expr, EP_xIsSelect))
				SelectDelete(ctx, expr->x.Select);
			else
				ListDelete(ctx, expr->x.List);
		}
		if (!ExprHasProperty(expr, EP_Static))
			SysEx::TagFree(ctx, expr);
	}

#pragma region Clone

	__device__ static int ExprStructSize(Expr *exor)
	{
		if (ExprHasProperty(expr, EP_TokenOnly)) return EXPR_TOKENONLYSIZE;
		if (ExprHasProperty(expr, EP_Reduced)) return EXPR_REDUCEDSIZE;
		return EXPR_FULLSIZE;
	}

	__device__ static int DupedExprStructSize(Expr *expr, int flags)
	{
		_assert(flags == EXPRDUP_REDUCE || flags == 0); // Only one flag value allowed
		int size;
		if (!(flags & EXPRDUP_REDUCE))
			size = EXPR_FULLSIZE;
		else
		{
			_assert(!ExprHasAnyProperty(expr, EP_TokenOnly|EP_Reduced));
			_assert(!ExprHasProperty(expr, EP_FromJoin)); 
			_assert(!(expr->Flags2 & EP2_MallocedToken));
			_assert(!(expr->Flags2 & EP2_Irreducible));
			if (expr->Left || expr->Right || expr->x.List)
				size = EXPR_REDUCEDSIZE | EP_Reduced;
			else
				size = EXPR_TOKENONLYSIZE | EP_TokenOnly;
		}
		return size;
	}

	__device__ static int DupedExprNodeSize(Expr *expr, int flags)
	{
		int bytes = DupedExprStructSize(expr, flags) & 0xfff;
		if (!ExprHasProperty(expr, EP_IntValue) && expr->u.Token)
			bytes += _strlen30(expr->u.Token) + 1;
		return SysEx_ROUND8(bytes);
	}

	__device__ static int DupedExprSize(Expr *expr, int flags)
	{
		int bytes = 0;
		if (expr)
		{
			bytes = DupedExprNodeSize(expr, flags);
			if (flags & EXPRDUP_REDUCE)
				bytes += DupedExprSize(expr->Left, flags) + DupedExprSize(expr->Right, flags);
		}
		return bytes;
	}

	__device__ static Expr *ExprDup2(Context *ctx, Expr *expr, int flags, uint8 **buffer)
	{
		Expr *newExpr = nullptr; // Value to return
		if (expr)
		{
			_assert(!buffer || isReduced);
			const bool isReduced = (flags & EXPRDUP_REDUCE);
			// Figure out where to write the new Expr structure.
			uint8 *alloc;
			uint32 staticFlag;
			if (buffer)
			{
				alloc = *buffer;
				staticFlag = EP_Static;
			}
			else
			{
				alloc = (uint8 *)SysEx::TagAlloc(ctx, DupedExprSize(expr, flags));
				staticFlag = 0;
			}
			newExpr = (Expr *)alloc;
			if (newExpr)
			{
				// Set nNewSize to the size allocated for the structure pointed to by pNew. This is either EXPR_FULLSIZE, EXPR_REDUCEDSIZE or
				// EXPR_TOKENONLYSIZE. nToken is set to the number of bytes consumed * by the copy of the p->u.zToken string (if any).
				const unsigned structSize = DupedExprStructSize(expr, flags);
				const int newSize = structSize & 0xfff;
				int tokenLength;
				int tokenLength = (!ExprHasProperty(expr, EP_IntValue) && expr->u.Token ? _strlen30(p->u.Token) + 1 : 0);
				if (isReduced)
				{
					_assert(!ExprHasProperty(expr, EP_Reduced));
					_memcpy(alloc, (uint8 *)expr, newSize);
				}
				else
				{
					int size = ExprStructSize(p);
					_memcpy(alloc, (uint8 *)expr, size);
					_memset(&alloc[size], 0, EXPR_FULLSIZE-size);
				}

				// Set the EP_Reduced, EP_TokenOnly, and EP_Static flags appropriately.
				newExpr->Flags &= ~(EP_Reduced|EP_TokenOnly|EP_Static);
				newExpr->Flags |= structSize & (EP_Reduced|EP_TokenOnly);
				newExpr->Flags |= staticFlag;
				// Copy the p->u.zToken string, if any.
				if (tokenLength)
				{
					char *token = newExpr->u.Token = (char *)&alloc[newSize];
					_memcpy(token, expr->u.Token, tokenLength);
				}
				if (!((expr->Flags | newExpr->Flags) & EP_TokenOnly))
				{
					// Fill in the pNew->x.pSelect or pNew->x.pList member.
					if (ExprHasProperty(expr, EP_xIsSelect))
						newExpr->x.Select = Expr::SelectDup(ctx, expr->x.Select, isReduced);
					else
						newExpr->x.List = Expr::ExprListDup(ctx, expr->x.List, isReduced);
				}
				// Fill in pNew->pLeft and pNew->pRight.
				if (ExprHasAnyProperty(newExpr, EP_Reduced|EP_TokenOnly))
				{
					alloc += DupedExprNodeSize(expr, flags);
					if (ExprHasProperty(newExpr, EP_Reduced))
					{
						newExpr->Left = ExprDup2(ctx, expr->Left, EXPRDUP_REDUCE, &alloc);
						newExpr->Right = ExprDup2(ctx, expr->Right, EXPRDUP_REDUCE, &alloc);
					}
					if (buffer)
						*buffer = alloc;
				}
				else
				{
					newExpr->Flags2 = (EP2)0;
					if (!ExprHasAnyProperty(expr, EP_TokenOnly))
					{
						newExpr->Left = ExprDup2(ctx, expr->Left, 0, nullptr);
						newExpr->Right = ExprDup2(ctx, expr->Right, 0, nullptr);
					}
				}
			}
		}
		return newExpr;
	}

	__device__ Expr *Expr::ExprDup(Context *ctx, Expr *expr, int flags) { return ExprDup2(ctx, expr, flags, nullptr); }
	__device__ ExprList *Expr::ExprListDup(Context *ctx, ExprList *list, int flags)
	{
		if (!list)
			return 0;
		ExprList *newList = (ExprList *)SysEx::TagAlloc(ctx, sizeof(*newList));
		if (!newList)
			return 0;
		int i;
		newList->ECursor = 0;
		newList->Exprs = i = list->Exprs;
		if ((flags & EXPRDUP_REDUCE) == 0)
		{
			for (i = 1; i < list->Exprs; i += i) { }
		}
		ExprList::ExprListItem *item;
		newList->Ids = item = (ExprList::ExprListItem *)SysEx::TagAlloc(ctx, i * sizeof(list->Ids[0]));
		if (!item)
		{
			SysEx::TagFree(ctx, newList);
			return 0;
		} 
		ExprList::ExprListItem *oldItem = list->a;
		for (i = 0; i < list->Exprs; i++, item++, oldItem++)
		{
			Expr *oldExpr = oldItem->Expr;
			item->Expr = ExprDup(db, oldExpr, flags);
			item->Name = SysEx::TagStrDup(ctx, oldItem->Name);
			item->Span = SysEx::TagStrDup(ctx, oldItem->Span);
			item->SortOrder = oldItem->SortOrder;
			item->Done = false;
			item->OrderByCol = oldItem->OrderByCol;
			item->Alias = oldItem->Alias;
		}
		return newList;
	}

#if !defined(OMIT_VIEW) || !defined(OMIT_TRIGGER) || !defined(OMIT_SUBQUERY)
	__device__ SrcList *Expr::SrcListDup(Context *ctx, SrcList *list, int flags)
	{
		if (!list)
			return 0;
		int bytes = sizeof(*list) + (list->Srcs > 0 ? sizeof(list->a[0]) * (list->Srcs-1) : 0);
		SrcList *newList = (SrcList *)SysEx::TagAlloc(ctx, bytes);
		if (!newList)
			return 0;
		newList->Srcs = newList->Allocs = list->Srcs;
		for (int i=0; i<p->nSrc; i++)
		{
			SrcList::SrcListItem *newItem = &newList->a[i];
			SrcList::SrcListItem *oldItem = &list->a[i];
			newItem->Schema = oldItem->Schema;
			newItem->Database = SysEx::TagStrDup(ctx, oldItem->Database);
			newItem->Name = SysEx::TagStrDup(ctx, oldItem->Name);
			newItem->Alias = SysEx::TagStrDup(ctx, oldItem->Alias);
			newItem->Jointype = oldItem->Jointype;
			newItem->Cursor = oldItem->Cursor;
			newItem->AddrFillSub = oldItem->AddrFillSub;
			newItem->RegReturn = oldItem->RegReturn;
			newItem->IsCorrelated = oldItem->IsCorrelated;
			newItem->ViaCoroutine = oldItem->ViaCoroutine;
			newItem->IndexName = SysEx::TagStrDup(ctx, oldItem->IndexName);
			newItem->NotIndexed = oldItem->NotIndexed;
			newItem->Index = oldItem->Index;
			Core::Table *table = newItem->Table = oldItem->Table;
			if (table)
				table->Refs++;
			newItem->Select = SelectDup(ctx, oldItem->Select, flags);
			newItem->On = ExprDup(db, oldItem->On, flags);
			newItem->Using = IdListDup(ctx, oldItem->Using);
			newItem->ColUsed = oldItem->ColUsed;
		}
		return newList;
	}
	__device__ IdList *Expr::IdListDup(Context *ctx, IdList *list)
	{
		if (!list)
			return 0;
		IdList *newList = (IdList *)SysEx::TagAlloc(ctx, sizeof(*newList));
		if (!newList)
			return 0;
		newList->Ids.length = list->Ids.length;
		newList->Ids.data = (IdList::IdListItem *)SysEx::TagAlloc(ctx, list->Ids.length * sizeof(list->Ids[0]));
		if (!newList->Ids.data)
		{
			SysEx::TagFree(ctx, newList);
			return 0;
		}
		// Note that because the size of the allocation for p->a[] is not necessarily a power of two, sqlite3IdListAppend() may not be called
		// on the duplicate created by this function.
		for (int i = 0; i < p->Id; i++)
		{
			IdList::IdListItem *newItem = &newList->Ids[i];
			IdList::IdListItem *oldItem = &list->Ids[i];
			newItem->Name = SysEx::TagStrDup(ctx, oldItem->Name);
			newItem->Idx = oldItem->Idx;
		}
		return newList;
	}
	__device__ Select *Expr::SelectDup(Context *ctx, Select *select, int flags)
	{
		if (!select)
			return 0;
		Select *newSelect = (Select *)SysEx::TagAlloc(ctx, sizeof(*select));
		if (!newSelect)
			return 0;
		newSelect->EList = ExprListDup(ctx, select->EList, flags);
		newSelect->Src = SrcListDup(ctx, select->Src, flags);
		newSelect->Where = ExprDup(ctx, select->Where, flags);
		newSelect->GroupBy = ExprListDup(ctx, select->GroupBy, flags);
		newSelect->Having = ExprDup(ctx, select->Having, flags);
		newSelect->OrderBy = ExprListDup(ctx, select->OrderBy, flags);
		newSelect->OP = select->OP;
		Select *prior;
		newSelect->Prior = prior = SelectDup(ctx, select->Prior, flags);
		if (prior)
			prior->Next = newSelect;
		newSelect->Next = nullptr;
		newSelect->Limit = ExprDup(ctx, p->pLimit, flags);
		newSelect->Offset = ExprDup(ctx, p->pOffset, flags);
		newSelect->Limit = 0;
		newSelect->Offset = 0;
		newSelect->SelFlags = (SF)(select->SelFlags & ~SF_UsesEphemeral);
		newSelect->Rightmost = 0;
		newSelect->AddrOpenEphm[0] = -1;
		newSelect->AddrOpenEphm[1] = -1;
		newSelect->AddrOpenEphm[2] = -1;
		return newSelect;
	}
#else
	__device__ Select *Expr::SelectDup(Context *ctx, Select *select, int flags) { _assert(select); return 0; }
#endif

#pragma endregion

	__device__ ExprList *Expr::ExprListAppend(Context *ctx, ExprList *list, Expr *expr)
	{
		if (!list)
		{
			list = (ExprList *)SysEx::TagAlloc(ctx, sizeof(ExprList));
			if (!list)
				goto no_mem;
			list->Ids = (ExprList::ExprListItem *)SysEx::TagAlloc(ctx, sizeof(list->Ids[0]));
			if (!list->Ids)
				goto no_mem;
		}
		else if ((list->Exprs & (list->Exprs-1)) == 0)
		{
			_assert(list->Exprs > 0);
			ExprList::ExprListItem *ids = (ExprList::ExprListItem *)SysEx::TagRealloc(ctx, list->Ids, list->Exprs*2*sizeof(list->Ids[0]));
			if (!ids)
				goto no_mem;
			list->Ids = ids;
		}
		_assert(list->Ids);
		ExprList::ExprListItem *item = &list->Ids[list->Exprs++];
		_memset(item, 0, sizeof(*item));
		item->Expr = expr;
		return list;

no_mem:     
		// Avoid leaking memory if malloc has failed.
		ExprDelete(ctx, expr);
		ExprListDelete(ctx, list);
		return 0;
	}

	__device__ void Parse::ExprListSetName(ExprList *list, Token *name, bool dequote)
	{
		Context *ctx = Ctx;
		_assert(list || ctx->MallocFailed);
		if (list)
		{
			_assert(list->Exprs > 0);
			ExprList::ExprListItem *item = &list->Ids[list->Exprs-1];
			_assert(!item->Name);
			item->Name = SysEx::TagStrNDup(ctx, name->data, name->length);
			if (dequote && item->Name)
				sqlite3Dequote(item->Name);
		}
	}

	__device__ void Parse::ExprListSetSpan(ExprList *list, ExprSpan *span)
	{
		Context *ctx = Ctx;
		_assert(list || ctx->MallocFailed);
		if (list)
		{
			ExprList::ExprListItem *item = &list->Ids[list->Exprs-1];
			_assert(list->Exprs > 0);
			_assert(ctx->MallocFailed || item->Expr == span->Expr);
			SysEx::TagFree(ctx, item->Span);
			item->Span = SysEx::TagStrNDup(ctx, (char *)span->Start, (int)(span->End - span->Start));
		}
	}

	__device__ void Parse::ExprListCheckLength(ExprList *lList, const char *object)
	{
		int max = Ctx->Limits[LIMIT_COLUMN];
		ASSERTCOVERAGE(list && list->Exprs == max);
		ASSERTCOVERAGE(list && list->Exprs == max+1);
		if (list && list->Expr > max)
			ErrorMsg("too many columns in %s", object);
	}

	__device__ void Expr::ExprListDelete(Context *ctx, ExprList *list)
	{
		if (!list)
			return;
		_assert(list->Ids || list->Exprs == 0);
		int i;
		for (ExprList::ExprListItem *item = list->Ids, i = 0; i < list->Exprs; i++, item++)
		{
			ExprDelete(ctx, item->Expr);
			SysEx::TagFree(ctx, item->Name);
			SysEx::TagFree(ctx, item->Span);
		}
		SysEx::TagFree(ctx, list->Ids);
		SysEx::TagFree(ctx, list);
	}

#pragma region Walk an expression tree

	__device__ static WRC ExprNodeIsConstant(Walker *walker, Expr *expr)
	{
		// If pWalker->u.i is 3 then any term of the expression that comes from the ON or USING clauses of a join disqualifies the expression
		// from being considered constant. */
		if (walker->u.I == 3 && ExprHasAnyProperty(expr, EP_FromJoin))
		{
			walker->u.I = 0;
			return WRC_Abort;
		}
		switch (expr->OP)
		{
		case TK_FUNCTION:
			// Consider functions to be constant if all their arguments are constant and pWalker->u.i==2
			if (walker->u.I == 2)
				return WRC_Continue;
			// Fall through
		case TK_ID:
		case TK_COLUMN:
		case TK_AGG_FUNCTION:
		case TK_AGG_COLUMN:
			ASSERTCOVERAGE(expr->OP == TK_ID);
			ASSERTCOVERAGE(expr->OP == TK_COLUMN);
			ASSERTCOVERAGE(expr->OP == TK_AGG_FUNCTION);
			ASSERTCOVERAGE(expr->OP == TK_AGG_COLUMN);
			walker->u.I = 0;
			return WRC_Abort;
		default:
			ASSERTCOVERAGE(expr->OP == TK_SELECT); // selectNodeIsConstant will disallow
			ASSERTCOVERAGE(expr->OP == TK_EXISTS); // selectNodeIsConstant will disallow
			return WRC_Continue;
		}
	}

	__device__ static WRC SelectNodeIsConstant(Walker *walker, Select *notUsed)
	{
		walker->u.I = 0;
		return WRC_Abort;
	}

	__device__ static int ExprIsConst(Expr *expr, int initFlag)
	{
		Walker w;
		w.u.I = initFlag;
		w.ExprCallback = ExprNodeIsConstant;
		w.SelectCallback = SelectNodeIsConstant;
		w.Expr(expr);
		return w.u.I;
	}

	__device__ bool Expr::IsConstant() { return ExprIsConst(this, 1); }
	__device__ bool Expr::IsConstantNotJoin() { return ExprIsConst(this, 3); }
	__device__ bool Expr::IsConstantOrFunction() { return ExprIsConst(this, 2); }
	__device__ bool Expr::IsInteger(int *value)
	{
		// If an expression is an integer literal that fits in a signed 32-bit integer, then the EP_IntValue flag will have already been set
		int rc = 0;
		_assert(OP != TK_INTEGER || (Flags & EP_IntValue) != 0 || !ConvertEx::Atoi(u.Token, &rc));
		if (Flags & EP_IntValue)
		{
			*value = u.I;
			return true;
		}
		switch (OP)
		{
		case TK_UPLUS: return Left->IsInteger(value);
		case TK_UMINUS:
			int v;
			if (Left->IsInteger(&v))
			{
				*value = -v;
				return true;
			}
			return false;
		default: return false;
		}
	}

#pragma endregion

	__device__ bool Expr::CanBeNull()
	{
		Expr *expr = this;
		while (expr->OP == TK_UPLUS || expr->OP == TK_UMINUS) { expr = expr->Left; }
		uint8 op = expr->OP;
		if (op == TK_REGISTER)
			op = expr->OP2;
		switch (op)
		{
		case TK_INTEGER:
		case TK_STRING:
		case TK_FLOAT:
		case TK_BLOB: return false;
		default: return true;
		}
	}

	__device__ void Expr::CodeIsNullJump(Vdbe *v, const Expr *expr, int reg, int dest)
	{
		if (((Expr *)expr)->CanBeNull())
			v->AddOp2(OP_IsNull, reg, dest);
	}

	__device__ bool Expr::NeedsNoAffinityChange(AFF aff)
	{
		if (aff == AFF_NONE)
			return true;
		Expr *expr = this;
		while (expr->OP == TK_UPLUS || expr->OP == TK_UMINUS) { expr = expr->Left; }
		uint8 op = expr->OP;
		if (op == TK_REGISTER)
			op = expr->OP2;
		switch (op)
		{
		case TK_INTEGER: return aff == AFF_INTEGER || aff == AFF_NUMERIC;
		case TK_FLOAT: return aff == AFF_REAL || aff == AFF_NUMERIC;
		case TK_STRING: return aff == AFF_TEXT;
		case TK_BLOB: return true;
		case TK_COLUMN:
			_assert(TableIdx >= 0); // p cannot be part of a CHECK constraint
			return ColumnIdx < 0 && (aff == AFF_INTEGER || aff == AFF_NUMERIC);
		default: return false;
		}
	}

#pragma region SUBQUERY

	__device__ int Parse::CodeOnce()
	{
		Vdbe *v = GetVdbe(); // Virtual machine being coded
		return v->AddOp1(OP_Once, Onces++);
	}

#ifndef OMIT_SUBQUERY
	__device__ static bool IsCandidateForInOpt(Select *select)
	{
		if (select) return false;					// right-hand side of IN is SELECT
		if (select->Prior) return false;			// Not a compound SELECT
		if (select->SelFlags & (SF_Distinct|SF_Aggregate))
		{
			ASSERTCOVERAGE((select->SelFlags & (SF_Distinct|SF_Aggregate)) == SF_Distinct);
			ASSERTCOVERAGE((select->SelFlags & (SF_Distinct|SF_Aggregate)) == SF_Aggregate);
			return false; // No DISTINCT keyword and no aggregate functions
		}
		_assert(!select->GroupBy);					// Has no GROUP BY clause
		if (select->Limit) return false;			// Has no LIMIT clause
		_assert(select->Offset == false);               // No LIMIT means no OFFSET
		if (select->Where) return 0;				// Has no WHERE clause
		SrcList *src = select->Src;
		_assert(src);
		if (src->Srcs != 1) return false;			// Single term in FROM clause
		if (src->Ids[0].Select) return false;		// FROM is not a subquery or view
		Table *table = src->Ids[0].Table;
		if (SysEx_NEVER(table == nullptr)) return false;
		_assert(!table->Select);					// FROM clause is not a view
		if (IsVirtual(table)) return false;			// FROM clause not a virtual table
		ExprList *list = select->EList;
		if (list->Exprs != 1) return false;			// One column in the result set
		if (list->Ids[0].Expr->OP != TK_COLUMN) return false; // Result is a column
		return true;
	}

	__device__ IN_INDEX Parse::FindInIndex(Expr *expr, int *notFound)
	{
		_assert(expr->OP == TK_IN);
		IN_INDEX type = 0; // Type of RHS table. IN_INDEX_*
		int tableIdx = Tabs++; // Cursor of the RHS table
		bool mustBeUnique = (notFound == 0);   // True if RHS must be unique
		Vdbe *v = GetVdbe(); // Virtual machine being coded

		// Check to see if an existing table or index can be used to satisfy the query.  This is preferable to generating a new ephemeral table.
		Select *select = (ExprHasProperty(expr, EP_xIsSelect) ? expr->x.Select : nullptr); // SELECT to the right of IN operator
		if (SysEx_ALWAYS(Errs == 0) && IsCandidateForInOpt(select))
		{
			_assert(select); // Because of isCandidateForInOpt(p)
			_assert(select->EList); // Because of isCandidateForInOpt(p)
			_assert(select->EList->Ids[0].Expr); // Because of isCandidateForInOpt(p)
			_assert(select->Src); // Because of isCandidateForInOpt(p)

			Context *ctx = Ctx; // Database connection
			Table *table = select->Src->Ids[0].Table; // Table <table>.
			Expr *expr2 = select->EList->Ids[0].Expr; // Expression <column>
			int col = expr2->ColumnIdx; // Index of column <column>

			// Code an OP_VerifyCookie and OP_TableLock for <table>.
			int db = SchemaToIndex(ctx, table->Schema); // Database idx for pTab
			CodeVerifySchema(db);
			this->TableLock(db, table->Id, 0, table->Name);

			// This function is only called from two places. In both cases the vdbe has already been allocated. So assume sqlite3GetVdbe() is always
			// successful here.
			_assert(v);
			if (col < 0)
			{
				int addr = CodeOnce();
				OpenTable(tableIdx, db, table, OP_OpenRead);
				type = IN_INDEX_ROWID;
				v->JumpHere(addr);
			}
			else
			{
				// The collation sequence used by the comparison. If an index is to be used in place of a temp-table, it must be ordered according
				// to this collation sequence.  */
				CollSeq *req = BinaryCompareCollSeq(expr->Left, expr2);
				// Check that the affinity that will be used to perform the comparison is the same as the affinity of the column. If
				// it is not, it is not possible to use any index.
				bool validAffinity = expr->ValidIndexAffinity(table->Cols[col].Affinity);
				for (Index *index = table->Index; index && type == 0 && validAffinity; index = index->Next)
					if (index->Columns[0] == col && FindCollSeq(ctx, ENC(ctx), index->CollNames[0], 0) == req && (!mustBeUnique || (index->Columns.length == 1 && index->OnError != OE_None)))
					{
						char *key = (char *)IndexKeyinfo(index);
						int addr = CodeOnce();
						v->AddOp4(OP_OpenRead, tableIdx, index->Id, db, key, Vdbe::P4T_KEYINFO_HANDOFF);
						v->VdbeComment("%s", index->Name);
						_assert(IN_INDEX_INDEX_DESC == IN_INDEX_INDEX_ASC+1);
						type = (IN_INDEX)IN_INDEX_INDEX_ASC + index->SortOrders[0];
						v->JumpHere(addr);
						if (notFound && !table->Cols[col].NotNull)
						{
							*notFound = ++Mems;
							v->AddOp2(OP_Null, 0, *notFound);
						}
					}
			}
		}

		if (type == 0)
		{
			// Could not found an existing table or index to use as the RHS b-tree. We will have to generate an ephemeral table to do the job.
			double savedQueryLoops = QueryLoops;
			int mayHaveNull = 0;
			type = IN_INDEX_EPH;
			if (notFound)
			{
				*notFound = mayHaveNull = ++Mems;
				v->AddOp2(OP_Null, 0, *notFound);
			}
			else
			{
				ASSERTCOVERAGE(QueryLoops > (double)1);
				QueryLoops = (double)1;
				if (expr->Left->ColumnIdx < 0 && !ExprHasAnyProperty(expr, EP_xIsSelect))
					type = IN_INDEX_ROWID;
			}
			CodeSubselect(expr, mayHaveNull, type == IN_INDEX_ROWID);
			QueryLoops = savedQueryLoops;
		}
		else
			expr->TableIdx = tableIdx;
		return type;
	}

	__device__ int Parse::CodeSubselect(Expr *expr, int mayHaveNull, bool isRowid)
	{
			int reg = 0; // Register storing resulting
			Vdbe *v = GetVdbe();
			if (SysEx_NEVER(!v))
				return 0;
			ExprCachePush();

			// This code must be run in its entirety every time it is encountered if any of the following is true:
			//    *  The right-hand side is a correlated subquery
			//    *  The right-hand side is an expression list containing variables
			//    *  We are inside a trigger
			// If all of the above are false, then we can run this code just once save the results, and reuse the same result on subsequent invocations.
			int testAddr = (!ExprHasAnyProperty(expr, EP_VarSelect) ? CodeOnce() : -1); // One-time test address

#ifndef OMIT_EXPLAIN
			if (Explain == 2)
			{
				char *msg = SysEx::Mprintf(Ctx, "EXECUTE %s%s SUBQUERY %d", (testAddr >= 0 ? "" : "CORRELATED "), (expr->OP == TK_IN ? "LIST" : "SCALAR"), NextSelectId);
				v->AddOp4(OP_Explain, SelectId, 0, 0, msg, Vdbe::P4T_DYNAMIC);
			}
#endif

			switch (expr->OP)
			{
			case TK_IN: {
				AFF affinity;              // Affinity of the LHS of the IN
				KeyInfo keyInfo;            // Keyinfo for the generated table
				static u8 sortOrder = 0;    // Fake aSortOrder for keyInfo
				int addr;                   // Address of OP_OpenEphemeral instruction
				Expr *pLeft = pExpr->pLeft; // the LHS of the IN operator

				if( rMayHaveNull ){
					sqlite3VdbeAddOp2(v, OP_Null, 0, rMayHaveNull);
				}

				affinity = sqlite3ExprAffinity(pLeft);

				/* Whether this is an 'x IN(SELECT...)' or an 'x IN(<exprlist>)'
				** expression it is handled the same way.  An ephemeral table is 
				** filled with single-field index keys representing the results
				** from the SELECT or the <exprlist>.
				**
				** If the 'x' expression is a column value, or the SELECT...
				** statement returns a column value, then the affinity of that
				** column is used to build the index keys. If both 'x' and the
				** SELECT... statement are columns, then numeric affinity is used
				** if either column has NUMERIC or INTEGER affinity. If neither
				** 'x' nor the SELECT... statement are columns, then numeric affinity
				** is used.
				*/
				pExpr->iTable = pParse->nTab++;
				addr = sqlite3VdbeAddOp2(v, OP_OpenEphemeral, pExpr->iTable, !isRowid);
				if( rMayHaveNull==0 ) sqlite3VdbeChangeP5(v, BTREE_UNORDERED);
				memset(&keyInfo, 0, sizeof(keyInfo));
				keyInfo.nField = 1;
				keyInfo.aSortOrder = &sortOrder;

				if( ExprHasProperty(pExpr, EP_xIsSelect) ){
					/* Case 1:     expr IN (SELECT ...)
					**
					** Generate code to write the results of the select into the temporary
					** table allocated and opened above.
					*/
					SelectDest dest;
					ExprList *pEList;

					assert( !isRowid );
					sqlite3SelectDestInit(&dest, SRT_Set, pExpr->iTable);
					dest.affSdst = (u8)affinity;
					assert( (pExpr->iTable&0x0000FFFF)==pExpr->iTable );
					pExpr->x.pSelect->iLimit = 0;
					if( sqlite3Select(pParse, pExpr->x.pSelect, &dest) ){
						return 0;
					}
					pEList = pExpr->x.pSelect->pEList;
					if( ALWAYS(pEList!=0 && pEList->nExpr>0) ){ 
						keyInfo.aColl[0] = sqlite3BinaryCompareCollSeq(pParse, pExpr->pLeft,
							pEList->a[0].pExpr);
					}
				}else if( ALWAYS(pExpr->x.pList!=0) ){
					/* Case 2:     expr IN (exprlist)
					**
					** For each expression, build an index key from the evaluation and
					** store it in the temporary table. If <expr> is a column, then use
					** that columns affinity when building index keys. If <expr> is not
					** a column, use numeric affinity.
					*/
					int i;
					ExprList *pList = pExpr->x.pList;
					struct ExprList_item *pItem;
					int r1, r2, r3;

					if( !affinity ){
						affinity = SQLITE_AFF_NONE;
					}
					keyInfo.aColl[0] = sqlite3ExprCollSeq(pParse, pExpr->pLeft);
					keyInfo.aSortOrder = &sortOrder;

					/* Loop through each expression in <exprlist>. */
					r1 = sqlite3GetTempReg(pParse);
					r2 = sqlite3GetTempReg(pParse);
					sqlite3VdbeAddOp2(v, OP_Null, 0, r2);
					for(i=pList->nExpr, pItem=pList->a; i>0; i--, pItem++){
						Expr *pE2 = pItem->pExpr;
						int iValToIns;

						/* If the expression is not constant then we will need to
						** disable the test that was generated above that makes sure
						** this code only executes once.  Because for a non-constant
						** expression we need to rerun this code each time.
						*/
						if( testAddr>=0 && !sqlite3ExprIsConstant(pE2) ){
							sqlite3VdbeChangeToNoop(v, testAddr);
							testAddr = -1;
						}

						/* Evaluate the expression and insert it into the temp table */
						if( isRowid && sqlite3ExprIsInteger(pE2, &iValToIns) ){
							sqlite3VdbeAddOp3(v, OP_InsertInt, pExpr->iTable, r2, iValToIns);
						}else{
							r3 = sqlite3ExprCodeTarget(pParse, pE2, r1);
							if( isRowid ){
								sqlite3VdbeAddOp2(v, OP_MustBeInt, r3,
									sqlite3VdbeCurrentAddr(v)+2);
								sqlite3VdbeAddOp3(v, OP_Insert, pExpr->iTable, r2, r3);
							}else{
								sqlite3VdbeAddOp4(v, OP_MakeRecord, r3, 1, r2, &affinity, 1);
								sqlite3ExprCacheAffinityChange(pParse, r3, 1);
								sqlite3VdbeAddOp2(v, OP_IdxInsert, pExpr->iTable, r2);
							}
						}
					}
					sqlite3ReleaseTempReg(pParse, r1);
					sqlite3ReleaseTempReg(pParse, r2);
				}
				if( !isRowid ){
					sqlite3VdbeChangeP4(v, addr, (void *)&keyInfo, P4_KEYINFO);
				}
				break;
						}

			case TK_EXISTS:
			case TK_SELECT:
			default: {
				/* If this has to be a scalar SELECT.  Generate code to put the
				** value of this select in a memory cell and record the number
				** of the memory cell in iColumn.  If this is an EXISTS, write
				** an integer 0 (not exists) or 1 (exists) into a memory cell
				** and record that memory cell in iColumn.
				*/
				Select *pSel;                         /* SELECT statement to encode */
				SelectDest dest;                      /* How to deal with SELECt result */

				testcase( pExpr->op==TK_EXISTS );
				testcase( pExpr->op==TK_SELECT );
				assert( pExpr->op==TK_EXISTS || pExpr->op==TK_SELECT );

				assert( ExprHasProperty(pExpr, EP_xIsSelect) );
				pSel = pExpr->x.pSelect;
				sqlite3SelectDestInit(&dest, 0, ++pParse->nMem);
				if( pExpr->op==TK_SELECT ){
					dest.eDest = SRT_Mem;
					sqlite3VdbeAddOp2(v, OP_Null, 0, dest.iSDParm);
					VdbeComment((v, "Init subquery result"));
				}else{
					dest.eDest = SRT_Exists;
					sqlite3VdbeAddOp2(v, OP_Integer, 0, dest.iSDParm);
					VdbeComment((v, "Init EXISTS result"));
				}
				sqlite3ExprDelete(pParse->db, pSel->pLimit);
				pSel->pLimit = sqlite3PExpr(pParse, TK_INTEGER, 0, 0,
					&sqlite3IntTokens[1]);
				pSel->iLimit = 0;
				if( sqlite3Select(pParse, pSel, &dest) ){
					return 0;
				}
				rReg = dest.iSDParm;
				ExprSetIrreducible(pExpr);
				break;
					 }
			}

			if( testAddr>=0 ){
				sqlite3VdbeJumpHere(v, testAddr);
			}
			sqlite3ExprCachePop(pParse, 1);

			return rReg;
	}

	static void sqlite3ExprCodeIN(
		Parse *pParse,        /* Parsing and code generating context */
		Expr *pExpr,          /* The IN expression */
		int destIfFalse,      /* Jump here if LHS is not contained in the RHS */
		int destIfNull        /* Jump here if the results are unknown due to NULLs */
		){
			int rRhsHasNull = 0;  /* Register that is true if RHS contains NULL values */
			char affinity;        /* Comparison affinity to use */
			int eType;            /* Type of the RHS */
			int r1;               /* Temporary use register */
			Vdbe *v;              /* Statement under construction */

			/* Compute the RHS.   After this step, the table with cursor
			** pExpr->iTable will contains the values that make up the RHS.
			*/
			v = pParse->pVdbe;
			assert( v!=0 );       /* OOM detected prior to this routine */
			VdbeNoopComment((v, "begin IN expr"));
			eType = sqlite3FindInIndex(pParse, pExpr, &rRhsHasNull);

			/* Figure out the affinity to use to create a key from the results
			** of the expression. affinityStr stores a static string suitable for
			** P4 of OP_MakeRecord.
			*/
			affinity = comparisonAffinity(pExpr);

			/* Code the LHS, the <expr> from "<expr> IN (...)".
			*/
			sqlite3ExprCachePush(pParse);
			r1 = sqlite3GetTempReg(pParse);
			sqlite3ExprCode(pParse, pExpr->pLeft, r1);

			/* If the LHS is NULL, then the result is either false or NULL depending
			** on whether the RHS is empty or not, respectively.
			*/
			if( destIfNull==destIfFalse ){
				/* Shortcut for the common case where the false and NULL outcomes are
				** the same. */
				sqlite3VdbeAddOp2(v, OP_IsNull, r1, destIfNull);
			}else{
				int addr1 = sqlite3VdbeAddOp1(v, OP_NotNull, r1);
				sqlite3VdbeAddOp2(v, OP_Rewind, pExpr->iTable, destIfFalse);
				sqlite3VdbeAddOp2(v, OP_Goto, 0, destIfNull);
				sqlite3VdbeJumpHere(v, addr1);
			}

			if( eType==IN_INDEX_ROWID ){
				/* In this case, the RHS is the ROWID of table b-tree
				*/
				sqlite3VdbeAddOp2(v, OP_MustBeInt, r1, destIfFalse);
				sqlite3VdbeAddOp3(v, OP_NotExists, pExpr->iTable, destIfFalse, r1);
			}else{
				/* In this case, the RHS is an index b-tree.
				*/
				sqlite3VdbeAddOp4(v, OP_Affinity, r1, 1, 0, &affinity, 1);

				/* If the set membership test fails, then the result of the 
				** "x IN (...)" expression must be either 0 or NULL. If the set
				** contains no NULL values, then the result is 0. If the set 
				** contains one or more NULL values, then the result of the
				** expression is also NULL.
				*/
				if( rRhsHasNull==0 || destIfFalse==destIfNull ){
					/* This branch runs if it is known at compile time that the RHS
					** cannot contain NULL values. This happens as the result
					** of a "NOT NULL" constraint in the database schema.
					**
					** Also run this branch if NULL is equivalent to FALSE
					** for this particular IN operator.
					*/
					sqlite3VdbeAddOp4Int(v, OP_NotFound, pExpr->iTable, destIfFalse, r1, 1);

				}else{
					/* In this branch, the RHS of the IN might contain a NULL and
					** the presence of a NULL on the RHS makes a difference in the
					** outcome.
					*/
					int j1, j2, j3;

					/* First check to see if the LHS is contained in the RHS.  If so,
					** then the presence of NULLs in the RHS does not matter, so jump
					** over all of the code that follows.
					*/
					j1 = sqlite3VdbeAddOp4Int(v, OP_Found, pExpr->iTable, 0, r1, 1);

					/* Here we begin generating code that runs if the LHS is not
					** contained within the RHS.  Generate additional code that
					** tests the RHS for NULLs.  If the RHS contains a NULL then
					** jump to destIfNull.  If there are no NULLs in the RHS then
					** jump to destIfFalse.
					*/
					j2 = sqlite3VdbeAddOp1(v, OP_NotNull, rRhsHasNull);
					j3 = sqlite3VdbeAddOp4Int(v, OP_Found, pExpr->iTable, 0, rRhsHasNull, 1);
					sqlite3VdbeAddOp2(v, OP_Integer, -1, rRhsHasNull);
					sqlite3VdbeJumpHere(v, j3);
					sqlite3VdbeAddOp2(v, OP_AddImm, rRhsHasNull, 1);
					sqlite3VdbeJumpHere(v, j2);

					/* Jump to the appropriate target depending on whether or not
					** the RHS contains a NULL
					*/
					sqlite3VdbeAddOp2(v, OP_If, rRhsHasNull, destIfNull);
					sqlite3VdbeAddOp2(v, OP_Goto, 0, destIfFalse);

					/* The OP_Found at the top of this branch jumps here when true, 
					** causing the overall IN expression evaluation to fall through.
					*/
					sqlite3VdbeJumpHere(v, j1);
				}
			}
			sqlite3ReleaseTempReg(pParse, r1);
			sqlite3ExprCachePop(pParse, 1);
			VdbeComment((v, "end IN expr"));
	}

#endif
#pragma endregion

	static char *dup8bytes(Vdbe *v, const char *in)
	{
		char *out = sqlite3DbMallocRaw(sqlite3VdbeDb(v), 8);
		if( out ){
			memcpy(out, in, 8);
		}
		return out;
	}

#ifndef SQLITE_OMIT_FLOATING_POINT
	/*
	** Generate an instruction that will put the floating point
	** value described by z[0..n-1] into register iMem.
	**
	** The z[] string will probably not be zero-terminated.  But the 
	** z[n] character is guaranteed to be something that does not look
	** like the continuation of the number.
	*/
	static void codeReal(Vdbe *v, const char *z, int negateFlag, int iMem){
		if( ALWAYS(z!=0) ){
			double value;
			char *zV;
			sqlite3AtoF(z, &value, sqlite3Strlen30(z), SQLITE_UTF8);
			assert( !sqlite3IsNaN(value) ); /* The new AtoF never returns NaN */
			if( negateFlag ) value = -value;
			zV = dup8bytes(v, (char*)&value);
			sqlite3VdbeAddOp4(v, OP_Real, 0, iMem, 0, zV, P4_REAL);
		}
	}
#endif


	/*
	** Generate an instruction that will put the integer describe by
	** text z[0..n-1] into register iMem.
	**
	** Expr.u.zToken is always UTF8 and zero-terminated.
	*/
	static void codeInteger(Parse *pParse, Expr *pExpr, int negFlag, int iMem){
		Vdbe *v = pParse->pVdbe;
		if( pExpr->flags & EP_IntValue ){
			int i = pExpr->u.iValue;
			assert( i>=0 );
			if( negFlag ) i = -i;
			sqlite3VdbeAddOp2(v, OP_Integer, i, iMem);
		}else{
			int c;
			i64 value;
			const char *z = pExpr->u.zToken;
			assert( z!=0 );
			c = sqlite3Atoi64(z, &value, sqlite3Strlen30(z), SQLITE_UTF8);
			if( c==0 || (c==2 && negFlag) ){
				char *zV;
				if( negFlag ){ value = c==2 ? SMALLEST_INT64 : -value; }
				zV = dup8bytes(v, (char*)&value);
				sqlite3VdbeAddOp4(v, OP_Int64, 0, iMem, 0, zV, P4_INT64);
			}else{
#ifdef SQLITE_OMIT_FLOATING_POINT
				sqlite3ErrorMsg(pParse, "oversized integer: %s%s", negFlag ? "-" : "", z);
#else
				codeReal(v, z, negFlag, iMem);
#endif
			}
		}
	}

	/*
	** Clear a cache entry.
	*/
	static void cacheEntryClear(Parse *pParse, struct yColCache *p){
		if( p->tempReg ){
			if( pParse->nTempReg<ArraySize(pParse->aTempReg) ){
				pParse->aTempReg[pParse->nTempReg++] = p->iReg;
			}
			p->tempReg = 0;
		}
	}


	/*
	** Record in the column cache that a particular column from a
	** particular table is stored in a particular register.
	*/
	void sqlite3ExprCacheStore(Parse *pParse, int iTab, int iCol, int iReg){
		int i;
		int minLru;
		int idxLru;
		struct yColCache *p;

		assert( iReg>0 );  /* Register numbers are always positive */
		assert( iCol>=-1 && iCol<32768 );  /* Finite column numbers */

		/* The SQLITE_ColumnCache flag disables the column cache.  This is used
		** for testing only - to verify that SQLite always gets the same answer
		** with and without the column cache.
		*/
		if( OptimizationDisabled(pParse->db, SQLITE_ColumnCache) ) return;

		/* First replace any existing entry.
		**
		** Actually, the way the column cache is currently used, we are guaranteed
		** that the object will never already be in cache.  Verify this guarantee.
		*/
#ifndef NDEBUG
		for(i=0, p=pParse->aColCache; i<SQLITE_N_COLCACHE; i++, p++){
			assert( p->iReg==0 || p->iTable!=iTab || p->iColumn!=iCol );
		}
#endif

		/* Find an empty slot and replace it */
		for(i=0, p=pParse->aColCache; i<SQLITE_N_COLCACHE; i++, p++){
			if( p->iReg==0 ){
				p->iLevel = pParse->iCacheLevel;
				p->iTable = iTab;
				p->iColumn = iCol;
				p->iReg = iReg;
				p->tempReg = 0;
				p->lru = pParse->iCacheCnt++;
				return;
			}
		}

		/* Replace the last recently used */
		minLru = 0x7fffffff;
		idxLru = -1;
		for(i=0, p=pParse->aColCache; i<SQLITE_N_COLCACHE; i++, p++){
			if( p->lru<minLru ){
				idxLru = i;
				minLru = p->lru;
			}
		}
		if( ALWAYS(idxLru>=0) ){
			p = &pParse->aColCache[idxLru];
			p->iLevel = pParse->iCacheLevel;
			p->iTable = iTab;
			p->iColumn = iCol;
			p->iReg = iReg;
			p->tempReg = 0;
			p->lru = pParse->iCacheCnt++;
			return;
		}
	}

	/*
	** Indicate that registers between iReg..iReg+nReg-1 are being overwritten.
	** Purge the range of registers from the column cache.
	*/
	void sqlite3ExprCacheRemove(Parse *pParse, int iReg, int nReg){
		int i;
		int iLast = iReg + nReg - 1;
		struct yColCache *p;
		for(i=0, p=pParse->aColCache; i<SQLITE_N_COLCACHE; i++, p++){
			int r = p->iReg;
			if( r>=iReg && r<=iLast ){
				cacheEntryClear(pParse, p);
				p->iReg = 0;
			}
		}
	}

	/*
	** Remember the current column cache context.  Any new entries added
	** added to the column cache after this call are removed when the
	** corresponding pop occurs.
	*/
	void sqlite3ExprCachePush(Parse *pParse){
		pParse->iCacheLevel++;
	}

	/*
	** Remove from the column cache any entries that were added since the
	** the previous N Push operations.  In other words, restore the cache
	** to the state it was in N Pushes ago.
	*/
	void sqlite3ExprCachePop(Parse *pParse, int N){
		int i;
		struct yColCache *p;
		assert( N>0 );
		assert( pParse->iCacheLevel>=N );
		pParse->iCacheLevel -= N;
		for(i=0, p=pParse->aColCache; i<SQLITE_N_COLCACHE; i++, p++){
			if( p->iReg && p->iLevel>pParse->iCacheLevel ){
				cacheEntryClear(pParse, p);
				p->iReg = 0;
			}
		}
	}

	/*
	** When a cached column is reused, make sure that its register is
	** no longer available as a temp register.  ticket #3879:  that same
	** register might be in the cache in multiple places, so be sure to
	** get them all.
	*/
	static void sqlite3ExprCachePinRegister(Parse *pParse, int iReg){
		int i;
		struct yColCache *p;
		for(i=0, p=pParse->aColCache; i<SQLITE_N_COLCACHE; i++, p++){
			if( p->iReg==iReg ){
				p->tempReg = 0;
			}
		}
	}

	/*
	** Generate code to extract the value of the iCol-th column of a table.
	*/
	void sqlite3ExprCodeGetColumnOfTable(
		Vdbe *v,        /* The VDBE under construction */
		Table *pTab,    /* The table containing the value */
		int iTabCur,    /* The cursor for this table */
		int iCol,       /* Index of the column to extract */
		int regOut      /* Extract the valud into this register */
		){
			if( iCol<0 || iCol==pTab->iPKey ){
				sqlite3VdbeAddOp2(v, OP_Rowid, iTabCur, regOut);
			}else{
				int op = IsVirtual(pTab) ? OP_VColumn : OP_Column;
				sqlite3VdbeAddOp3(v, op, iTabCur, iCol, regOut);
			}
			if( iCol>=0 ){
				sqlite3ColumnDefault(v, pTab, iCol, regOut);
			}
	}

	/*
	** Generate code that will extract the iColumn-th column from
	** table pTab and store the column value in a register.  An effort
	** is made to store the column value in register iReg, but this is
	** not guaranteed.  The location of the column value is returned.
	**
	** There must be an open cursor to pTab in iTable when this routine
	** is called.  If iColumn<0 then code is generated that extracts the rowid.
	*/
	int sqlite3ExprCodeGetColumn(
		Parse *pParse,   /* Parsing and code generating context */
		Table *pTab,     /* Description of the table we are reading from */
		int iColumn,     /* Index of the table column */
		int iTable,      /* The cursor pointing to the table */
		int iReg,        /* Store results here */
		u8 p5            /* P5 value for OP_Column */
		){
			Vdbe *v = pParse->pVdbe;
			int i;
			struct yColCache *p;

			for(i=0, p=pParse->aColCache; i<SQLITE_N_COLCACHE; i++, p++){
				if( p->iReg>0 && p->iTable==iTable && p->iColumn==iColumn ){
					p->lru = pParse->iCacheCnt++;
					sqlite3ExprCachePinRegister(pParse, p->iReg);
					return p->iReg;
				}
			}  
			assert( v!=0 );
			sqlite3ExprCodeGetColumnOfTable(v, pTab, iTable, iColumn, iReg);
			if( p5 ){
				sqlite3VdbeChangeP5(v, p5);
			}else{   
				sqlite3ExprCacheStore(pParse, iTable, iColumn, iReg);
			}
			return iReg;
	}

	/*
	** Clear all column cache entries.
	*/
	void sqlite3ExprCacheClear(Parse *pParse){
		int i;
		struct yColCache *p;

		for(i=0, p=pParse->aColCache; i<SQLITE_N_COLCACHE; i++, p++){
			if( p->iReg ){
				cacheEntryClear(pParse, p);
				p->iReg = 0;
			}
		}
	}

	/*
	** Record the fact that an affinity change has occurred on iCount
	** registers starting with iStart.
	*/
	void sqlite3ExprCacheAffinityChange(Parse *pParse, int iStart, int iCount){
		sqlite3ExprCacheRemove(pParse, iStart, iCount);
	}

	/*
	** Generate code to move content from registers iFrom...iFrom+nReg-1
	** over to iTo..iTo+nReg-1. Keep the column cache up-to-date.
	*/
	void sqlite3ExprCodeMove(Parse *pParse, int iFrom, int iTo, int nReg){
		int i;
		struct yColCache *p;
		assert( iFrom>=iTo+nReg || iFrom+nReg<=iTo );
		sqlite3VdbeAddOp3(pParse->pVdbe, OP_Move, iFrom, iTo, nReg-1);
		for(i=0, p=pParse->aColCache; i<SQLITE_N_COLCACHE; i++, p++){
			int x = p->iReg;
			if( x>=iFrom && x<iFrom+nReg ){
				p->iReg += iTo-iFrom;
			}
		}
	}

#if defined(SQLITE_DEBUG) || defined(SQLITE_COVERAGE_TEST)
	/*
	** Return true if any register in the range iFrom..iTo (inclusive)
	** is used as part of the column cache.
	**
	** This routine is used within assert() and testcase() macros only
	** and does not appear in a normal build.
	*/
	static int usedAsColumnCache(Parse *pParse, int iFrom, int iTo){
		int i;
		struct yColCache *p;
		for(i=0, p=pParse->aColCache; i<SQLITE_N_COLCACHE; i++, p++){
			int r = p->iReg;
			if( r>=iFrom && r<=iTo ) return 1;    /*NO_TEST*/
		}
		return 0;
	}
#endif /* SQLITE_DEBUG || SQLITE_COVERAGE_TEST */

	/*
	** Generate code into the current Vdbe to evaluate the given
	** expression.  Attempt to store the results in register "target".
	** Return the register where results are stored.
	**
	** With this routine, there is no guarantee that results will
	** be stored in target.  The result might be stored in some other
	** register if it is convenient to do so.  The calling function
	** must check the return code and move the results to the desired
	** register.
	*/
	int sqlite3ExprCodeTarget(Parse *pParse, Expr *pExpr, int target){
		Vdbe *v = pParse->pVdbe;  /* The VM under construction */
		int op;                   /* The opcode being coded */
		int inReg = target;       /* Results stored in register inReg */
		int regFree1 = 0;         /* If non-zero free this temporary register */
		int regFree2 = 0;         /* If non-zero free this temporary register */
		int r1, r2, r3, r4;       /* Various register numbers */
		sqlite3 *db = pParse->db; /* The database connection */

		assert( target>0 && target<=pParse->nMem );
		if( v==0 ){
			assert( pParse->db->mallocFailed );
			return 0;
		}

		if( pExpr==0 ){
			op = TK_NULL;
		}else{
			op = pExpr->op;
		}
		switch( op ){
		case TK_AGG_COLUMN: {
			AggInfo *pAggInfo = pExpr->pAggInfo;
			struct AggInfo_col *pCol = &pAggInfo->aCol[pExpr->iAgg];
			if( !pAggInfo->directMode ){
				assert( pCol->iMem>0 );
				inReg = pCol->iMem;
				break;
			}else if( pAggInfo->useSortingIdx ){
				sqlite3VdbeAddOp3(v, OP_Column, pAggInfo->sortingIdxPTab,
					pCol->iSorterColumn, target);
				break;
			}
			/* Otherwise, fall thru into the TK_COLUMN case */
							}
		case TK_COLUMN: {
			if( pExpr->iTable<0 ){
				/* This only happens when coding check constraints */
				assert( pParse->ckBase>0 );
				inReg = pExpr->iColumn + pParse->ckBase;
			}else{
				inReg = sqlite3ExprCodeGetColumn(pParse, pExpr->pTab,
					pExpr->iColumn, pExpr->iTable, target,
					pExpr->op2);
			}
			break;
						}
		case TK_INTEGER: {
			codeInteger(pParse, pExpr, 0, target);
			break;
						 }
#ifndef SQLITE_OMIT_FLOATING_POINT
		case TK_FLOAT: {
			assert( !ExprHasProperty(pExpr, EP_IntValue) );
			codeReal(v, pExpr->u.zToken, 0, target);
			break;
					   }
#endif
		case TK_STRING: {
			assert( !ExprHasProperty(pExpr, EP_IntValue) );
			sqlite3VdbeAddOp4(v, OP_String8, 0, target, 0, pExpr->u.zToken, 0);
			break;
						}
		case TK_NULL: {
			sqlite3VdbeAddOp2(v, OP_Null, 0, target);
			break;
					  }
#ifndef SQLITE_OMIT_BLOB_LITERAL
		case TK_BLOB: {
			int n;
			const char *z;
			char *zBlob;
			assert( !ExprHasProperty(pExpr, EP_IntValue) );
			assert( pExpr->u.zToken[0]=='x' || pExpr->u.zToken[0]=='X' );
			assert( pExpr->u.zToken[1]=='\'' );
			z = &pExpr->u.zToken[2];
			n = sqlite3Strlen30(z) - 1;
			assert( z[n]=='\'' );
			zBlob = sqlite3HexToBlob(sqlite3VdbeDb(v), z, n);
			sqlite3VdbeAddOp4(v, OP_Blob, n/2, target, 0, zBlob, P4_DYNAMIC);
			break;
					  }
#endif
		case TK_VARIABLE: {
			assert( !ExprHasProperty(pExpr, EP_IntValue) );
			assert( pExpr->u.zToken!=0 );
			assert( pExpr->u.zToken[0]!=0 );
			sqlite3VdbeAddOp2(v, OP_Variable, pExpr->iColumn, target);
			if( pExpr->u.zToken[1]!=0 ){
				assert( pExpr->u.zToken[0]=='?' 
					|| strcmp(pExpr->u.zToken, pParse->azVar[pExpr->iColumn-1])==0 );
				sqlite3VdbeChangeP4(v, -1, pParse->azVar[pExpr->iColumn-1], P4_STATIC);
			}
			break;
						  }
		case TK_REGISTER: {
			inReg = pExpr->iTable;
			break;
						  }
		case TK_AS: {
			inReg = sqlite3ExprCodeTarget(pParse, pExpr->pLeft, target);
			break;
					}
#ifndef SQLITE_OMIT_CAST
		case TK_CAST: {
			/* Expressions of the form:   CAST(pLeft AS token) */
			int aff, to_op;
			inReg = sqlite3ExprCodeTarget(pParse, pExpr->pLeft, target);
			assert( !ExprHasProperty(pExpr, EP_IntValue) );
			aff = sqlite3AffinityType(pExpr->u.zToken);
			to_op = aff - SQLITE_AFF_TEXT + OP_ToText;
			assert( to_op==OP_ToText    || aff!=SQLITE_AFF_TEXT    );
			assert( to_op==OP_ToBlob    || aff!=SQLITE_AFF_NONE    );
			assert( to_op==OP_ToNumeric || aff!=SQLITE_AFF_NUMERIC );
			assert( to_op==OP_ToInt     || aff!=SQLITE_AFF_INTEGER );
			assert( to_op==OP_ToReal    || aff!=SQLITE_AFF_REAL    );
			testcase( to_op==OP_ToText );
			testcase( to_op==OP_ToBlob );
			testcase( to_op==OP_ToNumeric );
			testcase( to_op==OP_ToInt );
			testcase( to_op==OP_ToReal );
			if( inReg!=target ){
				sqlite3VdbeAddOp2(v, OP_SCopy, inReg, target);
				inReg = target;
			}
			sqlite3VdbeAddOp1(v, to_op, inReg);
			testcase( usedAsColumnCache(pParse, inReg, inReg) );
			sqlite3ExprCacheAffinityChange(pParse, inReg, 1);
			break;
					  }
#endif /* SQLITE_OMIT_CAST */
		case TK_LT:
		case TK_LE:
		case TK_GT:
		case TK_GE:
		case TK_NE:
		case TK_EQ: {
			assert( TK_LT==OP_Lt );
			assert( TK_LE==OP_Le );
			assert( TK_GT==OP_Gt );
			assert( TK_GE==OP_Ge );
			assert( TK_EQ==OP_Eq );
			assert( TK_NE==OP_Ne );
			testcase( op==TK_LT );
			testcase( op==TK_LE );
			testcase( op==TK_GT );
			testcase( op==TK_GE );
			testcase( op==TK_EQ );
			testcase( op==TK_NE );
			r1 = sqlite3ExprCodeTemp(pParse, pExpr->pLeft, &regFree1);
			r2 = sqlite3ExprCodeTemp(pParse, pExpr->pRight, &regFree2);
			codeCompare(pParse, pExpr->pLeft, pExpr->pRight, op,
				r1, r2, inReg, SQLITE_STOREP2);
			testcase( regFree1==0 );
			testcase( regFree2==0 );
			break;
					}
		case TK_IS:
		case TK_ISNOT: {
			testcase( op==TK_IS );
			testcase( op==TK_ISNOT );
			r1 = sqlite3ExprCodeTemp(pParse, pExpr->pLeft, &regFree1);
			r2 = sqlite3ExprCodeTemp(pParse, pExpr->pRight, &regFree2);
			op = (op==TK_IS) ? TK_EQ : TK_NE;
			codeCompare(pParse, pExpr->pLeft, pExpr->pRight, op,
				r1, r2, inReg, SQLITE_STOREP2 | SQLITE_NULLEQ);
			testcase( regFree1==0 );
			testcase( regFree2==0 );
			break;
					   }
		case TK_AND:
		case TK_OR:
		case TK_PLUS:
		case TK_STAR:
		case TK_MINUS:
		case TK_REM:
		case TK_BITAND:
		case TK_BITOR:
		case TK_SLASH:
		case TK_LSHIFT:
		case TK_RSHIFT: 
		case TK_CONCAT: {
			assert( TK_AND==OP_And );
			assert( TK_OR==OP_Or );
			assert( TK_PLUS==OP_Add );
			assert( TK_MINUS==OP_Subtract );
			assert( TK_REM==OP_Remainder );
			assert( TK_BITAND==OP_BitAnd );
			assert( TK_BITOR==OP_BitOr );
			assert( TK_SLASH==OP_Divide );
			assert( TK_LSHIFT==OP_ShiftLeft );
			assert( TK_RSHIFT==OP_ShiftRight );
			assert( TK_CONCAT==OP_Concat );
			testcase( op==TK_AND );
			testcase( op==TK_OR );
			testcase( op==TK_PLUS );
			testcase( op==TK_MINUS );
			testcase( op==TK_REM );
			testcase( op==TK_BITAND );
			testcase( op==TK_BITOR );
			testcase( op==TK_SLASH );
			testcase( op==TK_LSHIFT );
			testcase( op==TK_RSHIFT );
			testcase( op==TK_CONCAT );
			r1 = sqlite3ExprCodeTemp(pParse, pExpr->pLeft, &regFree1);
			r2 = sqlite3ExprCodeTemp(pParse, pExpr->pRight, &regFree2);
			sqlite3VdbeAddOp3(v, op, r2, r1, target);
			testcase( regFree1==0 );
			testcase( regFree2==0 );
			break;
						}
		case TK_UMINUS: {
			Expr *pLeft = pExpr->pLeft;
			assert( pLeft );
			if( pLeft->op==TK_INTEGER ){
				codeInteger(pParse, pLeft, 1, target);
#ifndef SQLITE_OMIT_FLOATING_POINT
			}else if( pLeft->op==TK_FLOAT ){
				assert( !ExprHasProperty(pExpr, EP_IntValue) );
				codeReal(v, pLeft->u.zToken, 1, target);
#endif
			}else{
				regFree1 = r1 = sqlite3GetTempReg(pParse);
				sqlite3VdbeAddOp2(v, OP_Integer, 0, r1);
				r2 = sqlite3ExprCodeTemp(pParse, pExpr->pLeft, &regFree2);
				sqlite3VdbeAddOp3(v, OP_Subtract, r2, r1, target);
				testcase( regFree2==0 );
			}
			inReg = target;
			break;
						}
		case TK_BITNOT:
		case TK_NOT: {
			assert( TK_BITNOT==OP_BitNot );
			assert( TK_NOT==OP_Not );
			testcase( op==TK_BITNOT );
			testcase( op==TK_NOT );
			r1 = sqlite3ExprCodeTemp(pParse, pExpr->pLeft, &regFree1);
			testcase( regFree1==0 );
			inReg = target;
			sqlite3VdbeAddOp2(v, op, r1, inReg);
			break;
					 }
		case TK_ISNULL:
		case TK_NOTNULL: {
			int addr;
			assert( TK_ISNULL==OP_IsNull );
			assert( TK_NOTNULL==OP_NotNull );
			testcase( op==TK_ISNULL );
			testcase( op==TK_NOTNULL );
			sqlite3VdbeAddOp2(v, OP_Integer, 1, target);
			r1 = sqlite3ExprCodeTemp(pParse, pExpr->pLeft, &regFree1);
			testcase( regFree1==0 );
			addr = sqlite3VdbeAddOp1(v, op, r1);
			sqlite3VdbeAddOp2(v, OP_AddImm, target, -1);
			sqlite3VdbeJumpHere(v, addr);
			break;
						 }
		case TK_AGG_FUNCTION: {
			AggInfo *pInfo = pExpr->pAggInfo;
			if( pInfo==0 ){
				assert( !ExprHasProperty(pExpr, EP_IntValue) );
				sqlite3ErrorMsg(pParse, "misuse of aggregate: %s()", pExpr->u.zToken);
			}else{
				inReg = pInfo->aFunc[pExpr->iAgg].iMem;
			}
			break;
							  }
		case TK_CONST_FUNC:
		case TK_FUNCTION: {
			ExprList *pFarg;       /* List of function arguments */
			int nFarg;             /* Number of function arguments */
			FuncDef *pDef;         /* The function definition object */
			int nId;               /* Length of the function name in bytes */
			const char *zId;       /* The function name */
			int constMask = 0;     /* Mask of function arguments that are constant */
			int i;                 /* Loop counter */
			u8 enc = ENC(db);      /* The text encoding used by this database */
			CollSeq *pColl = 0;    /* A collating sequence */

			assert( !ExprHasProperty(pExpr, EP_xIsSelect) );
			testcase( op==TK_CONST_FUNC );
			testcase( op==TK_FUNCTION );
			if( ExprHasAnyProperty(pExpr, EP_TokenOnly) ){
				pFarg = 0;
			}else{
				pFarg = pExpr->x.pList;
			}
			nFarg = pFarg ? pFarg->nExpr : 0;
			assert( !ExprHasProperty(pExpr, EP_IntValue) );
			zId = pExpr->u.zToken;
			nId = sqlite3Strlen30(zId);
			pDef = sqlite3FindFunction(db, zId, nId, nFarg, enc, 0);
			if( pDef==0 ){
				sqlite3ErrorMsg(pParse, "unknown function: %.*s()", nId, zId);
				break;
			}

			/* Attempt a direct implementation of the built-in COALESCE() and
			** IFNULL() functions.  This avoids unnecessary evalation of
			** arguments past the first non-NULL argument.
			*/
			if( pDef->flags & SQLITE_FUNC_COALESCE ){
				int endCoalesce = sqlite3VdbeMakeLabel(v);
				assert( nFarg>=2 );
				sqlite3ExprCode(pParse, pFarg->a[0].pExpr, target);
				for(i=1; i<nFarg; i++){
					sqlite3VdbeAddOp2(v, OP_NotNull, target, endCoalesce);
					sqlite3ExprCacheRemove(pParse, target, 1);
					sqlite3ExprCachePush(pParse);
					sqlite3ExprCode(pParse, pFarg->a[i].pExpr, target);
					sqlite3ExprCachePop(pParse, 1);
				}
				sqlite3VdbeResolveLabel(v, endCoalesce);
				break;
			}


			if( pFarg ){
				r1 = sqlite3GetTempRange(pParse, nFarg);

				/* For length() and typeof() functions with a column argument,
				** set the P5 parameter to the OP_Column opcode to OPFLAG_LENGTHARG
				** or OPFLAG_TYPEOFARG respectively, to avoid unnecessary data
				** loading.
				*/
				if( (pDef->flags & (SQLITE_FUNC_LENGTH|SQLITE_FUNC_TYPEOF))!=0 ){
					u8 exprOp;
					assert( nFarg==1 );
					assert( pFarg->a[0].pExpr!=0 );
					exprOp = pFarg->a[0].pExpr->op;
					if( exprOp==TK_COLUMN || exprOp==TK_AGG_COLUMN ){
						assert( SQLITE_FUNC_LENGTH==OPFLAG_LENGTHARG );
						assert( SQLITE_FUNC_TYPEOF==OPFLAG_TYPEOFARG );
						testcase( pDef->flags==SQLITE_FUNC_LENGTH );
						pFarg->a[0].pExpr->op2 = pDef->flags;
					}
				}

				sqlite3ExprCachePush(pParse);     /* Ticket 2ea2425d34be */
				sqlite3ExprCodeExprList(pParse, pFarg, r1, 1);
				sqlite3ExprCachePop(pParse, 1);   /* Ticket 2ea2425d34be */
			}else{
				r1 = 0;
			}
#ifndef SQLITE_OMIT_VIRTUALTABLE
			/* Possibly overload the function if the first argument is
			** a virtual table column.
			**
			** For infix functions (LIKE, GLOB, REGEXP, and MATCH) use the
			** second argument, not the first, as the argument to test to
			** see if it is a column in a virtual table.  This is done because
			** the left operand of infix functions (the operand we want to
			** control overloading) ends up as the second argument to the
			** function.  The expression "A glob B" is equivalent to 
			** "glob(B,A).  We want to use the A in "A glob B" to test
			** for function overloading.  But we use the B term in "glob(B,A)".
			*/
			if( nFarg>=2 && (pExpr->flags & EP_InfixFunc) ){
				pDef = sqlite3VtabOverloadFunction(db, pDef, nFarg, pFarg->a[1].pExpr);
			}else if( nFarg>0 ){
				pDef = sqlite3VtabOverloadFunction(db, pDef, nFarg, pFarg->a[0].pExpr);
			}
#endif
			for(i=0; i<nFarg; i++){
				if( i<32 && sqlite3ExprIsConstant(pFarg->a[i].pExpr) ){
					constMask |= (1<<i);
				}
				if( (pDef->flags & SQLITE_FUNC_NEEDCOLL)!=0 && !pColl ){
					pColl = sqlite3ExprCollSeq(pParse, pFarg->a[i].pExpr);
				}
			}
			if( pDef->flags & SQLITE_FUNC_NEEDCOLL ){
				if( !pColl ) pColl = db->pDfltColl; 
				sqlite3VdbeAddOp4(v, OP_CollSeq, 0, 0, 0, (char *)pColl, P4_COLLSEQ);
			}
			sqlite3VdbeAddOp4(v, OP_Function, constMask, r1, target,
				(char*)pDef, P4_FUNCDEF);
			sqlite3VdbeChangeP5(v, (u8)nFarg);
			if( nFarg ){
				sqlite3ReleaseTempRange(pParse, r1, nFarg);
			}
			break;
						  }
#ifndef SQLITE_OMIT_SUBQUERY
		case TK_EXISTS:
		case TK_SELECT: {
			testcase( op==TK_EXISTS );
			testcase( op==TK_SELECT );
			inReg = sqlite3CodeSubselect(pParse, pExpr, 0, 0);
			break;
						}
		case TK_IN: {
			int destIfFalse = sqlite3VdbeMakeLabel(v);
			int destIfNull = sqlite3VdbeMakeLabel(v);
			sqlite3VdbeAddOp2(v, OP_Null, 0, target);
			sqlite3ExprCodeIN(pParse, pExpr, destIfFalse, destIfNull);
			sqlite3VdbeAddOp2(v, OP_Integer, 1, target);
			sqlite3VdbeResolveLabel(v, destIfFalse);
			sqlite3VdbeAddOp2(v, OP_AddImm, target, 0);
			sqlite3VdbeResolveLabel(v, destIfNull);
			break;
					}
#endif /* SQLITE_OMIT_SUBQUERY */


					/*
					**    x BETWEEN y AND z
					**
					** This is equivalent to
					**
					**    x>=y AND x<=z
					**
					** X is stored in pExpr->pLeft.
					** Y is stored in pExpr->pList->a[0].pExpr.
					** Z is stored in pExpr->pList->a[1].pExpr.
					*/
		case TK_BETWEEN: {
			Expr *pLeft = pExpr->pLeft;
			struct ExprList_item *pLItem = pExpr->x.pList->a;
			Expr *pRight = pLItem->pExpr;

			r1 = sqlite3ExprCodeTemp(pParse, pLeft, &regFree1);
			r2 = sqlite3ExprCodeTemp(pParse, pRight, &regFree2);
			testcase( regFree1==0 );
			testcase( regFree2==0 );
			r3 = sqlite3GetTempReg(pParse);
			r4 = sqlite3GetTempReg(pParse);
			codeCompare(pParse, pLeft, pRight, OP_Ge,
				r1, r2, r3, SQLITE_STOREP2);
			pLItem++;
			pRight = pLItem->pExpr;
			sqlite3ReleaseTempReg(pParse, regFree2);
			r2 = sqlite3ExprCodeTemp(pParse, pRight, &regFree2);
			testcase( regFree2==0 );
			codeCompare(pParse, pLeft, pRight, OP_Le, r1, r2, r4, SQLITE_STOREP2);
			sqlite3VdbeAddOp3(v, OP_And, r3, r4, target);
			sqlite3ReleaseTempReg(pParse, r3);
			sqlite3ReleaseTempReg(pParse, r4);
			break;
						 }
		case TK_COLLATE: 
		case TK_UPLUS: {
			inReg = sqlite3ExprCodeTarget(pParse, pExpr->pLeft, target);
			break;
					   }

		case TK_TRIGGER: {
			/* If the opcode is TK_TRIGGER, then the expression is a reference
			** to a column in the new.* or old.* pseudo-tables available to
			** trigger programs. In this case Expr.iTable is set to 1 for the
			** new.* pseudo-table, or 0 for the old.* pseudo-table. Expr.iColumn
			** is set to the column of the pseudo-table to read, or to -1 to
			** read the rowid field.
			**
			** The expression is implemented using an OP_Param opcode. The p1
			** parameter is set to 0 for an old.rowid reference, or to (i+1)
			** to reference another column of the old.* pseudo-table, where 
			** i is the index of the column. For a new.rowid reference, p1 is
			** set to (n+1), where n is the number of columns in each pseudo-table.
			** For a reference to any other column in the new.* pseudo-table, p1
			** is set to (n+2+i), where n and i are as defined previously. For
			** example, if the table on which triggers are being fired is
			** declared as:
			**
			**   CREATE TABLE t1(a, b);
			**
			** Then p1 is interpreted as follows:
			**
			**   p1==0   ->    old.rowid     p1==3   ->    new.rowid
			**   p1==1   ->    old.a         p1==4   ->    new.a
			**   p1==2   ->    old.b         p1==5   ->    new.b       
			*/
			Table *pTab = pExpr->pTab;
			int p1 = pExpr->iTable * (pTab->nCol+1) + 1 + pExpr->iColumn;

			assert( pExpr->iTable==0 || pExpr->iTable==1 );
			assert( pExpr->iColumn>=-1 && pExpr->iColumn<pTab->nCol );
			assert( pTab->iPKey<0 || pExpr->iColumn!=pTab->iPKey );
			assert( p1>=0 && p1<(pTab->nCol*2+2) );

			sqlite3VdbeAddOp2(v, OP_Param, p1, target);
			VdbeComment((v, "%s.%s -> $%d",
				(pExpr->iTable ? "new" : "old"),
				(pExpr->iColumn<0 ? "rowid" : pExpr->pTab->aCol[pExpr->iColumn].zName),
				target
				));

#ifndef SQLITE_OMIT_FLOATING_POINT
			/* If the column has REAL affinity, it may currently be stored as an
			** integer. Use OP_RealAffinity to make sure it is really real.  */
			if( pExpr->iColumn>=0 
				&& pTab->aCol[pExpr->iColumn].affinity==SQLITE_AFF_REAL
				){
					sqlite3VdbeAddOp1(v, OP_RealAffinity, target);
			}
#endif
			break;
						 }


						 /*
						 ** Form A:
						 **   CASE x WHEN e1 THEN r1 WHEN e2 THEN r2 ... WHEN eN THEN rN ELSE y END
						 **
						 ** Form B:
						 **   CASE WHEN e1 THEN r1 WHEN e2 THEN r2 ... WHEN eN THEN rN ELSE y END
						 **
						 ** Form A is can be transformed into the equivalent form B as follows:
						 **   CASE WHEN x=e1 THEN r1 WHEN x=e2 THEN r2 ...
						 **        WHEN x=eN THEN rN ELSE y END
						 **
						 ** X (if it exists) is in pExpr->pLeft.
						 ** Y is in pExpr->pRight.  The Y is also optional.  If there is no
						 ** ELSE clause and no other term matches, then the result of the
						 ** exprssion is NULL.
						 ** Ei is in pExpr->pList->a[i*2] and Ri is pExpr->pList->a[i*2+1].
						 **
						 ** The result of the expression is the Ri for the first matching Ei,
						 ** or if there is no matching Ei, the ELSE term Y, or if there is
						 ** no ELSE term, NULL.
						 */
		default: assert( op==TK_CASE ); {
			int endLabel;                     /* GOTO label for end of CASE stmt */
			int nextCase;                     /* GOTO label for next WHEN clause */
			int nExpr;                        /* 2x number of WHEN terms */
			int i;                            /* Loop counter */
			ExprList *pEList;                 /* List of WHEN terms */
			struct ExprList_item *aListelem;  /* Array of WHEN terms */
			Expr opCompare;                   /* The X==Ei expression */
			Expr cacheX;                      /* Cached expression X */
			Expr *pX;                         /* The X expression */
			Expr *pTest = 0;                  /* X==Ei (form A) or just Ei (form B) */
			VVA_ONLY( int iCacheLevel = pParse->iCacheLevel; )

				assert( !ExprHasProperty(pExpr, EP_xIsSelect) && pExpr->x.pList );
			assert((pExpr->x.pList->nExpr % 2) == 0);
			assert(pExpr->x.pList->nExpr > 0);
			pEList = pExpr->x.pList;
			aListelem = pEList->a;
			nExpr = pEList->nExpr;
			endLabel = sqlite3VdbeMakeLabel(v);
			if( (pX = pExpr->pLeft)!=0 ){
				cacheX = *pX;
				testcase( pX->op==TK_COLUMN );
				testcase( pX->op==TK_REGISTER );
				cacheX.iTable = sqlite3ExprCodeTemp(pParse, pX, &regFree1);
				testcase( regFree1==0 );
				cacheX.op = TK_REGISTER;
				opCompare.op = TK_EQ;
				opCompare.pLeft = &cacheX;
				pTest = &opCompare;
				/* Ticket b351d95f9cd5ef17e9d9dbae18f5ca8611190001:
				** The value in regFree1 might get SCopy-ed into the file result.
				** So make sure that the regFree1 register is not reused for other
				** purposes and possibly overwritten.  */
				regFree1 = 0;
			}
			for(i=0; i<nExpr; i=i+2){
				sqlite3ExprCachePush(pParse);
				if( pX ){
					assert( pTest!=0 );
					opCompare.pRight = aListelem[i].pExpr;
				}else{
					pTest = aListelem[i].pExpr;
				}
				nextCase = sqlite3VdbeMakeLabel(v);
				testcase( pTest->op==TK_COLUMN );
				sqlite3ExprIfFalse(pParse, pTest, nextCase, SQLITE_JUMPIFNULL);
				testcase( aListelem[i+1].pExpr->op==TK_COLUMN );
				testcase( aListelem[i+1].pExpr->op==TK_REGISTER );
				sqlite3ExprCode(pParse, aListelem[i+1].pExpr, target);
				sqlite3VdbeAddOp2(v, OP_Goto, 0, endLabel);
				sqlite3ExprCachePop(pParse, 1);
				sqlite3VdbeResolveLabel(v, nextCase);
			}
			if( pExpr->pRight ){
				sqlite3ExprCachePush(pParse);
				sqlite3ExprCode(pParse, pExpr->pRight, target);
				sqlite3ExprCachePop(pParse, 1);
			}else{
				sqlite3VdbeAddOp2(v, OP_Null, 0, target);
			}
			assert( db->mallocFailed || pParse->nErr>0 
				|| pParse->iCacheLevel==iCacheLevel );
			sqlite3VdbeResolveLabel(v, endLabel);
			break;
				 }
#ifndef SQLITE_OMIT_TRIGGER
		case TK_RAISE: {
			assert( pExpr->affinity==OE_Rollback 
				|| pExpr->affinity==OE_Abort
				|| pExpr->affinity==OE_Fail
				|| pExpr->affinity==OE_Ignore
				);
			if( !pParse->pTriggerTab ){
				sqlite3ErrorMsg(pParse,
					"RAISE() may only be used within a trigger-program");
				return 0;
			}
			if( pExpr->affinity==OE_Abort ){
				sqlite3MayAbort(pParse);
			}
			assert( !ExprHasProperty(pExpr, EP_IntValue) );
			if( pExpr->affinity==OE_Ignore ){
				sqlite3VdbeAddOp4(
					v, OP_Halt, SQLITE_OK, OE_Ignore, 0, pExpr->u.zToken,0);
			}else{
				sqlite3HaltConstraint(pParse, SQLITE_CONSTRAINT_TRIGGER,
					pExpr->affinity, pExpr->u.zToken, 0);
			}

			break;
					   }
#endif
		}
		sqlite3ReleaseTempReg(pParse, regFree1);
		sqlite3ReleaseTempReg(pParse, regFree2);
		return inReg;
	}

	/*
	** Generate code to evaluate an expression and store the results
	** into a register.  Return the register number where the results
	** are stored.
	**
	** If the register is a temporary register that can be deallocated,
	** then write its number into *pReg.  If the result register is not
	** a temporary, then set *pReg to zero.
	*/
	int sqlite3ExprCodeTemp(Parse *pParse, Expr *pExpr, int *pReg){
		int r1 = sqlite3GetTempReg(pParse);
		int r2 = sqlite3ExprCodeTarget(pParse, pExpr, r1);
		if( r2==r1 ){
			*pReg = r1;
		}else{
			sqlite3ReleaseTempReg(pParse, r1);
			*pReg = 0;
		}
		return r2;
	}

	/*
	** Generate code that will evaluate expression pExpr and store the
	** results in register target.  The results are guaranteed to appear
	** in register target.
	*/
	int sqlite3ExprCode(Parse *pParse, Expr *pExpr, int target){
		int inReg;

		assert( target>0 && target<=pParse->nMem );
		if( pExpr && pExpr->op==TK_REGISTER ){
			sqlite3VdbeAddOp2(pParse->pVdbe, OP_Copy, pExpr->iTable, target);
		}else{
			inReg = sqlite3ExprCodeTarget(pParse, pExpr, target);
			assert( pParse->pVdbe || pParse->db->mallocFailed );
			if( inReg!=target && pParse->pVdbe ){
				sqlite3VdbeAddOp2(pParse->pVdbe, OP_SCopy, inReg, target);
			}
		}
		return target;
	}

	/*
	** Generate code that evalutes the given expression and puts the result
	** in register target.
	**
	** Also make a copy of the expression results into another "cache" register
	** and modify the expression so that the next time it is evaluated,
	** the result is a copy of the cache register.
	**
	** This routine is used for expressions that are used multiple 
	** times.  They are evaluated once and the results of the expression
	** are reused.
	*/
	int sqlite3ExprCodeAndCache(Parse *pParse, Expr *pExpr, int target){
		Vdbe *v = pParse->pVdbe;
		int inReg;
		inReg = sqlite3ExprCode(pParse, pExpr, target);
		assert( target>0 );
		/* This routine is called for terms to INSERT or UPDATE.  And the only
		** other place where expressions can be converted into TK_REGISTER is
		** in WHERE clause processing.  So as currently implemented, there is
		** no way for a TK_REGISTER to exist here.  But it seems prudent to
		** keep the ALWAYS() in case the conditions above change with future
		** modifications or enhancements. */
		if( ALWAYS(pExpr->op!=TK_REGISTER) ){  
			int iMem;
			iMem = ++pParse->nMem;
			sqlite3VdbeAddOp2(v, OP_Copy, inReg, iMem);
			pExpr->iTable = iMem;
			pExpr->op2 = pExpr->op;
			pExpr->op = TK_REGISTER;
		}
		return inReg;
	}

#if defined(SQLITE_ENABLE_TREE_EXPLAIN)
	/*
	** Generate a human-readable explanation of an expression tree.
	*/
	void sqlite3ExplainExpr(Vdbe *pOut, Expr *pExpr){
		int op;                   /* The opcode being coded */
		const char *zBinOp = 0;   /* Binary operator */
		const char *zUniOp = 0;   /* Unary operator */
		if( pExpr==0 ){
			op = TK_NULL;
		}else{
			op = pExpr->op;
		}
		switch( op ){
		case TK_AGG_COLUMN: {
			sqlite3ExplainPrintf(pOut, "AGG{%d:%d}",
				pExpr->iTable, pExpr->iColumn);
			break;
							}
		case TK_COLUMN: {
			if( pExpr->iTable<0 ){
				/* This only happens when coding check constraints */
				sqlite3ExplainPrintf(pOut, "COLUMN(%d)", pExpr->iColumn);
			}else{
				sqlite3ExplainPrintf(pOut, "{%d:%d}",
					pExpr->iTable, pExpr->iColumn);
			}
			break;
						}
		case TK_INTEGER: {
			if( pExpr->flags & EP_IntValue ){
				sqlite3ExplainPrintf(pOut, "%d", pExpr->u.iValue);
			}else{
				sqlite3ExplainPrintf(pOut, "%s", pExpr->u.zToken);
			}
			break;
						 }
#ifndef SQLITE_OMIT_FLOATING_POINT
		case TK_FLOAT: {
			sqlite3ExplainPrintf(pOut,"%s", pExpr->u.zToken);
			break;
					   }
#endif
		case TK_STRING: {
			sqlite3ExplainPrintf(pOut,"%Q", pExpr->u.zToken);
			break;
						}
		case TK_NULL: {
			sqlite3ExplainPrintf(pOut,"NULL");
			break;
					  }
#ifndef SQLITE_OMIT_BLOB_LITERAL
		case TK_BLOB: {
			sqlite3ExplainPrintf(pOut,"%s", pExpr->u.zToken);
			break;
					  }
#endif
		case TK_VARIABLE: {
			sqlite3ExplainPrintf(pOut,"VARIABLE(%s,%d)",
				pExpr->u.zToken, pExpr->iColumn);
			break;
						  }
		case TK_REGISTER: {
			sqlite3ExplainPrintf(pOut,"REGISTER(%d)", pExpr->iTable);
			break;
						  }
		case TK_AS: {
			sqlite3ExplainExpr(pOut, pExpr->pLeft);
			break;
					}
#ifndef SQLITE_OMIT_CAST
		case TK_CAST: {
			/* Expressions of the form:   CAST(pLeft AS token) */
			const char *zAff = "unk";
			switch( sqlite3AffinityType(pExpr->u.zToken) ){
			case SQLITE_AFF_TEXT:    zAff = "TEXT";     break;
			case SQLITE_AFF_NONE:    zAff = "NONE";     break;
			case SQLITE_AFF_NUMERIC: zAff = "NUMERIC";  break;
			case SQLITE_AFF_INTEGER: zAff = "INTEGER";  break;
			case SQLITE_AFF_REAL:    zAff = "REAL";     break;
			}
			sqlite3ExplainPrintf(pOut, "CAST-%s(", zAff);
			sqlite3ExplainExpr(pOut, pExpr->pLeft);
			sqlite3ExplainPrintf(pOut, ")");
			break;
					  }
#endif /* SQLITE_OMIT_CAST */
		case TK_LT:      zBinOp = "LT";     break;
		case TK_LE:      zBinOp = "LE";     break;
		case TK_GT:      zBinOp = "GT";     break;
		case TK_GE:      zBinOp = "GE";     break;
		case TK_NE:      zBinOp = "NE";     break;
		case TK_EQ:      zBinOp = "EQ";     break;
		case TK_IS:      zBinOp = "IS";     break;
		case TK_ISNOT:   zBinOp = "ISNOT";  break;
		case TK_AND:     zBinOp = "AND";    break;
		case TK_OR:      zBinOp = "OR";     break;
		case TK_PLUS:    zBinOp = "ADD";    break;
		case TK_STAR:    zBinOp = "MUL";    break;
		case TK_MINUS:   zBinOp = "SUB";    break;
		case TK_REM:     zBinOp = "REM";    break;
		case TK_BITAND:  zBinOp = "BITAND"; break;
		case TK_BITOR:   zBinOp = "BITOR";  break;
		case TK_SLASH:   zBinOp = "DIV";    break;
		case TK_LSHIFT:  zBinOp = "LSHIFT"; break;
		case TK_RSHIFT:  zBinOp = "RSHIFT"; break;
		case TK_CONCAT:  zBinOp = "CONCAT"; break;

		case TK_UMINUS:  zUniOp = "UMINUS"; break;
		case TK_UPLUS:   zUniOp = "UPLUS";  break;
		case TK_BITNOT:  zUniOp = "BITNOT"; break;
		case TK_NOT:     zUniOp = "NOT";    break;
		case TK_ISNULL:  zUniOp = "ISNULL"; break;
		case TK_NOTNULL: zUniOp = "NOTNULL"; break;

		case TK_COLLATE: {
			sqlite3ExplainExpr(pOut, pExpr->pLeft);
			sqlite3ExplainPrintf(pOut,".COLLATE(%s)",pExpr->u.zToken);
			break;
						 }

		case TK_AGG_FUNCTION:
		case TK_CONST_FUNC:
		case TK_FUNCTION: {
			ExprList *pFarg;       /* List of function arguments */
			if( ExprHasAnyProperty(pExpr, EP_TokenOnly) ){
				pFarg = 0;
			}else{
				pFarg = pExpr->x.pList;
			}
			if( op==TK_AGG_FUNCTION ){
				sqlite3ExplainPrintf(pOut, "AGG_FUNCTION%d:%s(",
					pExpr->op2, pExpr->u.zToken);
			}else{
				sqlite3ExplainPrintf(pOut, "FUNCTION:%s(", pExpr->u.zToken);
			}
			if( pFarg ){
				sqlite3ExplainExprList(pOut, pFarg);
			}
			sqlite3ExplainPrintf(pOut, ")");
			break;
						  }
#ifndef SQLITE_OMIT_SUBQUERY
		case TK_EXISTS: {
			sqlite3ExplainPrintf(pOut, "EXISTS(");
			sqlite3ExplainSelect(pOut, pExpr->x.pSelect);
			sqlite3ExplainPrintf(pOut,")");
			break;
						}
		case TK_SELECT: {
			sqlite3ExplainPrintf(pOut, "(");
			sqlite3ExplainSelect(pOut, pExpr->x.pSelect);
			sqlite3ExplainPrintf(pOut, ")");
			break;
						}
		case TK_IN: {
			sqlite3ExplainPrintf(pOut, "IN(");
			sqlite3ExplainExpr(pOut, pExpr->pLeft);
			sqlite3ExplainPrintf(pOut, ",");
			if( ExprHasProperty(pExpr, EP_xIsSelect) ){
				sqlite3ExplainSelect(pOut, pExpr->x.pSelect);
			}else{
				sqlite3ExplainExprList(pOut, pExpr->x.pList);
			}
			sqlite3ExplainPrintf(pOut, ")");
			break;
					}
#endif /* SQLITE_OMIT_SUBQUERY */

					/*
					**    x BETWEEN y AND z
					**
					** This is equivalent to
					**
					**    x>=y AND x<=z
					**
					** X is stored in pExpr->pLeft.
					** Y is stored in pExpr->pList->a[0].pExpr.
					** Z is stored in pExpr->pList->a[1].pExpr.
					*/
		case TK_BETWEEN: {
			Expr *pX = pExpr->pLeft;
			Expr *pY = pExpr->x.pList->a[0].pExpr;
			Expr *pZ = pExpr->x.pList->a[1].pExpr;
			sqlite3ExplainPrintf(pOut, "BETWEEN(");
			sqlite3ExplainExpr(pOut, pX);
			sqlite3ExplainPrintf(pOut, ",");
			sqlite3ExplainExpr(pOut, pY);
			sqlite3ExplainPrintf(pOut, ",");
			sqlite3ExplainExpr(pOut, pZ);
			sqlite3ExplainPrintf(pOut, ")");
			break;
						 }
		case TK_TRIGGER: {
			/* If the opcode is TK_TRIGGER, then the expression is a reference
			** to a column in the new.* or old.* pseudo-tables available to
			** trigger programs. In this case Expr.iTable is set to 1 for the
			** new.* pseudo-table, or 0 for the old.* pseudo-table. Expr.iColumn
			** is set to the column of the pseudo-table to read, or to -1 to
			** read the rowid field.
			*/
			sqlite3ExplainPrintf(pOut, "%s(%d)", 
				pExpr->iTable ? "NEW" : "OLD", pExpr->iColumn);
			break;
						 }
		case TK_CASE: {
			sqlite3ExplainPrintf(pOut, "CASE(");
			sqlite3ExplainExpr(pOut, pExpr->pLeft);
			sqlite3ExplainPrintf(pOut, ",");
			sqlite3ExplainExprList(pOut, pExpr->x.pList);
			break;
					  }
#ifndef SQLITE_OMIT_TRIGGER
		case TK_RAISE: {
			const char *zType = "unk";
			switch( pExpr->affinity ){
			case OE_Rollback:   zType = "rollback";  break;
			case OE_Abort:      zType = "abort";     break;
			case OE_Fail:       zType = "fail";      break;
			case OE_Ignore:     zType = "ignore";    break;
			}
			sqlite3ExplainPrintf(pOut, "RAISE-%s(%s)", zType, pExpr->u.zToken);
			break;
					   }
#endif
		}
		if( zBinOp ){
			sqlite3ExplainPrintf(pOut,"%s(", zBinOp);
			sqlite3ExplainExpr(pOut, pExpr->pLeft);
			sqlite3ExplainPrintf(pOut,",");
			sqlite3ExplainExpr(pOut, pExpr->pRight);
			sqlite3ExplainPrintf(pOut,")");
		}else if( zUniOp ){
			sqlite3ExplainPrintf(pOut,"%s(", zUniOp);
			sqlite3ExplainExpr(pOut, pExpr->pLeft);
			sqlite3ExplainPrintf(pOut,")");
		}
	}
#endif /* defined(SQLITE_ENABLE_TREE_EXPLAIN) */

#if defined(SQLITE_ENABLE_TREE_EXPLAIN)
	/*
	** Generate a human-readable explanation of an expression list.
	*/
	void sqlite3ExplainExprList(Vdbe *pOut, ExprList *pList){
		int i;
		if( pList==0 || pList->nExpr==0 ){
			sqlite3ExplainPrintf(pOut, "(empty-list)");
			return;
		}else if( pList->nExpr==1 ){
			sqlite3ExplainExpr(pOut, pList->a[0].pExpr);
		}else{
			sqlite3ExplainPush(pOut);
			for(i=0; i<pList->nExpr; i++){
				sqlite3ExplainPrintf(pOut, "item[%d] = ", i);
				sqlite3ExplainPush(pOut);
				sqlite3ExplainExpr(pOut, pList->a[i].pExpr);
				sqlite3ExplainPop(pOut);
				if( pList->a[i].zName ){
					sqlite3ExplainPrintf(pOut, " AS %s", pList->a[i].zName);
				}
				if( pList->a[i].bSpanIsTab ){
					sqlite3ExplainPrintf(pOut, " (%s)", pList->a[i].zSpan);
				}
				if( i<pList->nExpr-1 ){
					sqlite3ExplainNL(pOut);
				}
			}
			sqlite3ExplainPop(pOut);
		}
	}
#endif /* SQLITE_DEBUG */

	/*
	** Return TRUE if pExpr is an constant expression that is appropriate
	** for factoring out of a loop.  Appropriate expressions are:
	**
	**    *  Any expression that evaluates to two or more opcodes.
	**
	**    *  Any OP_Integer, OP_Real, OP_String, OP_Blob, OP_Null, 
	**       or OP_Variable that does not need to be placed in a 
	**       specific register.
	**
	** There is no point in factoring out single-instruction constant
	** expressions that need to be placed in a particular register.  
	** We could factor them out, but then we would end up adding an
	** OP_SCopy instruction to move the value into the correct register
	** later.  We might as well just use the original instruction and
	** avoid the OP_SCopy.
	*/
	static int isAppropriateForFactoring(Expr *p){
		if( !sqlite3ExprIsConstantNotJoin(p) ){
			return 0;  /* Only constant expressions are appropriate for factoring */
		}
		if( (p->flags & EP_FixedDest)==0 ){
			return 1;  /* Any constant without a fixed destination is appropriate */
		}
		while( p->op==TK_UPLUS ) p = p->pLeft;
		switch( p->op ){
#ifndef SQLITE_OMIT_BLOB_LITERAL
		case TK_BLOB:
#endif
		case TK_VARIABLE:
		case TK_INTEGER:
		case TK_FLOAT:
		case TK_NULL:
		case TK_STRING: {
			testcase( p->op==TK_BLOB );
			testcase( p->op==TK_VARIABLE );
			testcase( p->op==TK_INTEGER );
			testcase( p->op==TK_FLOAT );
			testcase( p->op==TK_NULL );
			testcase( p->op==TK_STRING );
			/* Single-instruction constants with a fixed destination are
			** better done in-line.  If we factor them, they will just end
			** up generating an OP_SCopy to move the value to the destination
			** register. */
			return 0;
						}
		case TK_UMINUS: {
			if( p->pLeft->op==TK_FLOAT || p->pLeft->op==TK_INTEGER ){
				return 0;
			}
			break;
						}
		default: {
			break;
				 }
		}
		return 1;
	}

	/*
	** If pExpr is a constant expression that is appropriate for
	** factoring out of a loop, then evaluate the expression
	** into a register and convert the expression into a TK_REGISTER
	** expression.
	*/
	static int evalConstExpr(Walker *pWalker, Expr *pExpr){
		Parse *pParse = pWalker->pParse;
		switch( pExpr->op ){
		case TK_IN:
		case TK_REGISTER: {
			return WRC_Prune;
						  }
		case TK_COLLATE: {
			return WRC_Continue;
						 }
		case TK_FUNCTION:
		case TK_AGG_FUNCTION:
		case TK_CONST_FUNC: {
			/* The arguments to a function have a fixed destination.
			** Mark them this way to avoid generated unneeded OP_SCopy
			** instructions. 
			*/
			ExprList *pList = pExpr->x.pList;
			assert( !ExprHasProperty(pExpr, EP_xIsSelect) );
			if( pList ){
				int i = pList->nExpr;
				struct ExprList_item *pItem = pList->a;
				for(; i>0; i--, pItem++){
					if( ALWAYS(pItem->pExpr) ) pItem->pExpr->flags |= EP_FixedDest;
				}
			}
			break;
							}
		}
		if( isAppropriateForFactoring(pExpr) ){
			int r1 = ++pParse->nMem;
			int r2 = sqlite3ExprCodeTarget(pParse, pExpr, r1);
			/* If r2!=r1, it means that register r1 is never used.  That is harmless
			** but suboptimal, so we want to know about the situation to fix it.
			** Hence the following assert: */
			assert( r2==r1 );
			pExpr->op2 = pExpr->op;
			pExpr->op = TK_REGISTER;
			pExpr->iTable = r2;
			return WRC_Prune;
		}
		return WRC_Continue;
	}

	/*
	** Preevaluate constant subexpressions within pExpr and store the
	** results in registers.  Modify pExpr so that the constant subexpresions
	** are TK_REGISTER opcodes that refer to the precomputed values.
	**
	** This routine is a no-op if the jump to the cookie-check code has
	** already occur.  Since the cookie-check jump is generated prior to
	** any other serious processing, this check ensures that there is no
	** way to accidently bypass the constant initializations.
	**
	** This routine is also a no-op if the SQLITE_FactorOutConst optimization
	** is disabled via the sqlite3_test_control(SQLITE_TESTCTRL_OPTIMIZATIONS)
	** interface.  This allows test logic to verify that the same answer is
	** obtained for queries regardless of whether or not constants are
	** precomputed into registers or if they are inserted in-line.
	*/
	void sqlite3ExprCodeConstants(Parse *pParse, Expr *pExpr){
		Walker w;
		if( pParse->cookieGoto ) return;
		if( OptimizationDisabled(pParse->db, SQLITE_FactorOutConst) ) return;
		w.xExprCallback = evalConstExpr;
		w.xSelectCallback = 0;
		w.pParse = pParse;
		sqlite3WalkExpr(&w, pExpr);
	}


	/*
	** Generate code that pushes the value of every element of the given
	** expression list into a sequence of registers beginning at target.
	**
	** Return the number of elements evaluated.
	*/
	int sqlite3ExprCodeExprList(
		Parse *pParse,     /* Parsing context */
		ExprList *pList,   /* The expression list to be coded */
		int target,        /* Where to write results */
		int doHardCopy     /* Make a hard copy of every element */
		){
			struct ExprList_item *pItem;
			int i, n;
			assert( pList!=0 );
			assert( target>0 );
			assert( pParse->pVdbe!=0 );  /* Never gets this far otherwise */
			n = pList->nExpr;
			for(pItem=pList->a, i=0; i<n; i++, pItem++){
				Expr *pExpr = pItem->pExpr;
				int inReg = sqlite3ExprCodeTarget(pParse, pExpr, target+i);
				if( inReg!=target+i ){
					sqlite3VdbeAddOp2(pParse->pVdbe, doHardCopy ? OP_Copy : OP_SCopy,
						inReg, target+i);
				}
			}
			return n;
	}

	/*
	** Generate code for a BETWEEN operator.
	**
	**    x BETWEEN y AND z
	**
	** The above is equivalent to 
	**
	**    x>=y AND x<=z
	**
	** Code it as such, taking care to do the common subexpression
	** elementation of x.
	*/
	static void exprCodeBetween(
		Parse *pParse,    /* Parsing and code generating context */
		Expr *pExpr,      /* The BETWEEN expression */
		int dest,         /* Jump here if the jump is taken */
		int jumpIfTrue,   /* Take the jump if the BETWEEN is true */
		int jumpIfNull    /* Take the jump if the BETWEEN is NULL */
		){
			Expr exprAnd;     /* The AND operator in  x>=y AND x<=z  */
			Expr compLeft;    /* The  x>=y  term */
			Expr compRight;   /* The  x<=z  term */
			Expr exprX;       /* The  x  subexpression */
			int regFree1 = 0; /* Temporary use register */

			assert( !ExprHasProperty(pExpr, EP_xIsSelect) );
			exprX = *pExpr->pLeft;
			exprAnd.op = TK_AND;
			exprAnd.pLeft = &compLeft;
			exprAnd.pRight = &compRight;
			compLeft.op = TK_GE;
			compLeft.pLeft = &exprX;
			compLeft.pRight = pExpr->x.pList->a[0].pExpr;
			compRight.op = TK_LE;
			compRight.pLeft = &exprX;
			compRight.pRight = pExpr->x.pList->a[1].pExpr;
			exprX.iTable = sqlite3ExprCodeTemp(pParse, &exprX, &regFree1);
			exprX.op = TK_REGISTER;
			if( jumpIfTrue ){
				sqlite3ExprIfTrue(pParse, &exprAnd, dest, jumpIfNull);
			}else{
				sqlite3ExprIfFalse(pParse, &exprAnd, dest, jumpIfNull);
			}
			sqlite3ReleaseTempReg(pParse, regFree1);

			/* Ensure adequate test coverage */
			testcase( jumpIfTrue==0 && jumpIfNull==0 && regFree1==0 );
			testcase( jumpIfTrue==0 && jumpIfNull==0 && regFree1!=0 );
			testcase( jumpIfTrue==0 && jumpIfNull!=0 && regFree1==0 );
			testcase( jumpIfTrue==0 && jumpIfNull!=0 && regFree1!=0 );
			testcase( jumpIfTrue!=0 && jumpIfNull==0 && regFree1==0 );
			testcase( jumpIfTrue!=0 && jumpIfNull==0 && regFree1!=0 );
			testcase( jumpIfTrue!=0 && jumpIfNull!=0 && regFree1==0 );
			testcase( jumpIfTrue!=0 && jumpIfNull!=0 && regFree1!=0 );
	}

	/*
	** Generate code for a boolean expression such that a jump is made
	** to the label "dest" if the expression is true but execution
	** continues straight thru if the expression is false.
	**
	** If the expression evaluates to NULL (neither true nor false), then
	** take the jump if the jumpIfNull flag is SQLITE_JUMPIFNULL.
	**
	** This code depends on the fact that certain token values (ex: TK_EQ)
	** are the same as opcode values (ex: OP_Eq) that implement the corresponding
	** operation.  Special comments in vdbe.c and the mkopcodeh.awk script in
	** the make process cause these values to align.  Assert()s in the code
	** below verify that the numbers are aligned correctly.
	*/
	void sqlite3ExprIfTrue(Parse *pParse, Expr *pExpr, int dest, int jumpIfNull){
		Vdbe *v = pParse->pVdbe;
		int op = 0;
		int regFree1 = 0;
		int regFree2 = 0;
		int r1, r2;

		assert( jumpIfNull==SQLITE_JUMPIFNULL || jumpIfNull==0 );
		if( NEVER(v==0) )     return;  /* Existence of VDBE checked by caller */
		if( NEVER(pExpr==0) ) return;  /* No way this can happen */
		op = pExpr->op;
		switch( op ){
		case TK_AND: {
			int d2 = sqlite3VdbeMakeLabel(v);
			testcase( jumpIfNull==0 );
			sqlite3ExprCachePush(pParse);
			sqlite3ExprIfFalse(pParse, pExpr->pLeft, d2,jumpIfNull^SQLITE_JUMPIFNULL);
			sqlite3ExprIfTrue(pParse, pExpr->pRight, dest, jumpIfNull);
			sqlite3VdbeResolveLabel(v, d2);
			sqlite3ExprCachePop(pParse, 1);
			break;
					 }
		case TK_OR: {
			testcase( jumpIfNull==0 );
			sqlite3ExprIfTrue(pParse, pExpr->pLeft, dest, jumpIfNull);
			sqlite3ExprIfTrue(pParse, pExpr->pRight, dest, jumpIfNull);
			break;
					}
		case TK_NOT: {
			testcase( jumpIfNull==0 );
			sqlite3ExprIfFalse(pParse, pExpr->pLeft, dest, jumpIfNull);
			break;
					 }
		case TK_LT:
		case TK_LE:
		case TK_GT:
		case TK_GE:
		case TK_NE:
		case TK_EQ: {
			assert( TK_LT==OP_Lt );
			assert( TK_LE==OP_Le );
			assert( TK_GT==OP_Gt );
			assert( TK_GE==OP_Ge );
			assert( TK_EQ==OP_Eq );
			assert( TK_NE==OP_Ne );
			testcase( op==TK_LT );
			testcase( op==TK_LE );
			testcase( op==TK_GT );
			testcase( op==TK_GE );
			testcase( op==TK_EQ );
			testcase( op==TK_NE );
			testcase( jumpIfNull==0 );
			r1 = sqlite3ExprCodeTemp(pParse, pExpr->pLeft, &regFree1);
			r2 = sqlite3ExprCodeTemp(pParse, pExpr->pRight, &regFree2);
			codeCompare(pParse, pExpr->pLeft, pExpr->pRight, op,
				r1, r2, dest, jumpIfNull);
			testcase( regFree1==0 );
			testcase( regFree2==0 );
			break;
					}
		case TK_IS:
		case TK_ISNOT: {
			testcase( op==TK_IS );
			testcase( op==TK_ISNOT );
			r1 = sqlite3ExprCodeTemp(pParse, pExpr->pLeft, &regFree1);
			r2 = sqlite3ExprCodeTemp(pParse, pExpr->pRight, &regFree2);
			op = (op==TK_IS) ? TK_EQ : TK_NE;
			codeCompare(pParse, pExpr->pLeft, pExpr->pRight, op,
				r1, r2, dest, SQLITE_NULLEQ);
			testcase( regFree1==0 );
			testcase( regFree2==0 );
			break;
					   }
		case TK_ISNULL:
		case TK_NOTNULL: {
			assert( TK_ISNULL==OP_IsNull );
			assert( TK_NOTNULL==OP_NotNull );
			testcase( op==TK_ISNULL );
			testcase( op==TK_NOTNULL );
			r1 = sqlite3ExprCodeTemp(pParse, pExpr->pLeft, &regFree1);
			sqlite3VdbeAddOp2(v, op, r1, dest);
			testcase( regFree1==0 );
			break;
						 }
		case TK_BETWEEN: {
			testcase( jumpIfNull==0 );
			exprCodeBetween(pParse, pExpr, dest, 1, jumpIfNull);
			break;
						 }
#ifndef SQLITE_OMIT_SUBQUERY
		case TK_IN: {
			int destIfFalse = sqlite3VdbeMakeLabel(v);
			int destIfNull = jumpIfNull ? dest : destIfFalse;
			sqlite3ExprCodeIN(pParse, pExpr, destIfFalse, destIfNull);
			sqlite3VdbeAddOp2(v, OP_Goto, 0, dest);
			sqlite3VdbeResolveLabel(v, destIfFalse);
			break;
					}
#endif
		default: {
			r1 = sqlite3ExprCodeTemp(pParse, pExpr, &regFree1);
			sqlite3VdbeAddOp3(v, OP_If, r1, dest, jumpIfNull!=0);
			testcase( regFree1==0 );
			testcase( jumpIfNull==0 );
			break;
				 }
		}
		sqlite3ReleaseTempReg(pParse, regFree1);
		sqlite3ReleaseTempReg(pParse, regFree2);  
	}

	/*
	** Generate code for a boolean expression such that a jump is made
	** to the label "dest" if the expression is false but execution
	** continues straight thru if the expression is true.
	**
	** If the expression evaluates to NULL (neither true nor false) then
	** jump if jumpIfNull is SQLITE_JUMPIFNULL or fall through if jumpIfNull
	** is 0.
	*/
	void sqlite3ExprIfFalse(Parse *pParse, Expr *pExpr, int dest, int jumpIfNull){
		Vdbe *v = pParse->pVdbe;
		int op = 0;
		int regFree1 = 0;
		int regFree2 = 0;
		int r1, r2;

		assert( jumpIfNull==SQLITE_JUMPIFNULL || jumpIfNull==0 );
		if( NEVER(v==0) ) return; /* Existence of VDBE checked by caller */
		if( pExpr==0 )    return;

		/* The value of pExpr->op and op are related as follows:
		**
		**       pExpr->op            op
		**       ---------          ----------
		**       TK_ISNULL          OP_NotNull
		**       TK_NOTNULL         OP_IsNull
		**       TK_NE              OP_Eq
		**       TK_EQ              OP_Ne
		**       TK_GT              OP_Le
		**       TK_LE              OP_Gt
		**       TK_GE              OP_Lt
		**       TK_LT              OP_Ge
		**
		** For other values of pExpr->op, op is undefined and unused.
		** The value of TK_ and OP_ constants are arranged such that we
		** can compute the mapping above using the following expression.
		** Assert()s verify that the computation is correct.
		*/
		op = ((pExpr->op+(TK_ISNULL&1))^1)-(TK_ISNULL&1);

		/* Verify correct alignment of TK_ and OP_ constants
		*/
		assert( pExpr->op!=TK_ISNULL || op==OP_NotNull );
		assert( pExpr->op!=TK_NOTNULL || op==OP_IsNull );
		assert( pExpr->op!=TK_NE || op==OP_Eq );
		assert( pExpr->op!=TK_EQ || op==OP_Ne );
		assert( pExpr->op!=TK_LT || op==OP_Ge );
		assert( pExpr->op!=TK_LE || op==OP_Gt );
		assert( pExpr->op!=TK_GT || op==OP_Le );
		assert( pExpr->op!=TK_GE || op==OP_Lt );

		switch( pExpr->op ){
		case TK_AND: {
			testcase( jumpIfNull==0 );
			sqlite3ExprIfFalse(pParse, pExpr->pLeft, dest, jumpIfNull);
			sqlite3ExprIfFalse(pParse, pExpr->pRight, dest, jumpIfNull);
			break;
					 }
		case TK_OR: {
			int d2 = sqlite3VdbeMakeLabel(v);
			testcase( jumpIfNull==0 );
			sqlite3ExprCachePush(pParse);
			sqlite3ExprIfTrue(pParse, pExpr->pLeft, d2, jumpIfNull^SQLITE_JUMPIFNULL);
			sqlite3ExprIfFalse(pParse, pExpr->pRight, dest, jumpIfNull);
			sqlite3VdbeResolveLabel(v, d2);
			sqlite3ExprCachePop(pParse, 1);
			break;
					}
		case TK_NOT: {
			testcase( jumpIfNull==0 );
			sqlite3ExprIfTrue(pParse, pExpr->pLeft, dest, jumpIfNull);
			break;
					 }
		case TK_LT:
		case TK_LE:
		case TK_GT:
		case TK_GE:
		case TK_NE:
		case TK_EQ: {
			testcase( op==TK_LT );
			testcase( op==TK_LE );
			testcase( op==TK_GT );
			testcase( op==TK_GE );
			testcase( op==TK_EQ );
			testcase( op==TK_NE );
			testcase( jumpIfNull==0 );
			r1 = sqlite3ExprCodeTemp(pParse, pExpr->pLeft, &regFree1);
			r2 = sqlite3ExprCodeTemp(pParse, pExpr->pRight, &regFree2);
			codeCompare(pParse, pExpr->pLeft, pExpr->pRight, op,
				r1, r2, dest, jumpIfNull);
			testcase( regFree1==0 );
			testcase( regFree2==0 );
			break;
					}
		case TK_IS:
		case TK_ISNOT: {
			testcase( pExpr->op==TK_IS );
			testcase( pExpr->op==TK_ISNOT );
			r1 = sqlite3ExprCodeTemp(pParse, pExpr->pLeft, &regFree1);
			r2 = sqlite3ExprCodeTemp(pParse, pExpr->pRight, &regFree2);
			op = (pExpr->op==TK_IS) ? TK_NE : TK_EQ;
			codeCompare(pParse, pExpr->pLeft, pExpr->pRight, op,
				r1, r2, dest, SQLITE_NULLEQ);
			testcase( regFree1==0 );
			testcase( regFree2==0 );
			break;
					   }
		case TK_ISNULL:
		case TK_NOTNULL: {
			testcase( op==TK_ISNULL );
			testcase( op==TK_NOTNULL );
			r1 = sqlite3ExprCodeTemp(pParse, pExpr->pLeft, &regFree1);
			sqlite3VdbeAddOp2(v, op, r1, dest);
			testcase( regFree1==0 );
			break;
						 }
		case TK_BETWEEN: {
			testcase( jumpIfNull==0 );
			exprCodeBetween(pParse, pExpr, dest, 0, jumpIfNull);
			break;
						 }
#ifndef SQLITE_OMIT_SUBQUERY
		case TK_IN: {
			if( jumpIfNull ){
				sqlite3ExprCodeIN(pParse, pExpr, dest, dest);
			}else{
				int destIfNull = sqlite3VdbeMakeLabel(v);
				sqlite3ExprCodeIN(pParse, pExpr, dest, destIfNull);
				sqlite3VdbeResolveLabel(v, destIfNull);
			}
			break;
					}
#endif
		default: {
			r1 = sqlite3ExprCodeTemp(pParse, pExpr, &regFree1);
			sqlite3VdbeAddOp3(v, OP_IfNot, r1, dest, jumpIfNull!=0);
			testcase( regFree1==0 );
			testcase( jumpIfNull==0 );
			break;
				 }
		}
		sqlite3ReleaseTempReg(pParse, regFree1);
		sqlite3ReleaseTempReg(pParse, regFree2);
	}

	/*
	** Do a deep comparison of two expression trees.  Return 0 if the two
	** expressions are completely identical.  Return 1 if they differ only
	** by a COLLATE operator at the top level.  Return 2 if there are differences
	** other than the top-level COLLATE operator.
	**
	** Sometimes this routine will return 2 even if the two expressions
	** really are equivalent.  If we cannot prove that the expressions are
	** identical, we return 2 just to be safe.  So if this routine
	** returns 2, then you do not really know for certain if the two
	** expressions are the same.  But if you get a 0 or 1 return, then you
	** can be sure the expressions are the same.  In the places where
	** this routine is used, it does not hurt to get an extra 2 - that
	** just might result in some slightly slower code.  But returning
	** an incorrect 0 or 1 could lead to a malfunction.
	*/
	int sqlite3ExprCompare(Expr *pA, Expr *pB){
		if( pA==0||pB==0 ){
			return pB==pA ? 0 : 2;
		}
		assert( !ExprHasAnyProperty(pA, EP_TokenOnly|EP_Reduced) );
		assert( !ExprHasAnyProperty(pB, EP_TokenOnly|EP_Reduced) );
		if( ExprHasProperty(pA, EP_xIsSelect) || ExprHasProperty(pB, EP_xIsSelect) ){
			return 2;
		}
		if( (pA->flags & EP_Distinct)!=(pB->flags & EP_Distinct) ) return 2;
		if( pA->op!=pB->op ){
			if( pA->op==TK_COLLATE && sqlite3ExprCompare(pA->pLeft, pB)<2 ){
				return 1;
			}
			if( pB->op==TK_COLLATE && sqlite3ExprCompare(pA, pB->pLeft)<2 ){
				return 1;
			}
			return 2;
		}
		if( sqlite3ExprCompare(pA->pLeft, pB->pLeft) ) return 2;
		if( sqlite3ExprCompare(pA->pRight, pB->pRight) ) return 2;
		if( sqlite3ExprListCompare(pA->x.pList, pB->x.pList) ) return 2;
		if( pA->iTable!=pB->iTable || pA->iColumn!=pB->iColumn ) return 2;
		if( ExprHasProperty(pA, EP_IntValue) ){
			if( !ExprHasProperty(pB, EP_IntValue) || pA->u.iValue!=pB->u.iValue ){
				return 2;
			}
		}else if( pA->op!=TK_COLUMN && ALWAYS(pA->op!=TK_AGG_COLUMN) && pA->u.zToken){
			if( ExprHasProperty(pB, EP_IntValue) || NEVER(pB->u.zToken==0) ) return 2;
			if( strcmp(pA->u.zToken,pB->u.zToken)!=0 ){
				return pA->op==TK_COLLATE ? 1 : 2;
			}
		}
		return 0;
	}

	/*
	** Compare two ExprList objects.  Return 0 if they are identical and 
	** non-zero if they differ in any way.
	**
	** This routine might return non-zero for equivalent ExprLists.  The
	** only consequence will be disabled optimizations.  But this routine
	** must never return 0 if the two ExprList objects are different, or
	** a malfunction will result.
	**
	** Two NULL pointers are considered to be the same.  But a NULL pointer
	** always differs from a non-NULL pointer.
	*/
	int sqlite3ExprListCompare(ExprList *pA, ExprList *pB){
		int i;
		if( pA==0 && pB==0 ) return 0;
		if( pA==0 || pB==0 ) return 1;
		if( pA->nExpr!=pB->nExpr ) return 1;
		for(i=0; i<pA->nExpr; i++){
			Expr *pExprA = pA->a[i].pExpr;
			Expr *pExprB = pB->a[i].pExpr;
			if( pA->a[i].sortOrder!=pB->a[i].sortOrder ) return 1;
			if( sqlite3ExprCompare(pExprA, pExprB) ) return 1;
		}
		return 0;
	}

	/*
	** An instance of the following structure is used by the tree walker
	** to count references to table columns in the arguments of an 
	** aggregate function, in order to implement the
	** sqlite3FunctionThisSrc() routine.
	*/
	struct SrcCount {
		SrcList *pSrc;   /* One particular FROM clause in a nested query */
		int nThis;       /* Number of references to columns in pSrcList */
		int nOther;      /* Number of references to columns in other FROM clauses */
	};

	/*
	** Count the number of references to columns.
	*/
	static int exprSrcCount(Walker *pWalker, Expr *pExpr){
		/* The NEVER() on the second term is because sqlite3FunctionUsesThisSrc()
		** is always called before sqlite3ExprAnalyzeAggregates() and so the
		** TK_COLUMNs have not yet been converted into TK_AGG_COLUMN.  If
		** sqlite3FunctionUsesThisSrc() is used differently in the future, the
		** NEVER() will need to be removed. */
		if( pExpr->op==TK_COLUMN || NEVER(pExpr->op==TK_AGG_COLUMN) ){
			int i;
			struct SrcCount *p = pWalker->u.pSrcCount;
			SrcList *pSrc = p->pSrc;
			for(i=0; i<pSrc->nSrc; i++){
				if( pExpr->iTable==pSrc->a[i].iCursor ) break;
			}
			if( i<pSrc->nSrc ){
				p->nThis++;
			}else{
				p->nOther++;
			}
		}
		return WRC_Continue;
	}

	/*
	** Determine if any of the arguments to the pExpr Function reference
	** pSrcList.  Return true if they do.  Also return true if the function
	** has no arguments or has only constant arguments.  Return false if pExpr
	** references columns but not columns of tables found in pSrcList.
	*/
	int sqlite3FunctionUsesThisSrc(Expr *pExpr, SrcList *pSrcList){
		Walker w;
		struct SrcCount cnt;
		assert( pExpr->op==TK_AGG_FUNCTION );
		memset(&w, 0, sizeof(w));
		w.xExprCallback = exprSrcCount;
		w.u.pSrcCount = &cnt;
		cnt.pSrc = pSrcList;
		cnt.nThis = 0;
		cnt.nOther = 0;
		sqlite3WalkExprList(&w, pExpr->x.pList);
		return cnt.nThis>0 || cnt.nOther==0;
	}

	/*
	** Add a new element to the pAggInfo->aCol[] array.  Return the index of
	** the new element.  Return a negative number if malloc fails.
	*/
	static int addAggInfoColumn(sqlite3 *db, AggInfo *pInfo){
		int i;
		pInfo->aCol = sqlite3ArrayAllocate(
			db,
			pInfo->aCol,
			sizeof(pInfo->aCol[0]),
			&pInfo->nColumn,
			&i
			);
		return i;
	}    

	/*
	** Add a new element to the pAggInfo->aFunc[] array.  Return the index of
	** the new element.  Return a negative number if malloc fails.
	*/
	static int addAggInfoFunc(sqlite3 *db, AggInfo *pInfo){
		int i;
		pInfo->aFunc = sqlite3ArrayAllocate(
			db, 
			pInfo->aFunc,
			sizeof(pInfo->aFunc[0]),
			&pInfo->nFunc,
			&i
			);
		return i;
	}    

	/*
	** This is the xExprCallback for a tree walker.  It is used to
	** implement sqlite3ExprAnalyzeAggregates().  See sqlite3ExprAnalyzeAggregates
	** for additional information.
	*/
	static int analyzeAggregate(Walker *pWalker, Expr *pExpr){
		int i;
		NameContext *pNC = pWalker->u.pNC;
		Parse *pParse = pNC->pParse;
		SrcList *pSrcList = pNC->pSrcList;
		AggInfo *pAggInfo = pNC->pAggInfo;

		switch( pExpr->op ){
		case TK_AGG_COLUMN:
		case TK_COLUMN: {
			testcase( pExpr->op==TK_AGG_COLUMN );
			testcase( pExpr->op==TK_COLUMN );
			/* Check to see if the column is in one of the tables in the FROM
			** clause of the aggregate query */
			if( ALWAYS(pSrcList!=0) ){
				struct SrcList_item *pItem = pSrcList->a;
				for(i=0; i<pSrcList->nSrc; i++, pItem++){
					struct AggInfo_col *pCol;
					assert( !ExprHasAnyProperty(pExpr, EP_TokenOnly|EP_Reduced) );
					if( pExpr->iTable==pItem->iCursor ){
						/* If we reach this point, it means that pExpr refers to a table
						** that is in the FROM clause of the aggregate query.  
						**
						** Make an entry for the column in pAggInfo->aCol[] if there
						** is not an entry there already.
						*/
						int k;
						pCol = pAggInfo->aCol;
						for(k=0; k<pAggInfo->nColumn; k++, pCol++){
							if( pCol->iTable==pExpr->iTable &&
								pCol->iColumn==pExpr->iColumn ){
									break;
							}
						}
						if( (k>=pAggInfo->nColumn)
							&& (k = addAggInfoColumn(pParse->db, pAggInfo))>=0 
							){
								pCol = &pAggInfo->aCol[k];
								pCol->pTab = pExpr->pTab;
								pCol->iTable = pExpr->iTable;
								pCol->iColumn = pExpr->iColumn;
								pCol->iMem = ++pParse->nMem;
								pCol->iSorterColumn = -1;
								pCol->pExpr = pExpr;
								if( pAggInfo->pGroupBy ){
									int j, n;
									ExprList *pGB = pAggInfo->pGroupBy;
									struct ExprList_item *pTerm = pGB->a;
									n = pGB->nExpr;
									for(j=0; j<n; j++, pTerm++){
										Expr *pE = pTerm->pExpr;
										if( pE->op==TK_COLUMN && pE->iTable==pExpr->iTable &&
											pE->iColumn==pExpr->iColumn ){
												pCol->iSorterColumn = j;
												break;
										}
									}
								}
								if( pCol->iSorterColumn<0 ){
									pCol->iSorterColumn = pAggInfo->nSortingColumn++;
								}
						}
						/* There is now an entry for pExpr in pAggInfo->aCol[] (either
						** because it was there before or because we just created it).
						** Convert the pExpr to be a TK_AGG_COLUMN referring to that
						** pAggInfo->aCol[] entry.
						*/
						ExprSetIrreducible(pExpr);
						pExpr->pAggInfo = pAggInfo;
						pExpr->op = TK_AGG_COLUMN;
						pExpr->iAgg = (i16)k;
						break;
					} /* endif pExpr->iTable==pItem->iCursor */
				} /* end loop over pSrcList */
			}
			return WRC_Prune;
						}
		case TK_AGG_FUNCTION: {
			if( (pNC->ncFlags & NC_InAggFunc)==0
				&& pWalker->walkerDepth==pExpr->op2
				){
					/* Check to see if pExpr is a duplicate of another aggregate 
					** function that is already in the pAggInfo structure
					*/
					struct AggInfo_func *pItem = pAggInfo->aFunc;
					for(i=0; i<pAggInfo->nFunc; i++, pItem++){
						if( sqlite3ExprCompare(pItem->pExpr, pExpr)==0 ){
							break;
						}
					}
					if( i>=pAggInfo->nFunc ){
						/* pExpr is original.  Make a new entry in pAggInfo->aFunc[]
						*/
						u8 enc = ENC(pParse->db);
						i = addAggInfoFunc(pParse->db, pAggInfo);
						if( i>=0 ){
							assert( !ExprHasProperty(pExpr, EP_xIsSelect) );
							pItem = &pAggInfo->aFunc[i];
							pItem->pExpr = pExpr;
							pItem->iMem = ++pParse->nMem;
							assert( !ExprHasProperty(pExpr, EP_IntValue) );
							pItem->pFunc = sqlite3FindFunction(pParse->db,
								pExpr->u.zToken, sqlite3Strlen30(pExpr->u.zToken),
								pExpr->x.pList ? pExpr->x.pList->nExpr : 0, enc, 0);
							if( pExpr->flags & EP_Distinct ){
								pItem->iDistinct = pParse->nTab++;
							}else{
								pItem->iDistinct = -1;
							}
						}
					}
					/* Make pExpr point to the appropriate pAggInfo->aFunc[] entry
					*/
					assert( !ExprHasAnyProperty(pExpr, EP_TokenOnly|EP_Reduced) );
					ExprSetIrreducible(pExpr);
					pExpr->iAgg = (i16)i;
					pExpr->pAggInfo = pAggInfo;
					return WRC_Prune;
			}else{
				return WRC_Continue;
			}
							  }
		}
		return WRC_Continue;
	}
	static int analyzeAggregatesInSelect(Walker *pWalker, Select *pSelect){
		UNUSED_PARAMETER(pWalker);
		UNUSED_PARAMETER(pSelect);
		return WRC_Continue;
	}

	/*
	** Analyze the pExpr expression looking for aggregate functions and
	** for variables that need to be added to AggInfo object that pNC->pAggInfo
	** points to.  Additional entries are made on the AggInfo object as
	** necessary.
	**
	** This routine should only be called after the expression has been
	** analyzed by sqlite3ResolveExprNames().
	*/
	void sqlite3ExprAnalyzeAggregates(NameContext *pNC, Expr *pExpr){
		Walker w;
		memset(&w, 0, sizeof(w));
		w.xExprCallback = analyzeAggregate;
		w.xSelectCallback = analyzeAggregatesInSelect;
		w.u.pNC = pNC;
		assert( pNC->pSrcList!=0 );
		sqlite3WalkExpr(&w, pExpr);
	}

	/*
	** Call sqlite3ExprAnalyzeAggregates() for every expression in an
	** expression list.  Return the number of errors.
	**
	** If an error is found, the analysis is cut short.
	*/
	void sqlite3ExprAnalyzeAggList(NameContext *pNC, ExprList *pList){
		struct ExprList_item *pItem;
		int i;
		if( pList ){
			for(pItem=pList->a, i=0; i<pList->nExpr; i++, pItem++){
				sqlite3ExprAnalyzeAggregates(pNC, pItem->pExpr);
			}
		}
	}

	/*
	** Allocate a single new register for use to hold some intermediate result.
	*/
	int sqlite3GetTempReg(Parse *pParse){
		if( pParse->nTempReg==0 ){
			return ++pParse->nMem;
		}
		return pParse->aTempReg[--pParse->nTempReg];
	}

	/*
	** Deallocate a register, making available for reuse for some other
	** purpose.
	**
	** If a register is currently being used by the column cache, then
	** the dallocation is deferred until the column cache line that uses
	** the register becomes stale.
	*/
	void sqlite3ReleaseTempReg(Parse *pParse, int iReg){
		if( iReg && pParse->nTempReg<ArraySize(pParse->aTempReg) ){
			int i;
			struct yColCache *p;
			for(i=0, p=pParse->aColCache; i<SQLITE_N_COLCACHE; i++, p++){
				if( p->iReg==iReg ){
					p->tempReg = 1;
					return;
				}
			}
			pParse->aTempReg[pParse->nTempReg++] = iReg;
		}
	}

	/*
	** Allocate or deallocate a block of nReg consecutive registers
	*/
	int sqlite3GetTempRange(Parse *pParse, int nReg){
		int i, n;
		i = pParse->iRangeReg;
		n = pParse->nRangeReg;
		if( nReg<=n ){
			assert( !usedAsColumnCache(pParse, i, i+n-1) );
			pParse->iRangeReg += nReg;
			pParse->nRangeReg -= nReg;
		}else{
			i = pParse->nMem+1;
			pParse->nMem += nReg;
		}
		return i;
	}
	void sqlite3ReleaseTempRange(Parse *pParse, int iReg, int nReg){
		sqlite3ExprCacheRemove(pParse, iReg, nReg);
		if( nReg>pParse->nRangeReg ){
			pParse->nRangeReg = nReg;
			pParse->iRangeReg = iReg;
		}
	}

	/*
	** Mark all temporary registers as being unavailable for reuse.
	*/
	void sqlite3ClearTempRegCache(Parse *pParse){
		pParse->nTempReg = 0;
		pParse->nRangeReg = 0;
	}
