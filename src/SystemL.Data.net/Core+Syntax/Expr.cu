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

	__device__ static uint8 _codeSubselect_SortOrder = 0; // Fake aSortOrder for keyInfo
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
			Expr *left = expr->Left; // the LHS of the IN operator
			if (mayHaveNull)
				v->AddOp2(OP_Null, 0, mayHaveNull);
			AFF affinity = left->Affinity(); // Affinity of the LHS of the IN

			// Whether this is an 'x IN(SELECT...)' or an 'x IN(<exprlist>)' expression it is handled the same way.  An ephemeral table is 
			// filled with single-field index keys representing the results from the SELECT or the <exprlist>.
			//
			// If the 'x' expression is a column value, or the SELECT... statement returns a column value, then the affinity of that
			// column is used to build the index keys. If both 'x' and the SELECT... statement are columns, then numeric affinity is used
			// if either column has NUMERIC or INTEGER affinity. If neither 'x' nor the SELECT... statement are columns, then numeric affinity is used.
			expr->TableIdx = Tabs++;
			int addr = v->AddOp2(OP_OpenEphemeral, pExpr->iTable, !isRowid); // Address of OP_OpenEphemeral instruction
			if (!mayHaveNull) v->ChangeP5(BTREE_UNORDERED);
			KeyInfo keyInfo; // Keyinfo for the generated table
			_memset(&keyInfo, 0, sizeof(keyInfo));
			keyInfo.Fields = 1;
			keyInfo.SortOrders = &_codeSubselect_SortOrder;

			if (ExprHasProperty(expr, EP_xIsSelect))
			{
				// Case 1:     expr IN (SELECT ...)
				// Generate code to write the results of the select into the temporary table allocated and opened above.
				_assert(!isRowid);
				SelectDest dest;
				SelectDestInit(&dest, SRT_Set, pExpr->iTable);
				dest.AffSdst = affinity;
				_assert((expr->TableIdx&0x0000FFFF) == expr->TableIdx);
				expr->x.Select->LimitId = 0;
				if (Parse::Select(expr->x.Select, &dest))
					return 0;
				ExprList *elist = expr->x.Select->EList;
				if (SysEx_ALWAYS(elist && elist->Exprs > 0))
					keyInfo.Colls[0] = BinaryCompareCollSeq(expr->Left, elist->Ids[0].Expr);
			}
			else if (SysEx_ALWAYS(expr->x.List))
			{
				// Case 2:     expr IN (exprlist)
				// For each expression, build an index key from the evaluation and store it in the temporary table. If <expr> is a column, then use
				// that columns affinity when building index keys. If <expr> is not a column, use numeric affinity.
				ExprList *list = expr->x.List;
				if (!affinity)
					affinity = AFF_NONE;
				keyInfo.Colls[0] = ExprCollSeq(expr->Left);
				keyInfo.SortOrders = &_codeSubselect_SortOrder;

				// Loop through each expression in <exprlist>.
				int r1 = GetTempReg();
				int r2 = GetTempReg();
				v->AddOp2(OP_Null, 0, r2);
				int i;
				ExprList::ExprListItem *item;
				for (i = list->Exprs, item = list->Ids; i > 0; i--, item++)
				{
					Expr *e2 = item->Expr;
					// If the expression is not constant then we will need to disable the test that was generated above that makes sure
					// this code only executes once.  Because for a non-constant expression we need to rerun this code each time.
					if (testAddr >= 0 && !e2->IsConstant())
					{
						v->ChangeToNoop(testAddr);
						testAddr = -1;
					}
					// Evaluate the expression and insert it into the temp table
					int valToIns;
					if (isRowid && e2->IsInteger(&valToIns))
						v->AddOp3(OP_InsertInt, expr->TableIdx, r2, valToIns);
					else
					{
						int r3 = ExprCodeTarget(e2, r1);
						if (isRowid)
						{
							v->AddOp2(OP_MustBeInt, r3, v->CurrentAddr()+2);
							v->AddOp3(OP_Insert, pExpr->iTable, r2, r3);
						}
						else
						{
							v->AddOp4(OP_MakeRecord, r3, 1, r2, &affinity, 1);
							ExprCacheAffinityChange(r3, 1);
							v->AddOp2(OP_IdxInsert, expr->TableIdx, r2);
						}
					}
				}
				ReleaseTempReg(r1);
				ReleaseTempReg(r2);
			}
			if (!isRowid)
				v->ChangeP4(addr, (void *)&keyInfo, Vdbe::P4T_KEYINFO);
			break; }

		case TK_EXISTS:
		case TK_SELECT:
		default: {
			ASSERTCOVERAGE(expr->OP == TK_EXISTS);
			ASSERTCOVERAGE(expr->OP == TK_SELECT);
			_assert(expr->OP == TK_EXISTS || expr->OP == TK_SELECT);
			_assert(ExprHasProperty(expr, EP_xIsSelect));

			// If this has to be a scalar SELECT.  Generate code to put the value of this select in a memory cell and record the number
			// of the memory cell in iColumn.  If this is an EXISTS, write an integer 0 (not exists) or 1 (exists) into a memory cell
			// and record that memory cell in iColumn.
			Select *sel = expr->x.Select; // SELECT statement to encode
			SelectDest dest; // How to deal with SELECt result
			SelectDestInit(&dest, 0, ++Mems);
			if (expr->OP == TK_SELECT)
			{
				dest.Dest = SRT_Mem;
				v->AddOp2(OP_Null, 0, dest.SDParmId);
				VdbeComment(v, "Init subquery result");
			}
			else
			{
				dest.Dest = SRT_Exists;
				v->AddOp2(OP_Integer, 0, dest.SDParmId);
				VdbeComment(v, "Init EXISTS result");
			}
			Expr::Delete(Ctx, sel->Limit);
			sel->Limit = PExpr(TK_INTEGER, 0, 0, &sqlite3IntTokens[1]);
			sel->LimitId = 0;
			if (Select(sel, &dest))
				return 0;
			reg = dest.SDParmId;
			ExprSetIrreducible(expr);
			break; }
		}
		if (testAddr >= 0)
			v->JumpHere(testAddr);
		ExprCachePop(1);
		return reg;
	}

	__device__ void Parse::ExprCodeIN(Expr *expr, int destIfFalse, int destIfNull)
	{
		// Compute the RHS. After this step, the table with cursor expr->TableId will contains the values that make up the RHS.
		Vdbe *v = V; // Statement under construction
		_assert(v); // OOM detected prior to this routine
		VdbeNoopComment(v, "begin IN expr");
		int rhsHasNull = 0;  // Register that is true if RHS contains NULL values
		int type = sqlite3FindInIndex(expr, &rhsHasNull); // Type of the RHS

		// Figure out the affinity to use to create a key from the results of the expression. affinityStr stores a static string suitable for P4 of OP_MakeRecord.
		AFF affinity = ComparisonAffinity(expr); // Comparison affinity to use

		// Code the LHS, the <expr> from "<expr> IN (...)".
		ExprCachePush();
		int r1 = GetTempReg(); // Temporary use register
		ExprCode(expr->Left, r1);

		// If the LHS is NULL, then the result is either false or NULL depending on whether the RHS is empty or not, respectively.
		if (destIfNull == destIfFalse)
		{
			// Shortcut for the common case where the false and NULL outcomes are the same.
			v->AddOp2(OP_IsNull, r1, destIfNull);
		}
		else
		{
			int addr1 = v->AddOp1(OP_NotNull, r1);
			v->AddOp2(OP_Rewind, expr->TableIdx, destIfFalse);
			v->AddOp2(OP_Goto, 0, destIfNull);
			v->JumpHere(addr1);
		}

		if (type == IN_INDEX_ROWID)
		{
			// In this case, the RHS is the ROWID of table b-tree
			v->AddOp2(OP_MustBeInt, r1, destIfFalse);
			v->AddOp3(OP_NotExists, pExpr->iTable, destIfFalse, r1);
		}
		else
		{
			// In this case, the RHS is an index b-tree.
			v->AddOp4(OP_Affinity, r1, 1, 0, &affinity, 1);

			// If the set membership test fails, then the result of the  "x IN (...)" expression must be either 0 or NULL. If the set
			// contains no NULL values, then the result is 0. If the set contains one or more NULL values, then the result of the
			// expression is also NULL.
			if (rhsHasNull == 0 || destIfFalse == destIfNull)
			{
				// This branch runs if it is known at compile time that the RHS cannot contain NULL values. This happens as the result
				// of a "NOT NULL" constraint in the database schema.
				//
				// Also run this branch if NULL is equivalent to FALSE for this particular IN operator.
				v->AddOp4Int(OP_NotFound, expr->TableIdx, destIfFalse, r1, 1);
			}
			else
			{
				// In this branch, the RHS of the IN might contain a NULL and the presence of a NULL on the RHS makes a difference in the outcome.

				// First check to see if the LHS is contained in the RHS. If so, then the presence of NULLs in the RHS does not matter, so jump
				// over all of the code that follows.
				int j1 = v->AddOp4Int(OP_Found, expr->TableIdx, 0, r1, 1);

				// Here we begin generating code that runs if the LHS is not contained within the RHS.  Generate additional code that
				// tests the RHS for NULLs.  If the RHS contains a NULL then jump to destIfNull.  If there are no NULLs in the RHS then
				// jump to destIfFalse.
				int j2 = v->AddOp1(OP_NotNull, rhsHasNull);
				int j3 = v->AddOp4Int(OP_Found, expr->TableIdx, 0, rhsHasNull, 1);
				v->AddOp2(OP_Integer, -1, rhsHasNull);
				v->JumpHere(j3);
				v->AddOp2(OP_AddImm, rhsHasNull, 1);
				v->JumpHere(j2);

				// Jump to the appropriate target depending on whether or not the RHS contains a NULL
				v->AddOp2(OP_If, rhsHasNull, destIfNull);
				v->AddOp2(OP_Goto, 0, destIfFalse);

				// The OP_Found at the top of this branch jumps here when true, causing the overall IN expression evaluation to fall through.
				v->JumpHere(j1);
			}
		}
		ReleaseTempReg(r1);
		ExprCachePop(1);
		VdbeComment(v, "end IN expr");
	}

#endif
#pragma endregion

	__device__ static char *Dup8bytes(Vdbe *v, const char *in)
	{
		char *out = (char *)SysEx::TagAlloc(v->Db, 8);
		if (out)
			_memcpy(out, in, 8);
		return out;
	}

#ifndef OMIT_FLOATING_POINT
	//
	// Generate an instruction that will put the floating point value described by z[0..n-1] into register iMem.
	//
	// The z[] string will probably not be zero-terminated.  But the z[n] character is guaranteed to be something that does not look
	// like the continuation of the number.
	__device__ static void CodeReal(Vdbe *v, const char *z, int negateFlag, int mem)
	{
		if (SysEx_ALWAYS(z))
		{
			double value;
			ConvertEx::Atof(z, &value, _strlen30(z), TEXTENCODE_UTF8);
			_assert(!_isNaN(value)); // The new AtoF never returns NaN
			if (negateFlag) value = -value;
			char *value2 = Dup8bytes(v, (char *)&value);
			v->AddOp4(OP_Real, 0, mem, 0, value2, Vdbe::P4T_REAL);
		}
	}
#endif

	__device__ static void CodeInteger(Parse *parse, Expr *expr, int negFlag, int mem)
	{
		Vdbe *v = parse->V;
		if (expr->Flags & EP_IntValue)
		{
			int i = expr->u.I;
			_assert(i >= 0);
			if (negFlag) i = -i;
			v->AddOp2(OP_Integer, i, mem);
		}
		else
		{
			const char *z = expr->u.Token;
			_assert(z);
			int64 value;
			int c = ConvertEx::Atoi64(z, &value, _strlen30(z), TEXTENCODE_UTF8);
			if (c == 0 || (c == 2 && negFlag))
			{
				if (negFlag) { value = (c == 2 ? SMALLEST_INT64 : -value); }
				char *value2 = Dup8bytes(v, (char *)&value);
				v->AddOp4(OP_Int64, 0, mem, 0, value2, Vdbe::P4T_INT64);
			}
			else
			{
#ifdef OMIT_FLOATING_POINT
				parse->ErrorMsg("oversized integer: %s%s", negFlag ? "-" : "", z);
#else
				CodeReal(v, z, negFlag, mem);
#endif
			}
		}
	}

#pragma region Column Cache

	__device__ static void CacheEntryClear(Parse *parse, Parse::ColCache *p)
	{
		if (p->TempReg)
		{
			if (parse->TempReg.length < __arrayStaticLength(parse->TempReg.data))
				parse->TempReg[parse->TempReg.length++] = p->Reg;
			p->TempReg = 0;
		}
	}

	__device__ void Parse::ExprCacheStore(int table, int column, int reg)
	{
		_assert(reg > 0);  // Register numbers are always positive
		_assert(column >= -1 && column < 32768);  // Finite column numbers
		// The SQLITE_ColumnCache flag disables the column cache.  This is used for testing only - to verify that SQLite always gets the same answer
		// with and without the column cache.
		if (OptimizationDisabled(Ctx, SQLITE_ColumnCache))
			return;
		// First replace any existing entry.
		// Actually, the way the column cache is currently used, we are guaranteed that the object will never already be in cache.  Verify this guarantee.
		int i;
		ColCache *p;
#ifndef NDEBUG
		for (i = 0, p = ColCaches; i < N_COLCACHE; i++, p++)
			_assert(p->Reg == 0 || p->Table != table || p->Column != column);
#endif
		// Find an empty slot and replace it
		for (i = 0, p = ColCaches; i < N_COLCACHE; i++, p++)
		{
			if (p->Reg == 0)
			{
				p->Level = CacheLevel;
				p->Table = table;
				p->Column = column;
				p->Reg = reg;
				p->TempReg = 0;
				p->Lru = CacheCnt++;
				return;
			}
		}
		// Replace the last recently used
		int minLru = 0x7fffffff;
		int idxLru = -1;
		for (i = 0, p = ColCaches; i < N_COLCACHE; i++, p++)
		{
			if (p->Lru < minLru)
			{
				idxLru = i;
				minLru = p->Lru;
			}
		}
		if (SysEx_ALWAYS(idxLru >= 0))
		{
			p = &ColCaches[idxLru];
			p->Level = CacheLevel;
			p->Table = table;
			p->Column = column;
			p->Reg = reg;
			p->TempReg = 0;
			p->Lru = CacheCnt++;
			return;
		}
	}

	__device__ void Parse::ExprCacheRemove(int reg, int regs)
	{
		int last = reg + regs - 1;
		int i;
		ColCache *p;
		for (i = 0, p = ColCaches; i < N_COLCACHE; i++, p++)
		{
			int r = p->Reg;
			if (r >= reg && r <= last)
			{
				CacheEntryClear(this, p);
				p->Reg = 0;
			}
		}
	}

	__device__ void Parse::ExprCachePush()
	{
		CacheLevel++;
	}

	__device__ void Parse::ExprCachePop(int n)
	{
		_assert(n > 0);
		_assert(CacheLevel >= n);
		CacheLevel -= n;
		int i;
		ColCache *p;
		for (i = 0, p = ColCaches; i < N_COLCACHE; i++, p++)
		{
			if (p->Reg && p->Level > CacheLevel)
			{
				CacheEntryClear(this, p);
				p->Reg = 0;
			}
		}
	}

	__device__ void Parse::ExprCachePinRegister(int reg)
	{
		int i;
		ColCache *p;
		for (i =0 , p = ColCaches; i < N_COLCACHE; i++, p++)
		{
			if (p->Reg == reg)
				p->TempReg = 0;
		}
	}

	__device__ void Expr::CodeGetColumnOfTable(Vdbe *v, Table *table, int tableId, int column, int regOut)
	{
		if (column < 0 || column == table->PKey)
			v->AddOp2(OP_Rowid, tableId, regOut);
		else
		{
			int op = (IsVirtual(table) ? OP_VColumn : OP_Column);
			v->AddOp3(op, tableId, column, regOut);
		}
		if (column >= 0)
			ColumnDefault(v, table, column, regOut);
	}

	__device__ int Parse::ExprCodeGetColumn(Table *table, int column, int tableId, int reg, uint8 p5)
	{
		Vdbe *v = V;
		int i;
		ColCache *p;
		for (i = 0, p = ColCaches; i < N_COLCACHE; i++, p++)
		{
			if (p->Reg > 0 && p->Table == tableId && p->Column == column)
			{
				p->Lru = CacheCnt++;
				ExprCachePinRegister(p->Reg);
				return p->Reg;
			}
		}  
		_assert(v);
		Expr::CodeGetColumnOfTable(v, table, tableId, column, reg);
		if (p5)
			v->ChangeP5(p5);
		else
			ExprCacheStore(tableId, column, reg);
		return reg;
	}

	__device__ void Parse::ExprCacheClear()
	{
		int i;
		ColCache *p;
		for (i = 0, p = ColCaches; i < N_COLCACHE; i++, p++)
		{
			if (p->Reg)
			{
				CacheEntryClear(this, p);
				p->Reg = 0;
			}
		}
	}

	__device__ void Parse::ExprCacheAffinityChange(int start, int count) { ExprCacheRemove(start, count); }

	__device__ void Parse::ExprCodeMove(int from, int to, int regs)
	{
		assert(from >= to + regs || from + reg <= to);
		V->AddOp3(OP_Move, from, to, regs - 1);
		int i;
		ColCache *p;
		for (i = 0, p = ColCaches; i < N_COLCACHE; i++, p++)
		{
			int r = p->Reg;
			if (r >= from && r < from + regs)
				p->Reg += to - from;
		}
	}

#pragma endregion

#if defined(DEBUG) || defined(COVERAGE_TEST)
	__device__ static int UsedAsColumnCache(Parse *parse, int from, int to)
	{
		int i;
		Parse::ColCache *p;
		for (i = 0, p = parse->ColCaches; i < N_COLCACHE; i++, p++)
		{
			int r = p->Reg;
			if (r >= from && r <= to)
				return true;
		}
		return false;
	}
#endif


	__device__ int Parse::ExprCodeTarget(Expr *expr, int target)
	{
		_assert(target > 0 && target <= Mems);
		Context *ctx = Ctx; // The database connection
		Vdbe *v = V; // The VM under construction
		if (!v)
		{
			_assert(ctx->MallocFailed);
			return 0;
		}

		int inReg = target;       // Results stored in register inReg
		int regFree1 = 0;         // If non-zero free this temporary register
		int regFree2 = 0;         // If non-zero free this temporary register
		int r1, r2, r3, r4;       // Various register numbers

		int op = (!expr ? TK_NULL : expr->OP); // The opcode being coded
		switch (op)
		{
		case TK_AGG_COLUMN: {
			AggInfo *aggInfo = expr->AggInfo;
			AggInfo::AggInfoColumn *col = &aggInfo->Cols[expr->Agg];
			if (!aggInfo->DirectMode)
			{
				_assert(col->Mem > 0);
				inReg = col->Mem;
				break;
			}
			else if (aggInfo->UseSortingIdx)
			{
				v->AddOp3(OP_Column, aggInfo->SortingIdxPTab, col->SorterColumn, target);
				break;
			} }
							// Otherwise, fall thru into the TK_COLUMN case
		case TK_COLUMN: {
			if (expr->Table < 0)
			{
				// This only happens when coding check constraints
				_assert(CkBase > 0);
				inReg = expr->ColumnIdx + CkBase;
			}
			else
				inReg = ExprCodeGetColumn(expr->Table, expr->ColumnIdx, expr->TableIdx, target, expr->OP2);
			break; }

		case TK_INTEGER: {
			CodeInteger(this, expr, 0, target);
			break; }

#ifndef OMIT_FLOATING_POINT
		case TK_FLOAT: {
			_assert(!ExprHasProperty(pExpr, EP_IntValue));
			CodeReal(v, pExpr->u.Token, 0, target);
			break; }
#endif

		case TK_STRING: {
			_assert(!ExprHasProperty(expr, EP_IntValue));
			v->AddOp4(OP_String8, 0, target, 0, expr->u.Token, 0);
			break; }

		case TK_NULL: {
			v->AddOp2(OP_Null, 0, target);
			break; }

#ifndef OMIT_BLOB_LITERAL
		case TK_BLOB: {
			_assert(!ExprHasProperty(expr, EP_IntValue));
			_assert(expr->u.Token[0] == 'x' || expr->u.Token[0] == 'X');
			_assert(expr->u.Token[1] == '\'' );
			const char *z = &pExpr->u.Token[2];
			int n = _strlen30(z) - 1;
			_assert(z[n] == '\'');
			char *blob = ConvertEx::HexToBlob(v->Db, z, n);
			v->AddOp4(OP_Blob, n/2, target, 0, blob, Vdbe::P4T_DYNAMIC);
			break; }
#endif

		case TK_VARIABLE: {
			_assert(!ExprHasProperty(expr, EP_IntValue));
			_assert(expr->u.Token != 0);
			_assert(expr->u.Token[0] != 0);
			v->AddOp2(OP_Variable, expr->ColumnIdx, target);
			if (expr->u.Token[1] != 0)
			{
				_assert(expr->u.Token[0] == '?' || !_strcmp(expr->u.Token, Vars[expr->ColumnIdx-1]));
				v->ChangeP4(-1, Vars[expr->ColumnIdx-1], Vdbe::P4T_STATIC);
			}
			break; }

		case TK_REGISTER: {
			inReg = expr->TableIdx;
			break; }

		case TK_AS: {
			inReg = ExprCodeTarget(expr->Left, target);
			break; }

#ifndef OMIT_CAST
		case TK_CAST: {
			// Expressions of the form:   CAST(pLeft AS token)
			inReg = ExprCodeTarget(expr->Left, target);
			_assert(!ExprHasProperty(expr, EP_IntValue));
			AFF aff = AffinityType(expr->u.Token);
			int toOP = (int)aff - AFF_TEXT + OP_ToText;
			assert(toOP == OP_ToText	|| aff != AFF_TEXT);
			assert(toOP == OP_ToBlob    || aff != AFF_NONE);
			assert(toOP == OP_ToNumeric || aff != AFF_NUMERIC);
			assert(toOP == OP_ToInt     || aff != AFF_INTEGER);
			assert(toOP == OP_ToReal    || aff != AFF_REAL);
			ASSERTCOVERAGE(toOP == OP_ToText);
			ASSERTCOVERAGE(toOP == OP_ToBlob);
			ASSERTCOVERAGE(toOP == OP_ToNumeric);
			ASSERTCOVERAGE(toOP == OP_ToInt);
			ASSERTCOVERAGE(toOP == OP_ToReal);
			if (inReg != target)
			{
				v->AddOp2(OP_SCopy, inReg, target);
				inReg = target;
			}
			v->AddOp1(toOP, inReg);
			ASSERTCOVERAGE(UsedAsColumnCache(this, inReg, inReg));
			ExprCacheAffinityChange(inReg, 1);
			break; }
#endif

		case TK_LT:
		case TK_LE:
		case TK_GT:
		case TK_GE:
		case TK_NE:
		case TK_EQ: {
			_assert(TK_LT == OP_Lt);
			_assert(TK_LE == OP_Le);
			_assert(TK_GT == OP_Gt);
			_assert(TK_GE == OP_Ge);
			_assert(TK_EQ == OP_Eq);
			_assert(TK_NE == OP_Ne);
			ASSERTCOVERAGE(op == TK_LT);
			ASSERTCOVERAGE(op == TK_LE);
			ASSERTCOVERAGE(op == TK_GT);
			ASSERTCOVERAGE(op == TK_GE);
			ASSERTCOVERAGE(op == TK_EQ);
			ASSERTCOVERAGE(op == TK_NE);
			r1 = ExprCodeTemp(expr->Left, &regFree1);
			r2 = ExprCodeTemp(expr->Right, &regFree2);
			CodeCompare(this, expr->Left, expr->Right, op, r1, r2, inReg, SQLITE_STOREP2);
			ASSERTCOVERAGE(regFree1 == 0);
			ASSERTCOVERAGE(regFree2 == 0);
			break; }

		case TK_IS:
		case TK_ISNOT: {
			ASSERTCOVERAGE(op == TK_IS);
			ASSERTCOVERAGE(op == TK_ISNOT);
			r1 = ExprCodeTemp(expr->Left, &regFree1);
			r2 = ExprCodeTemp(expr->Right, &regFree2);
			op = (op==TK_IS) ? TK_EQ : TK_NE;
			CodeCompare(this, expr->Left, expr->Right, op, r1, r2, inReg, SQLITE_STOREP2 | SQLITE_NULLEQ);
			ASSERTCOVERAGE(regFree1 == 0);
			ASSERTCOVERAGE(regFree2 == 0);
			break; }

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
			_assert(TK_AND == OP_And);
			_assert(TK_OR == OP_Or);
			_assert(TK_PLUS == OP_Add);
			_assert(TK_MINUS == OP_Subtract);
			_assert(TK_REM == OP_Remainder);
			_assert(TK_BITAND == OP_BitAnd);
			_assert(TK_BITOR == OP_BitOr);
			_assert(TK_SLASH == OP_Divide);
			_assert(TK_LSHIFT == OP_ShiftLeft);
			_assert(TK_RSHIFT == OP_ShiftRight);
			_assert(TK_CONCAT == OP_Concat);
			ASSERTCOVERAGE(op == TK_AND);
			ASSERTCOVERAGE(op == TK_OR);
			ASSERTCOVERAGE(op == TK_PLUS);
			ASSERTCOVERAGE(op == TK_MINUS);
			ASSERTCOVERAGE(op == TK_REM);
			ASSERTCOVERAGE(op == TK_BITAND);
			ASSERTCOVERAGE(op == TK_BITOR);
			ASSERTCOVERAGE(op == TK_SLASH);
			ASSERTCOVERAGE(op == TK_LSHIFT);
			ASSERTCOVERAGE(op == TK_RSHIFT);
			ASSERTCOVERAGE(op == TK_CONCAT);
			r1 = ExprCodeTemp(expr->Left, &regFree1);
			r2 = ExprCodeTemp(expr->Right, &regFree2);
			v->AddOp3(op, r2, r1, target);
			ASSERTCOVERAGE(regFree1 == 0);
			ASSERTCOVERAGE(regFree2 == 0);
			break; }

		case TK_UMINUS: {
			Expr *left = expr->Left;
			_assert(left);
			if (left->OP == TK_INTEGER)
			{
				CodeInteger(this, left, 1, target);
#ifndef OMIT_FLOATING_POINT
			}
			else if (left->OP == TK_FLOAT)
			{
				_assert(!ExprHasProperty(expr, EP_IntValue));
				CodeReal(v, left->u.Token, 1, target);
#endif
			}
			else
			{
				regFree1 = r1 = GetTempReg();
				v->AddOp2(OP_Integer, 0, r1);
				r2 = ExprCodeTemp(expr->Left, &regFree2);
				v->AddOp3(OP_Subtract, r2, r1, target);
				ASSERTCOVERAGE(regFree2 == 0);
			}
			inReg = target;
			break; }

		case TK_BITNOT:
		case TK_NOT: {
			_assert(TK_BITNOT == OP_BitNot);
			_assert(TK_NOT == OP_Not);
			ASSERTCOVERAGE(op == TK_BITNOT);
			ASSERTCOVERAGE(op == TK_NOT);
			r1 = ExprCodeTemp(eExpr->Left, &regFree1);
			ASSERTCOVERAGE(regFree1 == 0);
			inReg = target;
			v->AddOp2(op, r1, inReg);
			break; }

		case TK_ISNULL:
		case TK_NOTNULL: {
			int addr;
			_assert(TK_ISNULL == OP_IsNull);
			_assert(TK_NOTNULL == OP_NotNull);
			ASSERTCOVERAGE(op == TK_ISNULL);
			ASSERTCOVERAGE(op == TK_NOTNULL);
			v->AddOp2(OP_Integer, 1, target);
			r1 = ExprCodeTemp(expr->Left, &regFree1);
			ASSERTCOVERAGE(regFree1 == 0);
			addr = v->AddOp1(op, r1);
			v->AddOp2(OP_AddImm, target, -1);
			v->JumpHere(addr);
			break; }

		case TK_AGG_FUNCTION: {
			AggInfo *info = expr->AggInfo;
			if (!info)
			{
				_assert(!ExprHasProperty(expr, EP_IntValue));
				ErrorMsg("misuse of aggregate: %s()", expr->u.Token);
			}
			else
				inReg = info->Funcs[expr->Agg].Mem;
			break; }

		case TK_CONST_FUNC:
		case TK_FUNCTION: {
			_assert(!ExprHasProperty(expr, EP_xIsSelect));
			ASSERTCOVERAGE(op == TK_CONST_FUNC);
			ASSERTCOVERAGE(op == TK_FUNCTION);
			ExprList *farg = (ExprHasAnyProperty(expr, EP_TokenOnly) ? nullptr : expr->x.List); // List of function arguments
			int fargLength = (farg ? farg->Exprs : 0); // Number of function arguments */
			_assert(!ExprHasProperty(expr, EP_IntValue));
			const char *id = pExpr->u.Token; // The function name
			int idLength = _strlen30(id); // Length of the function name in bytes
			uint8 enc = ENC(Ctx); // The text encoding used by this database
			FuncDef *def = FindFunction(Ctx, id, idLength, fargLength, enc, 0); // The function definition object
			if (!def)
			{
				ErrorMsg("unknown function: %.*s()", idLength, id);
				break;
			}

			// Attempt a direct implementation of the built-in COALESCE() and IFNULL() functions.  This avoids unnecessary evalation of
			// arguments past the first non-NULL argument.
			int i;
			if (def->Flags & FUNC_COALESCE)
			{
				int endCoalesce = v->MakeLabel();
				_assert(fargLength >= 2);
				ExprCode(farg->Ids[0].Expr, target);
				for (i = 1; i < fargLength; i++)
				{
					v->AddOp2(OP_NotNull, target, endCoalesce);
					ExprCacheRemove(target, 1);
					ExprCachePush();
					ExprCode(farg->Ids[i].Expr, target);
					ExprCachePop(1);
				}
				v->ResolveLabel(endCoalesce);
				break;
			}

			if (farg)
			{
				r1 = GetTempRange(fargLength);
				// For length() and typeof() functions with a column argument, set the P5 parameter to the OP_Column opcode to OPFLAG_LENGTHARG
				// or OPFLAG_TYPEOFARG respectively, to avoid unnecessary data loading.
				if ((def->Flags & (FUNC_LENGTH | FUNC_TYPEOF)) != 0)
				{
					_assert(fargLength == 1);
					_assert(farg->Ids[0].Expr);
					uint8 exprOP = farg->Ids[0].Expr->OP;
					if (exprOP == TK_COLUMN || exprOP == TK_AGG_COLUMN)
					{
						_assert(FUNC_LENGTH == OPFLAG_LENGTHARG);
						_assert(FUNC_TYPEOF == OPFLAG_TYPEOFARG);
						ASSERTCOVERAGE(def->Flags == FUNC_LENGTH);
						farg->Ids[0].Expr->OP2 = def->Flags;
					}
				}
				ExprCachePush(); // Ticket 2ea2425d34be
				ExprCodeExprList(farg, r1, 1);
				ExprCachePop(1); // Ticket 2ea2425d34be
			}
			else
				r1 = 0;
#ifndef OMIT_VIRTUALTABLE
			// Possibly overload the function if the first argument is a virtual table column.
			//
			// For infix functions (LIKE, GLOB, REGEXP, and MATCH) use the second argument, not the first, as the argument to test to
			// see if it is a column in a virtual table.  This is done because the left operand of infix functions (the operand we want to
			// control overloading) ends up as the second argument to the function.  The expression "A glob B" is equivalent to 
			// "glob(B,A).  We want to use the A in "A glob B" to test for function overloading.  But we use the B term in "glob(B,A)".
			if (fargLength >= 2 && (expr->Flags & EP_InfixFunc) ? 1 : 0)
				def = VTable::OverloadFunction(Ctx, def, fargLength, farg->Ids[1].Expr);
			else if (fargLength > 0)
				def = VTable::OverloadFunction(Ctx, def, fargLength, farg->Ids[0].Expr);
#endif

			int constMask = 0; // Mask of function arguments that are constant
			CollSeq *coll = nullptr; // A collating sequence
			for (i = 0; i < fargLength; i++)
			{
				if (i < 32 && farg->Ids[i].Expr->IsConstant())
					constMask |= (1<<i);
				if ((def->Flags & FUNC_NEEDCOLL) != 0 && !coll)
					coll = ExprCollSeq(farg->Ids[i].Expr);
			}
			if (def->Flags & FUNC_NEEDCOLL)
			{
				if (!coll)
					coll = Ctx->DfltColl; 
				v->AddOp4(OP_CollSeq, 0, 0, 0, (char *)coll, Vdbe::P4T_COLLSEQ);
			}
			v->AddOp4(OP_Function, constMask, r1, target, (char *)def, Vdbe::P4T_FUNCDEF);
			v->ChangeP5((uint8)fargLength);
			if (fargLength)
				ReleaseTempRange(r1, fargLength);
			break; }

#ifndef OMIT_SUBQUERY
		case TK_EXISTS:
		case TK_SELECT: {
			ASSERTCOVERAGE(op == TK_EXISTS);
			ASSERTCOVERAGE(op == TK_SELECT);
			inReg = CodeSubselect(expr, 0, false);
			break; }

		case TK_IN: {
			int destIfFalse = v->MakeLabel();
			int destIfNull = v->MakeLabel();
			v->AddOp2(OP_Null, 0, target);
			ExprCodeIN(expr, destIfFalse, destIfNull);
			v->AddOp2(OP_Integer, 1, target);
			v->ResolveLabel(destIfFalse);
			v->AddOp2(OP_AddImm, target, 0);
			v->ResolveLabel(destIfNull);
			break; }
#endif


					//
					//    x BETWEEN y AND z
					//
					// This is equivalent to
					//
					//    x>=y AND x<=z
					//
					// X is stored in pExpr->pLeft.
					// Y is stored in pExpr->pList->a[0].pExpr.
					// Z is stored in pExpr->pList->a[1].pExpr.
		case TK_BETWEEN: {
			Expr *left = expr->Left;
			ExprList::ExprListItem *item = expr->x.List->Ids;
			Expr *right = item->Expr;
			r1 = ExprCodeTemp(left, &regFree1);
			r2 = ExprCodeTemp(right, &regFree2);
			ASSERTCOVERAGE(regFree1 == 0);
			ASSERTCOVERAGE(regFree2 == 0);
			r3 = GetTempReg();
			r4 = GetTempReg();
			CodeCompare(this, left, right, OP_Ge, r1, r2, r3, SQLITE_STOREP2);
			item++;
			right = item->Expr;
			ReleaseTempReg(regFree2);
			r2 = ExprCodeTemp(right, &regFree2);
			ASSERTCOVERAGE(regFree2 == 0);
			CodeCompare(this, left, right, OP_Le, r1, r2, r4, SQLITE_STOREP2);
			v->AddOp3(OP_And, r3, r4, target);
			ReleaseTempReg(r3);
			ReleaseTempReg(r4);
			break; }

		case TK_COLLATE: 
		case TK_UPLUS: {
			inReg = ExprCodeTarget(expr->Left, target);
			break; }

		case TK_TRIGGER: {
			// If the opcode is TK_TRIGGER, then the expression is a reference to a column in the new.* or old.* pseudo-tables available to
			// trigger programs. In this case Expr.iTable is set to 1 for the new.* pseudo-table, or 0 for the old.* pseudo-table. Expr.iColumn
			// is set to the column of the pseudo-table to read, or to -1 to read the rowid field.
			//
			// The expression is implemented using an OP_Param opcode. The p1 parameter is set to 0 for an old.rowid reference, or to (i+1)
			// to reference another column of the old.* pseudo-table, where i is the index of the column. For a new.rowid reference, p1 is
			// set to (n+1), where n is the number of columns in each pseudo-table. For a reference to any other column in the new.* pseudo-table, p1
			// is set to (n+2+i), where n and i are as defined previously. For example, if the table on which triggers are being fired is
			// declared as:
			//
			//   CREATE TABLE t1(a, b);
			//
			// Then p1 is interpreted as follows:
			//
			//   p1==0   ->    old.rowid     p1==3   ->    new.rowid
			//   p1==1   ->    old.a         p1==4   ->    new.a
			//   p1==2   ->    old.b         p1==5   ->    new.b       
			Table *table = expr->Table;
			int p1 = expr->TableIdx * (table->Cols.length + 1) + 1 + expr->ColumnIdx;
			_assert(expr->TableIdx == 0 || expr->TableIdx == 1);
			_assert(expr->ColumnIdx >= -1 && expr->ColumnIdx < table->Cols.length);
			_assert(table->PKey < 0 || expr->ColumnIdx != table->PKey);
			_assert(p1 >= 0 && p1 < (table->Cols.ModuleArgs*2+2));
			v->AddOp2(OP_Param, p1, target);
			VdbeComment(v, "%s.%s -> $%d", (expr->TableIdx ? "new" : "old"), (expr->ColumnIdx < 0 ? "rowid" : expr->Table->Cols[expr->ColumnIdx].Name), target);

#ifndef OMIT_FLOATING_POINT
			// If the column has REAL affinity, it may currently be stored as an integer. Use OP_RealAffinity to make sure it is really real.
			if (expr->ColumnIdx >= 0 && table->Cols[expr->ColumnIdx].Affinity == AFF_REAL)
				v->AddOp1(OP_RealAffinity, target);
#endif
			break; }


						 // Form A:
						 //   CASE x WHEN e1 THEN r1 WHEN e2 THEN r2 ... WHEN eN THEN rN ELSE y END
						 //
						 // Form B:
						 //   CASE WHEN e1 THEN r1 WHEN e2 THEN r2 ... WHEN eN THEN rN ELSE y END
						 //
						 // Form A is can be transformed into the equivalent form B as follows:
						 //   CASE WHEN x=e1 THEN r1 WHEN x=e2 THEN r2 ...
						 //        WHEN x=eN THEN rN ELSE y END
						 //
						 // X (if it exists) is in pExpr->pLeft.
						 // Y is in pExpr->pRight.  The Y is also optional.  If there is no
						 // ELSE clause and no other term matches, then the result of the
						 // exprssion is NULL.
						 // Ei is in pExpr->pList->a[i*2] and Ri is pExpr->pList->a[i*2+1].
						 //
						 // The result of the expression is the Ri for the first matching Ei, or if there is no matching Ei, the ELSE term Y, or if there is
						 // no ELSE term, NULL.
		default: _assert(op == TK_CASE); {

			ASSERTONLY(int cacheLevel = CacheLevel);
			_assert(!ExprHasProperty(expr, EP_xIsSelect) && expr->x.List);
			_assert((expr->x.List->Exprs % 2) == 0);
			_assert(expr->x.List->Exprs > 0);
			ExprList *list = expr->x.List; // List of WHEN terms
			ExprList::ExprListItem *elems = list->Ids; // Array of WHEN terms
			int exprs = list->Exprs; // 2x number of WHEN terms
			int endLabel = v->MakeLabel(); // GOTO label for end of CASE stmt
			Expr *x; // The X expression
			Expr opCompare; // The X==Ei expression
			Expr *test = nullptr; // X==Ei (form A) or just Ei (form B)
			if ((x = expr->Left) != nullptr)
			{
				Expr cacheX = *x; // Cached expression X
				ASSERTCOVERAGE(x->OP == TK_COLUMN);
				ASSERTCOVERAGE(x->OP == TK_REGISTER);
				cacheX.TableIdx = ExprCodeTemp(x, &regFree1);
				ASSERTCOVERAGE(regFree1 == 0);
				cacheX.OP = TK_REGISTER;
				opCompare.OP = TK_EQ;
				opCompare.Left = &cacheX;
				test = &opCompare;
				// Ticket b351d95f9cd5ef17e9d9dbae18f5ca8611190001: The value in regFree1 might get SCopy-ed into the file result.
				// So make sure that the regFree1 register is not reused for other purposes and possibly overwritten.
				regFree1 = 0;
			}
			for (int i = 0; i < exprs; i += 2)
			{
				ExprCachePush();
				if (x)
				{
					_assert(test);
					opCompare.Right = elems[i].Expr;
				}
				else
					test = elems[i].Expr;
				int nextCase = v->MakeLabel(); // GOTO label for next WHEN clause
				ASSERTCOVERAGE(test->OP == TK_COLUMN );
				ExprIfFalse(test, nextCase, SQLITE_JUMPIFNULL);
				ASSERTCOVERAGE(elems[i+1].Expr->OP == TK_COLUMN);
				ASSERTCOVERAGE(elems[i+1].Expr->OP == TK_REGISTER);
				ExprCode(elems[i+1].Expr, target);
				v->AddOp2(OP_Goto, 0, endLabel);
				ExprCachePop(1);
				v->ResolveLabel(nextCase);
			}
			if (expr->Right)
			{
				ExprCachePush();
				ExprCode(expr->Right, target);
				ExprCachePop(1);
			}
			else
				v->AddOp2(OP_Null, 0, target);
			_assert(Ctx->MallocFailed || Errs > 0 || CacheLevel == cacheLevel);
			v->ResolveLabel(endLabel);
			break; }

#ifndef OMIT_TRIGGER
		case TK_RAISE: {
			_assert(expr->Affinity == OE_Rollback  || expr->Affinity == OE_Abort || expr->Affinity= = OE_Fail || expr->Affinity == OE_Ignore);
			if (!TriggerTab)
			{
				ErrorMsg("RAISE() may only be used within a trigger-program");
				return 0;
			}
			if (expr->Affinity == OE_Abort)
				MayAbort();
			_assert(!ExprHasProperty(expr, EP_IntValue));
			if (expr->Affinity == OE_Ignore)
				v->AddOp4(OP_Halt, RC_OK, OE_Ignore, 0, pExpr->u.zToken,0);
			else
				HaltConstraint(CONSTRAINT_TRIGGER, expr->Affinity, expr->u.Token, 0);
			break; }
#endif

		}
		ReleaseTempReg(regFree1);
		ReleaseTempReg(regFree2);
		return inReg;
	}

	__device__ int Parse::ExprCodeTemp(Expr *expr, int *reg)
	{
		int r1 = GetTempReg();
		int r2 = ExprCodeTarget(expr, r1);
		if (r2 == r1)
			*reg = r1;
		else
		{
			ReleaseTempReg(r1);
			*reg = 0;
		}
		return r2;
	}

	__device__ int Parse::ExprCode(Expr *expr, int target)
	{
		_assert(target > 0 && target <= Mems);
		if (expr && expr->OP == TK_REGISTER)
			V->AddOp2(OP_Copy, expr->TableIdx, target);
		else
		{
			int inReg = ExprCodeTarget(expr, target);
			_assert(V || Ctx->MallocFailed);
			if (inReg != target && V)
				V->AddOp2(OP_SCopy, inReg, target);
		}
		return target;
	}

	__device__ int Parse::ExprCodeAndCache(Expr *expr, int target)
	{
		Vdbe *v = V;
		int inReg = ExprCode(expr, target);
		_assert(target > 0);
		// This routine is called for terms to INSERT or UPDATE.  And the only other place where expressions can be converted into TK_REGISTER is
		// in WHERE clause processing.  So as currently implemented, there is no way for a TK_REGISTER to exist here.  But it seems prudent to
		// keep the ALWAYS() in case the conditions above change with future modifications or enhancements. */
		if (SysEx_ALWAYS(expr->OP != TK_REGISTER))
		{  
			int mem = ++Mems;
			v->AddOp2(OP_Copy, inReg, mem);
			expr->TableIdx = mem;
			expr->OP2 = pExpr->op;
			expr->OP = TK_REGISTER;
		}
		return inReg;
	}

#if defined(ENABLE_TREE_EXPLAIN)
	void sqlite3ExplainExpr(Vdbe *pOut, Expr *pExpr)
	{
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
#endif

	__device__ static int IsAppropriateForFactoring(Expr *p)
	{
		if (!p->IsConstantNotJoin())
			return 0;  // Only constant expressions are appropriate for factoring
		if ((p->Flags & EP_FixedDest) == 0)
			return 1;  /* Any constant without a fixed destination is appropriate */
		while (p->OP == TK_UPLUS) p = p->Left;
		switch (p->OP)
		{
#ifndef OMIT_BLOB_LITERAL
		case TK_BLOB:
#endif
		case TK_VARIABLE:
		case TK_INTEGER:
		case TK_FLOAT:
		case TK_NULL:
		case TK_STRING: {
			ASSERTCOVERAGE(p->OP == TK_BLOB);
			ASSERTCOVERAGE(p->OP == TK_VARIABLE);
			ASSERTCOVERAGE(p->OP == TK_INTEGER);
			ASSERTCOVERAGE(p->OP == TK_FLOAT);
			ASSERTCOVERAGE(p->OP == TK_NULL);
			ASSERTCOVERAGE(p->OP == TK_STRING);
			// Single-instruction constants with a fixed destination are better done in-line.  If we factor them, they will just end
			// up generating an OP_SCopy to move the value to the destination register.
			return 0; }

		case TK_UMINUS: {
			if (p->Left->OP == TK_FLOAT || p->Left->OP == TK_INTEGER)
				return 0;
			break; }

		default: {
			break; }
		}
		return 1;
	}

	__device__ static int EvalConstExpr(Walker *walker, Expr *expr)
	{
		Parse *parse = walker->Parse;
		switch (expr->OP)
		{
		case TK_IN:
		case TK_REGISTER: {
			return WRC_Prune; }
		case TK_COLLATE: {
			return WRC_Continue; }
		case TK_FUNCTION:
		case TK_AGG_FUNCTION:
		case TK_CONST_FUNC: {
			// The arguments to a function have a fixed destination. Mark them this way to avoid generated unneeded OP_SCopy
			// instructions. 
			ExprList *list = expr->x.List;
			_assert(!ExprHasProperty(expr, EP_xIsSelect));
			if (list)
			{
				int i = list->Exprs;
				ExprList::ExprListItem *item = list->Ids;
				for (; i > 0; i--, item++)
					if (SysEx_ALWAYS(item->Expr))
						item->Expr->Flags |= EP_FixedDest;
			}
			break; }
		}
		if (IsAppropriateForFactoring(expr))
		{
			int r1 = ++parse->Mems;
			int r2 = parse->ExprCodeTarget(expr, r1);
			// If r2!=r1, it means that register r1 is never used.  That is harmless but suboptimal, so we want to know about the situation to fix it.
			// Hence the following assert:
			_assert(r2 == r1);
			expr->OP2 = expr->OP;
			expr->OP = TK_REGISTER;
			expr->TableIdx = r2;
			return WRC_Prune;
		}
		return WRC_Continue;
	}

	__device__ void Parse::ExprCodeConstants(Expr *expr)
	{
		Walker w;
		if (CookieGoto || OptimizationDisabled(Ctx, SQLITE_FactorOutConst))
			return;
		w.ExprCallback = EvalConstExpr;
		w.SelectCallback = 0;
		w.Parse = this;
		WalkExpr(&w, pExpr);
	}

	__device__ int Parse::ExprCodeExprList(ExprList *list, int target, bool doHardCopy)
	{
		_assert(list);
		_assert(target>0 );
		_assert(V);  // Never gets this far otherwise
		int n = list->Exprs;
		int i;
		ExprList::ExprListItem *item;
		for (item = list->Ids, i = 0; i < n; i++, item++)
		{
			Expr *expr = item->Expr;
			int inReg = ExprCodeTarget(expr, target+i);
			if (inReg != target+i)
				V->AddOp2(doHardCopy ? OP_Copy : OP_SCopy, inReg, target+i);
		}
		return n;
	}

	__device__ static void ExprCodeBetween(Expr *expr, int dest, int jumpIfTrue, int jumpIfNull)
	{
		_assert(!ExprHasProperty(expr, EP_xIsSelect));
		Expr exprX = *expr->Left; // The  x  subexpression
		Expr compLeft; // The  x>=y  term
		Expr compRight;   // The  x<=z  term
		Expr exprAnd; // The AND operator in  x>=y AND x<=z
		exprAnd.OP = TK_AND;
		exprAnd.Left = &compLeft;
		exprAnd.Right = &compRight;
		compLeft.OP = TK_GE;
		compLeft.Left = &exprX;
		compLeft.Right = expr->x.List->Ids[0].Expr;
		compRight.OP = TK_LE;
		compRight.Left = &exprX;
		compRight.Right = expr->x.List->Ids[1].Expr;
		int regFree1 = 0; // Temporary use register
		exprX.TableIdx = ExprCodeTemp(&exprX, &regFree1);
		exprX.OP = TK_REGISTER;
		if (jumpIfTrue)
			ExprIfTrue(&exprAnd, dest, jumpIfNull);
		else
			ExprIfFalse(&exprAnd, dest, jumpIfNull);
		ReleaseTempReg(regFree1);

		// Ensure adequate test coverage
		ASSERTCOVERAGE(jumpIfTrue == 0 && jumpIfNull == 0 && regFree1 == 0);
		ASSERTCOVERAGE(jumpIfTrue == 0 && jumpIfNull == 0 && regFree1 != 0);
		ASSERTCOVERAGE(jumpIfTrue == 0 && jumpIfNull != 0 && regFree1 == 0);
		ASSERTCOVERAGE(jumpIfTrue == 0 && jumpIfNull != 0 && regFree1 != 0);
		ASSERTCOVERAGE(jumpIfTrue != 0 && jumpIfNull == 0 && regFree1 == 0);
		ASSERTCOVERAGE(jumpIfTrue != 0 && jumpIfNull == 0 && regFree1 != 0);
		ASSERTCOVERAGE(jumpIfTrue != 0 && jumpIfNull != 0 && regFree1 == 0);
		ASSERTCOVERAGE(jumpIfTrue != 0 && jumpIfNull != 0 && regFree1 != 0);
	}

	__device__ void Parse::ExprIfTrue(Expr *expr, int dest, int jumpIfNull)
	{
		_assert(jumpIfNull == SQLITE_JUMPIFNULL || jumpIfNull == 0);
		Vdbe *v = V;
		if (SysEx_NEVER(!v)) return;  // Existence of VDBE checked by caller
		if (SysEx_NEVER(!expr)) return;  // No way this can happen

		int regFree1 = 0;
		int regFree2 = 0;
		int r1, r2;

		int op = expr->OP;
		switch (op)
		{
		case TK_AND: {
			int d2 = v->MakeLabel();
			ASSERTCOVERAGE(jumpIfNull == 0);
			ExprCachePush();
			ExprIfFalse(expr->Left, d2, jumpIfNull ^ SQLITE_JUMPIFNULL);
			ExprIfTrue(expr->Right, dest, jumpIfNull);
			v->ResolveLabel(d2);
			ExprCachePop(1);
			break; }

		case TK_OR: {
			ASSERTCOVERAGE(jumpIfNull == 0);
			ExprIfTrue(expr->Left, dest, jumpIfNull);
			ExprIfTrue(expr->Right, dest, jumpIfNull);
			break; }

		case TK_NOT: {
			ASSERTCOVERAGE(jumpIfNull == 0);
			ExprIfFalse(expr->Left, dest, jumpIfNull);
			break; }

		case TK_LT:
		case TK_LE:
		case TK_GT:
		case TK_GE:
		case TK_NE:
		case TK_EQ: {
			_assert(TK_LT == OP_Lt);
			_assert(TK_LE == OP_Le);
			_assert(TK_GT == OP_Gt);
			_assert(TK_GE == OP_Ge);
			_assert(TK_EQ == OP_Eq);
			_assert(TK_NE == OP_Ne);
			ASSERTCOVERAGE(op == TK_LT);
			ASSERTCOVERAGE(op == TK_LE);
			ASSERTCOVERAGE(op == TK_GT);
			ASSERTCOVERAGE(op == TK_GE);
			ASSERTCOVERAGE(op == TK_EQ);
			ASSERTCOVERAGE(op == TK_NE);
			ASSERTCOVERAGE(jumpIfNull == 0);
			r1 = ExprCodeTemp(expr->Left, &regFree1);
			r2 = ExprCodeTemp(expr->Right, &regFree2);
			CodeCompare(this, expr->Left, expr->Right, op, r1, r2, dest, jumpIfNull);
			ASSERTCOVERAGE(regFree1 == 0);
			ASSERTCOVERAGE(regFree2 == 0);
			break; }

		case TK_IS:
		case TK_ISNOT: {
			ASSERTCOVERAGE(op == TK_IS);
			ASSERTCOVERAGE(op == TK_ISNOT);
			r1 = ExprCodeTemp(expr->Left, &regFree1);
			r2 = ExprCodeTemp(expr->Right, &regFree2);
			op = (op == TK_IS ? TK_EQ : TK_NE);
			CodeCompare(this, expr->Left, expr->Right, op, r1, r2, dest, SQLITE_NULLEQ);
			ASSERTCOVERAGE(regFree1 == 0);
			ASSERTCOVERAGE(regFree2 == 0);
			break; }

		case TK_ISNULL:
		case TK_NOTNULL: {
			_assert(TK_ISNULL == OP_IsNull);
			_assert(TK_NOTNULL == OP_NotNull);
			ASSERTCOVERAGE(op == TK_ISNULL);
			ASSERTCOVERAGE(op == TK_NOTNULL);
			r1 = ExprCodeTemp(expr->Left, &regFree1);
			v->AddOp2(op, r1, dest);
			ASSERTCOVERAGE(regFree1 == 0);
			break; }

		case TK_BETWEEN: {
			ASSERTCOVERAGE(jumpIfNull == 0);
			ExprCodeBetween(this, expr, dest, 1, jumpIfNull);
			break; }

#ifndef OMIT_SUBQUERY
		case TK_IN: {
			int destIfFalse = v->MakeLabel();
			int destIfNull = (jumpIfNull ? dest : destIfFalse);
			ExprCodeIN(this, expr, destIfFalse, destIfNull);
			v->AddOp2(OP_Goto, 0, dest);
			v->ResolveLabel(destIfFalse);
			break; }
#endif
		default: {
			r1 = ExprCodeTemp(this, expr, &regFree1);
			v->AddOp3(OP_If, r1, dest, jumpIfNull != 0);
			ASSERTCOVERAGE(regFree1 == 0);
			ASSERTCOVERAGE(jumpIfNull == 0);
			break; }
		}
		ReleaseTempReg(regFree1);
		ReleaseTempReg(regFree2);  
	}

	__device__ void Parse::ExprIfFalse(Expr *expr, int dest, int jumpIfNull)
	{
		Vdbe *v = V;
		_assert(jumpIfNull == SQLITE_JUMPIFNULL || jumpIfNull == 0);
		if (SysEx_NEVER(v == nullptr)) return; // Existence of VDBE checked by caller
		if (!expr) return;

		int op = 0;
		int regFree1 = 0;
		int regFree2 = 0;
		int r1, r2;

		// The value of pExpr->op and op are related as follows:
		//
		//       pExpr->op            op
		//       ---------          ----------
		//       TK_ISNULL          OP_NotNull
		//       TK_NOTNULL         OP_IsNull
		//       TK_NE              OP_Eq
		//       TK_EQ              OP_Ne
		//       TK_GT              OP_Le
		//       TK_LE              OP_Gt
		//       TK_GE              OP_Lt
		//       TK_LT              OP_Ge
		//
		// For other values of pExpr->op, op is undefined and unused. The value of TK_ and OP_ constants are arranged such that we
		// can compute the mapping above using the following expression. Assert()s verify that the computation is correct.
		op = ((expr->OP+(TK_ISNULL&1))^1) - (TK_ISNULL&1);

		// Verify correct alignment of TK_ and OP_ constants
		_assert(expr->OP != TK_ISNULL || op == OP_NotNull);
		_assert(expr->OP != TK_NOTNULL || op == OP_IsNull);
		_assert(expr->OP != TK_NE || op == OP_Eq);
		_assert(expr->OP != TK_EQ || op == OP_Ne);
		_assert(expr->OP != TK_LT || op == OP_Ge);
		_assert(expr->OP != TK_LE || op == OP_Gt);
		_assert(expr->OP != TK_GT || op == OP_Le);
		_assert(expr->OP != TK_GE || op == OP_Lt);

		switch (expr->OP)
		{
		case TK_AND: {
			ASSERTCOVERAGE(jumpIfNull == 0);
			ExprIfFalse(expr->Left, dest, jumpIfNull);
			ExprIfFalse(expr->Right, dest, jumpIfNull);
			break; }

		case TK_OR: {
			int d2 = v->MakeLabel();
			ASSERTCOVERAGE(jumpIfNull == 0);
			ExprCachePush();
			ExprIfTrue(expr->Left, d2, jumpIfNull ^ SQLITE_JUMPIFNULL);
			ExprIfFalse(expr->Right, dest, jumpIfNull);
			v->ResolveLabel(d2);
			ExprCachePop(1);
			break; }

		case TK_NOT: {
			ASSERTCOVERAGE(jumpIfNull == 0);
			ExprIfTrue(expr->Left, dest, jumpIfNull);
			break; }

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
#pragma region Registers

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

#pragma endregion