/***** This file contains automatically generated code ******
**
** The code in this file has been automatically generated by
**
**   sqlite/tool/mkkeywordhash.c
**
** The code in this file implements a function that determines whether
** or not a given identifier is really an SQL keyword.  The same thing
** might be implemented more directly using a hand-written hash table.
** But by using this automatically generated code, the size of the code
** is substantially reduced.  This is important for embedded applications
** on platforms with limited memory.
*/
/* Hash score: 175 */

/* zText[] encodes 811 bytes of keywords in 541 bytes */
/*   REINDEXEDESCAPEACHECKEYBEFOREIGNOREGEXPLAINSTEADDATABASELECT       */
/*   ABLEFTHENDEFERRABLELSEXCEPTRANSACTIONATURALTERAISEXCLUSIVE         */
/*   XISTSAVEPOINTERSECTRIGGEREFERENCESCONSTRAINTOFFSETEMPORARY         */
/*   UNIQUERYATTACHAVINGROUPDATEBEGINNERELEASEBETWEENOTNULLIKE          */
/*   CASCADELETECASECOLLATECREATECURRENT_DATEDETACHIMMEDIATEJOIN        */
/*   SERTMATCHPLANALYZEPRAGMABORTVALUESVIRTUALIMITWHENWHERENAME         */
/*   AFTEREPLACEANDEFAULTAUTOINCREMENTCASTCOLUMNCOMMITCONFLICTCROSS     */
/*   CURRENT_TIMESTAMPRIMARYDEFERREDISTINCTDROPFAILFROMFULLGLOBYIF      */
/*   ISNULLORDERESTRICTOUTERIGHTROLLBACKROWUNIONUSINGVACUUMVIEW         */
/*   INITIALLY                                                          */
__constant__ static const char zText[540] = {
	'R','E','I','N','D','E','X','E','D','E','S','C','A','P','E','A','C','H',
	'E','C','K','E','Y','B','E','F','O','R','E','I','G','N','O','R','E','G',
	'E','X','P','L','A','I','N','S','T','E','A','D','D','A','T','A','B','A',
	'S','E','L','E','C','T','A','B','L','E','F','T','H','E','N','D','E','F',
	'E','R','R','A','B','L','E','L','S','E','X','C','E','P','T','R','A','N',
	'S','A','C','T','I','O','N','A','T','U','R','A','L','T','E','R','A','I',
	'S','E','X','C','L','U','S','I','V','E','X','I','S','T','S','A','V','E',
	'P','O','I','N','T','E','R','S','E','C','T','R','I','G','G','E','R','E',
	'F','E','R','E','N','C','E','S','C','O','N','S','T','R','A','I','N','T',
	'O','F','F','S','E','T','E','M','P','O','R','A','R','Y','U','N','I','Q',
	'U','E','R','Y','A','T','T','A','C','H','A','V','I','N','G','R','O','U',
	'P','D','A','T','E','B','E','G','I','N','N','E','R','E','L','E','A','S',
	'E','B','E','T','W','E','E','N','O','T','N','U','L','L','I','K','E','C',
	'A','S','C','A','D','E','L','E','T','E','C','A','S','E','C','O','L','L',
	'A','T','E','C','R','E','A','T','E','C','U','R','R','E','N','T','_','D',
	'A','T','E','D','E','T','A','C','H','I','M','M','E','D','I','A','T','E',
	'J','O','I','N','S','E','R','T','M','A','T','C','H','P','L','A','N','A',
	'L','Y','Z','E','P','R','A','G','M','A','B','O','R','T','V','A','L','U',
	'E','S','V','I','R','T','U','A','L','I','M','I','T','W','H','E','N','W',
	'H','E','R','E','N','A','M','E','A','F','T','E','R','E','P','L','A','C',
	'E','A','N','D','E','F','A','U','L','T','A','U','T','O','I','N','C','R',
	'E','M','E','N','T','C','A','S','T','C','O','L','U','M','N','C','O','M',
	'M','I','T','C','O','N','F','L','I','C','T','C','R','O','S','S','C','U',
	'R','R','E','N','T','_','T','I','M','E','S','T','A','M','P','R','I','M',
	'A','R','Y','D','E','F','E','R','R','E','D','I','S','T','I','N','C','T',
	'D','R','O','P','F','A','I','L','F','R','O','M','F','U','L','L','G','L',
	'O','B','Y','I','F','I','S','N','U','L','L','O','R','D','E','R','E','S',
	'T','R','I','C','T','O','U','T','E','R','I','G','H','T','R','O','L','L',
	'B','A','C','K','R','O','W','U','N','I','O','N','U','S','I','N','G','V',
	'A','C','U','U','M','V','I','E','W','I','N','I','T','I','A','L','L','Y',
};
__constant__ static const unsigned char aHash[127] = {
	72, 101, 114,  70,   0,  45,   0,   0,  78,   0,  73,   0,   0,
	42,  12,  74,  15,   0, 113,  81,  50, 108,   0,  19,   0,   0,
	118,   0, 116, 111,   0,  22,  89,   0,   9,   0,   0,  66,  67,
	0,  65,   6,   0,  48,  86,  98,   0, 115,  97,   0,   0,  44,
	0,  99,  24,   0,  17,   0, 119,  49,  23,   0,   5, 106,  25,
	92,   0,   0, 121, 102,  56, 120,  53,  28,  51,   0,  87,   0,
	96,  26,   0,  95,   0,   0,   0,  91,  88,  93,  84, 105,  14,
	39, 104,   0,  77,   0,  18,  85, 107,  32,   0, 117,  76, 109,
	58,  46,  80,   0,   0,  90,  40,   0, 112,   0,  36,   0,   0,
	29,   0,  82,  59,  60,   0,  20,  57,   0,  52,
};
__constant__ static const unsigned char aNext[121] = {
	0,   0,   0,   0,   4,   0,   0,   0,   0,   0,   0,   0,   0,
	0,   2,   0,   0,   0,   0,   0,   0,  13,   0,   0,   0,   0,
	0,   7,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
	0,   0,   0,   0,  33,   0,  21,   0,   0,   0,  43,   3,  47,
	0,   0,   0,   0,  30,   0,  54,   0,  38,   0,   0,   0,   1,
	62,   0,   0,  63,   0,  41,   0,   0,   0,   0,   0,   0,   0,
	61,   0,   0,   0,   0,  31,  55,  16,  34,  10,   0,   0,   0,
	0,   0,   0,   0,  11,  68,  75,   0,   8,   0, 100,  94,   0,
	103,   0,  83,   0,  71,   0,   0, 110,  27,  37,  69,  79,   0,
	35,  64,   0,   0,
};
__constant__ static const unsigned char aLen[121] = {
	7,   7,   5,   4,   6,   4,   5,   3,   6,   7,   3,   6,   6,
	7,   7,   3,   8,   2,   6,   5,   4,   4,   3,  10,   4,   6,
	11,   6,   2,   7,   5,   5,   9,   6,   9,   9,   7,  10,  10,
	4,   6,   2,   3,   9,   4,   2,   6,   5,   6,   6,   5,   6,
	5,   5,   7,   7,   7,   3,   2,   4,   4,   7,   3,   6,   4,
	7,   6,  12,   6,   9,   4,   6,   5,   4,   7,   6,   5,   6,
	7,   5,   4,   5,   6,   5,   7,   3,   7,  13,   2,   2,   4,
	6,   6,   8,   5,  17,  12,   7,   8,   8,   2,   4,   4,   4,
	4,   4,   2,   2,   6,   5,   8,   5,   5,   8,   3,   5,   5,
	6,   4,   9,   3,
};
__constant__ static const unsigned short int aOffset[121] = {
	0,   2,   2,   8,   9,  14,  16,  20,  23,  25,  25,  29,  33,
	36,  41,  46,  48,  53,  54,  59,  62,  65,  67,  69,  78,  81,
	86,  91,  95,  96, 101, 105, 109, 117, 122, 128, 136, 142, 152,
	159, 162, 162, 165, 167, 167, 171, 176, 179, 184, 189, 194, 197,
	203, 206, 210, 217, 223, 223, 223, 226, 229, 233, 234, 238, 244,
	248, 255, 261, 273, 279, 288, 290, 296, 301, 303, 310, 315, 320,
	326, 332, 337, 341, 344, 350, 354, 361, 363, 370, 372, 374, 383,
	387, 393, 399, 407, 412, 412, 428, 435, 442, 443, 450, 454, 458,
	462, 466, 469, 471, 473, 479, 483, 491, 495, 500, 508, 511, 516,
	521, 527, 531, 536,
};
__constant__ static const unsigned char aCode[121] = {
	TK_REINDEX,    TK_INDEXED,    TK_INDEX,      TK_DESC,       TK_ESCAPE,     
	TK_EACH,       TK_CHECK,      TK_KEY,        TK_BEFORE,     TK_FOREIGN,    
	TK_FOR,        TK_IGNORE,     TK_LIKE_KW,    TK_EXPLAIN,    TK_INSTEAD,    
	TK_ADD,        TK_DATABASE,   TK_AS,         TK_SELECT,     TK_TABLE,      
	TK_JOIN_KW,    TK_THEN,       TK_END,        TK_DEFERRABLE, TK_ELSE,       
	TK_EXCEPT,     TK_TRANSACTION,TK_ACTION,     TK_ON,         TK_JOIN_KW,    
	TK_ALTER,      TK_RAISE,      TK_EXCLUSIVE,  TK_EXISTS,     TK_SAVEPOINT,  
	TK_INTERSECT,  TK_TRIGGER,    TK_REFERENCES, TK_CONSTRAINT, TK_INTO,       
	TK_OFFSET,     TK_OF,         TK_SET,        TK_TEMP,       TK_TEMP,       
	TK_OR,         TK_UNIQUE,     TK_QUERY,      TK_ATTACH,     TK_HAVING,     
	TK_GROUP,      TK_UPDATE,     TK_BEGIN,      TK_JOIN_KW,    TK_RELEASE,    
	TK_BETWEEN,    TK_NOTNULL,    TK_NOT,        TK_NO,         TK_NULL,       
	TK_LIKE_KW,    TK_CASCADE,    TK_ASC,        TK_DELETE,     TK_CASE,       
	TK_COLLATE,    TK_CREATE,     TK_CTIME_KW,   TK_DETACH,     TK_IMMEDIATE,  
	TK_JOIN,       TK_INSERT,     TK_MATCH,      TK_PLAN,       TK_ANALYZE,    
	TK_PRAGMA,     TK_ABORT,      TK_VALUES,     TK_VIRTUAL,    TK_LIMIT,      
	TK_WHEN,       TK_WHERE,      TK_RENAME,     TK_AFTER,      TK_REPLACE,    
	TK_AND,        TK_DEFAULT,    TK_AUTOINCR,   TK_TO,         TK_IN,         
	TK_CAST,       TK_COLUMNKW,   TK_COMMIT,     TK_CONFLICT,   TK_JOIN_KW,    
	TK_CTIME_KW,   TK_CTIME_KW,   TK_PRIMARY,    TK_DEFERRED,   TK_DISTINCT,   
	TK_IS,         TK_DROP,       TK_FAIL,       TK_FROM,       TK_JOIN_KW,    
	TK_LIKE_KW,    TK_BY,         TK_IF,         TK_ISNULL,     TK_ORDER,      
	TK_RESTRICT,   TK_JOIN_KW,    TK_JOIN_KW,    TK_ROLLBACK,   TK_ROW,        
	TK_UNION,      TK_USING,      TK_VACUUM,     TK_VIEW,       TK_INITIALLY,  
	TK_ALL,        
};

__device__ static int KeywordCode(const char *z, int n)
{
	int h, i;
	if( n<2 ) return TK_ID;
	h = ((_tolower(z[0])*4) ^
		(_tolower(z[n-1])*3) ^
		n) % 127;
	for(i=((int)aHash[h])-1; i>=0; i=((int)aNext[i])-1){
		if( aLen[i]==n && _strncmp(&zText[aOffset[i]],z,n)==0 ){
			ASSERTCOVERAGE( i==0 ); /* REINDEX */
			ASSERTCOVERAGE( i==1 ); /* INDEXED */
			ASSERTCOVERAGE( i==2 ); /* INDEX */
			ASSERTCOVERAGE( i==3 ); /* DESC */
			ASSERTCOVERAGE( i==4 ); /* ESCAPE */
			ASSERTCOVERAGE( i==5 ); /* EACH */
			ASSERTCOVERAGE( i==6 ); /* CHECK */
			ASSERTCOVERAGE( i==7 ); /* KEY */
			ASSERTCOVERAGE( i==8 ); /* BEFORE */
			ASSERTCOVERAGE( i==9 ); /* FOREIGN */
			ASSERTCOVERAGE( i==10 ); /* FOR */
			ASSERTCOVERAGE( i==11 ); /* IGNORE */
			ASSERTCOVERAGE( i==12 ); /* REGEXP */
			ASSERTCOVERAGE( i==13 ); /* EXPLAIN */
			ASSERTCOVERAGE( i==14 ); /* INSTEAD */
			ASSERTCOVERAGE( i==15 ); /* ADD */
			ASSERTCOVERAGE( i==16 ); /* DATABASE */
			ASSERTCOVERAGE( i==17 ); /* AS */
			ASSERTCOVERAGE( i==18 ); /* SELECT */
			ASSERTCOVERAGE( i==19 ); /* TABLE */
			ASSERTCOVERAGE( i==20 ); /* LEFT */
			ASSERTCOVERAGE( i==21 ); /* THEN */
			ASSERTCOVERAGE( i==22 ); /* END */
			ASSERTCOVERAGE( i==23 ); /* DEFERRABLE */
			ASSERTCOVERAGE( i==24 ); /* ELSE */
			ASSERTCOVERAGE( i==25 ); /* EXCEPT */
			ASSERTCOVERAGE( i==26 ); /* TRANSACTION */
			ASSERTCOVERAGE( i==27 ); /* ACTION */
			ASSERTCOVERAGE( i==28 ); /* ON */
			ASSERTCOVERAGE( i==29 ); /* NATURAL */
			ASSERTCOVERAGE( i==30 ); /* ALTER */
			ASSERTCOVERAGE( i==31 ); /* RAISE */
			ASSERTCOVERAGE( i==32 ); /* EXCLUSIVE */
			ASSERTCOVERAGE( i==33 ); /* EXISTS */
			ASSERTCOVERAGE( i==34 ); /* SAVEPOINT */
			ASSERTCOVERAGE( i==35 ); /* INTERSECT */
			ASSERTCOVERAGE( i==36 ); /* TRIGGER */
			ASSERTCOVERAGE( i==37 ); /* REFERENCES */
			ASSERTCOVERAGE( i==38 ); /* CONSTRAINT */
			ASSERTCOVERAGE( i==39 ); /* INTO */
			ASSERTCOVERAGE( i==40 ); /* OFFSET */
			ASSERTCOVERAGE( i==41 ); /* OF */
			ASSERTCOVERAGE( i==42 ); /* SET */
			ASSERTCOVERAGE( i==43 ); /* TEMPORARY */
			ASSERTCOVERAGE( i==44 ); /* TEMP */
			ASSERTCOVERAGE( i==45 ); /* OR */
			ASSERTCOVERAGE( i==46 ); /* UNIQUE */
			ASSERTCOVERAGE( i==47 ); /* QUERY */
			ASSERTCOVERAGE( i==48 ); /* ATTACH */
			ASSERTCOVERAGE( i==49 ); /* HAVING */
			ASSERTCOVERAGE( i==50 ); /* GROUP */
			ASSERTCOVERAGE( i==51 ); /* UPDATE */
			ASSERTCOVERAGE( i==52 ); /* BEGIN */
			ASSERTCOVERAGE( i==53 ); /* INNER */
			ASSERTCOVERAGE( i==54 ); /* RELEASE */
			ASSERTCOVERAGE( i==55 ); /* BETWEEN */
			ASSERTCOVERAGE( i==56 ); /* NOTNULL */
			ASSERTCOVERAGE( i==57 ); /* NOT */
			ASSERTCOVERAGE( i==58 ); /* NO */
			ASSERTCOVERAGE( i==59 ); /* NULL */
			ASSERTCOVERAGE( i==60 ); /* LIKE */
			ASSERTCOVERAGE( i==61 ); /* CASCADE */
			ASSERTCOVERAGE( i==62 ); /* ASC */
			ASSERTCOVERAGE( i==63 ); /* DELETE */
			ASSERTCOVERAGE( i==64 ); /* CASE */
			ASSERTCOVERAGE( i==65 ); /* COLLATE */
			ASSERTCOVERAGE( i==66 ); /* CREATE */
			ASSERTCOVERAGE( i==67 ); /* CURRENT_DATE */
			ASSERTCOVERAGE( i==68 ); /* DETACH */
			ASSERTCOVERAGE( i==69 ); /* IMMEDIATE */
			ASSERTCOVERAGE( i==70 ); /* JOIN */
			ASSERTCOVERAGE( i==71 ); /* INSERT */
			ASSERTCOVERAGE( i==72 ); /* MATCH */
			ASSERTCOVERAGE( i==73 ); /* PLAN */
			ASSERTCOVERAGE( i==74 ); /* ANALYZE */
			ASSERTCOVERAGE( i==75 ); /* PRAGMA */
			ASSERTCOVERAGE( i==76 ); /* ABORT */
			ASSERTCOVERAGE( i==77 ); /* VALUES */
			ASSERTCOVERAGE( i==78 ); /* VIRTUAL */
			ASSERTCOVERAGE( i==79 ); /* LIMIT */
			ASSERTCOVERAGE( i==80 ); /* WHEN */
			ASSERTCOVERAGE( i==81 ); /* WHERE */
			ASSERTCOVERAGE( i==82 ); /* RENAME */
			ASSERTCOVERAGE( i==83 ); /* AFTER */
			ASSERTCOVERAGE( i==84 ); /* REPLACE */
			ASSERTCOVERAGE( i==85 ); /* AND */
			ASSERTCOVERAGE( i==86 ); /* DEFAULT */
			ASSERTCOVERAGE( i==87 ); /* AUTOINCREMENT */
			ASSERTCOVERAGE( i==88 ); /* TO */
			ASSERTCOVERAGE( i==89 ); /* IN */
			ASSERTCOVERAGE( i==90 ); /* CAST */
			ASSERTCOVERAGE( i==91 ); /* COLUMN */
			ASSERTCOVERAGE( i==92 ); /* COMMIT */
			ASSERTCOVERAGE( i==93 ); /* CONFLICT */
			ASSERTCOVERAGE( i==94 ); /* CROSS */
			ASSERTCOVERAGE( i==95 ); /* CURRENT_TIMESTAMP */
			ASSERTCOVERAGE( i==96 ); /* CURRENT_TIME */
			ASSERTCOVERAGE( i==97 ); /* PRIMARY */
			ASSERTCOVERAGE( i==98 ); /* DEFERRED */
			ASSERTCOVERAGE( i==99 ); /* DISTINCT */
			ASSERTCOVERAGE( i==100 ); /* IS */
			ASSERTCOVERAGE( i==101 ); /* DROP */
			ASSERTCOVERAGE( i==102 ); /* FAIL */
			ASSERTCOVERAGE( i==103 ); /* FROM */
			ASSERTCOVERAGE( i==104 ); /* FULL */
			ASSERTCOVERAGE( i==105 ); /* GLOB */
			ASSERTCOVERAGE( i==106 ); /* BY */
			ASSERTCOVERAGE( i==107 ); /* IF */
			ASSERTCOVERAGE( i==108 ); /* ISNULL */
			ASSERTCOVERAGE( i==109 ); /* ORDER */
			ASSERTCOVERAGE( i==110 ); /* RESTRICT */
			ASSERTCOVERAGE( i==111 ); /* OUTER */
			ASSERTCOVERAGE( i==112 ); /* RIGHT */
			ASSERTCOVERAGE( i==113 ); /* ROLLBACK */
			ASSERTCOVERAGE( i==114 ); /* ROW */
			ASSERTCOVERAGE( i==115 ); /* UNION */
			ASSERTCOVERAGE( i==116 ); /* USING */
			ASSERTCOVERAGE( i==117 ); /* VACUUM */
			ASSERTCOVERAGE( i==118 ); /* VIEW */
			ASSERTCOVERAGE( i==119 ); /* INITIALLY */
			ASSERTCOVERAGE( i==120 ); /* ALL */
			return aCode[i];
		}
	}
	return TK_ID;
}
//int sqlite3KeywordCode(const unsigned char *z, int n)
//{
//	return KeywordCode((char *)z, n);
//}
#define SQLITE_N_KEYWORD 121
