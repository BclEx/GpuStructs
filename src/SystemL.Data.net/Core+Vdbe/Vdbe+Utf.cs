#define TRANSLATE_TRACE //1
using System;
using System.Diagnostics;
using System.Text;

namespace Core
{
  public partial class Vdbe
  {
#if !OMIT_UTF16
static int sqlite3VdbeMemTranslate(Mem mem, TEXTENCODE desiredEncode)
{
int len; // Maximum length of output string in bytes
Debugger.Break (); // TODO -
//unsigned char *zOut; // Output buffer
//unsigned char *zIn; // Input iterator
//unsigned char *zTerm; // End of input
//unsigned char *z; // Output iterator
//unsigned int c;

Debug.Assert( mem.Ctx==null || MutexEx.Held(mem.Ctx.Mutex) );
Debug.Assert( (mem.Flags & MEM.Str) != 0);
Debug.Assert( mem.Encode != desiredEncode);
Debug.Assert( mem.Encode != 0);
Debug.Assert( mem.N >= 0);

//#if TRANSLATE_TRACE && DEBUG
//{
//char zBuf[100];
//sqlite3VdbeMemPrettyPrint(mem, zBuf);
//fprintf(stderr, "INPUT:  %s\n", zBuf);
//}
#endif

// If the translation is between UTF-16 little and big endian, then all that is required is to swap the byte order. This case is handled
// differently from the others.
Debugger.Break (); // TODO -
//if( pMem->enc!=SQLITE_UTF8 && desiredEnc!=SQLITE_UTF8 ){
//  u8 temp;
//  int rc;
//  rc = sqlite3VdbeMemMakeWriteable(pMem);
//  if( rc!=SQLITE_OK ){
//    Debug.Assert( rc==SQLITE_NOMEM );
//    return SQLITE_NOMEM;
//  }
//  zIn = (u8*)pMem.z;
//  zTerm = &zIn[pMem->n&~1];
//  while( zIn<zTerm ){
//    temp = *zIn;
//    *zIn = *(zIn+1);
//    zIn++;
//    *zIn++ = temp;
//  }
//  pMem->enc = desiredEnc;
//  goto translate_out;
//}

// Set len to the maximum number of bytes required in the output buffer.
if( desiredEncode==TEXTENCODE.UTF8)
{
/* When converting from UTF-16, the maximum growth results from
** translating a 2-byte character to a 4-byte UTF-8 character.
** A single byte is required for the output string
** nul-terminator.
*/
mem.n &= ~1;
len = mem.n * 2 + 1;
}else{
/* When converting from UTF-8 to UTF-16 the maximum growth is caused
** when a 1-byte UTF-8 character is translated into a 2-byte UTF-16
** character. Two bytes are required in the output buffer for the
** nul-terminator.
*/
len = mem.n * 2 + 2;
}

/* Set zIn to point at the start of the input buffer and zTerm to point 1
** byte past the end.
**
** Variable zOut is set to point at the output buffer, space obtained
** from sqlite3Malloc().
*/
Debugger.Break (); // TODO -
//zIn = (u8*)pMem.z;
//zTerm = &zIn[pMem->n];
//zOut = sqlite3DbMallocRaw(pMem->db, len);
//if( !zOut ){
//  return SQLITE_NOMEM;
//}
//z = zOut;

//if( pMem->enc==SQLITE_UTF8 ){
//  if( desiredEnc==SQLITE_UTF16LE ){
//    /* UTF-8 -> UTF-16 Little-endian */
//    while( zIn<zTerm ){
///* c = sqlite3Utf8Read(zIn, zTerm, (const u8**)&zIn); */
//READ_UTF8(zIn, zTerm, c);
//      WRITE_UTF16LE(z, c);
//    }
//  }else{
//    Debug.Assert( desiredEnc==SQLITE_UTF16BE );
//    /* UTF-8 -> UTF-16 Big-endian */
//    while( zIn<zTerm ){
///* c = sqlite3Utf8Read(zIn, zTerm, (const u8**)&zIn); */
//READ_UTF8(zIn, zTerm, c);
//      WRITE_UTF16BE(z, c);
//    }
//  }
//  pMem->n = (int)(z - zOut);
//  *z++ = 0;
//}else{
//  Debug.Assert( desiredEnc==SQLITE_UTF8 );
//  if( pMem->enc==SQLITE_UTF16LE ){
//    /* UTF-16 Little-endian -> UTF-8 */
//    while( zIn<zTerm ){
//      READ_UTF16LE(zIn, zIn<zTerm, c); 
//      WRITE_UTF8(z, c);
//    }
//  }else{
//    /* UTF-16 Big-endian -> UTF-8 */
//    while( zIn<zTerm ){
//      READ_UTF16BE(zIn, zIn<zTerm, c); 
//      WRITE_UTF8(z, c);
//    }
//  }
//  pMem->n = (int)(z - zOut);
//}
//*z = 0;
//Debug.Assert( (pMem->n+(desiredEnc==SQLITE_UTF8?1:2))<=len );

//sqlite3VdbeMemRelease(pMem);
//pMem->flags &= ~(MEM_Static|MEM_Dyn|MEM_Ephem);
//pMem->enc = desiredEnc;
//pMem->flags |= (MEM_Term|MEM_Dyn);
//pMem.z = (char*)zOut;
//pMem.zMalloc = pMem.z;

translate_out:
#if TRANSLATE_TRACE && DEBUG
{
char zBuf[100];
sqlite3VdbeMemPrettyPrint(mem, zBuf);
fprintf(stderr, "OUTPUT: %s\n", zBuf);
}
#endif
return RC.OK;
}

static int sqlite3VdbeMemHandleBom(Mem pMem){
int rc = SQLITE_OK;
int bom = 0;
byte[] b01 = new byte[2];
Encoding.Unicode.GetBytes( pMem.z, 0, 1,b01,0 );
assert( pMem->n>=0 );
if( pMem->n>1 ){
//  u8 b1 = *(u8 *)pMem.z;
//  u8 b2 = *(((u8 *)pMem.z) + 1);
if( b01[0]==0xFE && b01[1]==0xFF ){//  if( b1==0xFE && b2==0xFF ){
bom = SQLITE_UTF16BE;
}
if( b01[0]==0xFF && b01[1]==0xFE ){  //  if( b1==0xFF && b2==0xFE ){
bom = SQLITE_UTF16LE;
}
}

if( bom!=0 ){
rc = sqlite3VdbeMemMakeWriteable(pMem);
if( rc==SQLITE_OK ){
pMem.n -= 2;
Debugger.Break (); // TODO -
//memmove(pMem.z, pMem.z[2], pMem.n);
//pMem.z[pMem.n] = '\0';
//pMem.z[pMem.n+1] = '\0';
pMem.flags |= MEM_Term;
pMem.enc = bom;
}
}
return rc;
}
#endif

#if !SQLITE_OMIT_UTF16
static string sqlite3Utf16to8(sqlite3 db, string z, int nByte, u8 enc){
Debugger.Break (); // TODO -
Mem m = Pool.Allocate_Mem();
//  memset(&m, 0, sizeof(m));
//  m.db = db;
//  sqlite3VdbeMemSetStr(&m, z, nByte, enc, SQLITE_STATIC);
//  sqlite3VdbeChangeEncoding(&m, SQLITE_UTF8);
//  if( db.mallocFailed !=0{
//    sqlite3VdbeMemRelease(&m);
//    m.z = 0;
//  }
//  Debug.Assert( (m.flags & MEM_Term)!=0 || db.mallocFailed !=0);
//  Debug.Assert( (m.flags & MEM_Str)!=0 || db.mallocFailed !=0);
  assert( (m.flags & MEM_Dyn)!=0 || db->mallocFailed );
  assert( m.z || db->mallocFailed );
  return m.z;
}

#if ENABLE_STAT3
char *sqlite3Utf8to16(sqlite3 db, u8 enc, char *z, int n, int *pnOut){
  Mem m;
  memset(&m, 0, sizeof(m));
  m.db = db;
  sqlite3VdbeMemSetStr(&m, z, n, SQLITE_UTF8, SQLITE_STATIC);
  if( sqlite3VdbeMemTranslate(&m, enc) ){
    assert( db->mallocFailed );
    return 0;
  }
  assert( m.z==m.zMalloc );
  *pnOut = m.n;
  return m.z;
}
#endif
#endif
  }
}
