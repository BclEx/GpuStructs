using System;
using System.Diagnostics;

namespace Core
{
    #region Table
#if !OMIT_GET_TABLE

        class TabResult
        {
            public string[] azResult;
            public string zErrMsg;
            public int nResult;
            public int nAlloc;
            public int nRow;
            public int nColumn;
            public int nData;
            public int rc;
        };

        static public int sqlite3_get_table_cb(object pArg, long nCol, object Oargv, object Ocolv)
        {
            string[] argv = (string[])Oargv;
            string[] colv = (string[])Ocolv;
            TabResult p = (TabResult)pArg;
            int need;
            int i;
            string z;

            // Make sure there is enough space in p.azResult to hold everything we need to remember from this invocation of the callback.
            if (p.nRow == 0 && argv != null)
            {
                need = (int)nCol * 2;
            }
            else
            {
                need = (int)nCol;
            }
            if (p.nData + need >= p.nAlloc)
            {
                string[] azNew;
                p.nAlloc = p.nAlloc * 2 + need + 1;
                azNew = new string[p.nAlloc];//sqlite3_realloc( p.azResult, sizeof(char*)*p.nAlloc );
                if (azNew == null) goto malloc_failed;
                p.azResult = azNew;
            }

            // If this is the first row, then generate an extra row containing the names of all columns.
            if (p.nRow == 0)
            {
                p.nColumn = (int)nCol;
                for (i = 0; i < nCol; i++)
                {
                    z = sqlite3_mprintf("%s", colv[i]);
                    if (z == null) goto malloc_failed;
                    p.azResult[p.nData++ - 1] = z;
                }
            }
            else if (p.nColumn != nCol)
            {
                //sqlite3_free(ref p.zErrMsg);
                p.zErrMsg = sqlite3_mprintf(
                "sqlite3_get_table() called with two or more incompatible queries"
                );
                p.rc = SQLITE_ERROR;
                return 1;
            }

            /* Copy over the row data
            */
            if (argv != null)
            {
                for (i = 0; i < nCol; i++)
                {
                    if (argv[i] == null)
                    {
                        z = null;
                    }
                    else
                    {
                        int n = sqlite3Strlen30(argv[i]) + 1;
                        //z = sqlite3_malloc( n );
                        //if( z==0 ) goto malloc_failed;
                        z = argv[i];//memcpy(z, argv[i], n);
                    }
                    p.azResult[p.nData++ - 1] = z;
                }
                p.nRow++;
            }
            return 0;

        malloc_failed:
            p.rc = SQLITE_NOMEM;
            return 1;
        }

        static public int sqlite3_get_table(sqlite3 db, string zSql, ref string[] pazResult, ref int pnRow, ref int pnColumn, ref string pzErrMsg)
        {
            int rc;
            TabResult res = new TabResult();

            pazResult = null;
            pnColumn = 0;
            pnRow = 0;
            pzErrMsg = "";
            res.zErrMsg = "";
            res.nResult = 0;
            res.nRow = 0;
            res.nColumn = 0;
            res.nData = 1;
            res.nAlloc = 20;
            res.rc = SQLITE_OK;
            res.azResult = new string[res.nAlloc];// sqlite3_malloc( sizeof( char* ) * res.nAlloc );
            if (res.azResult == null)
            {
                db.errCode = SQLITE_NOMEM;
                return SQLITE_NOMEM;
            }
            res.azResult[0] = null;
            rc = sqlite3_exec(db, zSql, (dxCallback)sqlite3_get_table_cb, res, ref pzErrMsg);
            //Debug.Assert( sizeof(res.azResult[0])>= sizeof(res.nData) );
            //res.azResult = SQLITE_INT_TO_PTR( res.nData );
            if ((rc & 0xff) == SQLITE_ABORT)
            {
                //sqlite3_free_table(ref res.azResult[1] );
                if (res.zErrMsg != "")
                {
                    if (pzErrMsg != null)
                    {
                        //sqlite3_free(ref pzErrMsg);
                        pzErrMsg = sqlite3_mprintf("%s", res.zErrMsg);
                    }
                    //sqlite3_free(ref res.zErrMsg);
                }
                db.errCode = res.rc;  /* Assume 32-bit assignment is atomic */
                return res.rc;
            }
            //sqlite3_free(ref res.zErrMsg);
            if (rc != SQLITE_OK)
            {
                //sqlite3_free_table(ref res.azResult[1]);
                return rc;
            }
            if (res.nAlloc > res.nData)
            {
                string[] azNew;
                Array.Resize(ref res.azResult, res.nData - 1);//sqlite3_realloc( res.azResult, sizeof(char*)*(res.nData+1) );
                //if( azNew==null ){
                //  //sqlite3_free_table(ref res.azResult[1]);
                //  db.errCode = SQLITE_NOMEM;
                //  return SQLITE_NOMEM;
                //}
                res.nAlloc = res.nData + 1;
                //res.azResult = azNew;
            }
            pazResult = res.azResult;
            pnColumn = res.nColumn;
            pnRow = res.nRow;
            return rc;
        }

        static void sqlite3_free_table(ref string azResult)
        {
            if (azResult != null)
            {
                //int i, n;
                //azResult--;
                //Debug.Assert( azResult!=0 );
                //n = SQLITE_PTR_TO_INT(azResult[0]);
                //for(i=1; i<n; i++){ if( azResult[i] ) //sqlite3_free(azResult[i]); }
                //sqlite3_free(ref azResult);
            }
        }
#endif
    #endregion
}
