#pragma region Foreign-License
/*
Copyright (c) 1997 Cornell University.
Copyright (c) 1997 Sun Microsystems, Inc.
Copyright (c) 2012 Sky Morey

See the file "license.terms" for information on usage and redistribution of this file, and for a DISCLAIMER OF ALL WARRANTIES.
*/
#pragma endregion
#include "Runtime.cu.h"

namespace Tcl { namespace Lang
{
	class BackSlashResult
	{
	public:
		char C;
		int NextIndex;
		bool IsWordSep;
		__device__ BackSlashResult(char c, int w)
		{
			C = c;
			NextIndex = w;
			IsWordSep = false;
		}
		__device__ BackSlashResult(char c, int w, bool b)
		{
			C = c;
			NextIndex = w;
			IsWordSep = b;
		}
	};
}}