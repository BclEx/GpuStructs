MEM_TAG( UNSET )		// This should never be used
	MEM_TAG( STATIC_EXE	)	// The static exe, generally how much memory we are using before our main() function ever runs
	MEM_TAG( DEBUG )		// Crap we don't care about, because it won't be in a retail build
	MEM_TAG( NEW )			// Crap allocated with new which hasn't been given an explicit tag
	MEM_TAG( BLOCKALLOC )	// Crap allocated with idBlockAlloc which hasn't been given an explicit tag
	MEM_TAG( PHYSICAL )
	MEM_TAG( TEMP )
	MEM_TAG( IDLIB_HASH )
	MEM_TAG( IDLIB_LIST )
	MEM_TAG( STRING )
