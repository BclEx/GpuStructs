#include "../Core+Pager/Core+Pager.cu.h"

#pragma region CollSeq
namespace Core {
	struct CollSeq
	{
		char *Name;				// Name of the collating sequence, UTF-8 encoded
		TEXTENCODE Encode;		// Text encoding handled by xCmp()
		void *User;				// First argument to xCmp()
		int (*Cmp)(void *, int, const void *, int, const void *);
		void (*Del)(void *);	// Destructor for pUser
	};

#pragma endregion

#pragma region ISchema

	enum SCHEMA : uint8
	{
		SCHEMA_SchemaLoaded = 0x0001,	// The schema has been loaded
		SCHEMA_UnresetViews = 0x0002,	// Some views have defined column names
		SCHEMA_Empty = 0x0004,			// The file is empty (length 0 bytes)
	};

#define DbHasProperty(D,I,P)     (((D)->DBs[I].Schema->Flags&(P))==(P))
#define DbHasAnyProperty(D,I,P)  (((D)->DBs[I].Schema->Flags&(P))!=0)
#define DbSetProperty(D,I,P)     (D)->DBs[I].Schema->Flags|=(P)
#define DbClearProperty(D,I,P)   (D)->DBs[I].Schema->Flags&=~(P)

	struct ISchema
	{
		//int ShemaCookie;	// Database schema version number for this file
		//int Generation;     // Generation counter.  Incremented with each change
		//Hash TableHash;     // All tables indexed by name
		//Hash IndexHash;     // All (named) indices indexed by name
		//Hash TriggerHash;   // All triggers indexed by name
		//Hash FKeyHash;      // All foreign keys by referenced table name
		//Table *SeqTable;    // The sqlite_sequence table used by AUTOINCREMENT
		//uint8 FileFormat;   // Schema format version for this file
		//uint8 Encode;       // Text encoding used by this database
		SCHEMA Flags;		// Flags associated with this schema
		int CacheSize;		// Number of pages to use in the cache
	};
}
#pragma endregion

#include "Context.cu.h"
#include "Btree.cu.h"

#pragma region IVdbe
namespace Core {
	class IVdbe
	{
	public:
		__device__ virtual UnpackedRecord *AllocUnpackedRecord(KeyInfo *keyInfo, char *space, int spaceLength, char **free);
		__device__ virtual void RecordUnpack(KeyInfo *keyInfo, int keyLength, const void *key, UnpackedRecord *p);
		__device__ virtual void DeleteUnpackedRecord(UnpackedRecord *r);
		__device__ virtual int RecordCompare(int cells, const void *cellKey, UnpackedRecord *idxKey);
	};
}
#pragma endregion
