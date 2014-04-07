using System;
namespace Core
{
    #region Savepoint

    public class Savepoint
    {
        public string Name;			// Savepoint name (nul-terminated)
        public long DeferredCons;	// Number of deferred fk violations
        public Savepoint Next;		// Parent savepoint (if any)
    }

    #endregion
}