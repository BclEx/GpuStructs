using System.Diagnostics;
namespace Core.Text
{
    public class StringBuilder
    {
        public object Ctx;			// Optional database for lookaside.  Can be NULL
        public string Base;		// A base allocation.  Not from malloc.
        public System.Text.StringBuilder Text; // The string collected so far
        public int Size;			// Amount of space allocated in zText
        public int MaxSize;		// Maximum allowed string length
        public bool MallocFailed;  // Becomes true if any memory allocation fails
        public bool Overflowed;        // Becomes true if string size exceeds limits

        public void Append(string z, int length)
        {
            Debug.Assert(z != null || length == 0);
            if (Overflowed)
            {
                //ASSERTCOVERAGE(Overflowed);
                return;
            }
            if (length < 0)
                length = z.Length;
            if (length == 0 || SysEx.NEVER(z == null))
                return;
            Text.Append(z.Substring(0, length <= z.Length ? length : z.Length));
        }

        public new string ToString()
        {
            return Text.ToString();
        }

        public void Reset()
        {
            Text.Length = 0;
        }

        public static void Init(StringBuilder b, int capacity, int maxSize)
        {
            b.Text.Length = 0;
            if (b.Text.Capacity < capacity)
                b.Text.Capacity = capacity;
            b.Ctx = null;
            b.MaxSize = maxSize;
        }
    }
}