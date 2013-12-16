using System;

namespace Core
{
    public struct array_t<T>
    {
        public int length;
        public T[] data;
        public array_t(T[] a) { data = a; length = 0; }
        public array_t(T[] a, int b) { data = a; length = b; }
        public static implicit operator T[](array_t<T> p) { return p.data; }
        public T this[int i] { get { return data[i]; } set { data[i] = value; } }
    };
    public struct array_t2<TLength, T> where TLength : struct
    {
        public TLength length;
        public T[] data;
        public array_t2(T[] a) { data = a; length = default(TLength); }
        public array_t2(T[] a, TLength b) { data = a; length = b; }
        public static implicit operator T[](array_t2<TLength, T> p) { return p.data; }
        public T this[int i] { get { return data[i]; } set { data[i] = value; } }
    };

    public static class cs
    {
        public static int memcmp(byte[] a, byte[] b, int limit)
        {
            if (a.Length < limit)
                return (a.Length < b.Length ? -1 : +1);
            if (b.Length < limit)
                return +1;
            for (int i = 0; i < limit; i++)
            {
                if (a[i] != b[i])
                    return (a[i] < b[i] ? -1 : 1);
            }
            return 0;
        }
        public static int memcmp(string a, byte[] b, int limit)
        {
            if (a.Length < limit)
                return (a.Length < b.Length ? -1 : +1);
            if (b.Length < limit)
                return +1;
            char[] cA = a.ToCharArray();
            for (int i = 0; i < limit; i++)
            {
                if (cA[i] != b[i])
                    return (cA[i] < b[i] ? -1 : 1);
            }
            return 0;
        }
        public static int memcmp(byte[] a, int aOffset, byte[] b, int limit)
        {
            if (a.Length < aOffset + limit)
                return (a.Length - aOffset < b.Length ? -1 : +1);
            if (b.Length < limit)
                return +1;
            for (int i = 0; i < limit; i++)
            {
                if (a[i + aOffset] != b[i])
                    return (a[i + aOffset] < b[i] ? -1 : 1);
            }
            return 0;
        }
        public static int memcmp(byte[] a, int aOffset, byte[] b, int bOffset, int limit)
        {
            if (a.Length < aOffset + limit)
                return (a.Length - aOffset < b.Length - bOffset ? -1 : +1);
            if (b.Length < bOffset + limit)
                return +1;
            for (int i = 0; i < limit; i++)
            {
                if (a[i + aOffset] != b[i + bOffset])
                    return (a[i + aOffset] < b[i + bOffset] ? -1 : 1);
            }
            return 0;
        }
        public static int memcmp(byte[] a, int aOffset, string b, int limit)
        {
            if (a.Length < aOffset + limit)
                return (a.Length - aOffset < b.Length ? -1 : +1);
            if (b.Length < limit)
                return +1;
            for (int i = 0; i < limit; i++)
            {
                if (a[i + aOffset] != b[i])
                    return (a[i + aOffset] < b[i] ? -1 : 1);
            }
            return 0;
        }
        public static int memcmp(string a, string b, int limit)
        {
            if (a.Length < limit)
                return (a.Length < b.Length ? -1 : +1);
            if (b.Length < limit)
                return +1;
            int rc;
            if ((rc = String.Compare(a, 0, b, 0, limit, StringComparison.Ordinal)) == 0)
                return 0;
            return (rc < 0 ? -1 : +1);
        }
    }
}