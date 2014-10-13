using System;
using System.Diagnostics;
using System.Text;

namespace Core
{
    public partial class SysEx
    {
        static byte[] _utf8Trans1 = new byte[] 
    {
0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
0x00, 0x01, 0x02, 0x03, 0x00, 0x01, 0x00, 0x00,
};

        public static uint Utf8Read(string z, ref string zNext)
        {
            // Same as READ_UTF8() above but without the zTerm parameter. For this routine, we assume the UTF8 string is always zero-terminated.
            if (string.IsNullOrEmpty(z)) return 0;
            int zIdx = 0;
            uint c = z[zIdx++];
            if (c >= 0xc0)
            {
                //c = _utf8Trans1[c - 0xc0];
                while (zIdx != z.Length && (z[zIdx] & 0xc0) == 0x80)
                    c = (uint)((c << 6) + (0x3f & z[zIdx++]));
                if (c < 0x80 || (c & 0xFFFFF800) == 0xD800 || (c & 0xFFFFFFFE) == 0xFFFE) c = 0xFFFD;
            }
            zNext = z.Substring(zIdx);
            return c;
        }

        public static int Utf8CharLen(string z, int bytes)
        {
            if (z.Length == 0) return 0;
            int zLength = z.Length;
            int zTerm = (bytes >= 0 && bytes <= zLength ? bytes : zLength);
            if (zTerm == zLength)
                return zLength - (z[zTerm - 1] == 0 ? 1 : 0);
            else
                return bytes;
        }

#if TEST && DEBUG
        public static int Utf8To8(byte[] z)
        {
            try
            {
                string z2 = Encoding.UTF8.GetString(z, 0, z.Length);
                byte[] zOut = Encoding.UTF8.GetBytes(z2);
                {
                    Array.Copy(zOut, 0, z, 0, z.Length);
                    return z.Length;
                }
            }
            catch (EncoderFallbackException) { return 0; }
        }
#endif

#if !OMIT_UTF16
        public static int Utf16ByteLen(byte[] z, int chars)
        {
            string z2 = Encoding.UTF32.GetString(z, 0, z.Length);
            return Encoding.UTF32.GetBytes(z2).Length;
        }
#endif
    }
}
