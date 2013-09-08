#region Foreign-License
/*
	Implements the built-in "binary" Tcl command.

Copyright (c) 1999 Christian Krone.
Copyright (c) 1997 by Sun Microsystems, Inc.
Copyright (c) 2012 Sky Morey

See the file "license.terms" for information on usage and redistribution of this file, and for a DISCLAIMER OF ALL WARRANTIES.
*/
#endregion
using System;
using System.Text;
using System.IO;
namespace Tcl.Lang
{
    /// <summary>
    /// This class implements the built-in "binary" command in Tcl.
    /// </summary>
    class BinaryCmd : ICommand
    {
        private static readonly string[] _validCmds = new string[] { "format", "scan" };
        private const string HEXDIGITS = "0123456789abcdef";
        private const int CMD_FORMAT = 0;
        private const int CMD_SCAN = 1;

        // The following constants are used by GetFormatSpec to indicate various special conditions in the parsing of a format specifier.
        private const int BINARY_ALL = -1; // Use all elements in the argument.
        private const int BINARY_NOCOUNT = -2; // No count was specified in format.
        private const char FORMAT_END = ' '; // End of format was found.

        public TCL.CompletionCode CmdProc(Interp interp, TclObject[] argv)
        {
            int arg; // Index of next argument to consume.
            char[] format = null; // User specified format string.
            char cmd; // Current format character.
            int cursor; // Current position within result buffer.
            int maxPos; // Greatest position within result buffer that cursor has visited.
            int value = 0; // Current integer value to be packed.
            int offset, size = 0, length; // Initialized to avoid compiler warning.
            if (argv.Length < 2)
                throw new TclNumArgsException(interp, 1, argv, "option ?arg arg ...?");
            int cmdIndex = TclIndex.Get(interp, argv[1], _validCmds, "option", 0);

            switch (cmdIndex)
            {
                case CMD_FORMAT:
                    {
                        if (argv.Length < 3)
                            throw new TclNumArgsException(interp, 2, argv, "formatString ?arg arg ...?");
                        // To avoid copying the data, we format the string in two passes. The first pass computes the size of the output buffer.  The
                        // second pass places the formatted data into the buffer.
                        format = argv[2].ToString().ToCharArray();
                        arg = 3;
                        length = 0;
                        offset = 0;
                        int parsePos = 0;
                        while ((cmd = GetFormatSpec(format, ref parsePos)) != FORMAT_END)
                        {
                            int count = GetFormatCount(format, ref parsePos);
                            switch (cmd)
                            {
                                case 'a':
                                case 'A':
                                case 'b':
                                case 'B':
                                case 'h':
                                case 'H':
                                    {
                                        // For string-type specifiers, the count corresponds to the number of bytes in a single argument.
                                        if (arg >= argv.Length)
                                            MissingArg(interp);
                                        if (count == BINARY_ALL)
                                            count = TclByteArray.getLength(interp, argv[arg]);
                                        else if (count == BINARY_NOCOUNT)
                                            count = 1;
                                        arg++;
                                        switch (cmd)
                                        {
                                            case 'a':
                                            case 'A':
                                                offset += count;
                                                break;
                                            case 'b':
                                            case 'B':
                                                offset += (count + 7) / 8;
                                                break;
                                            case 'h':
                                            case 'H':
                                                offset += (count + 1) / 2;
                                                break;
                                        }
                                        break;
                                    }
                                case 'c':
                                case 's':
                                case 'S':
                                case 'i':
                                case 'I':
                                case 'f':
                                case 'd':
                                    {
                                        if (arg >= argv.Length)
                                            MissingArg(interp);
                                        switch (cmd)
                                        {
                                            case 'c':
                                                size = 1;
                                                break;
                                            case 's':
                                            case 'S':
                                                size = 2;
                                                break;
                                            case 'i':
                                            case 'I':
                                                size = 4;
                                                break;
                                            case 'f':
                                                size = 4;
                                                break;
                                            case 'd':
                                                size = 8;
                                                break;
                                        }
                                        // For number-type specifiers, the count corresponds to the number of elements in the list stored in
                                        // a single argument.  If no count is specified, then the argument is taken as a single non-list value.
                                        if (count == BINARY_NOCOUNT)
                                        {
                                            arg++;
                                            count = 1;
                                        }
                                        else
                                        {
                                            int listc = TclList.getLength(interp, argv[arg++]);
                                            if (count == BINARY_ALL)
                                                count = listc;
                                            else if (count > listc)
                                                throw new TclException(interp, "number of elements in list" + " does not match count");
                                        }
                                        offset += count * size;
                                        break;
                                    }
                                case 'x':
                                    {
                                        if (count == BINARY_ALL)
                                            throw new TclException(interp, "cannot use \"*\"" + " in format string with \"x\"");
                                        if (count == BINARY_NOCOUNT)
                                            count = 1;
                                        offset += count;
                                        break;
                                    }

                                case 'X':
                                    {
                                        if (count == BINARY_NOCOUNT)
                                            count = 1;
                                        if (count > offset || count == BINARY_ALL)
                                            count = offset;
                                        if (offset > length)
                                            length = offset;
                                        offset -= count;
                                        break;
                                    }

                                case '@':
                                    {
                                        if (offset > length)
                                            length = offset;
                                        if (count == BINARY_ALL)
                                            offset = length;
                                        else if (count == BINARY_NOCOUNT)
                                            AlephWithoutCount(interp);
                                        else
                                            offset = count;
                                        break;
                                    }
                                default:
                                    {
                                        BadField(interp, cmd);
                                    }
                                    break;

                            }
                        }
                        if (offset > length)
                            length = offset;
                        if (length == 0)
                            return TCL.CompletionCode.RETURN;

                        // Prepare the result object by preallocating the calculated number of bytes and filling with nulls.
                        TclObject resultObj = TclByteArray.NewInstance();
                        byte[] resultBytes = TclByteArray.SetLength(interp, resultObj, length);
                        interp.SetResult(resultObj);
                        // Pack the data into the result object.  Note that we can skip the error checking during this pass, since we have already parsed the string once.
                        arg = 3;
                        cursor = 0;
                        maxPos = cursor;
                        parsePos = 0;
                        while ((cmd = GetFormatSpec(format, ref parsePos)) != FORMAT_END)
                        {
                            int count = GetFormatCount(format, ref parsePos);
                            if (count == 0 && cmd != '@')
                            {
                                arg++;
                                continue;
                            }
                            switch (cmd)
                            {
                                case 'a':
                                case 'A':
                                    {
                                        byte pad = (cmd == 'a' ? (byte)0 : (byte)SupportClass.Identity(' '));
                                        byte[] bytes = TclByteArray.getBytes(interp, argv[arg++]);
                                        length = bytes.Length;
                                        if (count == BINARY_ALL)
                                            count = length;
                                        else if (count == BINARY_NOCOUNT)
                                            count = 1;
                                        if (length >= count)
                                            Array.Copy(bytes, 0, resultBytes, cursor, count);
                                        else
                                        {
                                            Array.Copy(bytes, 0, resultBytes, cursor, length);
                                            for (int ix = 0; ix < count - length; ix++)
                                                resultBytes[cursor + length + ix] = pad;
                                        }
                                        cursor += count;
                                        break;
                                    }
                                case 'b':
                                case 'B':
                                    {
                                        char[] str = argv[arg++].ToString().ToCharArray();
                                        if (count == BINARY_ALL)
                                            count = str.Length;
                                        else if (count == BINARY_NOCOUNT)
                                            count = 1;
                                        int last = cursor + ((count + 7) / 8);
                                        if (count > str.Length)
                                            count = str.Length;
                                        if (cmd == 'B')
                                        {
                                            for (offset = 0; offset < count; offset++)
                                            {
                                                value <<= 1;
                                                if (str[offset] == '1')
                                                    value |= 1;
                                                else if (str[offset] != '0')
                                                    ExpectedButGot(interp, "binary", new string(str));
                                                if (((offset + 1) % 8) == 0)
                                                {
                                                    resultBytes[cursor++] = (byte)value;
                                                    value = 0;
                                                }
                                            }
                                        }
                                        else
                                        {
                                            for (offset = 0; offset < count; offset++)
                                            {
                                                value >>= 1;
                                                if (str[offset] == '1')
                                                    value |= 128;
                                                else if (str[offset] != '0')
                                                    ExpectedButGot(interp, "binary", new string(str));
                                                if (((offset + 1) % 8) == 0)
                                                {
                                                    resultBytes[cursor++] = (byte)value;
                                                    value = 0;
                                                }
                                            }
                                        }
                                        if ((offset % 8) != 0)
                                        {
                                            if (cmd == 'B')
                                                value <<= 8 - (offset % 8);
                                            else
                                                value >>= 8 - (offset % 8);
                                            resultBytes[cursor++] = (byte)value;
                                        }
                                        while (cursor < last)
                                            resultBytes[cursor++] = 0;
                                        break;
                                    }

                                case 'h':
                                case 'H':
                                    {
                                        char[] str = argv[arg++].ToString().ToCharArray();
                                        if (count == BINARY_ALL)
                                            count = str.Length;
                                        else if (count == BINARY_NOCOUNT)
                                            count = 1;
                                        int last = cursor + ((count + 1) / 2);
                                        if (count > str.Length)
                                            count = str.Length;
                                        if (cmd == 'H')
                                        {
                                            for (offset = 0; offset < count; offset++)
                                            {
                                                value <<= 4;
                                                int c = HEXDIGITS.IndexOf(Char.ToLower(str[offset]));
                                                if (c < 0)
                                                    ExpectedButGot(interp, "hexadecimal", new string(str));
                                                value |= (c & 0xf);
                                                if ((offset % 2) != 0)
                                                {
                                                    resultBytes[cursor++] = (byte)value;
                                                    value = 0;
                                                }
                                            }
                                        }
                                        else
                                        {
                                            for (offset = 0; offset < count; offset++)
                                            {
                                                value >>= 4;
                                                int c = HEXDIGITS.IndexOf(Char.ToLower(str[offset]));
                                                if (c < 0)
                                                    ExpectedButGot(interp, "hexadecimal", new string(str));
                                                value |= ((c << 4) & 0xf0);
                                                if ((offset % 2) != 0)
                                                {
                                                    resultBytes[cursor++] = (byte)value;
                                                    value = 0;
                                                }
                                            }
                                        }
                                        if ((offset % 2) != 0)
                                        {
                                            if (cmd == 'H')
                                                value <<= 4;
                                            else
                                                value >>= 4;
                                            resultBytes[cursor++] = (byte)value;
                                        }
                                        while (cursor < last)
                                            resultBytes[cursor++] = 0;
                                        break;
                                    }
                                case 'c':
                                case 's':
                                case 'S':
                                case 'i':
                                case 'I':
                                case 'f':
                                case 'd':
                                    {
                                        TclObject[] listv;
                                        if (count == BINARY_NOCOUNT)
                                        {
                                            listv = new TclObject[1];
                                            listv[0] = argv[arg++];
                                            count = 1;
                                        }
                                        else
                                        {
                                            listv = TclList.getElements(interp, argv[arg++]);
                                            if (count == BINARY_ALL)
                                                count = listv.Length;
                                        }
                                        for (int ix = 0; ix < count; ix++)
                                            cursor = FormatNumber(interp, cmd, listv[ix], resultBytes, cursor);
                                        break;
                                    }
                                case 'x':
                                    {
                                        if (count == BINARY_NOCOUNT)
                                            count = 1;
                                        for (int ix = 0; ix < count; ix++)
                                            resultBytes[cursor++] = 0;
                                        break;
                                    }

                                case 'X':
                                    {
                                        if (cursor > maxPos)
                                            maxPos = cursor;
                                        if (count == BINARY_NOCOUNT)
                                            count = 1;
                                        if (count == BINARY_ALL || count > cursor)
                                            cursor = 0;
                                        else
                                            cursor -= count;
                                        break;
                                    }
                                case '@':
                                    {
                                        if (cursor > maxPos)
                                            maxPos = cursor;
                                        if (count == BINARY_ALL)
                                            cursor = maxPos;
                                        else
                                            cursor = count;
                                        break;
                                    }
                            }
                        }
                        break;
                    }

                case CMD_SCAN:
                    {
                        if (argv.Length < 4)
                            throw new TclNumArgsException(interp, 2, argv, "value formatString ?varName varName ...?");
                        byte[] src = TclByteArray.getBytes(interp, argv[2]);
                        length = src.Length;
                        format = argv[3].ToString().ToCharArray();
                        arg = 4;
                        cursor = 0;
                        offset = 0;
                        int parsePos = 0;
                        while ((cmd = GetFormatSpec(format, ref parsePos)) != FORMAT_END)
                        {
                            int count = GetFormatCount(format, ref parsePos);
                            switch (cmd)
                            {
                                case 'a':
                                case 'A':
                                    {
                                        if (arg >= argv.Length)
                                            MissingArg(interp);
                                        if (count == BINARY_ALL)
                                            count = length - offset;
                                        else
                                        {
                                            if (count == BINARY_NOCOUNT)
                                                count = 1;
                                            if (count > length - offset)
                                                break;
                                        }
                                        size = count;
                                        // Trim trailing nulls and spaces, if necessary.
                                        if (cmd == 'A')
                                            while (size > 0)
                                            {
                                                if (src[offset + size - 1] != '\x0000' && src[offset + size - 1] != ' ')
                                                    break;
                                                size--;
                                            }
                                        interp.SetVar(argv[arg++], TclByteArray.NewInstance(src, offset, size), 0);
                                        offset += count;
                                        break;
                                    }
                                case 'b':
                                case 'B':
                                    {
                                        if (arg >= argv.Length)
                                            MissingArg(interp);
                                        if (count == BINARY_ALL)
                                            count = (length - offset) * 8;
                                        else
                                        {
                                            if (count == BINARY_NOCOUNT)
                                                count = 1;
                                            if (count > (length - offset) * 8)
                                                break;
                                        }
                                        StringBuilder s = new StringBuilder(count);
                                        int thisOffset = offset;
                                        if (cmd == 'b')
                                        {
                                            for (int ix = 0; ix < count; ix++)
                                            {
                                                if ((ix % 8) != 0)
                                                    value >>= 1;
                                                else
                                                    value = src[thisOffset++];
                                                s.Append((value & 1) != 0 ? '1' : '0');
                                            }
                                        }
                                        else
                                        {
                                            for (int ix = 0; ix < count; ix++)
                                            {
                                                if ((ix % 8) != 0)
                                                    value <<= 1;
                                                else
                                                    value = src[thisOffset++];
                                                s.Append((value & 0x80) != 0 ? '1' : '0');
                                            }
                                        }
                                        interp.SetVar(argv[arg++], TclString.NewInstance(s.ToString()), 0);
                                        offset += (count + 7) / 8;
                                        break;
                                    }
                                case 'h':
                                case 'H':
                                    {
                                        if (arg >= argv.Length)
                                            MissingArg(interp);
                                        if (count == BINARY_ALL)
                                            count = (length - offset) * 2;
                                        else
                                        {
                                            if (count == BINARY_NOCOUNT)
                                                count = 1;
                                            if (count > (length - offset) * 2)
                                                break;
                                        }
                                        StringBuilder s = new StringBuilder(count);
                                        int thisOffset = offset;
                                        if (cmd == 'h')
                                        {
                                            for (int ix = 0; ix < count; ix++)
                                            {
                                                if ((ix % 2) != 0)
                                                    value >>= 4;
                                                else
                                                    value = src[thisOffset++];
                                                s.Append(HEXDIGITS[value & 0xf]);
                                            }
                                        }
                                        else
                                        {
                                            for (int ix = 0; ix < count; ix++)
                                            {
                                                if ((ix % 2) != 0)
                                                    value <<= 4;
                                                else
                                                    value = src[thisOffset++];
                                                s.Append(HEXDIGITS[value >> 4 & 0xf]);
                                            }
                                        }
                                        interp.SetVar(argv[arg++], TclString.NewInstance(s.ToString()), 0);
                                        offset += (count + 1) / 2;
                                        break;
                                    }
                                case 'c':
                                case 's':
                                case 'S':
                                case 'i':
                                case 'I':
                                case 'f':
                                case 'd':
                                    {
                                        if (arg >= argv.Length)
                                            MissingArg(interp);
                                        switch (cmd)
                                        {
                                            case 'c':
                                                size = 1;
                                                break;
                                            case 's':
                                            case 'S':
                                                size = 2;
                                                break;
                                            case 'i':
                                            case 'I':
                                                size = 4;
                                                break;
                                            case 'f':
                                                size = 4;
                                                break;
                                            case 'd':
                                                size = 8;
                                                break;
                                        }
                                        TclObject valueObj;
                                        if (count == BINARY_NOCOUNT)
                                        {
                                            if (length - offset < size)
                                                break;
                                            valueObj = ScanNumber(src, offset, cmd);
                                            offset += size;
                                        }
                                        else
                                        {
                                            if (count == BINARY_ALL)
                                                count = (length - offset) / size;
                                            if (length - offset < count * size)
                                                break;
                                            valueObj = TclList.NewInstance();
                                            int thisOffset = offset;
                                            for (int ix = 0; ix < count; ix++)
                                            {
                                                TclList.Append(null, valueObj, ScanNumber(src, thisOffset, cmd));
                                                thisOffset += size;
                                            }
                                            offset += count * size;
                                        }
                                        interp.SetVar(argv[arg++], valueObj, 0);
                                        break;
                                    }
                                case 'x':
                                    {
                                        if (count == BINARY_NOCOUNT)
                                            count = 1;
                                        if (count == BINARY_ALL || count > length - offset)
                                            offset = length;
                                        else
                                            offset += count;
                                        break;
                                    }
                                case 'X':
                                    {
                                        if (count == BINARY_NOCOUNT)
                                            count = 1;
                                        if (count == BINARY_ALL || count > offset)
                                            offset = 0;
                                        else
                                            offset -= count;
                                        break;
                                    }
                                case '@':
                                    {
                                        if (count == BINARY_NOCOUNT)
                                            AlephWithoutCount(interp);
                                        if (count == BINARY_ALL || count > length)
                                            offset = length;
                                        else
                                            offset = count;
                                        break;
                                    }
                                default:
                                    {
                                        BadField(interp, cmd);
                                    }
                                    break;
                            }
                        }
                        // Set the result to the last position of the cursor.
                        interp.SetResult(arg - 4);
                    }
                    break;
            }
            return TCL.CompletionCode.RETURN;
        }

        private char GetFormatSpec(char[] format, ref int parsePos) // Current position in input.
        
        {
            int ix = parsePos;
            // Skip any leading blanks.
            while (ix < format.Length && format[ix] == ' ')
                ix++;
            // The string was empty, except for whitespace, so fail.
            if (ix >= format.Length)
            {
                parsePos = ix;
                return FORMAT_END;
            }
            // Extract the command character.
            parsePos = ix + 1;
            return format[ix++];
        }

        private int GetFormatCount(char[] format, ref int parsePos) // Current position in input.
        {
            int ix = parsePos;
            // Extract any trailing digits or '*'.
            if (ix < format.Length && format[ix] == '*')
            {
                parsePos = ix + 1;
                return BINARY_ALL;
            }
            else if (ix < format.Length && char.IsDigit(format[ix]))
            {
                int length = 1;
                while (ix + length < format.Length && char.IsDigit(format[ix + length]))
                    length++;
                parsePos = ix + length;
                return int.Parse(new string(format, ix, length));
            }
            else
                return BINARY_NOCOUNT;
        }

        internal static int FormatNumber(Interp interp, char type, TclObject src, byte[] resultBytes, int cursor)
        {
            if (type == 'd')
            {
                double dvalue = TclDouble.Get(interp, src);
                MemoryStream ms = new MemoryStream(resultBytes, cursor, 8);
                BinaryWriter writer = new BinaryWriter(ms);
                writer.Write(dvalue);
                cursor += 8;
                writer.Close();
                ms.Close();
            }
            else if (type == 'f')
            {
                float fvalue = (float)TclDouble.Get(interp, src);
                MemoryStream ms = new MemoryStream(resultBytes, cursor, 4);
                BinaryWriter writer = new BinaryWriter(ms);
                writer.Write(fvalue);
                cursor += 4;
                writer.Close();
                ms.Close();
            }
            else
            {
                int value = TclInteger.Get(interp, src);
                if (type == 'c')
                    resultBytes[cursor++] = (byte)value;
                else if (type == 's')
                {
                    resultBytes[cursor++] = (byte)value;
                    resultBytes[cursor++] = (byte)(value >> 8);
                }
                else if (type == 'S')
                {
                    resultBytes[cursor++] = (byte)(value >> 8);
                    resultBytes[cursor++] = (byte)value;
                }
                else if (type == 'i')
                {
                    resultBytes[cursor++] = (byte)value;
                    resultBytes[cursor++] = (byte)(value >> 8);
                    resultBytes[cursor++] = (byte)(value >> 16);
                    resultBytes[cursor++] = (byte)(value >> 24);
                }
                else if (type == 'I')
                {
                    resultBytes[cursor++] = (byte)(value >> 24);
                    resultBytes[cursor++] = (byte)(value >> 16);
                    resultBytes[cursor++] = (byte)(value >> 8);
                    resultBytes[cursor++] = (byte)value;
                }
            }
            return cursor;
        }

        private static TclObject ScanNumber(byte[] src, int pos, int type) // Format character from "binary scan"
        {
            switch (type)
            {
                case 'c':
                    {
                        return TclInteger.NewInstance((sbyte)src[pos]);
                    }
                case 's':
                    {
                        short value = (short)((src[pos] & 0xff) + ((src[pos + 1] & 0xff) << 8));
                        return TclInteger.NewInstance((int)value);
                    }
                case 'S':
                    {
                        short value = (short)((src[pos + 1] & 0xff) + ((src[pos] & 0xff) << 8));
                        return TclInteger.NewInstance((int)value);
                    }
                case 'i':
                    {
                        int value = (src[pos] & 0xff) + ((src[pos + 1] & 0xff) << 8) + ((src[pos + 2] & 0xff) << 16) + ((src[pos + 3] & 0xff) << 24);
                        return TclInteger.NewInstance(value);
                    }
                case 'I':
                    {
                        int value = (src[pos + 3] & 0xff) + ((src[pos + 2] & 0xff) << 8) + ((src[pos + 1] & 0xff) << 16) + ((src[pos] & 0xff) << 24);
                        return TclInteger.NewInstance(value);
                    }
                case 'f':
                    {
                        MemoryStream ms = new MemoryStream(src, pos, 4, false);
                        BinaryReader reader = new BinaryReader(ms);
                        double fvalue = reader.ReadSingle();
                        reader.Close();
                        ms.Close();
                        return TclDouble.NewInstance(fvalue);
                    }
                case 'd':
                    {
                        MemoryStream ms = new MemoryStream(src, pos, 8, false);
                        BinaryReader reader = new BinaryReader(ms);
                        double dvalue = reader.ReadDouble();
                        reader.Close();
                        ms.Close();
                        return TclDouble.NewInstance(dvalue);
                    }
            }
            return null;
        }

        /// <summary>
        /// Called whenever a format specifier was detected but there are not enough arguments specified.
        /// </summary>
        /// <param name="interp">
        /// The TclInterp which called the cmdProc method.
        /// </param>
        private static void MissingArg(Interp interp)
        {
            throw new TclException(interp, "not enough arguments for all format specifiers");
        }

        /// <summary>
        /// Called whenever an invalid format specifier was detected.
        /// </summary>
        /// <param name="interp">
        /// The TclInterp which called the cmdProc method.
        /// </param>
        /// <param name="cmd">
        /// The invalid field specifier.
        /// </param>
        private static void BadField(Interp interp, char cmd)
        {
            throw new TclException(interp, "bad field specifier \"" + cmd + "\"");
        }

        /// <summary>
        /// Called whenever a letter aleph character (@) was detected but there was no count specified.
        /// </summary>
        /// <param name="interp">
        /// The TclInterp which called the cmdProc method.
        /// </param>
        private static void AlephWithoutCount(Interp interp)
        {
            throw new TclException(interp, "missing count for \"@\" field specifier");
        }

        /// <summary>
        /// Called whenever a format was found which restricts the valid range of characters in the specified string, but the string contains
        /// at least one char not in this range.
        /// </summary>
        /// <param name="interp">
        /// The TclInterp which called the cmdProc method.
        /// </param>
        private static void ExpectedButGot(Interp interp, string expected, string str)
        {
            throw new TclException(interp, "expected " + expected + " string but got \"" + str + "\" instead");
        }
    }
}
