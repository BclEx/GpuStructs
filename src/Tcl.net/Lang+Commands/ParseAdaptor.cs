#undef DEBUG
#region Foreign-License
/*
	Temporary adaptor class that creates the interface from the current expression parser to the new Parser class.

Copyright (c) 1997 by Sun Microsystems, Inc.
Copyright (c) 2012 Sky Morey

See the file "license.terms" for information on usage and redistribution of this file, and for a DISCLAIMER OF ALL WARRANTIES.
*/
#endregion
using System;
namespace Tcl.Lang
{

    class ParseAdaptor
    {
        internal static ParseResult ParseVar(Interp interp, string inString, int index, int length)
        {
            ParseResult result;

            index--;
            result = Parser.parseVar(interp, inString.Substring(index, (length) - (index)));
            result.nextIndex += index;
            return (result);
        }
        internal static ParseResult parseNestedCmd(Interp interp, string inString, int index, int length)
        {
            CharPointer script;
            TclObject obj;

            // Check for the easy case where the last character in the string is '['.
            if (index == length)
            {
                throw new TclException(interp, "missing close-bracket");
            }

            script = new CharPointer(inString);
            script._index = index;

            interp._evalFlags |= Parser.TCL_BRACKET_TERM;
            Parser.eval2(interp, script._array, script._index, length - index, 0);
            obj = interp.GetResult();
            obj.Preserve();
            return (new ParseResult(obj, index + interp._termOffset + 1));
        }
        internal static ParseResult ParseQuotes(Interp interp, string inString, int index, int length)
        {
            TclObject obj;
            TclParse parse = null;
            TclToken token;
            CharPointer script;

            try
            {

                script = new CharPointer(inString);
                script._index = index;

                parse = new TclParse(interp, script._array, length, null, 0);

                System.Diagnostics.Debug.WriteLine("string is \"" + inString + "\"");
                System.Diagnostics.Debug.WriteLine("script.array is \"" + new string(script._array) + "\"");

                System.Diagnostics.Debug.WriteLine("index is " + index);
                System.Diagnostics.Debug.WriteLine("length is " + length);

                System.Diagnostics.Debug.WriteLine("parse.endIndex is " + parse.endIndex);


                parse.commandStart = script._index;
                token = parse.getToken(0);
                token.type = Parser.TCL_TOKEN_WORD;
                token.script_array = script._array;
                token.script_index = script._index;
                parse.numTokens++;
                parse.numWords++;
                parse = Parser.parseTokens(script._array, script._index, Parser.TYPE_QUOTE, parse);

                // Check for the error condition where the parse did not end on
                // a '"' char. Is this happened raise an error.

                if (script._array[parse.termIndex] != '"')
                {
                    throw new TclException(interp, "missing \"");
                }

                // if there was no error then parsing will continue after the
                // last char that was parsed from the string

                script._index = parse.termIndex + 1;

                // Finish filling in the token for the word and check for the
                // special case of a word consisting of a single range of
                // literal text.

                token = parse.getToken(0);
                token.size = script._index - token.script_index;
                token.numComponents = parse.numTokens - 1;
                if ((token.numComponents == 1) && (parse.getToken(1).type == Parser.TCL_TOKEN_TEXT))
                {
                    token.type = Parser.TCL_TOKEN_SIMPLE_WORD;
                }
                parse.commandSize = script._index - parse.commandStart;
                if (parse.numTokens > 0)
                {
                    obj = Parser.evalTokens(interp, parse.tokenList, 1, parse.numTokens - 1);
                }
                else
                {
                    throw new TclRuntimeError("parseQuotes error: null obj result");
                }
            }
            finally
            {
                parse.release();
            }

            return (new ParseResult(obj, script._index));
        }
        internal static ParseResult parseBraces(Interp interp, string str, int index, int length)
        {
            char[] arr = str.ToCharArray();
            int level = 1;

            for (int i = index; i < length; )
            {
                if (Parser.charType(arr[i]) == Parser.TYPE_NORMAL)
                {
                    i++;
                }
                else if (arr[i] == '}')
                {
                    level--;
                    if (level == 0)
                    {
                        str = new string(arr, index, i - index);
                        return new ParseResult(str, i + 1);
                    }
                    i++;
                }
                else if (arr[i] == '{')
                {
                    level++;
                    i++;
                }
                else if (arr[i] == '\\')
                {
                    BackSlashResult bs = Parser.backslash(arr, i);
                    i = bs.NextIndex;
                }
                else
                {
                    i++;
                }
            }

            //if you run off the end of the string you went too far
            throw new TclException(interp, "missing close-brace");
        }
    } // end ParseAdaptor
}
