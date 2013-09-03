#undef DEBUG
#region Foreign-License
/*
	A Class of the following type is filled in by Parser.parseCommand. It describes a single command parsed from an input string.

Copyright (c) 1997 by Sun Microsystems, Inc.
Copyright (c) 2012 Sky Morey

See the file "license.terms" for information on usage and redistribution of this file, and for a DISCLAIMER OF ALL WARRANTIES.
*/
#endregion
using System;
namespace Tcl.Lang
{

    public class TclParse
    {

        // The original command string passed to Parser.parseCommand. 

        public char[] inString;

        // Index into 'string' that is the character just after the last 
        // one in the command string.

        public int endIndex;

        // Index into 'string' that is the # that begins the first of 
        // one or more comments preceding the command. 

        public int commentStart;

        // Number of bytes in comments (up through newline character 
        // that terminates the last comment).  If there were no
        // comments, this field is 0.

        public int commentSize;

        // Index into 'string' that is the first character in first 
        // word of command.

        public int commandStart;

        // Number of bytes in command, including first character of 
        // first word, up through the terminating newline, close 
        // bracket, or semicolon. 

        public int commandSize;

        // Total number of words in command.  May be 0. 

        public int numWords;

        // Stores the tokens that compose the command.

        public TclToken[] tokenList;

        // Total number of tokens in command. 

        internal int numTokens;

        //  Total number of tokens available at token.

        internal int tokensAvailable;

        /*
        *----------------------------------------------------------------------
        *
        * The fields below are intended only for the private use of the
        * parser.  They should not be used by procedures that invoke
        * Tcl_ParseCommand.
        *
        *----------------------------------------------------------------------
        */

        // Interpreter to use for error reporting, or null.

        internal Interp interp;

        // Name of file from which script came, or null.  Used for error
        // messages.

        internal string fileName;

        // Line number corresponding to first character in string. 

        internal int lineNum;

        // Points to character in string that terminated most recent token. 
        // Filled in by Parser.parseTokens.  If an error occurs, points to
        // beginning of region where the error occurred (e.g. the open brace
        // if the close brace is missing).

        public int termIndex;

        // This field is set to true by Parser.parseCommand if the command
        // appears to be incomplete.  This information is used by 
        // Parser.commandComplete.

        internal bool incomplete;

        // When a TclParse is the return value of a method, result is set to
        // a standard Tcl result, indicating the return of the method.
        public TCL.CompletionCode result;

        // Default size of the tokenList array.

        private const int INITIAL_NUM_TOKENS = 20;
        private const int MAX_CACHED_TOKENS = 50; //my tests show 50 is best

        internal TclParse(Interp interp, char[] inString, int endIndex, string fileName, int lineNum)
        {
            this.interp = interp;
            this.inString = inString;
            this.endIndex = endIndex;
            this.fileName = fileName;
            this.lineNum = lineNum;
            this.tokenList = new TclToken[INITIAL_NUM_TOKENS];
            this.tokensAvailable = INITIAL_NUM_TOKENS;
            this.numTokens = 0;
            this.numWords = 0;
            this.commentStart = -1;
            this.commentSize = 0;
            this.commandStart = -1;
            this.commandSize = 0;
            this.incomplete = false;
        }
        internal TclToken getToken(int index)
        // The index into tokenList.
        {
            if (index >= tokensAvailable)
            {
                expandTokenArray(index);
            }

            if (tokenList[index] == null)
            {
                tokenList[index] = grabToken();
                tokenList[index].script_array = tokenList[0].script_array;
            }
            return tokenList[index];
        }


        // Release internal resources that this TclParser object might have allocated

        internal void release()
        {
            for (int index = 0; index < tokensAvailable; index++)
            {
                if (tokenList[index] != null)
                {
                    releaseToken(tokenList[index]);
                    tokenList[index] = null;
                }
            }
        }




        // Creating an interpreter will cause this init method to be called

        internal static void init(Interp interp)
        {
            TclToken[] TOKEN_CACHE = new TclToken[MAX_CACHED_TOKENS];
            for (int i = 0; i < MAX_CACHED_TOKENS; i++)
            {
                TOKEN_CACHE[i] = new TclToken();
            }

            interp._parserTokens = TOKEN_CACHE;
            interp._parserTokensUsed = 0;
        }


        private TclToken grabToken()
        {
            if (interp == null || interp._parserTokensUsed == MAX_CACHED_TOKENS)
            {
                // either we do not have a cache because the interp is null or we have already
                // used up all the open cache slots, we just allocate a new one in this case
                return new TclToken();
            }
            else
            {
                // the cache has an avaliable slot so grab it
                return interp._parserTokens[interp._parserTokensUsed++];
            }
        }

        private void releaseToken(TclToken token)
        {
            if (interp != null && interp._parserTokensUsed > 0)
            {
                // if cache is not full put the object back in the cache
                interp._parserTokensUsed -= 1;
                interp._parserTokens[interp._parserTokensUsed] = token;
            }
        }


        /*
        //uncommenting these methods will disable caching
		
        static void init(Interp interp) {}
        private TclToken grabToken() {return new TclToken();}
        private void releaseToken(TclToken token) {}
        */



        internal void expandTokenArray(int needed)
        {
            // Make sure there is at least enough room for needed tokens
            while (needed >= tokensAvailable)
            {
                tokensAvailable *= 2;
            }

            TclToken[] newList = new TclToken[tokensAvailable];
            Array.Copy((System.Array)tokenList, 0, (System.Array)newList, 0, tokenList.Length);
            tokenList = newList;
        }

        public override string ToString()
        {

            return (get().ToString());
        }

        public TclObject get()
        {
            TclObject obj;
            TclToken token;
            string typeString;
            int nextIndex;
            string cmd;
            int i;


            System.Diagnostics.Debug.WriteLine("Entered TclParse.get()");
            System.Diagnostics.Debug.WriteLine("numTokens is " + numTokens);

            obj = TclList.NewInstance();
            try
            {
                if (commentSize > 0)
                {
                    TclList.Append(interp, obj, TclString.NewInstance(new string(inString, commentStart, commentSize)));
                }
                else
                {
                    TclList.Append(interp, obj, TclString.NewInstance("-"));
                }

                if (commandStart >= (endIndex + 1))
                {
                    commandStart = endIndex;
                }
                cmd = new string(inString, commandStart, commandSize);
                TclList.Append(interp, obj, TclString.NewInstance(cmd));
                TclList.Append(interp, obj, TclInteger.NewInstance(numWords));

                for (i = 0; i < numTokens; i++)
                {
                    System.Diagnostics.Debug.WriteLine("processing token " + i);

                    token = tokenList[i];
                    switch (token.type)
                    {

                        case Parser.TCL_TOKEN_WORD:
                            typeString = "word";
                            break;

                        case Parser.TCL_TOKEN_SIMPLE_WORD:
                            typeString = "simple";
                            break;

                        case Parser.TCL_TOKEN_EXPAND_WORD:
                            typeString = "expand";
                            break;

                        case Parser.TCL_TOKEN_TEXT:
                            typeString = "text";
                            break;

                        case Parser.TCL_TOKEN_BS:
                            typeString = "backslash";
                            break;

                        case Parser.TCL_TOKEN_COMMAND:
                            typeString = "command";
                            break;

                        case Parser.TCL_TOKEN_VARIABLE:
                            typeString = "variable";
                            break;

                        default:
                            typeString = "??";
                            break;

                    }

                    System.Diagnostics.Debug.WriteLine("typeString is " + typeString);

                    TclList.Append(interp, obj, TclString.NewInstance(typeString));
                    TclList.Append(interp, obj, TclString.NewInstance(token.TokenString));
                    TclList.Append(interp, obj, TclInteger.NewInstance(token.numComponents));
                }
                nextIndex = commandStart + commandSize;
                TclList.Append(interp, obj, TclString.NewInstance(new string(inString, nextIndex, (endIndex - nextIndex))));
            }
            catch (TclException e)
            {
                // Do Nothing.
            }

            return obj;
        }
    } // end TclParse
}
