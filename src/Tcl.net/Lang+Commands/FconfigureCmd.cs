#region Foreign-License
/*
Copyright (c) 2001 Bruce A. Johnson
Copyright (c) 1997 Sun Microsystems, Inc.
Copyright (c) 2012 Sky Morey

See the file "license.terms" for information on usage and redistribution of this file, and for a DISCLAIMER OF ALL WARRANTIES.
*/
#endregion
using System;
using System.Text;

namespace Tcl.Lang
{

    /// <summary> This class implements the built-in "fconfigure" command in Tcl.</summary>

    class FconfigureCmd : ICommand
    {

        private static readonly string[] validCmds = new string[] { "-blocking", "-buffering", "-buffersize", "-encoding", "-eofchar", "-translation" };

        internal const int OPT_BLOCKING = 0;
        internal const int OPT_BUFFERING = 1;
        internal const int OPT_BUFFERSIZE = 2;
        internal const int OPT_ENCODING = 3;
        internal const int OPT_EOFCHAR = 4;
        internal const int OPT_TRANSLATION = 5;


        /// <summary> This procedure is invoked to process the "fconfigure" Tcl command.
        /// See the user documentation for details on what it does.
        /// 
        /// </summary>
        /// <param name="interp">the current interpreter.
        /// </param>
        /// <param name="argv">command arguments.
        /// </param>

        public TCL.CompletionCode CmdProc(Interp interp, TclObject[] argv)
        {

            Channel chan; // The channel being operated on this method

            if ((argv.Length < 2) || (((argv.Length % 2) == 1) && (argv.Length != 3)))
            {
                throw new TclNumArgsException(interp, 1, argv, "channelId ?optionName? ?value? ?optionName value?...");
            }


            chan = TclIO.getChannel(interp, argv[1].ToString());
            if (chan == null)
            {

                throw new TclException(interp, "can not find channel named \"" + argv[1].ToString() + "\"");
            }

            if (argv.Length == 2)
            {
                // return list of all name/value pairs for this channelId
                TclObject list = TclList.NewInstance();

                TclList.Append(interp, list, TclString.NewInstance("-blocking"));
                TclList.Append(interp, list, TclBoolean.newInstance(chan.Blocking));

                TclList.Append(interp, list, TclString.NewInstance("-buffering"));
                TclList.Append(interp, list, TclString.NewInstance(TclIO.getBufferingString(chan.Buffering)));

                TclList.Append(interp, list, TclString.NewInstance("-buffersize"));
                TclList.Append(interp, list, TclInteger.NewInstance(chan.BufferSize));

                // -encoding

                TclList.Append(interp, list, TclString.NewInstance("-encoding"));

                System.Text.Encoding javaEncoding = chan.Encoding;
                string tclEncoding;
                if ((System.Object)javaEncoding == null)
                {
                    tclEncoding = "binary";
                }
                else
                {
                    tclEncoding = EncodingCmd.getTclName(javaEncoding);
                }
                TclList.Append(interp, list, TclString.NewInstance(tclEncoding));

                // -eofchar

                TclList.Append(interp, list, TclString.NewInstance("-eofchar"));
                if (chan.ReadOnly)
                {
                    char eofChar = chan.InputEofChar;
                    TclList.Append(interp, list, (eofChar == 0) ? TclString.NewInstance("") : TclString.NewInstance(eofChar));
                }
                else if (chan.WriteOnly)
                {
                    char eofChar = chan.OutputEofChar;
                    TclList.Append(interp, list, (eofChar == 0) ? TclString.NewInstance("") : TclString.NewInstance(eofChar));
                }
                else if (chan.ReadWrite)
                {
                    char inEofChar = chan.InputEofChar;
                    char outEofChar = chan.OutputEofChar;

                    TclObject eofchar_pair = TclList.NewInstance();

                    TclList.Append(interp, eofchar_pair, (inEofChar == 0) ? TclString.NewInstance("") : TclString.NewInstance(inEofChar));

                    TclList.Append(interp, eofchar_pair, (outEofChar == 0) ? TclString.NewInstance("") : TclString.NewInstance(outEofChar));

                    TclList.Append(interp, list, eofchar_pair);
                }
                else
                {
                    // Not readable or writeable, do nothing
                }

                // -translation

                TclList.Append(interp, list, TclString.NewInstance("-translation"));

                if (chan.ReadOnly)
                {
                    TclList.Append(interp, list, TclString.NewInstance(TclIO.getTranslationString(chan.InputTranslation)));
                }
                else if (chan.WriteOnly)
                {
                    TclList.Append(interp, list, TclString.NewInstance(TclIO.getTranslationString(chan.OutputTranslation)));
                }
                else if (chan.ReadWrite)
                {
                    TclObject translation_pair = TclList.NewInstance();

                    TclList.Append(interp, translation_pair, TclString.NewInstance(TclIO.getTranslationString(chan.InputTranslation)));
                    TclList.Append(interp, translation_pair, TclString.NewInstance(TclIO.getTranslationString(chan.OutputTranslation)));

                    TclList.Append(interp, list, translation_pair);
                }
                else
                {
                    // Not readable or writeable, do nothing
                }

                interp.setResult(list);
            }

            if (argv.Length == 3)
            {
                // return value for supplied name

                int index = TclIndex.get(interp, argv[2], validCmds, "option", 0);

                switch (index)
                {

                    case OPT_BLOCKING:
                        {
                            // -blocking
                            interp.setResult(chan.Blocking);
                            break;
                        }

                    case OPT_BUFFERING:
                        {
                            // -buffering
                            interp.setResult(TclIO.getBufferingString(chan.Buffering));
                            break;
                        }

                    case OPT_BUFFERSIZE:
                        {
                            // -buffersize
                            interp.setResult(chan.BufferSize);
                            break;
                        }

                    case OPT_ENCODING:
                        {
                            // -encoding
                            System.Text.Encoding javaEncoding = chan.Encoding;
                            if ((System.Object)javaEncoding == null)
                            {
                                interp.setResult("binary");
                            }
                            else
                            {
                                interp.setResult(EncodingCmd.getTclName(javaEncoding));
                            }
                            break;
                        }

                    case OPT_EOFCHAR:
                        {
                            // -eofchar
                            if (chan.ReadOnly)
                            {
                                char eofChar = chan.InputEofChar;
                                interp.setResult((eofChar == 0) ? TclString.NewInstance("") : TclString.NewInstance(eofChar));
                            }
                            else if (chan.WriteOnly)
                            {
                                char eofChar = chan.OutputEofChar;
                                interp.setResult((eofChar == 0) ? TclString.NewInstance("") : TclString.NewInstance(eofChar));
                            }
                            else if (chan.ReadWrite)
                            {
                                char inEofChar = chan.InputEofChar;
                                char outEofChar = chan.OutputEofChar;

                                TclObject eofchar_pair = TclList.NewInstance();

                                TclList.Append(interp, eofchar_pair, (inEofChar == 0) ? TclString.NewInstance("") : TclString.NewInstance(inEofChar));

                                TclList.Append(interp, eofchar_pair, (outEofChar == 0) ? TclString.NewInstance("") : TclString.NewInstance(outEofChar));

                                interp.setResult(eofchar_pair);
                            }
                            else
                            {
                                // Not readable or writeable, do nothing
                            }

                            break;
                        }

                    case OPT_TRANSLATION:
                        {
                            // -translation
                            if (chan.ReadOnly)
                            {
                                interp.setResult(TclIO.getTranslationString(chan.InputTranslation));
                            }
                            else if (chan.WriteOnly)
                            {
                                interp.setResult(TclIO.getTranslationString(chan.OutputTranslation));
                            }
                            else if (chan.ReadWrite)
                            {
                                TclObject translation_pair = TclList.NewInstance();

                                TclList.Append(interp, translation_pair, TclString.NewInstance(TclIO.getTranslationString(chan.InputTranslation)));
                                TclList.Append(interp, translation_pair, TclString.NewInstance(TclIO.getTranslationString(chan.OutputTranslation)));

                                interp.setResult(translation_pair);
                            }
                            else
                            {
                                // Not readable or writeable, do nothing
                            }

                            break;
                        }

                    default:
                        {
                            throw new TclRuntimeError("Fconfigure.cmdProc() error: " + "incorrect index returned from TclIndex.get()");
                        }

                }
            }
            for (int i = 3; i < argv.Length; i += 2)
            {
                // Iterate through the list setting the name with the 
                // corresponding value.

                int index = TclIndex.get(interp, argv[i - 1], validCmds, "option", 0);

                switch (index)
                {

                    case OPT_BLOCKING:
                        {
                            // -blocking
                            chan.Blocking = TclBoolean.get(interp, argv[i]);
                            break;
                        }

                    case OPT_BUFFERING:
                        {
                            // -buffering

                            int id = TclIO.getBufferingID(argv[i].ToString());

                            if (id == -1)
                            {
                                throw new TclException(interp, "bad value for -buffering: must be " + "one of full, line, or none");
                            }

                            chan.Buffering = id;
                            break;
                        }

                    case OPT_BUFFERSIZE:
                        {
                            // -buffersize
                            chan.BufferSize = TclInteger.get(interp, argv[i]);
                            break;
                        }

                    case OPT_ENCODING:
                        {
                            // -encoding

                            string tclEncoding = argv[i].ToString();

                            if (tclEncoding.Equals("") || tclEncoding.Equals("binary"))
                            {
                                chan.Encoding = null;
                            }
                            else
                            {
                                System.Text.Encoding javaEncoding = EncodingCmd.getJavaName(tclEncoding);
                                if ((System.Object)javaEncoding == null)
                                {
                                    throw new TclException(interp, "unknown encoding \"" + tclEncoding + "\"");
                                }
                                chan.Encoding = javaEncoding;
                            }

                            break;
                        }

                    case OPT_EOFCHAR:
                        {
                            // -eofchar
                            TclList.setListFromAny(interp, argv[i]);
                            int length = TclList.getLength(interp, argv[i]);

                            if (length > 2)
                            {
                                throw new TclException(interp, "bad value for -eofchar: " + "should be a list of zero, one, or two elements");
                            }

                            char inputEofChar, outputEofChar;
                            string s;

                            if (length == 0)
                            {
                                inputEofChar = outputEofChar = (char)(0);
                            }
                            else if (length == 1)
                            {

                                s = TclList.index(interp, argv[i], 0).ToString();
                                inputEofChar = outputEofChar = s[0];
                            }
                            else
                            {

                                s = TclList.index(interp, argv[i], 0).ToString();
                                inputEofChar = s[0];


                                s = TclList.index(interp, argv[i], 1).ToString();
                                outputEofChar = s[0];
                            }

                            chan.InputEofChar = inputEofChar;
                            chan.OutputEofChar = outputEofChar;

                            break;
                        }

                    case OPT_TRANSLATION:
                        {
                            // -translation
                            TclList.setListFromAny(interp, argv[i]);
                            int length = TclList.getLength(interp, argv[i]);

                            if (length < 1 || length > 2)
                            {
                                throw new TclException(interp, "bad value for -translation: " + "must be a one or two element list");
                            }

                            string inputTranslationArg, outputTranslationArg;
                            int inputTranslation, outputTranslation;

                            if (length == 2)
                            {

                                inputTranslationArg = TclList.index(interp, argv[i], 0).ToString();
                                inputTranslation = TclIO.getTranslationID(inputTranslationArg);

                                outputTranslationArg = TclList.index(interp, argv[i], 1).ToString();
                                outputTranslation = TclIO.getTranslationID(outputTranslationArg);
                            }
                            else
                            {

                                outputTranslationArg = inputTranslationArg = argv[i].ToString();
                                outputTranslation = inputTranslation = TclIO.getTranslationID(outputTranslationArg);
                            }

                            if ((inputTranslation == -1) || (outputTranslation == -1))
                            {
                                throw new TclException(interp, "bad value for -translation: " + "must be one of auto, binary, cr, lf, " + "crlf, or platform");
                            }

                            if (outputTranslation == TclIO.TRANS_AUTO)
                                outputTranslation = TclIO.TRANS_PLATFORM;

                            if (chan.ReadOnly)
                            {
                                chan.InputTranslation = inputTranslation;
                                if (inputTranslationArg.Equals("binary"))
                                {
                                    chan.Encoding = null;
                                }
                            }
                            else if (chan.WriteOnly)
                            {
                                chan.OutputTranslation = outputTranslation;
                                if (outputTranslationArg.Equals("binary"))
                                {
                                    chan.Encoding = null;
                                }
                            }
                            else if (chan.ReadWrite)
                            {
                                chan.InputTranslation = inputTranslation;
                                chan.OutputTranslation = outputTranslation;
                                if (inputTranslationArg.Equals("binary") || outputTranslationArg.Equals("binary"))
                                {
                                    chan.Encoding = null;
                                }
                            }
                            else
                            {
                                // Not readable or writeable, do nothing
                            }

                            break;
                        }

                    default:
                        {
                            throw new TclRuntimeError("Fconfigure.cmdProc() error: " + "incorrect index returned from TclIndex.get()");
                        }

                }
            }
            return TCL.CompletionCode.RETURN;
        }
    }
}
