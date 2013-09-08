#region Foreign-License
/*
Copyright (c) 1997 Cornell University.
Copyright (c) 1997 Sun Microsystems, Inc.
Copyright (c) 2012 Sky Morey

See the file "license.terms" for information on usage and redistribution of this file, and for a DISCLAIMER OF ALL WARRANTIES.
*/
#endregion
using System;
using System.Collections;
namespace Tcl.Lang
{
    /// <summary>
    /// This class handles Tcl expressions.
    /// </summary>
    class Expression
    {
        // The token types are defined below.  In addition, there is a table associating a precedence with each operator.  The order
        // of types is important.  Consult the code before changing it.
        internal const int VALUE = 0;
        internal const int OPEN_PAREN = 1;
        internal const int CLOSE_PAREN = 2;
        internal const int COMMA = 3;
        internal const int END = 4;
        internal const int UNKNOWN = 5;
        // Binary operators:
        internal const int MULT = 8;
        internal const int DIVIDE = 9;
        internal const int MOD = 10;
        internal const int PLUS = 11;
        internal const int MINUS = 12;
        internal const int LEFT_SHIFT = 13;
        internal const int RIGHT_SHIFT = 14;
        internal const int LESS = 15;
        internal const int GREATER = 16;
        internal const int LEQ = 17;
        internal const int GEQ = 18;
        internal const int EQUAL = 19;
        internal const int NEQ = 20;
        internal const int BIT_AND = 21;
        internal const int BIT_XOR = 22;
        internal const int BIT_OR = 23;
        internal const int AND = 24;
        internal const int OR = 25;
        internal const int QUESTY = 26;
        internal const int COLON = 27;
        // Unary operators:
        internal const int UNARY_MINUS = 28;
        internal const int UNARY_PLUS = 29;
        internal const int NOT = 30;
        internal const int BIT_NOT = 31;
        internal const int EQ = 32;
        internal const int NE = 33;
        // Precedence table.  The values for non-operator token types are ignored.
        internal static int[] _precTable = new int[] { 0, 0, 0, 0, 0, 0, 0, 0, 12, 12, 12, 11, 11, 10, 10, 9, 9, 9, 9, 8, 8, 7, 6, 5, 4, 3, 2, 1, 13, 13, 13, 13 };
        // Mapping from operator numbers to strings;  used for error messages.
        internal static string[] _operatorStrings = new string[] { "VALUE", "(", ")", ",", "END", "UNKNOWN", "6", "7", "*", "/", "%", "+", "-", "<<", ">>", "<", ">", "<=", ">=", "==", "!=", "&", "^", "|", "&&", "||", "?", ":", "-", "+", "!", "~", "eq", "ne" };
        internal Hashtable _mathFuncTable;
        private string _expr; // The entire expression, as originally passed to eval et al.
        private int _len; // Length of the expression.
        internal int _token; // Type of the last token to be parsed from the expression. Corresponds to the characters just before expr.
        private int _ind; // Position to the next character to be scanned from the expression string.

        /// <summary>
        /// Evaluate a Tcl expression.
        /// </summary>
        /// <param name="interp">
        /// the context in which to evaluate the expression.
        /// </param>
        /// <param name="string">
        /// expression to evaluate.
        /// </param>
        /// <returns>
        /// the value of the expression.
        /// </returns>
        /// <exception cref="">
        /// TclException for malformed expressions.
        /// </exception>
        internal TclObject eval(Interp interp, string inString)
        {
            ExprValue value = ExprTopLevel(interp, inString);
            switch (value.Type)
            {
                case ExprValue.INT: return TclInteger.NewInstance((int)value.IntValue);
                case ExprValue.DOUBLE: return TclDouble.NewInstance(value.DoubleValue);
                case ExprValue.STRING: return TclString.NewInstance(value.StringValue);
                default: throw new TclRuntimeError("internal error: expression, unknown");
            }
        }

        /// <summary>
        /// Evaluate an Tcl expression.
        /// </summary>
        /// <param name="interp">
        /// the context in which to evaluate the expression.
        /// </param>
        /// <param name="string">
        /// expression to evaluate.
        /// </param>
        /// <exception cref="">
        /// TclException for malformed expressions.
        /// </exception>
        /// <returns>
        /// the value of the expression in boolean.
        /// </returns>
        internal bool EvalBoolean(Interp interp, string inString)
        {
            ExprValue value = ExprTopLevel(interp, inString);
            switch (value.Type)
            {
                case ExprValue.INT: return (value.IntValue != 0);
                case ExprValue.DOUBLE: return (value.DoubleValue != 0.0);
                case ExprValue.STRING: return Util.GetBoolean(interp, value.StringValue);
                default: throw new TclRuntimeError("internal error: expression, unknown");
            }
        }

        /// <summary>
        /// Constructor.
        /// </summary>
        internal Expression()
        {
            _mathFuncTable = new Hashtable();
            // rand  -- needs testing
            // srand -- needs testing
            // hypot -- needs testing
            // fmod  -- needs testing
            // try [expr fmod(4.67, 2.2)]
            // the answer should be .27, but I got .2699999999999996
            SupportClass.PutElement(_mathFuncTable, "atan2", new Atan2Function());
            SupportClass.PutElement(_mathFuncTable, "pow", new PowFunction());
            SupportClass.PutElement(_mathFuncTable, "acos", new AcosFunction());
            SupportClass.PutElement(_mathFuncTable, "asin", new AsinFunction());
            SupportClass.PutElement(_mathFuncTable, "atan", new AtanFunction());
            SupportClass.PutElement(_mathFuncTable, "ceil", new CeilFunction());
            SupportClass.PutElement(_mathFuncTable, "cos", new CosFunction());
            SupportClass.PutElement(_mathFuncTable, "cosh", new CoshFunction());
            SupportClass.PutElement(_mathFuncTable, "exp", new ExpFunction());
            SupportClass.PutElement(_mathFuncTable, "floor", new FloorFunction());
            SupportClass.PutElement(_mathFuncTable, "fmod", new FmodFunction());
            SupportClass.PutElement(_mathFuncTable, "hypot", new HypotFunction());
            SupportClass.PutElement(_mathFuncTable, "log", new LogFunction());
            SupportClass.PutElement(_mathFuncTable, "log10", new Log10Function());
            SupportClass.PutElement(_mathFuncTable, "rand", new RandFunction());
            SupportClass.PutElement(_mathFuncTable, "sin", new SinFunction());
            SupportClass.PutElement(_mathFuncTable, "sinh", new SinhFunction());
            SupportClass.PutElement(_mathFuncTable, "sqrt", new SqrtFunction());
            SupportClass.PutElement(_mathFuncTable, "srand", new SrandFunction());
            SupportClass.PutElement(_mathFuncTable, "tan", new TanFunction());
            SupportClass.PutElement(_mathFuncTable, "tanh", new TanhFunction());
            SupportClass.PutElement(_mathFuncTable, "abs", new AbsFunction());
            SupportClass.PutElement(_mathFuncTable, "double", new DoubleFunction());
            SupportClass.PutElement(_mathFuncTable, "int", new IntFunction());
            SupportClass.PutElement(_mathFuncTable, "round", new RoundFunction());
            SupportClass.PutElement(_mathFuncTable, "wide", new WideFunction());
            _expr = null;
            _ind = 0;
            _len = 0;
            _token = UNKNOWN;
        }

        /// <summary>
        /// Provides top-level functionality shared by procedures like ExprInt, ExprDouble, etc.
        /// </summary>
        /// <param name="interp">
        /// the context in which to evaluate the expression.
        /// </param>
        /// <param name="string">
        /// the expression.
        /// </param>
        /// <exception cref="">
        /// TclException for malformed expressions.
        /// </exception>
        /// <returns>
        /// the value of the expression.
        /// </returns>
        private ExprValue ExprTopLevel(Interp interp, string inString)
        {
            // Saved the state variables so that recursive calls to expr can work:
            // expr {[expr 1+2] + 3}
            string m_expr_saved = _expr;
            int _len_saved = _len;
            int _token_saved = _token;
            int _ind_saved = _ind;
            try
            {
                _expr = inString;
                _ind = 0;
                _len = inString.Length;
                _token = UNKNOWN;
                ExprValue val = ExprGetValue(interp, -1);
                if (_token != END)
                    SyntaxError(interp);
                return val;
            }
            finally
            {
                _expr = m_expr_saved;
                _len = _len_saved;
                _token = _token_saved;
                _ind = _ind_saved;
            }
        }

        internal static void IllegalType(Interp interp, int badType, int Operator)
        {
            throw new TclException(interp, "can't use " + (badType == ExprValue.DOUBLE ? "floating-point value" : "non-numeric string") + " as operand of \"" + _operatorStrings[Operator] + "\"");
        }

        internal void SyntaxError(Interp interp)
        {
            throw new TclException(interp, "syntax error in expression \"" + _expr + "\"");
        }

        internal static void DivideByZero(Interp interp)
        {
            interp.SetErrorCode(TclString.NewInstance("ARITH DIVZERO {divide by zero}"));
            throw new TclException(interp, "divide by zero");
        }

        internal static void IntegerTooLarge(Interp interp)
        {
            interp.SetErrorCode(TclString.NewInstance("ARITH IOVERFLOW {integer value too large to represent}"));
            throw new TclException(interp, "integer value too large to represent");
        }

        internal static void WideTooLarge(Interp interp)
        {
            interp.SetErrorCode(TclString.NewInstance("ARITH IOVERFLOW {wide value too large to represent}"));
            throw new TclException(interp, "wide value too large to represent");
        }

        internal static void DoubleTooLarge(Interp interp)
        {
            interp.SetErrorCode(TclString.NewInstance("ARITH OVERFLOW {floating-point value too large to represent}"));
            throw new TclException(interp, "floating-point value too large to represent");
        }

        internal static void DoubleTooSmall(Interp interp)
        {
            interp.SetErrorCode(TclString.NewInstance("ARITH UNDERFLOW {floating-point value too small to represent}"));
            throw new TclException(interp, "floating-point value too small to represent");
        }

        internal static void DomainError(Interp interp)
        {
            interp.SetErrorCode(TclString.NewInstance("ARITH DOMAIN {domain error: argument not in valid range}"));
            throw new TclException(interp, "domain error: argument not in valid range");
        }

        /// <summary>
        /// Given a string (such as one coming from command or variable substitution), make a Value based on the string.  The value
        /// be a floating-point or integer, if possible, or else it just be a copy of the string.
        /// </summary>
        /// <param name="interp">
        /// the context in which to evaluate the expression.
        /// </param>
        /// <param name="s">
        /// the string to parse.
        /// </param>
        /// <exception cref="">
        /// TclException for malformed expressions.
        /// </exception>
        /// <returns>
        /// the value of the expression.
        /// </returns>
        private ExprValue ExprParseString(Interp interp, string s)
        {
            int len = s.Length;
            /*System.out.println("now to ExprParseString ->" + s + "<- of length " + len);*/
            // Take shortcut when string is of length 0, as there is only a string rep for an empty string (no int or double rep)
            // this will happend a lot so this shortcut will speed things up!
            if (len == 0)
                return new ExprValue(s);
            // The strings "0" and "1" are going to occure a lot it might be wise to include shortcuts for these cases
            int i;
            if (LooksLikeInt(s, len, 0))
            {
                //System.out.println("string looks like an int");
                // Note: use strtoul instead of strtol for integer conversions to allow full-size unsigned numbers, but don't depend on
                // strtoul to handle sign characters;  it won't in some implementations.
                for (i = 0; char.IsWhiteSpace(s[i]); i++) { } // Empty loop body.
                StrtoulResult res;
                if (s[i] == '-')
                {
                    i++;
                    res = Util.Strtoul(s, i, 0);
                    res.value = -res.value;
                }
                else if (s[i] == '+')
                {
                    i++;
                    res = Util.Strtoul(s, i, 0);
                }
                else
                    res = Util.Strtoul(s, i, 0);
                if (res.errno == 0)
                {
                    // We treat this string as a number if all the charcters following the parsed number are a whitespace char
                    // E.g.: " 1", "1", "1 ", and " 1 " are all good numbers
                    bool trailing_blanks = true;
                    for (i = res.Index; i < len; i++)
                        if (!char.IsWhiteSpace(s[i]))
                            trailing_blanks = false;
                    if (trailing_blanks)
                    {
                        //System.out.println("string is an Integer of value " + res.value);
                        _token = VALUE;
                        return new ExprValue(res.value);
                    }
                }
                else if (res.errno == TCL.INTEGER_RANGE)
                    IntegerTooLarge(interp);
                /*
                if (res.index == len) {
                // We treat this string as a number only if the number ends at the end of the string. E.g.: " 1", "1" are
                // good numbers but "1 " is not.
                if (res.errno == TCL.INTEGER_RANGE) {
                IntegerTooLarge(interp);
                } else {
                m_token = VALUE;
                return new ExprValue(res.value);
                }
                }
                */
            }
            else
            {
                //System.out.println("string does not look like an int, checking for Double");
                StrtodResult res = Util.Strtod(s, 0);
                if (res.errno == 0)
                {
                    // Trailing whitespaces are treated just like the Integer case
                    bool trailing_blanks = true;
                    for (i = res.index; i < len; i++)
                        if (!char.IsWhiteSpace(s[i]))
                            trailing_blanks = false;
                    if (trailing_blanks)
                    {
                        //System.out.println("string is a Double of value " + res.value);
                        _token = VALUE;
                        return new ExprValue(res.value);
                    }
                }
                else if (res.errno == TCL.DOUBLE_RANGE)
                {
                    if (res.value != 0)
                        DoubleTooLarge(interp);
                    else
                        DoubleTooSmall(interp);
                }
                // if res.errno is any other value (like TCL.INVALID_DOUBLE) just fall through and use the string rep
                /*
                if (res.index == len) {
                if (res.errno == 0) {
                //System.out.println("string is a Double of value " + res.value);
                m_token = VALUE;
                return new ExprValue(res.value);
                } else if (res.errno == TCL.DOUBLE_RANGE) {
                DoubleTooLarge(interp);
                }
                }
                */
            }
            //System.out.println("string is not a valid number, returning as string");
            // Not a valid number.  Save a string value (but don't do anything if it's already the value).
            return new ExprValue(s);
        }

        /// <summary>
        /// Parse a "value" from the remainder of the expression.
        /// </summary>
        /// <param name="interp">
        /// the context in which to evaluate the expression.
        /// </param>
        /// <param name="prec">
        /// treat any un-parenthesized operator with precedence &lt;= this as the end of the expression.
        /// </param>
        /// <exception cref="">
        /// TclException for malformed expressions.
        /// </exception>
        /// <returns>
        /// the value of the expression.
        /// </returns>
        private ExprValue ExprGetValue(Interp interp, int prec)
        {
            int Operator;
            bool gotOp = false; // True means already lexed the
            // operator (while picking up value for unary operator).  Don't lex again.
            ExprValue value, value2;
            // There are two phases to this procedure.  First, pick off an initial value.  Then, parse (binary operator, value) pairs until done.
            value = ExprLex(interp);
            if (_token == OPEN_PAREN)
            {
                // Parenthesized sub-expression.
                value = ExprGetValue(interp, -1);
                if (_token != CLOSE_PAREN)
                    SyntaxError(interp);
            }
            else
            {
                if (_token == MINUS)
                    _token = UNARY_MINUS;
                if (_token == PLUS)
                    _token = UNARY_PLUS;
                if (_token >= UNARY_MINUS)
                {
                    // Process unary operators.
                    Operator = _token;
                    value = ExprGetValue(interp, _precTable[_token]);
                    if (interp.NoEval == 0)
                    {
                        switch (Operator)
                        {
                            case UNARY_MINUS:
                                if (value.Type == ExprValue.INT)
                                    value.IntValue = -value.IntValue;
                                else if (value.Type == ExprValue.DOUBLE)
                                    value.DoubleValue = -value.DoubleValue;
                                else
                                    IllegalType(interp, value.Type, Operator);
                                break;
                            case UNARY_PLUS:
                                if (value.Type != ExprValue.INT && value.Type != ExprValue.DOUBLE)
                                    IllegalType(interp, value.Type, Operator);
                                break;
                            case NOT:
                                if (value.Type == ExprValue.INT)
                                    value.IntValue = (value.IntValue != 0 ? 0 : 1);
                                else if (value.Type == ExprValue.DOUBLE)
                                {
                                    value.IntValue = (value.DoubleValue == 0.0 ? 1 : 0);
                                    value.Type = ExprValue.INT;
                                }
                                else
                                    IllegalType(interp, value.Type, Operator);
                                break;
                            case BIT_NOT:
                                if (value.Type == ExprValue.INT)
                                    value.IntValue = ~value.IntValue;
                                else
                                    IllegalType(interp, value.Type, Operator);
                                break;
                        }
                    }
                    gotOp = true;
                }
                else if (_token == CLOSE_PAREN) // Caller needs to deal with close paren token.
                    return null;
                else if (_token != VALUE)
                    SyntaxError(interp);
            }
            if (value == null)
                SyntaxError(interp);
            // Got the first operand.  Now fetch (operator, operand) pairs.
            if (!gotOp)
                value2 = ExprLex(interp);
            while (true)
            {
                Operator = _token;
                if (Operator < MULT || Operator >= UNARY_MINUS)
                {
                    if (Operator == END || Operator == CLOSE_PAREN || Operator == COMMA)
                        return value; // Goto Done
                    else
                        SyntaxError(interp);
                }
                if (_precTable[Operator] <= prec)
                    return value; // (goto done)

                // If we're doing an AND or OR and the first operand already determines the result, don't execute anything in the
                // second operand:  just parse.  Same style for ?: pairs.
                if (Operator == AND || Operator == OR || Operator == QUESTY)
                {
                    if (value.Type == ExprValue.DOUBLE)
                    {
                        value.IntValue = (value.DoubleValue != 0) ? 1 : 0;
                        value.Type = ExprValue.INT;
                    }
                    else if (value.Type == ExprValue.STRING)
                    {
                        try
                        {
                            bool b = Util.GetBoolean(null, value.StringValue);
                            value = new ExprValue(b ? 1 : 0);
                        }
                        catch (TclException)
                        {
                            if (interp.NoEval == 0)
                                IllegalType(interp, ExprValue.STRING, Operator);
                            // Must set value.intValue to avoid referencing uninitialized memory in the "if" below;  the actual
                            // value doesn't matter, since it will be ignored.
                            value.IntValue = 0;
                        }
                    }
                    if ((Operator == AND && value.IntValue == 0) || (Operator == OR && value.IntValue != 0))
                    {
                        interp.NoEval++;
                        try { value2 = ExprGetValue(interp, _precTable[Operator]); }
                        finally { interp.NoEval--; }
                        if (Operator == OR)
                            value.IntValue = 1;
                        continue;
                    }
                    else if (Operator == QUESTY)
                    {
                        // Special note:  ?: operators must associate right to left.  To make this happen, use a precedence one lower
                        // than QUESTY when calling ExprGetValue recursively.
                        if (value.IntValue != 0)
                        {
                            value = ExprGetValue(interp, _precTable[QUESTY] - 1);
                            if (_token != COLON)
                                SyntaxError(interp);
                            interp.NoEval++;
                            try { value2 = ExprGetValue(interp, _precTable[QUESTY] - 1); }
                            finally { interp.NoEval--; }
                        }
                        else
                        {
                            interp.NoEval++;
                            try { value2 = ExprGetValue(interp, _precTable[QUESTY] - 1); }
                            finally { interp.NoEval--; }
                            if (_token != COLON)
                                SyntaxError(interp);
                            value = ExprGetValue(interp, _precTable[QUESTY] - 1);
                        }
                        continue;
                    }
                    else
                        value2 = ExprGetValue(interp, _precTable[Operator]);
                }
                else
                    value2 = ExprGetValue(interp, _precTable[Operator]);
                if (_token < MULT && _token != VALUE && _token != END && _token != COMMA && _token != CLOSE_PAREN)
                    SyntaxError(interp);
                if (interp.NoEval != 0)
                    continue;
                // At this point we've got two values and an operator.  Check to make sure that the particular data types are appropriate
                // for the particular operator, and perform type conversion if necessary.
                switch (Operator)
                {
                    // For the operators below, no strings are allowed and ints get converted to floats if necessary.
                    case MULT:
                    case DIVIDE:
                    case PLUS:
                    case MINUS:
                        if (value.Type == ExprValue.STRING || value2.Type == ExprValue.STRING)
                            IllegalType(interp, ExprValue.STRING, Operator);
                        if (value.Type == ExprValue.DOUBLE)
                        {
                            if (value2.Type == ExprValue.INT)
                            {
                                value2.DoubleValue = value2.IntValue;
                                value2.Type = ExprValue.DOUBLE;
                            }
                        }
                        else if (value2.Type == ExprValue.DOUBLE)
                        {
                            if (value.Type == ExprValue.INT)
                            {
                                value.DoubleValue = value.IntValue;
                                value.Type = ExprValue.DOUBLE;
                            }
                        }
                        break;
                    // For the operators below, only integers are allowed.
                    case MOD:
                    case LEFT_SHIFT:
                    case RIGHT_SHIFT:
                    case BIT_AND:
                    case BIT_XOR:
                    case BIT_OR:
                        if (value.Type != ExprValue.INT)
                            IllegalType(interp, value.Type, Operator);
                        else if (value2.Type != ExprValue.INT)
                            IllegalType(interp, value2.Type, Operator);
                        break;
                    // For the operators below, any type is allowed but the two operands must have the same type.  Convert integers
                    // to floats and either to strings, if necessary.
                    case LESS:
                    case GREATER:
                    case LEQ:
                    case GEQ:
                    case EQUAL:
                    case EQ:
                    case NEQ:
                    case NE:
                        if (value.Type == ExprValue.STRING)
                        {
                            if (value2.Type != ExprValue.STRING)
                                ExprMakeString(interp, value2);
                        }
                        else if (value2.Type == ExprValue.STRING)
                        {
                            if (value.Type != ExprValue.STRING)
                                ExprMakeString(interp, value);
                        }
                        else if (value.Type == ExprValue.DOUBLE)
                        {
                            if (value2.Type == ExprValue.INT)
                            {
                                value2.DoubleValue = value2.IntValue;
                                value2.Type = ExprValue.DOUBLE;
                            }
                        }
                        else if (value2.Type == ExprValue.DOUBLE)
                        {
                            if (value.Type == ExprValue.INT)
                            {
                                value.DoubleValue = value.IntValue;
                                value.Type = ExprValue.DOUBLE;
                            }
                        }
                        break;

                    // For the operators below, no strings are allowed, but no int->double conversions are performed.
                    case AND:
                    case OR:
                        if (value.Type == ExprValue.STRING)
                            IllegalType(interp, value.Type, Operator);
                        if (value2.Type == ExprValue.STRING)
                            try
                            {
                                bool b = Util.GetBoolean(null, value2.StringValue);
                                value2 = new ExprValue(b ? 1 : 0);
                            }
                            catch (TclException) { IllegalType(interp, value2.Type, Operator); }
                        break;
                    // For the operators below, type and conversions are irrelevant:  they're handled elsewhere.
                    case QUESTY:
                    case COLON:
                        break;
                    // Any other operator is an error.
                    default:
                        throw new TclException(interp, "unknown operator in expression");
                }
                // Carry out the function of the specified operator.
                switch (Operator)
                {
                    case MULT:
                        if (value.Type == ExprValue.INT)
                            value.IntValue = value.IntValue * value2.IntValue;
                        else
                            value.DoubleValue *= value2.DoubleValue;
                        break;
                    case DIVIDE:
                    case MOD:
                        if (value.Type == ExprValue.INT)
                        {
                            if (value2.IntValue == 0)
                                DivideByZero(interp);
                            // The code below is tricky because C doesn't guarantee much about the properties of the quotient or
                            // remainder, but Tcl does:  the remainder always has the same sign as the divisor and a smaller absolute value.
                            long divisor = value2.IntValue;
                            bool negative = false;
                            if (divisor < 0)
                            {
                                divisor = -divisor;
                                value.IntValue = -value.IntValue;
                                negative = true;
                            }
                            long quot = value.IntValue / divisor;
                            long rem = value.IntValue % divisor;
                            if (rem < 0)
                            {
                                rem += divisor;
                                quot -= 1;
                            }
                            if (negative)
                                rem = -rem;
                            value.IntValue = (Operator == DIVIDE ? quot : rem);
                        }
                        else
                        {
                            if (value2.DoubleValue == 0.0)
                                DivideByZero(interp);
                            value.DoubleValue /= value2.DoubleValue;
                        }
                        break;
                    case PLUS:
                        if (value.Type == ExprValue.INT)
                            value.IntValue = value.IntValue + value2.IntValue;
                        else
                            value.DoubleValue += value2.DoubleValue;
                        break;
                    case MINUS:
                        if (value.Type == ExprValue.INT)
                            value.IntValue = value.IntValue - value2.IntValue;
                        else
                            value.DoubleValue -= value2.DoubleValue;
                        break;
                    case LEFT_SHIFT:
                        value.IntValue <<= (int)value2.IntValue;
                        break;
                    case RIGHT_SHIFT:
                        if (value.IntValue < 0)
                            value.IntValue = ~((~value.IntValue) >> (int)value2.IntValue);
                        else
                            value.IntValue >>= (int)value2.IntValue;
                        break;
                    case LESS:
                        if (value.Type == ExprValue.INT)
                            value.IntValue = (value.IntValue < value2.IntValue ? 1 : 0);
                        else if (value.Type == ExprValue.DOUBLE)
                            value.IntValue = (value.DoubleValue < value2.DoubleValue ? 1 : 0);
                        else
                            value.IntValue = (value.StringValue.CompareTo(value2.StringValue) < 0 ? 1 : 0);
                        value.Type = ExprValue.INT;
                        break;
                    case GREATER:
                        if (value.Type == ExprValue.INT)
                            value.IntValue = (value.IntValue > value2.IntValue ? 1 : 0);
                        else if (value.Type == ExprValue.DOUBLE)
                            value.IntValue = (value.DoubleValue > value2.DoubleValue ? 1 : 0);
                        else
                            value.IntValue = (value.StringValue.CompareTo(value2.StringValue) > 0 ? 1 : 0);
                        value.Type = ExprValue.INT;
                        break;

                    case LEQ:
                        if (value.Type == ExprValue.INT)
                            value.IntValue = (value.IntValue <= value2.IntValue ? 1 : 0);
                        else if (value.Type == ExprValue.DOUBLE)
                            value.IntValue = (value.DoubleValue <= value2.DoubleValue ? 1 : 0);
                        else
                            value.IntValue = (value.StringValue.CompareTo(value2.StringValue) <= 0 ? 1 : 0);
                        value.Type = ExprValue.INT;
                        break;
                    case GEQ:
                        if (value.Type == ExprValue.INT)
                            value.IntValue = (value.IntValue >= value2.IntValue ? 1 : 0);
                        else if (value.Type == ExprValue.DOUBLE)
                            value.IntValue = (value.DoubleValue >= value2.DoubleValue ? 1 : 0);
                        else
                            value.IntValue = (value.StringValue.CompareTo(value2.StringValue) >= 0 ? 1 : 0);
                        value.Type = ExprValue.INT;
                        break;
                    case EQUAL:
                    case EQ:
                        if (value.Type == ExprValue.INT)
                            value.IntValue = (value.IntValue == value2.IntValue ? 1 : 0);
                        else if (value.Type == ExprValue.DOUBLE)
                            value.IntValue = (value.DoubleValue == value2.DoubleValue ? 1 : 0);
                        else
                            value.IntValue = (value.StringValue.CompareTo(value2.StringValue) == 0 ? 1 : 0);
                        value.Type = ExprValue.INT;
                        break;
                    case NEQ:
                    case NE:
                        if (value.Type == ExprValue.INT)
                            value.IntValue = (value.IntValue != value2.IntValue ? 1 : 0);
                        else if (value.Type == ExprValue.DOUBLE)
                            value.IntValue = (value.DoubleValue != value2.DoubleValue ? 1 : 0);
                        else
                            value.IntValue = (value.StringValue.CompareTo(value2.StringValue) != 0 ? 1 : 0);
                        value.Type = ExprValue.INT;
                        break;
                    case BIT_AND:
                        value.IntValue &= value2.IntValue;
                        break;
                    case BIT_XOR:
                        value.IntValue ^= value2.IntValue;
                        break;
                    case BIT_OR:
                        value.IntValue |= value2.IntValue;
                        break;
                    // For AND and OR, we know that the first value has already been converted to an integer.  Thus we need only consider
                    // the possibility of int vs. double for the second value.
                    case AND:
                        if (value2.Type == ExprValue.DOUBLE)
                        {
                            value2.IntValue = (value2.DoubleValue != 0 ? 1 : 0);
                            value2.Type = ExprValue.INT;
                        }
                        value.IntValue = (value.IntValue != 0 && value2.IntValue != 0 ? 1 : 0);
                        break;
                    case OR:
                        if (value2.Type == ExprValue.DOUBLE)
                        {
                            value2.IntValue = (value2.DoubleValue != 0 ? 1 : 0);
                            value2.Type = ExprValue.INT;
                        }
                        value.IntValue = (value.IntValue != 0 || value2.IntValue != 0 ? 1 : 0);
                        break;
                    case COLON:
                        SyntaxError(interp);
                        break;
                }
            }
        }

        /// <summary>
        /// GetLexeme -> ExprLex
        /// Lexical analyzer for expression parser:  parses a single value, operator, or other syntactic element from an expression string.
        /// Size effects: the "m_token" member variable is set to the value of the current token.
        /// </summary>
        /// <param name="interp">
        /// the context in which to evaluate the expression.
        /// </param>
        /// <exception cref="">
        /// TclException for malformed expressions.
        /// </exception>
        /// <returns>
        /// the value of the expression.
        /// </returns>
        private ExprValue ExprLex(Interp interp)
        {
            while (_ind < _len && char.IsWhiteSpace(_expr[_ind]))
                _ind++;
            if (_ind >= _len)
            {
                _token = END;
                return null;
            }
            // First try to parse the token as an integer or floating-point number.  Don't want to check for a number if
            // the first character is "+" or "-".  If we do, we might treat a binary operator as unary by
            // mistake, which will eventually cause a syntax error.
            char c2;
            char c = _expr[_ind];
            if (_ind < _len - 1)
                c2 = _expr[_ind + 1];
            else
                c2 = '\x0000';
            if (c != '+' && c != '-')
            {
                bool startsWithDigit = char.IsDigit(c);
                if (startsWithDigit && LooksLikeInt(_expr, _len, _ind))
                {
                    StrtoulResult res = Util.Strtoul(_expr, _ind, 0);
                    if (res.errno == 0)
                    {
                        _ind = res.Index;
                        _token = VALUE;
                        return new ExprValue(res.value);
                    }
                    else if (res.errno == TCL.INTEGER_RANGE)
                        IntegerTooLarge(interp);
                }
                else if (startsWithDigit || c == '.' || c == 'n' || c == 'N')
                {
                    StrtodResult res = Util.Strtod(_expr, _ind);
                    if (res.errno == 0)
                    {
                        _ind = res.index;
                        _token = VALUE;
                        return new ExprValue(res.value);
                    }
                    else if (res.errno == TCL.DOUBLE_RANGE)
                    {
                        if (res.value != 0)
                            DoubleTooLarge(interp);
                        else
                            DoubleTooSmall(interp);
                    }
                }
            }
            ParseResult pres;
            ExprValue retval;
            _ind += 1; // ind is advanced to point to the next token
            switch (c)
            {
                case '$':
                    _token = VALUE;
                    pres = ParseAdaptor.ParseVar(interp, _expr, _ind, _len);
                    _ind = pres.nextIndex;
                    retval = (interp.NoEval != 0 ? new ExprValue(0) : ExprParseString(interp, pres.Value.ToString()));
                    pres.Release();
                    return retval;
                case '[':
                    _token = VALUE;
                    pres = ParseAdaptor.parseNestedCmd(interp, _expr, _ind, _len);
                    _ind = pres.nextIndex;
                    retval = (interp.NoEval != 0 ? new ExprValue(0) : ExprParseString(interp, pres.Value.ToString()));
                    pres.Release();
                    return retval;
                case '"':
                    _token = VALUE;
                    //System.out.println("now to parse from ->" + m_expr + "<- at index " + m_ind);
                    pres = ParseAdaptor.ParseQuotes(interp, _expr, _ind, _len);
                    _ind = pres.nextIndex;
                    //   System.out.println("after parse next index is " + m_ind);
                    //     System.out.println("returning value string ->" + pres.value.toString() + "<-" );
                    retval = (interp.NoEval != 0 ? new ExprValue(0) : ExprParseString(interp, pres.Value.ToString()));
                    pres.Release();
                    return retval;
                case '{':
                    _token = VALUE;
                    pres = ParseAdaptor.parseBraces(interp, _expr, _ind, _len);
                    _ind = pres.nextIndex;
                    retval = (interp.NoEval != 0 ? new ExprValue(0) : ExprParseString(interp, pres.Value.ToString()));
                    pres.Release();
                    return retval;
                case '(':
                    _token = OPEN_PAREN;
                    return null;
                case ')':
                    _token = CLOSE_PAREN;
                    return null;
                case ',':
                    _token = COMMA;
                    return null;
                case '*':
                    _token = MULT;
                    return null;
                case '/':
                    _token = DIVIDE;
                    return null;
                case '%':
                    _token = MOD;
                    return null;
                case '+':
                    _token = PLUS;
                    return null;
                case '-':
                    _token = MINUS;
                    return null;
                case '?':
                    _token = QUESTY;
                    return null;
                case ':':
                    _token = COLON;
                    return null;
                case '<':
                    switch (c2)
                    {
                        case '<':
                            _ind += 1;
                            _token = LEFT_SHIFT;
                            break;
                        case '=':
                            _ind += 1;
                            _token = LEQ;
                            break;
                        default:
                            _token = LESS;
                            break;
                    }
                    return null;
                case '>':
                    switch (c2)
                    {
                        case '>':
                            _ind += 1;
                            _token = RIGHT_SHIFT;
                            break;
                        case '=':
                            _ind += 1;
                            _token = GEQ;
                            break;
                        default:
                            _token = GREATER;
                            break;
                    }
                    return null;
                case '=':
                    if (c2 == '=')
                    {
                        _ind += 1;
                        _token = EQUAL;
                    }
                    else
                        _token = UNKNOWN;
                    return null;
                case 'e':
                    if (c2 == 'q')
                    {
                        _ind += 1;
                        _token = EQUAL;
                    }
                    else
                        _token = UNKNOWN;
                    return null;
                case 'n':
                    if (c2 == 'e')
                    {
                        _ind += 1;
                        _token = NEQ;
                    }
                    else
                        _token = UNKNOWN;
                    return null;
                case '!':
                    if (c2 == '=')
                    {
                        _ind += 1;
                        _token = NEQ;
                    }
                    else
                        _token = NOT;
                    return null;
                case '&':
                    if (c2 == '&')
                    {
                        _ind += 1;
                        _token = AND;
                    }
                    else
                        _token = BIT_AND;
                    return null;
                case '^':
                    _token = BIT_XOR;
                    return null;
                case '|':
                    if (c2 == '|')
                    {
                        _ind += 1;
                        _token = OR;
                    }
                    else
                        _token = BIT_OR;
                    return null;
                case '~':
                    _token = BIT_NOT;
                    return null;
                default:
                    if (char.IsLetter(c))
                    {
                        // Oops, re-adjust m_ind so that it points to the beginning of the function name.
                        _ind--;
                        return MathFunction(interp);
                    }
                    _token = UNKNOWN;
                    return null;
            }
        }

        /// <summary>
        /// Parses a math function from an expression string, carry out the function, and return the value computed.
        /// </summary>
        /// <param name="interp">
        /// current interpreter.
        /// </param>
        /// <returns>
        /// the value computed by the math function.
        /// </returns>
        /// <exception cref="">
        /// TclException if any error happens.
        /// </exception>
        internal ExprValue MathFunction(Interp interp)
        {
            int startIdx = _ind;
            // Find the end of the math function's name and lookup the MathFunc record for the function.  Search until the char at m_ind is not
            // alphanumeric or '_'
            for (; _ind < _len; _ind++)
                if (!(char.IsLetterOrDigit(_expr[_ind]) || _expr[_ind] == '_'))
                    break;
            // Get the funcName BEFORE calling ExprLex, so the funcName will not have trailing whitespace.
            string funcName = _expr.Substring(startIdx, (_ind) - (startIdx));
            // Parse errors are thrown BEFORE unknown function names
            ExprLex(interp);
            if (_token != OPEN_PAREN)
                SyntaxError(interp);
            // Now test for unknown funcName.  Doing the above statements out of order will cause some tests to fail.
            MathFunction mathFunc = (MathFunction)_mathFuncTable[funcName];
            if (mathFunc == null)
                throw new TclException(interp, "unknown math function \"" + funcName + "\"");
            // Scan off the arguments for the function, if there are any.
            int numArgs = mathFunc.ArgTypes.Length;
            TclObject[] argv = null;
            if (numArgs == 0)
            {
                ExprLex(interp);
                if (_token != CLOSE_PAREN)
                    SyntaxError(interp);
            }
            else
            {
                argv = new TclObject[numArgs];
                for (int i = 0; ; i++)
                {
                    ExprValue value = ExprGetValue(interp, -1);
                    // Handle close paren with no value % expr {srand()}
                    if (value == null && _token == CLOSE_PAREN)
                    {
                        if (i == numArgs)
                            break;
                        else
                            throw new TclException(interp, "too few arguments for math function");
                    }
                    if (value.Type == ExprValue.STRING)
                        throw new TclException(interp, "argument to math function didn't have numeric value");
                    // Copy the value to the argument record, converting it if necessary.
                    if (value.Type == ExprValue.INT)
                        argv[i] = (mathFunc.ArgTypes[i] == Lang.MathFunction.DOUBLE ? TclDouble.NewInstance((int)value.IntValue) : TclLong.newInstance(value.IntValue));
                    else
                        argv[i] = (mathFunc.ArgTypes[i] == Lang.MathFunction.INT ? TclInteger.NewInstance((int)value.DoubleValue) : TclDouble.NewInstance(value.DoubleValue));
                    // Check for a comma separator between arguments or a close-paren to end the argument list.
                    if (i == (numArgs - 1))
                    {
                        if (_token == CLOSE_PAREN)
                            break;
                        if (_token == COMMA)
                            throw new TclException(interp, "too many arguments for math function");
                        else
                            SyntaxError(interp);
                    }
                    if (_token != COMMA)
                    {
                        if (_token == CLOSE_PAREN)
                            throw new TclException(interp, "too few arguments for math function");
                        else
                            SyntaxError(interp);
                    }
                }
            }
            _token = VALUE;
            return (interp.NoEval != 0 ? new ExprValue(0) : mathFunc.Apply(interp, argv)); // Invoke the function and copy its result back into valuePtr.
        }

        /// <summary>
        /// This procedure decides whether the leading characters of a string look like an integer or something else (such as a
        /// floating-point number or string).
        /// </summary>
        /// <returns>
        /// a boolean value indicating if the string looks like an integer.
        /// </returns>
        internal static bool LooksLikeInt(string s, int len, int i)
        {
            while (i < len && char.IsWhiteSpace(s[i]))
                i++;
            if (i >= len)
                return false;
            char c = s[i];
            if (c == '+' || c == '-')
            {
                i++;
                if (i >= len)
                    return false;
                c = s[i];
            }
            if (!char.IsDigit(c))
                return false;
            while (i < len && char.IsDigit(s[i])) //System.out.println("'" + s.charAt(i) + "' is a digit");
                i++;
            if (i >= len)
                return true;
            c = s[i];
            if (c != '.' && c != 'e' && c != 'E')
                return true;
            return false;
        }

        /// <summary>
        /// Converts a value from int or double representation to a string.</summary>
        /// <param name="interp">
        /// interpreter to use for precision information.
        /// </param>
        /// <param name="value">
        /// Value to be converted.
        /// </param>
        internal static void ExprMakeString(Interp interp, ExprValue value)
        {
            if (value.Type == ExprValue.INT)
                value.StringValue = Convert.ToString(value.IntValue);
            else if (value.Type == ExprValue.DOUBLE)
                value.StringValue = value.DoubleValue.ToString();
            value.Type = ExprValue.STRING;
        }

        internal static void CheckIntegerRange(Interp interp, double d)
        {
            if (d < 0)
            {
                if (d < ((double)TCL.INT_MIN))
                    Expression.IntegerTooLarge(interp);
            }
            else
            {
                if (d > ((double)TCL.INT_MAX))
                    Expression.IntegerTooLarge(interp);
            }
        }

        internal static void CheckWideRange(Interp interp, double d)
        {
            if (d < 0)
            {
                if (d < Int64.MinValue)
                    Expression.WideTooLarge(interp);
            }
            else
            {
                if (d > Int64.MaxValue)
                    Expression.WideTooLarge(interp);
            }
        }

        internal static void CheckDoubleRange(Interp interp, double d)
        {
            if (d == double.NaN || d == double.NegativeInfinity || d == double.PositiveInfinity)
                Expression.DoubleTooLarge(interp);
        }
    }

    abstract class MathFunction
    {
        internal const int INT = 0;
        internal const int DOUBLE = 1;
        internal const int EITHER = 2;
        internal int[] ArgTypes;
        internal abstract ExprValue Apply(Interp interp, TclObject[] argv);
    }

    abstract class UnaryMathFunction : MathFunction
    {
        internal UnaryMathFunction()
        {
            ArgTypes = new int[1];
            ArgTypes[0] = DOUBLE;
        }
    }

    abstract class BinaryMathFunction : MathFunction
    {
        internal BinaryMathFunction()
        {
            ArgTypes = new int[2];
            ArgTypes[0] = DOUBLE;
            ArgTypes[1] = DOUBLE;
        }
    }

    abstract class NoArgMathFunction : MathFunction
    {
        internal NoArgMathFunction()
        {
            ArgTypes = new int[0];
        }
    }

    class Atan2Function : BinaryMathFunction
    {
        internal override ExprValue Apply(Interp interp, TclObject[] argv)
        {
            return new ExprValue(Math.Atan2(TclDouble.Get(interp, argv[0]), TclDouble.Get(interp, argv[1])));
        }
    }

    class AbsFunction : MathFunction
    {
        internal AbsFunction()
        {
            ArgTypes = new int[1];
            ArgTypes[0] = EITHER;
        }

        internal override ExprValue Apply(Interp interp, TclObject[] argv)
        {
            if (argv[0].InternalRep is TclDouble)
            {
                double d = TclDouble.Get(interp, argv[0]);
                return (d > 0 ? new ExprValue(d) : new ExprValue(-d));
            }
            else
            {
                int i = TclInteger.Get(interp, argv[0]);
                return (i > 0 ? new ExprValue(i) : new ExprValue(-i));
            }
        }
    }

    class DoubleFunction : MathFunction
    {
        internal DoubleFunction()
        {
            ArgTypes = new int[1];
            ArgTypes[0] = EITHER;
        }

        internal override ExprValue Apply(Interp interp, TclObject[] argv)
        {
            return new ExprValue(TclDouble.Get(interp, argv[0]));
        }
    }

    class IntFunction : MathFunction
    {
        internal IntFunction()
        {
            ArgTypes = new int[1];
            ArgTypes[0] = EITHER;
        }

        internal override ExprValue Apply(Interp interp, TclObject[] argv)
        {
            double d = TclDouble.Get(interp, argv[0]);
            Expression.CheckIntegerRange(interp, d);
            return new ExprValue((int)d);
        }
    }

    class WideFunction : MathFunction
    {
        internal WideFunction()
        {
            ArgTypes = new int[1];
            ArgTypes[0] = EITHER;
        }

        internal override ExprValue Apply(Interp interp, TclObject[] argv)
        {
            double d = TclDouble.Get(interp, argv[0]);
            Expression.CheckWideRange(interp, d);
            return new ExprValue((long)d);
        }
    }

    class RoundFunction : MathFunction
    {
        internal RoundFunction()
        {
            ArgTypes = new int[1];
            ArgTypes[0] = EITHER;
        }

        internal override ExprValue Apply(Interp interp, TclObject[] argv)
        {
            if (argv[0].InternalRep is TclDouble)
            {
                double d = TclDouble.Get(interp, argv[0]);
                if (d < 0)
                {
                    Expression.CheckIntegerRange(interp, d - 0.5);
                    return new ExprValue((int)(d - 0.5));
                }
                else
                {
                    Expression.CheckIntegerRange(interp, d + 0.5);
                    return new ExprValue((int)(d + 0.5));
                }
            }
            else
                return new ExprValue(TclInteger.Get(interp, argv[0]));
        }
    }

    class PowFunction : BinaryMathFunction
    {
        internal override ExprValue Apply(Interp interp, TclObject[] argv)
        {
            double d = System.Math.Pow(TclDouble.Get(interp, argv[0]), TclDouble.Get(interp, argv[1]));
            Expression.CheckDoubleRange(interp, d);
            return new ExprValue(d);
        }
    }

    class AcosFunction : UnaryMathFunction
    {
        internal override ExprValue Apply(Interp interp, TclObject[] argv)
        {
            double d = TclDouble.Get(interp, argv[0]);
            if (d < -1 || d > 1)
                Expression.DomainError(interp);
            return new ExprValue(Math.Acos(d));
        }
    }

    class AsinFunction : UnaryMathFunction
    {
        internal override ExprValue Apply(Interp interp, TclObject[] argv)
        {
            return new ExprValue(Math.Asin(TclDouble.Get(interp, argv[0])));
        }
    }

    class AtanFunction : UnaryMathFunction
    {
        internal override ExprValue Apply(Interp interp, TclObject[] argv)
        {
            return new ExprValue(Math.Atan(TclDouble.Get(interp, argv[0])));
        }
    }

    class CeilFunction : UnaryMathFunction
    {
        internal override ExprValue Apply(Interp interp, TclObject[] argv)
        {
            return new ExprValue(Math.Ceiling(TclDouble.Get(interp, argv[0])));
        }
    }

    class CosFunction : UnaryMathFunction
    {
        internal override ExprValue Apply(Interp interp, TclObject[] argv)
        {
            return new ExprValue(Math.Cos(TclDouble.Get(interp, argv[0])));
        }
    }

    class CoshFunction : UnaryMathFunction
    {
        internal override ExprValue Apply(Interp interp, TclObject[] argv)
        {
            double x = TclDouble.Get(interp, argv[0]);
            double d1 = Math.Pow(System.Math.E, x);
            double d2 = Math.Pow(System.Math.E, -x);
            Expression.CheckDoubleRange(interp, d1);
            Expression.CheckDoubleRange(interp, d2);
            return new ExprValue((d1 + d2) / 2);
        }
    }

    class ExpFunction : UnaryMathFunction
    {
        internal override ExprValue Apply(Interp interp, TclObject[] argv)
        {
            double d = Math.Exp(TclDouble.Get(interp, argv[0]));
            if (d == double.NaN || d == double.NegativeInfinity || d == double.PositiveInfinity)
                Expression.DoubleTooLarge(interp);
            return new ExprValue(d);
        }
    }

    class FloorFunction : UnaryMathFunction
    {
        internal override ExprValue Apply(Interp interp, TclObject[] argv)
        {
            return new ExprValue(Math.Floor(TclDouble.Get(interp, argv[0])));
        }
    }

    class FmodFunction : BinaryMathFunction
    {
        internal override ExprValue Apply(Interp interp, TclObject[] argv)
        {
            return new ExprValue(Math.IEEERemainder(TclDouble.Get(interp, argv[0]), TclDouble.Get(interp, argv[1])));
        }
    }

    class HypotFunction : BinaryMathFunction
    {
        internal override ExprValue Apply(Interp interp, TclObject[] argv)
        {
            double x = TclDouble.Get(interp, argv[0]);
            double y = TclDouble.Get(interp, argv[1]);
            return new ExprValue(Math.Sqrt((x * x) + (y * y)));
        }
    }

    class LogFunction : UnaryMathFunction
    {
        internal override ExprValue Apply(Interp interp, TclObject[] argv)
        {
            return new ExprValue(Math.Log(TclDouble.Get(interp, argv[0])));
        }
    }

    class Log10Function : UnaryMathFunction
    {
        private static readonly double log10 = Math.Log(10);

        internal override ExprValue Apply(Interp interp, TclObject[] argv)
        {
            return new ExprValue(Math.Log(TclDouble.Get(interp, argv[0])) / log10);
        }
    }

    class SinFunction : UnaryMathFunction
    {
        internal override ExprValue Apply(Interp interp, TclObject[] argv)
        {
            return new ExprValue(Math.Sin(TclDouble.Get(interp, argv[0])));
        }
    }

    class SinhFunction : UnaryMathFunction
    {
        internal override ExprValue Apply(Interp interp, TclObject[] argv)
        {
            double x = TclDouble.Get(interp, argv[0]);
            double d1 = Math.Pow(Math.E, x);
            double d2 = Math.Pow(Math.E, -x);
            Expression.CheckDoubleRange(interp, d1);
            Expression.CheckDoubleRange(interp, d2);
            return new ExprValue((d1 - d2) / 2);
        }
    }

    class SqrtFunction : UnaryMathFunction
    {
        internal override ExprValue Apply(Interp interp, TclObject[] argv)
        {
            return new ExprValue(Math.Sqrt(TclDouble.Get(interp, argv[0])));
        }
    }

    class TanFunction : UnaryMathFunction
    {
        internal override ExprValue Apply(Interp interp, TclObject[] argv)
        {
            return new ExprValue(Math.Tan(TclDouble.Get(interp, argv[0])));
        }
    }

    class TanhFunction : UnaryMathFunction
    {
        internal override ExprValue Apply(Interp interp, TclObject[] argv)
        {
            double x = TclDouble.Get(interp, argv[0]);
            if (x == 0)
                return new ExprValue(0.0);
            double d1 = Math.Pow(Math.E, x);
            double d2 = Math.Pow(Math.E, -x);
            Expression.CheckDoubleRange(interp, d1);
            Expression.CheckDoubleRange(interp, d2);
            return new ExprValue((d1 - d2) / (d1 + d2));
        }
    }

    class RandFunction : NoArgMathFunction
    {
        // Generate the random number using the linear congruential generator defined by the following recurrence:
        //		seed = ( IA * seed ) mod IM
        // where IA is 16807 and IM is (2^31) - 1.  In order to avoid potential problems with integer overflow, the  code uses
        // additional constants IQ and IR such that
        //		IM = IA*IQ + IR
        // For details on how this algorithm works, refer to the following papers: 
        //	S.K. Park & K.W. Miller, "Random number generators: good ones are hard to find," Comm ACM 31(10):1192-1201, Oct 1988
        //	W.H. Press & S.A. Teukolsky, "Portable random number generators," Computers in Physics 6(5):522-524, Sep/Oct 1992.
        private const int RandIA = 16807;
        private const int RandIM = 2147483647;
        private const int RandIQ = 127773;
        private const int RandIR = 2836;
        private static readonly DateTime _date = DateTime.Now;

        /// <summary>
        /// Srand calls the main algorithm for rand after it sets the seed. To facilitate this call, the method is static and can be used
        /// w/o creating a new object.  But we also need to maintain the inheritance hierarchy, thus the dynamic apply() calls the static statApply().
        /// </summary>
        internal override ExprValue Apply(Interp interp, TclObject[] argv)
        {
            return (StatApply(interp, argv));
        }

        internal static ExprValue StatApply(Interp interp, TclObject[] argv)
        {
            if (!(interp.RandSeedInit))
            {
                interp.RandSeedInit = true;
                interp.RandSeed = (int)_date.Ticks;
            }
            if (interp.RandSeed == 0) // Don't allow a 0 seed, since it breaks the generator.  Shift it to some other value.
                interp.RandSeed = 123459876;
            int tmp = (int)(interp.RandSeed / RandIQ);
            interp.RandSeed = ((RandIA * (interp.RandSeed - tmp * RandIQ)) - RandIR * tmp);
            if (interp.RandSeed < 0)
                interp.RandSeed += RandIM;
            return new ExprValue(interp.RandSeed * (1.0 / RandIM));
        }
    }

    class SrandFunction : UnaryMathFunction
    {
        internal override ExprValue Apply(Interp interp, TclObject[] argv)
        {
            // Reset the seed.
            interp.RandSeedInit = true;
            interp.RandSeed = (long)TclDouble.Get(interp, argv[0]);
            // To avoid duplicating the random number generation code we simply call the static random number generator in the RandFunction  class.
            return (RandFunction.StatApply(interp, null));
        }
    }
}
