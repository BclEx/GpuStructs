#ifndef __SYSTEM_STRING_H__
#define __SYSTEM_STRING_H__
#include "..\System.h"
namespace Sys { namespace Text {

#define ASSERT_ENUM_STRING(string, index) (1 / (int)!(string - index) ? #string : "")

	/// <summary>
	/// utf8Encoding_t
	/// </summary>
	enum utf8Encoding_t
	{
		UTF8_PURE_ASCII,		// no characters with values > 127
		UTF8_ENCODED_BOM,		// characters > 128 encoded with UTF8, but no byte-order-marker at the beginning
		UTF8_ENCODED_NO_BOM,	// characters > 128 encoded with UTF8, with a byte-order-marker at the beginning
		UTF8_INVALID,			// has values > 127 but isn't valid UTF8 
		UTF8_INVALID_BOM		// has a byte-order-marker at the beginning, but isn't valuid UTF8 -- it's messed up
	};

	// these library functions should not be used for cross platform compatibility
#define strcmp String::Cmp // use_String_Cmp
#define strncmp use_String_Cmpn

#if defined(StrCmpN)
#undef StrCmpN
#endif
#define StrCmpN use_String_Cmpn

#if defined(strcmpi)
#undef strcmpi
#endif
#define strcmpi use_String_Icmp

#if defined(StrCmpI)
#undef StrCmpI
#endif
#define StrCmpI	 use_String_Icmp

#if defined(StrCmpNI)
#undef StrCmpNI
#endif
#define StrCmpNI use_String_Icmpn

#define stricmp String::Icmp // use_String_Icmp
#define _stricmp use_String_Icmp
#define strcasecmp use_String_Icmp
#define strnicmp use_String_Icmpn
#define _strnicmp use_String_Icmpn
#define _memicmp use_String_Icmpn

#define snprintf use_String_snPrintf
#define _snprintf use_String_snPrintf
#define vsnprintf use_String_vsnPrintf
#define _vsnprintf use_String_vsnPrintf

#ifndef FILE_HASH_SIZE
#define FILE_HASH_SIZE 1024
#endif

	// make String a multiple of 16 bytes long don't make too large to keep memory requirements to a minimum
	const int STR_ALLOC_BASE = 20;
	const int STR_ALLOC_GRAN = 32;

	// color escape character
	const int C_COLOR_ESCAPE	= '^';
	const int C_COLOR_DEFAULT	= '0';
	const int C_COLOR_RED		= '1';
	const int C_COLOR_GREEN		= '2';
	const int C_COLOR_YELLOW	= '3';
	const int C_COLOR_BLUE		= '4';
	const int C_COLOR_CYAN		= '5';
	const int C_COLOR_ORANGE	= '6';
	const int C_COLOR_WHITE		= '7';
	const int C_COLOR_GRAY		= '8';
	const int C_COLOR_BLACK		= '9';

	// color escape string
#define S_COLOR_DEFAULT		"^0"
#define S_COLOR_RED			"^1"
#define S_COLOR_GREEN		"^2"
#define S_COLOR_YELLOW		"^3"
#define S_COLOR_BLUE		"^4"
#define S_COLOR_CYAN		"^5"
#define S_COLOR_ORANGE		"^6"
#define S_COLOR_WHITE		"^7"
#define S_COLOR_GRAY		"^8"
#define S_COLOR_BLACK		"^9"


	/// <summary>
	/// Measure_t
	/// </summary>
	typedef enum
	{
		MEASURE_SIZE = 0,
		MEASURE_BANDWIDTH
	} Measure_t;

	/// <summary>
	/// String
	/// </summary>
	class String
	{
	public:
		static const int INVALID_POSITION = -1;
		String();
		String(const String &text);
		String(const String &text, int start, int end);
		String(const char *text);
		String(const char *text, int start, int end);
		explicit String(const bool b);
		explicit String(const char c);
		explicit String(const int i);
		explicit String(const unsigned u);
		explicit String(const float f);
		~String();

		size_t Size() const;
		const char *c_str() const;
		operator const char *() const;
		operator const char *();

		char operator[](int index) const;
		char &operator[](int index);

		void operator=(const String &text);
		void operator=(const char *text);

		friend String operator+(const String &a, const String &b);
		friend String operator+(const String &a, const char *b);
		friend String operator+(const char *a, const String &b);

		friend String operator+(const String &a, const float b);
		friend String operator+(const String &a, const int b);
		friend String operator+(const String &a, const unsigned b);
		friend String operator+(const String &a, const bool b);
		friend String operator+(const String &a, const char b);

		String &operator+=(const String &a);
		String &operator+=(const char *a);
		String &operator+=(const float a);
		String &operator+=(const char a);
		String &operator+=(const int a);
		String &operator+=(const unsigned a);
		String &operator+=(const bool a);

		// case sensitive compare
		friend bool operator==(const String &a, const String &b);
		friend bool operator==(const String &a, const char *b);
		friend bool operator==(const char *a, const String &b);

		// case sensitive compare
		friend bool	operator!=(const String &a, const String &b);
		friend bool	operator!=(const String &a, const char *b);
		friend bool operator!=(const char *a, const String &b);

		// case sensitive compare
		int Cmp(const char *text) const;
		int Cmpn(const char *text, int n) const;
		int CmpPrefix(const char *text) const;

		// case insensitive compare
		int Icmp(const char *text) const;
		int Icmpn(const char *text, int n) const;
		int IcmpPrefix(const char *text) const;

		// case insensitive compare ignoring color
		int IcmpNoColor( const char *text ) const;

		// compares paths and makes sure folders come first
		int IcmpPath(const char *text) const;
		int IcmpnPath(const char *text, int n) const;
		int IcmpPrefixPath(const char *text) const;

		int Length() const;
		int Allocated() const;
		void Empty();
		bool IsEmpty() const;
		void Clear();
		void Append(const char a);
		void Append(const String &text);
		void Append(const char *text);
		void Append(const char *text, int len);
		void Insert(const char a, int index);
		void Insert(const char *text, int index);
		void ToLower();
		void ToUpper();
		bool IsNumeric() const;
		bool IsColor() const;
		bool HasLower() const;
		bool HasUpper() const;
		int LengthWithoutColors() const;
		String &RemoveColors();
		void CapLength(int);
		void Fill(const char ch, int newlen);

		inline int UTF8Length();
		inline uint32 UTF8Char(int &idx);
		static int UTF8Length(const byte * s);
		static inline uint32 UTF8Char(const char *s, int &idx);
		static uint32 UTF8Char(const byte *s, int &idx);
		void AppendUTF8Char(uint32 c);
		inline void ConvertToUTF8();
		static bool IsValidUTF8(const uint8 *s, const int maxLen, utf8Encoding_t &encoding);
		static inline bool IsValidUTF8(const char *s, const int maxLen, utf8Encoding_t &encoding) { return IsValidUTF8((const uint8 *)s, maxLen, encoding); }
		static inline bool IsValidUTF8(const uint8 *s, const int maxLen);
		static inline bool IsValidUTF8(const char *s, const int maxLen) { return IsValidUTF8((const uint8 *)s, maxLen); }

		int Find(const char c, int start = 0, int end = -1) const;
		int Find(const char *text, bool casesensitive = true, int start = 0, int end = -1) const;
		bool Filter(const char *filter, bool casesensitive) const;
		int Last(const char c) const;								// return the index to the last occurance of 'c', returns -1 if not found
		const char *Left(int len, String &result) const;			// store the leftmost 'len' characters in the result
		const char *Right(int len, String &result) const;			// store the rightmost 'len' characters in the result
		const char *Mid(int start, int len, String &result) const;	// store 'len' characters starting at 'start' in result
		String Left(int len) const;									// return the leftmost 'len' characters
		String Right(int len) const;								// return the rightmost 'len' characters
		String Mid(int start, int len) const;						// return 'len' characters starting at 'start'
		void Format(VERIFY_FORMAT_STRING const char *fmt, ...);		// perform a threadsafe sprintf to the string
		static String FormatInt(const int num, bool isCash = false);	// formats an integer as a value with commas
		static String FormatCash( const int num) { return FormatInt(num, true); }
		void StripLeading(const char c);							// strip char from front as many times as the char occurs
		void StripLeading(const char *string);						// strip string from front as many times as the string occurs
		bool StripLeadingOnce(const char *string);					// strip string from front just once if it occurs
		void StripTrailing(const char c);							// strip char from end as many times as the char occurs
		void StripTrailing(const char *string);						// strip string from end as many times as the string occurs
		bool StripTrailingOnce(const char *string);					// strip string from end just once if it occurs
		void Strip(const char c);									// strip char from front and end as many times as the char occurs
		void Strip(const char *string);								// strip string from front and end as many times as the string occurs
		void StripTrailingWhitespace();								// strip trailing white space characters
		String &StripQuotes();										// strip quotes around string
		bool Replace(const char *old, const char *nw);
		bool ReplaceChar(const char old, const char nw);
		inline void CopyRange(const char * text, int start, int end);

		// file name methods
		int FileNameHash() const;							// hash key for the filename (skips extension)
		String &BackSlashesToSlashes();						// convert slashes
		String &SlashesToBackSlashes();						// convert slashes
		String &SetFileExtension( const char *extension );	// set the given file extension
		String &StripFileExtension();						// remove any file extension
		String &StripAbsoluteFileExtension();				// remove any file extension looking from front (useful if there are multiple .'s)
		String &DefaultFileExtension( const char *extension );	// if there's no file extension use the default
		String &DefaultPath( const char *basepath );		// if there's no path use the default
		void AppendPath( const char *text );				// append a partial path
		String &StripFilename();							// remove the filename from a path
		String &StripPath();								// remove the path from the filename
		void ExtractFilePath(String &dest) const;			// copy the file path to another string
		void ExtractFileName(String &dest) const;			// copy the filename to another string
		void ExtractFileBase(String &dest) const;			// copy the filename minus the extension to another string
		void ExtractFileExtension(String &dest) const;		// copy the file extension to another string
		bool CheckExtension(const char *ext);

		// char * methods to replace library functions
		static int Length(const char *s);
		static char *ToLower(char *s);
		static char *ToUpper(char *s);
		static bool IsNumeric(const char *s);
		static bool IsColor(const char *s);
		static bool HasLower(const char *s);
		static bool	HasUpper(const char *s);
		static int LengthWithoutColors(const char *s);
		static char *RemoveColors(char *s);
		static int Cmp(const char *s1, const char *s2);
		static int Cmpn(const char *s1, const char *s2, int n);
		static int Icmp(const char *s1, const char *s2);
		static int Icmpn(const char *s1, const char *s2, int n);
		static int IcmpNoColor(const char *s1, const char *s2);
		static int IcmpPath(const char *s1, const char *s2);			// compares paths and makes sure folders come first
		static int IcmpnPath(const char *s1, const char *s2, int n);	// compares paths and makes sure folders come first
		static void Append(char *dest, int size, const char *src);
		static void Copynz(char *dest, const char *src, int destsize);
		static int snPrintf(char *dest, int size, VERIFY_FORMAT_STRING const char *fmt, ...);
		static int vsnPrintf(char *dest, int size, const char *fmt, va_list argptr);
		static int FindChar(const char *str, const char c, int start = 0, int end = -1);
		static int FindText(const char *str, const char *text, bool casesensitive = true, int start = 0, int end = -1);
		static bool Filter(const char *filter, const char *name, bool casesensitive);
		static void StripMediaName(const char *name, String &mediaName);
		static bool CheckExtension(const char *name, const char *ext);
		static const char *FloatArrayToString(const float *array, const int length, const int precision);
		static const char *CStyleQuote(const char *str);
		static const char *CStyleUnQuote(const char *str);

		// hash keys
		static int Hash(const char *string);
		static int Hash(const char *string, int length);
		static int IHash(const char *string); // case insensitive
		static int IHash(const char *string, int length); // case insensitive

		// character methods
		static char ToLower(char c);
		static char ToUpper(char c);
		static bool CharIsPrintable(int c);
		static bool CharIsLower(int c);
		static bool CharIsUpper(int c);
		static bool CharIsAlpha(int c);
		static bool CharIsNumeric(int c);
		static bool CharIsNewLine(char c);
		static bool CharIsTab(char c);
		static int ColorIndex(int c);

		friend int sprintf(String &dest, const char *fmt, ...);
		friend int vsprintf(String &dest, const char *fmt, va_list ap);

		void ReAllocate(int amount, bool keepold); // reallocate string data buffer
		void FreeData(); // free allocated string memory

		int BestUnit(const char *format, float value, Measure_t measure); // format value in the given measurement with the best unit, returns the best unit
		void SetUnit(const char *format, float value, int unit, Measure_t measure); // format value in the requested unit and measurement

		static void InitMemory();
		static void ShutdownMemory();
		static void PurgeMemory();

		int DynamicMemoryUsed() const;
		static String FormatNumber(int number);

	protected:
		int _len;
		char *_data;
		int _allocedAndFlag;	// top bit is used to store a flag that indicates if the string data is static or not
		char _baseBuffer[STR_ALLOC_BASE];

		void EnsureAlloced(int amount, bool keepold = true);	// ensure string data buffer is large anough

		// sets the data point to the specified buffer... note that this ignores makes the passed buffer empty and ignores
		// anything currently in the String's dynamic buffer.  This method is intended to be called only from a derived class's constructor.
		inline void SetStaticBuffer(char * buffer, const int bufferLength);

	private:
		// initialize string using base buffer... call ONLY FROM CONSTRUCTOR
		inline void Construct();										

		static const uint32 STATIC_BIT	= 31;
		static const uint32 STATIC_MASK	= 1u << STATIC_BIT;
		static const uint32 ALLOCED_MASK = STATIC_MASK - 1;

		inline int GetAlloced() const { return _allocedAndFlag & ALLOCED_MASK; }
		inline void SetAlloced(const int a) { _allocedAndFlag = (_allocedAndFlag & STATIC_MASK) | (a & ALLOCED_MASK); }

		inline bool IsStatic() const { return ( _allocedAndFlag & STATIC_MASK ) != 0; }
		inline void SetStatic(const bool isStatic) { _allocedAndFlag = (_allocedAndFlag & ALLOCED_MASK) | (isStatic << STATIC_BIT); }
	};

	char *va(VERIFY_FORMAT_STRING const char *fmt, ...);

	/// <summary>
	/// Sort_String, Sort routines for sorting List<String>
	/// </summary>
	class Sort_String : public Collections::Sort_Quick<String, Sort_String>
	{
	public:
		int Compare(const String &a, const String &b) const { return a.Icmp(b); }
	};

	/// <summary>
	/// Sort_PathString
	/// </summary>
	class Sort_PathString : public Collections::Sort_Quick<String, Sort_PathString>
	{
	public:
		int Compare(const String &a, const String &b) const { return a.IcmpPath(b); }
	};

	/// <summary>
	/// String
	/// </summary>
	inline String::String() { Construct(); }
	inline String::String(const String &text)
	{
		Construct();
		int l = text.Length();
		EnsureAlloced(l + 1);
		strcpy(_data, text._data);
		_len = l;
	}
	inline String::String(const String &text, int start, int end)
	{
		Construct();
		if (end > text.Length())
			end = text.Length();
		if (start > text.Length())
			start = text.Length();
		else if (start < 0)
			start = 0;
		int l = end - start;
		if (l < 0)
			l = 0;
		EnsureAlloced(l + 1);
		for (int i = 0; i < l; i++)
			_data[i] = text[start + i];
		_data[l] = '\0';
		_len = l;
	}
	inline String::String(const char *text)
	{
		Construct();
		if (text)
		{
			int l = strlen(text);
			EnsureAlloced(l + 1);
			strcpy(_data, text);
			_len = l;
		}
	}
	inline String::String(const char *text, int start, int end)
	{
		Construct();
		int l = strlen( text );
		if (end > l)
			end = l;
		if (start > l)
			start = l;
		else if (start < 0)
			start = 0;
		l = end - start;
		if (l < 0)
			l = 0;
		EnsureAlloced(l + 1);
		for (int i = 0; i < l; i++ )
			_data[i] = text[start + i];
		_data[l] = '\0';
		_len = l;
	}
	inline String::String(const bool b)
	{
		Construct();
		EnsureAlloced(2);
		_data[0] = (b ? '1' : '0');
		_data[1] = '\0';
		_len = 1;
	}
	inline String::String(const char c)
	{
		Construct();
		EnsureAlloced(2);
		_data[0] = c;
		_data[1] = '\0';
		_len = 1;
	}
	inline String::String(const int i)
	{
		Construct();
		char text[64];
		int l = ::sprintf(text, "%d", i);
		EnsureAlloced(l + 1);
		strcpy(_data, text);
		_len = l;
	}
	inline String::String(const unsigned u)
	{
		Construct();
		char text[64];
		int l = sprintf(text, "%u", u);
		EnsureAlloced(l + 1);
		strcpy(_data, text);
		_len = l;
	}
	inline String::String( const float f ) {
		Construct();
		char text[64];
		int l = String::snPrintf(text, sizeof(text), "%f", f);
		while (l > 0 && text[l - 1] == '0') text[--l] = '\0';
		while (l > 0 && text[l - 1] == '.') text[--l] = '\0';
		EnsureAlloced(l + 1);
		strcpy(_data, text);
		_len = l;
	}
	/// <summary>
	/// ~String
	/// </summary>
	inline String::~String() { FreeData(); }

	/// <summary>
	/// Construct
	/// </summary>
	inline void String::Construct()
	{
		SetStatic(false);
		SetAlloced(STR_ALLOC_BASE);
		_data = _baseBuffer;
		_len = 0;
		_data[0] = '\0';
#ifdef DEBUG_UNINITIALIZED_MEMORY
		memset(_baseBuffer, 0, sizeof(_baseBuffer));
#endif
	}

	/// <summary>
	/// EnsureAlloced
	/// </summary>
	inline void String::EnsureAlloced(int amount, bool keepold)
	{
		// static string's can't reallocate
		if (IsStatic())
		{
			release_assert(amount <= GetAlloced());
			return;
		}
		if (amount > GetAlloced())
			ReAllocate(amount, keepold);
	}

	/// <summary>
	/// SetStaticBuffer
	/// </summary>
	inline void String::SetStaticBuffer(char *buffer, const int bufferLength)
	{ 
		// this should only be called on a freshly constructed String
		assert(_data == _baseBuffer);
		_data = buffer;
		_len = 0;
		SetAlloced(bufferLength);
		SetStatic(true);
	}

	/// <summary>
	/// Size
	/// </summary>
	inline size_t String::Size() const { return sizeof(*this) + Allocated(); }

	/// <summary>
	/// c_str
	/// </summary>
	inline const char *String::c_str() const { return _data; }

	/// <summary>
	/// operators
	/// </summary>
	inline String::operator const char *() { return c_str(); }
	inline String::operator const char *() const { return c_str(); }
	inline char String::operator[](int index) const { assert(index >= 0 && index <= len); return _data[index]; }
	inline char &String::operator[]( int index ) { assert(index >= 0 && index <= len); return _data[index]; }
	inline void String::operator=(const String &text) { int l = text.Length(); EnsureAlloced(l + 1, false); memcpy(_data, text._data, l); _data[l] = '\0'; _len = l; }
	inline String operator+(const String &a, const String &b) { String s(a); s.Append(b); return s; }
	inline String operator+(const String &a, const char *b) { String s(a); s.Append(b); return s; }
	inline String operator+(const char *a, const String &b) { String s(a); s.Append(b); return s; }
	inline String operator+(const String &a, const bool b) { String s(a); s.Append(b ? "true" : "false"); return s; }
	inline String operator+(const String &a, const char b) { String s(a); s.Append(b); return s; }
	inline String operator+( const String &a, const float b ) { String s(a); char text[64]; sprintf(text, "%f", b); s.Append(text); return s; }
	inline String operator+(const String &a, const int b) { String s(a); char text[64]; sprintf(text, "%d", b); s.Append(text); return s; }
	inline String operator+(const String &a, const unsigned b) { String s(a); char text[64]; sprintf(text, "%u", b); s.Append(text); return s; }
	inline String &String::operator+=(const float a) { char text[64]; sprintf(text, "%f", a); Append(text); return *this; }
	inline String &String::operator+=(const int a) { char text[64]; sprintf(text, "%d", a); Append(text); return *this; }
	inline String &String::operator+=(const unsigned a) { char text[64]; sprintf(text, "%u", a); Append(text); return *this; }
	inline String &String::operator+=(const String &a) { Append(a); return *this; }
	inline String &String::operator+=(const char *a) { Append(a); return *this; }
	inline String &String::operator+=(const char a) { Append(a); return *this; }
	inline String &String::operator+=(const bool a) { Append(a ? "true" : "false"); return *this; }
	inline bool operator==(const String &a, const String &b) { return (!String::Cmp(a._data, b._data)); }
	inline bool operator==(const String &a, const char *b) { assert(b); return (!String::Cmp(a._data, b)); }
	inline bool operator==(const char *a, const String &b) { assert(a); return (!String::Cmp(a, b._data)); }
	inline bool operator!=(const String &a, const String &b) { return !(a == b); }
	inline bool operator!=(const String &a, const char *b) { return !(a == b); }
	inline bool operator!=(const char *a, const String &b) { return !(a == b); }

	/// <summary>
	/// Cmp
	/// </summary>
	inline int String::Cmp(const char *text) const { assert(text); return String::Cmp(_data, text); }
	inline int String::Cmpn(const char *text, int n) const { assert(text); return String::Cmpn(_data, text, n); }
	inline int String::CmpPrefix(const char *text) const { assert(text); return String::Cmpn(_data, text, strlen(text)); }
	inline int String::Icmp(const char *text) const { assert(text); return String::Icmp(_data, text); }
	inline int String::Icmpn(const char *text, int n) const { assert(text); return String::Icmpn(_data, text, n); }
	inline int String::IcmpPrefix(const char *text) const { assert(text); return String::Icmpn(_data, text, strlen(text)); }
	inline int String::IcmpNoColor(const char *text) const { assert(text); return String::IcmpNoColor(_data, text); }
	inline int String::IcmpPath(const char *text) const { assert(text); return String::IcmpPath(_data, text); }
	inline int String::IcmpnPath(const char *text, int n) const { assert(text); return String::IcmpnPath(_data, text, n); }
	inline int String::IcmpPrefixPath( const char *text ) const { assert(text); return String::IcmpnPath(_data, text, strlen(text)); }

	inline int String::Length() const { return _len; }
	inline int String::Allocated() const { return (_data != _baseBuffer ? GetAlloced() : 0); }
	inline void String::Empty() { EnsureAlloced(1); _data[0] = '\0'; _len = 0; }
	inline bool String::IsEmpty() const { return (String::Cmp(_data, "") == 0); }

	/// <summary>
	/// Clear
	/// </summary>
	inline void String::Clear()
	{
		if (IsStatic())
		{
			_len = 0;
			_data[0] = '\0';
			return;
		}
		FreeData();
		Construct();
	}

	/// <summary>
	/// Append
	/// </summary>
	inline void String::Append(const char a) { EnsureAlloced(_len + 2); _data[_len++] = a; _data[_len] = '\0'; }
	inline void String::Append(const String &text)
	{
		int newLen = _len + text.Length();
		EnsureAlloced(newLen + 1);
		for (int i = 0; i < text._len; i++)
			_data[_len + i] = text[i];
		_len = newLen;
		_data[_len] = '\0';
	}
	inline void String::Append(const char *text)
	{
		if (text)
		{
			int newLen = _len + strlen(text);
			EnsureAlloced(newLen + 1);
			for (int i = 0; text[i]; i++)
				_data[_len + i] = text[i];
			_len = newLen;
			_data[_len] = '\0';
		}
	}
	inline void String::Append(const char *text, int l)
	{
		if (text && l)
		{
			int newLen = _len + l;
			EnsureAlloced(newLen + 1);
			for (int i = 0; text[i] && i < l; i++)
				_data[_len + i] = text[i];
			_len = newLen;
			_data[_len] = '\0';
		}
	}

	/// <summary>
	/// Insert
	/// </summary>
	inline void String::Insert(const char a, int index)
	{
		if (index < 0)
			index = 0;
		else if (index > _len)
			index = _len;
		int l = 1;
		EnsureAlloced(_len + l + 1);
		for (int i = _len; i >= index; i--)
			_data[i + l] = _data[i];
		_data[index] = a;
		_len++;
	}
	inline void String::Insert(const char *text, int index)
	{
		if (index < 0)
			index = 0;
		else if (index > _len)
			index = _len;
		int l = strlen(text);
		EnsureAlloced(_len + l + 1);
		for (int i = _len; i >= index; i--) 
			_data[i + l] = _data[i];
		for (int i = 0; i < l; i++) 
			_data[index + i] = text[i];
		_len += l;
	}

	inline void String::ToLower() { for (int i = 0; _data[i]; i++) if (CharIsUpper(_data[i])) _data[i] += ('a' - 'A'); }
	inline void String::ToUpper() { for (int i = 0; _data[i]; i++) if (CharIsLower(_data[i])) _data[i] -= ('a' - 'A'); }
	inline bool String::IsNumeric() const { return String::IsNumeric(_data); }
	inline bool String::IsColor() const { return String::IsColor(_data); }
	inline bool String::HasLower() const { return String::HasLower(_data); }
	inline bool String::HasUpper() const { return String::HasUpper(_data); }
	inline String &String::RemoveColors() { String::RemoveColors(_data); _len = Length(_data); return *this; }
	inline int String::LengthWithoutColors() const { return String::LengthWithoutColors(_data); }
	inline void String::CapLength(int newlen) { if (_len <= newlen) return; _data[newlen] = 0; _len = newlen; }
	inline void String::Fill(const char ch, int newlen) { EnsureAlloced(newlen + 1); _len = newlen; memset(_data, ch, _len); _data[_len] = 0; }
	inline int String::UTF8Length() { return UTF8Length((byte *)_data); }
	inline uint32 String::UTF8Char(int &idx) { return UTF8Char((byte *)_data, idx); }
	inline void String::ConvertToUTF8() { Clear(); String s(*this); for (int index = 0; index < s.Length(); ++index) AppendUTF8Char(s[index]); }
	inline uint32 String::UTF8Char(const char *s, int &idx) { return UTF8Char((byte *)s, idx); }
	inline bool String::IsValidUTF8(const uint8 * s, const int maxLen) { utf8Encoding_t encoding; return IsValidUTF8(s, maxLen, encoding); }
	inline int String::Find(const char c, int start, int end) const { if (end == -1) end = _len; return String::FindChar(_data, c, start, end); }
	inline int String::Find(const char *text, bool casesensitive, int start, int end) const { if (end == -1) end = _len; return String::FindText(_data, text, casesensitive, start, end); }
	inline bool String::Filter(const char *filter, bool casesensitive) const { return String::Filter(filter, _data, casesensitive); }
	inline const char *String::Left(int len, String &result) const { return Mid(0, len, result); }
	inline const char *String::Right(int len, String &result) const
	{
		if (len >= Length())
		{
			result = *this;
			return result;
		}
		return Mid(Length() - len, len, result);
	}
	inline String String::Left(int len) const { return Mid(0, len); }
	inline String String::Right(int len) const { return (len >= Length() ? *this : Mid(Length() - len, len)); }
	inline void String::Strip(const char c) { StripLeading(c); StripTrailing(c); }
	inline void String::Strip(const char *string) { StripLeading(string); StripTrailing(string); }
	inline bool String::CheckExtension(const char *ext) { return String::CheckExtension(_data, ext); }
	inline int String::Length(const char *s) { int i; for (i = 0; s[i]; i++) { } return i; }
	inline char *String::ToLower(char *s) { for (int i = 0; s[i]; i++) if (CharIsUpper(s[i])) s[i] += ('a' - 'A'); return s; }
	inline char *String::ToUpper(char *s) { for (int i = 0; s[i]; i++) if (CharIsLower(s[i])) s[i] -= ('a' - 'A'); return s; }
	inline int String::Hash(const char *string) { int hash = 0; for (int i = 0; *string != '\0'; i++) hash += (*string++) * (i + 119); return hash; }
	inline int String::Hash(const char *string, int length) { int hash = 0; for (int i = 0; i < length; i++) hash += (*string++) * (i + 119); return hash; }
	inline int String::IHash(const char *string) { int hash = 0; for (int i = 0; *string != '\0'; i++) hash += ToLower(*string++) * (i + 119); return hash; }
	inline int String::IHash(const char *string, int length) { int hash = 0; for (int i = 0; i < length; i++) hash += ToLower(*string++) * (i + 119); return hash; }
	inline bool String::IsColor(const char *s) { return (s[0] == C_COLOR_ESCAPE && s[1] != '\0' && s[1] != ' '); }
	inline char String::ToLower(char c) { return (c <= 'Z' && c >= 'A' ? c + ('a' - 'A') : c); }
	inline char String::ToUpper(char c) { return (c >= 'a' && c <= 'z' ? c - ('a' - 'A') : c); }
	inline bool String::CharIsPrintable(int c) { return ((c >= 0x20 && c <= 0x7E) || (c >= 0xA1 && c <= 0xFF)); } // test for regular ascii and western European high-ascii chars
	inline bool String::CharIsLower(int c) { return ((c >= 'a' && c <= 'z') || (c >= 0xE0 && c <= 0xFF)); } // test for regular ascii and western European high-ascii chars
	inline bool String::CharIsUpper(int c) { return ((c <= 'Z' && c >= 'A') || (c >= 0xC0 && c <= 0xDF)); }// test for regular ascii and western European high-ascii chars
	inline bool String::CharIsAlpha(int c) { return ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= 0xC0 && c <= 0xFF)); } // test for regular ascii and western European high-ascii chars
	inline bool String::CharIsNumeric(int c) { return (c <= '9' && c >= '0'); }
	inline bool String::CharIsNewLine( char c ) { return (c == '\n' || c == '\r' || c == '\v'); }
	inline bool String::CharIsTab(char c) { return (c == '\t'); }
	inline int String::ColorIndex(int c) { return (c & 15); }
	inline int String::DynamicMemoryUsed() const { return (_data == _baseBuffer ? 0 : GetAlloced()); }

	inline void String::CopyRange(const char * text, int start, int end)
	{
		int l = end - start;
		if (l < 0)
			l = 0;
		EnsureAlloced( l + 1 );
		for (int i = 0; i < l; i++)
			_data[i] = text[start + i];
		_data[l] = '\0';
		_len = l;
	}

	/// <summary>
	/// StrStatic
	/// </summary>
	template<int Size>
	class StrStatic : public String
	{
	public:
		// we should only get here when the types, including the size, are identical
		inline void operator=(const StrStatic &text) { _len = text.Length(); memcpy(_data, text._data, _len + 1); }
		// all String operators are overloaded and the String default constructor is called so that the static buffer can be initialized in the body of the constructor before the data is ever copied.
		inline StrStatic() { _buffer[ 0 ] = '\0'; SetStaticBuffer(_buffer, Size); }
		inline StrStatic(const StrStatic &text) : String() { _buffer[0] = '\0'; SetStaticBuffer(_buffer, Size); String::operator=(text); }
		inline StrStatic(const String &text) : String() { _buffer[0] = '\0'; SetStaticBuffer(_buffer, Size); String::operator=(text); }
		inline StrStatic(const StrStatic &text, int start, int end) : String() { _buffer[0] = '\0'; SetStaticBuffer(_buffer, Size); CopyRange(text.c_str(), start, end); }
		inline StrStatic(const char *text) : Str() { _buffer[ 0 ] = '\0'; SetStaticBuffer(_buffer, Size); String::operator=(text); }
		inline StrStatic(const char *text, int start, int end) : String() { _buffer[0] = '\0'; SetStaticBuffer(_buffer, Size); CopyRange(text, start, end); }
		inline explicit StrStatic(const bool b) : String() { _buffer[0] = '\0'; SetStaticBuffer(_buffer, Size); String::operator=(b); }
		inline explicit StrStatic(const char c) : String() { _buffer[0] = '\0'; SetStaticBuffer(_buffer, Size); String::operator=(c); }
		inline explicit StrStatic(const int i) : String() { _buffer[0] = '\0'; SetStaticBuffer(_buffer, Size); String::operator=(i); }
		inline explicit StrStatic(const unsigned u) : String() { _buffer[0] = '\0'; SetStaticBuffer(_buffer, Size); String::operator=(u); }
		inline explicit StrStatic(const float f) : String() { _buffer[0] = '\0'; SetStaticBuffer(_buffer, Size); idStr::operator=(f); }

	private:
		char _buffer[Size];
	};

}}
#endif /* __SYSTEM_STRING_H__ */
