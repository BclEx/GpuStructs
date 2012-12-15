#ifndef __SYSTEM_EVENT_H__
#define __SYSTEM_EVENT_H__
//#include "..\Thunk.h"
#include "Collections\LinkedList.h"
namespace System {

	// if changed, enable the CREATE_EVENT_CODE define in Event.cpp to generate switch statement for Class::ProcessEventArgPtr.
	// running the game will then generate c:\doom\base\events.txt, the contents of which should be copied into the switch statement.
#define D_EVENT_MAXARGS				8			
#define D_EVENT_VOID				((char)0)
#define D_EVENT_INTEGER				'd'
#define D_EVENT_FLOAT				'f'
#define D_EVENT_STRING				's'
#define MAX_EVENTS					4096

	class Class;
	class Type;

	class Event
	{
	private:
		const char *_name;
		const char *_formatspec;
		unsigned int _formatspecIndex;
		int _returnType;
		int _numargs;
		size_t _argsize;
		int _argOffset[D_EVENT_MAXARGS];
		int _eventnum;
		const Event *_next;

		static Event *_eventDefList[MAX_EVENTS];
		static int _numEventDefs;

	public:
		Event(const char *command, const char *formatspec = nullptr, char returnType = 0);

		const char *GetName() const;
		const char *GetArgFormat() const;
		unsigned int GetFormatspecIndex() const;
		char GetReturnType() const;
		int GetEventNum() const;
		int GetNumArgs() const;
		size_t GetArgSize() const;
		int GetArgOffset( int arg ) const;

		static int NumEventCommands();
		static const Event *GetEventCommand(int eventnum);
		static const Event *FindEvent(const char *name);
	};

	class BinaryWriter;
	class BinaryReader;

	class EventManager
	{
	private:
		const Event *_eventdef;
		byte *_data;
		int _time;
		Class *_object;
		const Type *_typeinfo;
		Collections::LinkedList<EventManager> _eventNode;
		static DynamicBlockAlloc<byte, 16 * 1024, 256> _eventDataAllocator;

	public:
		static bool _initialized;
		~EventManager();

		static EventManager *Alloc(const Event *evdef, int args, va_list va);
		static void CopyArgs(const Event *evdef, int args, va_list va, int data[D_EVENT_MAXARGS]);

		void Free();
		void Schedule(Class *object, const Type *cls, int time);
		byte *GetData();

		static void CancelEvents(const Class *obj, const Event *evdef = nullptr);
		static void ClearEventList();
		static void ServiceEvents();
		static void ServiceFastEvents();
		static void Init();
		static void Shutdown();

		// save games
		static void Save(BinaryWriter *writer); // archives object for save game file
		static void Restore(BinaryReader *reader); // unarchives object from save game file
	};

	/// <summary>
	/// GetData
	/// </summary>
	inline byte *EventManager::GetData() { return _data; }

	/// <summary>
	/// GetName
	/// </summary>
	inline const char *Event::GetName() const { return _name; }

	/// <summary>
	/// GetArgFormat
	/// </summary>
	inline const char *Event::GetArgFormat() const { return _formatspec; }

	/// <summary>
	/// GetFormatspecIndex
	/// </summary>
	inline unsigned int Event::GetFormatspecIndex() const { return _formatspecIndex; }

	/// <summary>
	/// GetReturnType
	/// </summary>
	inline char Event::GetReturnType() const { return _returnType; }

	/// <summary>
	/// GetNumArgs
	/// </summary>
	inline int Event::GetNumArgs() const { return _numargs; }

	/// <summary>
	/// GetArgSize
	/// </summary>
	inline size_t Event::GetArgSize() const { return _argsize; }

	/// <summary>
	/// GetArgOffset
	/// </summary>
	inline int Event::GetArgOffset(int arg) const { assert(arg >= 0 && arg < D_EVENT_MAXARGS); return _argOffset[arg]; }

	/// <summary>
	/// GetEventNum
	/// </summary>
	inline int Event::GetEventNum() const { return _eventnum; }
}
#endif /* __SYSTEM_EVENT_H__ */
