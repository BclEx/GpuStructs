#ifndef __SYSTEM_CLASS_H__
#define __SYSTEM_CLASS_H__
#include "System.h"
namespace Sys {

	class Class;
	class Type;

	extern const Event EV_Remove;
	extern const Event EV_SafeRemove;

	typedef void (Class::*eventCallback_t)();
	template<class T>
	struct EventFunc
	{
		const Event *event;
		eventCallback_t function;
	};

	// added & so gcc could compile this
#define EVENT(event, function) { &(event), (void (Class::*)())(&function) },
#define END_CLASS { nullptr, nullptr } };

	/// <summary>
	/// EventArg
	/// </summary>
	class EventArg
	{
	public:
		int type;
		int value;
		EventArg()					{ type = D_EVENT_INTEGER; value = 0; };
		EventArg(int data)			{ type = D_EVENT_INTEGER; value = data; };
		EventArg(float data)		{ type = D_EVENT_FLOAT; value = *reinterpret_cast<int *>(&data); };
		EventArg(const Text::String &data){ type = D_EVENT_STRING; value = reinterpret_cast<int>(data.c_str()); };
		EventArg(const char *data)	{ type = D_EVENT_STRING; value = reinterpret_cast<int>(data); };
	};

	/// <summary>
	/// AllocException
	/// </summary>
	class AllocException : public Exception
	{
	public:
		AllocException(const char *text = "") : Exception(text) { }
	};

	/// <summary>
	/// CLASS_PROTOTYPE
	/// This macro must be included in the definition of any subclass of idClass. It prototypes variables used in class instanciation and type checking.
	/// Use this on single inheritance concrete classes only.
	/// </summary>
#define CLASS_PROTOTYPE(T) \
public: \
	static Type _type; \
	static Class *CreateInstance(); \
	virtual Type *GetType() const; \
	static EventFunc<T> _eventCallbacks[]

	/// <summary>
	/// CLASS_DECLARATION
	/// This macro must be included in the code to properly initialize variables used in type checking and run-time instanciation.  It also defines the list
	/// of events that the class responds to.  Take special care to ensure that the proper superclass is indicated or the run-time type information will be
	/// incorrect.  Use this on concrete classes only.
	/// </summary>
#define CLASS_DECLARATION(T, base) \
	Type T::_type(#T, #base, (EventFunc<Class> *)T::_eventCallbacks, T::CreateInstance, \
	(void (Class::*)())&T::Spawn, (void (Class::*)(BinaryWriter *) const)&T::Writer, (void (Class::*)(BinaryReader *))&T::Reader); \
	Class *T::CreateInstance() { try { T *p = new T; p->FindUninitializedMemory(); return p; }  catch(AllocException &) { return nullptr; } } \
	Type *T::GetType() const { return &(T::_type); } \
	EventFunc<T> T::_eventCallbacks[] = {

	/// <summary>
	/// ABSTRACT_PROTOTYPE
	/// This macro must be included in the definition of any abstract subclass of idClass. It prototypes variables used in class instanciation and type checking.
	/// Use this on single inheritance abstract classes only.
	/// </summary>
#define ABSTRACT_PROTOTYPE(T) \
	public: \
	static Type _type; \
	static Class *CreateInstance(); \
	virtual	Type *GetType() const; \
	static EventFunc<T> _eventCallbacks[]

	/// <summary>
	/// ABSTRACT_DECLARATION
	/// This macro must be included in the code to properly initialize variables used in type checking.  It also defines the list of events that the class
	/// responds to.  Take special care to ensure that the proper superclass is indicated or the run-time tyep information will be incorrect.  Use this
	/// on abstract classes only.
	/// </summary>
#define ABSTRACT_DECLARATION(T, base) \
	Type T::_type(#T, #base, \
	(EventFunc<Class> *)T::_eventCallbacks, T::CreateInstance, (void (Class::*)())&T::Spawn, \
	(void (Class::*)(BinaryWriter *) const)&T::Writer, (void (Class::*)(BinaryReader *))&T::Reader); \
	Class *T::CreateInstance() { gameLocal.Error("Cannot instanciate abstract class %s.", #T); return nullptr; } \
	Type *T::GetType() const { return &(T::_type); } \
	EventFunc<T> T::_eventCallbacks[] = {

	typedef void (Class::*classSpawnFunc_t)();
	class BinaryWriter;
	class BinaryReader;

	/// <summary>
	/// Class
	/// </summary>
	class Class {
	public:
		ABSTRACT_PROTOTYPE(Class);

		void *operator new(size_t);
		void operator delete(void *);
		virtual ~Class();

		void Spawn();
		void CallSpawn();
		bool IsType(const Type &c) const;
		const char *GetClassname() const;
		const char *GetSuperclass() const;
		void FindUninitializedMemory();

		void Writer(BinaryWriter *writer) const { };
		void Restore(BinaryReader *reader) { };

		bool HasEvent(const Event &ev) const;
		bool PostEventMS(const Event *ev, int time);
		bool PostEventMS(const Event *ev, int time, EventArg arg1);
		bool PostEventMS(const Event *ev, int time, EventArg arg1, EventArg arg2);
		bool PostEventMS(const Event *ev, int time, EventArg arg1, EventArg arg2, EventArg arg3);
		bool PostEventMS(const Event *ev, int time, EventArg arg1, EventArg arg2, EventArg arg3, EventArg arg4);
		bool PostEventMS(const Event *ev, int time, EventArg arg1, EventArg arg2, EventArg arg3, EventArg arg4, EventArg arg5);
		bool PostEventMS(const Event *ev, int time, EventArg arg1, EventArg arg2, EventArg arg3, EventArg arg4, EventArg arg5, EventArg arg6);
		bool PostEventMS(const Event *ev, int time, EventArg arg1, EventArg arg2, EventArg arg3, EventArg arg4, EventArg arg5, EventArg arg6, EventArg arg7);
		bool PostEventMS(const Event *ev, int time, EventArg arg1, EventArg arg2, EventArg arg3, EventArg arg4, EventArg arg5, EventArg arg6, EventArg arg7, EventArg arg8);
		bool PostEventSec(const Event *ev, float time);
		bool PostEventSec(const Event *ev, float time, EventArg arg1);
		bool PostEventSec(const Event *ev, float time, EventArg arg1, EventArg arg2);
		bool PostEventSec(const Event *ev, float time, EventArg arg1, EventArg arg2, EventArg arg3);
		bool PostEventSec(const Event *ev, float time, EventArg arg1, EventArg arg2, EventArg arg3, EventArg arg4);
		bool PostEventSec(const Event *ev, float time, EventArg arg1, EventArg arg2, EventArg arg3, EventArg arg4, EventArg arg5);
		bool PostEventSec(const Event *ev, float time, EventArg arg1, EventArg arg2, EventArg arg3, EventArg arg4, EventArg arg5, EventArg arg6);
		bool PostEventSec(const Event *ev, float time, EventArg arg1, EventArg arg2, EventArg arg3, EventArg arg4, EventArg arg5, EventArg arg6, EventArg arg7);
		bool PostEventSec(const Event *ev, float time, EventArg arg1, EventArg arg2, EventArg arg3, EventArg arg4, EventArg arg5, EventArg arg6, EventArg arg7, EventArg arg8);
		bool ProcessEvent(const Event *ev );
		bool ProcessEvent(const Event *ev, EventArg arg1);
		bool ProcessEvent(const Event *ev, EventArg arg1, EventArg arg2);
		bool ProcessEvent(const Event *ev, EventArg arg1, EventArg arg2, EventArg arg3);
		bool ProcessEvent(const Event *ev, EventArg arg1, EventArg arg2, EventArg arg3, EventArg arg4);
		bool ProcessEvent(const Event *ev, EventArg arg1, EventArg arg2, EventArg arg3, EventArg arg4, EventArg arg5);
		bool ProcessEvent(const Event *ev, EventArg arg1, EventArg arg2, EventArg arg3, EventArg arg4, EventArg arg5, EventArg arg6);
		bool ProcessEvent(const Event *ev, EventArg arg1, EventArg arg2, EventArg arg3, EventArg arg4, EventArg arg5, EventArg arg6, EventArg arg7);
		bool ProcessEvent(const Event *ev, EventArg arg1, EventArg arg2, EventArg arg3, EventArg arg4, EventArg arg5, EventArg arg6, EventArg arg7, EventArg arg8);
		bool ProcessEventArgPtr(const Event *ev, int *data);
		void CancelEvents(const Event *ev);
		void Event_Remove();

		// Static functions
		static void Init();
		static void Shutdown();
		static Type *GetClass(const char *name);
		static Class *CreateInstance(const char *name);
		static int GetNumTypes() { return _types.Num(); }
		static int GetTypeNumBits() { return _typeNumBits; }
		static Type *GetType(int num);

	private:
		classSpawnFunc_t CallSpawnFunc(Type *cls);

		bool PostEventArgs(const Event *ev, int time, int args, ...);
		bool ProcessEventArgs(const Event *ev, int args, ...);
		void Event_SafeRemove();

		static bool _initialized;
		static Collections::List<Type *, TAG_IDCLASS> _types;
		static Collections::List<Type *, TAG_IDCLASS> _typenums;
		static int _typeNumBits;
		static int _memused;
		static int _objects;
	};

	/// <summary>
	/// Type
	/// </summary>
	class Type {
	public:
		const char *_classname;
		const char *_superclass;
		Class *(*CreateInstance)();
		void (Class::*Spawn )();
		void (Class::*Writer)(BinaryWriter *writer) const;
		void (Class::*Reader)(BinaryReader *reader);

		EventFunc<Class> *_eventCallbacks;
		eventCallback_t *_eventMap;
		Type *_super;
		Type *_next;
		bool _freeEventMap;
		int _typeNum;
		int _lastChild;

		Hierarchy<Type> _node;

		Type(const char *classname, const char *superclass, EventFunc<Class> *eventCallbacks, Class *(*CreateInstance)(),
			void (Class::*Spawn)(), void (Class::*Writer)(BinaryWriter *writer) const, void	(Class::*Reader)(BinaryReader *reader));
		~Type();

		void Init();
		void Shutdown();

		bool IsType(const Type &superclass) const;
		bool HasEvent(const Event &ev) const;
	};

	/// <summary>
	/// Checks if the object's class is a subclass of the class defined by the  passed in idTypeInfo.
	/// <summary>
	inline bool Type::IsType(const Type &type) const { return (_typeNum >= type._typeNum && typeNum <= type._lastChild); }

	/// <summary>
	/// Checks if the object's class is a subclass of the class defined by the passed in idTypeInfo.
	/// </summary>
	inline bool Class::IsType(const Type &superclass) const { const Type *t = GetType(); return t->IsType(superclass); }

	/// <summary>
	/// Returns true if Class has event.
	/// </summary>
	inline bool Class::HasEvent(const Event &ev) const { assert(EventManager::_initialized); const Type *t = GetType(); return t->HasEvent(ev); }

	/// <summary>
	/// Returns true if Type has event.
	/// </summary>
	inline bool Type::HasEvent(const Event &ev) const { assert(EventManager::initialized); return (!eventMap[ev.GetEventNum()] ? false : true); }

	}
#endif /* __SYSTEM_CLASS_H__ */
