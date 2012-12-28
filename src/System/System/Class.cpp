//#include <string>
//#include "Class.h"
//#include "Collections\Hierarchy.h"
//using namespace Sys;
//
//// this is the head of a singly linked list of all the idTypes
//static Type *_typelist = nullptr;
//static Hierarchy<Type> _classHierarchy;
//static int _eventCallbackMemory = 0;
//
///// <summary>
///// Constructor for class. Should only be called from CLASS_DECLARATION macro. Handles linking class definition into class hierarchy.  This should only happen
///// at startup as idTypeInfos are statically defined.  Since static variables can be initialized in any order, the constructor must handle the case that subclasses
///// are initialized before superclasses.
///// </summary>
//Type::Type(const char *classname, const char *superclass, EventFunc<Class> *eventCallbacks, Class *(*CreateInstance)(), 
//		   void (Class::*Spawn)(), void (Class::*Writer)(BinaryWriter *writer) const, void (Class::*Reader)(BinaryReader *reader)) 
//{
//	this->_classname = classname;
//	this->_superclass = superclass;
//	this->_eventCallbacks = eventCallbacks;
//	this->_eventMap = nullptr;
//	this->Spawn = Spawn;
//	this->Writer = Writer;
//	this->Reader = Reader;
//	this->CreateInstance = CreateInstance;
//	this->_super = Class::GetClass(superclass);
//	this->_freeEventMap = false;
//	_typeNum = 0;
//	_lastChild = 0;
//	// Check if any subclasses were initialized before their superclass
//	for (Type *type = _typelist; type != nullptr; type = type->_next)
//		if (type->_super == nullptr && !String::Cmp(type->_superclass, this->_classname) && String::Cmp(type->_classname, "Class"))
//			type->_super = this;
//	// Insert sorted
//	Type **insert;
//	for (insert = &_typelist; *insert; insert = &(*insert)->_next)
//	{
//		assert(String::Cmp(classname, (*insert)->_classname));
//		if (String::Cmp(classname, (*insert)->_classname) < 0)
//		{
//			_next = *insert;
//			*insert = this;
//			break;
//		}
//	}
//	if (!*insert)
//	{
//		*insert = this;
//		_next = nullptr;
//	}
//}
//
///// <summary>
///// Type
///// </summary>
//Type::~Type() { Shutdown(); }
//
///// <summary>
///// Initializes the event callback table for the class.  Creates a table for fast lookups of event functions.  Should only be called once.
///// </summary>
//void Type::Init()
//{
//	// we've already been initialized by a subclass
//	if (_eventMap)
//		return;
//	// make sure our superclass is initialized first
//	if (_super && !_super->_eventMap)
//		_super->Init();
//	// add to our node hierarchy
//	_node.ParentTo(_super ? _super->_node : _classHierarchy);
//	_node.SetOwner(this);
//
//	// keep track of the number of children below each class
//	Type *c;
//	for (c = _super; c != nullptr; c = c->_super)
//		c->_lastChild++;
//	// if we're not adding any new event callbacks, we can just use our superclass's table
//	if ((!_eventCallbacks || !_eventCallbacks->event) && _super)
//	{
//		_eventMap = _super->_eventMap;
//		return;
//	}
//	// set a flag so we know to delete the eventMap table
//	_freeEventMap = true;
//	// Allocate our new table.  It has to have as many entries as there are events.  NOTE: could save some space by keeping track of the maximum
//	// event that the class responds to and doing range checking.
//	int num = Event::NumEventCommands();
//	_eventMap = new (TAG_SYSTEM) eventCallback_t[num];
//	memset(_eventMap, 0, sizeof(eventCallback_t) * num);
//	_eventCallbackMemory += sizeof( eventCallback_t) * num;
//
//	// allocate temporary memory for flags so that the subclass's event callbacks override the superclass's event callback
//	bool *set = new (TAG_SYSTEM) bool[num];
//	memset(set, 0, sizeof(bool) * num);
//
//	// go through the inheritence order and copies the event callback function into a list indexed by the event number.  This allows fast lookups of
//	// event functions.
//	for (c = this; c != nullptr; c = c->_super)
//	{
//		EventFunc<Class> *def = c->_eventCallbacks;
//		if (!def)
//			continue;
//		// go through each entry until we hit the NULL terminator
//		for (int i = 0; def[i].event != nullptr; i++)
//		{
//			int ev = def[i].event->GetEventNum();
//			if (set[ev])
//				continue;
//			set[ev] = true;
//			_eventMap[ev] = def[i].function;
//		}
//	}
//	delete[] set;
//}
//
///// <summary>
///// Should only be called when DLL or EXE is being shutdown. Although it cleans up any allocated memory, it doesn't bother to remove itself 
///// from the class list since the program is shutting down.
///// </summary>
//void Type::Shutdown()
//{
//	// free up the memory used for event lookups
//	if (_eventMap)
//	{
//		if (_freeEventMap)
//			delete[] _eventMap;
//		_eventMap = nullptr;
//	}
//	_typeNum = 0;
//	_lastChild = 0;
//}
//
//const Event EV_Remove("<immediateremove>", nullptr);
//const Event EV_SafeRemove("remove", nullptr);
//
//ABSTRACT_DECLARATION(Class, nullptr)
//	EVENT(EV_Remove, Class::Event_Remove)
//	EVENT(EV_SafeRemove, Class::Event_SafeRemove)
//	END_CLASS
//
//	// alphabetical order
//	List<Type *, TAG_IDCLASS> Class::_types;
//// typenum order
//List<Type *, TAG_IDCLASS> Class::_typenums;
//
//bool Class::_initialized = false;
//int Class::_typeNumBits	= 0;
//int Class::_memused = 0;
//int Class::_numobjects = 0;
//
///// <summary>
///// CallSpawn
///// </summary>
//void Class::CallSpawn() { Type *type = GetType(); CallSpawnFunc(type); }
//
///// <summary>
///// CallSpawnFunc
///// </summary>
//classSpawnFunc_t Class::CallSpawnFunc(Type *cls)
//{
//	if (cls->_super)
//	{
//		classSpawnFunc_t func = CallSpawnFunc(cls->_super);
//		// don't call the same function twice in a row. this can happen when subclasses don't have their own spawn function.
//		if (func == cls->Spawn)
//			return func;
//	}
//	(this->*cls->Spawn)();
//	return cls->Spawn;
//}
//
///// <summary>
///// FindUninitializedMemory
///// </summary>
//void Class::FindUninitializedMemory()
//{
//#ifdef ID_DEBUG_UNINITIALIZED_MEMORY
//	unsigned long *ptr = ( ( unsigned long * )this ) - 1;
//	int size = *ptr;
//	assert( ( size & 3 ) == 0 );
//	size >>= 2;
//	for ( int i = 0; i < size; i++ ) {
//		if ( ptr[i] == 0xcdcdcdcd ) {
//			const char *varName = GetTypeVariableName( GetClassname(), i << 2 );
//			gameLocal.Warning( "type '%s' has uninitialized variable %s (offset %d)", GetClassname(), varName, i << 2 );
//		}
//	}
//#endif
//}
//
///// <summary>
///// Spawn
///// </summary>
//void Class::Spawn() { }
//
///// <summary>
///// Destructor for object.  Cancels any events that depend on this object.
///// </summary>
//Class::~Class() { EventManager::CancelEvents(this); }
//
///// <summary>
///// CreateInstance
///// </summary>
//Class *Class::CreateInstance(const char *name)
//{
//	const Type *type = Class::GetClass(name);
//	if (!type)
//		return nullptr;
//	Class *obj = type->CreateInstance();
//	return obj;
//}
//
///// <summary>
///// Should be called after all idTypeInfos are initialized, so must be called manually upon game code initialization.  Tells all the idTypeInfos to initialize
///// their event callback table for the associated class.  This should only be called once during the execution of the program or DLL.
///// </summary>
//void Class::Init()
//{
//	//gameLocal.Printf("Initializing class hierarchy\n");
//	if (_initialized)
//	{
//		//gameLocal.Printf("...already initialized\n");
//		return;
//	}
//	// init the event callback tables for all the classes
//	Type *c;
//	for (c = _typelist; c != nullptr; c = c->_next)
//		c->Init();
//	// number the types according to the class hierarchy so we can quickly determine if a class
//	// is a subclass of another
//	int num = 0;
//	for (c = _classHierarchy.GetNext(); c != nullptr; c = c->_node.GetNext(), num++)
//	{
//		c->_typeNum = num;
//		c->_lastChild += num;
//	}
//	// number of bits needed to send types over network
//	_typeNumBits = Math::BitsForInteger(num);
//
//	// create a list of the types so we can do quick lookups one list in alphabetical order, one in typenum order
//	_types.SetGranularity(1);
//	_types.SetNum(num);
//	_typenums.SetGranularity(1);
//	_typenums.SetNum(num);
//	num = 0;
//	for (c = _typelist; c != nullptr; c = c->_next, num++)
//	{
//		_types[num] = c;
//		_typenums[c->_typeNum] = c;
//	}
//	_initialized = true;
//	//gameLocal.Printf("...%i classes, %i bytes for event callbacks\n", _types.Num(), _eventCallbackMemory);
//}
//
///// <summary>
///// Shutdown
///// </summary>
//void Class::Shutdown()
//{
//	for (Type *c = _typelist; c != nullptr; c = c->_next)
//		c->Shutdown();
//	_types.Clear();
//	_typenums.Clear();
//	_initialized = false;
//}
//
///// <summary>
///// new
///// </summary>
//void * Class::operator new(size_t s)
//{
//	s += sizeof( int );
//	int *p = (int *)Mem_Alloc(s, TAG_IDCLASS);
//	*p = s;
//	_memused += s;
//	_objects++;
//	return p + 1;
//}
//
///// <summary>
///// delete
///// </summary>
//void Class::operator delete(void *ptr)
//{
//	int *p;
//
//	if (ptr)
//	{
//		int *p = ((int *)ptr) - 1;
//		_memused -= *p;
//		_objects--;
//		Mem_Free(p);
//	}
//}
//
///// <summary>
///// GetClass
///// Returns the idTypeInfo for the name of the class passed in.  This is a static function so it must be called as idClass::GetClass( classname )
///// </summary>
//Type *Class::GetClass(const char *name)
//{
//	if (!_initialized)
//	{
//		// Class::Init hasn't been called yet, so do a slow lookup
//		for (Type *c = _typelist; c != nullptr; c = c->_next)
//			if (!String::Cmp(c->_classname, name))
//				return c;
//	} else {
//		// do a binary search through the list of types
//		int min = 0;
//		int max = _types.Num() - 1;
//		while (min <= max)
//		{
//			int mid = (min + max) / 2;
//			Type *c = _types[mid];
//			int order = String::Cmp(c->_classname, name);
//			if (!order)
//				return c;
//			if (order > 0)
//				max = mid - 1;
//			else
//				min = mid + 1;
//		}
//	}
//	return nullptr;
//}
//
///// <summary>
///// GetType
///// </summary>
//Type *Class::GetType(const int typeNum)
//{
//	if (!_initialized)
//	{
//		for (Type *c = _typelist; c != nullptr; c = c->_next)
//			if (c->_typeNum == typeNum)
//				return c;
//	} else if (typeNum >= 0 && typeNum < _types.Num())
//		return _typenums[ typeNum ];
//	return nullptr;
//}
//
///// <summary>
///// GetClassname
///// Returns the text classname of the object.
///// </summary>
//const char *Class::GetClassname() const { Type *type = GetType(); return type->_classname; }
//
///// <summary>
///// GetSuperclass
///// Returns the text classname of the superclass.
///// </summary>
//const char *Class::GetSuperclass() const { Type *cls = GetType(); return cls->_superclass; }
//
///// <summary>
///// CancelEvents
///// </summary>
//void Class::CancelEvents(const Event *ev) { EventManager::CancelEvents(this, ev); }
//
///// <summary>
///// PostEventArgs
///// </summary>
//bool Class::PostEventArgs(const Event *ev, int time, int numargs, ...)
//{
//	//assert(ev);
//	if (!EventManager::_initialized)
//		return false;
//	Type *c = GetType();
//	// we don't respond to this event, so ignore it
//	if (!c->_eventMap[ev->GetEventNum()])
//		return false;
//	//// If this is an entity with skipReplication, we want to process the event normally even on clients.
//	//bool isReplicated = !(IsType(Entity::Type) && (static_cast<Entity *>(this))->fl.skipReplication);
//	//// we service events on the client to avoid any bad code filling up the event pool we don't want them processed usually, unless when the map is (re)loading.
//	//// we allow threads to run fine, though.
//	//if (common->IsClient() && isReplicated && (gameLocal.GameState() != GAMESTATE_STARTUP) && !IsType(Thread::Type))
//	//	return true;
//	//
//	va_list args;
//	va_start(args, numargs);
//	Event *event = EventManager::Alloc(ev, numargs, args);
//	va_end(args);
//	event->Schedule(this, c, time);
//	return true;
//}
//
///// <summary>
///// PostEventMS
///// </summary>
//bool Class::PostEventMS(const Event *ev, int time) { return PostEventArgs(ev, time, 0); }
//bool Class::PostEventMS(const Event *ev, int time, EventArg arg1) { return PostEventArgs(ev, time, 1, &arg1); }
//bool Class::PostEventMS(const Event *ev, int time, EventArg arg1, EventArg arg2) { return PostEventArgs(ev, time, 2, &arg1, &arg2); }
//bool Class::PostEventMS(const Event *ev, int time, EventArg arg1, EventArg arg2, EventArg arg3) { return PostEventArgs(ev, time, 3, &arg1, &arg2, &arg3); }
//bool Class::PostEventMS(const Event *ev, int time, EventArg arg1, EventArg arg2, EventArg arg3, EventArg arg4) { return PostEventArgs(ev, time, 4, &arg1, &arg2, &arg3, &arg4); }
//bool Class::PostEventMS(const Event *ev, int time, EventArg arg1, EventArg arg2, EventArg arg3, EventArg arg4, EventArg arg5) { return PostEventArgs(ev, time, 5, &arg1, &arg2, &arg3, &arg4, &arg5); }
//bool Class::PostEventMS(const Event *ev, int time, EventArg arg1, EventArg arg2, EventArg arg3, EventArg arg4, EventArg arg5, EventArg arg6) { return PostEventArgs( ev, time, 6, &arg1, &arg2, &arg3, &arg4, &arg5, &arg6); }
//bool Class::PostEventMS(const Event *ev, int time, EventArg arg1, EventArg arg2, EventArg arg3, EventArg arg4, EventArg arg5, EventArg arg6, EventArg arg7) { return PostEventArgs(ev, time, 7, &arg1, &arg2, &arg3, &arg4, &arg5, &arg6, &arg7); }
//bool Class::PostEventMS(const Event *ev, int time, EventArg arg1, EventArg arg2, EventArg arg3, EventArg arg4, EventArg arg5, EventArg arg6, EventArg arg7, EventArg arg8) { return PostEventArgs(ev, time, 8, &arg1, &arg2, &arg3, &arg4, &arg5, &arg6, &arg7, &arg8); }
//
///// <summary>
///// PostEventMS
///// </summary>
//bool Class::PostEventSec(const Event *ev, float time) { return PostEventArgs(ev, SEC2MS(time), 0); }
//bool Class::PostEventSec(const Event *ev, float time, EventArg arg1) { return PostEventArgs(ev, SEC2MS(time), 1, &arg1); }
//bool Class::PostEventSec(const Event *ev, float time, EventArg arg1, EventArg arg2) { return PostEventArgs(ev, SEC2MS(time), 2, &arg1, &arg2); }
//bool Class::PostEventSec(const Event *ev, float time, EventArg arg1, EventArg arg2, EventArg arg3) { return PostEventArgs(ev, SEC2MS(time), 3, &arg1, &arg2, &arg3); }
//bool Class::PostEventSec(const Event *ev, float time, EventArg arg1, EventArg arg2, EventArg arg3, EventArg arg4) { return PostEventArgs(ev, SEC2MS(time), 4, &arg1, &arg2, &arg3, &arg4); }
//bool Class::PostEventSec(const Event *ev, float time, EventArg arg1, EventArg arg2, EventArg arg3, EventArg arg4, EventArg arg5) { return PostEventArgs(ev, SEC2MS(time), 5, &arg1, &arg2, &arg3, &arg4, &arg5); }
//bool Class::PostEventSec(const Event *ev, float time, EventArg arg1, EventArg arg2, EventArg arg3, EventArg arg4, EventArg arg5, EventArg arg6) { return PostEventArgs(ev, SEC2MS(time), 6, &arg1, &arg2, &arg3, &arg4, &arg5, &arg6); }
//bool Class::PostEventSec(const Event *ev, float time, EventArg arg1, EventArg arg2, EventArg arg3, EventArg arg4, EventArg arg5, EventArg arg6, EventArg arg7) { return PostEventArgs(ev, SEC2MS(time), 7, &arg1, &arg2, &arg3, &arg4, &arg5, &arg6, &arg7); }
//bool Class::PostEventSec(const Event *ev, float time, EventArg arg1, EventArg arg2, EventArg arg3, EventArg arg4, EventArg arg5, EventArg arg6, EventArg arg7, EventArg arg8) { return PostEventArgs(ev, SEC2MS(time), 8, &arg1, &arg2, &arg3, &arg4, &arg5, &arg6, &arg7, &arg8); }
//
///// <summary>
///// ProcessEventArgs
///// </summary>
//bool Class::ProcessEventArgs(const Event *ev, int numargs, ...)
//{
//	assert(ev);
//	assert(EventManager::_initialized);
//	Type *c = GetType();
//	int num = ev->GetEventNum();
//	// we don't respond to this event, so ignore it
//	if (!c->_eventMap[num])
//		return false;
//	va_list args;
//	va_start(args, numargs);
//	int data[D_EVENT_MAXARGS];
//	EventManager::CopyArgs(ev, numargs, args, data);
//	va_end(args);
//	ProcessEventArgPtr(ev, data);
//	return true;
//}
//
///// <summary>
///// ProcessEvent
///// </summary>
//bool Class::ProcessEvent(const Event *ev) { return ProcessEventArgs(ev, 0); }
//bool Class::ProcessEvent(const Event *ev, EventArg arg1) { return ProcessEventArgs(ev, 1, &arg1); }
//bool Class::ProcessEvent(const Event *ev, EventArg arg1, EventArg arg2) { return ProcessEventArgs(ev, 2, &arg1, &arg2); }
//bool Class::ProcessEvent(const Event *ev, EventArg arg1, EventArg arg2, EventArg arg3) { return ProcessEventArgs(ev, 3, &arg1, &arg2, &arg3); }
//bool Class::ProcessEvent(const Event *ev, EventArg arg1, EventArg arg2, EventArg arg3, EventArg arg4) { return ProcessEventArgs(ev, 4, &arg1, &arg2, &arg3, &arg4); }
//bool Class::ProcessEvent(const Event *ev, EventArg arg1, EventArg arg2, EventArg arg3, EventArg arg4, EventArg arg5) { return ProcessEventArgs(ev, 5, &arg1, &arg2, &arg3, &arg4, &arg5); }
//bool Class::ProcessEvent(const Event *ev, EventArg arg1, EventArg arg2, EventArg arg3, EventArg arg4, EventArg arg5, EventArg arg6) { return ProcessEventArgs(ev, 6, &arg1, &arg2, &arg3, &arg4, &arg5, &arg6); }
//bool Class::ProcessEvent(const Event *ev, EventArg arg1, EventArg arg2, EventArg arg3, EventArg arg4, EventArg arg5, EventArg arg6, EventArg arg7) { return ProcessEventArgs(ev, 7, &arg1, &arg2, &arg3, &arg4, &arg5, &arg6, &arg7); }
//bool Class::ProcessEvent(const Event *ev, EventArg arg1, EventArg arg2, EventArg arg3, EventArg arg4, EventArg arg5, EventArg arg6, EventArg arg7, EventArg arg8) { return ProcessEventArgs(ev, 8, &arg1, &arg2, &arg3, &arg4, &arg5, &arg6, &arg7, &arg8); }
//
///// <summary>
///// ProcessEventArgPtr
///// </summary>
//bool Class::ProcessEventArgPtr(const Event *ev, int *data)
//{
//	assert(ev);
//	assert(EventManager::_initialized);
//	//SetTimeState ts;
//	//if (IsType(Entity::Type))
//	//	ts.PushState(((Entity*)this)->timeGroup);
//	//if (g_debugTriggers.GetBool() && (ev == &EV_Activate) && IsType(Entity::Type))
//	//{
//	//	const Entity *ent = *reinterpret_cast<Entity **>(data);
//	//	gameLocal.Printf( "%d: '%s' activated by '%s'\n", gameLocal.framenum, static_cast<Entity *>(this)->GetName(), (ent ? ent->GetName() : "NULL"));
//	//}
//	Type *c = GetType();
//	int num = ev->GetEventNum();
//	// we don't respond to this event, so ignore it
//	if (!c->_eventMap[num])
//		return false;
//	eventCallback_t callback = c->_eventMap[num];
//#if false && !CPU_EASYARGS
//	// on ppc architecture, floats are passed in a seperate set of registers the function prototypes must have matching float declaration
//	// http://developer.apple.com/documentation/DeveloperTools/Conceptual/MachORuntime/2rt_powerpc_abi/chapter_9_section_5.html
//	switch(ev->GetFormatspecIndex())
//	{
//	case 1 << D_EVENT_MAXARGS: (this->*callback)(); break;
//
//		// generated file - see CREATE_EVENT_CODE
//#include "Callbacks.cpp"
//	default: gameLocal.Warning("Invalid formatspec on event '%s'", ev->GetName()); break;
//	}
//#else
//	assert(D_EVENT_MAXARGS == 8);
//	switch(ev->GetNumArgs())
//	{
//	case 0 : ( this->*callback )(); break;
//	case 1 :
//		typedef void (Class::*eventCallback_1_t)(const int);
//		(this->*(eventCallback_1_t)callback)(data[0]);
//		break;
//	case 2 :
//		typedef void (Class::*eventCallback_2_t)(const int, const int);
//		(this->*(eventCallback_2_t)callback)(data[0], data[1]);
//		break;
//	case 3 :
//		typedef void (Class::*eventCallback_3_t)(const int, const int, const int);
//		(this->*(eventCallback_3_t)callback)(data[0], data[1], data[2]);
//		break;
//	case 4 :
//		typedef void (Class::*eventCallback_4_t)(const int, const int, const int, const int);
//		( this->*(eventCallback_4_t )callback)(data[0], data[1], data[2], data[3]);
//		break;
//	case 5 :
//		typedef void (Class::*eventCallback_5_t)(const int, const int, const int, const int, const int);
//		( this->*(eventCallback_5_t)callback)(data[0], data[1], data[2], data[3], data[4]);
//		break;
//	case 6 :
//		typedef void (Class::*eventCallback_6_t)(const int, const int, const int, const int, const int, const int);
//		(this->*(eventCallback_6_t)callback)(data[0], data[1], data[2], data[3], data[4], data[5]);
//		break;
//	case 7 :
//		typedef void (Class::*eventCallback_7_t)(const int, const int, const int, const int, const int, const int, const int);
//		(this->*(eventCallback_7_t)callback)(data[ 0 ], data[ 1 ], data[ 2 ], data[ 3 ], data[ 4 ], data[ 5 ], data[ 6 ]);
//		break;
//	case 8 :
//		typedef void (Class::*eventCallback_8_t)(const int, const int, const int, const int, const int, const int, const int, const int);
//		(this->*(eventCallback_8_t)callback)(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]);
//		break;
//	default: gameLocal.Warning( "Invalid formatspec on event '%s'", ev->GetName() ); break;
//	}
//#endif
//	return true;
//}
//
///// <summary>
///// Event_Remove
///// </summary>
//void Class::Event_Remove() { delete this; }
//
///// <summary>
///// Event_SafeRemove
///// Forces the remove to be done at a safe time
///// </summary>
//void Class::Event_SafeRemove() { PostEventMS(&EV_Remove, 0); }
