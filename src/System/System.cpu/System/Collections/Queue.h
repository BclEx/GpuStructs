#ifndef __SYSTEM_QUEUE_H__
#define __SYSTEM_QUEUE_H__
namespace System { namespace Collections {

	/// <summary>
	/// QueueTemplate
	/// </summary>
	template<class T, int NextOffset>
	class QueueTemplate
	{
	public:
		QueueTemplate();
		void Add(T *element);
		T *Get();

	private:
		T *_first;
		T *_last;
	};

#define QUEUE_NEXT_PTR(element)	(*((T**)(((byte*)element) + nextOffset)))

	/// <summary>
	/// QueueTemplate
	/// </summary>
	template<class T, int NextOffset>
	QueueTemplate<T, NextOffset>::QueueTemplate() { _first = _last = nullptr; }

	/// <summary>
	/// Add
	/// </summary>
	template<class T, int NextOffset>
	void QueueTemplate<T, NextOffset>::Add(T *element)
	{
		QUEUE_NEXT_PTR(element) = nullptr;
		if (_last)
			QUEUE_NEXT_PTR(last) = element;
		else 
			_first = element;
		_last = element;
	}

	/// <summary>
	/// Get
	/// </summary>
	template<class T, int NextOffset>
	T *QueueTemplate<T, NextOffset>::Get()
	{
		T *element = _first;
		if (element)
		{
			_first = QUEUE_NEXT_PTR(_first);
			if (_last == element)
				_last = nullptr;
			QUEUE_NEXT_PTR(element) = nullptr;
		}
		return element;
	}

	/// <summary>
	/// A node of a Queue
	/// </summary>
	template<typename T>
	class QueueNode
	{
	public:
		QueueNode() { _next = nullptr; }
		T *GetNext() const { return _next; }
		void SetNext(T *next) { this->_next = next; }

	private:
		T *_next;
	};

	/// <summary>
	/// A Queue, idQueue, is a template Container class implementing the Queue abstract data type.
	/// </summary>
	template<typename T, QueueNode<T> T::*NodePtr>
	class Queue
	{
	public:
		Queue();
		void Add(T *element);
		type *RemoveFirst();
		type *Peek() const;
		bool IsEmpty();

	private:
		T *_first;
		T *_last;
	};

	/// <summary>
	/// Queue
	/// </summary>
	template<typename T, QueueNode<T> T::*NodePtr>
	Queue<T, NodePtr>::Queue() { _first = _last = nullptr; }

	/// <summary>
	/// Add
	/// </summary>
	template<typename T, QueueNode<T> T::*NodePtr>
	void Queue<T, NodePtr>::Add(T *element)
	{
		(element->*NodePtr).SetNext(nullptr);
		if (_last) 
			(_last->*NodePtr).SetNext(element);
		else 
			_first = element;
		_last = element;
	}

	/// <summary>
	/// RemoveFirst
	/// </summary>
	template<typename T, QueueNode<T> T::*NodePtr>
	T *Queue<T, NodePtr>::RemoveFirst()
	{
		T *element = _first;
		if (element)
		{
			_first = (_first->*NodePtr).GetNext();
			if (_last == element)
				_last = nullptr;
			(element->*NodePtr).SetNext(nullptr);
		}
		return element;
	}

	/// <summary>
	/// Peek
	/// </summary>
	template<typename T, QueueNode<T> T::*NodePtr>
	T *Queue<T, NodePtr>::Peek() const { return _first; }

	/// <summary>
	/// IsEmpty
	/// </summary>
	template<typename T, QueueNode<T> T::*NodePtr>
	bool Queue<T, NodePtr>::IsEmpty() { return (_first == nullptr); }

	///// <summary>
	///// Test
	///// </summary>
	//template<typename T, QueueNode<T> T::*NodePtr>
	//void Queue<T, NodePtr>::Test()
	//{
	//	class MyType
	//	{
	//	public:
	//		QueueNode<MyType> queueNode;
	//	};
	//	Queue<MyType, &MyType::queueNode> myQueue;
	//	MyType *element = new (TAG_IDLIB) MyType;
	//	myQueue.Add(element);
	//	element = myQueue.RemoveFirst();
	//	delete element;
	//}

}}
#endif /* __SYSTEM_QUEUE_H__ */
