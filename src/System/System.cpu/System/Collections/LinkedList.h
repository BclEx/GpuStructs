#ifndef __SYSTEM_LINKEDLIST_H__
#define __SYSTEM_LINKEDLIST_H__
namespace System { namespace Collections {

	/// <summary>
	/// Circular linked list template
	/// </summary>
	template<class T>
	class LinkedList
	{
	public:
		LinkedList();
		~LinkedList();

		bool IsListEmpty() const;
		bool InList() const;
		int Num() const;
		void Clear();

		void InsertBefore(LinkedList &node);
		void InsertAfter(LinkedList &node);
		void AddToEnd(LinkedList &node);
		void AddToFront(LinkedList &node);

		void Remove();

		T *Next() const;
		T *Prev() const;

		T *Owner() const;
		void SetOwner(T *object);

		LinkedList *ListHead() const;
		LinkedList *NextNode() const;
		LinkedList *PrevNode() const;

	private:
		LinkedList *_head;
		LinkedList *_next;
		LinkedList *_prev;
		T * _owner;
	};

	/// <summary>
	/// Node is initialized to be the head of an empty list
	/// </summary>
	template<class T>
	LinkedList<T>::LinkedList()
	{
		_owner = nullptr;
		_head = this;	
		_next = this;
		_prev = this;
	}

	/// <summary>
	/// Removes the node from the list, or if it's the head of a list, removes all the nodes from the list.
	/// </summary>
	template<class T>
	LinkedList<T>::~LinkedList() { Clear(); }

	/// <summary>
	/// Returns true if the list is empty.
	/// </summary>
	template<class T>
	bool LinkedList<T>::IsListEmpty() const { return _head->_next == _head; }

	/// <summary>
	/// Returns true if the node is in a list.  If called on the head of a list, will always return false.
	/// </summary>
	template<class T>
	bool LinkedList<T>::InList() const { return _head != this; }

	/// <summary>
	/// Returns the number of nodes in the list.
	/// </summary>
	template<class T>
	int LinkedList<T>::Num() const
	{
		int num = 0;
		for (LinkedList<T> *node = _head->_next; node != _head; node = node->_next)
			num++;
		return num;
	}

	/// <summary>
	/// If node is the head of the list, clears the list.  Otherwise it just removes the node from the list.
	/// </summary>
	template<class T>
	void LinkedList<T>::Clear()
	{
		if (_head == this)
			while (_next != this)
				_next->Remove();
		else
			Remove();
	}

	/// <summary>
	/// Removes node from list
	/// </summary>
	template<class T>
	void LinkedList<T>::Remove()
	{
		_prev->_next = _next;
		_next->_prev = _prev;
		_next = this;
		_prev = this;
		_head = this;
	}

	/// <summary>
	/// Places the node before the existing node in the list.  If the existing node is the head, then the new node is placed at the end of the list.
	/// </summary>
	template<class T>
	void LinkedList<T>::InsertBefore(LinkedList &node)
	{
		Remove();
		_next = &node;
		_prev = _node._prev;
		_node._prev = this;
		_prev->_next = this;
		_head = _node._head;
	}

	/// <summary>
	/// Places the node after the existing node in the list.  If the existing node is the head, then the new node is placed at the beginning of the list.
	/// </summary>
	template<class T>
	void LinkedList<T>::InsertAfter(LinkedList &node)
	{
		Remove();
		_prev = &node;
		_next = node._next;
		_node._next	= this;
		_next->_prev = this;
		_head = node._head;
	}

	/// <summary>
	/// Adds node at the end of the list
	/// </summary>
	template<class T>
	void LinkedList<T>::AddToEnd(LinkedList &node) { InsertBefore(*node._head); }

	/// <summary>
	/// Adds node at the beginning of the list
	/// </summary>
	template<class T>
	void LinkedList<T>::AddToFront(LinkedList &node) { InsertAfter(*node._head); }

	/// <summary>
	/// Returns the head of the list.  If the node isn't in a list, it returns a pointer to itself.
	/// </summary>
	template<class T>
	LinkedList<T> *LinkedList<T>::ListHead() const { return _head; }

	/// <summary>
	/// Returns the next object in the list, or NULL if at the end.
	/// </summary>
	template<class T>
	T *LinkedList<T>::Next() const { return (!_next || _next == _head ? nullptr : _next->_owner); }

	/// <summary>
	/// Returns the previous object in the list, or NULL if at the beginning.
	/// </summary>
	template<class T>
	T *LinkedList<T>::Prev() const { return (!_prev ||  _prev == _head  ? nullptr : _prev->_owner); }

	/// <summary>
	/// Returns the next node in the list, or NULL if at the end.
	/// </summary>
	template<class T>
	LinkedList<T> *LinkedList<T>::NextNode() const { return (_next == _head ? nullptr : _next); }

	/// <summary>
	/// Returns the previous node in the list, or NULL if at the beginning.
	/// </summary>
	template<class T>
	LinkedList<T> *LinkedList<T>::PrevNode() const { return (_prev == _head ? nullptr : _prev); }

	/// <summary>
	/// Gets the object that is associated with this node.
	/// </summary>
	template<class T>
	T *LinkedList<T>::Owner() const { return _owner; }

	/// <summary>
	/// Sets the object that this node is associated with.
	/// </summary>
	template<class T>
	void LinkedList<T>::SetOwner(T *object) { _owner = object; }

}}
#endif /* __SYSTEM_LINKEDLIST_H__ */
