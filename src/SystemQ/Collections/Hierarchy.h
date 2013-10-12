#ifndef __SYSTEM_HIERARCHY_H__
#define __SYSTEM_HIERARCHY_H__
namespace Sys { namespace Collections {

	/// <summary>
	/// Hierarchy
	/// </summary>
	template<class T>
	class Hierarchy
	{
	public:
		Hierarchy();
		~Hierarchy();

		void SetOwner(T *object);
		T *Owner() const;
		void ParentTo(Hierarchy &node);
		void MakeSiblingAfter(Hierarchy &node);
		bool ParentedBy(const Hierarchy &node) const;
		void RemoveFromParent();
		void RemoveFromHierarchy();

		T * GetParent() const;		// parent of this node
		T * GetChild() const;		// first child of this node
		T * GetSibling() const;		// next node with the same parent
		T * GetPriorSibling() const; // previous node with the same parent
		T * GetNext() const;		// goes through all nodes of the hierarchy
		T * GetNextLeaf() const;	// goes through all leaf nodes of the hierarchy

	private:
		Hierarchy *_parent;
		Hierarchy *_sibling;
		Hierarchy *_child;
		T *_owner;

		Hierarchy<T> *GetPriorSiblingNode() const; // previous node with the same parent
	};

	/// <summary>
	/// Hierarchy
	/// </summary>
	template<class T>
	Hierarchy<T>::Hierarchy() {
		_owner = nullptr;
		_parent = nullptr;	
		_sibling = nullptr;
		_child = nullptr;
	}

	/// <summary>
	/// ~Hierarchy
	/// </summary>
	template<class T>
	Hierarchy<T>::~Hierarchy() { RemoveFromHierarchy(); }

	/// <summary>
	/// Gets the object that is associated with this node.
	/// </summary>
	template<class T>
	T *Hierarchy<T>::Owner() const { return _owner; }

	/// <summary>
	/// Sets the object that this node is associated with.
	/// </summary>
	template<class T>
	void Hierarchy<T>::SetOwner(T *object) { _owner = object; }

	/// <summary>
	/// ParentedBy
	/// </summary>
	template<class T>
	bool Hierarchy<T>::ParentedBy(const Hierarchy &node) const
	{
		if (parent == &node)
			return true;
		else if (_parent)
			return _parent->ParentedBy(node);
		return false;
	}

	/// <summary>
	/// Makes the given node the parent.
	/// </summary>
	template<class T>
	void Hierarchy<T>::ParentTo(Hierarchy &node)
	{
		RemoveFromParent();
		_parent = &node;
		_sibling = node._child;
		_node._child = this;
	}

	/// <summary>
	/// Makes the given node a sibling after the passed in node.
	/// </summary>
	template<class T>
	void Hierarchy<T>::MakeSiblingAfter(Hierarchy &node)
	{
		RemoveFromParent();
		_parent	= node._parent;
		_sibling = node._sibling;
		node._sibling = this;
	}

	/// <summary>
	/// RemoveFromParent
	/// </summary>
	template<class T>
	void Hierarchy<T>::RemoveFromParent()
	{
		if (parent)
		{
			Hierarchy<T> *prev = GetPriorSiblingNode();
			if (prev)
				prev->_sibling = _sibling;
			else
				parent->_child = _sibling;

		}
		_parent = nullptr;
		_sibling = nullptr;
	}

	/// <summary>
	/// Removes the node from the hierarchy and adds it's children to the parent.
	/// </summary>
	template<class T>
	void Hierarchy<T>::RemoveFromHierarchy()
	{
		Hierarchy<T> *parentNode = _parent;
		RemoveFromParent();
		if (parentNode)
			while (_child)
			{
				Hierarchy<T> *node = _child;
				node->RemoveFromParent();
				node->ParentTo(*parentNode);
			}
		else
			while (_child)
				_child->RemoveFromParent();
	}

	/// <summary>
	/// GetParent
	/// </summary>
	template<class T>
	T *Hierarchy<T>::GetParent() const { return (_parent ? _parent->_owner : nullptr); }

	/// <summary>
	/// GetChild
	/// </summary>
	template<class T>
	T *Hierarchy<T>::GetChild() const { return (_child ? _child->_owner : nullptr); }

	/// <summary>
	/// GetSibling
	/// </summary>
	template<class T>
	T *Hierarchy<T>::GetSibling() const { return (_sibling ? _sibling->_owner : nullptr); }

	/// <summary>
	/// Returns NULL if no parent, or if it is the first child.
	/// </summary>
	template<class T>
	Hierarchy<T> *Hierarchy<T>::GetPriorSiblingNode() const
	{
		if (!_parent || _parent->child == this)
			return nullptr;
		Hierarchy<T> *node = _parent->child;
		Hierarchy<T> *prev = nullptr;
		while (node != this && node != nullptr)
		{
			prev = node;
			node = node->_sibling;
		}
		if (node != this)
			System::Error("Hierarchy::GetPriorSibling: could not find node in parent's list of children");
		return prev;
	}

	/// <summary>
	/// Returns NULL if no parent, or if it is the first child.
	/// </summary>
	template<class T>
	T *Hierarchy<T>::GetPriorSibling() const
	{
		Hierarchy<T> *prior = GetPriorSiblingNode();
		return (prior ? prior->_owner : nullptr);
	}

	/// <summary>
	/// Goes through all nodes of the hierarchy.
	/// </summary>
	template<class T>
	T *Hierarchy<t>::GetNext() const
	{
		if (_child)
			return _child->_owner;
		const Hierarchy<T> *node = this;
		while (node && node->_sibling == nullptr)
			node = node->_parent;
		return (node ? node->_sibling->_owner : nullptr);
	}

	/// <summary>
	/// Goes through all leaf nodes of the hierarchy.
	/// </summary>
	template<class T>
	T *Hierarchy<T>::GetNextLeaf() const
	{
		const Hierarchy<T> *node;
		if (_child)
		{
			node = _child;
			while (node->_child)
				node = node->_child;
			return node->_owner;
		}
		node = this;
		while (node && node->_sibling == nullptr)
			node = node->parent;
		if (node)
		{
			node = node->_sibling;
			while (node->_child) 
				node = node->_child;
			return node->_owner;
		}
		return nullptr;
	}

}}
#endif /* __SYSTEM_HIERARCHY_H__ */
