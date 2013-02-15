/// <summary>
/// Balanced Search Tree
/// </summary>
#ifndef __SYSTEM_BTREE_H__
#define __SYSTEM_BTREE_H__
namespace Sys { namespace Collections {

	/// <summary>
	/// BTreeNode
	/// </summary>
	template<class TObj, class TKey>
	class BTreeNode
	{
	public:
		TKey Key; // key used for sorting
		TObj *Object; // if != NULL pointer to object stored in leaf node
		BTreeNode *Parent; // parent node
		BTreeNode *Next; // next sibling
		BTreeNode *Prev; // prev sibling
		int NumChildren; // number of children
		BTreeNode *FirstChild; // first child
		BTreeNode *LastChild; // last child
	};

	/// <summary>
	/// BTreeNode
	/// </summary>
#define BTREE_CHECK
	template<class TObj, class TKey, int MaxChildrenPerNode>
	class BTree
	{
	public:
		BTree();
		~BTree();

		void Init();
		void Shutdown();

		BTreeNode<TObj, TKey> *Add(TObj *object, TKey key); // add an object to the tree
		void Remove(BTreeNode<TObj, TKey> *node); // remove an object node from the tree

		BTreeNode<TObj, TKey> *NodeFind(TKey key) const; // find an object using the given key
		BTreeNode<TObj, TKey> *NodeFindSmallestLargerEqual(TKey key) const; // find an object with the smallest key larger equal the given key
		BTreeNode<TObj, TKey> *NodeFindLargestSmallerEqual(TKey key) const; // find an object with the largest key smaller equal the given key

		TObj *Find(TKey key) const; // find an object using the given key
		TObj *FindSmallestLargerEqual(TKey key) const; // find an object with the smallest key larger equal the given key
		TObj *FindLargestSmallerEqual(TKey key) const; // find an object with the largest key smaller equal the given key

		BTreeNode<TObj, TKey> *GetRoot() const; // returns the root node of the tree
		int GetNodeCount() const; // returns the total number of nodes in the tree
		BTreeNode<TObj, TKey> *GetNext(BTreeNode<TObj, TKey> *node) const; // goes through all nodes of the tree
		BTreeNode<TObj, TKey> *GetNextLeaf(BTreeNode<TObj, TKey> *node) const; // goes through all leaf nodes of the tree

	private:
		BTreeNode<TObj, TKey> *_root;
		BlockAlloc<BTreeNode<TObj, TKey>, 128> _nodeAllocator;

		BTreeNode<TObj, TKey> *AllocNode();
		void FreeNode(BTreeNode<TObj, TKey> *node);
		void SplitNode(BTreeNode<TObj, TKey> *node);
		BTreeNode<TObj, TKey> *MergeNodes(BTreeNode<TObj, TKey> *node1, BTreeNode<TObj, TKey> *node2);

		void CheckTree_r(BTreeNode<TObj, TKey> *node, int &numNodes) const;
		void CheckTree() const;
	};

	/// <summary>
	/// BTree
	/// </summary>
	template<class TObj, class TKey, int MaxChildrenPerNode>
	inline BTree<TObj, TKey, MaxChildrenPerNode>::BTree() { assert(MaxChildrenPerNode >= 4); _root = nullptr; }
	template<class TObj, class TKey, int MaxChildrenPerNode>
	inline BTree<TObj, TKey, MaxChildrenPerNode>::~BTree() { Shutdown(); }

	/// <summary>
	/// Init
	/// </summary>
	template<class TObj, class TKey, int MaxChildrenPerNode>
	inline void BTree<TObj, TKey, MaxChildrenPerNode>::Init() { _root = AllocNode(); }

	/// <summary>
	/// Shutdown
	/// </summary>
	template<class TObj, class TKey, int MaxChildrenPerNode>
	inline void BTree<TObj, TKey, MaxChildrenPerNode>::Shutdown() { _nodeAllocator.Shutdown(); _root = nullptr; }

	/// <summary>
	/// Add
	/// </summary>
	template<class TObj, class TKey, int MaxChildrenPerNode>
	inline BTreeNode<TObj,TKey> *BTree<TObj, TKey, MaxChildrenPerNode>::Add(TObj *object, TKey key)
	{
		if (_root == nullptr)
			_root = AllocNode();
		BTreeNode<TObj, TKey> *newNode;
		if (_root->NumChildren >= MaxChildrenPerNode)
		{
			newNode = AllocNode();
			newNode->Key = _root->Key;
			newNode->FirstChild = _root;
			newNode->LastChild = _root;
			newNode->NumChildren = 1;
			_root->Parent = newNode;
			SplitNode(_root);
			_root = newNode;
		}
		newNode = AllocNode();
		newNode->Key = key;
		newNode->Object = object;
		BTreeNode<TObj, TKey> *child;
		for (BTreeNode<TObj, TKey> *node = _root; node->FirstChild != nullptr; node = child)
		{
			if (key > node->Key)
				node->Key = key;
			// find the first child with a key larger equal to the key of the new node
			for (child = node->FirstChild; child->Next; child = child->Next)
				if (key <= child->Key)
					break;
			if (child->Object)
			{
				if (key <= child->Key)
				{
					// insert new node before child
					if (child->Prev)
						child->Prev->Next = newNode;
					else
						node->FirstChild = newNode;
					newNode->Prev = child->Prev;
					newNode->Next = child;
					child->Prev = newNode;
				}
				else
				{
					// insert new node after child
					if (child->Next)
						child->Next->Prev = newNode;
					else
						node->LastChild = newNode;
					newNode->Prev = child;
					newNode->Next = child->Next;
					child->Next = newNode;
				}
				newNode->Parent = node;
				node->NumChildren++;
#ifdef BTREE_CHECK
				CheckTree();
#endif
				return newNode;
			}
			// make sure the child has room to store another node
			if (child->NumChildren >= MaxChildrenPerNode)
			{
				SplitNode(child);
				if (key <= child->Prev->Key)
					child = child->Prev;
			}
		}
		// we only end up here if the root node is empty
		newNode->Parent = _root;
		_root->Key = key;
		_root->FirstChild = newNode;
		_root->LastChild = newNode;
		_root->NumChildren++;
#ifdef BTREE_CHECK
		CheckTree();
#endif
		return newNode;
	}

	/// <summary>
	/// Remove
	/// </summary>
	template<class TObj, class TKey, int MaxChildrenPerNode>
	inline void BTree<TObj, TKey, MaxChildrenPerNode>::Remove(BTreeNode<TObj, TKey> *node)
	{
		assert(node->Object != nullptr);
		// unlink the node from it's parent
		if (node->Prev)
			node->Prev->Next = node->next;
		else
			node->Parent->FirstChild = node->Next;
		if (node->Next)
			node->Next->Prev = node->Prev;
		else
			node->Parent->LastChild = node->Prev;
		node->Parent->NumChildren--;
		// make sure there are no parent nodes with a single child
		BTreeNode<TObj, TKey> *parent;
		for (parent = node->Parent; parent != _root && parent->NumChildren <= 1; parent = parent->Parent)
		{

			if (parent->Next)
				parent = MergeNodes(parent, parent->Next);
			else if (parent->Prev)
				parent = MergeNodes(parent->Prev, parent);
			// a parent may not use a key higher than the key of it's last child
			if (parent->Key > parent->LastChild->Key)
				parent->Key = parent->LastChild->Key;
			if (parent->NumChildren > MaxChildrenPerNode)
			{
				SplitNode(parent);
				break;
			}
		}
		// a parent may not use a key higher than the key of it's last child
		for (; parent != nullptr && parent->LastChild != nullptr; parent = parent->Parent)
			if (parent->Key > parent->LastChild->Key)
				parent->Key = parent->LastChild->Key;
		// free the node
		FreeNode(node);
		// remove the root node if it has a single internal node as child
		if (_root->NumChildren == 1 && _root->FirstChild->Object == nullptr)
		{
			BTreeNode<TObj, TKey> *oldRoot = _root;
			_root->FirstChild->Parent = null;
			_root = _root->FirstChild;
			FreeNode(oldRoot);
		}
#ifdef BTREE_CHECK
		CheckTree();
#endif
	}

	/// <summary>
	/// NodeFind
	/// </summary>
	template<class TObj, class TKey, int MaxChildrenPerNode>
	inline BTreeNode<TObj, TKey> *BTree<TObj, TKey, MaxChildrenPerNode>::NodeFind(TKey key) const
	{
		if (_root == nullptr)
			return nullptr;
		for (BTreeNode<TObj, TKey> *node = _root->FirstChild; node != nullptr; node = node->FirstChild)
		{
			while (node->Next)
			{
				if (node->Key >= key)
					break;
				node = node->Next;
			}
			if (node->Object)
				return (node->Key == key ? node : nullptr);
		}
		return nullptr;
	}

	/// <summary>
	/// NodeFindSmallestLargerEqual
	/// </summary>
	template<class TObj, class TKey, int MaxChildrenPerNode>
	inline BTreeNode<TObj, TKey> *BTree<TObj, TKey, MaxChildrenPerNode>::NodeFindSmallestLargerEqual(TKey key) const
	{
		if (_root == nullptr)
			return nullptr;
		for (BTreeNode<TObj, TKey> *node = _root->FirstChild; node != nullptr; node = node->FirstChild)
		{
			while (node->Next)
			{
				if (node->Key >= key)
					break;
				node = node->Next;
			}
			if (node->Object) 
				return (node->Key >= key ? node : nullptr);
		}
		return nullptr;
	}

	/// <summary>
	/// NodeFindLargestSmallerEqual
	/// </summary>
	template<class TObj, class TKey, int MaxChildrenPerNode>
	inline BTreeNode<TObj, TKey> *BTree<TObj, TKey, MaxChildrenPerNode>::NodeFindLargestSmallerEqual(TKey key) const
	{
		if (_root == nullptr)
			return nullptr;
		BTreeNode<TObj, TKey> *smaller = nullptr;
		for (BTreeNode<TObj, TKey> *node = _root->FirstChild; node != nullptr; node = node->FirstChild)
		{
			while (node->Next)
			{
				if (node->Key >= key)
					break;
				smaller = node;
				node = node->Next;
			}
			if (node->Object)
			{
				if (node->Key <= key)
					return node;
				else if (smaller == null)
					return nullptr;
				else
				{
					node = smaller;
					if (node->Object)
						return node;
				}
			}
		}
		return nullptr;
	}

	/// <summary>
	/// Find
	/// </summary>
	template<class TObj, class TKey, int MaxChildrenPerNode>
	inline TObj *BTree<TObj, TKey, MaxChildrenPerNode>::Find(TKey key) const { BTreeNode<TObj, TKey> *node = NodeFind(key); return (node == nullptr ? nullptr : node->Object); }

	/// <summary>
	/// FindSmallestLargerEqual
	/// </summary>
	template<class TObj, class TKey, int MaxChildrenPerNode>
	inline TObj *BTree<TObj, TKey, MaxChildrenPerNode>::FindSmallestLargerEqual(TKey key) const { BTreeNode<TObj, TKey> *node = NodeFindSmallestLargerEqual(key); return (node == nullptr ? nullptr : node->Object); }

	/// <summary>
	/// FindLargestSmallerEqual
	/// </summary>
	template<class TObj, class TKey, int MaxChildrenPerNode>
	inline TObj *BTree<TObj, TKey, MaxChildrenPerNode>::FindLargestSmallerEqual(TKey key) const { BTreeNode<TObj, TKey> *node = NodeFindLargestSmallerEqual(key); return (node == nullptr ? nullptr : node->Object); }

	/// <summary>
	/// GetRoot
	/// </summary>
	template<class TObj, class TKey, int MaxChildrenPerNode>
	inline BTreeNode<TObj, TKey> *BTree<TObj, TKey, MaxChildrenPerNode>::GetRoot() const { return _root; }

	/// <summary>
	/// GetNodeCount
	/// </summary>
	template<class TObj, class TKey, int MaxChildrenPerNode>
	inline int BTree<TObj, TKey, MaxChildrenPerNode>::GetNodeCount() const { return _nodeAllocator.GetAllocCount(); }

	/// <summary>
	/// GetNext
	/// </summary>
	template<class TObj, class TKey, int MaxChildrenPerNode>
	inline BTreeNode<TObj, TKey> *BTree<TObj, TKey, MaxChildrenPerNode>::GetNext(BTreeNode<TObj, TKey> *node) const
	{
		if (node->FirstChild) 
			return node->FirstChild;
		while (node && node->Next == nullptr)
			node = node->Parent;
		return node;

	}

	/// <summary>
	/// GetNextLeaf
	/// </summary>
	template<class TObj, class TKey, int MaxChildrenPerNode>
	inline BTreeNode<TObj, TKey> *BTree<TObj, TKey, MaxChildrenPerNode>::GetNextLeaf(BTreeNode<TObj, TKey> *node) const
	{
		if (node->FirstChild)
		{
			while (node->FirstChild)
				node = node->FirstChild;
			return node;
		}
		while (node && node->Next == nullptr)
			node = node->Parent;
		if (node)
		{
			node = node->Next;
			while (node->FirstChild)
				node = node->FirstChild;
			return node;
		}
		return nullptr;
	}

	/// <summary>
	/// AllocNode
	/// </summary>
	template<class TObj, class TKey, int MaxChildrenPerNode>
	inline BTreeNode<TObj, TKey> *BTree<TObj, TKey, MaxChildrenPerNode>::AllocNode()
	{
		BTreeNode<TObj, TKey> *node = _nodeAllocator.Alloc();
		node->Key = 0;
		node->Parent = nullptr;
		node->Next = nullptr;
		node->Prev = nullptr;
		node->NumChildren = 0;
		node->FirstChild = nullptr;
		node->LastChild = nullptr;
		node->Object = nullptr;
		return node;
	}

	/// <summary>
	/// FreeNode
	/// </summary>
	template<class TObj, class TKey, int MaxChildrenPerNode>
	inline void BTree<TObj, TKey, MaxChildrenPerNode>::FreeNode(BTreeNode<TObj, TKey> *node) { _nodeAllocator.Free(node); }

	/// <summary>
	/// SplitNode
	/// </summary>
	template<class TObj, class TKey, int MaxChildrenPerNode>
	inline void BTree<TObj, TKey, MaxChildrenPerNode>::SplitNode(BTreeNode<TObj, TKey> *node)
	{
		// allocate a new node
		BTreeNode<TObj, TKey> *newNode = AllocNode();
		newNode->Parent = node->Parent;
		// divide the children over the two nodes
		BTreeNode<TObj, TKey> *child = node->FirstChild;
		child->Parent = newNode;
		for (int i = 3; i < node->NumChildren; i += 2)
		{
			child = child->Next;
			child->Parent = newNode;
		}
		newNode->Key = child->Key;
		newNode->NumChildren = node->NumChildren / 2;
		newNode->FirstChild = node->FirstChild;
		newNode->LastChild = child;
		node->NumChildren -= newNode->NumChildren;
		node->FirstChild = child->Next;
		child->Next->Prev = nullptr;
		child->Next = nullptr;
		// add the new child to the parent before the split node
		assert(node->Parent->NumChildren < MaxChildrenPerNode);
		if (node->Prev)
			node->Prev->Next = newNode;
		else
			node->Parent->FirstChild = newNode;
		newNode->Prev = node->Prev;
		newNode->Next = node;
		node->Prev = newNode;
		node->Parent->NumChildren++;
	}

	/// <summary>
	/// MergeNodes
	/// </summary>
	template<class TObj, class TKey, int MaxChildrenPerNode>
	inline BTreeNode<TObj, TKey> *BTree<TObj, TKey, MaxChildrenPerNode>::MergeNodes(BTreeNode<TObj, TKey> *node1, BTreeNode<TObj, TKey> *node2)
	{
		assert(node1->Parent == node2->Parent);
		assert(node1->Next == node2 && node2->Prev == node1);
		assert(node1->Object == NULL && node2->Object == nullptr);
		assert(node1->NumChildren >= 1 && node2->NumChildren >= 1);
		BTreeNode<TObj, TKey> *child;
		for (child = node1->FirstChild; child->Next; child = child->Next)
			child->Parent = node2;
		child->Parent = node2;
		child->Next = node2->FirstChild;
		node2->FirstChild->Prev = child;
		node2->FirstChild = node1->FirstChild;
		node2->NumChildren += node1->NumChildren;
		// unlink the first node from the parent
		if (node1->Prev)
			node1->Prev->Next = node2;
		else
			node1->Parent->FirstChild = node2;
		node2->Prev = node1->Prev;
		node2->Parent->NumChildren--;
		FreeNode(node1);
		return node2;
	}

	/// <summary>
	/// CheckTree_r
	/// </summary>
	template<class TObj, class TKey, int MaxChildrenPerNode>
	inline void BTree<TObj, TKey, MaxChildrenPerNode>::CheckTree_r(BTreeNode<TObj, TKey> *node, int &numNodes) const
	{
		numNodes++;
		// the root node may have zero children and leaf nodes always have zero children, all other nodes should have at least 2 and at most MaxChildrenPerNode children
		assert(node == _root || (node->Object != nullptr && node->NumChildren == 0) || (node->NumChildren >= 2 && node->NumChildren <= MaxChildrenPerNode));
		// the key of a node may never be larger than the key of it's last child
		assert(node->LastChild == nullptr || node->Key <= node->LastChild->Key);
		int numChildren = 0;
		for (BTreeNode<TObj, TKey> *child = node->FirstChild; child; child = child->Next)
		{
			numChildren++;
			// make sure the children are properly linked
			if (child->Prev == nullptr) { assert(node->FirstChild == child); }
			else { assert(child->Prev->Next == child); }
			if (child->next == nullptr) { assert(node->LastChild == child); }
			else { assert(child->Next->Prev == child); }
			// recurse down the tree
			CheckTree_r(child, numNodes);
		}
		// the number of children should equal the number of linked children
		assert(numChildren == node->NumChildren);
	}

	/// <summary>
	/// CheckTree
	/// </summary>
	template<class TObj, class TKey, int MaxChildrenPerNode>
	inline void BTree<TObj, TKey, MaxChildrenPerNode>::CheckTree() const
	{
		int numNodes = 0;
		CheckTree_r(_root, numNodes);
		// the number of nodes in the tree should equal the number of allocated nodes
		assert(numNodes == _nodeAllocator.GetAllocCount());
		// all the leaf nodes should be ordered
		BTreeNode<TObj, TKey> *lastNode = GetNextLeaf(GetRoot());
		if (lastNode)
			for (BTreeNode<TObj, TKey> *node = GetNextLeaf(lastNode); node; lastNode = node, node = GetNextLeaf(node))
				assert(lastNode->Key <= node->Key);
	}

}}
#endif /* __SYSTEM_BTREE_H__ */
