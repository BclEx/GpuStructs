using Xunit;
using System.Collections.Generic;
namespace System.Collections
{
    /// <summary>
    /// LinkedListNodeTests
    /// </summary>
    public class LinkedListNodeTests
    {
        [Fact]
        public void Constructor_Valid_NotNull()
        {
            var node = new LinkedListNode<object>(1);
            Assert.NotNull(node);
        }

        [Fact]
        public void List_SingleNode_Null()
        {
            var node = new LinkedListNode<object>(1);
            Assert.Null(node.List);
        }

        [Fact]
        public void Next_SingleNode_Null()
        {
            var node = new LinkedListNode<object>(1);
            Assert.Null(node.Next);
        }

        [Fact]
        public void Previous_SingleNode_Null()
        {
            var node = new LinkedListNode<object>(1);
            Assert.Null(node.Previous);
        }

        [Fact]
        public void Value_SingleNode_EqualsOne()
        {
            var node = new LinkedListNode<object>(1);
            Assert.Equal(1, node.Value);
        }
        
        [Fact]
        public void List_NodeFromList_SameAsList()
        {
            var linkedList = new LinkedList<object>(new object[] { 0, 1, 2 });
            var node = linkedList.Find(1);
            Assert.Same(linkedList, node.List);
        }

        [Fact]
        public void Next_NodeFromList_EqualsNull()
        {
            var linkedList = new LinkedList<object>(new object[] { 0, 1, 2 });
            var node = linkedList.Find(1);
            Assert.Equal(2, node.Next.Value);
        }

        [Fact]
        public void Previous_NodeFromList_EqualsNull()
        {
            var linkedList = new LinkedList<object>(new object[] { 0, 1, 2 });
            var node = linkedList.Find(1);
            Assert.Equal(0, node.Previous.Value);
        }
    }
}
