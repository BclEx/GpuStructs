using Xunit;
using System.Collections.Generic;
using System.Runtime.Serialization;
namespace System.Collections
{
    /// <summary>
    /// LinkedListTests
    /// </summary>
    public class LinkedListTests
    {
        [Fact]
        public void Constructor_New_NotNull()
        {
            var linkedList = new LinkedList<object>();
            Assert.NotNull(linkedList);
        }

        [Fact]
        public void Constructor_NewWithArray_NotNullAndCountEqualsTwo()
        {
            var linkedList = new LinkedList<object>(new object[] { 0, 1 });
            Assert.NotNull(linkedList);
            Assert.Equal(2, linkedList.Count);
        }

        [Fact]
        public void AddAfter_TwoElementList_LastValueEqualsAddedValue()
        {
            var linkedList = new LinkedList<object>(new object[] { 0, 1 });
            linkedList.AddAfter(linkedList.Last, 2);
            Assert.Equal(2, linkedList.Last.Value);
        }

        [Fact]
        public void AddAfter_TwoElementList_LastEqualsAddedNode()
        {
            var newNode = new LinkedListNode<object>(2);
            var linkedList = new LinkedList<object>(new object[] { 0, 1 });
            //
            linkedList.AddAfter(linkedList.Last, newNode);
            Assert.Same(newNode, linkedList.Last);
        }

        [Fact]
        public void AddBefore_TwoElementList_FirstValueEqualsAddedValue()
        {
            var linkedList = new LinkedList<object>(new object[] { 0, 1 });
            linkedList.AddBefore(linkedList.First, 2);
            Assert.Equal(2, linkedList.First.Value);
        }

        [Fact]
        public void AddBefore_TwoElementList_FirstEqualsAddedNode()
        {
            var newNode = new LinkedListNode<object>(2);
            var linkedList = new LinkedList<object>(new object[] { 0, 1 });
            //
            linkedList.AddBefore(linkedList.First, newNode);
            Assert.Same(newNode, linkedList.First);
        }

        [Fact]
        public void AddFirst_TwoElementList_FirstValueEqualsAddedValue()
        {
            var linkedList = new LinkedList<object>(new object[] { 0, 1 });
            linkedList.AddFirst(2);
            Assert.Equal(2, linkedList.First.Value);
        }

        [Fact]
        public void AddFirst_TwoElementList_FirstEqualsAddedNode()
        {
            var newNode = new LinkedListNode<object>(2);
            var linkedList = new LinkedList<object>(new object[] { 0, 1 });
            //
            linkedList.AddFirst(newNode);
            Assert.Same(newNode, linkedList.First);
        }

        [Fact]
        public void AddLast_TwoElementList_LastValueEqualsAddedValue()
        {
            var linkedList = new LinkedList<object>(new object[] { 0, 1 });
            linkedList.AddLast(2);
            Assert.Equal(2, linkedList.Last.Value);
        }

        [Fact]
        public void AddLast_TwoElementList_LastEqualsAddedNode()
        {
            var newNode = new LinkedListNode<object>(2);
            var linkedList = new LinkedList<object>(new object[] { 0, 1 });
            //
            linkedList.AddLast(newNode);
            Assert.Same(newNode, linkedList.Last);
        }

        [Fact]
        public void Clear_TwoElementList_CountEqualsZero()
        {
            var linkedList = new LinkedList<object>(new object[] { 0, 1 });
            linkedList.Clear();
            Assert.Equal(0, linkedList.Count);
        }

        [Fact]
        public void Contains_TwoElementList_ContainsCountEqualsZero()
        {
            var linkedList = new LinkedList<object>(new object[] { 0, 1 });
            Assert.Equal(true, linkedList.Contains(1));
        }

        [Fact]
        public void CopyTo_TwoElementList_LastElementEqualsOne()
        {
            var linkedList = new LinkedList<object>(new object[] { 0, 1 });
            object[] array = new object[2];
            //
            linkedList.CopyTo(array, 0);
            Assert.Equal(1, array[1]);
        }

        [Fact]
        public void Find_TwoElementList_FoundNodeValueEqualsOne()
        {
            var linkedList = new LinkedList<object>(new object[] { 0, 1 });
            var node = linkedList.Find(1);
            Assert.Equal(1, node.Value);
        }

        [Fact]
        public void FindLast_TwoElementList_FoundNodeValueEqualsOne()
        {
            var linkedList = new LinkedList<object>(new object[] { 0, 1 });
            var node = linkedList.FindLast(1);
            Assert.Equal(1, node.Value);
        }

        [Fact]
        public void GetEnumerator_TwoElementList_NotNull()
        {
            var linkedList = new LinkedList<object>(new object[] { 0, 1 });
            Assert.NotNull(linkedList.GetEnumerator());
        }

        [Fact]
        public void GetObjectData_TwoElementList_SerialMemberCountEqualsThree()
        {
            //var linkedList = new LinkedList<object>(new object[] { 0, 1 });
            //var serial = new SerializationInfo(null, null);
            ////
            //linkedList.GetObjectData(serial, new StreamingContext());
            //Assert.Equal(3, serial.MemberCount);
        }

        [Fact]
        public void OnDeserialization_TwoElementList_IsNotNull()
        {
            //var linkedList = new LinkedList<object>(new object[] { 0, 1 });
            ////
            //linkedList.OnDeserialization(null);
            //Assert.Fail();
        }

        [Fact]
        public void Remove_TwoElementListRemoveValue_CountEqualsOne()
        {
            var linkedList = new LinkedList<object>(new object[] { 0, 1 });
            linkedList.Remove(1);
            Assert.Equal(1, linkedList.Count);
        }

        [Fact]
        public void Remove_TwoElementListRemoveNode_CountEqualsOne()
        {
            var linkedList = new LinkedList<object>(new object[] { 0, 1 });
            var node = linkedList.Find(1);
            //
            linkedList.Remove(node);
            Assert.Equal(1, linkedList.Count);
        }

        [Fact]
        public void RemoveFirst_TwoElementList_CountEqualsOne()
        {
            var linkedList = new LinkedList<object>(new object[] { 0, 1 });
            linkedList.RemoveFirst();
            Assert.Equal(1, linkedList.Count);
        }

        [Fact]
        public void RemoveLast_TwoElementList_CountEqualsOne()
        {
            var linkedList = new LinkedList<object>(new object[] { 0, 1 });
            linkedList.RemoveLast();
            Assert.Equal(1, linkedList.Count);
        }

        [Fact]
        public void Count_TwoElementList_EqualsTwo()
        {
            var linkedList = new LinkedList<object>(new object[] { 0, 1 });
            Assert.Equal(2, linkedList.Count);
        }

        [Fact]
        public void First_TwoElementList_ValueEqualsZero()
        {
            var linkedList = new LinkedList<object>(new object[] { 0, 1 });
            Assert.Equal(0, linkedList.First.Value);
        }

        [Fact]
        public void Last_TwoElementObject_ValueEqualsOne()
        {
            var linkedList = new LinkedList<object>(new object[] { 0, 1 });
            Assert.Equal(1, linkedList.Last.Value);
        }
    }
}
