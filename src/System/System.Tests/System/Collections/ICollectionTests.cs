using Xunit;
using System.Collections.Generic;
namespace System.Collections
{
    /// <summary>
    /// ICollectionTests
    /// </summary>
    public class ICollectionTests
    {
        [Fact]
        public void Add_TwoElementList_CountEqualsThree()
        {
            var collection = (ICollection<int>)new List<int>(new int[] { 0, 1 });
            collection.Add(2);
            Assert.Equal(3, collection.Count);
        }

        [Fact]
        public void Clear_TwoElementList_CountEqualsZero()
        {
            var collection = (ICollection<int>)new List<int>(new int[] { 0, 1 });
            collection.Clear();
            Assert.Equal(0, collection.Count);
        }

        [Fact]
        public void Contains_TwoElementList_EqualsTrue()
        {
            var collection = (ICollection<int>)new List<int>(new int[] { 0, 1 });
            Assert.True(collection.Contains(1));
        }

        [Fact]
        public void CopyTo_TwoElementList_CountEqualsTrue()
        {
            var collection = (ICollection<int>)new List<int>(new int[] { 0, 1 });
            int[] array = new int[2];
            collection.CopyTo(array, 0);
            Assert.Equal(2, collection.Count);
        }

        [Fact]
        public void Remove_TwoElementList_CountEqualsOne()
        {
            var collection = (ICollection<int>)new List<int>(new int[] { 0, 1 });
            collection.Remove(1);
            Assert.Equal(1, collection.Count);
        }

        [Fact]
        public void getCount_TwoElementList_EqualsTwo()
        {
            var collection = (ICollection<int>)new List<int>(new int[] { 0, 1 });
            Assert.Equal(2, collection.Count);
        }

        [Fact]
        public void getReadOnly_TwoElementList_EqualsFalse()
        {
            var collection = (ICollection<int>)new List<int>(new int[] { 0, 1 });
            Assert.False(collection.IsReadOnly);
        }
    }
}
