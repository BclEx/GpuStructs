using Xunit;
using System.Collections.Generic;
namespace System.Collections
{
    /// <summary>
    /// IListTests
    /// </summary>
    public class IListTests
    {
        [Fact]
        public void IndexOf_TwoElementList_EqualsOne()
        {
            var list = (IList<int>)new List<int>(new int[] { 0, 1 });
            Assert.Equal(1, list.IndexOf(1));
        }

        [Fact]
        public void Insert_TwoElementList_CountEqualThree()
        {
            var list = (IList<int>)new List<int>(new int[] { 0, 1 });
            list.Insert(0, 2);
            Assert.Equal(3, list.Count);
        }

        [Fact]
        public void RemoveAt_TwoElementList_CountEqualOne()
        {
            var list = (IList<int>)new List<int>(new int[] { 0, 1 });
            list.RemoveAt(1);
            Assert.Equal(1, list.Count);
        }

        [Fact]
        public void getIndex_TwoElementList_EqualOne()
        {
            var list = (IList<int>)new List<int>(new int[] { 0, 1 });
            Assert.Equal(1, list[1]);
        }

        [Fact]
        public void setIndex_TwoElementList_IndexEqualTwo()
        {
            var list = (IList<int>)new List<int>(new int[] { 0, 1 });
            list[1] = 2;
            Assert.Equal(2, list[1]);
        }
    }
}
