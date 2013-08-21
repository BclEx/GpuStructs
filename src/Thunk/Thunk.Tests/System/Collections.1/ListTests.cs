using Xunit;
using System.Collections.Generic;
namespace System.Collections
{
    /// <summary>
    /// ListTests
    /// </summary>
    public class ListTests
    {
        #region Enumerator

        /// <summary>
        /// EnumeratorTests
        /// </summary>
        public class EnumeratorTests
        {
            [Fact]
            public void Dispose_OneElementList_NotNull()
            {
                var list = new List<int>(new int[] { 0 });
                var enumerator = list.GetEnumerator();
                //
                enumerator.Dispose();
            }

            [Fact]
            public void MoveNext_OneElementList_EqualsTrue()
            {
                var list = new List<int>(new int[] { 0 });
                var enumerator = list.GetEnumerator();
                //
                Assert.True(enumerator.MoveNext());
            }

            [Fact]
            public void getCurrent_OneElementList_NotNull()
            {
                var list = new List<int>(new int[] { 0 });
                var enumerator = list.GetEnumerator();
                enumerator.MoveNext();
                //
                Assert.NotNull(enumerator.Current);
            }
        }

        #endregion

        [Fact]
        public void Constructor_Valid_NotNull()
        {
            var list = new List<int>();
            Assert.NotNull(list);
        }

        [Fact]
        public void Constructor_TwoElementList_NotNull()
        {
            var list = new List<int>(new int[] { 0, 1 });
            Assert.NotNull(list);
            Assert.Equal(2, list.Count);
        }

        [Fact]
        public void Constructor_Valid2_NotNull()
        {
            var list = new List<int>(5);
            Assert.NotNull(list);
        }

        [Fact]
        public void Add_Valid_NotNull()
        {
            var list = new List<int>(new int[] { 0, 1 });
            list.Add(2);
            Assert.Equal(3, list.Count);
        }

        [Fact]
        public void AddRange_Valid_NotNull()
        {
            var list = new List<int>(new int[] { 0, 1 });
            list.AddRange(new int[] { 2, 3 });
            Assert.Equal(4, list.Count);
        }

        [Fact]
        public void AsReadOnly_Valid_NotNull()
        {
            var list = new List<int>(new int[] { 0, 1 });
            Assert.NotNull(list.AsReadOnly());
        }

        [Fact]
        public void BinarySearch_Item_EqualsOne()
        {
            var list = new List<int>(new int[] { 0, 1 });
            Assert.Equal(1, list.BinarySearch(1));
        }

        [Fact]
        public void BinarySearch_ItemAndComparer_EqualsOne()
        {
            var list = new List<int>(new int[] { 0, 1 });
            Assert.Equal(1, list.BinarySearch(1, Comparer<int>.Default));
        }

        [Fact]
        public void BinarySearch_IndexCountItemAndComparer_EqualsOne()
        {
            var list = new List<int>(new int[] { 0, 1 });
            Assert.Equal(-1, list.BinarySearch(0, 0, 1, Comparer<int>.Default));
        }

        [Fact]
        public void Clear_Valid_NotNull()
        {
            var list = new List<int>(new int[] { 0, 1 });
            list.Clear();
            Assert.Equal(0, list.Count);
        }

        [Fact]
        public void Contains_Valid_NotNull()
        {
            var list = new List<int>(new int[] { 0, 1 });
            Assert.True(list.Contains(1));
        }

        [Fact]
        public void ConvertAll_Valid_NotNull()
        {
        }

        [Fact]
        public void CopyTo_TwoElementList_LastElementEqualsOne()
        {
            var list = new List<int>(new int[] { 0, 1 });
            int[] array = new int[2];
            //
            list.CopyTo(array);
            Assert.Equal(1, array[1]);
        }

        [Fact]
        public void CopyTo_TwoElementList2_LastElementEqualsOne()
        {
            var list = new List<int>(new int[] { 0, 1 });
            int[] array = new int[2];
            //
            list.CopyTo(array, 0);
            Assert.Equal(1, array[1]);
        }

        [Fact]
        public void CopyTo_TwoElementList3_LastElementEqualsOne()
        {
            var list = new List<int>(new int[] { 0, 1 });
            int[] array = new int[2];
            //
            list.CopyTo(0, array, 0, 2);
            Assert.Equal(1, array[1]);
        }

        [Fact]
        public void Exists_TwoElementList_EqualTrue()
        {
            var list = new List<int>(new int[] { 0, 1 });
            Assert.True(list.Contains(1));
        }

        [Fact]
        public void Find_TwoElementListAndPredicate_EqualOne()
        {
            var list = new List<int>(new int[] { 0, 1 });
            Assert.Equal(1, list.Find(c => c == 1));
        }

        [Fact]
        public void FindAll_TwoElementListAndPredicate_CountEqualOne()
        {
            var list = new List<int>(new int[] { 0, 1 });
            Assert.Equal(1, list.FindAll(c => c == 1).Count);
        }

        [Fact]
        public void FindIndex_TwoElementListAndPredicate_EqualOne()
        {
            var list = new List<int>(new int[] { 0, 1 });
            Assert.Equal(1, list.FindIndex(c => c == 1));
        }

        [Fact]
        public void FindIndex_TwoElementListAndPredicateStartZero_EqualOne()
        {
            var list = new List<int>(new int[] { 0, 1 });
            Assert.Equal(1, list.FindIndex(0, c => c == 1));
        }

        [Fact]
        public void FindIndex_TwoElementListAndPredicateStartZeroSearchTwo_EqualOne()
        {
            var list = new List<int>(new int[] { 0, 1 });
            Assert.Equal(1, list.FindIndex(0, 2, c => c == 1));
        }

        [Fact]
        public void FindLast_TwoElementList_EqualOne()
        {
            var list = new List<int>(new int[] { 0, 1 });
            Assert.Equal(1, list.FindLast(c => c == 1));
        }

        [Fact]
        public void FindLastIndex_TwoElementListAndPredicate_EqualOne()
        {
            var list = new List<int>(new int[] { 0, 1 });
            Assert.Equal(1, list.FindLastIndex(c => c == 1));
        }

        [Fact]
        public void FindLastIndex_TwoElementListAndPredicateStartZero_EqualOne()
        {
            var list = new List<int>(new int[] { 0, 1 });
            Assert.Equal(1, list.FindLastIndex(1, c => c == 1));
        }

        [Fact]
        public void FindLastIndex_TwoElementListAndPredicateStartZeroSearchTwo_EqualOne()
        {
            var list = new List<int>(new int[] { 0, 1 });
            Assert.Equal(1, list.FindLastIndex(1, 2, c => c == 1));
        }

        [Fact]
        public void ForEach_TwoElementList_VariableEqualOne()
        {
            var list = new List<int>(new int[] { 0, 1 });
            //
            int variable = 0;
            list.ForEach(c => variable += c);
            Assert.Equal(1, variable);
        }

        [Fact]
        public void GetEnumerator_TwoElementList_NotNull()
        {
            var list = new List<int>(new int[] { 0, 1 });
            Assert.NotNull(list.GetEnumerator());
        }

        [Fact]
        public void GetRange_TwoElementList_NewListCountEqualsTwo()
        {
            var list = new List<int>(new int[] { 0, 1 });
            //
            var newList = list.GetRange(0, 2);
            Assert.Equal(2, newList.Count);
        }

        [Fact]
        public void IndexOf_TwoElementList_EqualsOne()
        {
            var list = new List<int>(new int[] { 0, 1 });
            Assert.Equal(1, list.IndexOf(1));
        }

        [Fact]
        public void IndexOf_TwoElementListStartAtZero_EqualsOne()
        {
            var list = new List<int>(new int[] { 0, 1 });
            Assert.Equal(1, list.IndexOf(1, 0));
        }

        [Fact]
        public void IndexOf_TwoElementListStartAtZeroForTwo_EqualsOne()
        {
            var list = new List<int>(new int[] { 0, 1 });
            Assert.Equal(1, list.IndexOf(1, 0, 2));
        }

        [Fact]
        public void Insert_TwoElementList_EqualsOne()
        {
            var list = new List<int>(new int[] { 0, 1 });
            //
            list.Insert(0, 2);
            Assert.Equal(2, list[0]);
        }

        [Fact]
        public void InsertRange_TwoElementList_EqualsOne()
        {
            var list = new List<int>(new int[] { 0, 1 });
            //
            list.InsertRange(0, new int[] { 2, 3 });
            Assert.Equal(2, list[0]);
        }

        [Fact]
        public void LastIndexOf_TwoElementList_EqualsOne()
        {
            var list = new List<int>(new int[] { 0, 1 });
            Assert.Equal(1, list.LastIndexOf(1));
        }

        [Fact]
        public void LastIndexOf_TwoElementListStartAtZero_EqualsOne()
        {
            var list = new List<int>(new int[] { 0, 1 });
            Assert.Equal(1, list.LastIndexOf(1, 1));
        }

        [Fact]
        public void LastIndexOf_TwoElementListStartAtZeroForTwo_EqualsOne()
        {
            var list = new List<int>(new int[] { 0, 1 });
            Assert.Equal(1, list.LastIndexOf(1, 1, 2));
        }

        [Fact]
        public void Remove_TwoElementList_CountEqualsOne()
        {
            var list = new List<int>(new int[] { 0, 1 });
            //
            list.Remove(1);
            Assert.Equal(1, list.Count);
        }

        [Fact]
        public void RemoveAll_TwoElementList_CountEqualsOne()
        {
            var list = new List<int>(new int[] { 0, 1 });
            //
            list.RemoveAll(c => true);
            Assert.Equal(0, list.Count);
        }

        [Fact]
        public void RemoveAt_TwoElementList_FirstElementEqualsOne()
        {
            var list = new List<int>(new int[] { 0, 1 });
            //
            list.RemoveAt(0);
            Assert.Equal(1, list[0]);
        }

        [Fact]
        public void RemoveRange_TwoElementList_CountEqualsOne()
        {
            var list = new List<int>(new int[] { 0, 1 });
            //
            list.RemoveRange(0, 1);
            Assert.Equal(1, list.Count);
        }

        [Fact]
        public void Reverse_TwoElementList_FirstElementEqualsOne()
        {
            var list = new List<int>(new int[] { 0, 1 });
            list.Reverse();
            Assert.Equal(1, list[0]);
        }

        [Fact]
        public void Reverse_TwoElementListStartZeroForTwoElements_FirstElementEqualsOne()
        {
            var list = new List<int>(new int[] { 0, 1 });
            list.Reverse(0, 2);
            Assert.Equal(1, list[0]);
        }

        [Fact]
        public void Sort_TwoElementList_FirstElementEqualsZero()
        {
            var list = new List<int>(new int[] { 0, 1 });
            list.Sort();
            Assert.Equal(0, list[0]);
        }

        [Fact]
        public void Sort_TwoElementListAndDefaultComparer_FirstElementEqualsZero()
        {
            var list = new List<int>(new int[] { 0, 1 });
            list.Sort(Comparer<int>.Default);
            Assert.Equal(0, list[0]);
        }

        [Fact]
        public void Sort_TwoElementListOffsetsAndDefaultComparer_FirstElementEqualsZero()
        {
            var list = new List<int>(new int[] { 0, 1 });
            list.Sort(0, 2, Comparer<int>.Default);
            Assert.Equal(0, list[0]);
        }

        [Fact]
        public void ToArray_TwoElementList_FirstElementEqualsZero()
        {
            var list = new List<int>(new int[] { 0, 1 });
            var array = list.ToArray();
            Assert.Equal(0, array[0]);
        }

        [Fact]
        public void TrimExcess_TwoElementList_CapacityEqualsTwo()
        {
            var list = new List<int>(new int[] { 0, 1 });
            list.TrimExcess();
            Assert.Equal(2, list.Capacity);
        }

        [Fact]
        public void TrueForAll_TwoElementList_EqualsTrue()
        {
            var list = new List<int>(new int[] { 0, 1 });
            Assert.True(list.TrueForAll(c => true));
        }

        [Fact]
        public void getCapacity_TwoElementList_EqualsTwo()
        {
            var list = new List<int>(new int[] { 0, 1 });
            Assert.Equal(2, list.Capacity);
        }

        [Fact]
        public void setCapacity_TwoElementList_CapacityEqualsFive()
        {
            var list = new List<int>(new int[] { 0, 1 });
            list.Capacity = 5;
            Assert.Equal(5, list.Capacity);
        }

        [Fact]
        public void getCount_TwoElementList_EqualsTwo()
        {
            var list = new List<int>(new int[] { 0, 1 });
            Assert.Equal(2, list.Count);
        }

        [Fact]
        public void getItem_TwoElementList_FirstElementEqualsZero()
        {
            var list = new List<int>(new int[] { 0, 1 });
            Assert.Equal(0, list[0]);
        }

        [Fact]
        public void setItem_TwoElementList_FirstElementEqualsTwo()
        {
            var list = new List<int>(new int[] { 0, 1 });
            list[0] = 2;
            Assert.Equal(2, list[0]);
        }
    }
}
