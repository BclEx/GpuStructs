using Xunit;
using System.Collections.Generic;
namespace System.Collections.Dictionary
{
    /// <summary>
    /// ValueCollectionTests
    /// </summary>
    public class ValueCollectionTests
    {
        #region Enumerator

        /// <summary>
        /// EnumeratorTests
        /// </summary>
        public class EnumeratorTests
        {
            [Fact]
            public void Dispose_OneElementDictionary_NotNull()
            {
                var dictionary = new Dictionary<int, int>();
                dictionary.Add(1, 1);
                var valueCollection = new Dictionary<int, int>.ValueCollection(dictionary);
                var enumerator = valueCollection.GetEnumerator();
                //
                enumerator.Dispose();
            }

            [Fact]
            public void MoveNext_OneElementDictionary_EqualsTrue()
            {
                var dictionary = new Dictionary<int, int>();
                dictionary.Add(1, 1);
                var valueCollection = new Dictionary<int, int>.ValueCollection(dictionary);
                var enumerator = valueCollection.GetEnumerator();
                //
                Assert.True(enumerator.MoveNext());
            }

            [Fact]
            public void getCurrent_OneElementDictionary_NotNull()
            {
                var dictionary = new Dictionary<int, int>();
                dictionary.Add(1, 1);
                var valueCollection = new Dictionary<int, int>.ValueCollection(dictionary);
                var enumerator = valueCollection.GetEnumerator();
                enumerator.MoveNext();
                //
                Assert.NotNull(enumerator.Current);
            }
        }

        #endregion

        [Fact]
        public void Constructor_Valid_NotNull()
        {
            var dictionary = new Dictionary<int, int>();
            var valueCollection = new Dictionary<int, int>.ValueCollection(dictionary);
            Assert.NotNull(valueCollection);
        }

        [Fact]
        public void CopyTo_KeyCollectionWithOneElementDictionary_FirstElementEqualsOne()
        {
            var dictionary = new Dictionary<int, int>();
            dictionary.Add(1, 1);
            var valueCollection = new Dictionary<int, int>.ValueCollection(dictionary);
            //
            int[] values = new int[1];
            valueCollection.CopyTo(values, 0);
            Assert.Equal(1, values[0]);
        }

        [Fact]
        public void GetEnumerator_KeyCollectionWithOneElementDictionary_NotNull()
        {
            var dictionary = new Dictionary<int, int>();
            dictionary.Add(1, 1);
            var valueCollection = new Dictionary<int, int>.ValueCollection(dictionary);
            //
            Assert.NotNull(valueCollection.GetEnumerator());
        }

        [Fact]
        public void getCount_KeyCollectionWithOneElementDictionary_CountEqualsOne()
        {
            var dictionary = new Dictionary<int, int>();
            dictionary.Add(1, 1);
            var valueCollection = new Dictionary<int, int>.ValueCollection(dictionary);
            //
            Assert.Equal(1, valueCollection.Count);
        }
    }
}
