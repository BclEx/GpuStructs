using Xunit;
using System.Collections.Generic;
namespace System.Collections.Dictionary
{
    /// <summary>
    /// KeyCollectionTests
    /// </summary>
    public class KeyCollectionTests
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
                var keyCollection = new Dictionary<int, int>.KeyCollection(dictionary);
                var enumerator = keyCollection.GetEnumerator();
                //
                enumerator.Dispose();
            }

            [Fact]
            public void MoveNext_OneElementDictionary_EqualsTrue()
            {
                var dictionary = new Dictionary<int, int>();
                dictionary.Add(1, 1);
                var keyCollection = new Dictionary<int, int>.KeyCollection(dictionary);
                var enumerator = keyCollection.GetEnumerator();
                //
                Assert.True(enumerator.MoveNext());
            }

            [Fact]
            public void getCurrent_OneElementDictionary_NotNull()
            {
                var dictionary = new Dictionary<int, int>();
                dictionary.Add(1, 1);
                var keyCollection = new Dictionary<int, int>.KeyCollection(dictionary);
                var enumerator = dictionary.GetEnumerator();
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
            var keyCollection = new Dictionary<int, int>.KeyCollection(dictionary);
            Assert.NotNull(keyCollection);
        }

        [Fact]
        public void CopyTo_KeyCollectionWithOneElementDictionary_FirstElementEqualsOne()
        {
            var dictionary = new Dictionary<int, int>();
            dictionary.Add(1, 1);
            var keyCollection = new Dictionary<int, int>.KeyCollection(dictionary);
            //
            int[] keys = new int[1];
            keyCollection.CopyTo(keys, 0);
            Assert.Equal(1, keys[0]);
        }

        [Fact]
        public void GetEnumerator_KeyCollectionWithOneElementDictionary_NotNull()
        {
            var dictionary = new Dictionary<int, int>();
            dictionary.Add(1, 1);
            var keyCollection = new Dictionary<int, int>.KeyCollection(dictionary);
            //
            Assert.NotNull(keyCollection.GetEnumerator());
        }

        [Fact]
        public void getCount_KeyCollectionWithOneElementDictionary_CountEqualsOne()
        {
            var dictionary = new Dictionary<int, int>();
            dictionary.Add(1, 1);
            var keyCollection = new Dictionary<int, int>.KeyCollection(dictionary);
            //
            Assert.Equal(1, keyCollection.Count);
        }
    }
}
