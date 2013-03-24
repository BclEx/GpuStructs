using Xunit;
using System.Collections.Generic;
namespace System.Collections
{
    /// <summary>
    /// DictionaryTests
    /// </summary>
    public class DictionaryTests
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
                var enumerator = dictionary.GetEnumerator();
                //
                enumerator.Dispose();
            }

            [Fact]
            public void MoveNext_OneElementDictionary_EqualsTrue()
            {
                var dictionary = new Dictionary<int, int>();
                dictionary.Add(1, 1);
                var enumerator = dictionary.GetEnumerator();
                //
                Assert.True(enumerator.MoveNext());
            }

            [Fact]
            public void getCurrent_OneElementDictionary_NotNull()
            {
                var dictionary = new Dictionary<int, int>();
                dictionary.Add(1, 1);
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
            Assert.NotNull(dictionary);
        }

        [Fact]
        public void Constructor_OneElementDictionary_ContainsKeyEqualsTrue()
        {
            var startDictionary = new Dictionary<int, int>();
            startDictionary.Add(1, 1);
            var dictionary = new Dictionary<int, int>(startDictionary);
            //
            Assert.NotNull(dictionary);
            Assert.True(dictionary.ContainsKey(1));
        }

        [Fact]
        public void Constructor_DefaultEqualityComparer_NotNull()
        {
            var dictionary = new Dictionary<int, int>(EqualityComparer<int>.Default);
            Assert.NotNull(dictionary);
        }

        [Fact]
        public void Constructor_CapacityOfOne_NotNull()
        {
            var dictionary = new Dictionary<int, int>(1);
            Assert.NotNull(dictionary);
        }

        [Fact]
        public void Constructor_OneElementDictionaryAndDefaultEqualityComparer_ContainsKeyEqualsTrue()
        {
            var startDictionary = new Dictionary<int, int>();
            startDictionary.Add(1, 1);
            var dictionary = new Dictionary<int, int>(startDictionary, EqualityComparer<int>.Default);
            //
            Assert.NotNull(dictionary);
            Assert.True(dictionary.ContainsKey(1));
        }

        [Fact]
        public void Constructor_CapacityOfOneAndDefaultEqualityComparer_NotNull()
        {
            var dictionary = new Dictionary<int, int>(1, EqualityComparer<int>.Default);
            Assert.NotNull(dictionary);
        }

        [Fact]
        public void Add_EmptyDictionary_CountEqualsOne()
        {
            var dictionary = new Dictionary<int, int>();
            dictionary.Add(1, 1);
            Assert.Equal(1, dictionary.Count);
        }

        [Fact]
        public void Clear_OneElementDictionary_CountEqualsZero()
        {
            var dictionary = new Dictionary<int, int>();
            dictionary.Add(1, 1);
            //
            dictionary.Clear();
            Assert.Equal(0, dictionary.Count);
        }

        [Fact]
        public void ContainsKey_OneElementDictionary_EqualsTrue()
        {
            var dictionary = new Dictionary<int, int>();
            dictionary.Add(1, 1);
            //
            Assert.True(dictionary.ContainsKey(1));
        }

        [Fact]
        public void ContainsValue_OneElementDictionary_EqualsTrue()
        {
            var dictionary = new Dictionary<int, int>();
            dictionary.Add(1, 1);
            //
            Assert.True(dictionary.ContainsValue(1));
        }

        [Fact]
        public void GetEnumerator_OneElementDictionary_NotNull()
        {
            var dictionary = new Dictionary<int, int>();
            dictionary.Add(1, 1);
            Assert.NotNull(dictionary.GetEnumerator());
        }

        [Fact]
        public void GetObjectData_OneElementDictionary_NotNull()
        {
        }

        [Fact]
        public void OnDeserialization_OneElementDictionary_NotNull()
        {
        }

        [Fact]
        public void Remove_OneElementDictionary_CountEqualsZero()
        {
            var dictionary = new Dictionary<int, int>();
            dictionary.Add(1, 1);
            //
            dictionary.Remove(1);
            Assert.Equal(0, dictionary.Count);
        }

        [Fact]
        public void TryGetValue_OneElementDictionary_EqualTrueAndValueEqualOne()
        {
            var dictionary = new Dictionary<int, int>();
            dictionary.Add(1, 1);
            //
            int value;
            Assert.Equal(true, dictionary.TryGetValue(1, out value));
            Assert.Equal(1, value);
        }

        [Fact]
        public void getComparer_Valid_IsNull()
        {
            var dictionary = new Dictionary<int, int>();
            Assert.Same(EqualityComparer<int>.Default, dictionary.Comparer);
        }

        [Fact]
        public void getCount_OneElementDictionary_EqualOne()
        {
            var dictionary = new Dictionary<int, int>();
            dictionary.Add(1, 1);
            //
            Assert.Equal(1, dictionary.Count);
        }

        [Fact]
        public void getItem_OneElementDictionary_EqualOne()
        {
            var dictionary = new Dictionary<int, int>();
            dictionary.Add(1, 1);
            //
            Assert.Equal(1, dictionary[1]);
        }

        [Fact]
        public void setItem_OneElementDictionary_EqualTwo()
        {
            var dictionary = new Dictionary<int, int>();
            dictionary.Add(1, 1);
            //
            dictionary[1] = 2;
            Assert.Equal(2, dictionary[1]);
        }

        [Fact]
        public void getKeys_OneElementDictionary_KeysCountEqualOne()
        {
            var dictionary = new Dictionary<int, int>();
            dictionary.Add(1, 1);
            //
            Assert.NotNull(dictionary.Keys);
            Assert.Equal(1, dictionary.Keys.Count);
        }

        [Fact]
        public void getValues_OneElementDictionary_ValuesCountEqualOne()
        {
            var dictionary = new Dictionary<int, int>();
            dictionary.Add(1, 1);
            //
            Assert.NotNull(dictionary.Values);
            Assert.Equal(1, dictionary.Values.Count);
        }
    }
}
