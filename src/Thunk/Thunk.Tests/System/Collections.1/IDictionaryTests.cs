using Xunit;
using System.Collections.Generic;
namespace System.Collections
{
    /// <summary>
    /// IDictionaryTests
    /// </summary>
    public class IDictionaryTests
    {
        [Fact]
        public void Add_EmptyDictionary_CountEqualsOne()
        {
            var startDictionary = new Dictionary<int, int>();
            var dictionary = (IDictionary<int, int>)startDictionary;
            //
            dictionary.Add(1, 1);
            Assert.Equal(1, dictionary.Count);
        }

        [Fact]
        public void ContainsKey_OneElementDictionary_EqualsTrue()
        {
            var startDictionary = new Dictionary<int, int>();
            startDictionary.Add(1, 1);
            var dictionary = (IDictionary<int, int>)startDictionary;
            //
            Assert.True(dictionary.ContainsKey(1));
        }

        [Fact]
        public void Remove_OneElementDictionary_CountEqualsZero()
        {
            var startDictionary = new Dictionary<int, int>();
            startDictionary.Add(1, 1);
            var dictionary = (IDictionary<int, int>)startDictionary;
            //
            dictionary.Remove(1);
            Assert.Equal(0, dictionary.Count);
        }

        [Fact]
        public void TryGetValue_OneElementDictionary_EqualTrueAndValueEqualOne()
        {
            var startDictionary = new Dictionary<int, int>();
            startDictionary.Add(1, 1);
            var dictionary = (IDictionary<int, int>)startDictionary;
            //
            int value;
            Assert.Equal(true, dictionary.TryGetValue(1, out value));
            Assert.Equal(1, value);
        }

        [Fact]
        public void getItem_OneElementDictionary_EqualOne()
        {
            var startDictionary = new Dictionary<int, int>();
            startDictionary.Add(1, 1);
            var dictionary = (IDictionary<int, int>)startDictionary;
            //
            Assert.Equal(1, dictionary[1]);
        }

        [Fact]
        public void setItem_OneElementDictionar_EqualTwo()
        {
            var startDictionary = new Dictionary<int, int>();
            startDictionary.Add(1, 1);
            var dictionary = (IDictionary<int, int>)startDictionary;
            //
            dictionary[1] = 2;
            Assert.Equal(2, dictionary[1]);
        }

        [Fact]
        public void getKeys_OneElementDictionary_KeysCountEqualOne()
        {
            var startDictionary = new Dictionary<int, int>();
            startDictionary.Add(1, 1);
            var dictionary = (IDictionary<int, int>)startDictionary;
            //
            Assert.NotNull(dictionary.Keys);
            Assert.Equal(1, dictionary.Keys.Count);
        }

        [Fact]
        public void getValues_OneElementDictionary_ValuesCountEqualOne()
        {
            var startDictionary = new Dictionary<int, int>();
            startDictionary.Add(1, 1);
            var dictionary = (IDictionary<int, int>)startDictionary;
            //
            Assert.NotNull(dictionary.Values);
            Assert.Equal(1, dictionary.Values.Count);
        }
    }
}
