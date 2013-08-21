using Xunit;
using System.Collections.Generic;
namespace System.Collections
{
    /// <summary>
    /// SortedDictionaryTests
    /// </summary>
    public class SortedDictionaryTests
    {
        [Fact]
        public void Constructor_Valid_NotNull()
        {
            var set = new SortedDictionary<string, object>();
            Assert.NotNull(set);
        }
    }
}
