using Xunit;
using System.Collections.Generic;
namespace System.Collections
{
    /// <summary>
    /// KeyValuePairTests
    /// </summary>
    public class KeyValuePairTests
    {
        [Fact]
        public void Constructor_Valid_NotNull()
        {
            var keyValuePair = new KeyValuePair<string, object>("test", 1);
            Assert.NotNull(keyValuePair);
        }

        [Fact]
        public void Key_Valid_EqualsTest()
        {
            var keyValuePair = new KeyValuePair<string, object>("test", 1);
            Assert.Equal("test", keyValuePair.Key);
        }

        [Fact]
        public void Value_Valid_EqualsOne()
        {
            var keyValuePair = new KeyValuePair<string, object>("test", 1);
            Assert.Equal(1, keyValuePair.Value);
        }

        [Fact]
        public void ToString_Valid_EqualsTestAndOne()
        {
            var keyValuePair = new KeyValuePair<string, object>("test", 1);
            Assert.Equal("[test, 1]", keyValuePair.ToString());
        }
    }
}
