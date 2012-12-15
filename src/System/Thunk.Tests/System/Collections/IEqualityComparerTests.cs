using Xunit;
using System.Collections.Generic;
namespace System.Collections
{
    /// <summary>
    /// IEqualityComparerTests
    /// </summary>
    public class IEqualityComparerTests
    {
        [Fact]
        public void Equals_Valid_EqualsTrue()
        {
            var comparer = (IEqualityComparer<string>)EqualityComparer<string>.Default;
            Assert.True(comparer.Equals("test", "test"));
        }

        [Fact]
        public void GetHashCode_Valid_NotEqualZero()
        {
            var comparer = (IEqualityComparer<string>)EqualityComparer<string>.Default;
            Assert.NotEqual(0, comparer.GetHashCode("test"));
        }
    }
}
