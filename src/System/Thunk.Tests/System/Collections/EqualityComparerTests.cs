using Xunit;
using System.Collections.Generic;
namespace System.Collections
{
    #region Fakes

    /// <summary>
    /// FakeEqualityComparer
    /// </summary>
    internal class FakeEqualityComparer : EqualityComparer<string>
    {
        public override bool Equals(string x, string y) { return (x == y); }
        public override int GetHashCode(string obj) { return 1; }
    }

    #endregion

    /// <summary>
    /// EqualityComparerTests
    /// </summary>
    public class EqualityComparerTests
    {
        [Fact]
        public void Constructor_Valid_NotNull()
        {
            var comparer = new FakeEqualityComparer();
            Assert.NotNull(comparer);
        }

        [Fact]
        public void Equals_Valid_EqualsTrue()
        {
            var comparer = new FakeEqualityComparer();
            Assert.True(comparer.Equals("test", "test"));
        }

        [Fact]
        public void GetHashCode_Valid_NotEqualZero()
        {
            var comparer = new FakeEqualityComparer();
            Assert.NotEqual(0, comparer.GetHashCode("test"));
        }

        [Fact]
        public void getDefault_Valid_NotNull()
        {
            Assert.NotNull(EqualityComparer<string>.Default);
        }
    }
}
