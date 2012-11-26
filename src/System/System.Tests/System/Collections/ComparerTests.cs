using Xunit;
using System.Collections.Generic;
namespace System.Collections
{
    #region Fakes

    /// <summary>
    /// FakeComparer
    /// </summary>
    internal class FakeComparer : Comparer<string>
    {
        public override int Compare(string x, string y) { return 0; }
    }

    #endregion

    /// <summary>
    /// ComparerTests
    /// </summary>
    public class ComparerTests
    {
        public void Constructor_Valid_NotNull()
        {
            var comparer = new FakeComparer();
            Assert.NotNull(comparer);
        }

        [Fact]
        public void Compare_Valid_EqualsZero()
        {
            var comparer = new FakeComparer();
            Assert.Equal(0, comparer.Compare("test", "test"));
        }

        [Fact]
        public void getDefault_Valid_NotNull()
        {
            Assert.NotNull(Comparer<string>.Default);
        }

        //[Fact]
        //public void Equals_Valid_EqualsTrue()
        //{
        //    var comparer = new FakeComparer();
        //    Assert.True(comparer.Equals(comparer));
        //}

        //[Fact]
        //public void GetHashCode_Valid_NotEqualsZero()
        //{
        //    var comparer = new FakeComparer();
        //    Assert.NotEqual(0, comparer.GetHashCode());
        //}

        //[Fact]
        //public void GetType_Valid_SameAsSampleComparerType()
        //{
        //    var comparer = new FakeComparer();
        //    Assert.Same(comparer.GetType(), typeof(SampleComparer));
        //}

        //[Fact]
        //public void ToString_Valid_EqualsSampleComparerTypeAsText()
        //{
        //    var comparer = new FakeComparer();
        //    Assert.Equal("ThunkC_.Collections_.SampleComparer", comparer.ToString());
        //}
    }
}
