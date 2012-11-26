using Xunit;
using System.Collections.Generic;
namespace System.Collections
{
    /// <summary>
    /// IComparerTests
    /// </summary>
    public class IComparerTests
    {
        [Fact]
        public void Compare_TwoEqualValues_EqualsZero()
        {
            var comparer = (IComparer<int>)Comparer<int>.Default;
            Assert.Equal(0, comparer.Compare(1, 1));
        }
    }
}
