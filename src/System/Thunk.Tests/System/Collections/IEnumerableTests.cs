using Xunit;
using System.Collections.Generic;
namespace System.Collections
{
    /// <summary>
    /// IEnumerableTests
    /// </summary>
    public class IEnumerableTests
    {
        [Fact]
        public void GetEnumerator_TwoElementList_NotNull()
        {
            var list = new List<int>(new int[] { 0, 1 });
            Assert.NotNull(list.GetEnumerator());
        }
    }
}
