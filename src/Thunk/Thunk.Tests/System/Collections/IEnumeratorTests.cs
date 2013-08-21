using Xunit;
using System.Collections.Generic;
namespace System.Collections
{
    /// <summary>
    /// IEnumeratorTests
    /// </summary>
    public class IEnumeratorTests
    {
        [Fact]
        public void Current_TwoElementList_NotNull()
        {
            var list = new List<int>(new int[] { 0, 1 });
            var enumerator = list.GetEnumerator();
            enumerator.MoveNext();
            var enumerator2 = (IEnumerator<int>)enumerator;
            Assert.NotNull(enumerator2.Current);
        }
    }
}
