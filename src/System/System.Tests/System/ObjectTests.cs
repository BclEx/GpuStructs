using Xunit;
namespace System
{
    /// <summary>
    /// ObjectTests
    /// </summary>
    public class ObjectTests
    {
        [Fact]
        public void Constructor_Valid_NotNull()
        {
            var obj = new object();
            Assert.NotNull(obj);
        }

        [Fact]
        public void Equals_OneObject_EqualsTrue()
        {
            var obj = new object();
            Assert.Equal(true, obj.Equals(obj));
        }

        [Fact]
        public void Equals2_OneObject_EqualsTrue()
        {
            var obj = new object();
            Assert.Equal(true, object.Equals(obj, obj));
        }

        [Fact]
        public void GetHashCode_OneObject_NotEqualsZero()
        {
            var obj = new object();
            Assert.NotEqual(0, obj.GetHashCode());
        }

        [Fact]
        public void GetType_OneObject_SameType()
        {
            var obj = new object();
            Assert.Same(obj.GetType(), typeof(object));
        }

        [Fact]
        public void ReferenceEquals_OneObject_EqualsTrue()
        {
            var obj = new object();
            Assert.Equal(true, object.ReferenceEquals(obj, obj));
        }

    }
}
