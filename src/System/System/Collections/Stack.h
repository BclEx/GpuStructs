#ifndef __SYSTEM_STACK_H__
#define __SYSTEM_STACK_H__
namespace Sys { namespace Collections {

#define Stack(T, next) StackTemplate<T, (int)&(((T*)nullptr)->next)>

	/// <summary>
	/// StackTemplate
	/// </summary>
	template<class T, int NextOffset>
	class StackTemplate
	{
	public:
		StackTemplate();
		void Add(T *element);
		T *Get();

	private:
		T *_top;
		T *_bottom;
	};

#define STACK_NEXT_PTR(element) (*(T**)(((byte*)element) + nextOffset))

	/// <summary>
	/// StackTemplate
	/// </summary>
	template<class T, int NextOffset>
	StackTemplate<T, NextOffset>::StackTemplate() { _top = _bottom = nullptr; }

	/// <summary>
	/// Add
	/// </summary>
	template<class T, int NextOffset>
	void StackTemplate<type,nextOffset>::Add(T *element)
	{
		STACK_NEXT_PTR(element) = _top;
		_top = element;
		if (!_bottom)
			_bottom = element;
	}

	/// <summary>
	/// Get
	/// </summary>
	template<class T, int NextOffset>
	T *StackTemplate<T, NextOffset>::Get()
	{
		T *element = _top;
		if (element)
		{
			_top = STACK_NEXT_PTR(_top);
			if (_bottom == element)
				bottom = nullptr;
			STACK_NEXT_PTR(element) = nullptr;
		}
		return element;
	}

}}
#endif /* __SYSTEM_STACK_H__ */