#ifndef __SYSTEM_PLANESET_H__
#define __SYSTEM_PLANESET_H__
#include "List.h"
namespace System { namespace Collections {
	/*
	/// <summary>
	/// PlaneSet
	/// </summary>
	class PlaneSet : public List<Plane>
	{
	public:
	void Clear() { List<Plane>::Clear(); _hash.Free(); }
	int FindPlane(const Plane &plane, const float normalEps, const float distEps);

	private:
	HashIndex _hash;
	};

	/// <summary>
	/// PlaneSet
	/// </summary>
	inline int PlaneSet::FindPlane(const Plane &plane, const float normalEps, const float distEps)
	{
	assert(distEps <= 0.125f);
	int i, border;
	int hashKey = (int)(Math::Fabs(plane.Dist()) * 0.125f);
	for (int border = -1; border <= 1; border++)
	for (int i = _hash.First(hashKey + border); i >= 0; i = _hash.Next(i)) 
	if ((*this)[i].Compare(plane, normalEps, distEps)) 
	return i;
	if (plane.Type() >= PLANETYPE_NEGX && plane.Type() < PLANETYPE_TRUEAXIAL)
	{
	Append(-plane);
	_hash.Add(hashKey, Num() - 1);
	Append(plane);
	_hash.Add( hashKey, Num() - 1);
	return (Num() - 1);
	}
	Append(plane);
	_hash.Add(hashKey, Num() - 1);
	Append(-plane);
	_hash.Add(hashKey, Num() - 1);
	return (Num() - 2);
	}
	*/
}}
#endif /* __SYSTEM_PLANESET_H__ */
