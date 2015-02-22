﻿// os.c
#include "Core.cu.h"
using namespace Core::IO;
namespace Core
{
	__device__ static VSystem *_WSD g_vfsList = nullptr;
#define _vfsList _GLOBAL(VSystem *, g_vfsList)

	__device__ VSystem *VSystem::FindVfs(const char *name)
	{
#ifndef OMIT_AUTOINIT
		RC rc = SysEx::AutoInitialize();
		if (rc) return nullptr;
#endif
		VSystem *vfs = nullptr;
		MutexEx mutex = MutexEx::Alloc(MutexEx::MUTEX_STATIC_MASTER);
		MutexEx::Enter(mutex);
		for (vfs = _vfsList; vfs && _strcmp(name, vfs->Name); vfs = vfs->Next) { }
		MutexEx::Leave(mutex);
		return vfs;
	}

	__device__ static void UnlinkVfs(VSystem *vfs)
	{
		_assert(MutexEx::Held(MutexEx::Alloc(MutexEx::MUTEX_STATIC_MASTER)));
		if (!vfs) { }
		else if (_vfsList == vfs)
			_vfsList = vfs->Next;
		else if (_vfsList)
		{
			VSystem *p = _vfsList;
			while (p->Next && p->Next != vfs)
				p = p->Next;
			if (p->Next == vfs)
				p->Next = vfs->Next;
		}
	}

	__device__ int VSystem::RegisterVfs(VSystem *vfs, bool default_)
	{
		MutexEx mutex = MutexEx::Alloc(MutexEx::MUTEX_STATIC_MASTER);
		MutexEx::Enter(mutex);
		UnlinkVfs(vfs);
		if (default_ || !_vfsList)
		{
			vfs->Next = _vfsList;
			_vfsList = vfs;
		}
		else
		{
			vfs->Next = _vfsList->Next;
			_vfsList->Next = vfs;
		}
		_assert(_vfsList != nullptr);
		MutexEx::Leave(mutex);
		return RC_OK;
	}

	__device__ int VSystem::UnregisterVfs(VSystem *vfs)
	{
		MutexEx mutex = MutexEx::Alloc(MutexEx::MUTEX_STATIC_MASTER);
		MutexEx::Enter(mutex);
		UnlinkVfs(vfs);
		MutexEx::Leave(mutex);
		return RC_OK;
	}

	// from main_c
#pragma region File

#ifdef ENABLE_8_3_NAMES
	__device__ void SysEx::FileSuffix3(const char *baseFilename, char *z)
	{
#if ENABLE_8_3_NAMES<2
		if (!UriBoolean(baseFilename, "8_3_names", 0)) return;
#endif
		int size = _strlen30(z);
		int i;
		for (i = size-1; i > 0 && z[i] != '/' && z[i] !='.'; i--) { }
		if (z[i] == '.' && _ALWAYS(size > i+4)) _memmove(&z[i+1], &z[size-3], 4);
	}
#endif

	struct OpenMode
	{
		const char *Z;
		VSystem::OPEN Mode;
	};


	//SKY TODO: transfer host to constant
//#if __CUDACC__
//	__constant__ static OpenMode _cacheModes[3];
//	__constant__ static OpenMode h_cacheModes[3] =
//#else
//	static OpenMode _cacheModes[] =
//#endif
	__constant__ static OpenMode _cacheModes[3] =
	{
		{ "shared",  VSystem::OPEN_SHAREDCACHE },
		{ "private", VSystem::OPEN_PRIVATECACHE },
		{ nullptr, (VSystem::OPEN)0 }
	};

#if __CUDACC__
	__constant__ static OpenMode _openModes[5];
	static OpenMode h_openModes[] =
#else
	static OpenMode _openModes[] =
#endif
	//__constant__ static OpenMode _openModes[5] =
	{
		{ "ro",  VSystem::OPEN_READONLY },
		{ "rw",  VSystem::OPEN_READWRITE }, 
		{ "rwc", VSystem::OPEN_READWRITE | VSystem::OPEN_CREATE },
		{ "memory", VSystem::OPEN_MEMORY },
		{ nullptr, (VSystem::OPEN)0 }
	};

	__device__ RC VSystem::ParseUri(const char *defaultVfsName, const char *uri, VSystem::OPEN *flagsRef, VSystem **vfsOut, char **fileNameOut, char **errMsgOut)
	{
		_assert(*errMsgOut == nullptr);

		VSystem::OPEN flags = *flagsRef;
		const char *vfsName = defaultVfsName;
		int uriLength = _strlen30(uri);

		RC rc = RC_OK;
		char *fileName;
		if (((flags & VSystem::OPEN_URI) || SysEx_GlobalStatics.OpenUri) && uriLength >= 5 && !_memcmp(uri, "file:", 5))
		{
			// Make sure the SQLITE_OPEN_URI flag is set to indicate to the VFS xOpen method that there may be extra parameters following the file-name.
			flags |= VSystem::OPEN_URI;

			int bytes = uriLength+2; // Bytes of space to allocate
			int uriIdx; // Input character index
			for (uriIdx = 0; uriIdx < uriLength; uriIdx++) bytes += (uri[uriIdx] == '&');
			fileName = (char *)_alloc(bytes);
			if (!fileName) return RC_NOMEM;

			// Discard the scheme and authority segments of the URI.
			if (uri[5] == '/' && uri[6] == '/')
			{
				uriIdx = 7;
				while (uri[uriIdx] && uri[uriIdx] != '/') uriIdx++;
				if (uriIdx != 7 && (uriIdx != 16 || _memcmp("localhost", &uri[7], 9)))
				{
					*errMsgOut = _mprintf("invalid uri authority: %.*s", uriIdx-7, &uri[7]);
					rc = RC_ERROR;
					goto parse_uri_out;
				}
			}
			else
				uriIdx = 5;

			// Copy the filename and any query parameters into the zFile buffer. Decode %HH escape codes along the way. 
			//
			// Within this loop, variable eState may be set to 0, 1 or 2, depending on the parsing context. As follows:
			//
			//   0: Parsing file-name.
			//   1: Parsing name section of a name=value query parameter.
			//   2: Parsing value section of a name=value query parameter.
			int state = 0; // Parser state when parsing URI
			char c;
			int fileNameIdx = 0; // Output character index
			while ((c = uri[uriIdx]) != 0 && c != '#')
			{
				uriIdx++;
				if (c == '%' && _isxdigit(uri[uriIdx]) && _isxdigit(uri[uriIdx+1]))
				{
					int octet = (_hextobyte(uri[uriIdx++]) << 4);
					octet += _hextobyte(uri[uriIdx++]);
					_assert(octet >= 0 && octet < 256);
					if (octet == 0)
					{
						// This branch is taken when "%00" appears within the URI. In this case we ignore all text in the remainder of the path, name or
						// value currently being parsed. So ignore the current character and skip to the next "?", "=" or "&", as appropriate.
						while ((c = uri[uriIdx]) != 0 && c !='#' && 
							(state != 0 || c != '?') && 
							(state != 1 || (c != '=' && c != '&')) && 
							(state != 2 || c != '&'))
							uriIdx++;
						continue;
					}
					c = octet;
				}
				else if (state == 1 && (c == '&' || c == '='))
				{
					if (fileName[fileNameIdx-1] == 0)
					{
						// An empty option name. Ignore this option altogether.
						while (uri[uriIdx] && uri[uriIdx] != '#' && uri[uriIdx-1] != '&') uriIdx++;
						continue;
					}
					if (c == '&')
						fileName[fileNameIdx++] = '\0';
					else
						state = 2;
					c = 0;
				}
				else if ((state == 0 && c == '?') || (state == 2 && c == '&'))
				{
					c = 0;
					state = 1;
				}
				fileName[fileNameIdx++] = c;
			}
			if (state == 1) fileName[fileNameIdx++] = '\0';
			fileName[fileNameIdx++] = '\0';
			fileName[fileNameIdx++] = '\0';

			// Check if there were any options specified that should be interpreted here. Options that are interpreted here include "vfs" and those that
			// correspond to flags that may be passed to the sqlite3_open_v2() method.
			char *opt = &fileName[_strlen30(fileName)+1];
			while (opt[0])
			{
				int optLength = _strlen30(opt);
				char *val = &opt[optLength+1];
				int valLength = _strlen30(val);
				if (optLength == 3 && !_memcmp("vfs", opt, 3))
					vfsName = val;
				else
				{
					OpenMode *modes = nullptr;
					char *modeType = nullptr;
					VSystem::OPEN mask = (VSystem::OPEN)0;
					VSystem::OPEN limit = (VSystem::OPEN)0;
					if (optLength == 5 && !_memcmp("cache", opt, 5))
					{
						mask = VSystem::OPEN_SHAREDCACHE | VSystem::OPEN_PRIVATECACHE;
						modes = _cacheModes;
						limit = mask;
						modeType = "cache";
					}
					if (optLength == 4 && !_memcmp("mode", opt, 4))
					{
						mask = VSystem::OPEN_READONLY | VSystem::OPEN_READWRITE | VSystem::OPEN_CREATE | VSystem::OPEN_MEMORY;
						modes = _openModes;
						limit = mask & flags;
						modeType = "access";
					}
					if (modes)
					{
						VSystem::OPEN mode = (VSystem::OPEN)0;
						for (int i = 0; modes[i].Z; i++)
						{
							const char *z = modes[i].Z;
							if (valLength == _strlen30(z) && !_memcmp(val, z, valLength))
							{
								mode = modes[i].Mode;
								break;
							}
						}
						if (mode == 0)
						{
							*errMsgOut = _mprintf("no such %s mode: %s", modeType, val);
							rc = RC_ERROR;
							goto parse_uri_out;
						}
						if ((mode & ~VSystem::OPEN_MEMORY) > limit)
						{
							*errMsgOut = _mprintf("%s mode not allowed: %s", modeType, val);
							rc = RC_PERM;
							goto parse_uri_out;
						}
						flags = (VSystem::OPEN)(flags & ~mask) | mode;
					}
				}
				opt = &val[valLength+1];
			}
		}
		else
		{
			fileName = (char *)_alloc(uriLength+2);
			if (!fileName) return RC_NOMEM;
			_memcpy(fileName, uri, uriLength);
			fileName[uriLength] = '\0';
			fileName[uriLength+1] = '\0';
			flags &= ~VSystem::OPEN_URI;
		}

		*vfsOut = FindVfs(vfsName);
		if (!*vfsOut)
		{
			*errMsgOut = _mprintf("no such vfs: %s", vfsName);
			rc = RC_ERROR;
		}

parse_uri_out:
		if (rc != RC_OK)
		{
			_free(fileName);
			fileName = nullptr;
		}
		*flagsRef = flags;
		*fileNameOut = fileName;
		return rc;
	}

	__device__ const char *VSystem::UriParameter(const char *filename, const char *param)
	{
		if (!filename) return nullptr;
		filename += _strlen30(filename) + 1;
		while (filename[0])
		{
			int x = _strcmp(filename, param);
			filename += _strlen30(filename) + 1;
			if (x == 0) return filename;
			filename += _strlen30(filename) + 1;
		}
		return nullptr;
	}

	__device__ bool VSystem::UriBoolean(const char *filename, const char *param, bool dflt)
	{
		const char *z = UriParameter(filename, param);
		return (z ? ConvertEx::GetBoolean(z, dflt) : dflt);
	}

	__device__ int64 VSystem::UriInt64(const char *filename, const char *param, int64 dflt)
	{
		const char *z = UriParameter(filename, param);
		int64 v;
		return (z && ConvertEx::Atoi64(z, &v, _strlen30(z), TEXTENCODE_UTF8) == RC_OK ? v : dflt);
	}

#pragma endregion
}