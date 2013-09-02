#region Foreign-License
/*
Port of the ChannelBuffer struct from tclIO.h/tclIO.c and associated functionality

Copyright (c) 2012 Sky Morey

See the file "license.terms" for information on usage and redistribution of this file, and for a DISCLAIMER OF ALL WARRANTIES.
 */
#endregion
using System;
namespace Tcl.Lang
{

    class ChannelBuffer
    {

        // The next position into which a character
        // will be put in the buffer.

        internal int nextAdded;

        // Position of next byte to be removed
        // from the buffer.

        internal int nextRemoved;

        // How big is the buffer?

        internal int bufLength;

        // Next buffer in chain.

        internal ChannelBuffer next;

        // The actual bytes stored in the buffer

        internal byte[] buf;

        // A channel buffer has BUFFER_PADDING bytes extra at beginning to
        // hold any bytes of a native-encoding character that got split by
        // the end of the previous buffer and need to be moved to the
        // beginning of the next buffer to make a contiguous string so it
        // can be converted to UTF-8.
        //
        // A channel buffer has BUFFER_PADDING bytes extra at the end to
        // hold any bytes of a native-encoding character (generated from a
        // UTF-8 character) that overflow past the end of the buffer and
        // need to be moved to the next buffer.

        internal const int BUFFER_PADDING = 16;

        /// <summary> AllocChannelBuffer -> ChannelBuffer
        /// 
        /// Create a new ChannelBuffer object
        /// </summary>

        internal ChannelBuffer(int length)
        {
            int n;

            n = length + BUFFER_PADDING + BUFFER_PADDING;
            buf = new byte[n];
            nextAdded = BUFFER_PADDING;
            nextRemoved = BUFFER_PADDING;
            bufLength = length + BUFFER_PADDING;
            next = null;
        }
    }
}
