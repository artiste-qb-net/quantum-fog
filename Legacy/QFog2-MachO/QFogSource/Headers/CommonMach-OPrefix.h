// ===========================================================================
//	CommonMach-OPrefix.h	©1999-2003 Metrowerks Corporation.  All rights reserved.
// ===========================================================================
//	Common options for a Mach-O Target

// ---------------------------------------------------------------------------
//	Mach-O Target

#define PP_Target_Carbon		1

#define PP_Target_Classic		(!PP_Target_Carbon)

#define TARGET_API_MAC_CARBON	PP_Target_Carbon
#define TARGET_API_MAC_OS8		PP_Target_Classic
#define TARGET_API_MAC_OSX		PP_Target_Carbon


// ---------------------------------------------------------------------------
//	Options
	
#define PP_Uses_PowerPlant_Namespace	0
#define PP_Supports_Pascal_Strings		1
#define PP_StdDialogs_Option			PP_StdDialogs_NavServicesOnly

#define	PP_Uses_Old_Integer_Types		0
#define PP_Obsolete_AllowTargetSwitch	0
#define PP_Obsolete_ThrowExceptionCode	0
#define PP_Warn_Obsolete_Classes		1

#define PP_Suppress_Notes_221			1

#include <MSLCarbonPrefix.h>

#include 	<cmath>//c language math
#include 	<fp.h>	//must be after #include <cmath>, both necessary
#include 	<complex>
//FIXEDDECIMAL or FLOATDECIMAL are used 
//as the second argument of LString::Assign ( double, char, SInt16 )
//Definition in fp.h but not loaded?
#ifndef FIXEDDECIMAL
	#define 	FIXEDDECIMAL 1
#endif


//see MacTypes.h header, linker couldn't find memmove and bzero
#define NO_BLOCKMOVE_INLINE 1


//-----------------------------------------------------------------
//NONTRIVIAL EXCERPT FROM PP 
//see  DebugSample_PrefixCommon.h

// _do_debug corresponds to __PP_SAMPLE_DEBUG__ in CDebugApp sample project
#ifndef _do_debug
#error "_do_debug is not yet #defined"
#endif

#if	 !(_do_debug==0 || _do_debug==1)
#error "_do_debug is not zero or one"
#endif


// MacOS Macros

#define OLDROUTINENAMES						0
#define OLDROUTINELOCATIONS					0
#define SystemSevenOrLater					1

// use new PP API

#define PP_Obsolete_Constants				0
#define PP_Obsolete_Stream_Creators			0
#define PP_Obsolete_Array_API				0

// for MSL

#if 0 // already defined in ansi_prefix.mach.h
#define	__dest_os							__mac_os
#endif

#if _do_debug

	//for UDebugging and UException which give Throw_() and Signal_() capabilities:
   #define Debug_Throw
   #define Debug_Signal
	
	//these debugging macros must be #defined to desired value
	
	#define	PP_DEBUG						1
//	#define	PP_USE_MOREFILES
	#define	PP_SPOTLIGHT_SUPPORT			1
	#define	PP_QC_SUPPORT					0
	#define PP_DEBUGNEW_SUPPORT				0

	//set DebugNew to full strength
	//could also be set to DEBUG_NEW_BASIC
	
//	#define DEBUG_NEW		DEBUG_NEW_LEAKS
 
#else

	// when not debugging, turn everything is off
	// (only used for final builds)
	
	#define	PP_DEBUG						0
	// we don't need MoreFiles in the final build, but if we
	// did, it would be safe to leave this #defined in.
//	#define	PP_USE_MOREFILES
	#define	PP_SPOTLIGHT_SUPPORT			0
	#define	PP_QC_SUPPORT					0
	#define PP_DEBUGNEW_SUPPORT				0

	//since we're not supporting DebugNew, no need to define DEBUG_NEW
	#define DEBUG_NEW		DEBUG_NEW_OFF
			
#endif
//After defining all the debugging macros,
//we are ready for this header
#include <PP_Debug.h>

#if _do_debug


	// A nifty DebugNew feature is the ability to generate a "leaks.log"
	// file detailing the leaks that it found. The way it does this is
	// through some preprocessor "magic" turning "new" into "NEW". This
	// is a hack, basically, but it can generally work. However, DebugNew
	// cannot work with array operator new (new[]). So if you use array
	// operator new anywhere, this trick will NOT work for you (in fact
	// it could cause trouble). About your only solution then is to
	// manually replace "new" with "NEW" throughout your code.
	//
	// new[] doesn't work because the MW C/C++ compiler (up until CWP3)
	// did not support overriding/overloading new[]. And DebugNew needs
	// to be updated to suit this (at the time of this writing, it was
	// not).
	//
	// Also, if you have your own operator new anywhere (or really any
	// use of the word "new" that's in source and not in a comment), the
	// preprocessor will replace "new" with "NEW". This could cause all
	// sorts of havoc (e.g. try this with LThread or LReentrantMemoryPool
	// (and it's LRMPObject class)).
	//
	// If you cannot utilize DebugNew's DEBUG_NEW_LEAKS functionality
	// then your best bet would probably be to use DebugNew at the
	// DEBUG_NEW_BASIC level and rely upon something like Spotlight
	// for your C++ leak checking.
	
	#if PP_DEBUGNEW_SUPPORT && DEBUG_NEW == DEBUG_NEW_LEAKS
		#include <new.h>
		// NEW is only useful for leaks
		#define new NEW
	#endif

#endif
//-----------------------------------------------------------------


//in addition to what is in PP_ClassHeaders.cp
#include <LTable.h>
#include <LTableView.h>
#include <LTableMultiSelector.h>
#include <LTableMultiGeometry.h>
#include <LTableMonoGeometry.h>
#include <LGroupBox.h>
#include <LIconPane.h>
#include <PP_DebugMacros.h>

#include <LPopupButton.h>
#include <UNavServicesDialogs.h>
#include <UControlRegistry.h>



#include	"my_notation.h"

#define		_mac_gui_app
/* 
The ansi application and the mac gui application each has parts that
are not in the other. 

The overlapping code will have no preprocessor
conditionals bracketing it. 

The parts that are in the ansi_app minus
the mac_gui_app will be bracketed by #ifdef _ansi_app.

The parts that are in the mac_gui_app minus
the ansi_app will be bracketed by #ifdef _mac_gui_app.

The ansi_app is non-graphical and ANSI compatible. 
It can run on any platform. 

*/







