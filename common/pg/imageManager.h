#pragma once

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <FreeImage.h>

class ImageManager
{
public :
	static FIBITMAP* GenericLoader(const char* lpszPathName, int flag);
	static bool GenericWriter(FIBITMAP* dib, const char* lpszPathName, int flag);
	static void FreeImageErrorHandler(FREE_IMAGE_FORMAT fif, const char *message);
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Generic loader. </summary>
/// <param name="lpszPathName">	Full pathname of the file. </param>
/// <param name="flag">	Optional load flag constant. </param>
/// <returns>Returns the loaded dib if successful, returns NULL otherwise</returns>
////////////////////////////////////////////////////////////////////////////////////////////////////
inline FIBITMAP* ImageManager::GenericLoader(const char* lpszPathName, int flag)
{
	FREE_IMAGE_FORMAT fif = FIF_UNKNOWN;
	fif = FreeImage_GetFileType(lpszPathName, 0);					// check the file signature and deduce its format (the second argument is currently not used by FreeImage)

	if(fif == FIF_UNKNOWN)											// no signature ? try to guess the file format from the file extension
	{
		fif = FreeImage_GetFIFFromFilename(lpszPathName);
	}
	
	if((fif != FIF_UNKNOWN) && FreeImage_FIFSupportsReading(fif))	// check that the plugin has reading capabilities ...
	{
		return FreeImage_Load(fif, lpszPathName, flag);
	}
	return NULL;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Generic writer. </summary>
/// <param name="dib">		   	[in,out] If non-null, the bitmap. </param>
/// <param name="lpszPathName">	Full pathname of the file. </param>
/// <param name="flag">		   	Optional save flag constant </param>
/// <returns>	Returns true if successful, returns false otherwise </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////
inline bool ImageManager::GenericWriter(FIBITMAP* dib, const char* lpszPathName, int flag) {
	FREE_IMAGE_FORMAT fif = FIF_UNKNOWN;
	BOOL bSuccess = FALSE;

	if(dib)
	{
		fif = FreeImage_GetFIFFromFilename(lpszPathName);			// try to guess the file format from the file extension
		if(fif != FIF_UNKNOWN )										// check that the plugin has sufficient writing and export capabilities ...
		{
			WORD bpp = FreeImage_GetBPP(dib);
			if(FreeImage_FIFSupportsWriting(fif) && FreeImage_FIFSupportsExportBPP(fif, bpp)) 
			{
				bSuccess = FreeImage_Save(fif, dib, lpszPathName, flag);
			}
		}
	}
	return (bSuccess == TRUE) ? true : false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Handler, called when the free image error. </summary>
/// <param name="fif">	  	fif Format / Plugin responsible for the error. </param>
/// <param name="message">	Error message. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////
inline void ImageManager::FreeImageErrorHandler(FREE_IMAGE_FORMAT fif, const char *message) {
	printf("\n*** "); 
	if(fif != FIF_UNKNOWN) {
		printf("%s Format\n", FreeImage_GetFormatFromFIF(fif));
	}
	printf(message);
	printf(" ***\n");
}


