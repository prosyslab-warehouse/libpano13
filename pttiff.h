/*
 *  pttiff.h
 *
 * 
 *  Copyright Helmut Dersch and Max Lyons
 *  
 *  May 2006
 *
 *  This program is free software; you can redistribute it and/or
 *  modify it under the terms of the GNU General Public
 *  License as published by the Free Software Foundation; either
 *  version 2 of the License, or (at your option) any later version.
 *
 *  This software is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *  General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public
 *  License along with this software; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 *
 *  Author: Max Lyons
 * 
 */

#ifndef __PTtiff_h__

#define __PTtiff_h__

#include <tiffio.h>

void getCropInformationFromTiff(TIFF *tif, CropInfo *c);
void setCropInformationInTiff(TIFF *tiffFile, CropInfo *crop_info);

int TiffGetImageParameters(TIFF *tiffFile, pt_tiff_parms *tiffData);
int TiffSetImageParameters(TIFF *tiffFile, pt_tiff_parms *tiffData);

int uncropTiff(char *inputFile, char *outputFile, char *messageBuffer);

#endif