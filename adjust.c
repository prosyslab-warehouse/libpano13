/* Panorama_Tools	-	Generate, Edit and Convert Panoramic Images
   Copyright (C) 1998,1999 - Helmut Dersch  der@fh-furtwangen.de
   
   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2, or (at your option)
   any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.  */

/*------------------------------------------------------------*/


#include "filter.h"
#include "f2c.h"
#include <float.h>

#define C_FACTOR	100.0

static  AlignInfo	*g;	// This struct holds all informations for the optimization




void 			ColCorrect( Image *im, double ColCoeff[3][2] );
void 			GetColCoeff( Image *src, Image *buf, double ColCoeff[3][2] );
void 			getControlPoints( Image *im, struct controlPoint *cp );
void 			writeControlPoints( struct controlPoint *cp,char* cdesc );
int 			CheckParams( AlignInfo *g );
static int		CheckMakeParams( aPrefs *aP);
static int		GetOverlapRect( PTRect *OvRect, PTRect *r1, PTRect *r2 );
int 			AddEdgePoints( AlignInfo *gl );
int 			pt_average( UCHAR* pixel, int BytesPerLine, double rgb[3], int bytesPerChannel );
double 			distLine(int N0, int N1);

void adjust(TrformStr *TrPtr, aPrefs *prefs)
{
	int	 	destwidth, destheight;
	aPrefs		aP, *aPtr;
#if 0
	int 		nt = 0;		// Morph  parameters
	PTTriangle 	*ts=NULL; 
	PTTriangle 	*td=NULL; 
#endif
	SetAdjustDefaults(&aP);

	//PrintError ("MRDL: In adjust");

	switch( prefs->mode & 7 )// Should we use prefs, or read from script?
	{
		case _insert:
		case _extract:
			if( prefs->mode & _useScript ){
				aPtr = readAdjustLine( &(prefs->scriptFile) );
				if(aPtr==NULL){
					PrintError("Error processing script file" );
					TrPtr->success = 0;
					return;
				}
				memcpy(&aP, aPtr, sizeof(aPrefs));
				free(aPtr); aPtr = &aP;

				if( (TrPtr->mode & 7) == _usedata ){ // Report panorama format and stitching info back to calling app.
					memcpy( &prefs->pano, &aP.pano, sizeof( Image ) );
					memcpy( &prefs->sBuf, &aP.sBuf, sizeof( stBuf ) );
				}

				TrPtr->interpolator = aP.interpolator;
				TrPtr->gamma	    = aP.gamma;
					
#if 0
				int readmode = 1;
				aPtr = &aP;
				gsPrPtr->interpolator 	= TrPtr->interpolator;
				gsPrPtr->gamma			= TrPtr->gamma;
				if( TrPtr->mode & _destSupplied ){
					PTRect* p = &TrPtr->dest->selection;
					if( !(p->bottom == 0 && p->right == 0) &&
					    !(p->right == TrPtr->dest->width &&
					     p->bottom == TrPtr->dest->height) )
						readmode = 0;
				}
				if( readAdjust( aPtr, &(prefs->scriptFile), readmode, gsPrPtr ) != 0 )
				{
					PrintError("Error processing script file" );
					TrPtr->success = 0;
					return;
				}
				if( (TrPtr->mode & 7) == _usedata ) // Report panorama format and stitching info back to calling app.
				{
					memcpy( &prefs->pano, &aP.pano, sizeof( Image ) );
					memcpy( &prefs->sBuf, &aP.sBuf, sizeof( stBuf ) );
				}
				// Use modevalues read from script
				TrPtr->interpolator = gsPrPtr->interpolator;
				TrPtr->gamma		= gsPrPtr->gamma;
				
				// Parse script again, now reading triangles if morphing requested
				if( aPtr->im.cP.correction_mode & correction_mode_morph )
				{
					char*				script;
					AlignInfo			ainf;
					int					nIm, nPts; // Number of image being processed
					Image				im[2];
					
					script = LoadScript( &(prefs->scriptFile) );
					if( script != NULL ) 					// We can read the scriptfile
					{	
						nIm = numLines( script, '!' ) - 1;
						
						if( nIm < 0)
							nIm = numLines( script, 'o' ) - 1;
					
						// Set ainf
						ainf.nt 	= 0;
						ainf.t		= NULL;
						ainf.numIm 	= 2;
						ainf.im		= im;
						memcpy( &ainf.pano, &aP.pano, sizeof( Image ));
						memcpy( &ainf.im[0], &aP.pano, sizeof( Image ));
						memcpy( &ainf.im[1], &aP.pano, sizeof( Image ));
						
						nPts = ReadMorphPoints( script, &ainf, nIm );
						if(nPts > 0) // Found Points
						{
							AddEdgePoints( &ainf );
							TriangulatePoints( &ainf, 1 );
							nt = ainf.nt;
							if(nt > 0)
							{
								SortControlPoints	( &ainf, 1 );
								SetSourceTriangles	( &ainf, 1, &td  );
								SetDestTriangles    ( &ainf, 1, &ts  );
							}
						}
						if(ainf.numPts > 0) free(ainf.cpt);
					}
					free( script );
				}
#endif
			}else{
			 	aPtr = prefs;
			}
			 break;
		default:
			break;
	}
	switch( prefs->mode & 7)
	{
		case _insert:			// Create a panoramic image using src; merge with buffer if required
			// Find brightest rectangle if this is a circular fishey image
			{
			Image ImCrop, *theSrc;
			
			if( aPtr->im.format ==_fisheye_circ	&& aPtr->im.cP.cutFrame )
			{
				int fwidth = TrPtr->src->width, fheight = TrPtr->src->height;
				
				if( aPtr->im.cP.frame ) // subtract framewidth from width/height
				{
					fwidth = TrPtr->src->width - aPtr->im.cP.frame;
					if( aPtr->im.cP.frame < fwidth ) fwidth -= aPtr->im.cP.frame;
					if( aPtr->im.cP.frame < fheight) fheight-= aPtr->im.cP.frame;
				}
				else
				{
					if( aPtr->im.cP.fwidth > 0)
						fwidth = aPtr->im.cP.fwidth;
					if( aPtr->im.cP.fheight > 0)
						fheight = aPtr->im.cP.fheight;
				}
					
				if( cutTheFrame( &ImCrop, TrPtr->src, fwidth, fheight, TrPtr->mode & _show_progress ) != 0 )
				{
					PrintError("Error Cropping Image");
					TrPtr->success = 0;
					return;
				}
				theSrc = TrPtr->src;
				TrPtr->src = &ImCrop;
				
			}
			// Image params are set as src 
			aPtr->im.width	= TrPtr->src->width;
			aPtr->im.height	= TrPtr->src->height;
			
			// Pano is set to buffer, if merging requested; else as prefs
			if( *aPtr->sBuf.srcName != 0 )
			{
				if (LoadBufImage( &(aPtr->pano), aPtr->sBuf.srcName, 0) != 0 )
				{
					PrintError( "Error loading Buffer; trying without" );
				}
			}
						
			if( aPtr->pano.width == 0 && aPtr->im.hfov != 0.0)
			{
				aPtr->pano.width = aPtr->im.width * aPtr->pano.hfov / aPtr->im.hfov;
				aPtr->pano.width/=10; aPtr->pano.width*=10;
			}
			if( aPtr->pano.height == 0 )
				aPtr->pano.height = aPtr->pano.width/2;

			destheight 				= aPtr->pano.height;
			destwidth 				= aPtr->pano.width;
			
			if( destheight == 0 || destwidth == 0 )
			{
				PrintError("Please set Panorama width/height" );
				TrPtr->success = 0;
				goto _insert_exit;
			}
			
		
			if( SetDestImage( TrPtr, destwidth, destheight) != 0)
			{
				PrintError("Could not allocate %ld bytes",TrPtr->dest->dataSize );
				TrPtr->success = 0;
				goto _insert_exit;
			}
			TrPtr->mode				|= _honor_valid;
			CopyPosition( TrPtr->src,  &(aPtr->im) );
			CopyPosition( TrPtr->dest, &(aPtr->pano) );
			addAlpha( TrPtr->src ); // Add alpha channel to indicate valid data
			
			aPtr->mode = prefs->mode; // For checkparam
			MakePano( TrPtr,  aPtr );
			
			if(aPtr->ts) free(aPtr->ts);
			if(aPtr->td) free(aPtr->td);

			// Stitch images; Proceed only if panoramic image valid

			if( TrPtr->success )
			{
				if( *(aPtr->sBuf.srcName) != 0 ){ // We have to merge in one images
					// Load the bufferimage
					if( LoadBufImage( &aPtr->pano, aPtr->sBuf.srcName, 1 ) != 0 )
					{
						PrintError( "Could not load buffer %s; Keeping Source",aPtr->sBuf.srcName );
						goto _insert_exit;
					}

					if( HaveEqualSize( &aPtr->pano, TrPtr->dest ))
					{
	
						// At this point we have two valid, equally sized images						
						// Do Colour Correction on one or both  images
						DoColorCorrection( TrPtr->dest, &aPtr->pano, aPtr->sBuf.colcorrect & 3);
						
						if( merge( TrPtr->dest , &aPtr->pano, aPtr->sBuf.feather, TrPtr->mode & _show_progress, aPtr->sBuf.seam ) != 0 )
						{
							PrintError( "Error merging images. Keeping Source" );
						}
					}
					myfree( (void**)aPtr->pano.data );
				} // src != 0
					
				if( *(aPtr->sBuf.destName) != 0 ) // save buffer image
				{
					if( SaveBufImage( TrPtr->dest, aPtr->sBuf.destName ) != 0 )
						PrintError( "Could not save to Buffer. Most likely your disk is full");
				}
			} // Tr.success 
				

			if( TrPtr->success == 0  && ! (TrPtr->mode & _destSupplied) )	
				myfree( (void**)TrPtr->dest->data );
				
		_insert_exit:
			if( aPtr->im.format ==_fisheye_circ	&& aPtr->im.cP.cutFrame )	// There is a cropped source image;
			{
				if( ImCrop.data != NULL )
					myfree( (void**) ImCrop.data );
				TrPtr->src = theSrc;
			}
			
			}
			break;
		
		case _extract:
				
			if( aPtr->im.width == 0 )
			{
				aPtr->im.width = 500 ;
			}
			if(  aPtr->im.height == 0 )
			{
				aPtr->im.height = aPtr->im.width * 4 / 5;
			}
				
			// Set pano-params to src-image irrespective of prefs
			aPtr->pano.width	= TrPtr->src->width;				//	width of panorama
			aPtr->pano.height	= TrPtr->src->height;				//  height of panorama
			
			CopyPosition( TrPtr->src, &(aPtr->pano) );
			addAlpha( TrPtr->src ); 
				
			if( *(aPtr->sBuf.destName) != 0 ) // save buffer image
			{
				if( SaveBufImage( TrPtr->src, aPtr->sBuf.destName ) != 0 )
					PrintError( "Could not save Buffer Image. Most likely your disk is full");
			} 
			
			// Set up Image Structure in TrPtr struct


			destheight 			= aPtr->im.height;
			destwidth 			= aPtr->im.width;


			if( SetDestImage( TrPtr, destwidth, destheight) != 0)
			{
				PrintError("Could not allocate %ld bytes",TrPtr->dest->dataSize );
				TrPtr->success = 0;
				return;
			}

			CopyPosition( TrPtr->dest, &(aPtr->im) );

			TrPtr->mode					|= _honor_valid;
			if( aPtr->pano.hfov == 360.0 )
				TrPtr->mode				|= _wrapX;
			
			aPtr->mode = prefs->mode; // For checkparam
			ExtractStill( TrPtr,  aPtr );
				
				
			if( TrPtr->success == 0 && ! (TrPtr->mode & _destSupplied))	
				myfree( (void**)TrPtr->dest->data );
			break;
		
		case _readControlPoints:
			{
				char			*script, *newscript, cdesc[1000];
				controlPoint 	cp[NUMPTS];			// List of Control points

				script = LoadScript( &(prefs->scriptFile) );
				if( script != NULL ) 					// We can read the scriptfile
				{
					newscript = (char*) malloc( strlen(script) + NUMPTS * 60 ); // One line per pair of points
					if( newscript != NULL )
					{
						readControlPoints( script, cp );		// If this is the second image: get coordinates in first
						getControlPoints( TrPtr->src, cp );		// Scan image and find control points
						writeControlPoints( cp, cdesc );		// format control point coordinates
						
						sprintf( newscript, "%s\n%s", script, cdesc );
						
						if( WriteScript( newscript,&( prefs->scriptFile), 0 ) != 0 )
										PrintError( "Could not write Scriptfile" );
						free( newscript );
					}
					free( script );
				}

			}
			TrPtr->success = 0;							// Don't destroy image!
			break;


		case _runOptimizer:
			// Run Optimizer; Dummy image needed but not changed
			{
				char*				script;
				OptInfo				opt;
				AlignInfo			ainf;

				script = LoadScript( &(prefs->scriptFile) );
				if( script != NULL ) 					// We can read the scriptfile
				{
					if (ParseScript( script, &ainf ) == 0)
					{
						if( CheckParams( &ainf ) == 0 ) 				// and it seems to make sense
						{
							ainf.fcn	= fcnPano;
							
							SetGlobalPtr( &ainf ); 
							
							opt.numVars 		= g->numParam;
							opt.numData 		= g->numPts;
							opt.SetVarsToX		= SetLMParams;
							opt.SetXToVars		= SetAlignParams;
							opt.fcn				= g->fcn;
							*opt.message		= 0;

							RunLMOptimizer( &opt );
							g->data				= opt.message;
							WriteResults( script, &(prefs->scriptFile), g, distSquared ,
							            ( TrPtr->mode & 7 ) != _usedata );
						}
						DisposeAlignInfo( &ainf );					// These were allocated by 'ParseScript()'
					}
					free( script );
				}
			}
				
			TrPtr->success = 0;							// Don't destroy Dummy image!
			break;
		default:
			TrPtr->success = 0;							
			break;

	}
}




// Make a pano in TrPtr->dest (must be allocated and all set!)
// using parameters in aPrefs (ignore image parameters in TrPtr !)

void MakePano( TrformStr *TrPtr, aPrefs *aP )
{
	struct 	MakeParams	mp;
	fDesc 	stack[15], fD;		// Parameters for execute 
	void	*morph[3];	

	int 	i,k, kstart, kend, color;

	TrPtr->success = 1;
	
	if( CheckMakeParams( aP) != 0)
	{
		TrPtr->success = 0;
		return;
	}


	if(  isColorSpecific( &(aP->im.cP) ) )			// Color dependent
	{
		kstart 	= 1; kend	= 4;
	}
	else 											// Color independent
	{
		kstart	= 0; kend	= 1;
	}
				
	for( k = kstart; k < kend; k++ )
	{
		color = k-1; if( color < 0 ) color = 0;
		SetMakeParams( stack, &mp, &(aP->im) , &(aP->pano), color );
		
		if( aP->nt > 0 )	// Morphing requested
		{
			morph[0] = (void*)aP->td;
			morph[1] = (void*)aP->ts;
			morph[2] = (void*)&aP->nt;

			i=0; while( stack[i].func != NULL && i<14 ) i++;
			if( i!=14 )
			{
				for(i=14; i>0; i--)
				{
					memcpy( &stack[i], &stack[i-1], sizeof( fDesc ));
				}
				stack[0].func 		= tmorph;
				stack[0].param 		= (void*)morph;
			}
		}
					
			
		
		if( TrPtr->success != 0)
		{
			fD.func = execute_stack; fD.param = stack;
			transForm( TrPtr,  &fD , k);
		}
	}
}

/*This function was added by Kekus Digital on 18/9/2002. This function takes the parameter 'imageNum' which repesents the index of the image that has to be converted.*/
void MyMakePano( TrformStr *TrPtr, aPrefs *aP, int imageNum )
{
	struct 	MakeParams	mp;
	fDesc 	stack[15], fD;		// Parameters for execute 
	void	*morph[3];	

	int 	i,k, kstart, kend, color;

	TrPtr->success = 1;
	
	if( CheckMakeParams( aP) != 0)
	{
		TrPtr->success = 0;
		return;
	}


	if(  isColorSpecific( &(aP->im.cP) ) )			// Color dependent
	{
		kstart 	= 1; kend	= 4;
	}
	else 											// Color independent
	{
		kstart	= 0; kend	= 1;
	}
				
	for( k = kstart; k < kend; k++ )
	{
		color = k-1; if( color < 0 ) color = 0;
		SetMakeParams( stack, &mp, &(aP->im) , &(aP->pano), color );
		
		if( aP->nt > 0 )	// Morphing requested
		{
			morph[0] = (void*)aP->td;
			morph[1] = (void*)aP->ts;
			morph[2] = (void*)&aP->nt;

			i=0; while( stack[i].func != NULL && i<14 ) i++;
			if( i!=14 )
			{
				for(i=14; i>0; i--)
				{
					memcpy( &stack[i], &stack[i-1], sizeof( fDesc ));
				}
				stack[0].func 		= tmorph;
				stack[0].param 		= (void*)morph;
			}
		}
					
			
		
		if( TrPtr->success != 0)
		{
			fD.func = execute_stack; fD.param = stack;
			MyTransForm( TrPtr,  &fD , k, imageNum);
		}
	}
}


// Extract image from pano in TrPtr->src 
// using parameters in prefs (ignore image parameters
// in TrPtr)

void ExtractStill( TrformStr *TrPtr , aPrefs *aP )
{
	struct 	MakeParams	mp;
	fDesc 	stack[15], fD;		// Parameters for execute 

	int 	k, kstart, kend, color;

	TrPtr->success = 1;

	if( CheckMakeParams( aP) != 0)
	{
		TrPtr->success = 0;
		return;
	}

		

	if( isColorSpecific( &(aP->im.cP) ) )			// Color dependent
	{
		kstart 	= 1; kend	= 4;
	}
	else 															// Color independent
	{
		kstart	= 0; kend	= 1;
	}
				
	for( k = kstart; k < kend; k++ )
	{
		color = k-1; if( color < 0 ) color = 0;
		SetInvMakeParams( stack, &mp,  &(aP->im) , &(aP->pano), color );
		
		if( TrPtr->success != 0)
		{
			fD.func = execute_stack; fD.param = stack;
			transForm( TrPtr,  &fD , k);
		}
	}
}


// Set Makeparameters depending on adjustprefs, color and source image

void SetMakeParams( struct fDesc *stack, struct MakeParams *mp, Image *im , Image *pn, int color )
{
	int 		i;
	double		a,b;						// field of view in rad


	a	=	 DEG_TO_RAD( im->hfov );	// field of view in rad		
	b	=	 DEG_TO_RAD( pn->hfov );

	SetMatrix(  	- DEG_TO_RAD( im->pitch ), 
					0.0, 
					- DEG_TO_RAD( im->roll ), 
					mp->mt, 
					0 );


	if(pn->format == _rectilinear)									// rectilinear panorama
	{
		mp->distance 	= (double) pn->width / (2.0 * tan(b/2.0));
		if(im->format == _rectilinear)										// rectilinear image
		{
			mp->scale[0] = ((double)pn->hfov / im->hfov) * 
						   (a /(2.0 * tan(a/2.0))) * ((double)im->width/(double) pn->width)
						   * 2.0 * tan(b/2.0) / b; 

		}
		else 																//  pamoramic or fisheye image
		{
			mp->scale[0] = ((double)pn->hfov / im->hfov) * ((double)im->width/ (double) pn->width)
						   * 2.0 * tan(b/2.0) / b; 
		}
	}
	else																// equirectangular or panoramic or fisheye
	{
		mp->distance 	= ((double) pn->width) / b;
		if(im->format == _rectilinear)										// rectilinear image
		{
			mp->scale[0] = ((double)pn->hfov / im->hfov) * (a /(2.0 * tan(a/2.0))) * ((double)im->width)/ ((double) pn->width); 

		}
		else 																//  pamoramic or fisheye image
		{
			mp->scale[0] = ((double)pn->hfov / im->hfov) * ((double)im->width)/ ((double) pn->width); 
		}
	}
	mp->scale[1] 	= mp->scale[0];


	mp->shear[0] 	= im->cP.shear_x / im->height;
	mp->shear[1] 	= im->cP.shear_y / im->width;
	mp->rot[0]		= mp->distance * PI;								// 180� in screenpoints
	mp->rot[1]		= -im->yaw *  mp->distance * PI / 180.0; 			//    rotation angle in screenpoints
	mp->perspect[0] = (void*)(mp->mt);
	mp->perspect[1] = (void*)&(mp->distance);
			
	for(i=0; i<4; i++)
		mp->rad[i] 	= im->cP.radial_params[color][i];
	mp->rad[5] = im->cP.radial_params[color][4];

	if( (im->cP.correction_mode & 3) == correction_mode_radial )
		mp->rad[4] 	= ( (double)( im->width < im->height ? im->width : im->height) ) / 2.0;
	else
		mp->rad[4] 	= ((double) im->height) / 2.0;
		

	mp->horizontal 	= im->cP.horizontal_params[color];
	mp->vertical 	= im->cP.vertical_params[color];
	
	i = 0;

	if(pn->format == _rectilinear)									// rectilinear panorama
	{
		SetDesc(stack[i],	erect_rect,		&(mp->distance)	); i++;	// Convert rectilinear to equirect
	}
	else if(pn->format == _panorama)
	{
		SetDesc(stack[i],	erect_pano,		&(mp->distance)	); i++;	// Convert panoramic to equirect
	}
	else if(pn->format == _fisheye_circ || pn->format == _fisheye_ff)
	{
		SetDesc(stack[i],	erect_sphere_tp,		&(mp->distance)	); i++;	// Convert panoramic to sphere
	}

	SetDesc(	stack[i],	rotate_erect,		mp->rot			); i++;	// Rotate equirect. image horizontally
	SetDesc(	stack[i],	sphere_tp_erect,	&(mp->distance)	); i++;	// Convert spherical image to equirect.
	SetDesc(	stack[i],	persp_sphere,		mp->perspect	); i++;	// Perspective Control spherical Image

	if(im->format 		== _rectilinear)									// rectilinear image
	{
		SetDesc(stack[i],	rect_sphere_tp,		&(mp->distance)	); i++;	// Convert rectilinear to spherical
	}
	else if	(im->format 	== _panorama)									//  pamoramic image
	{
		SetDesc(stack[i],	pano_sphere_tp,		&(mp->distance)	); i++;	// Convert panoramic to spherical
	}
	else if	(im->format 	== _equirectangular)							//  PSphere image
	{
		SetDesc(stack[i],	erect_sphere_tp,	&(mp->distance)	); i++;	// Convert PSphere to spherical
	}

	SetDesc(	stack[i],	resize,				mp->scale		); i++; // Scale image
	
	if( im->cP.radial )
	{
		switch( im->cP.correction_mode & 3 )
	{
			case correction_mode_radial:    SetDesc(stack[i],radial,mp->rad); 	  i++; break;
			case correction_mode_vertical:  SetDesc(stack[i],vertical,mp->rad);   i++; break;
			case correction_mode_deregister:SetDesc(stack[i],deregister,mp->rad); i++; break;
		}
	}
	if (  im->cP.vertical)
	{
		SetDesc(stack[i],vert,				&(mp->vertical)); 	i++;
	}
	if ( im->cP.horizontal )
		{
		SetDesc(stack[i],horiz,				&(mp->horizontal)); i++;
		}
	if( im->cP.shear )
	{
		SetDesc( stack[i],shear,			mp->shear		); i++;
	}

	stack[i].func  = (trfn)NULL;

// print stack for debugging
#if 0
	printf( "Rotate params: %lg  %lg\n" , mp->rot[0], mp->rot[1]);
	printf( "Distance     : %lg\n" , mp->distance);
	printf( "Perspect params: %lg  %lg  %lg\n",a, beta , gammar );  	
	if(aP->format 		== _rectilinear)									// rectilinear image
	{
		printf( "Rectilinear\n" );  	
	}
	else if	(aP->format 	== _panorama)									//  pamoramic image
	{
		printf( "Panorama\n" );  	
	}
	else
		printf( "Fisheye\n" );  	
	
	printf( "Scaling     : %lg\n" , mp->scale[0]);

	if(  aP->correct )
	{
		printf( "Correct:\n" );  	
		if( aP->c_prefs.shear )
		{
			printf( "Shear: %lg\n", mp->shear );  	
		}
		if ( aP->c_prefs.horizontal )
		{
			printf( "horiz:%lg\n", mp->horizontal );  
		}
		if (  aP->c_prefs.vertical)
		{
			printf( "vert:%lg\n", mp->vertical );  
		}
		if( aP->c_prefs.radial )
		{
			printf( "Polynomial:\n" );  	
			if( aP->c_prefs.isScanningSlit )
			{
				printf( "Scanning Slit:\n" );  	
			}
			else
			{
				printf( "Radial:\n" );  	
				printf( "Params: %lg %lg %lg %lg %lg\n", mp->rad[0],mp->rad[1],mp->rad[2],mp->rad[3],mp->rad[4] );  	
			}
		}
	}

#endif
}


// Set inverse Makeparameters depending on adjustprefs, color and source image

void 	SetInvMakeParams( struct fDesc *stack, struct MakeParams *mp, Image *im , Image *pn, int color )
{

	int 		i;
	double		a,b;							// field of view in rad


	a =	 DEG_TO_RAD( im->hfov );	// field of view in rad		
	b =	 DEG_TO_RAD( pn->hfov );


	SetMatrix( 	DEG_TO_RAD( im->pitch ), 
				0.0, 
				DEG_TO_RAD( im->roll ), 
				mp->mt, 
				1 );


	if(pn->format == _rectilinear)									// rectilinear panorama
	{
		mp->distance 	= (double) pn->width / (2.0 * tan(b/2.0));
		if(im->format == _rectilinear)										// rectilinear image
		{
			mp->scale[0] = ((double)pn->hfov / im->hfov) * 
						   (a /(2.0 * tan(a/2.0))) * ((double)im->width/(double) pn->width)
						   * 2.0 * tan(b/2.0) / b; 

		}
		else 																//  pamoramic or fisheye image
		{
			mp->scale[0] = ((double)pn->hfov / im->hfov) * ((double)im->width/ (double) pn->width)
						   * 2.0 * tan(b/2.0) / b; 
		}
	}
	else																// equirectangular or panoramic 
	{
		mp->distance 	= ((double) pn->width) / b;
		if(im->format == _rectilinear)										// rectilinear image
		{
			mp->scale[0] = ((double)pn->hfov / im->hfov) * (a /(2.0 * tan(a/2.0))) * ((double)im->width)/ ((double) pn->width); 

		}
		else 																//  pamoramic or fisheye image
		{
			mp->scale[0] = ((double)pn->hfov / im->hfov) * ((double)im->width)/ ((double) pn->width); 
		}
	}
	mp->shear[0] 	= -im->cP.shear_x / im->height;
	mp->shear[1] 	= -im->cP.shear_y / im->width;
	
	mp->scale[0] = 1.0 / mp->scale[0];
	mp->scale[1] 	= mp->scale[0];
	mp->horizontal 	= -im->cP.horizontal_params[color];
	mp->vertical 	= -im->cP.vertical_params[color];
	for(i=0; i<4; i++)
		mp->rad[i] 	= im->cP.radial_params[color][i];
	mp->rad[5] = im->cP.radial_params[color][4];
	
	switch( im->cP.correction_mode & 3 )
	{
		case correction_mode_radial: mp->rad[4] = ((double)(im->width < im->height ? im->width : im->height) ) / 2.0;break;
		case correction_mode_vertical: 
		case correction_mode_deregister: mp->rad[4] = ((double) im->height) / 2.0;break;
	}

	mp->rot[0]		= mp->distance * PI;								// 180� in screenpoints
	mp->rot[1]		= im->yaw *  mp->distance * PI / 180.0; 			//    rotation angle in screenpoints

	mp->perspect[0] = (void*)(mp->mt);
	mp->perspect[1] = (void*)&(mp->distance);




	i = 0;	// Stack counter
		
		// Perform radial correction
	if( im->cP.shear )
	{
		SetDesc( stack[i],shear,			mp->shear		); i++;
	}
		
	if ( im->cP.horizontal )
	{
		SetDesc(stack[i],horiz,				&(mp->horizontal)); i++;
	}
	if (  im->cP.vertical)
	{
		SetDesc(stack[i],vert,				&(mp->vertical)); 	i++;
	}
	if(   im->cP.radial )
	{
		switch( im->cP.correction_mode & 3)
		{
			case correction_mode_radial:   SetDesc(stack[i],inv_radial,mp->rad); 	i++; break;
			case correction_mode_vertical: SetDesc(stack[i],inv_vertical,mp->rad); 	i++; break;
			case correction_mode_deregister: break;
		}
	}
	
	SetDesc(	stack[i],	resize,				mp->scale		); i++; // Scale image
	
	if(im->format 		== _rectilinear)									// rectilinear image
	{
		SetDesc(stack[i],	sphere_tp_rect,		&(mp->distance)	); i++;	// 
	}
	else if	(im->format 	== _panorama)									//  pamoramic image
	{
		SetDesc(stack[i],	sphere_tp_pano,		&(mp->distance)	); i++;	// Convert panoramic to spherical
	}
	else if	(im->format 	== _equirectangular)							//  PSphere image
	{
		SetDesc(stack[i],	sphere_tp_erect,	&(mp->distance)	); i++;	// Convert Psphere to spherical
	}


	SetDesc(	stack[i],	persp_sphere,		mp->perspect	); i++;	// Perspective Control spherical Image
	SetDesc(	stack[i],	erect_sphere_tp,	&(mp->distance)	); i++;	// Convert spherical image to equirect.
	SetDesc(	stack[i],	rotate_erect,		mp->rot			); i++;	// Rotate equirect. image horizontally

	if(pn->format == _rectilinear)									// rectilinear panorama
	{
		SetDesc(stack[i],	rect_erect,		&(mp->distance)	); i++;	// Convert rectilinear to spherical
	}
	else if(pn->format == _panorama)
	{
		SetDesc(stack[i],	pano_erect,		&(mp->distance)	); i++;	// Convert rectilinear to spherical
	}
	else if(pn->format == _fisheye_circ || pn->format == _fisheye_ff )
	{
		SetDesc(stack[i],	sphere_tp_erect,		&(mp->distance)	); i++;	// Convert rectilinear to spherical
	}
	
	stack[i].func = (trfn)NULL;
}




	

	


// Add an alpha channel to the image, assuming rectangular or circular shape
// subtract frame 

void addAlpha( Image *im ){
	register int 		x,y,c1;
	int			framex, framey;
	register unsigned char	*src;
	
	src = *(im->data);
	framex = 0; framey = 0;
	
	if( im->cP.cutFrame ){
		if( im->cP.frame < 0 || im->cP.fwidth < 0 || im->cP.fheight < 0 ){	// Use supplied alpha channel
			return;
		}
		
		if( im->cP.frame != 0 ){
			framex = (im->width/2 	> im->cP.frame ? im->cP.frame : 0);
			framey = (im->height/2 	> im->cP.frame ? im->cP.frame : 0);
		}
		else{
			if( im->width > im->cP.fwidth )
				framex = (im->width - im->cP.fwidth) / 2;
			if( im->height > im->cP.fheight )
				framey = (im->height - im->cP.fheight) / 2;
		}
	}


	if( im->bitsPerPixel == 32 || im->bitsPerPixel == 64 ) // leave 24/48 bit images unchanged
	{
		if( im->format != _fisheye_circ )		// Rectangle valid
		{
			int yend = im->height - framey;
			int xend = im->width  - framex;
			
			if( im->bitsPerPixel == 32 )
			{
				for(y = 0; y < im->height; y++)
				{
					c1 = y * im->bytesPerLine;
					for(x = 0; x < im->width; x++)
						src[ c1 + 4 * x ] = 0;
				}
				for(y = framey; y < yend; y++)
				{
					c1 = y * im->bytesPerLine;
					for(x = framex; x < xend; x++)
						src[ c1 + 4 * x ] = UCHAR_MAX;
				}
			}
			else // im->bitsPerPixel == 64
			{
				for(y = 0; y < im->height; y++)
				{
					c1 = y * im->bytesPerLine;
					for(x = 0; x < im->width; x++)
						*((USHORT*)(src + c1 + 8 * x )) = 0;
				}
				for(y = framey; y < yend; y++)
				{
					c1 = y * im->bytesPerLine;
					for(x = framex; x < xend; x++)
						*((USHORT*)(src + c1 + 8 * x )) = USHRT_MAX;
				}
			}
		}
		else if( im->format == _fisheye_circ ) // Circle valid
		{
			int topCircle 	= ( im->height - im->width ) / 2; 	// top of circle
			int botCircle 	= topCircle + im->width ;			// bottom of circle
			int r			= ( im->width / 2 );				// radius of circle
			int x1, x2, h;
			
			if( im->bitsPerPixel == 32 )
			{
				for(y = 0; y < im->height  ; y++) 
				{
					if( (y < topCircle) || (y > botCircle) )  // Always invalid
					{
						for(x = 0; x < im->width; x++)
							src[y * im->bytesPerLine + 4 * x] = 0;
					}
					else
					{
						h 	=	y - im->height/2;
						if( h*h > r*r ) h = r;

						x1 = (int) (r - sqrt( r*r - h*h ));
						if( x1 < 0 ) x1 = 0;
						x2 = (int) (r + sqrt( r*r - h*h ));
						if( x2 > im->width ) x2 = im->width;
			
						for(x = 0; x < x1; x++)
							src[y * im->bytesPerLine + 4 * x] = 0;
						for(x = x1; x < x2; x++)
							src[y * im->bytesPerLine + 4 * x] = UCHAR_MAX;
						for(x = x2; x < im->width; x++)
							src[y * im->bytesPerLine + 4 * x] = 0;
					}
				}
			}
			else // im->bitsPerPixel == 64
			{
				for(y = 0; y < im->height  ; y++) 
				{
					if( (y < topCircle) || (y > botCircle) )  // Always invalid
					{
						for(x = 0; x < im->width; x++)
							*((USHORT*)(src + y * im->bytesPerLine + 8 * x)) = 0;
					}
					else
					{
						h 	=	y - im->height/2;
						if( h*h > r*r ) h = r;

						x1 = (int) (r - sqrt( r*r - h*h ));
						if( x1 < 0 ) x1 = 0;
						x2 = (int) (r + sqrt( r*r - h*h ));
						if( x2 > im->width ) x2 = im->width;
			
						for(x = 0; x < x1; x++)
							*((USHORT*)(src + y * im->bytesPerLine + 8 * x)) = 0;
						for(x = x1; x < x2; x++)
							*((USHORT*)(src + y * im->bytesPerLine + 8 * x)) = USHRT_MAX;
						for(x = x2; x < im->width; x++)
							*((USHORT*)(src + y * im->bytesPerLine + 8 * x)) = 0;
					}
				}
			}
		} // mode
	}	// pixelsize
}


// Angular Distance of Control point "num"
double distSphere( int num ){
	double 		x, y ; 	// Coordinates of control point in panorama
	double		w2, h2;
	int j;
	Image sph;
	int n[2];
	struct 	MakeParams	mp;
	struct  fDesc 		stack[15];
	CoordInfo b[2];
	

	// Get image position in imaginary spherical image
	
	SetImageDefaults( &sph );
	
	sph.width 			= 360;
	sph.height 			= 180;
	sph.format			= _equirectangular;
	sph.hfov			= 360.0;
	
	n[0] = g->cpt[num].num[0];
	n[1] = g->cpt[num].num[1];
	
	// Calculate coordinates x/y in panorama

	for(j=0; j<2; j++){
		SetInvMakeParams( stack, &mp, &g->im[ n[j] ], &sph, 0 );
		
		h2 	= (double)g->im[ n[j] ].height / 2.0 - 0.5;
		w2	= (double)g->im[ n[j] ].width  / 2.0 - 0.5;
		
		
		execute_stack( 	(double)g->cpt[num].x[j] - w2,		// cartesian x-coordinate src
						(double)g->cpt[num].y[j] - h2,		// cartesian y-coordinate src
						&x, &y, stack);

		x = DEG_TO_RAD( x ); 
		y = DEG_TO_RAD( y ) + PI/2.0;
		b[j].x[0] =   sin(x) * sin( y );
		b[j].x[1] =   cos( y );
		b[j].x[2] = - cos(x) * sin(y);
	}
	
	return acos( SCALAR_PRODUCT( &b[0], &b[1] ) ) * g->pano.width / ( 2.0 * PI );
}



void pt_getXY(int n, double x, double y, double *X, double *Y){
	struct 	MakeParams	mp;
	struct  fDesc 		stack[15];
	double h2,w2;

	SetInvMakeParams( stack, &mp, &g->im[ n ], &g->pano, 0 );
	h2 	= (double)g->im[ n ].height / 2.0 - 0.5;
	w2	= (double)g->im[ n ].width  / 2.0 - 0.5;


	execute_stack( 	x - w2,	y - h2,	X, Y, stack);
}

// Return distance of 2 lines
// The line through the two farthest apart points is calculated
// Returned is the distance of the other two points
double distLine(int N0, int N1){
	double x[4],y[4], del, delmax, A, B, C, mu, d0, d1;
	int n0, n1, n2, n3, i, k;

	pt_getXY(g->cpt[N0].num[0], (double)g->cpt[N0].x[0], (double)g->cpt[N0].y[0], &x[0], &y[0]);
	pt_getXY(g->cpt[N0].num[1], (double)g->cpt[N0].x[1], (double)g->cpt[N0].y[1], &x[1], &y[1]);
	pt_getXY(g->cpt[N1].num[0], (double)g->cpt[N1].x[0], (double)g->cpt[N1].y[0], &x[2], &y[2]);
	pt_getXY(g->cpt[N1].num[1], (double)g->cpt[N1].x[1], (double)g->cpt[N1].y[1], &x[3], &y[3]);

	delmax = 0.0;
	n0 = 0; n1 = 1;

	for(i=0; i<4; i++){
		for(k=i+1; k<4; k++){
			del = (x[i]-x[k])*(x[i]-x[k])+(y[i]-y[k])*(y[i]-y[k]);
			if(del>delmax){
				n0=i; n1=k; delmax=del;
			}
		}
	}
	if(delmax==0.0) return 0.0;

	for(i=0; i<4; i++){
		if(i!= n0 && i!= n1){
			n2 = i;
			break;
		}
	}
	for(i=0; i<4; i++){
		if(i!= n0 && i!= n1 && i!=n2){
			n3 = i;
		}
	}


	A=y[n1]-y[n0]; B=x[n0]-x[n1]; C=y[n0]*(x[n1]-x[n0])-x[n0]*(y[n1]-y[n0]);

	mu=1.0/sqrt(A*A+B*B);

	d0 = (A*x[n2]+B*y[n2]+C)*mu;
	d1 = (A*x[n3]+B*y[n3]+C)*mu;

	return d0*d0 + d1*d1;

}


// Calculate the distance of Control Point "num" between two images
// in final pano

double distSquared( int num ) 
{
	double 		x[2], y[2]; 				// Coordinates of control point in panorama
	double		w2, h2;
	int j, n[2];
	double result;

	struct 	MakeParams	mp;
	struct  fDesc 		stack[15];

	

	n[0] = g->cpt[num].num[0];
	n[1] = g->cpt[num].num[1];
	
	// Calculate coordinates x/y in panorama

	for(j=0; j<2; j++)
	{
		SetInvMakeParams( stack, &mp, &g->im[ n[j] ], &g->pano, 0 );
		
		h2 	= (double)g->im[ n[j] ].height / 2.0 - 0.5;
		w2	= (double)g->im[ n[j] ].width  / 2.0 - 0.5;
		

		execute_stack( 	(double)g->cpt[num].x[j] - w2,		// cartesian x-coordinate src
						(double)g->cpt[num].y[j] - h2,		// cartesian y-coordinate src
						&x[j], &y[j], stack);
		// test to check if inverse works
#if 0
		{
			double xt, yt;
			struct 	MakeParams	mtest;
			struct  fDesc 		stacktest[15];
			SetMakeParams( stacktest, &mtest, &g->im[ n[j] ], &g->pano, 0 );
			execute_stack( 	x[j],		// cartesian x-coordinate src
							y[j],		// cartesian y-coordinate src
						&xt, &yt, stacktest);
			
			printf("x= %lg,	y= %lg,  xb = %lg, yb = %lg \n", g->cpt[num].x[j], g->cpt[num].y[j], xt+w2, yt+h2);  
			
		}
#endif
	}
	
	
//	printf("Coordinates 0:   %lg:%lg	1:	%lg:%lg\n",x[0] + g->pano->width/2,y[0]+ g->pano->height/2, x[1] + g->pano->width/2,y[1]+ g->pano->height/2);


	// take care of wrapping and points at edge of panorama
	
	if( g->pano.hfov == 360.0 )
	{
		double delta = abs( x[0] - x[1] );
		
		if( delta > g->pano.width / 2 )
		{
			if( x[0] < x[1] )
				x[0] += g->pano.width;
			else
				x[1] += g->pano.width;
		}
	}


	switch( g->cpt[num].type )		// What do we want to optimize?
	{
		case 1:			// x difference
			result = ( x[0] - x[1] ) * ( x[0] - x[1] );
			break;
		case 2:			// y-difference
			result =  ( y[0] - y[1] ) * ( y[0] - y[1] );
			break;
		default:
			result = ( y[0] - y[1] ) * ( y[0] - y[1] ) + ( x[0] - x[1] ) * ( x[0] - x[1] ); // square of distance
			break;
	}
	

	return result;
}



// Calculate the distance of Control Point "num" between two images
// in image 0

double distSquared2( int num ) 
{
	double 		x[2], y[2]; 				// Coordinates of control point in panorama
	double		w2, h2;
	int n[2];
	double result;

	struct 	MakeParams	mp;
	struct  fDesc 		stack[15];

	

	n[0] = g->cpt[num].num[0];
	n[1] = g->cpt[num].num[1];
	
	// Calculate coordinates x/y in panorama

	SetInvMakeParams( stack, &mp, &g->im[ n[0] ], &g->pano, 0 );
		
	h2 	= (double)g->im[ n[0] ].height / 2.0 - 0.5;
	w2	= (double)g->im[ n[0] ].width  / 2.0 - 0.5;
		

	execute_stack( 	(double)g->cpt[num].x[0] - w2,		// cartesian x-coordinate src
					(double)g->cpt[num].y[0] - h2,		// cartesian y-coordinate src
					&x[0], &y[0], stack);

	// Calculate coordinates x/y in image 1

	SetMakeParams( stack, &mp,&g->im[ n[1] ], &g->pano, 0 );

	execute_stack( 	x[0], y[0],
					&x[1], &y[1], stack);

	h2 	= (double)g->im[ n[1] ].height / 2.0 - 0.5;
	w2	= (double)g->im[ n[1] ].width  / 2.0 - 0.5;
	
	x[0] = (double)g->cpt[num].x[1] - w2;
	y[0] = (double)g->cpt[num].y[1] - h2;
//	printf("Coordinates 0:   %lg:%lg	1:	%lg:%lg\n",x[0] + g->pano->width/2,y[0]+ g->pano->height/2, x[1] + g->pano->width/2,y[1]+ g->pano->height/2);

	switch( g->cpt[num].type )		// What do we want to optimize?
	{
		case 1:			// x difference
			result = ( x[0] - x[1] ) * ( x[0] - x[1] );
			break;
		case 2:			// y-difference
			result =  ( y[0] - y[1] ) * ( y[0] - y[1] );
			break;
		default:
			result = ( y[0] - y[1] ) * ( y[0] - y[1] ) + ( x[0] - x[1] ) * ( x[0] - x[1] ); // square of distance
			break;
	}
	

	return result;
}


// Levenberg-Marquardt function measuring the quality of the fit in fvec[]

int fcnPano(m,n,x,fvec,iflag)
int m,n;
int *iflag;
double x[],fvec[]; 
{
#pragma unused(n)
	int i;
	static int numIt;
	double result;
	
	if( *iflag == -100 ){ // reset
		numIt = 0;
		infoDlg ( _initProgress, "Optimizing Variables" );
		return 0;
	}
	if( *iflag == -99 ){ // 
		infoDlg ( _disposeProgress, "" );
		return 0;
	}


	if( *iflag == 0 )
	{
		char message[256];
		
		result = 0.0;
		for( i=0; i < g->numPts; i++)
		{
			result += fvec[i] ;
		}
		result = sqrt( result/ (double)g->numPts );
		
		sprintf( message, "Average Difference between Controlpoints \nafter %d iteration(s): %g pixels", numIt,result);//average);
		numIt += 10;
		if( !infoDlg ( _setProgress,message ) )
			*iflag = -1;
		return 0;
	}

	// Set Parameters


	SetAlignParams( x ) ;
	
	// Calculate distances
	
	result = 0.0;
	for( i=0; i < g->numPts; i++){
		int j;
		switch(g->cpt[i].type){
			case 0: fvec[i] = distSphere( i );
			        break;
			case 1:
			case 2: fvec[i] = distSquared( i );
				break;
			default:for(j=0; j<g->numPts; j++){
					if(j!=i && g->cpt[i].type == g->cpt[j].type){
						fvec[i] = distLine(i,j);
						break;
					}
				}
				break;
		}
		result += fvec[i] ;
	}
	result = result/ (double)g->numPts;
	
	for( i=g->numPts; i < m; i++)
	{
		fvec[i] = result ;
	}
		
	return 0;
}




// Find Colour correcting polynomial matching the overlap of src and buf
// using least square fit.
// Each RGB-Channel is fitted using the relation  
//      buf = coeff[0] * src + coeff[1]
#if 1
void GetColCoeff( Image *src, Image *buf, double ColCoeff[3][2] ){
	register int 		x,y,c1,c2,i, numPts;
	double 			xy[3], xi[3], xi2[3], yi[3], xav[3], yav[3];
	register unsigned char 	*source, *buff;
	int			BitsPerChannel,bpp;


	
	GetBitsPerChannel( src, BitsPerChannel );
	bpp = src->bitsPerPixel/8;
	

	source = *(src->data);
	buff   = *(buf->data);
	
	for(i=0;i<3;i++){
		xy[i] = xi[i] = xi2[i] = yi[i] = 0.0;
	}
	numPts = 0;	

	if( BitsPerChannel == 8 ){
		for( y=2; y<src->height-2; y++){
			c1 = y * src->bytesPerLine;
			for( x=2; x<src->width-2; x++){
				c2 = c1 + x*bpp;
				if( source[c2] != 0  &&  buff[c2] != 0 ){ // &&   // In overlap region?
				    //(source[c2] != UCHAR_MAX  ||  buff[c2] != UCHAR_MAX)){ // above seam?
					if( pt_average( source+c2, src->bytesPerLine, xav, 1 ) &&
					    pt_average( buff+c2, src->bytesPerLine, yav, 1 ) ){
						numPts++;
						for( i=0; i<3; i++){
							xi[i]	+= xav[i];
							yi[i]	+= yav[i];
							xi2[i] 	+= xav[i]*xav[i];
							xy[i]	+= xav[i]*yav[i];
						}
					}
				}
			}
		}
	}else{//16
		for( y=1; y<src->height-1; y++){
			c1 = y * src->bytesPerLine;
			for( x=1; x<src->width-1; x++){
				c2 = c1 + x*bpp;
				if( *((USHORT*)(source + c2)) != 0  &&  *((USHORT*)(buff + c2)) != 0 ) { //&& // In overlap region?
				 //( *((USHORT*)(source + c2)) != USHRT_MAX  ||  *((USHORT*)(buff + c2)) != USHRT_MAX ) ){ // above seam?
					if( pt_average( source + c2, src->bytesPerLine, xav, 2 ) &&
					    pt_average( buff + c2, src->bytesPerLine, yav, 2 )){
						numPts++;
						for( i=0; i<3; i++){
							xi[i]	+= xav[i];
							yi[i]	+= yav[i];
							xi2[i] 	+= xav[i]*xav[i];
							xy[i]	+= xav[i]*yav[i];
						}
					}
				}
			}
		}
	}
		
	
	if( numPts > 0 ){
		for( i=0; i<3; i++){
			ColCoeff[i][0] = ( numPts * xy[i] - xi[i] * yi[i] ) / ( numPts * xi2[i] - xi[i]*xi[i] );
			ColCoeff[i][1] = ( xi2[i] * yi[i] - xy[i] * xi[i] ) / ( numPts * xi2[i] - xi[i]*xi[i] );
		}
	}else{
		for( i=0; i<3; i++){
			ColCoeff[i][0] = 1.0;
			ColCoeff[i][1] = 0.0;
		}
	}
}
#endif
// Average 9 pixels
int pt_average( UCHAR* pixel, int BytesPerLine, double rgb[3], int bytesPerChannel ){
	int x, y, i;
	UCHAR *px;
	double sum = 1.0 + 4 * 0.5 + 8 * 0.2 + 8 * 0.1 ;//2.6;
#if 0
	double bl[3][3] =      {{ 0.1, 0.3, 0.1}, // Blurr overlap using this matrix
				{ 0.3, 1.0, 0.3},
				{ 0.1, 0.3, 0.1}};

#endif
	double bl[5][5] =      {{ 0.0, 0.1, 0.2, 0.1, 0.0},
				{ 0.1, 0.2, 0.5, 0.2, 0.1},
				{ 0.2, 0.5, 1.0, 0.5, 0.2},
				{ 0.1, 0.2, 0.5, 0.2, 0.1},
				{ 0.0, 0.1, 0.2, 0.1, 0.0}};


	rgb[0] = rgb[1] = rgb[2] = 0.0;
	if( bytesPerChannel != 1 ) return;

	for(y=0; y<5; y++){
		for(x=0; x<5; x++){
			px = pixel + (y-2)*BytesPerLine + x-2;
			if( *px == 0 ) return 0;
			rgb[0] +=  *(++px) * bl[y][x];
			rgb[1] +=  *(++px) * bl[y][x];
			rgb[2] +=  *(++px) * bl[y][x];
		}
	}
	for( i=0; i<3; i++) rgb[i]/=sum;

}


#if 0

// Backup

// Find Colour correcting polynomial matching the overlap of src and buf
// using least square fit.
// Each RGB-Channel is fitted using the relation  
//      buf = coeff[0] * src + coeff[1]

void GetColCoeff( Image *src, Image *buf, double ColCoeff[3][2] )
{
	register int x,y,c1,c2,i, numPts;
	double xy[3], xi[3], xi2[3], yi[3];
	register unsigned char *source, *buff;
	int		BitsPerChannel,bpp;
	
	GetBitsPerChannel( src, BitsPerChannel );
	bpp = src->bitsPerPixel/8;
	

	source = *(src->data);
	buff   = *(buf->data);
	for(i=0;i<3;i++)
	{
		xy[i] = xi[i] = xi2[i] = yi[i] = 0.0;
	}
	numPts = 0;	

	if( BitsPerChannel == 8 )
	{
		for( y=0; y<src->height; y++)
		{
			c1 = y * src->bytesPerLine;
			for( x=0; x<src->width; x++)
			{
				c2 = c1 + x*bpp;
				if( source[c2] != 0  &&  buff[c2] != 0 ) // In overlap region?
				{
					numPts++;
					for( i=0; i<3; i++)
					{
						c2++;
						xi[i]	+= (double)source[c2];
						yi[i]	+= (double)buff[c2];
						xi2[i] 	+= ((double)source[c2])*((double)source[c2]);
						xy[i]	+= ((double)source[c2])*((double)buff[c2]);
					}
				}
			}
		}
	}
	else // 16
	{
		for( y=0; y<src->height; y++)
		{
			c1 = y * src->bytesPerLine;
			for( x=0; x<src->width; x++)
			{
				c2 = c1 + x*bpp;
				if( *((USHORT*)(source + c2)) != 0  &&  *((USHORT*)(buff + c2)) != 0 ) // In overlap region?
				{
					numPts++;
					for( i=0; i<3; i++)
					{
						c2++;
						xi[i]	+= (double) *((USHORT*)(source + c2));
						yi[i]	+= (double) *((USHORT*)(buff + c2));
						xi2[i] 	+= ((double) *((USHORT*)(source + c2)))*((double) *((USHORT*)(source + c2)));
						xy[i]	+= ((double) *((USHORT*)(source + c2)))*((double) *((USHORT*)(buff + c2)));
					}
				}
			}
		}
	}
		
	
	if( numPts > 0 )
	{
		for( i=0; i<3; i++)
		{
			ColCoeff[i][0] = ( numPts * xy[i] - xi[i] * yi[i] ) / ( numPts * xi2[i] - xi[i]*xi[i] );
			ColCoeff[i][1] = ( xi2[i] * yi[i] - xy[i] * xi[i] ) / ( numPts * xi2[i] - xi[i]*xi[i] );
		}
	}
	else
	{
		for( i=0; i<3; i++)
		{
			ColCoeff[i][0] = 1.0;
			ColCoeff[i][1] = 0.0;
		}
	}
}

#endif


// Colourcorrect the image im using polynomial coefficients ColCoeff
// Each RGB-Channel is corrected using the relation  
//      new = coeff[0] * old + coeff[1]

void ColCorrect( Image *im, double ColCoeff[3][2] )
{
	register int x,y, c1, c2, i;
	register unsigned char* data;
	register double result;
	int bpp, BitsPerChannel;
	
	GetBitsPerChannel( im, BitsPerChannel );
	bpp = im->bitsPerPixel/8;

	data = *(im->data);

	if( BitsPerChannel == 8 )
	{
		for( y=0; y<im->height; y++)
		{
			c1 = y * im->bytesPerLine;
			for( x=0; x<im->width; x++ )
			{
				c2 = c1 + x * bpp;
				if( data[ c2 ] != 0 ) // Alpha channel set
				{
					for( i=0; i<3; i++)
					{
						c2++;
						result = ColCoeff[i][0] * data[ c2 ] + ColCoeff[i][1];
						DBL_TO_UC( data[ c2 ], result );
					}
				}
			}
		}
	}
	else // 16
	{
		for( y=0; y<im->height; y++)
		{
			c1 = y * im->bytesPerLine;
			for( x=0; x<im->width; x++ )
			{
				c2 = c1 + x * bpp;
				if( *((USHORT*)(data + c2 )) != 0 ) // Alpha channel set
				{
					for( i=0; i<3; i++)
					{
						c2++;
						result = ColCoeff[i][0] * *((USHORT*)(data + c2 )) + ColCoeff[i][1];
						DBL_TO_US( *((USHORT*)(data + c2 )) , result );
					}
				}
			}
		}
	}
}


void SetAdjustDefaults( aPrefs *prefs )
{

	prefs->magic		=	50;					//	File validity check, must be 50
	prefs->mode			= 	_insert;			//	
	
	SetImageDefaults( &(prefs->im) );
	SetImageDefaults( &(prefs->pano) );
	
	SetStitchDefaults( &(prefs->sBuf) );	

	memset( &(prefs->scriptFile), 0, sizeof( fullPath ) );
	
	prefs->nt = 0;
	prefs->ts = NULL;
	prefs->td = NULL;
	
	prefs->interpolator = _poly3;
	prefs->gamma = 1.0;
}




				

void 	DisposeAlignInfo( struct AlignInfo *g )
{
	if(g->im != NULL) free(g->im);
	if(g->opt!= NULL) free(g->opt);
	if(g->cpt!= NULL) free(g->cpt);
	if(g->t  != NULL) free(g->t);
	if(g->cim != NULL) free(g->cim);
}




// Set global preferences structures using LM-params

int	SetAlignParams( double *x )
{
	// Set Parameters
	int i,j,k;
	
	j = 0;
	
	for( i=0; i<g->numIm; i++ ){
		if( (k = g->opt[i].yaw) > 0 ){
			if( k == 1 ){	g->im[i].yaw  =	x[j++];	NORM_ANGLE( g->im[i].yaw );
			}else{	g->im[i].yaw  =	g->im[k-2].yaw;	}
		}
		if( (k = g->opt[i].pitch) > 0 ){
			if( k == 1 ){	g->im[i].pitch  =	x[j++];	NORM_ANGLE( g->im[i].pitch );
			}else{	g->im[i].pitch  =	g->im[k-2].pitch;	}
		}
		if( (k = g->opt[i].roll) > 0 ){
			if( k == 1 ){	g->im[i].roll  =	x[j++];	NORM_ANGLE( g->im[i].roll );
			}else{	g->im[i].roll  =	g->im[k-2].roll;	}
		}
		if( (k = g->opt[i].hfov) > 0 ){
			if( k == 1 ){	
				g->im[i].hfov  =	x[j++];	
				if( g->im[i].hfov < 0.0 )
					g->im[i].hfov = - g->im[i].hfov;
			}else{	g->im[i].hfov  = g->im[k-2].hfov;	}
		}
		if( (k = g->opt[i].a) > 0 ){
			if( k == 1 ){ g->im[i].cP.radial_params[0][3]  =	x[j++] / C_FACTOR;
			}else{	g->im[i].cP.radial_params[0][3] = g->im[k-2].cP.radial_params[0][3];}
		}
		if( (k = g->opt[i].b) > 0 ){
			if( k == 1 ){ g->im[i].cP.radial_params[0][2]  =	x[j++] / C_FACTOR;
			}else{	g->im[i].cP.radial_params[0][2] = g->im[k-2].cP.radial_params[0][2];}
		}
		if( (k = g->opt[i].c) > 0 ){
			if( k == 1 ){ g->im[i].cP.radial_params[0][1]  =	x[j++] / C_FACTOR;
			}else{	g->im[i].cP.radial_params[0][1] = g->im[k-2].cP.radial_params[0][1];}
		}
		if( (k = g->opt[i].d) > 0 ){
			if( k == 1 ){ g->im[i].cP.horizontal_params[0]  =	x[j++];
			}else{	g->im[i].cP.horizontal_params[0] = g->im[k-2].cP.horizontal_params[0];}
		}
		if( (k = g->opt[i].e) > 0 ){
			if( k == 1 ){ g->im[i].cP.vertical_params[0]  =	x[j++];
			}else{	g->im[i].cP.vertical_params[0] = g->im[k-2].cP.vertical_params[0];}
		}
		if( (k = g->opt[i].shear_x) > 0 ){
			if( k == 1 ){ g->im[i].cP.shear_x  =	x[j++];
			}else{	g->im[i].cP.shear_x = g->im[k-2].cP.shear_x;}
		}

		if( (k = g->opt[i].shear_y) > 0 ){
			if( k == 1 ){ g->im[i].cP.shear_y  =	x[j++];
			}else{	g->im[i].cP.shear_y = g->im[k-2].cP.shear_y;}
		}

		
		g->im[i].cP.radial_params[0][0] = 1.0 - ( g->im[i].cP.radial_params[0][3]
														+ g->im[i].cP.radial_params[0][2]
														+ g->im[i].cP.radial_params[0][1] ) ;

	}
	if( j != g->numParam )
		return -1;
	else
		return 0;

}

// Set LM params using global preferences structure
// Change to cover range 0....1 (roughly)

int SetLMParams( double *x )
{
	int i,j;
		
	j=0; // Counter for optimization parameters


	for( i=0; i<g->numIm; i++ ){
		if(g->opt[i].yaw == 1)  //  optimize alpha? 0-no 1-yes
			x[j++] = g->im[i].yaw;

		if(g->opt[i].pitch == 1)  //  optimize pitch? 0-no 1-yes
			x[j++] = g->im[i].pitch; 

		if(g->opt[i].roll == 1)  //  optimize gamma? 0-no 1-yes
			x[j++] = g->im[i].roll ; 

		if(g->opt[i].hfov == 1)  //  optimize hfov? 0-no 1-yes
			x[j++] = g->im[i].hfov ; 

		if(g->opt[i].a == 1)  //  optimize a? 0-no 1-yes
			x[j++] =  g->im[i].cP.radial_params[0][3] * C_FACTOR; 

		if(g->opt[i].b == 1)   //  optimize b? 0-no 1-yes
			x[j++] = g->im[i].cP.radial_params[0][2] * C_FACTOR; 

		if(g->opt[i].c == 1)  //  optimize c? 0-no 1-yes
			x[j++] = g->im[i].cP.radial_params[0][1] * C_FACTOR; 

		if(g->opt[i].d == 1)  //  optimize d? 0-no 1-yes
			x[j++] = g->im[i].cP.horizontal_params[0] ; 

		if(g->opt[i].e == 1)  //  optimize e? 0-no 1-yes
			x[j++] = g->im[i].cP.vertical_params[0]  ; 

		if(g->opt[i].shear_x == 1)  //  optimize shear_x? 0-no 1-yes
			x[j++] = g->im[i].cP.shear_x  ;

		if(g->opt[i].shear_y == 1)  //  optimize shear_y? 0-no 1-yes
			x[j++] = g->im[i].cP.shear_y  ;
	}
	
	if( j != g->numParam )
		return -1;
	else
		return 0;

}



		



#define DX 3
#define DY 14

// Read Control Point Position from flag pasted into image

void getControlPoints( Image *im, controlPoint *cp )
{
	int y, x, cy,cx, bpp, r,g,b,n, nim, k,i,np;
	register unsigned char *p,*ch;
	
	
	p = *(im->data);
	bpp = im->bitsPerPixel/8;
	if( bpp == 4 )
	{
		r = 1; g = 2; b = 3;
	}
	else if( bpp == 3 )
	{		
		r = 0; g = 1; b = 2;
	}
	else
	{
		PrintError("Can't read ControlPoints from images with %d Bytes per Pixel", bpp);
		return;
	}
	
	np = 0;
	for(y=0; y<im->height; y++)
	{
		cy = y * im->bytesPerLine;
		for(x=0; x<im->width; x++)
		{
			cx = cy + bpp * x;
			if( p[ cx 			+ r ] 	== 0  	&& p[ cx 			+ g ] 	== 255 	&& p[ cx 			+ b ] 	== 0   &&
				p[ cx + bpp 	+ r ] 	== 255  && p[ cx + bpp 		+ g ] 	== 0 	&& p[ cx + bpp 		+ b ] 	== 0   &&
				p[ cx + 2*bpp 	+ r ] 	== 0  	&& p[ cx + 2*bpp 	+ g ] 	== 0 	&& p[ cx + 2*bpp 	+ b ] 	== 255 &&
				p[ cx - bpp 	+ r ] 	== 0  	&& p[ cx - bpp 		+ g ] 	== 0 	&& p[ cx - bpp 		+ b ] 	== 0 )
			{
				if(p[cx + 3*bpp + r ] 	== 0  	&& p[ cx + 3*bpp 	+ g ] 	== 255 	&& p[ cx + 3*bpp 	+ b ] 	== 255)
				{	// Control Point
					ch = &(p[cx + 4*bpp + r ]);
					n = 0;
					while( ch[0] == 255 && ch[1] == 0 && ch[2] == 0 )
					{
						n++;
						ch += bpp;
					}
					if( n >= 0 )
					{
						k = 0;
						if( cp[n].num[0] != -1 )
							k = 1;
						cp[n].x[k] = x + DX;
						cp[n].y[k] = y + DY;
						np++;
					}
				}
				else if(p[cx+3*bpp +r] 	== 255  && p[ cx + 3*bpp 	+ g ] 	== 255 	&& p[ cx + 3*bpp 	+ b ] 	== 0)
				{	// Image number
					ch = &(p[cx + 4*bpp + r ]);
					n = 0;
					while( ch[0] == 255 && ch[1] == 0 && ch[2] == 0 )
					{
						n++;
						ch += bpp;
					}
					if( n >= 0 )
					{
						nim = n;
					}
				}
			}
		}
	}
	k = 0;
	if( cp[0].num[0] != -1 )
		k = 1;
	for(i=0; i<np; i++)
		cp[i].num[k] = nim;
	

}
			
			
// Write Control Point coordinates into script 

void writeControlPoints( controlPoint *cp,char* cdesc )
{
	int i;
	char line[80];
	
	*cdesc = 0;
	for(i=0; i<NUMPTS && cp[i].num[0] != -1; i++)
	{
		sprintf( line, "c n%d N%d x%d y%d X%d Y%d\n", cp[i].num[0], cp[i].num[1], 
													   cp[i].x[0], cp[i].y[0],
													   cp[i].x[1], cp[i].y[1]);
		strcat( cdesc, line );
	}
}


void	SetStitchDefaults( struct stitchBuffer *sBuf)
{
	*sBuf->srcName 		= 0;
	*sBuf->destName 	= 0;
	sBuf->feather		= 10;			
	sBuf->colcorrect	= 0;			
	sBuf->seam			= _middle;	
}

void		SetOptDefaults( optVars *opt )
{
	opt->hfov = opt->yaw = opt->pitch = opt->roll = opt->a = opt->b = opt->c = opt->d = opt->e = opt->shear_x = opt->shear_y = 0;
}

void DoColorCorrection( Image *im1, Image *im2, int mode )
{
	double 	ColCoeff [3][2];
	int 	i;

	switch( mode )
	{
		case 0: 
			break; // no correction
		case 1: // Correct im1
			GetColCoeff( im1, im2, ColCoeff );
			ColCorrect( im1, ColCoeff );
			break; 
		case 2: // Correct im2
			GetColCoeff( im1, im2, ColCoeff );
			// Invert coefficients
			for( i = 0;  i<3;  i++)
			{
				ColCoeff[i][1] = - ColCoeff[i][1] / ColCoeff[i][0];
				ColCoeff[i][0] = 1.0/ColCoeff[i][0];
			}
			ColCorrect( im2, ColCoeff );
			break; 
		case 3: // Correct both halfs									
			GetColCoeff( im1, im2, ColCoeff );
			for(i = 0; i<3; i++)
			{
				ColCoeff[i][1] =  ColCoeff[i][1] / 2.0 ;
				ColCoeff[i][0] = (ColCoeff[i][0] + 1.0 ) / 2.0;
			}
			ColCorrect( im1, ColCoeff );
			for(i = 0; i<3; i++)
			{
				ColCoeff[i][1] = - ColCoeff[i][1] / ColCoeff[i][0];
				ColCoeff[i][0] = 1.0 / ColCoeff[i][0];
			}
			ColCorrect( im2, ColCoeff );
			break;
		default: break;
	} // switch
}


// Do some checks on Optinfo structure and reject if obviously nonsense

int CheckParams( AlignInfo *g )
{
	int i;
	int		err = -1;
	char 	*errmsg[] = {
				"No Parameters to optimize",
				"No input images",
				"No Feature Points",
				"Image width must be positive",
				"Image height must be positive",
				"Field of View must be positive",
				"Field of View must be smaller than 180 degrees in rectilinear Images",
				"Unsupported Image Format (must be 0,1,2,3 or 4)",
				"Panorama Width must be positive",
				"Panorama Height must be positive",
				"Field of View must be smaller than 180 degrees in rectilinear Panos",
				"Unsupported Panorama Format",
				"Control Point Coordinates must be positive",
				"Invalid Image Number in Control Point Descriptions"
				};

	if( g->numParam == 0 )				err = 0;
	if( g->numIm	== 0 )				err = 1;
	if( g->numPts	== 0 )				err = 2;
	
	// Check images
	
	for( i=0; i<g->numIm; i++)
	{
		if( g->im[i].width  <= 0 )		err = 3;
		if( g->im[i].height <= 0 )		err = 4;
		if( g->im[i].hfov   <= 0.0 )	err = 5;
		if( g->im[i].format == _rectilinear && g->im[i].hfov >= 180.0 )	err = 6;
		if( g->im[i].format != _rectilinear && g->im[i].format != _panorama &&
		    g->im[i].format != _fisheye_circ && g->im[i].format != _fisheye_ff && g->im[i].format != _equirectangular)
										err = 7;
	}
	
	// Check Panorama specs
	
	if( g->pano.hfov <= 0.0 )	err = 5;
	if( g->pano.width <=0 )		err = 8;
	if( g->pano.height <=0 )		err = 9;
	if( g->pano.format == _rectilinear && g->pano.hfov >= 180.0 )	err = 10;
	if( g->pano.format != _rectilinear && g->pano.format != _panorama &&
		    g->pano.format != _equirectangular ) err = 11;
	
	// Check Control Points
	
	for( i=0; i<g->numPts; i++)
	{
		if( g->cpt[i].x[0] < 0 || g->cpt[i].y[0] < 0 || g->cpt[i].x[1] < 0 || g->cpt[i].y[1] < 0 )
			err = 12;
		if( g->cpt[i].num[0] < 0 || g->cpt[i].num[0] >= g->numIm ||
			g->cpt[i].num[1] < 0 || g->cpt[i].num[1] >= g->numIm )			err = 13;
	}
	
	if( err != -1 )
	{
		PrintError( errmsg[ err ] );
		return -1;
	}
	else
		return 0;
}
			

static int		CheckMakeParams( aPrefs *aP)
{
	
	if( (aP->pano.format == _rectilinear) && (aP->pano.hfov >= 180.0) )
	{
		PrintError("Rectilinear Panorama can not have 180 or more degrees field of view.");
		return -1;
	}
	if( (aP->im.format == _rectilinear) && (aP->im.hfov >= 180.0) )
	{
		PrintError("Rectilinear Image can not have 180 or more degrees field of view.");
		return -1;
	}
	if( (aP->mode & 7) == _insert ){
		if( (aP->im.format == _fisheye_circ || 	aP->im.format == _fisheye_ff) &&
		    (aP->im.hfov > MAX_FISHEYE_FOV) ){
				PrintError("Fisheye lens processing limited to fov <= %lg", MAX_FISHEYE_FOV);
				return -1;
		}
	}

	return 0;
	
}


			

// return 0, if overlap exists, else -1
static int GetOverlapRect( PTRect *OvRect, PTRect *r1, PTRect *r2 )
{
	OvRect->left 	= max( r1->left, r2->left );
	OvRect->right	= min( r1->right, r2->right );
	OvRect->top		= max( r1->top, r2->top );
	OvRect->bottom	= min( r1->bottom, r2->bottom );
	
	if( OvRect->right > OvRect->left && OvRect->bottom > OvRect->top )
		return 0;
	else
		return -1;
}

void SetGlobalPtr( AlignInfo *p )
{
	g = p;
}

void GetControlPointCoordinates(int i, double *x, double *y, AlignInfo *gl )
{
	double		w2, h2;
	int j, n[2];

	struct 	MakeParams	mp;
	struct  fDesc 		stack[15];

	

	n[0] = gl->cpt[i].num[0];
	n[1] = gl->cpt[i].num[1];
	
	// Calculate coordinates x/y in panorama

	for(j=0; j<2; j++)
	{
		SetInvMakeParams( stack, &mp, &gl->im[ n[j] ], &gl->pano, 0 );
		
		h2 	= (double)gl->im[ n[j] ].height / 2.0 - 0.5;
		w2	= (double)gl->im[ n[j] ].width  / 2.0 - 0.5;
		

		execute_stack( 	(double)gl->cpt[i].x[j] - w2,		// cartesian x-coordinate src
						(double)gl->cpt[i].y[j] - h2,		// cartesian y-coordinate src
						&x[j], &y[j], stack);

		h2 	= (double)gl->pano.height / 2.0 - 0.5;
		w2	= (double)gl->pano.width  / 2.0 - 0.5;
		x[j] += w2;
		y[j] += h2;
	}
}


int AddEdgePoints( AlignInfo *gl )
{
	void *tmp;

	tmp =  realloc( gl->cpt, (gl->numPts+4) * sizeof( controlPoint ) );
	if( tmp == NULL )	return -1;
	gl->numPts+=4; gl->cpt = (controlPoint*)tmp; 

	gl->cpt[gl->numPts-4].num[0] = 0;
	gl->cpt[gl->numPts-4].num[1] = 1;
	gl->cpt[gl->numPts-4].x[0] = gl->cpt[gl->numPts-4].x[1] = -9.0 * (double)gl->pano.width;
	gl->cpt[gl->numPts-4].y[0] = gl->cpt[gl->numPts-4].y[1] = -9.0 * (double)gl->pano.height;

	gl->cpt[gl->numPts-3].num[0] = 0;
	gl->cpt[gl->numPts-3].num[1] = 1;
	gl->cpt[gl->numPts-3].x[0] = gl->cpt[gl->numPts-3].x[1] = 10.0 * (double)gl->pano.width;
	gl->cpt[gl->numPts-3].y[0] = gl->cpt[gl->numPts-3].y[1] = -9.0 * (double)gl->pano.height;

	gl->cpt[gl->numPts-2].num[0] = 0;
	gl->cpt[gl->numPts-2].num[1] = 1;
	gl->cpt[gl->numPts-2].x[0] = gl->cpt[gl->numPts-2].x[1] = -9.0 * (double)gl->pano.width;
	gl->cpt[gl->numPts-2].y[0] = gl->cpt[gl->numPts-2].y[1] = 10.0 * (double)gl->pano.height;

	gl->cpt[gl->numPts-1].num[0] = 0;
	gl->cpt[gl->numPts-1].num[1] = 1;
	gl->cpt[gl->numPts-1].x[0] = gl->cpt[gl->numPts-1].x[1] = 10.0 * (double)gl->pano.width;
	gl->cpt[gl->numPts-1].y[0] = gl->cpt[gl->numPts-1].y[1] = 10.0 * (double)gl->pano.height;

	return 0;
}


