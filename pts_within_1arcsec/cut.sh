#!/bin/bash

set -e

galid=0
cat gals.reg | while read -r gal; 
   do
	  input_img='../../../eFEDS/img_0520_12346_excl_grp.fits'
	  output_img=cut_0520_${galid}.fits
          
	  dmcopy infile="${input_img}[pos=${gal}]" \
		  outfile=${output_img} clob+
          echo ${galid} "done"
	  let galid=${galid}+1
   done
