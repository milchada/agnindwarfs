#images pulled from here
#https://erosita.mpe.mpg.de/edr/eROSITAObservations/Catalogues/liuT/eFEDS_c001_images/

#create source mask
ermask expimage="eFEDS_c001_clean_0d2_2d3_ExpMap.fits" \
        detmask="detmask.fits" \
        threshold1=0.01 \
        threshold2=100.

#create sensitivity map for point source detection
#method 1 - ersensmap
ersensmap expimages="eFEDS_c001_clean_0d2_2d3_ExpMap.fits" \
          bkgimages="eFEDS_c001_clean_0d2_2d3_BkgMap.fits"  \
          detmasks="detmask.fits" \
          sensimage="sensmap.fits" \
          emin="200." \
          emax="2300." \
          ecf="1.22246e+12" \
          method="FIT" \
          aper_type="BOX" \
          aper_size=4.5 \
          likemin=6. \
          detmask_flag="Y" \
          shapelet_flag="N" \
          photon_flag="N" \
          area_flag="N" \
          ext_flag="N"

#read out at positions of galaxies
#see crossmatch