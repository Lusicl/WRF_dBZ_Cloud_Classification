# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 14:48:36 2021

@author: mrocq
"""

#############
import numpy as np  
from scipy.ndimage import label, generate_binary_structure, center_of_mass
import matplotlib.patches as patches
from numpy.linalg import eig
from wrf_convstrat import conv_strat_latlon
import wrf

#from pyart.retrieve._echo_steiner import classify as getCSmask
#############

#helper function
#takes in x and y points and returns parameters which define a ellipse that surrounds these points
#(Medioni et al. 2000;Nesbitt et al. 2006)
def get_ellipse(xs,ys):
    npts = len(xs)
    i11 = np.sum(ys**2) / npts
    i22 = np.sum(xs**2) / npts
    i12 = -np.sum(xs * ys) / npts
    tensor = [ [i11, i12], [i12,i22] ]
    eig_vals, eig_vecs = eig(tensor)
    semimajor = np.sqrt(eig_vals[0]) * 2.0
    semiminor = np.sqrt(eig_vals[1]) * 2.0
    major = semimajor * 2.0
    minor = semiminor * 2.0
    eig_vecs = eig_vecs[:,0]
    orientation = np.math.atan2(eig_vecs[1], eig_vecs[0]) - np.deg2rad(90.0)
    
    return major, minor, orientation


def PF_finder(refl, lon, lat, z, work_alt = 2., rainrate_array = None, thresh_dbz = 17., min_conv_n = 20):
    #change masked values (if there are any) to zero
    
    x = np.arange(0, np.shape(lat)[1], 1)
    y = np.arange(0, np.shape(lat)[0], 1)
    
    z1 = np.arange(500, 20001, 500)
    refl = wrf.interplevel(refl, z, z1)
    #refl = np.ma.filled(refl,0)

#get rid of negative dBZ
    #refl[refl < 0] = 0

    #get horizontal grid spacing
    #dx = np.diff(x)[0]
    dx = 3
    #calculate convective/stratiform areas using Steiner algorithm, baked into this Pyart function
    #returns an array where 0 = no reflectivity, 1 = stratiform, 2 = convection, 3 = other/unsure
    
    dbz_2km = refl[3,:,:]
    cs,cc,bkgnd = conv_strat_latlon(dbz_2km, lat, lon)

    strat_inds = cs == 0
    conv_inds  = cs > 0
    
    #index of desired working altitude
    z_ind = np.where(z1 == work_alt)[0][0]
    
    #pull out reflecitvity at desired altitude
    refl_z = refl[z_ind]
    #print (np.shape(refl_z))
    #turn x/y into 2D arrays
    x,y = np.meshgrid(x,y)
    
    #masking array for finding groups. Points will be considered contiguous if they
    #touch a pixel on any side, including diagonally
    
    label_mask = generate_binary_structure(2,2)
#calculates contiguous areas where reflectivty is higher than the given threshold
#assigns every group a unique number, and returns an array where each index is
#replaced with the number of the group it belongs to, or a zero if there was no data there
    pf_groups, n_groups = label(refl_z >= thresh_dbz, structure = label_mask)
    pf_groups_conv, n_groups_conv = label(conv_inds, structure = label_mask)
    #subtract 1 from pf_groups so that the indexing works out nicer
    pf_groups -= 1
    pf_groups_conv -= 1
    
    #calculate center of each PF
    #creates a list of (x,y) locations
    pf_locs = center_of_mass(refl_z >= thresh_dbz, pf_groups , np.arange(n_groups))
    pf_locs = [(x[int(l[0]),int(l[1])], y[int(l[0]),int(l[1])]) for l in pf_locs]
    #print (pf_locs)
    
    pf_locs_conv = center_of_mass(conv_inds, pf_groups_conv , np.arange(n_groups_conv))
    pf_locs_conv = [(x[int(l[0]),int(l[1])], y[int(l[0]),int(l[1])]) for l in pf_locs_conv]
    #print (pf_locs_conv)
    
    #create empty lists for filling in data
    area               = []
    conv_area          = []
    strat_area         = []
    mean_conv_rr       = []
    mean_strat_rr      = []
    mean_rr            = []
    mean_refl_by_alt   = []
    max_refl_by_alt    = []
    echo_top           = []
    
    ellipses = []
    all_conv_ellipses = []
    conv_ellipses = [[] for i in range(n_groups)]
    conv_ell_area = [[] for i in range(n_groups)]
    
    for group_num in np.arange(n_groups):
        
        #get indices of PF
        pf_inds = pf_groups == group_num
        print (type(pf_inds))
        print (np.shape(pf_inds))
        #print (refl[0,pf_inds])
        
        #get x and y of each point relative to the center point
        pf_xs = x[pf_inds] - pf_locs[group_num][0]
        pf_ys = y[pf_inds] - pf_locs[group_num][1]
        
        major, minor, orientation = get_ellipse(pf_xs, pf_ys)
        
        ellipses.append(patches.Ellipse(pf_locs[group_num], width = major, height = minor, 
                                  angle = np.rad2deg(orientation), facecolor = 'None', edgecolor = 'k', lw = 1.25))

#calculate ellipse for each convective feature in larger feature
        
        for conv_group_num in np.arange(n_groups_conv):
            if np.sum((pf_inds) & (pf_groups_conv == conv_group_num)) > min_conv_n:
                inds = (pf_inds) & (pf_groups_conv == conv_group_num)
                pf_xs = x[inds] - pf_locs_conv[conv_group_num][0]
                pf_ys = y[inds] - pf_locs_conv[conv_group_num][1]
                
                major, minor, orientation = get_ellipse(pf_xs, pf_ys)
                
                conv_ellipses[group_num].append(patches.Ellipse(pf_locs_conv[conv_group_num],width = major, height = minor, 
                                  angle = np.rad2deg(orientation), 
                                                                    facecolor = 'None', edgecolor = 'b', lw = 2))
                
                
                conv_ell_area[group_num].append(np.sum(inds)*dx**2)
    
        #print (conv_ell_area)
                

#calculate ellipse surrounding *all* of the convective pixels in each PF,
#whether thet are contiguous or not
        if np.sum((pf_inds) & (conv_inds)) > min_conv_n:
            inds = (pf_inds) & (conv_inds)
            pf_xs = x[inds] - pf_locs[group_num][0]
            pf_ys = y[inds] - pf_locs[group_num][1]
            
            major, minor, orientation = get_ellipse(pf_xs, pf_ys)
            
            all_conv_ellipses.append(patches.Ellipse(pf_locs[group_num],width = major, height = minor, 
                                  angle = np.rad2deg(orientation), 
                                                         facecolor = 'None', edgecolor = 'k', lw = 2))
        else:
            all_conv_ellipses.append(None)
        
        
        npix = np.sum(pf_inds)
        n_conv_pix  = np.sum((pf_inds) & (conv_inds))
        n_strat_pix  = np.sum((pf_inds) & (strat_inds))
        
        area.append(npix*dx**2) 
        conv_area.append(n_conv_pix*dx**2)
        strat_area.append(n_strat_pix*dx**2)
        #print ('got the areas!')

		#calculate rain_rate 
		#conv:  z = 130.51*RR**1.447
		#strat: z = 294.61*RR**1.548
        rain_rate = np.zeros_like(refl_z)
        
#         if rainrate_array == None:
#             rain_rate[(pf_inds) & (conv_inds)]  = 0.034*10.0**(0.0691*refl_z[(pf_inds) & (conv_inds)]) 
#             rain_rate[(pf_inds) & (strat_inds)] = 0.025*10.0**(0.0646*refl_z[(pf_inds) & (strat_inds)])

		#conv:  z = 100*RR**1.7
		#strat: z = 200*RR**1.49

        if rainrate_array == None:
            rain_rate[(pf_inds) & (conv_inds)]  = 0.0666*10.0**(0.0588*refl_z[(pf_inds) & (conv_inds)]) 
            rain_rate[(pf_inds) & (strat_inds)] = 0.0286*10.0**(0.0671*refl_z[(pf_inds) & (strat_inds)])
        else:
            rain_rate[(pf_inds) & (conv_inds)]  = rainrate_array[(pf_inds) & (conv_inds)]
            rain_rate[(pf_inds) & (strat_inds)] = rainrate_array[(pf_inds) & (strat_inds)]
        
        mean_rr.append(np.mean(rain_rate[pf_inds]))
        mean_conv_rr.append(np.mean(rain_rate[(pf_inds) & (conv_inds)]))
        mean_strat_rr.append(np.mean(rain_rate[(pf_inds) & (strat_inds)]))

		#calculate mean and maximum reflectivity in PF by altitude
        mean_refl_by_alt_pf = np.zeros(z1.shape)
        max_refl_by_alt_pf  = np.zeros(z1.shape)
        for zi in range(len(z1)):
            rain_inds = refl[zi, pf_inds] > 0
            #print (rain_inds)
            if np.sum(rain_inds) > 0:
                mean_refl_by_alt_pf[zi] = np.mean(refl[zi,pf_inds][rain_inds])
                max_refl_by_alt_pf[zi]  = np.max(refl[zi,pf_inds][rain_inds])
        
        mean_refl_by_alt.append(mean_refl_by_alt_pf)
        max_refl_by_alt.append(max_refl_by_alt_pf)

		#find highest measurable echo in PF
        echo_top_ind = len(max_refl_by_alt_pf) - np.argmax((max_refl_by_alt_pf > 0)[::-1]) - 1
        #print (len(max_refl_by_alt_pf))
        echo_top.append(z1[echo_top_ind])
    
    
    return pf_locs, ellipses, conv_ellipses, pf_locs_conv, area, conv_area, strat_area, mean_conv_rr, mean_strat_rr, mean_rr, mean_refl_by_alt, max_refl_by_alt, echo_top, conv_ell_area
