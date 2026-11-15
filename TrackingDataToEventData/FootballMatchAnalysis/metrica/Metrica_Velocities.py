#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 14:52:19 2020

Module for measuring player velocities, smoothed using a Savitzky-Golay filter, with Metrica tracking data.

Data can be found at: https://github.com/metrica-sports/sample-data

@author: Laurie Shaw (@EightyFivePoint)

"""
import numpy as np
import scipy.signal as signal

def _smooth_velocity_segment(values, filter_, window, polyorder):
    values = values.copy()
    values_for_filter = values.interpolate(limit_direction="both")
    valid_count = values_for_filter.dropna().shape[0]

    if valid_count < 2:
        return values

    if filter_ == 'Savitzky-Golay':
        segment_window = min(window, len(values_for_filter))
        if segment_window % 2 == 0:
            segment_window -= 1
        if segment_window <= polyorder:
            return values
        return signal.savgol_filter(values_for_filter, window_length=segment_window, polyorder=polyorder)

    if filter_ == 'moving average':
        segment_window = min(window, valid_count)
        ma_window = np.ones(segment_window) / segment_window
        return np.convolve(values_for_filter, ma_window, mode='same')

    return values

def calc_player_velocities(team, smoothing=True, filter_='Savitzky-Golay', window=7, polyorder=1, maxspeed = 12):
    """ calc_player_velocities( tracking_data )
    
    Calculate player velocities in x & y direciton, and total player speed at each timestamp of the tracking data
    
    Parameters
    -----------
        team: the tracking DataFrame for home or away team
        smoothing: boolean variable that determines whether velocity measures are smoothed. Default is True.
        filter: type of filter to use when smoothing the velocities. Default is Savitzky-Golay, which fits a polynomial of order 'polyorder' to the data within each window
        window: smoothing window size in # of frames
        polyorder: order of the polynomial for the Savitzky-Golay filter. Default is 1 - a linear fit to the velcoity, so gradient is the acceleration
        maxspeed: the maximum speed that a player can realisitically achieve (in meters/second). Speed measures that exceed maxspeed are tagged as outliers and set to NaN. 
        
    Returrns
    -----------
       team : the tracking DataFrame with columns for speed in the x & y direction and total speed added

    """
    # remove any velocity data already in the dataframe
    team = remove_player_velocities(team)
    
    # Get the player ids
    player_ids = np.unique( [ c[:-2] for c in team.columns if c[:4] in ['Home','Away'] ] )

    # Calculate the timestep from one frame to the next. Should always be 0.04 within the same half
    dt = team['Time [s]'].diff()
    
    # Smooth each period independently. Generated clips often contain only one
    # period, and short clips may be smaller than the default smoothing window.
    period_groups = [group.index for _, group in team.groupby("Period", sort=False)]
    
    # estimate velocities for players in team
    for player in player_ids: # cycle through players individually
        # difference player positions in timestep dt to get unsmoothed estimate of velicity
        vx = team[player+"_x"].diff() / dt
        vy = team[player+"_y"].diff() / dt

        if maxspeed>0:
            # remove unsmoothed data points that exceed the maximum speed (these are most likely position errors)
            raw_speed = np.sqrt( vx**2 + vy**2 )
            vx[ raw_speed>maxspeed ] = np.nan
            vy[ raw_speed>maxspeed ] = np.nan
            
        if smoothing:
            for period_idx in period_groups:
                vx.loc[period_idx] = _smooth_velocity_segment(vx.loc[period_idx], filter_, window, polyorder)
                vy.loc[period_idx] = _smooth_velocity_segment(vy.loc[period_idx], filter_, window, polyorder)
                
        
        # put player speed in x,y direction, and total speed back in the data frame
        team[player + "_vx"] = vx
        team[player + "_vy"] = vy
        team[player + "_speed"] = np.sqrt( vx**2 + vy**2 )

    return team

def remove_player_velocities(team):
    # remove player velocoties and acceleeration measures that are already in the 'team' dataframe
    columns = [c for c in team.columns if c.split('_')[-1] in ['vx','vy','ax','ay','speed','acceleration']] # Get the player ids
    team = team.drop(columns=columns)
    return team
    
