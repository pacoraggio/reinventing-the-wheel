import pandas as pd
import numpy as np
import altair as alt
from altair import datum #Needed for subsetting (transforming data)

from vega_datasets import data
import json


def plot_q25(df, points_list, cut_point = -40):

    if(len(points_list) == 2):
        p1 = points_list[0]
        p2 = points_list[1]
    else:
        p1 = points_list[0]
        p2 = points_list[1]
        p3 = points_list[2]
        

    df['category'] = np.where(df['x'] < cut_point, 'Below 1st Quartile', 'Above 1st Quartile')


    points = (alt
              .Chart(df)
              .mark_point(size = 220, 
                          filled=True,
                          stroke='black',
                          strokeWidth=1)
              .encode(
                  x = alt.X('x'),
                  color=alt.Color('category',
                                          scale=alt.Scale(domain=['Below 1st Quartile', 'Above 1st Quartile'],
                                                          range=['#17becf', '#f58518'])
                                         )
              )
             )

    v1 = (alt
          .Chart(pd.DataFrame({'p1': [p1]}))
          .mark_rule(
              color = 'black',
              strokeWidth = 0.7
          )
          .encode(
              x = 'p1:Q',
              y = alt.value(2),
              y2 = alt.value(38)
          )
         )

    a1 = (alt
          .Chart(df)
          .mark_text(
              align='center',
              baseline='top',
              fontSize = 12,
              dx = 0,
              dy = -30
          ).encode(
              x='x',
              text=alt.Text('x', format=',.0f')
          ).transform_filter(
              (datum.x == p1)
          )
         )

    v2 = (alt
          .Chart(pd.DataFrame({'p2': [p2]}))
          .mark_rule(
              color = 'black',
              strokeWidth = 0.7
          )
          .encode(
              x = 'p2:Q',
              y = alt.value(2),
              y2 = alt.value(38)
          )
         )

    a2 = (alt
          .Chart(df)
          .mark_text(
              align='center',
              baseline='top',
              fontSize = 12,
              dx = 0,
              dy = -30
          ).encode(
              x='x',
              text=alt.Text('x', format=',.0f')
          ).transform_filter(
              (datum.x == p2)
          )
         )

    if(len(points_list) == 3):
        v3 = (alt
              .Chart(pd.DataFrame({'p2': [p3]}))
              .mark_rule(
                  color = 'black',
                  strokeWidth = 0.7
              )
              .encode(
                  x = 'p2:Q',
                  y = alt.value(2),
                  y2 = alt.value(38)
              )
             )
        
        a3 = (alt
          .Chart(pd.DataFrame({'p3': [p3]}))
          .mark_text(
              align='left',
              baseline='top',
              fontSize = 12,
              dx = 3,
              dy = -3
          ).encode(
              x='p3',
              text=alt.Text('p3', format=',.1f')
          )
         )

        chart = (points + v1 + v2 + v3 + a1 + a2+ a3).properties(
            width = 620,
            height = 40
        )
        return(chart)
    
    chart = (points + v1 + v2 + a1 + a2).properties(
        width = 620,
        height = 40
    )
    return(chart)    


def mquintile(data, p):
    """
    data: np array, list, pandas series is an array of observations
    p: float between 0 and 1, is the percentage of samples you want to consider
    """

    samples = np.sort(data)
    # n is the position of the sorted array containing the samples in the desired quantile
    n = p*(len(samples)-1)
    if n%1 == 0:
        # if the position is a whole number, return the sample at that position 
        print("Whole number")
        print(samples[int(n)])
        return(samples[int(n)])
    else:
        # is the position is an odd number, we compute the the values of the sorted array
        # for the considered position
        pos = int(n)
        # compute the adiacent samples to interpole to compute the quartile
        lower_sample = samples[pos]
        upper_sample = samples[pos+1]
        print("lower sample = {}, upper sample {}".format(lower_sample, upper_sample))
        # compute the fraction of sample to use in the interpolation
        f = n-pos
        print("fraction = {}".format(f))
        # Finally, calculate the interpolated point representing the quantile
        quantile = lower_sample+(f * (upper_sample-lower_sample))
        print("quantile value = {}".format(quantile))
        return(quantile)
        

def sum_and_round(x):
    if np.issubdtype(x.dtype, np.floating):
        return round(x.sum(), 0)
    else:
        return x.sum()


def mean_and_round(x):
    if np.issubdtype(x.dtype, np.floating):
        return round(x.mean(), 4)
    else:
        return x.mean()


def plot_boxplot_with_points(df, obs, value, show_axis = True, title = "", show_title = True):
    obs_type = obs+':N'
    value_type = value + ':Q'

    if show_axis == True:
        points = (alt  
                  .Chart(df.sort_values(value))
                  .mark_point(size = 50, filled=True, stroke='black',
                              strokeWidth=1,color = '#9C755F')
                  .encode(
                      x = alt.X(value),
                      tooltip = [obs_type, value_type]
                  )
                 )
        
        box = (alt
               .Chart(df.sort_values(value))
               .mark_boxplot(size = 25, opacity = 0.8, color = '#76B7B2')
               .encode(
                   x = alt.X(value_type),
                   y = alt.Y('type:O', title='')
               )
              )
    
        #title = value + " by " + obs + " box plot"
        if show_title == False:
            title = ""

        if show_title == "":
            title = value + " by " + obs + " box plot"
        
        
        chart = (box + points).properties(
            title = title,
            width = 620,
            height = 80
        )
        

        return(chart)

    points = (alt  
                  .Chart(df.sort_values(value))
                  .mark_point(size = 50, filled=True, stroke='black',
                              strokeWidth=1,color = '#9C755F')
                  .encode(
                      x = alt.X(value, axis=alt.Axis(labels=False, title=None, grid=True)),
                      tooltip = [obs_type, value_type]
                  )
                 )
    box = (alt
               .Chart(df.sort_values(value))
               .mark_boxplot(size = 25, opacity = 0.8, color = '#76B7B2')
               .encode(
                   x = alt.X(value_type, axis=alt.Axis(labels=False, title=None, grid=True)),
                   y = alt.Y('type:O', title='')
               )
              )

    if title == "":
            title = value + " by " + obs + " box plot"
        
    chart = (box + points).properties(
            title = title,
            width = 620,
            height = 80
        )
    return(chart)
    