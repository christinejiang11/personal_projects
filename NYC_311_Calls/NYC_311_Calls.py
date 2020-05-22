#!/usr/bin/env python
# coding: utf-8

# In[1]:


#ideas for importing data:
#1. get data in batches: export to a list, export list to a csv, import csv for each new run - do data preprocessing
#2. get data in batches, process each batch, export to a csv, import csv with specified datatypes
#3. how big is a "big" dataset? whats the general limit for what i can export to work with on my computer?
#4. multithreading vs. multiprocessing?
#5. export data to a database? SQLalchemy
#6. work directly in colab, then download and export to github?
#
#call with ori:
#options:
#1. download locally, wait for however long it takes, large = when it stops fitting in ram
#mem = temp working space, fast to access but can't store much, might be erased; disk = long term storage
#goal = get something done quickly
#embarrassingly parallel problems: processes don't need to talk to each other

#2. BQ: data stays in google servers, might have to pay eventually, if you need to do a lot of processing / really large datasets
#take the processed results
#use api to query from BQ

#3. spark cluster on AWS

#keep raw data - quickly recover

#python stores each num as an obj; each obj has overhead involved (methods, value, ref count)
#numpy stores all of these values as one obj


# In[2]:


###QUESTIONS
#how to run bokeh server - will interactions work?
#how to create multi-index heatmap
#absolute paths?
#running on heap cluster?

#general questions:
#how to deal with large datasets?
#interactive sliders with bokeh server? i'm very confused about bokeh server
#code review?


# ## To Do:
# - multiprocessing or multithreading to get more data
# - save data to csv, use pandas to read in and indicate appropriate data types
# - data processing: add time, drop nulls
# - reduce memory
# - create cleaner categories; create function to indicate any new categories that need to be coded
#

# ## Import dataset, reduce memory

# In[53]:


import pandas as pd
import numpy as np
import requests
from sodapy import Socrata
import os
import seaborn as sns
from matplotlib import pyplot as plt
import sys
from fuzzywuzzy import process
from fuzzywuzzy import fuzz
import pygsheets
from datetime import datetime, date, time
import json

from bokeh.plotting import figure, output_file, output_notebook, show, save, reset_output, gmap
from bokeh.models import ColumnDataSource, GMapOptions, HoverTool, BasicTicker, ColorBar, LinearColorMapper, PrintfTickFormatter, Panel, Tabs
from bokeh.palettes import Spectral6, all_palettes, brewer
from bokeh.transform import factor_cmap, transform, linear_cmap
from bokeh.layouts import column, row, layout, WidgetBox
from bokeh.io import output_file, show, curdoc
from bokeh.sampledata.unemployment1948 import data
from bokeh.models import CheckboxButtonGroup, CheckboxGroup, TextInput, Slider, DateRangeSlider
from bokeh.application import Application
from bokeh.application.handlers import FunctionHandler

import warnings
warnings.simplefilter("ignore")

os.getcwd()
pd.options.display.max_columns=50


# In[4]:


#define parameters for endpoint, dataset, and app token
data_url = 'data.cityofnewyork.us'
dataset = 'erm2-nwe9'
app_token = 'dM7DDeidbAmgtydtJVV1epbiU'

#sets up the connection, need application token to override throttling limits
#username and password only required for creating or modifying data
client = Socrata(data_url, app_token)
client.timeout = 6000

#count number of records in desired dataset
record_count = client.get(dataset, select='count(*)', where="created_date >='2020-02-01'")
record_count


# In[5]:


def get_data(chunk_size=100000, total_rows=500000):
    start = 0
    results=[]

    #paginate through dataset in sets of 10000 to get all records since 2019
    while True:
        print(f'{start} rows retrieved')
        results.extend(client.get(dataset,where="created_date >= '2020-02-01'",
                                  limit=chunk_size, offset=start))
        start += chunk_size
        if start > total_rows:
            break
    return results


# In[6]:


# orig_results = get_data()
# orig_df = pd.DataFrame(orig_results)
# path = '/Users/christinejiang/Documents/Python/data/'
# orig_df.to_csv(path+'311_data.csv', index=False)


# In[7]:


def reduce_memory(df):
    """improved version of memory reduction function. uses pd.to_numeric to downcast types;
    also considers whether there are few enough unique values to warrant use of category vs. object."""
    orig_size = df.memory_usage().sum()/1024**2
    dtypes = df.dtypes.astype(str).unique()
    converted_float = pd.DataFrame()
    converted_int = pd.DataFrame()
    converted_obj = pd.DataFrame()
    converted_misc = pd.DataFrame()

    #convert floats
    selected_float = df.select_dtypes(include='float')
    converted_float = selected_float.apply(pd.to_numeric, downcast='float')
    float_size = selected_float.memory_usage().sum()/1024**2
    converted_float_size = converted_float.memory_usage().sum()/1024**2
    print(f'floats: {float_size:.2f} reduced to {converted_float_size:.2f} MB')

    #convert ints
    selected_int = df.select_dtypes(include='integer')
    converted_int = selected_int.apply(pd.to_numeric, downcast='integer')
    int_size = selected_int.memory_usage().sum()/1024**2
    converted_int_size = converted_int.memory_usage().sum()/1024**2
    print(f'ints: {int_size:.2f} reduced to {converted_int_size:.2f} MB')

    #convert objects / categories
    selected_object = df.select_dtypes(include=['object', 'category'])
    obj_size = selected_object.memory_usage().sum()/1024**2
    for col in selected_object.columns:
        count = len(selected_object[col])
        unique = len(selected_object[col].astype(str).unique())
        if unique < count/2:
            converted_obj[col] = selected_object[col].astype(str).astype('category')
        else:
            converted_obj[col] = selected_object[col].astype(str)
    converted_obj_size = converted_obj.memory_usage().sum()/1024**2
    print(f'object: {obj_size:.2f} reduced to {converted_obj_size:.2f} MB')

    #join floats, ints, and objects / categories
    float_int = converted_float.join(converted_int)
    float_int_obj = float_int.join(converted_obj)

    #for any columns of any other type, keep them the same and join to the converted dataframe
    no_change_cols = [x for x in df.columns if x not in float_int_obj.columns]
    reduced_df = float_int_obj.join(df[no_change_cols])

    #re-order columns to appear in original order
    reduced_df = reduced_df[df.columns]
    reduced_size = reduced_df.memory_usage().sum()/1024**2
    print(f'final df: {orig_size:.2f} reduced to {reduced_size:.2f} MB, {(orig_size-reduced_size)/orig_size*100:.1f}% reduction')
    return reduced_df


# In[8]:


#olddf - most dtypes are objects, some are ints and some are floats
#BUT olddf datatypes get changed inplace after running reduce_mem_usage! becomes mostly categorical
#use parse date
path = '/Users/christinejiang/Documents/Python/data/'
olddf = pd.read_csv(path+'311_data.csv')

#newdf - most dtypes are categorical, some ints and floats
newdf = reduce_memory(olddf)


# In[9]:


# #need to fix this - categorical blanks are not showing up as blanks
# plt.rc('figure',figsize=(15,4))
# #display(sns.heatmap(olddf.isnull()))
# display(sns.heatmap(newdf.isnull()))


# ## Data Processing
# - need to create a function that will show the new, unmapped categories for each new imported dataset
# - need to indicate column types upon reading in with pandas, use the reduce memory function after preprocessing

# In[10]:


##ingest all datetime as datetimes
#leave strings as objects, integers as ints, floats as floats
#specifically cast columns you want to change to diff dtype
#use categorical if you're going to be grouping by (cardinality<1000)


# In[11]:


def preprocess_df(df):
    '''drop columns that are mostly blank, create time-related columns from date columns'''
    #drop mostly blank columns
    cleandf = df.drop(['intersection_street_1',
               'intersection_street_2',
               'taxi_company_borough',
               'taxi_pick_up_location',
               'bridge_highway_name',
               'bridge_highway_direction',
               'road_ramp',
               'bridge_highway_segment',
               'landmark',
               'vehicle_type'], axis=1, inplace=False)

    #convert date related columns to datetime
    for col in cleandf.columns:
        if 'date' in col:
            cleandf[col] = pd.to_datetime(cleandf[col])

    #created time-related columns from date columns
    #look into vectorized functions
    #note: .apply not vectorized, slow
    cleandf['created_mdy'] = [x.strftime('%Y-%m-%d') for x in cleandf['created_date']]
    cleandf['created_year'] = [x.year for x in cleandf['created_date']]
    cleandf['created_month'] = [x.strftime('%b') for x in cleandf['created_date']]
    cleandf['created_day'] = [x.day for x in cleandf['created_date']]
    cleandf['created_weekday'] = [x.strftime('%a') for x in cleandf['created_date']]
    cleandf['created_week'] = [x.week for x in cleandf['created_date']]
    cleandf['created_hour'] = [x.hour for x in cleandf['created_date']]
    cleandf['closed_hour'] = [x.hour for x in cleandf['closed_date']]
    cleandf['time_to_close'] = cleandf['closed_date'] - cleandf['created_date']
    cleandf['count'] = 1
    cleandf['hour'] = [x.strftime('%I %p') for x in cleandf['created_date']]
    return cleandf


# In[12]:


def export_col_values(workbook, df, columns):
    """for a list of columns, creates a new sheet for each column and exports unique values and their counts to that sheet"""
    for col in columns:
        value_counts = df[col].value_counts()
        counts_df = pd.DataFrame(value_counts).reset_index()
        #was getting an error with using fillna for categorical column, need to cast to string
        counts_df['index'] = counts_df['index'].astype(str)
        try:
            worksheet = workbook.worksheet_by_title(col)
        except Exception:
            #ensure the error is in regards to missing the worksheet
            print(sys.exc_info())
            workbook.add_worksheet(col)
            worksheet = workbook.worksheet_by_title(col)
        worksheet.set_dataframe(counts_df, start='A1')
    print(f'{len(columns)} sets of column values exported.')

def get_valid_names(workbook, columns, start='D1'):
    """extracts the valid names manually entered by the user in column D of the workbook"""
    valid_names = {}
    for col in columns:
        worksheet = workbook.worksheet_by_title(col)
        valid_matrix = worksheet.get_values(start='D1', end='D100')
        valid_names[col] = [v[0] for v in valid_matrix]
    return valid_names

def fuzzy_match(value):
    """returns the best match for each column; fuzzy match score of < 90 will return 'Other'"""
    match = process.extract(query=value, choices=valid_names[col], limit=1)
    if match[0][1] < 90:
        return 'Other'
    else:
        return match[0][0]


# In[13]:


#use pygsheets to connect to workbook where we will export unique column values
client = pygsheets.authorize(service_account_file=path+'client_secret.json')
workbook = client.open('311_data_cleaning')
columns = ['agency_name','complaint_type','descriptor','location_type','city']

#export unique column values and their counts
export_col_values(workbook, newdf, columns)


# In[14]:


#COMMENT OUT THIS CELL IF DOING CATEGORY REVIEW
#get dictionary of lists with valid names for each column
valid_names = get_valid_names(workbook, columns, start='D1')

#fuzzy match each of the columns to the available values
for col in columns:
    newdf['cleaned_'+col] = newdf[col].apply(fuzzy_match)


# In[15]:


#preprocess dataset to remove mostly null columns and create date columns
processed_df = preprocess_df(newdf)
clean_processed = reduce_memory(processed_df)
nyc_311_calls = clean_processed.copy()


# ## Bokeh Visualizations
# - function to create bar graphs
# - function to create tables
# - function to create heatmaps
# - function to create geo map
# - function to create call outs
# - add visualizations and sliders to dashboard

# ### Heatmap with Controls

# In[16]:


def heatmap_tab(nyc_311_calls, x, y, value, title=None, x_ticks=None, y_ticks=None, exclude=None):

    def make_dataset(x, y, value):
        pivot = nyc_311_calls.pivot_table(values=value, index=x, columns=y, aggfunc='sum')
        pivot.columns = pivot.columns.astype(str)
        pivot.index = pivot.index.astype(str)
        if exclude:
            try:
                pivot = pivot.drop(exclude)
                print(f'{exclude} dropped from index.')
            except KeyError:
                print(f'{exclude} dropped from columns.')
                pivot = pivot.drop(exclude, axis=1)
            except:
                print('Exclusion does not exist in index or columns.')
        df_pivot = pd.DataFrame(pivot.stack()).reset_index()
        df_pivot.columns = ['y','x','value']
        return ColumnDataSource(df_pivot)

    def make_plot(src):
        #output_notebook()
        colors = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"]
        mapper = LinearColorMapper(palette=colors, low=src.data['value'].min(), high=src.data['value'].max())

        if x_ticks:
            x_range = x_ticks
        else:
            x_range = sorted(list(set(src.data['x'])))

        if y_ticks:
            y_range = y_ticks
        else:
            y_range = sorted(list(set(src.data['y'])))

        p = figure(plot_width=900, plot_height=500, x_range=x_range,
                   y_range = y_range, title = title)
        p.rect(x='x', y='y', width=1, height=1, source=src, line_color='white', fill_color=transform('value', mapper))
        color_bar = ColorBar(color_mapper=mapper, location=(0,0), ticker=BasicTicker(desired_num_ticks=len(colors)))
        p.add_layout(color_bar, 'right')
        hover = HoverTool(tooltips=[(x,'@x'),(y,'@y'), (value,'@value')])
        p.add_tools(hover)
        p.grid.visible=False
        p.axis.axis_line_color = None
        p.axis.major_tick_line_color = None
        p.axis.major_label_text_font_size = "11px"
        p.axis.major_label_standoff = 0
        p.xaxis.major_label_orientation = 1.0
        return p

    #def update(attr, old, new):
    #    DateRangeSlider.

    date_range_slider = DateRangeSlider(title="Date Range: ", start=date(2020, 1, 1), end=date.today(), value=(date(2020, 1, 1), date.today()), step=1)
    #date_range_slider.on_change('value',update)

    src = make_dataset(x, y, value)
    plot = make_plot(src)

    controls=WidgetBox(date_range_slider)
    layout = column(controls, plot)
    tab = Panel(child=layout, title='Heatmap')
    return tab


# ### Geomap with Controls

# In[54]:


#make_dataset() format specific data to be displayed
#make_plot() draw plot with specified data
#update() update plot based on user selections

def geomap_tab(nyc_311_calls):
    #function to make dataset for geomap based on number of points to display
    #make_dataset function has the params that can be used to change the underlying data
    #params: neighborhood, number of points
    def make_dataset(boroughs, display_num=10000):
        borough_filter = nyc_311_calls[nyc_311_calls['borough'].isin(boroughs)]
        borough_filter['lat_round'] = round(borough_filter['latitude'],3)
        borough_filter['lon_round'] = round(borough_filter['longitude'],3)
        latlon_df = pd.DataFrame(borough_filter.groupby(['lat_round', 'lon_round'])['count'].sum()).reset_index()
        latlon_df['sizes'] = latlon_df['count']/latlon_df['count'].max()*50
        latlon_sorted = latlon_df.sort_values('sizes', ascending=False)
        latlon_display = latlon_sorted[:display_num]
        return ColumnDataSource(latlon_display)

    def make_plot(src):
        #output_notebook()
        #function to generate the plot given a ColumnDataSource with specific parameters
        with open(path+'client_secret.json') as f:
            data = json.load(f)
        api_key = data['google_api_key']
        map_options = GMapOptions(lat=40.76, lng=-73.95, map_type='roadmap', zoom=11)
        call_map = gmap(api_key, map_options, title='NYC 311 Calls')
        call_map.circle(x='lon_round', y='lat_round', size='sizes', source=src, fill_alpha = 0.7,
                        fill_color='salmon', line_color='red')

        hover = HoverTool(tooltips=[('longitude','@lon_round'),('latitude','@lat_round'), ('count','@count')])
        call_map.add_tools(hover)
        return call_map

    def update(attr, old, new):
        #function to update the underlying dataset given any change in interface widgets (checkbox, slider)
        boroughs_to_plot = [borough_selection.labels[i] for i in borough_selection.active]
        num_display = display_selection.value
        new_src = make_dataset(boroughs_to_plot, display_num=num_display)
        src.data.update(new_src.data)

    available_boroughs = list(set(nyc_311_calls['borough']))
    available_boroughs.sort()

    borough_selection = CheckboxGroup(labels=available_boroughs, active=[0,1,2,3,4])
    borough_selection.on_change('active', update)

    display_selection = Slider(start=1000, end=5000, step=100, value=5000, title='Display Num')
    display_selection.on_change('value', update)

    date_range_slider = DateRangeSlider(title="Date Range: ", start=date(2020, 1, 1), end=date.today(), value=(date(2020, 1, 1), date.today()), step=1)

    initial_boroughs = [borough_selection.labels[i] for i in borough_selection.active]

    src = make_dataset(initial_boroughs, display_num=display_selection.value)
    p = make_plot(src)
    controls = WidgetBox(date_range_slider, borough_selection, display_selection)
    layout = row(controls, p)
    tab = Panel(child=layout, title='Geomap')
    tabs = Tabs(tabs=[tab])
    return tab


# ### Bar Graph with Controls

# In[19]:


def bargraph_tab(nyc_311_calls, categories=['cleaned_descriptor']):

    def make_dataset(category, value, num_categories):
        grouped_df = pd.DataFrame(nyc_311_calls.groupby(category)[value].sum()).reset_index()
        grouped_df[category] = grouped_df[category].astype(str)
        grouped_df = grouped_df.sort_values(value, ascending = False)[:num_categories]
        colors = ['#550b1d', '#933b41', '#cc7878', '#ddb7b1', '#dfccce', '#e2e2e2', '#c9d9d3', '#a5bab7', '#75968f']
        try:
            grouped_df['color'] = colors[:num_categories]
        except KeyError:
            print('Too many categories selected; select up to 9 categories to display')
        return ColumnDataSource(grouped_df)

    def make_plot(src):
        output_file = ('test.html')
        p = figure(y_range = list(reversed(src.data[category])), plot_height=300, plot_width=600,
                  x_axis_label='Calls')

#         mapper = LinearColorMapper(palette=colors, low=src.data['value'].min(), high=src.data['value'].max())
        p.hbar(y=category, right=value, source=src, height=0.8, color='color', hover_color='white')
        category_value = f'@{category}'
        value_value = f'@{value}'
        hover = HoverTool(tooltips=[(category, category_value), (value, value_value)])
        p.add_tools(hover)
        return p

    def update():
        pass

    value='count'
    num_categories=8

    date_range_slider = DateRangeSlider(title="Date Range: ", start=date(2020, 1, 1), end=date.today(), value=(date(2020, 1, 1), date.today()), step=1)
    controls = WidgetBox(date_range_slider)

    plots = {}
    for i, category in enumerate(categories):
        src = make_dataset(category, value, num_categories)
        p = make_plot(src)
        plots[i] = p

    row_1 = row(plots[2], plots[3])
    row_2 = row(plots[0], plots[1])
    layout = column(controls, row_1, row_2)
    tab = Panel(child=layout, title='Bargraph')
    tabs = Tabs(tabs=[tab])
    return tab


# In[65]:


time = ['11 PM', '10 PM', '09 PM', '08 PM', '07 PM', '06 PM', '05 PM', '04 PM', '03 PM', '02 PM', '01 PM', '12 PM', '11 AM', '10 AM', '09 AM', '08 AM', '07 AM', '06 AM', '05 AM', '04 AM', '03 AM', '02 AM', '01 AM', '12 AM']

heatmap = heatmap_tab(nyc_311_calls, 'hour','created_weekday','count', 'Calls by Day and Hour', y_ticks=time,
                     x_ticks=['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'])

heatmap2 = heatmap_tab(nyc_311_calls, 'cleaned_descriptor','created_week','count', 'Calls by Type', exclude='Other')

geomap = geomap_tab(nyc_311_calls)

bargraph = bargraph_tab(nyc_311_calls, ['cleaned_agency_name','cleaned_descriptor','cleaned_city', 'cleaned_location_type'])

tabs = Tabs(tabs=[bargraph, heatmap, heatmap2, geomap])
output_file('dashboard.html')
#show(tabs)
curdoc().add_root(tabs)


# In[22]:


# class BokehHeatmap:
#     def __init__(self, x, y, value, exclude=None):
#         self.x = x
#         self.y = y
#         self.value = value
#         self.exclude = exclude
#         self.pivot = self.make_dataset()[0]
#         self.df_pivot = self.make_dataset()[1]

#     def make_dataset(self):
#         value = self.value
#         x=self.x
#         y=self.y
#         exclude = self.exclude
#         pivot = clean_processed.pivot_table(values=value,index=x,columns=y,aggfunc='sum')
#         if exclude:
#             try:
#                 pivot = pivot.drop(exclude, axis=0)
#                 print(f'{exclude} dropped from x-axis.')
#             except KeyError:
#                 pivot = pivot.drop(exclude, axis=1)
#                 print(f'{exclude} dropped from y-axis.')
#             except:
#                 print('Exclusion does not exist in index or columns.')
#         pivot.columns = pivot.columns.astype(str)
#         pivot.index = pivot.index.astype(str)
#         list(pivot.index)
#         df_pivot = pd.DataFrame(pivot.stack(), columns=[value]).reset_index()
#         return pivot, df_pivot

#     def make_plot(self, title=None, x_ticks=None, y_ticks=None):
#         #reset_output()
#         #output_notebook()
#         df_pivot = self.df_pivot
#         pivot = self.pivot
#         x = self.x
#         y = self.y
#         value = self.value
#         source = ColumnDataSource(df_pivot)

#         if title:
#             graph_title = title
#         else:
#             graph_title= f'{value} by {x} and {y}'

#         if x_ticks:
#             x_range = x_ticks
#         else:
#             x_range = list(pivot.index)

#         if y_ticks:
#             y_range = y_ticks
#         else:
#             y_range = list((pivot.columns))

#         colors = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"]
#         mapper = LinearColorMapper(palette=colors, low=df_pivot[value].min(), high=df_pivot[value].max())

#         p = figure(plot_width=900, plot_height=500, title=graph_title,
#                    x_range=x_range, y_range=y_range,
#                    toolbar_location=None, x_axis_location="below")

#         p.rect(x=x, y=y, width=1, height=1, source=source, line_color='white', fill_color=transform(value, mapper))
#         color_bar = ColorBar(color_mapper=mapper, location=(0, 0),
#                              ticker=BasicTicker(desired_num_ticks=len(colors)))
#         p.add_layout(color_bar, 'right')

#         value_value = f'@{value}'
#         x_value = f'@{x}'
#         y_value = f'@{y}'

#         hover = HoverTool(tooltips=[(x, x_value),(y, y_value), (value, value_value)])
#         p.add_tools(hover)
#         p.grid.visible=False

#         p.axis.axis_line_color = None
#         p.axis.major_tick_line_color = None
#         p.axis.major_label_text_font_size = "11px"
#         p.axis.major_label_standoff = 0
#         p.xaxis.major_label_orientation = 1.0
#         return p

# class BokehGeoMap():
#     def __init__(self, color='salmon', display_number=5000):
#         self.color = color
#         self.display_number = display_number
#         self.dataset = self.make_dataset()

#     def make_dataset(self):
#         clean_processed['round_lat'] = round(clean_processed['latitude'],3)
#         clean_processed['round_lon'] = round(clean_processed['longitude'],3)
#         latlon_df = pd.DataFrame(clean_processed.groupby(['round_lat', 'round_lon'])['count'].sum()).reset_index()
#         latlon_df.sort_values('count', ascending=False)
#         latlon_df['scaled'] = (latlon_df['count']/latlon_df['count'].max()*50)
#         latlon_sorted = latlon_df.sort_values('scaled',ascending=False)
#         return latlon_sorted

#     def make_plot(self):
#         #output_file('maps.html')
#         color = self.color
#         display_number = self.display_number
#         dataset = self.dataset
#         sample_df = dataset[:display_number]

#         latitudes = sample_df['round_lat']
#         longitudes = sample_df['round_lon']
#         sizes = sample_df['scaled']

#         with open(path+'client_secret.json') as f:
#             data = json.load(f)
#         api_key = data["google_api_key"]

#         map_options = GMapOptions(lat=40.76, lng=-73.95, map_type="roadmap", zoom=11)
#         call_map = gmap(api_key, map_options, title="NYC 311 Calls")
#         source = ColumnDataSource(data=dict(lat=latitudes,lon=longitudes, sizes = sizes))
#         call_map.circle(x="lon", y="lat", size='sizes', fill_color=color, fill_alpha=0.7, line_color=color, source=source)
#         return call_map


#class BokehBarGraph:
#     def __init__(self, category, value, num_categories=9):
#         self.category = category
#         self.value = value
#         self.num_categories = num_categories
#         self.dataset = self.make_dataset()

#     def make_dataset(self):
#         category = self.category
#         value = self.value
#         num_categories = self.num_categories
#         grouped_df = pd.DataFrame(clean_processed.groupby(category)[value].sum()).reset_index()[0:num_categories]
#         grouped_df[category] = grouped_df[category].astype(str)
#         grouped_df = grouped_df.sort_values(value, ascending=True)
#         try:
#             grouped_df['color'] = brewer['YlGnBu'][num_categories]
#         except KeyError:
#             print('Too many categories selected; select up to 9 categories to display.')
#         return grouped_df

#     def make_plot(self):
#         category = self.category
#         value = self.value
#         dataset = self.dataset
#         #output_file('bargraph.html')
#         source = ColumnDataSource(dataset)

#         bargraph = figure(y_range=dataset[category], plot_height = 300, plot_width=600, background_fill_color="#000000",
#                    x_axis_label=value, y_axis_label=category)
#         bargraph.grid.visible=False

# #         color_map = factor_cmap(field_name=category, palette = Spectral6,
# #                                 factors=value_count[category].unique())
#         bargraph.hbar(y=category,right=value,source=source, height=.8, color='color', hover_color='white')
#         category_value = f'@{category}'
#         value_value = f'@{value}'
#         hover = HoverTool(tooltips=[(category,category_value),(value, value_value)])
#         bargraph.add_tools(hover)
#         return bargraph
#         #show(p1)


# In[23]:


# ##line graph
# clean_processed['cleaned_descriptor'] = clean_processed['cleaned_descriptor'].astype(str)
# clean_processed['created_mdy'] = clean_processed['created_mdy'].astype(str)
# pivot = clean_processed.pivot_table(values='count', index='created_mdy', columns='cleaned_descriptor', aggfunc=sum).reset_index()
# source = ColumnDataSource(pivot)

# reset_output()
# linegraph = figure(x_range=pivot['created_mdy'])
# for col in pivot.columns:
#     linegraph.line(x='created_mdy', y = col, source=source, color='red', legend='test')
# factor_cmap(field_name='test', palette='Viridis256', factors=pivot.columns)

# hover = HoverTool(tooltips=[("Date","@created_mdy"),("Ceiling","@Ceiling")])
# linegraph.add_tools(hover)
# show(linegraph)


# In[ ]:





# In[ ]:
