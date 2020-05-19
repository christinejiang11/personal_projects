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


# ## To Do:
# - multiprocessing or multithreading to get more data
# - save data to csv, use pandas to read in and indicate appropriate data types
# - data processing: add time, drop nulls
# - reduce memory
# - create cleaner categories; create function to indicate any new categories that need to be coded
# 

# ## Import dataset, reduce memory

# In[119]:


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
from bokeh.io import output_file, show
from bokeh.sampledata.unemployment1948 import data
from bokeh.models import CheckboxButtonGroup, CheckboxGroup, TextInput

os.getcwd()
pd.options.display.max_columns=50


# In[6]:


#define parameters for endpoint, dataset, and app token
data_url = 'data.cityofnewyork.us'
dataset = 'erm2-nwe9'
app_token = 'dM7DDeidbAmgtydtJVV1epbiU'

#sets up the connection, need application token to override throttling limits
#username and password only required for creating or modifying data
client = Socrata(data_url, app_token)
client.timeout = 6000

#count number of records in desired dataset
record_count = client.get(dataset, select='count(*)', where="created_date >='2019-11-01'")
record_count


# In[9]:


def get_data(chunk_size=100000, total_rows=1000000):
    start = 0
    results=[]

    #paginate through dataset in sets of 10000 to get all records since 2019
    while True:
        print(f'{start} rows retrieved')
        results.extend(client.get(dataset,where="created_date >= '2019-11-01'", 
                                  limit=chunk_size, offset=start))
        start += chunk_size
        if start > total_rows:
            break
    return results


# In[8]:


orig_results = get_data()
orig_df = pd.DataFrame(orig_results)
orig_df.to_csv('/Users/linyu/Documents/Python/data/311_data.csv', index=False)


# In[10]:


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


# In[12]:


#olddf - most dtypes are objects, some are ints and some are floats
#BUT olddf datatypes get changed inplace after running reduce_mem_usage! becomes mostly categorical
olddf = pd.read_csv('/Users/linyu/Documents/Python/data/311_data.csv')

#newdf - most dtypes are categorical, some ints and floats
newdf = reduce_memory(olddf)


# In[13]:


#need to fix this - categorical blanks are not showing up as blanks
plt.rc('figure',figsize=(15,4))
#display(sns.heatmap(olddf.isnull()))
display(sns.heatmap(newdf.isnull()))


# ## Data Processing
# - need to create a function that will show the new, unmapped categories for each new imported dataset
# - need to indicate column types upon reading in with pandas, use the reduce memory function after preprocessing

# In[14]:


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
    return cleandf


# In[102]:


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


# In[103]:


#use pygsheets to connect to workbook where we will export unique column values
client = pygsheets.authorize(service_account_file='/Users/linyu/Documents/Python/data/client_secret.json')
workbook = client.open('311_data_cleaning')
columns = ['agency_name','complaint_type','descriptor','location_type','city']

#export unique column values and their counts
export_col_values(workbook, newdf, columns)


# In[104]:


#COMMENT OUT THIS CELL IF DOING CATEGORY REVIEW
#get dictionary of lists with valid names for each column
valid_names = get_valid_names(workbook, columns, start='D1')

#fuzzy match each of the columns to the available values
for col in columns:
    newdf['cleaned_'+col] = newdf[col].apply(fuzzy_match)


# In[105]:


#preprocess dataset to remove mostly null columns and create date columns
processed_df = preprocess_df(newdf)
#nyc_311_calls = reduce_memory(processed_df)
clean_processed = reduce_memory(processed_df)


# ## Bokeh Visualizations
# - function to create bar graphs
# - function to create tables
# - function to create heatmaps
# - function to create geo map
# - function to create call outs
# - add visualizations and sliders to dashboard

# ### Bar Graph

# In[132]:


class BokehBarGraph:
    def __init__(self, category, value, num_categories=9):
        self.category = category
        self.value = value
        self.num_categories = num_categories
        self.dataset = self.make_dataset()
        
    def make_dataset(self):
        category = self.category
        value = self.value
        num_categories = self.num_categories
        grouped_df = pd.DataFrame(clean_processed.groupby(category)[value].sum()).reset_index()[0:num_categories]
        grouped_df[category] = grouped_df[category].astype(str)
        grouped_df = grouped_df.sort_values(value, ascending=True)
        try:
            grouped_df['color'] = brewer['YlGnBu'][num_categories]
        except KeyError:
            print('Too many categories selected; select up to 9 categories to display.')
        return grouped_df
    
    def make_plot(self):
        category = self.category
        value = self.value
        dataset = self.dataset
        #output_file('bargraph.html')
        source = ColumnDataSource(dataset)

        bargraph = figure(y_range=dataset[category], plot_height = 300, plot_width=600, background_fill_color="#000000",
                   x_axis_label=value, y_axis_label=category)
        bargraph.grid.visible=False
        
#         color_map = factor_cmap(field_name=category, palette = Spectral6, 
#                                 factors=value_count[category].unique())
        bargraph.hbar(y=category,right=value,source=source, height=.8, color='color', hover_color='white')
        category_value = f'@{category}'
        value_value = f'@{value}'
        hover = HoverTool(tooltips=[(category,category_value),(value, value_value)])
        bargraph.add_tools(hover)
        return bargraph
        #show(p1)


# ### Heatmap

# In[107]:


class BokehHeatmap:
    def __init__(self, x, y, value, exclude=None):
        self.x = x
        self.y = y
        self.value = value
        self.exclude = exclude
        self.pivot = self.make_dataset()[0]
        self.df_pivot = self.make_dataset()[1]
        
    def make_dataset(self):
        value = self.value
        x=self.x
        y=self.y
        exclude = self.exclude
        pivot = clean_processed.pivot_table(values=value,index=x,columns=y,aggfunc='sum')
        if exclude:
            try:
                pivot = pivot.drop(exclude, axis=0)
                print(f'{exclude} dropped from x-axis.')
            except KeyError:
                pivot = pivot.drop(exclude, axis=1)
                print(f'{exclude} dropped from y-axis.')
            except:
                print('Exclusion does not exist in index or columns.')
        pivot.columns = pivot.columns.astype(str)
        pivot.index = pivot.index.astype(str)
        list(pivot.index)
        df_pivot = pd.DataFrame(pivot.stack(), columns=[value]).reset_index()
        return pivot, df_pivot

    def make_plot(self, title=None, x_ticks=None, y_ticks=None):
        #reset_output()
        #output_notebook()
        df_pivot = self.df_pivot
        pivot = self.pivot
        x = self.x
        y = self.y
        value = self.value 
        source = ColumnDataSource(df_pivot)
        
        if title:
            graph_title = title
        else:
            graph_title= f'{value} by {x} and {y}'
        
        if x_ticks:
            x_range = x_ticks
        else:
            x_range = list(pivot.index)
            
        if y_ticks:
            y_range = y_ticks
        else:
            y_range = list((pivot.columns))

        colors = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"]
        mapper = LinearColorMapper(palette=colors, low=df_pivot[value].min(), high=df_pivot[value].max())

        p = figure(plot_width=900, plot_height=500, title=graph_title,
                   x_range=x_range, y_range=y_range,
                   toolbar_location=None, x_axis_location="below")

        p.rect(x=x, y=y, width=1, height=1, source=source, line_color='white', fill_color=transform(value, mapper))
        color_bar = ColorBar(color_mapper=mapper, location=(0, 0),
                             ticker=BasicTicker(desired_num_ticks=len(colors)))
        p.add_layout(color_bar, 'right')
        
        value_value = f'@{value}'
        x_value = f'@{x}'
        y_value = f'@{y}'

        hover = HoverTool(tooltips=[(x, x_value),(y, y_value), (value, value_value)])
        p.add_tools(hover)
        p.grid.visible=False

        p.axis.axis_line_color = None
        p.axis.major_tick_line_color = None
        p.axis.major_label_text_font_size = "11px"
        p.axis.major_label_standoff = 0
        p.xaxis.major_label_orientation = 1.0
        return p


# ### Geo Map

# In[134]:


class BokehGeoMap():
    def __init__(self, color='salmon', display_number=5000):
        self.color = color
        self.display_number = display_number
        self.dataset = self.make_dataset()
        
    def make_dataset(self):
        clean_processed['round_lat'] = round(clean_processed['latitude'],3)
        clean_processed['round_lon'] = round(clean_processed['longitude'],3)
        latlon_df = pd.DataFrame(clean_processed.groupby(['round_lat', 'round_lon'])['count'].sum()).reset_index()
        latlon_df.sort_values('count', ascending=False)
        latlon_df['scaled'] = (latlon_df['count']/latlon_df['count'].max()*50)
        latlon_sorted = latlon_df.sort_values('scaled',ascending=False)
        return latlon_sorted
    
    def make_plot(self):
        #output_file('maps.html')
        color = self.color
        display_number = self.display_number
        dataset = self.dataset
        sample_df = dataset[:display_number]
        
        latitudes = sample_df['round_lat']
        longitudes = sample_df['round_lon']
        sizes = sample_df['scaled']

        with open('/Users/linyu/Documents/Python/data/client_secret.json') as f:
            data = json.load(f)
        api_key = data["google_api_key"]

        map_options = GMapOptions(lat=40.76, lng=-73.95, map_type="roadmap", zoom=11)
        call_map = gmap(api_key, map_options, title="NYC 311 Calls")
        source = ColumnDataSource(data=dict(lat=latitudes,lon=longitudes, sizes = sizes))
        call_map.circle(x="lon", y="lat", size='sizes', fill_color=color, fill_alpha=0.7, line_color=color, source=source)
        return call_map


# ### Geomap with slider

# In[120]:


#make_dataset() format specific data to be displayed
#make_plot() draw plot with specified data
#update() update plot based on user selections

def geomap_tab(nyc_311_calls):
    #function to make dataset for geomap based on number of points to display
    #make_dataset function has the params that can be used to change the underlying data
    #params: neighborhood, number of points
    def make_dataset(boroughs, display_num=5000):
        borough_filter = nyc_311_calls[nyc_311_calls['borough'].isin(boroughs)]
        borough_filter['lat_round'] = round(borough_filter['latitude'],3)
        borough_filter['lon_round'] = round(borough_filter['longitude'],3)
        latlon_df = pd.DataFrame(borough_filter.groupby(['lat_round', 'lon_round'])['count'].sum()).reset_index()
        latlon_df['sizes'] = latlon_df['count']/latlon_df['count'].max()*50
        latlon_sorted = latlon_df.sort_values('sizes', ascending=False)
        latlon_display = latlon_sorted[:display_num]
        return ColumnDataSource(latlon_display)
    
    def make_plot(src):
        #function to generate the plot given a ColumnDataSource with specific parameters
        output_file('geomap_with_slider.html')
        with open('/Users/linyu/Documents/Python/data/client_secret.json') as f:
            data = json.load(f)
        api_key = data['google_api_key']
        map_options = GMapOptions(lat=40.76, lng=-73.95, map_type='roadmap', zoom=11)
        call_map = gmap(api_key, map_options, title='NYC 311 Calls')
        call_map.circle(x='lon_round', y='lat_round', size='sizes', source=src, fill_alpha = 0.7, 
                        fill_color='salmon', line_color='red')
        return call_map
    
    def update(attr, old, new):
        #function to update the underlying dataset given any change in interface widgets (checkbox, slider)
        boroughs_to_plot = [borough_selection.labels[i] for i in borough_selection.active]
        new_src = make_dataset(boroughs_to_plot, display_num=5000)
        src.data.update(new_src.data)
        
    available_boroughs = list(set(nyc_311_calls['borough']))
    available_boroughs.sort()
    
    borough_selection = CheckboxGroup(labels=available_boroughs, active=[0,1,2,3,4])
    borough_selection.on_change('active', update)
    
    initial_boroughs = [borough_selection.labels[i] for i in borough_selection.active]
    
    src = make_dataset(initial_boroughs, display_num=5000)
    p = make_plot(src)
    controls = WidgetBox(borough_selection)
    layout = row(controls, p)
    tab = Panel(child=layout, title='Geomap')
    tabs = Tabs(tabs=[tab])
    return tabs


# In[124]:


nyc_311_calls = clean_processed.copy()
test = geomap_tab(nyc_311_calls = nyc_311_calls)
show(test)


# In[126]:


available_boroughs = list(set(nyc_311_calls['borough']))
available_boroughs.sort()
borough_selection = CheckboxGroup(labels=available_boroughs, active=[0,1,2,3])
[borough_selection.labels[i] for i in borough_selection.active]


# In[127]:


def my_text_input_handler(attr, old, new):
    print("Previous label: " + old)
    print("Updated label: " + new)

text_input = TextInput(value="default", title="Label:")
text_input.on_change("value", my_text_input_handler)


# In[128]:


def make_dataset(boroughs, display_num=2000):
    borough_filter = nyc_311_calls[nyc_311_calls['city'].isin(boroughs)]
    #return borough_filter
    borough_filter['lat_round'] = round(borough_filter['latitude'],3)
    borough_filter['lon_round'] = round(borough_filter['longitude'],3)
    latlon_df = pd.DataFrame(borough_filter.groupby(['lat_round', 'lon_round'])['count'].sum()).reset_index()
    latlon_df['sizes'] = latlon_df['count']/latlon_df['count'].max()*50
    latlon_sorted = latlon_df.sort_values('sizes', ascending=False)
    latlon_display = latlon_sorted[:display_num]
    return (ColumnDataSource(latlon_display))

test = make_dataset(available_boroughs)
len(test.data['index'])

test2 = make_dataset(available_boroughs, 1000)
len(test2.data['index'])

test.data.update(test2.data)
len(test.data['index'])


# In[129]:


clean_processed['cleaned_descriptor'] = clean_processed['cleaned_descriptor'].astype(str)
clean_processed['created_mdy'] = clean_processed['created_mdy'].astype(str)
pivot = clean_processed.pivot_table(values='count', index='created_mdy', columns='cleaned_descriptor', aggfunc=sum).reset_index()
source = ColumnDataSource(pivot)

reset_output()
linegraph = figure(x_range=pivot['created_mdy'])
for col in pivot.columns:
    linegraph.line(x='created_mdy', y = col, source=source, color='red', legend='test')
factor_cmap(field_name='test', palette='Viridis256', factors=pivot.columns)

hover = HoverTool(tooltips=[("Date","@created_mdy"),("Ceiling","@Ceiling")])
linegraph.add_tools(hover)
show(linegraph)


# In[130]:


test = factor_cmap(field_name='test', palette='Viridis256', factors=pivot.columns)
test


# In[135]:


#create graphs of each kind, format into dashboard
complaint_count = BokehBarGraph(category='cleaned_descriptor', value='count').make_plot()
month_hour = BokehHeatmap(x='created_week', y='created_hour', value='count').make_plot()
geomap = BokehGeoMap().make_plot()
rowdash = row(complaint_count, month_hour)
#show(column(rowdash, month_hour, geomap))

output_file('tab_test.html')
tab1 = Panel(child=complaint_count, title='Complaint Count')
tab2 = Panel(child=month_hour, title='Month Hour')
tab3 = Panel(child=geomap, title='Geomap')

tabs = Tabs(tabs=[tab1, tab2, tab3], background='white')

show(tabs)

