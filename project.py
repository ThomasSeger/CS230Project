"""
Name:       Thomas Seger
CS230:      Section 2
Data:       Cambridge Property Database
URL:        Link to your web application online

Description:
This program runs some data analysis on residential housing in Cambridge, first showing a map of all properties listed
in the dataFrame, then gives functionality to see relationships between a couple variables and the cost of a house,
and finishes with a statement of a regression analysis that tells the user the most influential variables on the cost
of the house.

"""
import streamlit as st
import pydeck as pdk
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Import data
df_housing = pd.read_csv("Cambridge_Property_Database_FY2022_8000_sample.csv", index_col="PID")
# pd.set_option("display.max_rows", None, "display.max_columns", None, 'display.width', None, 'max_colwidth', None)

# Cleaning the data
# Remove unnecessary columns
df_housing = df_housing.drop(['GISID', 'BldgNum', 'Address', 'Unit', 'StateClassCode', 'PropertyClass', 'Zoning',
                              'Map/Lot', 'YearOfAssessment', 'TaxDistrict', 'ResidentialExemption', 'BuildingValue',
                              'LandValue', 'SalePrice', 'Book/Page', 'SaleDate', 'PreviousAssessedValue', 'Owner_Name',
                              'Owner_CoOwnerName', 'Owner_Address', 'Owner_Address2', 'Owner_City', 'Owner_State',
                              'Owner_Zip', 'Exterior_WallHeight', 'Exterior_FloorLocation', 'Exterior_View',
                              'Interior_Flooring', 'Interior_Layout', 'Interior_LaundryInUnit', 'Systems_Plumbing',
                              'Parking_Garage'], axis=1)
# Taking only Residential houses (contains the least NaN values)
# Drop rows with NaN values in filter column
df_housing.dropna(axis=0, how='any', thresh=1, subset=['Exterior_occupancy'], inplace=True)
# Filter by Exterior_occupancy ending in 'RES'
df_housing = df_housing[df_housing['Exterior_occupancy'].str.endswith('RES')]
df_housing = df_housing.loc[(df_housing['AssessedValue'] < 10000000) & (df_housing['Condition_YearBuilt'] > 1700) &
                            (df_housing['PropertyTaxAmount'] > 0), :]
# Remove blank rows
df_housing = df_housing.dropna()

# Intro and Description
st.title('Welcome to my CS230 Project')
st.subheader('Thomas Seger')
st.write('This website displays data analytics of residential properties in Cambridge, MA.')
st.write('Click on the sidebar to begin.')

# Sidebar
st.sidebar.header('Data Analysis will include:')

# Map
# Change Latitude and Longitude to latitude and longitude
df_housing.rename(columns={"Latitude": "latitude", "Longitude": "longitude"}, inplace=True)

st.sidebar.subheader('1. An interactive map of properties')
selected_map = st.sidebar.radio('Select a map to view:', ['', 'Simple', 'Advanced'])
if selected_map == 'Simple':
    st.title('Simple Map')
    st.map(df_housing)
elif selected_map == 'Advanced':
    st.title('Scatterplot Map')
    view_state = pdk.ViewState(
        latitude=df_housing['latitude'].mean(),
        longitude=df_housing['longitude'].mean(),
        zoom=12,
        pitch=0
    )
    layer1 = pdk.Layer(
        type='ScatterplotLayer',
        data=df_housing,
        get_position='[longitude, latitude]',
        get_radius=25,
        get_color=[0, 0, 255],
        pickable=True
    )
    layer2 = pdk.Layer(
        type='ScatterplotLayer',
        data=df_housing,
        get_position='[longitude, latitude]',
        get_radius=25,
        get_color=[0, 0, 255],
        pickable=True
    )
    tool_tip = {
        'html': 'Price:<br/> <b>{AssessedValue}</b>',
        'style': {'backgroundColor': 'blue', 'color': 'white'}
    }
    map = pdk.Deck(
        map_style='mapbox://styles/mapbox/outdoors-v11',
        initial_view_state=view_state,
        layers=[layer1, layer2],
        tooltip=tool_tip
    )
    st.pydeck_chart(map)

# Graphs
st.sidebar.subheader('2. Graphs displaying relationships between residential home prices and different characteristics')
# Variables and their names
variables = ['LandArea', 'Exterior_Style', 'Exterior_occupancy', 'Exterior_WallType', 'Exterior_RoofMaterial',
             'Interior_LivingArea', 'Interior_NumUnits', 'Interior_TotalRooms', 'Interior_Bedrooms',
             'Interior_FullBaths', 'Systems_HeatFuel', 'Systems_CentralAir', 'Condition_YearBuilt',
             'Condition_OverallCondition', 'Parking_Open', 'PropertyTaxAmount']
names = ['Land Area', 'Exterior Style', 'House Type', 'Wall Type', 'Roof Material', 'Square Footage',
         'Total Units', 'Total Rooms', 'Bedrooms', 'Bathrooms', 'Heating Type', 'Central Air',
         'Year Built', 'Overall Condition', 'Parking Spots', 'Property Taxes']

selection = st.sidebar.multiselect('Select the variables you would like to compare with the value of the home.',
                                   names)
scatter_selection = []
count = 0
for i in names:
    if i in selection:
        scatter_selection.append(variables[count])
    count += 1
y = list(df_housing.loc[:, 'AssessedValue'])
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
for i, c in zip(scatter_selection, colors):
    x = list(df_housing.loc[:, i])
    plt.scatter(x, y, color=c, s=50)

plt.xlabel(selection)
plt.ylabel('Price')
plt.title('Chosen Variables vs. Price')
plt.legend(labels=selection)
plt.savefig(f'plot_{i}.png')

if len(scatter_selection) != 0:
    st.image(f'plot_{i}.png')

# Regression
st.sidebar.subheader('3. A linear regression model to determine the extent of the relationship.')
# Initialize x values, y values, and regression model
reg_variables = ['LandArea', 'Interior_LivingArea', 'Interior_NumUnits', 'Interior_TotalRooms', 'Interior_Bedrooms',
                 'Interior_FullBaths', 'Condition_YearBuilt', 'Parking_Open', 'PropertyTaxAmount']
df_x = pd.DataFrame(df_housing, columns=reg_variables)
df_y = pd.DataFrame(df_housing['AssessedValue'])
reg = linear_model.LinearRegression()

# Split the data into 67% training and 33% testing data
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.33, random_state=42)
reg.fit(x_train, y_train)  # Train the model
y_pred = reg.predict(x_test)  # predictions of our test data
results = {}
for i in range(len(reg_variables)):
    results[reg_variables[i]] = reg.coef_[0][i]

choice = st.sidebar.radio('View Regression Results?', ('No', 'Yes'))
if choice == 'Yes':
    st.write('Here are the coefficients for each variable used:')
    st.write(results)
    st.write(f'Results found that the variable that increased property value the most was the number of bathrooms, '
             f'increasing property value by over $7,000 per bedroom.')
    st.write(f'The variable that actually decreased property value was the number of bedrooms in the house!')
    st.error('Note: Using the Mean Squared Error Method, this regression model is inaccurate, potential issues could '
             'be lack of variables used and outliers in the data.')
