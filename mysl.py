# -*- coding: utf-8 -*-
# Copyright 2018-2019 Streamlit Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""An example of showing geographic data."""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk

# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(layout="wide")


# LAYING OUT THE TOP SECTION OF THE APP
row1_1, row1_2 = st.columns((2,3))
with row1_1:
    st.write('Select a date in January 2019')
    selected_date = st.selectbox('Date', list(range(1,6)))

with row1_2:
    hour_selected = st.slider("Select Time (3 hour)", 0, 23, 0, 3)


@st.cache(allow_output_mutation=True)
def load_data(date):
###  https://github.com/Maplub/odsample/blob/master/20190102.csv?raw=true
    url = "https://github.com/Maplub/odsample/blob/master/2019010" + str(date) + ".csv?raw=true"
    df = pd.read_csv(url, header = 0)
    lowercase = lambda x: str(x).lower()
    df.rename(lowercase, axis="columns", inplace=True)
    df.drop(['unnamed: 0'], axis=1)
    #df[DATE_TIME] = pd.to_datetime(df[DATE_TIME])
    return df
df = load_data(selected_date)
data = df


###################
# CREATING FUNCTION FOR MAPS
def mapl(data, lat, lon, zoom):
    st.write(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state={
            "latitude": lat,
            "longitude": lon,
            "zoom": zoom,
            "pitch": 50,
        },
        layers=[
            pdk.Layer(
                "HexagonLayer",
                data=data,
                get_position=["lonstartl", "latstartl"],
                radius=100,
                elevation_scale=4,
                elevation_range=[0, 1000],
                pickable=True,
                extruded=True,
            ),
        ]
    ))

def mapr(data, lat, lon, zoom):
    st.write(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state={
            "latitude": lat,
            "longitude": lon,
            "zoom": zoom,
            "pitch": 50,
        },
        layers=[
            pdk.Layer(
                "HexagonLayer",
                data=data,
                get_position=["lonstop", "latstop"],
                radius=100,
                elevation_scale=4,
                elevation_range=[0, 1000],
                pickable=True,
                extruded=True,
            ),
        ]
    ))

#data = df[df[DATE_TIME].dt.hour == hour_selected]
data['begin'] = pd.to_datetime(data['begin'])
data['finish'] = pd.to_datetime(data['finish'])

# LAYING OUT THE TOP SECTION OF THE APP
begin = "begin"
finish = "finish"
point_A = data[data[begin].dt.hour <= hour_selected+3]
point_B = data[data[finish].dt.hour <= hour_selected+3]
midpointA = (np.average(point_A["latstartl"]), np.average(point_A["lonstartl"]))
midpointB = (np.average(point_B["latstop"]), np.average(point_B["lonstop"]))


row2_1, row2_2= st.columns((1,1))
with row2_1:
    st.write('**Starting Point** ',str(selected_date),'/1/2019')#str(selected_date)
    dataA = point_A[['latstartl', 'lonstartl','begin']]
    st.dataframe(dataA)

with row2_2:
    st.write('**Terminal Point** ',str(selected_date),'/1/2019')#str(selected_date)
    dataB = point_B[['latstop','lonstop','finish']]
    st.dataframe(dataB)

row3_1, row3_2= st.columns((1,1))
with row3_1:
    st.write("**Starting Point from %i:00 to %i:00**" % (hour_selected, (hour_selected+3) % 24))
    mapl(dataA, midpointA[0], midpointA[1], 11)

with row3_2:
    st.write("**Terminal Point from %i:00 to %i:00**" % (hour_selected, (hour_selected+3) % 24))
    mapr(dataB, midpointB[0], midpointB[1], 11)


# FILTERING DATA FOR THE HISTOGRAM #START
filtered = data[
    (data[begin].dt.hour >= hour_selected) & (data[begin].dt.hour < (hour_selected + 3))
    ]

hist = np.histogram(filtered[begin].dt.minute, bins=60, range=(0, 60))[0]

chart_data = pd.DataFrame({"minute": range(60), "volume": hist})

# FILTERING DATA FOR THE HISTOGRAM #STOP
filtered = data[
    (data[finish].dt.hour >= hour_selected) & (data[finish].dt.hour < (hour_selected + 3))
    ]

hist = np.histogram(filtered[finish].dt.minute, bins=60, range=(0, 60))[0]

chart_data = pd.DataFrame({"minute": range(60), "volume": hist})

# LAYING OUT THE HISTOGRAM SECTION

st.write("")

st.write("**rest of coordinate from %i:00 to %i:00**" % (hour_selected, (hour_selected + 3) % 24))

st.altair_chart(alt.Chart(chart_data)
    .mark_area(
        interpolate='step-after',
    ).encode(
        x=alt.X("minute:Q", scale=alt.Scale(nice=False)),
        y=alt.Y("volume:Q"),
        tooltip=['minute', 'volume']
    ).configure_mark(
        opacity=0.4,
        color='red'
    ), use_container_width=True)
