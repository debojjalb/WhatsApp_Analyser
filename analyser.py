import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np
import plotly.express as px
import seaborn as sns



add_selectbox = st.sidebar.selectbox(
    "Do you Mind Adding a review?",
    ("Yes", "Later")
)
if add_selectbox == 'Yes' :
    st.sidebar.markdown("## Click here for your feedback -> https://bit.ly/2AMcvvc")
else :
    st.sidebar.markdown("## I hope you would change your mind till the end of thsi project :)")

st.sidebar.markdown("# Contact Details:")
st.sidebar.markdown("## Debojjal Bagchi")
st.sidebar.markdown("#### Undergrad at IISc")
st.sidebar.markdown("## LinkedIn :")
st.sidebar.markdown("#### https://www.linkedin.com/in/debojjal-bagchi/")
st.sidebar.markdown("## Email :")
st.sidebar.markdown("#### debojjalb@gmail.com")

st.sidebar.markdown("##### Kindly Close this sidepanel to enjoy full experience")





st.markdown("# WhatsApp Chat Analyser")
st.markdown("## Independent Project By Debojjal Bagchi during Summer 2020")
st.markdown(" #### © Debojjal Bagchi")
st.markdown(" #### Licensed under the [4-Clause Berkeley Software Distribution (BSD) License](https://github.com/debojjalb/WhatsApp_Analyser/blob/master/LICENSE), University of California")

.
st.markdown("###### *This analyses only last 40,000 msgs. Works Best with Indiviual Chats & Small Group Chats. View on Desktop for Complete Experience. Built on Python Using Pandas, Streamlit & Plotly")

st.markdown("#### Send me your feedback, Ways to improve or any other suggestions at debojjalb@gmail.com")

import warnings
warnings.filterwarnings("ignore")

import pandas as pd

import streamlit as st
import os
st.markdown("# Upload file to be Analysed")
st.markdown("##### How to? - Open the WhatsApp Chat/Group Chat you want to analyse. Click on the 3 dots -> Export -> Without Media. Save this file in your files folder or preferrably Desktop if using PC & Upload here. YOUR PHONE LANGUAGE MUST BE ENGLISH(UK) WHILE EXPORTING THE CHAT")
st.markdown("###### *NO USER DATA IS TAKEN BY ME WHATSOEVER")


filename = st.file_uploader("Choose a TXT file", type="txt")

#filename = file_selector()
#st.markdown("##### Kindly Confirm file name is <something>.txt before proceeding")
#st.write('You selected `%s`' % filename)

st.markdown("### Please wait for 30s for the Chat to be Analysed Completely. Starting from now you will find lots of interactive graphs. Move the slider whenever necessary to get better Visualisations")

user_input = filename

df=pd.read_csv(user_input,header=None,error_bad_lines=False,encoding='utf8')
df= df.drop(0)
df.columns=['Date', 'Chat']
Message= df["Chat"].str.split("-", n = 1, expand = True)
df['Date']=df['Date'].str.replace(",","")
df['Time']=Message[0]
df['Text']=Message[1]
Message1= df["Text"].str.split(":", n = 1, expand = True)
df['Text']=Message1[1]
df['Name']=Message1[0]
df=df.drop(columns=['Chat'])
df['Text']=df['Text'].str.lower()
df['Text'] = df['Text'].str.replace('<media omitted>','MediaShared')
df['Text'] = df['Text'].str.replace('this message was deleted','DeletedMsg')

Time= df["Time"].str.split(":", n = 1, expand = True)
df['Hour']=Time[0]
df['Min']=Time[1]
Date = df["Date"].str.split("/", n = 2, expand = True)
df['Day'] = Date[0]
df['Month']= Date[1]
df['Year']= Date[2]
df = df.drop(['Min'], axis = 1)

data_columns = ['Hour', 'Day' ,'Month' , 'Year' ]
num_df = (df.drop(data_columns, axis=1)
         .join(df[data_columns].apply(pd.to_numeric, errors='coerce')))

num_df = num_df[num_df[data_columns].notnull().all(axis=1)]
df = num_df

df_new= df[['Day', 'Month', 'Year', 'Hour']].astype("int")
df_new[['Text','Name','Time', 'Date' ]] = df[['Text','Name', 'Time' , 'Date']]
df = df_new
df['counts'] = 1

#INTRODUCTION

days = df["Date"].nunique()
msgs = df["counts"].count()
average = msgs/days

df['totalwords'] = df['Text'].str.count(' ') + 1
words = df['totalwords'].sum()
avg_words = words/days

df['NAME_Count'] = df['Text'].str.len()
letters  = df['NAME_Count'].sum()
avg_let = letters/days

sub ='MediaShared'
  
df["Indexes"]= df["Text"].str.find(sub)
media = pd.DataFrame(df.groupby(['Indexes'])['counts' ].count())
media.reset_index(inplace=True)
media = media[media['Indexes']==1]

media = media['counts']

st.markdown('# 1. Some Introductory Analysis ')

st.markdown("### 1. Total Number of Chats : %i" %(days))
st.markdown("### 2. Number of Days Chatted : %i" %(msgs))
st.markdown('### 3. Average Number of Chats per Day: %i' %(average))
st.markdown('### 4. Total Number of Words : %i'%(words))
st.markdown('### 5. Total Number of Letters : %i'%(letters))
st.markdown('### 6. Average Number of Words per day : %i' %(avg_words))
st.markdown('### 7. Average Number of Letters per day : %i' %(avg_let))
#st.markdown('## Total Number of Media Shared : %i' %(media))
#st.markdown('## Average Number of Media Shared per day : %i')




#GRAPH 1
st.markdown('# 2. Which hour had how many messages? ')
hour = st.slider("Move the slider to select the hour", 0 , 23, 21)
data1 = df[df['Hour'] == hour]
data1['count'] = 1


st.markdown(" ## Who talks how much between %i:00 and %i:00?"  % ( hour, (hour+1) ))
st.markdown(" ##### Are the legends covering a part of the chart or hindering your Visualisations? Try using the Wide Mode. You will find a 'full screen' option just above the right top end of the legends if you hover yor mose pointer there. Click on that & the problem should be solved" )
st.markdown("#### Double Click on persons name from legends to select to compare. You can select any number of persons to compare!")

fig = px.pie(data1, values='count', names='Name', color_discrete_sequence=px.colors.sequential.RdBu, title='Percentage of Chat')
fig.update_layout(legend=dict( x=1 , font=dict(
           family="sans-serif",
           size=8,
           color="black"
    ),
)
)

st.write(fig)

if st.checkbox("Show filtered Chats for Graph 1", False) :
    st.write(data1[['Name', 'Date', 'Time', 'Text' ]])

#GRAPH 2

st.markdown(" # 3. Chats over various days of the week - TREND")


import calendar
import datetime

def findDay(day, month, year):
    born = datetime.date(year, month, day)
    return born.strftime("%A")

df['Week'] = df.apply(lambda row : findDay(row['Day'],
                     row['Month'], row['Year']), axis = 1)

data2 = pd.DataFrame(df.groupby(['Week', 'Name' ])['counts' ].count())
data2.index.name = 'Week'
data2.reset_index(inplace=True)

fig2 = px.line(data2, x="Week", y="counts", color = 'Name' )
fig2.update_layout(plot_bgcolor='white')
st.markdown(" ##### Are the legends covering a part of the chart or hindering your Visualisations? Try using the Wide Mode. You will find a 'full screen' option just above the right top end of the legends if you hover yor mose pointer there. Click on that & the problem should be solved" )
st.markdown("#### Double Click on persons name from legends to select to compare. You can select any number of persons to compare!")

#fig2.update_layout(legend_orientation="h")
fig2.update_layout(legend=dict(x=0, y=-2))
fig2['layout']['xaxis']['side'] = 'top'

st.write(fig2)

#Data 2
data2.sort_values('counts', inplace=True, ascending=False)

st.markdown(" ## The top Chats are wrt week: ")
user_input_2 = st.slider("Move the slider to select the how many top results you want", 0 , 20, 5)
st.markdown(" ## The top %i Chats wrt day of the week are: " %(int(user_input_2)))

st.write(data2.head(int(user_input_2)))


#GRAPH 3

st.markdown(" # 4. Chats over various time of the day - TREND")


import calendar
import datetime


data3 = pd.DataFrame(df.groupby(['Hour', 'Name' ])['counts' ].count())
data3.index.name = 'Hour'
data3.reset_index(inplace=True)

fig3 = px.bar(data3, x="Hour", y="counts", color = 'Name' )
fig3.update_layout(plot_bgcolor='white')
st.markdown(" ##### Are the legends covering a part of the chart or hindering your Visualisations? Try using the Wide Mode. You will find a 'full screen' option just above the right top end of the legends if you hover yor mose pointer there. Click on that & the problem should be solved" )
st.markdown("#### Double Click on persons name from legends to select to compare. You can select any number of persons to compare!")

fig3.update_layout(legend_orientation="h")
fig3['layout']['xaxis']['side'] = 'top'

st.write(fig3)

#Data 3

data3.sort_values('counts', inplace=True, ascending=False)

st.markdown(" ## The top Chats are wrt hours in a day are: ")
#user_input_2 = st.text_input("Top how many?", 5)
user_input_3 = st.slider("Move the slider to select the how many top results you want!", 0 , 40 , 15)
st.markdown(" ## The top %i Chats wrt hour of the day are: " %(int(user_input_3)))

st.write(data3.head(int(user_input_3)))

#Graph 4

st.markdown(" # 5. Percentage of Media Shared ")


sub ='MediaShared'
  
df["Indexes"]= df["Text"].str.find(sub)
data4 = pd.DataFrame(df.groupby(['Indexes','Name'])['counts' ].count())
df = df.drop('Indexes', axis=1)
data4.reset_index(inplace=True)
data4 = data4[data4['Indexes']==1]

fig4 = px.pie(data4, values='counts', names='Name', color_discrete_sequence=px.colors.sequential.RdBu, title='Percentage of Media Shared')
st.markdown(" ##### Are the legends covering a part of the chart or hindering your Visualisations? Try using the Wide Mode. You will find a 'full screen' option just above the right top end of the legends if you hover yor mose pointer there. Click on that & the problem should be solved" )
st.markdown("#### Double Click on persons name from legends to select to compare. You can select any number of persons to compare!")

st.write(fig4)

if st.checkbox("Show filtered Chats for Graph 4", False) :
    st.write(data4)
    

#Graph 5
st.markdown(" # 6. Maximum Words Used in Chat ")

user_input_5 = st.slider("Move the slider to select the how many top results you want :)", 0 , 50,  10)

from collections import Counter
#Common = Counter(" ".join((df["Text"]).astype("string")))
#Common= (Common.split()).most_common(user_input_5)

f1 =  (df.set_index('Name')['Text']
.str.lower()
.str.split(expand=True)
.stack()
.groupby(level=0)
.value_counts()
.groupby(level=0)
.head(user_input_5)
.rename_axis(('Name','Word'))
.reset_index(name='Frequence'))

f1 = f1.head(user_input_5)

#data5 = pd.DataFrame(Common, columns= ['Word','Frequence'])
#st.write(f1)
data5 = f1
select = st.selectbox('See Visualisation as', ['Pie Chart', 'Bar Chart'])

if select == 'Pie Chart' :
    fig5 = px.pie(data5, values='Frequence', names='Word', title='', color_discrete_sequence=px.colors.sequential.RdBu)
    fig5.update_layout(plot_bgcolor='white')
    st.markdown(" ##### Are the legends covering a part of the chart or hindering your Visualisations? Try using the Wide Mode. You will find a 'full screen' option just above the right top end of the legends if you hover yor mose pointer there. Click on that & the problem should be solved" )
    st.markdown("#### Double Click on persons name from legends to select to compare. You can select any number of persons to compare!")

    st.write(fig5)
elif select == 'Bar Chart' :

    fig6 = px.bar(data5, x="Word", y="Frequence" , color_discrete_sequence=px.colors.sequential.RdBu)
    fig6.update_layout(plot_bgcolor='white')
    st.markdown(" ##### Are the legends covering a part of the chart or hindering your Visualisations? Try using the Wide Mode. You will find a 'full screen' option just above the right top end of the legends if you hover yor mose pointer there. Click on that & the problem should be solved" )
    st.markdown("#### Double Click on persons name from legends to select to compare. You can select any number of persons to caompare!")

    st.write(fig6)

if st.checkbox("Show a list of top %i words used in chat & their frequencies & Sender's Name" %user_input_5, False) :
    st.write(data5)


#Trend
st.markdown(" # 7. Growth of Number of messages over time ")

data6 = pd.DataFrame(df.groupby(['Date'])['counts'].count())
data6.reset_index(inplace=True)
data6['Date'] = data6['Date'].astype('datetime64[ns]')
data6["Date"] = pd.to_datetime(data6["Date"] , format='%d/%m/%y')
df['Date'] = df['Date'].astype('datetime64[ns]')


data6 = data6.sort_values(by="Date")


fig7 = px.line(data6, x='Date', y='counts', title='Time Series with Rangeslider')
fig7.update_xaxes(rangeslider_visible=True)
fig7.update_layout(plot_bgcolor='white')

st.markdown(" ##### Are the legends covering a part of the chart or hindering your Visualisations? Try using the Wide Mode. You will find a 'full screen' option just above the right top end of the legends if you hover yor mose pointer there. Click on that & the problem should be solved" )

st.write(fig7)


#AnotherTrend

st.markdown(" # 8. Trend of Chat for Each Person ")


data7 = pd.DataFrame(df.groupby(['Date','Name'])['counts'].count())
data7.reset_index(inplace=True)
data7["TMP"] = data7.index.values
data7 = data7[data7.TMP.notnull()]
data7.drop(["TMP"], axis=1, inplace=True)
data7['Date'] = data7['Date'].astype('datetime64[ns]')

data7["Date"] = pd.to_datetime(data7["Date"] , format='%d/%m/%y')
data7 = data7.sort_values(by="Date")

select = st.selectbox('Select Person', data7['Name'].unique())


i = select
# individual
#for i in data7['Name'].unique() :
st.markdown(" ## Trend of Chat over time for person name : %s " %i)
figi = px.line(data7[data7['Name'] == i], x='Date', y='counts', title='Counts for Number of Chats',)
figi.update_xaxes(rangeslider_visible=True)
    #figi.update_layout(plot_bgcolor='white')
st.markdown(" ##### Are the legends covering a part of the chart or hindering your Visualisations? Try using the Wide Mode. You will find a 'full screen' option just above the right top end of the legends if you hover yor mose pointer there. Click on that & the problem should be solved" )

st.write(figi)



# Predictions!
    
st.markdown(" # 9. Chat Trend Predictions ")
st.markdown(" ### Note : I am still tuning this model. Rf regressor was used. Suggest a better source code to me at debojjalb@gmail.com (Due Credits will be given)")

X_train = data6['Date']
y_train = data6['counts']

import datetime

a = datetime.datetime.today()
numdays = 100
dateList = []
for x in range (0, numdays):
    dateList.append(a + datetime.timedelta(days = x))
    
df2 = pd.DataFrame(dateList, columns = ['Date'])
df2['Date'] = df2['Date'].map(lambda x: str(x)[:-16])
X_test = df2['Date']

import datetime as dt

#X_train = X_train.map(dt.datetime.toordinal)
X_test = pd.to_datetime(X_test)
#X_test = X_test.map(dt.datetime.toordinal)

X_test = X_test.values.reshape(-1, 1)
X_train = X_train.values.reshape(-1, 1)

from sklearn.ensemble import RandomForestRegressor


rfc = RandomForestRegressor()
rfc.fit(X_train, y_train)
y_pred=rfc.predict(X_test)


DATA = pd.DataFrame([df2['Date'], y_pred]).T
DATA.columns = ['Date', 'Counts']

fig10 = px.line(DATA, x='Date', y='Counts', title='RandomForestRegressor Predictions for next 100 days')
fig10.update_xaxes(rangeslider_visible=True)
fig10.update_layout(plot_bgcolor='white')

st.markdown(" ##### Are the legends covering a part of the chart or hindering your Visualisations? Try using the Wide Mode. You will find a 'full screen' option just above the right top end of the legends if you hover yor mose pointer there. Click on that & the problem should be solved" )
st.write(fig10)


# FeedBack

st.markdown(" # 10. Feedback")

st.markdown("## Plese provide your feedback")

st.markdown("### Visit Left Side Panel Please")

st.markdown(" ##### © Debojjal Bagchi")
st.markdown(" ##### Licensed under the [4-Clause Berkeley Software Distribution (BSD) License](https://github.com/debojjalb/WhatsApp_Analyser/blob/master/LICENSE), University of California")

