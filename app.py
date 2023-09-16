import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import yfinance as yf
import datetime
from keras.models import load_model
import streamlit as st
import warnings
warnings.filterwarnings("ignore")


#Estilizar o app com css (abrir o arquivo .css)
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

start = datetime.datetime(2015, 1, 1)
end = datetime.datetime(2023, 7, 31)

st.title('📈 Preço de fechamento da previsão de tendência de ações:')
st.markdown("Ações das empresas: Apple, Google, Tesla")

user_input = st.selectbox("Selecionar ações da empresa:", ('TSLA', 'GOOGL', 'AAPL'))
df = yf.download(user_input, start, end)

st.divider()
#Describing Data
st.subheader('Data from 2015-2023')
st.write(df.describe())

st.divider()
st.markdown("### Gráfico de preço de fechamento x tempo")
tab1, tab2, tab3, tab4 = st.tabs([":chart_with_upwards_trend: Preço Fechamento:" , 
":chart_with_downwards_trend: Closing Price 100MA:", ":bar_chart: Preço Fechamento 100MA & 200MA:", ":dart: Gráfico Original vs Previsão"])

#Visualizations
with tab1:
    st.markdown('Gráfico de preço de fechamento x tempo:')
    st.write("Empresa:", user_input)
    fig = plt.figure(figsize=(12,6))
    plt.plot(df.Close)
    st.pyplot(fig)

with tab2:
    st.markdown('Gráfico de preço de fechamento x tempo 100MA:')
    st.write("Empresa:", user_input)
    ma100 = df.Close.rolling(100).mean()
    fig = plt.figure(figsize=(12,6))
    plt.plot(ma100)
    plt.plot(df.Close)
    st.pyplot(fig)

with tab3:
    st.markdown('Gráfico de preço de fechamento x tempo 100MA % 200MA')
    st.write("Empresa:", user_input)
    ma100 = df.Close.rolling(100).mean()
    ma200 = df.Close.rolling(200).mean()
    fig = plt.figure(figsize=(12,6))
    plt.plot(ma100, 'r')
    plt.plot(ma200, 'g')
    plt.plot(df.Close, 'b')
    st.pyplot(fig)    

#Splitting Data into Training and Testing

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)

#Spliting data into x_train and y_train

x_train = []
y_train = []

for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100: i])
    y_train.append(data_training_array[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

#Load my model ML
model = load_model('keras_model.h5')

#Testing Part
past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)

#Scalling
scaler = scaler.scale_  
scale_factor = 1 /scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor 


#Final graph
with tab4:
    st.subheader('Previsto vs Original')
    fig2 = plt.figure(figsize=(12,6))
    plt.plot(y_test, 'b', label = 'Original Price')
    plt.plot(y_predicted, 'r', label = 'Predicted Price')
    plt.xlabel = ('Time')
    plt.ylabel = ('Price')
    plt.legend()
    st.pyplot(fig2)         

st.markdown("")
st.markdown("")
st.markdown("")
st.image("img/logo.png", width=250)    
