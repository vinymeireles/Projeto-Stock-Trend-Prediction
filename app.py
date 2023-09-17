import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import yfinance as yf
import datetime
import streamlit as st
from streamlit_option_menu import option_menu
import warnings
warnings.filterwarnings("ignore")

#Apps
st.set_page_config(page_title="Mercado de Ações", page_icon= ":bar_chart:")

# Style
with open('style.css')as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)

#Página Home
def home():
    st.subheader('📈 Mercado de ações 2022 - 2023:')
    st.markdown("Empresas: Apple, Amazon, Google, Microsoft, Netflix e Tesla.")    
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        start = st.date_input("📆 Data Inicio", value=datetime.date(2022, 1, 1))
    with col2:    
        end = st.date_input("📆 Data Fim", value=datetime.datetime(2023, 9, 15))
    with col3:
        user_input = st.selectbox("Selecionar ações da empresa:", ('AAPL', 'AMZN','GOOGL', 'MSFT', 'NFLX', 'TSLA'))
        df = yf.download(user_input, start, end)

    st.divider()
    #Describing Data
    st.subheader(f'Dados Estatísticos no período: \n({start} - {end})')
    st.dataframe(df.describe(), use_container_width=True)

#Menu Gráfico
def graphs():

    st.divider()
    st.write("🔎 Selecionar os dados:")
    col1, col2, col3 = st.columns(3)
    with col1:
        start = st.date_input("📆 Data Inicio", value=datetime.date(2022, 1, 1))
    with col2:    
        end = st.date_input("📆 Data Fim", value=datetime.datetime(2023, 9, 15))
    with col3:
        user_input = st.selectbox("Selecionar ações da empresa:", ('AAPL', 'AMZN','GOOGL', 'MSFT', 'NFLX', 'TSLA'))
        df = yf.download(user_input, start, end)
    
    st.divider()
    st.subheader(" 📈 Gráfico de Tendência do Mercado Financeiro")
    tab1, tab2, tab3, tab4 = st.tabs([":chart_with_upwards_trend: Preço Fechamento:" , 
    ":chart_with_downwards_trend: Closing Price 100MA:", ":bar_chart: Preço Fechamento 100MA & 200MA:", ":dart: Previsão"])

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
    from keras.models import load_model

    data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
    data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])


    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0,1))

    data_training_array = scaler.fit_transform(data_training)

    #Load my model ML
    model = load_model('keras_model.h5')

    #Testing Part

    input_data = scaler.fit_transform(data_testing)

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
        st.subheader('Previsão')
        st.write("Empresa:", user_input)
        fig2 = plt.figure(figsize=(12,6))
        plt.plot(y_test, 'b', label = 'Original Price')
        plt.plot(y_predicted, 'r', label = 'Predicted Price')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(fig2)         


def contact():
    st.markdown("<h2 style='text-align: center; color: red;'>🪧 Contatos</h2>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center; color: white;'> Para desenvolvimento de novos projeto - Dashboard utilizando Inteligência Articial: Machine Learning</h6>", unsafe_allow_html=True)
    st.markdown("")
    col1, col2, col3 = st.columns(3)
    with col2:    
        st.image("img/logo.png", width=250)    
    st.divider()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.image("icons/whatsapp.png", caption="28 99918-3961", width=90)
    with col2:
        st.image("icons/gmail.png", caption="viniciusmeireles@gmail.com", width=100)
    with col3:
        st.image("icons/location.png", caption="Vitória/ES", width=90)    
    with col4:
        st.image("icons/linkedin.png",caption= "/pviniciusmeireles", width=90)

    
#Menu Horizontal   
def sideBar():

    selected = option_menu(
        None,                                        #required (None/ "Nome Menu")
        options=["Home", "Gráficos", "Contatos"],    #required  (opções do menu)   
        icons=["house", "bar-chart-fill", "envelope"],         #optional
        menu_icon="cast",                            #optional
        default_index=0,                             #optional
        orientation="horizontal",                    #muda a orientação
    ) 
    if selected=="Home":
        home()    
    if selected=="Gráficos":
        graphs()
    if selected=="Contatos":
        contact()

sideBar()