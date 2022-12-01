import streamlit as st
import altair as alt
from PIL import Image, ImageDraw
import io
import ast
import requests
import pandas as pd
import numpy as np

url_class = "https://getprediction-qejkc246ba-tl.a.run.app"


st.markdown("# Cafe ❄️")
st.sidebar.markdown("# Cafe ❄️")

labels_classes = ['MarronAVinagre',
 'Pergamino',
 'BrocadoSevero',
 'Concha',
 'Negros',
 'Normales',
 'PMordidoCortado',
 'BrocadoLeve']


image = Image.open('img/cafe.jpg')
st.image(image, caption='Cafe')


uploaded_file = st.file_uploader("Carga una imagen")
bytes_data = None
bounding_image = None
decoded = None

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    decoded = Image.open(io.BytesIO(bytes_data))
    newsize = (256, 256)
    decoded = decoded.resize(newsize)

    buf = io.BytesIO()
    decoded.save(buf, format='JPEG')
    bytes_data = buf.getvalue()
    bounding_image = decoded.copy()

    st.image(decoded, caption='Cafe')
    #print(decoded.shape)
    
if st.button('Clasificar grano de cafe'):
    
    if bytes_data is not None and decoded is not None:
        with st.spinner('Procesando la imagen...'):
            resp = requests.post(url_class, files={ 'file' : bytes_data })
        pred = resp.json()
        probs = pred['ok']
        probs = ast.literal_eval(probs)
        #print(probs, type(probs))

        "Probabilidades por grano"
        source = pd.DataFrame({'Probabilidades': probs, 'Granos': labels_classes})
    
        bar_chart = alt.Chart(source).mark_bar().encode(
            y='Probabilidades:Q',
            x='Granos:O',
        )
    
        st.altair_chart(bar_chart, use_container_width=True)

        st.balloons()

        col1, col2 = st.columns(2)
        with col1:
            st.image(decoded, caption='Cafe')
        with col2:
            label_grano = labels_classes[np.argmax(probs)]
            st.info(f'El grano es del tipo {label_grano}', icon="ℹ️")

        
            
            
    else:
        st.write('Por favor sube una image!')