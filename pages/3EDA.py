# Import Library
import streamlit as st
import os
import time
from pydantic_settings import BaseSettings
from ydata_profiling import ProfileReport
from pycaret.classification import setup, compare_models, pull, save_model, load_model
from streamlit_pandas_profiling import st_profile_report

import pandas as pd
st.sidebar.image("Logo_Universitas_Jambi-removebg-preview.png", width= 150,caption="Universitas Jambi")

# Method atau Fungsi Save Dataset
def save_upload(uploadfile):
    with open(os.path.join("data_simpan", uploadfile.name), "wb") as f:
        f.write(uploadfile.getbuffer())
    return st.success("Berhasil Meload Dataset")


# EDA & Auto ML
st.title("App EDA & Auto Machine Learning")
st.markdown("**Upload Dataset**")
file = st.file_uploader("Upload")
if file:
    # Membaca Dataset
    df = pd.read_csv(file, index_col=None, encoding="utf-8",
                     delimiter=',', quotechar='"')
    # df.to_csv('sourcecode.csv', index=None)

    # Menampilkan isi data
    st.text("Isi Data")
    st.dataframe(df)
    save_upload(file)

    # Menampilkan data exploration
    st.text("Proses Exploration Data Analysis")
    
    profile_df = ProfileReport(df)
    st_profile_report(profile_df)

    # Pembuatan model
    st.text("Membuat Model")
    pilih = st.selectbox('Silahkan Pilih Kolom', df.columns)
    if st.button('Modeling'):
        setup(df, target=pilih)
        setup_df = pull()
        best_model = compare_models()
        compare_df = pull()

        # Menampilkan Prosess Bar
        my_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.1)
            my_bar.progress(percent_complete + 1)
        st.dataframe(compare_df)

        # Menyimpan Model
        save_model(best_model, './hasil_model/best_model')

        # Fungsi Download Model
        st.text("Download Model")
        with open('./hasil_model/best_model.pkl', 'rb') as f:
            st.download_button('Download Model', f, file_name="best_model.pkl")
