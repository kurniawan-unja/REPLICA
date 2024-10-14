import streamlit as st

st.set_page_config(
    page_title="Multipage App",
    page_icon="Logo_Universitas_Jambi-removebg-preview.png"
)

st.sidebar.image("Logo_Universitas_Jambi-removebg-preview.png", width= 150,caption="Universitas Jambi")

st.title("Recommended of Personalized Learning Content Analysis (REPLICA)")
st.write("Sistem REPLICA akan menyesuaikan materi atau metode pembelajaran berdasarkan hasil analisis kinerja dan preferensi belajar siswa.",
         " Misalnya, jika siswa mengalami kesulitan dalam konsep tertentu,",
         "sistem akan merekomendasikan konten yang relevan untuk memperkuat pemahaman siswa terhadap konsep tersebut.")
st.sidebar.success("Pilih halaman di atas.")
