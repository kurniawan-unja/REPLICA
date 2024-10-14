import streamlit as st

st.title('Tim Penelitian Aplikasi REPLICA')
# Membuat 2 kolom
col1, col2, col3 = st.columns(3)

# Menampilkan konten di kolom pertama
with col1:
    st.write("Ketua Tim Peneliti")
        #st.write("Ini adalah konten di kolom pertama.")
    st.image("eko.jpg", width= 150, caption="Pradita Eko Prasetio Utomo, S.Pd., M.Cs")

# Menampilkan konten di kolom kedua
with col2:
    st.write("Anggota Tim Peneliti")
    st.image("patoni.jpg",  width= 150,caption="Haerul Pathoni., S.Pd., M.Si")

# Menampilkan konten di kolom kKetiga
with col3:
    st.write("Anggota Tim Peneliti")
    st.image("cicin.jpg",  width=150, caption="Cicyn Riantoni, S.Pd., M.Pd")