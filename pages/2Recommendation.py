import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Judul aplikasi
st.title("Analisis Performa Siswa dan Sistem Rekomendasi")
st.sidebar.image("Logo_Universitas_Jambi-removebg-preview.png", width= 150,caption="Universitas Jambi")
# Mengunggah file CSV data siswa
uploaded_file = st.file_uploader("student_data.csv", type=["csv"])

if uploaded_file is not None:
    # Membaca data dari file CSV
    df = pd.read_csv(uploaded_file)

    # Menampilkan beberapa data untuk preview
    st.write("### Data Siswa")
    st.write(df.head())

    # Menampilkan statistik deskriptif
    st.write("### Statistik Deskriptif")
    st.write(df.describe())

    # Visualisasi distribusi nilai rata-rata
    st.write("### Distribusi Rata-rata Nilai")
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Rata-rata Nilai'], kde=True, color="blue")
    plt.title("Distribusi Rata-rata Nilai Siswa")
    st.pyplot(plt)

    # Visualisasi performa terhadap kelulusan
    st.write("### Hubungan antara Rata-rata Nilai dan Kelulusan")
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Status Kelulusan', y='Rata-rata Nilai', data=df)
    plt.title("Box Plot Rata-rata Nilai berdasarkan Status Kelulusan")
    st.pyplot(plt)

    # Memproses data untuk model
    st.write("### Sistem Rekomendasi Kelulusan Fisika")

    # Memetakan status kelulusan ke biner (1 = Lulus, 0 = Gagal)
    df['Status Kelulusan'] = df['Status Kelulusan'].map({'Lulus': 1, 'Gagal': 0})

    # Definisikan fitur (nilai teori, nilai praktik, tugas) dan target (status kelulusan)
    X = df[['Nilai Teori', 'Nilai Praktik', 'Tugas']]  # Fitur
    y = df['Status Kelulusan']  # Target

    # Membagi data menjadi data latih dan uji
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Membuat model Logistic Regression
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Melakukan prediksi terhadap data uji
    y_pred = model.predict(X_test)

    # Menampilkan akurasi model
    st.write(f"### Akurasi Model: {accuracy_score(y_test, y_pred):.2f}")

    # Menampilkan confusion matrix
    st.write("### Confusion Matrix")
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    st.pyplot(plt)

    # Menampilkan classification report
    st.write("### Classification Report")
    st.text(classification_report(y_test, y_pred))

    # Rekomendasi berdasarkan prediksi model
    st.write("### Prediksi Kelulusan:")

    # Membuat prediksi berdasarkan input nilai siswa yang baru
    nilai_teori = st.number_input("Masukkan Nilai Teori", min_value=0, max_value=100)
    nilai_praktik = st.number_input("Masukkan Nilai Praktik", min_value=0, max_value=100)
    nilai_tugas = st.number_input("Masukkan Nilai Tugas", min_value=0, max_value=100)

    if st.button("Cek Prediksi Kelulusan:"):
        new_data = pd.DataFrame({
            'Nilai Teori': [nilai_teori],
            'Nilai Praktik': [nilai_praktik],
            'Tugas': [nilai_tugas]
        })

        prediksi_kelulusan = model.predict(new_data)[0]

        if prediksi_kelulusan == 1:
            st.success("Siswa diprediksi LULUS dalam matapelajaran Fisika.")
        else:
            st.error("Siswa diprediksi GAGAL dalam matapelajaran Fisika. Rekomendasi: Siswa perlu memperbaiki pemahaman konsep dasar Fisika.")

data = {
    "Miskonsepsi": [
        "Gaya Gravitasi Bumi hanya berpengaruh di Bumi",
        "Kecepatan dan percepatan selalu searah",
        "Gaya aksi-reaksi saling meniadakan",
        "Energi tidak bisa berubah bentuk",
        "Hukum Newton hanya berlaku untuk benda diam"
    ],
    "Materi Remedial": [
        "Pelajari lebih lanjut tentang hukum gravitasi universal, yang menjelaskan bahwa gaya gravitasi bekerja di mana saja di alam semesta.",
        "Kecepatan adalah besaran vektor, dan arah kecepatan tidak selalu searah dengan percepatan. Lihat contoh gerak melingkar.",
        "Gaya aksi dan reaksi bekerja pada dua benda yang berbeda, jadi tidak saling meniadakan. Pahami contoh interaksi gaya dalam hukum ketiga Newton.",
        "Energi bisa berubah bentuk, misalnya energi kinetik bisa berubah menjadi energi potensial. Pelajari konsep kekekalan energi.",
        "Hukum Newton berlaku untuk semua benda, baik diam maupun bergerak. Pahami aplikasi hukum Newton dalam berbagai kondisi gerak."
    ]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# User input: Memilih miskonsepsi
miskonsepsi_input = st.selectbox("Pilih Miskonsepsi yang Dialami Siswa:", df["Miskonsepsi"].unique())

# Menemukan materi remedial yang sesuai
rekomendasi = df[df["Miskonsepsi"] == miskonsepsi_input]["Materi Remedial"].values[0]

# Menampilkan rekomendasi remedial
st.write("### Rekomendasi Remedial:")
st.write(rekomendasi)