import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
import plotly.express as px

from PIL import Image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns

st.title('**Analisis Big Data Untuk Mencegah Kanker Dalam Konteks Kesehatan**')

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Deskripsi", "Regresi", "Image", "Tabular", "Clustering"])

with tab1:
    st.markdown('''
        <div style="text-align: justify">
                Kanker merupakan penyakit yang muncul pada organ tubuh manusia, dengan ditandai tumbuhnya sel abnormal yang tidak bisa dikontrol sehingga dapat berpindah ke seluruh tubuh yang lain. Hal ini bisa disebabkan oleh kualitas taraf hidup seseorang antara lain, akibat perubahan fisik dan psikis. Penyakit ini tergolong beberapa jenis kategori penyakit diantaranya, kanker otak, kanker mulut, kanker nasofaring, kanker leher rahim, kanker ovarium, kanker payudara, kanker saluran pencernaan, kanker serviks, kanker kulit, kanker kolon (kanker usus besar), dan kanker kandung kemih atau ginjal. Menurut data Global Cancer Statistics (Globocan) yang dirilis oleh World Health Organization (WHO) pada tahun 2022 terdapat 19.976.499 total kasus kanker yang di mana 9.743.832 sudah dikatakan meninggal. Sedangkan untuk di Indonesia sendiri terdapat 4.647 penderita kanker dari total responden 768.635, data ini diperoleh dari survei Riset Kesehatan Dasar (Riskesdas) pada tahun 2007.
        </div><br>
    ''', unsafe_allow_html=True)
    st.markdown('''
        <div style="text-align: justify">
                Dalam konteks analisis data medis modern, big data memiliki peran penting dalam meningkatkan pemahaman kita tentang penyakit, mempercepat diagnosis, dan memperbaiki pengelolaan kesehatan. Salah satu tantangan utama dalam bidang ini adalah tumor otak, sebuah kondisi yang memerlukan pendekatan analitik yang kompleks untuk memprediksi, mendiagnosis, dan mengelompokkan pasien berdasarkan risiko dan karakteristik penyakit. Penelitian ini bertujuan untuk mengembangkan sistem analisis prediktif dan klasifikasi menggunakan data publik yang mencakup citra MRI serta beberapa data statistik lainnya terkait tumor otak. Sistem ini akan menggunakan metode ARIMA untuk memprediksi angka kematian akibat tumor otak, Convolutional Neural Network (CNN) untuk klasifikasi citra MRI, Naive Bayes untuk klasifikasi data tabular, dan K-Means untuk pengelompokan berdasarkan area tumor.
        </div><br>
    ''', unsafe_allow_html=True)
    st.markdown('''
        <div style="text-align: justify">
                Terdapat beberapa penelitian terdahulu terkait penelititan yang kami kerjakan, diantaranya penelitian terkait prediksi Wang et al. (2021) menggunakan model ARIMA untuk meramalkan tingkat kematian yang disebabkan oleh kanker terkait merokok di Qingdao, Tiongkok. Studi tersebut menunjukkan bahwa model ARIMA (2,1,0) × (3,1,0)12 menunjukkan akurasi yang luar biasa dalam prediksi jangka pendek, dengan jangka waktu satu hingga dua tahun. Namun, pada tahun ketiga, interval kepercayaan menunjukkan kecenderungan untuk melebar, sehingga mengurangi akurasi prediksi. Penurunan akurasi ini dapat dikaitkan dengan sifat inheren data yang tidak stabil dalam jangka waktu yang lama. Untuk mengatasi tantangan ini, para peneliti mengusulkan penggabungan pembaruan data secara berkala untuk meningkatkan keandalan prediksi jangka panjang. Penelitian terkait klasifikasi citra MRI kanker otak menunjukkan bahwa dengan menggunakan CNN dalam arsitektur VGG16 mencapai kinerja tertinggi dengan akurasi 93% dalam hal Identifikasi Tumor Otak Citra MRI. Penelitian terkait klasifikasi tabular menggunakan metode Naive Bayes dalam deteksi tumor yaitu deteksi tumor pada bagian otak tertentu oleh Hein et al. (2019). Data terdiri dari citra MRI pasien sejumlah 114 gambar, meliputi 24 gambar normal dan 90 terdeteksi tumor. Dari 114 gambar terdapat 221 objek setelah dilakukan proses segmentasi, 171 dari 221 objek digunakan untuk training dan 50 untuk proses testing. Hasil akhir didapat 47 dari 50 objek testing terdeteksi benar menggunakan metode Naive Bayes, sehingga akurasi total di dapat sebesar 94%. Dan yang terakhir penelitian terkait clustering efektivitas metode K-Means dalam berbagai bidang analisis data. Misalnya, studi oleh Juniar Hutagalung dan Fifin Sonata (2021) menggunakan K-Means untuk menganalisis minat nasabah pada produk asuransi, menghasilkan klaster produk unggulan yang memudahkan perusahaan dalam memilih produk yang diminati nasabah, seperti asuransi kebakaran dan kesehatan.
        </div>
    ''', unsafe_allow_html=True)

with tab2:
    # Menggunakan tema Bootstrap
    st.markdown("""
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <script src="https://kit.fontawesome.com/a076d05399.js" crossorigin="anonymous"></script>
        <style>
            body {
                background-color: #f8f9fa;
            }
            .stButton button {
                background-color: #198754;
                color: white;
                border-radius: 5px;
            }
        </style>
    """, unsafe_allow_html=True)

    # Header aplikasi
    st.markdown("""
    <div class="alert alert-primary text-center" role="alert">
        <h1><i class="fa fa-chart-bar"></i> Cancer Death Rates Dashboard</h1>
        <p>Analyze and forecast cancer death rates with ease!</p>
    </div>
    """, unsafe_allow_html=True)

    # Panduan pengguna
    st.markdown("""
    <div class="card mb-4">
        <div class="card-body">
            <h4 class="card-title">User Guidance</h4>
            <ul>
                <li>Upload a CSV file containing cancer death rates.</li>
                <li>Use the input fields to set ARIMA parameters for forecasting.</li>
                <li>Explore data visualization and forecasting insights.</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Upload file
    uploaded_file_c= st.file_uploader("Upload your CSV file", type=["csv"], key="forecasting")
    if uploaded_file_c is not None:
        data = pd.read_csv(uploaded_file_c)

        # Rename target column for easier reference
        data.rename(
            columns={"Deaths - Neoplasms - Sex: Both - Age: Age-standardized (Rate)": "Death_Rate"},
            inplace=True,
        )

        # Menampilkan data
        st.markdown("""
        <div class="card mb-4">
            <div class="card-body">
                <h5 class="card-title">Detailed Data View</h5>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(data)

        # Grup data berdasarkan tahun
        yearly_data = data.groupby("Year")["Death_Rate"].mean()

        # Visualisasi data historis
        st.markdown("""
        <div class="card mb-4">
            <div class="card-body">
                <h5 class="card-title">Mean Cancer Death Rates Over Time</h5>
            </div>
        </div>
        """, unsafe_allow_html=True)
        plt.figure(figsize=(10, 6))
        plt.plot(yearly_data, label="Mean Death Rate")
        plt.title("Mean Cancer Death Rates Over Time")
        plt.xlabel("Year")
        plt.ylabel("Death Rate")
        plt.legend()
        st.pyplot(plt)

        # Layout responsif untuk input dan evaluasi model
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<h5>Input Parameters</h5>", unsafe_allow_html=True)
            p = st.number_input("ARIMA p", min_value=0, max_value=10, value=5)
            d = st.number_input("ARIMA d", min_value=0, max_value=2, value=1)
            q = st.number_input("ARIMA q", min_value=0, max_value=10, value=0)

        with col2:
            st.markdown("<h5>Model Metrics</h5>", unsafe_allow_html=True)
            train_size = int(len(yearly_data) * 0.8)
            train, test = yearly_data[:train_size], yearly_data[train_size:]
            model_train = ARIMA(train, order=(p, d, q))
            model_train_fit = model_train.fit()
            predictions = model_train_fit.forecast(steps=len(test))
            mape = mean_absolute_percentage_error(test, predictions)
            st.write(f"Mean Absolute Percentage Error (MAPE): {mape * 100:.2f}%")

        # Peramalan data masa depan
        forecast_years = 10
        model_fit = ARIMA(yearly_data, order=(p, d, q)).fit()
        forecast = model_fit.forecast(steps=forecast_years)
        future_years = pd.Series(
            range(yearly_data.index[-1] + 1, yearly_data.index[-1] + 1 + forecast_years)
        )

        st.markdown("""
        <div class="card mb-4">
            <div class="card-body">
                <h5 class="card-title">Forecasting</h5>
            </div>
        </div>
        """, unsafe_allow_html=True)
        plt.figure(figsize=(10, 6))
        plt.plot(yearly_data, label="Historical Data")
        plt.plot(future_years, forecast, label="Forecast", linestyle="--", color="red")
        plt.title("Cancer Death Rates: Historical and Forecast")
        plt.xlabel("Year")
        plt.ylabel("Death Rate")
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)

        # Visualisasi peta angka kematian
        st.markdown("""
        <div class="card mb-4">
            <div class="card-body">
                <h5 class="card-title">Cancer Death Rates Map</h5>
            </div>
        </div>
        """, unsafe_allow_html=True)
        fig_map = px.choropleth(
            data,
            locations="Entity",
            locationmode="country names",
            color="Death_Rate",
            hover_name="Entity",
            color_continuous_scale=px.colors.sequential.Plasma,
            labels={"Death_Rate": "Death Rate per 100,000"},
            title="Cancer Death Rates by Country",
        )
        st.plotly_chart(fig_map)

with tab3:
    # Load the saved model
    filename = 'model-image.pkl'
    model_image = pickle.load(open(filename, 'rb'))

    # Load the pre-trained VGG16 model (without the top classification layers)
    base_model = VGG16(weights='imagenet', include_top=False)

    def extract_features(image_path):
        """Extracts features from an image using VGG16."""
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        features = base_model.predict(img_array)
        return features.flatten()

    # Streamlit app

    #st.title("Deteksi Brain MRI Anomaly")
    st.write("Deteksi Brain MRI Anomaly")

    uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_container_width=True, width=5)

        # Perform prediction
        try:
            image_path = 'temp_image.jpg'  # Temporary file path
            image.save(image_path)

            new_image_features = extract_features(image_path)
            new_image_prediction = model_image.predict(np.array([new_image_features]))
            predicted_label = np.argmax(new_image_prediction)

            label_map = {'normal': 0, 'anomaly': 1}  # Replace with your actual label map
            reverse_label_map = {i: label for label, i in label_map.items()}
            predicted_label_original = reverse_label_map[predicted_label]

            st.write(f"Prediction: {predicted_label_original}")
        except Exception as e:
            st.write(f"Error during prediction: {e}")


with tab4:
    st.write('KLASIFIKASI LOKASI TUMOR')
    age = st.radio('Usia', ('<30', '30-59', '>=60'), horizontal=True)
    sex = st.radio('Jenis Kelamin', ('laki-laki', 'perempuan'), horizontal=True)
    histologic_type = st.radio('Jenis sel yang membentuk tumor', ('epidermoid', 'adeno', 'anaplastic'), horizontal=True)
    degree_of_diffe = st.radio('Tingkat kemiripan sel tumor dengan sel aslinya', ('sangat mirip', 'normal', 'tidak mirip'), horizontal=True)
    st.write('')
    st.write('Kemunculan tanda penyebaran pada:')
    bone = st.radio('a. Tulang', ('ya', 'tidak'), horizontal=True)
    bone_marrow = st.radio('b. Sumsum Tulang', ('ya', 'tidak'), horizontal=True)
    lung = st.radio('c. Paru-Paru', ('ya', 'tidak'), horizontal=True)
    pleura = st.radio('d. Pleura', ('ya', 'tidak'), horizontal=True)
    peritoneum = st.radio('e. Peritoneum', ('ya', 'tidak'), horizontal=True)
    liver = st.radio('f. Hati', ('ya', 'tidak'), horizontal=True)
    brain = st.radio('g. Otak', ('ya', 'tidak'), horizontal=True)
    skin = st.radio('h. Kulit', ('ya', 'tidak'), horizontal=True)
    neck = st.radio('i. Leher', ('ya', 'tidak'), horizontal=True)
    supraclavicular = st.radio('j. Supraclavicular (Tulang Selangka)', ('ya', 'tidak'), horizontal=True)
    axillar = st.radio('k. Axillar (Bawah Sendi Bahu)', ('ya', 'tidak'), horizontal=True)
    mediastinum = st.radio('l. Mediastinum (Rongga Tengah Dada)', ('ya', 'tidak'), horizontal=True)   
    abdominal = st.radio('m. Abdominal (Antara Dada & Panggul)', ('ya', 'tidak'), horizontal=True)   

    input = {
        'Age':age,
        'Sex':sex,
        'Histologic-Type':histologic_type,
        'Degree-of-Diffe':degree_of_diffe,
        'Bone':bone,
        'Bone-Marrow':bone_marrow,
        'Lung':lung,
        'Pleura':pleura,
        'Peritoneum':peritoneum,
        'Liver':liver,
        'Brain':brain,
        'Skin':skin,
        'Neck':neck,
        'Supraclavicular':supraclavicular,
        'Axillar':axillar,
        'Mediastinum':mediastinum,
        'Abdominal':abdominal
    }

    fitur = pd.DataFrame(input, index=[0])

    st.write('Data anda:',fitur)

    data = [age, sex, histologic_type, degree_of_diffe, bone, bone_marrow, lung, pleura, peritoneum, liver, brain, skin, neck, supraclavicular, axillar, mediastinum, abdominal]

    for i in data:
        if i == "30-59" or i == "perempuan" or i == "adeno" or i == "normal" or i == "tidak":
            data[data.index(i)] = 0
        elif i == "<30" or i == "laki-laki" or i == "anaplastic" or i == "tidak mirip" or i == "ya":
            data[data.index(i)] = 1
        elif i == ">=60" or i == "epidermoid" or i == "sangat mirip":
            data[data.index(i)] = 2


    data_sc = np.array([data])

    data_sc = np.delete(data_sc, [5, 11]).reshape(1, -1)

    if st.button("Klasifikasi"):
        clf = pickle.load(open('model-tabular.pkl', 'rb'))

        predict = clf.predict(data_sc)

        st.write('Hasil Prediksi: ', predict[0])

with tab5:


    # Menggunakan tema Bootstrap
    st.markdown("""
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {
                background-color: #f8f9fa;
            }
            .stButton button {
                background-color: #198754;
                color: white;
                border-radius: 5px;
            }
        </style>
    """, unsafe_allow_html=True)

    # Header aplikasi
    st.markdown("""
    <div class="alert alert-primary text-center" role="alert">
        <h1><i class="fa fa-chart-bar"></i> Cancer Death Clustering Dashboard</h1>
        <p>Perform KMeans Clustering and Visualize Cancer Death Rates Globally!</p>
    </div>
    """, unsafe_allow_html=True)

    # Load data
    data_path = "clustering.csv"  # File harus berada di folder yang sama
    df = pd.read_csv(data_path)

    # Menampilkan data awal
    st.markdown("""
    <div class="card mb-4">
        <div class="card-body">
            <h5 class="card-title">Loaded Data</h5>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.dataframe(df.head())

    # Filter data berdasarkan tahun dan penyakit
    selected_year = st.selectbox("Select Year", sorted(df['Year'].unique()))
    selected_disease = st.selectbox("Select Disease", df.columns[3:])  # Kolom penyakit mulai dari indeks ke-3
    filtered_df = df[df['Year'] == selected_year][['Country', 'Code', 'Year', selected_disease]]

    # Menampilkan data terfilter
    st.markdown(f"### Filtered Data for {selected_year} - {selected_disease}")
    st.dataframe(filtered_df)

    # Tangani outlier tanpa menghapus data
    Q1 = filtered_df[selected_disease].quantile(0.25)
    Q3 = filtered_df[selected_disease].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Ganti outlier dengan batas yang diperbolehkan
    filtered_df[selected_disease] = filtered_df[selected_disease].clip(lower=lower_bound, upper=upper_bound)

    # Menampilkan data setelah menangani outlier
    st.markdown("### Data After Handling Outliers")
    st.dataframe(filtered_df)

    # Standardisasi data
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(filtered_df[[selected_disease]])

    # Clustering menggunakan KMeans dengan 4 cluster
    kmeans = KMeans(n_clusters=4, random_state=42)
    filtered_df['Cluster'] = kmeans.fit_predict(scaled_features)

    # Menentukan rentang nilai setiap cluster
    cluster_stats = filtered_df.groupby('Cluster')[selected_disease].agg(['min', 'max', 'count']).reset_index()

    st.markdown("""
    <div class="card mb-4">
        <div class="card-body">
            <h5 class="card-title">Cluster Statistics (Min, Max, Count)</h5>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.dataframe(cluster_stats)

    # Visualisasi Map
    st.markdown("""
    <div class="card mb-4">
        <div class="card-body">
            <h5 class="card-title">World Map of Cancer Death Clusters</h5>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Plot map menggunakan Plotly
    fig_map = px.choropleth(
        filtered_df,
        locations="Country",
        locationmode="country names",
        color="Cluster",
        title="Global Cancer Death Clustering",
        color_continuous_scale=px.colors.sequential.Viridis,
        hover_data={selected_disease: True, 'Cluster': True}
    )
    st.plotly_chart(fig_map)

    # Visualisasi Clustering
    st.markdown("""
    <div class="card mb-4">
        <div class="card-body">
            <h5 class="card-title">Cluster Visualization</h5>
        </div>
    </div>
    """, unsafe_allow_html=True)

    fig_scatter = px.scatter(
        filtered_df,
        x=filtered_df.index,
        y=selected_disease,
        color="Cluster",
        title="Scatter Plot of Clusters",
        labels={"x": "Index", selected_disease: selected_disease},
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    st.plotly_chart(fig_scatter)

    # Save hasil clustering ke CSV
    st.markdown("""
    <div class="card mb-4">
        <div class="card-body">
            <h5 class="card-title">Save Results</h5>
        </div>
    </div>
    """, unsafe_allow_html=True)
    output_file = "clustered_data_filtered.csv"
    filtered_df.to_csv(output_file, index=False)
    st.success(f"Clustered data saved to {output_file}!")
