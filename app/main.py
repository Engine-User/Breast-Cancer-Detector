import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from io import BytesIO

def get_clean_data():
    data_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'data.csv')
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found at {data_file}")
    data = pd.read_csv(data_file)
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({ 'M': 1, 'B': 0 })
    return data

def add_sidebar():
    st.sidebar.markdown("""
    <style>
    .sidebar-button {
        background-color: #ff69b4;
        color: white;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 4px;
    }
    </style>
    """, unsafe_allow_html=True)

    if st.sidebar.button("About", key="about", help="Learn more", type="primary", use_container_width=True):
        st.sidebar.info("This is a Breast Cancer Risk Predictor app that uses Machine Learning to assess the risk of breast cancer based on various factors.")

    if st.sidebar.button("How to Use", key="how_to_use", help="Learn how", type="primary", use_container_width=True):
        st.sidebar.markdown("""
        ### How to Use
        
        This tool uses a Logistic Regression model trained on a dataset of breast cancer patients. Here's how it works:

        1. Input patient data using the sliders in the sidebar.
        2. The app scales the input data using StandardScaler.
        3. The scaled data is then fed into the trained model.
        4. The model predicts the probability of breast cancer.
        5. Results are displayed as a radar chart and probability scores.

        **Scientific Details:**
        - Model: Logistic Regression
        - Scaling: StandardScaler
        - Features: Various cell nuclei measurements
        - Accuracy: 96.7%

        Remember, this tool is for educational purposes and should not replace professional medical advice.
        """)

    if st.sidebar.button("Contact", key="Connect?", help="Contact information", type="primary", use_container_width=True):
        st.sidebar.info("For any queries or feedback, please contact: ggengineerco@gmail.com.")
        st.sidebar.markdown("<span style='color: red; font-size: 22px;'><strong><em>From Engineer</em></strong></span>", unsafe_allow_html=True)

    st.sidebar.header("Cell Nuclei Measurements")
    
    data = get_clean_data()
    if data is None:
        st.stop()
    
    input_dict = {}

    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=float(data[key].min()),
            max_value=float(data[key].max()),
            value=float(data[key].mean()),
            help=f"Enter the {label.lower()} of the cell nuclei"
        )
    
    return input_dict

def get_radar_chart(input_data):
    categories = list(input_data.keys())
    values = list(input_data.values())

    # Normalize the values to a 0-1 scale for better visualization
    min_vals = [6.981, 9.71, 43.79, 143.5, 0.05263, 0.01938, 0.0, 0.0, 0.106, 0.04996, 0.1115, 0.36, 0.757, 6.802, 0.001713, 0.002252, 0.0, 0.0, 0.007882, 0.0008948, 7.93, 12.02, 50.41, 185.2, 0.07117, 0.02729, 0.0, 0.0, 0.1565, 0.05504]
    max_vals = [28.11, 39.28, 188.5, 2501.0, 0.1634, 0.3454, 0.4268, 0.2012, 0.304, 0.09744, 2.873, 4.885, 21.98, 542.2, 0.03113, 0.1354, 0.396, 0.05279, 0.07895, 0.02984, 36.04, 49.54, 251.2, 4254.0, 0.2226, 1.058, 1.252, 0.291, 0.6638, 0.2075]

    normalized_values = [(v - min_val) / (max_val - min_val) for v, min_val, max_val in zip(values, min_vals, max_vals)]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=normalized_values,
        theta=categories,
        fill='toself',
        name='Patient Data'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False
    )
    
    return fig

def add_predictions(input_data):
    model_file = os.path.join(os.path.dirname(__file__), '..', 'model', 'model.pkl')
    scaler_file = os.path.join(os.path.dirname(__file__), '..', 'model', 'scaler.pkl')

    if not os.path.exists(model_file) or not os.path.exists(scaler_file):
        st.error("Model or scaler file not found. Please check if the files exist.")
        return

    with open(model_file, 'rb') as f:
        model = pickle.load(f)

    with open(scaler_file, 'rb') as f:
        scaler = pickle.load(f)
  
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    input_df = pd.DataFrame(input_array, columns=input_data.keys())  # Create DataFrame with feature names
    input_array_scaled = scaler.transform(input_df)
    prediction = model.predict(input_array_scaled)
  
    st.subheader("Breast Cancer Risk Prediction")
    st.markdown("The prediction is:")
  
    if prediction[0] == 0:
        st.markdown("<span class='diagnosis benign'>Benign</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='diagnosis malicious'>Malignant</span>", unsafe_allow_html=True)
    
    proba = model.predict_proba(input_array_scaled)[0]
    st.markdown(f"Probability of being benign: **{proba[0]:.2f}**")
    st.markdown(f"Probability of being malignant: <span style='color: #ff69b4;'>**{proba[1]:.2f}**</span>", unsafe_allow_html=True)
  
    st.markdown("This app can assist medical professionals in assessing breast cancer risk, but should not be used as a substitute for a professional diagnosis.")

    return prediction, proba

def visualize_data(data, input_data=None, proba=None):
    st.subheader("Data Visualization")

    # Distribution of diagnosis
    st.write("Distribution of Diagnosis")
    if input_data is not None and proba is not None:
        # Create a pie chart for the single prediction
        labels = ['Benign', 'Malignant']
        values = [proba[0], proba[1]]
        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
        fig.update_layout(title_text="Prediction Probability")
    else:
        # Show the distribution for the entire dataset
        benign_count = (data['diagnosis'] == 0).sum()
        malignant_count = (data['diagnosis'] == 1).sum()
        fig = go.Figure(data=[go.Pie(labels=['Benign', 'Malignant'], 
                                     values=[benign_count, malignant_count], hole=.3)])
        fig.update_layout(title_text="Distribution of Benign vs Malignant Cases")
    
    st.plotly_chart(fig)
    st.write("This pie chart shows the distribution of benign and malignant cases. "
             "For user input, it shows the predicted probabilities.")

    # Correlation heatmap
    st.write("Feature Correlations")
    corr = data.corr()
    fig = px.imshow(corr, title="Correlation Heatmap")
    st.plotly_chart(fig)
    st.write("This heatmap shows how different features are correlated with each other. "
             "Darker colors indicate stronger correlations. This can help identify which features might be most important for diagnosis.")

    # Feature importance plot
    st.write("Feature Importance")
    model_file = os.path.join(os.path.dirname(__file__), '..', 'model', 'model.pkl')
    
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    
    importance = pd.DataFrame({'feature': data.columns[1:], 'importance': abs(model.coef_[0])})
    importance = importance.sort_values('importance', ascending=False)
    fig = px.bar(importance.head(10), x='feature', y='importance', title='Top 10 Most Important Features')
    st.plotly_chart(fig)
    st.write("This bar chart shows the most important features for predicting breast cancer. "
             "The height of each bar indicates how important that feature is in the model's decision-making process.")

def collect_historical_data():
    st.subheader("Historical Data Collection")

    # User input fields
    medical_history = st.text_area("Medical History", placeholder="Enter patient's medical history...")
    family_history = st.text_area("Family History", placeholder="Enter patient's family history of breast cancer...")
    previous_biopsies = st.text_area("Previous Biopsies", placeholder="Enter details of any previous breast biopsies...")
    hormone_replacement_therapy = st.selectbox("Hormone Replacement Therapy", ["No", "Yes"])
    menopause_status = st.selectbox("Menopause Status", ["Premenopausal", "Perimenopausal", "Postmenopausal"])
    breast_density = st.selectbox("Breast Density", ["Almost entirely fatty", "Scattered areas of fibroglandular density", "Heterogeneously dense", "Extremely dense"])

    # File upload
    uploaded_files = st.file_uploader("Upload relevant documents (PDF, DOCX, Images)", accept_multiple_files=True, type=['pdf', 'docx', 'png', 'jpg', 'jpeg'])

    if st.button("Generate Action Steps"):
        # Process the inputs and generate action steps
        action_steps = generate_action_steps(medical_history, family_history, previous_biopsies, hormone_replacement_therapy, menopause_status, breast_density, uploaded_files)
        
        st.subheader("Recommended Action Steps")
        for step in action_steps:
            st.write(f"- {step}")

def generate_action_steps(medical_history, family_history, previous_biopsies, hormone_replacement_therapy, menopause_status, breast_density, uploaded_files):
    # This function would contain logic to generate action steps based on the inputs
    # For now, we'll return some placeholder steps
    steps = [
        "Schedule a mammogram and clinical breast exam",
        "Discuss genetic testing options if family history indicates high risk",
        "Review hormone replacement therapy if currently in use",
        "Consider additional screening methods based on breast density",
        "Implement lifestyle changes to reduce risk factors",
        "Schedule follow-up appointments to monitor any changes"
    ]
    return steps

def generate_pdf_report(input_data, prediction, proba):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Breast Cancer Risk Prediction Report")

    # Patient Data
    c.setFont("Helvetica", 12)
    y = height - 80
    for key, value in input_data.items():
        c.drawString(50, y, f"{key}: {value}")
        y -= 20

    # Prediction
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y - 20, "Prediction:")
    c.setFont("Helvetica", 12)
    prediction_text = "Benign" if prediction[0] == 0 else "Malignant"
    c.drawString(50, y - 40, prediction_text)

    # Probabilities
    c.drawString(50, y - 60, f"Probability of being benign: {proba[0]:.2f}")
    c.drawString(50, y - 80, f"Probability of being malignant: {proba[1]:.2f}")

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

def main():
    st.set_page_config(
        page_title="Breast Cancer Risk Predictor",
        page_icon=":female-doctor:",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    css_file = os.path.join(os.path.dirname(__file__), '..', 'assets', 'style.css')
    if os.path.exists(css_file):
        with open(css_file) as f:
            st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
    else:
        st.warning("Style file not found. The app will continue without custom styling.")
    
    input_data = add_sidebar()
    
    with st.container():
        st.markdown("<h1 style='color: #ff69b4;'>Breast Cancer Risk Predictor</h1>", unsafe_allow_html=True)
        st.write("This app predicts the risk of breast cancer based on cell nuclei measurements. Please input the patient's data using the sidebar for an accurate prediction.")
        st.write("The prediction is based on machine learning algorithms trained on historical data.")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Prediction", "Data Visualization", "Diagnose Yourself", "Additional Information"])
    
    with tab1:
        col1, col2 = st.columns([4,1])
        
        with col1:
            radar_chart = get_radar_chart(input_data)
            st.plotly_chart(radar_chart, use_container_width=True)
        with col2:
            prediction, proba = add_predictions(input_data)

        if st.button("Generate PDF Report"):
            pdf = generate_pdf_report(input_data, prediction, proba)
            st.download_button(
                label="Download PDF Report",
                data=pdf,
                file_name="breast_cancer_risk_report.pdf",
                mime="application/pdf"
            )

    with tab2:
        data = get_clean_data()
        visualize_data(data)

    with tab3:
        collect_historical_data()

    with tab4:
        st.subheader("General Information about Breast Cancer")
        st.write("""
        Breast cancer is a type of cancer that starts in the breast. It can start in one or both breasts. It occurs when breast cells mutate (change) and grow out of control, creating a mass of tissue (tumor).

        **Risk Factors:**
        - Increasing age
        - Personal history of breast conditions
        - Family history of breast cancer
        - Inherited genes that increase cancer risk
        - Radiation exposure
        - Obesity
        - Beginning your period at a younger age
        - Beginning menopause at an older age
        - Having your first child at an older age
        - Postmenopausal hormone therapy

        **Prevention Methods:**
        - Maintain a healthy weight
        - Be physically active
        - Limit or avoid alcohol
        - Limit postmenopausal hormone therapy
        - Choose to breastfeed
        - Be vigilant about breast cancer detection

        For more information, please visit:
        - [American Cancer Society](https://www.cancer.org/cancer/breast-cancer.html)
        - [National Cancer Institute](https://www.cancer.gov/types/breast)
        - [World Health Organization](https://www.who.int/cancer/prevention/diagnosis-screening/breast-cancer/en/)
        """)

if __name__ == '__main__':
    main()
