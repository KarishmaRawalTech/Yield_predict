import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the dataset for EDA (example dataset, replace 'crop' with your actual dataset if needed)
# data = sns.load_dataset('crop')  # Uncomment if using seaborn's sample dataset
data = pd.read_csv('agricultural_yield_train.csv')  # Replace with your actual dataset path

# Set up the page configuration
st.set_page_config(page_title="Crop Yield Prediction", layout="wide")

# Title for the app
st.title('# 🌿💡 Agri-Analytics: Forecasting Yields from Soil to Sun 🌞💧')

# Sidebar navigation
page = st.sidebar.selectbox('Select Option', ['Home', 'EDA'])

# Home Page
if page == 'Home':
    st.markdown("""
    ## Problem Statement  
    The problem at hand is predicting the **crop yield (kg per hectare)** based on various factors, such as:  
    - 🌱 **Soil quality**  
    - 🌾 **Seed variety**  
    - 🧪 **Fertilizer usage**  
    - 🌞 **Weather conditions** (sunny days and rainfall)  
    - 💧 **Irrigation practices**  

    The goal is to understand how each feature influences the crop yield and make accurate predictions based on these features.

    ## Objective of the Model  

    - 🎯 The primary objective is to predict the **crop yield (kg per hectare)** based on input features.  
    - 🔧 By analyzing the impact of factors like soil quality, fertilizer usage, and irrigation schedules, the model aims to help **optimize farming practices** for better yields.  
    - 🧠 Provide insights to farmers and agricultural planners on how to adjust factors such as fertilizer amounts or irrigation schedules based on **expected weather conditions and soil quality**.  
    - 📊 Help farmers plan their resources effectively by predicting yields under varying conditions.
    """)

     
# EDA Page
elif page == 'EDA':
    # Dataset Description
    st.markdown("""
    ## Dataset Columns Description 🌾📋  

    | **Column Name**                          | **Description**                                                                                     |  
    |------------------------------------------|-----------------------------------------------------------------------------------------------------|  
    | 🌱 **Soil_Quality**                       | Represents the quality of soil, measured on a scale (e.g., nutrient content, pH levels).             |  
    | 🌾 **Seed_Variety**                       | Indicates the type of seed used (e.g., 1 for hybrid, 0 for non-hybrid).                              |  
    | 🧪 **Fertilizer_Amount_kg_per_hectare**  | Quantity of fertilizer applied per hectare (in kilograms).                                          |  
    | 🌞 **Sunny_Days**                         | Total number of sunny days during the crop-growing season.                                          |  
    | 🌧️ **Rainfall_mm**                       | Amount of rainfall received (in millimeters) during the crop-growing season.                        |  
    | 💧 **Irrigation_Schedule**                | Represents the irrigation frequency/schedule (e.g., number of irrigations during the growing period).|  
    | 🎯 **Yield_kg_per_hectare**               | The target variable: Crop yield per hectare (in kilograms).                                         |  
    """)


    st.markdown("### Exploratory Data Analysis (EDA) 📊🔍📊")

    # Show the first few rows of the dataset
    st.markdown("<h3>First Few Rows of the Dataset:</h3>", unsafe_allow_html=True)
    st.write(data.head())
    
    # Shape of the dataset
    st.markdown("<h3>Rows and Columns:</h3>", unsafe_allow_html=True)
    Rows, Columns = data.shape
    st.write(f'Rows: {Rows} \nColumns: {Columns}')

    # Check for missing values and duplicates
    st.write("Missing values:")
    st.write(data.isnull().sum())

    st.write("Duplicates:")
    st.write(data.duplicated().sum())

    # Correlation of features
    st.write("Correlation of features:")
    st.write(data.corr())

    # Summary statistics
    st.write("Summary Statistics:")
    st.write(data.describe(include='all').fillna('-'))

    # Visualizations
    st.markdown("### 📊 Univariate Distribution of Agricultural Features:")

    # Set a dark theme with dark background and adjust color palette
    sns.set_theme(style="dark", palette="dark")

    # Create a list of features to visualize
    features = [
        "Soil_Quality", 
        "Seed_Variety", 
        "Fertilizer_Amount_kg_per_hectare", 
        "Sunny_Days", 
        "Rainfall_mm", 
        "Irrigation_Schedule", 
        "Yield_kg_per_hectare"
    ]

    # Create subplots for univariate analysis
    fig, axes = plt.subplots(4, 2, figsize=(8, 10))
    for i, feature in enumerate(features, 1):
        plt.subplot(4, 2, i)
        sns.histplot(data[feature], kde=True, color=sns.color_palette("Set2")[i % len(features)])
        plt.title(f"{feature}", fontsize=14)
        plt.xlabel(feature, fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.grid(False)  # Disable the grid

    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("""
- **Interpretation:**
    - **Soil Quality:** Even distribution of soil quality 🌱.
    - **Seed Variety:** Binary distribution, indicating two distinct seed varieties 🌾🌾.
    - **Fertilizer Amount:** Uniform spread of fertilizer 💧.
    - **Sunny Days:** Normally distributed with a peak around 100 sunny days ☀️.
    - **Rainfall:** Normally distributed with a peak around 500 mm 🌧️.
    - **Irrigation Schedule:** Multiple peaks, indicating variable irrigation practices 💦.
    - **Yield:** Normally distributed with a peak around 800 kg/ha 🌾.
""")


    # Box Plot
    st.markdown("### 📊 Box Plot for Distribution of Data (Outliers Included): 📉")
    fig = plt.figure(figsize=(15, 4))
    sns.boxplot(data=data, palette="Set2", fliersize=5)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("""
- **Interpretation:**
  - **Soil Quality:** Low variability 🌱
  - **Seed Variety:** No variation 🌾
  - **Fertilizer Amount (kg/ha):** Significant variability 💩
  - **Sunny Days:** Consistent 🌞
  - **Rainfall (mm):** High variability 🌧️
  - **Irrigation Schedule:** No variation 💧
  - **Yield (kg/ha):** Significant variability 🌾📊
""")


    # Violin Plot for Yield by Seed Variety
    st.markdown("### 📊 Violin Plot for Yield by Seed Variety: 🌱")
    fig = plt.figure(figsize=(10, 6))
    sns.violinplot(x='Seed_Variety', y='Yield_kg_per_hectare', data=data, palette='Set2')
    plt.title('Violin Plot: Yield by Seed Variety')
    plt.xlabel('Seed Variety')
    plt.ylabel('Yield (kg/ha)')
    st.pyplot(fig)
    
    st.markdown("""
- **Interpretation** 🌱📊
    - The yield distribution for Seed Variety 1 🌾 is wider with a higher median compared to Seed Variety 0 🌾.
""")


    # Violin Plot for Yield Distribution by Irrigation Schedule
    st.markdown("### 📊 Violin Plot for Yield Distribution by Irrigation Schedule: 💧")
    fig = plt.figure(figsize=(10, 6))
    sns.violinplot(data=data, x='Irrigation_Schedule', y='Yield_kg_per_hectare', palette='muted', inner='quart', scale='width')
    st.pyplot(fig)

    st.markdown("""
- **Interpretation** 🌾
- The violin plot shows increasing crop yields 📈 with more irrigation events 💧, highlighting variability in distribution.
- It underscores irrigation's significant impact on yields, providing insights for optimizing water usage and agricultural productivity 💡🌱.
""")


    # Scatter Plot for Fertilizer Amount vs. Yield
    st.markdown("### 📊 Scatter Plot for Fertilizer Amount vs. Yield: 🌱")
    fig = plt.figure(figsize=(15, 5))
    sns.scatterplot(x='Fertilizer_Amount_kg_per_hectare', y='Yield_kg_per_hectare', data=data, color='orange')
    sns.regplot(x='Fertilizer_Amount_kg_per_hectare', y='Yield_kg_per_hectare', data=data, scatter=False, color='darkred', line_kws={'linewidth': 2, 'linestyle': '-'})
    plt.title('Yield vs. Fertilizer Amount')
    plt.xlabel('Fertilizer Amount (kg/ha)')
    plt.ylabel('Yield (kg/ha)')
    st.pyplot(fig)
    
    st.markdown("""
- **Interpretation** 🌱
- The scatter plot shows a relationship between fertilizer amount (kg/ha) and crop yield (kg/ha) 📊.
- The red regression line indicates a positive correlation 🔴; as the fertilizer amount increases, yield tends to increase slightly 📈.
""")


    # Scatter Plot for Rainfall vs. Yield
    st.markdown("### 📊 Scatter Plot for Rainfall vs. Yield: 🌧️🌾")
    fig = plt.figure(figsize=(15, 5))
    sns.scatterplot(x='Rainfall_mm', y='Yield_kg_per_hectare', data=data, color='royalblue', s=100, edgecolor='black', linewidth=1, alpha=0.5)
    sns.regplot(x='Rainfall_mm', y='Yield_kg_per_hectare', data=data, scatter=False, color='darkred', line_kws={'linewidth': 2, 'linestyle': '-'})
    plt.title('Yield vs. Rainfall', fontsize=18, fontweight='bold', color='darkslategray')
    plt.xlabel('Rainfall (mm)', fontsize=14, fontweight='bold', color='darkslategray')
    plt.ylabel('Yield (kg/ha)', fontsize=14, fontweight='bold', color='darkslategray')
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("""
- **Interpretation** 📊
    - The scatterplot shows a negative linear relationship between rainfall (in mm) and yield (in kg/ha) 🌧️📉, as indicated by the downward-sloping red regression line.
    - Excessive rainfall might negatively impact yield 🌧️➡️❌, possibly due to over-saturation of soil or crop damage 🌱💧.
""")


    # Correlation Heatmap
    st.markdown("### 📊 Correlation Heatmap for Numeric Variables: 🔍")
    fig = plt.figure(figsize=(8, 6))
    correlation = data.corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap')
    st.pyplot(fig)
    
    st.markdown("""
- **Interpretation:**
  - 🌱 **Seed Variety** has the strongest positive correlation with yield (𝑟=0.68), indicating it is an important factor for yield prediction.  
  - 🚿 **Irrigation Schedule** (𝑟=0.56) and 🌾 **Fertilizer Amount** (𝑟=0.28) also positively influence yield.  
  - 🌧️ **Rainfall** has a weak negative correlation (𝑟=−0.25), confirming its potential adverse effects at high levels.  
  - ☀️ **Sunny Days** and 🌍 **Soil Quality** show negligible correlations (𝑟=0.10 and 𝑟=0.11).
""")


    # 3D Scatter Plot
    st.markdown("### 📊 3D Scatter Plot for Yield vs. Seed Variety & Irrigation Schedule: 🌱")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(
        data['Seed_Variety'], data['Irrigation_Schedule'], data['Yield_kg_per_hectare'],
        c=data['Yield_kg_per_hectare'], cmap='viridis', edgecolor='k', s=50
    )
    ax.set_xlabel("Seed Variety")
    ax.set_ylabel("Irrigation Schedule")
    ax.set_zlabel("Yield (kg/ha)")
    ax.set_title("3D Plot: Yield vs. Seed Variety & Irrigation")
    plt.colorbar(scatter, label='Yield (kg/ha)')
    st.pyplot(fig)

    st.markdown("""
- **Interpretation:**
  - 🌱 Yield increases as both seed variety improves (closer to 1.0) and irrigation frequency increases.
  - 🌾 Seed varieties closer to higher values (e.g., 0.8–1.0) combined with higher irrigation schedules (10–14) result in maximum yields.
  - 💧 Poor irrigation (e.g., <5) yields minimal output, regardless of seed variety.
""")



 
