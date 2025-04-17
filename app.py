import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import joblib
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="AML Course Project",
    page_icon="🎓",
    layout="wide"
)

# Title and description
st.title("Preferred Career Domain Prediction")
st.markdown("""
This application analyzes engineering student's data to predict their preferred career domains.
""")

# Sidebar
st.sidebar.header("Pages")
page = st.sidebar.radio("Go to", ["Data Exploration", "Model Training", "Predictions"])

# Load data function
@st.cache_data
def load_data():
    df = pd.read_csv('df11.csv')
    return df

def main():
    # Initialize session state variables if they don't exist
    if 'trained_model' not in st.session_state:
        st.session_state.trained_model = None
    if 'le' not in st.session_state:
        st.session_state.le = None
    if 'ct' not in st.session_state:
        st.session_state.ct = None
    if 'model_metrics' not in st.session_state:
        st.session_state.model_metrics = {}
    
    if page == "Data Exploration":
        data_exploration()
    elif page == "Model Training":
        model_training()
    elif page == "Predictions":
        make_predictions()
    
    # Add footer to the bottom-left corner of the sidebar
st.sidebar.markdown(
    """
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            text-align: left;
            padding: 10px;
            font-size: 12px;
            color: #888;
        }
    </style>
    <div class="footer">
         Vibe Coded with ❤️ by <a href="https://github.com/aashu-0" target="_blank">aashu-0</a>
    </div>
    """,
    unsafe_allow_html=True
)
    
def data_exploration():
    st.header("Data Exploration")
    
    df = load_data()
    

    # Display raw data
    st.subheader("Collected Data")
    st.dataframe(df.head())
    
    # Display data information
    st.subheader("Data Information")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Data Shape:", df.shape)
        st.write("Columns:", df.columns.tolist())
    with col2:
        buffer = pd.DataFrame({
            'Column': df.columns,
            'Non-Null Count': df.count().values,
            'Data Type': df.dtypes.values
        })
        st.dataframe(buffer)
    
    # Data visualizations
    st.subheader("Data Visualizations")
    
    viz_type = st.selectbox(
        "Select Visualization",
        ["Career Domain Distribution", "Gender Distribution", "Department Distribution", 
         "CGPA Distribution", "State Distribution", "Higher Studies vs CGPA", "Technical vs Skill Rating",
         "Career Preferences by Department", "Urban vs Rural Distribution"]
    )
    
    if viz_type == "Career Domain Distribution":
        fig = px.histogram(df, x="preferred_career_domains", title="Distribution of Preferred Career Domains")
        st.plotly_chart(fig, use_container_width=True)
        
    elif viz_type == "Gender Distribution":
        fig = px.pie(df, names="gender", title="Gender Distribution")
        st.plotly_chart(fig, use_container_width=True)
        
    elif viz_type == "Department Distribution":
        fig = px.histogram(df, x="department", title="Distribution by Department")
        fig.update_xaxes(categoryorder="total descending")
        st.plotly_chart(fig, use_container_width=True)
        
    elif viz_type == "CGPA Distribution":
        fig = px.histogram(df, x="cgpa", title="CGPA Distribution", nbins=20)
        st.plotly_chart(fig, use_container_width=True)
        
    elif viz_type == "State Distribution":
        # Top 10 states
        top_states = df['state'].value_counts().head(10).reset_index()
        top_states.columns = ['state', 'count']
        fig = px.bar(top_states, x='state', y='count', title="Top 10 States")
        st.plotly_chart(fig, use_container_width=True)
        
    elif viz_type == "Higher Studies vs CGPA":
        fig = px.scatter(df, x="higher_studies", y="cgpa", color="gender", title="Higher Studies vs CGPA")
        st.plotly_chart(fig, use_container_width=True)
        
    elif viz_type == "Technical vs Skill Rating":
        fig = px.scatter(df, x="avg_technical_rating", y="avg_skill_rating", 
                         color="preferred_career_domains", title="Technical vs Skill Rating")
        st.plotly_chart(fig, use_container_width=True)
        
    elif viz_type == "Career Preferences by Department":
        dept_career = pd.crosstab(df['department'], df['preferred_career_domains'])
        fig = px.imshow(dept_career, aspect="auto", title="Career Preferences by Department")
        st.plotly_chart(fig, use_container_width=True)
        
    elif viz_type == "Urban vs Rural Distribution":
        fig = px.pie(df, names="city", title="Urban vs Rural Distribution")
        st.plotly_chart(fig, use_container_width=True)

    # Correlation matrix
    st.subheader("Correlation Matrix")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    corr_matrix = df[numeric_cols].corr()
    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", title="Correlation Matrix")
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary statistics
    st.subheader("Summary Statistics of Numerical Columns")
    st.dataframe(df.describe())
    st.subheader("Summary Statistics of Categorical Columns")
    categorical_cols = df.select_dtypes(include=['object']).columns
    cat_summary = df[categorical_cols].describe()
    st.dataframe(cat_summary)

def preprocess_data(df):
    # Create label encoder
    le = LabelEncoder()
    df['encoded_career_domains'] = le.fit_transform(df['preferred_career_domains'])
    st.session_state.le = le
    
    # Define features and target
    X = df.drop(columns=['preferred_career_domains', 'encoded_career_domains'])
    y = df['encoded_career_domains']
    
    # Create column transformer
    ohe = OneHotEncoder(sparse_output=False, drop="first")
    oe = OrdinalEncoder(categories=[["Rural", "Tier 3 City", "Tier 2 City", "Metro City"], ['No', 'Yes']])
    
    numeric_transf = Pipeline(steps=[
        ("num_scaler", StandardScaler()),
        ("pca", PCA(n_components=4))
    ])
    
    ct = ColumnTransformer([
        ("num", numeric_transf, ['age', 'studying_yr', 'cgpa', 'avg_technical_rating', 'avg_skill_rating', 'avg_carrer_choice_rating']),
        ("one_hot", ohe, ["gender", "state", "department", "work_style", "course_outside_curriculum", "higher_studies", "family_influence"]),
        ("ord_enc", oe, ["city", "career_counseling_exp"]),
    ], remainder="passthrough")
    
    st.session_state.ct = ct
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    
    # Apply transformations
    X_train = ct.fit_transform(X_train)
    X_test = ct.transform(X_test)
    
    # # Apply SMOTE with lower k_neighbors to fix the ValueEror
    # smote = SMOTE(random_state=123, k_neighbors=3)  # Fix for the SMOTE error
    # X_train, y_train = smote.fit_resample(X_train, y_train)
    
    return X_train, y_train, X_test, y_test, le.classes_

def model_training():
    st.header("Model Training")
    
    df = load_data()
    
    # Display class distribution
    st.subheader("Class Distribution (Career Domains)")
    fig = px.histogram(df, x="preferred_career_domains", title="Distribution of Preferred Career Domains")
    st.plotly_chart(fig, use_container_width=True)
    
    # Preprocessing data
    with st.spinner("Preprocessing data..."):
        X_train, y_train, X_test, y_test, class_names = preprocess_data(df)
        st.success("Data preprocessing completed!")
    
    # Model selection
    st.subheader("Select Model")
    model_option = st.selectbox(
        "Choose a classification algorithm",
        ["Random Forest", "Gradient Boosting", "CatBoostClassifier", "Decision Tree", "Logistic Regression", "Multi Layer Perceptron"]
    )
    
    # Model training
    if st.button("Train Model"):
        with st.spinner(f"Training {model_option} model..."):
            if model_option == "Random Forest":
                model = RandomForestClassifier(class_weight='balanced', random_state=123)
            elif model_option == "Gradient Boosting":
                model = GradientBoostingClassifier(random_state=123)
            elif model_option == "CatBoostClassifier":
                model = CatBoostClassifier(iterations=1000,learning_rate=0.05,depth=6,
                                           loss_function='MultiClass',eval_metric='Accuracy',verbose=100,
                                           random_seed=123)
            elif model_option == "Decision Tree":
                model = DecisionTreeClassifier(class_weight='balanced',random_state=123)
            elif model_option == "Multi Layer Perceptron":
                model = MLPClassifier(hidden_layer_sizes=(100,50), activation='relu',
                                      solver='adam',learning_rate='adaptive',max_iter=1000,
                                      early_stopping=True, random_state=123)
            else:
                model = LogisticRegression(multi_class='multinomial', max_iter=1000, random_state=123)
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Save the trained model
            st.session_state.trained_model = model
            
            # Make predictions on test set
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, target_names=class_names, 
                                           labels=np.unique(y_test),  # Ensure only labels present in y_test are used,
                                           output_dict=True)
            cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))
            
            # Save metrics
            st.session_state.model_metrics = {
                "accuracy": accuracy,
                "report": report,
                "cm": cm,
                "class_names": class_names
            }
            
            st.success(f"Model training completed with accuracy: {accuracy:.2f}")
    
    # Display metrics if model has been trained
    if st.session_state.trained_model is not None:
        st.subheader("Model Evaluation")
        
        st.write(f"Accuracy: {st.session_state.model_metrics['accuracy']:.4f}")
        
        # Classification report
        report_df = pd.DataFrame(st.session_state.model_metrics['report']).transpose()
        st.write("Classification Report:")
        st.dataframe(report_df)
        
        # Confusion Matrix
        st.write("Confusion Matrix:")
        cm = st.session_state.model_metrics['cm']
        class_names = st.session_state.model_metrics['class_names']
        
        unique_classes = np.unique(y_test)
        fig = px.imshow(
            cm,
            labels=dict(x="Predicted", y="True"),
            x=[class_names[i] for i in unique_classes],
            y=[class_names[i] for i in unique_classes],
            text_auto=True,
            title="Confusion Matrix"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance for tree-based models
        if model_option in ["Random Forest", "Gradient Boosting","CatBoostClassifier", "Decision Tree"]:
            st.subheader("Feature Importance")
            
            # Get feature names
            feature_names = []
            # Add numeric features
            feature_names.extend(['PCA_1', 'PCA_2', 'PCA_3', 'PCA_4'])
            # Add one-hot encoded features
            for col in ["gender", "state", "department", "work_style", "course_outside_curriculum", "higher_studies", "family_influence"]:
                unique_vals = df[col].nunique() - 1
                feature_names.extend([f"{col}_{i}" for i in range(unique_vals)])
            # Add ordinal encoded features
            feature_names.extend(["city", "career_counseling_exp"])
            
            # Get importances
            importances = st.session_state.trained_model.feature_importances_
            
            # Create a DataFrame for the feature importances
            if len(importances) == len(feature_names):
                feature_imp = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values(by='Importance', ascending=False)
                
                # Display top 10 features
                top_features = feature_imp.head(10)
                fig = px.bar(top_features, x='Importance', y='Feature', orientation='h',
                             title='Top 10 Feature Importances')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Feature names and importance values don't match in length. Cannot display feature importance.")

def make_predictions():
    st.header("Make Predictions")
    
    if st.session_state.trained_model is None:
        st.warning("Please train a model first in the 'Model Training' section.")
        return
    
    df = load_data()
    
    st.subheader("Enter Student Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=17, max_value=30, value=20)
        gender = st.selectbox("Gender", df['gender'].unique())
        state = st.selectbox("State", df['state'].unique())
        city_type = st.selectbox("City Type", ["Rural", "Tier 3 City", "Tier 2 City", "Metro City"])
        studying_yr = st.number_input("Year of Study", min_value=1, max_value=5, value=2)
        department = st.selectbox("Department", df['department'].unique())
        course_outside = st.selectbox("Course Outside Curriculum", ["Yes", "No", "Maybe"])
    
    with col2:
        cgpa = st.slider("CGPA", min_value=5.5, max_value=10.0, value=8.0, step=0.25)
        work_style = st.selectbox("Work Style", df['work_style'].unique())
        family_influence = st.selectbox("Family Influence", ["Yes", "No", "Maybe"])
        higher_studies = st.selectbox("Higher Studies", ["Yes", "No", "Maybe"])
        career_counseling = st.selectbox("Career Counseling Experience", ["Yes", "No"])
        technical_rating = st.slider("Technical Rating (1-5)", min_value=1.0, max_value=5.0, value=2.5, step=0.2)
        skill_rating = st.slider("Skill Rating (1-5)", min_value=1.0, max_value=5.0, value=3.0, step=0.2)
        career_choice_rating = st.slider("Career Choice Rating (1-5)", min_value=1.0, max_value=5.0, value=4.0, step=0.2)
    
    # Create a DataFrame for the input data
    input_data = pd.DataFrame({
        'age': [age],
        'gender': [gender],
        'state': [state],
        'city': [city_type],
        'studying_yr': [studying_yr],
        'department': [department],
        'course_outside_curriculum': [course_outside],
        'cgpa': [cgpa],
        'work_style': [work_style],
        'family_influence': [family_influence],
        'higher_studies': [higher_studies],
        'career_counseling_exp': [career_counseling],
        'avg_technical_rating': [technical_rating],
        'avg_skill_rating': [skill_rating],
        'avg_carrer_choice_rating': [career_choice_rating]
    })
    
    if st.button("Predict Career Domain"):
        # Transform input data
        input_transformed = st.session_state.ct.transform(input_data)
        
        # Make prediction
        prediction_idx = st.session_state.trained_model.predict(input_transformed)[0]
        prediction_proba = st.session_state.trained_model.predict_proba(input_transformed)[0]
        
        # Get predicted class name
        predicted_career = st.session_state.le.inverse_transform([prediction_idx])[0]
        
        # Display prediction
        st.subheader("Prediction Result")
        st.markdown(f"### Predicted Career Domain: **{predicted_career}**")
        
        # Display probability distribution
        st.subheader("Probability Distribution")
        
        proba_df = pd.DataFrame({
            'Career Domain': st.session_state.le.classes_,
            'Probability': prediction_proba
        }).sort_values(by='Probability', ascending=False)
        
        fig = px.bar(proba_df, x='Career Domain', y='Probability', 
                     title='Career Domain Probabilities')
        st.plotly_chart(fig, use_container_width=True)
        
        # Display recommendations based on predicted career domain
        st.subheader("Recommendations")
        
        recommendations = {
            'IT/Tech': [
                "Focus on programming skills and coding projects",
                "Consider certifications in cloud computing, data science, or software development",
                "Join tech communities and participate in hackathons",
                "Explore internships with tech companies"
            ],
            'Engineering/Manufacturing': [
                "Focus on industry-specific skills relevant to your department",
                "Consider certifications in CAD, project management, or quality control",
                "Seek internships in manufacturing or engineering firms",
                "Work on practical engineering projects"
            ],
            'Government/Public Sector': [
                "Prepare for competitive government exams",
                "Focus on public policy and governance knowledge",
                "Consider internships in government departments",
                "Develop administrative and management skills"
            ],
            'Startups/Innovation': [
                "Build entrepreneurial skills and business acumen",
                "Work on innovative projects or participate in startup competitions",
                "Network with entrepreneurs and join incubation programs",
                "Learn about funding, pitching, and business development"
            ],
            'Creative Arts/Design': [
                "Build a portfolio showcasing your creative work",
                "Learn design tools and software relevant to your field",
                "Participate in design competitions and workshops",
                "Network with professionals in creative industries"
            ],
            'Academia/Education': [
                "Focus on research methodology and academic writing",
                "Consider higher education opportunities",
                "Participate in teaching assistantships or mentoring programs",
                "Publish papers or present at academic conferences"
            ],
            'Finance/Banking': [
                "Learn financial analysis and investment principles",
                "Consider certifications like CFA, CFP, or FRM",
                "Develop quantitative and analytical skills",
                "Seek internships in financial institutions"
            ]
        }
        
        if predicted_career in recommendations:
            for rec in recommendations[predicted_career]:
                st.markdown(f"- {rec}")
        else:
            st.write("No specific recommendations available for this career domain.")

if __name__ == "__main__":
    main()
