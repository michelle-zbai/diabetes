# 🩺 Diabetes Risk Checker
A comprehensive interactive web application for diabetes risk assessment, personalized health guidance, and evidence-based management recommendations powered by machine learning and AI.

### 🎯 Core Functionality
- **Risk Assessment**: Machine learning model predicts diabetes risk based on health metrics
- **Personalized Guidance**: AI-powered suggestions tailored to your health profile
- **Diabetes Management**: Special mode for users already diagnosed with diabetes
- **Risk Trending**: 10-year projection of diabetes risk trajectory
- **Research Insights**: Peer-reviewed journal articles with working links
- **Professional Guidance**: Forum-style advice on connecting with healthcare providers

### 📊 Health Metrics Tracked
- Age, BMI, HbA1c Level
- Blood Glucose Level
- Gender, Smoking History
- Hypertension & Heart Disease Status
- Diabetes Type (if applicable)

### 🤖 AI-Powered Features
- **Health Suggestions**: Specific, measurable recommendations (e.g., "5-10% weight loss over 3-6 months")
- **Risk Projections**: Year-by-year risk forecasts with contextual tips
- **Journal Recommendations**: 3 peer-reviewed studies matched to your diabetes type and status
- **Professional Consultation Guide**: Personalized talking points for doctor visits

### Prerequisites
- Python 3.10+
- pip (Python package manager)
- OpenAI API key (for AI features)

### Setup

1. **Clone the repository** (if applicable) or navigate to the project directory:
   ```bash
   cd c:\Users\zymic\Desktop\s1\tech\diabetes
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   
   Or install manually:
   ```bash
   pip install streamlit pandas numpy scikit-learn seaborn matplotlib altair openai
   ```

4. **Configure OpenAI API Key**:
   - Create or edit `.streamlit/secrets.toml` in the project directory
   - Add your OpenAI API key:
     ```toml
     OPENAI_API_KEY = "sk-proj-YOUR_API_KEY_HERE"
     ```
   - Get your API key from [OpenAI Platform](https://platform.openai.com/account/api-keys)

5. **Prepare the dataset**:
   - Ensure `diabetes.csv` is in the project root directory
   - Required columns: `age`, `bmi`, `HbA1c_level`, `blood_glucose_level`, `gender`, `smoking_history`, `hypertension`, `heart_disease`, `diabetes`

## How to Use
### For Risk Assessment (No Diabetes)

1. **Enter Your Health Information**:
   - Input age, BMI, HbA1c level, blood glucose level
   - Select gender, smoking history, hypertension/heart disease status
   - Select "No" for "Do you already have diabetes?"

2. **Calculate Risk**:
   - Click "Calculate Risk" button
   - View your risk percentage and category (Low/Medium/High)

3. **Explore Results**:
   - **💊 Health Suggestions**: Get specific actionable recommendations
   - **📊 Your Health Metrics**: See key indicators with status
   - **📈 Projected Risk Trend**: View 10-year risk forecast with tips on hover
   - **Recent Entries**: Track history of assessments
   - **Peer-Reviewed Research Insights**: Read 3 relevant studies with links
   - **Connect with Professionals**: Get guidance on finding and talking to doctors

### For Diabetes Management (Already Diagnosed)

1. **Enter Your Health Information**:
   - Input your current health metrics
   - Select "Yes" for "Do you already have diabetes?"
   - Select your diabetes type (Type 1, Type 2, Gestational, Other)

2. **Calculate**:
   - Click "Calculate Risk"
   - Risk will display as "Already Diagnosed"
   - Suggestions shift to management focus

3. **Get Management Guidance**:
   - **Health Suggestions**: Focus on glycemic control, screening, medication adherence
   - **Research Insights**: Type-specific articles on diabetes management
   - **Professional Guidance**: Talk to specialists (endocrinologist, diabetes educator, etc.)

## Technical Stack

- **Frontend**: Streamlit (Python web framework)
- **ML Model**: Logistic Regression (scikit-learn)
- **Data Processing**: pandas, numpy
- **Visualization**: Altair, Matplotlib, Seaborn
- **AI Integration**: OpenAI GPT-4o-mini
- **Scaling**: StandardScaler (feature normalization)

## Model Details

- **Algorithm**: Logistic Regression with class weight balancing
- **Training Set**: Diabetes dataset with 80/20 train-test split
- **Features**: One-hot encoded categorical variables (gender, smoking_history)
- **Output**: Diabetes risk probability (0-100%)