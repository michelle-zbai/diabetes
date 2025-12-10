import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from openai import OpenAI

st.set_page_config(page_title="Diabetes Risk Checker", page_icon="🩺", layout="centered")
st.title("🩺 Diabetes Risk Checker")
st.caption("Simple interactive tool to estimate diabetes risk, offer suggestions, and visualize your trend.")

# Fixed footer visible at all times
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f0f2f6;
        color: #333;
        text-align: center;
        padding: 12px;
        font-size: 16px;
        font-weight: 600;
        border-top: 1px solid #ddd;
        z-index: 999;
    }
    </style>
    <div class="footer">
        ⚠️ This tool provides an estimate based on patterns in the dataset and is not a medical diagnosis.
    </div>
    """,
    unsafe_allow_html=True
)

# Initialize session state
if 'show_results' not in st.session_state:
    st.session_state['show_results'] = False

@st.cache_data
def load_data(path: str = "diabetes.csv"):
    df = pd.read_csv(path)
    # Normalize categorical text
    df['smoking_history'] = df['smoking_history'].replace('No Info', 'no info')
    df['gender'] = df['gender'].str.lower().replace({'female':'female','male':'male','other':'other'})
    return df

@st.cache_resource
def train_model(df: pd.DataFrame):
    # One-hot encode categoricals
    df_proc = pd.get_dummies(df, columns=['gender', 'smoking_history'], drop_first=True)
    X = df_proc.drop('diabetes', axis=1)
    y = df_proc['diabetes']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # Scale only numeric columns (LogReg benefits from scaling). Keep column names.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = LogisticRegression(max_iter=1000, solver='liblinear', class_weight='balanced')
    model.fit(X_train_scaled, y_train)
    return model, scaler, X.columns.tolist()

# Load and train
df = load_data()
model, scaler, feature_cols = train_model(df)

# LLM-based suggestions function
def get_llm_suggestions(risk_pct, age, bmi, HbA1c, glucose, gender, smoking, hypertension, heart_disease, has_diabetes=False, diabetes_type="Not sure"):
    """Generate personalized health suggestions using OpenAI API.
    Returns a tuple (content, error_code, error_message) where error_code is one of:
    - 'no_key' if OPENAI_API_KEY is missing
    - 'api_error' if API call fails (error_message will have details)
    - None if successful
    """
    # Get API key from Streamlit secrets or environment
    api_key = st.secrets.get("OPENAI_API_KEY", None)
    if not api_key:
        return None, 'no_key', 'Missing OPENAI_API_KEY in .streamlit/secrets.toml'

    try:
        client = OpenAI(api_key=api_key)

        focus = "manage existing diabetes (type: " + diabetes_type + ") with evidence-based care" if has_diabetes else "reduce diabetes risk"
        prompt = f"""Based on this patient data, provide 3-4 SPECIFIC, personalized, actionable suggestions to {focus}. Avoid generic health tips—be concrete and measurable.

Patient Profile: Risk {risk_pct:.1f}% | Age {age} | BMI {bmi} | HbA1c {HbA1c}% | Glucose {glucose} mg/dL | Smoking: {smoking} | Hypertension: {hypertension} | Heart Disease: {heart_disease} | Has Diabetes: {has_diabetes} | Diabetes Type: {diabetes_type}

CRITICAL: Give SPECIFIC recommendations, not generic advice:
- Instead of "reduce weight", suggest: "Aim for 5-10% weight loss (X-Y lbs) over 3-6 months via portion control and 150 min/week moderate activity"
- Instead of "monitor glucose", suggest: "Check fasting glucose 2x/week before breakfast; target <110 mg/dL (non-diabetic) or <130 mg/dL (with diabetes)"
- Instead of "eat healthy", suggest: "Swap refined carbs for whole grains; limit added sugars to <6 tsp/day; increase fiber to 25-30g/day"
- Instead of "exercise", suggest: "Do 30-min brisk walks (Zone 2) 5x/week + 2x/week strength (15-20 min). Start Week 1 with 3x/week if sedentary."
- Instead of "talk to doctor", suggest: "Schedule visit to discuss: HbA1c target ({HbA1c}% is [high/normal/low] for your type), lipid panel, kidney function (eGFR/albuminuria), annual eye/foot exam"
- For medications: "Ask about [specific class: metformin dose, GLP-1 receptor agonist, SGLT2i, insulin type] and how to monitor for side effects"

Format as concise, numbered bullet points. Each recommendation should include metric/number/timeframe."""


        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a concise health advisor. Provide brief, actionable diabetes prevention advice with bold formatting for key actions."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=250,
            temperature=0.7
        )

        return response.choices[0].message.content, None, None
    except Exception as e:
        return None, 'api_error', str(e)

# Visual icon mapping for suggestion keywords
def decorate_suggestions(text: str):
    """Convert LLM suggestion text (markdown/bullets) into visual, icon-prefixed bullets.
    Returns a list of markdown bullet strings with emojis."""
    if not text:
        return []
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    icon_map = [
        ("glucose", "🩸"), ("blood sugar", "🩸"), ("hba1c", "🔬"),
        ("exercise", "🏃"), ("walk", "🚶"), ("activity", "🏃"),
        ("diet", "🥗"), ("meal", "🥗"), ("carb", "🥖"),
        ("weight", "⚖️"), ("bmi", "⚖️"),
        ("salt", "🧂"), ("pressure", "💓"), ("hypertension", "💓"),
        ("sleep", "🛌"), ("stress", "🧘"),
        ("smok", "🚭"), ("quit", "🚭"),
        ("doctor", "👩‍⚕️"), ("check", "📅"), ("monitor", "📟")
    ]
    decorated = []
    for ln in lines:
        lower = ln.lower()
        icon = "✅"
        for key, ic in icon_map:
            if key in lower:
                icon = ic
                break
        # ensure single-line bullet without leading dashes duplication
        content = ln.lstrip("- ")
        decorated.append(f"- {icon} {content}")
    return decorated

# LLM-based trend suggestion (short, one-line)
def get_trend_suggestions_batch(years, risks, ages):
    """Generate brief suggestions for multiple future points in a single LLM call to avoid rate limits.
    Returns (list_of_suggestions, error_code, error_message)
    """
    api_key = st.secrets.get("OPENAI_API_KEY", None)
    if not api_key:
        return None, 'no_key', 'Missing OPENAI_API_KEY in .streamlit/secrets.toml'

    try:
        client = OpenAI(api_key=api_key)
        # Build a compact prompt asking for N tips (one per year index)
        items = "\n".join([f"- Year {y}: age {a}, risk {r:.1f}%" for y, r, a in zip(years, risks, ages)])
        prompt = f"""Provide ONE ultra-short actionable diabetes prevention tip (max 8 words) for each line below.
Return as numbered list 0..{len(years)-1}, no extra commentary.

{items}
"""
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a health advisor. Provide concise, varied, actionable tips."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.9
        )
        content = response.choices[0].message.content
        # Parse lines into list
        lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
        # Extract tips after numbering if present
        tips = []
        for i in range(len(years)):
            # find line starting with i or fallback to ith line
            found = None
            for ln in lines:
                if ln.startswith(str(i)):
                    # remove leading numbering like "0." or "0)"
                    tip = ln.lstrip(str(i)).lstrip('.').lstrip(')').strip()
                    found = tip if tip else ln
                    break
            tips.append(found or (lines[i] if i < len(lines) else ""))
        return tips, None, None
    except Exception as e:
        return None, 'api_error', str(e)

# Always show input form
st.write("### Enter Your Health Information")
col1, col2 = st.columns(2)
age = col1.number_input("Age", min_value=0, max_value=120, value=40, key='age')
bmi = col1.number_input("BMI", min_value=0.0, max_value=80.0, value=25.0, step=0.1, key='bmi')
HbA1c_level = col2.number_input("HbA1c Level", min_value=0.0, max_value=20.0, value=5.8, step=0.1, key='hba1c')
blood_glucose_level = col2.number_input("Blood Glucose Level", min_value=0.0, max_value=500.0, value=110.0, step=1.0, key='glucose')

col3, col4 = st.columns(2)
gender_display = col3.selectbox("Gender", options=["Female", "Male", "Other"], index=0, key='gender_display')
gender = gender_display.lower()
smoking_display = col4.selectbox("Smoking History", options=["Never", "Former", "Current", "Not Current", "Ever", "No Info"], index=0, key='smoking_display')
smoking_history = smoking_display.lower()

col5, col6 = st.columns(2)
hypertension_label = col5.selectbox("Hypertension", options=["No", "Yes"], index=0, key='hypertension_label')
heart_disease_label = col6.selectbox("Heart Disease", options=["No", "Yes"], index=0, key='heart_disease_label')

col7, col8 = st.columns(2)
has_diabetes_label = col7.selectbox("Do you already have diabetes?", options=["No", "Yes"], index=0, key='has_diabetes_label')
if has_diabetes_label == "Yes":
    diabetes_type = col8.selectbox("If yes, what type?", options=["Type 1", "Type 2", "Gestational", "Other", "Not sure"], index=1, key='diabetes_type')
else:
    diabetes_type = "Not sure"

# Map to binary values expected by the model/dataset
hypertension_bin = 1 if hypertension_label == "Yes" else 0
heart_disease_bin = 1 if heart_disease_label == "Yes" else 0
has_diabetes = has_diabetes_label == "Yes"

# Compute on click
if st.button("Calculate Risk", type="primary"):
    # Persist current inputs to session for results display
    st.session_state['inputs'] = {
        'age': age,
        'bmi': bmi,
        'HbA1c_level': HbA1c_level,
        'blood_glucose_level': blood_glucose_level,
        'gender': gender,
        'smoking_history': smoking_history,
        'hypertension_label': hypertension_label,
        'heart_disease_label': heart_disease_label,
        'hypertension_bin': hypertension_bin,
        'heart_disease_bin': heart_disease_bin,
        'has_diabetes': has_diabetes,
        'diabetes_type': diabetes_type
    }
    st.session_state['show_results'] = True
    st.rerun()

# Build feature vector matching training columns
def build_feature_row():
    row = {"age": age,
           "bmi": bmi,
           "HbA1c_level": HbA1c_level,
           "blood_glucose_level": blood_glucose_level,
           "hypertension": hypertension_bin,
           "heart_disease": heart_disease_bin,
           # one-hot gender (drop_first=True -> use male, other vs base female)
           "gender_male": 1 if gender == "male" else 0,
           "gender_other": 1 if gender == "other" else 0,
           # one-hot smoking_history (drop_first=True -> base is first alphabetically; we'll mirror training via get_dummies drop_first)
           "smoking_history_current": 1 if smoking_history == "current" else 0,
           "smoking_history_ever": 1 if smoking_history == "ever" else 0,
           "smoking_history_former": 1 if smoking_history == "former" else 0,
           "smoking_history_never": 1 if smoking_history == "never" else 0,
           "smoking_history_no info": 1 if smoking_history == "no info" else 0,
           "smoking_history_not current": 1 if smoking_history == "not current" else 0}
    # Fill any missing columns with 0
    for col in feature_cols:
        if col not in row:
            row[col] = 0
    # Ensure correct column order
    X_user = pd.DataFrame([row], columns=feature_cols)
    return X_user

if 'history' not in st.session_state:
    st.session_state['history'] = []

# Display results only when Calculate Risk has been clicked
if st.session_state['show_results'] and 'inputs' in st.session_state:
    inp = st.session_state['inputs']
    age = inp['age']; bmi = inp['bmi']; HbA1c_level = inp['HbA1c_level']; blood_glucose_level = inp['blood_glucose_level']
    gender = inp['gender']; smoking_history = inp['smoking_history']
    hypertension_label = inp['hypertension_label']; heart_disease_label = inp['heart_disease_label']
    hypertension_bin = inp['hypertension_bin']; heart_disease_bin = inp['heart_disease_bin']
    has_diabetes = inp.get('has_diabetes', False)
    diabetes_type = inp.get('diabetes_type', 'Not sure')

    # Build features and predict only if user is not already diagnosed
    if not has_diabetes:
        X_user = build_feature_row()
        X_user_scaled = scaler.transform(X_user)
        proba = model.predict_proba(X_user_scaled)[0, 1]
        risk_pct = float(proba * 100)
    else:
        risk_pct = 100.0

    st.markdown("<br>", unsafe_allow_html=True)
    st.write("### Your Estimated Diabetes Risk")
    # Determine matching color based on legend thresholds
    if has_diabetes:
        risk_color = "#CC0000"
        risk_label = "Already Diagnosed"
    elif risk_pct < 10:
        risk_color = "#0A8A0A"  # Low Risk (accessible green)
        risk_label = "Low Risk"
    elif risk_pct < 30:
        risk_color = "#CC5500"  # Medium Risk (accessible orange)
        risk_label = "Medium Risk"
    else:
        risk_color = "#CC0000"  # High Risk (accessible red)
        risk_label = "High Risk"

    # Large, color-matched risk display
    st.markdown(
        f"""
        <div style='padding: 12px; border-radius: 10px; border: 2px solid {risk_color};'>
          <div style='font-size: 22px; font-weight: 700; color: #333;'>Risk Probability</div>
          <div style='font-size: 42px; font-weight: 800; color: {risk_color}; line-height: 1.2;'>
                        {risk_pct:.1f}%
          </div>
          <div style='font-size: 16px; font-weight: 600; color: {risk_color};'>
            {risk_label}
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    if not has_diabetes:
                st.markdown("""
**Risk Level Legend:**
- <span style='color:#0A8A0A; font-weight: bold;'>Low Risk</span>: <10%
- <span style='color:#CC5500; font-weight: bold;'>Medium Risk</span>: 10–30%
- <span style='color:#CC0000; font-weight: bold;'>High Risk</span>: >30%
""", unsafe_allow_html=True)
    else:
                st.info(f"You reported existing diabetes ({diabetes_type}). Focus below on management guidance instead of risk reduction.")

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.write("### Health Suggestions")

    # Try LLM-generated suggestions first
    with st.spinner("🤖 Generating personalized suggestions..."):
        llm_content, llm_error, llm_error_msg = get_llm_suggestions(
            risk_pct, age, bmi, HbA1c_level, blood_glucose_level,
            gender, smoking_history, hypertension_label, heart_disease_label,
            has_diabetes, diabetes_type
        )

    # Show status badge about LLM config
    if llm_error == 'no_key':
        st.info("🔑 LLM: Not configured (missing API key)")
    elif llm_error == 'api_error':
        st.warning("LLM error encountered. Showing fallback suggestions.")
        if llm_error_msg:
            st.caption(f"LLM error details: {llm_error_msg}")
    else:
        pass

    if llm_content:
        # Render LLM suggestions with visual icons for clarity
        for item in decorate_suggestions(llm_content):
            st.markdown(item)

        # Add visual health metrics
        st.write("#### 📊 Your Health Metrics")
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("🩸 Glucose", f"{blood_glucose_level:.0f} mg/dL",
                      delta="Normal" if blood_glucose_level < 126 else "High",
                      delta_color="normal" if blood_glucose_level < 126 else "inverse")
        col_m2.metric("⚖️ BMI", f"{bmi:.1f}",
                      delta="Normal" if bmi < 25 else "Elevated" if bmi < 30 else "High",
                      delta_color="normal" if bmi < 25 else "inverse")
        col_m3.metric("🔬 HbA1c", f"{HbA1c_level:.1f}%",
                      delta="Normal" if HbA1c_level < 5.7 else "Prediabetic" if HbA1c_level < 6.5 else "High",
                      delta_color="normal" if HbA1c_level < 5.7 else "inverse")
    else:
        # Fallback to rule-based suggestions with icons
        suggestions = []
        if blood_glucose_level >= 126 or HbA1c_level >= 6.5:
            suggestions.append("🩸 **Monitor glucose regularly** and consult your doctor.")
        if bmi >= 30:
            suggestions.append("⚖️ **Reduce weight** through diet and **30min daily exercise**.")
        if hypertension_bin == 1:
            suggestions.append("💓 **Control blood pressure**: low-salt diet and stay active.")
        if heart_disease_bin == 1:
            suggestions.append("❤️ **See your cardiologist** about diabetes prevention.")
        if smoking_history in ["current", "ever", "former"]:
            suggestions.append("🚭 **Quit smoking** to lower cardiovascular risk.")
        if not suggestions:
            suggestions.append("✅ **Keep up healthy habits**: balanced diet, stay active, routine checkups.")

        for s in suggestions:
            st.markdown(f"- {s}")

        if llm_error == 'no_key':
            st.info("💡 Add your OpenAI API key to `.streamlit/secrets.toml` for AI-powered personalized suggestions!")

    # Save to session history for trend visualization (only if not already diagnosed)
    if not has_diabetes:
        st.session_state['history'].append({
            'risk_pct': risk_pct,
            'age': age,
            'bmi': bmi,
            'HbA1c_level': HbA1c_level,
            'blood_glucose_level': blood_glucose_level
        })

# Trend chart
if len(st.session_state['history']) > 0 and not (st.session_state.get('inputs', {}).get('has_diabetes', False)):
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.write("### Projected Diabetes Risk Trend (Next 10 Years)")
    # Use the latest entry for projection
    latest = st.session_state['history'][-1]
    years = np.arange(0, 11)
    # Simple projection: risk stays the same, or increases slightly with age
    projected_risk = []
    suggestions = []
    base_risk = latest['risk_pct']
    base_age = latest['age']
    
    # Generate LLM suggestions for each year
    with st.spinner("🤖 Generating personalized timeline suggestions..."):
        # compute risks and ages upfront
        ages = []
        for y in years:
            age = base_age + y
            ages.append(age)
            if age > 40:
                projected_risk.append(min(base_risk + (age - 40) * 1.0, 100.0))
            else:
                projected_risk.append(base_risk)
        # Single LLM call for all points (avoids rate limits)
        tips, err, err_msg = get_trend_suggestions_batch(years, projected_risk, ages)
        if tips and not err:
            suggestions = tips
        else:
            # Fallback list using simple rules
            for r in projected_risk:
                if r < 10:
                    suggestions.append("Maintain healthy habits")
                elif r < 20:
                    suggestions.append("Monitor glucose regularly")
                elif r < 30:
                    suggestions.append("Improve diet & exercise")
                else:
                    suggestions.append("Consult doctor for prevention plan")
            if err == 'api_error' and err_msg:
                st.caption(f"LLM timeline error: {err_msg}")
    
    # Interactive Altair chart with tooltips and axis labels
    import altair as alt
    proj_df = pd.DataFrame({
        'Years from Now': years, 
        'Diabetes Risk (%)': projected_risk,
        'Suggestion': suggestions
    })
    line = alt.Chart(proj_df).mark_line(point=True, size=4).encode(
        x=alt.X('Years from Now:Q', 
                title='Years from Now',
                axis=alt.Axis(labelFontSize=14, titleFontSize=16)),
        y=alt.Y('Diabetes Risk (%):Q', 
                title='Diabetes Risk (%)',
                axis=alt.Axis(labelFontSize=14, titleFontSize=16),
                scale=alt.Scale(domain=[0, 100])),
        tooltip=[
            alt.Tooltip('Years from Now:Q', title='Year'),
            alt.Tooltip('Diabetes Risk (%):Q', title='Risk (%)', format='.1f'),
            alt.Tooltip('Suggestion:N', title='Suggestion')
        ]
    ).properties(
        width=900, 
        height=500, 
        title=alt.Title('Projected Risk Over Time', fontSize=18, fontWeight='bold')
    ).configure_point(
        size=120
    ).configure_mark(
        fontSize=20,
        tooltip=alt.TooltipContent('encoding')
    )

    st.altair_chart(line, use_container_width=True)
    
    st.caption("⚠️ This projection assumes your health status stays the same. Improving habits may lower future risk!")

    with st.expander("Recent Entries"):
        hist_df = pd.DataFrame(st.session_state['history'])
        display_df = hist_df[['risk_pct','age','bmi','HbA1c_level','blood_glucose_level']].round(2).rename(columns={
            'risk_pct': 'Risk %',
            'age': 'Age',
            'bmi': 'BMI',
            'HbA1c_level': 'HbA1c Level',
            'blood_glucose_level': 'Blood Glucose Level'
        })
        st.dataframe(display_df)

# Peer-reviewed journals section
if st.session_state['show_results'] and 'inputs' in st.session_state:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.write("### Peer-Reviewed Research Insights")
    
    def get_journal_recommendations(risk_pct, age, bmi, HbA1c, glucose, gender, smoking, hypertension, heart_disease, has_diabetes=False, diabetes_type="Not sure"):
        """Generate peer-reviewed journal recommendations based on user health profile.
        Returns (content, error_code, error_message)
        """
        api_key = st.secrets.get("OPENAI_API_KEY", None)
        if not api_key:
            return None, 'no_key', 'Missing OPENAI_API_KEY in .streamlit/secrets.toml'

        try:
            client = OpenAI(api_key=api_key)

            prompt = f"""Based on this patient profile, recommend EXACTLY 3 most relevant peer-reviewed research studies related to diabetes prevention and management. No more, no less.

Patient Profile: Risk {risk_pct:.1f}% | Age {age} | BMI {bmi} | HbA1c {HbA1c} | Glucose {glucose} mg/dL | Smoking: {smoking} | Hypertension: {hypertension} | Heart Disease: {heart_disease} | Has Diabetes: {has_diabetes} | Diabetes Type: {diabetes_type}

Rules:
- If Has Diabetes is True and Diabetes Type is provided, pick 3 studies SPECIFIC to that type (e.g., Type 2 glycemic control, cardiometabolic risk; Type 1 insulin/CGM/ketone monitoring; Gestational pregnancy-specific care).
- If Has Diabetes is False, focus on 3 studies about prevention/early intervention.
- CRITICAL: Provide REAL, VALID PubMed or DOI links that work and are current (published 2015-2024).

For EACH of the 3 studies, provide:
1. Journal name (e.g., Diabetes Care, Lancet Diabetes & Endocrinology)
2. Brief study title or finding (1-2 lines)
3. Key actionable insight for the patient
4. A REAL, VALID PubMed link (format: https://pubmed.ncbi.nlm.nih.gov/PMID) OR valid DOI link (format: https://doi.org/10.XXXX/XXXXX)

IMPORTANT: Only use real PubMed IDs or DOI links. Do NOT invent fake links. If uncertain, use a real landmark study you know exists.

Format as Markdown bullet points with clickable links like: [Study Title](https://link.com). Return exactly 3 articles with WORKING links."""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a medical research advisor. Provide accurate, evidence-based recommendations with REAL PubMed and DOI links that can be directly accessed. Only cite real, peer-reviewed studies with valid working URLs. Links must be functional and current (2015-2024)."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=400,
                temperature=0.6
            )

            return response.choices[0].message.content, None, None
        except Exception as e:
            return None, 'api_error', str(e)
    
    # Generate journal recommendations
    with st.spinner("📖 Fetching peer-reviewed research insights..."):
        journal_content, journal_error, journal_error_msg = get_journal_recommendations(
            risk_pct, age, bmi, HbA1c_level, blood_glucose_level,
            gender, smoking_history, hypertension_label, heart_disease_label,
            has_diabetes, diabetes_type
        )
    
    if journal_content:
        st.markdown(journal_content)
    elif journal_error == 'api_error':
        st.warning("Unable to retrieve journal recommendations at this time.")
        if journal_error_msg:
            st.caption(f"Error details: {journal_error_msg}")
    elif journal_error == 'no_key':
        st.info("📖 Configure OpenAI API key to receive peer-reviewed research recommendations.")

    # Connect with professionals / forum-style guidance
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.write("### Connect with Professionals (Forum-style Guidance)")

    col_forum1, col_forum2 = st.columns(2)
    user_location = col_forum1.text_input("Your city/region", value="", key="forum_location")
    doctor_focus = col_forum2.selectbox(
        "What type of doctor do you want to discuss?",
        ["Endocrinologist", "Primary Care", "Cardiologist", "Diabetes Educator", "Dietitian"],
        index=0,
        key="forum_doctor_focus"
    )
    forum_question = st.text_area(
        "What would you ask peers or doctors? (e.g., medication options, CGM/insulin pumps, lifestyle support)",
        value="",
        key="forum_question",
        height=100
    )

    def get_forum_recommendations(location, focus, question):
        """Generate suggested talking points and professional types to consult."""
        api_key = st.secrets.get("OPENAI_API_KEY", None)
        if not api_key:
            return None, 'no_key', 'Missing OPENAI_API_KEY in .streamlit/secrets.toml'
        try:
            client = OpenAI(api_key=api_key)
            prompt = f"""Act as a helpful forum moderator guiding patients on how to find and talk to clinicians.
Location: {location or 'Not provided'}
Doctor focus: {focus}
User question: {question or 'Not provided'}

Provide 4-5 bullets:
- What type of clinician to see and why (aligned to diabetes type/status)
- What to ask in a first visit
- What records/labs to bring (A1c, lipids, kidney, BP, meds)
- How to vet a specialist (board certs, experience with Type 1/2/gestational)
- A reminder to use reputable directories (e.g., local hospital systems, insurance network, ADA/Endocrine Society find-a-doctor) and to avoid sharing private data in forums.
Keep bullets concise."""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a concise, safety-focused health forum moderator."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=280,
                temperature=0.6
            )
            return response.choices[0].message.content, None, None
        except Exception as e:
            return None, 'api_error', str(e)

    if st.button("Get forum-style guidance", type="secondary"):
        with st.spinner("💬 Drafting talking points..."):
            forum_content, forum_err, forum_err_msg = get_forum_recommendations(user_location, doctor_focus, forum_question)
        if forum_content:
            st.markdown(forum_content)
        elif forum_err == 'no_key':
            st.info("🔑 Add your OpenAI API key to get tailored forum-style guidance.")
        else:
            st.warning("Could not generate guidance right now. Try again in a moment.")
            if forum_err_msg:
                st.caption(f"Error: {forum_err_msg}")