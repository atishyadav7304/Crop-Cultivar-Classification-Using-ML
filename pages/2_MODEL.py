import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, cohen_kappa_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_decomposition import PLSRegression
from xgboost import XGBClassifier

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, LSTM
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

st.title("Step 2: Model Training & Evaluation")
st.write("Upload your split datasets (Train, Validation, Test) and train your classification models.")

# --- 1. Data Upload ---
col1, col2, col3 = st.columns(3)
train_file = col1.file_uploader("Upload Training Data", type=["csv", "xlsx"])
val_file = col2.file_uploader("Upload Validation Data", type=["csv", "xlsx"])
test_file = col3.file_uploader("Upload Testing Data", type=["csv", "xlsx"])

if train_file and val_file and test_file:
    train_df = pd.read_csv(train_file) if train_file.name.endswith('.csv') else pd.read_excel(train_file)
    val_df = pd.read_csv(val_file) if val_file.name.endswith('.csv') else pd.read_excel(val_file)
    test_df = pd.read_csv(test_file) if test_file.name.endswith('.csv') else pd.read_excel(test_file)
    
    X_train, y_train = train_df.iloc[:, 1:], train_df.iloc[:, 0]
    X_val, y_val = val_df.iloc[:, 1:], val_df.iloc[:, 0]
    X_test, y_test = test_df.iloc[:, 1:], test_df.iloc[:, 0]

    # Convert column names to strings just in case
    X_train.columns = X_train.columns.astype(str)
    X_test.columns = X_test.columns.astype(str)

    # Label Encoding
    le = LabelEncoder()
    le.fit(pd.concat([y_train, y_val, y_test]))
    y_train_enc = le.transform(y_train)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)
    n_classes = len(le.classes_)

    st.success("Datasets loaded and encoded successfully!")

    # --- 2. Model Selection ---
    st.markdown("---")
    model_choice = st.selectbox("Select Machine Learning Model", 
                                ["Random Forest", "XGBoost", "Logistic Regression", "Linear SVM", "PLS-DA", "1D-CNN", "LSTM"])
    
    if st.button(f"Train {model_choice} Model"):
        with st.spinner(f"Training {model_choice}... This may take a moment."):
            
            y_pred_enc = None
            
            # --- Traditional ML Models ---
            if model_choice == "Random Forest":
                model = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
                model.fit(X_train, y_train_enc)
                y_pred_enc = model.predict(X_test)
                
            elif model_choice == "XGBoost":
                model = XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.05, objective='multi:softmax', num_class=n_classes, random_state=42, n_jobs=-1)
                model.fit(X_train, y_train_enc)
                y_pred_enc = model.predict(X_test)
                
            elif model_choice in ["Logistic Regression", "Linear SVM", "PLS-DA"]:
                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_test_s = scaler.transform(X_test)
                
                if model_choice == "Logistic Regression":
                    model = LogisticRegression(penalty='l2', C=1.0, max_iter=5000, random_state=42)
                    model.fit(X_train_s, y_train_enc)
                    y_pred_enc = model.predict(X_test_s)
                    
                elif model_choice == "Linear SVM":
                    model = LinearSVC(C=1.0, max_iter=10000, random_state=42)
                    model.fit(X_train_s, y_train_enc)
                    y_pred_enc = model.predict(X_test_s)
                    
                elif model_choice == "PLS-DA":
                    Y_train_bin = label_binarize(y_train_enc, classes=np.unique(y_train_enc))
                    model = PLSRegression(n_components=min(20, X_train.shape[1]))
                    model.fit(X_train_s, Y_train_bin)
                    y_pred_enc = np.argmax(model.predict(X_test_s), axis=1)

            # --- Deep Learning Models ---
            elif model_choice in ["1D-CNN", "LSTM"]:
                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_val_s = scaler.transform(X_val)
                X_test_s = scaler.transform(X_test)
                
                X_train_dl = X_train_s.reshape(X_train_s.shape[0], X_train_s.shape[1], 1)
                X_val_dl = X_val_s.reshape(X_val_s.shape[0], X_val_s.shape[1], 1)
                X_test_dl = X_test_s.reshape(X_test_s.shape[0], X_test_s.shape[1], 1)
                
                y_train_cat = to_categorical(y_train_enc)
                y_val_cat = to_categorical(y_val_enc)
                
                model = Sequential()
                if model_choice == "1D-CNN":
                    model.add(Conv1D(filters=32, kernel_size=5, activation='relu', input_shape=(X_train_dl.shape[1], 1)))
                    model.add(BatchNormalization())
                    model.add(MaxPooling1D(2))
                    model.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
                    model.add(BatchNormalization())
                    model.add(MaxPooling1D(2))
                    model.add(Flatten())
                    model.add(Dense(128, activation='relu'))
                    model.add(Dropout(0.3))
                    model.add(Dense(n_classes, activation='softmax'))
                    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                
                elif model_choice == "LSTM":
                    model.add(LSTM(128, return_sequences=True, input_shape=(X_train_dl.shape[1], 1)))
                    model.add(Dropout(0.3))
                    model.add(LSTM(64))
                    model.add(Dropout(0.3))
                    model.add(Dense(64, activation='relu'))
                    model.add(Dense(n_classes, activation='softmax'))
                    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy' if len(y_train_cat.shape)==1 else 'categorical_crossentropy', metrics=['accuracy'])
                
                early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                model.fit(X_train_dl, y_train_cat, validation_data=(X_val_dl, y_val_cat), epochs=50, batch_size=32, callbacks=[early_stop], verbose=0)
                y_pred_enc = np.argmax(model.predict(X_test_dl), axis=1)

            # --- 3. Evaluation & Metrics ---
            accuracy = accuracy_score(y_test_enc, y_pred_enc)
            precision = precision_score(y_test_enc, y_pred_enc, average='weighted', zero_division=0)
            recall = recall_score(y_test_enc, y_pred_enc, average='weighted', zero_division=0)
            f1 = f1_score(y_test_enc, y_pred_enc, average='weighted', zero_division=0)
            kappa = cohen_kappa_score(y_test_enc, y_pred_enc)

            st.markdown("---")
            st.write(f"### {model_choice} Testing Results")
            col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
            col_m1.metric("Accuracy", f"{accuracy:.4f}")
            col_m2.metric("Precision", f"{precision:.4f}")
            col_m3.metric("Recall", f"{recall:.4f}")
            col_m4.metric("F1 Score", f"{f1:.4f}")
            col_m5.metric("Kappa", f"{kappa:.4f}")

            # Store results in Session State for Comparison Page
            if 'model_results' not in st.session_state:
                st.session_state.model_results = []
            
            st.session_state.model_results = [res for res in st.session_state.model_results if res['Model'] != model_choice]
            st.session_state.model_results.append({
                "Model": model_choice, "Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1 Score": f1, "Kappa": kappa
            })

            # --- 4. Wavelength Importance Extraction ---
            st.markdown("---")
            st.write(f"### 🌟 Top 20 Important Wavelengths ({model_choice})")
            
            importances = None
            importance_type = "Importance Factor"
            
            if model_choice in ["Random Forest", "XGBoost"]:
                importances = model.feature_importances_
                importance_type = "Gini / Gain Importance"
                
            elif model_choice in ["Logistic Regression", "Linear SVM"]:
                importances = np.mean(np.abs(model.coef_), axis=0)
                importance_type = "Mean Absolute Coefficient"
                
            elif model_choice == "PLS-DA":
                def calculate_vip(pls_model):
                    t = pls_model.x_scores_
                    w = pls_model.x_weights_
                    q = pls_model.y_loadings_
                    p, h = w.shape
                    s = np.diag(t.T @ t @ q.T @ q)
                    total_s = np.sum(s)
                    vip = np.zeros(p)
                    for i in range(p):
                        weight = np.array([(w[i,j]**2)*s[j] for j in range(h)])
                        vip[i] = np.sqrt(p*np.sum(weight)/total_s)
                    return vip
                importances = calculate_vip(model)
                importance_type = "VIP Score"
                
            elif model_choice in ["1D-CNN", "LSTM"]:
                st.info("Calculating Permutation Importance for Deep Learning model... This might take a few extra seconds.")
                baseline_acc = accuracy_score(y_test_enc, y_pred_enc)
                importances = []
                for i in range(X_test.shape[1]):
                    X_perm = X_test_s.copy()
                    np.random.shuffle(X_perm[:, i])
                    X_perm_dl = X_perm.reshape(X_perm.shape[0], X_perm.shape[1], 1)
                    pred_perm = np.argmax(model.predict(X_perm_dl, verbose=0), axis=1)
                    acc_perm = accuracy_score(y_test_enc, pred_perm)
                    importances.append(baseline_acc - acc_perm)
                importances = np.array(importances)
                importance_type = "Permutation Importance (Accuracy Drop)"

            if importances is not None:
                # Create Importance DataFrame
                importance_df = pd.DataFrame({
                    "Wavelength (nm)": X_train.columns,
                    importance_type: importances
                })
                
                # Sort and select Top 20
                importance_df = importance_df.sort_values(by=importance_type, ascending=False).reset_index(drop=True)
                top_20_df = importance_df.head(20).copy()
                top_20_df.index = top_20_df.index + 1  # Make index start at 1 for ranking
                
                # Plot Bar Chart
                fig, ax = plt.subplots(figsize=(12, 5))
                sns.barplot(data=top_20_df, x="Wavelength (nm)", y=importance_type, palette="viridis", ax=ax)
                plt.xticks(rotation=45)
                plt.title(f"Top 20 Wavelengths - {model_choice} ({importance_type})")
                plt.tight_layout()
                st.pyplot(fig)
                
                # Display Dataframe
                st.dataframe(top_20_df, use_container_width=True)

                # --- 5. Export Data ---
                buffer_export = io.BytesIO()
                with pd.ExcelWriter(buffer_export, engine='xlsxwriter') as writer:
                    # Metrics Sheet
                    metrics_df = pd.DataFrame([{"Metric": "Overall Accuracy", "Value": accuracy},
                                               {"Metric": "Precision", "Value": precision},
                                               {"Metric": "Recall", "Value": recall},
                                               {"Metric": "F1 Score", "Value": f1},
                                               {"Metric": "Kappa", "Value": kappa}])
                    metrics_df.to_excel(writer, index=False, sheet_name='Metrics')
                    
                    # Top 20 Sheet
                    top_20_df.to_excel(writer, index=True, index_label="Rank", sheet_name='Top_20_Wavelengths')
                    
                    # All Wavelengths Sheet
                    importance_df.to_excel(writer, index=True, index_label="Rank", sheet_name='All_Wavelengths')
                
                st.download_button(
                    label=f"📥 Download {model_choice} Metrics & Top 20 Wavelengths (.xlsx)",
                    data=buffer_export.getvalue(),
                    file_name=f'{model_choice}_Metrics_and_Wavelengths.xlsx',
                    mime='application/vnd.ms-excel'
                )