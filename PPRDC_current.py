import streamlit as st
import torch
import transformers
from huggingface_hub import get_full_repo_name
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertForSequenceClassification
import pandas as pd
import numpy as np
import time
from PIL import Image



# Function to load models
@st.cache_resource
def bring_in_models():
    favoritemodel = "SAdeosun/Pharmacy-Practice-Research-Domain-Classifier_PPRDC_Abstract"
    tokenizer = AutoTokenizer.from_pretrained("bioformers/bioformer-16L")
    model = AutoModelForSequenceClassification.from_pretrained(favoritemodel, num_labels=4)
    return tokenizer, model






def main():
    

    img = Image.open("n2.jpg")
    st.image (img)
    
    st.markdown("""<hr style="height:8px;border:none;color:#993399;background-color:#993399;" /> """, unsafe_allow_html=True)
    st.subheader ("About PPRDC")
    st.markdown ("This web application would accept an abstract text (or batches of abstract text in an Excel file) and will classify them into research domains including Clinical, Education, Social & Administrative, and Basic & Translational. It will also return the respective probabilities/scores for the abstract for the respective domains") 
    st.markdown("""<hr style="height:8px;border:none;color:#993399;background-color:#993399;" /> """, unsafe_allow_html=True)
    # Radio button to choose input type
    st.subheader ("Abstract input")
    input_type = st.radio("Select how you want to input your abstracts:", ["Text", "Excel file"])

    if input_type == "Text":
        # Text input
        #user_text = st.text_area("Paste the abstract text here:")
        form_submitted = False
        with st.form(key='my_form'):
          abstract_text_submitted = st.text_input ('Paste abstract text here and press the Analyze button below:')
          submit_button = st.form_submit_button(label = 'Analyze')
        if submit_button:
          form_submitted = True
        if form_submitted:

        #if user_text and st.button:
            split_text = abstract_text_submitted.split('©')
            if len (split_text) > 1:
                abstract_text_submitted = split_text[0]
                abstract_text_ignored = '©' + split_text[1]
            else:
                abstract_text_submitted = split_text[0]
                abstract_text_ignored = None

            st.write(f"**Abstract text analyzed**: \n {abstract_text_submitted}")
            st.markdown(f"**Abstract_text_ignored**: \n <span style='color:red;'>{abstract_text_ignored} </span>", unsafe_allow_html=True)

            tokenizer, model = bring_in_models()

            label_map = {0: "Clinical", 1: "Education", 2: "Social", 3: "Translational"}

            # Single text instance
            inputs_single = tokenizer(abstract_text_submitted, return_tensors="pt").to(model.device)

            # Predict
            model.eval()
            with torch.no_grad():
                outputs_single = model(**inputs_single)

            # Get probabilities and predicted labels
            probabilities_single = torch.nn.functional.softmax(outputs_single.logits, dim=-1)
            predicted_index = torch.argmax(outputs_single.logits, dim=-1).item()

            # Map the predicted index to its corresponding string label
            predicted_class = label_map[predicted_index]

            # Extract the probabilities for each class and convert them to a list
            probs = probabilities_single.cpu().detach().numpy().flatten().tolist()

            # Construct a pandas DataFrame containing the prediction result
            result_df = pd.DataFrame({
                "predicted_label": [predicted_class],
                "probability_Clinical": [probs[0]],
                "probability_Education": [probs[1]],
                "probability_Social": [probs[2]],
                "probability_Translational": [probs[3]]
            })
            st.markdown("""<hr style="height:4px;border:none;color:#993399;background-color:#993399;" /> """, unsafe_allow_html=True)

            st.subheader("Predicted Domain and Probabilities:")
            result_df.set_index ("predicted_label", inplace = True) 
            st.table(result_df)

    elif input_type == "Excel file":
        # DataFrame input
        st.markdown("""<hr style="height:4px;border:none;color:#993399;background-color:#993399;" /> """, unsafe_allow_html=True)
        st.subheader("Upload an Excel file:")
        uploaded_file = st.file_uploader("Choose an Excel file. The column containing the abstracts MUST be named: 'Abstract' (without the quotation marks)", type=["xlsx"])

        if uploaded_file:
            
            df = pd.read_excel(uploaded_file)
            st.write("Uploaded file:")
            st.write(df)

            tokenizer, model = bring_in_models()

            #abstract_column = st.text_input ("Abstract column name")
            df_copy = df.copy()
            df_copy['Abstract_text_removed'] = df_copy['Abstract'].apply(lambda x: '©' + x.split('©')[1] if '©' in x else None)
            df_copy['Abstract'] = df_copy['Abstract'].str.split('©').str.get(0)
            preds = []
            probs = []
            texts = df_copy['Abstract'].values.tolist()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            for i, t in enumerate(texts):
                inputs = tokenizer(t, return_tensors="pt").to(model.device)

                # Predict
                model.eval()
                with torch.no_grad():
                    outputs = model(**inputs)

                # Get probabilities and predicted labels
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
                predicted_idx = torch.argmax(outputs.logits, dim=-1).item()

                # Append predictions and probabilities to lists
                preds.append(predicted_idx)
                probs.append(probabilities.cpu().detach().numpy().tolist())

                # Update progress bar
                #print(f"\rProgress: {i+1}/{len(texts)} ({((i+1)/len(texts)):.2%})", end="")

            # Assign predictions and probabilities to DataFrame columns
            classes_map = {0:"Clinical", 1:"Education", 2:"Social", 3:"Translational"}
            df_copy["pred_index"] = preds
            df_copy["predicted_domain"] = df_copy["pred_index"].map(classes_map)          #[classes_map].preds
            df_copy[[f"probability_{c}" for c in ["Clinical","Education","Social","Translational"]]] = np.array(probs)
            df_copy.drop("pred_index", axis=1, inplace=True)

            st.markdown("""<hr style="height:4px;border:none;color:#993399;background-color:#993399;" /> """, unsafe_allow_html=True)

            st.subheader("Predicted Domains and Probabilities:")

            progress_bar = st.progress (0)
            for perc_completed in range (100):
              time.sleep(0.05)
              progress_bar.progress(perc_completed +1)
            st.success ("Analysis completed!")

            
            st.write (df_copy)
            st.write ("To download the Table of Predicted Domains and Probabilities above, hover on the top right corner of the table and click 'Download as CSV'") 

            
            
            
            
            st.markdown("""<hr style="height:4px;border:none;color:#993399;background-color:#993399;" /> """, unsafe_allow_html=True)

            
            st.write("Summary of Domain Predictions:")
            results_table = df_copy['predicted_domain'].value_counts().reset_index ()
            st.bar_chart (results_table)
            
            #st.table (results_table)
                       


if __name__ == "__main__":
    main()

st.markdown("""<hr style="height:4px;border:none;color:#993399;background-color:#993399;" /> """, unsafe_allow_html=True)

#st.markdown ("**Powered by:**  https://huggingface.co/SAdeosun/Pharmacy-Practice-Research-Domain-Classifier_PPRDC_Abstract")
