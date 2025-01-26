import os

import pandas as pd
from fpdf import FPDF


model_names = ["Llama", "Mistral", "Olmo"]
modes = ["math", "multihop"]
save_directory = "outputs/pdf_results"
metrics = ["Accuracy"]


def create_pdf_with_table(pdf_filename, title, model_df):
    os.makedirs(os.path.dirname(pdf_filename), exist_ok=True)

    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.add_page()

    pdf.set_font("Arial", size=16)
    pdf.cell(0, 10, title, ln=True, align="C")
    pdf.ln(10)

    pdf.set_font("Arial", size=10)

    col_width = 277 / len(model_df.columns)
    for col in model_df.columns:
        pdf.cell(col_width, 8, col, border=1, align="C")
    pdf.ln()

    for row in model_df.values:
        for item in row:
            pdf.cell(col_width, 8, str(item), border=1, align="C")
        pdf.ln()

    pdf.output(pdf_filename)


for mode in modes:
    for model_name in model_names:
        for metric in metrics:
            if mode == "math":

                if metric == "Accuracy":
                    df = pd.read_csv(
                        "inputs/csv_inputs/Decoding - Zero Shot Results.csv")

                df = df.drop(columns=['GSM8K'])
                df = df.rename(columns={'Unnamed: 8': 'GSM8K'})

                df = df.drop(columns=['MultiArith'])
                df = df.rename(columns={'Unnamed: 14': 'MultiArith'})

                df = df.drop(columns=['MATH'])
                df = df.rename(columns={'Unnamed: 17': 'MATH'})

                df = df.drop(columns=['Math QA'])
                df = df.rename(columns={'Unnamed: 20': 'Math QA'})

                model_df = df[df['model'].str.contains(model_name, na=False)]
                model_df = model_df[['Method', 'GSM8K',
                                    'MultiArith', 'MATH', 'Math QA']]

                model_df = model_df.reset_index(drop=True)

                model_df.loc[0, 'Method'] = "Self_Consistency"
                model_df.loc[1, 'Method'] = "COT_DECODING"
                model_df.loc[2, 'Method'] = "P(True)"
                model_df.loc[3, 'Method'] = "Greedy"
                model_df.loc[4, 'Method'] = "TEMP_Log_COT_DECODING"
                model_df.loc[5, 'Method'] = "GREEDY_Log_COT_DECODING"
                model_df.loc[6, 'Method'] = "MINIMUM_COT_DECODING"
                model_df.loc[7, 'Method'] = "MAXIMUM_COT_DECODING"
                model_df.loc[8, 'Method'] = "Harmonic_COT_DECODING"
                model_df.loc[9, 'Method'] = "Log_+P(t)"
                model_df.loc[10, 'Method'] = "Log_*P(t)"
                model_df.loc[11, 'Method'] = "Log_P(t)*COT_DECODING"
                model_df.loc[12, 'Method'] = "Entropy_*P(t)"
                model_df.loc[13, 'Method'] = "MINIMUM_+P(t)"
                model_df.loc[14, 'Method'] = "Weighted_*P(t)"
                model_df.loc[15, 'Method'] = "MINIMUM_*P(t)"
                model_df.loc[16, 'Method'] = "Harmonic_*P(t)"
                model_df = model_df.fillna('-')

                pdf_filename = f"{save_directory}/{mode}/{metric}/{mode}_{model_name}_{metric}.pdf"
                os.makedirs(os.path.dirname(pdf_filename), exist_ok=True)

                pdf_filename = f"{save_directory}/{mode}/{metric}/{mode}_{model_name}_{metric}.pdf"
                title = f"{mode} {model_name} {metric}"
                create_pdf_with_table(pdf_filename, title, model_df)

            elif mode == "multihop":

                if metric == "Accuracy":
                    df = pd.read_csv(
                        "inputs/csv_inputs/Decoding - Multihop Zero Shot Results.csv")

                df = df.drop(columns=['Trivia'])
                df = df.rename(columns={'Unnamed: 8': 'Trivia'})

                df = df.drop(columns=['HotPot'])
                df = df.rename(columns={'Unnamed: 12': 'HotPot'})

                model_df = df[df['model'].str.contains(model_name, na=False)]
                model_df = model_df[['Method', 'Trivia',
                                    'Trivia Random S=10', 'HotPot']]

                model_df = model_df.reset_index(drop=True)

                model_df.loc[0, 'Method'] = "Self_Consistency"
                model_df.loc[1, 'Method'] = "COT_DECODING"
                model_df.loc[2, 'Method'] = "P(True)"
                model_df.loc[3, 'Method'] = "Greedy"
                model_df.loc[4, 'Method'] = "TEMP_Log_COT_DECODING"
                model_df.loc[5, 'Method'] = "GREEDY_Log_COT_DECODING"
                model_df.loc[6, 'Method'] = "MINIMUM_COT_DECODING"
                model_df.loc[7, 'Method'] = "MAXIMUM_COT_DECODING"
                model_df.loc[8, 'Method'] = "Harmonic_COT_DECODING"
                model_df.loc[9, 'Method'] = "Log_+P(t)"
                model_df.loc[10, 'Method'] = "Log_*P(t)"
                model_df.loc[11, 'Method'] = "Log_P(t)*COT_DECODING"
                model_df.loc[12, 'Method'] = "Entropy_*P(t)"
                model_df.loc[13, 'Method'] = "MINIMUM_+P(t)"
                model_df.loc[14, 'Method'] = "Weighted_*P(t)"
                model_df.loc[15, 'Method'] = "MINIMUM_*P(t)"
                model_df.loc[16, 'Method'] = "Harmonic_*P(t)"
                model_df = model_df.fillna('-')

                pdf_filename = f"{save_directory}/{mode}/{metric}/{mode}_{model_name}_{metric}.pdf"
                os.makedirs(os.path.dirname(pdf_filename), exist_ok=True)

                pdf_filename = f"{save_directory}/{mode}/{metric}/{mode}_{model_name}_{metric}.pdf"
                title = f"{mode} {model_name} {metric}"
                create_pdf_with_table(pdf_filename, title, model_df)
