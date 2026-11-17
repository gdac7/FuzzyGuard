import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from loia_senatore import LoiaSenatoreBaseline

def load_sentiment140_sample(csv_path, sample_size=5000):
    """
    Carrega uma amostra do Sentiment140. 
    0: target (0 = negativo, 4 = positivo)
    5: text (o texto do tweet)
    """
    column_names = ['target', 'ids', 'date', 'flag', 'user', 'text']
    df = pd.read_csv(csv_path, encoding='latin-1', names=column_names)
    
    df['label'] = df['target'].apply(lambda x: 1 if x == 4 else 0)
    
    df = df[df['target'].isin([0, 4])]
    
    return df.sample(n=min(sample_size, len(df)), random_state=42)

def evaluate_thresholds(pipeline, dataset):
    """
    Reproduz o experimento da Tabela 10 variando o limiar t 
    em t = ±0.1, ±0.2, ±0.4, ±0.6
    """
    thresholds = [0.1, 0.2, 0.4, 0.6]
    results = []

    print("Processando tweets pelo pipeline fuzzy...")
    predictions = []
    ground_truth = []

    for idx, row in dataset.iterrows():
        text = row['text']
        true_label = row['label']
        
        pred_polarity, _ = pipeline.analyze_text(text)
        predictions.append(pred_polarity)
        ground_truth.append(true_label)

    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)

    total_pos_real = (ground_truth == 1).sum()
    total_neg_real = (ground_truth == 0).sum()

    for t in thresholds:
        pred_pos_mask = predictions > t
        true_pos_mask = ground_truth == 1
        
        tp_pos = (pred_pos_mask & true_pos_mask).sum()
        total_retrieved_pos = pred_pos_mask.sum()
        
        p_pos = tp_pos / total_retrieved_pos if total_retrieved_pos > 0 else 0.0
        r_pos = tp_pos / total_pos_real if total_pos_real > 0 else 0.0
        f_pos = (2 * p_pos * r_pos) / (p_pos + r_pos) if (p_pos + r_pos) > 0 else 0.0

        results.append({
            "Threshold": f"±{t}", "Polarity": "Positive",
            "Precision": round(p_pos, 2), "Recall": round(r_pos, 2), "F-measure": round(f_pos, 2)
        })

        pred_neg_mask = predictions < -t
        true_neg_mask = ground_truth == 0
        
        tp_neg = (pred_neg_mask & true_neg_mask).sum()
        total_retrieved_neg = pred_neg_mask.sum()
        
        p_neg = tp_neg / total_retrieved_neg if total_retrieved_neg > 0 else 0.0
        r_neg = tp_neg / total_neg_real if total_neg_real > 0 else 0.0
        f_neg = (2 * p_neg * r_neg) / (p_neg + r_neg) if (p_neg + r_neg) > 0 else 0.0

        results.append({
            "Threshold": f"±{t}", "Polarity": "Negative",
            "Precision": round(p_neg, 2), "Recall": round(r_neg, 2), "F-measure": round(f_neg, 2)
        })

    return pd.DataFrame(results)

if __name__ == "__main__":
    pipeline = LoiaSenatoreBaseline(alpha_cut=0.5)
    
    csv_file_path = "training.1600000.processed.noemoticon.csv" 
    
    try:
        print("Carregando amostra do Sentiment140...")
        sample_df = load_sentiment140_sample(csv_file_path, sample_size=1000)
        
        df_results = evaluate_thresholds(pipeline, sample_df)
        
        print("\n=== Tabela de Desempenho (Replicação da Tabela 10) ===")
        print(df_results.to_string(index=False))
        
    except FileNotFoundError:
        print(f"Aviso: O arquivo '{csv_file_path}' não foi encontrado.")
        print("Baixe o dataset Sentiment140 gratuitamente no Kaggle e coloque na pasta do projeto para rodar a avaliação.")