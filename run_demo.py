from src.fuzzy_sentic.config import DIMENSIONS, SENTIMENT_LEVELS
from src.fuzzy_sentic.interfaces import SenticPipelineInterface
from src.fuzzy_sentic.models.baseline import LoiaSenatoreBaseline
# from src.fuzzy_sentic.models.neosentic import NeoSenticFuzzy


def print_analysis(pipeline: SenticPipelineInterface, text: str) -> None:
    polarity, emotion_matrix = pipeline.analyze_text(text)

    print(f"\nModelo: {pipeline.__class__.__name__}")
    print(f"Texto analisado: '{text}'")
    print(f"Polaridade Global (Eq. 5): {polarity:.4f}\n")
    print("Matriz Emocional Final (6 Níveis Sentic x 4 Dimensões):")
    print(f"{'Nível':<8} | {DIMENSIONS[0]:<12} | {DIMENSIONS[1]:<10} | {DIMENSIONS[2]:<11} | {DIMENSIONS[3]:<10}")
    print("-" * 62)

    for idx, lvl in enumerate(SENTIMENT_LEVELS):
        row = emotion_matrix[idx]
        print(f"{lvl:<8} | {row[0]:<12.3f} | {row[1]:<10.3f} | {row[2]:<11.3f} | {row[3]:<10.3f}")


if __name__ == "__main__":
    sample_text = "He is happy to go home. However, the weather is not very good."

    baseline_model = LoiaSenatoreBaseline(alpha_cut=0.5)
    print_analysis(baseline_model, sample_text)

    # modern_model = NeoSenticFuzzy(alpha_cut=0.5)
    # print_analysis(modern_model, sample_text)