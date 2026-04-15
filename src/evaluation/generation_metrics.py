def exact_match(prediction: str, reference: str) -> float:
    return float(prediction.strip() == reference.strip())
