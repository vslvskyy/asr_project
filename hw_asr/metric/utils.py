import editdistance

# Don't forget to support cases when target_text == ''


def calc_cer(target_text, predicted_text) -> float:
    if target_text:
        return editdistance.distance(target_text, predicted_text) / len(target_text)
    return 1.0


def calc_wer(target_text, predicted_text) -> float:
    target_words = target_text.split()
    predicted_words = predicted_text.split()
    if target_words:
        return editdistance.distance(target_words, predicted_words) / len(target_words)
    return 1.0
