def classify_gender_from_f0(avg_f0):
    if avg_f0 == 0:
        return "unknown"
    elif avg_f0 < 170:
        return "male"
    elif avg_f0 < 300:
        return "female"
    else:
        return "child"