import pandas as pd

df = pd.read_excel("results/features_summary.xlsx")

df = df[df["actual_gender"].isin(["male", "female", "child"])]

summary = []

for gender in ["male", "female", "child"]:
    subset = df[df["actual_gender"] == gender]

    if len(subset) == 0:
        continue

    avg_f0 = subset["avg_f0"].mean()
    std_f0 = subset["avg_f0"].std()

    success = (subset["actual_gender"] == subset["predicted_gender"]).mean() * 100

    summary.append({
        "Class": gender,
        "Avg F0": avg_f0,
        "Std": std_f0,
        "Success (%)": success
    })

summary_df = pd.DataFrame(summary)
print(summary_df)