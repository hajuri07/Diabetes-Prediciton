from flask import Flask, render_template, request, send_file, redirect, url_for, flash
import pandas as pd
import numpy as np
import joblib
import io

app = Flask(__name__)
app.secret_key = "supersecret"  # for flash messages

# ============ LOAD ARTIFACTS ============
pre = joblib.load("preprocessor.joblib")
model = joblib.load("model_xgb.joblib")
cfg = joblib.load("config.joblib")
BEST_THR = float(cfg.get("threshold", 0.35))

# ============ UTILS ============
def load_csv(file_storage):
    df = pd.read_csv(file_storage)
    # clean headers (strip spaces/BOM)
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()
    return df

def detect_and_extract_target(df):
    # find churn col if present (optional for scoring)
    cand = [c for c in df.columns if "churn" in c.lower()]
    if cand:
        tcol = cand[0]
        y = (df[tcol].astype(str).str.strip()
             .map({"Yes":1,"No":0}).fillna(df[tcol]).astype(int, errors="ignore"))
        X = df.drop(columns=[tcol])
        return X, y, tcol
    return df.copy(), None, None

def score_dataframe(df_features, thr=BEST_THR):
    X_proc = pre.transform(df_features)
    proba = model.predict_proba(X_proc)[:, 1]
    pred = (proba >= thr).astype(int)
    risk = pd.cut(proba, bins=[0, thr-0.1, thr+0.1, 1.0],
                  labels=["Low","Medium","High"], include_lowest=True)
    out = df_features.copy()
    out["Churn_Probability"] = proba
    out["Prediction"] = pred
    out["Risk_Level"] = risk
    return out

def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = np.sum((y_true==1)&(y_pred==1))
    tn = np.sum((y_true==0)&(y_pred==0))
    fp = np.sum((y_true==0)&(y_pred==1))
    fn = np.sum((y_true==1)&(y_pred==0))
    precision = tp/(tp+fp+1e-8)
    recall = tp/(tp+fn+1e-8)
    f1 = 2*precision*recall/(precision+recall+1e-8)
    return dict(tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn),
                precision=float(precision), recall=float(recall), f1=float(f1))

# ============ ROUTES ============
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        thr = float(request.form.get("threshold", BEST_THR))
        file = request.files.get("csv_file")
        if not file or file.filename == "":
            flash("Please upload a CSV file.")
            return redirect(url_for("index"))

        try:
            df = load_csv(file)
            X, y, tcol = detect_and_extract_target(df)
            results = score_dataframe(X, thr=thr)

            metrics = None
            if y is not None and len(y)==len(results):
                metrics = compute_metrics(y, results["Prediction"])

            # preview top rows
            preview = results.head(20).to_html(classes="table table-striped table-sm", index=False)

            # store CSV in memory for download
            csv_buf = io.StringIO()
            results.to_csv(csv_buf, index=False)
            csv_bytes = io.BytesIO(csv_buf.getvalue().encode("utf-8"))

            return render_template(
                "results.html",
                threshold=thr,
                metrics=metrics,
                preview_table=preview,
                has_download=True
            ), 200, { "X-Download-File": "scored_customers.csv", "X-Download-Bytes": str(len(csv_bytes.getvalue())) }

        except Exception as e:
            flash(f"Error processing file: {e}")
            return redirect(url_for("index"))

    return render_template("index.html", best_thr=BEST_THR)

@app.route("/download", methods=["POST"])
def download():
    # Re-score and send file (stateless)
    thr = float(request.form.get("threshold_dl", BEST_THR))
    file = request.files.get("csv_file_dl")
    if not file or file.filename == "":
        flash("Please upload the same CSV again to download.")
        return redirect(url_for("index"))

    df = load_csv(file)
    X, _, _ = detect_and_extract_target(df)
    results = score_dataframe(X, thr=thr)

    buf = io.BytesIO()
    results.to_csv(buf, index=False)
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name="scored_customers.csv", mimetype="text/csv")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
