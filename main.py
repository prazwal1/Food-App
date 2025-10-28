from flask import request, jsonify, Flask, render_template, send_from_directory
import os
import tempfile
import pandas as pd
from src import train_demand_model, predict_next_hour, predict_next_24_hours, generate_realistic_hourly_orders
import pathlib

app = Flask(__name__)


MODEL_PATH = "./model/trained_model.pkl"
DF_PATH = "./static/training_data_for_api.csv"

model = None
df = None
feature_cols = None
dish_cols = None


if not os.path.exists("./model"):
    os.makedirs("./model")

if not os.path.exists("./static"):
    os.makedirs("./static")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})

@app.route("/train", methods=["POST"])
def train():
    global model, df, feature_cols, dish_cols

    # file upload
    if "file" in request.files:
        f = request.files["file"]
        if f.filename == "":
            return jsonify({"error": "empty filename"}), 400
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp_path = tmp.name
            f.save(tmp_path)
        csv_path = tmp_path
    else:
        data = request.get_json(silent=True) or {}
        csv_path = data.get("csv_path")
        if not csv_path:
            return jsonify({"error": "provide file upload 'file' or JSON 'csv_path'"}), 400
        if not os.path.exists(csv_path):
            return jsonify({"error": f"csv_path not found: {csv_path}"}), 400

    try:
        model, df, feature_cols, dish_cols = train_demand_model(csv_path, save_path=MODEL_PATH)
        # persist df for later use by predict endpoints
        df.to_csv(DF_PATH, index=False)
        return jsonify({
            "status": "trained",
            "model_path": MODEL_PATH,
            "training_rows": len(df),
            "dish_columns": dish_cols
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # cleanup temp file if used
        if "tmp_path" in locals() and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

@app.route("/predict/next_hour", methods=["GET"])
def api_predict_next_hour():
    if model is None or df is None or feature_cols is None or dish_cols is None:
        return jsonify({"error": "model not loaded. Train or load a model first."}), 400
    try:
        result = predict_next_hour(model, df, feature_cols, dish_cols)
        # ensure native types
        result = {k: (int(v) if hasattr(v, "__int__") else float(v)) for k, v in result.items()}
        return jsonify({"timestamp": str(df['order_placed_at'].max() + pd.Timedelta(hours=1)), "prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict/next_24_hours", methods=["GET"])
def api_predict_next_24_hours():
    if model is None or df is None or feature_cols is None or dish_cols is None:
        return jsonify({"error": "model not loaded. Train or load a model first."}), 400
    try:
        pred_df = predict_next_24_hours(model, df, feature_cols, dish_cols)
        # convert timestamps and numpy types to native
        records = []
        for r in pred_df.to_dict(orient="records"):
            rec = {}
            for k, v in r.items():
                if k == "timestamp":
                    rec[k] = pd.to_datetime(v).isoformat()
                else:
                    try:
                        rec[k] = int(v)
                    except Exception:
                        try:
                            rec[k] = float(v)
                        except Exception:
                            rec[k] = v
            records.append(rec)
        return jsonify({"predictions": records})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route("/static/<path:filename>")
def static_file(filename):
    # Serve files from a local "static" directory next to this file if present,
    # otherwise fall back to Flask's built-in static handling.
    root = pathlib.Path(__file__).parent
    static_dir = root / "static"
    if static_dir.exists():
        return send_from_directory(str(static_dir), filename)
    # fallback to Flask static folder (Flask sets this up by default to "static")
    try:
        return app.send_static_file(filename)
    except Exception:
        return jsonify({"error": "static file not found"}), 404

@app.route("/favicon.ico")
def favicon():
    root = pathlib.Path(__file__).parent
    static_dir = root / "static"
    ico_name = "favicon.ico"
    if (static_dir / ico_name).exists():
        return send_from_directory(str(static_dir), ico_name)
    try:
        return app.send_static_file(ico_name)
    except Exception:
        return ("", 204)
    
@app.route("/generate", methods=["POST"])
def generate():
    """
    Generate a synthetic dataset only (do not train). Accepts optional JSON {"rows": <int>}
    and saves the generated dataframe to DF_PATH for use by predict endpoints.
    """
    global model, df, feature_cols, dish_cols

    data = request.get_json(silent=True) or {}
    rows = int(data.get("rows", 1008))

    try:
        df_gen = generate_realistic_hourly_orders(target_rows=rows)
        # set server-side df and column metadata but do NOT train a model
        df = df_gen
        feature_cols = ['hour', 'day_of_week', 'is_weekend'] + [c for c in df.columns if '_lag' in c]
        dish_cols = [c for c in df.columns if c not in ['order_placed_at', 'hour', 'day_of_week', 'is_weekend'] and '_lag' not in c]
        df.to_csv(DF_PATH, index=False)
        return jsonify({
            "status": "generated",
            "generated_rows": len(df),
            "dish_columns": dish_cols,
            "model_loaded": model is not None
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route("/", methods=["GET"])
def index():
    """
    Render index.html from templates/ via render_template.
    If that fails (no template folder), try to serve index.html located next to main.py.
    """
    try:
        return render_template("index.html")
    except Exception:
        root = pathlib.Path(__file__).parent
        index_path = root / "index.html"
        if index_path.exists():
            return send_from_directory(str(root), "index.html")
        return ("<html><body><h1>Index</h1><p>No index.html found.</p></body></html>", 200)

if __name__ == "__main__":
    # run with: python main.py
    app.run(host="0.0.0.0", debug=True, port=int(os.environ.get("PORT", 5000)))